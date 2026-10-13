//! Native registered image mosaics for the desktop sequence workflow.
//!
//! FITS pixels are kept in float32 working planes, registered with the same
//! bilinear, valid-weight-normalized sampling used by the former SciPy path,
//! then combined per output pixel with the median/MAD contract.

use crate::{fits, Result};
use png::{BitDepth, ColorType, Encoder};
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

const CACHE_VERSION: u32 = 1;
const PREVIEW_SAMPLE_LIMIT: usize = 1 << 18;

#[derive(Debug)]
struct Frame {
    values: Vec<f32>,
    path: String,
}

#[derive(Debug)]
struct Mosaic {
    image: Vec<f32>,
    coverage: Vec<u16>,
    scatter: Vec<f32>,
    source_height: usize,
    source_width: usize,
    output_height: usize,
    output_width: usize,
    origin_x: f64,
    origin_y: f64,
    frame_origins: Vec<[f64; 2]>,
    shifts: Vec<[f64; 2]>,
    frame_paths: Vec<String>,
    mode: String,
    clip_sigma: f64,
    interpolation_order: usize,
    frame_valid_counts: Vec<u64>,
}

type MosaicBounds = (usize, usize, f64, f64, Vec<[f64; 2]>);

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn axis_samples(coordinate: f32, order: usize) -> ([(isize, f32); 4], usize) {
    let floor = coordinate.floor() as isize;
    match order {
        0 => {
            let index = (coordinate + 0.5).floor() as isize;
            ([(index, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)], 1)
        }
        1 => {
            let fraction = coordinate - floor as f32;
            (
                [
                    (floor, 1.0 - fraction),
                    (floor + 1, fraction),
                    (0, 0.0),
                    (0, 0.0),
                ],
                2,
            )
        }
        2 | 3 => {
            let count = if order == 2 { 3 } else { 4 };
            let first = floor - 1;
            let mut samples = [(0, 0.0); 4];
            for (slot, index) in (first..first + count as isize).enumerate() {
                let distance = (coordinate - index as f32).abs();
                let weight = if order == 2 {
                    if distance < 0.5 {
                        0.75 - distance * distance
                    } else if distance < 1.5 {
                        0.5 * (1.5 - distance).powi(2)
                    } else {
                        0.0
                    }
                } else if distance < 1.0 {
                    2.0 / 3.0 - distance * distance + 0.5 * distance.powi(3)
                } else if distance < 2.0 {
                    (2.0 - distance).powi(3) / 6.0
                } else {
                    0.0
                };
                samples[slot] = (index, weight);
            }
            (samples, count)
        }
        _ => unreachable!("interpolation order is validated before sampling"),
    }
}

fn sample(
    frame: &Frame,
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    interpolation_order: usize,
) -> Option<f32> {
    // scipy.ndimage.map_coordinates(mode="constant") rejects coordinates
    // outside the image instead of partially extrapolating at its boundary.
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x > (width - 1) as f32
        || y > (height - 1) as f32
    {
        return None;
    }
    let (x_samples, x_count) = axis_samples(x, interpolation_order);
    let (y_samples, y_count) = axis_samples(y, interpolation_order);
    let mut weighted = 0.0f32;
    let mut weight = 0.0f32;
    for (py, wy) in y_samples.iter().take(y_count) {
        if *wy == 0.0 || *py < 0 || *py >= height as isize {
            continue;
        }
        for (px, wx) in x_samples.iter().take(x_count) {
            let sample_weight = wx * wy;
            if sample_weight == 0.0 || *px < 0 || *px >= width as isize {
                continue;
            }
            let value = frame.values[*py as usize * width + *px as usize];
            if value.is_finite() {
                weighted += value * sample_weight;
                weight += sample_weight;
            }
        }
    }
    (weight > 1.0e-6).then_some(weighted / weight)
}

fn mosaic_bounds(height: usize, width: usize, shifts: &[[f64; 2]]) -> Result<MosaicBounds> {
    if shifts.is_empty() {
        return Err("at least one frame is required".into());
    }
    let min_x = shifts
        .iter()
        .map(|shift| -shift[0])
        .fold(f64::INFINITY, f64::min);
    let max_x = shifts
        .iter()
        .map(|shift| (width - 1) as f64 - shift[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = shifts
        .iter()
        .map(|shift| -shift[1])
        .fold(f64::INFINITY, f64::min);
    let max_y = shifts
        .iter()
        .map(|shift| (height - 1) as f64 - shift[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let origin_x = min_x.floor();
    let origin_y = min_y.floor();
    let output_width = (max_x.ceil() - origin_x + 1.0) as usize;
    let output_height = (max_y.ceil() - origin_y + 1.0) as usize;
    let checked = output_width
        .checked_mul(output_height)
        .ok_or("mosaic output dimensions overflow")?;
    if checked > isize::MAX as usize / 4 {
        return Err("mosaic output is too large".into());
    }
    let frame_origins = shifts
        .iter()
        .map(|shift| [-shift[0] - origin_x, -shift[1] - origin_y])
        .collect();
    Ok((
        output_height,
        output_width,
        origin_x,
        origin_y,
        frame_origins,
    ))
}

fn build_with_shape(
    frames: Vec<Frame>,
    source_height: usize,
    source_width: usize,
    shifts: Vec<[f64; 2]>,
    mode: &str,
    clip_sigma: f64,
    interpolation_order: usize,
) -> Result<Mosaic> {
    if frames.is_empty() {
        return Err("at least one frame is required".into());
    }
    if frames.len() != shifts.len() {
        return Err("frames and cumulative_shifts must have the same length".into());
    }
    if frames.len() > u16::MAX as usize {
        return Err("mosaic supports at most 65535 frames".into());
    }
    if !matches!(mode, "robust_mean" | "median") {
        return Err("combine_mode must be 'robust_mean' or 'median'".into());
    }
    if !clip_sigma.is_finite() || clip_sigma <= 0.0 {
        return Err("clip_sigma must be positive".into());
    }
    if interpolation_order > 3 {
        return Err("interpolation_order must be between 0 and 3".into());
    }
    if shifts.iter().flatten().any(|value| !value.is_finite()) {
        return Err("cumulative_shifts must be finite (frame_count, 2) values".into());
    }
    let source_size = source_height
        .checked_mul(source_width)
        .ok_or("mosaic source dimensions overflow")?;
    if source_height == 0
        || source_width == 0
        || frames.iter().any(|frame| frame.values.len() != source_size)
    {
        return Err("all mosaic frames must have the same image shape".into());
    }
    let (output_height, output_width, origin_x, origin_y, frame_origins) =
        mosaic_bounds(source_height, source_width, &shifts)?;
    let output_size = output_height
        .checked_mul(output_width)
        .ok_or("mosaic output dimensions overflow")?;
    let mut image = vec![f32::NAN; output_size];
    let mut coverage = vec![0u16; output_size];
    let mut scatter = vec![f32::NAN; output_size];
    let mut frame_valid_counts = vec![0u64; frames.len()];
    let mut samples = Vec::with_capacity(frames.len());
    let mut deviations = Vec::with_capacity(frames.len());
    let mut kept = Vec::with_capacity(frames.len());

    for out_y in 0..output_height {
        let common_y = (out_y as f32) + (origin_y as f32);
        for out_x in 0..output_width {
            let common_x = (out_x as f32) + (origin_x as f32);
            samples.clear();
            for (frame_index, (frame, shift)) in frames.iter().zip(&shifts).enumerate() {
                let raw_x = common_x + shift[0] as f32;
                let raw_y = common_y + shift[1] as f32;
                if let Some(value) = sample(
                    frame,
                    source_width,
                    source_height,
                    raw_x,
                    raw_y,
                    interpolation_order,
                ) {
                    samples.push(value);
                    frame_valid_counts[frame_index] += 1;
                }
            }
            if samples.is_empty() {
                continue;
            }
            let offset = out_y * output_width + out_x;
            coverage[offset] = samples.len() as u16;
            let center = median(&mut samples);
            deviations.clear();
            deviations.extend(samples.iter().map(|value| (value - center).abs()));
            let robust_scatter = median(&mut deviations) * 1.4826f32;
            scatter[offset] = robust_scatter;
            if mode == "median" {
                image[offset] = center;
                continue;
            }
            kept.clear();
            let limit = clip_sigma as f32 * robust_scatter.max(1.0e-3);
            kept.extend(
                samples
                    .iter()
                    .copied()
                    .filter(|value| ((*value - center).abs() <= limit) || samples.len() <= 2),
            );
            image[offset] = if kept.is_empty() {
                center
            } else {
                kept.iter().copied().sum::<f32>() / kept.len() as f32
            };
        }
    }
    Ok(Mosaic {
        image,
        coverage,
        scatter,
        source_height,
        source_width,
        output_height,
        output_width,
        origin_x,
        origin_y,
        frame_origins,
        frame_paths: frames.iter().map(|frame| frame.path.clone()).collect(),
        shifts,
        mode: mode.to_owned(),
        clip_sigma,
        interpolation_order,
        frame_valid_counts,
    })
}

fn metadata(mosaic: &Mosaic) -> Value {
    let covered = mosaic.coverage.iter().filter(|value| **value > 0).count();
    let overlap = mosaic.coverage.iter().filter(|value| **value >= 2).count();
    let max_coverage = mosaic.coverage.iter().copied().max().unwrap_or(0);
    let mean_coverage = if covered == 0 {
        0.0
    } else {
        mosaic
            .coverage
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>()
            / covered as f64
    };
    let mut min_x = usize::MAX;
    let mut min_y = usize::MAX;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    for y in 0..mosaic.output_height {
        for x in 0..mosaic.output_width {
            if mosaic.coverage[y * mosaic.output_width + x] > 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    let bbox = (min_x != usize::MAX).then(|| json!([min_x, min_y, max_x, max_y]));
    json!({
        "mosaic_cache_version": CACHE_VERSION,
        "frame_count": mosaic.shifts.len(),
        "source_shape": [mosaic.source_height, mosaic.source_width],
        "output_shape": [mosaic.output_height, mosaic.output_width],
        "common_origin_xy": [mosaic.origin_x, mosaic.origin_y],
        "frame_origins_xy": mosaic.frame_origins,
        "cumulative_shifts": mosaic.shifts,
        "frame_paths": mosaic.frame_paths,
        "combine_mode": mosaic.mode,
        "clip_sigma": mosaic.clip_sigma,
        "interpolation_order": mosaic.interpolation_order,
        "masked_zero_pixels": true,
        "masked_auxiliary_pixels": true,
        "frame_valid_pixel_counts": mosaic.frame_valid_counts,
        "covered_pixel_count": covered,
        "overlap_pixel_count": overlap,
        "max_coverage": max_coverage,
        "mean_coverage": mean_coverage,
        "covered_bbox_xy": bbox,
        "uncovered_pixel_count": mosaic.coverage.len() - covered,
        "union_area_ratio": covered as f64 / (mosaic.source_height * mosaic.source_width).max(1) as f64
    })
}

fn npy_header(descriptor: &str, height: usize, width: usize) -> Vec<u8> {
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({}, {}), }}",
        descriptor, height, width
    );
    let preamble = 10usize;
    let padding = (16 - ((preamble + header.len() + 1) % 16)) % 16;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    header.into_bytes()
}

fn write_npy_f32(path: &Path, values: &[f32], height: usize, width: usize) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path).map_err(|error| error.to_string())?);
    let header = npy_header("<f4", height, width);
    writer
        .write_all(b"\x93NUMPY\x01\x00")
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&(header.len() as u16).to_le_bytes())
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&header)
        .map_err(|error| error.to_string())?;
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn write_npy_u16(path: &Path, values: &[u16], height: usize, width: usize) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path).map_err(|error| error.to_string())?);
    let header = npy_header("<u2", height, width);
    writer
        .write_all(b"\x93NUMPY\x01\x00")
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&(header.len() as u16).to_le_bytes())
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&header)
        .map_err(|error| error.to_string())?;
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn percentile(sorted: &[f32], q: f64) -> f32 {
    if sorted.len() <= 1 {
        return sorted.first().copied().unwrap_or(0.0);
    }
    let rank = q * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = (rank - lower as f64) as f32;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

fn write_png_rgba(path: &Path, width: usize, height: usize, rgba: &[u8]) -> Result<()> {
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = Encoder::new(file, width as u32, height as u32);
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
    writer
        .write_image_data(rgba)
        .map_err(|error| error.to_string())
}

fn write_png_gray(path: &Path, width: usize, height: usize, gray: &[u8]) -> Result<()> {
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = Encoder::new(file, width as u32, height as u32);
    encoder.set_color(ColorType::Grayscale);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
    writer
        .write_image_data(gray)
        .map_err(|error| error.to_string())
}

fn write_artifacts(
    out_dir: &Path,
    mosaic: &Mosaic,
    metadata: &Value,
    raw_preview: bool,
) -> Result<()> {
    if !mosaic.image.iter().any(|value| value.is_finite()) {
        return Err("mosaic image contains no finite pixel".into());
    }
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    write_npy_f32(
        &out_dir.join("registered_mosaic_adu.npy"),
        &mosaic.image,
        mosaic.output_height,
        mosaic.output_width,
    )?;
    write_npy_u16(
        &out_dir.join("registered_mosaic_coverage.npy"),
        &mosaic.coverage,
        mosaic.output_height,
        mosaic.output_width,
    )?;
    write_npy_f32(
        &out_dir.join("registered_mosaic_scatter.npy"),
        &mosaic.scatter,
        mosaic.output_height,
        mosaic.output_width,
    )?;
    let metadata_path = out_dir.join("registered_mosaic_metadata.json");
    fs::write(
        metadata_path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(metadata).map_err(|error| error.to_string())?
        ),
    )
    .map_err(|error| error.to_string())?;

    let mut footprint = Vec::with_capacity(mosaic.coverage.len());
    let mut rgba = Vec::with_capacity(mosaic.image.len() * 4);
    let mut finite = mosaic
        .image
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .step_by(mosaic.image.len().div_ceil(PREVIEW_SAMPLE_LIMIT).max(1))
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Err("mosaic image contains no finite pixel".into());
    }
    finite.sort_by(f32::total_cmp);
    let low = percentile(&finite, 0.01);
    let center = percentile(&finite, 0.5);
    let q25 = percentile(&finite, 0.25);
    let q75 = percentile(&finite, 0.75);
    let high = percentile(&finite, 0.995);
    let noise = ((q75 - q25) / 1.3489795).max(f32::EPSILON);
    let enhanced_low = low.max(center - 0.5 * noise);
    let enhanced_high = high.max(center + 12.0 * noise).max(enhanced_low + 1.0);
    let raw_high = high.max(low + 1.0);
    let mut raw_rgba = raw_preview.then(|| Vec::with_capacity(mosaic.image.len() * 4));
    for (index, value) in mosaic.image.iter().copied().enumerate() {
        let covered = mosaic.coverage[index] > 0;
        let alpha = if value.is_finite() && covered { 255 } else { 0 };
        let enhanced = if value.is_finite() {
            ((value - enhanced_low) / (enhanced_high - enhanced_low))
                .clamp(0.0, 1.0)
                .powf(1.15)
        } else {
            0.0
        };
        let gray = (enhanced * 255.0).round() as u8;
        footprint.push(if covered { 255 } else { 0 });
        rgba.extend_from_slice(&[gray, gray, gray, alpha]);
        if let Some(raw_rgba) = raw_rgba.as_mut() {
            let raw = if value.is_finite() {
                ((value - low) / (raw_high - low)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let raw_gray = (raw * 255.0).round() as u8;
            raw_rgba.extend_from_slice(&[raw_gray, raw_gray, raw_gray, alpha]);
        }
    }
    write_png_gray(
        &out_dir.join("registered_mosaic_footprint.png"),
        mosaic.output_width,
        mosaic.output_height,
        &footprint,
    )?;
    write_png_rgba(
        &out_dir.join("registered_mosaic_preview.png"),
        mosaic.output_width,
        mosaic.output_height,
        &rgba,
    )?;
    if let Some(raw_rgba) = raw_rgba {
        write_png_rgba(
            &out_dir.join("registered_mosaic_raw_preview.png"),
            mosaic.output_width,
            mosaic.output_height,
            &raw_rgba,
        )?;
    }
    Ok(())
}

fn parse_shifts(path: &Path) -> Result<Vec<[f64; 2]>> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read sequence JSON {}: {error}", path.display()))?;
    let mut payload: Value = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    if let Some(sequence) = payload.get("sequence").filter(|value| value.is_object()) {
        payload = sequence.clone();
    }
    let rows = payload
        .get("cumulative_shifts")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("JSON missing cumulative_shifts: {}", path.display()))?;
    rows.iter()
        .map(|row| {
            let values = row
                .as_array()
                .filter(|values| values.len() == 2)
                .ok_or("each cumulative shift must have two values")?;
            let x = values[0]
                .as_f64()
                .ok_or("cumulative shifts must be numeric")?;
            let y = values[1]
                .as_f64()
                .ok_or("cumulative shifts must be numeric")?;
            if !x.is_finite() || !y.is_finite() {
                return Err("cumulative_shifts must be finite (frame_count, 2) values".into());
            }
            Ok([x, y])
        })
        .collect()
}

fn list_fits(data_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(data_dir)
        .map_err(|error| format!("cannot read FITS directory {}: {error}", data_dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext.to_string_lossy().eq_ignore_ascii_case("fits"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn run_paths(
    paths: &[PathBuf],
    shifts: Vec<[f64; 2]>,
    mode: &str,
    clip_sigma: f64,
    interpolation_order: usize,
    raw_preview: bool,
    out_dir: &Path,
) -> Result<Value> {
    if paths.len() < 2 {
        return Err(format!(
            "at least 2 FITS files are required: {}",
            paths
                .first()
                .and_then(|path| path.parent())
                .unwrap_or(Path::new("."))
                .display()
        ));
    }
    if paths.len() != shifts.len() {
        return Err(format!(
            "sequence JSON has {} shifts but data directory has {} frames",
            shifts.len(),
            paths.len()
        ));
    }
    let mut frames = Vec::with_capacity(paths.len());
    let mut shape = None;
    for path in paths {
        let fits = fits::read_fits(path, false)?;
        if shape.is_some_and(|expected| expected != (fits.height, fits.width)) {
            return Err("all mosaic frames must have the same image shape".into());
        }
        shape = Some((fits.height, fits.width));
        let mask = fits::auxiliary_mask(fits.height, fits.width)?;
        let mut values = Vec::with_capacity(fits.data.len());
        for (index, masked) in mask.iter().enumerate() {
            let value = fits.data.value(index) as f32;
            values.push(if !value.is_finite() || value == 0.0 || *masked {
                f32::NAN
            } else {
                value
            });
        }
        let resolved = fs::canonicalize(path).unwrap_or_else(|_| path.clone());
        frames.push(Frame {
            values,
            path: resolved.to_string_lossy().into_owned(),
        });
    }
    let (height, width) = shape.ok_or("at least one frame is required")?;
    let mosaic = build_with_shape(
        frames,
        height,
        width,
        shifts,
        mode,
        clip_sigma,
        interpolation_order,
    )?;
    let summary = metadata(&mosaic);
    write_artifacts(out_dir, &mosaic, &summary, raw_preview)?;
    Ok(json!({
        "tool":"rst19-mosaic",
        "out_dir":out_dir,
        "result_path":out_dir.join("registered_mosaic_metadata.json"),
        "result":summary
    }))
}

#[derive(Debug)]
struct Cli {
    data_dir: PathBuf,
    sequence_json: Option<PathBuf>,
    out_dir: PathBuf,
    mode: String,
    clip_sigma: f64,
    interpolation_order: usize,
    raw_preview: bool,
}

fn parse_args(args: &[String]) -> Result<Cli> {
    let mut data_dir = None;
    let mut sequence_json = None;
    let mut out_dir = PathBuf::from("tmp/registered-mosaic");
    let mut mode = "robust_mean".to_owned();
    let mut clip_sigma = 3.0;
    let mut interpolation_order = 1;
    let mut raw_preview = false;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline) = arg.strip_prefix("--").map_or(("", None), |rest| {
            rest.split_once('=')
                .map_or((rest, None), |(name, value)| (name, Some(value)))
        });
        if name.is_empty() {
            if data_dir.replace(PathBuf::from(arg)).is_some() {
                return Err("only one FITS directory is accepted".into());
            }
            index += 1;
            continue;
        }
        if name == "raw-preview" {
            if inline.is_some() {
                return Err("--raw-preview does not take a value".into());
            }
            raw_preview = true;
            index += 1;
            continue;
        }
        let value = if let Some(value) = inline {
            value.to_owned()
        } else {
            index += 1;
            args.get(index)
                .cloned()
                .ok_or_else(|| format!("--{name} requires a value"))?
        };
        match name {
            "sequence-json" => sequence_json = Some(PathBuf::from(value)),
            "out-dir" => out_dir = PathBuf::from(value),
            "mode" => mode = value,
            "clip-sigma" => {
                clip_sigma = value.parse().map_err(|_| "clip_sigma must be positive")?
            }
            "interpolation-order" => {
                interpolation_order = value
                    .parse()
                    .map_err(|_| "interpolation_order must be between 0 and 3")?;
                if interpolation_order > 3 {
                    return Err("interpolation_order must be between 0 and 3".into());
                }
            }
            "tile-rows" => {
                let _: usize = value.parse().map_err(|_| "tile_rows must be positive")?;
            }
            _ => return Err(format!("unknown option --{name}")),
        }
        index += 1;
    }
    Ok(Cli {
        data_dir: data_dir.ok_or("a FITS directory is required")?,
        sequence_json,
        out_dir,
        mode,
        clip_sigma,
        interpolation_order,
        raw_preview,
    })
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let cli = parse_args(args)?;
    let paths = list_fits(&cli.data_dir)?;
    if paths.len() < 2 {
        return Err(format!(
            "at least 2 FITS files are required: {}",
            cli.data_dir.display()
        ));
    }
    let sequence_path = cli
        .sequence_json
        .ok_or("a sequence JSON is required to avoid inventing registration shifts")?;
    let shifts = parse_shifts(&sequence_path)?;
    run_paths(
        &paths,
        shifts,
        &cli.mode,
        cli.clip_sigma,
        cli.interpolation_order,
        cli.raw_preview,
        &cli.out_dir,
    )
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("Usage: rst19-mosaic DATA_DIR --sequence-json FILE [OPTIONS]\n\n--out-dir DIR --mode robust_mean|median --clip-sigma N --interpolation-order 0|1|2|3 --tile-rows N --raw-preview");
        return Ok(());
    }
    let result = run_desktop(args)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|error| error.to_string())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEST_DIR: AtomicUsize = AtomicUsize::new(0);

    fn frame(values: &[f32], path: &str) -> Frame {
        Frame {
            values: values.to_vec(),
            path: path.into(),
        }
    }

    fn fits_card(key: &str, value: &str) -> [u8; 80] {
        let mut card = [b' '; 80];
        let text = format!("{key:<8}= {value:>20}");
        card[..text.len()].copy_from_slice(text.as_bytes());
        card
    }

    fn write_test_fits(path: &Path, width: usize, height: usize, values: &[i16]) {
        assert_eq!(values.len(), width * height);
        let mut bytes = Vec::new();
        for (key, value) in [
            ("SIMPLE", "T".to_owned()),
            ("BITPIX", "16".to_owned()),
            ("NAXIS", "2".to_owned()),
            ("NAXIS1", width.to_string()),
            ("NAXIS2", height.to_string()),
        ] {
            bytes.extend_from_slice(&fits_card(key, &value));
        }
        let mut end = [b' '; 80];
        end[..3].copy_from_slice(b"END");
        bytes.extend_from_slice(&end);
        bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
        bytes.extend(values.iter().flat_map(|value| value.to_be_bytes()));
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn union_footprint_and_median_values_match_python_fixture() {
        let result = build_with_shape(
            vec![
                frame(
                    &[
                        f32::NAN,
                        f32::NAN,
                        f32::NAN,
                        f32::NAN,
                        10.0,
                        10.0,
                        10.0,
                        10.0,
                        10.0,
                        10.0,
                        10.0,
                        10.0,
                    ],
                    "a.fits",
                ),
                frame(
                    &[
                        f32::NAN,
                        f32::NAN,
                        f32::NAN,
                        f32::NAN,
                        20.0,
                        20.0,
                        20.0,
                        20.0,
                        20.0,
                        20.0,
                        20.0,
                        20.0,
                    ],
                    "b.fits",
                ),
            ],
            3,
            4,
            vec![[0.0, 0.0], [1.0, 0.0]],
            "median",
            3.0,
            1,
        )
        .unwrap();
        assert_eq!((result.output_height, result.output_width), (3, 5));
        assert_eq!((result.origin_x, result.origin_y), (-1.0, 0.0));
        assert_eq!(&result.coverage[5..10], &[1, 2, 2, 2, 1]);
        assert_eq!(&result.image[5..10], &[20.0, 15.0, 15.0, 15.0, 10.0]);
        let summary = metadata(&result);
        assert_eq!(summary["overlap_pixel_count"], 6);
        assert_eq!(summary["covered_bbox_xy"], json!([0, 1, 4, 2]));
        assert_eq!(summary["uncovered_pixel_count"], 5);
    }

    #[test]
    fn masked_neighbors_are_renormalized_and_zero_coverage_stays_nan() {
        let result = build_with_shape(
            vec![frame(&[10.0, f32::NAN, 30.0, 40.0], "a.fits")],
            2,
            2,
            vec![[0.0, 0.0]],
            "median",
            3.0,
            1,
        )
        .unwrap();
        assert_eq!(result.coverage, vec![1, 0, 1, 1]);
        assert!(result.image[1].is_nan());
        assert_eq!(result.image[0], 10.0);
    }

    #[test]
    fn scipy_spline_orders_zero_through_three_are_available() {
        let values: Vec<_> = (0..4)
            .flat_map(|y| (0..4).map(move |x| ((y * 10 + x) as f32).powi(2)))
            .collect();
        let input = frame(&values, "spline.fits");
        let sampled: Vec<_> = (0..=3)
            .map(|order| sample(&input, 4, 4, 1.3, 1.7, order).unwrap())
            .collect();

        assert_eq!(sampled[0], 441.0);
        assert!((sampled[1] - 356.1).abs() < 1e-3);
        assert!(sampled.iter().all(|value| value.is_finite()));
        assert_ne!(sampled[2], sampled[1]);
        assert_ne!(sampled[3], sampled[1]);

        let mut args = vec!["frames".into(), "--interpolation-order".into(), "3".into()];
        args.extend(["--sequence-json".into(), "sequence.json".into()]);
        assert_eq!(parse_args(&args).unwrap().interpolation_order, 3);
        args[2] = "4".into();
        assert_eq!(
            parse_args(&args).unwrap_err(),
            "interpolation_order must be between 0 and 3"
        );
    }

    #[test]
    fn robust_mean_rejects_single_frame_outlier_when_mad_is_nonzero() {
        let result = build_with_shape(
            vec![
                frame(&[10.0], "a.fits"),
                frame(&[11.0], "b.fits"),
                frame(&[100.0], "c.fits"),
            ],
            1,
            1,
            vec![[0.0, 0.0]; 3],
            "robust_mean",
            1.0,
            1,
        )
        .unwrap();
        assert_eq!(result.image, vec![10.5]);
        assert_eq!(result.coverage, vec![3]);
    }

    #[test]
    fn desktop_args_require_a_sequence_json_argument() {
        assert_eq!(
            run_desktop(&[]).unwrap_err(),
            "a FITS directory is required"
        );
        let parsed = parse_args(&["frames".into()]).unwrap();
        assert!(parsed.sequence_json.is_none());
        let parsed = parse_args(&[
            "frames".into(),
            "--sequence-json".into(),
            "sequence.json".into(),
            "--raw-preview".into(),
        ])
        .unwrap();
        assert!(parsed.raw_preview);
    }

    #[test]
    fn desktop_mosaic_reads_fits_and_writes_numpy_png_and_metadata_artifacts() {
        let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rst19-mosaic-native-{}-{sequence}",
            std::process::id()
        ));
        let frames = root.join("frames");
        let output = root.join("output");
        fs::create_dir_all(&frames).unwrap();
        write_test_fits(&frames.join("frame-1.fits"), 4, 3, &[10; 12]);
        write_test_fits(&frames.join("frame-2.fits"), 4, 3, &[20; 12]);
        let sequence_json = root.join("sequence.json");
        fs::write(&sequence_json, r#"{"cumulative_shifts":[[0,0],[1,0]]}"#).unwrap();
        let result = run_desktop(&[
            frames.to_string_lossy().into_owned(),
            "--sequence-json".into(),
            sequence_json.to_string_lossy().into_owned(),
            "--out-dir".into(),
            output.to_string_lossy().into_owned(),
            "--mode".into(),
            "median".into(),
        ])
        .unwrap();
        assert_eq!(result["result"]["output_shape"], json!([3, 5]));
        assert_eq!(result["result"]["overlap_pixel_count"], 6);
        for name in [
            "registered_mosaic_adu.npy",
            "registered_mosaic_coverage.npy",
            "registered_mosaic_scatter.npy",
            "registered_mosaic_footprint.png",
            "registered_mosaic_preview.png",
            "registered_mosaic_metadata.json",
        ] {
            assert!(output.join(name).is_file(), "missing {name}");
        }
        let numpy = fs::read(output.join("registered_mosaic_adu.npy")).unwrap();
        assert_eq!(&numpy[..8], b"\x93NUMPY\x01\x00");
        let header_len = u16::from_le_bytes([numpy[8], numpy[9]]) as usize;
        assert!(String::from_utf8_lossy(&numpy[10..10 + header_len]).contains("(3, 5)"));
        let data_start = 10 + header_len;
        let row_one = (0..5)
            .map(|index| {
                let start = data_start + (5 + index) * 4;
                f32::from_le_bytes(numpy[start..start + 4].try_into().unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(row_one, [20.0, 15.0, 15.0, 15.0, 10.0]);
        let footprint = fs::read(output.join("registered_mosaic_footprint.png")).unwrap();
        assert_eq!(&footprint[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(footprint[25], 0, "footprint PNG must remain grayscale");
        let preview = fs::read(output.join("registered_mosaic_preview.png")).unwrap();
        assert_eq!(preview[25], 6, "preview PNG must preserve transparent RGBA");
        let _ = fs::remove_dir_all(root);
    }
}
