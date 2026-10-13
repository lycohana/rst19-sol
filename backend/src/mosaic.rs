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
            // SciPy's cardinal B-spline support moves when a fractional
            // coordinate crosses a half-pixel boundary. This matters for the
            // quadratic (odd-width) kernel; anchoring it at floor(x) shifts
            // one tap for coordinates such as 0.6.
            let first = (coordinate - count as f32 * 0.5).floor() as isize + 1;
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

fn mirror_index(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * (length as isize - 1);
    let reflected = index.rem_euclid(period);
    if reflected < length as isize {
        reflected as usize
    } else {
        (period - reflected) as usize
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
        if *wy == 0.0 {
            continue;
        }
        let py = if interpolation_order >= 2 {
            mirror_index(*py, height)
        } else if *py >= 0 && *py < height as isize {
            *py as usize
        } else {
            continue;
        };
        for (px, wx) in x_samples.iter().take(x_count) {
            let sample_weight = wx * wy;
            if sample_weight == 0.0 {
                continue;
            }
            let px = if interpolation_order >= 2 {
                mirror_index(*px, width)
            } else if *px >= 0 && *px < width as isize {
                *px as usize
            } else {
                continue;
            };
            let value = frame.values[py * width + px];
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

fn preview_dimensions(width: usize, height: usize, max_side: usize) -> (usize, usize) {
    let limit = max_side.max(64);
    let scale = (limit as f64 / width as f64)
        .min(limit as f64 / height as f64)
        .min(1.0);
    (
        ((width as f64 * scale).round() as usize).max(1),
        ((height as f64 * scale).round() as usize).max(1),
    )
}

fn lanczos3(distance: f64) -> f64 {
    let x = distance.abs();
    if x == 0.0 {
        return 1.0;
    }
    if x >= 3.0 {
        return 0.0;
    }
    let pix = std::f64::consts::PI * x;
    pix.sin() * (pix / 3.0).sin() / (pix * pix / 3.0)
}

const LANCZOS_PRECISION_BITS: u32 = 22;

fn resize_contributions(source_size: usize, target_size: usize) -> Vec<Vec<(usize, i64)>> {
    let scale = source_size as f64 / target_size as f64;
    let filter_scale = scale.max(1.0);
    let support = 3.0 * filter_scale;
    (0..target_size)
        .map(|target| {
            let center = (target as f64 + 0.5) * scale;
            let start = (center - support + 0.5).floor() as isize;
            let stop = (center + support + 0.5).floor() as isize;
            let mut values = (start.max(0)..stop.min(source_size as isize))
                .map(|source| {
                    (
                        source as usize,
                        lanczos3((source as f64 + 0.5 - center) / filter_scale),
                    )
                })
                .filter(|(_, weight)| *weight != 0.0)
                .collect::<Vec<_>>();
            let sum = values.iter().map(|(_, weight)| weight).sum::<f64>();
            if sum.abs() > f64::EPSILON {
                for (_, weight) in &mut values {
                    *weight /= sum;
                }
            }
            let coefficient_scale = (1_u64 << LANCZOS_PRECISION_BITS) as f64;
            values
                .into_iter()
                .map(|(source, weight)| {
                    let scaled = weight * coefficient_scale;
                    let fixed = if scaled < 0.0 {
                        (scaled - 0.5) as i64
                    } else {
                        (scaled + 0.5) as i64
                    };
                    (source, fixed)
                })
                .collect()
        })
        .collect()
}

fn resize_rgba_axis(
    input: &[u8],
    width: usize,
    height: usize,
    target_size: usize,
    horizontal: bool,
) -> Result<Vec<u8>> {
    let source_size = if horizontal { width } else { height };
    let contributions = resize_contributions(source_size, target_size);
    let output_width = if horizontal { target_size } else { width };
    let output_height = if horizontal { height } else { target_size };
    let output_len = output_width
        .checked_mul(output_height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or("preview dimensions overflow")?;
    let mut output = vec![0u8; output_len];
    for y in 0..output_height {
        for x in 0..output_width {
            let target = if horizontal { x } else { y };
            let output_offset = (y * output_width + x) * 4;
            for channel in 0..4 {
                let mut value = 1_i64 << (LANCZOS_PRECISION_BITS - 1);
                for (source, weight) in &contributions[target] {
                    let (source_x, source_y) = if horizontal {
                        (*source, y)
                    } else {
                        (x, *source)
                    };
                    value += input[(source_y * width + source_x) * 4 + channel] as i64 * weight;
                }
                output[output_offset + channel] =
                    (value >> LANCZOS_PRECISION_BITS).clamp(0, 255) as u8;
            }
        }
    }
    Ok(output)
}

fn resize_rgba_lanczos3(
    rgba: &[u8],
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
) -> Result<Vec<u8>> {
    if width == 0
        || height == 0
        || target_width == 0
        || target_height == 0
        || rgba.len() != width.saturating_mul(height).saturating_mul(4)
    {
        return Err("invalid RGBA preview dimensions".into());
    }
    // Pillow resamples RGBA through premultiplied-alpha RGBa. This prevents
    // invisible RGB values outside the mosaic footprint from bleeding into
    // visible edge pixels.
    let mut premultiplied = rgba.to_vec();
    for pixel in premultiplied.chunks_exact_mut(4) {
        let alpha = u16::from(pixel[3]);
        for channel in &mut pixel[..3] {
            *channel = ((u16::from(*channel) * alpha + 127) / 255) as u8;
        }
    }

    // Pillow 12.2 applies the horizontal pass before the vertical pass for
    // all two-axis resizes; the 8-bit intermediate rounding makes order
    // observable for RGBA previews.
    let horizontal = if target_width == width {
        premultiplied
    } else {
        resize_rgba_axis(&premultiplied, width, height, target_width, true)?
    };
    let resized = if target_height == height {
        horizontal
    } else {
        resize_rgba_axis(&horizontal, target_width, height, target_height, false)?
    };

    let mut output = resized;
    for pixel in output.chunks_exact_mut(4) {
        let alpha = u32::from(pixel[3]);
        if alpha == 0 {
            pixel[..3].fill(0);
        } else {
            for channel in &mut pixel[..3] {
                *channel = ((u32::from(*channel) * 255) / alpha).min(255) as u8;
            }
        }
    }
    Ok(output)
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
    let (preview_width, preview_height) =
        preview_dimensions(mosaic.output_width, mosaic.output_height, 4096);
    if preview_width == mosaic.output_width && preview_height == mosaic.output_height {
        write_png_rgba(
            &out_dir.join("registered_mosaic_preview.png"),
            preview_width,
            preview_height,
            &rgba,
        )?;
    } else {
        let preview = resize_rgba_lanczos3(
            &rgba,
            mosaic.output_width,
            mosaic.output_height,
            preview_width,
            preview_height,
        )?;
        write_png_rgba(
            &out_dir.join("registered_mosaic_preview.png"),
            preview_width,
            preview_height,
            &preview,
        )?;
    }
    if let Some(raw_rgba) = raw_rgba {
        if preview_width == mosaic.output_width && preview_height == mosaic.output_height {
            write_png_rgba(
                &out_dir.join("registered_mosaic_raw_preview.png"),
                preview_width,
                preview_height,
                &raw_rgba,
            )?;
        } else {
            let preview = resize_rgba_lanczos3(
                &raw_rgba,
                mosaic.output_width,
                mosaic.output_height,
                preview_width,
                preview_height,
            )?;
            write_png_rgba(
                &out_dir.join("registered_mosaic_raw_preview.png"),
                preview_width,
                preview_height,
                &preview,
            )?;
        }
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
    fn all_interpolation_orders_and_combine_modes_match_frozen_python_arrays() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/mosaic_parity.json")).unwrap();
        let frame_rows = fixture["frames"].as_array().unwrap();
        let source_height = frame_rows[0].as_array().unwrap().len();
        let source_width = frame_rows[0].as_array().unwrap()[0]
            .as_array()
            .unwrap()
            .len();
        let frames: Vec<Vec<f32>> = frame_rows
            .iter()
            .map(|frame| {
                frame
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| row.as_array().unwrap())
                    .map(|value| value.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect();
        let shifts: Vec<[f64; 2]> = fixture["shifts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|shift| {
                let values = shift.as_array().unwrap();
                [values[0].as_f64().unwrap(), values[1].as_f64().unwrap()]
            })
            .collect();

        for case in fixture["cases"].as_array().unwrap() {
            let mode = case["mode"].as_str().unwrap();
            let interpolation_order = case["interpolation_order"].as_u64().unwrap() as usize;
            let actual = build_with_shape(
                frames.iter().map(|values| frame(values, "")).collect(),
                source_height,
                source_width,
                shifts.clone(),
                mode,
                fixture["clip_sigma"].as_f64().unwrap(),
                interpolation_order,
            )
            .unwrap();

            let expected_image: Vec<f32> = case["image"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap())
                .map(|value| value.as_f64().map_or(f32::NAN, |value| value as f32))
                .collect();
            let expected_coverage: Vec<u16> = case["coverage"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap())
                .map(|value| value.as_u64().unwrap() as u16)
                .collect();
            let expected_scatter: Vec<f32> = case["scatter"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap())
                .map(|value| value.as_f64().map_or(f32::NAN, |value| value as f32))
                .collect();

            assert_eq!(
                actual.coverage, expected_coverage,
                "{mode}, order {interpolation_order}"
            );
            assert_eq!(actual.image.len(), expected_image.len());
            for (index, (actual, expected)) in actual.image.iter().zip(expected_image).enumerate() {
                if expected.is_nan() {
                    assert!(
                        actual.is_nan(),
                        "{mode}, order {interpolation_order}, image {index}"
                    );
                } else {
                    let tolerance = 2.0e-5 + 2.0e-6 * expected.abs();
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{mode}, order {interpolation_order}, image {index}: Rust={actual}, Python={expected}"
                    );
                }
            }
            for (index, (actual, expected)) in
                actual.scatter.iter().zip(expected_scatter).enumerate()
            {
                if expected.is_nan() {
                    assert!(
                        actual.is_nan(),
                        "{mode}, order {interpolation_order}, scatter {index}"
                    );
                } else {
                    let tolerance = 2.0e-5 + 2.0e-6 * expected.abs();
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{mode}, order {interpolation_order}, scatter {index}: Rust={actual}, Python={expected}"
                    );
                }
            }
        }
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
    fn lanczos_preview_resize_preserves_constant_rgba_and_aspect_ratio() {
        assert_eq!(preview_dimensions(8192, 4096, 4096), (4096, 2048));
        assert_eq!(preview_dimensions(320, 240, 12), (64, 48));

        let source = vec![48, 96, 144, 255].repeat(5 * 3);
        let resized = resize_rgba_lanczos3(&source, 5, 3, 3, 2).unwrap();
        assert_eq!(resized.len(), 3 * 2 * 4);
        assert!(resized
            .chunks_exact(4)
            .all(|pixel| pixel == [48, 96, 144, 255]));
        assert_eq!(
            resize_rgba_lanczos3(&source, 5, 3, 0, 2).unwrap_err(),
            "invalid RGBA preview dimensions"
        );
    }

    #[test]
    fn lanczos_preview_downscale_matches_frozen_pillow_rgba_pixels() {
        // Frozen from Pillow 12.2.0 Image.Resampling.LANCZOS on a 7x5 RGBA
        // image. Preview pixels are part of the Python desktop output contract.
        let source = (0..5)
            .flat_map(|y| {
                (0..7).flat_map(move |x| {
                    (0..4).map(move |channel| (x * 31 + y * 7 + channel * 53) as u8)
                })
            })
            .collect::<Vec<_>>();
        let resized = resize_rgba_lanczos3(&source, 7, 5, 4, 3).unwrap();
        assert_eq!(
            resized,
            [
                15, 69, 121, 177, 67, 120, 175, 208, 117, 167, 188, 61, 183, 247, 17, 71, 32, 85,
                138, 202, 69, 121, 180, 161, 236, 255, 200, 14, 183, 204, 47, 96, 44, 96, 149, 212,
                83, 134, 195, 180, 185, 255, 77, 33, 197, 91, 52, 106,
            ]
        );

        // The tall anisotropic shape catches Pillow's fixed horizontal-then-
        // vertical 8-bit intermediate rounding contract.
        let tall_source = (0..9)
            .flat_map(|y| {
                (0..3).flat_map(move |x| {
                    (0..4).map(move |channel| (x * 47 + y * 23 + channel * 61) as u8)
                })
            })
            .collect::<Vec<_>>();
        let vertical_only = resize_rgba_lanczos3(&tall_source, 3, 9, 3, 3).unwrap();
        assert_eq!(
            vertical_only,
            [
                30, 91, 153, 224, 59, 118, 192, 162, 106, 243, 112, 43, 74, 134, 187, 106, 160,
                255, 37, 54, 207, 142, 49, 113, 185, 247, 3, 70, 205, 92, 70, 138, 171, 51, 121,
                183,
            ]
        );
        let horizontal_only = resize_rgba_lanczos3(&tall_source, 3, 9, 2, 9).unwrap();
        assert_eq!(
            horizontal_only,
            [
                20, 81, 142, 214, 60, 120, 181, 97, 43, 104, 164, 237, 87, 148, 208, 120, 45, 106,
                176, 153, 170, 233, 0, 36, 70, 131, 202, 176, 177, 237, 43, 59, 111, 167, 71, 32,
                173, 234, 35, 100, 129, 236, 125, 55, 196, 72, 58, 123, 150, 251, 16, 78, 220, 99,
                85, 146, 174, 126, 40, 101, 242, 37, 108, 169, 228, 152, 63, 124, 87, 59, 131, 192,
            ]
        );
        let tall_resized = resize_rgba_lanczos3(&tall_source, 3, 9, 2, 3).unwrap();
        assert_eq!(
            tall_resized,
            [
                38, 97, 166, 211, 79, 165, 172, 80, 86, 168, 162, 83, 206, 171, 37, 89, 201, 178,
                28, 90, 183, 59, 108, 172,
            ]
        );
    }

    #[test]
    fn large_preview_artifacts_are_downsampled_without_resizing_science_arrays() {
        let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
        let output = std::env::temp_dir().join(format!(
            "rst19-mosaic-large-preview-{}-{sequence}",
            std::process::id()
        ));
        let width = 4097;
        let mosaic = Mosaic {
            image: vec![32.0; width],
            coverage: vec![1; width],
            scatter: vec![0.0; width],
            source_height: 1,
            source_width: width,
            output_height: 1,
            output_width: width,
            origin_x: 0.0,
            origin_y: 0.0,
            frame_origins: vec![[0.0, 0.0]],
            shifts: vec![[0.0, 0.0]],
            frame_paths: vec!["synthetic.fits".into()],
            mode: "median".into(),
            clip_sigma: 3.0,
            interpolation_order: 1,
            frame_valid_counts: vec![width as u64],
        };

        write_artifacts(&output, &mosaic, &metadata(&mosaic), true).unwrap();
        let preview = fs::read(output.join("registered_mosaic_preview.png")).unwrap();
        let raw = fs::read(output.join("registered_mosaic_raw_preview.png")).unwrap();
        assert_eq!(
            u32::from_be_bytes(preview[16..20].try_into().unwrap()),
            4096
        );
        assert_eq!(u32::from_be_bytes(preview[20..24].try_into().unwrap()), 1);
        assert_eq!(u32::from_be_bytes(raw[16..20].try_into().unwrap()), 4096);
        assert_eq!(
            mosaic.image.len(),
            width,
            "science array dimensions stay intact"
        );
        fs::remove_dir_all(output).unwrap();
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
