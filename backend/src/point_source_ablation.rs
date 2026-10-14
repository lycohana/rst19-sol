//! Shared-input point-source ablation, implemented without a Python runtime.
//!
//! A, B and C consume the same frozen float32 control and injection images.
//! C filters the exact B detections through their existing quality flag; it
//! does not run a second detector or reinterpret the injected truth as stars.

use crate::{
    detection::{self, DetectionOptions, DetectionResult},
    fits,
    measurement::{Detection, MeasurementOptions},
    models::PixelData,
    source_geometry_audit::NumpyPcg64,
    Result,
};
use csv::{Terminator, WriterBuilder};
use png::{BitDepth, ColorType, Encoder};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const SUMMARY_COLUMNS: &[&str] = &[
    "gaussian_center_amplitude_adu",
    "version",
    "matched_targets",
    "target_total",
    "recall",
    "wilson95_low_descriptive",
    "wilson95_high_descriptive",
    "unmatched_targets",
    "median_position_error_px",
    "p90_position_error_px",
    "extra_candidates_within_match_radius",
    "full_frame_candidate_count",
    "full_frame_quality_count",
    "detector_seconds",
];
const CONTROL_COLUMNS: &[&str] = &[
    "version",
    "input",
    "candidate_count",
    "returned_count",
    "quality_count",
    "output_count",
    "target_site_candidates_within_match_radius",
    "detector_seconds",
    "reused_version",
];
const TARGET_COLUMNS: &[&str] = &[
    "target_id",
    "x_px",
    "y_px",
    "psf_fwhm_px",
    "local_background_adu",
    "local_noise_adu",
    "control_candidate_within_match_radius",
    "truth_type",
    "gaussian_center_amplitude_adu",
    "sampled_peak_pixel_increment_adu",
    "sampled_peak_fraction_of_center_amplitude",
    "control_candidates_within_match_radius",
    "control_gaussian_candidates_within_match_radius",
    "control_hybrid_candidates_within_match_radius",
    "control_quality_candidates_within_match_radius",
    "gaussian_candidate_hit",
    "gaussian_nearby_candidate_count",
    "hybrid_candidate_hit",
    "hybrid_nearby_candidate_count",
    "hybrid_matched_candidate_quality_passed",
    "quality_gate_hit",
    "quality_gate_loss_from_hybrid_candidate",
    "quality_matching_assignment_changed",
    "quality_rejection_flags",
    "hybrid_candidate_status",
    "quality_gate_status",
    "gaussian_detection_id",
    "gaussian_x_px",
    "gaussian_y_px",
    "gaussian_error_px",
    "gaussian_quality_passed",
    "gaussian_flags",
    "hybrid_detection_id",
    "hybrid_x_px",
    "hybrid_y_px",
    "hybrid_error_px",
    "hybrid_quality_passed",
    "hybrid_flags",
    "quality_detection_id",
    "quality_x_px",
    "quality_y_px",
    "quality_error_px",
    "quality_quality_passed",
    "quality_flags",
    "frozen_injection_image",
    "frozen_injection_image_sha256",
];

#[derive(Clone, Debug)]
struct Options {
    peak_levels: Vec<f64>,
    target_count: usize,
    seed: u64,
    psf_fwhm: f64,
    injected_psf_fwhm: Option<f64>,
    threshold_sigma: f64,
    min_distance: usize,
    aperture_radius: usize,
    background_box_size: usize,
    min_flux_snr: f64,
    min_psf_support_pixels: usize,
    match_radius: Option<f64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            peak_levels: vec![12.0, 24.0, 56.0],
            target_count: 8,
            seed: 19019,
            psf_fwhm: 2.0,
            injected_psf_fwhm: None,
            threshold_sigma: 4.0,
            min_distance: 4,
            aperture_radius: 4,
            background_box_size: 128,
            min_flux_snr: 5.0,
            min_psf_support_pixels: 3,
            match_radius: None,
        }
    }
}

#[derive(Clone, Debug)]
struct Target {
    id: String,
    x: f64,
    y: f64,
    local_background: f64,
    local_noise: f64,
}

#[derive(Default)]
struct CliArgs {
    fits_path: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    report: Option<PathBuf>,
    reuse_manifest: Option<PathBuf>,
    options: Options,
    argv: Vec<String>,
}

fn finite_value(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

fn sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256(&fs::read(path).map_err(|error| error.to_string())?))
}

fn array_hash_f32(values: &[f32], height: usize, width: usize) -> String {
    let mut digest = Sha256::new();
    digest.update(b"<f4");
    digest.update(format!("[{height},{width}]").as_bytes());
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn array_hash_pixels(data: &PixelData, height: usize, width: usize) -> String {
    let mut digest = Sha256::new();
    digest.update(data.dtype().as_bytes());
    digest.update(format!("[{height},{width}]").as_bytes());
    match data {
        PixelData::U8(values) => digest.update(values),
        PixelData::I16(values) => values
            .iter()
            .for_each(|value| digest.update(value.to_be_bytes())),
        PixelData::I32(values) => values
            .iter()
            .for_each(|value| digest.update(value.to_be_bytes())),
        PixelData::I64(values) => values
            .iter()
            .for_each(|value| digest.update(value.to_be_bytes())),
        PixelData::F32(values) => values
            .iter()
            .for_each(|value| digest.update(value.to_be_bytes())),
        PixelData::F64(values) => values
            .iter()
            .for_each(|value| digest.update(value.to_be_bytes())),
    }
    format!("{:x}", digest.finalize())
}

fn display_path(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn compact_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1.0e15 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn npy_header(height: usize, width: usize) -> Vec<u8> {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({height}, {width}), }}");
    let padding = (64 - ((10 + header.len() + 1) % 64)) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    header.into_bytes()
}

fn write_npy_f32(path: &Path, values: &[f32], height: usize, width: usize) -> Result<()> {
    if height.checked_mul(width) != Some(values.len()) {
        return Err("float32 image shape does not match its values".into());
    }
    let mut writer = BufWriter::new(File::create(path).map_err(|error| error.to_string())?);
    let header = npy_header(height, width);
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

fn write_csv(path: &Path, columns: &[&str], rows: &[Value]) -> Result<()> {
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .terminator(Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(columns)
        .map_err(|error| error.to_string())?;
    for row in rows {
        let fields = columns
            .iter()
            .map(|column| csv_value(&row[*column]))
            .collect::<Vec<_>>();
        writer
            .write_record(fields)
            .map_err(|error| error.to_string())?;
    }
    let mut bytes = vec![0xef, 0xbb, 0xbf];
    bytes.extend(writer.into_inner().map_err(|error| error.to_string())?);
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn csv_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn detector_options(options: &Options, proposal_mode: &str) -> DetectionOptions {
    let mut detector = DetectionOptions::default();
    detector.measurement = MeasurementOptions {
        aperture_radius: options.aperture_radius,
        min_flux_snr: options.min_flux_snr,
        min_psf_support_pixels: options.min_psf_support_pixels,
        psf_fwhm: options.psf_fwhm,
        ..MeasurementOptions::default()
    };
    detector.threshold_sigma = options.threshold_sigma;
    detector.min_distance = options.min_distance;
    detector.background_box_size = options.background_box_size;
    detector.reject_linear_artifacts = true;
    detector.proposal_mode = proposal_mode.to_owned();
    detector.use_float32 = false;
    detector
}

fn detect(
    image: &[f32],
    height: usize,
    width: usize,
    mask: &[bool],
    options: &Options,
    mode: &str,
) -> Result<DetectionResult> {
    let data = PixelData::F32(image.to_vec());
    detection::detect_sources(
        &data,
        Some(mask),
        height,
        width,
        &detector_options(options, mode),
        None,
    )
}

fn inject_gaussian(
    image: &mut [f32],
    height: usize,
    width: usize,
    x: f64,
    y: f64,
    amplitude: f64,
    fwhm: f64,
) -> f64 {
    let sigma = (fwhm / 2.354820045).max(f64::EPSILON);
    let radius = 4usize.max((4.0 * sigma).ceil() as usize);
    let x0 = (x.floor() as isize - radius as isize).max(0) as usize;
    let y0 = (y.floor() as isize - radius as isize).max(0) as usize;
    let x1 = (x.floor() as isize + radius as isize + 1)
        .min(width as isize)
        .max(0) as usize;
    let y1 = (y.floor() as isize + radius as isize + 1)
        .min(height as isize)
        .max(0) as usize;
    let denom = 2.0 * sigma * sigma;
    let mut peak_fraction = 0.0_f64;
    for py in y0..y1 {
        for px in x0..x1 {
            let distance2 = (px as f64 - x).powi(2) + (py as f64 - y).powi(2);
            let kernel = (-distance2 / denom).exp();
            peak_fraction = peak_fraction.max(kernel);
            let index = py * width + px;
            image[index] = (f64::from(image[index]) + amplitude * kernel) as f32;
        }
    }
    peak_fraction
}

fn target_json(target: &Target, fwhm: f64) -> Value {
    json!({
        "target_id":target.id,
        "x_px":target.x,
        "y_px":target.y,
        "psf_fwhm_px":fwhm,
        "local_background_adu":target.local_background,
        "local_noise_adu":target.local_noise,
        "control_candidate_within_match_radius":false,
        "truth_type":"synthetic Gaussian point-source injection"
    })
}

fn manifest_targets(
    path: &Path,
    source_hash: &str,
    count: usize,
    fwhm: f64,
    height: usize,
    width: usize,
) -> Result<Vec<Target>> {
    let bytes = fs::read(path)
        .map_err(|error| format!("could not read frozen target manifest: {error}"))?;
    let root: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid target manifest JSON: {error}"))?;
    if root["source_fits"]["sha256"].as_str() != Some(source_hash) {
        return Err("fixed target manifest input SHA-256 does not match the current FITS".into());
    }
    let raw = root["targets"]
        .as_array()
        .ok_or("fixed target manifest is missing targets")?;
    if raw.len() != count {
        return Err("fixed target manifest target count differs from --targets".into());
    }
    let mut ids = HashSet::new();
    let mut targets = Vec::with_capacity(count);
    for item in raw {
        let id = item["target_id"]
            .as_str()
            .ok_or("fixed target is missing target_id")?
            .to_owned();
        let x = item["x_px"]
            .as_f64()
            .ok_or("fixed target x_px must be finite")?;
        let y = item["y_px"]
            .as_f64()
            .ok_or("fixed target y_px must be finite")?;
        let background = item["local_background_adu"]
            .as_f64()
            .ok_or("fixed target local background is missing")?;
        let noise = item["local_noise_adu"]
            .as_f64()
            .ok_or("fixed target local noise is missing")?;
        if !x.is_finite()
            || !y.is_finite()
            || !background.is_finite()
            || !noise.is_finite()
            || x < 0.0
            || y < 0.0
            || x >= width as f64
            || y >= height as f64
        {
            return Err("fixed target manifest contains an invalid or out-of-bounds target".into());
        }
        if !ids.insert(id.clone()) {
            return Err("fixed target manifest contains duplicate target IDs".into());
        }
        targets.push(Target {
            id,
            x,
            y,
            local_background: background,
            local_noise: noise,
        });
    }
    let _ = fwhm;
    Ok(targets)
}

fn quantile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = (sorted.len() - 1) as f64 * q;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    Some(sorted[low] + (sorted[high] - sorted[low]) * (index - low as f64))
}

fn local_patch_stats(
    image: &[f32],
    height: usize,
    width: usize,
    x: f64,
    y: f64,
) -> Option<(f64, f64, f64)> {
    let cx = x.round_ties_even() as isize;
    let cy = y.round_ties_even() as isize;
    let x0 = (cx - 5).max(0) as usize;
    let y0 = (cy - 5).max(0) as usize;
    let x1 = (cx + 6).min(width as isize).max(0) as usize;
    let y1 = (cy + 6).min(height as isize).max(0) as usize;
    let values = (y0..y1)
        .flat_map(|row| (x0..x1).map(move |column| image[row * width + column] as f64))
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.len() < 16 {
        return None;
    }
    let median = quantile(&values, 0.5)?;
    let deviations = values
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    let noise = (1.4826 * quantile(&deviations, 0.5)?).max(f64::EPSILON);
    let upper_excess = quantile(&values, 0.98)? - median;
    Some((median, noise, upper_excess))
}

fn select_targets(
    image: &[f32],
    height: usize,
    width: usize,
    existing: &[Detection],
    noise: f64,
    options: &Options,
    radius: f64,
) -> Result<Vec<Target>> {
    let margin = 14usize;
    if height <= 2 * margin || width <= 2 * margin {
        return Err("image is too small for real-background injection margins".into());
    }
    let exclusion = (12.0_f64).max(2.0 * options.psf_fwhm + radius);
    let mut rng = NumpyPcg64::new(u128::from(options.seed));
    let mut targets: Vec<Target> = Vec::with_capacity(options.target_count);
    for _ in 0..5000usize.max(options.target_count * 1500) {
        if targets.len() == options.target_count {
            break;
        }
        let x = rng.uniform(margin as f64, (width - margin) as f64);
        let y = rng.uniform(margin as f64, (height - margin) as f64);
        if targets
            .iter()
            .any(|other| (other.x - x).hypot(other.y - y) < exclusion)
            || existing
                .iter()
                .any(|source| (source.x - x).hypot(source.y - y) < exclusion)
        {
            continue;
        }
        let Some((background, local_noise, upper_excess)) =
            local_patch_stats(image, height, width, x, y)
        else {
            continue;
        };
        if upper_excess > (3.0 * local_noise).max(3.0 * noise).max(f64::EPSILON) {
            continue;
        }
        let index = targets.len() + 1;
        targets.push(Target {
            id: format!("PSF-{index:03}"),
            x,
            y,
            local_background: background,
            local_noise,
        });
    }
    if targets.len() != options.target_count {
        return Err(format!("无法在真实背景中找到 {} 个合规注入位置，仅找到 {} 个；请减少每次注入数量或先检查图像拥挤/条纹区域", options.target_count, targets.len()));
    }
    Ok(targets)
}

fn match_detections(
    targets: &[Target],
    detections: &[Detection],
    radius: f64,
) -> Result<(BTreeMap<String, Detection>, BTreeMap<String, usize>)> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err("match radius must be positive".into());
    }
    let mut edges: Vec<(f64, String, usize)> = Vec::new();
    let mut nearby = BTreeMap::new();
    for target in targets {
        let mut count = 0;
        for (index, detection) in detections.iter().enumerate() {
            let distance = (detection.x - target.x).hypot(detection.y - target.y);
            if distance <= radius {
                count += 1;
                edges.push((distance, target.id.clone(), index));
            }
        }
        nearby.insert(target.id.clone(), count);
    }
    edges.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then(left.1.cmp(&right.1))
            .then(left.2.cmp(&right.2))
    });
    let mut used_targets = HashSet::new();
    let mut used_detections = HashSet::new();
    let mut matches = BTreeMap::new();
    for (_, target_id, index) in edges {
        if used_targets.contains(&target_id) || used_detections.contains(&index) {
            continue;
        }
        used_targets.insert(target_id.clone());
        used_detections.insert(index);
        matches.insert(target_id, detections[index].clone());
    }
    Ok((matches, nearby))
}

fn wilson(successes: usize, total: usize) -> (Value, Value) {
    if total == 0 {
        return (Value::Null, Value::Null);
    }
    let z = 1.959963984540054_f64;
    let p = successes as f64 / total as f64;
    let denominator = 1.0 + z * z / total as f64;
    let center = (p + z * z / (2.0 * total as f64)) / denominator;
    let half = z * ((p * (1.0 - p) / total as f64 + z * z / (4.0 * (total * total) as f64)).sqrt())
        / denominator;
    (
        json!((center - half).max(0.0)),
        json!((center + half).min(1.0)),
    )
}

fn summary_row(
    level: f64,
    version: &str,
    matches: &BTreeMap<String, Detection>,
    targets: &[Target],
    nearby: &BTreeMap<String, usize>,
    seconds: f64,
    candidate_count: usize,
    quality_count: usize,
) -> Value {
    let errors = targets
        .iter()
        .filter_map(|target| {
            matches
                .get(&target.id)
                .map(|source| (source.x - target.x).hypot(source.y - target.y))
        })
        .collect::<Vec<_>>();
    let (low, high) = wilson(matches.len(), targets.len());
    let duplicates = nearby
        .values()
        .map(|count| count.saturating_sub(1))
        .sum::<usize>();
    json!({
        "gaussian_center_amplitude_adu":level,"version":version,"matched_targets":matches.len(),"target_total":targets.len(),
        "recall":if targets.is_empty(){Value::Null}else{json!(matches.len() as f64/targets.len() as f64)},
        "wilson95_low_descriptive":low,"wilson95_high_descriptive":high,"unmatched_targets":targets.len()-matches.len(),
        "median_position_error_px":quantile(&errors,0.5).map_or(Value::Null,|v|json!(v)),
        "p90_position_error_px":quantile(&errors,0.9).map_or(Value::Null,|v|json!(v)),
        "extra_candidates_within_match_radius":duplicates,"full_frame_candidate_count":candidate_count,
        "full_frame_quality_count":quality_count,"detector_seconds":seconds
    })
}

fn detection_record(
    detection: Option<&Detection>,
    x: f64,
    y: f64,
    prefix: &str,
    row: &mut Map<String, Value>,
) {
    let keys = [
        "detection_id",
        "x_px",
        "y_px",
        "error_px",
        "quality_passed",
        "flags",
    ];
    let values = if let Some(source) = detection {
        [
            json!(source.detection_id),
            finite_value(source.x),
            finite_value(source.y),
            finite_value((source.x - x).hypot(source.y - y)),
            json!(source.quality_passed),
            json!(source.flags.join("|")),
        ]
    } else {
        [
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            json!(""),
        ]
    };
    for (key, value) in keys.iter().zip(values) {
        row.insert(format!("{prefix}_{key}"), value);
    }
}

fn set_pixel(pixels: &mut [u8], width: usize, height: usize, x: i32, y: i32, color: [u8; 3]) {
    if x < 0 || y < 0 || x as usize >= width || y as usize >= height {
        return;
    }
    let index = (y as usize * width + x as usize) * 3;
    pixels[index..index + 3].copy_from_slice(&color);
}

fn circle(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    cx: i32,
    cy: i32,
    radius: i32,
    color: [u8; 3],
) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let d = dx * dx + dy * dy;
            if (d - radius * radius).abs() <= radius * 2 {
                set_pixel(pixels, width, height, cx + dx, cy + dy, color);
            }
        }
    }
}

fn write_case_sheet(
    path: &Path,
    image: &[f32],
    height: usize,
    width: usize,
    rows: &[Value],
    level: f64,
) -> Result<()> {
    let selected = rows
        .iter()
        .filter(|row| row["gaussian_center_amplitude_adu"].as_f64() == Some(level))
        .collect::<Vec<_>>();
    let tile = 124usize;
    let panel = 116usize;
    let columns = 4usize;
    let sheet_rows = selected.len().div_ceil(columns).max(1);
    let sheet_width = columns * tile;
    let sheet_height = sheet_rows * tile;
    let mut canvas = vec![0_u8; sheet_width * sheet_height * 3];
    for pixel in canvas.chunks_exact_mut(3) {
        pixel.copy_from_slice(&[24, 31, 42]);
    }
    for (index, row) in selected.iter().enumerate() {
        let Some(x) = row["x_px"].as_f64() else {
            continue;
        };
        let Some(y) = row["y_px"].as_f64() else {
            continue;
        };
        let cx = x.round_ties_even() as isize;
        let cy = y.round_ties_even() as isize;
        let x0 = (cx - 28).max(0) as usize;
        let y0 = (cy - 28).max(0) as usize;
        let x1 = (cx + 29).min(width as isize).max(0) as usize;
        let y1 = (cy + 29).min(height as isize).max(0) as usize;
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let patch = (y0..y1)
            .flat_map(|yy| (x0..x1).map(move |xx| image[yy * width + xx] as f64))
            .filter(|v| v.is_finite())
            .collect::<Vec<_>>();
        let lo = quantile(&patch, 0.05).unwrap_or(0.0);
        let hi = quantile(&patch, 0.997)
            .filter(|v| *v > lo)
            .unwrap_or(lo + 1.0);
        let pw = x1 - x0;
        let ph = y1 - y0;
        let scale = (108.0 / pw as f64).min(92.0 / ph as f64).min(1.0);
        let dw = (pw as f64 * scale).round().max(1.0) as usize;
        let dh = (ph as f64 * scale).round().max(1.0) as usize;
        let col = index % columns;
        let row_index = index / columns;
        let cell_x = col * tile;
        let cell_y = row_index * tile;
        let ox = cell_x + 4 + (panel - dw) / 2;
        let oy = cell_y + 4 + 2;
        for dy in 0..dh {
            for dx in 0..dw {
                let sx = x0 + ((dx as f64 / scale).floor() as usize).min(pw - 1);
                let sy = y0 + ((dy as f64 / scale).floor() as usize).min(ph - 1);
                let value = image[sy * width + sx] as f64;
                let gray = if value.is_finite() {
                    (((value - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0).round() as u8
                } else {
                    0
                };
                set_pixel(
                    &mut canvas,
                    sheet_width,
                    sheet_height,
                    (ox + dx) as i32,
                    (oy + dy) as i32,
                    [gray, gray, gray],
                );
            }
        }
        let marker = |px: Option<f64>, py: Option<f64>| -> Option<(i32, i32)> {
            let (px, py) = (px?, py?);
            Some((
                (ox as f64 + (px - x0 as f64) * scale).round() as i32,
                (oy as f64 + (py - y0 as f64) * scale).round() as i32,
            ))
        };
        if let Some((mx, my)) = marker(Some(x), Some(y)) {
            circle(
                &mut canvas,
                sheet_width,
                sheet_height,
                mx,
                my,
                5,
                [255, 222, 80],
            );
        }
        for (prefix, color) in [
            ("gaussian", [78, 168, 255]),
            ("hybrid", [72, 224, 161]),
            ("quality", [255, 157, 66]),
        ] {
            if let Some((mx, my)) = marker(
                row[format!("{prefix}_x_px")].as_f64(),
                row[format!("{prefix}_y_px")].as_f64(),
            ) {
                circle(&mut canvas, sheet_width, sheet_height, mx, my, 4, color);
            }
        }
    }
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = Encoder::new(file, sheet_width as u32, sheet_height as u32);
    encoder.set_color(ColorType::Rgb);
    encoder.set_depth(BitDepth::Eight);
    encoder
        .write_header()
        .map_err(|error| error.to_string())?
        .write_image_data(&canvas)
        .map_err(|error| error.to_string())
}

fn target_row(
    target: &Target,
    fwhm: f64,
    level: f64,
    peak_fraction: f64,
    control_counts: [usize; 4],
    gaussian: Option<&Detection>,
    gaussian_nearby: usize,
    hybrid: Option<&Detection>,
    hybrid_nearby: usize,
    quality: Option<&Detection>,
    image_path: &str,
    image_hash: &str,
) -> Value {
    let mut row = Map::new();
    let matched_quality = hybrid.map(|source| source.quality_passed);
    row.insert("target_id".into(), json!(target.id));
    row.insert("x_px".into(), json!(target.x));
    row.insert("y_px".into(), json!(target.y));
    row.insert("psf_fwhm_px".into(), json!(fwhm));
    row.insert(
        "local_background_adu".into(),
        json!(target.local_background),
    );
    row.insert("local_noise_adu".into(), json!(target.local_noise));
    row.insert("control_candidate_within_match_radius".into(), json!(false));
    row.insert(
        "truth_type".into(),
        json!("synthetic Gaussian point-source injection"),
    );
    row.insert("gaussian_center_amplitude_adu".into(), json!(level));
    row.insert(
        "sampled_peak_pixel_increment_adu".into(),
        json!(level * peak_fraction),
    );
    row.insert(
        "sampled_peak_fraction_of_center_amplitude".into(),
        json!(peak_fraction),
    );
    for (key, count) in [
        "control_candidates_within_match_radius",
        "control_gaussian_candidates_within_match_radius",
        "control_hybrid_candidates_within_match_radius",
        "control_quality_candidates_within_match_radius",
    ]
    .iter()
    .zip(control_counts)
    {
        row.insert((*key).into(), json!(count));
    }
    row.insert("gaussian_candidate_hit".into(), json!(gaussian.is_some()));
    row.insert(
        "gaussian_nearby_candidate_count".into(),
        json!(gaussian_nearby),
    );
    row.insert("hybrid_candidate_hit".into(), json!(hybrid.is_some()));
    row.insert("hybrid_nearby_candidate_count".into(), json!(hybrid_nearby));
    row.insert(
        "hybrid_matched_candidate_quality_passed".into(),
        matched_quality.map_or(Value::Null, |v| json!(v)),
    );
    row.insert("quality_gate_hit".into(), json!(quality.is_some()));
    row.insert(
        "quality_gate_loss_from_hybrid_candidate".into(),
        json!(hybrid.is_some() && quality.is_none()),
    );
    row.insert(
        "quality_matching_assignment_changed".into(),
        json!(hybrid.map(|v| v.detection_id) != quality.map(|v| v.detection_id)),
    );
    row.insert(
        "quality_rejection_flags".into(),
        json!(hybrid
            .filter(|v| !v.quality_passed)
            .map_or(String::new(), |v| v.flags.join("|"))),
    );
    row.insert(
        "hybrid_candidate_status".into(),
        json!(if hybrid.is_some() {
            "recovered"
        } else {
            "missed"
        }),
    );
    row.insert(
        "quality_gate_status".into(),
        json!(if quality.is_some() {
            "passed"
        } else if hybrid.is_some() {
            "rejected"
        } else {
            "not reached"
        }),
    );
    detection_record(gaussian, target.x, target.y, "gaussian", &mut row);
    detection_record(hybrid, target.x, target.y, "hybrid", &mut row);
    detection_record(quality, target.x, target.y, "quality", &mut row);
    row.insert("frozen_injection_image".into(), json!(image_path));
    row.insert("frozen_injection_image_sha256".into(), json!(image_hash));
    Value::Object(row)
}

fn control_row(
    version: &str,
    candidate_count: usize,
    returned_count: usize,
    quality_count: usize,
    output_count: usize,
    target_count: usize,
    seconds: f64,
    reused: &str,
) -> Value {
    json!({"version":version,"input":"same float32 no-injection control","candidate_count":candidate_count,"returned_count":returned_count,
        "quality_count":quality_count,"output_count":output_count,"target_site_candidates_within_match_radius":target_count,
        "detector_seconds":seconds,"reused_version":reused})
}

fn resolve_input_path(path: &Path) -> Result<PathBuf> {
    fs::canonicalize(path).map_err(|error| format!("FITS input path cannot be resolved: {error}"))
}

fn validate_options(options: &Options) -> Result<f64> {
    if options.peak_levels.is_empty()
        || options
            .peak_levels
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err("peak levels must be finite and positive".into());
    }
    if options.target_count == 0
        || options.psf_fwhm <= 0.0
        || options
            .injected_psf_fwhm
            .is_some_and(|v| !v.is_finite() || v <= 0.0)
        || options.threshold_sigma <= 0.0
        || options.min_distance == 0
    {
        return Err("detector and PSF parameters must be positive".into());
    }
    if options.aperture_radius == 0
        || options.background_box_size < 8
        || options.min_flux_snr < 0.0
        || options.min_psf_support_pixels == 0
    {
        return Err("quality parameters are outside the supported range".into());
    }
    if options
        .match_radius
        .is_some_and(|v| !v.is_finite() || v <= 0.0)
    {
        return Err("match radius must be positive".into());
    }
    Ok(options.match_radius.unwrap_or(options.psf_fwhm).max(3.0))
}

fn git_revision(root: &Path) -> Option<String> {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn source_execution(root: &Path, argv: &[String]) -> Value {
    let revision = git_revision(root);
    let paths = [
        "backend/src/point_source_ablation.rs",
        "backend/src/detection.rs",
        "backend/src/fits.rs",
    ];
    let mut hashes = Map::new();
    for relative in paths {
        if let Ok(hash) = sha256_file(&root.join(relative)) {
            hashes.insert(relative.into(), json!(hash));
        }
    }
    let dirty = Command::new("git")
        .args([
            "status",
            "--porcelain",
            "--",
            "backend/src",
            "desktop/src",
            "pyproject.toml",
        ])
        .current_dir(root)
        .output()
        .ok()
        .is_some_and(|o| !o.stdout.is_empty());
    json!({"argv":argv,"runtime":"rust","platform":std::env::consts::OS,"source_revision":revision,"source_worktree_dirty":dirty,"source_file_sha256":hashes})
}

fn run_analysis(
    fits_path: &Path,
    out_dir: &Path,
    options: &Options,
    fixed_manifest: Option<&Path>,
    argv: &[String],
) -> Result<Value> {
    let actual_radius = validate_options(options)?;
    let source = resolve_input_path(fits_path)?;
    if !source.is_file() {
        return Err(format!("FITS file not found: {}", source.display()));
    }
    if out_dir.exists()
        && fs::read_dir(out_dir)
            .map_err(|e| e.to_string())?
            .next()
            .is_some()
    {
        return Err(format!(
            "refusing to overwrite non-empty experiment directory: {}",
            out_dir.display()
        ));
    }
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let output = fs::canonicalize(out_dir).map_err(|e| e.to_string())?;
    let frozen = output.join("frozen_inputs");
    fs::create_dir_all(&frozen).map_err(|e| e.to_string())?;
    let frame = fits::read_fits(&source, false)?;
    let height = frame.height;
    let width = frame.width;
    if height == 0 || width == 0 || frame.data.len() != height * width {
        return Err("expected a non-empty 2-D FITS image".into());
    }
    let raw_hash = sha256_file(&source)?;
    let raw_array_hash = array_hash_pixels(&frame.data, height, width);
    let control = frame
        .data
        .to_f64()
        .into_iter()
        .map(|v| v as f32)
        .collect::<Vec<_>>();
    let mask = fits::auxiliary_mask(height, width)?;
    let control_path = frozen.join("control-float32.npy");
    write_npy_f32(&control_path, &control, height, width)?;
    let started_all = Instant::now();
    let started = Instant::now();
    let ensemble = detect(&control, height, width, &mask, options, "ensemble")?;
    let ensemble_seconds = started.elapsed().as_secs_f64();
    let mut detector_parameters = Map::new();
    detector_parameters.insert(
        "site_selection_ensemble".into(),
        ensemble.parameters.clone(),
    );
    let mut timings = Map::new();
    timings.insert("site_selection_ensemble".into(), json!([ensemble_seconds]));
    let mut control_versions = BTreeMap::new();
    let mut control_secs = BTreeMap::new();
    for mode in ["gaussian", "hybrid"] {
        let started = Instant::now();
        let result = detect(&control, height, width, &mask, options, mode)?;
        let elapsed = started.elapsed().as_secs_f64();
        detector_parameters.insert(format!("control_{mode}"), result.parameters.clone());
        control_secs.insert(mode.to_owned(), elapsed);
        control_versions.insert(mode.to_owned(), result);
        timings.insert(format!("control_{mode}"), json!([elapsed]));
    }
    let injected_fwhm = options.injected_psf_fwhm.unwrap_or(options.psf_fwhm);
    let targets = if let Some(path) = fixed_manifest {
        manifest_targets(
            path,
            &raw_hash,
            options.target_count,
            injected_fwhm,
            height,
            width,
        )?
    } else {
        let existing = ensemble
            .sources
            .iter()
            .chain(control_versions["gaussian"].sources.iter())
            .chain(control_versions["hybrid"].sources.iter())
            .cloned()
            .collect::<Vec<_>>();
        select_targets(
            &control,
            height,
            width,
            &existing,
            ensemble.noise,
            options,
            actual_radius,
        )?
    };
    let (ensemble_matches, ensemble_nearby) =
        match_detections(&targets, &ensemble.sources, actual_radius)?;
    if !ensemble_matches.is_empty() || ensemble_nearby.values().any(|v| *v > 0) {
        return Err(
            "the site-selection ensemble has a candidate within the true-match radius".into(),
        );
    }
    let gaussian_control = &control_versions["gaussian"];
    let hybrid_control = &control_versions["hybrid"];
    let (gaussian_control_matches, gaussian_control_nearby) =
        match_detections(&targets, &gaussian_control.sources, actual_radius)?;
    let (hybrid_control_matches, hybrid_control_nearby) =
        match_detections(&targets, &hybrid_control.sources, actual_radius)?;
    let quality_control = hybrid_control
        .sources
        .iter()
        .filter(|s| s.quality_passed)
        .cloned()
        .collect::<Vec<_>>();
    let (_, quality_control_nearby) = match_detections(&targets, &quality_control, actual_radius)?;
    if !gaussian_control_matches.is_empty()
        || !hybrid_control_matches.is_empty()
        || gaussian_control_nearby.values().any(|v| *v > 0)
        || hybrid_control_nearby.values().any(|v| *v > 0)
    {
        return Err("a Gaussian/Hybrid control candidate lies within the true-match radius".into());
    }
    let control_rows = vec![
        control_row(
            "A 基线 Gaussian 提案",
            gaussian_control.candidate_count,
            gaussian_control.sources.len(),
            gaussian_control.quality_count,
            gaussian_control.sources.len(),
            gaussian_control_nearby.values().sum(),
            control_secs["gaussian"],
            "",
        ),
        control_row(
            "B Gaussian+DoG 多尺度提案",
            hybrid_control.candidate_count,
            hybrid_control.sources.len(),
            hybrid_control.quality_count,
            hybrid_control.sources.len(),
            hybrid_control_nearby.values().sum(),
            control_secs["hybrid"],
            "",
        ),
        control_row(
            "C 对 B 候选应用现有质量门",
            hybrid_control.candidate_count,
            hybrid_control.sources.len(),
            hybrid_control.quality_count,
            quality_control.len(),
            quality_control_nearby.values().sum(),
            control_secs["hybrid"],
            "B candidate objects",
        ),
    ];
    let control_csv = output.join("control_summary.csv");
    write_csv(&control_csv, CONTROL_COLUMNS, &control_rows)?;
    let mut target_rows = Vec::<Value>::new();
    let mut summary_rows = Vec::<Value>::new();
    let mut image_artifacts = Map::new();
    let mut case_sheet_paths = Map::new();
    let mut gaussian_times = Vec::new();
    let mut hybrid_times = Vec::new();
    for level in &options.peak_levels {
        let mut image = control.clone();
        let mut peak_fractions = Vec::with_capacity(targets.len());
        for target in &targets {
            peak_fractions.push(inject_gaussian(
                &mut image,
                height,
                width,
                target.x,
                target.y,
                *level,
                injected_fwhm,
            ));
        }
        let level_key = compact_number(*level);
        let image_path = frozen.join(format!("injected-peak-{level_key}-adu.npy"));
        write_npy_f32(&image_path, &image, height, width)?;
        let image_file_hash = sha256_file(&image_path)?;
        let image_array_hash = array_hash_f32(&image, height, width);
        let start = Instant::now();
        let gaussian = detect(&image, height, width, &mask, options, "gaussian")?;
        let gaussian_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let hybrid = detect(&image, height, width, &mask, options, "hybrid")?;
        let hybrid_seconds = start.elapsed().as_secs_f64();
        gaussian_times.push(gaussian_seconds);
        hybrid_times.push(hybrid_seconds);
        detector_parameters.insert("gaussian".into(), gaussian.parameters.clone());
        detector_parameters.insert("hybrid".into(), hybrid.parameters.clone());
        let (gaussian_matches, gaussian_nearby) =
            match_detections(&targets, &gaussian.sources, actual_radius)?;
        let (hybrid_matches, hybrid_nearby) =
            match_detections(&targets, &hybrid.sources, actual_radius)?;
        let quality = hybrid
            .sources
            .iter()
            .filter(|s| s.quality_passed)
            .cloned()
            .collect::<Vec<_>>();
        let (quality_matches, quality_nearby) =
            match_detections(&targets, &quality, actual_radius)?;
        summary_rows.push(summary_row(
            *level,
            "A 基线 Gaussian 提案",
            &gaussian_matches,
            &targets,
            &gaussian_nearby,
            gaussian_seconds,
            gaussian.candidate_count,
            gaussian.quality_count,
        ));
        summary_rows.push(summary_row(
            *level,
            "B Gaussian+DoG 多尺度提案",
            &hybrid_matches,
            &targets,
            &hybrid_nearby,
            hybrid_seconds,
            hybrid.candidate_count,
            hybrid.quality_count,
        ));
        summary_rows.push(summary_row(
            *level,
            "C 对 B 候选应用现有质量门",
            &quality_matches,
            &targets,
            &quality_nearby,
            hybrid_seconds,
            hybrid.candidate_count,
            hybrid.quality_count,
        ));
        let mut level_rows = Vec::new();
        for (index, target) in targets.iter().enumerate() {
            let gm = gaussian_matches.get(&target.id);
            let hm = hybrid_matches.get(&target.id);
            let qm = quality_matches.get(&target.id);
            let row = target_row(
                target,
                injected_fwhm,
                *level,
                peak_fractions[index],
                [
                    ensemble_nearby[&target.id],
                    gaussian_control_nearby[&target.id],
                    hybrid_control_nearby[&target.id],
                    quality_control_nearby[&target.id],
                ],
                gm,
                gaussian_nearby[&target.id],
                hm,
                hybrid_nearby[&target.id],
                qm,
                &display_path(&image_path),
                &image_file_hash,
            );
            target_rows.push(row.clone());
            level_rows.push(row);
        }
        let image_name = format!("cases-{level_key}-adu.png");
        let case_path = output.join(&image_name);
        write_case_sheet(&case_path, &image, height, width, &level_rows, *level)?;
        case_sheet_paths.insert(level_key.clone(), json!(display_path(&case_path)));
        image_artifacts.insert(level_key.clone(),json!({"frozen_image":display_path(&image_path),"frozen_image_sha256":image_file_hash,"array_sha256":image_array_hash,"dtype":"float32","shape":[height,width],
            "gaussian_detection_seconds":gaussian_seconds,"hybrid_detection_seconds":hybrid_seconds,"gaussian_candidate_count":gaussian.candidate_count,"gaussian_quality_count":gaussian.quality_count,
            "hybrid_candidate_count":hybrid.candidate_count,"hybrid_quality_count":hybrid.quality_count}));
    }
    let summary_path = output.join("summary.csv");
    let target_path = output.join("target_results.csv");
    write_csv(&summary_path, SUMMARY_COLUMNS, &summary_rows)?;
    write_csv(&target_path, TARGET_COLUMNS, &target_rows)?;
    timings.insert("gaussian".into(), json!(gaussian_times));
    timings.insert("hybrid".into(), json!(hybrid_times));
    let raw_header = frame
        .header
        .iter()
        .filter(|(key, _)| {
            [
                "BITPIX", "BZERO", "BSCALE", "DATE-OBS", "EXPTIME", "EXPOSURE",
            ]
            .contains(&key.as_str())
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<Map<_, _>>();
    let target_values = targets
        .iter()
        .map(|target| target_json(target, injected_fwhm))
        .collect::<Vec<_>>();
    let target_count_collisions = 0usize;
    let manifest = json!({
        "schema_version":3,"experiment":"point_source_detection_ablation","status":"completed","created_local_time":local_time_iso(),
        "source_fits":{"path":display_path(&source),"sha256":raw_hash,"size_bytes":fs::metadata(&source).map_err(|e|e.to_string())?.len()},
        "source_image":{"shape":[height,width],"dtype":frame.data.dtype(),"array_sha256":raw_array_hash,"fits_header":raw_header},
        "shared_control":{"path":display_path(&control_path),"file_sha256":sha256_file(&control_path)?,"array_sha256":array_hash_f32(&control,height,width),"dtype":"float32","shape":[height,width],
            "selection_method":if fixed_manifest.is_some(){format!("reused frozen coordinates from {}",display_path(fixed_manifest.unwrap()))}else{"Ensemble/Gaussian/Hybrid candidates plus local 98th-percentile blank patch screen; positions frozen before injected versions".to_owned()},
            "selection_seed":options.seed,"control_detector_seconds":ensemble_seconds,"control_candidate_count":ensemble.candidate_count,"control_quality_count":ensemble.quality_count,
            "background_candidate_collisions_within_match_radius":target_count_collisions,"versions":control_rows,"versions_summary_csv":display_path(&control_csv)},
        "targets":target_values,"gaussian_center_amplitudes_adu":options.peak_levels,
        "versions":[{"id":"A","name":"基线 Gaussian 提案","candidate_proposal":"gaussian","quality_gate":"measured and reported, not used for candidate recall"},
            {"id":"B","name":"Gaussian + DoG 多尺度提案","candidate_proposal":"hybrid","quality_gate":"same existing gate, reported separately"},
            {"id":"C","name":"B 候选 + 现有质量门","candidate_pool":"same B result objects","quality_gate":"Detection.quality_passed; rejected flags retained"}],
        "parameters_requested":{"threshold_sigma":options.threshold_sigma,"psf_fwhm_px":options.psf_fwhm,"injected_psf_fwhm_px":injected_fwhm,"min_distance_px":options.min_distance,
            "aperture_radius_px":options.aperture_radius,"background_box_size_px":options.background_box_size,"min_flux_snr":options.min_flux_snr,"min_psf_support_pixels":options.min_psf_support_pixels,
            "reject_linear_artifacts":true,"match_radius_requested_px":options.match_radius,"match_radius_actual_px":actual_radius,
            "match_rule":"minimum-distance sorted greedy one-to-one candidate-truth assignment; maximum allowed distance is actual match radius",
            "signal_definition":"continuous Gaussian center amplitude added to the original image at the subpixel centroid, ADU; the theoretical maximum sampled pixel increment is lower according to the exact discrete kernel response, recorded for each target",
            "dtype_for_all_detector_versions":"float32","background_handling":"original raw FITS values converted once to float32; no global background subtraction before injection; detector local background estimation unchanged",
            "psf_model":"discrete Gaussian kernel","negative_control":"same float32 original image is run through A and B; C reuses B candidates. All frozen target sites must have zero A/B/C candidates within match radius or the run aborts"},
        "detector_parameters_actual":detector_parameters,"injected_images":image_artifacts,
        "metrics":{"candidate_recall":"one-to-one matched injected truth targets / injected target count","quality_recall":"one-to-one matches among the same version's quality-passed detections / injected target count",
            "quality_gate_loss":"B candidate matched but no quality-passed B candidate matches; divided by B candidate matched targets and separately by all injected targets","position_error":"Euclidean distance between measured centroid and injected center; median and P90 among matched targets, with misses reported separately",
            "duplicates":"additional detector candidates inside a target's match radius after one-to-one assignment; not interpreted as false positives","runtime_attribution":"C reuses B detector output and reports the B detector runtime; the post-hoc quality-gate filtering time is not measured separately",
            "interval":"95% Wilson interval shown descriptively by target at each signal level; target positions share a single image and repeated levels, so it is not a population-level independent-sample interval"},
        "runtime":{"total_experiment_seconds":started_all.elapsed().as_secs_f64(),"detector_seconds_by_version":timings},
        "outputs":{"summary_csv":display_path(&summary_path),"target_csv":display_path(&target_path),"control_summary_csv":display_path(&control_csv),"case_sheets":case_sheet_paths},
        "limitations":["The truth is an artificial Gaussian injection, not an independently catalogued physical star.","No full-image precision or false-positive rate is reported because the real background is not fully labelled.","All positions are on one FITS background; repeated signal levels and nearby pixels are not independent samples.","Motion proposal and track association are not measured in this point-source experiment."],
        "execution":source_execution(&Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap_or(Path::new(".")),argv)
    });
    let manifest_path = output.join("manifest.json");
    fs::write(
        &manifest_path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?
        ),
    )
    .map_err(|e| e.to_string())?;
    Ok(
        json!({"status":"completed","run_dir":display_path(&output),"manifest":display_path(&manifest_path),"summary":display_path(&summary_path),"target_count":options.target_count,"levels":options.peak_levels}),
    )
}

fn local_time_iso() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
        + 8 * 3600;
    let days = seconds.div_euclid(86400);
    let rem = seconds.rem_euclid(86400);
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 }.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += if month <= 2 { 1 } else { 0 };
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}+08:00",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn take_value(
    args: &[String],
    index: &mut usize,
    inline: Option<&str>,
    flag: &str,
) -> Result<String> {
    if let Some(v) = inline {
        return Ok(v.to_owned());
    }
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_args(args: &[String]) -> Result<CliArgs> {
    let mut parsed = CliArgs {
        options: Options::default(),
        argv: std::iter::once("rst19-ablation".to_owned())
            .chain(args.iter().cloned())
            .collect(),
        ..CliArgs::default()
    };
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if !arg.starts_with('-') {
            if parsed.fits_path.is_some() {
                return Err(format!("unexpected positional argument: {arg}"));
            }
            parsed.fits_path = Some(PathBuf::from(arg));
            i += 1;
            continue;
        }
        if arg == "--help" || arg == "-h" {
            return Err("__help__".into());
        }
        let (name, inline) = arg
            .strip_prefix("--")
            .ok_or_else(|| format!("unknown option {arg}"))?
            .split_once('=')
            .map_or((arg.trim_start_matches('-'), None), |(a, b)| (a, Some(b)));
        if name == "peak-levels" {
            let mut values = Vec::new();
            if let Some(raw) = inline {
                values.push(
                    raw.parse::<f64>()
                        .map_err(|_| "--peak-levels values must be numeric")?,
                );
            } else {
                i += 1;
                while i < args.len() && !args[i].starts_with('-') {
                    values.push(
                        args[i]
                            .parse::<f64>()
                            .map_err(|_| "--peak-levels values must be numeric")?,
                    );
                    i += 1;
                }
                i -= 1;
            }
            if values.is_empty() {
                return Err("--peak-levels requires one or more values".into());
            }
            parsed.options.peak_levels = values;
            i += 1;
            continue;
        }
        let raw = take_value(args, &mut i, inline, &format!("--{name}"))?;
        match name {
            "out-dir" => parsed.out_dir = Some(raw.into()),
            "report" => parsed.report = Some(raw.into()),
            "reuse-manifest" => parsed.reuse_manifest = Some(raw.into()),
            "targets" => {
                parsed.options.target_count = raw
                    .parse()
                    .map_err(|_| "--targets must be a positive integer")?
            }
            "seed" => {
                parsed.options.seed = raw
                    .parse()
                    .map_err(|_| "--seed must be a non-negative integer")?
            }
            "psf-fwhm" => {
                parsed.options.psf_fwhm = raw.parse().map_err(|_| "--psf-fwhm must be numeric")?
            }
            "injected-psf-fwhm" => {
                parsed.options.injected_psf_fwhm = Some(
                    raw.parse()
                        .map_err(|_| "--injected-psf-fwhm must be numeric")?,
                )
            }
            "threshold-sigma" => {
                parsed.options.threshold_sigma = raw
                    .parse()
                    .map_err(|_| "--threshold-sigma must be numeric")?
            }
            "min-distance" => {
                parsed.options.min_distance = raw
                    .parse()
                    .map_err(|_| "--min-distance must be an integer")?
            }
            "aperture-radius" => {
                parsed.options.aperture_radius = raw
                    .parse()
                    .map_err(|_| "--aperture-radius must be an integer")?
            }
            "background-box-size" => {
                parsed.options.background_box_size = raw
                    .parse()
                    .map_err(|_| "--background-box-size must be an integer")?
            }
            "min-flux-snr" => {
                parsed.options.min_flux_snr =
                    raw.parse().map_err(|_| "--min-flux-snr must be numeric")?
            }
            "min-psf-support-pixels" => {
                parsed.options.min_psf_support_pixels = raw
                    .parse()
                    .map_err(|_| "--min-psf-support-pixels must be an integer")?
            }
            "match-radius-px" => {
                parsed.options.match_radius = Some(
                    raw.parse()
                        .map_err(|_| "--match-radius-px must be numeric")?,
                )
            }
            _ => return Err(format!("unknown option --{name}")),
        }
        i += 1;
    }
    validate_options(&parsed.options)?;
    Ok(parsed)
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn write_report(run_dir: &Path, report_path: &Path) -> Result<PathBuf> {
    let manifest_path = run_dir.join("manifest.json");
    let manifest: Value =
        serde_json::from_slice(&fs::read(&manifest_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if manifest["schema_version"].as_i64() != Some(3)
        || manifest["experiment"].as_str() != Some("point_source_detection_ablation")
    {
        return Err(
            "point-ablation report requires schema_version 3 point-source ablation output".into(),
        );
    }
    let report = if report_path.is_absolute() {
        report_path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(report_path)
    };
    let parent = report
        .parent()
        .ok_or("report path has no parent directory")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let evidence = parent.join("evidence/point");
    fs::create_dir_all(&evidence).map_err(|e| e.to_string())?;
    for name in [
        "manifest.json",
        "summary.csv",
        "target_results.csv",
        "control_summary.csv",
    ] {
        let from = run_dir.join(name);
        if from.is_file() {
            fs::copy(&from, evidence.join(name)).map_err(|e| e.to_string())?;
        }
    }
    if let Some(sheets) = manifest["outputs"]["case_sheets"].as_object() {
        for value in sheets.values() {
            if let Some(path) = value.as_str() {
                if let Some(name) = Path::new(path).file_name() {
                    let from = run_dir.join(name);
                    if from.is_file() {
                        fs::copy(&from, evidence.join(name)).map_err(|e| e.to_string())?;
                    }
                }
            }
        }
    }
    let section=format!("<!-- RST19-POINT-ABLATION-START --><section id=\"rst19-point-ablation\"><h2>点源提案与质量门消融</h2><p>在同一真实 FITS float32 背景与冻结位置上比较 Gaussian 候选、Gaussian+DoG 候选和对同一 B 候选应用现有质量门。注入真值是合成 Gaussian 控制，不代表物理恒星。</p><ul><li><a href=\"evidence/point/summary.csv\">信号档位汇总</a></li><li><a href=\"evidence/point/target_results.csv\">逐目标回收结果</a></li><li><a href=\"evidence/point/control_summary.csv\">无注入控制</a></li><li><a href=\"evidence/point/manifest.json\">输入哈希与参数清单</a></li></ul><p>运行目录：<code>{}</code></p></section><!-- RST19-POINT-ABLATION-END -->",html_escape(&display_path(run_dir)));
    let mut html = if report.is_file() {
        fs::read_to_string(&report).map_err(|e| e.to_string())?
    } else {
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>RST19 消融报告</title></head><body><main><h1>RST19 消融报告</h1></main></body></html>".to_owned()
    };
    if let Some(start) = html.find("<!-- RST19-POINT-ABLATION-START -->") {
        if let Some(end) = html[start..].find("<!-- RST19-POINT-ABLATION-END -->") {
            let end = start + end + "<!-- RST19-POINT-ABLATION-END -->".len();
            html.replace_range(start..end, &section);
        } else {
            html.push_str(&section);
        }
    } else if let Some(end) = html.find("</body>") {
        html.insert_str(end, &section);
    } else {
        html.push_str(&section);
    }
    fs::write(&report, html).map_err(|e| e.to_string())?;
    Ok(report)
}

fn default_out_dir() -> PathBuf {
    PathBuf::from(format!(
        "tmp/ablation/run-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    ))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let parsed = parse_args(args)?;
    let source = parsed
        .fits_path
        .as_ref()
        .ok_or("FITS input path is required")?;
    let out = parsed.out_dir.unwrap_or_else(default_out_dir);
    let result = run_analysis(
        source,
        &out,
        &parsed.options,
        parsed.reuse_manifest.as_deref(),
        &parsed.argv,
    )?;
    let report = write_report(
        Path::new(result["run_dir"].as_str().ok_or("missing run directory")?),
        parsed
            .report
            .as_deref()
            .unwrap_or(Path::new("deliverables/ablation/index.html")),
    )?;
    let mut result = result;
    result["report"] = json!(display_path(&report));
    Ok(result)
}

pub fn run_cli(args: &[String]) -> i32 {
    let parsed = match parse_args(args) {
        Ok(value) => value,
        Err(error) if error == "__help__" => {
            println!("在同一真实 FITS float32 背景与冻结真值位置上比较 Gaussian、Gaussian+DoG 候选和现有质量门\n\nUsage: rst19-ablation FITS [--out-dir DIR] [--report FILE] [--peak-levels ADU ...] [--targets N] [--seed N] [--reuse-manifest FILE]");
            return 0;
        }
        Err(error) => {
            eprintln!("rst19-ablation: {error}");
            return 2;
        }
    };
    let Some(source) = parsed.fits_path.as_ref() else {
        eprintln!("rst19-ablation: FITS input path is required");
        return 2;
    };
    let out = parsed.out_dir.unwrap_or_else(default_out_dir);
    match run_analysis(
        source,
        &out,
        &parsed.options,
        parsed.reuse_manifest.as_deref(),
        &parsed.argv,
    ) {
        Ok(result) => {
            let report = write_report(
                Path::new(result["run_dir"].as_str().unwrap_or_default()),
                parsed
                    .report
                    .as_deref()
                    .unwrap_or(Path::new("deliverables/ablation/index.html")),
            );
            match report {
                Ok(path) => {
                    let mut result = result;
                    result["report"] = json!(display_path(&path));
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&result).unwrap_or_default()
                    );
                    0
                }
                Err(error) => {
                    eprintln!("rst19-ablation: failed to write report: {error}");
                    2
                }
            }
        }
        Err(error) => {
            eprintln!("rst19-ablation: failed: {error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shortest_in_radius_pairs_are_assigned_once() {
        let targets = vec![
            Target {
                id: "T1".into(),
                x: 5.0,
                y: 5.0,
                local_background: 0.0,
                local_noise: 1.0,
            },
            Target {
                id: "T2".into(),
                x: 6.0,
                y: 5.0,
                local_background: 0.0,
                local_noise: 1.0,
            },
        ];
        let make = |id, x| Detection {
            detection_id: id,
            x,
            y: 5.0,
            peak: 1.0,
            flux: 1.0,
            background: 0.0,
            noise: 1.0,
            snr: 1.0,
            fwhm: None,
            flags: vec![],
            flux_error: None,
            flux_snr: None,
            filter_snr: None,
            fwhm_x: None,
            fwhm_y: None,
            ellipticity: None,
            sharpness: None,
            footprint_pixels: None,
            psf_support_pixels: None,
            quality_passed: true,
            peak_x: None,
            peak_y: None,
            centroid_shift_px: None,
            proposal_methods: vec![],
            proposal_scales: vec![],
            proposal_snr: None,
            nearest_gaussian_px: None,
            deblend_delta_bic: None,
            deblend_component_snr: None,
            repeated_code_count: None,
            range_anomaly_pixel_count: None,
            repeated_code_values: vec![],
        };
        let (matches, nearby) =
            match_detections(&targets, &[make(1, 5.2), make(2, 5.8)], 1.0).unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(nearby["T1"], 2);
        assert_eq!(nearby["T2"], 2);
    }
    #[test]
    fn gaussian_injection_and_array_hash_are_deterministic() {
        let mut image = vec![0.0f32; 25 * 25];
        let fraction = inject_gaussian(&mut image, 25, 25, 12.3, 12.5, 24.0, 2.0);
        let sigma = 2.0 / 2.354820045;
        let expected = (-((12.0_f64 - 12.3).powi(2) + (12.0_f64 - 12.5).powi(2))
            / (2.0 * sigma * sigma))
            .exp();
        assert!((fraction - expected).abs() < 1e-12);
        assert_eq!(image[12 * 25 + 12], (24.0 * fraction) as f32);
        assert_eq!(
            array_hash_f32(&image, 25, 25),
            array_hash_f32(&image, 25, 25)
        );
    }
    #[test]
    fn parser_keeps_ablation_defaults_and_manifest_flag() {
        let p = parse_args(&[
            "image.fits".into(),
            "--reuse-manifest".into(),
            "frozen.json".into(),
            "--peak-levels".into(),
            "12".into(),
            "24".into(),
        ])
        .unwrap();
        assert_eq!(p.options.peak_levels, vec![12.0, 24.0]);
        assert_eq!(p.options.target_count, 8);
        assert_eq!(p.reuse_manifest, Some(PathBuf::from("frozen.json")));
    }
}
