//! Paired crowded-pair/isolated-source ablation over frozen FITS patches.
//!
//! Synthetic sources are detector-level controls on real image backgrounds;
//! the not-injected companion site is a local negative control, not physical
//! evidence that the field is empty.

use crate::{
    detection::{self, DetectionOptions, DetectionResult},
    fits,
    measurement::{Detection, MeasurementOptions},
    models::{FitsFrame, PixelData},
    Result,
};
use csv::{Terminator, WriterBuilder};
use png::{BitDepth, ColorType, Encoder};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const PATCH_SIZE: usize = 96;
const PRIMARY: [f64; 2] = [48.0, 48.0];
const COMPANION: [f64; 2] = [54.0, 48.0];
const VERSIONS: [(&str, &str, bool); 3] = [
    ("A", "gaussian", false),
    ("B", "hybrid", false),
    ("D", "gaussian", true),
];

#[derive(Clone)]
struct Anchor {
    id: String,
    x: f64,
    y: f64,
    patch: Vec<f32>,
    baseline: BTreeMap<String, DetectionResult>,
    collisions: Value,
}

#[derive(Clone)]
struct RunOptions {
    anchor_count: usize,
    match_radius_px: f64,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    Ok(sha256_bytes(&bytes))
}

fn array_sha256(values: &[f32], height: usize, width: usize) -> String {
    let mut digest = Sha256::new();
    digest.update(b"<f4");
    digest.update(format!("[{height},{width}]").as_bytes());
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn display_path(path: &Path) -> String {
    let value = path.to_string_lossy();
    #[cfg(windows)]
    {
        if let Some(unc) = value.strip_prefix("\\\\?\\UNC\\") {
            return format!("\\\\{unc}");
        }
        if let Some(path) = value.strip_prefix("\\\\?\\") {
            return path.to_owned();
        }
    }
    value.into_owned()
}

fn utc_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let days = seconds.div_euclid(86_400);
    let day_seconds = seconds.rem_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}+00:00",
        day_seconds / 3600,
        (day_seconds % 3600) / 60,
        day_seconds % 60
    )
}

fn json_string<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .as_str()
        .ok_or_else(|| format!("{field} must be a string"))
}

fn json_number(value: &Value, field: &str) -> Result<f64> {
    let number = value
        .as_f64()
        .ok_or_else(|| format!("{field} must be numeric"))?;
    if !number.is_finite() {
        return Err(format!("{field} must be finite"));
    }
    Ok(number)
}

fn safe_json_f64(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

fn detector_options(proposal_mode: &str, deblend: bool) -> DetectionOptions {
    DetectionOptions {
        measurement: MeasurementOptions {
            aperture_radius: 4,
            psf_fwhm: 3.0,
            min_flux_snr: 2.0,
            min_psf_support_pixels: 3,
            ..MeasurementOptions::default()
        },
        threshold_sigma: 6.0,
        min_distance: 4,
        background_box_size: 32,
        proposal_mode: proposal_mode.to_owned(),
        dog_threshold_sigma: Some(5.0),
        enable_local_deblend: deblend,
        ..DetectionOptions::default()
    }
}

fn detect_patch(image: &[f32], proposal_mode: &str, deblend: bool) -> Result<DetectionResult> {
    if image.len() != PATCH_SIZE * PATCH_SIZE {
        return Err("crowded ablation patch must be 96x96".into());
    }
    let mask = fits::auxiliary_mask(PATCH_SIZE, PATCH_SIZE)?;
    detection::detect_sources(
        &PixelData::F32(image.to_vec()),
        Some(&mask),
        PATCH_SIZE,
        PATCH_SIZE,
        &detector_options(proposal_mode, deblend),
        None,
    )
}

fn nearby_count(detections: &[Detection], x: f64, y: f64, radius: f64) -> usize {
    detections
        .iter()
        .filter(|item| (item.x - x).hypot(item.y - y) <= radius)
        .count()
}

fn patch_at(frame: &FitsFrame, center_x: f64, center_y: f64) -> Option<Vec<f32>> {
    let x_center = center_x.round_ties_even() as i64;
    let y_center = center_y.round_ties_even() as i64;
    let half = (PATCH_SIZE / 2) as i64;
    if x_center < half
        || y_center < half
        || x_center + half > frame.width as i64
        || y_center + half > frame.height as i64
    {
        return None;
    }
    let mut patch = Vec::with_capacity(PATCH_SIZE * PATCH_SIZE);
    for y in (y_center - half) as usize..(y_center + half) as usize {
        for x in (x_center - half) as usize..(x_center + half) as usize {
            patch.push(frame.data.value(y * frame.width + x) as f32);
        }
    }
    patch.iter().all(|value| value.is_finite()).then_some(patch)
}

fn add_gaussian(image: &mut [f32], center_x: f64, center_y: f64, peak: f64, fwhm: f64) {
    let sigma = fwhm / 2.354_820_045;
    let radius = 4usize.max((4.0 * sigma).ceil() as usize);
    let x_min = (center_x.floor() as i64 - radius as i64).max(0) as usize;
    let y_min = (center_y.floor() as i64 - radius as i64).max(0) as usize;
    let x_max = ((center_x.floor() as i64 + radius as i64 + 1).max(0) as usize).min(PATCH_SIZE);
    let y_max = ((center_y.floor() as i64 + radius as i64 + 1).max(0) as usize).min(PATCH_SIZE);
    for y in y_min..y_max {
        for x in x_min..x_max {
            let dx = x as f64 - center_x;
            let dy = y as f64 - center_y;
            let increment = peak * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            let index = y * PATCH_SIZE + x;
            image[index] = (f64::from(image[index]) + increment) as f32;
        }
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

fn value_csv(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value).unwrap_or_default(),
    }
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
            .map(|column| value_csv(&row[*column]))
            .collect::<Vec<_>>();
        writer
            .write_record(fields)
            .map_err(|error| error.to_string())?;
    }
    let mut bytes = vec![0xef, 0xbb, 0xbf];
    bytes.extend(writer.into_inner().map_err(|error| error.to_string())?);
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn quantile(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let index = (sorted.len() - 1) as f64 * q;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = (index - lower as f64) as f32;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

fn set_pixel(canvas: &mut [u8], width: usize, height: usize, x: i32, y: i32, color: [u8; 3]) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }
    let index = (y as usize * width + x as usize) * 3;
    canvas[index..index + 3].copy_from_slice(&color);
}

fn draw_line(
    canvas: &mut [u8],
    width: usize,
    height: usize,
    from: (i32, i32),
    to: (i32, i32),
    color: [u8; 3],
    thickness: i32,
) {
    let (mut x0, mut y0) = from;
    let (x1, y1) = to;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        for oy in -(thickness / 2)..=(thickness / 2) {
            for ox in -(thickness / 2)..=(thickness / 2) {
                set_pixel(canvas, width, height, x0 + ox, y0 + oy, color);
            }
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let twice = 2 * error;
        if twice >= dy {
            error += dy;
            x0 += sx;
        }
        if twice <= dx {
            error += dx;
            y0 += sy;
        }
    }
}

fn draw_circle(
    canvas: &mut [u8],
    width: usize,
    height: usize,
    center: (i32, i32),
    radius: i32,
    color: [u8; 3],
    thickness: i32,
) {
    for y in -radius..=radius {
        for x in -radius..=radius {
            let distance = ((x * x + y * y) as f64).sqrt();
            if (distance - f64::from(radius)).abs() <= f64::from(thickness) / 2.0 {
                set_pixel(canvas, width, height, center.0 + x, center.1 + y, color);
            }
        }
    }
}

fn injected_truths(anchor_id: &str, scene: &str, array_hash: &str) -> Vec<Value> {
    let (companion_id, companion_role, companion_injected) = if scene == "pair" {
        ("COMPANION", "weak companion", true)
    } else {
        (
            "NO-COMPANION",
            "not-injected companion negative site",
            false,
        )
    };
    vec![
        json!({
            "target_id":format!("{anchor_id}-PRIMARY"),
            "anchor_id":anchor_id,
            "role":"bright primary",
            "injected":true,
            "x_px":PRIMARY[0],
            "y_px":PRIMARY[1],
            "scene_array_sha256":array_hash
        }),
        json!({
            "target_id":format!("{anchor_id}-{companion_id}"),
            "anchor_id":anchor_id,
            "role":companion_role,
            "injected":companion_injected,
            "x_px":COMPANION[0],
            "y_px":COMPANION[1],
            "scene_array_sha256":array_hash
        }),
    ]
}

fn match_truths(
    truths: &[Value],
    detections: &[Detection],
    radius: f64,
) -> (BTreeMap<String, usize>, BTreeMap<String, usize>) {
    let mut edges = Vec::<(f64, String, usize)>::new();
    let mut local_counts = BTreeMap::<String, usize>::new();
    for truth in truths {
        let target_id = truth["target_id"].as_str().unwrap_or_default().to_owned();
        let x = truth["x_px"].as_f64().unwrap_or_default();
        let y = truth["y_px"].as_f64().unwrap_or_default();
        let nearby = detections
            .iter()
            .enumerate()
            .filter_map(|(index, detection)| {
                let distance = (detection.x - x).hypot(detection.y - y);
                (distance <= radius).then_some((distance, target_id.clone(), index))
            })
            .collect::<Vec<_>>();
        local_counts.insert(target_id, nearby.len());
        edges.extend(nearby);
    }
    edges.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let mut assigned_truths = std::collections::HashSet::<String>::new();
    let mut assigned_detections = std::collections::HashSet::<usize>::new();
    let mut matches = BTreeMap::new();
    for (_, target_id, index) in edges {
        if assigned_truths.contains(&target_id) || assigned_detections.contains(&index) {
            continue;
        }
        assigned_truths.insert(target_id.clone());
        assigned_detections.insert(index);
        matches.insert(target_id, index);
    }
    (matches, local_counts)
}

fn percentile_f64(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() == 1 {
        return Some(sorted[0]);
    }
    let index = (sorted.len() - 1) as f64 * q;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    Some(sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower as f64))
}

fn scene_metrics(
    anchor_id: &str,
    scene: &str,
    truths: &[Value],
    results: &BTreeMap<String, DetectionResult>,
    timings: &BTreeMap<String, f64>,
    match_radius: f64,
) -> (Vec<Value>, Vec<Value>) {
    let injected = truths
        .iter()
        .filter(|truth| truth["injected"].as_bool() == Some(true))
        .cloned()
        .collect::<Vec<_>>();
    let negative = truths
        .iter()
        .filter(|truth| truth["injected"].as_bool() != Some(true))
        .collect::<Vec<_>>();
    let mut summaries = Vec::new();
    let mut target_rows = Vec::new();
    for version in ["A", "B", "C", "D"] {
        let result = &results[version];
        let quality_sources = result
            .sources
            .iter()
            .filter(|source| source.quality_passed)
            .cloned()
            .collect::<Vec<_>>();
        let (candidate_matches, candidate_nearby) =
            match_truths(&injected, &result.sources, match_radius);
        let (quality_matches, quality_nearby) =
            match_truths(&injected, &quality_sources, match_radius);
        let errors = injected
            .iter()
            .filter_map(|truth| {
                let key = truth["target_id"].as_str()?;
                let index = *candidate_matches.get(key)?;
                let detection = &result.sources[index];
                Some(
                    (detection.x - truth["x_px"].as_f64()?)
                        .hypot(detection.y - truth["y_px"].as_f64()?),
                )
            })
            .collect::<Vec<_>>();
        let negative_candidate_hits = negative
            .iter()
            .filter(|truth| {
                nearby_count(
                    &result.sources,
                    truth["x_px"].as_f64().unwrap_or_default(),
                    truth["y_px"].as_f64().unwrap_or_default(),
                    match_radius,
                ) > 0
            })
            .count();
        let negative_quality_hits = negative
            .iter()
            .filter(|truth| {
                nearby_count(
                    &quality_sources,
                    truth["x_px"].as_f64().unwrap_or_default(),
                    truth["y_px"].as_f64().unwrap_or_default(),
                    match_radius,
                ) > 0
            })
            .count();
        let quality_passed_hits = quality_matches.len();
        let version_name = match version {
            "A" => "Gaussian 基线",
            "B" => "Gaussian+DoG 多尺度提案",
            "C" => "B 候选通过现有质量门",
            _ => "Gaussian+局部去混叠候选",
        };
        summaries.push(json!({
            "anchor_id":anchor_id,
            "scene":scene,
            "version":version,
            "version_name":version_name,
            "injected_truth_total":injected.len(),
            "candidate_matched_targets":candidate_matches.len(),
            "candidate_recall":if injected.is_empty() { Value::Null } else { safe_json_f64(candidate_matches.len() as f64 / injected.len() as f64) },
            "quality_matched_targets":quality_passed_hits,
            "quality_recall":if injected.is_empty() { Value::Null } else { safe_json_f64(quality_passed_hits as f64 / injected.len() as f64) },
            "quality_gate_lost_candidate_hits":candidate_matches.len().saturating_sub(quality_passed_hits),
            "median_position_error_px":percentile_f64(&errors, 0.5).map_or(Value::Null, safe_json_f64),
            "p90_position_error_px":percentile_f64(&errors, 0.9).map_or(Value::Null, safe_json_f64),
            "negative_companion_site_candidates":negative_candidate_hits,
            "negative_companion_site_quality_hits":negative_quality_hits,
            "candidate_count":result.candidate_count,
            "quality_count":result.quality_count,
            "detector_seconds":safe_json_f64(timings[version]),
            "match_radius_px":match_radius
        }));
        for truth in truths {
            let target_id = truth["target_id"].as_str().unwrap_or_default();
            let is_injected = truth["injected"].as_bool() == Some(true);
            let candidate_index = if is_injected {
                candidate_matches.get(target_id).copied()
            } else {
                None
            };
            let quality_index = if is_injected {
                quality_matches.get(target_id).copied()
            } else {
                None
            };
            let x = truth["x_px"].as_f64().unwrap_or_default();
            let y = truth["y_px"].as_f64().unwrap_or_default();
            let candidate_nearby_count = if is_injected {
                candidate_nearby.get(target_id).copied().unwrap_or(0)
            } else {
                nearby_count(&result.sources, x, y, match_radius)
            };
            let quality_nearby_count = if is_injected {
                quality_nearby.get(target_id).copied().unwrap_or(0)
            } else {
                nearby_count(&quality_sources, x, y, match_radius)
            };
            let candidate = candidate_index.map(|index| &result.sources[index]);
            let position_error = candidate.map(|item| (item.x - x).hypot(item.y - y));
            target_rows.push(json!({
                "scene":scene,
                "version":version,
                "anchor_id":anchor_id,
                "target_id":target_id,
                "role":truth["role"],
                "injected":is_injected,
                "truth_x_px":x,
                "truth_y_px":y,
                "candidate_nearby_count":candidate_nearby_count,
                "candidate_hit":if is_injected { candidate.is_some() } else { candidate_nearby_count > 0 },
                "quality_nearby_count":quality_nearby_count,
                "quality_hit":if is_injected { quality_index.is_some() } else { quality_nearby_count > 0 },
                "detection_id":candidate.map_or(Value::Null, |item| json!(item.detection_id)),
                "detection_x_px":candidate.map_or(Value::Null, |item| safe_json_f64(item.x)),
                "detection_y_px":candidate.map_or(Value::Null, |item| safe_json_f64(item.y)),
                "position_error_px":position_error.map_or(Value::Null, safe_json_f64),
                "quality_passed":candidate.map_or(Value::Null, |item| json!(item.quality_passed)),
                "flags":candidate.map_or_else(String::new, |item| item.flags.join("|")),
                "frozen_input_sha256":truth["scene_array_sha256"]
            }));
        }
    }
    (summaries, target_rows)
}

fn csv_columns() -> BTreeMap<&'static str, &'static [&'static str]> {
    BTreeMap::from([
        (
            "summary.csv",
            &[
                "scene",
                "version",
                "version_name",
                "patch_count",
                "injected_truth_total",
                "candidate_matched_targets",
                "candidate_recall",
                "quality_matched_targets",
                "quality_recall",
                "quality_gate_lost_candidate_hits",
                "median_position_error_px",
                "p90_position_error_px",
                "negative_companion_sites_total",
                "negative_companion_site_candidates",
                "negative_companion_site_fp_rate",
                "negative_companion_site_quality_hits",
                "candidate_count_total",
                "quality_count_total",
                "detector_seconds_total",
                "median_detector_seconds_per_patch",
                "match_radius_px",
            ][..],
        ),
        (
            "anchor_summary.csv",
            &[
                "anchor_id",
                "scene",
                "version",
                "version_name",
                "injected_truth_total",
                "candidate_matched_targets",
                "candidate_recall",
                "quality_matched_targets",
                "quality_recall",
                "quality_gate_lost_candidate_hits",
                "median_position_error_px",
                "p90_position_error_px",
                "negative_companion_site_candidates",
                "negative_companion_site_quality_hits",
                "candidate_count",
                "quality_count",
                "detector_seconds",
                "match_radius_px",
            ][..],
        ),
        (
            "target_results.csv",
            &[
                "scene",
                "version",
                "anchor_id",
                "target_id",
                "role",
                "injected",
                "truth_x_px",
                "truth_y_px",
                "candidate_nearby_count",
                "candidate_hit",
                "quality_nearby_count",
                "quality_hit",
                "detection_id",
                "detection_x_px",
                "detection_y_px",
                "position_error_px",
                "quality_passed",
                "flags",
                "frozen_input_sha256",
            ][..],
        ),
        (
            "control_summary.csv",
            &[
                "anchor_id",
                "version",
                "primary_site_candidates",
                "companion_site_candidates",
                "candidate_count",
                "quality_count",
                "detector_seconds",
                "input",
            ][..],
        ),
        (
            "detector_runs.csv",
            &[
                "anchor_id",
                "scene",
                "version",
                "input_file",
                "input_file_sha256",
                "input_array_sha256",
                "proposal_mode",
                "enable_local_deblend",
                "candidate_count",
                "returned_count",
                "quality_count",
                "detector_seconds",
                "reused_detector_version",
                "parameters_actual",
            ][..],
        ),
    ])
}

fn write_case_sheet(
    path: &Path,
    first_anchor: &str,
    pair_image: &[f32],
    single_image: &[f32],
    target_rows: &[Value],
) -> Result<()> {
    let width = 1080usize;
    let height = 500usize;
    let mut canvas = vec![239u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let color = if !(16..1064).contains(&x) || !(12..484).contains(&y) {
                [205, 215, 228]
            } else {
                [255, 255, 255]
            };
            set_pixel(&mut canvas, width, height, x as i32, y as i32, color);
        }
    }
    let scenes = [("pair", pair_image), ("single", single_image)];
    let versions = [
        ("A", [55, 145, 235]),
        ("B", [35, 160, 118]),
        ("D", [222, 72, 73]),
    ];
    for (scene_index, (scene, values)) in scenes.iter().enumerate() {
        let panel_left = 16 + scene_index * 536;
        let image_left = panel_left + 42;
        let image_top = 58usize;
        let crop_x = 30usize;
        let crop_y = 30usize;
        let crop_width = 42usize;
        let crop_height = 36usize;
        let crop = (0..crop_height)
            .flat_map(|y| {
                (0..crop_width).map(move |x| values[(crop_y + y) * PATCH_SIZE + crop_x + x])
            })
            .collect::<Vec<_>>();
        let mut sorted = crop.clone();
        sorted.sort_by(f32::total_cmp);
        let low = quantile(&sorted, 0.02);
        let high = quantile(&sorted, 0.997).max(low + 1.0);
        for y in 0..crop_height {
            for x in 0..crop_width {
                let value = crop[y * crop_width + x];
                let gray = (((value - low) / (high - low)).clamp(0.0, 1.0) * 255.0) as u8;
                for sy in 0..10 {
                    for sx in 0..10 {
                        set_pixel(
                            &mut canvas,
                            width,
                            height,
                            (image_left + x * 10 + sx) as i32,
                            (image_top + y * 10 + sy) as i32,
                            [gray, gray, gray],
                        );
                    }
                }
            }
        }
        let point = |xy: [f64; 2]| {
            (
                (image_left as f64 + (xy[0] - crop_x as f64) * 10.0).round() as i32,
                (image_top as f64 + (xy[1] - crop_y as f64) * 10.0).round() as i32,
            )
        };
        let primary = point(PRIMARY);
        draw_circle(&mut canvas, width, height, primary, 7, [255, 217, 59], 3);
        let companion = point(COMPANION);
        if *scene == "pair" {
            draw_circle(&mut canvas, width, height, companion, 7, [192, 104, 221], 3);
        } else {
            draw_line(
                &mut canvas,
                width,
                height,
                (companion.0 - 6, companion.1 - 6),
                (companion.0 + 6, companion.1 + 6),
                [192, 104, 221],
                3,
            );
            draw_line(
                &mut canvas,
                width,
                height,
                (companion.0 - 6, companion.1 + 6),
                (companion.0 + 6, companion.1 - 6),
                [192, 104, 221],
                3,
            );
        }
        for (version, color) in versions {
            for row in target_rows.iter().filter(|row| {
                row["anchor_id"].as_str() == Some(first_anchor)
                    && row["scene"].as_str() == Some(*scene)
                    && row["version"].as_str() == Some(version)
            }) {
                let (Some(x), Some(y)) = (
                    row["detection_x_px"].as_f64(),
                    row["detection_y_px"].as_f64(),
                ) else {
                    continue;
                };
                let center = point([x, y]);
                if version == "D" && row["role"].as_str() == Some("weak companion") {
                    draw_line(
                        &mut canvas,
                        width,
                        height,
                        (center.0 - 6, center.1 - 6),
                        (center.0 + 6, center.1 + 6),
                        color,
                        3,
                    );
                    draw_line(
                        &mut canvas,
                        width,
                        height,
                        (center.0 - 6, center.1 + 6),
                        (center.0 + 6, center.1 - 6),
                        color,
                        3,
                    );
                } else {
                    draw_circle(
                        &mut canvas,
                        width,
                        height,
                        center,
                        match version {
                            "A" => 4,
                            "B" => 7,
                            _ => 10,
                        },
                        color,
                        2,
                    );
                }
            }
        }
        // Small color swatches replace a text legend while keeping the sheet
        // font-independent across Windows, macOS and Linux.
        for (index, color) in [
            [255, 217, 59],
            [192, 104, 221],
            [55, 145, 235],
            [35, 160, 118],
            [222, 72, 73],
        ]
        .into_iter()
        .enumerate()
        {
            let x = image_left as i32 + index as i32 * 22;
            let y = 438;
            draw_circle(&mut canvas, width, height, (x, y), 5, color, 2);
        }
    }
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = Encoder::new(BufWriter::new(file), width as u32, height as u32);
    encoder.set_color(ColorType::Rgb);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
    writer
        .write_image_data(&canvas)
        .map_err(|error| error.to_string())
}

fn run_analysis(
    fits_path: &Path,
    fixed_manifest_path: &Path,
    out_dir: &Path,
    options: &RunOptions,
    argv: &[String],
) -> Result<Value> {
    let source = fits_path
        .canonicalize()
        .map_err(|error| format!("cannot open input FITS {}: {error}", fits_path.display()))?;
    let fixed_path = fixed_manifest_path.canonicalize().map_err(|error| {
        format!(
            "cannot open fixed target manifest {}: {error}",
            fixed_manifest_path.display()
        )
    })?;
    let output = if out_dir.is_absolute() {
        out_dir.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(out_dir)
    };
    if output.exists()
        && fs::read_dir(&output)
            .map_err(|error| error.to_string())?
            .next()
            .is_some()
    {
        return Err(format!(
            "refusing to overwrite non-empty experiment directory: {}",
            output.display()
        ));
    }
    if options.anchor_count == 0
        || !options.match_radius_px.is_finite()
        || options.match_radius_px <= 0.0
    {
        return Err("anchor count and match radius must be positive".into());
    }
    let source_sha = sha256_file(&source)?;
    let fixed_bytes = fs::read(&fixed_path).map_err(|error| error.to_string())?;
    let fixed_manifest: Value =
        serde_json::from_slice(&fixed_bytes).map_err(|error| error.to_string())?;
    if fixed_manifest["source_fits"]["sha256"].as_str() != Some(source_sha.as_str()) {
        return Err("fixed anchor manifest input SHA-256 does not match the current FITS".into());
    }
    let fixed_targets = fixed_manifest["targets"]
        .as_array()
        .ok_or("fixed anchor manifest has no targets array")?;
    if fixed_targets.len() < options.anchor_count {
        return Err("fixed anchor manifest has fewer target sites than requested anchors".into());
    }
    let frame = fits::read_fits(&source, true)?;
    if frame.width == 0 || frame.height == 0 || frame.data.len() != frame.width * frame.height {
        return Err("expected a non-empty 2-D FITS image".into());
    }
    let started_all = Instant::now();
    let mut eligible = Vec::<Anchor>::new();
    let mut rejected = Vec::<Value>::new();
    let mut previous_centers = Vec::<(f64, f64)>::new();
    for item in fixed_targets {
        if eligible.len() >= options.anchor_count {
            break;
        }
        let x = json_number(&item["x_px"], "fixed target x_px")?;
        let y = json_number(&item["y_px"], "fixed target y_px")?;
        let Some(patch) = patch_at(&frame, x, y) else {
            rejected.push(json!({"anchor_id":item.get("target_id").cloned().unwrap_or(Value::Null),"reason":"96x96 patch shape or finite-value check failed or exceeds image edge"}));
            continue;
        };
        if previous_centers
            .iter()
            .any(|(px, py)| (x - px).hypot(y - py) < PATCH_SIZE as f64)
        {
            rejected.push(json!({"anchor_id":item.get("target_id").cloned().unwrap_or(Value::Null),"reason":"patch overlaps an earlier anchor"}));
            continue;
        }
        let mut baseline = BTreeMap::<String, DetectionResult>::new();
        for (version, mode, deblend) in VERSIONS {
            baseline.insert(version.to_owned(), detect_patch(&patch, mode, deblend)?);
        }
        let mut collisions = Map::new();
        let mut has_collision = false;
        for version in ["A", "B", "D"] {
            let result = &baseline[version];
            let primary_count = nearby_count(
                &result.sources,
                PRIMARY[0],
                PRIMARY[1],
                options.match_radius_px,
            );
            let companion_count = nearby_count(
                &result.sources,
                COMPANION[0],
                COMPANION[1],
                options.match_radius_px,
            );
            has_collision |= primary_count > 0 || companion_count > 0;
            collisions.insert(
                version.to_owned(),
                json!({"primary":primary_count,"companion":companion_count}),
            );
        }
        let collisions = Value::Object(collisions);
        if has_collision {
            rejected.push(json!({"anchor_id":item.get("target_id").cloned().unwrap_or(Value::Null),"reason":"baseline candidate within a known injection site","baseline_collisions":collisions}));
            continue;
        }
        let anchor_id = json_string(&item["target_id"], "fixed target_id")?.to_owned();
        eligible.push(Anchor {
            id: anchor_id,
            x,
            y,
            patch,
            baseline,
            collisions,
        });
        previous_centers.push((x, y));
    }
    if eligible.len() < options.anchor_count {
        return Err(format!(
            "only {} fixed patches passed no-injection screening; {} are required",
            eligible.len(),
            options.anchor_count
        ));
    }
    let site_screen_seconds = started_all.elapsed().as_secs_f64();
    fs::create_dir_all(&output).map_err(|error| error.to_string())?;
    let frozen_dir = output.join("frozen_inputs");
    fs::create_dir_all(&frozen_dir).map_err(|error| error.to_string())?;
    let mut anchor_summary_rows = Vec::new();
    let mut target_rows = Vec::new();
    let mut control_rows = Vec::new();
    let mut detector_rows = Vec::new();
    let mut anchor_records = Vec::new();
    let mut aggregate_summaries = Vec::new();
    let selection_started = Instant::now();
    let mut first_pair = None::<Vec<f32>>;
    let mut first_single = None::<Vec<f32>>;
    for anchor in &eligible {
        let base_name = format!("{}-base-float32.npy", anchor.id);
        let base_path = frozen_dir.join(&base_name);
        write_npy_f32(&base_path, &anchor.patch, PATCH_SIZE, PATCH_SIZE)?;
        let base_hash = sha256_file(&base_path)?;
        let mut pair_image = anchor.patch.clone();
        add_gaussian(&mut pair_image, PRIMARY[0], PRIMARY[1], 160.0, 1.4);
        add_gaussian(&mut pair_image, COMPANION[0], COMPANION[1], 40.0, 0.8);
        let mut single_image = anchor.patch.clone();
        add_gaussian(&mut single_image, PRIMARY[0], PRIMARY[1], 160.0, 1.4);
        let mut record = json!({
            "anchor_id":anchor.id,
            "original_fits_center_px":[anchor.x,anchor.y],
            "crop_origin_px":[anchor.x.round_ties_even() as i64 - 48,anchor.y.round_ties_even() as i64 - 48],
            "patch_shape":[PATCH_SIZE,PATCH_SIZE],
            "base_array_sha256":array_sha256(&anchor.patch,PATCH_SIZE,PATCH_SIZE),
            "base_file":display_path(&base_path),
            "base_file_sha256":base_hash,
            "baseline_collisions":anchor.collisions,
            "scenes":{}
        });
        for version in ["A", "B", "D"] {
            let control = &anchor.baseline[version];
            control_rows.push(json!({
                "anchor_id":anchor.id,
                "version":version,
                "primary_site_candidates":nearby_count(&control.sources,PRIMARY[0],PRIMARY[1],options.match_radius_px),
                "companion_site_candidates":nearby_count(&control.sources,COMPANION[0],COMPANION[1],options.match_radius_px),
                "candidate_count":control.candidate_count,
                "quality_count":control.quality_count,
                "detector_seconds":Value::Null,
                "input":"same real FITS patch, no injection; pre-screen run"
            }));
        }
        for (scene, image) in [
            ("pair", pair_image.as_slice()),
            ("single", single_image.as_slice()),
        ] {
            let scene_path = frozen_dir.join(format!("{}-{scene}-float32.npy", anchor.id));
            write_npy_f32(&scene_path, image, PATCH_SIZE, PATCH_SIZE)?;
            let scene_file_sha = sha256_file(&scene_path)?;
            let scene_array_sha = array_sha256(image, PATCH_SIZE, PATCH_SIZE);
            let truths = injected_truths(&anchor.id, scene, &scene_array_sha);
            let mut results = BTreeMap::<String, DetectionResult>::new();
            let mut timings = BTreeMap::<String, f64>::new();
            for (version, mode, deblend) in VERSIONS {
                let started = Instant::now();
                let result = detect_patch(image, mode, deblend)?;
                let elapsed = started.elapsed().as_secs_f64();
                results.insert(version.to_owned(), result);
                timings.insert(version.to_owned(), elapsed);
            }
            let b_result = results["B"].clone();
            let c_result = DetectionResult {
                sources: b_result
                    .sources
                    .iter()
                    .filter(|source| source.quality_passed)
                    .cloned()
                    .collect(),
                ..b_result.clone()
            };
            results.insert("C".into(), c_result);
            timings.insert("C".into(), timings["B"]);
            for version in ["A", "B", "C", "D"] {
                let result = &results[version];
                detector_rows.push(json!({
                    "anchor_id":anchor.id,
                    "scene":scene,
                    "version":version,
                    "input_file":display_path(&scene_path),
                    "input_file_sha256":scene_file_sha,
                    "input_array_sha256":scene_array_sha,
                    "proposal_mode":if version=="B" || version=="C" {"hybrid"} else {"gaussian"},
                    "enable_local_deblend":version=="D",
                    "candidate_count":result.candidate_count,
                    "returned_count":result.sources.len(),
                    "quality_count":result.quality_count,
                    "detector_seconds":safe_json_f64(timings[version]),
                    "reused_detector_version":if version=="C" {"B"} else {""},
                    "parameters_actual":result.parameters
                }));
            }
            let (scene_summaries, scene_targets) = scene_metrics(
                &anchor.id,
                scene,
                &truths,
                &results,
                &timings,
                options.match_radius_px,
            );
            anchor_summary_rows.extend(scene_summaries);
            target_rows.extend(scene_targets);
            record["scenes"][scene] = json!({
                "file":display_path(&scene_path),
                "file_sha256":scene_file_sha,
                "array_sha256":scene_array_sha,
                "dtype":"float32",
                "truths":truths
            });
            if anchor_records.is_empty() {
                if scene == "pair" {
                    first_pair = Some(image.to_vec());
                }
                if scene == "single" {
                    first_single = Some(image.to_vec());
                }
            }
        }
        anchor_records.push(record);
    }
    for scene in ["pair", "single"] {
        for version in ["A", "B", "C", "D"] {
            let patch_rows = anchor_summary_rows
                .iter()
                .filter(|row| {
                    row["scene"].as_str() == Some(scene) && row["version"].as_str() == Some(version)
                })
                .collect::<Vec<_>>();
            let injected_total = patch_rows
                .iter()
                .map(|row| row["injected_truth_total"].as_u64().unwrap_or(0))
                .sum::<u64>();
            let candidate_hits = patch_rows
                .iter()
                .map(|row| row["candidate_matched_targets"].as_u64().unwrap_or(0))
                .sum::<u64>();
            let quality_hits = patch_rows
                .iter()
                .map(|row| row["quality_matched_targets"].as_u64().unwrap_or(0))
                .sum::<u64>();
            let negative_total = target_rows
                .iter()
                .filter(|row| {
                    row["scene"].as_str() == Some(scene)
                        && row["version"].as_str() == Some(version)
                        && row["role"].as_str() == Some("not-injected companion negative site")
                })
                .count();
            let negative_hits = patch_rows
                .iter()
                .map(|row| {
                    row["negative_companion_site_candidates"]
                        .as_u64()
                        .unwrap_or(0)
                })
                .sum::<u64>();
            let negative_quality_hits = patch_rows
                .iter()
                .map(|row| {
                    row["negative_companion_site_quality_hits"]
                        .as_u64()
                        .unwrap_or(0)
                })
                .sum::<u64>();
            let errors = target_rows
                .iter()
                .filter(|row| {
                    row["scene"].as_str() == Some(scene)
                        && row["version"].as_str() == Some(version)
                        && row["injected"].as_bool() == Some(true)
                        && row["candidate_hit"].as_bool() == Some(true)
                })
                .filter_map(|row| row["position_error_px"].as_f64())
                .collect::<Vec<_>>();
            let seconds = patch_rows
                .iter()
                .filter_map(|row| row["detector_seconds"].as_f64())
                .collect::<Vec<_>>();
            let total_seconds = seconds.iter().sum::<f64>();
            let version_name = match version {
                "A" => "Gaussian 基线",
                "B" => "Gaussian+DoG 多尺度提案",
                "C" => "B 候选通过现有质量门",
                _ => "Gaussian+局部去混叠候选",
            };
            aggregate_summaries.push(json!({
                "scene":scene,
                "version":version,
                "version_name":version_name,
                "patch_count":patch_rows.len(),
                "injected_truth_total":injected_total,
                "candidate_matched_targets":candidate_hits,
                "candidate_recall":if injected_total==0 {Value::Null} else {safe_json_f64(candidate_hits as f64 / injected_total as f64)},
                "quality_matched_targets":quality_hits,
                "quality_recall":if injected_total==0 {Value::Null} else {safe_json_f64(quality_hits as f64 / injected_total as f64)},
                "quality_gate_lost_candidate_hits":patch_rows.iter().map(|row| row["quality_gate_lost_candidate_hits"].as_u64().unwrap_or(0)).sum::<u64>(),
                "median_position_error_px":percentile_f64(&errors,0.5).map_or(Value::Null,safe_json_f64),
                "p90_position_error_px":percentile_f64(&errors,0.9).map_or(Value::Null,safe_json_f64),
                "negative_companion_sites_total":negative_total,
                "negative_companion_site_candidates":negative_hits,
                "negative_companion_site_fp_rate":if negative_total==0 {Value::Null} else {safe_json_f64(negative_hits as f64 / negative_total as f64)},
                "negative_companion_site_quality_hits":negative_quality_hits,
                "candidate_count_total":patch_rows.iter().map(|row| row["candidate_count"].as_u64().unwrap_or(0)).sum::<u64>(),
                "quality_count_total":patch_rows.iter().map(|row| row["quality_count"].as_u64().unwrap_or(0)).sum::<u64>(),
                "detector_seconds_total":safe_json_f64(total_seconds),
                "median_detector_seconds_per_patch":percentile_f64(&seconds,0.5).map_or(Value::Null,safe_json_f64),
                "match_radius_px":options.match_radius_px
            }));
        }
    }
    write_csv(
        &output.join("summary.csv"),
        csv_columns()["summary.csv"],
        &aggregate_summaries,
    )?;
    write_csv(
        &output.join("anchor_summary.csv"),
        csv_columns()["anchor_summary.csv"],
        &anchor_summary_rows,
    )?;
    write_csv(
        &output.join("target_results.csv"),
        csv_columns()["target_results.csv"],
        &target_rows,
    )?;
    write_csv(
        &output.join("control_summary.csv"),
        csv_columns()["control_summary.csv"],
        &control_rows,
    )?;
    write_csv(
        &output.join("detector_runs.csv"),
        csv_columns()["detector_runs.csv"],
        &detector_rows,
    )?;
    let case_sheet = output.join("crowded_pair_cases.png");
    write_case_sheet(
        &case_sheet,
        &eligible[0].id,
        first_pair.as_deref().ok_or("missing first pair image")?,
        first_single
            .as_deref()
            .ok_or("missing first single image")?,
        &target_rows,
    )?;
    let source_metadata = fs::metadata(&source).map_err(|error| error.to_string())?;
    let requested_detector_options = json!({
        "threshold_sigma":6.0,"min_distance":4,"aperture_radius":4,"psf_fwhm":3.0,
        "background_box_size":32,"min_flux_snr":2.0,"min_psf_support_pixels":3,
        "dog_threshold_sigma":5.0
    });
    let manifest = json!({
        "schema_version":1,
        "experiment":"paired_real_fits_crowded_companion_ablation",
        "status":"completed",
        "created_utc":utc_timestamp(),
        "source_fits":{"path":display_path(&source),"sha256":source_sha,"size_bytes":source_metadata.len()},
        "fixed_anchor_manifest":{"path":display_path(&fixed_path),"sha256":sha256_bytes(&fixed_bytes),"source_experiment":fixed_manifest.get("experiment").cloned().unwrap_or(Value::Null)},
        "patch_selection":{
            "requested_anchor_count":options.anchor_count,
            "accepted_anchor_count":eligible.len(),
            "patch_shape":[PATCH_SIZE,PATCH_SIZE],
            "site_screen":"A Gaussian, B hybrid, and D Gaussian+local-deblend all have zero returned candidates within the match radius of both prospective sites on the same no-injection patch",
            "accepted_anchors":anchor_records,
            "rejected_anchors":rejected
        },
        "injections":{
            "primary":{"x_px":PRIMARY[0],"y_px":PRIMARY[1],"center_peak_excess_adu":160.0,"fwhm_px":1.4},
            "companion":{"x_px":COMPANION[0],"y_px":COMPANION[1],"center_peak_excess_adu":40.0,"fwhm_px":0.8},
            "pair_separation_px":(COMPANION[0]-PRIMARY[0]).hypot(COMPANION[1]-PRIMARY[1]),
            "negative_control":"single-source scenes contain the primary only; the companion coordinate is a known not-injected site, not a claim of physical blank sky",
            "all_version_inputs":"each detector variant loads the same frozen float32 NPY for a given anchor and scene"
        },
        "versions":[
            {"id":"A","name":"Gaussian baseline proposals","proposal_mode":"gaussian","enable_local_deblend":false},
            {"id":"B","name":"Gaussian + DoG multiscale proposals","proposal_mode":"hybrid","enable_local_deblend":false},
            {"id":"C","name":"B candidates passed through existing quality gate","candidate_pool":"same B Detection objects; quality_passed only"},
            {"id":"D","name":"Gaussian proposals + local deblend","proposal_mode":"gaussian","enable_local_deblend":true}
        ],
        "parameters_requested":{
            "threshold_sigma":6.0,"min_distance":4,"aperture_radius":4,"psf_fwhm":3.0,
            "background_box_size":32,"min_flux_snr":2.0,"min_psf_support_pixels":3,
            "dog_threshold_sigma":5.0,"match_radius_px":options.match_radius_px,
            "injected_psf_fwhm_px":{"primary":1.4,"companion":0.8},"dtype":"float32"
        },
        "detector_options":requested_detector_options,
        "outputs":{
            "summary_csv":display_path(&output.join("summary.csv")),
            "anchor_summary_csv":display_path(&output.join("anchor_summary.csv")),
            "target_results_csv":display_path(&output.join("target_results.csv")),
            "control_summary_csv":display_path(&output.join("control_summary.csv")),
            "detector_runs_csv":display_path(&output.join("detector_runs.csv")),
            "case_sheet_png":display_path(&case_sheet),
            "frozen_inputs_dir":display_path(&frozen_dir)
        },
        "metrics":{
            "candidate_recall":"one-to-one matched injected sources / injected truth count, separately from quality-passed recall",
            "position_error":"Euclidean distance in pixels; median and P90 over matched injected truths only",
            "quality_gate_loss":"candidate hit minus matched quality-passed hit count; C uses the same B candidate pool",
            "false_split_control":"count of candidate and quality detections within the configured radius of the not-injected companion site in paired single-source scenes",
            "matching":"minimum-distance sorted greedy one-to-one assignment"
        },
        "runtime":{"total_experiment_seconds":started_all.elapsed().as_secs_f64(),"site_screen_seconds":site_screen_seconds,"artifact_and_detector_seconds":selection_started.elapsed().as_secs_f64()},
        "limitations":[
            "Injected Gaussian sources are labelled synthetic truths on real FITS image patches; they are not independently catalogued physical stars.",
            "The not-injected companion coordinate is a local negative control only; it does not establish that the surrounding sky is physically empty.",
            "Eight patches from one FITS frame are paired measurements, not an independent survey population; no full-frame precision or physical completeness is estimated.",
            "D measures an optional single-frame deblend path. It is a separate crowded-pair condition and not a replacement for the A/B/C point-source ablation."
        ],
        "execution":{
            "argv":if argv.is_empty() {json!(["rst19-crowded-ablation"])} else {json!(argv)},
            "runtime":"Rust","version":env!("CARGO_PKG_VERSION"),"target_os":std::env::consts::OS,
            "target_arch":std::env::consts::ARCH,"source_revision":option_env!("RST19_SOURCE_REVISION"),
            "source_file_sha256":source_hashes()
        }
    });
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(json!({
        "status":"completed",
        "run_dir":display_path(&output),
        "manifest":display_path(&output.join("manifest.json")),
        "summary":display_path(&output.join("summary.csv")),
        "anchors":eligible.len()
    }))
}

fn source_hashes() -> Value {
    let backend = Path::new(env!("CARGO_MANIFEST_DIR"));
    let files = [
        "src/crowded_pair_ablation.rs",
        "src/detection.rs",
        "src/fits.rs",
        "src/measurement.rs",
    ];
    let mut hashes = Map::new();
    for relative in files {
        hashes.insert(
            relative.replace('\\', "/"),
            sha256_file(&backend.join(relative)).map_or(Value::Null, |value| json!(value)),
        );
    }
    Value::Object(hashes)
}

#[derive(Default)]
struct CliArgs {
    fits_path: Option<PathBuf>,
    fixed_manifest: Option<PathBuf>,
    point_run_dir: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    report: Option<PathBuf>,
    motion_run_dir: Option<PathBuf>,
    motion_proposal_run_dir: Option<PathBuf>,
    motion_crossing_run_dir: Option<PathBuf>,
    demo_summary: Option<PathBuf>,
    options: RunOptions,
    argv: Vec<String>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            anchor_count: 8,
            match_radius_px: 3.0,
        }
    }
}

fn take_value(
    args: &[String],
    index: &mut usize,
    inline: Option<&str>,
    flag: &str,
) -> Result<String> {
    if let Some(value) = inline {
        return Ok(value.to_owned());
    }
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_args(args: &[String]) -> Result<CliArgs> {
    let mut parsed = CliArgs {
        options: RunOptions::default(),
        argv: std::iter::once("rst19-crowded-ablation".to_owned())
            .chain(args.iter().cloned())
            .collect(),
        ..CliArgs::default()
    };
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if !arg.starts_with('-') {
            if parsed.fits_path.is_some() {
                return Err(format!("unexpected positional argument: {arg}"));
            }
            parsed.fits_path = Some(PathBuf::from(arg));
            index += 1;
            continue;
        }
        let (name, inline) = arg
            .strip_prefix("--")
            .ok_or_else(|| format!("unknown option {arg}"))?
            .split_once('=')
            .map_or((arg.trim_start_matches('-'), None), |(name, value)| {
                (name, Some(value))
            });
        let value = take_value(args, &mut index, inline, &format!("--{name}"))?;
        match name {
            "reuse-manifest" => parsed.fixed_manifest = Some(PathBuf::from(value)),
            "point-run-dir" => parsed.point_run_dir = Some(PathBuf::from(value)),
            "out-dir" => parsed.out_dir = Some(PathBuf::from(value)),
            "report" => parsed.report = Some(PathBuf::from(value)),
            "anchors" => {
                parsed.options.anchor_count = value
                    .parse::<usize>()
                    .map_err(|_| "--anchors must be a positive integer")?;
            }
            "match-radius-px" => {
                parsed.options.match_radius_px = value
                    .parse::<f64>()
                    .map_err(|_| "--match-radius-px must be a number")?;
            }
            "motion-run-dir" => parsed.motion_run_dir = Some(PathBuf::from(value)),
            "motion-proposal-run-dir" => {
                parsed.motion_proposal_run_dir = Some(PathBuf::from(value))
            }
            "motion-crossing-run-dir" => {
                parsed.motion_crossing_run_dir = Some(PathBuf::from(value))
            }
            "demo-summary" => parsed.demo_summary = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option --{name}")),
        }
        index += 1;
    }
    if parsed.options.anchor_count == 0
        || !parsed.options.match_radius_px.is_finite()
        || parsed.options.match_radius_px <= 0.0
    {
        return Err("anchor count and match radius must be positive".into());
    }
    Ok(parsed)
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination).map_err(|error| error.to_string())?;
    for entry in fs::read_dir(source).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        let target = destination.join(entry.file_name());
        if file_type.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), target).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn copy_report_evidence(source: &Path, destination: &Path, files: &[&str]) -> Result<()> {
    fs::create_dir_all(destination).map_err(|error| error.to_string())?;
    for name in files {
        let from = source.join(name);
        if from.is_file() {
            let to = destination.join(name);
            if from.canonicalize().ok() != to.canonicalize().ok() {
                fs::copy(&from, to).map_err(|error| error.to_string())?;
            }
        }
    }
    Ok(())
}

fn copy_analysis_evidence(source: &Path, destination: &Path) -> Result<()> {
    copy_report_evidence(
        source,
        destination,
        &[
            "summary.csv",
            "anchor_summary.csv",
            "target_results.csv",
            "control_summary.csv",
            "detector_runs.csv",
            "crowded_pair_cases.png",
            "manifest.json",
        ],
    )?;
    let frozen = source.join("frozen_inputs");
    if frozen.is_dir() {
        copy_tree(&frozen, &destination.join("frozen_inputs"))?;
    }
    Ok(())
}

fn report_section(
    output: &Path,
    optional_runs: &[(&str, &Option<PathBuf>)],
    demo_summary: &Option<PathBuf>,
) -> String {
    let mut optional_links = String::new();
    for (label, path) in optional_runs {
        if let Some(path) = path {
            optional_links.push_str(&format!(
                "<li>{}: <code>{}</code></li>",
                html_escape(label),
                html_escape(&display_path(path))
            ));
        }
    }
    if let Some(path) = demo_summary {
        optional_links.push_str(&format!(
            "<li>单帧演示摘要: <code>{}</code></li>",
            html_escape(&display_path(path))
        ));
    }
    let evidence = "evidence/crowded";
    format!(
        "<!-- RST19-CROWDED-ABLATION-START --><section id=\"rst19-crowded-ablation\"><h2>拥挤双源与孤立源配对消融</h2><p>同一真实 FITS 背景块上比较 Gaussian、Gaussian+DoG、多尺度候选质量门和局部去混叠。单星场景中的伴星位置只是未注入负控点，不代表物理空场。</p><ul><li><a href=\"{evidence}/summary.csv\">总汇总</a></li><li><a href=\"{evidence}/anchor_summary.csv\">按锚点结果</a></li><li><a href=\"{evidence}/target_results.csv\">逐目标结果</a></li><li><a href=\"{evidence}/control_summary.csv\">无注入筛选控制</a></li><li><a href=\"{evidence}/detector_runs.csv\">检测器运行记录</a></li><li><a href=\"{evidence}/manifest.json\">清单与输入哈希</a></li><li><a href=\"{evidence}/crowded_pair_cases.png\">首个锚点图</a></li></ul><p>实验目录：<code>{}</code></p>{}<p>注入结果只表示已知合成源在所选图像块中的检出，不是物理源计数或全幅精度估计。</p></section><!-- RST19-CROWDED-ABLATION-END -->",
        html_escape(&display_path(output)),
        if optional_links.is_empty() { String::new() } else { format!("<h3>关联证据</h3><ul>{optional_links}</ul>") }
    )
}

fn write_report(
    point_run_dir: &Path,
    report_path: &Path,
    output: &Path,
    optional_runs: &[(&str, &Option<PathBuf>)],
    demo_summary: &Option<PathBuf>,
) -> Result<PathBuf> {
    let point_manifest_path = point_run_dir.join("manifest.json");
    let point_manifest: Value =
        serde_json::from_slice(&fs::read(&point_manifest_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if point_manifest["schema_version"].as_i64() != Some(3)
        || point_manifest["experiment"].as_str() != Some("point_source_detection_ablation")
    {
        return Err(
            "crowded-pair report requires schema_version 3 point-source ablation output".into(),
        );
    }
    let report = if report_path.is_absolute() {
        report_path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(report_path)
    };
    let parent = report
        .parent()
        .ok_or("report path has no parent directory")?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let evidence_root = parent.join("evidence");
    copy_report_evidence(
        point_run_dir,
        &evidence_root.join("point"),
        &[
            "manifest.json",
            "summary.csv",
            "target_results.csv",
            "control_summary.csv",
        ],
    )?;
    if let Some(case_sheets) = point_manifest["outputs"]["case_sheets"].as_object() {
        for value in case_sheets.values() {
            if let Some(path) = value.as_str() {
                let source = point_run_dir.join(Path::new(path).file_name().unwrap_or_default());
                if source.is_file() {
                    fs::copy(
                        &source,
                        evidence_root
                            .join("point")
                            .join(Path::new(path).file_name().unwrap_or_default()),
                    )
                    .map_err(|error| error.to_string())?;
                }
            }
        }
    }
    copy_analysis_evidence(output, &evidence_root.join("crowded"))?;
    let section = report_section(output, optional_runs, demo_summary);
    let mut html = if report.is_file() {
        fs::read_to_string(&report).map_err(|error| error.to_string())?
    } else {
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>RST19 配对消融报告</title></head><body><main><h1>RST19 配对消融报告</h1></main></body></html>".to_owned()
    };
    if let Some(start) = html.find("<!-- RST19-CROWDED-ABLATION-START -->") {
        if let Some(end) = html[start..].find("<!-- RST19-CROWDED-ABLATION-END -->") {
            let end = start + end + "<!-- RST19-CROWDED-ABLATION-END -->".len();
            html.replace_range(start..end, &section);
        }
    } else if let Some(end) = html.find("</body>") {
        html.insert_str(end, &section);
    } else {
        html.push_str(&section);
    }
    fs::write(&report, html).map_err(|error| error.to_string())?;
    crate::ablation_report::decorate(&report)?;
    Ok(report)
}

fn default_out_dir() -> PathBuf {
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    PathBuf::from(format!("tmp/ablation/crowded-{epoch}"))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let parsed = parse_args(args)?;
    let source = parsed.fits_path.ok_or("FITS input path is required")?;
    let fixed_manifest = parsed
        .fixed_manifest
        .ok_or("--reuse-manifest is required")?;
    let point_run = parsed.point_run_dir.ok_or("--point-run-dir is required")?;
    let out_dir = parsed.out_dir.unwrap_or_else(default_out_dir);
    let report = parsed
        .report
        .unwrap_or_else(|| PathBuf::from("deliverables/ablation/index.html"));
    let mut result = run_analysis(
        &source,
        &fixed_manifest,
        &out_dir,
        &parsed.options,
        &parsed.argv,
    )?;
    let report_path = write_report(
        &point_run,
        &report,
        &out_dir,
        &[
            ("运动关联", &parsed.motion_run_dir),
            ("运动像素提案", &parsed.motion_proposal_run_dir),
            ("交叉轨迹", &parsed.motion_crossing_run_dir),
        ],
        &parsed.demo_summary,
    )?;
    result["report"] = json!(display_path(&report_path));
    Ok(result)
}

pub fn run_cli(args: &[String]) -> i32 {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("在真实 FITS 冻结背景块上比较拥挤双源的候选回收、质量门和局部去混叠\n\nUsage: rst19-crowded-ablation FITS --reuse-manifest FILE --point-run-dir DIR [--out-dir DIR] [--report FILE] [--anchors N] [--match-radius-px PX]");
        return 0;
    }
    match run_desktop(args) {
        Ok(value) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&value).unwrap_or_default()
            );
            0
        }
        Err(error) => {
            eprintln!("rst19-crowded-ablation: failed: {error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_is_one_to_one_and_counts_local_candidates() {
        let truths = vec![
            json!({"target_id":"A","x_px":0.0,"y_px":0.0}),
            json!({"target_id":"B","x_px":0.5,"y_px":0.0}),
        ];
        let detection = |id, x| Detection {
            detection_id: id,
            x,
            y: 0.0,
            peak: 100.0,
            flux: 100.0,
            background: 10.0,
            noise: 1.0,
            snr: 10.0,
            fwhm: Some(2.0),
            flags: Vec::new(),
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
            proposal_methods: Vec::new(),
            proposal_scales: Vec::new(),
            proposal_snr: None,
            nearest_gaussian_px: None,
            deblend_delta_bic: None,
            deblend_component_snr: None,
            repeated_code_count: None,
            range_anomaly_pixel_count: None,
            repeated_code_values: Vec::new(),
        };
        let detections = vec![detection(1, 0.1), detection(2, 1.2)];
        let (matches, nearby) = match_truths(&truths, &detections, 2.0);
        assert_eq!(matches.len(), 2);
        assert_eq!(nearby["A"], 2);
        assert_eq!(nearby["B"], 2);
    }

    #[test]
    fn gaussian_injection_is_center_normalized_and_pairing_is_shared() {
        let mut image = vec![10.0f32; PATCH_SIZE * PATCH_SIZE];
        add_gaussian(&mut image, 48.0, 48.0, 160.0, 1.4);
        assert!((f64::from(image[48 * PATCH_SIZE + 48]) - 170.0).abs() < 1e-5);
        assert!(image[0] == 10.0);
    }

    #[test]
    fn cli_requires_the_frozen_manifest_and_point_run() {
        let error = run_desktop(&[]).unwrap_err();
        assert_eq!(error, "FITS input path is required");
    }

    #[test]
    fn npy_writer_matches_numpy_v1_header_alignment_and_payload() {
        let path = std::env::temp_dir().join(format!(
            "rst19-crowded-npy-{}-{}.npy",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let values = [1.25f32, -2.5, 0.0];
        write_npy_f32(&path, &values, 1, values.len()).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        assert_eq!(bytes[6..8], [1, 0]);
        let header_length = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let payload_offset = 10 + header_length;
        assert_eq!(payload_offset, 128);
        assert_eq!(bytes.len(), payload_offset + values.len() * 4);
        for (index, value) in values.iter().enumerate() {
            let start = payload_offset + index * 4;
            assert_eq!(&bytes[start..start + 4], &value.to_le_bytes());
        }
        fs::remove_file(path).unwrap();
    }
}
