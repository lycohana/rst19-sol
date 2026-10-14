//! Frame-by-frame forced photometry for candidates from registered sequences.
//!
//! This is an audit layer only. It does not propose detections, change the
//! quality flag, or identify physical sources.

use crate::{fits, models::FitsFrame, sequence_photometry, Result};
use csv::{ReaderBuilder, Terminator, WriterBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const FEATURE_CLASSES: &[(&str, &str)] = &[
    ("range_anomaly", "数据有效性/范围异常邻域"),
    ("linear_artifact", "线状/拖影候选"),
    ("masked_or_edge", "边缘/掩膜候选"),
    ("crowded_blend", "拥挤/未分辨近邻"),
    ("spike_or_support", "尖峰/PSF 支持不足"),
    ("weak_or_background", "弱通量/背景不确定"),
    ("shape_outlier", "形状异常"),
    ("compact_quality", "质量通过的紧凑候选"),
    ("other_rejected", "其他拒绝候选"),
];

const FRAME_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "frame_index",
    "predicted_x",
    "predicted_y",
    "flux_snr",
    "support_3x3",
    "center_valid",
    "fwhm",
    "ellipticity",
    "quality_like",
    "local_peak_x",
    "local_peak_y",
    "local_peak_flux_snr",
    "local_peak_support_3x3",
    "local_peak_center_valid",
    "local_peak_fwhm",
    "local_peak_ellipticity",
    "local_peak_quality_like",
    "matched_response_snr",
    "local_peak_response_snr",
    "relocalization_dx_px",
    "relocalization_dy_px",
    "relocalization_distance_px",
    "relocalized",
];
const SUMMARY_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "sample_count",
    "frame_count",
    "median_frame_valid_fraction",
    "median_quality_like_presence",
    "quality_like_presence_ge_12_count",
    "median_flux_snr_across_frames",
    "robust_flux_snr_cv_median",
    "median_support_3x3",
    "median_fwhm",
    "median_ellipticity",
    "median_local_peak_quality_like_presence",
    "local_peak_quality_like_presence_ge_12_count",
    "median_local_peak_flux_snr_across_frames",
    "robust_local_peak_flux_snr_cv_median",
    "median_local_peak_support_3x3",
    "median_local_peak_fwhm",
    "median_local_peak_ellipticity",
    "median_matched_response_snr",
    "median_local_peak_response_snr",
    "median_local_peak_flux_gain_fraction",
    "median_modal_relocalization_fraction",
    "median_relocalization_distance_px",
    "median_relocalized_frame_fraction",
    "sample_detection_ids",
];
const PAIR_FRAME_FIELDS: &[&str] = &[
    "detection_id_a",
    "detection_id_b",
    "frame_index",
    "anchor_separation_px",
    "local_separation_px",
    "separation_change_px",
    "a_shift_x_px",
    "a_shift_y_px",
    "b_shift_x_px",
    "b_shift_y_px",
    "a_toward_b_projection_px",
    "b_toward_a_projection_px",
    "both_shift_toward_each_other",
    "local_offset_same",
    "anchor_order_preserved",
];
const PAIR_SUMMARY_FIELDS: &[&str] = &[
    "detection_id_a",
    "detection_id_b",
    "feature_class_a",
    "feature_class_b",
    "frame_count",
    "valid_frame_count",
    "anchor_separation_px",
    "median_local_separation_px",
    "median_separation_change_px",
    "local_separation_contract_fraction",
    "both_shift_toward_each_other_fraction",
    "local_offset_same_fraction",
    "anchor_order_reversal_count",
];

pub const HELP: &str = "对序列候选执行逐帧强制测光稳定性审计\n\nUsage: rst19-forced-stability SOURCE_CATALOG.csv PERSISTENCE.json [OPTIONS]\n\nOptions: --base-dir DIR --max-per-class N --anchor-coordinate peak|centroid\n         --include-detection-id ID (repeatable) --aperture-radius N\n         --min-psf-support-pixels N --min-flux-snr N --min-fwhm N --max-fwhm N\n         --max-ellipticity N --local-peak-search-radius 0|1 --out-dir DIR --quiet";

#[derive(Clone, Debug, Serialize)]
pub struct ForcedAnchor {
    pub detection_id: i64,
    pub feature_class: String,
    pub feature_class_label: String,
    pub x: f64,
    pub y: f64,
    pub flux_snr: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ForcedFrameMetric {
    pub detection_id: i64,
    pub feature_class: String,
    pub feature_class_label: String,
    pub frame_index: usize,
    pub predicted_x: f64,
    pub predicted_y: f64,
    pub flux_snr: Option<f64>,
    pub support_3x3: Option<i16>,
    pub center_valid: bool,
    pub fwhm: Option<f64>,
    pub ellipticity: Option<f64>,
    pub quality_like: bool,
    pub local_peak_x: Option<f64>,
    pub local_peak_y: Option<f64>,
    pub local_peak_flux_snr: Option<f64>,
    pub local_peak_support_3x3: Option<i16>,
    pub local_peak_center_valid: Option<bool>,
    pub local_peak_fwhm: Option<f64>,
    pub local_peak_ellipticity: Option<f64>,
    pub local_peak_quality_like: Option<bool>,
    pub matched_response_snr: Option<f64>,
    pub local_peak_response_snr: Option<f64>,
    pub relocalization_dx_px: Option<f64>,
    pub relocalization_dy_px: Option<f64>,
    pub relocalization_distance_px: Option<f64>,
    pub relocalized: Option<bool>,
}

#[derive(Clone, Debug)]
struct FrameSpec {
    frame_index: usize,
    path: PathBuf,
    background_adu: f64,
    noise_adu: f64,
    shift_x: f64,
    shift_y: f64,
}

#[derive(Clone, Debug)]
struct Options {
    source_catalog: PathBuf,
    persistence_json: PathBuf,
    base_dir: PathBuf,
    max_per_class: usize,
    anchor_coordinate: String,
    include_detection_ids: Vec<i64>,
    aperture_radius: usize,
    min_psf_support_pixels: usize,
    min_flux_snr: f64,
    min_fwhm: f64,
    max_fwhm: f64,
    max_ellipticity: f64,
    local_peak_search_radius: usize,
    local_peak_psf_fwhm: f64,
    out_dir: PathBuf,
    quiet: bool,
}

#[derive(Clone, Debug)]
struct AuditResult {
    source_catalog: PathBuf,
    persistence_json: PathBuf,
    anchor_coordinate: String,
    included_detection_ids: Vec<i64>,
    frame_count: usize,
    anchors: Vec<ForcedAnchor>,
    frame_metrics: Vec<ForcedFrameMetric>,
    summary_rows: Vec<Value>,
    pair_frame_rows: Vec<Value>,
    pair_summary_rows: Vec<Value>,
    options: Options,
}

fn finite_float(value: &str) -> Option<f64> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    value
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
}

fn required_float(value: Option<&Value>, name: &str, context: &str) -> Result<f64> {
    let number = match value {
        Some(Value::Number(number)) => number.as_f64(),
        Some(Value::String(text)) => text.parse::<f64>().ok(),
        _ => None,
    }
    .filter(|number| number.is_finite())
    .ok_or_else(|| format!("{context}: {name} must be finite numeric data"))?;
    Ok(number)
}

fn feature_order(observed: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut classes: Vec<String> = FEATURE_CLASSES
        .iter()
        .map(|(name, _)| (*name).to_owned())
        .collect();
    let known: HashSet<&str> = FEATURE_CLASSES.iter().map(|(name, _)| *name).collect();
    let mut extras: BTreeSet<String> = observed
        .into_iter()
        .filter(|value| !known.contains(value.as_str()))
        .collect();
    classes.extend(extras.iter().cloned());
    extras.clear();
    classes
}

fn feature_label(feature_class: &str, fallback: &str) -> String {
    FEATURE_CLASSES
        .iter()
        .find(|(name, _)| *name == feature_class)
        .map(|(_, label)| (*label).to_owned())
        .unwrap_or_else(|| fallback.to_owned())
}

fn load_anchors(
    path: &Path,
    max_per_class: usize,
    coordinate: &str,
    included: &[i64],
) -> Result<Vec<ForcedAnchor>> {
    if max_per_class == 0 {
        return Err("max_per_class must be positive".into());
    }
    if !matches!(coordinate, "peak" | "centroid") {
        return Err("anchor_coordinate must be 'peak' or 'centroid'".into());
    }
    let mut csv_bytes = fs::read(path)
        .map_err(|e| format!("cannot read source catalog {}: {e}", path.display()))?;
    if csv_bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        csv_bytes.drain(..3);
    }
    let mut reader = ReaderBuilder::new()
        .flexible(true)
        .from_reader(csv_bytes.as_slice());
    let headers = reader.headers().map_err(|e| e.to_string())?.clone();
    let index = |name: &str| headers.iter().position(|field| field == name);
    for required in ["detection_id", "feature_class", "x", "y"] {
        if index(required).is_none() {
            return Err(format!("source catalog is missing fields: {required}"));
        }
    }
    let mut grouped: BTreeMap<String, Vec<ForcedAnchor>> = BTreeMap::new();
    let mut observed = HashSet::new();
    for (row_i, row) in reader.records().enumerate() {
        let row = row.map_err(|e| format!("source catalog row {}: {e}", row_i + 2))?;
        let get = |name: &str| index(name).and_then(|i| row.get(i)).unwrap_or("").trim();
        let feature_class = get("feature_class").to_owned();
        if feature_class.is_empty() {
            return Err(format!(
                "source catalog row {} has empty feature_class",
                row_i + 2
            ));
        }
        let id_text = get("detection_id");
        let detection_id = id_text
            .parse::<i64>()
            .map_err(|_| format!("row {} has invalid detection_id: {id_text:?}", row_i + 2))?;
        observed.insert(detection_id);
        let (primary_x, fallback_x, primary_y, fallback_y) = if coordinate == "peak" {
            ("peak_x", "x", "peak_y", "y")
        } else {
            ("x", "peak_x", "y", "peak_y")
        };
        let x = finite_float(get(primary_x))
            .or_else(|| finite_float(get(fallback_x)))
            .ok_or_else(|| format!("row {} has no finite {primary_x}/{fallback_x}", row_i + 2))?;
        let y = finite_float(get(primary_y))
            .or_else(|| finite_float(get(fallback_y)))
            .ok_or_else(|| format!("row {} has no finite {primary_y}/{fallback_y}", row_i + 2))?;
        let label = get("feature_class_label");
        grouped
            .entry(feature_class.clone())
            .or_default()
            .push(ForcedAnchor {
                detection_id,
                feature_class: feature_class.clone(),
                feature_class_label: if label.is_empty() {
                    feature_label(&feature_class, &feature_class)
                } else {
                    label.to_owned()
                },
                x,
                y,
                flux_snr: finite_float(get("flux_snr")),
            });
    }
    let included_set: HashSet<i64> = included.iter().copied().collect();
    let missing: Vec<String> = included_set
        .difference(&observed)
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|id| id.to_string())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog does not contain included detection_id(s): {}",
            missing.join(", ")
        ));
    }
    let classes = feature_order(grouped.keys().cloned());
    let mut selected = Vec::new();
    for feature_class in classes {
        let Some(mut members) = grouped.remove(&feature_class) else {
            continue;
        };
        members.sort_by(|a, b| {
            a.flux_snr
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&b.flux_snr.unwrap_or(f64::NEG_INFINITY))
                .then(a.detection_id.cmp(&b.detection_id))
        });
        if members.len() > max_per_class {
            let mut keep = HashSet::new();
            let step = if max_per_class == 1 {
                0.0
            } else {
                (members.len() - 1) as f64 / (max_per_class - 1) as f64
            };
            for i in 0..max_per_class {
                let fraction = if max_per_class == 1 {
                    0.0
                } else {
                    i as f64 * step
                };
                let index = if max_per_class > 1 && i + 1 == max_per_class {
                    members.len() - 1
                } else {
                    fraction.round_ties_even() as usize
                };
                keep.insert(index);
            }
            selected.extend(members.into_iter().enumerate().filter_map(|(i, member)| {
                (keep.contains(&i) || included_set.contains(&member.detection_id)).then_some(member)
            }));
        } else {
            selected.extend(members);
        }
    }
    Ok(selected)
}

fn resolve_path(text: &str, base_dir: &Path) -> PathBuf {
    let path = PathBuf::from(text);
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

fn load_frame_specs(path: &Path, base_dir: &Path) -> Result<(usize, Vec<FrameSpec>)> {
    let payload: Value = serde_json::from_slice(&fs::read(path).map_err(|e| {
        format!(
            "persistence JSON does not exist or cannot be read: {}: {e}",
            path.display()
        )
    })?)
    .map_err(|e| format!("invalid persistence JSON: {e}"))?;
    let frame_count = payload
        .get("frame_count")
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str().and_then(|text| text.parse::<u64>().ok()))
        })
        .unwrap_or(0) as usize;
    if frame_count == 0 {
        return Err("persistence JSON has invalid frame_count".into());
    }
    let shifts = payload
        .get("cumulative_shifts")
        .and_then(Value::as_array)
        .ok_or("cumulative_shifts must be an array")?;
    if shifts.len() != frame_count {
        return Err("cumulative_shifts length must match frame_count".into());
    }
    let mut parsed_shifts = Vec::with_capacity(frame_count);
    for (i, shift) in shifts.iter().enumerate() {
        let pair = shift
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| format!("cumulative_shifts[{}] must contain x/y", i + 1))?;
        parsed_shifts.push((
            required_float(
                pair.first(),
                "shift_x",
                &format!("cumulative_shifts[{}]", i + 1),
            )?,
            required_float(
                pair.get(1),
                "shift_y",
                &format!("cumulative_shifts[{}]", i + 1),
            )?,
        ));
    }
    let rows = payload
        .get("frame_rows")
        .and_then(Value::as_array)
        .ok_or("persistence JSON is missing frame_rows")?;
    let mut by_index: BTreeMap<usize, FrameSpec> = BTreeMap::new();
    for row in rows {
        if !row.is_object() {
            continue;
        }
        let frame_index =
            row.get("frame_index")
                .and_then(Value::as_u64)
                .ok_or("frame_rows contains an invalid frame_index")? as usize;
        if frame_index == 0 || frame_index > frame_count {
            return Err(format!(
                "frame_rows contains out-of-range frame_index={frame_index}"
            ));
        }
        let path_text = row.get("path").and_then(Value::as_str).unwrap_or("");
        let frame_path = resolve_path(path_text, base_dir);
        let context = format!("frame {frame_index}");
        let spec = FrameSpec {
            frame_index,
            path: frame_path,
            background_adu: required_float(row.get("background_adu"), "background_adu", &context)?,
            noise_adu: required_float(row.get("noise_adu"), "noise_adu", &context)?
                .max(f64::EPSILON),
            shift_x: parsed_shifts[frame_index - 1].0,
            shift_y: parsed_shifts[frame_index - 1].1,
        };
        if let Some(previous) = by_index.get(&frame_index) {
            if previous.path != spec.path
                || (previous.background_adu - spec.background_adu).abs()
                    > 1e-8 + 1e-5 * previous.background_adu.abs()
                || (previous.noise_adu - spec.noise_adu).abs()
                    > 1e-8 + 1e-5 * previous.noise_adu.abs()
            {
                return Err(format!(
                    "frame_rows has inconsistent duplicate frame {frame_index}"
                ));
            }
        } else {
            by_index.insert(frame_index, spec);
        }
    }
    let missing: Vec<String> = (1..=frame_count)
        .filter(|index| !by_index.contains_key(index))
        .map(|index| index.to_string())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "frame_rows is missing frame(s): {}",
            missing.join(", ")
        ));
    }
    Ok((frame_count, by_index.into_values().collect()))
}

fn median(values: &[f64]) -> Option<f64> {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(f64::total_cmp);
    let n = finite.len();
    Some(if n % 2 == 0 {
        (finite[n / 2 - 1] + finite[n / 2]) * 0.5
    } else {
        finite[n / 2]
    })
}

fn robust_cv(values: &[f64]) -> Option<f64> {
    let center = median(values)?;
    let deviations: Vec<f64> = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v - center).abs())
        .collect();
    let scale = 1.4826 * median(&deviations)?;
    (center.abs() > f64::EPSILON).then_some(scale / center.abs())
}

fn local_offsets(radius: usize) -> Result<Vec<(i32, i32)>> {
    if radius > 1 {
        return Err("local_peak_search_radius must be 0 or 1 px".into());
    }
    let mut offsets = vec![(0, 0)];
    for dy in -(radius as i32)..=(radius as i32) {
        for dx in -(radius as i32)..=(radius as i32) {
            if dx != 0 || dy != 0 {
                offsets.push((dx, dy));
            }
        }
    }
    Ok(offsets)
}

fn select_local_peak(values: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    let mut found = false;
    for (index, value) in values.iter().enumerate() {
        if value.is_finite() && (!found || *value > best_value) {
            best = index;
            best_value = *value;
            found = true;
        }
    }
    best
}

fn gaussian_constant(
    values: &[f64],
    valid: &[bool],
    height: usize,
    width: usize,
    sigma: f64,
) -> (Vec<f64>, Vec<f64>) {
    let radius = (4.0 * sigma + 0.5).floor() as isize;
    let mut kernel: Vec<f64> = (-radius..=radius)
        .map(|offset| (-0.5 * (offset as f64 / sigma).powi(2)).exp())
        .collect();
    let sum = kernel.iter().sum::<f64>();
    for value in &mut kernel {
        *value /= sum;
    }
    let pass = |input: &[f64]| {
        let mut out = vec![0.0; input.len()];
        for y in 0..height {
            for x in 0..width {
                let mut value = 0.0;
                for (k, weight) in kernel.iter().enumerate() {
                    let d = k as isize - radius;
                    let yy = y as isize + d;
                    if yy >= 0 && yy < height as isize {
                        value += input[yy as usize * width + x] * weight;
                    }
                }
                out[y * width + x] = value;
            }
        }
        let mut second = vec![0.0; input.len()];
        for y in 0..height {
            for x in 0..width {
                let mut value = 0.0;
                for (k, weight) in kernel.iter().enumerate() {
                    let d = k as isize - radius;
                    let xx = x as isize + d;
                    if xx >= 0 && xx < width as isize {
                        value += out[y * width + xx as usize] * weight;
                    }
                }
                second[y * width + x] = value;
            }
        }
        second
    };
    let residual: Vec<f64> = values
        .iter()
        .zip(valid)
        .map(|(value, ok)| if *ok { *value } else { 0.0 })
        .collect();
    let weights: Vec<f64> = valid.iter().map(|ok| if *ok { 1.0 } else { 0.0 }).collect();
    (pass(&residual), pass(&weights))
}

fn bilinear(map: &[f64], height: usize, width: usize, x: f64, y: f64) -> f64 {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x > (width - 1) as f64
        || y > (height - 1) as f64
    {
        return f64::NEG_INFINITY;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let dx = x - x0 as f64;
    let dy = y - y0 as f64;
    let a = map[y0 * width + x0] * (1.0 - dx) + map[y0 * width + x1] * dx;
    let b = map[y1 * width + x0] * (1.0 - dx) + map[y1 * width + x1] * dx;
    a * (1.0 - dy) + b * dy
}

fn local_responses(
    frame: &FitsFrame,
    image: &[f64],
    mask: &[bool],
    xs: &[f64],
    ys: &[f64],
    bg: f64,
    noise: f64,
    psf_fwhm: f64,
) -> Result<Vec<f64>> {
    let height = frame.height;
    let width = frame.width;
    if xs.len() != ys.len() {
        return Err("xs and ys lengths must match".into());
    }
    let valid: Vec<bool> = image
        .iter()
        .zip(mask)
        .map(|(v, m)| !*m && v.is_finite())
        .collect();
    let mut response_map = vec![f64::NEG_INFINITY; image.len()];
    let scales = [0.65, 1.0, 1.35].map(|factor| (psf_fwhm * factor / 2.35482).max(0.5));
    let safe_noise = noise.max(f64::EPSILON);
    for sigma in scales {
        let residual: Vec<f64> = image
            .iter()
            .map(|v| if v.is_finite() { *v - bg } else { 0.0 })
            .collect();
        let (filtered, coverage) = gaussian_constant(&residual, &valid, height, width, sigma);
        let radius = (4.0 * sigma).ceil() as isize;
        let mut one_d: Vec<f64> = (-radius..=radius)
            .map(|offset| (-0.5 * (offset as f64 / sigma).powi(2)).exp())
            .collect();
        let norm = one_d.iter().sum::<f64>();
        for value in &mut one_d {
            *value /= norm;
        }
        let kernel_l2 = one_d.iter().map(|v| v * v).sum::<f64>().max(f64::EPSILON);
        for i in 0..image.len() {
            if coverage[i] < 0.5 {
                continue;
            }
            let value = filtered[i] / (safe_noise * kernel_l2 * coverage[i].sqrt());
            if value > response_map[i] {
                response_map[i] = value;
            }
        }
    }
    Ok(xs
        .iter()
        .zip(ys)
        .map(|(x, y)| bilinear(&response_map, height, width, *x, *y))
        .collect())
}

fn measurement_quality(
    flux: Option<f64>,
    support: Option<i16>,
    center_valid: bool,
    fwhm: Option<f64>,
    ellipticity: Option<f64>,
    options: &Options,
) -> bool {
    center_valid
        && flux.is_some_and(|v| v >= options.min_flux_snr)
        && support.is_some_and(|v| v as usize >= options.min_psf_support_pixels)
        && fwhm.is_some_and(|v| v >= options.min_fwhm && v <= options.max_fwhm)
        && ellipticity.is_some_and(|v| v <= options.max_ellipticity)
}

fn median_value(values: &[f64]) -> Value {
    median(values).map(Value::from).unwrap_or(Value::Null)
}

fn summaries(
    anchors: &[ForcedAnchor],
    metrics: &[ForcedFrameMetric],
    frame_count: usize,
) -> Result<Vec<Value>> {
    let mut by_id: HashMap<i64, Vec<&ForcedFrameMetric>> = HashMap::new();
    let mut by_class: BTreeMap<String, Vec<&ForcedAnchor>> = BTreeMap::new();
    for anchor in anchors {
        by_class
            .entry(anchor.feature_class.clone())
            .or_default()
            .push(anchor);
    }
    for metric in metrics {
        by_id.entry(metric.detection_id).or_default().push(metric);
    }
    let mut rows = Vec::new();
    for feature_class in feature_order(by_class.keys().cloned()) {
        let Some(class_anchors) = by_class.get(&feature_class) else {
            continue;
        };
        let mut valid_fraction = Vec::new();
        let mut quality_presence = Vec::new();
        let mut flux = Vec::new();
        let mut cv = Vec::new();
        let mut support = Vec::new();
        let mut fwhm = Vec::new();
        let mut ellipticity = Vec::new();
        let mut local_quality_presence = Vec::new();
        let mut local_flux = Vec::new();
        let mut local_cv = Vec::new();
        let mut local_support = Vec::new();
        let mut local_fwhm = Vec::new();
        let mut local_ellipticity = Vec::new();
        let mut response = Vec::new();
        let mut local_response = Vec::new();
        let mut local_gain = Vec::new();
        let mut modal_relocalization = Vec::new();
        let mut relocation_distance = Vec::new();
        let mut relocalized_fraction = Vec::new();
        for anchor in class_anchors {
            let mut anchor_metrics = by_id.get(&anchor.detection_id).cloned().unwrap_or_default();
            anchor_metrics.sort_by_key(|metric| metric.frame_index);
            if anchor_metrics.len() != frame_count {
                return Err(format!(
                    "anchor {} has {} frame metrics; expected {frame_count}",
                    anchor.detection_id,
                    anchor_metrics.len()
                ));
            }
            let values: Vec<f64> = anchor_metrics.iter().filter_map(|m| m.flux_snr).collect();
            if let Some(value) = median(&values) {
                flux.push(value);
            }
            if let Some(value) = robust_cv(&values) {
                cv.push(value);
            }
            let s: Vec<f64> = anchor_metrics
                .iter()
                .filter_map(|m| m.support_3x3.map(|v| v as f64))
                .collect();
            if let Some(v) = median(&s) {
                support.push(v);
            }
            let fw: Vec<f64> = anchor_metrics.iter().filter_map(|m| m.fwhm).collect();
            if let Some(v) = median(&fw) {
                fwhm.push(v);
            }
            let el: Vec<f64> = anchor_metrics
                .iter()
                .filter_map(|m| m.ellipticity)
                .collect();
            if let Some(v) = median(&el) {
                ellipticity.push(v);
            }
            valid_fraction.push(
                anchor_metrics.iter().filter(|m| m.center_valid).count() as f64
                    / frame_count as f64,
            );
            let q = anchor_metrics.iter().filter(|m| m.quality_like).count();
            quality_presence.push(q as f64);
            let local: Vec<f64> = anchor_metrics
                .iter()
                .filter_map(|m| m.local_peak_flux_snr.or(m.flux_snr))
                .collect();
            if let Some(v) = median(&local) {
                local_flux.push(v);
            }
            if let Some(v) = robust_cv(&local) {
                local_cv.push(v);
            }
            let ls: Vec<f64> = anchor_metrics
                .iter()
                .filter_map(|m| m.local_peak_support_3x3.or(m.support_3x3).map(|v| v as f64))
                .collect();
            if let Some(v) = median(&ls) {
                local_support.push(v);
            }
            let lf: Vec<f64> = anchor_metrics
                .iter()
                .filter_map(|m| m.local_peak_fwhm.or(m.fwhm))
                .collect();
            if let Some(v) = median(&lf) {
                local_fwhm.push(v);
            }
            let le: Vec<f64> = anchor_metrics
                .iter()
                .filter_map(|m| m.local_peak_ellipticity.or(m.ellipticity))
                .collect();
            if let Some(v) = median(&le) {
                local_ellipticity.push(v);
            }
            let lq = anchor_metrics
                .iter()
                .filter(|m| m.local_peak_quality_like.unwrap_or(m.quality_like))
                .count();
            local_quality_presence.push(lq as f64);
            let rv: Vec<f64> = anchor_metrics
                .iter()
                .filter_map(|m| m.matched_response_snr)
                .collect();
            if let Some(v) = median(&rv) {
                response.push(v);
            }
            let lrv: Vec<f64> = anchor_metrics
                .iter()
                .filter_map(|m| m.local_peak_response_snr.or(m.matched_response_snr))
                .collect();
            if let Some(v) = median(&lrv) {
                local_response.push(v);
            }
            if let Some(fixed) = median(&values).filter(|v| *v > f64::EPSILON) {
                if let Some(local) = median(&local) {
                    local_gain.push(local / fixed - 1.0);
                }
            }
            let mut offset_counts: HashMap<(i64, i64), usize> = HashMap::new();
            for m in &anchor_metrics {
                let dx = (m.relocalization_dx_px.unwrap_or(0.0) * 1e6).round_ties_even() as i64;
                let dy = (m.relocalization_dy_px.unwrap_or(0.0) * 1e6).round_ties_even() as i64;
                *offset_counts.entry((dx, dy)).or_default() += 1;
            }
            modal_relocalization.push(
                offset_counts.values().copied().max().unwrap_or(0) as f64 / frame_count as f64,
            );
            let distances: Vec<f64> = anchor_metrics
                .iter()
                .map(|m| m.relocalization_distance_px.unwrap_or(0.0))
                .collect();
            if let Some(v) = median(&distances) {
                relocation_distance.push(v);
            }
            relocalized_fraction.push(
                anchor_metrics
                    .iter()
                    .filter(|m| m.relocalized.unwrap_or(false))
                    .count() as f64
                    / frame_count as f64,
            );
        }
        let label = feature_label(&feature_class, &class_anchors[0].feature_class_label);
        let sample_ids = class_anchors
            .iter()
            .map(|a| a.detection_id.to_string())
            .collect::<Vec<_>>()
            .join("|");
        rows.push(json!({
            "feature_class":feature_class,"feature_class_label":label,"sample_count":class_anchors.len(),"frame_count":frame_count,
            "median_frame_valid_fraction":median_value(&valid_fraction),"median_quality_like_presence":median_value(&quality_presence),
            "quality_like_presence_ge_12_count":quality_presence.iter().filter(|v|**v>=12.0).count(),"median_flux_snr_across_frames":median_value(&flux),
            "robust_flux_snr_cv_median":median_value(&cv),"median_support_3x3":median_value(&support),"median_fwhm":median_value(&fwhm),"median_ellipticity":median_value(&ellipticity),
            "median_local_peak_quality_like_presence":median_value(&local_quality_presence),"local_peak_quality_like_presence_ge_12_count":local_quality_presence.iter().filter(|v|**v>=12.0).count(),
            "median_local_peak_flux_snr_across_frames":median_value(&local_flux),"robust_local_peak_flux_snr_cv_median":median_value(&local_cv),"median_local_peak_support_3x3":median_value(&local_support),
            "median_local_peak_fwhm":median_value(&local_fwhm),"median_local_peak_ellipticity":median_value(&local_ellipticity),"median_matched_response_snr":median_value(&response),
            "median_local_peak_response_snr":median_value(&local_response),"median_local_peak_flux_gain_fraction":median_value(&local_gain),"median_modal_relocalization_fraction":median_value(&modal_relocalization),
            "median_relocalization_distance_px":median_value(&relocation_distance),"median_relocalized_frame_fraction":median_value(&relocalized_fraction),"sample_detection_ids":sample_ids
        }));
    }
    Ok(rows)
}

fn pair_frame_rows(
    anchors: &[ForcedAnchor],
    metrics: &[ForcedFrameMetric],
    ids: &[i64],
    frame_count: usize,
) -> Result<Vec<Value>> {
    let mut ids = ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() < 2 {
        return Ok(Vec::new());
    }
    let by_id: HashMap<i64, &ForcedAnchor> = anchors.iter().map(|a| (a.detection_id, a)).collect();
    let by_frame: HashMap<(i64, usize), &ForcedFrameMetric> = metrics
        .iter()
        .map(|m| ((m.detection_id, m.frame_index), m))
        .collect();
    let mut rows = Vec::new();
    for i in 0..ids.len() {
        for j in i + 1..ids.len() {
            let aid = ids[i];
            let bid = ids[j];
            let a = by_id
                .get(&aid)
                .ok_or_else(|| format!("pair anchor missing for detection_id(s) {aid}, {bid}"))?;
            let b = by_id
                .get(&bid)
                .ok_or_else(|| format!("pair anchor missing for detection_id(s) {aid}, {bid}"))?;
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let separation = dx.hypot(dy);
            if separation <= f64::EPSILON {
                return Err(format!(
                    "pair anchors {aid}, {bid} have identical coordinates"
                ));
            }
            let ux = dx / separation;
            let uy = dy / separation;
            for frame in 1..=frame_count {
                let ma = by_frame
                    .get(&(aid, frame))
                    .ok_or_else(|| format!("pair {aid}/{bid} is missing frame {frame}"))?;
                let mb = by_frame
                    .get(&(bid, frame))
                    .ok_or_else(|| format!("pair {aid}/{bid} is missing frame {frame}"))?;
                let ax = ma.local_peak_x.unwrap_or(ma.predicted_x);
                let ay = ma.local_peak_y.unwrap_or(ma.predicted_y);
                let bx = mb.local_peak_x.unwrap_or(mb.predicted_x);
                let by = mb.local_peak_y.unwrap_or(mb.predicted_y);
                let finite = ax.is_finite() && ay.is_finite() && bx.is_finite() && by.is_finite();
                let (local_sep, change, asx, asy, bsx, bsy, proj_a, proj_b, inward, same, order) =
                    if finite {
                        let vx = bx - ax;
                        let vy = by - ay;
                        let sep = vx.hypot(vy);
                        let asx = ax - ma.predicted_x;
                        let asy = ay - ma.predicted_y;
                        let bsx = bx - mb.predicted_x;
                        let bsy = by - mb.predicted_y;
                        let pa = asx * ux + asy * uy;
                        let pb = bsx * (-ux) + bsy * (-uy);
                        (
                            Some(sep),
                            Some(sep - separation),
                            Some(asx),
                            Some(asy),
                            Some(bsx),
                            Some(bsy),
                            Some(pa),
                            Some(pb),
                            pa > 0.0 && pb > 0.0,
                            (asx - bsx).abs() <= 1e-6 && (asy - bsy).abs() <= 1e-6,
                            Some(vx * ux + vy * uy > 0.0),
                        )
                    } else {
                        (
                            None, None, None, None, None, None, None, None, false, false, None,
                        )
                    };
                rows.push(json!({"detection_id_a":aid,"detection_id_b":bid,"frame_index":frame,"anchor_separation_px":separation,"local_separation_px":local_sep,"separation_change_px":change,"a_shift_x_px":asx,"a_shift_y_px":asy,"b_shift_x_px":bsx,"b_shift_y_px":bsy,"a_toward_b_projection_px":proj_a,"b_toward_a_projection_px":proj_b,"both_shift_toward_each_other":inward,"local_offset_same":same,"anchor_order_preserved":order}));
            }
        }
    }
    Ok(rows)
}

fn pair_summaries(
    anchors: &[ForcedAnchor],
    frame_rows: &[Value],
    ids: &[i64],
    frame_count: usize,
) -> Vec<Value> {
    let mut ids = ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() < 2 {
        return Vec::new();
    }
    let by_id: HashMap<i64, &ForcedAnchor> = anchors.iter().map(|a| (a.detection_id, a)).collect();
    let mut rows = Vec::new();
    for i in 0..ids.len() {
        for j in i + 1..ids.len() {
            let aid = ids[i];
            let bid = ids[j];
            let pair: Vec<&Value> = frame_rows
                .iter()
                .filter(|r| r["detection_id_a"] == aid && r["detection_id_b"] == bid)
                .collect();
            let valid: Vec<&Value> = pair
                .iter()
                .copied()
                .filter(|r| !r["local_separation_px"].is_null())
                .collect();
            let sep: Vec<f64> = valid
                .iter()
                .filter_map(|r| r["local_separation_px"].as_f64())
                .collect();
            let change: Vec<f64> = valid
                .iter()
                .filter_map(|r| r["separation_change_px"].as_f64())
                .collect();
            let anchor_sep = pair
                .first()
                .and_then(|r| r["anchor_separation_px"].as_f64())
                .unwrap_or(0.0);
            let fraction = |key: &str| {
                if valid.is_empty() {
                    Value::Null
                } else {
                    Value::from(
                        valid.iter().filter(|r| r[key] == true).count() as f64 / valid.len() as f64,
                    )
                }
            };
            let reversals = valid
                .iter()
                .filter(|r| r["anchor_order_preserved"] == false)
                .count();
            rows.push(json!({"detection_id_a":aid,"detection_id_b":bid,"feature_class_a":by_id[&aid].feature_class,"feature_class_b":by_id[&bid].feature_class,"frame_count":frame_count,"valid_frame_count":valid.len(),"anchor_separation_px":anchor_sep,"median_local_separation_px":median_value(&sep),"median_separation_change_px":median_value(&change),"local_separation_contract_fraction":if change.is_empty(){Value::Null}else{Value::from(change.iter().filter(|v|**v<0.0).count() as f64/change.len() as f64)},"both_shift_toward_each_other_fraction":fraction("both_shift_toward_each_other"),"local_offset_same_fraction":fraction("local_offset_same"),"anchor_order_reversal_count":reversals}));
        }
    }
    rows
}

fn parse_options(args: &[String]) -> Result<Options> {
    let mut positional = Vec::new();
    let mut base_dir = PathBuf::from(".");
    let mut max_per_class = 32usize;
    let mut anchor_coordinate = "peak".to_owned();
    let mut include_detection_ids = Vec::new();
    let mut aperture_radius = 4usize;
    let mut min_psf_support_pixels = 3usize;
    let mut min_flux_snr = 5.0f64;
    let mut min_fwhm = 0.8f64;
    let mut max_fwhm = 12.0f64;
    let mut max_ellipticity = 0.65f64;
    let mut local_peak_search_radius = 1usize;
    let local_peak_psf_fwhm = 3.0f64;
    let mut out_dir = PathBuf::from("tmp/forced-stability-audit");
    let mut quiet = false;
    let mut i = 0usize;
    while i < args.len() {
        let arg = &args[i];
        if !arg.starts_with('-') {
            positional.push(PathBuf::from(arg));
            i += 1;
            continue;
        }
        if arg == "--quiet" {
            quiet = true;
            i += 1;
            continue;
        }
        if arg == "-h" || arg == "--help" {
            return Err(HELP.to_owned());
        }
        let (name, inline) = arg[2..]
            .split_once('=')
            .map(|(n, v)| (n, Some(v.to_owned())))
            .unwrap_or((&arg[2..], None));
        i += 1;
        let mut value = || -> Result<String> {
            if let Some(v) = &inline {
                return Ok(v.clone());
            }
            let v = args
                .get(i)
                .filter(|v| !v.starts_with('-'))
                .cloned()
                .ok_or_else(|| format!("--{name} requires a value"))?;
            i += 1;
            Ok(v)
        };
        match name {
            "base-dir" => base_dir = PathBuf::from(value()?),
            "max-per-class" => {
                max_per_class = value()?.parse().map_err(|_| "invalid --max-per-class")?
            }
            "anchor-coordinate" => anchor_coordinate = value()?,
            "include-detection-id" => include_detection_ids.push(
                value()?
                    .parse()
                    .map_err(|_| "invalid --include-detection-id")?,
            ),
            "aperture-radius" => {
                aperture_radius = value()?.parse().map_err(|_| "invalid --aperture-radius")?
            }
            "min-psf-support-pixels" => {
                min_psf_support_pixels = value()?
                    .parse()
                    .map_err(|_| "invalid --min-psf-support-pixels")?
            }
            "min-flux-snr" => {
                min_flux_snr = value()?.parse().map_err(|_| "invalid --min-flux-snr")?
            }
            "min-fwhm" => min_fwhm = value()?.parse().map_err(|_| "invalid --min-fwhm")?,
            "max-fwhm" => max_fwhm = value()?.parse().map_err(|_| "invalid --max-fwhm")?,
            "max-ellipticity" => {
                max_ellipticity = value()?.parse().map_err(|_| "invalid --max-ellipticity")?
            }
            "local-peak-search-radius" => {
                local_peak_search_radius = value()?
                    .parse()
                    .map_err(|_| "invalid --local-peak-search-radius")?
            }
            "out-dir" => out_dir = PathBuf::from(value()?),
            _ => return Err(format!("unknown option: {arg}")),
        }
    }
    if positional.len() != 2 {
        return Err(
            "usage: rst19-forced-stability SOURCE_CATALOG.csv PERSISTENCE.json [OPTIONS]".into(),
        );
    }
    if aperture_radius < 1 || min_psf_support_pixels < 1 || max_per_class < 1 {
        return Err(
            "aperture_radius, min_psf_support_pixels and max_per_class must be positive".into(),
        );
    }
    if !min_flux_snr.is_finite()
        || !min_fwhm.is_finite()
        || !max_fwhm.is_finite()
        || !max_ellipticity.is_finite()
        || min_flux_snr <= 0.0
        || min_fwhm <= 0.0
        || max_fwhm < min_fwhm
        || max_ellipticity <= 0.0
    {
        return Err("invalid forced-stability quality-style thresholds".into());
    }
    if !local_peak_psf_fwhm.is_finite() || local_peak_psf_fwhm <= 0.0 {
        return Err("local_peak_psf_fwhm must be positive".into());
    }
    local_offsets(local_peak_search_radius)?;
    if !matches!(anchor_coordinate.as_str(), "peak" | "centroid") {
        return Err("anchor_coordinate must be 'peak' or 'centroid'".into());
    }
    Ok(Options {
        source_catalog: positional[0].clone(),
        persistence_json: positional[1].clone(),
        base_dir,
        max_per_class,
        anchor_coordinate,
        include_detection_ids,
        aperture_radius,
        min_psf_support_pixels,
        min_flux_snr,
        min_fwhm,
        max_fwhm,
        max_ellipticity,
        local_peak_search_radius,
        local_peak_psf_fwhm,
        out_dir,
        quiet,
    })
}

fn run(options: Options) -> Result<AuditResult> {
    if !options.source_catalog.is_file() {
        return Err(format!(
            "source catalog does not exist: {}",
            options.source_catalog.display()
        ));
    }
    if !options.persistence_json.is_file() {
        return Err(format!(
            "persistence JSON does not exist: {}",
            options.persistence_json.display()
        ));
    }
    let anchors = load_anchors(
        &options.source_catalog,
        options.max_per_class,
        &options.anchor_coordinate,
        &options.include_detection_ids,
    )?;
    if anchors.is_empty() {
        return Err("source catalog has no anchors".into());
    }
    let (frame_count, frames) = load_frame_specs(&options.persistence_json, &options.base_dir)?;
    let offsets = local_offsets(options.local_peak_search_radius)?;
    let mut metrics = Vec::with_capacity(anchors.len() * frame_count);
    let anchor_x: Vec<f64> = anchors.iter().map(|a| a.x).collect();
    let anchor_y: Vec<f64> = anchors.iter().map(|a| a.y).collect();
    for spec in &frames {
        let frame = fits::read_fits(&spec.path, false)?;
        let image = frame.data.to_f64();
        let mask = fits::auxiliary_mask(frame.height, frame.width)?;
        let xs: Vec<f64> = anchor_x.iter().map(|x| x + spec.shift_x).collect();
        let ys: Vec<f64> = anchor_y.iter().map(|y| y + spec.shift_y).collect();
        let mut window_x = Vec::with_capacity(anchors.len() * offsets.len());
        let mut window_y = Vec::with_capacity(window_x.capacity());
        for (x, y) in xs.iter().zip(&ys) {
            for (dx, dy) in &offsets {
                window_x.push(*x + *dx as f64);
                window_y.push(*y + *dy as f64);
            }
        }
        let measurements = sequence_photometry::stack_forced_measure_batch(
            sequence_photometry::FramePixels {
                image: &image,
                mask: &mask,
                height: frame.height,
                width: frame.width,
            },
            &window_x,
            &window_y,
            sequence_photometry::StackMeasurementOptions {
                aperture_radius: options.aperture_radius,
                global_background: spec.background_adu,
                global_noise: spec.noise_adu,
            },
        )?;
        let responses = local_responses(
            &frame,
            &image,
            &mask,
            &window_x,
            &window_y,
            spec.background_adu,
            spec.noise_adu,
            options.local_peak_psf_fwhm,
        )?;
        for (anchor_index, anchor) in anchors.iter().enumerate() {
            let begin = anchor_index * offsets.len();
            let end = begin + offsets.len();
            let local_index = select_local_peak(&responses[begin..end]);
            let fixed = measurements[begin];
            let local = measurements[begin + local_index];
            let (fixed_flux, fixed_support, fixed_valid, fixed_fwhm, fixed_ellipticity) = fixed;
            let (local_flux, local_support, local_valid, local_fwhm, local_ellipticity) = local;
            let fixed_response = responses[begin];
            let local_response = responses[begin + local_index];
            let (dx, dy) = offsets[local_index];
            metrics.push(ForcedFrameMetric {
                detection_id: anchor.detection_id,
                feature_class: anchor.feature_class.clone(),
                feature_class_label: anchor.feature_class_label.clone(),
                frame_index: spec.frame_index,
                predicted_x: xs[anchor_index],
                predicted_y: ys[anchor_index],
                flux_snr: fixed_flux,
                support_3x3: fixed_valid.then_some(fixed_support),
                center_valid: fixed_valid,
                fwhm: fixed_fwhm,
                ellipticity: fixed_ellipticity,
                quality_like: measurement_quality(
                    fixed_flux,
                    fixed_valid.then_some(fixed_support),
                    fixed_valid,
                    fixed_fwhm,
                    fixed_ellipticity,
                    &options,
                ),
                local_peak_x: Some(xs[anchor_index] + dx as f64),
                local_peak_y: Some(ys[anchor_index] + dy as f64),
                local_peak_flux_snr: local_flux,
                local_peak_support_3x3: local_valid.then_some(local_support),
                local_peak_center_valid: Some(local_valid),
                local_peak_fwhm: local_fwhm,
                local_peak_ellipticity: local_ellipticity,
                local_peak_quality_like: Some(measurement_quality(
                    local_flux,
                    local_valid.then_some(local_support),
                    local_valid,
                    local_fwhm,
                    local_ellipticity,
                    &options,
                )),
                matched_response_snr: fixed_response.is_finite().then_some(fixed_response),
                local_peak_response_snr: local_response.is_finite().then_some(local_response),
                relocalization_dx_px: Some(dx as f64),
                relocalization_dy_px: Some(dy as f64),
                relocalization_distance_px: Some((dx as f64).hypot(dy as f64)),
                relocalized: Some(local_index != 0),
            });
        }
        if !options.quiet {
            eprintln!(
                "forced stability: frame {}/{}",
                spec.frame_index, frame_count
            );
        }
    }
    let summary_rows = summaries(&anchors, &metrics, frame_count)?;
    let pair_frame_rows = pair_frame_rows(
        &anchors,
        &metrics,
        &options.include_detection_ids,
        frame_count,
    )?;
    let pair_summary_rows = pair_summaries(
        &anchors,
        &pair_frame_rows,
        &options.include_detection_ids,
        frame_count,
    );
    Ok(AuditResult {
        source_catalog: options.source_catalog.clone(),
        persistence_json: options.persistence_json.clone(),
        anchor_coordinate: options.anchor_coordinate.clone(),
        included_detection_ids: {
            let mut ids = options.include_detection_ids.clone();
            ids.sort_unstable();
            ids.dedup();
            ids
        },
        frame_count,
        anchors,
        frame_metrics: metrics,
        summary_rows,
        pair_frame_rows,
        pair_summary_rows,
        options,
    })
}

fn csv_field(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(v) => v.clone(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.to_string(),
        _ => value.to_string(),
    }
}

fn write_value_csv(path: &Path, fields: &[&str], rows: &[Value]) -> Result<()> {
    let mut file =
        File::create(path).map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    file.write_all(&[0xEF, 0xBB, 0xBF])
        .map_err(|e| e.to_string())?;
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .terminator(Terminator::CRLF)
        .from_writer(file);
    writer.write_record(fields).map_err(|e| e.to_string())?;
    for row in rows {
        let values: Vec<String> = fields.iter().map(|field| csv_field(&row[*field])).collect();
        writer.write_record(values).map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn write_artifacts(result: &AuditResult) -> Result<(PathBuf, Value)> {
    fs::create_dir_all(&result.options.out_dir).map_err(|e| e.to_string())?;
    let output = result
        .options
        .out_dir
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let frame_rows: Vec<Value> = result
        .frame_metrics
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| e.to_string())?;
    write_value_csv(
        &output.join("forced_stability_frame_metrics.csv"),
        FRAME_FIELDS,
        &frame_rows,
    )?;
    write_value_csv(
        &output.join("forced_stability_summary.csv"),
        SUMMARY_FIELDS,
        &result.summary_rows,
    )?;
    if !result.pair_frame_rows.is_empty() {
        write_value_csv(
            &output.join("forced_stability_pair_frame_metrics.csv"),
            PAIR_FRAME_FIELDS,
            &result.pair_frame_rows,
        )?;
        write_value_csv(
            &output.join("forced_stability_pair_summary.csv"),
            PAIR_SUMMARY_FIELDS,
            &result.pair_summary_rows,
        )?;
    }
    let coordinate_definition = if result.anchor_coordinate == "peak" {
        "source_catalog peak_x/peak_y with measured x/y fallback"
    } else {
        "source_catalog measured x/y with peak_x/peak_y fallback"
    };
    let payload = json!({"source_catalog":result.source_catalog,"persistence_json":result.persistence_json,"frame_count":result.frame_count,"anchor_count":result.anchors.len(),"frame_metric_count":result.frame_metrics.len(),"parameters":{"aperture_radius":result.options.aperture_radius,"min_psf_support_pixels":result.options.min_psf_support_pixels,"min_flux_snr":result.options.min_flux_snr,"min_fwhm":result.options.min_fwhm,"max_fwhm":result.options.max_fwhm,"max_ellipticity":result.options.max_ellipticity,"anchor_coordinate":result.anchor_coordinate,"local_peak_search_radius":result.options.local_peak_search_radius,"local_peak_psf_fwhm":result.options.local_peak_psf_fwhm,"local_peak_selection":"argmax of three-scale normalized Gaussian matched response in the bounded integer-offset window; center offset is first for deterministic ties","anchor_coordinate_definition":coordinate_definition,"included_detection_ids":result.included_detection_ids,"raw_measurement":"unscaled FITS; per-frame background_adu/noise_adu from persistence JSON"},"anchors":result.anchors,"summary_rows":result.summary_rows,"pair_relationship_summary":result.pair_summary_rows,"note":"Fixed fields (flux_snr/support_3x3/fwhm/ellipticity/quality_like) are forced photometry at predicted registered positions. local_peak_* is a bounded diagnostic relocation within at most ±1 px, not a reclassification or a WCS/catalog identity match. Taking the maximum over multiple positions has a look-elsewhere bias over positions and scales, so fixed and local values must be reported together. quality_like is not quality_passed; value-domain flags and deblending evidence must be applied separately. When two detection IDs are explicitly included, pair relationship files also report registered local separation and inward-shift fractions. They are bounded diagnostics for shared structure or model mismatch, not evidence of a binary or two independent catalog sources."});
    fs::write(
        output.join("forced_stability.json"),
        serde_json::to_vec_pretty(&payload).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok((output, payload))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let options = parse_options(args)?;
    let result = run(options)?;
    let (_output, payload) = write_artifacts(&result)?;
    Ok(payload)
}

pub fn run_cli(args: &[String]) -> i32 {
    let result = (|| -> Result<(PathBuf, AuditResult, Value)> {
        let options = parse_options(args)?;
        let result = run(options)?;
        let (output, payload) = write_artifacts(&result)?;
        Ok((output, result, payload))
    })();
    match result {
        Ok((output, result, _)) => {
            println!("rst19-forced-stability: 审计已写入 {}", output.display());
            println!(
                "anchors={} frames={} metrics={}",
                result.anchors.len(),
                result.frame_count,
                result.frame_metrics.len()
            );
            for row in &result.summary_rows {
                println!("{}: n={} fixed_flux_snr={} local_flux_snr={} quality_like_ge12={} local_quality_like_ge12={} reloc_dist={}",row["feature_class"],row["sample_count"],row["median_flux_snr_across_frames"],row["median_local_peak_flux_snr_across_frames"],row["quality_like_presence_ge_12_count"],row["local_peak_quality_like_presence_ge_12_count"],row["median_relocalization_distance_px"]);
            }
            0
        }
        Err(error) => {
            eprintln!("rst19-forced-stability: {error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{FitsFrame, PixelData};
    use serde_json::Map;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    fn temp_dir(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "rst19-forced-stability-{label}-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn options(root: &Path) -> Options {
        Options {
            source_catalog: root.join("source_catalog.csv"),
            persistence_json: root.join("persistence.json"),
            base_dir: root.to_path_buf(),
            max_per_class: 32,
            anchor_coordinate: "peak".into(),
            include_detection_ids: Vec::new(),
            aperture_radius: 4,
            min_psf_support_pixels: 3,
            min_flux_snr: 5.0,
            min_fwhm: 0.8,
            max_fwhm: 12.0,
            max_ellipticity: 0.65,
            local_peak_search_radius: 1,
            local_peak_psf_fwhm: 3.0,
            out_dir: root.join("out"),
            quiet: true,
        }
    }

    fn anchor(id: i64, class: &str, x: f64, y: f64) -> ForcedAnchor {
        ForcedAnchor {
            detection_id: id,
            feature_class: class.into(),
            feature_class_label: class.into(),
            x,
            y,
            flux_snr: Some(10.0),
        }
    }

    fn metric(
        anchor: &ForcedAnchor,
        frame_index: usize,
        flux_snr: f64,
        quality: bool,
    ) -> ForcedFrameMetric {
        ForcedFrameMetric {
            detection_id: anchor.detection_id,
            feature_class: anchor.feature_class.clone(),
            feature_class_label: anchor.feature_class_label.clone(),
            frame_index,
            predicted_x: anchor.x,
            predicted_y: anchor.y,
            flux_snr: Some(flux_snr),
            support_3x3: Some(5),
            center_valid: true,
            fwhm: Some(2.0),
            ellipticity: Some(0.1),
            quality_like: quality,
            local_peak_x: Some(anchor.x),
            local_peak_y: Some(anchor.y),
            local_peak_flux_snr: None,
            local_peak_support_3x3: None,
            local_peak_center_valid: None,
            local_peak_fwhm: None,
            local_peak_ellipticity: None,
            local_peak_quality_like: None,
            matched_response_snr: None,
            local_peak_response_snr: None,
            relocalization_dx_px: Some(0.0),
            relocalization_dy_px: Some(0.0),
            relocalization_distance_px: Some(0.0),
            relocalized: Some(false),
        }
    }

    #[test]
    fn local_offsets_are_bounded_and_center_wins_ties_or_all_invalid() {
        let offsets = local_offsets(1).unwrap();
        assert_eq!(offsets.len(), 9);
        assert_eq!(offsets[0], (0, 0));
        assert_eq!(select_local_peak(&[4.0, 4.0, f64::NAN]), 0);
        assert_eq!(select_local_peak(&[f64::NAN; 9]), 0);
        assert!(local_offsets(2).unwrap_err().contains("0 or 1"));
    }

    #[test]
    fn anchor_loading_uses_peak_fallback_inclusion_and_deterministic_quantiles() {
        let root = temp_dir("anchors");
        let catalog = root.join("sources.csv");
        fs::write(&catalog, "detection_id,x,y,peak_x,peak_y,flux_snr,feature_class,feature_class_label\n1,10.25,20.75,10,21,1,compact_quality,compact\n2,30.25,40.75,30,41,2,compact_quality,compact\n3,50.25,60.75,50,61,3,compact_quality,compact\n4,70.25,80.75,,,4,range_anomaly,range\n").unwrap();
        let peaks = load_anchors(&catalog, 2, "peak", &[]).unwrap();
        let centroids = load_anchors(&catalog, 2, "centroid", &[]).unwrap();
        let included = load_anchors(&catalog, 1, "peak", &[2]).unwrap();
        assert_eq!(
            peaks
                .iter()
                .map(|a| a.detection_id)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([1, 3, 4])
        );
        let centroid = centroids.iter().find(|a| a.detection_id == 1).unwrap();
        assert_eq!((centroid.x, centroid.y), (10.25, 20.75));
        assert_eq!(
            included
                .iter()
                .map(|a| a.detection_id)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([1, 2, 4])
        );
        assert!(load_anchors(&catalog, 1, "peak", &[999])
            .unwrap_err()
            .contains("included detection_id"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn grouped_summary_and_pair_geometry_keep_diagnostic_boundaries() {
        let a = anchor(11, "range_anomaly", 20.0, 30.0);
        let b = anchor(12, "crowded_blend", 16.0, 30.0);
        let mut ma = metric(&a, 1, 10.0, true);
        let mut mb = metric(&b, 1, 10.0, true);
        ma.local_peak_x = Some(19.0);
        mb.local_peak_x = Some(17.0);
        let pair_rows = pair_frame_rows(
            &[a.clone(), b.clone()],
            &[ma.clone(), mb.clone()],
            &[11, 12],
            1,
        )
        .unwrap();
        let pair = pair_summaries(&[a.clone(), b.clone()], &pair_rows, &[11, 12], 1);
        assert_eq!(pair[0]["anchor_separation_px"], 4.0);
        assert_eq!(pair[0]["median_local_separation_px"], 2.0);
        assert_eq!(pair[0]["both_shift_toward_each_other_fraction"], 1.0);
        assert_eq!(pair[0]["local_offset_same_fraction"], 0.0);

        let one = anchor(1, "compact_quality", 10.0, 20.0);
        let values = vec![
            metric(&one, 1, 10.0, true),
            metric(&one, 2, 11.0, true),
            metric(&one, 3, 9.0, false),
        ];
        let rows = summaries(&[one], &values, 3).unwrap();
        assert_eq!(rows[0]["median_quality_like_presence"], 2.0);
        assert_eq!(rows[0]["median_flux_snr_across_frames"], 10.0);
        assert!((rows[0]["robust_flux_snr_cv_median"].as_f64().unwrap() - 0.14826).abs() < 1e-12);
        assert!(summaries(&[a], &[ma, mb], 2)
            .unwrap_err()
            .contains("expected 2"));
    }

    #[test]
    fn local_matched_response_selects_a_shifted_gaussian_center() {
        let width = 64;
        let height = 64;
        let image: Vec<f64> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    10.0 + 100.0
                        * (-0.5
                            * (((x as f64 - 33.0) / 1.2).powi(2)
                                + ((y as f64 - 30.0) / 1.2).powi(2)))
                        .exp()
                })
            })
            .collect();
        let frame = FitsFrame {
            path: PathBuf::new(),
            header: Map::new(),
            data: PixelData::F64(image.clone()),
            auxiliary: None,
            data_offset: 0,
            width,
            height,
        };
        let offsets = local_offsets(1).unwrap();
        let xs: Vec<f64> = offsets.iter().map(|(dx, _)| 32.0 + *dx as f64).collect();
        let ys: Vec<f64> = offsets.iter().map(|(_, dy)| 30.0 + *dy as f64).collect();
        let response = local_responses(
            &frame,
            &image,
            &vec![false; image.len()],
            &xs,
            &ys,
            10.0,
            1.0,
            3.0,
        )
        .unwrap();
        assert_eq!(offsets[select_local_peak(&response)], (1, 0));
    }

    #[test]
    fn full_audit_reads_fits_and_writes_csv_json_without_python() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let frame =
            workspace.join("backend/tests/fixtures/empirical_pair_audit_parity/frames/frame.fits");
        let root = temp_dir("e2e");
        fs::write(root.join("source_catalog.csv"), "detection_id,x,y,peak_x,peak_y,flux_snr,feature_class,feature_class_label\n1,46,48,46,48,10,compact_quality,compact\n").unwrap();
        let persistence = json!({"frame_count":1,"cumulative_shifts":[[0.0,0.0]],"frame_rows":[{"frame_index":1,"path":frame,"background_adu":100.0,"noise_adu":3.0}]});
        fs::write(
            root.join("persistence.json"),
            serde_json::to_vec(&persistence).unwrap(),
        )
        .unwrap();
        let mut options = options(&root);
        options.persistence_json = root.join("persistence.json");
        options.base_dir = root.clone();
        let result = run(options).unwrap();
        assert_eq!(result.frame_count, 1);
        assert_eq!(result.frame_metrics.len(), 1);
        let (output, payload) = write_artifacts(&result).unwrap();
        assert_eq!(payload["anchor_count"], 1);
        for file in [
            "forced_stability_frame_metrics.csv",
            "forced_stability_summary.csv",
            "forced_stability.json",
        ] {
            assert!(output.join(file).is_file(), "missing {file}");
        }
        assert_eq!(
            fs::read(output.join("forced_stability_frame_metrics.csv"))
                .unwrap()
                .get(..3),
            Some(&[0xEF, 0xBB, 0xBF][..])
        );
        fs::remove_dir_all(root).unwrap();
    }
}
