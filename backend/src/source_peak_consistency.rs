//! Compare catalog candidate peaks with local maxima in the original FITS pixels.
//!
//! These measurements describe detector-level pixel geometry. Raw-pixel maxima
//! are not sufficient evidence of a physical source and do not change the
//! detector cache or quality decisions.

use crate::{filtering, fits, models::PixelData, Result};
use csv::{StringRecord, Terminator, WriterBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub const DEFAULT_LOCAL_MAX_RADII: &[usize] = &[1, 2, 3];

const REQUIRED_COLUMNS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "peak_x",
    "peak_y",
];

const ROW_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "peak_x",
    "peak_y",
    "raw_peak_adu",
    "raw_valid",
    "raw_local_maximum_r1",
    "raw_unique_local_maximum_r1",
    "raw_local_max_adu_r1",
    "raw_local_max_gap_adu_r1",
    "raw_local_maximum_r2",
    "raw_unique_local_maximum_r2",
    "raw_local_max_adu_r2",
    "raw_local_max_gap_adu_r2",
    "raw_local_maximum_r3",
    "raw_unique_local_maximum_r3",
    "raw_local_max_adu_r3",
    "raw_local_max_gap_adu_r3",
];

const CLASS_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "candidate_count",
    "quality_count",
    "raw_valid_count",
    "raw_local_maximum_count_r1",
    "raw_local_maximum_fraction_r1",
    "raw_local_maximum_count_r2",
    "raw_local_maximum_fraction_r2",
    "raw_local_maximum_count_r3",
    "raw_local_maximum_fraction_r3",
    "raw_unique_local_maximum_count_r1",
    "raw_unique_local_maximum_count_r2",
    "raw_unique_local_maximum_count_r3",
    "nonlocal_peak_count_r1",
    "nonlocal_peak_gap_median_adu_r1",
    "nonlocal_peak_gap_p90_adu_r1",
    "nonlocal_peak_count_r2",
    "nonlocal_peak_gap_median_adu_r2",
    "nonlocal_peak_gap_p90_adu_r2",
    "nonlocal_peak_count_r3",
    "nonlocal_peak_gap_median_adu_r3",
    "nonlocal_peak_gap_p90_adu_r3",
];

const TARGET_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "peak_x",
    "peak_y",
    "radius_px",
    "local_max_x",
    "local_max_y",
    "raw_peak_adu",
    "local_max_adu",
    "local_max_gap_adu",
    "local_max_offset_px",
    "raw_local_maximum",
    "raw_unique_local_maximum",
];

const INTERPRETATION_BOUNDARY: &str = "raw 局部极大值只是像素拓扑诊断，不是恒星的充分条件；滤波候选可能因 PSF 混叠、探测器值域、背景结构或模型失配而偏离原始像素峰；不推导 precision、FDR、恒星概率或物理源数";
const CONCLUSION_PREFIX: &str = "原始像素局部峰审计显示，匹配滤波候选与 raw 像素峰可以发生位置分歧；该分歧在范围异常、线状或拥挤结构中应作为专项复核信号，而不能用 raw 局部极大值单独决定真星。目标结果";

#[derive(Clone, Debug)]
struct CatalogRecord {
    detection_id: i64,
    feature_class: String,
    feature_class_label: String,
    quality_passed: bool,
    peak_x: i64,
    peak_y: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourcePeakConsistencyRow {
    pub detection_id: i64,
    pub feature_class: String,
    pub feature_class_label: String,
    pub quality_passed: bool,
    pub peak_x: usize,
    pub peak_y: usize,
    pub raw_peak_adu: f64,
    pub raw_valid: bool,
    pub raw_local_maximum_r1: bool,
    pub raw_unique_local_maximum_r1: bool,
    pub raw_local_max_adu_r1: f64,
    pub raw_local_max_gap_adu_r1: f64,
    pub raw_local_maximum_r2: bool,
    pub raw_unique_local_maximum_r2: bool,
    pub raw_local_max_adu_r2: f64,
    pub raw_local_max_gap_adu_r2: f64,
    pub raw_local_maximum_r3: bool,
    pub raw_unique_local_maximum_r3: bool,
    pub raw_local_max_adu_r3: f64,
    pub raw_local_max_gap_adu_r3: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourcePeakConsistencyClassSummary {
    pub feature_class: String,
    pub feature_class_label: String,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub raw_valid_count: usize,
    pub raw_local_maximum_count_r1: usize,
    pub raw_local_maximum_fraction_r1: Option<f64>,
    pub raw_local_maximum_count_r2: usize,
    pub raw_local_maximum_fraction_r2: Option<f64>,
    pub raw_local_maximum_count_r3: usize,
    pub raw_local_maximum_fraction_r3: Option<f64>,
    pub raw_unique_local_maximum_count_r1: usize,
    pub raw_unique_local_maximum_count_r2: usize,
    pub raw_unique_local_maximum_count_r3: usize,
    pub nonlocal_peak_count_r1: usize,
    pub nonlocal_peak_gap_median_adu_r1: Option<f64>,
    pub nonlocal_peak_gap_p90_adu_r1: Option<f64>,
    pub nonlocal_peak_count_r2: usize,
    pub nonlocal_peak_gap_median_adu_r2: Option<f64>,
    pub nonlocal_peak_gap_p90_adu_r2: Option<f64>,
    pub nonlocal_peak_count_r3: usize,
    pub nonlocal_peak_gap_median_adu_r3: Option<f64>,
    pub nonlocal_peak_gap_p90_adu_r3: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourcePeakConsistencyTargetRow {
    pub detection_id: i64,
    pub feature_class: String,
    pub feature_class_label: String,
    pub quality_passed: bool,
    pub peak_x: usize,
    pub peak_y: usize,
    pub radius_px: usize,
    pub local_max_x: usize,
    pub local_max_y: usize,
    pub raw_peak_adu: f64,
    pub local_max_adu: f64,
    pub local_max_gap_adu: f64,
    pub local_max_offset_px: f64,
    pub raw_local_maximum: bool,
    pub raw_unique_local_maximum: bool,
}

#[derive(Clone, Debug)]
pub struct SourcePeakConsistencyResult {
    pub catalog_path: String,
    pub fits_path: String,
    pub image_shape: (usize, usize),
    pub local_max_radii_px: Vec<usize>,
    pub rows: Vec<SourcePeakConsistencyRow>,
    pub class_summaries: Vec<SourcePeakConsistencyClassSummary>,
    pub targets: Vec<SourcePeakConsistencyTargetRow>,
    pub conclusion: String,
}

impl SourcePeakConsistencyResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "fits_path": self.fits_path,
            "image_shape": [self.image_shape.0, self.image_shape.1],
            "local_max_radii_px": self.local_max_radii_px,
            "rows": self.rows,
            "class_summaries": self.class_summaries,
            "targets": self.targets,
            "conclusion": self.conclusion,
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        })
    }
}

#[derive(Clone, Debug)]
struct RadiusData {
    radius: usize,
    maxima: Vec<bool>,
    unique_maxima: Vec<bool>,
    max_values: Vec<f64>,
}

fn python_repr(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("'{escaped}'")
}

fn required_value(record: &StringRecord, columns: &BTreeMap<String, usize>, key: &str) -> String {
    columns
        .get(key)
        .and_then(|index| record.get(*index))
        .unwrap_or_default()
        .trim()
        .to_owned()
}

fn required_integer(
    record: &StringRecord,
    columns: &BTreeMap<String, usize>,
    key: &str,
    row_number: usize,
) -> Result<i64> {
    let value = required_value(record, columns, key);
    if value.is_empty() {
        return Err(format!("catalog row {row_number}: empty {key}"));
    }
    let parsed = value.parse::<f64>().map_err(|_| {
        format!(
            "catalog row {row_number}: invalid {key}={}",
            python_repr(&value)
        )
    })?;
    if !parsed.is_finite()
        || parsed.fract() != 0.0
        || parsed < i64::MIN as f64
        || parsed >= -(i64::MIN as f64)
    {
        return Err(format!(
            "catalog row {row_number}: {key} must be an integer, got {}",
            python_repr(&value)
        ));
    }
    Ok(parsed as i64)
}

fn required_coordinate(
    record: &StringRecord,
    columns: &BTreeMap<String, usize>,
    key: &str,
    row_number: usize,
) -> Result<i64> {
    required_integer(record, columns, key, row_number)
}

fn required_bool(
    record: &StringRecord,
    columns: &BTreeMap<String, usize>,
    key: &str,
    row_number: usize,
) -> Result<bool> {
    let value = required_value(record, columns, key);
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(format!(
            "catalog row {row_number}: invalid {key}={}",
            python_repr(&value)
        )),
    }
}

fn load_catalog(path: &Path) -> Result<Vec<CatalogRecord>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|error| error.to_string())?;
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let normalized: Vec<_> = headers
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if index == 0 {
                value.trim_start_matches('\u{feff}').to_owned()
            } else {
                value.to_owned()
            }
        })
        .collect();
    let columns: BTreeMap<_, _> = normalized
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let present: BTreeSet<_> = normalized.iter().map(String::as_str).collect();
    let missing: Vec<_> = REQUIRED_COLUMNS
        .iter()
        .copied()
        .filter(|key| !present.contains(key))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let mut ids = BTreeSet::new();
    let mut records = Vec::new();
    for (row_index, row) in reader.records().enumerate() {
        let row = row.map_err(|error| error.to_string())?;
        let row_number = row_index + 2;
        let detection_id = required_integer(&row, &columns, "detection_id", row_number)?;
        if !ids.insert(detection_id) {
            return Err(format!(
                "source catalog contains duplicate detection_id={detection_id}"
            ));
        }
        let feature_class = required_value(&row, &columns, "feature_class");
        let feature_class_label = required_value(&row, &columns, "feature_class_label");
        let quality_passed = required_bool(&row, &columns, "quality_passed", row_number)?;
        let peak_x = required_coordinate(&row, &columns, "peak_x", row_number)?;
        let peak_y = required_coordinate(&row, &columns, "peak_y", row_number)?;
        if feature_class.is_empty() {
            return Err(format!("catalog row {row_number}: empty feature_class"));
        }
        records.push(CatalogRecord {
            detection_id,
            feature_class,
            feature_class_label,
            quality_passed,
            peak_x,
            peak_y,
        });
    }
    if records.is_empty() {
        return Err("source catalog is empty".into());
    }
    Ok(records)
}

fn tuple_text(values: &[usize]) -> String {
    if values.len() == 1 {
        format!("({},)", values[0])
    } else {
        format!(
            "({})",
            values
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn validate_radii(radii: &[usize]) -> Result<Vec<usize>> {
    if radii != DEFAULT_LOCAL_MAX_RADII {
        return Err(format!(
            "source peak consistency uses the stable radius set (1, 2, 3); got {}",
            tuple_text(radii)
        ));
    }
    Ok(radii.to_vec())
}

fn resolved_path(path: &Path) -> Result<PathBuf> {
    let canonical = fs::canonicalize(path).map_err(|error| error.to_string())?;
    #[cfg(windows)]
    {
        let text = canonical.to_string_lossy();
        let normalized = text.strip_prefix("\\\\?\\").unwrap_or(text.as_ref());
        Ok(PathBuf::from(normalized))
    }
    #[cfg(not(windows))]
    {
        Ok(canonical)
    }
}

fn integer_sentinel(data: &PixelData) -> f64 {
    match data {
        PixelData::U8(_) => u8::MIN as f64,
        PixelData::I16(_) => i16::MIN as f64,
        PixelData::I32(_) => i32::MIN as f64,
        PixelData::I64(_) => i64::MIN as f64,
        PixelData::F32(_) | PixelData::F64(_) => f64::NEG_INFINITY,
    }
}

fn clamp_offset(center: usize, offset: isize, limit: usize) -> usize {
    (center as isize + offset).clamp(0, limit as isize - 1) as usize
}

/// Separable box convolution with nearest-edge extension, matching ndimage.convolve.
fn box_sum_nearest(values: &[bool], height: usize, width: usize, radius: usize) -> Vec<u16> {
    let mut horizontal = vec![0_u16; values.len()];
    let window = radius as isize;
    for y in 0..height {
        let mut sum = (-(window)..=window)
            .filter(|offset| values[y * width + clamp_offset(0, *offset, width)])
            .count() as u16;
        for x in 0..width {
            horizontal[y * width + x] = sum;
            if x + 1 < width {
                let removed_x = clamp_offset(x, -(window), width);
                let added_x = clamp_offset(x + 1, window, width);
                sum -= u16::from(values[y * width + removed_x]);
                sum += u16::from(values[y * width + added_x]);
            }
        }
    }
    let mut output = vec![0_u16; values.len()];
    for x in 0..width {
        let mut sum = (-(window)..=window)
            .map(|offset| horizontal[clamp_offset(0, offset, height) * width + x])
            .sum::<u16>();
        for y in 0..height {
            output[y * width + x] = sum;
            if y + 1 < height {
                let removed_y = clamp_offset(y, -(window), height);
                let added_y = clamp_offset(y + 1, window, height);
                sum -= horizontal[removed_y * width + x];
                sum += horizontal[added_y * width + x];
            }
        }
    }
    output
}

fn radius_data(
    image: &[f64],
    valid: &[bool],
    height: usize,
    width: usize,
    radius: usize,
    sentinel: f64,
) -> Result<RadiusData> {
    let masked: Vec<_> = image
        .iter()
        .zip(valid)
        .map(|(value, is_valid)| if *is_valid { *value } else { sentinel })
        .collect();
    let max_values = filtering::maximum_filter(&masked, height, width, radius)?;
    let equal_maxima: Vec<_> = image
        .iter()
        .zip(valid)
        .zip(&max_values)
        .map(|((value, is_valid), maximum)| *is_valid && *value == *maximum)
        .collect();
    let counts = box_sum_nearest(&equal_maxima, height, width, radius);
    let unique_maxima: Vec<_> = equal_maxima
        .iter()
        .zip(counts)
        .map(|(is_maximum, count)| *is_maximum && count == 1)
        .collect();
    Ok(RadiusData {
        radius,
        maxima: equal_maxima,
        unique_maxima,
        max_values,
    })
}

fn gap_record(maximum: f64, raw: f64) -> f64 {
    if maximum.is_finite() {
        let difference = maximum - raw;
        if difference > 0.0 {
            difference
        } else {
            0.0
        }
    } else {
        f64::NAN
    }
}

fn gap_summary(maximum: f64, raw: f64) -> f64 {
    let difference = maximum - raw;
    if difference.is_nan() {
        f64::NAN
    } else {
        difference.max(0.0)
    }
}

fn linear_quantile(values: &[f64], probability: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    if values.iter().any(|value| value.is_nan()) {
        return Some(f64::NAN);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * probability;
    let left = position.floor() as usize;
    let right = position.ceil() as usize;
    Some(sorted[left] + (sorted[right] - sorted[left]) * (position - left as f64))
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    if values.iter().any(|value| value.is_nan()) {
        return Some(f64::NAN);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Some((sorted[middle - 1] + sorted[middle]) / 2.0)
    } else {
        Some(sorted[middle])
    }
}

fn target_local_maximum(
    image: &[f64],
    valid: &[bool],
    height: usize,
    width: usize,
    x: usize,
    y: usize,
    radius: usize,
) -> (usize, usize, f64) {
    let index = y * width + x;
    if !valid[index] {
        return (x, y, f64::NAN);
    }
    let y0 = y.saturating_sub(radius);
    let y1 = y.saturating_add(radius).min(height - 1);
    let x0 = x.saturating_sub(radius);
    let x1 = x.saturating_add(radius).min(width - 1);
    let mut found = false;
    let mut best_value = f64::NEG_INFINITY;
    let mut best_x = x;
    let mut best_y = y;
    for py in y0..=y1 {
        for px in x0..=x1 {
            let pixel_index = py * width + px;
            if !valid[pixel_index] {
                continue;
            }
            let value = image[pixel_index];
            if !found || value > best_value {
                found = true;
                best_value = value;
                best_x = px;
                best_y = py;
            }
        }
    }
    if found {
        (best_x, best_y, best_value)
    } else {
        (x, y, f64::NAN)
    }
}

fn make_row(
    record: &CatalogRecord,
    image: &[f64],
    valid: &[bool],
    width: usize,
    radii: &[RadiusData],
) -> SourcePeakConsistencyRow {
    let index = record.peak_y as usize * width + record.peak_x as usize;
    let raw = image[index];
    let mut values = Vec::with_capacity(radii.len());
    for data in radii {
        let maximum = data.max_values[index];
        values.push((
            data.maxima[index],
            data.unique_maxima[index],
            maximum,
            gap_record(maximum, raw),
        ));
    }
    SourcePeakConsistencyRow {
        detection_id: record.detection_id,
        feature_class: record.feature_class.clone(),
        feature_class_label: record.feature_class_label.clone(),
        quality_passed: record.quality_passed,
        peak_x: record.peak_x as usize,
        peak_y: record.peak_y as usize,
        raw_peak_adu: raw,
        raw_valid: valid[index],
        raw_local_maximum_r1: values[0].0,
        raw_unique_local_maximum_r1: values[0].1,
        raw_local_max_adu_r1: values[0].2,
        raw_local_max_gap_adu_r1: values[0].3,
        raw_local_maximum_r2: values[1].0,
        raw_unique_local_maximum_r2: values[1].1,
        raw_local_max_adu_r2: values[1].2,
        raw_local_max_gap_adu_r2: values[1].3,
        raw_local_maximum_r3: values[2].0,
        raw_unique_local_maximum_r3: values[2].1,
        raw_local_max_adu_r3: values[2].2,
        raw_local_max_gap_adu_r3: values[2].3,
    }
}

fn make_class_summary(
    feature_class: &str,
    records: &[CatalogRecord],
    indexes: &[usize],
    image: &[f64],
    valid: &[bool],
    width: usize,
    radii: &[RadiusData],
) -> SourcePeakConsistencyClassSummary {
    let count = indexes.len();
    let quality_count = indexes
        .iter()
        .filter(|index| records[**index].quality_passed)
        .count();
    let raw_valid_count = indexes
        .iter()
        .filter(|index| {
            valid[records[**index].peak_y as usize * width + records[**index].peak_x as usize]
        })
        .count();
    let mut local_count = [0; 3];
    let mut local_fraction = [None; 3];
    let mut unique_count = [0; 3];
    let mut nonlocal_count = [0; 3];
    let mut gap_median = [None; 3];
    let mut gap_p90 = [None; 3];
    for (radius_index, data) in radii.iter().enumerate() {
        let mut gaps = Vec::new();
        for source_index in indexes {
            let record = &records[*source_index];
            let index = record.peak_y as usize * width + record.peak_x as usize;
            if data.maxima[index] {
                local_count[radius_index] += 1;
            }
            if data.unique_maxima[index] {
                unique_count[radius_index] += 1;
            }
            let gap = gap_summary(data.max_values[index], image[index]);
            if !data.maxima[index] {
                nonlocal_count[radius_index] += 1;
                gaps.push(gap);
            }
        }
        local_fraction[radius_index] = if count == 0 {
            None
        } else {
            Some(local_count[radius_index] as f64 / count as f64)
        };
        gap_median[radius_index] = median(&gaps);
        gap_p90[radius_index] = linear_quantile(&gaps, 0.90);
    }
    let label = records
        .iter()
        .find(|record| record.feature_class == feature_class)
        .map(|record| record.feature_class_label.clone())
        .unwrap_or_default();
    SourcePeakConsistencyClassSummary {
        feature_class: feature_class.to_owned(),
        feature_class_label: label,
        candidate_count: count,
        quality_count,
        raw_valid_count,
        raw_local_maximum_count_r1: local_count[0],
        raw_local_maximum_fraction_r1: local_fraction[0],
        raw_local_maximum_count_r2: local_count[1],
        raw_local_maximum_fraction_r2: local_fraction[1],
        raw_local_maximum_count_r3: local_count[2],
        raw_local_maximum_fraction_r3: local_fraction[2],
        raw_unique_local_maximum_count_r1: unique_count[0],
        raw_unique_local_maximum_count_r2: unique_count[1],
        raw_unique_local_maximum_count_r3: unique_count[2],
        nonlocal_peak_count_r1: nonlocal_count[0],
        nonlocal_peak_gap_median_adu_r1: gap_median[0],
        nonlocal_peak_gap_p90_adu_r1: gap_p90[0],
        nonlocal_peak_count_r2: nonlocal_count[1],
        nonlocal_peak_gap_median_adu_r2: gap_median[1],
        nonlocal_peak_gap_p90_adu_r2: gap_p90[1],
        nonlocal_peak_count_r3: nonlocal_count[2],
        nonlocal_peak_gap_median_adu_r3: gap_median[2],
        nonlocal_peak_gap_p90_adu_r3: gap_p90[2],
    }
}

fn target_rows(
    records: &[CatalogRecord],
    image: &[f64],
    valid: &[bool],
    height: usize,
    width: usize,
    radii: &[RadiusData],
    target_ids: &[i64],
) -> Result<Vec<SourcePeakConsistencyTargetRow>> {
    let by_id: BTreeMap<_, _> = records
        .iter()
        .map(|record| (record.detection_id, record))
        .collect();
    let requested: BTreeSet<_> = target_ids.iter().copied().collect();
    let catalog_ids: BTreeSet<_> = by_id.keys().copied().collect();
    let missing: Vec<_> = requested.difference(&catalog_ids).copied().collect();
    if !missing.is_empty() {
        return Err(format!(
            "target detection_id not in source catalog: {}",
            missing
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let mut ordered_ids = Vec::new();
    for id in target_ids {
        if !ordered_ids.contains(id) {
            ordered_ids.push(*id);
        }
    }
    let mut rows = Vec::with_capacity(ordered_ids.len() * radii.len());
    for id in ordered_ids {
        let record = by_id[&id];
        let x = record.peak_x as usize;
        let y = record.peak_y as usize;
        let center = y * width + x;
        let raw = image[center];
        for data in radii {
            let (local_x, local_y, local_max) =
                target_local_maximum(image, valid, height, width, x, y, data.radius);
            let offset =
                ((local_x as f64 - x as f64).powi(2) + (local_y as f64 - y as f64).powi(2)).sqrt();
            rows.push(SourcePeakConsistencyTargetRow {
                detection_id: id,
                feature_class: record.feature_class.clone(),
                feature_class_label: record.feature_class_label.clone(),
                quality_passed: record.quality_passed,
                peak_x: x,
                peak_y: y,
                radius_px: data.radius,
                local_max_x: local_x,
                local_max_y: local_y,
                raw_peak_adu: raw,
                local_max_adu: local_max,
                local_max_gap_adu: if local_max.is_finite() {
                    gap_record(local_max, raw)
                } else {
                    f64::NAN
                },
                local_max_offset_px: offset,
                raw_local_maximum: data.maxima[center],
                raw_unique_local_maximum: data.unique_maxima[center],
            });
        }
    }
    Ok(rows)
}

pub fn run(
    catalog_path: &Path,
    fits_path: &Path,
    target_ids: &[i64],
    local_max_radii_px: &[usize],
) -> Result<SourcePeakConsistencyResult> {
    let radii = validate_radii(local_max_radii_px)?;
    let catalog = resolved_path(catalog_path)?;
    let fits_path = resolved_path(fits_path)?;
    let records = load_catalog(&catalog)?;
    let frame = fits::read_fits(&fits_path, false)?;
    let height = frame.height;
    let width = frame.width;
    let image = frame.data.to_f64();
    let valid: Vec<_> = fits::auxiliary_mask(height, width)?
        .into_iter()
        .zip(&image)
        .map(|(masked, value)| !masked && value.is_finite())
        .collect();
    for record in &records {
        if record.peak_x < 0
            || record.peak_y < 0
            || record.peak_x as usize >= width
            || record.peak_y as usize >= height
        {
            return Err(format!(
                "detection_id={} peak coordinate ({}, {}) is outside image shape ({}, {})",
                record.detection_id, record.peak_x, record.peak_y, height, width
            ));
        }
    }
    let sentinel = integer_sentinel(&frame.data);
    let radius_data: Vec<_> = radii
        .iter()
        .map(|radius| radius_data(&image, &valid, height, width, *radius, sentinel))
        .collect::<Result<_>>()?;
    let rows: Vec<_> = records
        .iter()
        .map(|record| make_row(record, &image, &valid, width, &radius_data))
        .collect();
    let classes: BTreeSet<_> = records
        .iter()
        .map(|record| record.feature_class.as_str())
        .collect();
    let class_summaries: Vec<_> = classes
        .iter()
        .map(|feature_class| {
            let indexes: Vec<_> = records
                .iter()
                .enumerate()
                .filter(|(_, record)| record.feature_class == *feature_class)
                .map(|(index, _)| index)
                .collect();
            make_class_summary(
                feature_class,
                &records,
                &indexes,
                &image,
                &valid,
                width,
                &radius_data,
            )
        })
        .collect();
    let targets = target_rows(
        &records,
        &image,
        &valid,
        height,
        width,
        &radius_data,
        target_ids,
    )?;
    let target_text = if targets.is_empty() {
        "未指定目标。".to_owned()
    } else {
        let joined = targets
            .iter()
            .map(|target| {
                let local_maximum = if target.raw_local_maximum {
                    "True"
                } else {
                    "False"
                };
                format!(
                    "ID {} 在 r={} px 的原始局部极大值={local_maximum}",
                    target.detection_id, target.radius_px
                )
            })
            .collect::<Vec<_>>()
            .join("；");
        format!("：{joined}。")
    };
    Ok(SourcePeakConsistencyResult {
        catalog_path: catalog.to_string_lossy().into_owned(),
        fits_path: fits_path.to_string_lossy().into_owned(),
        image_shape: (height, width),
        local_max_radii_px: radii,
        rows,
        class_summaries,
        targets,
        conclusion: format!("{CONCLUSION_PREFIX}{target_text}"),
    })
}

fn csv_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn write_csv_rows<T: Serialize>(path: &Path, headers: &[&str], rows: &[T]) -> Result<()> {
    let mut bytes = b"\xEF\xBB\xBF".to_vec();
    {
        let mut writer = WriterBuilder::new()
            .terminator(Terminator::CRLF)
            .from_writer(&mut bytes);
        writer
            .write_record(headers)
            .map_err(|error| error.to_string())?;
        for row in rows {
            let value = serde_json::to_value(row).map_err(|error| error.to_string())?;
            writer
                .write_record(headers.iter().map(|header| csv_value(&value[*header])))
                .map_err(|error| error.to_string())?;
        }
        writer.flush().map_err(|error| error.to_string())?;
    }
    fs::write(path, bytes).map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &SourcePeakConsistencyResult, output: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    write_csv_rows(
        &output.join("source_peak_consistency.csv"),
        ROW_FIELDS,
        &result.rows,
    )?;
    write_csv_rows(
        &output.join("source_peak_consistency_class_summary.csv"),
        CLASS_FIELDS,
        &result.class_summaries,
    )?;
    write_csv_rows(
        &output.join("source_peak_consistency_targets.csv"),
        TARGET_FIELDS,
        &result.targets,
    )?;
    fs::write(
        output.join("source_peak_consistency.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut positional = Vec::new();
    let mut targets = Vec::new();
    let mut output = PathBuf::from("tmp/source-peak-consistency");
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        let (name, inline) = argument.split_once('=').unwrap_or((argument, ""));
        match name {
            "--target-id" | "--out-dir" => {
                let value = if !inline.is_empty() {
                    inline
                } else {
                    index += 1;
                    args.get(index).ok_or("option requires a value")?
                };
                if name == "--target-id" {
                    targets.push(value.parse::<i64>().map_err(|error| error.to_string())?);
                } else {
                    output = PathBuf::from(value);
                }
            }
            _ if name.starts_with('-') => return Err(format!("unknown argument {name}")),
            _ => positional.push(PathBuf::from(name)),
        }
        index += 1;
    }
    if positional.len() != 2 {
        return Err("expected source catalog and FITS path".into());
    }
    let result = run(
        &positional[0],
        &positional[1],
        &targets,
        DEFAULT_LOCAL_MAX_RADII,
    )?;
    write_artifacts(&result, &output)?;
    Ok(json!({
        "tool": "rst19-source-peak-consistency",
        "out_dir": output,
        "result_path": output.join("source_peak_consistency.json"),
        "result": result.as_dict(),
    }))
}

const CLI_HELP: &str = "把 source_catalog 候选放回原始 FITS，审计 3x3/5x5/7x7 raw 局部峰一致性\n\nUSAGE: rst19-source-peak-consistency CATALOG.csv FRAME.fits [OPTIONS]\n\nOptions:\n  --target-id ID     重复传入以导出指定 detection_id 的局部峰位置\n  --out-dir DIR      [default: tmp/source-peak-consistency]\n  -h, --help";

pub fn run_cli(args: &[String]) -> i32 {
    if args
        .iter()
        .any(|argument| argument == "-h" || argument == "--help")
    {
        println!("{CLI_HELP}");
        return 0;
    }
    match run_desktop(args) {
        Ok(payload) => {
            println!(
                "rst19-source-peak-consistency: 结果已写入 {}",
                payload["out_dir"].as_str().unwrap_or("")
            );
            if let Some(rows) = payload["result"]["class_summaries"].as_array() {
                for row in rows {
                    println!(
                        "{}: n={} raw_local_max(r1/r2/r3)={}/{}/{}",
                        row["feature_class"].as_str().unwrap_or(""),
                        row["candidate_count"].as_u64().unwrap_or(0),
                        row["raw_local_maximum_count_r1"].as_u64().unwrap_or(0),
                        row["raw_local_maximum_count_r2"].as_u64().unwrap_or(0),
                        row["raw_local_maximum_count_r3"].as_u64().unwrap_or(0),
                    );
                }
            }
            println!("{}", payload["result"]["conclusion"].as_str().unwrap_or(""));
            0
        }
        Err(error) => {
            eprintln!("rst19-source-peak-consistency: {error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_root() -> PathBuf {
        std::env::temp_dir().join(format!(
            "rst19-peak-consistency-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn standalone_cli_help_is_available_without_python() {
        assert_eq!(run_cli(&["--help".into()]), 0);
    }

    fn fits_card(key: &str, value: &str) -> String {
        let mut card = format!("{key:<8}= {value:>20}");
        card.push_str(&" ".repeat(80 - card.len()));
        card
    }

    fn write_f32_fits(path: &Path, width: usize, height: usize, image: &[f32]) {
        let mut header = [
            fits_card("SIMPLE", "T"),
            fits_card("BITPIX", "-32"),
            fits_card("NAXIS", "2"),
            fits_card("NAXIS1", &width.to_string()),
            fits_card("NAXIS2", &height.to_string()),
            format!("{:<80}", "END"),
        ]
        .concat()
        .into_bytes();
        header.resize(
            header.len().div_ceil(fits::BLOCK_SIZE) * fits::BLOCK_SIZE,
            b' ',
        );
        let mut bytes = header;
        for value in image {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        bytes.resize(bytes.len().div_ceil(fits::BLOCK_SIZE) * fits::BLOCK_SIZE, 0);
        fs::write(path, bytes).unwrap();
    }

    fn write_catalog(path: &Path) {
        fs::write(
            path,
            "detection_id,feature_class,feature_class_label,quality_passed,peak_x,peak_y\r\n1,compact_quality,compact,True,5,5\r\n2,range_anomaly,range,False,8,5\r\n3,crowded_blend,blend,False,12,5\r\n",
        )
        .unwrap();
    }

    #[test]
    fn reports_raw_maximum_gaps_classes_and_targets() {
        let root = test_root();
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        let fits_path = root.join("frame.fits");
        write_catalog(&catalog);
        let mut image = vec![10.0_f32; 20 * 20];
        image[5 * 20 + 5] = 100.0;
        image[5 * 20 + 7] = 30.0;
        image[5 * 20 + 8] = 20.0;
        image[5 * 20 + 9] = 90.0;
        image[5 * 20 + 12] = 80.0;
        write_f32_fits(&fits_path, 20, 20, &image);
        let result = run(&catalog, &fits_path, &[2, 3], DEFAULT_LOCAL_MAX_RADII).unwrap();
        assert_eq!(result.rows[1].raw_local_max_gap_adu_r1, 70.0);
        assert!(result.rows[0].raw_unique_local_maximum_r1);
        assert_eq!(
            result.class_summaries[2].nonlocal_peak_gap_median_adu_r1,
            Some(70.0)
        );
        assert_eq!(result.targets.len(), 6);
        assert_eq!(
            (result.targets[0].local_max_x, result.targets[0].local_max_y),
            (9, 5)
        );
        assert_eq!(result.targets[0].local_max_gap_adu, 70.0);
        let out = write_artifacts(&result, &root.join("out")).unwrap();
        assert!(out.join("source_peak_consistency.csv").is_file());
        assert!(out
            .join("source_peak_consistency_class_summary.csv")
            .is_file());
        assert!(out.join("source_peak_consistency_targets.csv").is_file());
        assert!(out.join("source_peak_consistency.json").is_file());
        let cli_out = root.join("cli-out");
        let args = vec![
            catalog.to_string_lossy().into_owned(),
            fits_path.to_string_lossy().into_owned(),
            "--target-id".into(),
            "2".into(),
            "--out-dir".into(),
            cli_out.to_string_lossy().into_owned(),
        ];
        assert_eq!(run_cli(&args), 0);
        assert!(cli_out.join("source_peak_consistency.json").is_file());
    }

    #[test]
    fn rejects_bad_radius_target_and_coordinate_contracts() {
        let root = test_root();
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        let fits_path = root.join("frame.fits");
        write_catalog(&catalog);
        write_f32_fits(&fits_path, 20, 20, &vec![10.0; 20 * 20]);
        assert!(run(&catalog, &fits_path, &[], &[1, 2])
            .unwrap_err()
            .contains("stable radius set"));
        assert_eq!(
            run(&catalog, &fits_path, &[99], DEFAULT_LOCAL_MAX_RADII).unwrap_err(),
            "target detection_id not in source catalog: 99"
        );
        fs::write(
            &catalog,
            "detection_id,feature_class,feature_class_label,quality_passed,peak_x,peak_y\n1,compact,c,True,20,5\n",
        )
        .unwrap();
        assert!(run(&catalog, &fits_path, &[], DEFAULT_LOCAL_MAX_RADII)
            .unwrap_err()
            .contains("is outside image shape (20, 20)"));
    }
}
