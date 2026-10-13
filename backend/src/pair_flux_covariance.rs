//! Cross-frame forced-photometry covariance for a nearby pair of detections.
//!
//! The audit reads an existing forced-stability CSV and describes how two
//! detector responses co-vary relative to complete control-source pairs. It
//! does not rerun detection or interpret correlation as physical identity.

use crate::Result;
use csv::{Terminator, WriterBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

pub const PAIR_FLUX_METRICS: [&str; 2] = ["flux_snr", "local_peak_flux_snr"];

const CSV_HEADERS: [&str; 29] = [
    "metric_name",
    "valid_frame_count",
    "primary_pearson",
    "primary_spearman",
    "frame_median_normalized_pearson",
    "frame_median_normalized_spearman",
    "below_pair_total_median_frame_count",
    "above_pair_total_median_frame_count",
    "below_pair_total_median_pearson",
    "above_pair_total_median_pearson",
    "control_source_count",
    "control_pair_count",
    "control_pearson_p05",
    "control_pearson_median",
    "control_pearson_p95",
    "control_normalized_pearson_p05",
    "control_normalized_pearson_median",
    "control_normalized_pearson_p95",
    "raw_upper_tail_fraction",
    "normalized_upper_tail_fraction",
    "primary_fraction_median",
    "primary_fraction_min",
    "primary_fraction_max",
    "primary_fraction_range",
    "primary_fraction_mad_scaled",
    "below_pair_total_median_primary_fraction_median",
    "above_pair_total_median_primary_fraction_median",
    "primary_fraction_shift_high_minus_low",
    "control_fraction_shift_pair_count",
];
const CSV_HEADERS_EXTRA: [&str; 3] = [
    "control_fraction_shift_abs_median",
    "control_fraction_shift_abs_p95",
    "absolute_fraction_shift_upper_tail_fraction",
];
const NOTE: &str = "这里的量是同一强制测光流程产生的 flux_snr/local_peak_flux_snr；控制分布用于发现异常共变，不是正式显著性检验，也不是双星概率。";
const INTERPRETATION: &str = "共变只能说明局部响应可能共享结构、孔径或值域状态，不能单独区分混合星、异常码和真实近邻，也不是双星概率；正式结论仍需原始值域、PSF、位置协方差、WCS/星表唯一性或已知真值注入共同支持。";

#[derive(Debug, Clone, Serialize)]
pub struct ForcedStabilityFluxRow {
    pub detection_id: i64,
    pub frame_index: i64,
    pub flux_snr: Option<f64>,
    pub local_peak_flux_snr: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairFluxCovarianceMetric {
    pub metric_name: String,
    pub valid_frame_count: usize,
    pub primary_pearson: Option<f64>,
    pub primary_spearman: Option<f64>,
    pub frame_median_normalized_pearson: Option<f64>,
    pub frame_median_normalized_spearman: Option<f64>,
    pub below_pair_total_median_frame_count: usize,
    pub above_pair_total_median_frame_count: usize,
    pub below_pair_total_median_pearson: Option<f64>,
    pub above_pair_total_median_pearson: Option<f64>,
    pub control_source_count: usize,
    pub control_pair_count: usize,
    pub control_pearson_p05: Option<f64>,
    pub control_pearson_median: Option<f64>,
    pub control_pearson_p95: Option<f64>,
    pub control_normalized_pearson_p05: Option<f64>,
    pub control_normalized_pearson_median: Option<f64>,
    pub control_normalized_pearson_p95: Option<f64>,
    pub raw_upper_tail_fraction: Option<f64>,
    pub normalized_upper_tail_fraction: Option<f64>,
    pub primary_fraction_median: Option<f64>,
    pub primary_fraction_min: Option<f64>,
    pub primary_fraction_max: Option<f64>,
    pub primary_fraction_range: Option<f64>,
    pub primary_fraction_mad_scaled: Option<f64>,
    pub below_pair_total_median_primary_fraction_median: Option<f64>,
    pub above_pair_total_median_primary_fraction_median: Option<f64>,
    pub primary_fraction_shift_high_minus_low: Option<f64>,
    pub control_fraction_shift_pair_count: usize,
    pub control_fraction_shift_abs_median: Option<f64>,
    pub control_fraction_shift_abs_p95: Option<f64>,
    pub absolute_fraction_shift_upper_tail_fraction: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairFluxCovarianceAuditResult {
    pub frame_count: usize,
    pub primary_detection_id: i64,
    pub secondary_detection_id: i64,
    pub control_source_count: usize,
    pub metrics: Vec<PairFluxCovarianceMetric>,
    pub parameters: Value,
    pub conclusion: String,
}

impl PairFluxCovarianceAuditResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "frame_count": self.frame_count,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "control_source_count": self.control_source_count,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "conclusion": self.conclusion,
        })
    }
}

fn optional_float(raw: Option<&str>, field: &str) -> Result<Option<f64>> {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let value = raw
        .parse::<f64>()
        .map_err(|_| format!("forced stability field {field:?} is not numeric: {raw:?}"))?;
    Ok(value.is_finite().then_some(value))
}

fn optional_integer(raw: Option<&str>) -> Result<i64> {
    let raw = raw.unwrap_or_default().trim();
    let value = raw
        .parse::<f64>()
        .map_err(|_| "id/frame is invalid".to_owned())?;
    if !value.is_finite() {
        return Err("id/frame is invalid".into());
    }
    Ok(value.trunc() as i64)
}

pub fn load_forced_stability_flux_csv(path: &Path) -> Result<Vec<ForcedStabilityFluxRow>> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let mut reader = csv::ReaderBuilder::new().from_reader(bytes);
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let index = |name: &str| headers.iter().position(|header| header == name);
    let required = [
        "detection_id",
        "frame_index",
        "flux_snr",
        "local_peak_flux_snr",
    ];
    let missing: Vec<_> = required
        .iter()
        .filter(|name| index(name).is_none())
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "forced stability CSV is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let mut rows = Vec::new();
    for (row_index, record) in reader.records().enumerate() {
        let record = record.map_err(|error| error.to_string())?;
        let get = |name: &str| index(name).and_then(|column| record.get(column));
        let detection_id = optional_integer(get("detection_id")).map_err(|_| {
            format!(
                "invalid forced stability row {}: id/frame is invalid",
                row_index + 2
            )
        })?;
        let frame_index = optional_integer(get("frame_index")).map_err(|_| {
            format!(
                "invalid forced stability row {}: id/frame is invalid",
                row_index + 2
            )
        })?;
        rows.push(ForcedStabilityFluxRow {
            detection_id,
            frame_index,
            flux_snr: optional_float(get("flux_snr"), "flux_snr")?,
            local_peak_flux_snr: optional_float(get("local_peak_flux_snr"), "local_peak_flux_snr")?,
        });
    }
    Ok(rows)
}

fn metric_value(row: &ForcedStabilityFluxRow, metric: &str) -> Result<Option<f64>> {
    match metric {
        "flux_snr" => Ok(row.flux_snr),
        "local_peak_flux_snr" => Ok(row.local_peak_flux_snr),
        _ => Err(format!("unsupported pair flux metric: {metric:?}")),
    }
}

fn finite_pair(left: &[Option<f64>], right: &[Option<f64>]) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (x, y) in left.iter().zip(right) {
        if let (Some(x), Some(y)) = (x, y) {
            if x.is_finite() && y.is_finite() {
                xs.push(*x);
                ys.push(*y);
            }
        }
    }
    (xs, ys)
}

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 3 {
        return None;
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return None;
    }
    let mean_x = left.iter().sum::<f64>() / left.len() as f64;
    let mean_y = right.iter().sum::<f64>() / right.len() as f64;
    let centered_x: Vec<_> = left.iter().map(|value| value - mean_x).collect();
    let centered_y: Vec<_> = right.iter().map(|value| value - mean_y).collect();
    let sum_x = centered_x.iter().map(|value| value * value).sum::<f64>();
    let sum_y = centered_y.iter().map(|value| value * value).sum::<f64>();
    let denominator = (sum_x * sum_y).sqrt();
    if !denominator.is_finite() || denominator <= f64::EPSILON {
        return None;
    }
    let numerator = centered_x
        .iter()
        .zip(&centered_y)
        .map(|(x, y)| x * y)
        .sum::<f64>();
    let value = numerator / denominator;
    value.is_finite().then_some(value)
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start;
        while end + 1 < order.len() && values[order[end + 1]] == values[order[start]] {
            end += 1;
        }
        let rank = (start + end) as f64 / 2.0 + 1.0;
        for index in start..=end {
            ranks[order[index]] = rank;
        }
        start = end + 1;
    }
    ranks
}

fn spearman(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 3 {
        return None;
    }
    pearson(&average_ranks(left), &average_ranks(right))
}

fn percentile(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let location = (sorted.len() - 1) as f64 * percentile / 100.0;
    let low = location.floor() as usize;
    let high = location.ceil() as usize;
    if low == high {
        return Some(sorted[low]);
    }
    Some(sorted[low] + (sorted[high] - sorted[low]) * (location - low as f64))
}

fn upper_tail(target: Option<f64>, controls: &[f64]) -> Option<f64> {
    let target = target?;
    if controls.is_empty() {
        return None;
    }
    let count = controls.iter().filter(|value| **value >= target).count();
    Some((1 + count) as f64 / (1 + controls.len()) as f64)
}

fn scaled_mad(values: &[f64]) -> Option<f64> {
    let median = percentile(values, 50.0)?;
    let deviations: Vec<_> = values.iter().map(|value| (value - median).abs()).collect();
    percentile(&deviations, 50.0).map(|value| 1.4826 * value)
}

fn fraction_shift_high_minus_low(primary: &[f64], secondary: &[f64]) -> Option<f64> {
    if primary.len() != secondary.len() || primary.len() < 3 {
        return None;
    }
    let mut totals = Vec::with_capacity(primary.len());
    let mut fractions = Vec::with_capacity(primary.len());
    for (primary_value, secondary_value) in primary.iter().zip(secondary) {
        if !primary_value.is_finite() || !secondary_value.is_finite() {
            return None;
        }
        let total = primary_value + secondary_value;
        if !total.is_finite() || total.abs() <= f64::EPSILON {
            return None;
        }
        totals.push(total);
        fractions.push(primary_value / total);
    }
    let total_median = percentile(&totals, 50.0)?;
    let low: Vec<_> = totals
        .iter()
        .zip(&fractions)
        .filter(|(total, _)| **total < total_median)
        .map(|(_, fraction)| *fraction)
        .collect();
    let high: Vec<_> = totals
        .iter()
        .zip(&fractions)
        .filter(|(total, _)| **total >= total_median)
        .map(|(_, fraction)| *fraction)
        .collect();
    Some(percentile(&high, 50.0)? - percentile(&low, 50.0)?)
}

fn python_float_string(value: f64) -> String {
    let mut raw = format!("{value:?}");
    let negative = raw.starts_with('-');
    if negative {
        raw.remove(0);
    }
    let (mantissa, exponent) = raw
        .split_once(['e', 'E'])
        .map(|(mantissa, exponent)| (mantissa, exponent.parse::<i32>().unwrap_or(0)))
        .unwrap_or((raw.as_str(), 0));
    let decimal_position = mantissa.find('.').unwrap_or(mantissa.len()) as i32 + exponent;
    let mut digits: String = mantissa
        .chars()
        .filter(|character| *character != '.')
        .collect();
    let leading_zeroes = digits.chars().take_while(|digit| *digit == '0').count();
    digits.drain(..leading_zeroes);
    let decimal_position = decimal_position - leading_zeroes as i32;
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    if digits.is_empty() {
        return if negative { "-0.0" } else { "0.0" }.into();
    }
    let sign = if negative { "-" } else { "" };
    if !(-3..=15).contains(&decimal_position) {
        let scientific_exponent = decimal_position - 1;
        let exponent_sign = if scientific_exponent >= 0 { "+" } else { "-" };
        let exponent_digits = format!("{:02}", scientific_exponent.abs());
        let fraction = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        return format!(
            "{sign}{}{fraction}e{exponent_sign}{exponent_digits}",
            &digits[..1]
        );
    }
    if decimal_position <= 0 {
        return format!(
            "{sign}0.{}{}",
            "0".repeat((-decimal_position) as usize),
            digits
        );
    }
    if decimal_position as usize >= digits.len() {
        return format!(
            "{sign}{}{}.0",
            digits,
            "0".repeat(decimal_position as usize - digits.len())
        );
    }
    format!(
        "{sign}{}.{}",
        &digits[..decimal_position as usize],
        &digits[decimal_position as usize..]
    )
}

fn optional_python_float(value: Option<f64>) -> String {
    value
        .map(python_float_string)
        .unwrap_or_else(|| "None".into())
}

fn empty_metric(metric_name: &str, valid_frame_count: usize) -> PairFluxCovarianceMetric {
    PairFluxCovarianceMetric {
        metric_name: metric_name.into(),
        valid_frame_count,
        primary_pearson: None,
        primary_spearman: None,
        frame_median_normalized_pearson: None,
        frame_median_normalized_spearman: None,
        below_pair_total_median_frame_count: 0,
        above_pair_total_median_frame_count: 0,
        below_pair_total_median_pearson: None,
        above_pair_total_median_pearson: None,
        control_source_count: 0,
        control_pair_count: 0,
        control_pearson_p05: None,
        control_pearson_median: None,
        control_pearson_p95: None,
        control_normalized_pearson_p05: None,
        control_normalized_pearson_median: None,
        control_normalized_pearson_p95: None,
        raw_upper_tail_fraction: None,
        normalized_upper_tail_fraction: None,
        primary_fraction_median: None,
        primary_fraction_min: None,
        primary_fraction_max: None,
        primary_fraction_range: None,
        primary_fraction_mad_scaled: None,
        below_pair_total_median_primary_fraction_median: None,
        above_pair_total_median_primary_fraction_median: None,
        primary_fraction_shift_high_minus_low: None,
        control_fraction_shift_pair_count: 0,
        control_fraction_shift_abs_median: None,
        control_fraction_shift_abs_p95: None,
        absolute_fraction_shift_upper_tail_fraction: None,
    }
}

pub fn run(
    rows: &[ForcedStabilityFluxRow],
    primary_detection_id: i64,
    secondary_detection_id: i64,
    metrics: &[String],
) -> Result<PairFluxCovarianceAuditResult> {
    if primary_detection_id == secondary_detection_id {
        return Err("primary and secondary detection IDs must differ".into());
    }
    if metrics.is_empty() {
        return Err("metrics cannot be empty".into());
    }
    let mut metric_names = Vec::new();
    for metric in metrics {
        if !PAIR_FLUX_METRICS.contains(&metric.as_str()) {
            return Err(format!("unsupported pair flux metric: {metric:?}"));
        }
        if !metric_names.contains(metric) {
            metric_names.push(metric.clone());
        }
    }
    let mut by_id: BTreeMap<i64, BTreeMap<i64, &ForcedStabilityFluxRow>> = BTreeMap::new();
    for row in rows {
        let frame_rows = by_id.entry(row.detection_id).or_default();
        if frame_rows.insert(row.frame_index, row).is_some() {
            return Err(format!(
                "duplicate forced stability row for id={}, frame={}",
                row.detection_id, row.frame_index
            ));
        }
    }
    let primary_rows = by_id
        .get(&primary_detection_id)
        .ok_or("both target detection IDs must be present in forced stability rows")?;
    let secondary_rows = by_id
        .get(&secondary_detection_id)
        .ok_or("both target detection IDs must be present in forced stability rows")?;
    let frame_indices: Vec<i64> = primary_rows
        .keys()
        .filter(|frame| secondary_rows.contains_key(frame))
        .copied()
        .collect();
    if frame_indices.len() < 3 {
        return Err("target pair needs at least three common frames".into());
    }
    let target_ids = BTreeSet::from([primary_detection_id, secondary_detection_id]);
    let control_ids: Vec<i64> = by_id
        .iter()
        .filter(|(source_id, frame_rows)| {
            !target_ids.contains(source_id)
                && frame_indices
                    .iter()
                    .all(|frame_index| frame_rows.contains_key(frame_index))
        })
        .map(|(source_id, _)| *source_id)
        .collect();
    let mut metrics_out = Vec::new();
    for metric_name in &metric_names {
        let primary_values: Vec<_> = frame_indices
            .iter()
            .map(|frame| metric_value(primary_rows[frame], metric_name))
            .collect::<Result<_>>()?;
        let secondary_values: Vec<_> = frame_indices
            .iter()
            .map(|frame| metric_value(secondary_rows[frame], metric_name))
            .collect::<Result<_>>()?;
        let (primary_series, _secondary_series) = finite_pair(&primary_values, &secondary_values);
        if primary_series.len() < 3 {
            metrics_out.push(empty_metric(metric_name, primary_series.len()));
            continue;
        }
        let mut normalized_control_ids = Vec::new();
        for source_id in &control_ids {
            let values: Vec<_> = frame_indices
                .iter()
                .map(|frame| metric_value(by_id[source_id][frame], metric_name))
                .collect::<Result<_>>()?;
            if values
                .iter()
                .all(|value| value.is_some_and(|value| value.is_finite()))
            {
                normalized_control_ids.push(*source_id);
            }
        }
        let mut frame_medians = BTreeMap::new();
        for frame in &frame_indices {
            let values: Vec<f64> = normalized_control_ids
                .iter()
                .filter_map(|source_id| {
                    metric_value(by_id[source_id][frame], metric_name)
                        .ok()
                        .flatten()
                })
                .filter(|value| value.is_finite())
                .collect();
            if let Some(median) = percentile(&values, 50.0) {
                frame_medians.insert(*frame, median);
            }
        }
        let mut normalized_primary = Vec::new();
        let mut normalized_secondary = Vec::new();
        let mut raw_primary = Vec::new();
        let mut raw_secondary = Vec::new();
        let mut pair_totals = Vec::new();
        let mut split_fractions = Vec::new();
        let mut pair_fractions: Vec<Option<f64>> = Vec::new();
        for (index, frame) in frame_indices.iter().enumerate() {
            let (Some(primary_value), Some(secondary_value)) =
                (primary_values[index], secondary_values[index])
            else {
                continue;
            };
            if !primary_value.is_finite() || !secondary_value.is_finite() {
                continue;
            }
            raw_primary.push(primary_value);
            raw_secondary.push(secondary_value);
            let total = primary_value + secondary_value;
            pair_totals.push(total);
            if total.abs() > f64::EPSILON {
                let fraction = primary_value / total;
                split_fractions.push(fraction);
                pair_fractions.push(Some(fraction));
            } else {
                pair_fractions.push(None);
            }
            if let Some(scale) = frame_medians.get(frame) {
                if scale.abs() > f64::EPSILON {
                    normalized_primary.push(primary_value / scale);
                    normalized_secondary.push(secondary_value / scale);
                }
            }
        }
        let mut raw_controls = Vec::new();
        let mut normalized_controls = Vec::new();
        let mut control_fraction_shift_abs = Vec::new();
        for left_index in 0..normalized_control_ids.len() {
            for right_index in left_index + 1..normalized_control_ids.len() {
                let left_id = normalized_control_ids[left_index];
                let right_id = normalized_control_ids[right_index];
                let left: Vec<_> = frame_indices
                    .iter()
                    .map(|frame| metric_value(by_id[&left_id][frame], metric_name))
                    .collect::<Result<_>>()?;
                let right: Vec<_> = frame_indices
                    .iter()
                    .map(|frame| metric_value(by_id[&right_id][frame], metric_name))
                    .collect::<Result<_>>()?;
                let (left_series, right_series) = finite_pair(&left, &right);
                if let Some(value) = pearson(&left_series, &right_series) {
                    raw_controls.push(value);
                }
                if let Some(value) = fraction_shift_high_minus_low(&left_series, &right_series) {
                    control_fraction_shift_abs.push(value.abs());
                }
                if frame_medians.len() == frame_indices.len() {
                    let left_normalized: Vec<_> = left_series
                        .iter()
                        .zip(&frame_indices)
                        .map(|(value, frame)| value / frame_medians[frame])
                        .collect();
                    let right_normalized: Vec<_> = right_series
                        .iter()
                        .zip(&frame_indices)
                        .map(|(value, frame)| value / frame_medians[frame])
                        .collect();
                    if let Some(value) = pearson(&left_normalized, &right_normalized) {
                        normalized_controls.push(value);
                    }
                }
            }
        }
        let primary_pearson = pearson(&raw_primary, &raw_secondary);
        let normalized_pearson = pearson(&normalized_primary, &normalized_secondary);
        let total_median = percentile(&pair_totals, 50.0).unwrap();
        let below_indices: Vec<_> = pair_totals
            .iter()
            .enumerate()
            .filter(|(_, total)| **total < total_median)
            .map(|(index, _)| index)
            .collect();
        let above_indices: Vec<_> = pair_totals
            .iter()
            .enumerate()
            .filter(|(_, total)| **total >= total_median)
            .map(|(index, _)| index)
            .collect();
        let below_pearson = pearson(
            &below_indices
                .iter()
                .map(|index| raw_primary[*index])
                .collect::<Vec<_>>(),
            &below_indices
                .iter()
                .map(|index| raw_secondary[*index])
                .collect::<Vec<_>>(),
        );
        let above_pearson = pearson(
            &above_indices
                .iter()
                .map(|index| raw_primary[*index])
                .collect::<Vec<_>>(),
            &above_indices
                .iter()
                .map(|index| raw_secondary[*index])
                .collect::<Vec<_>>(),
        );
        let below_fractions: Vec<_> = below_indices
            .iter()
            .filter_map(|index| pair_fractions[*index])
            .collect();
        let above_fractions: Vec<_> = above_indices
            .iter()
            .filter_map(|index| pair_fractions[*index])
            .collect();
        let primary_fraction_median = percentile(&split_fractions, 50.0);
        let primary_fraction_min = percentile(&split_fractions, 0.0);
        let primary_fraction_max = percentile(&split_fractions, 100.0);
        let primary_fraction_range = primary_fraction_min
            .zip(primary_fraction_max)
            .map(|(minimum, maximum)| maximum - minimum);
        let below_fraction_median = percentile(&below_fractions, 50.0);
        let above_fraction_median = percentile(&above_fractions, 50.0);
        let primary_fraction_shift = below_fraction_median
            .zip(above_fraction_median)
            .map(|(low, high)| high - low);
        let absolute_fraction_shift = primary_fraction_shift.map(f64::abs);
        metrics_out.push(PairFluxCovarianceMetric {
            metric_name: metric_name.clone(),
            valid_frame_count: raw_primary.len(),
            primary_pearson,
            primary_spearman: spearman(&raw_primary, &raw_secondary),
            frame_median_normalized_pearson: normalized_pearson,
            frame_median_normalized_spearman: spearman(&normalized_primary, &normalized_secondary),
            below_pair_total_median_frame_count: below_indices.len(),
            above_pair_total_median_frame_count: above_indices.len(),
            below_pair_total_median_pearson: below_pearson,
            above_pair_total_median_pearson: above_pearson,
            control_source_count: normalized_control_ids.len(),
            control_pair_count: raw_controls.len(),
            control_pearson_p05: percentile(&raw_controls, 5.0),
            control_pearson_median: percentile(&raw_controls, 50.0),
            control_pearson_p95: percentile(&raw_controls, 95.0),
            control_normalized_pearson_p05: percentile(&normalized_controls, 5.0),
            control_normalized_pearson_median: percentile(&normalized_controls, 50.0),
            control_normalized_pearson_p95: percentile(&normalized_controls, 95.0),
            raw_upper_tail_fraction: upper_tail(primary_pearson, &raw_controls),
            normalized_upper_tail_fraction: upper_tail(normalized_pearson, &normalized_controls),
            primary_fraction_median,
            primary_fraction_min,
            primary_fraction_max,
            primary_fraction_range,
            primary_fraction_mad_scaled: scaled_mad(&split_fractions),
            below_pair_total_median_primary_fraction_median: below_fraction_median,
            above_pair_total_median_primary_fraction_median: above_fraction_median,
            primary_fraction_shift_high_minus_low: primary_fraction_shift,
            control_fraction_shift_pair_count: control_fraction_shift_abs.len(),
            control_fraction_shift_abs_median: percentile(&control_fraction_shift_abs, 50.0),
            control_fraction_shift_abs_p95: percentile(&control_fraction_shift_abs, 95.0),
            absolute_fraction_shift_upper_tail_fraction: upper_tail(
                absolute_fraction_shift,
                &control_fraction_shift_abs,
            ),
        });
    }
    let descriptions = metrics_out
        .iter()
        .map(|metric| {
            format!(
                "{}: Pearson={}, 按帧中位数归一后={}, 控制上尾={}",
                metric.metric_name,
                optional_python_float(metric.primary_pearson),
                optional_python_float(metric.frame_median_normalized_pearson),
                optional_python_float(metric.normalized_upper_tail_fraction)
            )
        })
        .collect::<Vec<_>>();
    let conclusion = format!(
        "pair 强制测光共变审计完成：{} 帧，控制源 {} 个；{}。{}",
        frame_indices.len(),
        control_ids.len(),
        descriptions.join("；"),
        INTERPRETATION
    );
    Ok(PairFluxCovarianceAuditResult {
        frame_count: frame_indices.len(),
        primary_detection_id,
        secondary_detection_id,
        control_source_count: control_ids.len(),
        metrics: metrics_out,
        parameters: json!({
            "metric_names": metric_names,
            "frame_indices": frame_indices,
            "normalization": "divide by the same-frame median across other complete sources",
            "control_pair_scope": "all complete source pairs excluding the two targets",
            "upper_tail_is_not_p_value": true,
            "interpretation_boundary": "diagnostic of shared response, not binary-star probability",
            "primary_fraction_definition": "primary metric / (primary metric + secondary metric) per common frame",
            "primary_fraction_mad_scale": 1.4826,
            "pair_total_split": "below < median and above >= median of the target pair total",
            "control_fraction_shift_definition": "absolute median(f_high) - median(f_low) over complete control pairs",
            "control_fraction_shift_upper_tail_is_not_p_value": true
        }),
        conclusion,
    })
}

pub fn write_artifacts(
    result: &PairFluxCovarianceAuditResult,
    output_dir: &Path,
) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    let mut file = File::create(output_dir.join("pair_flux_covariance.csv"))
        .map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .terminator(Terminator::CRLF)
        .from_writer(file);
    let headers: Vec<_> = CSV_HEADERS
        .iter()
        .chain(CSV_HEADERS_EXTRA.iter())
        .copied()
        .collect();
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for metric in &result.metrics {
        writer
            .serialize(metric)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    let mut payload = result.as_dict();
    payload["note"] = Value::String(NOTE.into());
    fs::write(
        output_dir.join("pair_flux_covariance.json"),
        serde_json::to_vec_pretty(&payload).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    fs::canonicalize(output_dir).map_err(|error| error.to_string())
}

fn option_value<'a>(
    args: &'a [String],
    index: &mut usize,
    name: &str,
    inline: Option<&'a str>,
) -> Result<&'a str> {
    if let Some(value) = inline {
        Ok(value)
    } else {
        *index += 1;
        args.get(*index)
            .map(String::as_str)
            .ok_or_else(|| format!("{name} requires a value"))
    }
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut csv_path = None;
    let mut primary_id = None;
    let mut secondary_id = None;
    let mut metrics = Vec::<String>::new();
    let mut output_dir = PathBuf::from("tmp/pair-flux-covariance-audit");
    let mut index = 0;
    while index < args.len() {
        let (name, inline) = args[index]
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((args[index].as_str(), None));
        match name {
            "--primary-id" | "--secondary-id" | "--metric" | "--out-dir" => {
                let raw = option_value(args, &mut index, name, inline)?;
                match name {
                    "--primary-id" => {
                        primary_id = Some(
                            raw.parse::<i64>()
                                .map_err(|error| format!("invalid value for {name}: {error}"))?,
                        )
                    }
                    "--secondary-id" => {
                        secondary_id = Some(
                            raw.parse::<i64>()
                                .map_err(|error| format!("invalid value for {name}: {error}"))?,
                        )
                    }
                    "--metric" => metrics.push(raw.into()),
                    _ => output_dir = PathBuf::from(raw),
                }
            }
            _ if name.starts_with('-') => return Err(format!("unknown argument {name}")),
            _ => {
                if csv_path.replace(PathBuf::from(name)).is_some() {
                    return Err("only one forced stability CSV is allowed".into());
                }
            }
        }
        index += 1;
    }
    let csv_path = csv_path.ok_or("a forced stability CSV is required")?;
    let primary_id = primary_id.ok_or("--primary-id is required")?;
    let secondary_id = secondary_id.ok_or("--secondary-id is required")?;
    if metrics.is_empty() {
        metrics.extend(PAIR_FLUX_METRICS.iter().map(|metric| (*metric).into()));
    }
    let rows = load_forced_stability_flux_csv(&csv_path)?;
    let result = run(&rows, primary_id, secondary_id, &metrics)?;
    let output_dir = write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-pair-covariance-audit",
        "out_dir": output_dir,
        "result_path": output_dir.join("pair_flux_covariance.json"),
        "result": result.as_dict(),
        "note": NOTE
    }))
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("rst19-pair-covariance-audit FORCED_METRICS_CSV --primary-id ID --secondary-id ID [--metric flux_snr|local_peak_flux_snr]... [--out-dir DIR]");
        return Ok(());
    }
    let result = run_desktop(args)?;
    println!(
        "rst19-pair-covariance-audit: 审计已写入 {}",
        result["out_dir"].as_str().unwrap_or_default()
    );
    println!(
        "{}",
        result["result"]["conclusion"].as_str().unwrap_or_default()
    );
    if let Some(metrics) = result["result"]["metrics"].as_array() {
        for metric in metrics {
            println!(
                "{}: pearson={} normalized={} control_tail={}",
                metric["metric_name"].as_str().unwrap_or_default(),
                metric["primary_pearson"],
                metric["frame_median_normalized_pearson"],
                metric["normalized_upper_tail_fraction"]
            );
        }
    }
    Ok(())
}

pub fn handle_request(request: &Value) -> Result<Value> {
    let csv_path = request["forced_metrics_csv"]
        .as_str()
        .ok_or("missing forced_metrics_csv")?;
    let primary_id = request["primary_detection_id"]
        .as_i64()
        .ok_or("missing primary_detection_id")?;
    let secondary_id = request["secondary_detection_id"]
        .as_i64()
        .ok_or("missing secondary_detection_id")?;
    let metrics = request
        .get("metrics")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .ok_or("metrics must contain strings")
                })
                .collect::<std::result::Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_else(|| {
            PAIR_FLUX_METRICS
                .iter()
                .map(|metric| (*metric).into())
                .collect()
        });
    let rows = load_forced_stability_flux_csv(Path::new(csv_path))?;
    let result = run(&rows, primary_id, secondary_id, &metrics)?;
    let output = if let Some(path) = request.get("output_dir").and_then(Value::as_str) {
        let output = write_artifacts(&result, Path::new(path))?;
        json!({"out_dir":output, "result_path":output.join("pair_flux_covariance.json")})
    } else {
        Value::Null
    };
    Ok(json!({"result":result.as_dict(), "output":output, "note":NOTE}))
}

#[cfg(test)]
mod tests {
    use super::python_float_string;

    #[test]
    fn conclusion_float_text_matches_python_exponent_rules() {
        assert_eq!(python_float_string(1.0), "1.0");
        assert_eq!(python_float_string(0.0001), "0.0001");
        assert_eq!(python_float_string(0.00001), "1e-05");
        assert_eq!(python_float_string(-0.00001), "-1e-05");
        assert_eq!(python_float_string(1e16), "1e+16");
    }
}
