//! Temporal consistency summaries over an existing per-frame aperture CSV.
//!
//! The audit joins frame measurements to a source catalog by detection ID and
//! keeps catalog classes separate from explicitly artifact-local frame classes.
//! Its summaries describe detector responses, not physical-source identity.

use csv::{Terminator, WriterBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

pub const TEMPORAL_REQUIRED_FIELDS: [&str; 4] = [
    "flux_snr_raw",
    "negative_extreme_count",
    "repeated_code_count",
    "zero_count",
];
pub const FEATURE_CLASS_SOURCE_CHOICES: [&str; 2] = ["catalog", "frame"];
type SourceCatalog = BTreeMap<i64, (String, String)>;
type FeatureClassLabels = HashMap<String, String>;
const GUARDRAILS: [&str; 5] = [
    "逐帧响应来自固定 detector 坐标/孔径口径，不是星表身份匹配。",
    "稳健波动、正值比例和帧间重复是描述量，不是恒星概率、precision 或 FDR。",
    "feature_class 是检测器旗标优先级标签；部分 SNR/值域字段与其存在定义重叠。",
    "缺少 WCS、空间变化 PSF 和官方真值时，不能把响应持久升级为物理恒星确认。",
    "detection_id 只在同一实验产物内有意义；跨产物连接必须先通过类别血缘校验。",
];

#[derive(Debug, Clone, Serialize)]
pub struct TemporalConsistencySource {
    pub detection_id: i64,
    pub feature_class: String,
    pub feature_class_label: String,
    pub expected_frame_count: usize,
    pub frame_count: usize,
    pub finite_flux_frame_count: usize,
    pub coverage_fraction: f64,
    pub flux_snr_median: f64,
    pub flux_snr_mad_scaled: f64,
    pub flux_snr_relative_mad: f64,
    pub flux_snr_positive_fraction: f64,
    pub flux_snr_ge5_fraction: f64,
    pub flux_snr_sign_flip_count: usize,
    pub zero_frame_fraction: f64,
    pub negative_extreme_frame_fraction: f64,
    pub repeated_code_frame_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TemporalConsistencyClassSummary {
    pub feature_class: String,
    pub feature_class_label: String,
    pub source_count: usize,
    pub frame_count_median: f64,
    pub flux_snr_median_median: f64,
    pub flux_snr_relative_mad_p10: f64,
    pub flux_snr_relative_mad_median: f64,
    pub flux_snr_relative_mad_p90: f64,
    pub flux_snr_positive_fraction_median: f64,
    pub flux_snr_ge5_fraction_median: f64,
    pub sign_flip_count_median: f64,
    pub source_any_sign_flip_fraction: f64,
    pub source_all_positive_fraction: f64,
    pub source_any_zero_fraction: f64,
    pub source_any_negative_extreme_fraction: f64,
    pub source_any_repeated_code_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TemporalConsistencyAuditResult {
    pub catalog_path: String,
    pub frame_metrics_path: String,
    pub feature_class_source: String,
    pub expected_frame_count: usize,
    pub selected_detection_ids: Vec<i64>,
    pub feature_classes: Vec<String>,
    pub source_rows: Vec<TemporalConsistencySource>,
    pub class_summaries: Vec<TemporalConsistencyClassSummary>,
}

impl TemporalConsistencyAuditResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "frame_metrics_path": self.frame_metrics_path,
            "feature_class_source": self.feature_class_source,
            "expected_frame_count": self.expected_frame_count,
            "selected_detection_ids": self.selected_detection_ids,
            "feature_classes": self.feature_classes,
            "source_rows": self.source_rows,
            "class_summaries": self.class_summaries,
            "interpretation_guardrails": GUARDRAILS,
        })
    }
}

#[derive(Debug, Clone)]
struct FrameMetricRow {
    detection_id: i64,
    frame_index: i64,
    feature_class: Option<String>,
    flux_snr_raw: f64,
    negative_extreme_count: f64,
    repeated_code_count: f64,
    zero_count: f64,
}

fn csv_reader(path: &Path) -> Result<csv::Reader<std::io::Cursor<Vec<u8>>>, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    Ok(csv::ReaderBuilder::new().from_reader(std::io::Cursor::new(bytes.to_vec())))
}

fn headers<R: std::io::Read>(
    reader: &mut csv::Reader<R>,
) -> Result<(csv::StringRecord, HashMap<String, usize>), String> {
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let index = headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.to_owned(), index))
        .collect();
    Ok((headers, index))
}

fn field<'a>(record: &'a csv::StringRecord, index: &HashMap<String, usize>, name: &str) -> &'a str {
    index
        .get(name)
        .and_then(|column| record.get(*column))
        .unwrap_or_default()
}

fn required_id(raw: &str, field_name: &str) -> Result<i64, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(format!("CSV row has empty {field_name}"));
    }
    let value = raw
        .parse::<f64>()
        .map_err(|_| format!("CSV row has invalid {field_name}={raw:?}"))?;
    if !value.is_finite() {
        return Err(format!("CSV row has invalid {field_name}={raw:?}"));
    }
    let value = value.trunc();
    if value < 0.0 {
        return Err(format!("CSV row has negative {field_name}={value:.0}"));
    }
    Ok(value as i64)
}

fn required_float(raw: &str, field_name: &str) -> Result<f64, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(format!("CSV row has empty {field_name}"));
    }
    let value = raw
        .parse::<f64>()
        .map_err(|_| format!("CSV row has invalid {field_name}={raw:?}"))?;
    if !value.is_finite() {
        return Err(format!("CSV row has non-finite {field_name}={raw:?}"));
    }
    Ok(value)
}

fn read_catalog(path: &Path) -> Result<(SourceCatalog, FeatureClassLabels), String> {
    let mut reader =
        csv_reader(path).map_err(|_| format!("无法读取源级 CSV：{}", path.display()))?;
    let (_, index) = headers(&mut reader)?;
    let mut missing: Vec<_> = ["detection_id", "feature_class", "feature_class_label"]
        .into_iter()
        .filter(|name| !index.contains_key(*name))
        .collect();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let mut records = BTreeMap::new();
    let mut labels = HashMap::new();
    for record in reader.records() {
        let record = record.map_err(|error| error.to_string())?;
        let detection_id = required_id(field(&record, &index, "detection_id"), "detection_id")?;
        if records.contains_key(&detection_id) {
            return Err(format!(
                "source catalog contains duplicate detection_id={detection_id}"
            ));
        }
        let feature_class = field(&record, &index, "feature_class").trim().to_owned();
        if feature_class.is_empty() {
            return Err(format!(
                "source catalog has empty feature_class for detection_id={detection_id}"
            ));
        }
        let label = field(&record, &index, "feature_class_label").trim();
        let label = if label.is_empty() {
            feature_class.clone()
        } else {
            label.to_owned()
        };
        records.insert(detection_id, (feature_class.clone(), label.clone()));
        labels.entry(feature_class).or_insert(label);
    }
    if records.is_empty() {
        return Err(format!("source catalog is empty: {}", path.display()));
    }
    Ok((records, labels))
}

fn aliased_field<'a>(
    record: &'a csv::StringRecord,
    index: &HashMap<String, usize>,
    field_name: &str,
) -> Result<&'a str, String> {
    let candidates: &[&str] = match field_name {
        "repeated_code_count" => &["repeated_code_count", "repeated_3990_3993_count"],
        _ => &[field_name],
    };
    candidates
        .iter()
        .find(|candidate| index.contains_key(**candidate))
        .map(|candidate| field(record, index, candidate))
        .ok_or_else(|| {
            format!(
                "CSV row has no field alias for {field_name}: expected {}",
                candidates.join("/")
            )
        })
}

fn read_frame_metrics(
    path: &Path,
    catalog: &SourceCatalog,
) -> Result<(BTreeMap<i64, Vec<FrameMetricRow>>, bool), String> {
    let mut reader =
        csv_reader(path).map_err(|_| format!("无法读取逐帧摘要 CSV：{}", path.display()))?;
    let (_, index) = headers(&mut reader)?;
    let required = [
        "detection_id",
        "frame_index",
        "flux_snr_raw",
        "negative_extreme_count",
        "zero_count",
    ];
    let mut missing: Vec<_> = required
        .into_iter()
        .filter(|name| !index.contains_key(*name))
        .collect();
    missing.sort_unstable();
    let has_repeated =
        index.contains_key("repeated_code_count") || index.contains_key("repeated_3990_3993_count");
    if !has_repeated {
        missing.push("repeated_code_count/repeated_3990_3993_count");
    }
    if !missing.is_empty() {
        return Err(format!(
            "frame metrics are missing required columns: {}",
            missing.join(", ")
        ));
    }
    let has_feature_class = index.contains_key("feature_class");
    let mut grouped: BTreeMap<i64, Vec<FrameMetricRow>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for record in reader.records() {
        let record = record.map_err(|error| error.to_string())?;
        let detection_id = required_id(field(&record, &index, "detection_id"), "detection_id")?;
        if !catalog.contains_key(&detection_id) {
            return Err(format!(
                "frame metrics detection_id={detection_id} is absent from source catalog"
            ));
        }
        let frame_index = required_id(field(&record, &index, "frame_index"), "frame_index")?;
        if !seen.insert((detection_id, frame_index)) {
            return Err(format!("frame metrics contains duplicate detection_id={detection_id}, frame_index={frame_index}"));
        }
        let flux_snr_raw = required_float(field(&record, &index, "flux_snr_raw"), "flux_snr_raw")?;
        let negative_extreme_count = required_float(
            field(&record, &index, "negative_extreme_count"),
            "negative_extreme_count",
        )?;
        let repeated_code_count = required_float(
            aliased_field(&record, &index, "repeated_code_count")?,
            "repeated_code_count",
        )?;
        let zero_count = required_float(field(&record, &index, "zero_count"), "zero_count")?;
        let feature_class =
            has_feature_class.then(|| field(&record, &index, "feature_class").trim().to_owned());
        grouped
            .entry(detection_id)
            .or_default()
            .push(FrameMetricRow {
                detection_id,
                frame_index,
                feature_class,
                flux_snr_raw,
                negative_extreme_count,
                repeated_code_count,
                zero_count,
            });
    }
    if grouped.is_empty() {
        return Err(format!("frame metrics are empty: {}", path.display()));
    }
    for rows in grouped.values_mut() {
        rows.sort_by_key(|row| row.frame_index);
    }
    Ok((grouped, has_feature_class))
}

fn percentile(values: &[f64], probability: f64) -> Result<f64, String> {
    if values.is_empty() {
        return Err("cannot summarize an empty class".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        Ok(sorted[lower])
    } else {
        Ok(sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64))
    }
}

fn source_summary(
    rows: &[FrameMetricRow],
    feature_class: &str,
    label: &str,
    expected_frame_count: usize,
) -> Result<TemporalConsistencySource, String> {
    let flux_values: Vec<_> = rows.iter().map(|row| row.flux_snr_raw).collect();
    let median = percentile(&flux_values, 0.5)?;
    let deviations: Vec<_> = flux_values
        .iter()
        .map(|value| (value - median).abs())
        .collect();
    let mad_scaled = 1.4826 * percentile(&deviations, 0.5)?;
    let positive = flux_values.iter().filter(|value| **value > 0.0).count();
    let ge5 = flux_values.iter().filter(|value| **value >= 5.0).count();
    let sign_flips = flux_values
        .windows(2)
        .filter(|pair| (pair[0] > 0.0) != (pair[1] > 0.0))
        .count();
    let count_fraction = |predicate: fn(&FrameMetricRow) -> bool| {
        rows.iter().filter(|row| predicate(row)).count() as f64 / rows.len() as f64
    };
    Ok(TemporalConsistencySource {
        detection_id: rows[0].detection_id,
        feature_class: feature_class.to_owned(),
        feature_class_label: label.to_owned(),
        expected_frame_count,
        frame_count: rows.len(),
        finite_flux_frame_count: flux_values.len(),
        coverage_fraction: rows.len() as f64 / expected_frame_count as f64,
        flux_snr_median: median,
        flux_snr_mad_scaled: mad_scaled,
        flux_snr_relative_mad: mad_scaled / median.abs().max(1.0),
        flux_snr_positive_fraction: positive as f64 / rows.len() as f64,
        flux_snr_ge5_fraction: ge5 as f64 / rows.len() as f64,
        flux_snr_sign_flip_count: sign_flips,
        zero_frame_fraction: count_fraction(|row| row.zero_count > 0.0),
        negative_extreme_frame_fraction: count_fraction(|row| row.negative_extreme_count > 0.0),
        repeated_code_frame_fraction: count_fraction(|row| row.repeated_code_count > 0.0),
    })
}

fn class_summary(
    feature_class: &str,
    rows: &[TemporalConsistencySource],
) -> Result<TemporalConsistencyClassSummary, String> {
    if rows.is_empty() {
        return Err(format!(
            "feature class has no temporal source rows: {feature_class:?}"
        ));
    }
    let label = &rows[0].feature_class_label;
    if rows.iter().any(|row| &row.feature_class_label != label) {
        return Err(format!(
            "feature class has inconsistent labels: {feature_class:?}"
        ));
    }
    let median_field = |select: fn(&TemporalConsistencySource) -> f64| -> Result<f64, String> {
        percentile(&rows.iter().map(select).collect::<Vec<_>>(), 0.5)
    };
    let fraction = |predicate: fn(&TemporalConsistencySource) -> bool| {
        rows.iter().filter(|row| predicate(row)).count() as f64 / rows.len() as f64
    };
    let relative_mads: Vec<_> = rows.iter().map(|row| row.flux_snr_relative_mad).collect();
    Ok(TemporalConsistencyClassSummary {
        feature_class: feature_class.into(),
        feature_class_label: label.clone(),
        source_count: rows.len(),
        frame_count_median: median_field(|row| row.frame_count as f64)?,
        flux_snr_median_median: median_field(|row| row.flux_snr_median)?,
        flux_snr_relative_mad_p10: percentile(&relative_mads, 0.10)?,
        flux_snr_relative_mad_median: percentile(&relative_mads, 0.5)?,
        flux_snr_relative_mad_p90: percentile(&relative_mads, 0.90)?,
        flux_snr_positive_fraction_median: median_field(|row| row.flux_snr_positive_fraction)?,
        flux_snr_ge5_fraction_median: median_field(|row| row.flux_snr_ge5_fraction)?,
        sign_flip_count_median: median_field(|row| row.flux_snr_sign_flip_count as f64)?,
        source_any_sign_flip_fraction: fraction(|row| row.flux_snr_sign_flip_count > 0),
        source_all_positive_fraction: fraction(|row| row.flux_snr_positive_fraction == 1.0),
        source_any_zero_fraction: fraction(|row| row.zero_frame_fraction > 0.0),
        source_any_negative_extreme_fraction: fraction(|row| {
            row.negative_extreme_frame_fraction > 0.0
        }),
        source_any_repeated_code_fraction: fraction(|row| row.repeated_code_frame_fraction > 0.0),
    })
}

fn resolve_feature_class(
    detection_id: i64,
    rows: &[FrameMetricRow],
    catalog: &SourceCatalog,
    feature_class_source: &str,
    frame_has_feature_class: bool,
) -> Result<String, String> {
    let catalog_class = &catalog[&detection_id].0;
    let frame_classes: BTreeSet<_> = rows
        .iter()
        .filter_map(|row| row.feature_class.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect();
    if feature_class_source == "catalog" {
        if !frame_classes.is_empty() && frame_classes != BTreeSet::from([catalog_class.clone()]) {
            return Err(format!("feature_class mismatch for detection_id={detection_id}: catalog={catalog_class:?}, frame={:?}; use feature_class_source='frame' only for an explicitly artifact-local table", frame_classes.iter().next().unwrap()));
        }
        return Ok(catalog_class.clone());
    }
    if feature_class_source != "frame" {
        return Err(format!(
            "feature_class_source must be one of {}",
            FEATURE_CLASS_SOURCE_CHOICES.join(", ")
        ));
    }
    if !frame_has_feature_class || frame_classes.is_empty() {
        return Err(
            "feature_class_source='frame' requires a non-empty feature_class column".into(),
        );
    }
    if frame_classes.len() != 1
        || rows.iter().any(|row| {
            row.feature_class
                .as_deref()
                .unwrap_or_default()
                .trim()
                .is_empty()
        })
    {
        return Err(format!(
            "frame metrics has inconsistent feature_class for detection_id={detection_id}"
        ));
    }
    Ok(frame_classes.into_iter().next().unwrap())
}

pub fn run(
    catalog_path: &Path,
    frame_metrics_path: &Path,
    expected_frame_count: usize,
    feature_classes: Option<&[String]>,
    feature_class_source: &str,
    detection_ids: Option<&[i64]>,
) -> Result<TemporalConsistencyAuditResult, String> {
    if expected_frame_count == 0 {
        return Err("expected_frame_count 必须为正整数".into());
    }
    if !FEATURE_CLASS_SOURCE_CHOICES.contains(&feature_class_source) {
        return Err(format!(
            "feature_class_source must be one of {}",
            FEATURE_CLASS_SOURCE_CHOICES.join(", ")
        ));
    }
    let (catalog, labels) = read_catalog(catalog_path)?;
    let (grouped, frame_has_feature_class) = read_frame_metrics(frame_metrics_path, &catalog)?;
    let selected_ids: Vec<i64> = if let Some(ids) = detection_ids {
        let mut seen = BTreeSet::new();
        let mut unique = Vec::new();
        for id in ids {
            if seen.insert(*id) {
                unique.push(*id);
            }
        }
        if unique.iter().any(|id| !grouped.contains_key(id)) {
            let id = unique.iter().find(|id| !grouped.contains_key(id)).unwrap();
            return Err(format!("frame metrics has no detection_id={id}"));
        }
        if unique.iter().any(|id| *id < 0) {
            return Err("detection_ids 必须为非负整数".into());
        }
        if unique.is_empty() {
            return Err("detection_ids 不能为空".into());
        }
        unique
    } else {
        grouped.keys().copied().collect()
    };
    let resolved: BTreeMap<i64, String> = selected_ids
        .iter()
        .map(|id| {
            Ok((
                *id,
                resolve_feature_class(
                    *id,
                    &grouped[id],
                    &catalog,
                    feature_class_source,
                    frame_has_feature_class,
                )?,
            ))
        })
        .collect::<Result<_, String>>()?;
    let sampled_classes: BTreeSet<_> = resolved
        .values()
        .filter(|feature_class| feature_class.as_str() != "other_rejected")
        .cloned()
        .collect();
    let requested_classes: Vec<String> = if let Some(classes) = feature_classes {
        let mut seen = BTreeSet::new();
        classes
            .iter()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .filter(|value| seen.insert((*value).to_owned()))
            .map(str::to_owned)
            .collect()
    } else {
        sampled_classes.into_iter().collect()
    };
    if requested_classes.is_empty() {
        return Err("feature_classes 不能为空，且逐帧表中没有可用类别".into());
    }
    let mut available_classes: BTreeSet<_> = catalog.values().map(|row| row.0.clone()).collect();
    available_classes.extend(resolved.values().cloned());
    if let Some(unknown) = requested_classes
        .iter()
        .find(|name| !available_classes.contains(*name))
    {
        return Err(format!("source catalog has no feature_class={unknown:?}"));
    }
    for name in &requested_classes {
        if !resolved.values().any(|value| value == name) {
            return Err(format!(
                "frame metrics has no sampled rows for feature_class={name:?}"
            ));
        }
    }
    let mut source_rows = Vec::new();
    for id in &selected_ids {
        let feature_class = &resolved[id];
        if requested_classes.contains(feature_class) {
            source_rows.push(source_summary(
                &grouped[id],
                feature_class,
                labels
                    .get(feature_class)
                    .map(String::as_str)
                    .unwrap_or(feature_class),
                expected_frame_count,
            )?);
        }
    }
    let mut class_summaries = Vec::new();
    for feature_class in &requested_classes {
        let rows: Vec<_> = source_rows
            .iter()
            .filter(|row| &row.feature_class == feature_class)
            .cloned()
            .collect();
        class_summaries.push(class_summary(feature_class, &rows)?);
    }
    Ok(TemporalConsistencyAuditResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        frame_metrics_path: frame_metrics_path.to_string_lossy().into_owned(),
        feature_class_source: feature_class_source.into(),
        expected_frame_count,
        selected_detection_ids: selected_ids,
        feature_classes: requested_classes,
        source_rows,
        class_summaries,
    })
}

pub fn write_artifacts(
    result: &TemporalConsistencyAuditResult,
    output_dir: &Path,
) -> Result<PathBuf, String> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    fs::write(
        output_dir.join("feature_temporal_consistency_audit.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    write_csv(
        &output_dir.join("feature_temporal_consistency_sources.csv"),
        &[
            "detection_id",
            "feature_class",
            "feature_class_label",
            "expected_frame_count",
            "frame_count",
            "finite_flux_frame_count",
            "coverage_fraction",
            "flux_snr_median",
            "flux_snr_mad_scaled",
            "flux_snr_relative_mad",
            "flux_snr_positive_fraction",
            "flux_snr_ge5_fraction",
            "flux_snr_sign_flip_count",
            "zero_frame_fraction",
            "negative_extreme_frame_fraction",
            "repeated_code_frame_fraction",
        ],
        &result.source_rows,
    )?;
    write_csv(
        &output_dir.join("feature_temporal_consistency_summary.csv"),
        &[
            "feature_class",
            "feature_class_label",
            "source_count",
            "frame_count_median",
            "flux_snr_median_median",
            "flux_snr_relative_mad_p10",
            "flux_snr_relative_mad_median",
            "flux_snr_relative_mad_p90",
            "flux_snr_positive_fraction_median",
            "flux_snr_ge5_fraction_median",
            "sign_flip_count_median",
            "source_any_sign_flip_fraction",
            "source_all_positive_fraction",
            "source_any_zero_fraction",
            "source_any_negative_extreme_fraction",
            "source_any_repeated_code_fraction",
        ],
        &result.class_summaries,
    )?;
    Ok(output_dir.to_path_buf())
}

fn write_csv<T: Serialize>(path: &Path, columns: &[&str], rows: &[T]) -> Result<(), String> {
    let mut file = File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .terminator(Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(columns)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer.serialize(row).map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    Ok(())
}

fn option_value<'a>(
    args: &'a [String],
    index: &mut usize,
    name: &str,
    inline: Option<&'a str>,
) -> Result<&'a str, String> {
    if let Some(value) = inline {
        Ok(value)
    } else {
        *index += 1;
        args.get(*index)
            .map(String::as_str)
            .ok_or_else(|| format!("{name} requires a value"))
    }
}

pub fn run_desktop(args: &[String]) -> Result<Value, String> {
    let mut catalog = None;
    let mut metrics = None;
    let mut classes = Vec::new();
    let mut expected_frames = 15usize;
    let mut class_source = "catalog".to_owned();
    let mut ids = Vec::new();
    let mut output_dir = PathBuf::from("tmp/feature-temporal-consistency-audit");
    let mut index = 0;
    while index < args.len() {
        let (name, inline) = args[index]
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((args[index].as_str(), None));
        match name {
            "--feature-class"
            | "--expected-frames"
            | "--feature-class-source"
            | "--detection-id"
            | "--out-dir" => {
                let raw = option_value(args, &mut index, name, inline)?;
                match name {
                    "--feature-class" => classes.push(raw.into()),
                    "--expected-frames" => {
                        expected_frames = raw
                            .parse()
                            .map_err(|error| format!("invalid value for {name}: {error}"))?
                    }
                    "--feature-class-source" => class_source = raw.into(),
                    "--detection-id" => ids.push(
                        raw.parse()
                            .map_err(|error| format!("invalid value for {name}: {error}"))?,
                    ),
                    _ => output_dir = PathBuf::from(raw),
                }
            }
            _ if name.starts_with('-') => return Err(format!("unknown argument {name}")),
            _ if catalog.is_none() => catalog = Some(PathBuf::from(name)),
            _ if metrics.is_none() => metrics = Some(PathBuf::from(name)),
            _ => return Err("only catalog and frame-metrics CSV inputs are accepted".into()),
        }
        index += 1;
    }
    let catalog = catalog.ok_or("a source catalog CSV is required")?;
    let metrics = metrics.ok_or("a frame-metrics CSV is required")?;
    let result = run(
        &catalog,
        &metrics,
        expected_frames,
        (!classes.is_empty()).then_some(classes.as_slice()),
        &class_source,
        (!ids.is_empty()).then_some(ids.as_slice()),
    )?;
    let output_dir = write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool":"rst19-feature-temporal-consistency-audit",
        "out_dir":output_dir,
        "result_path":output_dir.join("feature_temporal_consistency_audit.json"),
        "result":result.as_dict()
    }))
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    if args
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!(
            "rst19-feature-temporal-consistency-audit CATALOG.csv FRAME_METRICS.csv [options]"
        );
        return Ok(());
    }
    let result = run_desktop(args)?;
    println!(
        "rst19-feature-temporal-consistency-audit: 结果已写入 {}",
        result["out_dir"].as_str().unwrap_or_default()
    );
    let summaries = result["result"]["class_summaries"].as_array().unwrap();
    println!(
        "类别摘要：{}",
        summaries
            .iter()
            .map(|row| format!(
                "{}={}",
                row["feature_class"].as_str().unwrap_or_default(),
                row["source_count"]
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

pub fn handle_request(request: &Value) -> Result<Value, String> {
    let catalog = request["catalog_path"]
        .as_str()
        .ok_or("missing catalog_path")?;
    let metrics = request["frame_metrics_path"]
        .as_str()
        .ok_or("missing frame_metrics_path")?;
    let expected_frames = request
        .get("expected_frame_count")
        .and_then(Value::as_u64)
        .unwrap_or(15) as usize;
    let classes = request
        .get("feature_classes")
        .filter(|value| !value.is_null())
        .cloned()
        .map(serde_json::from_value::<Vec<String>>)
        .transpose()
        .map_err(|error| error.to_string())?;
    let class_source = request
        .get("feature_class_source")
        .and_then(Value::as_str)
        .unwrap_or("catalog");
    let ids = request
        .get("detection_ids")
        .filter(|value| !value.is_null())
        .cloned()
        .map(serde_json::from_value::<Vec<i64>>)
        .transpose()
        .map_err(|error| error.to_string())?;
    let result = run(
        Path::new(catalog),
        Path::new(metrics),
        expected_frames,
        classes.as_deref(),
        class_source,
        ids.as_deref(),
    )?;
    let output = if let Some(path) = request.get("output_dir").and_then(Value::as_str) {
        let output = write_artifacts(&result, Path::new(path))?;
        json!({"out_dir":output, "result_path":output.join("feature_temporal_consistency_audit.json")})
    } else {
        Value::Null
    };
    Ok(json!({"result":result.as_dict(), "output":output}))
}
