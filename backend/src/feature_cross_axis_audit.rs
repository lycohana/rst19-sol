//! Detector-level cross-axis audit joining the existing raw, peak, and temporal
//! summaries by detection ID. This is a descriptive evidence view, not a star
//! classifier or a physical-truth estimate.

use crate::Result;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};

const CATALOG_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
];
const RAW_FIELDS: &[&str] = &[
    "detection_id",
    "core_fraction_median",
    "noise_median",
    "frames_support_ge3",
    "frames_flux_snr_ge5",
    "frames_negative_extreme_count_gt0",
    "frames_repeated_code_count_gt0",
];
const PEAK_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "raw_valid",
    "raw_local_maximum_r1",
    "raw_local_max_gap_adu_r1",
];
const TEMPORAL_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "expected_frame_count",
    "frame_count",
    "coverage_fraction",
    "flux_snr_relative_mad",
    "flux_snr_positive_fraction",
    "flux_snr_ge5_fraction",
    "flux_snr_sign_flip_count",
];

#[derive(Debug, Clone)]
struct CsvTable {
    path: PathBuf,
    rows: Vec<BTreeMap<String, String>>,
    columns: Vec<String>,
    sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureCrossAxisSource {
    pub detection_id: i64,
    pub feature_class: String,
    pub feature_class_label: String,
    pub quality_passed: bool,
    pub raw_valid: bool,
    pub raw_local_maximum_r1: bool,
    pub raw_local_max_gap_adu_r1: f64,
    pub core_fraction_median: f64,
    pub local_noise_median: f64,
    pub support_frames_ge3: f64,
    pub flux_snr_frames_ge5: f64,
    pub negative_extreme_frames: f64,
    pub repeated_code_frames: f64,
    pub expected_frame_count: i64,
    pub temporal_frame_count: i64,
    pub temporal_coverage_fraction: f64,
    pub temporal_flux_snr_relative_mad: f64,
    pub temporal_positive_fraction: f64,
    pub temporal_flux_snr_ge5_fraction: f64,
    pub temporal_sign_flip_count: i64,
    pub raw_peak_status: String,
    pub value_domain_status: String,
    pub support_status: String,
    pub evidence_pattern: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureCrossAxisClassSummary {
    pub feature_class: String,
    pub feature_class_label: String,
    pub source_count: usize,
    pub quality_passed_fraction: f64,
    pub raw_valid_fraction: f64,
    pub raw_local_maximum_fraction: f64,
    pub raw_non_local_maximum_fraction: f64,
    pub core_fraction_median: f64,
    pub local_noise_median: f64,
    pub support_frames_ge3_median: f64,
    pub flux_snr_frames_ge5_median: f64,
    pub value_domain_anomaly_fraction: f64,
    pub temporal_coverage_median: f64,
    pub temporal_flux_snr_relative_mad_median: f64,
    pub temporal_positive_fraction_median: f64,
    pub temporal_flux_snr_ge5_fraction_median: f64,
    pub temporal_any_sign_flip_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureCrossAxisPatternSummary {
    pub feature_class: String,
    pub feature_class_label: String,
    pub evidence_pattern: String,
    pub source_count: usize,
    pub source_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureCrossAxisInputProvenance {
    pub name: String,
    pub path: String,
    pub sha256: String,
    pub row_count: usize,
    pub columns: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FeatureCrossAxisAuditResult {
    pub catalog_path: String,
    pub raw_summary_path: String,
    pub peak_consistency_path: String,
    pub temporal_sources_path: String,
    pub selected_detection_ids: Vec<i64>,
    pub feature_classes: Vec<String>,
    pub source_rows: Vec<FeatureCrossAxisSource>,
    pub class_summaries: Vec<FeatureCrossAxisClassSummary>,
    pub pattern_summaries: Vec<FeatureCrossAxisPatternSummary>,
    pub input_provenance: Vec<FeatureCrossAxisInputProvenance>,
}

impl FeatureCrossAxisAuditResult {
    pub fn as_dict(&self) -> Value {
        let provenance: BTreeMap<_, _> = self
            .input_provenance
            .iter()
            .map(|row| (row.name.clone(), row))
            .collect();
        json!({
            "catalog_path": self.catalog_path,
            "raw_summary_path": self.raw_summary_path,
            "peak_consistency_path": self.peak_consistency_path,
            "temporal_sources_path": self.temporal_sources_path,
            "selected_detection_ids": self.selected_detection_ids,
            "feature_classes": self.feature_classes,
            "source_rows": self.source_rows,
            "class_summaries": self.class_summaries,
            "pattern_summaries": self.pattern_summaries,
            "input_provenance": provenance,
            "interpretation_guardrails": [
                "feature_class 是检测器优先级旗标，不是物理恒星类别。",
                "raw 局部峰、二维支持、值域计数和逐帧响应不是彼此独立的真值标签。",
                "evidence_pattern 只描述证据并置，不是分类器、恒星概率或伪影率。",
                "跨表连接要求抽样 detection_id 集合一致，避免把不同实验产物静默拼接。",
                "input_provenance 记录输入表的精确字节 SHA-256、行数和列顺序，便于复核数据血缘。",
            ],
        })
    }
}

fn python_repr(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
        .replace('\'', "\\'");
    format!("'{escaped}'")
}

fn py_float(value: f64) -> String {
    let mut rendered = format!("{value:?}");
    if let Some((mantissa, exponent)) = rendered.split_once('e') {
        let exponent = exponent.parse::<i32>().unwrap_or(0);
        rendered = format!(
            "{mantissa}e{}{:02}",
            if exponent >= 0 { "+" } else { "-" },
            exponent.abs()
        );
    }
    rendered
}

fn py_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

fn read_csv(path: &Path, name: &str) -> Result<CsvTable> {
    let bytes = fs::read(path).map_err(|_| format!("无法读取{name}：{}", path.display()))?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(bytes.as_slice());
    let raw_headers = reader.headers().map_err(|error| error.to_string())?.clone();
    if raw_headers.is_empty() {
        return Err(format!("{name} has no header: {}", path.display()));
    }
    let columns: Vec<String> = raw_headers
        .iter()
        .enumerate()
        .map(|(index, header)| {
            if index == 0 {
                header.trim_start_matches('\u{feff}').to_owned()
            } else {
                header.to_owned()
            }
        })
        .collect();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|error| error.to_string())?;
        let mut row = BTreeMap::new();
        for (index, column) in columns.iter().enumerate() {
            row.insert(
                column.clone(),
                record.get(index).unwrap_or_default().to_owned(),
            );
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{name} is empty: {}", path.display()));
    }
    Ok(CsvTable {
        path: path.to_path_buf(),
        rows,
        columns,
        sha256: digest,
    })
}

fn required_columns(table: &CsvTable, fields: &[&str], name: &str) -> Result<()> {
    let columns: HashSet<_> = table.rows[0].keys().map(String::as_str).collect();
    let missing: Vec<_> = fields
        .iter()
        .filter(|field| !columns.contains(**field))
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{name} is missing required columns: {}",
            missing.join(", ")
        ));
    }
    Ok(())
}

fn required_id(row: &BTreeMap<String, String>, field: &str, name: &str) -> Result<i64> {
    let raw = row
        .get(field)
        .map(String::as_str)
        .unwrap_or_default()
        .trim();
    if raw.is_empty() {
        return Err(format!("{name} has empty {field}"));
    }
    let parsed = raw
        .parse::<f64>()
        .map_err(|_| format!("{name} has invalid {field}={}", python_repr(raw)))?;
    if !parsed.is_finite() || parsed < i64::MIN as f64 || parsed > i64::MAX as f64 {
        return Err(format!("{name} has invalid {field}={}", python_repr(raw)));
    }
    let value = parsed.trunc() as i64;
    if value < 0 {
        return Err(format!("{name} has negative {field}={value}"));
    }
    Ok(value)
}

fn required_float(row: &BTreeMap<String, String>, field: &str, name: &str) -> Result<f64> {
    let raw = row
        .get(field)
        .map(String::as_str)
        .unwrap_or_default()
        .trim();
    if raw.is_empty() {
        return Err(format!("{name} has empty {field}"));
    }
    let value = raw
        .parse::<f64>()
        .map_err(|_| format!("{name} has invalid {field}={}", python_repr(raw)))?;
    if !value.is_finite() {
        return Err(format!(
            "{name} has non-finite {field}={}",
            python_repr(raw)
        ));
    }
    Ok(value)
}

fn required_bool(row: &BTreeMap<String, String>, field: &str, name: &str) -> Result<bool> {
    let raw = row
        .get(field)
        .map(String::as_str)
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    match raw.as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(format!(
            "{name} has invalid boolean {field}={}",
            python_repr(&raw)
        )),
    }
}

fn index_unique(table: &CsvTable, name: &str) -> Result<BTreeMap<i64, BTreeMap<String, String>>> {
    let mut indexed = BTreeMap::new();
    for row in &table.rows {
        let detection_id = required_id(row, "detection_id", name)?;
        if indexed.contains_key(&detection_id) {
            return Err(format!(
                "{name} contains duplicate detection_id={detection_id}"
            ));
        }
        indexed.insert(detection_id, row.clone());
    }
    Ok(indexed)
}

fn check_same_ids(reference: &BTreeSet<i64>, candidate: &BTreeSet<i64>, name: &str) -> Result<()> {
    let missing: Vec<_> = reference.difference(candidate).take(3).copied().collect();
    let extra: Vec<_> = candidate.difference(reference).take(3).copied().collect();
    if !missing.is_empty() || !extra.is_empty() {
        let mut details = Vec::new();
        if !missing.is_empty() {
            details.push(format!("missing={missing:?}"));
        }
        if !extra.is_empty() {
            details.push(format!("extra={extra:?}"));
        }
        return Err(format!(
            "{name} detection_id set does not match raw sample ({})",
            details.join("; ")
        ));
    }
    Ok(())
}

fn pattern(
    raw_peak: bool,
    value_anomaly: bool,
    support_seen: bool,
) -> (String, String, String, String) {
    let raw_status = if raw_peak {
        "raw_local_peak"
    } else {
        "raw_non_local_peak"
    };
    let domain_status = if value_anomaly {
        "value_domain_anomaly"
    } else {
        "value_domain_clean"
    };
    let support_status = if support_seen {
        "support_seen"
    } else {
        "no_support_seen"
    };
    (
        raw_status.to_owned(),
        domain_status.to_owned(),
        support_status.to_owned(),
        format!("{raw_status}+{domain_status}+{support_status}"),
    )
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn make_class_summary(
    feature_class: String,
    label: String,
    rows: &[FeatureCrossAxisSource],
) -> FeatureCrossAxisClassSummary {
    let count = rows.len();
    let fraction = |selected: usize| selected as f64 / count as f64;
    let mut core: Vec<_> = rows.iter().map(|row| row.core_fraction_median).collect();
    let mut noise: Vec<_> = rows.iter().map(|row| row.local_noise_median).collect();
    let mut support: Vec<_> = rows.iter().map(|row| row.support_frames_ge3).collect();
    let mut flux_support: Vec<_> = rows.iter().map(|row| row.flux_snr_frames_ge5).collect();
    let mut coverage: Vec<_> = rows
        .iter()
        .map(|row| row.temporal_coverage_fraction)
        .collect();
    let mut relative_mad: Vec<_> = rows
        .iter()
        .map(|row| row.temporal_flux_snr_relative_mad)
        .collect();
    let mut positive: Vec<_> = rows
        .iter()
        .map(|row| row.temporal_positive_fraction)
        .collect();
    let mut strong: Vec<_> = rows
        .iter()
        .map(|row| row.temporal_flux_snr_ge5_fraction)
        .collect();
    FeatureCrossAxisClassSummary {
        feature_class,
        feature_class_label: label,
        source_count: count,
        quality_passed_fraction: fraction(rows.iter().filter(|row| row.quality_passed).count()),
        raw_valid_fraction: fraction(rows.iter().filter(|row| row.raw_valid).count()),
        raw_local_maximum_fraction: fraction(
            rows.iter().filter(|row| row.raw_local_maximum_r1).count(),
        ),
        raw_non_local_maximum_fraction: fraction(
            rows.iter().filter(|row| !row.raw_local_maximum_r1).count(),
        ),
        core_fraction_median: median(&mut core),
        local_noise_median: median(&mut noise),
        support_frames_ge3_median: median(&mut support),
        flux_snr_frames_ge5_median: median(&mut flux_support),
        value_domain_anomaly_fraction: fraction(
            rows.iter()
                .filter(|row| row.value_domain_status == "value_domain_anomaly")
                .count(),
        ),
        temporal_coverage_median: median(&mut coverage),
        temporal_flux_snr_relative_mad_median: median(&mut relative_mad),
        temporal_positive_fraction_median: median(&mut positive),
        temporal_flux_snr_ge5_fraction_median: median(&mut strong),
        temporal_any_sign_flip_fraction: fraction(
            rows.iter()
                .filter(|row| row.temporal_sign_flip_count > 0)
                .count(),
        ),
    }
}

fn provenance(name: &str, table: &CsvTable) -> FeatureCrossAxisInputProvenance {
    FeatureCrossAxisInputProvenance {
        name: name.to_owned(),
        path: table.path.to_string_lossy().into_owned(),
        sha256: table.sha256.clone(),
        row_count: table.rows.len(),
        columns: table.columns.clone(),
    }
}

pub fn run(
    catalog_path: &Path,
    raw_summary_path: &Path,
    peak_consistency_path: &Path,
    temporal_sources_path: &Path,
    feature_classes: Option<&[String]>,
    detection_ids: Option<&[i64]>,
) -> Result<FeatureCrossAxisAuditResult> {
    let catalog_table = read_csv(catalog_path, "source catalog")?;
    let raw_table = read_csv(raw_summary_path, "raw source summary")?;
    let peak_table = read_csv(peak_consistency_path, "peak consistency table")?;
    let temporal_table = read_csv(temporal_sources_path, "temporal source table")?;
    required_columns(&catalog_table, CATALOG_FIELDS, "source catalog")?;
    required_columns(&raw_table, RAW_FIELDS, "raw source summary")?;
    required_columns(&peak_table, PEAK_FIELDS, "peak consistency table")?;
    required_columns(&temporal_table, TEMPORAL_FIELDS, "temporal source table")?;
    let catalog = index_unique(&catalog_table, "source catalog")?;
    let raw = index_unique(&raw_table, "raw source summary")?;
    let peak = index_unique(&peak_table, "peak consistency table")?;
    let temporal = index_unique(&temporal_table, "temporal source table")?;
    let sample_ids: BTreeSet<_> = raw.keys().copied().collect();
    check_same_ids(
        &sample_ids,
        &temporal.keys().copied().collect(),
        "temporal source table",
    )?;
    if let Some(missing) = sample_ids.iter().find(|id| !catalog.contains_key(id)) {
        return Err(format!(
            "source catalog is missing raw sample detection_id={missing}"
        ));
    }
    if let Some(missing) = sample_ids.iter().find(|id| !peak.contains_key(id)) {
        return Err(format!(
            "peak consistency table is missing raw sample detection_id={missing}"
        ));
    }

    let selected_ids: Vec<i64> = if let Some(ids) = detection_ids {
        let mut unique = Vec::new();
        for id in ids {
            if !unique.contains(id) {
                unique.push(*id);
            }
        }
        if unique.is_empty() {
            return Err("detection_ids 不能为空".into());
        }
        if unique.iter().any(|id| *id < 0) {
            return Err("detection_ids 必须为非负整数".into());
        }
        if let Some(missing) = unique.iter().find(|id| !sample_ids.contains(id)) {
            return Err(format!("raw source summary has no detection_id={missing}"));
        }
        unique
    } else {
        sample_ids.iter().copied().collect()
    };

    let selected_classes: BTreeSet<String> = selected_ids
        .iter()
        .map(|id| {
            catalog[id]
                .get("feature_class")
                .cloned()
                .unwrap_or_default()
                .trim()
                .to_owned()
        })
        .collect();
    let requested_classes: Vec<String> = if let Some(classes) = feature_classes {
        let mut unique = Vec::new();
        for value in classes
            .iter()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        {
            if !unique.iter().any(|item: &String| item == value) {
                unique.push(value.to_owned());
            }
        }
        unique
    } else {
        selected_classes.iter().cloned().collect()
    };
    if requested_classes.is_empty() {
        return Err("feature_classes 不能为空".into());
    }
    if let Some(missing) = requested_classes
        .iter()
        .find(|name| !selected_classes.contains(*name))
    {
        return Err(format!(
            "raw sample has no feature_class={}",
            python_repr(missing)
        ));
    }

    let mut source_rows = Vec::new();
    for detection_id in &selected_ids {
        let catalog_row = &catalog[detection_id];
        let feature_class = catalog_row
            .get("feature_class")
            .map(String::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned();
        if !requested_classes.contains(&feature_class) {
            continue;
        }
        let raw_row = &raw[detection_id];
        let peak_row = &peak[detection_id];
        let temporal_row = &temporal[detection_id];
        let temporal_class = temporal_row
            .get("feature_class")
            .map(String::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned();
        if !temporal_class.is_empty() && temporal_class != feature_class {
            return Err(format!(
                "feature_class mismatch for detection_id={detection_id}: catalog={}, temporal={}",
                python_repr(&feature_class),
                python_repr(&temporal_class)
            ));
        }
        let peak_class = peak_row
            .get("feature_class")
            .map(String::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned();
        if peak_class != feature_class {
            return Err(format!(
                "feature_class mismatch for detection_id={detection_id}: catalog={}, peak={}",
                python_repr(&feature_class),
                python_repr(&peak_class)
            ));
        }
        let catalog_label = catalog_row
            .get("feature_class_label")
            .map(String::as_str)
            .unwrap_or_default()
            .trim();
        let catalog_label = if catalog_label.is_empty() {
            &feature_class
        } else {
            catalog_label
        };
        let peak_label = peak_row
            .get("feature_class_label")
            .map(String::as_str)
            .unwrap_or_default()
            .trim();
        let peak_label = if peak_label.is_empty() {
            &peak_class
        } else {
            peak_label
        };
        if peak_label != catalog_label {
            return Err(format!(
                "feature_class_label mismatch for detection_id={detection_id}: catalog={}, peak={}",
                python_repr(catalog_label),
                python_repr(peak_label)
            ));
        }
        let catalog_quality = required_bool(catalog_row, "quality_passed", "source catalog")?;
        let peak_quality = required_bool(peak_row, "quality_passed", "peak consistency table")?;
        if peak_quality != catalog_quality {
            return Err(format!(
                "quality_passed mismatch for detection_id={detection_id}: catalog={}, peak={}",
                py_bool(catalog_quality),
                py_bool(peak_quality)
            ));
        }
        let expected_frames = required_float(
            temporal_row,
            "expected_frame_count",
            "temporal source table",
        )?
        .trunc() as i64;
        let frame_count =
            required_float(temporal_row, "frame_count", "temporal source table")?.trunc() as i64;
        if expected_frames < 1 || frame_count < 0 || frame_count > expected_frames {
            return Err(format!(
                "temporal frame counts are invalid for detection_id={detection_id}"
            ));
        }
        let negative_frames = required_float(
            raw_row,
            "frames_negative_extreme_count_gt0",
            "raw source summary",
        )?;
        let repeated_frames = required_float(
            raw_row,
            "frames_repeated_code_count_gt0",
            "raw source summary",
        )?;
        let support_frames = required_float(raw_row, "frames_support_ge3", "raw source summary")?;
        let raw_peak = required_bool(peak_row, "raw_local_maximum_r1", "peak consistency table")?;
        let (raw_status, domain_status, support_status, evidence_pattern) = pattern(
            raw_peak,
            negative_frames > 0.0 || repeated_frames > 0.0,
            support_frames > 0.0,
        );
        source_rows.push(FeatureCrossAxisSource {
            detection_id: *detection_id,
            feature_class: feature_class.clone(),
            feature_class_label: catalog_label.to_owned(),
            quality_passed: catalog_quality,
            raw_valid: required_bool(peak_row, "raw_valid", "peak consistency table")?,
            raw_local_maximum_r1: raw_peak,
            raw_local_max_gap_adu_r1: required_float(
                peak_row,
                "raw_local_max_gap_adu_r1",
                "peak consistency table",
            )?,
            core_fraction_median: required_float(
                raw_row,
                "core_fraction_median",
                "raw source summary",
            )?,
            local_noise_median: required_float(raw_row, "noise_median", "raw source summary")?,
            support_frames_ge3: support_frames,
            flux_snr_frames_ge5: required_float(
                raw_row,
                "frames_flux_snr_ge5",
                "raw source summary",
            )?,
            negative_extreme_frames: negative_frames,
            repeated_code_frames: repeated_frames,
            expected_frame_count: expected_frames,
            temporal_frame_count: frame_count,
            temporal_coverage_fraction: required_float(
                temporal_row,
                "coverage_fraction",
                "temporal source table",
            )?,
            temporal_flux_snr_relative_mad: required_float(
                temporal_row,
                "flux_snr_relative_mad",
                "temporal source table",
            )?,
            temporal_positive_fraction: required_float(
                temporal_row,
                "flux_snr_positive_fraction",
                "temporal source table",
            )?,
            temporal_flux_snr_ge5_fraction: required_float(
                temporal_row,
                "flux_snr_ge5_fraction",
                "temporal source table",
            )?,
            temporal_sign_flip_count: required_float(
                temporal_row,
                "flux_snr_sign_flip_count",
                "temporal source table",
            )?
            .trunc() as i64,
            raw_peak_status: raw_status,
            value_domain_status: domain_status,
            support_status,
            evidence_pattern,
        });
    }
    if source_rows.is_empty() {
        return Err("selected sources have no requested feature class".into());
    }

    let mut by_class: BTreeMap<String, Vec<FeatureCrossAxisSource>> = BTreeMap::new();
    let mut class_labels: BTreeMap<String, String> = BTreeMap::new();
    for row in &source_rows {
        by_class
            .entry(row.feature_class.clone())
            .or_default()
            .push(row.clone());
        class_labels
            .entry(row.feature_class.clone())
            .or_insert_with(|| row.feature_class_label.clone());
    }
    let class_summaries = requested_classes
        .iter()
        .filter_map(|name| {
            by_class
                .get(name)
                .map(|rows| make_class_summary(name.clone(), class_labels[name].clone(), rows))
        })
        .collect();
    let mut pattern_summaries = Vec::new();
    for feature_class in &requested_classes {
        if let Some(rows) = by_class.get(feature_class) {
            let mut counts: BTreeMap<String, usize> = BTreeMap::new();
            for row in rows {
                *counts.entry(row.evidence_pattern.clone()).or_default() += 1;
            }
            for (evidence_pattern, count) in counts {
                pattern_summaries.push(FeatureCrossAxisPatternSummary {
                    feature_class: feature_class.clone(),
                    feature_class_label: class_labels[feature_class].clone(),
                    evidence_pattern,
                    source_count: count,
                    source_fraction: count as f64 / rows.len() as f64,
                });
            }
        }
    }
    let provenance = vec![
        provenance("catalog", &catalog_table),
        provenance("raw_summary", &raw_table),
        provenance("peak_consistency", &peak_table),
        provenance("temporal_sources", &temporal_table),
    ];
    Ok(FeatureCrossAxisAuditResult {
        catalog_path: catalog_table.path.to_string_lossy().into_owned(),
        raw_summary_path: raw_table.path.to_string_lossy().into_owned(),
        peak_consistency_path: peak_table.path.to_string_lossy().into_owned(),
        temporal_sources_path: temporal_table.path.to_string_lossy().into_owned(),
        selected_detection_ids: source_rows.iter().map(|row| row.detection_id).collect(),
        feature_classes: requested_classes,
        source_rows,
        class_summaries,
        pattern_summaries,
        input_provenance: provenance,
    })
}

const SOURCE_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "raw_valid",
    "raw_local_maximum_r1",
    "raw_local_max_gap_adu_r1",
    "core_fraction_median",
    "local_noise_median",
    "support_frames_ge3",
    "flux_snr_frames_ge5",
    "negative_extreme_frames",
    "repeated_code_frames",
    "expected_frame_count",
    "temporal_frame_count",
    "temporal_coverage_fraction",
    "temporal_flux_snr_relative_mad",
    "temporal_positive_fraction",
    "temporal_flux_snr_ge5_fraction",
    "temporal_sign_flip_count",
    "raw_peak_status",
    "value_domain_status",
    "support_status",
    "evidence_pattern",
];
const CLASS_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "source_count",
    "quality_passed_fraction",
    "raw_valid_fraction",
    "raw_local_maximum_fraction",
    "raw_non_local_maximum_fraction",
    "core_fraction_median",
    "local_noise_median",
    "support_frames_ge3_median",
    "flux_snr_frames_ge5_median",
    "value_domain_anomaly_fraction",
    "temporal_coverage_median",
    "temporal_flux_snr_relative_mad_median",
    "temporal_positive_fraction_median",
    "temporal_flux_snr_ge5_fraction_median",
    "temporal_any_sign_flip_fraction",
];
const PATTERN_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "evidence_pattern",
    "source_count",
    "source_fraction",
];

fn records_sources(rows: &[FeatureCrossAxisSource]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|row| {
            vec![
                row.detection_id.to_string(),
                row.feature_class.clone(),
                row.feature_class_label.clone(),
                py_bool(row.quality_passed).to_owned(),
                py_bool(row.raw_valid).to_owned(),
                py_bool(row.raw_local_maximum_r1).to_owned(),
                py_float(row.raw_local_max_gap_adu_r1),
                py_float(row.core_fraction_median),
                py_float(row.local_noise_median),
                py_float(row.support_frames_ge3),
                py_float(row.flux_snr_frames_ge5),
                py_float(row.negative_extreme_frames),
                py_float(row.repeated_code_frames),
                row.expected_frame_count.to_string(),
                row.temporal_frame_count.to_string(),
                py_float(row.temporal_coverage_fraction),
                py_float(row.temporal_flux_snr_relative_mad),
                py_float(row.temporal_positive_fraction),
                py_float(row.temporal_flux_snr_ge5_fraction),
                row.temporal_sign_flip_count.to_string(),
                row.raw_peak_status.clone(),
                row.value_domain_status.clone(),
                row.support_status.clone(),
                row.evidence_pattern.clone(),
            ]
        })
        .collect()
}

fn records_classes(rows: &[FeatureCrossAxisClassSummary]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|row| {
            vec![
                row.feature_class.clone(),
                row.feature_class_label.clone(),
                row.source_count.to_string(),
                py_float(row.quality_passed_fraction),
                py_float(row.raw_valid_fraction),
                py_float(row.raw_local_maximum_fraction),
                py_float(row.raw_non_local_maximum_fraction),
                py_float(row.core_fraction_median),
                py_float(row.local_noise_median),
                py_float(row.support_frames_ge3_median),
                py_float(row.flux_snr_frames_ge5_median),
                py_float(row.value_domain_anomaly_fraction),
                py_float(row.temporal_coverage_median),
                py_float(row.temporal_flux_snr_relative_mad_median),
                py_float(row.temporal_positive_fraction_median),
                py_float(row.temporal_flux_snr_ge5_fraction_median),
                py_float(row.temporal_any_sign_flip_fraction),
            ]
        })
        .collect()
}

fn records_patterns(rows: &[FeatureCrossAxisPatternSummary]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|row| {
            vec![
                row.feature_class.clone(),
                row.feature_class_label.clone(),
                row.evidence_pattern.clone(),
                row.source_count.to_string(),
                py_float(row.source_fraction),
            ]
        })
        .collect()
}

fn write_csv(path: &Path, headers: &[&str], rows: Vec<Vec<String>>) -> Result<()> {
    let mut file = fs::File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    if rows.is_empty() {
        writer
            .write_record([""])
            .map_err(|error| error.to_string())?;
    } else {
        writer
            .write_record(headers)
            .map_err(|error| error.to_string())?;
        for row in rows {
            writer
                .write_record(row)
                .map_err(|error| error.to_string())?;
        }
    }
    writer.flush().map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &FeatureCrossAxisAuditResult, output: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    fs::write(
        output.join("feature_cross_axis_audit.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    write_csv(
        &output.join("feature_cross_axis_sources.csv"),
        SOURCE_FIELDS,
        records_sources(&result.source_rows),
    )?;
    write_csv(
        &output.join("feature_cross_axis_summary.csv"),
        CLASS_FIELDS,
        records_classes(&result.class_summaries),
    )?;
    write_csv(
        &output.join("feature_cross_axis_patterns.csv"),
        PATTERN_FIELDS,
        records_patterns(&result.pattern_summaries),
    )?;
    Ok(output.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut positional = Vec::new();
    let mut feature_classes = Vec::new();
    let mut detection_ids = Vec::new();
    let mut output = PathBuf::from("tmp/feature-cross-axis-audit");
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline) = arg
            .split_once('=')
            .map(|(n, v)| (n, Some(v)))
            .unwrap_or((arg.as_str(), None));
        match name {
            "--feature-class" | "--detection-id" | "--out-dir" => {
                let value = if let Some(value) = inline {
                    value
                } else {
                    index += 1;
                    args.get(index).ok_or("option requires a value")?
                };
                match name {
                    "--feature-class" => feature_classes.push(value.to_owned()),
                    "--detection-id" => detection_ids.push(
                        value
                            .parse::<i64>()
                            .map_err(|error| format!("invalid detection_id: {error}"))?,
                    ),
                    _ => output = PathBuf::from(value),
                }
            }
            _ if name.starts_with('-') => return Err(format!("unknown argument {name}")),
            _ => positional.push(PathBuf::from(name)),
        }
        index += 1;
    }
    if positional.len() != 4 {
        return Err(
            "expected catalog, raw summary, peak consistency, and temporal source paths".into(),
        );
    }
    let result = run(
        &positional[0],
        &positional[1],
        &positional[2],
        &positional[3],
        (!feature_classes.is_empty()).then_some(feature_classes.as_slice()),
        (!detection_ids.is_empty()).then_some(detection_ids.as_slice()),
    )?;
    write_artifacts(&result, &output)?;
    Ok(
        json!({"tool":"rst19-feature-cross-axis-audit", "out_dir":output,
        "result_path":output.join("feature_cross_axis_audit.json"), "result":result.as_dict()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn aligns_inputs_patterns_provenance_and_artifact_fields() {
        let root = std::env::temp_dir().join(format!(
            "rst19-cross-axis-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        let raw = root.join("raw.csv");
        let peak = root.join("peak.csv");
        let temporal = root.join("temporal.csv");
        fs::write(&catalog, "\u{feff}detection_id,feature_class,feature_class_label,quality_passed\r\n1,compact_quality,紧凑,True\r\n2,crowded_blend,拥挤,False\r\n").unwrap();
        fs::write(&raw, "detection_id,core_fraction_median,noise_median,frames_support_ge3,frames_flux_snr_ge5,frames_negative_extreme_count_gt0,frames_repeated_code_count_gt0\n1,0.7,8,15,15,0,0\n2,0.2,14,0,2,1,1\n").unwrap();
        fs::write(&peak, "detection_id,feature_class,feature_class_label,quality_passed,raw_valid,raw_local_maximum_r1,raw_local_max_gap_adu_r1\n1,compact_quality,紧凑,True,True,True,0\n2,crowded_blend,拥挤,False,True,False,100\n").unwrap();
        fs::write(&temporal, "detection_id,feature_class,expected_frame_count,frame_count,coverage_fraction,flux_snr_relative_mad,flux_snr_positive_fraction,flux_snr_ge5_fraction,flux_snr_sign_flip_count\n1,compact_quality,15,15,1,0.1,1,1,0\n2,crowded_blend,15,10,0.6667,0.3,0.8,0.1,2\n").unwrap();
        let result = run(&catalog, &raw, &peak, &temporal, None, None).unwrap();
        assert_eq!(result.selected_detection_ids, [1, 2]);
        assert_eq!(
            result.source_rows[0].evidence_pattern,
            "raw_local_peak+value_domain_clean+support_seen"
        );
        assert_eq!(
            result.source_rows[1].evidence_pattern,
            "raw_non_local_peak+value_domain_anomaly+no_support_seen"
        );
        assert_eq!(
            result.class_summaries[1].temporal_any_sign_flip_fraction,
            1.0
        );
        let out = root.join("out");
        write_artifacts(&result, &out).unwrap();
        assert!(out.join("feature_cross_axis_sources.csv").is_file());
        assert!(out.join("feature_cross_axis_summary.csv").is_file());
        assert!(out.join("feature_cross_axis_patterns.csv").is_file());
        let _ = fs::remove_dir_all(root);
    }
}
