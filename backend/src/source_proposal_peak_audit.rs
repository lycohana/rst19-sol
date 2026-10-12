//! Join proposal-method provenance with raw-pixel peak-consistency evidence.
//!
//! Proposal methods are correlated detector proposal sources, not independent
//! votes. This audit only summarizes detector-level records and does not infer
//! physical source identity or catalog quality.

use crate::Result;
use csv::{Terminator, WriterBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub const METHOD_GROUP_NAMES: [&str; 5] = [
    "all_three",
    "gaussian_only",
    "dog_only",
    "partial",
    "no_method",
];

const CATALOG_REQUIRED: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "flags",
    "proposal_methods",
    "flux_snr",
    "filter_snr",
];
const PEAK_REQUIRED: &[&str] = &[
    "detection_id",
    "raw_valid",
    "raw_local_maximum_r1",
    "raw_unique_local_maximum_r1",
    "raw_local_max_gap_adu_r1",
];
const ROW_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "method_group",
    "method_count",
    "proposal_methods",
    "quality_passed",
    "flags",
    "flux_snr",
    "filter_snr",
    "raw_valid",
    "raw_local_maximum_r1",
    "raw_unique_local_maximum_r1",
    "raw_local_max_gap_adu_r1",
];
const SUMMARY_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "method_group",
    "candidate_count",
    "class_count",
    "class_fraction",
    "quality_count",
    "quality_fraction",
    "raw_valid_count",
    "raw_local_maximum_count_r1",
    "raw_local_maximum_fraction_r1",
    "raw_unique_local_maximum_count_r1",
    "nonlocal_peak_count_r1",
    "nonlocal_peak_fraction_r1",
    "nonlocal_peak_gap_median_adu_r1",
    "nonlocal_peak_gap_p90_adu_r1",
    "median_flux_snr",
    "median_filter_snr",
];
const TARGET_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "method_group",
    "proposal_methods",
    "quality_passed",
    "flags",
    "flux_snr",
    "filter_snr",
    "raw_local_maximum_r1",
    "raw_unique_local_maximum_r1",
    "raw_local_max_gap_adu_r1",
    "method_group_count",
    "method_group_share_in_class",
];
const INTERPRETATION_BOUNDARY: &str = "proposal_methods 是相关提议器来源记录，不是独立投票；raw 局部峰是像素拓扑诊断，不是恒星充分条件；本审计不推导 precision、FDR、恒星概率、完备率或物理源数";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SourceProposalPeakRow {
    pub detection_id: i64,
    pub feature_class: String,
    pub feature_class_label: String,
    pub method_group: String,
    pub method_count: usize,
    pub proposal_methods: String,
    pub quality_passed: bool,
    pub flags: String,
    pub flux_snr: Option<f64>,
    pub filter_snr: Option<f64>,
    pub raw_valid: bool,
    pub raw_local_maximum_r1: bool,
    pub raw_unique_local_maximum_r1: bool,
    pub raw_local_max_gap_adu_r1: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SourceProposalPeakSummary {
    pub feature_class: String,
    pub feature_class_label: String,
    pub method_group: String,
    pub candidate_count: usize,
    pub class_count: usize,
    pub class_fraction: f64,
    pub quality_count: usize,
    pub quality_fraction: f64,
    pub raw_valid_count: usize,
    pub raw_local_maximum_count_r1: usize,
    pub raw_local_maximum_fraction_r1: f64,
    pub raw_unique_local_maximum_count_r1: usize,
    pub nonlocal_peak_count_r1: usize,
    pub nonlocal_peak_fraction_r1: f64,
    pub nonlocal_peak_gap_median_adu_r1: Option<f64>,
    pub nonlocal_peak_gap_p90_adu_r1: Option<f64>,
    pub median_flux_snr: Option<f64>,
    pub median_filter_snr: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SourceProposalPeakTarget {
    pub detection_id: i64,
    pub feature_class: String,
    pub feature_class_label: String,
    pub method_group: String,
    pub proposal_methods: String,
    pub quality_passed: bool,
    pub flags: String,
    pub flux_snr: Option<f64>,
    pub filter_snr: Option<f64>,
    pub raw_local_maximum_r1: bool,
    pub raw_unique_local_maximum_r1: bool,
    pub raw_local_max_gap_adu_r1: Option<f64>,
    pub method_group_count: usize,
    pub method_group_share_in_class: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SourceProposalPeakAuditResult {
    pub catalog_path: String,
    pub peak_consistency_path: String,
    pub rows: Vec<SourceProposalPeakRow>,
    pub summaries: Vec<SourceProposalPeakSummary>,
    pub targets: Vec<SourceProposalPeakTarget>,
    pub conclusion: String,
}

impl SourceProposalPeakAuditResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "peak_consistency_path": self.peak_consistency_path,
            "rows": self.rows,
            "summaries": self.summaries,
            "targets": self.targets,
            "conclusion": self.conclusion,
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        })
    }
}

type CsvRow = BTreeMap<String, Option<String>>;

#[derive(Clone, Debug)]
struct PeakRecord {
    raw_valid: bool,
    raw_local_maximum_r1: bool,
    raw_unique_local_maximum_r1: bool,
    raw_local_max_gap_adu_r1: Option<f64>,
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

fn resolved_path(path: &Path) -> Result<PathBuf> {
    let canonical = fs::canonicalize(path).map_err(|error| error.to_string())?;
    #[cfg(windows)]
    {
        let text = canonical.to_string_lossy();
        Ok(PathBuf::from(
            text.strip_prefix("\\\\?\\").unwrap_or(text.as_ref()),
        ))
    }
    #[cfg(not(windows))]
    {
        Ok(canonical)
    }
}

fn python_value(row: &CsvRow, field: &str) -> String {
    match row.get(field) {
        Some(Some(value)) => value.clone(),
        Some(None) => "None".to_owned(),
        None => String::new(),
    }
}

fn proposal_method_group(raw_methods: &str) -> String {
    let methods = raw_methods
        .split('|')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>();
    if ["gaussian", "dog_narrow", "dog_broad"]
        .iter()
        .all(|method| methods.contains(method))
    {
        "all_three".to_owned()
    } else if methods.len() == 1 && methods.contains("gaussian") {
        "gaussian_only".to_owned()
    } else if !methods.is_empty() && !methods.contains("gaussian") {
        "dog_only".to_owned()
    } else if !methods.is_empty() {
        "partial".to_owned()
    } else {
        "no_method".to_owned()
    }
}

fn finite_float(row: &CsvRow, field: &str) -> Option<f64> {
    let raw = python_value(row, field).trim().to_owned();
    if raw.is_empty() || matches!(raw.to_ascii_lowercase().as_str(), "nan" | "none" | "null") {
        return None;
    }
    raw.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn required_bool(row: &CsvRow, field: &str, row_number: usize) -> Result<bool> {
    let value = python_value(row, field).trim().to_ascii_lowercase();
    match value.as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(format!(
            "CSV row {row_number}: invalid {field}={}",
            python_repr(&value)
        )),
    }
}

fn read_table(path: &Path, kind: &str, required: &[&str]) -> Result<Vec<(i64, CsvRow)>> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let header_set = headers.iter().collect::<BTreeSet<_>>();
    let missing = required
        .iter()
        .filter(|field| !header_set.contains(**field))
        .copied()
        .collect::<BTreeSet<_>>();
    if !missing.is_empty() {
        let prefix = if kind == "source catalog" {
            "source catalog is missing required columns"
        } else {
            "peak consistency CSV is missing required columns"
        };
        return Err(format!(
            "{prefix}: {}",
            missing.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    let mut records = Vec::new();
    let mut seen_ids = BTreeSet::new();
    for (offset, record) in reader.records().enumerate() {
        let record = record.map_err(|error| error.to_string())?;
        let row_number = offset + 2;
        let mut row = CsvRow::new();
        for (index, header) in headers.iter().enumerate() {
            // DictReader preserves the last duplicate header and distinguishes
            // a short row's missing value from an explicit empty field.
            row.insert(header.to_owned(), record.get(index).map(str::to_owned));
        }
        row.insert("__row_number".to_owned(), Some(row_number.to_string()));
        let raw_id = python_value(&row, "detection_id").trim().to_owned();
        let detection_id = raw_id.parse::<i64>().map_err(|_| {
            format!(
                "{} row {row_number}: invalid detection_id",
                if kind == "source catalog" {
                    "catalog"
                } else {
                    "peak"
                }
            )
        })?;
        if !seen_ids.insert(detection_id) {
            let message = if kind == "source catalog" {
                format!("source catalog contains duplicate detection_id={detection_id}")
            } else {
                format!("peak consistency CSV contains duplicate detection_id={detection_id}")
            };
            return Err(message);
        }
        records.push((detection_id, row));
    }
    if records.is_empty() {
        let message = if kind == "source catalog" {
            format!("source catalog is empty: {}", path.display())
        } else {
            format!("peak consistency CSV is empty: {}", path.display())
        };
        return Err(message);
    }
    Ok(records)
}

fn parse_catalog(path: &Path) -> Result<BTreeMap<i64, SourceProposalPeakRow>> {
    let records = read_table(path, "source catalog", CATALOG_REQUIRED)?;
    let mut output = BTreeMap::new();
    for (detection_id, row) in records {
        let row_number = row
            .get("__row_number")
            .and_then(Option::as_deref)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2);
        let feature_class = python_value(&row, "feature_class").trim().to_owned();
        if feature_class.is_empty() {
            return Err(format!("catalog row {row_number}: empty feature_class"));
        }
        let methods = python_value(&row, "proposal_methods").trim().to_owned();
        // Python's method_count removes empty strings but deliberately does not
        // strip individual tokens; proposal_method_group does strip them.
        let method_count = methods
            .split('|')
            .filter(|item| !item.is_empty())
            .collect::<BTreeSet<_>>()
            .len();
        output.insert(
            detection_id,
            SourceProposalPeakRow {
                detection_id,
                feature_class,
                feature_class_label: python_value(&row, "feature_class_label").trim().to_owned(),
                method_group: proposal_method_group(&methods),
                method_count,
                proposal_methods: methods,
                quality_passed: required_bool(&row, "quality_passed", row_number)?,
                flags: python_value(&row, "flags").trim().to_owned(),
                flux_snr: finite_float(&row, "flux_snr"),
                filter_snr: finite_float(&row, "filter_snr"),
                raw_valid: false,
                raw_local_maximum_r1: false,
                raw_unique_local_maximum_r1: false,
                raw_local_max_gap_adu_r1: None,
            },
        );
    }
    Ok(output)
}

fn parse_peak(path: &Path) -> Result<BTreeMap<i64, PeakRecord>> {
    let records = read_table(path, "peak consistency CSV", PEAK_REQUIRED)?;
    let mut output = BTreeMap::new();
    for (detection_id, row) in records {
        // Row numbering comes from original CSV order, not the sorted ID map.
        let row_number = row
            .get("__row_number")
            .and_then(Option::as_deref)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2);
        output.insert(
            detection_id,
            PeakRecord {
                raw_valid: required_bool(&row, "raw_valid", row_number)?,
                raw_local_maximum_r1: required_bool(&row, "raw_local_maximum_r1", row_number)?,
                raw_unique_local_maximum_r1: required_bool(
                    &row,
                    "raw_unique_local_maximum_r1",
                    row_number,
                )?,
                raw_local_max_gap_adu_r1: finite_float(&row, "raw_local_max_gap_adu_r1"),
            },
        );
    }
    Ok(output)
}

fn median(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut values = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn quantile90(values: &[f64]) -> Option<f64> {
    let mut values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let position = (values.len() - 1) as f64 * 0.9;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    let left = values[lower];
    let right = values[upper];
    Some(if fraction >= 0.5 {
        right - (right - left) * (1.0 - fraction)
    } else {
        left + (right - left) * fraction
    })
}

fn joined_rows(
    mut catalog: BTreeMap<i64, SourceProposalPeakRow>,
    peak: BTreeMap<i64, PeakRecord>,
) -> Result<Vec<SourceProposalPeakRow>> {
    let catalog_ids = catalog.keys().copied().collect::<BTreeSet<_>>();
    let peak_ids = peak.keys().copied().collect::<BTreeSet<_>>();
    if catalog_ids != peak_ids {
        let catalog_only = catalog_ids
            .difference(&peak_ids)
            .take(5)
            .map(i64::to_string)
            .collect::<Vec<_>>();
        let peak_only = peak_ids
            .difference(&catalog_ids)
            .take(5)
            .map(i64::to_string)
            .collect::<Vec<_>>();
        return Err(format!(
            "catalog and peak consistency IDs do not match: catalog_only=[{}], peak_only=[{}]",
            catalog_only.join(", "),
            peak_only.join(", ")
        ));
    }
    for (id, record) in peak {
        if let Some(row) = catalog.get_mut(&id) {
            row.raw_valid = record.raw_valid;
            row.raw_local_maximum_r1 = record.raw_local_maximum_r1;
            row.raw_unique_local_maximum_r1 = record.raw_unique_local_maximum_r1;
            row.raw_local_max_gap_adu_r1 = record.raw_local_max_gap_adu_r1;
        }
    }
    Ok(catalog.into_values().collect())
}

fn summarize(rows: &[SourceProposalPeakRow]) -> Vec<SourceProposalPeakSummary> {
    let mut by_class: BTreeMap<&str, Vec<&SourceProposalPeakRow>> = BTreeMap::new();
    let mut by_group: BTreeMap<(&str, &str), Vec<&SourceProposalPeakRow>> = BTreeMap::new();
    let mut labels: BTreeMap<&str, &str> = BTreeMap::new();
    for row in rows {
        by_class.entry(&row.feature_class).or_default().push(row);
        by_group
            .entry((&row.feature_class, &row.method_group))
            .or_default()
            .push(row);
        labels.insert(&row.feature_class, &row.feature_class_label);
    }
    by_group
        .into_iter()
        .map(|((feature_class, method_group), members)| {
            let class_count = by_class[feature_class].len();
            let candidate_count = members.len();
            let quality_count = members.iter().filter(|row| row.quality_passed).count();
            let raw_valid_count = members.iter().filter(|row| row.raw_valid).count();
            let local_count = members
                .iter()
                .filter(|row| row.raw_local_maximum_r1)
                .count();
            let unique_count = members
                .iter()
                .filter(|row| row.raw_unique_local_maximum_r1)
                .count();
            let gaps = members
                .iter()
                .filter(|row| row.raw_valid && !row.raw_local_maximum_r1)
                .filter_map(|row| row.raw_local_max_gap_adu_r1)
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            SourceProposalPeakSummary {
                feature_class: feature_class.to_owned(),
                feature_class_label: labels[feature_class].to_owned(),
                method_group: method_group.to_owned(),
                candidate_count,
                class_count,
                class_fraction: candidate_count as f64 / class_count as f64,
                quality_count,
                quality_fraction: quality_count as f64 / candidate_count as f64,
                raw_valid_count,
                raw_local_maximum_count_r1: local_count,
                raw_local_maximum_fraction_r1: local_count as f64 / candidate_count as f64,
                raw_unique_local_maximum_count_r1: unique_count,
                nonlocal_peak_count_r1: gaps.len(),
                nonlocal_peak_fraction_r1: gaps.len() as f64 / candidate_count as f64,
                nonlocal_peak_gap_median_adu_r1: median(gaps.iter().copied()),
                nonlocal_peak_gap_p90_adu_r1: quantile90(&gaps),
                median_flux_snr: median(members.iter().filter_map(|row| row.flux_snr)),
                median_filter_snr: median(members.iter().filter_map(|row| row.filter_snr)),
            }
        })
        .collect()
}

fn targets(
    rows: &[SourceProposalPeakRow],
    target_ids: &[i64],
) -> Result<Vec<SourceProposalPeakTarget>> {
    let mut requested = Vec::new();
    for id in target_ids {
        if !requested.contains(id) {
            requested.push(*id);
        }
    }
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let by_id = rows
        .iter()
        .map(|row| (row.detection_id, row))
        .collect::<BTreeMap<_, _>>();
    let missing = requested
        .iter()
        .copied()
        .filter(|id| !by_id.contains_key(id))
        .collect::<BTreeSet<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "target detection_id not found in joined rows: {}",
            missing
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let mut class_group_counts = BTreeMap::new();
    let mut class_counts = BTreeMap::new();
    for row in rows {
        *class_group_counts
            .entry((row.feature_class.as_str(), row.method_group.as_str()))
            .or_insert(0usize) += 1;
        *class_counts
            .entry(row.feature_class.as_str())
            .or_insert(0usize) += 1;
    }
    Ok(requested
        .into_iter()
        .map(|id| {
            let row = by_id[&id];
            let method_group_count =
                class_group_counts[&(row.feature_class.as_str(), row.method_group.as_str())];
            SourceProposalPeakTarget {
                detection_id: row.detection_id,
                feature_class: row.feature_class.clone(),
                feature_class_label: row.feature_class_label.clone(),
                method_group: row.method_group.clone(),
                proposal_methods: row.proposal_methods.clone(),
                quality_passed: row.quality_passed,
                flags: row.flags.clone(),
                flux_snr: row.flux_snr,
                filter_snr: row.filter_snr,
                raw_local_maximum_r1: row.raw_local_maximum_r1,
                raw_unique_local_maximum_r1: row.raw_unique_local_maximum_r1,
                raw_local_max_gap_adu_r1: row.raw_local_max_gap_adu_r1,
                method_group_count,
                method_group_share_in_class: method_group_count as f64
                    / class_counts[row.feature_class.as_str()] as f64,
            }
        })
        .collect())
}

pub fn proposal_method_group_name(raw_methods: &str) -> String {
    proposal_method_group(raw_methods)
}

pub fn run(
    catalog_path: &Path,
    peak_consistency_path: &Path,
    target_ids: &[i64],
) -> Result<SourceProposalPeakAuditResult> {
    let catalog_path = resolved_path(catalog_path)?;
    let peak_consistency_path = resolved_path(peak_consistency_path)?;
    let catalog = parse_catalog(&catalog_path)?;
    let peak = parse_peak(&peak_consistency_path)?;
    let rows = joined_rows(catalog, peak)?;
    let summaries = summarize(&rows);
    let targets = targets(&rows, target_ids)?;
    let target_text = targets
        .iter()
        .map(|row| {
            format!(
                "ID {}={}, raw r1局部峰={}",
                row.detection_id,
                row.method_group,
                if row.raw_local_maximum_r1 {
                    "True"
                } else {
                    "False"
                }
            )
        })
        .collect::<Vec<_>>()
        .join("；");
    let mut conclusion = "提议器来源与 raw 峰交叉审计显示，同一 feature_class 内不同来源子组的质量和像素拓扑不能直接合并解释；来源组合只用于复核排序，不是独立投票。".to_owned();
    if !target_text.is_empty() {
        conclusion.push_str(&format!("目标结果：{target_text}。"));
    }
    Ok(SourceProposalPeakAuditResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        peak_consistency_path: peak_consistency_path.to_string_lossy().into_owned(),
        rows,
        summaries,
        targets,
        conclusion,
    })
}

fn python_float(value: f64) -> String {
    let raw = format!("{value:?}");
    if let Some((mantissa, exponent)) = raw.split_once('e') {
        let exponent = exponent.parse::<i32>().unwrap_or(0);
        format!("{mantissa}e{exponent:+03}")
    } else {
        raw
    }
}

fn csv_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(number) if number.is_i64() || number.is_u64() => number.to_string(),
        Value::Number(number) => number
            .as_f64()
            .map(python_float)
            .unwrap_or_else(|| number.to_string()),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn write_csv<T: Serialize>(path: &Path, headers: &[&str], rows: &[T]) -> Result<()> {
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
                .write_record(headers.iter().map(|field| csv_value(&value[*field])))
                .map_err(|error| error.to_string())?;
        }
        writer.flush().map_err(|error| error.to_string())?;
    }
    fs::write(path, bytes).map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &SourceProposalPeakAuditResult, output: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    write_csv(
        &output.join("source_proposal_peak_rows.csv"),
        ROW_FIELDS,
        &result.rows,
    )?;
    write_csv(
        &output.join("source_proposal_peak_summary.csv"),
        SUMMARY_FIELDS,
        &result.summaries,
    )?;
    write_csv(
        &output.join("source_proposal_peak_targets.csv"),
        TARGET_FIELDS,
        &result.targets,
    )?;
    fs::write(
        output.join("source_proposal_peak_audit.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut positional = Vec::<PathBuf>::new();
    let mut target_ids = Vec::new();
    let mut output = PathBuf::from("tmp/source-proposal-peak-audit");
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline) = arg
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((arg.as_str(), None));
        match name {
            "--target-id" | "--out-dir" => {
                let value = if let Some(value) = inline {
                    value
                } else {
                    index += 1;
                    args.get(index).ok_or("option requires a value")?
                };
                if name == "--target-id" {
                    target_ids.push(value.parse::<i64>().map_err(|error| error.to_string())?);
                } else {
                    output = PathBuf::from(value);
                }
            }
            "--" => {
                index += 1;
                positional.extend(args[index..].iter().map(PathBuf::from));
                break;
            }
            _ if name.starts_with('-') => return Err(format!("unknown argument {name}")),
            _ => positional.push(PathBuf::from(name)),
        }
        index += 1;
    }
    if positional.len() != 2 {
        return Err("expected source catalog and peak consistency CSV".into());
    }
    let result = run(&positional[0], &positional[1], &target_ids)?;
    write_artifacts(&result, &output)?;
    Ok(json!({
        "tool": "rst19-source-proposal-peak-audit",
        "out_dir": output,
        "result_path": output.join("source_proposal_peak_audit.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposal_groups_match_python_rules() {
        assert_eq!(
            proposal_method_group_name("gaussian|dog_narrow|dog_broad"),
            "all_three"
        );
        assert_eq!(proposal_method_group_name("gaussian"), "gaussian_only");
        assert_eq!(
            proposal_method_group_name("dog_narrow|dog_broad"),
            "dog_only"
        );
        assert_eq!(proposal_method_group_name("gaussian|dog_narrow"), "partial");
        assert_eq!(proposal_method_group_name(""), "no_method");
        assert_eq!(
            proposal_method_group_name(" gaussian |dog_narrow|dog_broad|other"),
            "all_three"
        );
    }

    #[test]
    fn linear_quantile_uses_numpy_interpolation() {
        assert_eq!(quantile90(&[0.0, 10.0]), Some(9.0));
        assert_eq!(median([1.0, 3.0]), Some(2.0));
        assert_eq!(median([]), None);
    }
}
