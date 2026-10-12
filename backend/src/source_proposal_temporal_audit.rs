//! Join proposal provenance with source-exact and compact-subgroup temporal evidence.
//!
//! The temporal values are registered-coordinate neighborhood responses. They
//! are detector-level descriptions and do not establish catalog identity.

use crate::Result;
use csv::{Terminator, WriterBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub const COMPACT_FEATURE_CLASS: &str = "compact_quality";
const COMPACT_SUBGROUPS: [&str; 5] = [
    "all_three",
    "compact_other",
    "dog_only_deblend",
    "dog_only_no_deblend",
    "gaussian_only",
];
const CATALOG_REQUIRED: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "flags",
    "proposal_methods",
];
const DIAGNOSTIC_REQUIRED: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "diagnostic_subgroup",
    "diagnostic_subgroup_label",
    "quality_passed",
    "proposal_methods",
    "candidate_presence",
    "candidate_same_subgroup_presence",
    "quality_presence",
    "quality_same_subgroup_presence",
];
const COMPACT_REQUIRED: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "proposal_subgroup",
    "proposal_subgroup_label",
    "anchor_count",
    "anchor_quality_count",
    "frame_count",
    "required_presence",
    "association_radius_px",
    "candidate_mean_presence",
    "candidate_presence_ge_required_count",
    "candidate_presence_all_frames_count",
    "quality_mean_presence",
    "quality_presence_ge_required_count",
    "quality_presence_all_frames_count",
];
const OPTIONAL_TEMPORAL_FIELDS: [&str; 3] =
    ["frame_count", "required_presence", "association_radius_px"];
const INTERPRETATION_GUARDRAILS: [&str; 5] = [
    "proposal_methods 是来源记录，不是独立投票或概率",
    "candidate/quality presence 是注册坐标邻域响应，不是星表身份",
    "compact_quality 只有来源子组聚合，空的逐源时序字段不是零响应",
    "同一来源组合内的持久性也不能替代 WCS、星表、PSF 和注入真值",
    "任何比例都不是噪点概率、precision、FDR、完备率或确认星数",
];
const ROW_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "diagnostic_subgroup",
    "diagnostic_subgroup_label",
    "method_group",
    "proposal_methods",
    "quality_passed",
    "candidate_presence",
    "candidate_same_subgroup_presence",
    "quality_presence",
    "quality_same_subgroup_presence",
    "frame_count",
    "required_presence",
    "association_radius_px",
    "detail_level",
];
const SUMMARY_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "method_group",
    "detail_level",
    "aggregation_basis",
    "candidate_count",
    "quality_count",
    "class_count",
    "class_fraction",
    "frame_count",
    "required_presence",
    "association_radius_px",
    "candidate_presence_median",
    "candidate_presence_mean",
    "candidate_presence_ge_required_count",
    "candidate_presence_ge_required_fraction",
    "candidate_presence_all_frames_count",
    "candidate_presence_all_frames_fraction",
    "candidate_same_subgroup_presence_median",
    "candidate_same_subgroup_presence_mean",
    "candidate_same_subgroup_presence_ge_required_count",
    "candidate_same_subgroup_presence_ge_required_fraction",
    "candidate_same_subgroup_presence_all_frames_count",
    "candidate_same_subgroup_presence_all_frames_fraction",
    "quality_presence_median",
    "quality_presence_mean",
    "quality_presence_ge_required_count",
    "quality_presence_ge_required_fraction",
    "quality_presence_all_frames_count",
    "quality_presence_all_frames_fraction",
    "quality_same_subgroup_presence_median",
    "quality_same_subgroup_presence_mean",
    "quality_same_subgroup_presence_ge_required_count",
    "quality_same_subgroup_presence_ge_required_fraction",
    "quality_same_subgroup_presence_all_frames_count",
    "quality_same_subgroup_presence_all_frames_fraction",
];
const TARGET_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "method_group",
    "proposal_methods",
    "quality_passed",
    "flags",
    "temporal_detail_available",
    "detail_level",
    "diagnostic_subgroup",
    "candidate_presence",
    "candidate_same_subgroup_presence",
    "quality_presence",
    "quality_same_subgroup_presence",
    "frame_count",
    "required_presence",
    "association_radius_px",
    "method_group_count",
    "method_group_share_in_class",
    "note",
];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SourceProposalTemporalRow {
    pub detection_id: i64,
    pub feature_class: String,
    pub feature_class_label: String,
    pub diagnostic_subgroup: String,
    pub diagnostic_subgroup_label: String,
    pub method_group: String,
    pub proposal_methods: String,
    pub quality_passed: bool,
    pub candidate_presence: usize,
    pub candidate_same_subgroup_presence: usize,
    pub quality_presence: usize,
    pub quality_same_subgroup_presence: usize,
    pub frame_count: usize,
    pub required_presence: usize,
    pub association_radius_px: f64,
    pub detail_level: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SourceProposalTemporalSummary {
    pub feature_class: String,
    pub feature_class_label: String,
    pub method_group: String,
    pub detail_level: String,
    pub aggregation_basis: String,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub class_count: usize,
    pub class_fraction: f64,
    pub frame_count: usize,
    pub required_presence: usize,
    pub association_radius_px: f64,
    pub candidate_presence_median: Option<f64>,
    pub candidate_presence_mean: Option<f64>,
    pub candidate_presence_ge_required_count: usize,
    pub candidate_presence_ge_required_fraction: f64,
    pub candidate_presence_all_frames_count: usize,
    pub candidate_presence_all_frames_fraction: f64,
    pub candidate_same_subgroup_presence_median: Option<f64>,
    pub candidate_same_subgroup_presence_mean: Option<f64>,
    pub candidate_same_subgroup_presence_ge_required_count: Option<usize>,
    pub candidate_same_subgroup_presence_ge_required_fraction: Option<f64>,
    pub candidate_same_subgroup_presence_all_frames_count: Option<usize>,
    pub candidate_same_subgroup_presence_all_frames_fraction: Option<f64>,
    pub quality_presence_median: Option<f64>,
    pub quality_presence_mean: Option<f64>,
    pub quality_presence_ge_required_count: usize,
    pub quality_presence_ge_required_fraction: f64,
    pub quality_presence_all_frames_count: usize,
    pub quality_presence_all_frames_fraction: f64,
    pub quality_same_subgroup_presence_median: Option<f64>,
    pub quality_same_subgroup_presence_mean: Option<f64>,
    pub quality_same_subgroup_presence_ge_required_count: Option<usize>,
    pub quality_same_subgroup_presence_ge_required_fraction: Option<f64>,
    pub quality_same_subgroup_presence_all_frames_count: Option<usize>,
    pub quality_same_subgroup_presence_all_frames_fraction: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SourceProposalTemporalTarget {
    pub detection_id: i64,
    pub feature_class: String,
    pub feature_class_label: String,
    pub method_group: String,
    pub proposal_methods: String,
    pub quality_passed: bool,
    pub flags: String,
    pub temporal_detail_available: bool,
    pub detail_level: String,
    pub diagnostic_subgroup: String,
    pub candidate_presence: Option<usize>,
    pub candidate_same_subgroup_presence: Option<usize>,
    pub quality_presence: Option<usize>,
    pub quality_same_subgroup_presence: Option<usize>,
    pub frame_count: Option<usize>,
    pub required_presence: Option<usize>,
    pub association_radius_px: Option<f64>,
    pub method_group_count: usize,
    pub method_group_share_in_class: f64,
    pub note: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SourceProposalTemporalAuditResult {
    pub catalog_path: String,
    pub diagnostic_sources_path: String,
    pub subgroup_persistence_path: String,
    pub frame_count: usize,
    pub required_presence: usize,
    pub association_radius_px: f64,
    pub source_rows: Vec<SourceProposalTemporalRow>,
    pub summaries: Vec<SourceProposalTemporalSummary>,
    pub targets: Vec<SourceProposalTemporalTarget>,
    pub conclusion: String,
}

impl SourceProposalTemporalAuditResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "diagnostic_sources_path": self.diagnostic_sources_path,
            "subgroup_persistence_path": self.subgroup_persistence_path,
            "frame_count": self.frame_count,
            "required_presence": self.required_presence,
            "association_radius_px": self.association_radius_px,
            "source_row_count": self.source_rows.len(),
            "summary_rows": self.summaries,
            "target_rows": self.targets,
            "conclusion": self.conclusion,
            "interpretation_guardrails": INTERPRETATION_GUARDRAILS,
        })
    }
}

#[derive(Clone, Debug)]
struct CsvTable {
    headers: Vec<String>,
    rows: Vec<BTreeMap<String, Option<String>>>,
}

#[derive(Clone, Debug)]
struct CatalogSource {
    detection_id: i64,
    feature_class: String,
    feature_class_label: String,
    method_group: String,
    proposal_methods: String,
    quality_passed: bool,
    flags: String,
}

#[derive(Clone, Debug)]
struct CompactSubgroup {
    method_group: String,
    anchor_count: usize,
    anchor_quality_count: usize,
    candidate_mean_presence: Option<f64>,
    candidate_presence_ge_required_count: usize,
    candidate_presence_all_frames_count: usize,
    quality_mean_presence: Option<f64>,
    quality_presence_ge_required_count: usize,
    quality_presence_all_frames_count: usize,
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

fn python_repr(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("'{escaped}'")
}

fn value_text(row: &BTreeMap<String, Option<String>>, field: &str) -> String {
    match row.get(field) {
        Some(Some(value)) => value.clone(),
        Some(None) => "None".to_owned(),
        None => String::new(),
    }
}

fn read_csv(path: &Path, required: &[&str]) -> Result<CsvTable> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader
        .headers()
        .map_err(|error| error.to_string())?
        .iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let present = headers.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let missing = required
        .iter()
        .filter(|field| !present.contains(**field))
        .copied()
        .collect::<BTreeSet<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "{}: missing required columns: {}",
            path.display(),
            missing.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|error| error.to_string())?;
        let mut row = BTreeMap::new();
        for (index, header) in headers.iter().enumerate() {
            row.insert(header.clone(), record.get(index).map(str::to_owned));
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{}: CSV has no rows", path.display()));
    }
    Ok(CsvTable { headers, rows })
}

fn integer(value: &str, field: &str, path: &Path, row: usize) -> Result<i64> {
    let parsed = value.trim().parse::<f64>().map_err(|_| {
        format!(
            "{} row {row}: field {} is not an integer",
            path.display(),
            python_repr(field)
        )
    })?;
    if !parsed.is_finite()
        || parsed.fract() != 0.0
        || parsed < i64::MIN as f64
        || parsed >= -(i64::MIN as f64)
    {
        return Err(format!(
            "{} row {row}: field {} is not an integer",
            path.display(),
            python_repr(field)
        ));
    }
    Ok(parsed as i64)
}

fn optional_float(value: &str, field: &str, path: &Path, row: usize) -> Result<Option<f64>> {
    let value = value.trim();
    if value.is_empty() || matches!(value.to_ascii_lowercase().as_str(), "nan" | "none" | "null") {
        return Ok(None);
    }
    let parsed = value.parse::<f64>().map_err(|_| {
        format!(
            "{} row {row}: field {} is not numeric",
            path.display(),
            python_repr(field)
        )
    })?;
    Ok(parsed.is_finite().then_some(parsed))
}

fn required_bool(value: &str, field: &str, path: &Path, row: usize) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(format!(
            "{} row {row}: field {} is not boolean",
            path.display(),
            python_repr(field)
        )),
    }
}

fn subgroup_method_group(subgroup: &str) -> Option<&'static str> {
    match subgroup {
        "all_three" => Some("all_three"),
        "gaussian_only" => Some("gaussian_only"),
        "dog_only_deblend" | "dog_only_no_deblend" => Some("dog_only"),
        "compact_other" => Some("partial"),
        _ => None,
    }
}

fn read_catalog(path: &Path) -> Result<Vec<CatalogSource>> {
    let table = read_csv(path, CATALOG_REQUIRED)?;
    let mut output = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, row) in table.rows.iter().enumerate() {
        let row_number = index + 2;
        let detection_id = integer(
            &value_text(row, "detection_id"),
            "detection_id",
            path,
            row_number,
        )?;
        if !seen.insert(detection_id) {
            return Err(format!(
                "{}: duplicate detection_id={detection_id}",
                path.display()
            ));
        }
        let feature_class = value_text(row, "feature_class").trim().to_owned();
        if feature_class.is_empty() {
            return Err(format!(
                "{} row {row_number}: empty feature_class",
                path.display()
            ));
        }
        let methods = value_text(row, "proposal_methods").trim().to_owned();
        output.push(CatalogSource {
            detection_id,
            feature_class,
            feature_class_label: value_text(row, "feature_class_label").trim().to_owned(),
            method_group: crate::source_proposal_peak_audit::proposal_method_group_name(&methods),
            proposal_methods: methods,
            quality_passed: required_bool(
                &value_text(row, "quality_passed"),
                "quality_passed",
                path,
                row_number,
            )?,
            flags: value_text(row, "flags").trim().to_owned(),
        });
    }
    Ok(output)
}

fn validate_temporal_bounds(
    candidate: i64,
    candidate_same: i64,
    quality: i64,
    quality_same: i64,
    frame_count: usize,
    required: usize,
    context: &str,
) -> Result<()> {
    if frame_count < 1 || required < 1 || required > frame_count {
        return Err(format!("{context}: invalid frame_count/required_presence"));
    }
    for (field, value) in [
        ("candidate_presence", candidate),
        ("candidate_same_subgroup_presence", candidate_same),
        ("quality_presence", quality),
        ("quality_same_subgroup_presence", quality_same),
    ] {
        if value < 0 || value > frame_count as i64 {
            return Err(format!(
                "{context}: {field}={value} outside [0, {frame_count}]"
            ));
        }
    }
    if candidate_same > candidate {
        return Err(format!(
            "{context}: same-subgroup candidate presence exceeds candidate presence"
        ));
    }
    if quality_same > quality {
        return Err(format!(
            "{context}: same-subgroup quality presence exceeds quality presence"
        ));
    }
    Ok(())
}

fn read_diagnostic_sources(
    path: &Path,
    catalog: &[CatalogSource],
    frame_count: usize,
    required: usize,
    radius: f64,
) -> Result<BTreeMap<i64, SourceProposalTemporalRow>> {
    let table = read_csv(path, DIAGNOSTIC_REQUIRED)?;
    let catalog_by_id = catalog
        .iter()
        .map(|source| (source.detection_id, source))
        .collect::<BTreeMap<_, _>>();
    let optional_count = OPTIONAL_TEMPORAL_FIELDS
        .iter()
        .filter(|field| table.headers.iter().any(|header| header == **field))
        .count();
    if optional_count != 0 && optional_count != OPTIONAL_TEMPORAL_FIELDS.len() {
        let mut missing = OPTIONAL_TEMPORAL_FIELDS
            .iter()
            .filter(|field| !table.headers.iter().any(|header| header == **field))
            .copied()
            .collect::<Vec<_>>();
        missing.sort_unstable();
        return Err(format!(
            "{}: optional temporal metadata must be supplied as a complete set; missing {}",
            path.display(),
            missing.join(", ")
        ));
    }
    let mut output = BTreeMap::new();
    for (index, row) in table.rows.iter().enumerate() {
        let row_number = index + 2;
        let detection_id = integer(
            &value_text(row, "detection_id"),
            "detection_id",
            path,
            row_number,
        )?;
        if output.contains_key(&detection_id) {
            return Err(format!(
                "{}: duplicate detection_id={detection_id}",
                path.display()
            ));
        }
        let source = catalog_by_id.get(&detection_id).ok_or_else(|| {
            format!(
                "{} row {row_number}: detection_id={detection_id} is not in source catalog",
                path.display()
            )
        })?;
        let feature_class = value_text(row, "feature_class").trim().to_owned();
        let feature_label = value_text(row, "feature_class_label").trim().to_owned();
        let methods = value_text(row, "proposal_methods").trim().to_owned();
        if feature_class != source.feature_class
            || feature_label != source.feature_class_label
            || crate::source_proposal_peak_audit::proposal_method_group_name(&methods)
                != source.method_group
        {
            return Err(format!(
                "{} row {row_number}: catalog/diagnostic source mismatch for detection_id={detection_id}",
                path.display()
            ));
        }
        let quality_passed = required_bool(
            &value_text(row, "quality_passed"),
            "quality_passed",
            path,
            row_number,
        )?;
        if quality_passed != source.quality_passed {
            return Err(format!(
                "{} row {row_number}: quality_passed mismatch for detection_id={detection_id}",
                path.display()
            ));
        }
        if optional_count != 0 {
            let row_frames = integer(
                &value_text(row, "frame_count"),
                "frame_count",
                path,
                row_number,
            )?;
            let row_required = integer(
                &value_text(row, "required_presence"),
                "required_presence",
                path,
                row_number,
            )?;
            let row_radius = optional_float(
                &value_text(row, "association_radius_px"),
                "association_radius_px",
                path,
                row_number,
            )?;
            if row_frames != frame_count as i64
                || row_required != required as i64
                || row_radius.is_none_or(|value| (value - radius).abs() > 1e-9)
            {
                return Err(format!(
                    "{} row {row_number}: temporal metadata disagrees with subgroup persistence",
                    path.display()
                ));
            }
        }
        let candidate = integer(
            &value_text(row, "candidate_presence"),
            "candidate_presence",
            path,
            row_number,
        )?;
        let candidate_same = integer(
            &value_text(row, "candidate_same_subgroup_presence"),
            "candidate_same_subgroup_presence",
            path,
            row_number,
        )?;
        let quality = integer(
            &value_text(row, "quality_presence"),
            "quality_presence",
            path,
            row_number,
        )?;
        let quality_same = integer(
            &value_text(row, "quality_same_subgroup_presence"),
            "quality_same_subgroup_presence",
            path,
            row_number,
        )?;
        validate_temporal_bounds(
            candidate,
            candidate_same,
            quality,
            quality_same,
            frame_count,
            required,
            &format!("{} row {row_number}", path.display()),
        )?;
        output.insert(
            detection_id,
            SourceProposalTemporalRow {
                detection_id,
                feature_class,
                feature_class_label: feature_label,
                diagnostic_subgroup: value_text(row, "diagnostic_subgroup").trim().to_owned(),
                diagnostic_subgroup_label: value_text(row, "diagnostic_subgroup_label")
                    .trim()
                    .to_owned(),
                method_group: source.method_group.clone(),
                proposal_methods: source.proposal_methods.clone(),
                quality_passed,
                candidate_presence: candidate as usize,
                candidate_same_subgroup_presence: candidate_same as usize,
                quality_presence: quality as usize,
                quality_same_subgroup_presence: quality_same as usize,
                frame_count,
                required_presence: required,
                association_radius_px: radius,
                detail_level: "source_exact".to_owned(),
            },
        );
    }
    Ok(output)
}

fn read_compact_subgroups(path: &Path) -> Result<(Vec<CompactSubgroup>, usize, usize, f64)> {
    let table = read_csv(path, COMPACT_REQUIRED)?;
    let mut output = Vec::new();
    let mut seen = BTreeSet::new();
    let mut frames = BTreeSet::new();
    let mut required_values = BTreeSet::new();
    let mut radii = Vec::new();
    for (index, row) in table.rows.iter().enumerate() {
        let row_number = index + 2;
        let feature_class = value_text(row, "feature_class").trim().to_owned();
        let subgroup = value_text(row, "proposal_subgroup").trim().to_owned();
        if !seen.insert((feature_class.clone(), subgroup.clone())) {
            return Err(format!(
                "{}: duplicate compact subgroup {feature_class}/{subgroup}",
                path.display()
            ));
        }
        if feature_class != COMPACT_FEATURE_CLASS {
            return Err(format!(
                "{} row {row_number}: expected {}, got {}",
                path.display(),
                python_repr(COMPACT_FEATURE_CLASS),
                python_repr(&feature_class)
            ));
        }
        let method_group = subgroup_method_group(&subgroup).ok_or_else(|| {
            format!(
                "{} row {row_number}: unknown compact proposal_subgroup={}; use {}",
                path.display(),
                python_repr(&subgroup),
                COMPACT_SUBGROUPS.join(", ")
            )
        })?;
        let parse_usize = |field: &str| -> Result<usize> {
            usize::try_from(integer(&value_text(row, field), field, path, row_number)?).map_err(
                |_| {
                    format!(
                        "{} row {row_number}: field {} is not an integer",
                        path.display(),
                        python_repr(field)
                    )
                },
            )
        };
        let anchor_count = parse_usize("anchor_count")?;
        let anchor_quality_count = parse_usize("anchor_quality_count")?;
        let frame_count = parse_usize("frame_count")?;
        let required_presence = parse_usize("required_presence")?;
        let radius = optional_float(
            &value_text(row, "association_radius_px"),
            "association_radius_px",
            path,
            row_number,
        )?;
        let candidate_mean_presence = optional_float(
            &value_text(row, "candidate_mean_presence"),
            "candidate_mean_presence",
            path,
            row_number,
        )?;
        let candidate_presence_ge_required_count =
            parse_usize("candidate_presence_ge_required_count")?;
        let candidate_presence_all_frames_count =
            parse_usize("candidate_presence_all_frames_count")?;
        let quality_mean_presence = optional_float(
            &value_text(row, "quality_mean_presence"),
            "quality_mean_presence",
            path,
            row_number,
        )?;
        let quality_presence_ge_required_count = parse_usize("quality_presence_ge_required_count")?;
        let quality_presence_all_frames_count = parse_usize("quality_presence_all_frames_count")?;
        if anchor_count < 1 || anchor_quality_count > anchor_count {
            return Err(format!(
                "{} row {row_number}: invalid anchor counts",
                path.display()
            ));
        }
        if frame_count < 1 || required_presence < 1 || required_presence > frame_count {
            return Err(format!(
                "{} row {row_number}: invalid frame_count/required_presence",
                path.display()
            ));
        }
        let Some(association_radius_px) = radius.filter(|value| *value > 0.0) else {
            return Err(format!(
                "{} row {row_number}: association_radius_px must be positive",
                path.display()
            ));
        };
        if candidate_mean_presence.is_some_and(|value| value < 0.0 || value > frame_count as f64) {
            return Err(format!(
                "{} row {row_number}: candidate_mean_presence outside frame range",
                path.display()
            ));
        }
        if quality_mean_presence.is_some_and(|value| value < 0.0 || value > frame_count as f64) {
            return Err(format!(
                "{} row {row_number}: quality_mean_presence outside frame range",
                path.display()
            ));
        }
        for (field, value) in [
            (
                "candidate_presence_ge_required_count",
                candidate_presence_ge_required_count,
            ),
            (
                "candidate_presence_all_frames_count",
                candidate_presence_all_frames_count,
            ),
            (
                "quality_presence_ge_required_count",
                quality_presence_ge_required_count,
            ),
            (
                "quality_presence_all_frames_count",
                quality_presence_all_frames_count,
            ),
        ] {
            if value > anchor_count {
                return Err(format!(
                    "{} row {row_number}: {field} outside anchor_count",
                    path.display()
                ));
            }
        }
        frames.insert(frame_count);
        required_values.insert(required_presence);
        radii.push(association_radius_px);
        output.push(CompactSubgroup {
            method_group: method_group.to_owned(),
            anchor_count,
            anchor_quality_count,
            candidate_mean_presence,
            candidate_presence_ge_required_count,
            candidate_presence_all_frames_count,
            quality_mean_presence,
            quality_presence_ge_required_count,
            quality_presence_all_frames_count,
        });
    }
    radii.sort_by(f64::total_cmp);
    radii.dedup_by(|left, right| left == right);
    if frames.len() != 1 || required_values.len() != 1 || radii.len() != 1 {
        return Err(format!(
            "{}: temporal parameters are not constant across compact subgroup rows",
            path.display()
        ));
    }
    Ok((
        output,
        *frames.first().unwrap(),
        *required_values.first().unwrap(),
        radii[0],
    ))
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn weighted_mean(rows: &[&CompactSubgroup], field: &str) -> Option<f64> {
    let weighted = rows
        .iter()
        .filter_map(|row| {
            let value = match field {
                "candidate_mean_presence" => row.candidate_mean_presence,
                "quality_mean_presence" => row.quality_mean_presence,
                _ => None,
            }?;
            Some((row.anchor_count, value))
        })
        .collect::<Vec<_>>();
    if weighted.is_empty() {
        return None;
    }
    let total = weighted.iter().map(|(weight, _)| *weight).sum::<usize>();
    (total != 0).then(|| {
        weighted
            .iter()
            .map(|(weight, value)| *weight as f64 * *value)
            .sum::<f64>()
            / total as f64
    })
}

type GroupKey = (String, String);
type CatalogGroups = (BTreeMap<GroupKey, Vec<usize>>, BTreeMap<String, Vec<usize>>);

fn group_catalog(catalog: &[CatalogSource]) -> CatalogGroups {
    let mut by_group: BTreeMap<GroupKey, Vec<usize>> = BTreeMap::new();
    let mut by_class: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, source) in catalog.iter().enumerate() {
        by_group
            .entry((source.feature_class.clone(), source.method_group.clone()))
            .or_default()
            .push(index);
        by_class
            .entry(source.feature_class.clone())
            .or_default()
            .push(index);
    }
    (by_group, by_class)
}

fn exact_summary(
    members: &[SourceProposalTemporalRow],
    catalog_group: &[usize],
    catalog: &[CatalogSource],
    class_count: usize,
    frame_count: usize,
    required_presence: usize,
    association_radius_px: f64,
) -> Result<SourceProposalTemporalSummary> {
    let first = members
        .first()
        .ok_or("cannot summarize an empty exact source group")?;
    let candidate = members
        .iter()
        .map(|row| row.candidate_presence as f64)
        .collect::<Vec<_>>();
    let candidate_same = members
        .iter()
        .map(|row| row.candidate_same_subgroup_presence as f64)
        .collect::<Vec<_>>();
    let quality = members
        .iter()
        .map(|row| row.quality_presence as f64)
        .collect::<Vec<_>>();
    let quality_same = members
        .iter()
        .map(|row| row.quality_same_subgroup_presence as f64)
        .collect::<Vec<_>>();
    let candidate_ge = members
        .iter()
        .filter(|row| row.candidate_presence >= required_presence)
        .count();
    let candidate_all = members
        .iter()
        .filter(|row| row.candidate_presence == frame_count)
        .count();
    let candidate_same_ge = members
        .iter()
        .filter(|row| row.candidate_same_subgroup_presence >= required_presence)
        .count();
    let candidate_same_all = members
        .iter()
        .filter(|row| row.candidate_same_subgroup_presence == frame_count)
        .count();
    let quality_ge = members
        .iter()
        .filter(|row| row.quality_presence >= required_presence)
        .count();
    let quality_all = members
        .iter()
        .filter(|row| row.quality_presence == frame_count)
        .count();
    let quality_same_ge = members
        .iter()
        .filter(|row| row.quality_same_subgroup_presence >= required_presence)
        .count();
    let quality_same_all = members
        .iter()
        .filter(|row| row.quality_same_subgroup_presence == frame_count)
        .count();
    let count = members.len();
    let quality_count = catalog_group
        .iter()
        .filter(|index| catalog[**index].quality_passed)
        .count();
    let fraction = |value: usize| value as f64 / count as f64;
    Ok(SourceProposalTemporalSummary {
        feature_class: first.feature_class.clone(),
        feature_class_label: first.feature_class_label.clone(),
        method_group: first.method_group.clone(),
        detail_level: "source_exact".to_owned(),
        aggregation_basis: "diagnostic_source_rows".to_owned(),
        candidate_count: count,
        quality_count,
        class_count,
        class_fraction: count as f64 / class_count as f64,
        frame_count,
        required_presence,
        association_radius_px,
        candidate_presence_median: median(&candidate),
        candidate_presence_mean: mean(&candidate),
        candidate_presence_ge_required_count: candidate_ge,
        candidate_presence_ge_required_fraction: fraction(candidate_ge),
        candidate_presence_all_frames_count: candidate_all,
        candidate_presence_all_frames_fraction: fraction(candidate_all),
        candidate_same_subgroup_presence_median: median(&candidate_same),
        candidate_same_subgroup_presence_mean: mean(&candidate_same),
        candidate_same_subgroup_presence_ge_required_count: Some(candidate_same_ge),
        candidate_same_subgroup_presence_ge_required_fraction: Some(fraction(candidate_same_ge)),
        candidate_same_subgroup_presence_all_frames_count: Some(candidate_same_all),
        candidate_same_subgroup_presence_all_frames_fraction: Some(fraction(candidate_same_all)),
        quality_presence_median: median(&quality),
        quality_presence_mean: mean(&quality),
        quality_presence_ge_required_count: quality_ge,
        quality_presence_ge_required_fraction: fraction(quality_ge),
        quality_presence_all_frames_count: quality_all,
        quality_presence_all_frames_fraction: fraction(quality_all),
        quality_same_subgroup_presence_median: median(&quality_same),
        quality_same_subgroup_presence_mean: mean(&quality_same),
        quality_same_subgroup_presence_ge_required_count: Some(quality_same_ge),
        quality_same_subgroup_presence_ge_required_fraction: Some(fraction(quality_same_ge)),
        quality_same_subgroup_presence_all_frames_count: Some(quality_same_all),
        quality_same_subgroup_presence_all_frames_fraction: Some(fraction(quality_same_all)),
    })
}

fn compact_summary(
    members: &[&CompactSubgroup],
    catalog_group: &[usize],
    catalog: &[CatalogSource],
    catalog_class: &[usize],
    frame_count: usize,
    required_presence: usize,
    association_radius_px: f64,
) -> Result<SourceProposalTemporalSummary> {
    if members.is_empty() {
        return Err("cannot summarize an empty compact subgroup aggregation".into());
    }
    let count = members.iter().map(|row| row.anchor_count).sum::<usize>();
    let quality_count = members
        .iter()
        .map(|row| row.anchor_quality_count)
        .sum::<usize>();
    let catalog_quality = catalog_group
        .iter()
        .filter(|index| catalog[**index].quality_passed)
        .count();
    if count != catalog_group.len() || quality_count != catalog_quality {
        return Err("compact subgroup anchor counts do not match source catalog".into());
    }
    let candidate_ge = members
        .iter()
        .map(|row| row.candidate_presence_ge_required_count)
        .sum::<usize>();
    let candidate_all = members
        .iter()
        .map(|row| row.candidate_presence_all_frames_count)
        .sum::<usize>();
    let quality_ge = members
        .iter()
        .map(|row| row.quality_presence_ge_required_count)
        .sum::<usize>();
    let quality_all = members
        .iter()
        .map(|row| row.quality_presence_all_frames_count)
        .sum::<usize>();
    let label = catalog_class
        .first()
        .map(|index| catalog[*index].feature_class_label.clone())
        .ok_or("compact feature class is absent from source catalog")?;
    let fraction = |value: usize| value as f64 / count as f64;
    Ok(SourceProposalTemporalSummary {
        feature_class: COMPACT_FEATURE_CLASS.to_owned(),
        feature_class_label: label,
        method_group: members[0].method_group.clone(),
        detail_level: "subgroup_aggregate".to_owned(),
        aggregation_basis: "compact_proposal_subgroup_persistence".to_owned(),
        candidate_count: count,
        quality_count,
        class_count: catalog_class.len(),
        class_fraction: count as f64 / catalog_class.len() as f64,
        frame_count,
        required_presence,
        association_radius_px,
        candidate_presence_median: None,
        candidate_presence_mean: weighted_mean(members, "candidate_mean_presence"),
        candidate_presence_ge_required_count: candidate_ge,
        candidate_presence_ge_required_fraction: fraction(candidate_ge),
        candidate_presence_all_frames_count: candidate_all,
        candidate_presence_all_frames_fraction: fraction(candidate_all),
        candidate_same_subgroup_presence_median: None,
        candidate_same_subgroup_presence_mean: None,
        candidate_same_subgroup_presence_ge_required_count: None,
        candidate_same_subgroup_presence_ge_required_fraction: None,
        candidate_same_subgroup_presence_all_frames_count: None,
        candidate_same_subgroup_presence_all_frames_fraction: None,
        quality_presence_median: None,
        quality_presence_mean: weighted_mean(members, "quality_mean_presence"),
        quality_presence_ge_required_count: quality_ge,
        quality_presence_ge_required_fraction: fraction(quality_ge),
        quality_presence_all_frames_count: quality_all,
        quality_presence_all_frames_fraction: fraction(quality_all),
        quality_same_subgroup_presence_median: None,
        quality_same_subgroup_presence_mean: None,
        quality_same_subgroup_presence_ge_required_count: None,
        quality_same_subgroup_presence_ge_required_fraction: None,
        quality_same_subgroup_presence_all_frames_count: None,
        quality_same_subgroup_presence_all_frames_fraction: None,
    })
}

fn repr_pairs(values: &[(String, String)]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|(left, right)| format!("({}, {})", python_repr(left), python_repr(right)))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn repr_count_map(values: &BTreeMap<String, usize>) -> String {
    format!(
        "{{{}}}",
        values
            .iter()
            .map(|(key, value)| format!("{}: {value}", python_repr(key)))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn build_summaries(
    catalog: &[CatalogSource],
    source_rows: &BTreeMap<i64, SourceProposalTemporalRow>,
    compact_rows: &[CompactSubgroup],
    frames: usize,
    required: usize,
    radius: f64,
) -> Result<Vec<SourceProposalTemporalSummary>> {
    let (by_group, by_class) = group_catalog(catalog);
    let mut exact: BTreeMap<GroupKey, Vec<SourceProposalTemporalRow>> = BTreeMap::new();
    for row in source_rows.values() {
        exact
            .entry((row.feature_class.clone(), row.method_group.clone()))
            .or_default()
            .push(row.clone());
    }
    let mut output = Vec::new();
    for ((feature_class, method_group), mut members) in exact {
        members.sort_by_key(|row| row.detection_id);
        let group = &by_group[&(feature_class.clone(), method_group.clone())];
        output.push(exact_summary(
            &members,
            group,
            catalog,
            by_class[&feature_class].len(),
            frames,
            required,
            radius,
        )?);
    }
    let mut compact: BTreeMap<String, Vec<&CompactSubgroup>> = BTreeMap::new();
    for row in compact_rows {
        compact
            .entry(row.method_group.clone())
            .or_default()
            .push(row);
    }
    if let Some(class) = by_class.get(COMPACT_FEATURE_CLASS) {
        for (method_group, members) in compact {
            let group = &by_group[&(COMPACT_FEATURE_CLASS.to_owned(), method_group)];
            output.push(compact_summary(
                &members, group, catalog, class, frames, required, radius,
            )?);
        }
    }
    let expected = by_group.keys().cloned().collect::<BTreeSet<_>>();
    let actual = output
        .iter()
        .map(|row| (row.feature_class.clone(), row.method_group.clone()))
        .collect::<BTreeSet<_>>();
    if expected != actual {
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let extra = actual.difference(&expected).cloned().collect::<Vec<_>>();
        return Err(format!(
            "summary source groups are incomplete: missing={}, extra={}",
            repr_pairs(&missing),
            repr_pairs(&extra)
        ));
    }
    output.sort_by(|left, right| {
        (&left.feature_class, &left.method_group).cmp(&(&right.feature_class, &right.method_group))
    });
    Ok(output)
}

fn build_targets(
    catalog: &[CatalogSource],
    source_rows: &BTreeMap<i64, SourceProposalTemporalRow>,
    summaries: &[SourceProposalTemporalSummary],
    target_ids: &[i64],
) -> Result<Vec<SourceProposalTemporalTarget>> {
    let mut requested = Vec::new();
    for id in target_ids {
        if !requested.contains(id) {
            requested.push(*id);
        }
    }
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let by_id = catalog
        .iter()
        .map(|source| (source.detection_id, source))
        .collect::<BTreeMap<_, _>>();
    let missing = requested
        .iter()
        .copied()
        .filter(|id| !by_id.contains_key(id))
        .collect::<BTreeSet<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "target detection_id not found in source catalog: {}",
            missing
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let group_counts = summaries
        .iter()
        .map(|row| {
            (
                (row.feature_class.as_str(), row.method_group.as_str()),
                row.candidate_count,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let class_counts = summaries
        .iter()
        .map(|row| (row.feature_class.as_str(), row.class_count))
        .collect::<BTreeMap<_, _>>();
    Ok(requested.into_iter().map(|id| {
        let source = by_id[&id];
        let group_count = group_counts[&(source.feature_class.as_str(), source.method_group.as_str())];
        let share = group_count as f64 / class_counts[source.feature_class.as_str()] as f64;
        if let Some(temporal) = source_rows.get(&id) {
            SourceProposalTemporalTarget {
                detection_id: id,
                feature_class: source.feature_class.clone(),
                feature_class_label: source.feature_class_label.clone(),
                method_group: source.method_group.clone(),
                proposal_methods: source.proposal_methods.clone(),
                quality_passed: source.quality_passed,
                flags: source.flags.clone(),
                temporal_detail_available: true,
                detail_level: temporal.detail_level.clone(),
                diagnostic_subgroup: temporal.diagnostic_subgroup.clone(),
                candidate_presence: Some(temporal.candidate_presence),
                candidate_same_subgroup_presence: Some(temporal.candidate_same_subgroup_presence),
                quality_presence: Some(temporal.quality_presence),
                quality_same_subgroup_presence: Some(temporal.quality_same_subgroup_presence),
                frame_count: Some(temporal.frame_count),
                required_presence: Some(temporal.required_presence),
                association_radius_px: Some(temporal.association_radius_px),
                method_group_count: group_count,
                method_group_share_in_class: share,
                note: "非紧凑候选有逐源诊断时序".to_owned(),
            }
        } else {
            SourceProposalTemporalTarget {
                detection_id: id,
                feature_class: source.feature_class.clone(),
                feature_class_label: source.feature_class_label.clone(),
                method_group: source.method_group.clone(),
                proposal_methods: source.proposal_methods.clone(),
                quality_passed: source.quality_passed,
                flags: source.flags.clone(),
                temporal_detail_available: false,
                detail_level: "subgroup_aggregate".to_owned(),
                diagnostic_subgroup: String::new(),
                candidate_presence: None,
                candidate_same_subgroup_presence: None,
                quality_presence: None,
                quality_same_subgroup_presence: None,
                frame_count: None,
                required_presence: None,
                association_radius_px: None,
                method_group_count: group_count,
                method_group_share_in_class: share,
                note: "该来源组只有 compact proposal subgroup 汇总，未提供逐源时序；空值不是零响应".to_owned(),
            }
        }
    }).collect())
}

pub fn run(
    catalog_path: &Path,
    diagnostic_path: &Path,
    subgroup_path: &Path,
    target_ids: &[i64],
) -> Result<SourceProposalTemporalAuditResult> {
    let catalog_path = resolved_path(catalog_path)?;
    let diagnostic_path = resolved_path(diagnostic_path)?;
    let subgroup_path = resolved_path(subgroup_path)?;
    let catalog = read_catalog(&catalog_path)?;
    let (compact_rows, frames, required, radius) = read_compact_subgroups(&subgroup_path)?;
    let source_rows =
        read_diagnostic_sources(&diagnostic_path, &catalog, frames, required, radius)?;
    let noncompact = catalog
        .iter()
        .filter(|row| row.feature_class != COMPACT_FEATURE_CLASS)
        .map(|row| row.detection_id)
        .collect::<BTreeSet<_>>();
    let diagnostic_ids = source_rows.keys().copied().collect::<BTreeSet<_>>();
    if noncompact != diagnostic_ids {
        let catalog_only = noncompact
            .difference(&diagnostic_ids)
            .take(5)
            .map(i64::to_string)
            .collect::<Vec<_>>();
        let diagnostic_only = diagnostic_ids
            .difference(&noncompact)
            .take(5)
            .map(i64::to_string)
            .collect::<Vec<_>>();
        return Err(format!(
            "diagnostic source IDs must equal all non-compact catalog IDs: catalog_only=[{}], diagnostic_only=[{}]",
            catalog_only.join(", "), diagnostic_only.join(", ")
        ));
    }
    let (by_group, _) = group_catalog(&catalog);
    let expected = by_group
        .iter()
        .filter(|((class, _), _)| class == COMPACT_FEATURE_CLASS)
        .map(|((_, method), members)| (method.clone(), members.len()))
        .collect::<BTreeMap<_, _>>();
    let mut actual = BTreeMap::new();
    for row in &compact_rows {
        *actual.entry(row.method_group.clone()).or_insert(0usize) += row.anchor_count;
    }
    if expected != actual {
        return Err(format!(
            "compact subgroup anchor counts do not cover source catalog groups: expected={}, actual={}",
            repr_count_map(&expected), repr_count_map(&actual)
        ));
    }
    let summaries = build_summaries(
        &catalog,
        &source_rows,
        &compact_rows,
        frames,
        required,
        radius,
    )?;
    let targets = build_targets(&catalog, &source_rows, &summaries, target_ids)?;
    let target_text = targets
        .iter()
        .map(|row| {
            format!(
                "ID {}={}, candidate={}, quality={}",
                row.detection_id,
                row.method_group,
                row.candidate_presence
                    .map_or_else(|| "None".to_owned(), |v| v.to_string()),
                row.quality_presence
                    .map_or_else(|| "None".to_owned(), |v| v.to_string())
            )
        })
        .collect::<Vec<_>>()
        .join("；");
    let mut conclusion = "来源—时序交叉审计表明，来源交集、候选位置持久和质量响应持久是三条不同证据轴；非紧凑候选可逐源核对，compact_quality 只能按来源子组汇总，不能把缺失的逐源时序写成 0。".to_owned();
    if !target_text.is_empty() {
        conclusion.push_str(&format!("目标结果：{target_text}。"));
    }
    Ok(SourceProposalTemporalAuditResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        diagnostic_sources_path: diagnostic_path.to_string_lossy().into_owned(),
        subgroup_persistence_path: subgroup_path.to_string_lossy().into_owned(),
        frame_count: frames,
        required_presence: required,
        association_radius_px: radius,
        source_rows: source_rows.into_values().collect(),
        summaries,
        targets,
        conclusion,
    })
}

fn python_float(value: f64) -> String {
    let raw = format!("{value:?}");
    if let Some((mantissa, exponent)) = raw.split_once('e') {
        format!("{mantissa}e{:+03}", exponent.parse::<i32>().unwrap_or(0))
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

pub fn write_artifacts(
    result: &SourceProposalTemporalAuditResult,
    output: &Path,
) -> Result<PathBuf> {
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    write_csv(
        &output.join("source_proposal_temporal_rows.csv"),
        ROW_FIELDS,
        &result.source_rows,
    )?;
    write_csv(
        &output.join("source_proposal_temporal_summary.csv"),
        SUMMARY_FIELDS,
        &result.summaries,
    )?;
    write_csv(
        &output.join("source_proposal_temporal_targets.csv"),
        TARGET_FIELDS,
        &result.targets,
    )?;
    fs::write(
        output.join("source_proposal_temporal_audit.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut positional = Vec::<PathBuf>::new();
    let mut targets = Vec::new();
    let mut output = PathBuf::from("tmp/source-proposal-temporal-audit");
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
                    targets.push(value.parse::<i64>().map_err(|error| error.to_string())?);
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
    if positional.len() != 3 {
        return Err(
            "expected source catalog, diagnostic sources, and subgroup persistence CSVs".into(),
        );
    }
    let result = run(&positional[0], &positional[1], &positional[2], &targets)?;
    write_artifacts(&result, &output)?;
    Ok(json!({
        "tool": "rst19-source-proposal-temporal-audit",
        "out_dir": output,
        "result_path": output.join("source_proposal_temporal_audit.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_bounds_reject_invalid_presence_relationships() {
        assert!(validate_temporal_bounds(3, 2, 2, 1, 5, 4, "case").is_ok());
        assert!(validate_temporal_bounds(3, 4, 2, 1, 5, 4, "case").is_err());
        assert!(validate_temporal_bounds(3, 2, 2, 1, 5, 6, "case").is_err());
    }

    #[test]
    fn compact_subgroups_map_to_source_proposal_methods() {
        assert_eq!(subgroup_method_group("all_three"), Some("all_three"));
        assert_eq!(subgroup_method_group("dog_only_deblend"), Some("dog_only"));
        assert_eq!(
            subgroup_method_group("dog_only_no_deblend"),
            Some("dog_only")
        );
        assert_eq!(subgroup_method_group("compact_other"), Some("partial"));
        assert_eq!(subgroup_method_group("unknown"), None);
    }
}
