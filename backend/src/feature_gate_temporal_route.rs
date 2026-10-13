//! Joins detector quality-gate routes to precomputed temporal diagnostic rows.
//!
//! This is a CSV/JSON postprocessor. It never reopens FITS files or runs a
//! detector; temporal presence remains a registered-coordinate response and
//! does not establish catalog identity or physical source truth.

use csv::{Terminator, WriterBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const ROUTE_FIELDS: [&str; 8] = [
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "route",
    "numeric_failure_reasons",
    "structural_flags",
    "flags",
];
const DIAGNOSTIC_FIELDS: [&str; 10] = [
    "detection_id",
    "feature_class",
    "feature_class_label",
    "diagnostic_subgroup",
    "diagnostic_subgroup_label",
    "diagnostic_subgroup_definition",
    "candidate_presence",
    "candidate_same_subgroup_presence",
    "quality_presence",
    "quality_same_subgroup_presence",
];
const ROUTES: [&str; 5] = [
    "quality_passed",
    "rejected_numeric_only",
    "rejected_structural_only",
    "rejected_numeric_and_structural",
    "rejected_unexplained",
];
const GUARDRAILS: [&str; 6] = [
    "temporal presence is a registered-coordinate neighborhood response, not a catalog identity",
    "blank temporal fields mean no source-level diagnostic row was supplied, not zero response",
    "a quality gate route is an observed detector-rule decomposition, not a counterfactual cause",
    "same_subgroup_presence is a mechanism-consistency diagnostic, not a physical artifact label",
    "target distribution fractions are tie-inclusive empirical descriptions, not probabilities or confidence intervals",
    "none of these counts is a noise probability, completeness, precision, or confirmed star count",
];

#[derive(Clone, Debug, Serialize)]
pub struct FeatureGateTemporalSourceRow {
    pub detection_id: String,
    pub feature_class: String,
    pub feature_class_label: String,
    pub quality_passed: bool,
    pub route: String,
    pub numeric_failure_reasons: String,
    pub structural_flags: String,
    pub flags: String,
    pub temporal_source_covered: bool,
    pub diagnostic_subgroup: String,
    pub diagnostic_subgroup_label: String,
    pub diagnostic_subgroup_definition: String,
    pub candidate_presence: Option<i64>,
    pub candidate_same_subgroup_presence: Option<i64>,
    pub quality_presence: Option<i64>,
    pub quality_same_subgroup_presence: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeatureGateTemporalRouteSummaryRow {
    pub feature_class: String,
    pub feature_class_label: String,
    pub route: String,
    pub route_count: usize,
    pub temporal_source_count: usize,
    pub temporal_coverage_fraction: f64,
    pub candidate_presence_median: Option<f64>,
    pub candidate_presence_mean: Option<f64>,
    pub candidate_presence_ge_required_count: Option<usize>,
    pub candidate_presence_ge_required_fraction: Option<f64>,
    pub candidate_presence_all_frames_count: Option<usize>,
    pub candidate_presence_all_frames_fraction: Option<f64>,
    pub candidate_same_subgroup_ge_required_count: Option<usize>,
    pub candidate_same_subgroup_ge_required_fraction: Option<f64>,
    pub quality_presence_median: Option<f64>,
    pub quality_presence_mean: Option<f64>,
    pub quality_presence_ge_required_count: Option<usize>,
    pub quality_presence_ge_required_fraction: Option<f64>,
    pub quality_presence_all_frames_count: Option<usize>,
    pub quality_presence_all_frames_fraction: Option<f64>,
    pub quality_presence_any_count: Option<usize>,
    pub quality_presence_any_fraction: Option<f64>,
    pub quality_same_subgroup_ge_required_count: Option<usize>,
    pub quality_same_subgroup_ge_required_fraction: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeatureGateTemporalSubgroupSummaryRow {
    pub feature_class: String,
    pub feature_class_label: String,
    pub diagnostic_subgroup: String,
    pub diagnostic_subgroup_label: String,
    pub diagnostic_subgroup_definition: String,
    pub subgroup_count: usize,
    pub route_quality_passed_count: usize,
    pub route_rejected_numeric_only_count: usize,
    pub route_rejected_structural_only_count: usize,
    pub route_rejected_numeric_and_structural_count: usize,
    pub route_rejected_unexplained_count: usize,
    pub candidate_presence_median: Option<f64>,
    pub candidate_presence_ge_required_count: Option<usize>,
    pub candidate_presence_ge_required_fraction: Option<f64>,
    pub candidate_presence_all_frames_count: Option<usize>,
    pub candidate_same_subgroup_ge_required_count: Option<usize>,
    pub candidate_same_subgroup_ge_required_fraction: Option<f64>,
    pub mechanism_gap_fraction: Option<f64>,
    pub quality_presence_ge_required_count: Option<usize>,
    pub quality_presence_ge_required_fraction: Option<f64>,
    pub quality_same_subgroup_ge_required_count: Option<usize>,
    pub quality_same_subgroup_ge_required_fraction: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeatureGateTemporalTargetRow {
    pub detection_id: String,
    pub feature_class: String,
    pub feature_class_label: String,
    pub route: String,
    pub temporal_source_covered: bool,
    pub diagnostic_subgroup: String,
    pub diagnostic_subgroup_label: String,
    pub diagnostic_subgroup_definition: String,
    pub subgroup_count: usize,
    pub candidate_presence: Option<i64>,
    pub candidate_presence_le_fraction: Option<f64>,
    pub candidate_presence_ge_fraction: Option<f64>,
    pub candidate_same_subgroup_presence: Option<i64>,
    pub candidate_same_subgroup_presence_le_fraction: Option<f64>,
    pub candidate_same_subgroup_presence_ge_fraction: Option<f64>,
    pub quality_presence: Option<i64>,
    pub quality_presence_le_fraction: Option<f64>,
    pub quality_presence_ge_fraction: Option<f64>,
    pub candidate_presence_ge_required: Option<bool>,
    pub candidate_same_subgroup_ge_required: Option<bool>,
    pub quality_presence_ge_required: Option<bool>,
    pub candidate_and_same_ge_required: Option<bool>,
    pub candidate_same_and_quality_ge_required: Option<bool>,
    pub all_three_ge_required: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct FeatureGateTemporalRouteResult {
    pub route_csv: String,
    pub diagnostic_sources_csv: String,
    pub persistence_json: String,
    pub frame_count: i64,
    pub required_presence: i64,
    pub association_radius_px: f64,
    pub association_method: String,
    pub source_rows: Vec<FeatureGateTemporalSourceRow>,
    pub summary_rows: Vec<FeatureGateTemporalRouteSummaryRow>,
    pub subgroup_summary_rows: Vec<FeatureGateTemporalSubgroupSummaryRow>,
    pub target_rows: Vec<FeatureGateTemporalTargetRow>,
}

impl FeatureGateTemporalRouteResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "route_csv": self.route_csv,
            "diagnostic_sources_csv": self.diagnostic_sources_csv,
            "persistence_json": self.persistence_json,
            "frame_count": self.frame_count,
            "required_presence": self.required_presence,
            "association_radius_px": self.association_radius_px,
            "association_method": self.association_method,
            "source_row_count": self.source_rows.len(),
            "temporal_source_row_count": self.source_rows.iter().filter(|row| row.temporal_source_covered).count(),
            "summary_rows": self.summary_rows,
            "subgroup_summary_rows": self.subgroup_summary_rows,
            "target_rows": self.target_rows,
            "interpretation_guardrails": GUARDRAILS,
        })
    }
}

type CsvRow = HashMap<String, String>;
type SubgroupKey = (String, String, String, String, String);

fn read_csv(path: &Path, required_fields: &[&str]) -> Result<Vec<CsvRow>, String> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|error| format!("{}: {error}", path.display()))?
        .iter()
        .map(|header| header.trim_start_matches('\u{feff}').to_owned())
        .collect::<Vec<_>>();
    let mut missing = required_fields
        .iter()
        .filter(|field| !headers.iter().any(|header| header == **field))
        .map(|field| (*field).to_owned())
        .collect::<Vec<_>>();
    missing.sort();
    if !missing.is_empty() {
        return Err(format!(
            "{}: missing required columns: {}",
            path.display(),
            missing.join(", ")
        ));
    }
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|error| format!("{}: {error}", path.display()))?;
        let mut row = CsvRow::new();
        for (index, header) in headers.iter().enumerate() {
            row.insert(
                header.clone(),
                record.get(index).unwrap_or_default().to_owned(),
            );
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{}: CSV has no rows", path.display()));
    }
    Ok(rows)
}

fn value_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(value)) => value.clone(),
        Some(Value::Bool(value)) => (if *value { "True" } else { "False" }).to_owned(),
        Some(value) => value.to_string(),
    }
}

fn integer_text(text: &str, field: &str, path: &Path) -> Result<i64, String> {
    let parsed = text
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("{}: field {field:?} is not an integer", path.display()))?;
    if !parsed.is_finite()
        || parsed.fract() != 0.0
        || parsed < i64::MIN as f64
        || parsed >= i64::MAX as f64
    {
        return Err(format!(
            "{}: field {field:?} is not an integer",
            path.display()
        ));
    }
    Ok(parsed as i64)
}

fn optional_integer(row: Option<&CsvRow>, field: &str, path: &Path) -> Result<Option<i64>, String> {
    let Some(value) = row.and_then(|row| row.get(field)) else {
        return Ok(None);
    };
    if value.trim().is_empty() {
        return Ok(None);
    }
    let parsed = integer_text(value, field, path)?;
    if parsed < 0 {
        return Err(format!(
            "{}: field {field:?} must not be negative",
            path.display()
        ));
    }
    Ok(Some(parsed))
}

fn optional_float_text(text: &str, field: &str, path: &Path) -> Result<Option<f64>, String> {
    if text.trim().is_empty() {
        return Ok(None);
    }
    let parsed = text
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("{}: field {field:?} is not numeric", path.display()))?;
    if !parsed.is_finite() {
        return Err(format!("{}: field {field:?} is not finite", path.display()));
    }
    Ok(Some(parsed))
}

fn metadata_integer(value: Option<&Value>, field: &str, path: &Path) -> Result<i64, String> {
    integer_text(&value_text(value), field, path)
}

fn metadata(path: &Path) -> Result<(i64, i64, f64, String), String> {
    let text = fs::read_to_string(path)
        .map_err(|_| format!("{}: invalid persistence JSON", path.display()))?;
    let payload: Value = serde_json::from_str(&text)
        .map_err(|_| format!("{}: invalid persistence JSON", path.display()))?;
    let frame_count = metadata_integer(payload.get("frame_count"), "frame_count", path)?;
    let required_presence =
        metadata_integer(payload.get("required_presence"), "required_presence", path)?;
    let association_radius = optional_float_text(
        &value_text(payload.get("association_radius_px")),
        "association_radius_px",
        path,
    )?
    .ok_or_else(|| format!("{}: association_radius_px is required", path.display()))?;
    let association_method = match payload.get("association_method") {
        None => String::new(),
        Some(Value::Null) => "None".to_owned(),
        Some(value) => value_text(Some(value)).trim().to_owned(),
    };
    if frame_count < 1
        || required_presence < 1
        || required_presence > frame_count
        || association_radius <= 0.0
    {
        return Err(format!(
            "{}: invalid frame/presence/association metadata",
            path.display()
        ));
    }
    Ok((
        frame_count,
        required_presence,
        association_radius,
        association_method,
    ))
}

fn field<'a>(row: &'a CsvRow, name: &str) -> &'a str {
    row.get(name).map(String::as_str).unwrap_or_default()
}

fn bool_value(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "1" | "yes"
    )
}

fn median(values: &[i64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    let middle = ordered.len() / 2;
    Some(if ordered.len() % 2 == 1 {
        ordered[middle] as f64
    } else {
        (ordered[middle - 1] as f64 + ordered[middle] as f64) / 2.0
    })
}

fn mean(values: &[i64]) -> Option<f64> {
    (!values.is_empty())
        .then(|| values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64)
}

fn fraction(count: Option<usize>, denominator: usize) -> Option<f64> {
    count
        .filter(|_| denominator != 0)
        .map(|count| count as f64 / denominator as f64)
}

fn empirical_fraction(value: Option<i64>, values: &[i64], relation: &str) -> Option<f64> {
    let value = value?;
    if values.is_empty() {
        return None;
    }
    let count = match relation {
        "le" => values
            .iter()
            .filter(|candidate| **candidate <= value)
            .count(),
        "ge" => values
            .iter()
            .filter(|candidate| **candidate >= value)
            .count(),
        _ => return None,
    };
    Some(count as f64 / values.len() as f64)
}

fn covered_values<F>(members: &[FeatureGateTemporalSourceRow], select: F) -> Vec<i64>
where
    F: Fn(&FeatureGateTemporalSourceRow) -> Option<i64>,
{
    members
        .iter()
        .filter(|row| row.temporal_source_covered)
        .filter_map(select)
        .collect()
}

fn ge_count(values: &[i64], threshold: i64) -> Option<usize> {
    (!values.is_empty()).then(|| values.iter().filter(|value| **value >= threshold).count())
}

fn all_frames_count(values: &[i64], frame_count: i64) -> Option<usize> {
    ge_count(values, frame_count)
}

fn any_count(values: &[i64]) -> Option<usize> {
    (!values.is_empty()).then(|| values.iter().filter(|value| **value > 0).count())
}

fn summarize_group(
    members: &[FeatureGateTemporalSourceRow],
    frame_count: i64,
    required_presence: i64,
) -> FeatureGateTemporalRouteSummaryRow {
    let candidate = covered_values(members, |row| row.candidate_presence);
    let same = covered_values(members, |row| row.candidate_same_subgroup_presence);
    let quality = covered_values(members, |row| row.quality_presence);
    let quality_same = covered_values(members, |row| row.quality_same_subgroup_presence);
    let covered_count = members
        .iter()
        .filter(|row| row.temporal_source_covered)
        .count();
    FeatureGateTemporalRouteSummaryRow {
        feature_class: members[0].feature_class.clone(),
        feature_class_label: members[0].feature_class_label.clone(),
        route: members[0].route.clone(),
        route_count: members.len(),
        temporal_source_count: covered_count,
        temporal_coverage_fraction: covered_count as f64 / members.len() as f64,
        candidate_presence_median: median(&candidate),
        candidate_presence_mean: mean(&candidate),
        candidate_presence_ge_required_count: ge_count(&candidate, required_presence),
        candidate_presence_ge_required_fraction: fraction(
            ge_count(&candidate, required_presence),
            candidate.len(),
        ),
        candidate_presence_all_frames_count: all_frames_count(&candidate, frame_count),
        candidate_presence_all_frames_fraction: fraction(
            all_frames_count(&candidate, frame_count),
            candidate.len(),
        ),
        candidate_same_subgroup_ge_required_count: ge_count(&same, required_presence),
        candidate_same_subgroup_ge_required_fraction: fraction(
            ge_count(&same, required_presence),
            same.len(),
        ),
        quality_presence_median: median(&quality),
        quality_presence_mean: mean(&quality),
        quality_presence_ge_required_count: ge_count(&quality, required_presence),
        quality_presence_ge_required_fraction: fraction(
            ge_count(&quality, required_presence),
            quality.len(),
        ),
        quality_presence_all_frames_count: all_frames_count(&quality, frame_count),
        quality_presence_all_frames_fraction: fraction(
            all_frames_count(&quality, frame_count),
            quality.len(),
        ),
        quality_presence_any_count: any_count(&quality),
        quality_presence_any_fraction: fraction(any_count(&quality), quality.len()),
        quality_same_subgroup_ge_required_count: ge_count(&quality_same, required_presence),
        quality_same_subgroup_ge_required_fraction: fraction(
            ge_count(&quality_same, required_presence),
            quality_same.len(),
        ),
    }
}

fn route_count(members: &[FeatureGateTemporalSourceRow], route: &str) -> usize {
    members
        .iter()
        .filter(|member| member.route == route)
        .count()
}

fn summarize_subgroup(
    members: &[FeatureGateTemporalSourceRow],
    frame_count: i64,
    required_presence: i64,
) -> FeatureGateTemporalSubgroupSummaryRow {
    let candidate = covered_values(members, |row| row.candidate_presence);
    let same = covered_values(members, |row| row.candidate_same_subgroup_presence);
    let quality = covered_values(members, |row| row.quality_presence);
    let quality_same = covered_values(members, |row| row.quality_same_subgroup_presence);
    let candidate_ge = ge_count(&candidate, required_presence);
    let candidate_same_ge = ge_count(&same, required_presence);
    let quality_ge = ge_count(&quality, required_presence);
    let quality_same_ge = ge_count(&quality_same, required_presence);
    let mechanism_gap = match (candidate_ge, candidate_same_ge) {
        (Some(candidates), Some(same_count)) if candidate.len() == same.len() => {
            Some((candidates as f64 - same_count as f64) / candidate.len() as f64)
        }
        _ => None,
    };
    let first = &members[0];
    FeatureGateTemporalSubgroupSummaryRow {
        feature_class: first.feature_class.clone(),
        feature_class_label: first.feature_class_label.clone(),
        diagnostic_subgroup: first.diagnostic_subgroup.clone(),
        diagnostic_subgroup_label: first.diagnostic_subgroup_label.clone(),
        diagnostic_subgroup_definition: first.diagnostic_subgroup_definition.clone(),
        subgroup_count: members.len(),
        route_quality_passed_count: route_count(members, ROUTES[0]),
        route_rejected_numeric_only_count: route_count(members, ROUTES[1]),
        route_rejected_structural_only_count: route_count(members, ROUTES[2]),
        route_rejected_numeric_and_structural_count: route_count(members, ROUTES[3]),
        route_rejected_unexplained_count: route_count(members, ROUTES[4]),
        candidate_presence_median: median(&candidate),
        candidate_presence_ge_required_count: candidate_ge,
        candidate_presence_ge_required_fraction: fraction(candidate_ge, candidate.len()),
        candidate_presence_all_frames_count: all_frames_count(&candidate, frame_count),
        candidate_same_subgroup_ge_required_count: candidate_same_ge,
        candidate_same_subgroup_ge_required_fraction: fraction(candidate_same_ge, same.len()),
        mechanism_gap_fraction: mechanism_gap,
        quality_presence_ge_required_count: quality_ge,
        quality_presence_ge_required_fraction: fraction(quality_ge, quality.len()),
        quality_same_subgroup_ge_required_count: quality_same_ge,
        quality_same_subgroup_ge_required_fraction: fraction(quality_same_ge, quality_same.len()),
    }
}

fn target_row(
    target: &FeatureGateTemporalSourceRow,
    members: &[FeatureGateTemporalSourceRow],
    required_presence: i64,
) -> FeatureGateTemporalTargetRow {
    let candidate = members
        .iter()
        .filter_map(|row| row.candidate_presence)
        .collect::<Vec<_>>();
    let same = members
        .iter()
        .filter_map(|row| row.candidate_same_subgroup_presence)
        .collect::<Vec<_>>();
    let quality = members
        .iter()
        .filter_map(|row| row.quality_presence)
        .collect::<Vec<_>>();
    let candidate_ready = target
        .candidate_presence
        .map(|value| value >= required_presence);
    let same_ready = target
        .candidate_same_subgroup_presence
        .map(|value| value >= required_presence);
    let quality_ready = target
        .quality_presence
        .map(|value| value >= required_presence);
    let candidate_and_same = candidate_ready.zip(same_ready).map(|(a, b)| a && b);
    let candidate_same_and_quality = candidate_ready
        .zip(same_ready)
        .zip(quality_ready)
        .map(|((a, b), c)| a && b && c);
    FeatureGateTemporalTargetRow {
        detection_id: target.detection_id.clone(),
        feature_class: target.feature_class.clone(),
        feature_class_label: target.feature_class_label.clone(),
        route: target.route.clone(),
        temporal_source_covered: target.temporal_source_covered,
        diagnostic_subgroup: target.diagnostic_subgroup.clone(),
        diagnostic_subgroup_label: target.diagnostic_subgroup_label.clone(),
        diagnostic_subgroup_definition: target.diagnostic_subgroup_definition.clone(),
        subgroup_count: members.len(),
        candidate_presence: target.candidate_presence,
        candidate_presence_le_fraction: empirical_fraction(
            target.candidate_presence,
            &candidate,
            "le",
        ),
        candidate_presence_ge_fraction: empirical_fraction(
            target.candidate_presence,
            &candidate,
            "ge",
        ),
        candidate_same_subgroup_presence: target.candidate_same_subgroup_presence,
        candidate_same_subgroup_presence_le_fraction: empirical_fraction(
            target.candidate_same_subgroup_presence,
            &same,
            "le",
        ),
        candidate_same_subgroup_presence_ge_fraction: empirical_fraction(
            target.candidate_same_subgroup_presence,
            &same,
            "ge",
        ),
        quality_presence: target.quality_presence,
        quality_presence_le_fraction: empirical_fraction(target.quality_presence, &quality, "le"),
        quality_presence_ge_fraction: empirical_fraction(target.quality_presence, &quality, "ge"),
        candidate_presence_ge_required: candidate_ready,
        candidate_same_subgroup_ge_required: same_ready,
        quality_presence_ge_required: quality_ready,
        candidate_and_same_ge_required: candidate_and_same,
        candidate_same_and_quality_ge_required: candidate_same_and_quality,
        all_three_ge_required: candidate_same_and_quality,
    }
}

fn subgroup_key(row: &FeatureGateTemporalSourceRow) -> SubgroupKey {
    (
        row.feature_class.clone(),
        row.feature_class_label.clone(),
        row.diagnostic_subgroup.clone(),
        row.diagnostic_subgroup_label.clone(),
        row.diagnostic_subgroup_definition.clone(),
    )
}

pub fn run(
    route_csv: &Path,
    diagnostic_sources_csv: &Path,
    persistence_json: &Path,
    target_ids: &[String],
) -> Result<FeatureGateTemporalRouteResult, String> {
    let (frame_count, required_presence, association_radius_px, association_method) =
        metadata(persistence_json)?;
    let route_records = read_csv(route_csv, &ROUTE_FIELDS)?;
    let diagnostic_records = read_csv(diagnostic_sources_csv, &DIAGNOSTIC_FIELDS)?;
    let mut seen_targets = HashSet::new();
    let normalized_targets = target_ids
        .iter()
        .map(|target| target.trim())
        .filter(|target| !target.is_empty())
        .filter(|target| seen_targets.insert((*target).to_owned()))
        .map(str::to_owned)
        .collect::<Vec<_>>();

    let mut diagnostic_by_id = HashMap::new();
    for record in diagnostic_records {
        let id = field(&record, "detection_id").trim().to_owned();
        if id.is_empty() {
            return Err(format!(
                "{}: empty detection_id",
                diagnostic_sources_csv.display()
            ));
        }
        if diagnostic_by_id.insert(id.clone(), record).is_some() {
            return Err(format!(
                "{}: duplicate detection_id={id}",
                diagnostic_sources_csv.display()
            ));
        }
    }

    let mut source_rows = Vec::new();
    let mut seen_route_ids = HashSet::new();
    let mut grouped: BTreeMap<(String, String, String), Vec<FeatureGateTemporalSourceRow>> =
        BTreeMap::new();
    let mut subgrouped: BTreeMap<SubgroupKey, Vec<FeatureGateTemporalSourceRow>> = BTreeMap::new();
    for record in route_records {
        let detection_id = field(&record, "detection_id").trim().to_owned();
        if detection_id.is_empty() {
            return Err(format!("{}: empty detection_id", route_csv.display()));
        }
        if !seen_route_ids.insert(detection_id.clone()) {
            return Err(format!(
                "{}: duplicate detection_id={detection_id}",
                route_csv.display()
            ));
        }
        let feature_class = field(&record, "feature_class").trim().to_owned();
        let feature_class_label = {
            let label = field(&record, "feature_class_label").trim();
            if label.is_empty() {
                feature_class.clone()
            } else {
                label.to_owned()
            }
        };
        let route = field(&record, "route").trim().to_owned();
        if feature_class.is_empty() || route.is_empty() {
            return Err(format!(
                "{}: row {detection_id} has empty feature_class or route",
                route_csv.display()
            ));
        }
        let diagnostic = diagnostic_by_id.get(&detection_id);
        if diagnostic.is_none() && route != "quality_passed" {
            return Err(format!(
                "{}: missing temporal row for rejected route candidate {detection_id}",
                diagnostic_sources_csv.display()
            ));
        }
        if let Some(diagnostic) = diagnostic {
            if field(diagnostic, "feature_class").trim() != feature_class {
                return Err(format!(
                    "feature class mismatch for detection_id={detection_id}"
                ));
            }
        }
        let row = FeatureGateTemporalSourceRow {
            detection_id,
            feature_class,
            feature_class_label,
            quality_passed: bool_value(field(&record, "quality_passed")),
            route,
            numeric_failure_reasons: field(&record, "numeric_failure_reasons").trim().to_owned(),
            structural_flags: field(&record, "structural_flags").trim().to_owned(),
            flags: field(&record, "flags").trim().to_owned(),
            temporal_source_covered: diagnostic.is_some(),
            diagnostic_subgroup: diagnostic
                .map(|row| field(row, "diagnostic_subgroup").trim().to_owned())
                .unwrap_or_default(),
            diagnostic_subgroup_label: diagnostic
                .map(|row| field(row, "diagnostic_subgroup_label").trim().to_owned())
                .unwrap_or_default(),
            diagnostic_subgroup_definition: diagnostic
                .map(|row| {
                    field(row, "diagnostic_subgroup_definition")
                        .trim()
                        .to_owned()
                })
                .unwrap_or_default(),
            candidate_presence: optional_integer(
                diagnostic,
                "candidate_presence",
                diagnostic_sources_csv,
            )?,
            candidate_same_subgroup_presence: optional_integer(
                diagnostic,
                "candidate_same_subgroup_presence",
                diagnostic_sources_csv,
            )?,
            quality_presence: optional_integer(
                diagnostic,
                "quality_presence",
                diagnostic_sources_csv,
            )?,
            quality_same_subgroup_presence: optional_integer(
                diagnostic,
                "quality_same_subgroup_presence",
                diagnostic_sources_csv,
            )?,
        };
        grouped
            .entry((
                row.feature_class.clone(),
                row.feature_class_label.clone(),
                row.route.clone(),
            ))
            .or_default()
            .push(row.clone());
        if row.temporal_source_covered && !row.diagnostic_subgroup.is_empty() {
            subgrouped
                .entry(subgroup_key(&row))
                .or_default()
                .push(row.clone());
        }
        source_rows.push(row);
    }

    let mut unknown_ids = diagnostic_by_id
        .keys()
        .filter(|id| !seen_route_ids.contains(*id))
        .cloned()
        .collect::<Vec<_>>();
    unknown_ids.sort();
    if !unknown_ids.is_empty() {
        return Err(format!(
            "{}: contains {} IDs absent from the route CSV",
            diagnostic_sources_csv.display(),
            unknown_ids.len()
        ));
    }

    let mut summary_rows = grouped
        .values()
        .map(|members| summarize_group(members, frame_count, required_presence))
        .collect::<Vec<_>>();
    summary_rows.sort_by(|a, b| {
        a.feature_class.cmp(&b.feature_class).then_with(|| {
            let a_rank = ROUTES
                .iter()
                .position(|route| *route == a.route)
                .unwrap_or(ROUTES.len());
            let b_rank = ROUTES
                .iter()
                .position(|route| *route == b.route)
                .unwrap_or(ROUTES.len());
            a_rank.cmp(&b_rank).then_with(|| a.route.cmp(&b.route))
        })
    });
    let mut subgroup_summary_rows = subgrouped
        .values()
        .map(|members| summarize_subgroup(members, frame_count, required_presence))
        .collect::<Vec<_>>();
    subgroup_summary_rows.sort_by(|a, b| {
        a.feature_class
            .cmp(&b.feature_class)
            .then_with(|| a.diagnostic_subgroup.cmp(&b.diagnostic_subgroup))
    });

    let source_by_id = source_rows
        .iter()
        .map(|row| (row.detection_id.as_str(), row))
        .collect::<HashMap<_, _>>();
    let targets: HashSet<&str> = normalized_targets.iter().map(String::as_str).collect();
    let mut missing_targets = targets
        .iter()
        .filter(|target| !source_by_id.contains_key(**target))
        .copied()
        .collect::<Vec<_>>();
    missing_targets.sort();
    if !missing_targets.is_empty() {
        return Err(format!(
            "{}: target IDs absent from the route CSV: {}",
            route_csv.display(),
            missing_targets.join(", ")
        ));
    }
    let target_rows = normalized_targets
        .iter()
        .map(|target_id| {
            let target = source_by_id[target_id.as_str()];
            let members =
                if target.temporal_source_covered && !target.diagnostic_subgroup.is_empty() {
                    subgrouped
                        .get(&subgroup_key(target))
                        .map(Vec::as_slice)
                        .unwrap_or_default()
                } else {
                    &[]
                };
            target_row(target, members, required_presence)
        })
        .collect();

    Ok(FeatureGateTemporalRouteResult {
        route_csv: route_csv.to_string_lossy().into_owned(),
        diagnostic_sources_csv: diagnostic_sources_csv.to_string_lossy().into_owned(),
        persistence_json: persistence_json.to_string_lossy().into_owned(),
        frame_count,
        required_presence,
        association_radius_px,
        association_method,
        source_rows,
        summary_rows,
        subgroup_summary_rows,
        target_rows,
    })
}

const SOURCE_CSV_FIELDS: [&str; 16] = [
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "route",
    "numeric_failure_reasons",
    "structural_flags",
    "flags",
    "temporal_source_covered",
    "diagnostic_subgroup",
    "diagnostic_subgroup_label",
    "diagnostic_subgroup_definition",
    "candidate_presence",
    "candidate_same_subgroup_presence",
    "quality_presence",
    "quality_same_subgroup_presence",
];
const SUMMARY_CSV_FIELDS: [&str; 24] = [
    "feature_class",
    "feature_class_label",
    "route",
    "route_count",
    "temporal_source_count",
    "temporal_coverage_fraction",
    "candidate_presence_median",
    "candidate_presence_mean",
    "candidate_presence_ge_required_count",
    "candidate_presence_ge_required_fraction",
    "candidate_presence_all_frames_count",
    "candidate_presence_all_frames_fraction",
    "candidate_same_subgroup_ge_required_count",
    "candidate_same_subgroup_ge_required_fraction",
    "quality_presence_median",
    "quality_presence_mean",
    "quality_presence_ge_required_count",
    "quality_presence_ge_required_fraction",
    "quality_presence_all_frames_count",
    "quality_presence_all_frames_fraction",
    "quality_presence_any_count",
    "quality_presence_any_fraction",
    "quality_same_subgroup_ge_required_count",
    "quality_same_subgroup_ge_required_fraction",
];
const SUBGROUP_CSV_FIELDS: [&str; 22] = [
    "feature_class",
    "feature_class_label",
    "diagnostic_subgroup",
    "diagnostic_subgroup_label",
    "diagnostic_subgroup_definition",
    "subgroup_count",
    "route_quality_passed_count",
    "route_rejected_numeric_only_count",
    "route_rejected_structural_only_count",
    "route_rejected_numeric_and_structural_count",
    "route_rejected_unexplained_count",
    "candidate_presence_median",
    "candidate_presence_ge_required_count",
    "candidate_presence_ge_required_fraction",
    "candidate_presence_all_frames_count",
    "candidate_same_subgroup_ge_required_count",
    "candidate_same_subgroup_ge_required_fraction",
    "mechanism_gap_fraction",
    "quality_presence_ge_required_count",
    "quality_presence_ge_required_fraction",
    "quality_same_subgroup_ge_required_count",
    "quality_same_subgroup_ge_required_fraction",
];
const TARGET_CSV_FIELDS: [&str; 24] = [
    "detection_id",
    "feature_class",
    "feature_class_label",
    "route",
    "temporal_source_covered",
    "diagnostic_subgroup",
    "diagnostic_subgroup_label",
    "diagnostic_subgroup_definition",
    "subgroup_count",
    "candidate_presence",
    "candidate_presence_le_fraction",
    "candidate_presence_ge_fraction",
    "candidate_same_subgroup_presence",
    "candidate_same_subgroup_presence_le_fraction",
    "candidate_same_subgroup_presence_ge_fraction",
    "quality_presence",
    "quality_presence_le_fraction",
    "quality_presence_ge_fraction",
    "candidate_presence_ge_required",
    "candidate_same_subgroup_ge_required",
    "quality_presence_ge_required",
    "candidate_and_same_ge_required",
    "candidate_same_and_quality_ge_required",
    "all_three_ge_required",
];

fn csv_cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn write_csv<T: Serialize>(
    output: &Path,
    name: &str,
    fields: &[&str],
    rows: &[T],
) -> Result<(), String> {
    let path = output.join(name);
    let mut file = File::create(&path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = WriterBuilder::new()
        .terminator(Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(fields)
        .map_err(|error| error.to_string())?;
    for row in rows {
        let value = serde_json::to_value(row).map_err(|error| error.to_string())?;
        let record = fields
            .iter()
            .map(|field| csv_cell(value.get(*field).unwrap_or(&Value::Null)))
            .collect::<Vec<_>>();
        writer
            .write_record(record)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    Ok(())
}

pub fn write_artifacts(
    result: &FeatureGateTemporalRouteResult,
    output_dir: &Path,
) -> Result<PathBuf, String> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    write_csv(
        output_dir,
        "feature_gate_temporal_route_sources.csv",
        &SOURCE_CSV_FIELDS,
        &result.source_rows,
    )?;
    write_csv(
        output_dir,
        "feature_gate_temporal_route_summary.csv",
        &SUMMARY_CSV_FIELDS,
        &result.summary_rows,
    )?;
    write_csv(
        output_dir,
        "feature_gate_temporal_subgroup_summary.csv",
        &SUBGROUP_CSV_FIELDS,
        &result.subgroup_summary_rows,
    )?;
    write_csv(
        output_dir,
        "feature_gate_temporal_targets.csv",
        &TARGET_CSV_FIELDS,
        &result.target_rows,
    )?;
    fs::write(
        output_dir.join("feature_gate_temporal_routes.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

fn option_value<'a>(
    args: &'a [String],
    index: &mut usize,
    name: &str,
    inline: Option<&'a str>,
) -> Result<&'a str, String> {
    if let Some(value) = inline {
        return Ok(value);
    }
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| format!("{name} requires a value"))
}

pub fn run_desktop(args: &[String]) -> Result<Value, String> {
    let mut positional = Vec::new();
    let mut targets = Vec::new();
    let mut out_dir = PathBuf::from("tmp/feature-gate-temporal-route");
    let mut index = 0;
    while index < args.len() {
        let (name, inline) = args[index]
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((args[index].as_str(), None));
        if name == "--out-dir" {
            out_dir = PathBuf::from(option_value(args, &mut index, name, inline)?);
        } else if name == "--target-id" {
            targets.push(option_value(args, &mut index, name, inline)?.to_owned());
        } else if name.starts_with('-') {
            return Err(format!("unknown argument {name}"));
        } else {
            positional.push(name.to_owned());
        }
        index += 1;
    }
    if positional.len() != 3 {
        return Err("expected route CSV, diagnostic sources CSV, and persistence JSON".to_owned());
    }
    let result = run(
        Path::new(&positional[0]),
        Path::new(&positional[1]),
        Path::new(&positional[2]),
        &targets,
    )?;
    let output = write_artifacts(&result, &out_dir)?;
    Ok(json!({
        "tool": "rst19-feature-gate-temporal-route",
        "out_dir": output,
        "result_path": output.join("feature_gate_temporal_routes.json"),
        "result": result.as_dict()
    }))
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("rst19-feature-gate-temporal-route ROUTE.csv DIAGNOSTICS.csv PERSISTENCE.json [--target-id ID] [--out-dir DIR]");
        return Ok(());
    }
    let result = run_desktop(args)?;
    println!(
        "feature gate temporal route: {} frames={} required={} radius={} subgroups={}",
        result["out_dir"].as_str().unwrap_or_default(),
        result["result"]["frame_count"],
        result["result"]["required_presence"],
        result["result"]["association_radius_px"],
        result["result"]["subgroup_summary_rows"]
            .as_array()
            .map(Vec::len)
            .unwrap_or_default()
    );
    for row in result["result"]["summary_rows"]
        .as_array()
        .into_iter()
        .flatten()
    {
        println!(
            "{}/{}: n={} temporal={} candidate>={}:{} quality>={}:{}",
            row["feature_class"].as_str().unwrap_or_default(),
            row["route"].as_str().unwrap_or_default(),
            row["route_count"],
            row["temporal_source_count"],
            result["result"]["required_presence"],
            display_value(&row["candidate_presence_ge_required_count"], "-"),
            result["result"]["required_presence"],
            display_value(&row["quality_presence_ge_required_count"], "-")
        );
    }
    for row in result["result"]["target_rows"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let subgroup = row["diagnostic_subgroup"].as_str().unwrap_or_default();
        let subgroup = if subgroup.is_empty() {
            "unavailable"
        } else {
            subgroup
        };
        println!(
            "target {}/{}: candidate={} same={} quality={} all_three={}",
            row["detection_id"].as_str().unwrap_or_default(),
            subgroup,
            display_value(&row["candidate_presence"], "-"),
            display_value(&row["candidate_same_subgroup_presence"], "-"),
            display_value(&row["quality_presence"], "-"),
            display_python_bool(&row["all_three_ge_required"])
        );
    }
    Ok(())
}

fn display_value(value: &Value, missing: &str) -> String {
    match value {
        Value::Null => missing.to_owned(),
        Value::String(value) => value.clone(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        value => value.to_string(),
    }
}

fn display_python_bool(value: &Value) -> String {
    match value {
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Null => "None".to_owned(),
        value => value.to_string(),
    }
}

pub fn handle_request(request: &Value) -> Result<Value, String> {
    let route = request["route_csv_path"]
        .as_str()
        .ok_or("missing route_csv_path")?;
    let diagnostic = request["diagnostic_sources_csv_path"]
        .as_str()
        .ok_or("missing diagnostic_sources_csv_path")?;
    let persistence = request["persistence_json_path"]
        .as_str()
        .ok_or("missing persistence_json_path")?;
    let targets: Vec<String> = request
        .get("target_ids")
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| "target_ids must be an array".to_owned())
                .map(|values| {
                    values
                        .iter()
                        .map(|value| value_text(Some(value)))
                        .collect::<Vec<_>>()
                })
        })
        .transpose()?
        .unwrap_or_default();
    let result = run(
        Path::new(route),
        Path::new(diagnostic),
        Path::new(persistence),
        &targets,
    )?;
    let mut response = json!({"result": result.as_dict()});
    if let Some(output_dir) = request.get("output_dir").and_then(Value::as_str) {
        let output = write_artifacts(&result, Path::new(output_dir))?;
        response["out_dir"] = json!(output);
        response["result_path"] = json!(output.join("feature_gate_temporal_routes.json"));
    }
    Ok(response)
}
