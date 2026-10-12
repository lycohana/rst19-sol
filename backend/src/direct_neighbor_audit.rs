//! Direct, non-transitive neighbor edges among detector candidates.
//!
//! The audit reports image-plane proximity and detector-level evidence only.
//! An independent-pair classification is an engineering candidate, not a
//! physical binary or a confirmed stellar identity.

use crate::{spatial::KdTree, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};

const REQUIRED_FIELDS: &[&str] = &[
    "detection_id",
    "x",
    "y",
    "peak_x",
    "peak_y",
    "quality_passed",
    "feature_class",
    "flags",
];
const OPTIONAL_FLOAT_FIELDS: &[&str] = &[
    "peak",
    "flux_snr",
    "filter_snr",
    "fwhm",
    "deblend_delta_bic",
    "deblend_component_snr",
];
const INDEPENDENCE_FLAGS: &[&str] = &[
    "CODE_PATTERN",
    "NEGATIVE_OVERFLOW",
    "MASKED",
    "SATURATED",
    "LINE_ARTIFACT",
    "UNRESOLVED_BLEND",
    "INSUFFICIENT_PSF_SUPPORT",
    "NARROW",
    "BROAD",
    "ELONGATED",
    "DIFFUSE",
];
const PAIR_FIELDS: &[&str] = &[
    "detection_id_a",
    "detection_id_b",
    "centroid_distance_px",
    "peak_distance_px",
    "feature_class_a",
    "feature_class_b",
    "same_feature_class",
    "quality_passed_a",
    "quality_passed_b",
    "both_quality_passed",
    "flags_a",
    "flags_b",
    "independent_psf_evidence",
    "classification",
    "reason",
    "is_target_pair",
];
const SUMMARY_FIELDS: &[&str] = &[
    "feature_class_a",
    "feature_class_b",
    "pair_count",
    "cross_class_count",
    "both_quality_count",
    "unresolved_pair_count",
    "independent_pair_candidate_count",
    "centroid_distance_median_px",
    "peak_distance_median_px",
];
const TARGET_FIELDS: &[&str] = &[
    "detection_id_a",
    "detection_id_b",
    "centroid_distance_px",
    "peak_distance_px",
    "feature_class_a",
    "feature_class_b",
    "quality_passed_a",
    "quality_passed_b",
    "flags_a",
    "flags_b",
    "independent_psf_evidence",
    "in_pair_radius",
    "classification",
    "reason",
];

#[derive(Debug, Clone)]
struct Source {
    detection_id: i64,
    x: f64,
    y: f64,
    peak_x: f64,
    peak_y: f64,
    quality_passed: bool,
    feature_class: String,
    flags: Vec<String>,
    deblend_delta_bic: Option<f64>,
    deblend_component_snr: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DirectNeighborPairRow {
    pub detection_id_a: i64,
    pub detection_id_b: i64,
    pub centroid_distance_px: f64,
    pub peak_distance_px: f64,
    pub feature_class_a: String,
    pub feature_class_b: String,
    pub same_feature_class: bool,
    pub quality_passed_a: bool,
    pub quality_passed_b: bool,
    pub both_quality_passed: bool,
    pub flags_a: String,
    pub flags_b: String,
    pub independent_psf_evidence: bool,
    pub classification: String,
    pub reason: String,
    pub is_target_pair: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DirectNeighborSummaryRow {
    pub feature_class_a: String,
    pub feature_class_b: String,
    pub pair_count: usize,
    pub cross_class_count: usize,
    pub both_quality_count: usize,
    pub unresolved_pair_count: usize,
    pub independent_pair_candidate_count: usize,
    pub centroid_distance_median_px: Option<f64>,
    pub peak_distance_median_px: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DirectNeighborTargetRow {
    pub detection_id_a: i64,
    pub detection_id_b: i64,
    pub centroid_distance_px: f64,
    pub peak_distance_px: f64,
    pub feature_class_a: String,
    pub feature_class_b: String,
    pub quality_passed_a: bool,
    pub quality_passed_b: bool,
    pub flags_a: String,
    pub flags_b: String,
    pub independent_psf_evidence: bool,
    pub in_pair_radius: bool,
    pub classification: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct DirectNeighborAuditResult {
    catalog_path: String,
    source_count: usize,
    quality_count: usize,
    pair_radius_px: f64,
    direct_pair_count: usize,
    unresolved_pair_count: usize,
    independent_pair_candidate_count: usize,
    target_ids: Vec<i64>,
    pairs: Vec<DirectNeighborPairRow>,
    summaries: Vec<DirectNeighborSummaryRow>,
    target: Option<DirectNeighborTargetRow>,
    parameters: Value,
    conclusion: String,
}

impl DirectNeighborAuditResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "source_count": self.source_count,
            "quality_count": self.quality_count,
            "pair_radius_px": self.pair_radius_px,
            "direct_pair_count": self.direct_pair_count,
            "unresolved_pair_count": self.unresolved_pair_count,
            "independent_pair_candidate_count": self.independent_pair_candidate_count,
            "target_ids": self.target_ids,
            "pairs": self.pairs,
            "summaries": self.summaries,
            "target": self.target,
            "parameters": self.parameters,
            "conclusion": self.conclusion,
            "interpretation_guardrails": [
                "直接近邻边不包含传递连通关系；边数不能与父源组数相加。",
                "feature_class 和 flags 是 detector-level 审计标签，不是物理天体类别。",
                "independent_pair_candidate 仍需要真实 PSF、注入、星表/WCS 或人工真值确认。",
                "直接近邻距离、候选数和质量数都不是双星概率、precision、FDR 或物理星数。",
            ],
        })
    }
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

fn optional_csv(value: Option<f64>) -> String {
    value.map(py_float).unwrap_or_default()
}

fn py_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn format_general(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let mut mantissa = mantissa
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned();
    if !(-4..6).contains(&exponent) {
        return format!(
            "{mantissa}e{}{:02}",
            if exponent >= 0 { "+" } else { "-" },
            exponent.abs()
        );
    }
    let decimals = (5 - exponent).max(0) as usize;
    mantissa = format!("{:.*}", decimals, value);
    mantissa
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

fn comma_group(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn parse_bool(raw: &str) -> std::result::Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "y" => Ok(true),
        "false" | "0" | "no" | "n" => Ok(false),
        _ => Err(format!(
            "catalog field 'quality_passed' is not boolean: {}",
            python_repr(raw)
        )),
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

fn optional_float(raw: &str, field: &str, row: usize) -> std::result::Result<Option<f64>, String> {
    let value = raw.trim();
    if value.is_empty()
        || ["none", "nan", "null", "—", "-"].contains(&value.to_ascii_lowercase().as_str())
    {
        return Ok(None);
    }
    let number = value.parse::<f64>().map_err(|_| {
        format!(
            "catalog row {row}: {field} must be numeric, got {}",
            python_repr(value)
        )
    })?;
    if !number.is_finite() {
        return Err(format!("catalog row {row}: {field} must be finite"));
    }
    Ok(Some(number))
}

fn required_float(raw: &str, field: &str, row: usize) -> std::result::Result<f64, String> {
    optional_float(raw, field, row)?.ok_or_else(|| format!("catalog row {row}: missing {field}"))
}

fn field<'a>(
    record: &'a csv::StringRecord,
    indexes: &BTreeMap<String, usize>,
    name: &str,
) -> &'a str {
    indexes
        .get(name)
        .and_then(|index| record.get(*index))
        .unwrap_or("")
}

fn read_sources(path: &Path) -> Result<Vec<Source>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|error| error.to_string())?;
    let raw_headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let headers: Vec<String> = raw_headers
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
    let mut indexes = BTreeMap::new();
    for (index, name) in headers.iter().enumerate() {
        indexes.insert(name.clone(), index);
    }
    let present: BTreeSet<_> = headers.iter().cloned().collect();
    let missing: Vec<_> = REQUIRED_FIELDS
        .iter()
        .filter(|field| !present.contains(**field))
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing fields: {}",
            missing.join(", ")
        ));
    }
    let mut sources = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, record) in reader.records().enumerate() {
        let row = index + 2;
        let record = record.map_err(|error| error.to_string())?;
        let id_raw = field(&record, &indexes, "detection_id").trim();
        let detection_id = id_raw
            .parse::<i64>()
            .map_err(|_| format!("catalog row {row}: detection_id must be an integer"))?;
        if !seen.insert(detection_id) {
            return Err(format!(
                "catalog row {row}: duplicate detection_id {detection_id}"
            ));
        }
        let feature_class = field(&record, &indexes, "feature_class").trim().to_owned();
        if feature_class.is_empty() {
            return Err(format!("catalog row {row}: feature_class is empty"));
        }
        let x = required_float(field(&record, &indexes, "x"), "x", row)?;
        let y = required_float(field(&record, &indexes, "y"), "y", row)?;
        let peak_x = required_float(field(&record, &indexes, "peak_x"), "peak_x", row)?;
        let peak_y = required_float(field(&record, &indexes, "peak_y"), "peak_y", row)?;
        let quality_passed = parse_bool(field(&record, &indexes, "quality_passed"))?;
        let flags = field(&record, &indexes, "flags")
            .split(['|', ',', ';'])
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect();
        let mut optional = BTreeMap::new();
        for field in OPTIONAL_FLOAT_FIELDS {
            optional.insert(
                (*field).to_owned(),
                optional_float(self::field(&record, &indexes, field), field, row)?,
            );
        }
        sources.push(Source {
            detection_id,
            x,
            y,
            peak_x,
            peak_y,
            quality_passed,
            feature_class,
            flags,
            deblend_delta_bic: optional["deblend_delta_bic"],
            deblend_component_snr: optional["deblend_component_snr"],
        });
    }
    if sources.is_empty() {
        return Err("source catalog contains no sources".into());
    }
    Ok(sources)
}

fn validate_radius(psf_fwhm: f64, pair_radius: Option<f64>) -> Result<(f64, f64)> {
    if !psf_fwhm.is_finite() || psf_fwhm <= 0.0 {
        return Err("psf_fwhm must be finite and positive".into());
    }
    let radius = pair_radius.unwrap_or_else(|| 2.5_f64.max(1.5 * psf_fwhm));
    if !radius.is_finite() || radius <= 0.0 {
        return Err("pair_radius_px must be finite and positive".into());
    }
    Ok((psf_fwhm, radius))
}

fn validate_targets(target_ids: &[i64]) -> Result<Vec<i64>> {
    let mut values = Vec::new();
    for value in target_ids {
        if !values.contains(value) {
            values.push(*value);
        }
    }
    if !values.is_empty() && values.len() != 2 {
        return Err("target_ids must contain either zero or two distinct detection IDs".into());
    }
    Ok(values)
}

fn psf_evidence(source: &Source, delta_bic_min: f64, component_snr_min: f64) -> bool {
    source
        .deblend_delta_bic
        .zip(source.deblend_component_snr)
        .is_some_and(|(delta, snr)| delta >= delta_bic_min && snr >= component_snr_min)
}

fn pair_fields(
    first: &Source,
    second: &Source,
    target_ids: &[i64],
    delta_bic_min: f64,
    component_snr_min: f64,
) -> DirectNeighborPairRow {
    let (first, second) = if first.detection_id <= second.detection_id {
        (first, second)
    } else {
        (second, first)
    };
    let centroid_distance = (first.x - second.x).hypot(first.y - second.y);
    let peak_distance = (first.peak_x - second.peak_x).hypot(first.peak_y - second.peak_y);
    let flags: BTreeSet<_> = first.flags.iter().chain(&second.flags).collect();
    let blocking_flags: Vec<_> = flags
        .iter()
        .filter(|flag| INDEPENDENCE_FLAGS.contains(&flag.as_str()))
        .map(|flag| (*flag).clone())
        .collect();
    let both_quality = first.quality_passed && second.quality_passed;
    let has_psf = psf_evidence(first, delta_bic_min, component_snr_min)
        || psf_evidence(second, delta_bic_min, component_snr_min);
    let independent = both_quality && blocking_flags.is_empty() && has_psf;
    let (classification, reason) = if independent {
        (
            "independent_pair_candidate".to_owned(),
            "两端质量、结构旗标和当前双PSF工程线满足；仍需真值确认".to_owned(),
        )
    } else {
        let mut reasons = vec!["两端为直接近邻边".to_owned()];
        if !both_quality {
            reasons.push("至少一端未通过质量层".to_owned());
        }
        if !blocking_flags.is_empty() {
            reasons.push(format!("含结构/值域旗标 {}", blocking_flags.join(",")));
        }
        if !has_psf {
            reasons.push("没有达到双PSF独立证据线".to_owned());
        }
        ("unresolved_pair".into(), reasons.join("；"))
    };
    DirectNeighborPairRow {
        detection_id_a: first.detection_id,
        detection_id_b: second.detection_id,
        centroid_distance_px: centroid_distance,
        peak_distance_px: peak_distance,
        feature_class_a: first.feature_class.clone(),
        feature_class_b: second.feature_class.clone(),
        same_feature_class: first.feature_class == second.feature_class,
        quality_passed_a: first.quality_passed,
        quality_passed_b: second.quality_passed,
        both_quality_passed: both_quality,
        flags_a: first.flags.join("|"),
        flags_b: second.flags.join("|"),
        independent_psf_evidence: independent,
        classification,
        reason,
        is_target_pair: target_ids.len() == 2
            && ((target_ids[0] == first.detection_id && target_ids[1] == second.detection_id)
                || (target_ids[0] == second.detection_id && target_ids[1] == first.detection_id)),
    }
}

fn target_row(pair: &DirectNeighborPairRow, in_radius: bool) -> DirectNeighborTargetRow {
    let (classification, reason) = if in_radius {
        (pair.classification.clone(), pair.reason.clone())
    } else {
        (
            "outside_pair_radius".into(),
            format!(
                "直接质心距离 {:.3}px 超出当前半径",
                pair.centroid_distance_px
            ),
        )
    };
    DirectNeighborTargetRow {
        detection_id_a: pair.detection_id_a,
        detection_id_b: pair.detection_id_b,
        centroid_distance_px: pair.centroid_distance_px,
        peak_distance_px: pair.peak_distance_px,
        feature_class_a: pair.feature_class_a.clone(),
        feature_class_b: pair.feature_class_b.clone(),
        quality_passed_a: pair.quality_passed_a,
        quality_passed_b: pair.quality_passed_b,
        flags_a: pair.flags_a.clone(),
        flags_b: pair.flags_b.clone(),
        independent_psf_evidence: pair.independent_psf_evidence,
        in_pair_radius: in_radius,
        classification,
        reason,
    }
}

fn summarize(pairs: &[DirectNeighborPairRow]) -> Vec<DirectNeighborSummaryRow> {
    let mut groups: BTreeMap<(String, String), Vec<&DirectNeighborPairRow>> = BTreeMap::new();
    for pair in pairs {
        let (first, second) = if pair.feature_class_a <= pair.feature_class_b {
            (pair.feature_class_a.clone(), pair.feature_class_b.clone())
        } else {
            (pair.feature_class_b.clone(), pair.feature_class_a.clone())
        };
        groups.entry((first, second)).or_default().push(pair);
    }
    groups
        .into_iter()
        .map(|((feature_class_a, feature_class_b), members)| {
            let mut centroid: Vec<_> = members
                .iter()
                .map(|pair| pair.centroid_distance_px)
                .collect();
            let mut peak: Vec<_> = members.iter().map(|pair| pair.peak_distance_px).collect();
            DirectNeighborSummaryRow {
                feature_class_a,
                feature_class_b,
                pair_count: members.len(),
                cross_class_count: members
                    .iter()
                    .filter(|pair| !pair.same_feature_class)
                    .count(),
                both_quality_count: members
                    .iter()
                    .filter(|pair| pair.both_quality_passed)
                    .count(),
                unresolved_pair_count: members
                    .iter()
                    .filter(|pair| pair.classification == "unresolved_pair")
                    .count(),
                independent_pair_candidate_count: members
                    .iter()
                    .filter(|pair| pair.classification == "independent_pair_candidate")
                    .count(),
                centroid_distance_median_px: median(&mut centroid),
                peak_distance_median_px: median(&mut peak),
            }
        })
        .collect()
}

pub fn run(
    catalog_csv: &Path,
    target_ids: &[i64],
    psf_fwhm: f64,
    pair_radius_px: Option<f64>,
    delta_bic_min: f64,
    component_snr_min: f64,
) -> Result<DirectNeighborAuditResult> {
    let (psf_fwhm, pair_radius_px) = validate_radius(psf_fwhm, pair_radius_px)?;
    if !delta_bic_min.is_finite() || delta_bic_min <= 0.0 {
        return Err("delta_bic_min must be finite and positive".into());
    }
    if !component_snr_min.is_finite() || component_snr_min <= 0.0 {
        return Err("component_snr_min must be finite and positive".into());
    }
    let target_ids = validate_targets(target_ids)?;
    let sources = read_sources(catalog_csv)?;
    let source_by_id: BTreeMap<_, _> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| (source.detection_id, index))
        .collect();
    let missing: Vec<_> = target_ids
        .iter()
        .filter(|target| !source_by_id.contains_key(target))
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "target detection_id values are not in source catalog: {}",
            missing
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let points: Vec<[f64; 2]> = sources.iter().map(|source| [source.x, source.y]).collect();
    let tree = KdTree::new(&points)?;
    let mut pairs = Vec::new();
    for (index, point) in points.iter().enumerate() {
        for (neighbor, _) in tree.within_radius(*point, pair_radius_px)? {
            if neighbor > index {
                pairs.push(pair_fields(
                    &sources[index],
                    &sources[neighbor],
                    &target_ids,
                    delta_bic_min,
                    component_snr_min,
                ));
            }
        }
    }
    pairs.sort_by_key(|pair| (pair.detection_id_a, pair.detection_id_b));
    let target = if target_ids.is_empty() {
        None
    } else {
        let first = &sources[source_by_id[&target_ids[0]]];
        let second = &sources[source_by_id[&target_ids[1]]];
        let direct = pair_fields(first, second, &target_ids, delta_bic_min, component_snr_min);
        Some(target_row(
            &direct,
            direct.centroid_distance_px <= pair_radius_px,
        ))
    };
    let unresolved_pair_count = pairs
        .iter()
        .filter(|pair| pair.classification == "unresolved_pair")
        .count();
    let independent_pair_candidate_count = pairs
        .iter()
        .filter(|pair| pair.classification == "independent_pair_candidate")
        .count();
    let conclusion = format!(
        "在 {}px 质心半径下保留 {} 条直接近邻边，其中未分辨边 {} 条、独立候选边 {} 条；直接边不经过传递合并，不能与父源组数或物理恒星数互换。",
        format_general(pair_radius_px),
        comma_group(pairs.len()),
        comma_group(unresolved_pair_count),
        comma_group(independent_pair_candidate_count)
    );
    let summaries = summarize(&pairs);
    Ok(DirectNeighborAuditResult {
        catalog_path: catalog_csv.to_string_lossy().into_owned(),
        source_count: sources.len(),
        quality_count: sources
            .iter()
            .filter(|source| source.quality_passed)
            .count(),
        pair_radius_px,
        direct_pair_count: pairs.len(),
        unresolved_pair_count,
        independent_pair_candidate_count,
        target_ids,
        pairs,
        summaries,
        target,
        parameters: json!({
            "psf_fwhm": psf_fwhm,
            "pair_radius_px": pair_radius_px,
            "pair_radius_definition": "max(2.5 px, 1.5 * psf_fwhm) unless explicitly overridden",
            "delta_bic_min": delta_bic_min,
            "component_snr_min": component_snr_min,
            "edge_definition": "centroid distance <= pair_radius_px; no transitive union",
            "interpretation_boundary": "detector-level direct-neighbor audit; no pair truth or star probability",
        }),
        conclusion,
    })
}

fn csv_record_pair(row: &DirectNeighborPairRow) -> Vec<String> {
    vec![
        row.detection_id_a.to_string(),
        row.detection_id_b.to_string(),
        py_float(row.centroid_distance_px),
        py_float(row.peak_distance_px),
        row.feature_class_a.clone(),
        row.feature_class_b.clone(),
        py_bool(row.same_feature_class).to_owned(),
        py_bool(row.quality_passed_a).to_owned(),
        py_bool(row.quality_passed_b).to_owned(),
        py_bool(row.both_quality_passed).to_owned(),
        row.flags_a.clone(),
        row.flags_b.clone(),
        py_bool(row.independent_psf_evidence).to_owned(),
        row.classification.clone(),
        row.reason.clone(),
        py_bool(row.is_target_pair).to_owned(),
    ]
}

fn csv_record_summary(row: &DirectNeighborSummaryRow) -> Vec<String> {
    vec![
        row.feature_class_a.clone(),
        row.feature_class_b.clone(),
        row.pair_count.to_string(),
        row.cross_class_count.to_string(),
        row.both_quality_count.to_string(),
        row.unresolved_pair_count.to_string(),
        row.independent_pair_candidate_count.to_string(),
        optional_csv(row.centroid_distance_median_px),
        optional_csv(row.peak_distance_median_px),
    ]
}

fn csv_record_target(row: &DirectNeighborTargetRow) -> Vec<String> {
    vec![
        row.detection_id_a.to_string(),
        row.detection_id_b.to_string(),
        py_float(row.centroid_distance_px),
        py_float(row.peak_distance_px),
        row.feature_class_a.clone(),
        row.feature_class_b.clone(),
        py_bool(row.quality_passed_a).to_owned(),
        py_bool(row.quality_passed_b).to_owned(),
        row.flags_a.clone(),
        row.flags_b.clone(),
        py_bool(row.independent_psf_evidence).to_owned(),
        py_bool(row.in_pair_radius).to_owned(),
        row.classification.clone(),
        row.reason.clone(),
    ]
}

fn write_csv(path: &Path, headers: &[&str], records: Vec<Vec<String>>) -> Result<()> {
    let mut file = fs::File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for record in records {
        writer
            .write_record(record)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn write_empty_bom(path: &Path) -> Result<()> {
    fs::write(path, [0xef, 0xbb, 0xbf]).map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &DirectNeighborAuditResult, output_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    write_csv(
        &output_dir.join("direct_neighbor_pairs.csv"),
        PAIR_FIELDS,
        result.pairs.iter().map(csv_record_pair).collect(),
    )?;
    write_csv(
        &output_dir.join("direct_neighbor_summaries.csv"),
        SUMMARY_FIELDS,
        result.summaries.iter().map(csv_record_summary).collect(),
    )?;
    if let Some(target) = &result.target {
        write_csv(
            &output_dir.join("direct_neighbor_targets.csv"),
            TARGET_FIELDS,
            vec![csv_record_target(target)],
        )?;
    } else {
        write_empty_bom(&output_dir.join("direct_neighbor_targets.csv"))?;
    }
    fs::write(
        output_dir.join("direct_neighbor_audit.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut catalog = None;
    let mut target_ids = Vec::new();
    let mut psf_fwhm = 2.0;
    let mut pair_radius_px = None;
    let mut delta_bic_min = 10.0;
    let mut component_snr_min = 5.0;
    let mut output_dir = PathBuf::from("tmp/direct-neighbor-audit");
    let mut index = 0;
    while index < args.len() {
        let (name, inline_value) = args[index]
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((args[index].as_str(), None));
        match name {
            "--target-id"
            | "--psf-fwhm"
            | "--pair-radius"
            | "--delta-bic-min"
            | "--component-snr-min"
            | "--out-dir" => {
                let value = if let Some(value) = inline_value {
                    value
                } else {
                    index += 1;
                    args.get(index)
                        .ok_or_else(|| format!("{name} requires a value"))?
                };
                match name {
                    "--target-id" => target_ids.push(
                        value
                            .parse::<i64>()
                            .map_err(|error| format!("invalid target detection ID: {error}"))?,
                    ),
                    "--psf-fwhm" => psf_fwhm = value.parse::<f64>().map_err(|e| e.to_string())?,
                    "--pair-radius" => {
                        pair_radius_px = Some(value.parse::<f64>().map_err(|e| e.to_string())?)
                    }
                    "--delta-bic-min" => {
                        delta_bic_min = value.parse::<f64>().map_err(|e| e.to_string())?
                    }
                    "--component-snr-min" => {
                        component_snr_min = value.parse::<f64>().map_err(|e| e.to_string())?
                    }
                    _ => output_dir = PathBuf::from(value),
                }
            }
            _ if name.starts_with('-') => return Err(format!("unknown argument {name}")),
            _ => {
                if catalog.replace(PathBuf::from(name)).is_some() {
                    return Err("only one source catalog is allowed".into());
                }
            }
        }
        index += 1;
    }
    let catalog = catalog.ok_or("a source catalog CSV is required")?;
    let result = run(
        &catalog,
        &target_ids,
        psf_fwhm,
        pair_radius_px,
        delta_bic_min,
        component_snr_min,
    )?;
    write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-direct-neighbor-audit",
        "out_dir": output_dir,
        "result_path": output_dir.join("direct_neighbor_audit.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "rst19-direct-neighbor-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn direct_edges_classification_and_target_match_python_contract() {
        let root = temp_root();
        let catalog = root.join("catalog.csv");
        fs::write(
            &catalog,
            "\u{feff}detection_id,x,y,peak_x,peak_y,peak,flux_snr,filter_snr,fwhm,deblend_delta_bic,deblend_component_snr,quality_passed,feature_class,flags\r\n1,0,0,0,0,3992,204.6,1195,2.59,-4.09,0.87,False,range_anomaly,CODE_PATTERN\r\n2,2.738,0,4,0,569,115.5,300,2.44,-4.09,0.87,False,crowded_blend,UNRESOLVED_BLEND\r\n3,10,0,10,0,250,12,20,2,,,True,compact_quality,\r\n4,20,0,20,0,900,20,100,2,12,6,True,compact_quality,\r\n5,23,0,23,0,400,10,60,2,,,True,compact_quality,\r\n",
        )
        .unwrap();
        let result = run(&catalog, &[1, 2], 2.0, Some(4.0), 10.0, 5.0).unwrap();
        assert_eq!(result.source_count, 5);
        assert_eq!(result.direct_pair_count, 2);
        assert_eq!(result.unresolved_pair_count, 1);
        assert_eq!(result.independent_pair_candidate_count, 1);
        assert_eq!(
            result
                .pairs
                .iter()
                .map(|pair| (pair.detection_id_a, pair.detection_id_b))
                .collect::<Vec<_>>(),
            [(1, 2), (4, 5)]
        );
        assert_eq!(
            result.target.as_ref().unwrap().classification,
            "unresolved_pair"
        );
        assert!((result.target.as_ref().unwrap().centroid_distance_px - 2.738).abs() < 1e-12);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn target_outside_radius_keeps_distance_and_is_not_a_direct_edge() {
        let root = temp_root();
        let catalog = root.join("catalog.csv");
        fs::write(
            &catalog,
            "detection_id,x,y,peak_x,peak_y,quality_passed,feature_class,flags\n1,0,0,0,0,True,a,\n2,3,4,3,4,True,b,\n",
        )
        .unwrap();
        let result = run(&catalog, &[2, 1], 2.0, Some(2.0), 10.0, 5.0).unwrap();
        assert_eq!(result.direct_pair_count, 0);
        let target = result.target.unwrap();
        assert!(!target.in_pair_radius);
        assert_eq!(target.classification, "outside_pair_radius");
        assert_eq!(target.detection_id_a, 1);
        assert_eq!(target.detection_id_b, 2);
        assert_eq!(target.centroid_distance_px, 5.0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn validates_targets_and_duplicates_without_mutating_input() {
        let root = temp_root();
        let catalog = root.join("catalog.csv");
        fs::write(
            &catalog,
            "detection_id,x,y,peak_x,peak_y,quality_passed,feature_class,flags\n1,0,0,0,0,True,a,\n2,1,0,1,0,True,b,\n",
        )
        .unwrap();
        let before = fs::read(&catalog).unwrap();
        assert_eq!(
            run(&catalog, &[1, 1], 2.0, None, 10.0, 5.0).unwrap_err(),
            "target_ids must contain either zero or two distinct detection IDs"
        );
        assert_eq!(
            run(&catalog, &[1, 9], 2.0, None, 10.0, 5.0).unwrap_err(),
            "target detection_id values are not in source catalog: 9"
        );
        assert_eq!(fs::read(&catalog).unwrap(), before);
        let _ = fs::remove_dir_all(root);
    }
}
