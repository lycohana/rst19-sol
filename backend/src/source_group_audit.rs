//! Detector-level grouping of nearby source-catalog candidates.
//!
//! Connected components organize nearby detections for review. Group labels do
//! not claim physical star identity, and the representative is only for display.

use crate::{spatial::KdTree, Result};
use csv::StringRecord;
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
const BLOCKING_FLAGS: &[&str] = &[
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

#[derive(Debug, Clone)]
pub struct CandidateGroupSource {
    pub detection_id: i64,
    pub x: f64,
    pub y: f64,
    pub peak_x: f64,
    pub peak_y: f64,
    pub quality_passed: bool,
    pub feature_class: String,
    pub flags: Vec<String>,
    pub peak: Option<f64>,
    pub flux_snr: Option<f64>,
    pub filter_snr: Option<f64>,
    pub fwhm: Option<f64>,
    pub deblend_delta_bic: Option<f64>,
    pub deblend_component_snr: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateGroupRow {
    pub group_id: usize,
    pub detection_ids: String,
    pub member_count: usize,
    pub representative_detection_id: i64,
    pub classification: String,
    pub quality_member_count: usize,
    pub centroid_span_px: f64,
    pub peak_span_px: f64,
    pub feature_classes: String,
    pub flags: String,
    pub independent_psf_evidence: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateGroupClassRow {
    pub feature_class: String,
    pub source_count: usize,
    pub quality_source_count: usize,
    pub multi_group_source_count: usize,
    pub multi_group_source_fraction: f64,
    pub unresolved_group_source_count: usize,
    pub independent_group_source_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateGroupTargetRow {
    pub group_id: usize,
    pub target_detection_ids: String,
    pub group_detection_ids: String,
    pub classification: String,
    pub representative_detection_id: i64,
    pub member_count: usize,
    pub centroid_span_px: f64,
    pub peak_span_px: f64,
    pub feature_classes: String,
    pub flags: String,
    pub independent_psf_evidence: bool,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct CandidateGroupAuditResult {
    pub catalog_path: String,
    pub source_count: usize,
    pub quality_count: usize,
    pub group_count: usize,
    pub multi_member_group_count: usize,
    pub unresolved_group_count: usize,
    pub independent_group_candidate_count: usize,
    pub groups: Vec<CandidateGroupRow>,
    pub classes: Vec<CandidateGroupClassRow>,
    pub targets: Vec<CandidateGroupTargetRow>,
    pub parameters: Value,
    pub conclusion: String,
}

impl CandidateGroupAuditResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "source_count": self.source_count,
            "quality_count": self.quality_count,
            "group_count": self.group_count,
            "multi_member_group_count": self.multi_member_group_count,
            "unresolved_group_count": self.unresolved_group_count,
            "independent_group_candidate_count": self.independent_group_candidate_count,
            "groups": self.groups,
            "classes": self.classes,
            "targets": self.targets,
            "parameters": self.parameters,
            "conclusion": self.conclusion,
            "interpretation_guardrails": [
                "源组是 detector-level 近邻组织，不是物理恒星分组。",
                "isolated 只表示没有进入当前组半径，不表示已完成星表身份确认。",
                "independent_group_candidate 仍是候选，不能直接写成双星或多星。",
                "代表源只用于显示/排序，不覆盖组内子候选和质量拒绝原因。",
            ],
        })
    }
}

fn repr(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn field(record: &StringRecord, headers: &StringRecord, name: &str) -> String {
    headers
        .iter()
        .position(|header| header == name)
        .and_then(|index| record.get(index))
        .unwrap_or_default()
        .to_owned()
}

fn optional_float(raw: &str, name: &str, row_number: usize) -> Result<Option<f64>> {
    let raw = raw.trim();
    if raw.is_empty()
        || ["none", "nan", "null", "—", "-"].contains(&raw.to_ascii_lowercase().as_str())
    {
        return Ok(None);
    }
    let value = raw.parse::<f64>().map_err(|_| {
        format!(
            "catalog row {row_number}: {name} must be numeric, got {}",
            repr(raw)
        )
    })?;
    if !value.is_finite() {
        return Err(format!("catalog row {row_number}: {name} must be finite"));
    }
    Ok(Some(value))
}

fn required_float(raw: &str, name: &str, row_number: usize) -> Result<f64> {
    optional_float(raw, name, row_number)?
        .ok_or_else(|| format!("catalog row {row_number}: missing {name}"))
}

fn parse_bool(raw: &str, field_name: &str) -> Result<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "y" => Ok(true),
        "false" | "0" | "no" | "n" => Ok(false),
        _ => Err(format!(
            "catalog field {} is not boolean: {}",
            repr(field_name),
            repr(raw)
        )),
    }
}

fn parse_flags(value: &str) -> Vec<String> {
    value
        .split(['|', ',', ';'])
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
        .collect()
}

pub fn load_candidate_group_sources(path: &Path) -> Result<Vec<CandidateGroupSource>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|error| error.to_string())?;
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let names: BTreeSet<_> = headers
        .iter()
        .enumerate()
        .map(|(index, name)| {
            if index == 0 {
                name.trim_start_matches('\u{feff}')
            } else {
                name
            }
        })
        .collect();
    // csv::Reader strips a leading BOM from the first field on supported builds;
    // use the same normalized names for subsequent column lookups.
    let normalized_headers: StringRecord = headers
        .iter()
        .enumerate()
        .map(|(index, name)| {
            if index == 0 {
                name.trim_start_matches('\u{feff}').to_owned()
            } else {
                name.to_owned()
            }
        })
        .collect();
    let missing: Vec<_> = REQUIRED_FIELDS
        .iter()
        .filter(|name| !names.contains(**name))
        .copied()
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
        let row_number = index + 2;
        let record = record.map_err(|error| error.to_string())?;
        let detection_id_raw = field(&record, &normalized_headers, "detection_id");
        let detection_id = detection_id_raw
            .trim()
            .parse::<i64>()
            .map_err(|_| format!("catalog row {row_number}: detection_id must be an integer"))?;
        if !seen.insert(detection_id) {
            return Err(format!(
                "catalog row {row_number}: duplicate detection_id {detection_id}"
            ));
        }
        let feature_class = field(&record, &normalized_headers, "feature_class")
            .trim()
            .to_owned();
        if feature_class.is_empty() {
            return Err(format!("catalog row {row_number}: feature_class is empty"));
        }
        let quality_raw = field(&record, &normalized_headers, "quality_passed");
        let flags_raw = field(&record, &normalized_headers, "flags");
        let optional = |name: &str| -> Result<Option<f64>> {
            optional_float(&field(&record, &normalized_headers, name), name, row_number)
        };
        sources.push(CandidateGroupSource {
            detection_id,
            x: required_float(&field(&record, &normalized_headers, "x"), "x", row_number)?,
            y: required_float(&field(&record, &normalized_headers, "y"), "y", row_number)?,
            peak_x: required_float(
                &field(&record, &normalized_headers, "peak_x"),
                "peak_x",
                row_number,
            )?,
            peak_y: required_float(
                &field(&record, &normalized_headers, "peak_y"),
                "peak_y",
                row_number,
            )?,
            quality_passed: parse_bool(&quality_raw, "quality_passed")?,
            feature_class,
            flags: parse_flags(&flags_raw),
            peak: optional("peak")?,
            flux_snr: optional("flux_snr")?,
            filter_snr: optional("filter_snr")?,
            fwhm: optional("fwhm")?,
            deblend_delta_bic: optional("deblend_delta_bic")?,
            deblend_component_snr: optional("deblend_component_snr")?,
        });
    }
    if sources.is_empty() {
        return Err("source catalog contains no sources".into());
    }
    Ok(sources)
}

fn validate_parameters(
    psf_fwhm: f64,
    group_radius_px: Option<f64>,
    delta_bic_min: f64,
    component_snr_min: f64,
) -> Result<(f64, f64, f64, f64)> {
    for (name, value) in [
        ("psf_fwhm", psf_fwhm),
        ("delta_bic_min", delta_bic_min),
        ("component_snr_min", component_snr_min),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{name} must be finite and positive"));
        }
    }
    let radius = group_radius_px.unwrap_or((1.5 * psf_fwhm).max(2.5));
    if !radius.is_finite() || radius <= 0.0 {
        return Err("group_radius_px must be finite and positive".into());
    }
    Ok((psf_fwhm, radius, delta_bic_min, component_snr_min))
}

fn connected_components(size: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut parent: Vec<_> = (0..size).collect();
    let mut rank = vec![0u8; size];
    fn find(parent: &mut [usize], node: usize) -> usize {
        if parent[node] != node {
            parent[node] = find(parent, parent[node]);
        }
        parent[node]
    }
    for &(left, right) in edges {
        let mut left_root = find(&mut parent, left);
        let mut right_root = find(&mut parent, right);
        if left_root == right_root {
            continue;
        }
        if rank[left_root] < rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        parent[right_root] = left_root;
        if rank[left_root] == rank[right_root] {
            rank[left_root] += 1;
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for index in 0..size {
        let root = find(&mut parent, index);
        groups.entry(root).or_default().push(index);
    }
    groups.into_values().collect()
}

fn span(points: &[[f64; 2]]) -> f64 {
    let mut maximum = 0.0f64;
    for left in 0..points.len() {
        for right in (left + 1)..points.len() {
            maximum = maximum.max(
                (points[left][0] - points[right][0]).hypot(points[left][1] - points[right][1]),
            );
        }
    }
    maximum
}

fn independence_evidence(
    source: &CandidateGroupSource,
    delta_bic_min: f64,
    component_snr_min: f64,
) -> bool {
    source
        .deblend_delta_bic
        .zip(source.deblend_component_snr)
        .is_some_and(|(delta, snr)| delta >= delta_bic_min && snr >= component_snr_min)
}

fn strength(source: &CandidateGroupSource, delta_bic_min: f64, component_snr_min: f64) -> [f64; 5] {
    let floor = |value: Option<f64>| {
        value
            .filter(|value| value.is_finite())
            .unwrap_or(f64::NEG_INFINITY)
    };
    [
        if source.quality_passed { 1.0 } else { 0.0 },
        if independence_evidence(source, delta_bic_min, component_snr_min) {
            1.0
        } else {
            0.0
        },
        floor(source.filter_snr),
        floor(source.flux_snr),
        floor(source.peak),
    ]
}

fn compare_strength(left: [f64; 5], right: [f64; 5]) -> std::cmp::Ordering {
    for (left, right) in left.into_iter().zip(right) {
        let order = left
            .partial_cmp(&right)
            .unwrap_or(std::cmp::Ordering::Equal);
        if !order.is_eq() {
            return order;
        }
    }
    std::cmp::Ordering::Equal
}

fn group_row(
    group_id: usize,
    sources: &[CandidateGroupSource],
    indices: &[usize],
    delta_bic_min: f64,
    component_snr_min: f64,
) -> CandidateGroupRow {
    let mut members: Vec<_> = indices.iter().map(|index| &sources[*index]).collect();
    members.sort_by_key(|source| source.detection_id);
    let mut representative = members[0];
    for source in members.iter().skip(1) {
        if compare_strength(
            strength(source, delta_bic_min, component_snr_min),
            strength(representative, delta_bic_min, component_snr_min),
        )
        .is_gt()
        {
            representative = source;
        }
    }
    let quality_count = members
        .iter()
        .filter(|source| source.quality_passed)
        .count();
    let flags: BTreeSet<_> = members
        .iter()
        .flat_map(|source| source.flags.iter().cloned())
        .collect();
    let classes: BTreeSet<_> = members
        .iter()
        .map(|source| source.feature_class.clone())
        .collect();
    let psf_evidence = members
        .iter()
        .any(|source| independence_evidence(source, delta_bic_min, component_snr_min));
    let blocking: Vec<_> = flags
        .iter()
        .filter(|flag| BLOCKING_FLAGS.contains(&flag.as_str()))
        .cloned()
        .collect();
    let independent =
        members.len() > 1 && quality_count == members.len() && blocking.is_empty() && psf_evidence;
    let (classification, reason) = if members.len() == 1 {
        (
            "isolated".to_owned(),
            "组内只有一个候选；仍需外部身份核验".to_owned(),
        )
    } else if independent {
        (
            "independent_group_candidate".to_owned(),
            "当前质量、值域、双PSF工程线均满足；仍需星表/注入真值确认".to_owned(),
        )
    } else {
        let mut reasons = vec!["多个候选共享当前近邻组".to_owned()];
        if quality_count < members.len() {
            reasons.push("至少一个成员未通过质量层".into());
        }
        if !blocking.is_empty() {
            reasons.push(format!("含结构/值域旗标 {}", blocking.join(",")));
        }
        if !psf_evidence {
            reasons.push("没有达到双PSF独立证据线".into());
        }
        ("unresolved_group".to_owned(), reasons.join("；"))
    };
    let centroids: Vec<_> = members.iter().map(|source| [source.x, source.y]).collect();
    let peaks: Vec<_> = members
        .iter()
        .map(|source| [source.peak_x, source.peak_y])
        .collect();
    CandidateGroupRow {
        group_id,
        detection_ids: members
            .iter()
            .map(|source| source.detection_id.to_string())
            .collect::<Vec<_>>()
            .join("|"),
        member_count: members.len(),
        representative_detection_id: representative.detection_id,
        classification,
        quality_member_count: quality_count,
        centroid_span_px: span(&centroids),
        peak_span_px: span(&peaks),
        feature_classes: classes.into_iter().collect::<Vec<_>>().join("|"),
        flags: flags.into_iter().collect::<Vec<_>>().join("|"),
        independent_psf_evidence: independent,
        reason,
    }
}

fn class_rows(
    sources: &[CandidateGroupSource],
    groups: &[CandidateGroupRow],
) -> Vec<CandidateGroupClassRow> {
    let mut source_group = BTreeMap::new();
    for group in groups {
        for id in group
            .detection_ids
            .split('|')
            .filter_map(|value| value.parse::<i64>().ok())
        {
            source_group.insert(id, group);
        }
    }
    let classes: BTreeSet<_> = sources
        .iter()
        .map(|source| source.feature_class.as_str())
        .collect();
    classes
        .into_iter()
        .map(|feature_class| {
            let members: Vec<_> = sources
                .iter()
                .filter(|source| source.feature_class == feature_class)
                .collect();
            let multi: Vec<_> = members
                .iter()
                .filter(|source| source_group[&source.detection_id].member_count > 1)
                .collect();
            let unresolved = multi
                .iter()
                .filter(|source| {
                    source_group[&source.detection_id].classification == "unresolved_group"
                })
                .count();
            let independent = multi
                .iter()
                .filter(|source| {
                    source_group[&source.detection_id].classification
                        == "independent_group_candidate"
                })
                .count();
            CandidateGroupClassRow {
                feature_class: feature_class.to_owned(),
                source_count: members.len(),
                quality_source_count: members
                    .iter()
                    .filter(|source| source.quality_passed)
                    .count(),
                multi_group_source_count: multi.len(),
                multi_group_source_fraction: multi.len() as f64 / members.len() as f64,
                unresolved_group_source_count: unresolved,
                independent_group_source_count: independent,
            }
        })
        .collect()
}

pub fn run(
    catalog_csv: &Path,
    target_ids: &[i64],
    psf_fwhm: f64,
    group_radius_px: Option<f64>,
    delta_bic_min: f64,
    component_snr_min: f64,
) -> Result<CandidateGroupAuditResult> {
    let (psf_fwhm, group_radius_px, delta_bic_min, component_snr_min) =
        validate_parameters(psf_fwhm, group_radius_px, delta_bic_min, component_snr_min)?;
    let sources = load_candidate_group_sources(catalog_csv)?;
    let source_by_id: BTreeMap<_, _> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| (source.detection_id, index))
        .collect();
    let targets: Vec<_> = target_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let missing: Vec<_> = targets
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
    let points: Vec<_> = sources.iter().map(|source| [source.x, source.y]).collect();
    let tree = KdTree::new(&points)?;
    let mut edges = Vec::new();
    for (index, point) in points.iter().enumerate() {
        for (neighbor, _) in tree.within_radius(*point, group_radius_px)? {
            if neighbor > index {
                edges.push((index, neighbor));
            }
        }
    }
    edges.sort_unstable();
    let mut components = connected_components(sources.len(), &edges);
    components.sort_by_key(|indices| {
        indices
            .iter()
            .map(|index| sources[*index].detection_id)
            .min()
            .unwrap_or(i64::MAX)
    });
    let groups: Vec<_> = components
        .iter()
        .enumerate()
        .map(|(index, indices)| {
            group_row(
                index + 1,
                &sources,
                indices,
                delta_bic_min,
                component_snr_min,
            )
        })
        .collect();
    let mut target_rows = Vec::new();
    for group in &groups {
        let matching: Vec<_> = group
            .detection_ids
            .split('|')
            .filter_map(|value| value.parse::<i64>().ok())
            .filter(|id| targets.contains(id))
            .collect();
        if matching.is_empty() {
            continue;
        }
        target_rows.push(CandidateGroupTargetRow {
            group_id: group.group_id,
            target_detection_ids: matching
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join("|"),
            group_detection_ids: group.detection_ids.clone(),
            classification: group.classification.clone(),
            representative_detection_id: group.representative_detection_id,
            member_count: group.member_count,
            centroid_span_px: group.centroid_span_px,
            peak_span_px: group.peak_span_px,
            feature_classes: group.feature_classes.clone(),
            flags: group.flags.clone(),
            independent_psf_evidence: group.independent_psf_evidence,
            reason: group.reason.clone(),
        });
    }
    let multi: Vec<_> = groups
        .iter()
        .filter(|group| group.member_count > 1)
        .collect();
    let unresolved = multi
        .iter()
        .filter(|group| group.classification == "unresolved_group")
        .count();
    let independent = multi
        .iter()
        .filter(|group| group.classification == "independent_group_candidate")
        .count();
    let conclusion = format!(
        "源级近邻图将 {} 个候选组织为 {} 个父源组，其中多成员组 {} 个、未分辨组 {} 个、当前独立候选组 {} 个。该结果只改变解释层次，不改变候选数、质量数或物理恒星真值。",
        comma_group(sources.len()), comma_group(groups.len()), comma_group(multi.len()),
        comma_group(unresolved), comma_group(independent)
    );
    let classes = class_rows(&sources, &groups);
    Ok(CandidateGroupAuditResult {
        catalog_path: catalog_csv.to_string_lossy().into_owned(),
        source_count: sources.len(),
        quality_count: sources
            .iter()
            .filter(|source| source.quality_passed)
            .count(),
        group_count: groups.len(),
        multi_member_group_count: multi.len(),
        unresolved_group_count: unresolved,
        independent_group_candidate_count: independent,
        groups,
        classes,
        targets: target_rows,
        parameters: json!({
            "psf_fwhm": psf_fwhm,
            "group_radius_px": group_radius_px,
            "group_radius_definition": "max(2.5 px, 1.5 * psf_fwhm) unless explicitly overridden",
            "delta_bic_min": delta_bic_min,
            "component_snr_min": component_snr_min,
            "edge_definition": "centroid distance <= group_radius_px; connected components are transitive",
            "independence_definition": "multi-member group + every member quality_passed + no blocking flag + at least one member reaches delta_bic and component_snr thresholds",
            "interpretation_boundary": "detector-level parent grouping; no star probability, pair truth, precision, or FDR",
        }),
        conclusion,
    })
}

fn comma_group(value: usize) -> String {
    let digits = value.to_string();
    let mut output = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(character);
    }
    output
}

const GROUP_FIELDS: &[&str] = &[
    "group_id",
    "detection_ids",
    "member_count",
    "representative_detection_id",
    "classification",
    "quality_member_count",
    "centroid_span_px",
    "peak_span_px",
    "feature_classes",
    "flags",
    "independent_psf_evidence",
    "reason",
];
const CLASS_FIELDS: &[&str] = &[
    "feature_class",
    "source_count",
    "quality_source_count",
    "multi_group_source_count",
    "multi_group_source_fraction",
    "unresolved_group_source_count",
    "independent_group_source_count",
];
const TARGET_FIELDS: &[&str] = &[
    "group_id",
    "target_detection_ids",
    "group_detection_ids",
    "classification",
    "representative_detection_id",
    "member_count",
    "centroid_span_px",
    "peak_span_px",
    "feature_classes",
    "flags",
    "independent_psf_evidence",
    "reason",
];

fn csv_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(number) => number
            .as_i64()
            .map(|value| value.to_string())
            .or_else(|| number.as_u64().map(|value| value.to_string()))
            .or_else(|| number.as_f64().map(|value| format!("{value:?}")))
            .unwrap_or_default(),
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

fn write_records(path: &Path, headers: &[&str], rows: &[Value]) -> Result<()> {
    let mut file = fs::File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for row in rows {
        let values: Vec<_> = headers
            .iter()
            .map(|header| csv_value(&row[*header]))
            .collect();
        writer
            .write_record(values)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &CandidateGroupAuditResult, output: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    let groups: Vec<_> = result
        .groups
        .iter()
        .map(|row| serde_json::to_value(row).unwrap())
        .collect();
    let classes: Vec<_> = result
        .classes
        .iter()
        .map(|row| serde_json::to_value(row).unwrap())
        .collect();
    let targets: Vec<_> = result
        .targets
        .iter()
        .map(|row| serde_json::to_value(row).unwrap())
        .collect();
    write_records(&output.join("source_groups.csv"), GROUP_FIELDS, &groups)?;
    write_records(
        &output.join("source_group_class_summary.csv"),
        CLASS_FIELDS,
        &classes,
    )?;
    write_records(
        &output.join("source_group_targets.csv"),
        TARGET_FIELDS,
        &targets,
    )?;
    fs::write(
        output.join("source_group_summary.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut positional = Vec::new();
    let mut target_ids = Vec::new();
    let mut psf_fwhm = 2.0;
    let mut group_radius = None;
    let mut delta_bic_min = 10.0;
    let mut component_snr_min = 5.0;
    let mut output = PathBuf::from("tmp/source-group-audit");
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        let (name, inline) = argument.split_once('=').unwrap_or((argument, ""));
        let option = match name {
            "--target-id"
            | "--psf-fwhm"
            | "--group-radius"
            | "--delta-bic-min"
            | "--component-snr-min"
            | "--out-dir" => Some(name),
            _ => None,
        };
        if let Some(option) = option {
            let value = if !inline.is_empty() {
                inline
            } else {
                index += 1;
                args.get(index).ok_or("option requires a value")?
            };
            match option {
                "--target-id" => {
                    target_ids.push(value.parse::<i64>().map_err(|error| error.to_string())?)
                }
                "--psf-fwhm" => {
                    psf_fwhm = value.parse::<f64>().map_err(|error| error.to_string())?
                }
                "--group-radius" => {
                    group_radius = Some(value.parse::<f64>().map_err(|error| error.to_string())?)
                }
                "--delta-bic-min" => {
                    delta_bic_min = value.parse::<f64>().map_err(|error| error.to_string())?
                }
                "--component-snr-min" => {
                    component_snr_min = value.parse::<f64>().map_err(|error| error.to_string())?
                }
                _ => output = PathBuf::from(value),
            }
        } else if name.starts_with('-') {
            return Err(format!("unknown argument {name}"));
        } else {
            positional.push(PathBuf::from(name));
        }
        index += 1;
    }
    if positional.len() != 1 {
        return Err("expected one source catalog path".into());
    }
    let result = run(
        &positional[0],
        &target_ids,
        psf_fwhm,
        group_radius,
        delta_bic_min,
        component_snr_min,
    )?;
    write_artifacts(&result, &output)?;
    Ok(json!({"tool":"rst19-source-group-audit", "out_dir":output,
        "result_path":output.join("source_group_summary.json"), "result":result.as_dict()}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn groups_are_transitive_and_keep_targets_and_artifacts_aligned() {
        let root = std::env::temp_dir().join(format!(
            "rst19-source-groups-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        fs::write(&catalog, "\u{feff}detection_id,x,y,peak_x,peak_y,peak,flux_snr,filter_snr,fwhm,deblend_delta_bic,deblend_component_snr,quality_passed,feature_class,flags\r\n1,0,0,0,0,3992,204.6,1195,2.59,-4.09,0.87,False,range_anomaly,CODE_PATTERN\r\n2,2.738,0,4,0,569,115.5,300,2.44,-4.09,0.87,False,crowded_blend,UNRESOLVED_BLEND\r\n3,10,0,10,0,250,12,20,2,,,True,compact_quality,\r\n4,20,0,20,0,900,20,100,2,12,6,True,compact_quality,\r\n5,23,0,23,0,400,10,60,2,,,True,compact_quality,\r\n").unwrap();
        let result = run(&catalog, &[1, 2], 2.0, None, 10.0, 5.0).unwrap();
        assert_eq!(result.group_count, 3);
        assert_eq!(result.multi_member_group_count, 2);
        assert_eq!(result.unresolved_group_count, 1);
        assert_eq!(result.independent_group_candidate_count, 1);
        assert_eq!(result.targets[0].target_detection_ids, "1|2");
        assert_eq!(result.targets[0].classification, "unresolved_group");
        let out = root.join("out");
        write_artifacts(&result, &out).unwrap();
        assert!(out.join("source_groups.csv").is_file());
        assert!(out.join("source_group_class_summary.csv").is_file());
        assert!(out.join("source_group_targets.csv").is_file());
        assert!(out.join("source_group_summary.json").is_file());
        let _ = fs::remove_dir_all(root);
    }
}
