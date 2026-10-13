//! Feature-class anchor selection around the native contaminated-pair audit.
//!
//! Feature classes remain detector-level labels. Injection recovery here is
//! conditional on the selected anchors and does not estimate physical truth.

use crate::{
    contaminated_pair_injection::{
        self, ContaminatedPairAnchor, ContaminatedPairInjectionResult,
        ContaminatedPairInjectionRow, Options as InjectionOptions,
    },
    fits,
    models::FitsFrame,
    Result,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

pub const CLASS_SELECTION_METRICS: [&str; 3] = ["filter_snr", "flux_snr", "peak"];
pub const CLASS_SELECTION_MODES: [&str; 2] = ["median", "high"];
pub const DEFAULT_FEATURE_CLASSES: [&str; 8] = [
    "compact_quality",
    "crowded_blend",
    "linear_artifact",
    "masked_or_edge",
    "range_anomaly",
    "shape_outlier",
    "spike_or_support",
    "weak_or_background",
];
const CLASS_SUMMARY_FIELDS: [&str; 20] = [
    "feature_class",
    "feature_class_label",
    "separation_px",
    "secondary_to_primary_ratio",
    "anchor_count",
    "condition_count",
    "anchor_detection_ids",
    "baseline_candidate_pair_count",
    "injected_candidate_pair_count",
    "new_candidate_pair_count",
    "baseline_quality_pair_count",
    "injected_quality_pair_count",
    "new_quality_pair_count",
    "merged_candidate_count",
    "merged_quality_count",
    "candidate_pair_resolved_fraction",
    "quality_pair_resolved_fraction",
    "new_candidate_pair_fraction",
    "new_quality_pair_fraction",
    "merged_candidate_fraction",
];
const QUALITY_REASON_SUMMARY_FIELDS: [&str; 10] = [
    "feature_class",
    "feature_class_label",
    "separation_px",
    "secondary_to_primary_ratio",
    "anchor_count",
    "condition_count",
    "endpoint_count",
    "primary_quality_reason_counts",
    "secondary_quality_reason_counts",
    "combined_quality_reason_counts",
];
const INTERPRETATION_BOUNDARY: &str = "feature class labels and anchor selection are detector-level diagnostics; injection recovery is not precision, FDR, or physical star truth";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureClassAnchorSelection {
    pub feature_class: String,
    pub feature_class_label: String,
    pub detection_id: i64,
    pub x: f64,
    pub y: f64,
    pub selection_metric: String,
    pub selection_mode: String,
    pub metric_value: f64,
    pub class_metric_median: f64,
    pub quality_passed: bool,
    pub flags: String,
    pub class_count: usize,
    pub eligible_count: usize,
    pub selection_rank: usize,
    pub direction_x: f64,
    pub direction_y: f64,
}

impl FeatureClassAnchorSelection {
    pub fn to_anchor(&self) -> ContaminatedPairAnchor {
        ContaminatedPairAnchor {
            name: format!("{}_{}", self.feature_class, self.detection_id),
            x: self.x,
            y: self.y,
            direction_x: self.direction_x,
            direction_y: self.direction_y,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub feature_classes: Option<Vec<String>>,
    pub per_class: usize,
    pub selection_metric: String,
    pub selection_mode: String,
    pub separations_px: Vec<f64>,
    pub secondary_to_primary_ratios: Vec<f64>,
    pub total_peak_excess_adu: f64,
    pub psf_fwhm: f64,
    pub max_sources: Option<usize>,
    pub analysis_scope: String,
    pub roi_half_size_px: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            feature_classes: None,
            per_class: 1,
            selection_metric: "filter_snr".into(),
            selection_mode: "median".into(),
            separations_px: vec![2.738, 4.123],
            secondary_to_primary_ratios: vec![0.143, 1.0],
            total_peak_excess_adu: 4096.0,
            psf_fwhm: 2.0,
            max_sources: None,
            analysis_scope: "full_frame".into(),
            roi_half_size_px: 192,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FeatureClassContaminatedPairResult {
    pub catalog_path: String,
    pub source_path: String,
    pub selection_metric: String,
    pub selection_mode: String,
    pub selected_rows: Vec<FeatureClassAnchorSelection>,
    pub injection: ContaminatedPairInjectionResult,
}

impl FeatureClassContaminatedPairResult {
    pub fn as_dict(&self) -> Result<Value> {
        Ok(json!({
            "catalog_path": self.catalog_path,
            "source_path": self.source_path,
            "selection_metric": self.selection_metric,
            "selection_mode": self.selection_mode,
            "selected_rows": self.selected_rows,
            "injection": self.injection,
            "class_summary": summarize_feature_class_contaminated_pair(self)?,
            "quality_reason_summary": summarize_feature_class_quality_reasons(self)?,
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        }))
    }
}

#[derive(Clone, Debug)]
struct CatalogRow {
    fields: HashMap<String, String>,
}

fn field<'a>(row: &'a CatalogRow, name: &str) -> &'a str {
    row.fields.get(name).map(String::as_str).unwrap_or("")
}

fn finite_float(row: &CatalogRow, name: &str) -> Option<f64> {
    field(row, name)
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn bool_value(row: &CatalogRow, name: &str) -> bool {
    matches!(
        field(row, name).trim().to_ascii_lowercase().as_str(),
        "true" | "1" | "yes" | "y" | "是"
    )
}

fn read_catalog(path: &Path) -> Result<(Vec<String>, Vec<CatalogRow>)> {
    let bytes = fs::read(path).map_err(|_| format!("无法读取源级 CSV：{}", path.display()))?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
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
    let required = [
        "detection_id",
        "feature_class",
        "x",
        "y",
        "quality_passed",
        "flags",
    ];
    let mut missing = required
        .iter()
        .filter(|name| !headers.iter().any(|header| header == **name))
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    missing.sort();
    if !missing.is_empty() {
        return Err(format!("源级 CSV 缺少列：{}", missing.join(", ")));
    }
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|error| error.to_string())?;
        let fields = headers
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), record.get(index).unwrap_or("").to_owned()))
            .collect();
        rows.push(CatalogRow { fields });
    }
    if rows.is_empty() {
        return Err(format!("源级 CSV 为空：{}", path.display()));
    }
    Ok((headers, rows))
}

fn normalize_direction(direction_x: f64, direction_y: f64) -> Result<(f64, f64)> {
    let norm = direction_x.hypot(direction_y);
    if !norm.is_finite() || norm <= 0.0 {
        return Err("direction must be finite and non-zero".into());
    }
    Ok((direction_x / norm, direction_y / norm))
}

fn in_bounds(
    x: f64,
    y: f64,
    image_shape: Option<(usize, usize)>,
    half_separation: f64,
    direction_x: f64,
    direction_y: f64,
) -> bool {
    let Some((height, width)) = image_shape else {
        return true;
    };
    [-1.0, 1.0].into_iter().all(|sign| {
        let point_x = x + sign * half_separation * direction_x;
        let point_y = y + sign * half_separation * direction_y;
        0.0 <= point_x && point_x < width as f64 && 0.0 <= point_y && point_y < height as f64
    })
}

#[allow(clippy::too_many_arguments)]
pub fn select_feature_class_anchors(
    catalog_path: &Path,
    feature_classes: Option<&[String]>,
    per_class: usize,
    selection_metric: &str,
    selection_mode: &str,
    image_shape: Option<(usize, usize)>,
    max_separation_px: f64,
    direction_x: f64,
    direction_y: f64,
) -> Result<Vec<FeatureClassAnchorSelection>> {
    if per_class == 0 {
        return Err("per_class must be positive".into());
    }
    if !CLASS_SELECTION_METRICS.contains(&selection_metric) {
        return Err(format!(
            "unsupported selection_metric: {selection_metric:?}"
        ));
    }
    if !CLASS_SELECTION_MODES.contains(&selection_mode) {
        return Err(format!("unsupported selection_mode: {selection_mode:?}"));
    }
    if !max_separation_px.is_finite() || max_separation_px <= 0.0 {
        return Err("max_separation_px must be finite and positive".into());
    }
    let (unit_x, unit_y) = normalize_direction(direction_x, direction_y)?;
    let (_headers, rows) = read_catalog(catalog_path)?;
    let mut by_class: BTreeMap<String, Vec<&CatalogRow>> = BTreeMap::new();
    for row in &rows {
        let class_name = field(row, "feature_class").trim();
        if !class_name.is_empty() && class_name != "other_rejected" {
            by_class.entry(class_name.to_owned()).or_default().push(row);
        }
    }
    let requested = if let Some(feature_classes) = feature_classes {
        let mut seen = HashSet::new();
        feature_classes
            .iter()
            .map(|name| name.trim())
            .filter(|name| !name.is_empty() && seen.insert((*name).to_owned()))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    } else {
        let mut names = DEFAULT_FEATURE_CLASSES
            .iter()
            .filter(|name| by_class.contains_key(**name))
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        let mut additional = by_class
            .keys()
            .filter(|name| !names.contains(name))
            .cloned()
            .collect::<Vec<_>>();
        names.append(&mut additional);
        names
    };
    if requested.is_empty() {
        return Err("feature_classes cannot be empty".into());
    }
    if let Some(unknown) = requested.iter().find(|name| !by_class.contains_key(*name)) {
        return Err(format!("source catalog has no feature_class={unknown:?}"));
    }

    let half_separation = 0.5 * max_separation_px;
    let mut selected = Vec::new();
    for class_name in requested {
        let class_rows = &by_class[&class_name];
        let mut eligible = Vec::<(f64, i64, &CatalogRow)>::new();
        for row in class_rows {
            let (Some(value), Some(x), Some(y)) = (
                finite_float(row, selection_metric),
                finite_float(row, "x"),
                finite_float(row, "y"),
            ) else {
                continue;
            };
            let detection_id = field(row, "detection_id")
                .trim()
                .parse::<i64>()
                .map_err(|_| format!("invalid detection_id in class {class_name:?}"))?;
            if in_bounds(x, y, image_shape, half_separation, unit_x, unit_y) {
                eligible.push((value, detection_id, row));
            }
        }
        if eligible.is_empty() {
            return Err(format!(
                "class {class_name:?} has no eligible in-bounds rows"
            ));
        }
        let mut values = eligible.iter().map(|item| item.0).collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        let middle = values.len() / 2;
        let class_median = if values.len().is_multiple_of(2) {
            (values[middle - 1] + values[middle]) / 2.0
        } else {
            values[middle]
        };
        if selection_mode == "median" {
            eligible.sort_by(|left, right| {
                (left.0 - class_median)
                    .abs()
                    .total_cmp(&(right.0 - class_median).abs())
                    .then_with(|| left.1.cmp(&right.1))
            });
        } else {
            let rejected = eligible
                .iter()
                .any(|item| !bool_value(item.2, "quality_passed"));
            if rejected {
                eligible.retain(|item| !bool_value(item.2, "quality_passed"));
            }
            eligible.sort_by(|left, right| {
                right
                    .0
                    .total_cmp(&left.0)
                    .then_with(|| left.1.cmp(&right.1))
            });
        }
        for (rank, (metric_value, detection_id, row)) in
            eligible.into_iter().take(per_class).enumerate()
        {
            let x = finite_float(row, "x").expect("eligible rows have finite x");
            let y = finite_float(row, "y").expect("eligible rows have finite y");
            let label = if row.fields.contains_key("feature_class_label") {
                field(row, "feature_class_label").to_owned()
            } else {
                class_name.clone()
            };
            selected.push(FeatureClassAnchorSelection {
                feature_class: class_name.clone(),
                feature_class_label: label,
                detection_id,
                x,
                y,
                selection_metric: selection_metric.to_owned(),
                selection_mode: selection_mode.to_owned(),
                metric_value,
                class_metric_median: class_median,
                quality_passed: bool_value(row, "quality_passed"),
                flags: field(row, "flags").to_owned(),
                class_count: class_rows.len(),
                eligible_count: values.len(),
                selection_rank: rank + 1,
                direction_x: unit_x,
                direction_y: unit_y,
            });
        }
    }
    Ok(selected)
}

fn fraction(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 / total as f64
    }
}

struct SummaryGroup<'a> {
    feature_class: String,
    feature_class_label: String,
    separation_px: f64,
    ratio: f64,
    rows: Vec<&'a ContaminatedPairInjectionRow>,
    selection_ids: BTreeSet<i64>,
}

fn summary_groups(result: &FeatureClassContaminatedPairResult) -> Result<Vec<SummaryGroup<'_>>> {
    let selection_by_anchor = result
        .selected_rows
        .iter()
        .map(|selection| (selection.to_anchor().name, selection))
        .collect::<HashMap<_, _>>();
    let mut keyed = result
        .injection
        .rows
        .iter()
        .map(|row| {
            let selection = selection_by_anchor.get(&row.anchor_name).ok_or_else(|| {
                format!("injection row has unknown anchor: {:?}", row.anchor_name)
            })?;
            Ok((
                selection.feature_class.clone(),
                selection.feature_class_label.clone(),
                selection.detection_id,
                row.separation_px,
                row.secondary_to_primary_ratio,
                row,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    keyed.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.3.total_cmp(&right.3))
            .then_with(|| left.4.total_cmp(&right.4))
    });
    let mut groups: Vec<SummaryGroup<'_>> = Vec::new();
    for (class, label, detection_id, separation, ratio, row) in keyed {
        let matches_last = groups.last().is_some_and(|group| {
            group.feature_class == class
                && group.separation_px == separation
                && group.ratio == ratio
        });
        if !matches_last {
            groups.push(SummaryGroup {
                feature_class: class,
                feature_class_label: label,
                separation_px: separation,
                ratio,
                rows: Vec::new(),
                selection_ids: BTreeSet::new(),
            });
        }
        let group = groups.last_mut().unwrap();
        group.selection_ids.insert(detection_id);
        group.rows.push(row);
    }
    Ok(groups)
}

pub fn summarize_feature_class_contaminated_pair(
    result: &FeatureClassContaminatedPairResult,
) -> Result<Vec<Value>> {
    summary_groups(result)?
        .into_iter()
        .map(|group| {
            let rows = group.rows;
            let total = rows.len();
            let ids = group
                .selection_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("|");
            Ok(json!({
                "feature_class": group.feature_class,
                "feature_class_label": group.feature_class_label,
                "separation_px": group.separation_px,
                "secondary_to_primary_ratio": group.ratio,
                "anchor_count": group.selection_ids.len(),
                "condition_count": total,
                "anchor_detection_ids": ids,
                "baseline_candidate_pair_count": pair_count(&rows, "baseline_candidate_primary", "baseline_candidate_secondary"),
                "injected_candidate_pair_count": flag_count(&rows, "candidate_pair_resolved"),
                "new_candidate_pair_count": pair_count(&rows, "new_candidate_primary", "new_candidate_secondary"),
                "baseline_quality_pair_count": pair_count(&rows, "baseline_quality_primary", "baseline_quality_secondary"),
                "injected_quality_pair_count": flag_count(&rows, "quality_pair_resolved"),
                "new_quality_pair_count": pair_count(&rows, "new_quality_primary", "new_quality_secondary"),
                "merged_candidate_count": flag_count(&rows, "merged_candidate"),
                "merged_quality_count": flag_count(&rows, "merged_quality"),
                "candidate_pair_resolved_fraction": fraction(flag_count(&rows, "candidate_pair_resolved"), total),
                "quality_pair_resolved_fraction": fraction(flag_count(&rows, "quality_pair_resolved"), total),
                "new_candidate_pair_fraction": fraction(pair_count(&rows, "new_candidate_primary", "new_candidate_secondary"), total),
                "new_quality_pair_fraction": fraction(pair_count(&rows, "new_quality_primary", "new_quality_secondary"), total),
                "merged_candidate_fraction": fraction(flag_count(&rows, "merged_candidate"), total),
            }))
        })
        .collect()
}

fn flag_count(rows: &[&ContaminatedPairInjectionRow], name: &str) -> usize {
    rows.iter()
        .filter(|row| match name {
            "candidate_pair_resolved" => row.candidate_pair_resolved,
            "quality_pair_resolved" => row.quality_pair_resolved,
            "merged_candidate" => row.merged_candidate,
            "merged_quality" => row.merged_quality,
            _ => false,
        })
        .count()
}

fn pair_count(rows: &[&ContaminatedPairInjectionRow], primary: &str, secondary: &str) -> usize {
    rows.iter()
        .filter(|row| {
            let primary_value = match primary {
                "baseline_candidate_primary" => row.baseline_candidate_primary,
                "new_candidate_primary" => row.new_candidate_primary,
                "new_quality_primary" => row.new_quality_primary,
                "baseline_quality_primary" => row.baseline_quality_primary,
                _ => false,
            };
            let secondary_value = match secondary {
                "baseline_candidate_secondary" => row.baseline_candidate_secondary,
                "new_candidate_secondary" => row.new_candidate_secondary,
                "new_quality_secondary" => row.new_quality_secondary,
                "baseline_quality_secondary" => row.baseline_quality_secondary,
                _ => false,
            };
            primary_value && secondary_value
        })
        .count()
}

fn reason_tokens(reason: &str) -> Vec<&str> {
    let value = reason.trim();
    if value.is_empty() {
        return vec!["UNKNOWN"];
    }
    let tokens = value
        .split('|')
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        vec!["UNKNOWN"]
    } else {
        tokens
    }
}

fn reason_counts_text(values: impl IntoIterator<Item = String>) -> Result<String> {
    let mut counts = BTreeMap::<String, usize>::new();
    for value in values {
        for token in reason_tokens(&value) {
            *counts.entry(token.to_owned()).or_default() += 1;
        }
    }
    serde_json::to_string(&counts).map_err(|error| error.to_string())
}

pub fn summarize_feature_class_quality_reasons(
    result: &FeatureClassContaminatedPairResult,
) -> Result<Vec<Value>> {
    summary_groups(result)?
        .into_iter()
        .map(|group| {
            let rows = group.rows;
            let primary = reason_counts_text(
                rows.iter()
                    .map(|row| row.injected_primary_quality_reason.clone()),
            )?;
            let secondary = reason_counts_text(
                rows.iter()
                    .map(|row| row.injected_secondary_quality_reason.clone()),
            )?;
            let combined = reason_counts_text(rows.iter().flat_map(|row| {
                [
                    row.injected_primary_quality_reason.clone(),
                    row.injected_secondary_quality_reason.clone(),
                ]
            }))?;
            Ok(json!({
                "feature_class": group.feature_class,
                "feature_class_label": group.feature_class_label,
                "separation_px": group.separation_px,
                "secondary_to_primary_ratio": group.ratio,
                "anchor_count": group.selection_ids.len(),
                "condition_count": rows.len(),
                "endpoint_count": 2 * rows.len(),
                "primary_quality_reason_counts": primary,
                "secondary_quality_reason_counts": secondary,
                "combined_quality_reason_counts": combined,
            }))
        })
        .collect()
}

pub fn run(
    frame: &FitsFrame,
    catalog_path: &Path,
    options: &Options,
) -> Result<FeatureClassContaminatedPairResult> {
    let max_separation = options
        .separations_px
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or("separations_px cannot be empty")?;
    let selected_rows = select_feature_class_anchors(
        catalog_path,
        options.feature_classes.as_deref(),
        options.per_class,
        &options.selection_metric,
        &options.selection_mode,
        Some((frame.height, frame.width)),
        max_separation,
        1.0,
        0.0,
    )?;
    let anchors = selected_rows
        .iter()
        .map(FeatureClassAnchorSelection::to_anchor)
        .collect::<Vec<_>>();
    let injection_options = InjectionOptions {
        separations_px: options.separations_px.clone(),
        secondary_to_primary_ratios: options.secondary_to_primary_ratios.clone(),
        total_peak_excess_adu: options.total_peak_excess_adu,
        psf_fwhm: options.psf_fwhm,
        max_sources: options.max_sources,
        analysis_scope: options.analysis_scope.clone(),
        roi_half_size_px: options.roi_half_size_px,
        ..InjectionOptions::default()
    };
    let injection = contaminated_pair_injection::run(frame, &anchors, &injection_options)?;
    Ok(FeatureClassContaminatedPairResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        source_path: frame.path.to_string_lossy().into_owned(),
        selection_metric: options.selection_metric.clone(),
        selection_mode: options.selection_mode.clone(),
        selected_rows,
        injection,
    })
}

fn write_csv(path: &Path, fields: &[&str], rows: &[Vec<String>]) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(fields)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(row)
            .map_err(|error| error.to_string())?;
    }
    let mut bytes = vec![0xef, 0xbb, 0xbf];
    bytes.extend(writer.into_inner().map_err(|error| error.to_string())?);
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn value_cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(values) => values.iter().map(value_cell).collect::<Vec<_>>().join("|"),
        Value::Object(_) => serde_json::to_string(value).unwrap_or_default(),
    }
}

pub fn write_artifacts(
    result: &FeatureClassContaminatedPairResult,
    output_dir: &Path,
) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    contaminated_pair_injection::write_artifacts(&result.injection, output_dir)?;
    // Keep the dataclass field order instead of JSON map order.
    let selection_fields = [
        "feature_class",
        "feature_class_label",
        "detection_id",
        "x",
        "y",
        "selection_metric",
        "selection_mode",
        "metric_value",
        "class_metric_median",
        "quality_passed",
        "flags",
        "class_count",
        "eligible_count",
        "selection_rank",
        "direction_x",
        "direction_y",
    ];
    let ordered_selection_rows = result
        .selected_rows
        .iter()
        .map(|row| {
            let value = serde_json::to_value(row).unwrap();
            selection_fields
                .iter()
                .map(|name| value_cell(&value[*name]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    write_csv(
        &output_dir.join("contaminated_pair_class_selection.csv"),
        &selection_fields,
        &ordered_selection_rows,
    )?;
    let summaries = summarize_feature_class_contaminated_pair(result)?;
    write_csv(
        &output_dir.join("contaminated_pair_class_summary.csv"),
        &CLASS_SUMMARY_FIELDS,
        &summaries
            .iter()
            .map(|row| {
                CLASS_SUMMARY_FIELDS
                    .iter()
                    .map(|name| value_cell(&row[*name]))
                    .collect()
            })
            .collect::<Vec<Vec<String>>>(),
    )?;
    let reasons = summarize_feature_class_quality_reasons(result)?;
    write_csv(
        &output_dir.join("contaminated_pair_quality_reason_summary.csv"),
        &QUALITY_REASON_SUMMARY_FIELDS,
        &reasons
            .iter()
            .map(|row| {
                QUALITY_REASON_SUMMARY_FIELDS
                    .iter()
                    .map(|name| value_cell(&row[*name]))
                    .collect()
            })
            .collect::<Vec<Vec<String>>>(),
    )?;
    let json_bytes =
        serde_json::to_vec_pretty(&result.as_dict()?).map_err(|error| error.to_string())?;
    fs::write(
        output_dir.join("contaminated_pair_class_audit.json"),
        json_bytes,
    )
    .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

#[derive(Clone, Debug)]
struct ParsedArgs {
    frame: PathBuf,
    catalog: PathBuf,
    output_dir: PathBuf,
    options: Options,
}

fn next_value<'a>(
    args: &'a [String],
    index: &mut usize,
    inline: Option<&'a str>,
    flag: &str,
) -> Result<&'a str> {
    if let Some(value) = inline {
        return Ok(value);
    }
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_float_list(
    args: &[String],
    index: &mut usize,
    inline: Option<&str>,
    flag: &str,
) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    if let Some(value) = inline {
        values.push(
            value
                .parse::<f64>()
                .map_err(|_| format!("invalid value for {flag}: {value}"))?,
        );
    }
    while *index + 1 < args.len() {
        if args[*index + 1].parse::<f64>().is_err() {
            break;
        }
        *index += 1;
        values.push(
            args[*index]
                .parse::<f64>()
                .map_err(|_| format!("invalid value for {flag}"))?,
        );
    }
    if values.is_empty() {
        return Err(format!("{flag} requires at least one value"));
    }
    Ok(values)
}

fn parse_args(args: &[String]) -> Result<ParsedArgs> {
    let mut frame = None;
    let mut catalog = None;
    let mut output_dir = PathBuf::from("tmp/contaminated-pair-class-audit");
    let mut options = Options::default();
    let mut feature_classes = Vec::new();
    let mut feature_class_seen = false;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline) = arg
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((arg.as_str(), None));
        match name {
            "--source-catalog" => {
                catalog = Some(PathBuf::from(next_value(args, &mut index, inline, name)?));
            }
            "--feature-class" => {
                feature_class_seen = true;
                feature_classes.push(next_value(args, &mut index, inline, name)?.to_owned());
            }
            "--per-class" => {
                options.per_class = next_value(args, &mut index, inline, name)?
                    .parse()
                    .map_err(|_| "--per-class must be an integer")?;
            }
            "--selection-metric" => {
                options.selection_metric = next_value(args, &mut index, inline, name)?.to_owned();
            }
            "--selection-mode" => {
                options.selection_mode = next_value(args, &mut index, inline, name)?.to_owned();
            }
            "--separations-px" => {
                options.separations_px = parse_float_list(args, &mut index, inline, name)?;
            }
            "--ratios" => {
                options.secondary_to_primary_ratios =
                    parse_float_list(args, &mut index, inline, name)?;
            }
            "--total-peak-adu" => {
                options.total_peak_excess_adu = next_value(args, &mut index, inline, name)?
                    .parse()
                    .map_err(|_| "--total-peak-adu must be numeric")?;
            }
            "--psf-fwhm" => {
                options.psf_fwhm = next_value(args, &mut index, inline, name)?
                    .parse()
                    .map_err(|_| "--psf-fwhm must be numeric")?;
            }
            "--max-sources" => {
                options.max_sources = Some(
                    next_value(args, &mut index, inline, name)?
                        .parse()
                        .map_err(|_| "--max-sources must be a positive integer")?,
                );
            }
            "--analysis-scope" => {
                options.analysis_scope = next_value(args, &mut index, inline, name)?.to_owned();
            }
            "--roi-half-size-px" => {
                options.roi_half_size_px = next_value(args, &mut index, inline, name)?
                    .parse()
                    .map_err(|_| "--roi-half-size-px must be a positive integer")?;
            }
            "--out-dir" => output_dir = PathBuf::from(next_value(args, &mut index, inline, name)?),
            "--" => {
                for value in &args[index + 1..] {
                    if frame.replace(PathBuf::from(value)).is_some() {
                        return Err("only one FITS frame is allowed".into());
                    }
                }
                break;
            }
            _ if arg.starts_with('-') => return Err(format!("unknown argument {arg}")),
            _ => {
                if frame.replace(PathBuf::from(arg)).is_some() {
                    return Err("only one FITS frame is allowed".into());
                }
            }
        }
        index += 1;
    }
    options.feature_classes = feature_class_seen.then_some(feature_classes);
    Ok(ParsedArgs {
        frame: frame.ok_or("a FITS frame is required")?,
        catalog: catalog.ok_or("--source-catalog is required")?,
        output_dir,
        options,
    })
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let parsed = parse_args(args)?;
    let frame = fits::read_fits(&parsed.frame, false)?;
    let result = run(&frame, &parsed.catalog, &parsed.options)?;
    let output = write_artifacts(&result, &parsed.output_dir)?;
    Ok(json!({
        "tool": "rst19-contaminated-pair-class",
        "out_dir": output,
        "result_path": output.join("contaminated_pair_class_audit.json"),
        "result": result.as_dict()?,
    }))
}

const CLI_HELP: &str = "从 source_catalog.csv 按类别选择代表锚点，复用统一的真实污染背景双源注入审计\n\nUsage: rst19-contaminated-pair-class FRAME --source-catalog CSV [OPTIONS]\n\nOptions:\n  --feature-class CLASS (repeatable)\n  --per-class N\n  --selection-metric filter_snr|flux_snr|peak\n  --selection-mode median|high\n  --separations-px PX ...\n  --ratios R ...\n  --total-peak-adu ADU\n  --psf-fwhm PIXELS\n  --max-sources N\n  --analysis-scope full_frame|local_roi\n  --roi-half-size-px PIXELS\n  --out-dir DIR\n  -h, --help\n";

pub fn run_cli(args: &[String]) -> i32 {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        println!("{CLI_HELP}");
        return 0;
    }
    match run_desktop(args) {
        Ok(output) => {
            println!(
                "contaminated pair class audit: {}",
                output["out_dir"].as_str().unwrap_or_default()
            );
            println!(
                "{}",
                output["result"]["injection"]["conclusion"]
                    .as_str()
                    .unwrap_or_default()
            );
            if let Some(rows) = output["result"]["selected_rows"].as_array() {
                for row in rows {
                    let class_name = row["feature_class"].as_str().unwrap_or_default();
                    let detection_id = row["detection_id"].as_i64().unwrap_or_default();
                    let metric = row["selection_metric"].as_str().unwrap_or_default();
                    let value = row["metric_value"].as_f64().unwrap_or_default();
                    let quality = if row["quality_passed"].as_bool().unwrap_or(false) {
                        "True"
                    } else {
                        "False"
                    };
                    let flags = row["flags"]
                        .as_str()
                        .filter(|value| !value.is_empty())
                        .unwrap_or("none");
                    println!("{class_name}: ID {detection_id}, {metric}={value:.4}, quality={quality}, flags={flags}");
                }
            }
            0
        }
        Err(error) => {
            eprintln!("rst19-contaminated-pair-class: {error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_uses_median_and_high_negative_ties_deterministically() {
        let root = std::env::temp_dir().join(format!("rst19-pair-class-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("catalog.csv");
        fs::write(
            &path,
            "detection_id,feature_class,feature_class_label,x,y,filter_snr,flux_snr,peak,quality_passed,flags\n1,compact_quality,compact,10,10,20,10,100,true,\n2,compact_quality,compact,30,30,40,20,200,true,\n3,linear_artifact,line,20,20,100,50,300,false,LINE_ARTIFACT\n4,linear_artifact,line,25,25,100,50,300,false,EDGE\n",
        )
        .unwrap();
        let median = select_feature_class_anchors(
            &path,
            Some(&["compact_quality".into(), "linear_artifact".into()]),
            1,
            "filter_snr",
            "median",
            Some((40, 40)),
            4.123,
            1.0,
            0.0,
        )
        .unwrap();
        assert_eq!(
            median
                .iter()
                .map(|row| row.detection_id)
                .collect::<Vec<_>>(),
            [1, 3]
        );
        assert_eq!(median[0].class_metric_median, 30.0);
        assert_eq!(median[1].selection_rank, 1);
        assert!(!median[1].quality_passed);
        let high = select_feature_class_anchors(
            &path,
            Some(&["linear_artifact".into()]),
            2,
            "filter_snr",
            "high",
            None,
            4.123,
            3.0,
            4.0,
        )
        .unwrap();
        assert_eq!(
            high.iter().map(|row| row.detection_id).collect::<Vec<_>>(),
            [3, 4]
        );
        assert_eq!((high[0].direction_x, high[0].direction_y), (0.6, 0.8));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn selection_rejects_invalid_catalog_contracts_and_out_of_bounds_pairs() {
        let root =
            std::env::temp_dir().join(format!("rst19-pair-class-errors-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("catalog.csv");
        fs::write(
            &path,
            "detection_id,feature_class,x,y,quality_passed,flags,filter_snr\n1,x,1,1,true,,3\n",
        )
        .unwrap();
        assert!(select_feature_class_anchors(
            &path,
            Some(&[]),
            1,
            "filter_snr",
            "median",
            None,
            4.0,
            1.0,
            0.0
        )
        .unwrap_err()
        .contains("feature_classes cannot be empty"));
        assert!(select_feature_class_anchors(
            &path,
            None,
            1,
            "filter_snr",
            "median",
            Some((5, 5)),
            4.0,
            1.0,
            0.0
        )
        .unwrap_err()
        .contains("no eligible in-bounds rows"));
        assert!(select_feature_class_anchors(
            &path, None, 1, "flux_snr", "median", None, 4.0, 1.0, 0.0
        )
        .unwrap_err()
        .contains("no eligible in-bounds rows"));
        fs::remove_dir_all(root).unwrap();
    }
}
