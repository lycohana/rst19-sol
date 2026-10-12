//! Population context for candidate-source nearest-neighbor spacing.
//!
//! Centroid and integer-peak coordinates are analyzed separately. A nearest
//! neighbor is detector geometry context, not evidence of physical identity.

use crate::{spatial::KdTree, Result};
use csv::StringRecord;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub const DEFAULT_RADII_PX: &[f64] = &[3.0, 4.0, 4.5, 5.0, 6.0];
const REQUIRED_FIELDS: &[&str] = &[
    "detection_id",
    "x",
    "y",
    "peak_x",
    "peak_y",
    "quality_passed",
    "feature_class",
];
const QUANTILE_LABELS: &[&str] = &[
    "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99",
];
const QUANTILE_LEVELS: &[f64] = &[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];

#[derive(Debug, Clone, Serialize)]
pub struct SourceSeparationThresholdRow {
    pub radius_px: f64,
    pub centroid_endpoint_count: usize,
    pub centroid_endpoint_fraction: f64,
    pub centroid_quality_endpoint_count: usize,
    pub centroid_quality_endpoint_fraction: f64,
    pub peak_endpoint_count: usize,
    pub peak_endpoint_fraction: f64,
    pub peak_quality_endpoint_count: usize,
    pub peak_quality_endpoint_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceSeparationClassRow {
    pub feature_class: String,
    pub source_count: usize,
    pub quality_count: usize,
    pub centroid_nn_median_px: Option<f64>,
    pub centroid_nn_p05_px: Option<f64>,
    pub centroid_nn_p95_px: Option<f64>,
    pub centroid_nn_le_4px_count: usize,
    pub peak_nn_median_px: Option<f64>,
    pub peak_nn_p05_px: Option<f64>,
    pub peak_nn_p95_px: Option<f64>,
    pub peak_nn_le_4px_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceSeparationTargetRow {
    pub detection_id: i64,
    pub feature_class: String,
    pub quality_passed: bool,
    pub x: f64,
    pub y: f64,
    pub peak_x: f64,
    pub peak_y: f64,
    pub centroid_nn_distance_px: Option<f64>,
    pub centroid_nn_detection_id: Option<i64>,
    pub centroid_nn_percentile: Option<f64>,
    pub peak_nn_distance_px: Option<f64>,
    pub peak_nn_detection_id: Option<i64>,
    pub peak_nn_percentile: Option<f64>,
}

#[derive(Debug, Clone)]
struct Source {
    detection_id: i64,
    x: f64,
    y: f64,
    peak_x: f64,
    peak_y: f64,
    quality_passed: bool,
    feature_class: String,
}

#[derive(Debug, Clone)]
pub struct SourceSeparationAuditResult {
    pub catalog_path: String,
    pub source_count: usize,
    pub quality_count: usize,
    pub thresholds: Vec<SourceSeparationThresholdRow>,
    pub classes: Vec<SourceSeparationClassRow>,
    pub targets: Vec<SourceSeparationTargetRow>,
    pub centroid_quantiles_px: BTreeMap<String, f64>,
    pub peak_quantiles_px: BTreeMap<String, f64>,
    pub parameters: Value,
    pub conclusion: String,
}

impl SourceSeparationAuditResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "source_count": self.source_count,
            "quality_count": self.quality_count,
            "thresholds": self.thresholds,
            "classes": self.classes,
            "targets": self.targets,
            "centroid_quantiles_px": self.centroid_quantiles_px,
            "peak_quantiles_px": self.peak_quantiles_px,
            "parameters": self.parameters,
            "conclusion": self.conclusion,
        })
    }
}

fn field(record: &StringRecord, indexes: &BTreeMap<String, usize>, name: &str) -> String {
    indexes
        .get(name)
        .and_then(|index| record.get(*index))
        .unwrap_or_default()
        .to_owned()
}

fn load_catalog(path: &Path) -> Result<Vec<Source>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|error| error.to_string())?;
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let normalized: Vec<_> = headers
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
    let indexes: BTreeMap<_, _> = normalized
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let present: BTreeSet<_> = normalized.iter().cloned().collect();
    let mut missing: Vec<_> = REQUIRED_FIELDS
        .iter()
        .filter(|name| !present.contains(**name))
        .copied()
        .collect();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing fields: {}",
            missing.join(", ")
        ));
    }
    let records: Vec<_> = reader
        .records()
        .map(|record| record.map_err(|error| error.to_string()))
        .collect::<Result<_>>()?;
    if records.len() < 2 {
        return Err("source catalog must contain at least two sources".into());
    }
    records
        .iter()
        .map(|record| {
            let parsed = (|| -> std::result::Result<Source, ()> {
                let detection_id = field(record, &indexes, "detection_id")
                    .trim()
                    .parse::<i64>()
                    .map_err(|_| ())?;
                let x = field(record, &indexes, "x")
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| ())?;
                let y = field(record, &indexes, "y")
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| ())?;
                let peak_x = field(record, &indexes, "peak_x")
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| ())?;
                let peak_y = field(record, &indexes, "peak_y")
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| ())?;
                let raw_quality = field(record, &indexes, "quality_passed");
                let quality_passed = match raw_quality.trim().to_ascii_lowercase().as_str() {
                    "true" | "1" | "yes" | "y" => true,
                    "false" | "0" | "no" | "n" => false,
                    _ => return Err(()),
                };
                Ok(Source {
                    detection_id,
                    x,
                    y,
                    peak_x,
                    peak_y,
                    quality_passed,
                    feature_class: field(record, &indexes, "feature_class"),
                })
            })();
            parsed.map_err(|_| "source catalog contains an invalid numeric or boolean row".into())
        })
        .collect()
}

fn validate_radii(radii_px: &[f64]) -> Result<Vec<f64>> {
    let mut radii = radii_px.to_vec();
    radii.sort_by(f64::total_cmp);
    radii.dedup_by(|left, right| *left == *right);
    if radii.is_empty()
        || radii
            .iter()
            .any(|radius| !radius.is_finite() || *radius <= 0.0)
    {
        return Err("radii_px must contain positive values".into());
    }
    Ok(radii)
}

fn nearest_other(points: &[[f64; 2]]) -> Result<(Vec<f64>, Vec<usize>)> {
    if points.len() < 2 {
        return Err("points must be an N×2 array with N≥2".into());
    }
    let tree = KdTree::new(points)?;
    let mut distances = Vec::with_capacity(points.len());
    let mut indexes = Vec::with_capacity(points.len());
    for (index, point) in points.iter().copied().enumerate() {
        let (nearest_index, distance) = tree
            .nearest_excluding(point, index)
            .ok_or("nearest neighbor query returned no other point")?;
        indexes.push(nearest_index);
        distances.push(distance);
    }
    Ok((distances, indexes))
}

fn quantile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let position = (sorted.len() - 1) as f64 * probability;
    let left = position.floor() as usize;
    let right = position.ceil() as usize;
    let fraction = position - left as f64;
    Some(sorted[left] + (sorted[right] - sorted[left]) * fraction)
}

fn median(values: &[f64]) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    quantile(&sorted, 0.5)
}

fn quantile_map(values: &[f64]) -> BTreeMap<String, f64> {
    let mut finite: Vec<_> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    finite.sort_by(f64::total_cmp);
    QUANTILE_LABELS
        .iter()
        .zip(QUANTILE_LEVELS)
        .filter_map(|(label, probability)| {
            quantile(&finite, *probability).map(|value| ((*label).to_owned(), value))
        })
        .collect()
}

pub fn run(
    catalog_csv: &Path,
    target_ids: &[i64],
    radii_px: &[f64],
) -> Result<SourceSeparationAuditResult> {
    let rows = load_catalog(catalog_csv)?;
    let mut targets = Vec::new();
    for target in target_ids {
        if !targets.contains(target) {
            targets.push(*target);
        }
    }
    let radii = validate_radii(radii_px)?;
    let detection_ids: Vec<_> = rows.iter().map(|row| row.detection_id).collect();
    let mut unique_ids = BTreeSet::new();
    if detection_ids.iter().any(|id| !unique_ids.insert(*id)) {
        return Err("source catalog contains duplicate detection_id values".into());
    }
    let centroid_points: Vec<_> = rows.iter().map(|row| [row.x, row.y]).collect();
    let peak_points: Vec<_> = rows.iter().map(|row| [row.peak_x, row.peak_y]).collect();
    if centroid_points
        .iter()
        .chain(&peak_points)
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err("source catalog coordinates must be finite".into());
    }
    let (centroid_distance, centroid_index) = nearest_other(&centroid_points)?;
    let (peak_distance, peak_index) = nearest_other(&peak_points)?;
    let source_count = rows.len();
    let quality_count = rows.iter().filter(|row| row.quality_passed).count();

    let thresholds = radii
        .iter()
        .map(|radius| {
            let centroid_endpoints = centroid_distance
                .iter()
                .filter(|distance| **distance <= *radius)
                .count();
            let centroid_quality = rows
                .iter()
                .zip(&centroid_distance)
                .filter(|(row, distance)| row.quality_passed && **distance <= *radius)
                .count();
            let peak_endpoints = peak_distance
                .iter()
                .filter(|distance| **distance <= *radius)
                .count();
            let peak_quality = rows
                .iter()
                .zip(&peak_distance)
                .filter(|(row, distance)| row.quality_passed && **distance <= *radius)
                .count();
            SourceSeparationThresholdRow {
                radius_px: *radius,
                centroid_endpoint_count: centroid_endpoints,
                centroid_endpoint_fraction: centroid_endpoints as f64 / source_count as f64,
                centroid_quality_endpoint_count: centroid_quality,
                centroid_quality_endpoint_fraction: if quality_count == 0 {
                    0.0
                } else {
                    centroid_quality as f64 / quality_count as f64
                },
                peak_endpoint_count: peak_endpoints,
                peak_endpoint_fraction: peak_endpoints as f64 / source_count as f64,
                peak_quality_endpoint_count: peak_quality,
                peak_quality_endpoint_fraction: if quality_count == 0 {
                    0.0
                } else {
                    peak_quality as f64 / quality_count as f64
                },
            }
        })
        .collect();

    let classes: BTreeSet<_> = rows.iter().map(|row| row.feature_class.as_str()).collect();
    let class_rows: Vec<_> = classes
        .into_iter()
        .map(|feature_class| {
            let indexes: Vec<_> = rows
                .iter()
                .enumerate()
                .filter(|(_, row)| row.feature_class == feature_class)
                .map(|(index, _)| index)
                .collect();
            let class_centroid: Vec<_> = indexes
                .iter()
                .map(|index| centroid_distance[*index])
                .collect();
            let class_peak: Vec<_> = indexes.iter().map(|index| peak_distance[*index]).collect();
            SourceSeparationClassRow {
                feature_class: feature_class.to_owned(),
                source_count: indexes.len(),
                quality_count: indexes
                    .iter()
                    .filter(|index| rows[**index].quality_passed)
                    .count(),
                centroid_nn_median_px: median(&class_centroid),
                centroid_nn_p05_px: {
                    let mut sorted = class_centroid.clone();
                    sorted.sort_by(f64::total_cmp);
                    quantile(&sorted, 0.05)
                },
                centroid_nn_p95_px: {
                    let mut sorted = class_centroid.clone();
                    sorted.sort_by(f64::total_cmp);
                    quantile(&sorted, 0.95)
                },
                centroid_nn_le_4px_count: class_centroid
                    .iter()
                    .filter(|value| **value <= 4.0)
                    .count(),
                peak_nn_median_px: median(&class_peak),
                peak_nn_p05_px: {
                    let mut sorted = class_peak.clone();
                    sorted.sort_by(f64::total_cmp);
                    quantile(&sorted, 0.05)
                },
                peak_nn_p95_px: {
                    let mut sorted = class_peak.clone();
                    sorted.sort_by(f64::total_cmp);
                    quantile(&sorted, 0.95)
                },
                peak_nn_le_4px_count: class_peak.iter().filter(|value| **value <= 4.0).count(),
            }
        })
        .collect();

    let id_to_index: BTreeMap<_, _> = detection_ids
        .iter()
        .enumerate()
        .map(|(index, detection_id)| (*detection_id, index))
        .collect();
    let requested_ids: BTreeSet<_> = targets.iter().copied().collect();
    let catalog_ids: BTreeSet<_> = id_to_index.keys().copied().collect();
    let missing: Vec<_> = requested_ids.difference(&catalog_ids).copied().collect();
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
    let target_rows: Vec<_> = targets
        .iter()
        .map(|target| {
            let index = id_to_index[target];
            let centroid_percentile = centroid_distance
                .iter()
                .filter(|distance| **distance <= centroid_distance[index])
                .count() as f64
                / source_count as f64;
            let peak_percentile = peak_distance
                .iter()
                .filter(|distance| **distance <= peak_distance[index])
                .count() as f64
                / source_count as f64;
            SourceSeparationTargetRow {
                detection_id: *target,
                feature_class: rows[index].feature_class.clone(),
                quality_passed: rows[index].quality_passed,
                x: rows[index].x,
                y: rows[index].y,
                peak_x: rows[index].peak_x,
                peak_y: rows[index].peak_y,
                centroid_nn_distance_px: Some(centroid_distance[index]),
                centroid_nn_detection_id: Some(detection_ids[centroid_index[index]]),
                centroid_nn_percentile: Some(centroid_percentile),
                peak_nn_distance_px: Some(peak_distance[index]),
                peak_nn_detection_id: Some(detection_ids[peak_index[index]]),
                peak_nn_percentile: Some(peak_percentile),
            }
        })
        .collect();
    let conclusion = "nearest-neighbor counts are candidate-geometry controls: they show how often local responses approach a radius, but do not establish independent stars, double-source identity, or physical truth. Peak and centroid coordinates must be interpreted separately.".to_owned();
    Ok(SourceSeparationAuditResult {
        catalog_path: catalog_csv.to_string_lossy().into_owned(),
        source_count,
        quality_count,
        thresholds,
        classes: class_rows,
        targets: target_rows,
        centroid_quantiles_px: quantile_map(&centroid_distance),
        peak_quantiles_px: quantile_map(&peak_distance),
        parameters: json!({
            "radii_px": radii,
            "coordinate_note": "x/y are measured centroids; peak_x/peak_y are integer peak coordinates",
            "endpoint_fraction_note": "each source contributes only its nearest-other endpoint; this is not an unordered pair count",
        }),
        conclusion,
    })
}

const THRESHOLD_FIELDS: &[&str] = &[
    "radius_px",
    "centroid_endpoint_count",
    "centroid_endpoint_fraction",
    "centroid_quality_endpoint_count",
    "centroid_quality_endpoint_fraction",
    "peak_endpoint_count",
    "peak_endpoint_fraction",
    "peak_quality_endpoint_count",
    "peak_quality_endpoint_fraction",
];
const CLASS_FIELDS: &[&str] = &[
    "feature_class",
    "source_count",
    "quality_count",
    "centroid_nn_median_px",
    "centroid_nn_p05_px",
    "centroid_nn_p95_px",
    "centroid_nn_le_4px_count",
    "peak_nn_median_px",
    "peak_nn_p05_px",
    "peak_nn_p95_px",
    "peak_nn_le_4px_count",
];
const TARGET_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "quality_passed",
    "x",
    "y",
    "peak_x",
    "peak_y",
    "centroid_nn_distance_px",
    "centroid_nn_detection_id",
    "centroid_nn_percentile",
    "peak_nn_distance_px",
    "peak_nn_detection_id",
    "peak_nn_percentile",
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
    let file = fs::File::create(path).map_err(|error| error.to_string())?;
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

pub fn write_artifacts(result: &SourceSeparationAuditResult, output: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    let thresholds: Vec<_> = result
        .thresholds
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<_, _>>()
        .map_err(|error| error.to_string())?;
    let classes: Vec<_> = result
        .classes
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<_, _>>()
        .map_err(|error| error.to_string())?;
    let targets: Vec<_> = result
        .targets
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<_, _>>()
        .map_err(|error| error.to_string())?;
    write_records(
        &output.join("source_separation_thresholds.csv"),
        THRESHOLD_FIELDS,
        &thresholds,
    )?;
    write_records(
        &output.join("source_separation_classes.csv"),
        CLASS_FIELDS,
        &classes,
    )?;
    write_records(
        &output.join("source_separation_targets.csv"),
        TARGET_FIELDS,
        &targets,
    )?;
    let mut payload = result.as_dict();
    payload["outputs"] = json!([
        "source_separation_thresholds.csv",
        "source_separation_classes.csv",
        "source_separation_targets.csv",
        "source_separation_summary.json",
    ]);
    fs::write(
        output.join("source_separation_summary.json"),
        serde_json::to_vec_pretty(&payload).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut positional = Vec::new();
    let mut target_ids = Vec::new();
    let mut radii = Vec::new();
    let mut output = PathBuf::from("tmp/source-separation-audit");
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
                    target_ids.push(value.parse::<i64>().map_err(|error| error.to_string())?);
                } else {
                    output = PathBuf::from(value);
                }
            }
            "--radii-px" => {
                if !inline.is_empty() {
                    radii.push(inline.parse::<f64>().map_err(|error| error.to_string())?);
                }
                while index + 1 < args.len() && !args[index + 1].starts_with("--") {
                    index += 1;
                    radii.push(
                        args[index]
                            .parse::<f64>()
                            .map_err(|error| error.to_string())?,
                    );
                }
            }
            _ if name.starts_with('-') => return Err(format!("unknown argument {name}")),
            _ => positional.push(PathBuf::from(name)),
        }
        index += 1;
    }
    if positional.len() != 1 {
        return Err("expected one source catalog path".into());
    }
    if radii.is_empty() {
        radii = DEFAULT_RADII_PX.to_vec();
    }
    let result = run(&positional[0], &target_ids, &radii)?;
    write_artifacts(&result, &output)?;
    Ok(
        json!({"tool":"rst19-source-separation-audit", "out_dir":output,
        "result_path":output.join("source_separation_summary.json"), "result":result.as_dict()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn centroid_peak_neighbors_and_artifacts_keep_separate_contracts() {
        let root = std::env::temp_dir().join(format!(
            "rst19-source-separation-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        fs::write(&catalog, "\u{feff}detection_id,x,y,peak_x,peak_y,quality_passed,feature_class\r\n1,0,0,0,0,True,compact_quality\r\n2,2.5,0,3,0,True,compact_quality\r\n3,10,0,10,0,False,weak_or_background\r\n4,10,5,10,5,False,shape_outlier\r\n").unwrap();
        let result = run(&catalog, &[1, 2], &[6.0, 3.0, 4.0, 3.0]).unwrap();
        assert_eq!(
            result
                .thresholds
                .iter()
                .map(|row| row.radius_px)
                .collect::<Vec<_>>(),
            [3.0, 4.0, 6.0]
        );
        assert_eq!(result.thresholds[0].centroid_endpoint_count, 2);
        assert_eq!(result.targets[0].centroid_nn_detection_id, Some(2));
        assert_eq!(result.targets[1].peak_nn_distance_px, Some(3.0));
        let output = root.join("out");
        write_artifacts(&result, &output).unwrap();
        assert!(output.join("source_separation_thresholds.csv").is_file());
        assert!(output.join("source_separation_classes.csv").is_file());
        assert!(output.join("source_separation_targets.csv").is_file());
        assert!(output.join("source_separation_summary.json").is_file());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn missing_target_and_duplicate_ids_match_catalog_contract() {
        let root = std::env::temp_dir().join(format!(
            "rst19-source-separation-errors-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        fs::write(&catalog, "detection_id,x,y,peak_x,peak_y,quality_passed,feature_class\n1,0,0,0,0,true,a\n2,1,0,1,0,false,b\n").unwrap();
        assert!(run(&catalog, &[99], DEFAULT_RADII_PX)
            .unwrap_err()
            .starts_with("target detection_id values are not in source catalog"));
        fs::write(&catalog, "detection_id,x,y,peak_x,peak_y,quality_passed,feature_class\n1,0,0,0,0,true,a\n1,1,0,1,0,false,b\n").unwrap();
        assert_eq!(
            run(&catalog, &[], DEFAULT_RADII_PX).unwrap_err(),
            "source catalog contains duplicate detection_id values"
        );
        let _ = fs::remove_dir_all(root);
    }
}
