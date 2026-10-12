use crate::Result;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

pub const DEFAULT_CORRELATION_FEATURES: [&str; 9] = [
    "peak",
    "flux_snr",
    "filter_snr",
    "fwhm",
    "ellipticity",
    "sharpness",
    "footprint_pixels",
    "psf_support_pixels",
    "centroid_shift_px",
];

const INTERPRETATION_BOUNDARY: &str = "descriptive source-field dependence and class medians; not causality, classifier accuracy, precision, FDR, or star probability";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureCorrelationRow {
    pub feature_a: String,
    pub feature_b: String,
    pub sample_count: usize,
    pub spearman_rho: Option<f64>,
    pub absolute_spearman_rho: Option<f64>,
    pub strong_threshold: f64,
    pub strong_absolute_correlation: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureClassMedianRow {
    pub feature_class: String,
    pub feature_class_label: String,
    pub sample_count: usize,
    pub feature_names: Vec<String>,
    pub medians: Vec<Option<f64>>,
}

impl FeatureClassMedianRow {
    fn as_dict(&self) -> Value {
        let mut result = Map::new();
        result.insert("feature_class".into(), json!(self.feature_class));
        result.insert(
            "feature_class_label".into(),
            json!(self.feature_class_label),
        );
        result.insert("sample_count".into(), json!(self.sample_count));
        for (feature, median) in self.feature_names.iter().zip(&self.medians) {
            result.insert(feature.clone(), json!(median));
        }
        Value::Object(result)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureClassCorrelationRow {
    pub feature_class: String,
    pub feature_class_label: String,
    pub feature_a: String,
    pub feature_b: String,
    pub sample_count: usize,
    pub spearman_rho: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureCorrelationAuditResult {
    pub catalog_path: String,
    pub features: Vec<String>,
    pub strong_threshold: f64,
    pub correlation_rows: Vec<FeatureCorrelationRow>,
    pub class_median_rows: Vec<FeatureClassMedianRow>,
    pub class_correlation_rows: Vec<FeatureClassCorrelationRow>,
}

impl FeatureCorrelationAuditResult {
    pub fn as_dict(&self) -> Value {
        let strong_correlations: Vec<&FeatureCorrelationRow> = self
            .correlation_rows
            .iter()
            .filter(|row| row.strong_absolute_correlation == Some(true))
            .collect();
        json!({
            "catalog_path": self.catalog_path,
            "features": self.features,
            "strong_threshold": self.strong_threshold,
            "correlations": self.correlation_rows,
            "strong_correlations": strong_correlations,
            "class_medians": self.class_median_rows.iter().map(FeatureClassMedianRow::as_dict).collect::<Vec<_>>(),
            "class_correlations": self.class_correlation_rows,
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        })
    }
}

#[derive(Clone, Debug)]
struct CatalogRecord {
    feature_class: String,
    feature_class_label: String,
    values: Vec<Option<f64>>,
}

#[derive(Clone, Debug)]
struct ClassGroup {
    feature_class: String,
    feature_class_label: String,
    record_indices: Vec<usize>,
}

fn parse_finite(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
}

fn normalize_features(features: &[String]) -> Result<Vec<String>> {
    let mut normalized = Vec::new();
    for feature in features {
        let feature = feature.trim();
        if !feature.is_empty() && !normalized.iter().any(|existing| existing == feature) {
            normalized.push(feature.to_owned());
        }
    }
    if normalized.len() < 2 {
        return Err("features must contain at least two non-empty names".into());
    }
    Ok(normalized)
}

fn read_catalog(path: &Path, features: &[String]) -> Result<Vec<CatalogRecord>> {
    let bytes = fs::read(path).map_err(|error| format!("无法读取源级 CSV {path:?}: {error}"))?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let names: std::collections::HashSet<&str> = headers.iter().collect();
    let mut required = vec!["feature_class".to_owned()];
    required.extend(features.iter().cloned());
    let mut missing: Vec<String> = required
        .into_iter()
        .filter(|name| !names.contains(name.as_str()))
        .collect();
    missing.sort();
    missing.dedup();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let mut indices = HashMap::new();
    for (index, name) in headers.iter().enumerate() {
        indices.insert(name, index);
    }
    let feature_indices: Vec<usize> = features
        .iter()
        .map(|feature| indices[feature.as_str()])
        .collect();
    let class_index = indices["feature_class"];
    let label_index = indices.get("feature_class_label").copied();

    let mut records = Vec::new();
    for row in reader.records() {
        let row = row.map_err(|error| error.to_string())?;
        let get = |index: usize| row.get(index).unwrap_or("None");
        let feature_class = get(class_index).trim().to_owned();
        let feature_class_label = label_index
            .map(|index| get(index).trim().to_owned())
            .unwrap_or_default();
        let values = feature_indices
            .iter()
            .map(|index| parse_finite(get(*index)))
            .collect();
        records.push(CatalogRecord {
            feature_class,
            feature_class_label,
            values,
        });
    }
    Ok(records)
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|left, right| values[*left].partial_cmp(&values[*right]).unwrap());
    let mut result = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let average_rank = ((start + 1 + end) as f64) / 2.0;
        for index in start..end {
            result[order[index]] = average_rank;
        }
        start = end;
    }
    result
}

fn spearman(values_a: &[f64], values_b: &[f64]) -> Option<f64> {
    if values_a.len() < 2 || values_a.len() != values_b.len() {
        return None;
    }
    let ranks_a = ranks(values_a);
    let ranks_b = ranks(values_b);
    let mean_a = ranks_a.iter().sum::<f64>() / ranks_a.len() as f64;
    let mean_b = ranks_b.iter().sum::<f64>() / ranks_b.len() as f64;
    let centered_a: Vec<f64> = ranks_a.iter().map(|rank| rank - mean_a).collect();
    let centered_b: Vec<f64> = ranks_b.iter().map(|rank| rank - mean_b).collect();
    let sum_a = centered_a.iter().map(|value| value * value).sum::<f64>();
    let sum_b = centered_b.iter().map(|value| value * value).sum::<f64>();
    let denominator = (sum_a * sum_b).sqrt();
    if denominator <= 0.0 {
        return None;
    }
    let covariance = centered_a
        .iter()
        .zip(&centered_b)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    Some(covariance / denominator)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[middle])
    } else {
        Some((values[middle - 1] + values[middle]) / 2.0)
    }
}

fn correlated_values(
    records: &[CatalogRecord],
    indices: &[usize],
    first_feature: usize,
    second_feature: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut values_a = Vec::new();
    let mut values_b = Vec::new();
    for index in indices {
        if let (Some(first), Some(second)) = (
            records[*index].values[first_feature],
            records[*index].values[second_feature],
        ) {
            values_a.push(first);
            values_b.push(second);
        }
    }
    (values_a, values_b)
}

pub fn run(
    catalog_path: &Path,
    features: &[String],
    strong_threshold: f64,
) -> Result<FeatureCorrelationAuditResult> {
    if !(0.0 < strong_threshold && strong_threshold <= 1.0) {
        return Err("strong_threshold must be in (0, 1]".into());
    }
    let features = normalize_features(features)?;
    let records = read_catalog(catalog_path, &features)?;
    let mut correlation_rows = Vec::new();
    for first_feature in 0..features.len() {
        for second_feature in first_feature + 1..features.len() {
            let indices: Vec<usize> = (0..records.len()).collect();
            let (values_a, values_b) =
                correlated_values(&records, &indices, first_feature, second_feature);
            let rho = spearman(&values_a, &values_b);
            let absolute = rho.map(f64::abs);
            correlation_rows.push(FeatureCorrelationRow {
                feature_a: features[first_feature].clone(),
                feature_b: features[second_feature].clone(),
                sample_count: values_a.len(),
                spearman_rho: rho,
                absolute_spearman_rho: absolute,
                strong_threshold,
                strong_absolute_correlation: absolute.map(|value| value >= strong_threshold),
            });
        }
    }

    let mut class_groups = Vec::<ClassGroup>::new();
    let mut class_indices = HashMap::<String, usize>::new();
    for (record_index, record) in records.iter().enumerate() {
        if record.feature_class.is_empty() {
            continue;
        }
        let group_index = if let Some(index) = class_indices.get(&record.feature_class) {
            *index
        } else {
            let index = class_groups.len();
            class_indices.insert(record.feature_class.clone(), index);
            class_groups.push(ClassGroup {
                feature_class: record.feature_class.clone(),
                feature_class_label: record.feature_class_label.clone(),
                record_indices: Vec::new(),
            });
            index
        };
        class_groups[group_index].record_indices.push(record_index);
    }

    let feature_names: Vec<String> = features.clone();
    let mut class_median_rows = Vec::new();
    let mut class_correlation_rows = Vec::new();
    for group in &class_groups {
        let medians = (0..features.len())
            .map(|feature_index| {
                let mut values: Vec<f64> = group
                    .record_indices
                    .iter()
                    .filter_map(|record_index| records[*record_index].values[feature_index])
                    .collect();
                median(&mut values)
            })
            .collect();
        class_median_rows.push(FeatureClassMedianRow {
            feature_class: group.feature_class.clone(),
            feature_class_label: group.feature_class_label.clone(),
            sample_count: group.record_indices.len(),
            feature_names: feature_names.clone(),
            medians,
        });
        for first_feature in 0..features.len() {
            for second_feature in first_feature + 1..features.len() {
                let (values_a, values_b) = correlated_values(
                    &records,
                    &group.record_indices,
                    first_feature,
                    second_feature,
                );
                class_correlation_rows.push(FeatureClassCorrelationRow {
                    feature_class: group.feature_class.clone(),
                    feature_class_label: group.feature_class_label.clone(),
                    feature_a: features[first_feature].clone(),
                    feature_b: features[second_feature].clone(),
                    sample_count: values_a.len(),
                    spearman_rho: spearman(&values_a, &values_b),
                });
            }
        }
    }

    Ok(FeatureCorrelationAuditResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        features,
        strong_threshold,
        correlation_rows,
        class_median_rows,
        class_correlation_rows,
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

fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(String::new, python_float)
}

fn optional_bool(value: Option<bool>) -> String {
    match value {
        Some(true) => "True".to_owned(),
        Some(false) => "False".to_owned(),
        None => String::new(),
    }
}

fn write_csv(path: &Path, headers: &[String], rows: Vec<Vec<String>>) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(headers)
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

pub fn write_artifacts(
    result: &FeatureCorrelationAuditResult,
    output_dir: &Path,
) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    write_csv(
        &output_dir.join("feature_correlations.csv"),
        &[
            "feature_a",
            "feature_b",
            "sample_count",
            "spearman_rho",
            "absolute_spearman_rho",
            "strong_threshold",
            "strong_absolute_correlation",
        ]
        .map(str::to_owned),
        result
            .correlation_rows
            .iter()
            .map(|row| {
                vec![
                    row.feature_a.clone(),
                    row.feature_b.clone(),
                    row.sample_count.to_string(),
                    optional_float(row.spearman_rho),
                    optional_float(row.absolute_spearman_rho),
                    python_float(row.strong_threshold),
                    optional_bool(row.strong_absolute_correlation),
                ]
            })
            .collect(),
    )?;
    let mut median_headers = vec![
        "feature_class".to_owned(),
        "feature_class_label".to_owned(),
        "sample_count".to_owned(),
    ];
    median_headers.extend(result.features.iter().cloned());
    write_csv(
        &output_dir.join("feature_class_medians.csv"),
        &median_headers,
        result
            .class_median_rows
            .iter()
            .map(|row| {
                let mut values = vec![
                    row.feature_class.clone(),
                    row.feature_class_label.clone(),
                    row.sample_count.to_string(),
                ];
                values.extend(row.medians.iter().copied().map(optional_float));
                values
            })
            .collect(),
    )?;
    write_csv(
        &output_dir.join("feature_class_correlations.csv"),
        &[
            "feature_class",
            "feature_class_label",
            "feature_a",
            "feature_b",
            "sample_count",
            "spearman_rho",
        ]
        .map(str::to_owned),
        result
            .class_correlation_rows
            .iter()
            .map(|row| {
                vec![
                    row.feature_class.clone(),
                    row.feature_class_label.clone(),
                    row.feature_a.clone(),
                    row.feature_b.clone(),
                    row.sample_count.to_string(),
                    optional_float(row.spearman_rho),
                ]
            })
            .collect(),
    )?;
    let json_bytes =
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?;
    fs::write(
        output_dir.join("feature_correlation_audit.json"),
        json_bytes,
    )
    .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut catalog_path = None;
    let mut output_dir = PathBuf::from("tmp/feature-correlation-audit");
    let mut features = DEFAULT_CORRELATION_FEATURES
        .iter()
        .map(|feature| (*feature).to_owned())
        .collect::<Vec<_>>();
    let mut strong_threshold = 0.45;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline_value) = arg
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((arg.as_str(), None));
        match name {
            "--out-dir" => {
                let value = if let Some(value) = inline_value {
                    value
                } else {
                    index += 1;
                    args.get(index).ok_or("--out-dir requires a value")?
                };
                output_dir = PathBuf::from(value);
            }
            "--strong-threshold" => {
                index += usize::from(inline_value.is_none());
                strong_threshold = inline_value
                    .unwrap_or_else(|| args.get(index).map(String::as_str).unwrap_or(""))
                    .parse::<f64>()
                    .map_err(|error| format!("invalid strong threshold: {error}"))?;
            }
            "--features" => {
                let mut values = Vec::new();
                if let Some(value) = inline_value {
                    if !value.is_empty() {
                        values.push(value.to_owned());
                    }
                }
                if inline_value.is_none() || index + 1 < args.len() {
                    while index + 1 < args.len() && !args[index + 1].starts_with('-') {
                        index += 1;
                        values.push(args[index].clone());
                    }
                }
                features = values;
            }
            "--" => {
                index += 1;
                while index < args.len() {
                    if catalog_path.replace(PathBuf::from(&args[index])).is_some() {
                        return Err("only one source catalog is allowed".into());
                    }
                    index += 1;
                }
                break;
            }
            _ if arg.starts_with('-') => return Err(format!("unknown argument {arg}")),
            _ => {
                if catalog_path.replace(PathBuf::from(arg)).is_some() {
                    return Err("only one source catalog is allowed".into());
                }
            }
        }
        index += 1;
    }
    let catalog_path = catalog_path.ok_or("a source catalog CSV is required")?;
    let result = run(&catalog_path, &features, strong_threshold)?;
    write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-feature-correlation",
        "out_dir": output_dir,
        "result_path": output_dir.join("feature_correlation_audit.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tied_ranks_and_first_seen_class_order_match_the_audit_contract() {
        let root =
            std::env::temp_dir().join(format!("rst19-feature-correlation-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        fs::write(
            &catalog,
            "feature_class,feature_class_label,a,b,c\ncompact,紧凑,1,1,5\ncompact,compact,2,2,4\nweak,弱,2,3,3\nweak,弱,4,4,2\n",
        )
        .unwrap();
        let result = run(&catalog, &["a".into(), "b".into(), "c".into()], 0.75).unwrap();
        assert_eq!(
            result.correlation_rows[0].spearman_rho,
            Some(0.9486832980505138)
        );
        assert_eq!(result.class_median_rows[0].feature_class, "compact");
        assert_eq!(result.class_median_rows[0].feature_class_label, "紧凑");
        assert_eq!(
            result.class_median_rows[0].medians,
            [Some(1.5), Some(1.5), Some(4.5)]
        );
    }

    #[test]
    fn insufficient_or_constant_samples_have_null_correlation() {
        let root = std::env::temp_dir().join(format!(
            "rst19-feature-correlation-null-{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        fs::write(&catalog, "feature_class,a,b\nx,1,2\nx,1,3\ny,NaN,5\n").unwrap();
        let result = run(&catalog, &["a".into(), "b".into()], 0.45).unwrap();
        assert_eq!(result.correlation_rows[0].spearman_rho, None);
        assert_eq!(result.correlation_rows[0].sample_count, 2);
        assert_eq!(result.class_correlation_rows[1].sample_count, 0);
    }
}
