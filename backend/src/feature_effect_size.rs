use crate::Result;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

pub const DEFAULT_EFFECT_FEATURES: [&str; 11] = [
    "flux_snr",
    "filter_snr",
    "peak",
    "fwhm",
    "ellipticity",
    "sharpness",
    "footprint_pixels",
    "psf_support_pixels",
    "centroid_shift_px",
    "repeated_code_count",
    "range_anomaly_pixel_count",
];

const INTERPRETATION_BOUNDARY: &str = "descriptive detector-rule effect sizes; not physical truth, classifier accuracy, precision, FDR, or star probability";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureEffectSizeRow {
    pub reference_class: String,
    pub comparison_class: String,
    pub feature: String,
    pub reference_count: usize,
    pub comparison_count: usize,
    pub reference_median: f64,
    pub comparison_median: f64,
    pub reference_iqr: f64,
    pub comparison_iqr: f64,
    pub auc_reference_greater: f64,
    pub cliffs_delta: f64,
    pub robust_median_gap: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureEffectSizeResult {
    pub catalog_path: String,
    pub reference_class: String,
    pub comparison_classes: Vec<String>,
    pub features: Vec<String>,
    pub rows: Vec<FeatureEffectSizeRow>,
    pub top_features_by_class: BTreeMap<String, Vec<String>>,
}

impl FeatureEffectSizeResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "reference_class": self.reference_class,
            "comparison_classes": self.comparison_classes,
            "features": self.features,
            "rows": self.rows,
            "top_features_by_class": self.top_features_by_class,
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        })
    }
}

struct CatalogRecord {
    feature_class: String,
    values: BTreeMap<String, Option<f64>>,
}

fn finite_float(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
}

fn field(record: &csv::StringRecord, index: usize) -> &str {
    record.get(index).unwrap_or("None")
}

fn unique_trimmed(values: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty() && seen.insert((*value).to_owned()))
        .map(str::to_owned)
        .collect()
}

fn read_catalog(path: &Path, required_features: &[String]) -> Result<Vec<CatalogRecord>> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let required = std::iter::once("feature_class".to_owned())
        .chain(required_features.iter().cloned())
        .collect::<HashSet<_>>();
    let mut indices = BTreeMap::new();
    for required_name in &required {
        if let Some(index) = headers
            .iter()
            .enumerate()
            .filter_map(|(index, header)| (header == required_name).then_some(index))
            .last()
        {
            indices.insert(required_name.clone(), index);
        }
    }
    let mut missing = required
        .iter()
        .filter(|name| !indices.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    missing.sort();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let feature_names = required_features
        .iter()
        .filter(|feature| !feature.trim().is_empty())
        .cloned()
        .collect::<HashSet<_>>();
    let mut records = Vec::new();
    for row in reader.records() {
        let row = row.map_err(|error| error.to_string())?;
        let mut values = BTreeMap::new();
        for feature in &feature_names {
            values.insert(feature.clone(), finite_float(field(&row, indices[feature])));
        }
        records.push(CatalogRecord {
            feature_class: field(&row, indices["feature_class"]).trim().to_owned(),
            values,
        });
    }
    Ok(records)
}

fn values_for(records: &[&CatalogRecord], feature: &str) -> Vec<f64> {
    records
        .iter()
        .filter_map(|record| record.values.get(feature).copied().flatten())
        .collect()
}

fn quantile(values: &[f64], probability: f64) -> Result<f64> {
    if values.is_empty() {
        return Err("cannot compute a quantile for an empty sequence".into());
    }
    let position = (values.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return Ok(values[lower]);
    }
    let weight = position - lower as f64;
    Ok(values[lower] * (1.0 - weight) + values[upper] * weight)
}

fn median(values: &[f64]) -> Result<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap());
    if sorted.len() % 2 == 1 {
        Ok(sorted[sorted.len() / 2])
    } else {
        let middle = sorted.len() / 2;
        Ok((sorted[middle - 1] + sorted[middle]) / 2.0)
    }
}

fn auc_reference_greater(reference: &[f64], comparison: &[f64]) -> f64 {
    let mut combined = reference
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, index, true))
        .chain(
            comparison
                .iter()
                .copied()
                .enumerate()
                .map(|(index, value)| (value, reference.len() + index, false)),
        )
        .collect::<Vec<_>>();
    combined.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap());
    let mut ranks = vec![0.0; combined.len()];
    let mut start = 0;
    while start < combined.len() {
        let mut end = start + 1;
        while end < combined.len() && combined[start].0 == combined[end].0 {
            end += 1;
        }
        let average_rank = (start as f64 + 1.0 + end as f64) / 2.0;
        for (_, original_index, _) in &combined[start..end] {
            ranks[*original_index] = average_rank;
        }
        start = end;
    }
    let rank_sum = ranks[..reference.len()].iter().sum::<f64>();
    let count = reference.len() as f64;
    let u = rank_sum - count * (count + 1.0) / 2.0;
    u / (count * comparison.len() as f64)
}

pub fn run(
    catalog_path: &Path,
    reference_class: &str,
    comparison_classes: Option<&[String]>,
    requested_features: &[String],
    top_n: usize,
) -> Result<FeatureEffectSizeResult> {
    let records = read_catalog(catalog_path, requested_features)?;
    let mut by_class: BTreeMap<String, Vec<&CatalogRecord>> = BTreeMap::new();
    for record in &records {
        if !record.feature_class.is_empty() {
            by_class
                .entry(record.feature_class.clone())
                .or_default()
                .push(record);
        }
    }
    let reference_rows = by_class
        .get(reference_class)
        .ok_or_else(|| format!("source catalog has no feature_class={reference_class:?}"))?;
    let comparisons = if let Some(classes) = comparison_classes {
        unique_trimmed(classes)
    } else {
        by_class
            .keys()
            .filter(|name| name.as_str() != reference_class && name.as_str() != "other_rejected")
            .cloned()
            .collect()
    };
    for name in &comparisons {
        if !by_class.contains_key(name) {
            return Err(format!("source catalog has no comparison class: {name:?}"));
        }
    }
    let features = unique_trimmed(requested_features);
    if features.is_empty() {
        return Err("features must not be empty".into());
    }
    if top_n == 0 {
        return Err("top_n must be positive".into());
    }

    let mut rows = Vec::new();
    let mut top_features_by_class = BTreeMap::new();
    for comparison_class in &comparisons {
        let comparison_rows = &by_class[comparison_class];
        let mut class_rows = Vec::new();
        for feature in &features {
            let reference_values = values_for(reference_rows, feature);
            let comparison_values = values_for(comparison_rows, feature);
            if reference_values.is_empty() || comparison_values.is_empty() {
                continue;
            }
            let auc = auc_reference_greater(&reference_values, &comparison_values);
            let reference_median = median(&reference_values)?;
            let comparison_median = median(&comparison_values)?;
            let reference_sorted = sorted_values(&reference_values);
            let reference_iqr =
                quantile(&reference_sorted, 0.75)? - quantile(&reference_sorted, 0.25)?;
            let comparison_sorted = sorted_values(&comparison_values);
            let comparison_iqr =
                quantile(&comparison_sorted, 0.75)? - quantile(&comparison_sorted, 0.25)?;
            let scale = ((reference_iqr + comparison_iqr) / 2.0).max(1e-12);
            class_rows.push(FeatureEffectSizeRow {
                reference_class: reference_class.to_owned(),
                comparison_class: comparison_class.clone(),
                feature: feature.clone(),
                reference_count: reference_values.len(),
                comparison_count: comparison_values.len(),
                reference_median,
                comparison_median,
                reference_iqr,
                comparison_iqr,
                auc_reference_greater: auc,
                cliffs_delta: 2.0 * auc - 1.0,
                robust_median_gap: (reference_median - comparison_median) / scale,
            });
        }
        class_rows.sort_by(|left, right| {
            right
                .cliffs_delta
                .abs()
                .partial_cmp(&left.cliffs_delta.abs())
                .unwrap()
                .then_with(|| left.feature.cmp(&right.feature))
        });
        top_features_by_class.insert(
            comparison_class.clone(),
            class_rows
                .iter()
                .take(top_n)
                .map(|row| row.feature.clone())
                .collect(),
        );
        rows.extend(class_rows);
    }
    Ok(FeatureEffectSizeResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        reference_class: reference_class.to_owned(),
        comparison_classes: comparisons,
        features,
        rows,
        top_features_by_class,
    })
}

fn sorted_values(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap());
    sorted
}

pub fn write_artifacts(result: &FeatureEffectSizeResult, output_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    const HEADERS: [&str; 12] = [
        "reference_class",
        "comparison_class",
        "feature",
        "reference_count",
        "comparison_count",
        "reference_median",
        "comparison_median",
        "reference_iqr",
        "comparison_iqr",
        "auc_reference_greater",
        "cliffs_delta",
        "robust_median_gap",
    ];
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(HEADERS)
        .map_err(|error| error.to_string())?;
    for row in &result.rows {
        writer
            .write_record([
                row.reference_class.clone(),
                row.comparison_class.clone(),
                row.feature.clone(),
                row.reference_count.to_string(),
                row.comparison_count.to_string(),
                json!(row.reference_median).to_string(),
                json!(row.comparison_median).to_string(),
                json!(row.reference_iqr).to_string(),
                json!(row.comparison_iqr).to_string(),
                json!(row.auc_reference_greater).to_string(),
                json!(row.cliffs_delta).to_string(),
                json!(row.robust_median_gap).to_string(),
            ])
            .map_err(|error| error.to_string())?;
    }
    let csv_bytes = writer.into_inner().map_err(|error| error.to_string())?;
    let mut bom_csv = vec![0xef, 0xbb, 0xbf];
    bom_csv.extend(csv_bytes);
    fs::write(output_dir.join("feature_effect_sizes.csv"), bom_csv)
        .map_err(|error| error.to_string())?;
    let mut json_bytes =
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?;
    json_bytes.push(b'\n');
    fs::write(output_dir.join("feature_effect_sizes.json"), json_bytes)
        .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut catalog_path = None;
    let mut output_dir = PathBuf::from("tmp/feature-effect-size");
    let mut reference_class = "compact_quality".to_owned();
    let mut comparison_classes = None;
    let mut features = DEFAULT_EFFECT_FEATURES
        .iter()
        .map(|feature| (*feature).to_owned())
        .collect::<Vec<_>>();
    let mut top_n = 5_usize;
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
            "--reference-class" => {
                index += usize::from(inline_value.is_none());
                reference_class = inline_value
                    .unwrap_or_else(|| args.get(index).map(String::as_str).unwrap_or(""))
                    .to_owned();
            }
            "--top-n" => {
                index += usize::from(inline_value.is_none());
                let parsed = inline_value
                    .unwrap_or_else(|| args.get(index).map(String::as_str).unwrap_or(""))
                    .parse::<i64>()
                    .map_err(|error| format!("invalid top N: {error}"))?;
                if parsed < 1 {
                    return Err("top_n must be positive".into());
                }
                top_n = parsed as usize;
            }
            "--comparison-classes" | "--features" => {
                let values = if let Some(value) = inline_value {
                    vec![value.to_owned()]
                } else {
                    let mut values = Vec::new();
                    while index + 1 < args.len() && !args[index + 1].starts_with('-') {
                        index += 1;
                        values.push(args[index].clone());
                    }
                    values
                };
                if name == "--comparison-classes" {
                    comparison_classes = Some(values);
                } else {
                    features = values;
                }
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
    let result = run(
        &catalog_path,
        &reference_class,
        comparison_classes.as_deref(),
        &features,
        top_n,
    )?;
    write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-feature-effect-size",
        "out_dir": output_dir,
        "result_path": output_dir.join("feature_effect_sizes.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_medians_iqrs_and_top_features_preserve_direction() {
        let dir = std::env::temp_dir().join(format!("rst19-effect-size-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let catalog = dir.join("catalog.csv");
        fs::write(
            &catalog,
            "feature_class,flux_snr,fwhm\ncompact_quality,10,2\ncompact_quality,20,2.2\ncompact_quality,30,2.1\nweak_or_background,1,3\nweak_or_background,2,3.2\nspike_or_support,5,4\n",
        )
        .unwrap();
        let result = run(
            &catalog,
            "compact_quality",
            Some(&["weak_or_background".into()]),
            &["flux_snr".into(), "fwhm".into()],
            2,
        )
        .unwrap();
        let auc_flux = result
            .rows
            .iter()
            .find(|row| row.feature == "flux_snr")
            .unwrap()
            .auc_reference_greater;
        let auc_fwhm = result
            .rows
            .iter()
            .find(|row| row.feature == "fwhm")
            .unwrap()
            .auc_reference_greater;
        assert_eq!(auc_flux, 1.0);
        assert_eq!(auc_fwhm, 0.0);
        assert_eq!(
            result.top_features_by_class["weak_or_background"],
            ["flux_snr", "fwhm"]
        );
    }
}
