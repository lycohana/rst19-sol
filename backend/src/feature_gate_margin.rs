use crate::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

const INTERPRETATION_BOUNDARY: [&str; 4] = [
    "a positive margin means the selected numeric field is inside the configured gate",
    "a negative margin means that field is outside the configured gate",
    "margins have different units and must not be summed or ranked across metrics",
    "quality_passed and feature_class remain detector-level labels, not physical truth",
];

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct FeatureGateMarginThresholds {
    pub min_flux_snr: f64,
    pub min_psf_support_pixels: f64,
    pub min_fwhm: f64,
    pub max_fwhm: f64,
    pub max_ellipticity: f64,
    pub min_sharpness: f64,
    pub max_sharpness: f64,
    pub min_footprint_pixels: f64,
}

impl Default for FeatureGateMarginThresholds {
    fn default() -> Self {
        Self {
            min_flux_snr: 5.0,
            min_psf_support_pixels: 3.0,
            min_fwhm: 0.8,
            max_fwhm: 12.0,
            max_ellipticity: 0.65,
            min_sharpness: 0.005,
            max_sharpness: 0.85,
            min_footprint_pixels: 2.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureGateMarginRow {
    pub feature_class: String,
    pub feature_class_label: Option<String>,
    pub class_count: usize,
    pub quality_count: usize,
    pub metric: String,
    pub value_count: usize,
    pub negative_margin_count: usize,
    pub negative_margin_fraction: f64,
    pub median_margin: f64,
    pub p10_margin: f64,
    pub p90_margin: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureGateMarginResult {
    pub catalog_path: String,
    pub thresholds: FeatureGateMarginThresholds,
    pub rows: Vec<FeatureGateMarginRow>,
}

impl FeatureGateMarginResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "thresholds": self.thresholds,
            "rows": self.rows,
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        })
    }
}

struct CatalogRecord {
    detection_id: String,
    feature_class: String,
    feature_class_label: Option<String>,
    quality: bool,
    values: [Option<f64>; 6],
}

fn finite_float(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
}

fn quality(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "1" | "yes"
    )
}

fn field(record: &csv::StringRecord, index: usize) -> &str {
    record.get(index).unwrap_or("None")
}

fn read_catalog(path: &Path) -> Result<Vec<CatalogRecord>> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let required = [
        "detection_id",
        "feature_class",
        "feature_class_label",
        "quality_passed",
        "flux_snr",
        "psf_support_pixels",
        "fwhm",
        "ellipticity",
        "sharpness",
        "footprint_pixels",
    ];
    let mut indices = BTreeMap::new();
    for name in required {
        if let Some(index) = headers
            .iter()
            .enumerate()
            .filter_map(|(index, header)| (header == name).then_some(index))
            .last()
        {
            indices.insert(name, index);
        }
    }
    let mut missing = required
        .iter()
        .filter(|name| !indices.contains_key(**name))
        .copied()
        .collect::<Vec<_>>();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }

    let mut records = Vec::new();
    for row in reader.records() {
        let row = row.map_err(|error| error.to_string())?;
        let raw_label = row.get(indices["feature_class_label"]);
        records.push(CatalogRecord {
            detection_id: field(&row, indices["detection_id"]).trim().to_owned(),
            feature_class: field(&row, indices["feature_class"]).trim().to_owned(),
            feature_class_label: raw_label.map(str::to_owned),
            quality: quality(field(&row, indices["quality_passed"])),
            values: [
                finite_float(field(&row, indices["flux_snr"])),
                finite_float(field(&row, indices["psf_support_pixels"])),
                finite_float(field(&row, indices["fwhm"])),
                finite_float(field(&row, indices["ellipticity"])),
                finite_float(field(&row, indices["sharpness"])),
                finite_float(field(&row, indices["footprint_pixels"])),
            ],
        });
    }
    if records.is_empty() {
        return Err("source catalog is empty".into());
    }
    Ok(records)
}

fn validate_thresholds(thresholds: &FeatureGateMarginThresholds) -> Result<()> {
    if thresholds.min_flux_snr <= 0.0 {
        return Err("min_flux_snr must be positive".into());
    }
    if thresholds.min_psf_support_pixels < 1.0 {
        return Err("min_psf_support_pixels must be positive".into());
    }
    if thresholds.min_fwhm <= 0.0 || thresholds.max_fwhm < thresholds.min_fwhm {
        return Err("fwhm thresholds are invalid".into());
    }
    if !(0.0..=1.0).contains(&thresholds.max_ellipticity) {
        return Err("max_ellipticity must be between 0 and 1".into());
    }
    if thresholds.min_sharpness < 0.0 || thresholds.max_sharpness <= thresholds.min_sharpness {
        return Err("sharpness thresholds are invalid".into());
    }
    if thresholds.min_footprint_pixels < 1.0 {
        return Err("min_footprint_pixels must be positive".into());
    }
    Ok(())
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

pub fn run(
    catalog_path: &Path,
    thresholds: FeatureGateMarginThresholds,
) -> Result<FeatureGateMarginResult> {
    validate_thresholds(&thresholds)?;
    let records = read_catalog(catalog_path)?;
    let mut seen_ids = HashSet::new();
    let mut classes: BTreeMap<String, Vec<CatalogRecord>> = BTreeMap::new();
    for record in records {
        if record.detection_id.is_empty() {
            return Err("source catalog contains an empty detection_id".into());
        }
        if !seen_ids.insert(record.detection_id.clone()) {
            return Err(format!(
                "source catalog contains duplicate detection_id: {}",
                record.detection_id
            ));
        }
        if !record.feature_class.is_empty() {
            classes
                .entry(record.feature_class.clone())
                .or_default()
                .push(record);
        }
    }

    let mut rows = Vec::new();
    for (feature_class, members) in classes {
        let mut margins: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
        for member in &members {
            let [flux_snr, psf_support, fwhm, ellipticity, sharpness, footprint] = member.values;
            if let Some(value) = flux_snr {
                margins
                    .entry("flux_snr")
                    .or_default()
                    .push(value - thresholds.min_flux_snr);
            }
            if let Some(value) = psf_support {
                margins
                    .entry("psf_support_pixels")
                    .or_default()
                    .push(value - thresholds.min_psf_support_pixels);
            }
            if let Some(value) = fwhm {
                margins
                    .entry("fwhm")
                    .or_default()
                    .push((value - thresholds.min_fwhm).min(thresholds.max_fwhm - value));
            }
            if let Some(value) = ellipticity {
                margins
                    .entry("ellipticity")
                    .or_default()
                    .push(thresholds.max_ellipticity - value);
            }
            if let Some(value) = sharpness {
                margins
                    .entry("sharpness")
                    .or_default()
                    .push((value - thresholds.min_sharpness).min(thresholds.max_sharpness - value));
            }
            if let Some(value) = footprint {
                margins
                    .entry("footprint_pixels")
                    .or_default()
                    .push(value - thresholds.min_footprint_pixels);
            }
        }
        let quality_count = members.iter().filter(|member| member.quality).count();
        let label = members[0].feature_class_label.clone();
        for (metric, mut values) in margins {
            values.sort_by(|left, right| left.partial_cmp(right).unwrap());
            let negative_margin_count = values.iter().filter(|value| **value < 0.0).count();
            rows.push(FeatureGateMarginRow {
                feature_class: feature_class.clone(),
                feature_class_label: label.clone(),
                class_count: members.len(),
                quality_count,
                metric: metric.into(),
                value_count: values.len(),
                negative_margin_count,
                negative_margin_fraction: negative_margin_count as f64 / values.len() as f64,
                median_margin: quantile(&values, 0.5)?,
                p10_margin: quantile(&values, 0.1)?,
                p90_margin: quantile(&values, 0.9)?,
            });
        }
    }
    rows.sort_by(|left, right| {
        left.feature_class
            .cmp(&right.feature_class)
            .then_with(|| left.metric.cmp(&right.metric))
    });
    Ok(FeatureGateMarginResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        thresholds,
        rows,
    })
}

fn float_cell(value: f64) -> String {
    json!(value).to_string()
}

fn csv_row(row: &FeatureGateMarginRow) -> Vec<String> {
    vec![
        row.feature_class.clone(),
        row.feature_class_label.clone().unwrap_or_default(),
        row.class_count.to_string(),
        row.quality_count.to_string(),
        row.metric.clone(),
        row.value_count.to_string(),
        row.negative_margin_count.to_string(),
        float_cell(row.negative_margin_fraction),
        float_cell(row.median_margin),
        float_cell(row.p10_margin),
        float_cell(row.p90_margin),
    ]
}

pub fn write_artifacts(result: &FeatureGateMarginResult, output_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    const HEADERS: [&str; 11] = [
        "feature_class",
        "feature_class_label",
        "class_count",
        "quality_count",
        "metric",
        "value_count",
        "negative_margin_count",
        "negative_margin_fraction",
        "median_margin",
        "p10_margin",
        "p90_margin",
    ];
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(HEADERS)
        .map_err(|error| error.to_string())?;
    for row in &result.rows {
        writer
            .write_record(csv_row(row))
            .map_err(|error| error.to_string())?;
    }
    let csv_bytes = writer.into_inner().map_err(|error| error.to_string())?;
    let mut bom_csv = vec![0xef, 0xbb, 0xbf];
    bom_csv.extend(csv_bytes);
    fs::write(output_dir.join("feature_gate_margins.csv"), bom_csv)
        .map_err(|error| error.to_string())?;
    let mut json_bytes =
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?;
    json_bytes.push(b'\n');
    fs::write(output_dir.join("feature_gate_margins.json"), json_bytes)
        .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

fn parse_f64(value: &str, name: &str) -> Result<f64> {
    value
        .parse::<f64>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn parse_int(value: &str, name: &str) -> Result<f64> {
    value
        .parse::<i64>()
        .map(|number| number as f64)
        .map_err(|error| format!("invalid {name}: {error}"))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut catalog_path = None;
    let mut output_dir = PathBuf::from("tmp/feature-gate-margin");
    let mut thresholds = FeatureGateMarginThresholds::default();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline_value) = arg
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((arg.as_str(), None));
        if name == "--out-dir" {
            let value = if let Some(value) = inline_value {
                value
            } else {
                index += 1;
                args.get(index).ok_or("--out-dir requires a value")?
            };
            output_dir = PathBuf::from(value);
        } else {
            let (target, integer) = match name {
                "--min-flux-snr" => (&mut thresholds.min_flux_snr, false),
                "--min-psf-support-pixels" => (&mut thresholds.min_psf_support_pixels, true),
                "--min-fwhm" => (&mut thresholds.min_fwhm, false),
                "--max-fwhm" => (&mut thresholds.max_fwhm, false),
                "--max-ellipticity" => (&mut thresholds.max_ellipticity, false),
                "--min-sharpness" => (&mut thresholds.min_sharpness, false),
                "--max-sharpness" => (&mut thresholds.max_sharpness, false),
                "--min-footprint-pixels" => (&mut thresholds.min_footprint_pixels, true),
                _ if arg.starts_with('-') => return Err(format!("unknown argument {arg}")),
                _ => {
                    if catalog_path.replace(PathBuf::from(arg)).is_some() {
                        return Err("only one source catalog is allowed".into());
                    }
                    index += 1;
                    continue;
                }
            };
            let value = if let Some(value) = inline_value {
                value
            } else {
                index += 1;
                args.get(index)
                    .ok_or_else(|| format!("{name} requires a value"))?
            };
            *target = if integer {
                parse_int(value, name)?
            } else {
                parse_f64(value, name)?
            };
        }
        index += 1;
    }
    let catalog_path = catalog_path.ok_or("a source catalog CSV is required")?;
    let result = run(&catalog_path, thresholds)?;
    write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-feature-gate-margin",
        "out_dir": output_dir,
        "result_path": output_dir.join("feature_gate_margins.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_margins_and_linear_quantiles_match_python_contract() {
        let dir = std::env::temp_dir().join(format!("rst19-gate-margin-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let catalog = dir.join("catalog.csv");
        fs::write(
            &catalog,
            "detection_id,feature_class,feature_class_label,quality_passed,flux_snr,psf_support_pixels,fwhm,ellipticity,sharpness,footprint_pixels\n1,compact,Compact,true,12,5,2.5,0.1,0.3,20\n2,compact,Ignored,false,3,4,2.5,0.1,0.3,20\n3,shape,Shape,false,25,6,3.5,0.7,0.3,20\n",
        )
        .unwrap();
        let result = run(&catalog, FeatureGateMarginThresholds::default()).unwrap();
        let snr = result
            .rows
            .iter()
            .find(|row| row.feature_class == "compact" && row.metric == "flux_snr")
            .unwrap();
        assert_eq!(snr.median_margin, 2.5);
        assert_eq!(snr.p10_margin, -1.1);
        let ellipticity = result
            .rows
            .iter()
            .find(|row| row.feature_class == "shape" && row.metric == "ellipticity")
            .unwrap();
        assert!(ellipticity.median_margin < 0.0);
    }
}
