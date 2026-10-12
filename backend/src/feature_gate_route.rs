use crate::{feature_gate_margin::FeatureGateMarginThresholds, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

const REQUIRED_FIELDS: [&str; 11] = [
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "flags",
    "flux_snr",
    "psf_support_pixels",
    "fwhm",
    "ellipticity",
    "sharpness",
    "footprint_pixels",
];

const NUMERIC_FLAG_ALIASES: [(&str, &str); 3] = [
    ("LOW_FLUX_SNR", "flux_snr"),
    ("INSUFFICIENT_PSF_SUPPORT", "psf_support_pixels"),
    ("SMALL_FOOTPRINT", "footprint_pixels"),
];

const INTERPRETATION_BOUNDARY: [&str; 4] = [
    "route is an observed detector-rule decomposition, not a counterfactual ablation",
    "flags and quality_passed can share implementation logic",
    "numeric and structural routes are not noise probabilities or physical star classes",
    "quality_passed remains a detector-level label until truth, WCS, catalog and injection checks close",
];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureGateRouteSourceRow {
    pub detection_id: String,
    pub feature_class: String,
    pub feature_class_label: String,
    pub quality_passed: bool,
    pub route: String,
    pub numeric_failure_reasons: String,
    pub structural_flags: String,
    pub flags: String,
    pub flux_snr: String,
    pub psf_support_pixels: String,
    pub fwhm: String,
    pub ellipticity: String,
    pub sharpness: String,
    pub footprint_pixels: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureGateRouteSummaryRow {
    pub feature_class: String,
    pub feature_class_label: String,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub rejected_count: usize,
    pub numeric_failure_count: usize,
    pub structural_flag_count: usize,
    pub rejected_numeric_only_count: usize,
    pub rejected_structural_only_count: usize,
    pub rejected_numeric_and_structural_count: usize,
    pub rejected_unexplained_count: usize,
    pub quality_passed_with_numeric_failure_count: usize,
    pub quality_passed_with_structural_flag_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureGateRouteResult {
    pub catalog_path: String,
    pub thresholds: FeatureGateMarginThresholds,
    pub source_rows: Vec<FeatureGateRouteSourceRow>,
    pub summary_rows: Vec<FeatureGateRouteSummaryRow>,
}

impl FeatureGateRouteResult {
    pub fn as_dict(&self) -> Value {
        let aliases: BTreeMap<&str, &str> = NUMERIC_FLAG_ALIASES.into_iter().collect();
        json!({
            "catalog_path": self.catalog_path,
            "thresholds": self.thresholds,
            "numeric_alias_flags": aliases,
            "source_rows": self.source_rows,
            "summary_rows": self.summary_rows,
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        })
    }
}

type Record = HashMap<String, Option<String>>;

fn raw<'a>(record: &'a Record, name: &str) -> Option<&'a str> {
    record.get(name).and_then(Option::as_deref)
}

fn python_str(record: &Record, name: &str) -> String {
    raw(record, name).unwrap_or("None").to_owned()
}

fn finite_float(record: &Record, name: &str) -> Option<f64> {
    raw(record, name)?
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn quality(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "1" | "yes"
    )
}

fn normalized_flags(value: &str) -> Vec<String> {
    value
        .replace(';', "|")
        .split('|')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn read_catalog(path: &Path) -> Result<Vec<Record>> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let mut indices = HashMap::<&str, usize>::new();
    for required in REQUIRED_FIELDS {
        if let Some(index) = headers
            .iter()
            .enumerate()
            .filter_map(|(index, header)| (header == required).then_some(index))
            .last()
        {
            indices.insert(required, index);
        }
    }
    let mut missing = REQUIRED_FIELDS
        .iter()
        .filter(|field| !indices.contains_key(**field))
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
        let mut record = Record::new();
        for field in REQUIRED_FIELDS {
            let index = indices[field];
            record.insert(field.to_owned(), row.get(index).map(str::to_owned));
        }
        records.push(record);
    }
    if records.is_empty() {
        return Err("source catalog is empty".into());
    }
    Ok(records)
}

fn numeric_reasons(record: &Record, thresholds: FeatureGateMarginThresholds) -> Vec<String> {
    let mut margins = BTreeMap::<&str, f64>::new();
    if let Some(value) = finite_float(record, "flux_snr") {
        margins.insert("flux_snr", value - thresholds.min_flux_snr);
    }
    if let Some(value) = finite_float(record, "psf_support_pixels") {
        margins.insert(
            "psf_support_pixels",
            value - thresholds.min_psf_support_pixels,
        );
    }
    if let Some(value) = finite_float(record, "fwhm") {
        margins.insert(
            "fwhm",
            (value - thresholds.min_fwhm).min(thresholds.max_fwhm - value),
        );
    }
    if let Some(value) = finite_float(record, "ellipticity") {
        margins.insert("ellipticity", thresholds.max_ellipticity - value);
    }
    if let Some(value) = finite_float(record, "sharpness") {
        margins.insert(
            "sharpness",
            (value - thresholds.min_sharpness).min(thresholds.max_sharpness - value),
        );
    }
    if let Some(value) = finite_float(record, "footprint_pixels") {
        margins.insert("footprint_pixels", value - thresholds.min_footprint_pixels);
    }
    margins
        .into_iter()
        .filter_map(|(metric, margin)| (margin < 0.0).then_some(metric.to_owned()))
        .collect()
}

fn structural_flags(flags: &[String], numeric_reasons: &[String]) -> Vec<String> {
    flags
        .iter()
        .filter(|flag| {
            NUMERIC_FLAG_ALIASES
                .iter()
                .find_map(|(alias, metric)| (*alias == flag.as_str()).then_some(*metric))
                .is_none_or(|metric| !numeric_reasons.iter().any(|reason| reason == metric))
        })
        .cloned()
        .collect()
}

fn route(numeric: &[String], structural: &[String], quality_passed: bool) -> &'static str {
    if quality_passed {
        "quality_passed"
    } else if !numeric.is_empty() && !structural.is_empty() {
        "rejected_numeric_and_structural"
    } else if !numeric.is_empty() {
        "rejected_numeric_only"
    } else if !structural.is_empty() {
        "rejected_structural_only"
    } else {
        "rejected_unexplained"
    }
}

pub fn run(catalog_path: &Path) -> Result<FeatureGateRouteResult> {
    let records = read_catalog(catalog_path)?;
    let thresholds = FeatureGateMarginThresholds::default();
    let mut seen_ids = HashSet::new();
    let mut source_rows = Vec::new();
    let mut classes = BTreeMap::<String, Vec<FeatureGateRouteSourceRow>>::new();

    for record in records {
        let detection_id = python_str(&record, "detection_id").trim().to_owned();
        if detection_id.is_empty() {
            return Err("source catalog contains an empty detection_id".into());
        }
        if !seen_ids.insert(detection_id.clone()) {
            return Err(format!(
                "source catalog contains duplicate detection_id: {detection_id}"
            ));
        }

        let feature_class = python_str(&record, "feature_class").trim().to_owned();
        if feature_class.is_empty() {
            return Err(format!(
                "source catalog row {detection_id} has an empty feature_class"
            ));
        }
        let label = python_str(&record, "feature_class_label").trim().to_owned();
        let label = if label.is_empty() {
            feature_class.clone()
        } else {
            label
        };
        let quality_passed = quality(&python_str(&record, "quality_passed"));
        let flags = normalized_flags(&python_str(&record, "flags"));
        let numeric = numeric_reasons(&record, thresholds);
        let structural = structural_flags(&flags, &numeric);
        let row = FeatureGateRouteSourceRow {
            detection_id,
            feature_class: feature_class.clone(),
            feature_class_label: label,
            quality_passed,
            route: route(&numeric, &structural, quality_passed).to_owned(),
            numeric_failure_reasons: numeric.join("|"),
            structural_flags: structural.join("|"),
            flags: flags.join("|"),
            flux_snr: python_str(&record, "flux_snr"),
            psf_support_pixels: python_str(&record, "psf_support_pixels"),
            fwhm: python_str(&record, "fwhm"),
            ellipticity: python_str(&record, "ellipticity"),
            sharpness: python_str(&record, "sharpness"),
            footprint_pixels: python_str(&record, "footprint_pixels"),
        };
        source_rows.push(row.clone());
        classes.entry(feature_class).or_default().push(row);
    }

    let mut summary_rows = Vec::new();
    for (feature_class, members) in classes {
        let mut route_counts = HashMap::<&str, usize>::new();
        let mut numeric_failure_count = 0;
        let mut structural_flag_count = 0;
        for member in &members {
            *route_counts.entry(&member.route).or_default() += 1;
            numeric_failure_count += usize::from(!member.numeric_failure_reasons.is_empty());
            structural_flag_count += usize::from(!member.structural_flags.is_empty());
        }
        let quality_count = members
            .iter()
            .filter(|member| member.quality_passed)
            .count();
        summary_rows.push(FeatureGateRouteSummaryRow {
            feature_class,
            feature_class_label: members[0].feature_class_label.clone(),
            candidate_count: members.len(),
            quality_count,
            rejected_count: members.len() - quality_count,
            numeric_failure_count,
            structural_flag_count,
            rejected_numeric_only_count: route_counts
                .get("rejected_numeric_only")
                .copied()
                .unwrap_or(0),
            rejected_structural_only_count: route_counts
                .get("rejected_structural_only")
                .copied()
                .unwrap_or(0),
            rejected_numeric_and_structural_count: route_counts
                .get("rejected_numeric_and_structural")
                .copied()
                .unwrap_or(0),
            rejected_unexplained_count: route_counts
                .get("rejected_unexplained")
                .copied()
                .unwrap_or(0),
            quality_passed_with_numeric_failure_count: members
                .iter()
                .filter(|member| {
                    member.route == "quality_passed" && !member.numeric_failure_reasons.is_empty()
                })
                .count(),
            quality_passed_with_structural_flag_count: members
                .iter()
                .filter(|member| {
                    member.route == "quality_passed" && !member.structural_flags.is_empty()
                })
                .count(),
        });
    }

    Ok(FeatureGateRouteResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        thresholds,
        source_rows,
        summary_rows,
    })
}

fn csv_row(row: &FeatureGateRouteSourceRow) -> Vec<String> {
    vec![
        row.detection_id.clone(),
        row.feature_class.clone(),
        row.feature_class_label.clone(),
        if row.quality_passed { "True" } else { "False" }.to_owned(),
        row.route.clone(),
        row.numeric_failure_reasons.clone(),
        row.structural_flags.clone(),
        row.flags.clone(),
        row.flux_snr.clone(),
        row.psf_support_pixels.clone(),
        row.fwhm.clone(),
        row.ellipticity.clone(),
        row.sharpness.clone(),
        row.footprint_pixels.clone(),
    ]
}

fn summary_csv_row(row: &FeatureGateRouteSummaryRow) -> Vec<String> {
    vec![
        row.feature_class.clone(),
        row.feature_class_label.clone(),
        row.candidate_count.to_string(),
        row.quality_count.to_string(),
        row.rejected_count.to_string(),
        row.numeric_failure_count.to_string(),
        row.structural_flag_count.to_string(),
        row.rejected_numeric_only_count.to_string(),
        row.rejected_structural_only_count.to_string(),
        row.rejected_numeric_and_structural_count.to_string(),
        row.rejected_unexplained_count.to_string(),
        row.quality_passed_with_numeric_failure_count.to_string(),
        row.quality_passed_with_structural_flag_count.to_string(),
    ]
}

fn write_csv(path: &Path, headers: &[&str], rows: Vec<Vec<String>>) -> Result<()> {
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

pub fn write_artifacts(result: &FeatureGateRouteResult, output_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    write_csv(
        &output_dir.join("feature_gate_route_sources.csv"),
        &[
            "detection_id",
            "feature_class",
            "feature_class_label",
            "quality_passed",
            "route",
            "numeric_failure_reasons",
            "structural_flags",
            "flags",
            "flux_snr",
            "psf_support_pixels",
            "fwhm",
            "ellipticity",
            "sharpness",
            "footprint_pixels",
        ],
        result.source_rows.iter().map(csv_row).collect(),
    )?;
    write_csv(
        &output_dir.join("feature_gate_route_summary.csv"),
        &[
            "feature_class",
            "feature_class_label",
            "candidate_count",
            "quality_count",
            "rejected_count",
            "numeric_failure_count",
            "structural_flag_count",
            "rejected_numeric_only_count",
            "rejected_structural_only_count",
            "rejected_numeric_and_structural_count",
            "rejected_unexplained_count",
            "quality_passed_with_numeric_failure_count",
            "quality_passed_with_structural_flag_count",
        ],
        result.summary_rows.iter().map(summary_csv_row).collect(),
    )?;
    fs::write(
        output_dir.join("feature_gate_routes.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut catalog_path = None;
    let mut output_dir = PathBuf::from("tmp/feature-gate-route");
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
    let result = run(&catalog_path)?;
    write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-feature-gate-route",
        "out_dir": output_dir,
        "result_path": output_dir.join("feature_gate_routes.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_separate_numeric_flags_from_structural_flags() {
        let root =
            std::env::temp_dir().join(format!("rst19-feature-gate-route-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("source_catalog.csv");
        fs::write(
            &catalog,
            "detection_id,feature_class,feature_class_label,quality_passed,flags,flux_snr,psf_support_pixels,fwhm,ellipticity,sharpness,footprint_pixels\n\
             1,compact_quality,紧凑,True,,12,5,2.5,0.1,0.3,20\n\
             2,weak_or_background,弱,False,LOW_FLUX_SNR,3,5,2.5,0.1,0.3,20\n\
             3,linear_artifact,线状,False,LINE_ARTIFACT,20,5,2.5,0.1,0.3,20\n\
             4,crowded_blend,拥挤,False,LOW_FLUX_SNR|UNRESOLVED_BLEND,3,2,2.5,0.1,0.3,20\n",
        )
        .unwrap();
        let result = run(&catalog).unwrap();
        assert_eq!(result.source_rows[0].route, "quality_passed");
        assert_eq!(result.source_rows[1].numeric_failure_reasons, "flux_snr");
        assert_eq!(result.source_rows[1].structural_flags, "");
        assert_eq!(result.source_rows[2].route, "rejected_structural_only");
        assert_eq!(
            result.source_rows[3].numeric_failure_reasons,
            "flux_snr|psf_support_pixels"
        );
        assert_eq!(result.source_rows[3].structural_flags, "UNRESOLVED_BLEND");
        let _ = fs::remove_dir_all(root);
    }
}
