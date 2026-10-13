use crate::Result;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

const DEFAULT_TARGETS: [&str; 2] = ["82931", "82934"];
const INTERPRETATION_BOUNDARY: &str = "descriptive aperture/PSF contamination diagnostic; depends on aperture, background, PSF and feature class; not star probability, artifact probability, completeness, precision, or FDR";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Summary {
    pub feature_class: String,
    pub sample_count: usize,
    pub ratio_median: f64,
    pub ratio_q90: f64,
    pub ratio_q95: f64,
    pub ratio_q99: f64,
    pub ratio_gt10_count: usize,
    pub ratio_gt10_fraction: f64,
    pub ratio_gt20_count: usize,
    pub ratio_gt20_fraction: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MethodSummary {
    pub feature_class: String,
    pub method_group: String,
    pub sample_count: usize,
    pub ratio_median: f64,
    pub ratio_q90: f64,
    pub ratio_q95: f64,
    pub ratio_gt10_count: usize,
    pub ratio_gt10_fraction: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Target {
    pub detection_id: String,
    pub feature_class: String,
    pub peak: f64,
    pub flux: f64,
    pub flux_to_peak: f64,
    pub class_sample_count: usize,
    pub class_percentile: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AperturePeakRatioResult {
    pub catalog_path: String,
    pub targets: Vec<String>,
    pub summaries: Vec<Summary>,
    pub method_summaries: Vec<MethodSummary>,
    pub target_rows: Vec<Target>,
}

impl AperturePeakRatioResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "targets": self.targets,
            "summaries": self.summaries,
            "method_summaries": self.method_summaries,
            "target_rows": self.target_rows,
            "ratio_definition": "finite aperture flux divided by positive peak value",
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        })
    }
}

#[derive(Clone, Debug)]
struct RatioRecord {
    detection_id: String,
    feature_class: String,
    peak: f64,
    flux: f64,
    ratio: f64,
    method_group: String,
}

fn field(record: &csv::StringRecord, index: Option<usize>) -> &str {
    index.map_or("", |index| record.get(index).unwrap_or("None"))
}

fn proposal_method_group(raw_methods: &str) -> String {
    let methods: HashSet<&str> = raw_methods
        .split('|')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect();
    if methods.len() == 3
        && methods.contains("gaussian")
        && methods.contains("dog_narrow")
        && methods.contains("dog_broad")
    {
        "all_three".to_owned()
    } else if methods.len() == 1 && methods.contains("gaussian") {
        "gaussian_only".to_owned()
    } else if !methods.is_empty() && !methods.contains("gaussian") {
        "dog_only".to_owned()
    } else if !methods.is_empty() {
        "partial".to_owned()
    } else {
        "no_method".to_owned()
    }
}

fn parse_finite(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn read_records(path: &Path) -> Result<Vec<RatioRecord>> {
    let bytes = fs::read(path).map_err(|error| format!("无法读取源级 CSV {path:?}: {error}"))?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let names: HashSet<&str> = headers.iter().collect();
    let required = ["detection_id", "feature_class", "peak", "flux"];
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|name| !names.contains(name))
        .collect();
    if !missing.is_empty() {
        let mut missing = missing;
        missing.sort_unstable();
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let mut indices = HashMap::new();
    for (index, name) in headers.iter().enumerate() {
        indices.insert(name, index);
    }

    let mut records = Vec::new();
    for row in reader.records() {
        let row = row.map_err(|error| error.to_string())?;
        let feature_class = field(&row, indices.get("feature_class").copied())
            .trim()
            .to_owned();
        let detection_id = field(&row, indices.get("detection_id").copied())
            .trim()
            .to_owned();
        if feature_class.is_empty() || detection_id.is_empty() {
            continue;
        }
        let Some(peak) = parse_finite(field(&row, indices.get("peak").copied())) else {
            continue;
        };
        let Some(flux) = parse_finite(field(&row, indices.get("flux").copied())) else {
            continue;
        };
        if peak <= 0.0 {
            continue;
        }
        let ratio = flux / peak;
        if !ratio.is_finite() {
            continue;
        }
        let methods = field(&row, indices.get("proposal_methods").copied());
        records.push(RatioRecord {
            detection_id,
            feature_class,
            peak,
            flux,
            ratio,
            method_group: proposal_method_group(methods),
        });
    }
    if records.is_empty() {
        return Err("source catalog has no rows with finite flux and positive peak".into());
    }
    Ok(records)
}

fn quantile(values: &[f64], probability: f64) -> f64 {
    let position = (values.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return values[lower];
    }
    let weight = position - lower as f64;
    values[lower] + (values[upper] - values[lower]) * weight
}

fn summary_counts(values: &[f64], threshold: f64) -> (usize, f64) {
    let count = values.iter().filter(|value| **value > threshold).count();
    (count, count as f64 / values.len() as f64)
}

pub fn run(catalog_path: &Path, target_ids: &[String]) -> Result<AperturePeakRatioResult> {
    let records = read_records(catalog_path)?;
    let mut targets = Vec::new();
    let mut target_seen = HashSet::new();
    for target in target_ids {
        let target = target.trim();
        if !target.is_empty() && target_seen.insert(target.to_owned()) {
            targets.push(target.to_owned());
        }
    }

    let mut by_class: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut by_method: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    for record in &records {
        by_class
            .entry(record.feature_class.clone())
            .or_default()
            .push(record.ratio);
        by_method
            .entry((record.feature_class.clone(), record.method_group.clone()))
            .or_default()
            .push(record.ratio);
    }
    for values in by_class.values_mut().chain(by_method.values_mut()) {
        values.sort_by(|left, right| left.partial_cmp(right).unwrap());
    }

    let summaries = by_class
        .iter()
        .map(|(feature_class, values)| {
            let (ratio_gt10_count, ratio_gt10_fraction) = summary_counts(values, 10.0);
            let (ratio_gt20_count, ratio_gt20_fraction) = summary_counts(values, 20.0);
            Summary {
                feature_class: feature_class.clone(),
                sample_count: values.len(),
                ratio_median: quantile(values, 0.50),
                ratio_q90: quantile(values, 0.90),
                ratio_q95: quantile(values, 0.95),
                ratio_q99: quantile(values, 0.99),
                ratio_gt10_count,
                ratio_gt10_fraction,
                ratio_gt20_count,
                ratio_gt20_fraction,
            }
        })
        .collect();
    let method_summaries = by_method
        .iter()
        .map(|((feature_class, method_group), values)| {
            let (ratio_gt10_count, ratio_gt10_fraction) = summary_counts(values, 10.0);
            MethodSummary {
                feature_class: feature_class.clone(),
                method_group: method_group.clone(),
                sample_count: values.len(),
                ratio_median: quantile(values, 0.50),
                ratio_q90: quantile(values, 0.90),
                ratio_q95: quantile(values, 0.95),
                ratio_gt10_count,
                ratio_gt10_fraction,
            }
        })
        .collect();

    let wanted: HashSet<&str> = targets.iter().map(String::as_str).collect();
    let mut target_rows: Vec<Target> = records
        .iter()
        .filter(|record| wanted.contains(record.detection_id.as_str()))
        .map(|record| {
            let values = &by_class[&record.feature_class];
            let rank = values.partition_point(|value| *value <= record.ratio);
            Target {
                detection_id: record.detection_id.clone(),
                feature_class: record.feature_class.clone(),
                peak: record.peak,
                flux: record.flux,
                flux_to_peak: record.ratio,
                class_sample_count: values.len(),
                class_percentile: rank as f64 / values.len() as f64,
            }
        })
        .collect();
    target_rows.sort_by(|left, right| {
        (&left.detection_id, &left.feature_class).cmp(&(&right.detection_id, &right.feature_class))
    });

    Ok(AperturePeakRatioResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        targets,
        summaries,
        method_summaries,
        target_rows,
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

fn write_json(path: &Path, result: &AperturePeakRatioResult) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?;
    fs::write(path, bytes).map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &AperturePeakRatioResult, output_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    write_csv(
        &output_dir.join("aperture_peak_ratio_summary.csv"),
        &[
            "feature_class",
            "sample_count",
            "ratio_median",
            "ratio_q90",
            "ratio_q95",
            "ratio_q99",
            "ratio_gt10_count",
            "ratio_gt10_fraction",
            "ratio_gt20_count",
            "ratio_gt20_fraction",
        ],
        result
            .summaries
            .iter()
            .map(|row| {
                vec![
                    row.feature_class.clone(),
                    row.sample_count.to_string(),
                    python_float(row.ratio_median),
                    python_float(row.ratio_q90),
                    python_float(row.ratio_q95),
                    python_float(row.ratio_q99),
                    row.ratio_gt10_count.to_string(),
                    python_float(row.ratio_gt10_fraction),
                    row.ratio_gt20_count.to_string(),
                    python_float(row.ratio_gt20_fraction),
                ]
            })
            .collect(),
    )?;
    write_csv(
        &output_dir.join("aperture_peak_ratio_method_summary.csv"),
        &[
            "feature_class",
            "method_group",
            "sample_count",
            "ratio_median",
            "ratio_q90",
            "ratio_q95",
            "ratio_gt10_count",
            "ratio_gt10_fraction",
        ],
        result
            .method_summaries
            .iter()
            .map(|row| {
                vec![
                    row.feature_class.clone(),
                    row.method_group.clone(),
                    row.sample_count.to_string(),
                    python_float(row.ratio_median),
                    python_float(row.ratio_q90),
                    python_float(row.ratio_q95),
                    row.ratio_gt10_count.to_string(),
                    python_float(row.ratio_gt10_fraction),
                ]
            })
            .collect(),
    )?;
    write_csv(
        &output_dir.join("aperture_peak_ratio_targets.csv"),
        &[
            "detection_id",
            "feature_class",
            "peak",
            "flux",
            "flux_to_peak",
            "class_sample_count",
            "class_percentile",
        ],
        result
            .target_rows
            .iter()
            .map(|row| {
                vec![
                    row.detection_id.clone(),
                    row.feature_class.clone(),
                    python_float(row.peak),
                    python_float(row.flux),
                    python_float(row.flux_to_peak),
                    row.class_sample_count.to_string(),
                    python_float(row.class_percentile),
                ]
            })
            .collect(),
    )?;
    write_json(&output_dir.join("aperture_peak_ratio.json"), result)?;
    Ok(output_dir.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut catalog_path = None;
    let mut output_dir = PathBuf::from("tmp/aperture-peak-ratio-audit");
    let mut target_ids = DEFAULT_TARGETS
        .iter()
        .map(|target| (*target).to_owned())
        .collect::<Vec<_>>();
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
            "--target-ids" => {
                let mut values = Vec::new();
                if let Some(value) = inline_value {
                    if !value.is_empty() {
                        values.push(value.to_owned());
                    }
                } else {
                    while index + 1 < args.len() && !args[index + 1].starts_with('-') {
                        index += 1;
                        values.push(args[index].clone());
                    }
                }
                target_ids = values;
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
    let result = run(&catalog_path, &target_ids)?;
    write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-aperture-peak-ratio",
        "out_dir": output_dir,
        "result_path": output_dir.join("aperture_peak_ratio.json"),
        "result": result.as_dict(),
    }))
}

const CLI_HELP: &str = "按特征类别审计 source_catalog 的 flux/peak，不生成物理恒星概率\n\nUsage: rst19-aperture-peak-ratio <SOURCE_CATALOG.csv> [OPTIONS]\n\nOptions:\n  --target-ids [ID ...]\n  --out-dir DIR\n  -h, --help\n";

/// Run the standalone Rust command while preserving the Python CLI's output
/// and exit-code contract.
pub fn run_cli(args: &[String]) -> i32 {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        println!("{CLI_HELP}");
        return 0;
    }
    let result = match run_desktop(args) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("rst19-aperture-peak-ratio: {error}");
            return 2;
        }
    };
    let output = result["out_dir"].as_str().unwrap_or_default();
    println!("aperture/peak ratio audit: {output}");
    if let Some(rows) = result["result"]["target_rows"].as_array() {
        for row in rows {
            let detection_id = row["detection_id"].as_str().unwrap_or_default();
            let feature_class = row["feature_class"].as_str().unwrap_or_default();
            let ratio = row["flux_to_peak"].as_f64().unwrap_or_default();
            let percentile = row["class_percentile"].as_f64().unwrap_or_default();
            println!(
                "{detection_id}: {feature_class}, flux/peak={ratio:.6}, class percentile={percentile:.6}"
            );
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_and_method_summaries_and_target_percentiles_are_deterministic() {
        let root =
            std::env::temp_dir().join(format!("rst19-aperture-peak-ratio-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        fs::write(
            &catalog,
            "detection_id,feature_class,peak,flux,proposal_methods\n1,range,4,24,gaussian|dog_narrow\n2,range,10,20,gaussian\n3,crowded,2,50,gaussian\n4,crowded,5,5,dog_narrow\n5,crowded,0,99,gaussian\n",
        )
        .unwrap();
        let result = run(&catalog, &["1".into(), "3".into()]).unwrap();
        assert_eq!(result.summaries[1].feature_class, "range");
        assert_eq!(result.summaries[1].ratio_median, 4.0);
        assert_eq!(result.summaries[0].ratio_gt20_count, 1);
        assert_eq!(result.target_rows[0].flux_to_peak, 6.0);
        assert_eq!(result.target_rows[0].class_percentile, 1.0);
        assert!(result.method_summaries.iter().any(|row| {
            row.feature_class == "crowded"
                && row.method_group == "dog_only"
                && row.sample_count == 1
        }));
    }

    #[test]
    fn skips_nonfinite_rows_but_requires_the_catalog_contract() {
        let root = std::env::temp_dir().join(format!(
            "rst19-aperture-peak-ratio-invalid-{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        fs::write(
            &catalog,
            "detection_id,feature_class,peak,flux\na,x,1,NaN\nb,x,1,3\n",
        )
        .unwrap();
        assert_eq!(run(&catalog, &[]).unwrap().summaries[0].sample_count, 1);
        fs::write(&catalog, "detection_id,peak,flux\na,1,2\n").unwrap();
        assert!(run(&catalog, &[])
            .unwrap_err()
            .contains("missing required columns: feature_class"));
    }

    #[test]
    fn standalone_cli_help_documents_the_rust_command() {
        assert!(CLI_HELP.contains("Usage: rst19-aperture-peak-ratio"));
        assert!(CLI_HELP.contains("--target-ids"));
        assert!(CLI_HELP.contains("--out-dir"));
    }
}
