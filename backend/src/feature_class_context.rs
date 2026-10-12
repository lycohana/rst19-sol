use crate::Result;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

pub const CONTEXT_METRIC_FIELDS: [&str; 9] = [
    "flux_snr",
    "filter_snr",
    "fwhm",
    "ellipticity",
    "sharpness",
    "psf_support_pixels",
    "footprint_pixels",
    "centroid_shift_px",
    "peak",
];

const INTERPRETATION_GUARDRAILS: [&str; 3] = [
    "经验分位数只表示目标在当前算法类别中的相对位置，不是真星概率。",
    "feature_class 是按旗标优先级生成的审计标签，不是物理类别。",
    "高 SNR 或高 PSF 支持不能抵消值域、共享孔径和去混叠证据。",
];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureClassContextMetric {
    pub detection_id: i64,
    pub feature_class: String,
    pub metric: String,
    pub value: f64,
    pub class_count: usize,
    pub class_p10: f64,
    pub class_median: f64,
    pub class_p90: f64,
    pub empirical_percentile: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureClassContextTarget {
    pub detection_id: i64,
    pub feature_class: String,
    pub feature_class_label: String,
    pub quality_passed: bool,
    pub flags: String,
    pub class_count: usize,
    pub class_quality_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureClassContextResult {
    pub catalog_path: String,
    pub targets: Vec<FeatureClassContextTarget>,
    pub metrics: Vec<FeatureClassContextMetric>,
}

impl FeatureClassContextResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "targets": self.targets,
            "metrics": self.metrics,
            "interpretation_guardrails": INTERPRETATION_GUARDRAILS,
        })
    }
}

struct CatalogRecord {
    detection_id: i64,
    feature_class: String,
    feature_class_label: String,
    quality_passed: String,
    flags: String,
    values: BTreeMap<String, Option<f64>>,
}

struct CatalogIndex {
    records: Vec<CatalogRecord>,
    by_id: HashMap<i64, usize>,
    by_class: HashMap<String, Vec<usize>>,
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

fn quantile(values: &[f64], probability: f64) -> Result<f64> {
    if values.is_empty() {
        return Err("无法对空指标集合计算经验分位数".into());
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

fn read_catalog(path: &Path, metric_fields: &[String]) -> Result<CatalogIndex> {
    let bytes = fs::read(path).map_err(|error| format!("无法读取源级 CSV：{path:?}: {error}"))?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let mut records = Vec::new();
    for row in reader.records() {
        let row = row.map_err(|error| error.to_string())?;
        let mut indices = BTreeMap::new();
        for name in [
            "detection_id",
            "feature_class",
            "feature_class_label",
            "quality_passed",
            "flags",
        ]
        .into_iter()
        .chain(metric_fields.iter().map(String::as_str))
        {
            if let Some(index) = headers
                .iter()
                .enumerate()
                .filter_map(|(index, header)| (header == name).then_some(index))
                .last()
            {
                indices.insert(name, index);
            }
        }
        let required_columns = [
            "detection_id",
            "feature_class",
            "feature_class_label",
            "quality_passed",
            "flags",
        ];
        let mut missing = required_columns
            .iter()
            .filter(|name| !indices.contains_key(**name))
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        missing.sort();
        if !missing.is_empty() {
            return Err(format!("源级 CSV 缺少列：{}", missing.join(", ")));
        }
        let detection_id_raw = field(&row, indices["detection_id"]);
        let detection_id = detection_id_raw
            .parse::<i64>()
            .map_err(|_| format!("源级 CSV 存在非法 detection_id：{detection_id_raw:?}"))?;
        let values = metric_fields
            .iter()
            .filter(|metric| !metric.trim().is_empty())
            .map(|metric| {
                (
                    metric.clone(),
                    indices
                        .get(metric.as_str())
                        .and_then(|index| finite_float(field(&row, *index))),
                )
            })
            .collect();
        records.push(CatalogRecord {
            detection_id,
            feature_class: field(&row, indices["feature_class"]).to_owned(),
            feature_class_label: field(&row, indices["feature_class_label"]).to_owned(),
            quality_passed: field(&row, indices["quality_passed"]).to_owned(),
            flags: field(&row, indices["flags"]).to_owned(),
            values,
        });
    }
    if records.is_empty() {
        return Err(format!("源级 CSV 为空：{path:?}"));
    }
    let mut by_id = HashMap::new();
    let mut by_class: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, record) in records.iter().enumerate() {
        if by_id.insert(record.detection_id, index).is_some() {
            return Err(format!(
                "源级 CSV 存在重复 detection_id：{}",
                record.detection_id
            ));
        }
        by_class
            .entry(record.feature_class.clone())
            .or_default()
            .push(index);
    }
    Ok(CatalogIndex {
        records,
        by_id,
        by_class,
    })
}

pub fn run(
    catalog_path: &Path,
    target_ids: &[i64],
    metric_fields: &[String],
) -> Result<FeatureClassContextResult> {
    let mut requested_ids = Vec::new();
    let mut seen_targets = HashSet::new();
    for target_id in target_ids {
        if seen_targets.insert(*target_id) {
            requested_ids.push(*target_id);
        }
    }
    if requested_ids.is_empty() {
        return Err("至少需要一个 target detection ID".into());
    }
    let supported = CONTEXT_METRIC_FIELDS.into_iter().collect::<HashSet<_>>();
    let mut unknown_metrics = metric_fields
        .iter()
        .filter(|metric| !supported.contains(metric.as_str()))
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    unknown_metrics.sort();
    if !unknown_metrics.is_empty() {
        return Err(format!(
            "不支持的类别上下文指标：{}",
            unknown_metrics.join(", ")
        ));
    }
    let catalog = read_catalog(catalog_path, metric_fields)?;
    let mut missing_targets = requested_ids
        .iter()
        .filter(|target_id| !catalog.by_id.contains_key(target_id))
        .copied()
        .collect::<Vec<_>>();
    missing_targets.sort_unstable();
    if !missing_targets.is_empty() {
        return Err(format!("源级 CSV 找不到 detection_id：{missing_targets:?}"));
    }

    let mut targets = Vec::new();
    let mut metrics = Vec::new();
    for detection_id in requested_ids {
        let row = &catalog.records[catalog.by_id[&detection_id]];
        let class_rows = &catalog.by_class[&row.feature_class];
        let class_quality_count = class_rows
            .iter()
            .filter(|index| {
                catalog.records[**index]
                    .quality_passed
                    .eq_ignore_ascii_case("true")
            })
            .count();
        targets.push(FeatureClassContextTarget {
            detection_id,
            feature_class: row.feature_class.clone(),
            feature_class_label: row.feature_class_label.clone(),
            quality_passed: row.quality_passed.eq_ignore_ascii_case("true"),
            flags: row.flags.clone(),
            class_count: class_rows.len(),
            class_quality_count,
        });
        for metric in metric_fields {
            let value = row.values.get(metric).copied().flatten();
            let mut values = class_rows
                .iter()
                .filter_map(|index| {
                    catalog.records[*index]
                        .values
                        .get(metric)
                        .copied()
                        .flatten()
                })
                .collect::<Vec<_>>();
            let Some(value) = value else {
                continue;
            };
            if values.is_empty() {
                continue;
            }
            values.sort_by(|left, right| left.partial_cmp(right).unwrap());
            let empirical_percentile = values
                .iter()
                .filter(|candidate| **candidate <= value)
                .count() as f64
                / values.len() as f64;
            metrics.push(FeatureClassContextMetric {
                detection_id,
                feature_class: row.feature_class.clone(),
                metric: metric.clone(),
                value,
                class_count: values.len(),
                class_p10: quantile(&values, 0.1)?,
                class_median: quantile(&values, 0.5)?,
                class_p90: quantile(&values, 0.9)?,
                empirical_percentile,
            });
        }
    }
    Ok(FeatureClassContextResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        targets,
        metrics,
    })
}

fn text_cell(value: &str) -> String {
    value.to_owned()
}

fn write_csv<T>(
    path: &Path,
    headers: &[&str],
    rows: &[T],
    cells: impl Fn(&T) -> Vec<String>,
) -> Result<()>
where
    T: Serialize,
{
    if rows.is_empty() {
        fs::write(path, [vec![0xef, 0xbb, 0xbf], vec![b'\n']].concat())
            .map_err(|error| error.to_string())?;
        return Ok(());
    }
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(cells(row))
            .map_err(|error| error.to_string())?;
    }
    let csv_bytes = writer.into_inner().map_err(|error| error.to_string())?;
    let mut bom_csv = vec![0xef, 0xbb, 0xbf];
    bom_csv.extend(csv_bytes);
    fs::write(path, bom_csv).map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &FeatureClassContextResult, output_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    let json_bytes =
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?;
    fs::write(output_dir.join("feature_class_context.json"), json_bytes)
        .map_err(|error| error.to_string())?;
    const TARGET_HEADERS: [&str; 7] = [
        "detection_id",
        "feature_class",
        "feature_class_label",
        "quality_passed",
        "flags",
        "class_count",
        "class_quality_count",
    ];
    write_csv(
        &output_dir.join("feature_class_context_targets.csv"),
        &TARGET_HEADERS,
        &result.targets,
        |row| {
            vec![
                row.detection_id.to_string(),
                text_cell(&row.feature_class),
                text_cell(&row.feature_class_label),
                if row.quality_passed { "True" } else { "False" }.into(),
                text_cell(&row.flags),
                row.class_count.to_string(),
                row.class_quality_count.to_string(),
            ]
        },
    )?;
    const METRIC_HEADERS: [&str; 9] = [
        "detection_id",
        "feature_class",
        "metric",
        "value",
        "class_count",
        "class_p10",
        "class_median",
        "class_p90",
        "empirical_percentile",
    ];
    write_csv(
        &output_dir.join("feature_class_context_metrics.csv"),
        &METRIC_HEADERS,
        &result.metrics,
        |row| {
            vec![
                row.detection_id.to_string(),
                row.feature_class.clone(),
                row.metric.clone(),
                json!(row.value).to_string(),
                row.class_count.to_string(),
                json!(row.class_p10).to_string(),
                json!(row.class_median).to_string(),
                json!(row.class_p90).to_string(),
                json!(row.empirical_percentile).to_string(),
            ]
        },
    )?;
    Ok(output_dir.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut catalog_path = None;
    let mut output_dir = PathBuf::from("tmp/feature-class-context");
    let mut target_ids = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--out-dir" || arg.starts_with("--out-dir=") {
            let value = if let Some(value) = arg.strip_prefix("--out-dir=") {
                value
            } else {
                index += 1;
                args.get(index).ok_or("--out-dir requires a value")?
            };
            output_dir = PathBuf::from(value);
        } else if arg == "--target-id" || arg.starts_with("--target-id=") {
            let value = if let Some(value) = arg.strip_prefix("--target-id=") {
                value
            } else {
                index += 1;
                args.get(index).ok_or("--target-id requires a value")?
            };
            target_ids.push(
                value
                    .parse::<i64>()
                    .map_err(|error| format!("invalid target detection ID: {error}"))?,
            );
        } else if arg.starts_with('-') {
            return Err(format!("unknown argument {arg}"));
        } else if catalog_path.replace(PathBuf::from(arg)).is_some() {
            return Err("only one source catalog is allowed".into());
        }
        index += 1;
    }
    let catalog_path = catalog_path.ok_or("a source catalog CSV is required")?;
    let metric_fields = CONTEXT_METRIC_FIELDS
        .iter()
        .map(|field| (*field).to_owned())
        .collect::<Vec<_>>();
    let result = run(&catalog_path, &target_ids, &metric_fields)?;
    write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-feature-class-context",
        "out_dir": output_dir,
        "result_path": output_dir.join("feature_class_context.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_percentiles_and_duplicate_targets_match_python_contract() {
        let dir = std::env::temp_dir().join(format!("rst19-class-context-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let catalog = dir.join("catalog.csv");
        fs::write(
            &catalog,
            "detection_id,feature_class,feature_class_label,quality_passed,flags,flux_snr\n1,compact,Compact,true,,5\n2,compact,Compact,false,LOW,10\n3,compact,Compact,true,,15\n",
        )
        .unwrap();
        let result = run(&catalog, &[2, 1, 2], &["flux_snr".into()]).unwrap();
        assert_eq!(result.targets.len(), 2);
        assert_eq!(result.targets[0].detection_id, 2);
        assert_eq!(result.targets[0].class_quality_count, 2);
        assert_eq!(result.metrics[0].class_p10, 6.0);
        assert_eq!(result.metrics[0].class_median, 10.0);
        assert_eq!(result.metrics[0].class_p90, 14.0);
        assert_eq!(result.metrics[0].empirical_percentile, 2.0 / 3.0);
    }
}
