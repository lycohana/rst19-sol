use crate::Result;
use csv::{StringRecord, Terminator, WriterBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

pub const RAW_EVIDENCE_METRICS: [&str; 9] = [
    "core_fraction_median",
    "noise_median",
    "zero_count_max",
    "frames_zero_count_gt0",
    "frames_negative_extreme_count_gt0",
    "frames_repeated_code_count_gt0",
    "frames_peak_snr_ge5",
    "frames_flux_snr_ge5",
    "frames_support_ge3",
];

const INTERPRETATION_GUARDRAILS: [&str; 4] = [
    "raw summary 是有限抽样，不是全图 precision、完备率或真星率。",
    "feature_class 是检测器旗标优先级产生的互斥审计标签，不是物理类别。",
    "definition_overlap 显式标记了与类别/质量规则共享的字段，不能当作独立验证。",
    "core_fraction 和局部噪声可补充类别解释，但也不能单独区分恒星、噪声和伪影。",
];

const SUMMARY_HEADERS: [&str; 16] = [
    "feature_class",
    "feature_class_label",
    "sample_count",
    "control_sample_count",
    "diagnostic_sample_count",
    "metric",
    "evidence_axis",
    "metric_note",
    "definition_overlap",
    "finite_value_count",
    "p10",
    "median",
    "p90",
    "minimum",
    "maximum",
    "nonzero_fraction",
];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RawEvidenceSummary {
    pub feature_class: String,
    pub feature_class_label: String,
    pub sample_count: usize,
    pub control_sample_count: usize,
    pub diagnostic_sample_count: usize,
    pub metric: String,
    pub evidence_axis: String,
    pub metric_note: String,
    pub definition_overlap: String,
    pub finite_value_count: usize,
    pub p10: f64,
    pub median: f64,
    pub p90: f64,
    pub minimum: f64,
    pub maximum: f64,
    pub nonzero_fraction: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct RawEvidenceAuditResult {
    catalog_path: String,
    raw_summary_path: String,
    feature_classes: Vec<String>,
    metrics: Vec<String>,
    sample_counts: BTreeMap<String, usize>,
    origin_counts: BTreeMap<String, BTreeMap<String, usize>>,
    rows: Vec<RawEvidenceSummary>,
    interpretation_guardrails: Vec<String>,
}

fn metric_metadata(metric: &str) -> Option<(&'static str, &'static str, &'static str)> {
    Some(match metric {
        "core_fraction_median" => (
            "raw_profile",
            "局部核心能量占比；不直接等同于 PSF 拟合或恒星身份。",
            "间接相关：与形态/支持有关，但未直接作为首要类别旗标。",
        ),
        "noise_median" => (
            "local_noise",
            "局部背景环噪声；用于判断相同峰值在不同背景中的可比性。",
            "低重叠：局部噪声未直接定义首要类别。",
        ),
        "zero_count_max" => (
            "validity",
            "单源孔径内零值计数最大值；是原始有效性线索，不是坏点真值。",
            "低重叠：零值是值域/有效孔径线索，未直接定义首要类别。",
        ),
        "frames_zero_count_gt0" => (
            "validity_temporal",
            "15 帧中出现零值的帧数；描述值域/孔径状态的时间重复。",
            "低重叠：用于补充值域时间行为，未直接定义首要类别。",
        ),
        "frames_negative_extreme_count_gt0" => (
            "signed_value",
            "15 帧中出现极端负码的帧数；需结合编码说明解释。",
            "高重叠：与 NEGATIVE_OVERFLOW/范围审计共享值域信息。",
        ),
        "frames_repeated_code_count_gt0" => (
            "value_pattern",
            "15 帧中出现重复码的帧数；是工程模式线索，不是坏点证明。",
            "部分重叠：CODE_PATTERN 使用重复码与范围异常的组合。",
        ),
        "frames_peak_snr_ge5" => (
            "temporal_peak",
            "15 帧峰值 SNR 达到当前工程线的帧数；不是恒星概率。",
            "部分重叠：峰值 SNR 与候选显著性相关，但不是唯一类别条件。",
        ),
        "frames_flux_snr_ge5" => (
            "temporal_flux",
            "15 帧通量 SNR 达到当前工程线的帧数；受孔径和背景估计影响。",
            "高重叠：LOW_FLUX_SNR 直接使用通量 SNR。",
        ),
        "frames_support_ge3" => (
            "temporal_support",
            "15 帧满足 3×3 支持下限的帧数；是形态支持线索，不是独立真值。",
            "高重叠：INSUFFICIENT_PSF_SUPPORT 与支持像素直接相关。",
        ),
        _ => return None,
    })
}

fn path_text(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn csv_data(path: &Path) -> Result<(Vec<String>, Vec<StringRecord>)> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|_| format!("无法读取 CSV：{}", path_text(path)))?;
    let raw_headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let headers = raw_headers
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if index == 0 {
                value.strip_prefix('\u{feff}').unwrap_or(value).to_owned()
            } else {
                value.to_owned()
            }
        })
        .collect();
    let records = reader
        .records()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok((headers, records))
}

fn column_indices(headers: &[String]) -> HashMap<String, usize> {
    let mut indices = HashMap::new();
    for (index, header) in headers.iter().enumerate() {
        indices.insert(header.clone(), index);
    }
    indices
}

fn row_value<'a>(
    record: &'a StringRecord,
    indices: &HashMap<String, usize>,
    field: &str,
) -> Option<&'a str> {
    indices.get(field).and_then(|index| record.get(*index))
}

fn missing_columns(headers: &[String], required: impl IntoIterator<Item = String>) -> Vec<String> {
    let available: BTreeSet<&str> = headers.iter().map(String::as_str).collect();
    required
        .into_iter()
        .filter(|field| !available.contains(field.as_str()))
        .collect()
}

fn parse_id(raw: Option<&str>, field: &str) -> Result<i64> {
    let text = raw.unwrap_or("None").trim();
    if text.is_empty() {
        return Err(format!("CSV row has empty {field}"));
    }
    let value = text
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .map(|value| value.trunc() as i64)
        .ok_or_else(|| format!("CSV row has invalid {field}={text:?}"))?;
    if value < 0 {
        return Err(format!("CSV row has negative {field}={value}"));
    }
    Ok(value)
}

fn read_catalog(path: &Path) -> Result<BTreeMap<i64, (String, String)>> {
    let (headers, rows) =
        csv_data(path).map_err(|_| format!("无法读取源级 CSV：{}", path_text(path)))?;
    let indices = column_indices(&headers);
    let missing = missing_columns(
        &headers,
        ["detection_id", "feature_class", "feature_class_label"].map(str::to_owned),
    );
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let mut catalog = BTreeMap::new();
    for row in rows {
        let detection_id = parse_id(row_value(&row, &indices, "detection_id"), "detection_id")?;
        if catalog.contains_key(&detection_id) {
            return Err(format!(
                "source catalog contains duplicate detection_id={detection_id}"
            ));
        }
        let feature_class = row_value(&row, &indices, "feature_class")
            .unwrap_or("None")
            .trim()
            .to_owned();
        if feature_class.is_empty() {
            return Err(format!(
                "source catalog has empty feature_class for detection_id={detection_id}"
            ));
        }
        let label = row_value(&row, &indices, "feature_class_label")
            .unwrap_or("None")
            .trim();
        catalog.insert(
            detection_id,
            (
                feature_class.clone(),
                if label.is_empty() {
                    feature_class
                } else {
                    label.to_owned()
                },
            ),
        );
    }
    if catalog.is_empty() {
        return Err(format!("source catalog is empty: {}", path_text(path)));
    }
    Ok(catalog)
}

#[derive(Clone)]
struct RawRecord {
    detection_id: i64,
    origin: String,
    values: HashMap<String, String>,
}

fn read_raw_summary(
    path: &Path,
    catalog: &BTreeMap<i64, (String, String)>,
    metrics: &[String],
) -> Result<Vec<RawRecord>> {
    let (headers, rows) =
        csv_data(path).map_err(|_| format!("无法读取原始抽样摘要：{}", path_text(path)))?;
    let indices = column_indices(&headers);
    let mut required = BTreeSet::from(["detection_id".to_owned(), "origin".to_owned()]);
    required.extend(metrics.iter().cloned());
    let missing = missing_columns(&headers, required);
    if !missing.is_empty() {
        return Err(format!(
            "raw source summary is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let mut seen = HashSet::new();
    let mut records = Vec::with_capacity(rows.len());
    for row in rows {
        let detection_id = parse_id(row_value(&row, &indices, "detection_id"), "detection_id")?;
        if !seen.insert(detection_id) {
            return Err(format!(
                "raw source summary contains duplicate detection_id={detection_id}"
            ));
        }
        if !catalog.contains_key(&detection_id) {
            return Err(format!(
                "raw source summary detection_id={detection_id} is absent from source catalog"
            ));
        }
        let origin = row_value(&row, &indices, "origin")
            .unwrap_or("None")
            .trim()
            .to_owned();
        if origin.is_empty() {
            return Err(format!(
                "raw source summary has empty origin for detection_id={detection_id}"
            ));
        }
        let values = indices
            .iter()
            .filter_map(|(field, index)| {
                row.get(*index)
                    .map(|value| (field.clone(), value.to_owned()))
            })
            .collect();
        records.push(RawRecord {
            detection_id,
            origin,
            values,
        });
    }
    if records.is_empty() {
        return Err(format!("raw source summary is empty: {}", path_text(path)));
    }
    Ok(records)
}

fn finite_value(record: &RawRecord, metric: &str) -> Option<f64> {
    record
        .values
        .get(metric)
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = (sorted.len() - 1) as f64 * q;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

fn normalize_names(values: Option<Vec<String>>) -> Vec<String> {
    let mut seen = HashSet::new();
    values
        .unwrap_or_default()
        .into_iter()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty() && seen.insert(value.clone()))
        .collect()
}

pub fn run(
    catalog_path: &Path,
    raw_summary_path: &Path,
    feature_classes: Option<Vec<String>>,
    metrics: Option<Vec<String>>,
) -> Result<Value> {
    let requested_metrics = if let Some(metrics) = metrics {
        normalize_names(Some(metrics))
    } else {
        RAW_EVIDENCE_METRICS
            .iter()
            .map(|value| (*value).to_owned())
            .collect()
    };
    if requested_metrics.is_empty() {
        return Err("metrics 不能为空".into());
    }
    if let Some(unknown) = requested_metrics
        .iter()
        .find(|metric| !RAW_EVIDENCE_METRICS.contains(&metric.as_str()))
    {
        return Err(format!("unknown raw evidence metric: {unknown:?}"));
    }
    let catalog = read_catalog(catalog_path)?;
    let raw_records = read_raw_summary(raw_summary_path, &catalog, &requested_metrics)?;
    let sampled_classes: BTreeSet<String> = raw_records
        .iter()
        .map(|record| catalog[&record.detection_id].0.clone())
        .filter(|feature_class| feature_class != "other_rejected")
        .collect();
    let requested_classes = match feature_classes {
        None => sampled_classes.iter().cloned().collect::<Vec<_>>(),
        Some(classes) => normalize_names(Some(classes)),
    };
    if requested_classes.is_empty() {
        return Err("feature_classes 不能为空，且抽样表中没有可用类别".into());
    }
    let catalog_classes: BTreeSet<&str> = catalog.values().map(|item| item.0.as_str()).collect();
    if let Some(unknown) = requested_classes
        .iter()
        .find(|feature_class| !catalog_classes.contains(feature_class.as_str()))
    {
        return Err(format!("source catalog has no feature_class={unknown:?}"));
    }
    if let Some(missing) = requested_classes
        .iter()
        .find(|feature_class| !sampled_classes.contains(feature_class.as_str()))
    {
        return Err(format!(
            "raw source summary has no sampled rows for feature_class={missing:?}"
        ));
    }

    let requested_class_set: HashSet<&str> = requested_classes.iter().map(String::as_str).collect();
    let mut sample_counts = BTreeMap::new();
    let mut origin_counts: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut values_by_class_metric: HashMap<(String, String), Vec<f64>> = HashMap::new();
    let mut labels_by_class: HashMap<String, BTreeSet<String>> = HashMap::new();
    for feature_class in &requested_classes {
        sample_counts.insert(feature_class.clone(), 0);
        origin_counts.insert(feature_class.clone(), BTreeMap::new());
        labels_by_class.insert(feature_class.clone(), BTreeSet::new());
    }
    for record in &raw_records {
        let (feature_class, label) = &catalog[&record.detection_id];
        if !requested_class_set.contains(feature_class.as_str()) {
            continue;
        }
        *sample_counts
            .get_mut(feature_class)
            .expect("requested class initialized") += 1;
        *origin_counts
            .get_mut(feature_class)
            .expect("requested class initialized")
            .entry(record.origin.clone())
            .or_default() += 1;
        labels_by_class
            .get_mut(feature_class)
            .expect("requested class initialized")
            .insert(label.clone());
        for metric in &requested_metrics {
            if let Some(value) = finite_value(record, metric) {
                values_by_class_metric
                    .entry((feature_class.clone(), metric.clone()))
                    .or_default()
                    .push(value);
            }
        }
    }

    let mut rows = Vec::new();
    for feature_class in &requested_classes {
        let labels = &labels_by_class[feature_class];
        let label = labels
            .iter()
            .next()
            .ok_or_else(|| {
                format!("raw source summary has no label for feature_class={feature_class:?}")
            })?
            .clone();
        for metric in &requested_metrics {
            let Some(values) =
                values_by_class_metric.get_mut(&(feature_class.clone(), metric.clone()))
            else {
                continue;
            };
            values.sort_by(f64::total_cmp);
            let (evidence_axis, metric_note, definition_overlap) = metric_metadata(metric)
                .ok_or_else(|| format!("unknown raw evidence metric: {metric:?}"))?;
            let nonzero_count = values.iter().filter(|value| **value > 0.0).count();
            rows.push(RawEvidenceSummary {
                feature_class: feature_class.clone(),
                feature_class_label: label.clone(),
                sample_count: sample_counts[feature_class],
                control_sample_count: origin_counts[feature_class]
                    .get("control")
                    .copied()
                    .unwrap_or(0),
                diagnostic_sample_count: origin_counts[feature_class]
                    .get("diagnostic")
                    .copied()
                    .unwrap_or(0),
                metric: metric.clone(),
                evidence_axis: evidence_axis.to_owned(),
                metric_note: metric_note.to_owned(),
                definition_overlap: definition_overlap.to_owned(),
                finite_value_count: values.len(),
                p10: quantile(values, 0.1),
                median: quantile(values, 0.5),
                p90: quantile(values, 0.9),
                minimum: values[0],
                maximum: values[values.len() - 1],
                nonzero_fraction: nonzero_count as f64 / values.len() as f64,
            });
        }
    }
    if rows.is_empty() {
        return Err("raw source summary has no finite requested metrics".into());
    }
    serde_json::to_value(RawEvidenceAuditResult {
        catalog_path: path_text(catalog_path),
        raw_summary_path: path_text(raw_summary_path),
        feature_classes: requested_classes,
        metrics: requested_metrics,
        sample_counts,
        origin_counts,
        rows,
        interpretation_guardrails: INTERPRETATION_GUARDRAILS
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
    })
    .map_err(|error| error.to_string())
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

fn write_csv(path: &Path, rows: &[RawEvidenceSummary]) -> Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .terminator(Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(SUMMARY_HEADERS)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record([
                row.feature_class.clone(),
                row.feature_class_label.clone(),
                row.sample_count.to_string(),
                row.control_sample_count.to_string(),
                row.diagnostic_sample_count.to_string(),
                row.metric.clone(),
                row.evidence_axis.clone(),
                row.metric_note.clone(),
                row.definition_overlap.clone(),
                row.finite_value_count.to_string(),
                python_float(row.p10),
                python_float(row.median),
                python_float(row.p90),
                python_float(row.minimum),
                python_float(row.maximum),
                python_float(row.nonzero_fraction),
            ])
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    Ok(())
}

pub fn write_artifacts(result: &Value, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let rows: Vec<RawEvidenceSummary> =
        serde_json::from_value(result["rows"].clone()).map_err(|error| error.to_string())?;
    write_csv(&out_dir.join("feature_raw_evidence_summary.csv"), &rows)?;
    fs::write(
        out_dir.join("feature_raw_evidence_audit.json"),
        serde_json::to_vec_pretty(result).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(out_dir.to_owned())
}

struct CliArgs {
    catalog: PathBuf,
    raw_summary: PathBuf,
    feature_classes: Vec<String>,
    out_dir: PathBuf,
}

fn parse_args(args: &[String]) -> Result<CliArgs> {
    let mut positional = Vec::new();
    let mut feature_classes = Vec::new();
    let mut out_dir = PathBuf::from("tmp/feature-raw-evidence-audit");
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if let Some(value) = arg.strip_prefix("--feature-class=") {
            feature_classes.push(value.to_owned());
        } else if arg == "--feature-class" {
            index += 1;
            feature_classes.push(args.get(index).ok_or("--feature-class 缺少值")?.clone());
        } else if let Some(value) = arg.strip_prefix("--out-dir=") {
            out_dir = PathBuf::from(value);
        } else if arg == "--out-dir" {
            index += 1;
            out_dir = PathBuf::from(args.get(index).ok_or("--out-dir 缺少值")?);
        } else if arg == "--" {
            positional.extend(args[index + 1..].iter().cloned());
            break;
        } else if arg.starts_with('-') {
            return Err(format!("unrecognized argument: {arg}"));
        } else {
            positional.push(arg.clone());
        }
        index += 1;
    }
    if positional.len() != 2 {
        return Err("需要 catalog 与 raw_summary 两个 CSV 路径".into());
    }
    Ok(CliArgs {
        catalog: PathBuf::from(&positional[0]),
        raw_summary: PathBuf::from(&positional[1]),
        feature_classes,
        out_dir,
    })
}

fn execute_cli(args: &[String]) -> Result<(Value, PathBuf)> {
    let parsed = parse_args(args)?;
    let feature_classes = (!parsed.feature_classes.is_empty()).then_some(parsed.feature_classes);
    let result = run(&parsed.catalog, &parsed.raw_summary, feature_classes, None)?;
    let out_dir = write_artifacts(&result, &parsed.out_dir)?;
    Ok((result, out_dir))
}

fn print_summary(result: &Value, output: &Path) -> Result<()> {
    println!(
        "rst19-feature-raw-evidence-audit: 结果已写入 {}",
        path_text(output)
    );
    let classes = result["feature_classes"]
        .as_array()
        .ok_or("feature_classes missing")?;
    let counts = &result["sample_counts"];
    let summaries = classes
        .iter()
        .filter_map(Value::as_str)
        .map(|feature_class| {
            format!(
                "{feature_class}={}",
                counts[feature_class].as_u64().unwrap_or(0)
            )
        })
        .collect::<Vec<_>>();
    println!("类别样本数：{}", summaries.join(", "));
    let metrics = result["metrics"].as_array().ok_or("metrics missing")?;
    let metrics = metrics.iter().filter_map(Value::as_str).collect::<Vec<_>>();
    let rows = result["rows"].as_array().ok_or("rows missing")?;
    println!(
        "证据轴记录：{} 条；可用字段：{}",
        rows.len(),
        metrics.join(", ")
    );
    Ok(())
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("连接 source_catalog 与原始 FITS 抽样摘要，比较各特征类别的局部证据\n\nUsage: rst19-feature-raw-evidence-audit <catalog.csv> <raw_summary.csv> [--feature-class CLASS]... [--out-dir PATH]");
        return Ok(());
    }
    let (result, output) = execute_cli(args)?;
    print_summary(&result, &output)
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let (result, output) = execute_cli(args)?;
    Ok(json!({
        "tool": "rst19-feature-raw-evidence-audit",
        "out_dir": path_text(&output),
        "result_path": path_text(&output.join("feature_raw_evidence_audit.json")),
        "result": result,
    }))
}

#[cfg(test)]
mod tests {
    use super::{quantile, RAW_EVIDENCE_METRICS};

    #[test]
    fn linear_quantiles_match_numpy_percentile_interpolation() {
        assert!((quantile(&[0.7, 0.8], 0.1) - 0.71).abs() < 1e-12);
        assert!((quantile(&[0.7, 0.8], 0.5) - 0.75).abs() < 1e-12);
        assert!((quantile(&[0.7, 0.8], 0.9) - 0.79).abs() < 1e-12);
    }

    #[test]
    fn metric_order_matches_the_public_csv_contract() {
        assert_eq!(RAW_EVIDENCE_METRICS[0], "core_fraction_median");
        assert_eq!(RAW_EVIDENCE_METRICS[8], "frames_support_ge3");
    }
}
