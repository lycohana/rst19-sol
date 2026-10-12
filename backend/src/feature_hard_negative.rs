use crate::Result;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

pub const HARD_NEGATIVE_METRIC_FIELDS: [&str; 3] = ["flux_snr", "filter_snr", "peak"];
const REQUIRED_FIELDS: [&str; 5] = [
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "flags",
];
pub const HARD_NEGATIVE_DETAIL_FIELDS: [&str; 9] = [
    "flux_snr",
    "filter_snr",
    "filter_flux_snr_ratio",
    "peak",
    "fwhm",
    "ellipticity",
    "sharpness",
    "psf_support_pixels",
    "footprint_pixels",
];
const INTERPRETATION_GUARDRAILS: [&str; 4] = [
    "quality_passed=False 是当前算法质量层的结果，不是物理伪影真值。",
    "类别内排名只表示高显著性与拒绝规则的冲突，不能解释为恒星概率。",
    "高 SNR 候选仍需值域、独占像素、去混叠、PSF 留出和跨帧身份复核。",
    "feature_class 是按旗标和形态生成的算法审计标签，不是物理类别。",
];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureHardNegativeSummary {
    pub feature_class: String,
    pub feature_class_label: Option<String>,
    pub class_count: usize,
    pub class_quality_count: usize,
    pub class_rejected_count: usize,
    pub class_rejected_fraction: f64,
    pub metric: String,
    pub top_metric_value: Option<f64>,
    pub top_detection_id: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureHardNegativeRow {
    pub detection_id: i64,
    pub rank_within_class: usize,
    pub feature_class: String,
    pub feature_class_label: Option<String>,
    pub flags: Option<String>,
    pub metric: String,
    pub metric_value: f64,
    pub flux_snr: Option<f64>,
    pub filter_snr: Option<f64>,
    pub filter_flux_snr_ratio: Option<f64>,
    pub peak: Option<f64>,
    pub fwhm: Option<f64>,
    pub ellipticity: Option<f64>,
    pub sharpness: Option<f64>,
    pub psf_support_pixels: Option<f64>,
    pub footprint_pixels: Option<f64>,
    pub class_count: usize,
    pub class_rejected_count: usize,
    pub class_quality_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureHardNegativeAuditResult {
    pub catalog_path: String,
    pub metric: String,
    pub top_n: usize,
    pub excluded_classes: Vec<String>,
    pub summaries: Vec<FeatureHardNegativeSummary>,
    pub rows: Vec<FeatureHardNegativeRow>,
}

impl FeatureHardNegativeAuditResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "catalog_path": self.catalog_path,
            "metric": self.metric,
            "top_n": self.top_n,
            "excluded_classes": self.excluded_classes,
            "summaries": self.summaries,
            "rows": self.rows,
            "interpretation_guardrails": INTERPRETATION_GUARDRAILS,
        })
    }
}

type Record = HashMap<String, Option<String>>;

fn row_value<'a>(row: &'a Record, field: &str) -> Option<&'a str> {
    row.get(field).and_then(Option::as_deref)
}

fn finite_float(row: &Record, field: &str) -> Option<f64> {
    let raw = row_value(row, field)?;
    if matches!(raw, "" | "nan" | "NaN" | "None") {
        return None;
    }
    raw.trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn is_quality(row: &Record) -> bool {
    row_value(row, "quality_passed")
        .unwrap_or("")
        .trim()
        .eq_ignore_ascii_case("true")
}

fn read_rows(catalog_path: &Path) -> Result<Vec<Record>> {
    let bytes = fs::read(catalog_path)
        .map_err(|_| format!("无法读取源级 CSV：{}", catalog_path.to_string_lossy()))?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let mut rows = Vec::new();
    for csv_row in reader.records() {
        let csv_row = csv_row.map_err(|error| error.to_string())?;
        let mut row = Record::new();
        for (index, header) in headers.iter().enumerate() {
            // DictReader gives the final duplicate field name precedence.
            row.insert(header.to_owned(), csv_row.get(index).map(str::to_owned));
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("源级 CSV 为空：{}", catalog_path.to_string_lossy()));
    }
    let first_keys: HashSet<&str> = rows[0].keys().map(String::as_str).collect();
    let mut missing = REQUIRED_FIELDS
        .iter()
        .filter(|field| !first_keys.contains(**field))
        .copied()
        .collect::<Vec<_>>();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(format!("源级 CSV 缺少列：{}", missing.join(", ")));
    }
    Ok(rows)
}

fn python_int(value: Option<&str>) -> std::result::Result<i64, String> {
    let raw = value.unwrap_or("None").trim();
    raw.parse::<i64>().map_err(|_| raw.to_owned())
}

pub fn run(
    catalog_path: &Path,
    top_n: usize,
    metric: &str,
    exclude_classes: &[String],
) -> Result<FeatureHardNegativeAuditResult> {
    if top_n == 0 {
        return Err("top_n 必须为正整数".into());
    }
    if !HARD_NEGATIVE_METRIC_FIELDS.contains(&metric) {
        return Err(format!(
            "不支持的 hard-negative 排名指标：{metric:?}；可选项为 {}",
            HARD_NEGATIVE_METRIC_FIELDS.join(", ")
        ));
    }
    let mut excluded = Vec::new();
    for class in exclude_classes {
        if !excluded.contains(class) {
            excluded.push(class.clone());
        }
    }
    let rows = read_rows(catalog_path)?;
    let mut by_class = BTreeMap::<String, Vec<(i64, Record)>>::new();
    let mut seen_ids = HashSet::new();
    for row in rows {
        let raw_id = row_value(&row, "detection_id");
        let detection_id = python_int(raw_id).map_err(|_| {
            format!(
                "源级 CSV 存在非法 detection_id：{:?}",
                raw_id.unwrap_or("None")
            )
        })?;
        if !seen_ids.insert(detection_id) {
            return Err(format!("源级 CSV 存在重复 detection_id：{detection_id}"));
        }
        let feature_class = row_value(&row, "feature_class")
            .unwrap_or("None")
            .to_owned();
        if !excluded.contains(&feature_class) {
            by_class
                .entry(feature_class)
                .or_default()
                .push((detection_id, row));
        }
    }

    let mut summaries = Vec::new();
    let mut audit_rows = Vec::new();
    for (feature_class, class_rows) in by_class {
        let rejected: Vec<&(i64, Record)> = class_rows
            .iter()
            .filter(|(_, row)| !is_quality(row))
            .collect();
        let mut ranked: Vec<(f64, i64, &Record)> = rejected
            .iter()
            .filter_map(|(id, row)| finite_float(row, metric).map(|value| (value, *id, row)))
            .collect();
        ranked.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap()
                .then_with(|| left.1.cmp(&right.1))
        });
        let quality_count = class_rows.len() - rejected.len();
        let first = &class_rows[0].1;
        summaries.push(FeatureHardNegativeSummary {
            feature_class: feature_class.clone(),
            feature_class_label: row_value(first, "feature_class_label").map(str::to_owned),
            class_count: class_rows.len(),
            class_quality_count: quality_count,
            class_rejected_count: rejected.len(),
            class_rejected_fraction: rejected.len() as f64 / class_rows.len() as f64,
            metric: metric.to_owned(),
            top_metric_value: ranked.first().map(|item| item.0),
            top_detection_id: ranked.first().map(|item| item.1),
        });

        for (rank, (metric_value, detection_id, row)) in ranked.into_iter().take(top_n).enumerate()
        {
            let flux_snr = finite_float(row, "flux_snr");
            let filter_snr = finite_float(row, "filter_snr");
            let filter_flux_snr_ratio = match (flux_snr, filter_snr) {
                (Some(flux), Some(filter)) if flux.abs() > 1e-12 => Some(filter / flux),
                _ => None,
            };
            audit_rows.push(FeatureHardNegativeRow {
                detection_id,
                rank_within_class: rank + 1,
                feature_class: feature_class.clone(),
                feature_class_label: row_value(row, "feature_class_label").map(str::to_owned),
                flags: row_value(row, "flags").map(str::to_owned),
                metric: metric.to_owned(),
                metric_value,
                flux_snr,
                filter_snr,
                filter_flux_snr_ratio,
                peak: finite_float(row, "peak"),
                fwhm: finite_float(row, "fwhm"),
                ellipticity: finite_float(row, "ellipticity"),
                sharpness: finite_float(row, "sharpness"),
                psf_support_pixels: finite_float(row, "psf_support_pixels"),
                footprint_pixels: finite_float(row, "footprint_pixels"),
                class_count: class_rows.len(),
                class_rejected_count: rejected.len(),
                class_quality_count: quality_count,
            });
        }
    }
    Ok(FeatureHardNegativeAuditResult {
        catalog_path: catalog_path.to_string_lossy().into_owned(),
        metric: metric.to_owned(),
        top_n,
        excluded_classes: excluded,
        summaries,
        rows: audit_rows,
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

fn optional_id(value: Option<i64>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn write_csv(path: &Path, headers: &[&str], rows: Vec<Vec<String>>) -> Result<()> {
    if rows.is_empty() {
        return fs::write(path, [vec![0xef, 0xbb, 0xbf], b"\n".to_vec()].concat())
            .map_err(|error| error.to_string());
    }
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
    result: &FeatureHardNegativeAuditResult,
    output_dir: &Path,
) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    let summary_rows = result
        .summaries
        .iter()
        .map(|row| {
            vec![
                row.feature_class.clone(),
                row.feature_class_label.clone().unwrap_or_default(),
                row.class_count.to_string(),
                row.class_quality_count.to_string(),
                row.class_rejected_count.to_string(),
                python_float(row.class_rejected_fraction),
                row.metric.clone(),
                optional_float(row.top_metric_value),
                optional_id(row.top_detection_id),
            ]
        })
        .collect();
    write_csv(
        &output_dir.join("feature_hard_negative_summary.csv"),
        &[
            "feature_class",
            "feature_class_label",
            "class_count",
            "class_quality_count",
            "class_rejected_count",
            "class_rejected_fraction",
            "metric",
            "top_metric_value",
            "top_detection_id",
        ],
        summary_rows,
    )?;
    let detail_rows = result
        .rows
        .iter()
        .map(|row| {
            vec![
                row.detection_id.to_string(),
                row.rank_within_class.to_string(),
                row.feature_class.clone(),
                row.feature_class_label.clone().unwrap_or_default(),
                row.flags.clone().unwrap_or_default(),
                row.metric.clone(),
                python_float(row.metric_value),
                optional_float(row.flux_snr),
                optional_float(row.filter_snr),
                optional_float(row.filter_flux_snr_ratio),
                optional_float(row.peak),
                optional_float(row.fwhm),
                optional_float(row.ellipticity),
                optional_float(row.sharpness),
                optional_float(row.psf_support_pixels),
                optional_float(row.footprint_pixels),
                row.class_count.to_string(),
                row.class_rejected_count.to_string(),
                row.class_quality_count.to_string(),
            ]
        })
        .collect();
    write_csv(
        &output_dir.join("feature_hard_negative_rows.csv"),
        &[
            "detection_id",
            "rank_within_class",
            "feature_class",
            "feature_class_label",
            "flags",
            "metric",
            "metric_value",
            "flux_snr",
            "filter_snr",
            "filter_flux_snr_ratio",
            "peak",
            "fwhm",
            "ellipticity",
            "sharpness",
            "psf_support_pixels",
            "footprint_pixels",
            "class_count",
            "class_rejected_count",
            "class_quality_count",
        ],
        detail_rows,
    )?;
    fs::write(
        output_dir.join("feature_hard_negative.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut catalog_path = None;
    let mut output_dir = PathBuf::from("tmp/feature-hard-negative");
    let mut top_n = 5usize;
    let mut metric = "flux_snr".to_owned();
    let mut include_other_rejected = false;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline_value) = arg
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((arg.as_str(), None));
        match name {
            "--out-dir" | "--top-n" | "--metric" => {
                let value = if let Some(value) = inline_value {
                    value
                } else {
                    index += 1;
                    args.get(index)
                        .ok_or_else(|| format!("{name} requires a value"))?
                };
                match name {
                    "--out-dir" => output_dir = PathBuf::from(value),
                    "--top-n" => {
                        top_n = value
                            .parse::<usize>()
                            .map_err(|error| format!("invalid top_n: {error}"))?
                    }
                    _ => metric = value.to_owned(),
                }
            }
            "--include-other-rejected" => include_other_rejected = true,
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
    let excluded = if include_other_rejected {
        Vec::new()
    } else {
        vec!["other_rejected".to_owned()]
    };
    let result = run(&catalog_path, top_n, &metric, &excluded)?;
    write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-feature-hard-negative",
        "out_dir": output_dir,
        "result_path": output_dir.join("feature_hard_negative.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_rejected_sources_by_metric_and_excludes_other_bucket() {
        let root = std::env::temp_dir().join(format!(
            "rst19-feature-hard-negative-{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("source.csv");
        fs::write(
            &catalog,
            "detection_id,feature_class,feature_class_label,quality_passed,flags,flux_snr,filter_snr,peak\n\
             1,compact_quality,紧凑,True,,2,4,20\n\
             2,compact_quality,紧凑,False,LOW_FLUX_SNR,12,24,120\n\
             3,compact_quality,紧凑,False,UNRESOLVED_BLEND,20,40,200\n\
             4,other_rejected,other,False,OTHER,100,200,1000\n",
        )
        .unwrap();
        let result = run(&catalog, 1, "flux_snr", &["other_rejected".into()]).unwrap();
        assert_eq!(result.summaries.len(), 1);
        assert_eq!(result.summaries[0].top_detection_id, Some(3));
        assert_eq!(result.rows[0].detection_id, 3);
        assert_eq!(result.rows[0].filter_flux_snr_ratio, Some(2.0));
        let _ = fs::remove_dir_all(root);
    }
}
