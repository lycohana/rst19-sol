//! Category-level synchronization among candidate persistence, quality response,
//! and local PSF evidence axes.
//!
//! This audit reads only the existing feature evidence matrix. Its conditional
//! Q|P value is detector-level descriptive context, not per-source precision,
//! false-discovery rate, a star probability, or physical identity.

use crate::Result;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeSet, HashSet},
    fs,
    path::{Path, PathBuf},
};

const REQUIRED_COLUMNS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "candidate_count",
    "candidate_presence_ge_required_count",
    "quality_response_ge_required_count",
    "spatial_local_correlation_median",
    "spatial_local_residual_median",
];

const SUMMARY_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "candidate_count",
    "candidate_persistent_count",
    "quality_response_count",
    "candidate_persistence_fraction",
    "quality_response_fraction",
    "quality_response_given_candidate_persistent_fraction",
    "spatial_local_correlation_median",
    "spatial_local_residual_median",
    "evidence_axis_pattern",
    "evidence_axis_pattern_label",
    "interpretation",
    "caution",
];

const SENSITIVITY_FIELDS: &[&str] = &[
    "configuration",
    "aligned_candidate_persistence_min",
    "aligned_quality_response_min",
    "aligned_spatial_correlation_min",
    "persistent_candidate_persistence_min",
    "persistent_quality_response_max",
    "feature_class",
    "evidence_axis_pattern",
    "evidence_axis_pattern_label",
];

const CAUTION: &str = "Q|P=质量邻域持久数/候选位置持久数，仅为当前注册与规则口径下的描述性比值；质量邻域响应可能属于其它类别，不能解释为逐源质量通过率或真星概率。";
const CONCLUSION: &str = "类别证据轴关系已汇总：紧凑质量类是候选位置、质量邻域和局部 PSF 同向的对照；拥挤、尖峰、弱背景和形状类可出现位置持久但质量不同步；边缘/掩膜类处于部分同步；线状与范围异常类更适合走专项机制审计。Q|P 只描述 detector-level 证据同步，不提供逐源 precision、FDR、恒星概率或物理类别真值。";
const INTERPRETATION_BOUNDARY: &str = "all persistence and quality-response values are detector-level; quality response may come from another feature class in the same registered neighborhood; no star probability, precision, FDR, or truth";

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EvidenceAxisThresholds {
    pub aligned_candidate_persistence_min: f64,
    pub aligned_quality_response_min: f64,
    pub aligned_spatial_correlation_min: f64,
    pub persistent_candidate_persistence_min: f64,
    pub persistent_quality_response_max: f64,
}

const CONFIGS: [(&str, EvidenceAxisThresholds); 5] = [
    (
        "baseline",
        EvidenceAxisThresholds {
            aligned_candidate_persistence_min: 0.80,
            aligned_quality_response_min: 0.80,
            aligned_spatial_correlation_min: 0.80,
            persistent_candidate_persistence_min: 0.50,
            persistent_quality_response_max: 0.05,
        },
    ),
    (
        "strict_psf",
        EvidenceAxisThresholds {
            aligned_candidate_persistence_min: 0.80,
            aligned_quality_response_min: 0.80,
            aligned_spatial_correlation_min: 0.85,
            persistent_candidate_persistence_min: 0.50,
            persistent_quality_response_max: 0.05,
        },
    ),
    (
        "loose_persistence",
        EvidenceAxisThresholds {
            aligned_candidate_persistence_min: 0.80,
            aligned_quality_response_min: 0.80,
            aligned_spatial_correlation_min: 0.80,
            persistent_candidate_persistence_min: 0.40,
            persistent_quality_response_max: 0.03,
        },
    ),
    (
        "strict_quality_gap",
        EvidenceAxisThresholds {
            aligned_candidate_persistence_min: 0.80,
            aligned_quality_response_min: 0.80,
            aligned_spatial_correlation_min: 0.80,
            persistent_candidate_persistence_min: 0.50,
            persistent_quality_response_max: 0.03,
        },
    ),
    (
        "loose_quality_gap",
        EvidenceAxisThresholds {
            aligned_candidate_persistence_min: 0.80,
            aligned_quality_response_min: 0.80,
            aligned_spatial_correlation_min: 0.80,
            persistent_candidate_persistence_min: 0.50,
            persistent_quality_response_max: 0.08,
        },
    ),
];

#[derive(Debug, Clone, Serialize)]
pub struct FeatureEvidenceAxesRow {
    pub feature_class: String,
    pub feature_class_label: String,
    pub candidate_count: i64,
    pub candidate_persistent_count: i64,
    pub quality_response_count: i64,
    pub candidate_persistence_fraction: f64,
    pub quality_response_fraction: f64,
    pub quality_response_given_candidate_persistent_fraction: Option<f64>,
    pub spatial_local_correlation_median: Option<f64>,
    pub spatial_local_residual_median: Option<f64>,
    pub evidence_axis_pattern: String,
    pub evidence_axis_pattern_label: String,
    pub interpretation: String,
    pub caution: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureEvidenceAxesSensitivityRow {
    pub configuration: String,
    pub aligned_candidate_persistence_min: f64,
    pub aligned_quality_response_min: f64,
    pub aligned_spatial_correlation_min: f64,
    pub persistent_candidate_persistence_min: f64,
    pub persistent_quality_response_max: f64,
    pub feature_class: String,
    pub evidence_axis_pattern: String,
    pub evidence_axis_pattern_label: String,
}

#[derive(Debug, Clone)]
pub struct FeatureEvidenceAxesResult {
    pub matrix_path: String,
    pub thresholds: EvidenceAxisThresholds,
    pub rows: Vec<FeatureEvidenceAxesRow>,
    pub sensitivity_rows: Vec<FeatureEvidenceAxesSensitivityRow>,
    pub conclusion: String,
}

impl FeatureEvidenceAxesResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "matrix_path": self.matrix_path,
            "thresholds": self.thresholds,
            "rows": self.rows,
            "sensitivity_rows": self.sensitivity_rows,
            "conclusion": self.conclusion,
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
        })
    }
}

fn pattern_text(pattern: &str) -> (&'static str, &'static str) {
    match pattern {
        "aligned_quality_psf" => (
            "候选、质量邻域与局部 PSF 同向",
            "位置持久、质量邻域响应充分，且局部 PSF 相关度达到当前审计线；可进入身份核验队列。",
        ),
        "persistent_without_quality" => (
            "位置持久但质量不同步",
            "位置响应达到持久线，但质量邻域响应很少；应优先检查值域、支持、混叠或局部噪声。",
        ),
        "location_persistent_quality_sparse" => (
            "位置持久、质量部分同步",
            "位置响应较稳定，但质量邻域只部分同步；应结合有效孔径、边缘/掩膜和结构证据分流。",
        ),
        "sparse_or_structure_sensitive" => (
            "位置稀疏或结构敏感",
            "位置持久线本身较少，或响应主要受线几何/值域等专项机制控制；不宜用通用持久率解释。",
        ),
        "no_sample" => (
            "无样本",
            "当前矩阵没有该类别候选；缺失样本不支持任何物理结论。",
        ),
        _ => ("", ""),
    }
}

fn classify(
    candidate_count: i64,
    candidate_persistence: f64,
    quality_response: f64,
    spatial_correlation: Option<f64>,
    thresholds: EvidenceAxisThresholds,
) -> &'static str {
    if candidate_count == 0 {
        return "no_sample";
    }
    if candidate_persistence >= thresholds.aligned_candidate_persistence_min
        && quality_response >= thresholds.aligned_quality_response_min
        && spatial_correlation
            .is_some_and(|value| value >= thresholds.aligned_spatial_correlation_min)
    {
        return "aligned_quality_psf";
    }
    if candidate_persistence >= thresholds.persistent_candidate_persistence_min
        && quality_response <= thresholds.persistent_quality_response_max
    {
        return "persistent_without_quality";
    }
    if candidate_persistence >= 0.50 {
        return "location_persistent_quality_sparse";
    }
    "sparse_or_structure_sensitive"
}

fn required_int(row: &csv::StringRecord, index: usize, key: &str) -> Result<i64> {
    let value = row.get(index).unwrap_or("").trim();
    if value.is_empty() {
        return Err(format!("feature evidence matrix has empty {key}"));
    }
    let parsed = value
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
        .map(|number| number.trunc())
        .filter(|number| *number >= i64::MIN as f64 && *number <= i64::MAX as f64)
        .map(|number| number as i64)
        .ok_or_else(|| format!("feature evidence matrix has invalid {key}='{value}'"))?;
    if parsed < 0 {
        return Err(format!(
            "feature evidence matrix has negative {key}={parsed}"
        ));
    }
    Ok(parsed)
}

fn optional_float(row: &csv::StringRecord, index: usize, key: &str) -> Result<Option<f64>> {
    let value = row.get(index).unwrap_or("").trim();
    if value.is_empty()
        || ["none", "nan", "null"]
            .iter()
            .any(|missing| value.eq_ignore_ascii_case(missing))
    {
        return Ok(None);
    }
    value
        .parse::<f64>()
        .map(Some)
        .map_err(|_| format!("feature evidence matrix has invalid {key}='{value}'"))
}

fn column_index(headers: &csv::StringRecord, name: &str) -> Result<usize> {
    headers
        .iter()
        .enumerate()
        .filter_map(|(index, header)| (header == name).then_some(index))
        .last()
        .ok_or_else(|| format!("feature evidence matrix is missing required columns: {name}"))
}

/// Summarize candidate, quality-response, and local-PSF evidence axes by class.
pub fn run(matrix_path: &Path) -> Result<FeatureEvidenceAxesResult> {
    let bytes = fs::read(matrix_path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let present: HashSet<&str> = headers.iter().collect();
    let mut missing: Vec<&str> = REQUIRED_COLUMNS
        .iter()
        .copied()
        .filter(|column| !present.contains(column))
        .collect();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(format!(
            "feature evidence matrix is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let indexes = REQUIRED_COLUMNS
        .iter()
        .map(|column| column_index(&headers, column))
        .collect::<Result<Vec<_>>>()?;
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for record in reader.records() {
        let record = record.map_err(|error| error.to_string())?;
        let feature_class = record.get(indexes[0]).unwrap_or("").trim().to_owned();
        if feature_class.is_empty() {
            return Err("feature evidence matrix has an empty feature_class".into());
        }
        if !seen.insert(feature_class.clone()) {
            return Err(format!(
                "feature evidence matrix contains duplicate feature_class={feature_class}"
            ));
        }
        let candidate_count = required_int(&record, indexes[2], "candidate_count")?;
        let candidate_persistent_count =
            required_int(&record, indexes[3], "candidate_presence_ge_required_count")?;
        let quality_response_count =
            required_int(&record, indexes[4], "quality_response_ge_required_count")?;
        if candidate_persistent_count > candidate_count {
            return Err(format!(
                "{feature_class}: candidate persistent count exceeds candidate count ({candidate_persistent_count}>{candidate_count})"
            ));
        }
        if quality_response_count > candidate_count {
            return Err(format!(
                "{feature_class}: quality response count exceeds candidate count ({quality_response_count}>{candidate_count})"
            ));
        }
        let candidate_persistence = if candidate_count > 0 {
            candidate_persistent_count as f64 / candidate_count as f64
        } else {
            0.0
        };
        let quality_response = if candidate_count > 0 {
            quality_response_count as f64 / candidate_count as f64
        } else {
            0.0
        };
        let quality_given_candidate = (candidate_persistent_count > 0)
            .then(|| quality_response_count as f64 / candidate_persistent_count as f64);
        let spatial_correlation =
            optional_float(&record, indexes[5], "spatial_local_correlation_median")?;
        let spatial_residual =
            optional_float(&record, indexes[6], "spatial_local_residual_median")?;
        let pattern = classify(
            candidate_count,
            candidate_persistence,
            quality_response,
            spatial_correlation,
            CONFIGS[0].1,
        );
        let (label, interpretation) = pattern_text(pattern);
        rows.push(FeatureEvidenceAxesRow {
            feature_class,
            feature_class_label: record.get(indexes[1]).unwrap_or("").trim().to_owned(),
            candidate_count,
            candidate_persistent_count,
            quality_response_count,
            candidate_persistence_fraction: candidate_persistence,
            quality_response_fraction: quality_response,
            quality_response_given_candidate_persistent_fraction: quality_given_candidate,
            spatial_local_correlation_median: spatial_correlation,
            spatial_local_residual_median: spatial_residual,
            evidence_axis_pattern: pattern.to_owned(),
            evidence_axis_pattern_label: label.to_owned(),
            interpretation: interpretation.to_owned(),
            caution: CAUTION.to_owned(),
        });
    }
    if rows.is_empty() {
        return Err("feature evidence matrix is empty".into());
    }
    let mut sensitivity_rows = Vec::with_capacity(CONFIGS.len() * rows.len());
    for (configuration, thresholds) in CONFIGS {
        for row in &rows {
            let pattern = classify(
                row.candidate_count,
                row.candidate_persistence_fraction,
                row.quality_response_fraction,
                row.spatial_local_correlation_median,
                thresholds,
            );
            sensitivity_rows.push(FeatureEvidenceAxesSensitivityRow {
                configuration: configuration.to_owned(),
                aligned_candidate_persistence_min: thresholds.aligned_candidate_persistence_min,
                aligned_quality_response_min: thresholds.aligned_quality_response_min,
                aligned_spatial_correlation_min: thresholds.aligned_spatial_correlation_min,
                persistent_candidate_persistence_min: thresholds
                    .persistent_candidate_persistence_min,
                persistent_quality_response_max: thresholds.persistent_quality_response_max,
                feature_class: row.feature_class.clone(),
                evidence_axis_pattern: pattern.to_owned(),
                evidence_axis_pattern_label: pattern_text(pattern).0.to_owned(),
            });
        }
    }
    Ok(FeatureEvidenceAxesResult {
        matrix_path: matrix_path.to_string_lossy().into_owned(),
        thresholds: CONFIGS[0].1,
        rows,
        sensitivity_rows,
        conclusion: CONCLUSION.to_owned(),
    })
}

fn csv_float(value: f64) -> String {
    let text = if value.is_nan() {
        "nan".into()
    } else if value == f64::INFINITY {
        "inf".into()
    } else if value == f64::NEG_INFINITY {
        "-inf".into()
    } else {
        value.to_string()
    };
    if !text.contains(['.', 'e', 'E']) && !text.contains("inf") && text != "nan" {
        format!("{text}.0")
    } else {
        text
    }
}

fn summary_record(row: &FeatureEvidenceAxesRow) -> Vec<String> {
    vec![
        row.feature_class.clone(),
        row.feature_class_label.clone(),
        row.candidate_count.to_string(),
        row.candidate_persistent_count.to_string(),
        row.quality_response_count.to_string(),
        csv_float(row.candidate_persistence_fraction),
        csv_float(row.quality_response_fraction),
        row.quality_response_given_candidate_persistent_fraction
            .map(csv_float)
            .unwrap_or_default(),
        row.spatial_local_correlation_median
            .map(csv_float)
            .unwrap_or_default(),
        row.spatial_local_residual_median
            .map(csv_float)
            .unwrap_or_default(),
        row.evidence_axis_pattern.clone(),
        row.evidence_axis_pattern_label.clone(),
        row.interpretation.clone(),
        row.caution.clone(),
    ]
}

fn sensitivity_record(row: &FeatureEvidenceAxesSensitivityRow) -> Vec<String> {
    vec![
        row.configuration.clone(),
        csv_float(row.aligned_candidate_persistence_min),
        csv_float(row.aligned_quality_response_min),
        csv_float(row.aligned_spatial_correlation_min),
        csv_float(row.persistent_candidate_persistence_min),
        csv_float(row.persistent_quality_response_max),
        row.feature_class.clone(),
        row.evidence_axis_pattern.clone(),
        row.evidence_axis_pattern_label.clone(),
    ]
}

fn write_csv(path: &Path, headers: &[&str], records: Vec<Vec<String>>) -> Result<()> {
    let mut file = fs::File::create(path).map_err(|error| error.to_string())?;
    use std::io::Write as _;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for record in records {
        writer
            .write_record(record)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &FeatureEvidenceAxesResult, output_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    write_csv(
        &output_dir.join("feature_evidence_axes.csv"),
        SUMMARY_FIELDS,
        result.rows.iter().map(summary_record).collect(),
    )?;
    write_csv(
        &output_dir.join("feature_evidence_axes_sensitivity.csv"),
        SENSITIVITY_FIELDS,
        result
            .sensitivity_rows
            .iter()
            .map(sensitivity_record)
            .collect(),
    )?;
    fs::write(
        output_dir.join("feature_evidence_axes.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

pub fn run_desktop(args: &[String], job_dir: &Path) -> Result<Value> {
    let mut matrix = None;
    let mut output_dir = PathBuf::from("tmp/feature-evidence-axes");
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--out-dir" {
            index += 1;
            output_dir = PathBuf::from(args.get(index).ok_or("missing value for --out-dir")?);
        } else if let Some(value) = argument.strip_prefix("--out-dir=") {
            output_dir = PathBuf::from(value);
        } else if argument.starts_with('-') {
            return Err(format!("unrecognized arguments: {argument}"));
        } else if matrix.replace(PathBuf::from(argument)).is_some() {
            return Err("unrecognized arguments: too many input paths".into());
        }
        index += 1;
    }
    let matrix = matrix.ok_or("the following arguments are required: matrix")?;
    let result = run(&matrix)?;
    write_artifacts(&result, &output_dir)?;
    fs::create_dir_all(job_dir).map_err(|error| error.to_string())?;
    let result_path = job_dir.join("feature_evidence_axes_result.json");
    fs::write(
        &result_path,
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(json!({
        "tool": "rst19-feature-evidence-axes",
        "out_dir": output_dir.to_string_lossy(),
        "result_path": result_path.to_string_lossy(),
        "result": result.as_dict(),
    }))
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args
        .iter()
        .any(|argument| argument == "-h" || argument == "--help")
    {
        println!("从 feature_evidence_matrix.csv 汇总候选位置与质量响应的证据轴关系\n\nUsage: rst19-feature-evidence-axes MATRIX [--out-dir DIR]");
        return Ok(());
    }

    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let job_dir = std::env::temp_dir().join(format!(
        "rst19-feature-evidence-axes-cli-{}-{nonce}",
        std::process::id()
    ));
    let outcome = run_desktop(args, &job_dir);
    let _ = fs::remove_dir_all(&job_dir);
    let response = outcome?;
    let output_dir = response
        .get("out_dir")
        .and_then(Value::as_str)
        .ok_or("native CLI response is missing out_dir")?;
    let result = response
        .get("result")
        .and_then(Value::as_object)
        .ok_or("native CLI response is missing result")?;
    let rows = result
        .get("rows")
        .and_then(Value::as_array)
        .ok_or("native CLI response is missing rows")?;
    let sensitivity_rows = result
        .get("sensitivity_rows")
        .and_then(Value::as_array)
        .ok_or("native CLI response is missing sensitivity_rows")?;
    let configurations = sensitivity_rows
        .iter()
        .filter_map(|row| row.get("configuration").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    println!("rst19-feature-evidence-axes: 结果已写入 {output_dir}");
    println!(
        "rst19-feature-evidence-axes: 敏感性配置 {} 组",
        configurations.len()
    );
    println!(
        "{}",
        result
            .get("conclusion")
            .and_then(Value::as_str)
            .ok_or("native CLI response is missing conclusion")?
    );
    for row in rows {
        let feature_class = row
            .get("feature_class")
            .and_then(Value::as_str)
            .ok_or("native CLI row is missing feature_class")?;
        let pattern = row
            .get("evidence_axis_pattern")
            .and_then(Value::as_str)
            .ok_or("native CLI row is missing evidence_axis_pattern")?;
        let candidate_count = row
            .get("candidate_persistent_count")
            .and_then(Value::as_i64)
            .ok_or("native CLI row is missing candidate_persistent_count")?;
        let quality_count = row
            .get("quality_response_count")
            .and_then(Value::as_i64)
            .ok_or("native CLI row is missing quality_response_count")?;
        let conditional = row
            .get("quality_response_given_candidate_persistent_fraction")
            .and_then(Value::as_f64)
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| "NA".to_owned());
        println!(
            "{feature_class}: pattern={pattern} candidate_persistent={candidate_count} quality_response={quality_count} Q|P={conditional}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn classifies_default_and_sensitive_patterns_in_input_order() {
        let root = std::env::temp_dir().join(format!(
            "rst19-evidence-axes-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let matrix = root.join("matrix.csv");
        fs::write(
            &matrix,
            "feature_class,feature_class_label,candidate_count,candidate_presence_ge_required_count,quality_response_ge_required_count,spatial_local_correlation_median,spatial_local_residual_median\ncompact_quality,compact,100,95,85,0.84,0.54\ncrowded_blend,blend,100,60,1,0.17,0.98\nmasked_or_edge,masked,100,80,20,0.67,0.74\nlinear_artifact,line,10,2,1,0.4,0.91\nother_rejected,other,0,0,0,,\n",
        )
        .unwrap();
        let result = run(&matrix).unwrap();
        assert_eq!(
            result
                .rows
                .iter()
                .map(|row| row.evidence_axis_pattern.as_str())
                .collect::<Vec<_>>(),
            [
                "aligned_quality_psf",
                "persistent_without_quality",
                "location_persistent_quality_sparse",
                "sparse_or_structure_sensitive",
                "no_sample",
            ]
        );
        assert_eq!(
            result.sensitivity_rows[5].evidence_axis_pattern,
            "location_persistent_quality_sparse"
        );
        assert_eq!(result.rows.len(), 5);
        assert_eq!(result.sensitivity_rows.len(), 25);

        let cli_output = root.join("cli-output");
        run_cli(&[
            matrix.to_string_lossy().into_owned(),
            "--out-dir".into(),
            cli_output.to_string_lossy().into_owned(),
        ])
        .unwrap();
        assert!(cli_output.join("feature_evidence_axes.csv").is_file());
        assert!(cli_output
            .join("feature_evidence_axes_sensitivity.csv")
            .is_file());
        assert!(cli_output.join("feature_evidence_axes.json").is_file());
        let _ = fs::remove_dir_all(root);
    }
}
