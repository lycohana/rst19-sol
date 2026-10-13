//! Joins existing feature-class, temporal-response, and leave-one-out PSF CSVs.
//!
//! This module organizes detector-level evidence. Its patterns and audit routes
//! are descriptive; they never establish physical source identity.

use csv::{Terminator, WriterBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

type DominantSecondary = (Option<String>, Option<String>, Option<f64>);

const FEATURE_ORDER: [(&str, &str); 9] = [
    ("range_anomaly", "数据有效性/范围异常邻域"),
    ("linear_artifact", "线状/拖影候选"),
    ("masked_or_edge", "边缘/掩膜候选"),
    ("crowded_blend", "拥挤/未分辨近邻"),
    ("spike_or_support", "尖峰/PSF 支持不足"),
    ("weak_or_background", "弱通量/背景不确定"),
    ("shape_outlier", "形状异常"),
    ("compact_quality", "质量通过的紧凑候选"),
    ("other_rejected", "其他拒绝候选"),
];
const SECONDARY_FEATURES: [(&str, &str); 7] = [
    ("range_anomaly", "数据有效性/范围异常邻域"),
    ("linear_artifact", "线状/拖影候选"),
    ("masked_or_edge", "边缘/掩膜候选"),
    ("crowded_blend", "拥挤/未分辨近邻"),
    ("spike_or_support", "尖峰/PSF 支持不足"),
    ("weak_or_background", "弱通量/背景不确定"),
    ("shape_outlier", "形状异常"),
];

#[derive(Debug, Clone, Serialize)]
pub struct FeatureEvidenceRow {
    pub feature_class: String,
    pub feature_class_label: String,
    pub candidate_count: Option<i64>,
    pub quality_count: Option<i64>,
    pub quality_fraction: Option<f64>,
    pub high_flux_snr_rejection_fraction: Option<f64>,
    pub dominant_secondary_feature: Option<String>,
    pub dominant_secondary_feature_label: Option<String>,
    pub dominant_secondary_fraction: Option<f64>,
    pub candidate_presence_ge_required_count: Option<i64>,
    pub candidate_persistence_fraction: Option<f64>,
    pub quality_response_ge_required_count: Option<i64>,
    pub quality_response_fraction: Option<f64>,
    pub candidate_any_response_fraction: Option<f64>,
    pub candidate_same_class_response_fraction: Option<f64>,
    pub quality_any_response_fraction: Option<f64>,
    pub quality_same_class_response_fraction: Option<f64>,
    pub leaveout_sample_count: Option<i64>,
    pub leaveout_valid_count: Option<i64>,
    pub leaveout_correlation_median: Option<f64>,
    pub leaveout_correlation_ge_threshold_fraction: Option<f64>,
    pub leaveout_residual_median: Option<f64>,
    pub spatial_local_template_available_cell_count: Option<i64>,
    pub spatial_local_template_source_count_median: Option<f64>,
    pub spatial_local_valid_count: Option<i64>,
    pub spatial_local_fallback_count: Option<i64>,
    pub spatial_paired_count: Option<i64>,
    pub spatial_local_correlation_median: Option<f64>,
    pub spatial_local_residual_median: Option<f64>,
    pub spatial_local_correlation_improved_fraction: Option<f64>,
    pub spatial_local_residual_reduced_fraction: Option<f64>,
    pub spatial_median_correlation_delta: Option<f64>,
    pub spatial_median_residual_delta: Option<f64>,
    pub candidate_persistence_wilson95_low: Option<f64>,
    pub candidate_persistence_wilson95_high: Option<f64>,
    pub quality_fraction_wilson95_low: Option<f64>,
    pub quality_fraction_wilson95_high: Option<f64>,
    pub quality_response_wilson95_low: Option<f64>,
    pub quality_response_wilson95_high: Option<f64>,
    pub evidence_pattern: String,
    pub note: String,
    pub recommended_audit_stage: String,
    pub recommended_audit_stage_label: String,
    pub counting_policy: String,
    pub counting_policy_label: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureEvidenceMatrixResult {
    pub source_morphology_path: String,
    pub feature_cross_path: String,
    pub psf_leaveout_path: String,
    pub psf_spatial_path: Option<String>,
    pub correlation_threshold: f64,
    pub rows: Vec<FeatureEvidenceRow>,
    pub parameters: Value,
    pub conclusion: String,
}

impl FeatureEvidenceMatrixResult {
    pub fn as_dict(&self) -> Value {
        serde_json::to_value(self).expect("finite feature evidence matrix")
    }
}

type CsvRow = HashMap<String, String>;
type IndexedRows = BTreeMap<String, CsvRow>;

fn read_csv(path: &Path) -> Result<Vec<CsvRow>, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let mut reader = csv::ReaderBuilder::new().from_reader(std::io::Cursor::new(bytes.to_vec()));
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    if headers.is_empty() || !headers.iter().any(|name| name == "feature_class") {
        return Err(format!("{} must contain feature_class", path.display()));
    }
    reader
        .records()
        .map(|record| {
            let record = record.map_err(|error| error.to_string())?;
            Ok(headers
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    (
                        name.to_owned(),
                        record.get(index).unwrap_or_default().to_owned(),
                    )
                })
                .collect())
        })
        .collect()
}

fn index_rows(rows: Vec<CsvRow>, path: &Path) -> Result<IndexedRows, String> {
    let mut indexed = BTreeMap::new();
    for row in rows {
        let feature_class = row
            .get("feature_class")
            .map(|value| value.trim().to_owned())
            .unwrap_or_default();
        if feature_class.is_empty() {
            continue;
        }
        if indexed.insert(feature_class.clone(), row).is_some() {
            return Err(format!(
                "{} contains duplicate feature_class={feature_class}",
                path.display()
            ));
        }
    }
    Ok(indexed)
}

fn optional_float(row: Option<&CsvRow>, key: &str) -> Result<Option<f64>, String> {
    let Some(row) = row else { return Ok(None) };
    let value = row.get(key).map(|value| value.trim()).unwrap_or_default();
    if value.is_empty() {
        return Ok(None);
    }
    let converted = value
        .parse::<f64>()
        .map_err(|_| format!("invalid number {key}={value:?}"))?;
    Ok(converted.is_finite().then_some(converted))
}

fn optional_int(row: Option<&CsvRow>, key: &str) -> Result<Option<i64>, String> {
    let Some(row) = row else { return Ok(None) };
    let value = row.get(key).map(|value| value.trim()).unwrap_or_default();
    if value.is_empty() {
        return Ok(None);
    }
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("invalid integer {key}={value:?}"))?;
    if !parsed.is_finite() || parsed < i64::MIN as f64 || parsed >= i64::MAX as f64 {
        return Err(format!("invalid integer {key}={value:?}"));
    }
    Ok(Some(parsed.trunc() as i64))
}

fn wilson95_interval(
    successes: Option<i64>,
    trials: Option<i64>,
) -> Result<(Option<f64>, Option<f64>), String> {
    let (Some(successes), Some(trials)) = (successes, trials) else {
        return Ok((None, None));
    };
    if trials < 0 || successes < 0 || successes > trials {
        return Err(format!(
            "invalid proportion counts: successes={successes}, trials={trials}"
        ));
    }
    if trials == 0 {
        return Ok((None, None));
    }
    // Matches statistics.NormalDist().inv_cdf(0.975) used by the Python reference.
    let z = 1.959_963_984_540_053_6_f64;
    let proportion = successes as f64 / trials as f64;
    let denominator = 1.0 + z * z / trials as f64;
    let center = (proportion + z * z / (2.0 * trials as f64)) / denominator;
    let half_width = z
        * (proportion * (1.0 - proportion) / trials as f64
            + z * z / (4.0 * (trials as f64).powi(2)))
        .sqrt()
        / denominator;
    let lower = if successes == 0 {
        0.0
    } else {
        (center - half_width).max(0.0)
    };
    let upper = if successes == trials {
        1.0
    } else {
        (center + half_width).min(1.0)
    };
    Ok((Some(lower), Some(upper)))
}

fn label_for(
    default_label: &str,
    morphology: Option<&CsvRow>,
    cross: Option<&CsvRow>,
    psf: Option<&CsvRow>,
) -> String {
    for row in [morphology, cross, psf].into_iter().flatten() {
        if let Some(label) = row.get("feature_class_label") {
            let label = label.trim();
            if !label.is_empty() {
                return label.to_owned();
            }
        }
    }
    default_label.to_owned()
}

fn dominant_secondary(
    feature_class: &str,
    morphology: Option<&CsvRow>,
) -> Result<DominantSecondary, String> {
    let Some(row) = morphology else {
        return Ok((None, None, None));
    };
    let mut best: Option<(f64, &str, &str)> = None;
    for (secondary, label) in SECONDARY_FEATURES {
        if secondary == feature_class {
            continue;
        }
        if let Some(value) = optional_float(Some(row), &format!("{secondary}_fraction"))? {
            if value > 0.0 && best.is_none_or(|current| value > current.0) {
                best = Some((value, secondary, label));
            }
        }
    }
    Ok(
        best.map_or((None, None, None), |(value, secondary, label)| {
            (Some(secondary.into()), Some(label.into()), Some(value))
        }),
    )
}

fn pattern(
    feature_class: &str,
    quality_fraction: Option<f64>,
    candidate_persistence: Option<f64>,
    quality_response: Option<f64>,
    correlation: Option<f64>,
    correlation_threshold: f64,
) -> (&'static str, &'static str) {
    if feature_class == "other_rejected" && quality_fraction.is_none() {
        return (
            "无非空样本",
            "当前表没有该类别的候选，不能从缺失数据推出物理结论。",
        );
    }
    if feature_class == "range_anomaly" {
        return ("值域优先审计", "范围/编码风险必须先于时序和 PSF 指标解释。");
    }
    if quality_fraction.is_some_and(|value| value >= 0.8)
        && quality_response.is_some_and(|value| value >= 0.5)
        && correlation.is_some_and(|value| value >= correlation_threshold)
    {
        return (
            "质量/时序/PSF 三层一致",
            "三层证据在当前抽样中相互支持，但仍需星表或注入真值。",
        );
    }
    if candidate_persistence.is_some_and(|value| value >= 0.5)
        && (quality_fraction.is_none_or(|value| value < 0.1)
            || quality_response.is_some_and(|value| value < 0.1))
        && correlation.is_none_or(|value| value < 0.7)
    {
        return (
            "候选持续但质量/PSF 不支持",
            "跨帧重复不能单独升级为恒星，应优先复核噪声、结构、掩膜或混合。",
        );
    }
    if correlation.is_some_and(|value| value < 0.7) {
        return (
            "形状/结构优先复核",
            "局部裁剪不像标准孤立 PSF，需结合几何、值域和邻峰证据。",
        );
    }
    (
        "证据未闭环",
        "当前三层指标不足以完成身份判断，应做留出注入或星表/WCS 核验。",
    )
}

fn audit_route(feature_class: &str) -> (&'static str, &'static str, &'static str, &'static str) {
    match feature_class {
        "range_anomaly" => (
            "value_domain",
            "原始值域/编码核验",
            "candidate_only",
            "仅保留候选，不直接计星",
        ),
        "linear_artifact" => (
            "line_geometry",
            "线几何与跨帧运动核验",
            "candidate_only",
            "仅保留候选，不直接计星",
        ),
        "masked_or_edge" => (
            "aperture_validity",
            "有效孔径、边缘与掩膜核验",
            "candidate_only",
            "仅保留候选，不直接计星",
        ),
        "crowded_blend" => (
            "joint_psf_deblend",
            "联合单/双 PSF 去混叠",
            "candidate_only",
            "仅保留候选，不直接计星",
        ),
        "spike_or_support" => (
            "psf_support",
            "核心支持与高频结构核验",
            "candidate_only",
            "仅保留候选，不直接计星",
        ),
        "weak_or_background" => (
            "local_noise_injection",
            "分区噪声与注入回收核验",
            "candidate_only",
            "仅保留候选，不直接计星",
        ),
        "shape_outlier" => (
            "shape_psf",
            "FWHM、椭圆率与径向 PSF 核验",
            "candidate_only",
            "仅保留候选，不直接计星",
        ),
        "compact_quality" => (
            "catalog_wcs",
            "星表/WCS 与留出注入核验",
            "identity_review_queue",
            "仅可进入身份核验队列",
        ),
        _ => (
            "manual_review",
            "规则缺口与人工复核",
            "candidate_only",
            "仅保留候选，不直接计星",
        ),
    }
}

fn opt_i(row: Option<&CsvRow>, key: &str) -> Result<Option<i64>, String> {
    optional_int(row, key)
}

fn opt_f(row: Option<&CsvRow>, key: &str) -> Result<Option<f64>, String> {
    optional_float(row, key)
}

pub fn run(
    source_morphology_path: &Path,
    feature_cross_path: &Path,
    psf_leaveout_path: &Path,
    correlation_threshold: f64,
    psf_spatial_path: Option<&Path>,
) -> Result<FeatureEvidenceMatrixResult, String> {
    if !(0.0 < correlation_threshold && correlation_threshold <= 1.0) {
        return Err("correlation_threshold must be between 0 and 1".into());
    }
    let morphology = index_rows(read_csv(source_morphology_path)?, source_morphology_path)?;
    let cross = index_rows(read_csv(feature_cross_path)?, feature_cross_path)?;
    let psf = index_rows(read_csv(psf_leaveout_path)?, psf_leaveout_path)?;
    let spatial = if let Some(path) = psf_spatial_path {
        index_rows(read_csv(path)?, path)?
    } else {
        BTreeMap::new()
    };
    let mut rows = Vec::with_capacity(FEATURE_ORDER.len());
    for (feature_class, default_label) in FEATURE_ORDER {
        let morphology_row = morphology.get(feature_class);
        let cross_row = cross.get(feature_class);
        let psf_row = psf.get(feature_class);
        let spatial_row = spatial.get(feature_class);
        let quality_fraction = opt_f(morphology_row, "quality_fraction")?;
        let (
            dominant_secondary_feature,
            dominant_secondary_feature_label,
            dominant_secondary_fraction,
        ) = dominant_secondary(feature_class, morphology_row)?;
        let candidate_persistence = opt_f(cross_row, "candidate_presence_fraction")?;
        let quality_response = opt_f(cross_row, "quality_presence_fraction")?;
        let correlation = opt_f(psf_row, "leaveout_correlation_median")?;
        let candidate_count = opt_i(morphology_row, "candidate_count")?;
        let quality_count = opt_i(morphology_row, "quality_count")?;
        let candidate_presence_count = opt_i(cross_row, "candidate_presence_ge_required_count")?;
        let anchor_candidate_count = opt_i(cross_row, "anchor_candidate_count")?;
        let quality_response_count = opt_i(cross_row, "quality_presence_ge_required_count")?;
        let (candidate_persistence_wilson95_low, candidate_persistence_wilson95_high) =
            wilson95_interval(candidate_presence_count, anchor_candidate_count)?;
        let (quality_fraction_wilson95_low, quality_fraction_wilson95_high) =
            wilson95_interval(quality_count, candidate_count)?;
        let (quality_response_wilson95_low, quality_response_wilson95_high) =
            wilson95_interval(quality_response_count, anchor_candidate_count)?;
        let (evidence_pattern, note) = pattern(
            feature_class,
            quality_fraction,
            candidate_persistence,
            quality_response,
            correlation,
            correlation_threshold,
        );
        let (
            recommended_audit_stage,
            recommended_audit_stage_label,
            counting_policy,
            counting_policy_label,
        ) = audit_route(feature_class);
        rows.push(FeatureEvidenceRow {
            feature_class: feature_class.into(),
            feature_class_label: label_for(default_label, morphology_row, cross_row, psf_row),
            candidate_count,
            quality_count,
            quality_fraction,
            high_flux_snr_rejection_fraction: opt_f(
                morphology_row,
                "high_flux_snr_rejection_fraction",
            )?,
            dominant_secondary_feature,
            dominant_secondary_feature_label,
            dominant_secondary_fraction,
            candidate_presence_ge_required_count: candidate_presence_count,
            candidate_persistence_fraction: candidate_persistence,
            quality_response_ge_required_count: quality_response_count,
            quality_response_fraction: quality_response,
            candidate_any_response_fraction: opt_f(cross_row, "candidate_any_response_fraction")?,
            candidate_same_class_response_fraction: opt_f(
                cross_row,
                "candidate_same_class_response_fraction",
            )?,
            quality_any_response_fraction: opt_f(cross_row, "quality_any_response_fraction")?,
            quality_same_class_response_fraction: opt_f(
                cross_row,
                "quality_same_class_response_fraction",
            )?,
            leaveout_sample_count: opt_i(psf_row, "sample_count")?,
            leaveout_valid_count: opt_i(psf_row, "leaveout_valid_count")?,
            leaveout_correlation_median: correlation,
            leaveout_correlation_ge_threshold_fraction: opt_f(
                psf_row,
                "leaveout_correlation_ge_threshold_fraction",
            )?,
            leaveout_residual_median: opt_f(psf_row, "leaveout_residual_median")?,
            spatial_local_template_available_cell_count: opt_i(
                spatial_row,
                "local_template_available_cell_count",
            )?,
            spatial_local_template_source_count_median: opt_f(
                spatial_row,
                "local_template_source_count_median",
            )?,
            spatial_local_valid_count: opt_i(spatial_row, "local_valid_count")?,
            spatial_local_fallback_count: opt_i(spatial_row, "local_fallback_count")?,
            spatial_paired_count: opt_i(spatial_row, "paired_correlation_count")?,
            spatial_local_correlation_median: opt_f(spatial_row, "local_correlation_median")?,
            spatial_local_residual_median: opt_f(spatial_row, "local_residual_median")?,
            spatial_local_correlation_improved_fraction: opt_f(
                spatial_row,
                "local_correlation_improved_fraction",
            )?,
            spatial_local_residual_reduced_fraction: opt_f(
                spatial_row,
                "local_residual_reduced_fraction",
            )?,
            spatial_median_correlation_delta: opt_f(
                spatial_row,
                "median_correlation_delta_local_minus_global",
            )?,
            spatial_median_residual_delta: opt_f(
                spatial_row,
                "median_residual_delta_local_minus_global",
            )?,
            candidate_persistence_wilson95_low,
            candidate_persistence_wilson95_high,
            quality_fraction_wilson95_low,
            quality_fraction_wilson95_high,
            quality_response_wilson95_low,
            quality_response_wilson95_high,
            evidence_pattern: evidence_pattern.into(),
            note: note.into(),
            recommended_audit_stage: recommended_audit_stage.into(),
            recommended_audit_stage_label: recommended_audit_stage_label.into(),
            counting_policy: counting_policy.into(),
            counting_policy_label: counting_policy_label.into(),
        });
    }
    let conclusion = concat!(
        "该矩阵只把三类已有研究量按首要特征类别对齐：形态/质量层、15 帧候选与质量层邻域响应、",
        "以及逐源留一经验 PSF。当前数据中，候选层的跨帧重复在多个被拒类别中仍然较高，",
        "而质量层邻域响应和标准 PSF 形状并未同步出现；因此“可重复”不能单独等同于“真星”。",
        "若提供空间 PSF 表，矩阵还会保留局部模板可得率、局部相关度/残差及相对全局模板的变化；",
        "这些字段只检验空间校正是否改变解释，不把模板不足或局部改善直接转成真星结论。",
        "矩阵是证据组织工具，不是加权分类器、概率模型或物理真值表。",
        "recommended_audit_stage 只是确定性的下一步复核路由，counting_policy 只规定当前研究的计数边界，",
        "二者都不产生恒星概率。"
    )
    .to_owned();
    Ok(FeatureEvidenceMatrixResult {
        source_morphology_path: source_morphology_path.to_string_lossy().into_owned(),
        feature_cross_path: feature_cross_path.to_string_lossy().into_owned(),
        psf_leaveout_path: psf_leaveout_path.to_string_lossy().into_owned(),
        psf_spatial_path: psf_spatial_path.map(|path| path.to_string_lossy().into_owned()),
        correlation_threshold,
        rows,
        parameters: json!({
            "feature_order": FEATURE_ORDER.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
            "temporal_required_presence": "沿用 feature_cross_audit.csv 的 required_presence，当前为 12/15 帧",
            "quality_response_definition": "feature_cross_audit.csv 的 quality_presence_* 是质量层在锚点邻域内的响应，不等价于同类质量通过或同类跨帧持久性",
            "same_class_response_definition": "feature_cross_audit.csv 的 *_same_class_response_fraction 只表示响应类别仍与锚点首要类别相同，不是真阳性率",
            "psf_definition": "逐源排除自身、最多 20 个隔离质量模板、留一相关度与相对残差",
            "spatial_psf_definition": "可选 source_feature_psf_spatial.csv；局部模板按空间网格留出比较，local_valid/fallback 先于局部相关度解释",
            "spatial_psf_is_diagnostic_only": true,
            "proportion_interval_definition": "95% Wilson interval from observed anchor counts; descriptive only, not a confidence interval for physical truth because shared image/systematic effects remain",
            "pattern_is_not_truth": true,
            "routing_definition": "按 feature_class 确定下一步复核模块；不使用指标加权，不输出概率，不改写 detection 或 quality_count",
            "counting_policy_definition": "只有 compact_quality 可进入星表/WCS 或留出注入身份核验队列；所有其他类别保留候选但不直接计星"
        }),
        conclusion,
    })
}

fn python_float_string(value: f64) -> String {
    let mut raw = format!("{value:?}");
    let negative = raw.starts_with('-');
    if negative {
        raw.remove(0);
    }
    let (mantissa, exponent) = raw
        .split_once(['e', 'E'])
        .map(|(mantissa, exponent)| (mantissa, exponent.parse::<i32>().unwrap_or(0)))
        .unwrap_or((raw.as_str(), 0));
    let decimal_position = mantissa.find('.').unwrap_or(mantissa.len()) as i32 + exponent;
    let mut digits: String = mantissa
        .chars()
        .filter(|character| *character != '.')
        .collect();
    let leading_zeroes = digits.chars().take_while(|digit| *digit == '0').count();
    digits.drain(..leading_zeroes);
    let decimal_position = decimal_position - leading_zeroes as i32;
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    if digits.is_empty() {
        return if negative { "-0.0" } else { "0.0" }.into();
    }
    let sign = if negative { "-" } else { "" };
    if !(-3..=15).contains(&decimal_position) {
        let scientific_exponent = decimal_position - 1;
        let exponent_sign = if scientific_exponent >= 0 { "+" } else { "-" };
        let fraction = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        return format!(
            "{sign}{}{fraction}e{exponent_sign}{:02}",
            &digits[..1],
            scientific_exponent.abs()
        );
    }
    if decimal_position <= 0 {
        return format!(
            "{sign}0.{}{}",
            "0".repeat((-decimal_position) as usize),
            digits
        );
    }
    if decimal_position as usize >= digits.len() {
        return format!(
            "{sign}{}{}.0",
            digits,
            "0".repeat(decimal_position as usize - digits.len())
        );
    }
    format!(
        "{sign}{}.{}",
        &digits[..decimal_position as usize],
        &digits[decimal_position as usize..]
    )
}

fn csv_cell(value: &Value, float_field: bool) -> String {
    match value {
        Value::Null => String::new(),
        Value::Number(number) if float_field => {
            python_float_string(number.as_f64().unwrap_or_default())
        }
        Value::Number(number) => number.to_string(),
        Value::String(value) => value.clone(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        _ => value.to_string(),
    }
}

const CSV_COLUMNS: [&str; 45] = [
    "feature_class",
    "feature_class_label",
    "candidate_count",
    "quality_count",
    "quality_fraction",
    "high_flux_snr_rejection_fraction",
    "dominant_secondary_feature",
    "dominant_secondary_feature_label",
    "dominant_secondary_fraction",
    "candidate_presence_ge_required_count",
    "candidate_persistence_fraction",
    "quality_response_ge_required_count",
    "quality_response_fraction",
    "candidate_any_response_fraction",
    "candidate_same_class_response_fraction",
    "quality_any_response_fraction",
    "quality_same_class_response_fraction",
    "leaveout_sample_count",
    "leaveout_valid_count",
    "leaveout_correlation_median",
    "leaveout_correlation_ge_threshold_fraction",
    "leaveout_residual_median",
    "spatial_local_template_available_cell_count",
    "spatial_local_template_source_count_median",
    "spatial_local_valid_count",
    "spatial_local_fallback_count",
    "spatial_paired_count",
    "spatial_local_correlation_median",
    "spatial_local_residual_median",
    "spatial_local_correlation_improved_fraction",
    "spatial_local_residual_reduced_fraction",
    "spatial_median_correlation_delta",
    "spatial_median_residual_delta",
    "candidate_persistence_wilson95_low",
    "candidate_persistence_wilson95_high",
    "quality_fraction_wilson95_low",
    "quality_fraction_wilson95_high",
    "quality_response_wilson95_low",
    "quality_response_wilson95_high",
    "evidence_pattern",
    "note",
    "recommended_audit_stage",
    "recommended_audit_stage_label",
    "counting_policy",
    "counting_policy_label",
];
const FLOAT_COLUMNS: [&str; 25] = [
    "quality_fraction",
    "high_flux_snr_rejection_fraction",
    "dominant_secondary_fraction",
    "candidate_persistence_fraction",
    "quality_response_fraction",
    "candidate_any_response_fraction",
    "candidate_same_class_response_fraction",
    "quality_any_response_fraction",
    "quality_same_class_response_fraction",
    "leaveout_correlation_median",
    "leaveout_correlation_ge_threshold_fraction",
    "leaveout_residual_median",
    "spatial_local_template_source_count_median",
    "spatial_local_correlation_median",
    "spatial_local_residual_median",
    "spatial_local_correlation_improved_fraction",
    "spatial_local_residual_reduced_fraction",
    "spatial_median_correlation_delta",
    "spatial_median_residual_delta",
    "candidate_persistence_wilson95_low",
    "candidate_persistence_wilson95_high",
    "quality_fraction_wilson95_low",
    "quality_fraction_wilson95_high",
    "quality_response_wilson95_low",
    "quality_response_wilson95_high",
];

pub fn write_artifacts(
    result: &FeatureEvidenceMatrixResult,
    out_dir: &Path,
) -> Result<PathBuf, String> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let output = fs::canonicalize(out_dir).map_err(|error| error.to_string())?;
    let rows = &result.rows;
    let csv_path = output.join("feature_evidence_matrix.csv");
    let mut file = File::create(&csv_path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .terminator(Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(CSV_COLUMNS)
        .map_err(|error| error.to_string())?;
    for row in rows {
        let object = serde_json::to_value(row).map_err(|error| error.to_string())?;
        let object = object
            .as_object()
            .ok_or("feature evidence row is not an object")?;
        let values: Vec<String> = CSV_COLUMNS
            .iter()
            .map(|column| {
                object
                    .get(*column)
                    .map(|value| csv_cell(value, FLOAT_COLUMNS.contains(column)))
                    .unwrap_or_default()
            })
            .collect();
        writer
            .write_record(values)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    fs::write(
        output.join("feature_evidence_matrix.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output)
}

fn option_value<'a>(
    args: &'a [String],
    index: &mut usize,
    name: &str,
    inline: Option<&'a str>,
) -> Result<&'a str, String> {
    if let Some(value) = inline {
        return Ok(value);
    }
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| format!("{name} requires a value"))
}

pub fn run_desktop(args: &[String]) -> Result<Value, String> {
    let mut source_morphology = None;
    let mut feature_cross = None;
    let mut psf_leaveout = None;
    let mut psf_spatial = None;
    let mut correlation_threshold = 0.8;
    let mut out_dir = PathBuf::from("tmp/feature-evidence-matrix");
    let mut index = 0;
    while index < args.len() {
        let (name, inline) = args[index]
            .split_once('=')
            .map(|(a, b)| (a, Some(b)))
            .unwrap_or((args[index].as_str(), None));
        if [
            "--source-morphology",
            "--feature-cross",
            "--psf-leaveout",
            "--psf-spatial",
            "--correlation-threshold",
            "--out-dir",
        ]
        .contains(&name)
        {
            let raw = option_value(args, &mut index, name, inline)?;
            match name {
                "--source-morphology" => source_morphology = Some(PathBuf::from(raw)),
                "--feature-cross" => feature_cross = Some(PathBuf::from(raw)),
                "--psf-leaveout" => psf_leaveout = Some(PathBuf::from(raw)),
                "--psf-spatial" => psf_spatial = Some(PathBuf::from(raw)),
                "--correlation-threshold" => {
                    correlation_threshold = raw
                        .parse()
                        .map_err(|error| format!("invalid value for {name}: {error}"))?
                }
                _ => out_dir = PathBuf::from(raw),
            }
        } else if name.starts_with('-') {
            return Err(format!("unknown argument {name}"));
        } else {
            return Err(format!("unexpected positional argument {name}"));
        }
        index += 1;
    }
    let morphology = source_morphology.ok_or("--source-morphology is required")?;
    let cross = feature_cross.ok_or("--feature-cross is required")?;
    let psf = psf_leaveout.ok_or("--psf-leaveout is required")?;
    let result = run(
        &morphology,
        &cross,
        &psf,
        correlation_threshold,
        psf_spatial.as_deref(),
    )?;
    let output = write_artifacts(&result, &out_dir)?;
    Ok(
        json!({"tool":"rst19-feature-evidence", "out_dir":output, "result_path":output.join("feature_evidence_matrix.json"), "result":result.as_dict()}),
    )
}

pub fn run_cli(args: &[String]) -> Result<(), String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("rst19-feature-evidence --source-morphology CSV --feature-cross CSV --psf-leaveout CSV [--psf-spatial CSV] [options]");
        return Ok(());
    }
    let result = run_desktop(args)?;
    println!(
        "rst19-feature-evidence: 矩阵已写入 {}",
        result["out_dir"].as_str().unwrap_or_default()
    );
    println!(
        "{}",
        result["result"]["conclusion"].as_str().unwrap_or_default()
    );
    for row in result["result"]["rows"].as_array().unwrap() {
        println!("{}: pattern={} candidate_persistence={} quality_response={} candidate_same_class={} quality_same_class={} leaveout_corr={} spatial_local_corr={} audit_stage={} counting_policy={}",
            row["feature_class"].as_str().unwrap_or_default(), row["evidence_pattern"], row["candidate_persistence_fraction"], row["quality_response_fraction"], row["candidate_same_class_response_fraction"], row["quality_same_class_response_fraction"], row["leaveout_correlation_median"], row["spatial_local_correlation_median"], row["recommended_audit_stage"], row["counting_policy"]);
    }
    Ok(())
}

pub fn handle_request(request: &Value) -> Result<Value, String> {
    let morphology = request["source_morphology_path"]
        .as_str()
        .ok_or("missing source_morphology_path")?;
    let cross = request["feature_cross_path"]
        .as_str()
        .ok_or("missing feature_cross_path")?;
    let psf = request["psf_leaveout_path"]
        .as_str()
        .ok_or("missing psf_leaveout_path")?;
    let threshold = request["correlation_threshold"].as_f64().unwrap_or(0.8);
    let spatial = request["psf_spatial_path"].as_str().map(Path::new);
    let result = run(
        Path::new(morphology),
        Path::new(cross),
        Path::new(psf),
        threshold,
        spatial,
    )?;
    let output = if let Some(path) = request["output_dir"].as_str() {
        let output = write_artifacts(&result, Path::new(path))?;
        json!({"out_dir":output, "result_path":output.join("feature_evidence_matrix.json")})
    } else {
        Value::Null
    };
    Ok(json!({"result":result.as_dict(), "output":output}))
}
