//! Rust writers for the tabular, detection-derived source audit artifacts.
//!
//! Frame-dependent cutouts, empirical-PSF products, and charts are separate
//! layers; this module writes only tables that can be derived from a completed
//! `DetectionResult` without rerunning detection or changing quality decisions.

use crate::{
    detection::DetectionResult,
    measurement::Detection,
    source_classification::{
        self, FeatureMorphologyRow, FeatureSpatialRow, FeatureSubclassSummaryRow,
    },
    Result,
};
use serde::Serialize;
use serde_json::{Map, Value};
use std::{fs::File, io::Write, path::Path};

const SOURCE_FIELDS: &[&str] = &[
    "detection_id",
    "x",
    "y",
    "peak_x",
    "peak_y",
    "centroid_shift_px",
    "peak",
    "flux",
    "background",
    "noise",
    "snr",
    "flux_error",
    "flux_snr",
    "filter_snr",
    "proposal_snr",
    "proposal_methods",
    "proposal_scales",
    "nearest_gaussian_px",
    "deblend_delta_bic",
    "deblend_component_snr",
    "repeated_code_count",
    "range_anomaly_pixel_count",
    "repeated_code_values",
    "fwhm",
    "fwhm_x",
    "fwhm_y",
    "ellipticity",
    "sharpness",
    "footprint_pixels",
    "psf_support_pixels",
    "quality_passed",
    "quality_class",
    "feature_class",
    "feature_class_label",
    "flags",
];

const FEATURE_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "candidate_count",
    "quality_count",
    "rejected_count",
    "quality_fraction",
    "median_flux_snr",
    "max_flux_snr",
    "common_flags",
];

const METHOD_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "candidate_count",
    "quality_count",
    "gaussian_present_count",
    "dog_narrow_present_count",
    "dog_broad_present_count",
    "all_three_count",
    "gaussian_only_count",
    "dog_only_count",
    "partial_combination_count",
    "no_gaussian_count",
    "no_method_count",
    "no_gaussian_quality_count",
    "no_gaussian_median_flux_snr",
];

const MORPHOLOGY_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "candidate_count",
    "quality_count",
    "rejected_count",
    "quality_fraction",
    "high_flux_snr_threshold",
    "high_flux_snr_count",
    "high_flux_snr_rejected_count",
    "high_flux_snr_rejection_fraction",
    "median_peak_snr",
    "median_flux_snr",
    "common_flags",
    "peak_snr_p10",
    "peak_snr_median",
    "peak_snr_p90",
    "flux_snr_p10",
    "flux_snr_median",
    "flux_snr_p90",
    "filter_snr_p10",
    "filter_snr_median",
    "filter_snr_p90",
    "fwhm_px_p10",
    "fwhm_px_median",
    "fwhm_px_p90",
    "ellipticity_p10",
    "ellipticity_median",
    "ellipticity_p90",
    "sharpness_p10",
    "sharpness_median",
    "sharpness_p90",
    "psf_support_pixels_p10",
    "psf_support_pixels_median",
    "psf_support_pixels_p90",
    "footprint_pixels_p10",
    "footprint_pixels_median",
    "footprint_pixels_p90",
    "centroid_shift_px_p10",
    "centroid_shift_px_median",
    "centroid_shift_px_p90",
    "x_px_p10",
    "x_px_median",
    "x_px_p90",
    "y_px_p10",
    "y_px_median",
    "y_px_p90",
    "range_anomaly_fraction",
    "linear_artifact_fraction",
    "masked_or_edge_fraction",
    "crowded_blend_fraction",
    "spike_or_support_fraction",
    "weak_or_background_fraction",
    "shape_outlier_fraction",
];

const SUBCLASS_FIELDS: &[&str] = &[
    "subclass",
    "subclass_label",
    "definition",
    "candidate_count",
    "quality_count",
    "rejected_count",
    "quality_fraction",
    "high_flux_snr_threshold",
    "high_snr_rejected_count",
    "median_flux_snr",
    "primary_feature_classes",
    "common_flags",
];

const SPATIAL_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "grid_size",
    "grid_row",
    "grid_column",
    "x_min_px",
    "x_max_px",
    "y_min_px",
    "y_max_px",
    "candidate_count",
    "quality_count",
    "rejected_count",
    "quality_fraction",
    "high_flux_snr_threshold",
    "high_flux_snr_count",
    "high_flux_snr_rejected_count",
    "median_flux_snr",
    "flux_snr_p90",
];

const SOURCE_GRID_FIELDS: &[&str] = &[
    "grid_row",
    "grid_column",
    "x_min_px",
    "x_max_px",
    "y_min_px",
    "y_max_px",
    "returned_count",
    "quality_count",
    "rejected_count",
    "quality_fraction",
];

const QUALITY_SUMMARY_FIELDS: &[&str] = &["metric", "value", "definition"];

#[derive(Debug, Clone, PartialEq, Serialize)]
struct QualityMetricRow {
    metric: String,
    value: String,
    definition: String,
}

fn scalar_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(values) => values.iter().map(scalar_text).collect::<Vec<_>>().join("|"),
        Value::Object(_) => value.to_string(),
    }
}

fn quality_metric(
    metric: impl Into<String>,
    value: Value,
    definition: impl Into<String>,
) -> QualityMetricRow {
    QualityMetricRow {
        metric: metric.into(),
        value: scalar_text(&value),
        definition: definition.into(),
    }
}

fn optional_float_text(value: Option<f64>) -> String {
    value.map_or_else(
        || "nan".into(),
        |number| scalar_text(&serde_json::json!(number)),
    )
}

fn quality_metric_text(
    metric: impl Into<String>,
    value: impl Into<String>,
    definition: impl Into<String>,
) -> QualityMetricRow {
    QualityMetricRow {
        metric: metric.into(),
        value: value.into(),
        definition: definition.into(),
    }
}

fn quality_summary_rows(
    detection: &DetectionResult,
    features: &[source_classification::FeatureSummaryRow],
    spatial: &SourceSpatialGridSummary,
) -> Vec<QualityMetricRow> {
    let mut flag_counts = std::collections::BTreeMap::<&str, usize>::new();
    let mut support_counts = [0_usize; 10];
    for source in &detection.sources {
        for flag in &source.flags {
            *flag_counts.entry(flag).or_default() += 1;
        }
        if source.quality_passed {
            if let Some(support) = source.psf_support_pixels.filter(|value| *value < 10) {
                support_counts[support] += 1;
            }
        }
    }
    let support_ge_4 = support_counts
        .iter()
        .enumerate()
        .filter(|(support, _)| *support >= 4)
        .map(|(_, count)| count)
        .sum::<usize>();
    let support_ge_5 = support_counts
        .iter()
        .enumerate()
        .filter(|(support, _)| *support >= 5)
        .map(|(_, count)| count)
        .sum::<usize>();
    let mut rows = vec![
        quality_metric(
            "candidate_count",
            serde_json::json!(detection.candidate_count),
            "NMS 后的候选峰数量",
        ),
        quality_metric(
            "returned_count",
            serde_json::json!(detection.sources.len()),
            "已计算源级属性的返回数量",
        ),
        quality_metric(
            "quality_count",
            serde_json::json!(detection.quality_count),
            "通过通量、形状、掩膜和有效性规则的数量",
        ),
        quality_metric(
            "rejected_count",
            serde_json::json!(detection.sources.len() - detection.quality_count),
            "返回但未通过质量规则的数量",
        ),
        quality_metric(
            "quality_count_support_ge_4",
            serde_json::json!(support_ge_4),
            "诊断口径：质量源中 3×3 PSF 支持像素至少 4 个；不改变 quality_count",
        ),
        quality_metric(
            "quality_count_support_ge_5",
            serde_json::json!(support_ge_5),
            "诊断口径：质量源中 3×3 PSF 支持像素至少 5 个；不因接近外部数量而自动采用",
        ),
        quality_metric(
            "background_adu",
            serde_json::json!(detection.background),
            "全图稳健背景基线",
        ),
        quality_metric(
            "noise_adu",
            serde_json::json!(detection.noise),
            "全图稳健噪声尺度",
        ),
    ];
    let parameter_rows = [
        ("threshold_sigma", "匹配滤波候选阈值"),
        ("proposal_mode", "宽筛选提案模式"),
        ("proposal_count_gaussian", "Gaussian 算法内 NMS 后提案数"),
        ("proposal_count_dog_narrow", "DoG 窄尺度算法内 NMS 后提案数"),
        ("proposal_count_dog_broad", "DoG 宽尺度算法内 NMS 后提案数"),
        (
            "proposal_count_starlet_s1",
            "Starlet 第一尺度算法内 NMS 后提案数",
        ),
        (
            "proposal_count_starlet_s2",
            "Starlet 第二尺度算法内 NMS 后提案数",
        ),
        ("min_distance", "候选峰最小间距"),
        ("psf_fwhm", "Gaussian 匹配滤波 PSF FWHM"),
        ("min_flux_snr", "质量层通量 SNR 下限"),
        (
            "negative_overflow_limit",
            "有符号整型图像测光孔径内的极端负码审计线；不是相机满阱标定",
        ),
        (
            "repeated_high_code_values",
            "全幅频次异常的高位整型码；只作数据有效性先验，不是饱和标定",
        ),
        (
            "repeated_high_code_min_count",
            "全幅重复高位码的工程审计最小频次",
        ),
        (
            "close_pair_guarded_source_count",
            "近邻 Gaussian 双 PSF 证据不足而从质量层降级的源数；候选仍保留",
        ),
        (
            "min_psf_support_pixels",
            "候选峰中心 3×3 核心中必须超过支持阈值的最少像素数",
        ),
    ];
    for (key, definition) in parameter_rows {
        let metric = match key {
            "min_distance" => "min_distance_px",
            "psf_fwhm" => "psf_fwhm_px",
            "repeated_high_code_values" => key,
            _ => key,
        };
        rows.push(quality_metric(
            metric,
            detection
                .parameters
                .get(key)
                .cloned()
                .unwrap_or(Value::Null),
            definition,
        ));
    }
    rows.extend([
        quality_metric(
            "truncated",
            serde_json::json!(u8::from(
                detection.candidate_count > detection.sources.len()
            )),
            "是否因显式返回上限截断",
        ),
        quality_metric_text(
            "spatial_grid_size",
            spatial.grid_size.to_string(),
            "空间诊断每轴网格数量",
        ),
        quality_metric_text(
            "quality_density_cv",
            optional_float_text(spatial.quality_density_cv),
            "16×16 网格质量源计数的变异系数；描述空间不均匀，不等同于灵敏度",
        ),
        quality_metric_text(
            "quality_fraction_median",
            optional_float_text(spatial.quality_fraction_median),
            "非空网格中质量通过率中位数",
        ),
        quality_metric_text(
            "quality_fraction_min",
            optional_float_text(spatial.quality_fraction_min),
            "非空网格中质量通过率最小值；需结合边缘/掩膜解释",
        ),
        quality_metric_text(
            "quality_fraction_max",
            optional_float_text(spatial.quality_fraction_max),
            "非空网格中质量通过率最大值",
        ),
        quality_metric(
            "feature_class_count",
            serde_json::json!(features.len()),
            "互斥首要特征类别数量；类别内仍保留原始重叠 flags",
        ),
    ]);
    for row in features {
        rows.push(quality_metric(
            format!("feature:{}:candidate_count", row.feature_class),
            serde_json::json!(row.candidate_count),
            format!("首要特征类别：{}；互斥审计计数", row.feature_class_label),
        ));
    }
    for row in features {
        rows.push(quality_metric(
            format!("feature:{}:quality_count", row.feature_class),
            serde_json::json!(row.quality_count),
            format!(
                "首要特征类别：{}；其中通过质量层的数量",
                row.feature_class_label
            ),
        ));
    }
    for (flag, count) in flag_counts {
        rows.push(quality_metric(
            format!("flag:{flag}"),
            serde_json::json!(count),
            "可重叠的源拒绝/审计标志计数",
        ));
    }
    for (support, count) in support_counts.into_iter().enumerate() {
        rows.push(quality_metric(
            format!("psf_support_pixels:{support}"),
            serde_json::json!(count),
            "质量源的 3×3 PSF 支持像素分布；用于人工抽检，不是独立真值标签",
        ));
    }
    rows
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SourceSpatialGridRow {
    pub grid_row: usize,
    pub grid_column: usize,
    pub x_min_px: f64,
    pub x_max_px: f64,
    pub y_min_px: f64,
    pub y_max_px: f64,
    pub returned_count: usize,
    pub quality_count: usize,
    pub rejected_count: usize,
    pub quality_fraction: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SourceSpatialGridSummary {
    pub grid_size: usize,
    pub quality_density_cv: Option<f64>,
    pub quality_fraction_median: Option<f64>,
    pub quality_fraction_min: Option<f64>,
    pub quality_fraction_max: Option<f64>,
}

pub fn summarize_source_spatial_grid(
    detection: &DetectionResult,
    grid_size: usize,
) -> Result<(Vec<SourceSpatialGridRow>, SourceSpatialGridSummary)> {
    if grid_size == 0 {
        return Err("grid_size must be positive".into());
    }
    let [height, width] = detection.image_shape;
    if height == 0 || width == 0 {
        return Err("image_shape must contain two positive dimensions".into());
    }
    let mut returned = vec![0_usize; grid_size * grid_size];
    let mut quality = vec![0_usize; grid_size * grid_size];
    for source in &detection.sources {
        if !source.x.is_finite() || !source.y.is_finite() {
            return Err("source coordinates must be finite".into());
        }
        let column = ((source.x / width.max(1) as f64 * grid_size as f64) as isize)
            .clamp(0, grid_size as isize - 1) as usize;
        let row = ((source.y / height.max(1) as f64 * grid_size as f64) as isize)
            .clamp(0, grid_size as isize - 1) as usize;
        let index = row * grid_size + column;
        returned[index] += 1;
        if source.quality_passed {
            quality[index] += 1;
        }
    }
    let mut rows = Vec::with_capacity(grid_size * grid_size);
    let mut nonempty_fractions = Vec::new();
    for row in 0..grid_size {
        for column in 0..grid_size {
            let index = row * grid_size + column;
            let fraction =
                (returned[index] != 0).then_some(quality[index] as f64 / returned[index] as f64);
            if let Some(fraction) = fraction {
                nonempty_fractions.push(fraction);
            }
            rows.push(SourceSpatialGridRow {
                grid_row: row,
                grid_column: column,
                x_min_px: column as f64 * width as f64 / grid_size as f64,
                x_max_px: (column + 1) as f64 * width as f64 / grid_size as f64,
                y_min_px: row as f64 * height as f64 / grid_size as f64,
                y_max_px: (row + 1) as f64 * height as f64 / grid_size as f64,
                returned_count: returned[index],
                quality_count: quality[index],
                rejected_count: returned[index] - quality[index],
                quality_fraction: fraction,
            });
        }
    }
    let quality_as_f64 = quality
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let density_mean = quality_as_f64.iter().sum::<f64>() / quality_as_f64.len() as f64;
    let quality_density_cv = (density_mean > 0.0).then(|| {
        let variance = quality_as_f64
            .iter()
            .map(|value| (value - density_mean).powi(2))
            .sum::<f64>()
            / quality_as_f64.len() as f64;
        variance.sqrt() / density_mean
    });
    let quality_fraction_median = if nonempty_fractions.is_empty() {
        None
    } else {
        Some(crate::statistics::median(&nonempty_fractions)?)
    };
    let quality_fraction_min = nonempty_fractions.iter().copied().min_by(f64::total_cmp);
    let quality_fraction_max = nonempty_fractions.iter().copied().max_by(f64::total_cmp);
    Ok((
        rows,
        SourceSpatialGridSummary {
            grid_size,
            quality_density_cv,
            quality_fraction_median,
            quality_fraction_min,
            quality_fraction_max,
        },
    ))
}

fn csv_cell(value: &Value, field: &str) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) if field == "quality_passed" => {
            if *value { "True" } else { "False" }.to_owned()
        }
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(values) => values
            .iter()
            .map(|value| csv_cell(value, ""))
            .collect::<Vec<_>>()
            .join("|"),
        Value::Object(_) => value.to_string(),
    }
}

fn write_rows<T: Serialize>(path: &Path, fields: &[&str], rows: &[T]) -> Result<()> {
    let mut file = File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xEF, 0xBB, 0xBF])
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(fields)
        .map_err(|error| error.to_string())?;
    for row in rows {
        let value = serde_json::to_value(row).map_err(|error| error.to_string())?;
        let object = value
            .as_object()
            .ok_or("source artifact row must serialize to an object")?;
        let cells = fields
            .iter()
            .map(|field| csv_cell(object.get(*field).unwrap_or(&Value::Null), field))
            .collect::<Vec<_>>();
        writer
            .write_record(cells)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn source_catalog_rows(
    sources: &[Detection],
    features: &[source_classification::FeatureSummaryRow],
) -> Result<Vec<Map<String, Value>>> {
    let labels = features
        .iter()
        .map(|row| (row.feature_class, row.feature_class_label))
        .collect::<std::collections::HashMap<_, _>>();
    sources
        .iter()
        .map(|source| {
            let mut row = serde_json::to_value(source)
                .map_err(|error| error.to_string())?
                .as_object()
                .cloned()
                .ok_or("detection source must serialize to an object")?;
            let feature_class = source_classification::classify_source_feature(source);
            row.insert("flags".into(), Value::String(source.flags.join("|")));
            row.insert(
                "proposal_methods".into(),
                Value::String(source.proposal_methods.join("|")),
            );
            row.insert(
                "proposal_scales".into(),
                Value::String(
                    source
                        .proposal_scales
                        .iter()
                        .map(|scale| format!("{scale:.3}"))
                        .collect::<Vec<_>>()
                        .join("|"),
                ),
            );
            row.insert(
                "repeated_code_values".into(),
                Value::String(
                    source
                        .repeated_code_values
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("|"),
                ),
            );
            row.insert(
                "quality_passed".into(),
                Value::String(
                    if source.quality_passed {
                        "True"
                    } else {
                        "False"
                    }
                    .into(),
                ),
            );
            row.insert(
                "quality_class".into(),
                Value::String(
                    if source.quality_passed {
                        "quality"
                    } else {
                        "rejected"
                    }
                    .into(),
                ),
            );
            row.insert("feature_class".into(), Value::String(feature_class.into()));
            row.insert(
                "feature_class_label".into(),
                Value::String(
                    labels
                        .get(feature_class)
                        .ok_or("feature class label is missing")?
                        .to_string(),
                ),
            );
            Ok(row)
        })
        .collect()
}

/// Write the source tables that depend only on the completed detection result.
/// This does not detect again, read the FITS image, or alter quality decisions.
pub fn write_detection_source_tables(detection: &DetectionResult, out_dir: &Path) -> Result<()> {
    if out_dir.as_os_str().is_empty() {
        return Err("output directory must not be empty".into());
    }
    std::fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;

    let feature_rows = source_classification::summarize_source_features(&detection.sources);
    let method_rows = source_classification::summarize_source_proposal_methods(&detection.sources);
    let morphology_rows: Vec<FeatureMorphologyRow> =
        source_classification::summarize_feature_morphology(&detection.sources, 10.0)?;
    let subclass_rows: Vec<FeatureSubclassSummaryRow> =
        source_classification::summarize_source_feature_subclasses(&detection.sources, 10.0)?;
    let spatial_rows: Vec<FeatureSpatialRow> =
        source_classification::summarize_feature_spatial_distribution(
            &detection.sources,
            detection.image_shape,
            4,
            10.0,
        )?;
    let (source_grid_rows, grid_summary) = summarize_source_spatial_grid(detection, 16)?;
    let quality_rows = quality_summary_rows(detection, &feature_rows, &grid_summary);

    let catalog = source_catalog_rows(&detection.sources, &feature_rows)?;
    write_rows(&out_dir.join("source_catalog.csv"), SOURCE_FIELDS, &catalog)?;
    write_rows(
        &out_dir.join("source_feature_summary.csv"),
        FEATURE_FIELDS,
        &feature_rows,
    )?;
    write_rows(
        &out_dir.join("source_feature_method_summary.csv"),
        METHOD_FIELDS,
        &method_rows,
    )?;
    write_rows(
        &out_dir.join("source_feature_morphology.csv"),
        MORPHOLOGY_FIELDS,
        &morphology_rows,
    )?;
    write_rows(
        &out_dir.join("source_feature_subclass_summary.csv"),
        SUBCLASS_FIELDS,
        &subclass_rows,
    )?;
    write_rows(
        &out_dir.join("source_feature_spatial.csv"),
        SPATIAL_FIELDS,
        &spatial_rows,
    )?;
    write_rows(
        &out_dir.join("source_spatial_grid.csv"),
        SOURCE_GRID_FIELDS,
        &source_grid_rows,
    )?;
    write_rows(
        &out_dir.join("source_quality_summary.csv"),
        QUALITY_SUMMARY_FIELDS,
        &quality_rows,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::DetectionResult;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    fn fixture_result() -> DetectionResult {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/source_classification_parity.json"
        ))
        .unwrap();
        let sources: Vec<Detection> = serde_json::from_value(fixture["sources"].clone()).unwrap();
        let quality_count = sources
            .iter()
            .filter(|source| source.quality_passed)
            .count();
        DetectionResult {
            image_shape: [128, 128],
            background: 2.0,
            noise: 3.0,
            threshold: 12.0,
            candidate_count: sources.len(),
            sources,
            quality_count,
            parameters: json!({}),
            candidate_peaks: Vec::new(),
            line_artifact_coordinates: Vec::new(),
        }
    }

    fn temp_dir() -> std::path::PathBuf {
        let sequence = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rst19-source-tables-{}-{sequence}",
            std::process::id()
        ))
    }

    #[test]
    fn writes_all_detection_derived_source_tables() {
        let root = temp_dir();
        write_detection_source_tables(&fixture_result(), &root).unwrap();
        let baselines: [(&str, &[u8]); 8] = [
            (
                "source_catalog.csv",
                include_bytes!("../tests/fixtures/source_tables/source_catalog.csv"),
            ),
            (
                "source_feature_summary.csv",
                include_bytes!("../tests/fixtures/source_tables/source_feature_summary.csv"),
            ),
            (
                "source_feature_method_summary.csv",
                include_bytes!("../tests/fixtures/source_tables/source_feature_method_summary.csv"),
            ),
            (
                "source_feature_morphology.csv",
                include_bytes!("../tests/fixtures/source_tables/source_feature_morphology.csv"),
            ),
            (
                "source_feature_subclass_summary.csv",
                include_bytes!(
                    "../tests/fixtures/source_tables/source_feature_subclass_summary.csv"
                ),
            ),
            (
                "source_feature_spatial.csv",
                include_bytes!("../tests/fixtures/source_tables/source_feature_spatial.csv"),
            ),
            (
                "source_spatial_grid.csv",
                include_bytes!("../tests/fixtures/source_tables/source_spatial_grid.csv"),
            ),
            (
                "source_quality_summary.csv",
                include_bytes!("../tests/fixtures/source_tables/source_quality_summary.csv"),
            ),
        ];
        for (name, baseline) in baselines {
            let bytes = std::fs::read(root.join(name)).unwrap();
            assert_eq!(bytes, baseline, "legacy CSV bytes differ for {name}");
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
