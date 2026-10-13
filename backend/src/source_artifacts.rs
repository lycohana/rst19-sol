//! Rust writers for the tabular, detection-derived source audit artifacts.
//!
//! Frame-dependent cutouts and empirical-PSF products are separate layers;
//! this module writes detection-derived source tables and diagnostic charts
//! without rerunning detection or changing quality decisions.

use crate::{
    detection::DetectionResult,
    measurement::Detection,
    models::FitsFrame,
    source_classification::{
        self, FeatureMorphologyRow, FeatureSpatialRow, FeatureSubclassSummaryRow,
    },
    Result,
};
use serde::Serialize;
use serde_json::{json, Map, Value};
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

#[derive(Debug, Clone)]
struct Chart {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl Chart {
    fn new(width: usize, height: usize, background: [u8; 3]) -> Self {
        let mut pixels = vec![0; width * height * 3];
        for pixel in pixels.chunks_exact_mut(3) {
            pixel.copy_from_slice(&background);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    fn pixel(&mut self, x: i32, y: i32, color: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let offset = (y as usize * self.width + x as usize) * 3;
        self.pixels[offset..offset + 3].copy_from_slice(&color);
    }

    fn rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: [u8; 3]) {
        for y in y0.min(y1)..=y0.max(y1) {
            for x in x0.min(x1)..=x0.max(x1) {
                self.pixel(x, y, color);
            }
        }
    }

    fn line(&mut self, start: (i32, i32), end: (i32, i32), color: [u8; 3]) {
        let (mut x0, mut y0) = start;
        let (x1, y1) = end;
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            self.pixel(x0, y0, color);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let doubled = 2 * error;
            if doubled >= dy {
                error += dy;
                x0 += sx;
            }
            if doubled <= dx {
                error += dx;
                y0 += sy;
            }
        }
    }

    fn text(&mut self, x: i32, y: i32, text: &str, color: [u8; 3]) {
        for (character_index, character) in text.chars().enumerate() {
            let rows = glyph(character.to_ascii_uppercase());
            for (row_index, row) in rows.into_iter().enumerate() {
                for column in 0..5 {
                    if row & (1 << (4 - column)) != 0 {
                        self.pixel(
                            x + character_index as i32 * 6 + column,
                            y + row_index as i32,
                            color,
                        );
                    }
                }
            }
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path).map_err(|error| error.to_string())?;
        let mut encoder = png::Encoder::new(file, self.width as u32, self.height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .map_err(|error| error.to_string())?
            .write_image_data(&self.pixels)
            .map_err(|error| error.to_string())
    }
}

fn glyph(character: char) -> [u8; 7] {
    match character {
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [14, 17, 16, 16, 16, 17, 14],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [14, 17, 16, 23, 17, 17, 15],
        'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [14, 4, 4, 4, 4, 4, 14],
        'J' => [7, 2, 2, 2, 18, 18, 12],
        'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 25, 25, 21, 19, 19, 17],
        'O' => [14, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'Q' => [14, 17, 17, 17, 21, 18, 13],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'X' => [17, 17, 10, 4, 10, 17, 17],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        'Z' => [31, 1, 2, 4, 8, 16, 31],
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        '_' => [0, 0, 0, 0, 0, 0, 31],
        '.' => [0, 0, 0, 0, 0, 6, 6],
        ',' => [0, 0, 0, 0, 6, 6, 4],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '/' => [1, 2, 2, 4, 8, 8, 16],
        ':' => [0, 6, 6, 0, 6, 6, 0],
        '(' => [2, 4, 8, 8, 8, 4, 2],
        ')' => [8, 4, 2, 2, 2, 4, 8],
        '>' => [16, 8, 4, 2, 4, 8, 16],
        '<' => [1, 2, 4, 8, 4, 2, 1],
        '=' => [0, 31, 0, 31, 0, 0, 0],
        '+' => [0, 4, 4, 31, 4, 4, 0],
        _ => [0; 7],
    }
}

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

fn grouped_count(value: usize) -> String {
    let digits = value.to_string();
    let first_group = digits.len() % 3;
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index != 0
            && (index == first_group || (index > first_group && (index - first_group) % 3 == 0))
        {
            formatted.push(',');
        }
        formatted.push(character);
    }
    formatted
}

fn write_quality_flag_chart(path: &Path, detection: &DetectionResult) -> Result<()> {
    const WIDTH: i32 = 1200;
    const HEIGHT: i32 = 680;
    const LEFT: i32 = 90;
    const TOP: i32 = 75;
    const RIGHT: i32 = WIDTH - 45;
    const BOTTOM: i32 = HEIGHT - 110;
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for source in &detection.sources {
        for flag in &source.flags {
            *counts.entry(flag).or_default() += 1;
        }
    }
    let mut rows = vec![("QUALITY_PASS".to_owned(), detection.quality_count)];
    let mut flags = counts.into_iter().collect::<Vec<_>>();
    flags.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
    rows.extend(
        flags
            .into_iter()
            .map(|(name, count)| (name.to_owned(), count)),
    );

    let background = [244, 240, 231];
    let mut chart = Chart::new(WIDTH as usize, HEIGHT as usize, background);
    chart.text(
        LEFT,
        22,
        "SOURCE QUALITY - OVERLAPPING FLAGS (LOG1P)",
        [32, 41, 61],
    );
    let transformed = rows
        .iter()
        .map(|(_, count)| (*count as f64).ln_1p())
        .collect::<Vec<_>>();
    let maximum = transformed.iter().copied().fold(0.0, f64::max).max(1.0);
    let slot = (RIGHT - LEFT) as f64 / rows.len().max(1) as f64;
    for (index, ((label, raw_count), value)) in rows.iter().zip(transformed).enumerate() {
        let x0 = (LEFT as f64 + index as f64 * slot + 0.12 * slot).floor() as i32;
        let x1 = (LEFT as f64 + (index + 1) as f64 * slot - 0.12 * slot).ceil() as i32;
        let y = (BOTTOM as f64 - value / maximum * (BOTTOM - TOP) as f64).round() as i32;
        let color = if label == "QUALITY_PASS" {
            [79, 155, 131]
        } else {
            [215, 148, 50]
        };
        chart.rect(x0, y, x1, BOTTOM, color);
        chart.text(
            x0,
            TOP.max(y - 16),
            &grouped_count(*raw_count),
            [39, 49, 71],
        );
        let short_label = if label.chars().count() <= 15 {
            label.clone()
        } else {
            format!("{}...", label.chars().take(12).collect::<String>())
        };
        chart.text(x0, BOTTOM + 10, &short_label, [98, 107, 125]);
    }
    chart.line((LEFT, BOTTOM), (RIGHT, BOTTOM), [98, 107, 125]);
    chart.text(
        LEFT,
        HEIGHT - 30,
        "FLAGS MAY OVERLAP; LOG1P BAR HEIGHTS PRESERVE RAW COUNTS",
        [98, 107, 125],
    );
    chart.save(path)
}

fn write_feature_class_chart(
    path: &Path,
    rows: &[source_classification::FeatureSummaryRow],
) -> Result<()> {
    const WIDTH: i32 = 1280;
    const HEIGHT: i32 = 720;
    const LEFT: i32 = 82;
    const TOP: i32 = 80;
    const RIGHT: i32 = WIDTH - 42;
    const BOTTOM: i32 = HEIGHT - 120;
    let background = [244, 240, 231];
    let mut chart = Chart::new(WIDTH as usize, HEIGHT as usize, background);
    chart.text(
        LEFT,
        24,
        "SOURCE FEATURE CLASSES - MUTUALLY EXCLUSIVE PRIMARY AUDIT",
        [32, 41, 61],
    );
    if rows.is_empty() {
        return chart.save(path);
    }
    let transformed = rows
        .iter()
        .map(|row| (row.candidate_count as f64).ln_1p())
        .collect::<Vec<_>>();
    let maximum = transformed.iter().copied().fold(0.0, f64::max).max(1.0);
    let slot = (RIGHT - LEFT) as f64 / rows.len() as f64;
    for (index, (row, scaled)) in rows.iter().zip(transformed).enumerate() {
        let x0 = (LEFT as f64 + index as f64 * slot + 0.10 * slot).floor() as i32;
        let x1 = (LEFT as f64 + (index + 1) as f64 * slot - 0.10 * slot).ceil() as i32;
        let y = (BOTTOM as f64 - scaled / maximum * (BOTTOM - TOP) as f64).round() as i32;
        let color = if row.quality_count != 0 {
            [79, 155, 131]
        } else {
            [215, 148, 50]
        };
        chart.rect(x0, y, x1, BOTTOM, color);
        chart.text(
            x0,
            TOP.max(y - 17),
            &grouped_count(row.candidate_count),
            [39, 49, 71],
        );
        let short_label = if row.feature_class.chars().count() <= 15 {
            row.feature_class.to_owned()
        } else {
            format!(
                "{}...",
                row.feature_class.chars().take(12).collect::<String>()
            )
        };
        chart.text(x0, BOTTOM + 10, &short_label, [98, 107, 125]);
    }
    chart.line((LEFT, BOTTOM), (RIGHT, BOTTOM), [98, 107, 125]);
    chart.text(
        LEFT,
        HEIGHT - 34,
        "PRIMARY CLASSES EXCLUSIVE; FLAGS CAN OVERLAP; BAR HEIGHTS USE LOG1P",
        [98, 107, 125],
    );
    chart.save(path)
}

fn interpolated_color(low: [u8; 3], high: [u8; 3], fraction: f64) -> [u8; 3] {
    std::array::from_fn(|index| {
        (low[index] as f64 + fraction.clamp(0.0, 1.0) * (high[index] as f64 - low[index] as f64))
            .round_ties_even()
            .clamp(0.0, 255.0) as u8
    })
}

fn write_source_spatial_chart(path: &Path, rows: &[SourceSpatialGridRow]) -> Result<()> {
    const WIDTH: i32 = 1200;
    const HEIGHT: i32 = 680;
    const PANEL_SIZE: i32 = 480;
    const PANEL_TOP: i32 = 82;
    const GRID: usize = 16;
    const BACKGROUND: [u8; 3] = [244, 240, 231];
    const TEXT: [u8; 3] = [98, 107, 125];
    if rows.len() != GRID * GRID {
        return Err(format!(
            "source spatial chart requires {} grid rows",
            GRID * GRID
        ));
    }
    let mut chart = Chart::new(WIDTH as usize, HEIGHT as usize, BACKGROUND);
    chart.text(
        75,
        22,
        "SOURCE SPATIAL AUDIT - RETURNED DENSITY / QUALITY FRACTION",
        [32, 41, 61],
    );
    let returned = rows
        .iter()
        .map(|row| row.returned_count)
        .collect::<Vec<_>>();
    let fractions = rows
        .iter()
        .map(|row| row.quality_fraction)
        .collect::<Vec<_>>();
    let density_max = returned.iter().copied().max().unwrap_or(0) as f64;
    let density_scale = (density_max.ln_1p()).max(1.0);
    let finite_fractions = fractions.iter().flatten().copied().collect::<Vec<_>>();
    let panel_data = [
        (
            75,
            "RETURNED DENSITY - LOG1P(COUNT)",
            [244, 240, 231],
            [215, 148, 50],
        ),
        (635, "QUALITY FRACTION", [222, 216, 202], [79, 155, 131]),
    ];
    for (panel_index, (panel_left, title, low, high)) in panel_data.into_iter().enumerate() {
        chart.text(panel_left, 55, title, [39, 49, 71]);
        let cell_size = PANEL_SIZE / GRID as i32;
        for row in 0..GRID {
            for column in 0..GRID {
                let index = row * GRID + column;
                let scaled = if panel_index == 0 {
                    (returned[index] as f64).ln_1p() / density_scale
                } else {
                    fractions[index].unwrap_or(0.0).clamp(0.0, 1.0)
                };
                let color = interpolated_color(low, high, scaled);
                let x0 = panel_left + column as i32 * cell_size;
                let y0 = PANEL_TOP + row as i32 * cell_size;
                chart.rect(x0, y0, x0 + cell_size - 1, y0 + cell_size - 1, color);
            }
        }
        for index in 1..GRID {
            let offset = index as i32 * cell_size;
            chart.line(
                (panel_left + offset, PANEL_TOP),
                (panel_left + offset, PANEL_TOP + PANEL_SIZE),
                BACKGROUND,
            );
            chart.line(
                (panel_left, PANEL_TOP + offset),
                (panel_left + PANEL_SIZE, PANEL_TOP + offset),
                BACKGROUND,
            );
        }
        chart.line(
            (panel_left, PANEL_TOP),
            (panel_left + PANEL_SIZE, PANEL_TOP),
            [98, 107, 125],
        );
        chart.line(
            (panel_left, PANEL_TOP + PANEL_SIZE),
            (panel_left + PANEL_SIZE, PANEL_TOP + PANEL_SIZE),
            [98, 107, 125],
        );
        chart.line(
            (panel_left, PANEL_TOP),
            (panel_left, PANEL_TOP + PANEL_SIZE),
            [98, 107, 125],
        );
        chart.line(
            (panel_left + PANEL_SIZE, PANEL_TOP),
            (panel_left + PANEL_SIZE, PANEL_TOP + PANEL_SIZE),
            [98, 107, 125],
        );
        chart.text(panel_left, PANEL_TOP + PANEL_SIZE + 12, "X ->", TEXT);
        chart.text(panel_left - 34, PANEL_TOP - 14, "Y v", TEXT);
        let (minimum, median, maximum) = if panel_index == 0 {
            let values = returned
                .iter()
                .map(|count| *count as f64)
                .collect::<Vec<_>>();
            (
                values.iter().copied().reduce(f64::min),
                crate::statistics::median(&values).ok(),
                values.iter().copied().reduce(f64::max),
            )
        } else if finite_fractions.is_empty() {
            (None, None, None)
        } else {
            (
                finite_fractions.iter().copied().reduce(f64::min),
                crate::statistics::median(&finite_fractions).ok(),
                finite_fractions.iter().copied().reduce(f64::max),
            )
        };
        if let (Some(minimum), Some(median), Some(maximum)) = (minimum, median, maximum) {
            chart.text(
                panel_left + 90,
                PANEL_TOP + PANEL_SIZE + 12,
                &format!("MIN {:.3} MED {:.3} MAX {:.3}", minimum, median, maximum),
                TEXT,
            );
        }
    }
    chart.text(
        75,
        HEIGHT - 32,
        "RAW COUNTS ARE IN SOURCE_SPATIAL_GRID.CSV; MAPS DESCRIBE RETURNED SOURCES",
        TEXT,
    );
    chart.save(path)
}

#[derive(Debug, Clone)]
struct ChartSeries {
    name: String,
    values: Vec<Option<f64>>,
    color: [u8; 3],
}

fn chart_tick(value: f64) -> String {
    if value.abs() >= 1000.0 {
        format!(
            "{}{}",
            if value.is_sign_negative() { "-" } else { "" },
            grouped_count(value.abs().round() as usize)
        )
    } else if value.abs() >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

fn chart_x_tick(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn write_line_chart(
    path: &Path,
    title: &str,
    x_values: &[f64],
    series: &[ChartSeries],
    y_label: &str,
    x_label: &str,
) -> Result<()> {
    const WIDTH: i32 = 1200;
    const HEIGHT: i32 = 680;
    const LEFT: i32 = 105;
    const TOP: i32 = 74;
    const RIGHT: i32 = WIDTH - 50;
    const BOTTOM: i32 = HEIGHT - 82;
    let background = [244, 240, 231];
    let mut chart = Chart::new(WIDTH as usize, HEIGHT as usize, background);
    chart.text(LEFT, 22, title, [32, 41, 61]);
    let axis = [98, 107, 125];
    let grid = [222, 216, 202];
    for offset in 0..2 {
        chart.line((LEFT + offset, TOP), (LEFT + offset, BOTTOM), axis);
        chart.line((LEFT, BOTTOM + offset), (RIGHT, BOTTOM + offset), axis);
    }
    let finite_y = series
        .iter()
        .flat_map(|item| item.values.iter().flatten().copied())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if x_values.is_empty() || finite_y.is_empty() {
        return chart.save(path);
    }
    let x_min = x_values.iter().copied().reduce(f64::min).unwrap_or(0.0);
    let mut x_max = x_values.iter().copied().reduce(f64::max).unwrap_or(1.0);
    let raw_y_min = finite_y.iter().copied().reduce(f64::min).unwrap_or(0.0);
    let y_min = 0.0_f64.min(raw_y_min);
    let mut y_max = finite_y.iter().copied().reduce(f64::max).unwrap_or(1.0);
    if x_max <= x_min {
        x_max = x_min + 1.0;
    }
    if y_max <= y_min {
        y_max = y_min + 1.0;
    }
    let x_screen = |value: f64| {
        LEFT + ((value - x_min) / (x_max - x_min) * (RIGHT - LEFT) as f64).round() as i32
    };
    let y_screen = |value: f64| {
        BOTTOM - ((value - y_min) / (y_max - y_min) * (BOTTOM - TOP) as f64).round() as i32
    };
    for tick_index in 0..6 {
        let fraction = tick_index as f64 / 5.0;
        let tick_value = y_max - fraction * (y_max - y_min);
        let y = TOP + (fraction * (BOTTOM - TOP) as f64).round() as i32;
        chart.line((LEFT + 1, y), (RIGHT, y), grid);
        let label = chart_tick(tick_value);
        let label_width = label.chars().count() as i32 * 6;
        chart.text(LEFT - 10 - label_width, y - 3, &label, axis);
    }
    let mut x_ticks = x_values.to_vec();
    if x_ticks.len() > 20 {
        let step = x_ticks.len().div_ceil(10).max(1);
        x_ticks = x_ticks.into_iter().step_by(step).collect();
        if x_ticks.last().copied() != x_values.last().copied() {
            x_ticks.push(*x_values.last().unwrap());
        }
    }
    for x in x_ticks {
        let sx = x_screen(x);
        chart.line((sx, BOTTOM), (sx, BOTTOM + 5), axis);
        let label = chart_x_tick(x);
        let label_width = label.chars().count() as i32 * 6;
        chart.text(sx - label_width / 2, BOTTOM + 8, &label, axis);
    }
    chart.text(
        RIGHT - (x_label.chars().count() as i32 * 6),
        BOTTOM + 36,
        x_label,
        axis,
    );
    chart.text(8, TOP - 22, y_label, axis);

    for (series_index, item) in series.iter().enumerate() {
        let points = x_values
            .iter()
            .zip(&item.values)
            .filter_map(|(x, y)| {
                y.filter(|value| value.is_finite())
                    .map(|value| (x_screen(*x), y_screen(value)))
            })
            .collect::<Vec<_>>();
        for pair in points.windows(2) {
            chart.line(pair[0], pair[1], item.color);
            chart.line(
                (pair[0].0, pair[0].1 + 1),
                (pair[1].0, pair[1].1 + 1),
                item.color,
            );
            chart.line(
                (pair[0].0, pair[0].1 - 1),
                (pair[1].0, pair[1].1 - 1),
                item.color,
            );
        }
        for (cx, cy) in points {
            for dy in -3..=3 {
                for dx in -3..=3 {
                    if dx * dx + dy * dy <= 9 {
                        chart.pixel(cx + dx, cy + dy, item.color);
                    }
                }
            }
        }
        let legend_x = RIGHT - 180;
        let legend_y = TOP + 18 * series_index as i32;
        chart.line(
            (legend_x, legend_y + 5),
            (legend_x + 18, legend_y + 5),
            item.color,
        );
        chart.text(legend_x + 25, legend_y, &item.name, [39, 49, 71]);
    }
    Ok(chart.save(path)?)
}

fn write_source_snr_charts(detection: &DetectionResult, out_dir: &Path) -> Result<()> {
    let source_snr = detection
        .sources
        .iter()
        .filter_map(|source| source.flux_snr)
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    let filter_snr = detection
        .sources
        .iter()
        .filter_map(|source| source.filter_snr)
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    let rank_count = source_snr.len().max(filter_snr.len());
    let rank = (1..=rank_count)
        .map(|value| value as f64)
        .collect::<Vec<_>>();
    let descending_log = |values: &[f64]| {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted.reverse();
        sorted
            .into_iter()
            .map(|value| Some(value.log10()))
            .collect::<Vec<_>>()
    };
    let mut flux_rank = descending_log(&source_snr);
    let mut filter_rank = descending_log(&filter_snr);
    flux_rank.resize(rank_count, None);
    filter_rank.resize(rank_count, None);
    write_line_chart(
        &out_dir.join("source_snr_rank.png"),
        "SOURCE QUALITY AUDIT - SNR RANK",
        &rank,
        &[
            ChartSeries {
                name: "log10 flux SNR".into(),
                values: flux_rank,
                color: [79, 155, 131],
            },
            ChartSeries {
                name: "log10 filter SNR".into(),
                values: filter_rank,
                color: [215, 148, 50],
            },
        ],
        "log10 SNR",
        "source rank (descending)",
    )?;

    let bins = 32_usize;
    let mut centers = Vec::new();
    let mut counts = Vec::new();
    if !source_snr.is_empty() {
        let percentile = crate::statistics::percentile(&source_snr, 99.5)?;
        let upper = 10.0_f64.max(percentile);
        let width = upper / bins as f64;
        let mut histogram = vec![0_u64; bins];
        for value in &source_snr {
            if *value < 0.0 || *value > upper {
                continue;
            }
            let index = ((*value / width).floor() as usize).min(bins - 1);
            histogram[index] += 1;
        }
        centers = (0..bins)
            .map(|index| (index as f64 + 0.5) * width)
            .collect();
        counts = histogram
            .into_iter()
            .map(|count| Some(count as f64))
            .collect();
    }
    write_line_chart(
        &out_dir.join("source_flux_snr_distribution.png"),
        "SOURCE QUALITY AUDIT - FLUX SNR DISTRIBUTION",
        &centers,
        &[ChartSeries {
            name: "sources / bin".into(),
            values: counts,
            color: [79, 155, 131],
        }],
        "sources / bin",
        "flux SNR",
    )
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
    write_quality_flag_chart(&out_dir.join("source_quality_flags.png"), detection)?;
    write_feature_class_chart(&out_dir.join("source_feature_classes.png"), &feature_rows)?;
    write_source_spatial_chart(
        &out_dir.join("source_spatial_density.png"),
        &source_grid_rows,
    )?;
    write_source_snr_charts(detection, out_dir)?;
    Ok(())
}

fn source_signal_snr(source: &Detection) -> f64 {
    match source.flux_snr {
        Some(value) if value.is_finite() => value,
        Some(_) => f64::NEG_INFINITY,
        None if source.snr.is_finite() => source.snr,
        None => f64::NEG_INFINITY,
    }
}

fn source_cutout_group_matches(group: &str, source: &Detection) -> bool {
    let snr = source_signal_snr(source);
    let has = |flag: &str| source.flags.iter().any(|value| value == flag);
    let supplemental_only = !source.proposal_methods.is_empty()
        && !source
            .proposal_methods
            .iter()
            .any(|method| method == "gaussian");
    match group {
        "quality_near_threshold" => source.quality_passed && (5.0..6.0).contains(&snr),
        "quality_mid_snr" => source.quality_passed && (6.0..20.0).contains(&snr),
        "quality_bright" => source.quality_passed && snr >= 20.0,
        "quality_extra_only" => source.quality_passed && supplemental_only,
        "rejected_extra_only" => !source.quality_passed && supplemental_only,
        "quality_starlet_added" => {
            source.quality_passed
                && supplemental_only
                && source
                    .proposal_methods
                    .iter()
                    .any(|method| method.starts_with("starlet_"))
        }
        "rejected_starlet_added" => {
            !source.quality_passed
                && supplemental_only
                && source
                    .proposal_methods
                    .iter()
                    .any(|method| method.starts_with("starlet_"))
        }
        "quality_deblended" => {
            source.quality_passed && source.deblend_delta_bic.is_some() && !has("UNRESOLVED_BLEND")
        }
        "rejected_deblended" => {
            !source.quality_passed && source.deblend_delta_bic.is_some() && !has("UNRESOLVED_BLEND")
        }
        "rejected_low_flux_snr" => has("LOW_FLUX_SNR"),
        "rejected_negative_overflow" => has("NEGATIVE_OVERFLOW"),
        "rejected_spike" => has("SPIKE"),
        "rejected_masked" => has("MASKED"),
        "rejected_line_artifact" => has("LINE_ARTIFACT"),
        "rejected_edge" => has("EDGE"),
        value => value.strip_prefix("feature_").is_some_and(|feature_class| {
            source_classification::classify_source_feature(source) == feature_class
        }),
    }
}

fn cutout_outline(chart: &mut Chart, center_x: f64, center_y: f64, radius: i32, color: [u8; 3]) {
    let steps = (radius.max(4) * 8) as usize;
    for step in 0..steps {
        let angle = std::f64::consts::TAU * step as f64 / steps as f64;
        let x = (center_x + f64::from(radius) * angle.cos()).round() as i32;
        let y = (center_y + f64::from(radius) * angle.sin()).round() as i32;
        chart.pixel(x, y, color);
    }
    let reach = (f64::from(radius) * 1.4).round() as i32;
    let x = center_x.round() as i32;
    let y = center_y.round() as i32;
    chart.line((x - reach, y), (x + reach, y), color);
    chart.line((x, y - reach), (x, y + reach), color);
}

fn render_source_cutout(
    frame: &FitsFrame,
    source: &Detection,
    median_value: f64,
    radius_px: usize,
    scale: usize,
) -> Result<Chart> {
    let side = radius_px
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or("source cutout dimensions overflow")?;
    let image_width = side
        .checked_mul(scale)
        .ok_or("source cutout dimensions overflow")?;
    let image_height = side
        .checked_mul(scale)
        .ok_or("source cutout dimensions overflow")?;
    let center = match (source.peak_x, source.peak_y) {
        (Some(x), Some(y)) if x.is_finite() && y.is_finite() => [x, y],
        _ => [source.x, source.y],
    };
    let center_x = center[0].round_ties_even() as isize;
    let center_y = center[1].round_ties_even() as isize;
    let x_start = center_x - radius_px as isize;
    let y_start = center_y - radius_px as isize;
    let mut crop = Vec::with_capacity(side * side);
    for y in 0..side {
        for x in 0..side {
            let frame_x = x_start + x as isize;
            let frame_y = y_start + y as isize;
            let value = if frame_x >= 0
                && frame_y >= 0
                && frame_x < frame.width as isize
                && frame_y < frame.height as isize
            {
                frame
                    .data
                    .value(frame_y as usize * frame.width + frame_x as usize)
            } else {
                median_value
            };
            crop.push(value);
        }
    }
    let finite = crop
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let (low, mut high) = if finite.is_empty() {
        (median_value, median_value + 1.0)
    } else {
        (
            crate::statistics::percentile(&finite, 5.0)?,
            crate::statistics::percentile(&finite, 99.7)?,
        )
    };
    if high <= low {
        high = low + 1.0;
    }
    let mut chart = Chart::new(image_width, image_height, [0, 0, 0]);
    for y in 0..side {
        for x in 0..side {
            let value = crop[y * side + x];
            let normalized = if value.is_finite() {
                ((value - low) / (high - low)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let gray = (normalized * 255.0).round() as u8;
            chart.rect(
                (x * scale) as i32,
                (y * scale) as i32,
                ((x + 1) * scale - 1) as i32,
                ((y + 1) * scale - 1) as i32,
                [gray; 3],
            );
        }
    }
    let marker_x = (center[0] - x_start as f64) * scale as f64;
    let marker_y = (center[1] - y_start as f64) * scale as f64;
    let marker_radius = (scale * 2).max(4) as i32;
    let color = if source.quality_passed {
        [79, 214, 170]
    } else {
        [240, 179, 75]
    };
    cutout_outline(&mut chart, marker_x, marker_y, marker_radius, color);
    Ok(chart)
}

/// Write deterministic frame-dependent cutouts and their contact sheet.
/// The samples remain audit illustrations and do not label physical truth.
pub fn write_detection_source_cutouts(
    frame: &FitsFrame,
    detection: &DetectionResult,
    out_dir: &Path,
    per_group: usize,
    radius_px: usize,
    scale: usize,
) -> Result<()> {
    if per_group == 0 || radius_px < 3 || scale == 0 {
        return Err("per_group and scale must be positive; radius_px must be at least 3".into());
    }
    if frame.width != detection.image_shape[1]
        || frame.height != detection.image_shape[0]
        || frame.width.checked_mul(frame.height) != Some(frame.data.len())
    {
        return Err("source cutout frame and detection shapes do not match".into());
    }
    let cutout_dir = out_dir.join("source_cutouts");
    std::fs::create_dir_all(&cutout_dir).map_err(|error| error.to_string())?;
    let finite_frame = (0..frame.data.len())
        .map(|index| frame.data.value(index))
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let median_value = if finite_frame.is_empty() {
        0.0
    } else {
        crate::statistics::median(&finite_frame)?
    };
    let groups = [
        ("quality_near_threshold", "QUALITY 5<=flux SNR<6"),
        ("quality_mid_snr", "QUALITY 6<=flux SNR<20"),
        ("quality_bright", "QUALITY flux SNR>=20"),
        (
            "quality_extra_only",
            "QUALITY proposed only by supplemental algorithms",
        ),
        (
            "rejected_extra_only",
            "REJECTED proposed only by supplemental algorithms",
        ),
        (
            "quality_starlet_added",
            "QUALITY non-Gaussian candidate hit by Starlet",
        ),
        (
            "rejected_starlet_added",
            "REJECTED non-Gaussian candidate hit by Starlet",
        ),
        ("quality_deblended", "QUALITY pair-PSF deblend supported"),
        ("rejected_deblended", "REJECTED despite pair-PSF support"),
        ("rejected_low_flux_snr", "REJECTED LOW_FLUX_SNR"),
        ("rejected_negative_overflow", "REJECTED NEGATIVE_OVERFLOW"),
        ("rejected_spike", "REJECTED SPIKE"),
        ("rejected_masked", "REJECTED MASKED"),
        ("rejected_line_artifact", "REJECTED LINE_ARTIFACT"),
        ("rejected_edge", "REJECTED EDGE"),
    ];
    let mut group_names = groups
        .into_iter()
        .map(|(name, label)| (name.to_owned(), label.to_owned()))
        .collect::<Vec<_>>();
    group_names.extend(
        source_classification::SOURCE_FEATURE_CLASSES
            .iter()
            .map(|(class, label)| (format!("feature_{class}"), format!("FEATURE {label}"))),
    );

    let fields = [
        "group",
        "group_label",
        "sample_index",
        "detection_id",
        "centroid_x",
        "centroid_y",
        "peak_x",
        "peak_y",
        "centroid_shift_px",
        "flux_snr",
        "proposal_snr",
        "proposal_methods",
        "nearest_gaussian_px",
        "deblend_delta_bic",
        "deblend_component_snr",
        "fwhm",
        "ellipticity",
        "sharpness",
        "quality_passed",
        "flags",
        "file",
    ];
    let mut manifest = Vec::<Value>::new();
    let mut rendered = Vec::<(String, String, String, i64, Chart, f64)>::new();
    for (group, label) in group_names {
        let mut sources = detection
            .sources
            .iter()
            .filter(|source| source_cutout_group_matches(&group, source))
            .collect::<Vec<_>>();
        sources.sort_by(|left, right| {
            source_signal_snr(left)
                .total_cmp(&source_signal_snr(right))
                .then_with(|| left.detection_id.cmp(&right.detection_id))
        });
        if sources.len() > per_group {
            let count = sources.len();
            sources = (0..per_group)
                .map(|index| {
                    let position = if per_group == 1 {
                        0
                    } else {
                        (index as f64 * (count - 1) as f64 / (per_group - 1) as f64)
                            .round_ties_even() as usize
                    };
                    sources[position]
                })
                .collect();
        }
        for (index, source) in sources.into_iter().enumerate() {
            let image = render_source_cutout(frame, source, median_value, radius_px, scale)?;
            let file_name = format!("{group}_{:02}_id{:05}.png", index + 1, source.detection_id);
            image.save(&cutout_dir.join(&file_name))?;
            let peak = match (source.peak_x, source.peak_y) {
                (Some(x), Some(y)) if x.is_finite() && y.is_finite() => [x, y],
                _ => [source.x, source.y],
            };
            let method_list = source.proposal_methods.join("|");
            let flag_list = source.flags.join("|");
            manifest.push(json!({
                "group":group,
                "group_label":label,
                "sample_index":index + 1,
                "detection_id":source.detection_id,
                "centroid_x":source.x,
                "centroid_y":source.y,
                "peak_x":peak[0],
                "peak_y":peak[1],
                "centroid_shift_px":source.centroid_shift_px,
                "flux_snr":source.flux_snr,
                "proposal_snr":source.proposal_snr,
                "proposal_methods":method_list,
                "nearest_gaussian_px":source.nearest_gaussian_px,
                "deblend_delta_bic":source.deblend_delta_bic,
                "deblend_component_snr":source.deblend_component_snr,
                "fwhm":source.fwhm,
                "ellipticity":source.ellipticity,
                "sharpness":source.sharpness,
                "quality_passed":source.quality_passed,
                "flags":flag_list,
                "file":format!("source_cutouts/{file_name}"),
            }));
            rendered.push((
                group.clone(),
                label.clone(),
                file_name,
                source.detection_id,
                image,
                source_signal_snr(source),
            ));
        }
    }
    write_rows(
        &out_dir.join("source_cutout_manifest.csv"),
        &fields,
        &manifest,
    )?;

    let tile_width = (2 * radius_px + 1)
        .checked_mul(scale)
        .ok_or("source contact sheet dimensions overflow")?;
    let tile_height = tile_width + 30;
    let columns = 4usize;
    let row_count = rendered.len().div_ceil(columns).max(1);
    let sheet_width = columns
        .checked_mul(tile_width)
        .ok_or("source contact sheet dimensions overflow")?;
    let sheet_height = 32usize
        .checked_add(
            row_count
                .checked_mul(tile_height)
                .ok_or("source contact sheet dimensions overflow")?,
        )
        .ok_or("source contact sheet dimensions overflow")?;
    let mut sheet = Chart::new(sheet_width, sheet_height, [24, 32, 51]);
    sheet.text(
        6,
        8,
        "SOURCE CUTOUT AUDIT - QUALITY CLASS IS NOT TRUTH",
        [246, 241, 231],
    );
    for (index, (group, _label, _file_name, detection_id, image, snr)) in
        rendered.iter().enumerate()
    {
        let x0 = (index % columns * tile_width) as i32;
        let y0 = (32 + index / columns * tile_height) as i32;
        for y in 0..image.height {
            for x in 0..image.width {
                let source_offset = (y * image.width + x) * 3;
                let color = [
                    image.pixels[source_offset],
                    image.pixels[source_offset + 1],
                    image.pixels[source_offset + 2],
                ];
                sheet.pixel(x0 + x as i32, y0 + y as i32, color);
            }
        }
        let snr_label = if snr.is_finite() {
            format!("{snr:.1}")
        } else {
            "NA".into()
        };
        let group_label = if group.chars().count() > 25 {
            format!("{}~", group.chars().take(24).collect::<String>())
        } else {
            group.clone()
        };
        sheet.text(
            x0 + 3,
            y0 + tile_width as i32 + 2,
            &group_label,
            [188, 224, 209],
        );
        sheet.text(
            x0 + 3,
            y0 + tile_width as i32 + 14,
            &format!("ID {detection_id} SNR {snr_label}"),
            [188, 224, 209],
        );
    }
    sheet.save(&out_dir.join("source_cutout_contact_sheet.png"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::DetectionResult;
    use crate::models::PixelData;
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
        for (name, expected_width, expected_height) in [
            ("source_quality_flags.png", 1200, 680),
            ("source_feature_classes.png", 1280, 720),
            ("source_spatial_density.png", 1200, 680),
            ("source_snr_rank.png", 1200, 680),
            ("source_flux_snr_distribution.png", 1200, 680),
        ] {
            let decoder = png::Decoder::new(std::io::BufReader::new(
                File::open(root.join(name)).unwrap(),
            ));
            let mut reader = decoder.read_info().unwrap();
            let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut pixels).unwrap();
            assert_eq!(info.width, expected_width, "width mismatch in {name}");
            assert_eq!(info.height, expected_height, "height mismatch in {name}");
            assert_eq!(
                info.color_type,
                png::ColorType::Rgb,
                "color type mismatch in {name}"
            );
            pixels.truncate(info.buffer_size());
            assert!(
                pixels.chunks_exact(3).any(|pixel| pixel == [244, 240, 231]),
                "missing background in {name}"
            );
            assert!(
                pixels
                    .chunks_exact(3)
                    .any(|pixel| pixel == [79, 155, 131] || pixel == [215, 148, 50]),
                "missing data colors in {name}"
            );
        }
        if std::env::var_os("RST19_KEEP_SOURCE_ARTIFACTS").is_none() {
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn writes_deterministic_frame_source_cutouts_and_manifest() {
        let root = temp_dir();
        let result = fixture_result();
        let frame = FitsFrame {
            path: root.join("fixture.fits"),
            header: Map::new(),
            data: PixelData::F32(
                (0..128 * 128)
                    .map(|index| ((index % 97) as f32) - 48.0)
                    .collect(),
            ),
            auxiliary: None,
            data_offset: 0,
            width: 128,
            height: 128,
        };
        write_detection_source_cutouts(&frame, &result, &root, 8, 20, 4).unwrap();
        let manifest = std::fs::read(root.join("source_cutout_manifest.csv")).unwrap();
        assert!(manifest.starts_with(&[0xEF, 0xBB, 0xBF]));
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(manifest.as_slice());
        let rows = reader
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert!(!rows.is_empty());
        assert_eq!(
            std::fs::read_dir(root.join("source_cutouts"))
                .unwrap()
                .count(),
            rows.len()
        );
        let decoder = png::Decoder::new(std::io::BufReader::new(
            File::open(root.join(rows[0].get(20).unwrap())).unwrap(),
        ));
        let png_reader = decoder.read_info().unwrap();
        let info = png_reader.info();
        assert_eq!((info.width, info.height), (164, 164));
        assert_eq!(info.color_type, png::ColorType::Rgb);
        let contact = png::Decoder::new(std::io::BufReader::new(
            File::open(root.join("source_cutout_contact_sheet.png")).unwrap(),
        ));
        let contact_reader = contact.read_info().unwrap();
        assert_eq!(contact_reader.info().width, 656);
        if std::env::var_os("RST19_KEEP_SOURCE_ARTIFACTS").is_none() {
            let _ = std::fs::remove_dir_all(root);
        }
    }
}
