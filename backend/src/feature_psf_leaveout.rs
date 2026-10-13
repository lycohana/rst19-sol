//! Per-source leave-one-out empirical PSF shape audit.
//!
//! These are detector-level shape diagnostics only. The audit neither changes
//! quality decisions nor establishes physical source identity.
use crate::{fits, measurement::Detection, spatial::KdTree, Result};
use csv::{Terminator, WriterBuilder};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const FEATURE_CLASSES: [(&str, &str); 9] = [
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
const FEATURE_FLAGS: [(&str, &[&str]); 7] = [
    (
        "range_anomaly",
        &["SATURATED", "NEGATIVE_OVERFLOW", "CODE_PATTERN"],
    ),
    ("linear_artifact", &["LINE_ARTIFACT"]),
    ("masked_or_edge", &["MASKED", "PARTIAL_MASKED", "EDGE"]),
    ("crowded_blend", &["UNRESOLVED_BLEND"]),
    (
        "spike_or_support",
        &[
            "SPIKE",
            "NARROW",
            "SMALL_FOOTPRINT",
            "INSUFFICIENT_PSF_SUPPORT",
        ],
    ),
    (
        "weak_or_background",
        &["LOW_FLUX_SNR", "NON_POSITIVE_FLUX", "BACKGROUND_UNCERTAIN"],
    ),
    (
        "shape_outlier",
        &["NO_SHAPE", "BROAD", "ELONGATED", "DIFFUSE"],
    ),
];
const ELIGIBILITY_EXCLUDED_FLAGS: [&str; 6] = [
    "EDGE",
    "MASKED",
    "PARTIAL_MASKED",
    "SATURATED",
    "LINE_ARTIFACT",
    "NEGATIVE_OVERFLOW",
];
const CONCLUSION: &str = "留一法只把被评价源从经验 PSF 模板候选池排除，并继续用全幅候选表做邻峰隔离。如果 compact_quality 的留一相关度仍高而线状、拥挤或尖峰类仍低，类别差异就不主要是模板自相似；但这仍只是 detector-level 形状证据，不能替代星表/WCS 一对一匹配、独立注入真值或完整噪声似然。";

#[derive(Clone, Debug, Serialize)]
pub struct PsfSummary {
    model: &'static str,
    support_radius: usize,
    source_count: usize,
    median_fwhm_px: Option<f64>,
    median_ellipticity: Option<f64>,
    kernel_shape: [usize; 2],
    kernel_sum: f64,
}

#[derive(Clone, Debug)]
struct EmpiricalPsf {
    kernel: Vec<f32>,
    support_radius: usize,
    source_count: usize,
    median_fwhm_px: Option<f64>,
    median_ellipticity: Option<f64>,
}

impl EmpiricalPsf {
    fn summary(&self) -> PsfSummary {
        let kernel_sum = self.kernel.iter().copied().sum::<f32>() as f64;
        let side = 2 * self.support_radius + 1;
        PsfSummary {
            model: "empirical",
            support_radius: self.support_radius,
            source_count: self.source_count,
            median_fwhm_px: self.median_fwhm_px,
            median_ellipticity: self.median_ellipticity,
            kernel_shape: [side, side],
            kernel_sum,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaturePsfLeaveOutSourceRow {
    pub detection_id: i64,
    pub feature_class: String,
    pub feature_class_label: String,
    pub quality_passed: bool,
    pub in_sample_correlation: Option<f64>,
    pub leaveout_correlation: Option<f64>,
    pub correlation_delta_leaveout_minus_in_sample: Option<f64>,
    pub in_sample_residual: Option<f64>,
    pub leaveout_residual: Option<f64>,
    pub residual_delta_leaveout_minus_in_sample: Option<f64>,
    pub in_sample_central_energy_fraction: Option<f64>,
    pub leaveout_central_energy_fraction: Option<f64>,
    pub leaveout_amplitude_adu: Option<f64>,
    pub leaveout_template_source_count: usize,
    pub leaveout_template_median_fwhm_px: Option<f64>,
    pub leaveout_template_median_ellipticity: Option<f64>,
    pub in_sample_valid: bool,
    pub leaveout_valid: bool,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaturePsfLeaveOutClassRow {
    pub feature_class: String,
    pub feature_class_label: String,
    pub global_psf_source_count: usize,
    pub sample_count: usize,
    pub in_sample_valid_count: usize,
    pub leaveout_valid_count: usize,
    pub paired_count: usize,
    pub leaveout_template_built_count: usize,
    pub leaveout_template_failed_count: usize,
    pub leaveout_template_source_count_median: Option<f64>,
    pub correlation_threshold: f64,
    pub in_sample_correlation_median: Option<f64>,
    pub leaveout_correlation_p10: Option<f64>,
    pub leaveout_correlation_median: Option<f64>,
    pub leaveout_correlation_p90: Option<f64>,
    pub leaveout_correlation_ge_threshold_count: usize,
    pub leaveout_correlation_ge_threshold_fraction: Option<f64>,
    pub in_sample_residual_median: Option<f64>,
    pub leaveout_residual_p10: Option<f64>,
    pub leaveout_residual_median: Option<f64>,
    pub leaveout_residual_p90: Option<f64>,
    pub paired_correlation_delta_median: Option<f64>,
    pub paired_residual_delta_median: Option<f64>,
    pub sample_detection_ids: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeaturePsfLeaveOutResult {
    pub frame_path: String,
    pub catalog_path: Option<String>,
    pub image_shape: [usize; 2],
    pub per_class: usize,
    pub support_radius: usize,
    pub max_template_sources: usize,
    pub correlation_threshold: f64,
    pub global_psf: Option<PsfSummary>,
    pub class_rows: Vec<FeaturePsfLeaveOutClassRow>,
    pub source_rows: Vec<FeaturePsfLeaveOutSourceRow>,
    pub parameters: Value,
    pub conclusion: String,
}

impl FeaturePsfLeaveOutResult {
    pub fn as_dict(&self) -> Value {
        serde_json::to_value(self).expect("finite feature PSF leave-out result")
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub per_class: usize,
    pub support_radius: usize,
    pub max_template_sources: usize,
    pub correlation_threshold: f64,
    pub out_dir: PathBuf,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            per_class: 16,
            support_radius: 7,
            max_template_sources: 20,
            correlation_threshold: 0.8,
            out_dir: PathBuf::from("tmp/feature-psf-leaveout-audit"),
        }
    }
}

fn python_round(value: f64) -> i64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor as i64
    } else if fraction > 0.5 {
        floor as i64 + 1
    } else {
        let lower = floor as i64;
        if lower.rem_euclid(2) == 0 {
            lower
        } else {
            lower + 1
        }
    }
}

fn interpolate_shift(input: &[f64], side: usize, shift_y: f64, shift_x: f64) -> Vec<f64> {
    let mut output = vec![0.0; side * side];
    for y in 0..side {
        for x in 0..side {
            let source_y = y as f64 - shift_y;
            let source_x = x as f64 - shift_x;
            if source_y < 0.0
                || source_x < 0.0
                || source_y > (side - 1) as f64
                || source_x > (side - 1) as f64
            {
                continue;
            }
            let y0 = source_y.floor() as usize;
            let x0 = source_x.floor() as usize;
            let y1 = (y0 + 1).min(side - 1);
            let x1 = (x0 + 1).min(side - 1);
            let dy = source_y - y0 as f64;
            let dx = source_x - x0 as f64;
            let top = input[y0 * side + x0] * (1.0 - dx) + input[y0 * side + x1] * dx;
            let bottom = input[y1 * side + x0] * (1.0 - dx) + input[y1 * side + x1] * dx;
            output[y * side + x] = top * (1.0 - dy) + bottom * dy;
        }
    }
    output
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let upper = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Some((sorted[upper - 1] + sorted[upper]) * 0.5)
    } else {
        Some(sorted[upper])
    }
}

fn percentile(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * percentile / 100.0;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    Some(sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64))
}

fn classify(source: &Detection) -> &'static str {
    for (feature_class, flags) in FEATURE_FLAGS {
        if source
            .flags
            .iter()
            .any(|flag| flags.contains(&flag.as_str()))
        {
            return feature_class;
        }
    }
    if source.quality_passed {
        "compact_quality"
    } else {
        "other_rejected"
    }
}

fn signal_snr(source: &Detection) -> f64 {
    let value = source.flux_snr.unwrap_or(source.snr);
    if value.is_finite() {
        value
    } else {
        f64::NEG_INFINITY
    }
}

fn select_samples(sources: &[Detection], per_class: usize) -> HashMap<&'static str, Vec<usize>> {
    let mut selected = HashMap::new();
    for (feature_class, _) in FEATURE_CLASSES {
        let mut members = sources
            .iter()
            .enumerate()
            .filter_map(|(index, source)| (classify(source) == feature_class).then_some(index))
            .collect::<Vec<_>>();
        members.sort_by(|&left, &right| {
            signal_snr(&sources[left])
                .total_cmp(&signal_snr(&sources[right]))
                .then_with(|| sources[left].detection_id.cmp(&sources[right].detection_id))
        });
        if members.len() > per_class {
            let count = members.len();
            let mut indices = Vec::with_capacity(per_class);
            for sample in 0..per_class {
                let position = (count - 1) as f64 * sample as f64 / (per_class - 1).max(1) as f64;
                let lower = position.floor() as usize;
                let fraction = position - lower as f64;
                let rounded = if fraction < 0.5 {
                    lower
                } else if fraction > 0.5 || lower % 2 == 1 {
                    lower + 1
                } else {
                    lower
                };
                if indices.last() != Some(&rounded) {
                    indices.push(rounded);
                }
            }
            members = indices.into_iter().map(|index| members[index]).collect();
        }
        selected.insert(feature_class, members);
    }
    selected
}

fn extract_stamp(
    image: &[f64],
    height: usize,
    width: usize,
    source: &Detection,
    radius: usize,
) -> Option<(Vec<f64>, i64, i64)> {
    let center_x = python_round(source.x);
    let center_y = python_round(source.y);
    let x0 = center_x - radius as i64;
    let y0 = center_y - radius as i64;
    let side = 2 * radius + 1;
    if x0 < 0 || y0 < 0 || x0 + side as i64 > width as i64 || y0 + side as i64 > height as i64 {
        return None;
    }
    let mut stamp = Vec::with_capacity(side * side);
    for y in y0 as usize..y0 as usize + side {
        for x in x0 as usize..x0 as usize + side {
            stamp.push(image[y * width + x]);
        }
    }
    Some((stamp, center_x, center_y))
}

#[allow(clippy::too_many_arguments)]
fn estimate_psf(
    image: &[f64],
    height: usize,
    width: usize,
    sources: &[Detection],
    spatial: &KdTree,
    aux_mask: &[bool],
    radius: usize,
    max_sources: usize,
    excluded_ids: &HashSet<i64>,
) -> Option<EmpiricalPsf> {
    let margin = radius + 1;
    let mut eligible = sources
        .iter()
        .enumerate()
        .filter_map(|(index, source)| {
            (source.x.is_finite()
                && source.y.is_finite()
                && source.x >= margin as f64
                && source.x < width.saturating_sub(margin) as f64
                && source.y >= margin as f64
                && source.y < height.saturating_sub(margin) as f64
                && !excluded_ids.contains(&source.detection_id)
                && source.quality_passed
                && !source
                    .flags
                    .iter()
                    .any(|flag| ELIGIBILITY_EXCLUDED_FLAGS.contains(&flag.as_str()))
                && source.ellipticity.is_some_and(|value| value <= 0.45)
                && source
                    .fwhm
                    .is_some_and(|value| (1.0..=8.0).contains(&value)))
            .then_some((index, source))
        })
        .collect::<Vec<_>>();
    eligible.sort_by(|(_, left), (_, right)| signal_snr(right).total_cmp(&signal_snr(left)));
    let mut stamps: Vec<Vec<f32>> = Vec::new();
    let mut fwhm_values = Vec::new();
    let mut ellipticity_values = Vec::new();
    let side = 2 * radius + 1;
    let isolation_radius = (3 * radius).max(18) as f64;
    for (source_index, source) in eligible {
        if stamps.len() >= max_sources {
            break;
        }
        let nearest_other = spatial
            .nearest_excluding([source.x, source.y], source_index)
            .map_or(f64::INFINITY, |(_, distance)| distance);
        if nearest_other < isolation_radius {
            continue;
        }
        let Some((mut stamp, center_x, center_y)) =
            extract_stamp(image, height, width, source, radius)
        else {
            continue;
        };
        if stamp.iter().any(|value| !value.is_finite()) {
            continue;
        }
        let x0 = (center_x - radius as i64) as usize;
        let y0 = (center_y - radius as i64) as usize;
        if (y0..y0 + side).any(|y| (x0..x0 + side).any(|x| aux_mask[y * width + x])) {
            continue;
        }
        let background = if source.background.is_finite() {
            source.background
        } else {
            median(&stamp)?
        };
        for value in &mut stamp {
            *value = (*value - background).max(0.0);
        }
        let shifted = interpolate_shift(
            &stamp,
            side,
            center_y as f64 - source.y,
            center_x as f64 - source.x,
        );
        let peak = shifted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !peak.is_finite() || peak <= 0.0 {
            continue;
        }
        stamps.push(
            shifted
                .into_iter()
                .map(|value| (value / peak) as f32)
                .collect(),
        );
        fwhm_values.push(source.fwhm?);
        if let Some(value) = source.ellipticity.filter(|value| value.is_finite()) {
            ellipticity_values.push(value);
        }
    }
    if stamps.len() < 3 {
        return None;
    }
    let mut kernel = vec![0.0f32; side * side];
    for pixel in 0..kernel.len() {
        let mut values = stamps.iter().map(|stamp| stamp[pixel]).collect::<Vec<_>>();
        values.sort_by(f32::total_cmp);
        let upper = values.len() / 2;
        kernel[pixel] = if values.len() % 2 == 0 {
            ((values[upper - 1] as f64 + values[upper] as f64) * 0.5) as f32
        } else {
            values[upper]
        };
        kernel[pixel] = kernel[pixel].max(0.0);
    }
    let peak = kernel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !peak.is_finite() || peak <= 0.0 {
        return None;
    }
    for value in &mut kernel {
        *value /= peak;
    }
    Some(EmpiricalPsf {
        kernel,
        support_radius: radius,
        source_count: stamps.len(),
        median_fwhm_px: median(&fwhm_values),
        median_ellipticity: median(&ellipticity_values),
    })
}

fn psf_similarity_metrics(
    image: &[f64],
    height: usize,
    width: usize,
    source: &Detection,
    psf: &EmpiricalPsf,
    aux_mask: &[bool],
) -> Option<[f64; 4]> {
    let radius = psf.support_radius;
    let side = 2 * radius + 1;
    let (stamp, center_x, center_y) = extract_stamp(image, height, width, source, radius)?;
    let x0 = (center_x - radius as i64) as usize;
    let y0 = (center_y - radius as i64) as usize;
    let valid = stamp
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value.is_finite() && !aux_mask[(y0 + index / side) * width + x0 + index % side]
        })
        .collect::<Vec<_>>();
    if valid.iter().filter(|&&value| value).count() < (0.8 * stamp.len() as f64).ceil() as usize {
        return None;
    }
    let mut finite_samples = stamp
        .iter()
        .zip(&valid)
        .filter_map(|(&value, &is_valid)| is_valid.then_some(value))
        .collect::<Vec<_>>();
    let fallback_background = median(&finite_samples)?;
    let background = if source.background.is_finite() {
        source.background
    } else {
        fallback_background
    };
    let observed = stamp
        .iter()
        .zip(&valid)
        .map(|(&value, &is_valid)| {
            if is_valid {
                (value - background).max(0.0)
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let shifted = interpolate_shift(
        &observed,
        side,
        center_y as f64 - source.y,
        center_x as f64 - source.x,
    );
    let template = psf
        .kernel
        .iter()
        .enumerate()
        .map(|(index, &value)| if valid[index] { value as f64 } else { 0.0 })
        .collect::<Vec<_>>();
    let data_norm = shifted
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let template_norm = template
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let denominator = data_norm * template_norm;
    let epsilon = f64::EPSILON;
    if denominator <= epsilon {
        return None;
    }
    let dot = shifted
        .iter()
        .zip(&template)
        .map(|(data, model)| data * model)
        .sum::<f64>();
    let template_square_sum = template.iter().map(|value| value * value).sum::<f64>();
    let amplitude = (dot / template_square_sum.max(epsilon)).max(0.0);
    let residual = shifted
        .iter()
        .zip(&template)
        .map(|(data, model)| (data - amplitude * model).powi(2))
        .sum::<f64>()
        .sqrt()
        / data_norm.max(epsilon);
    let positive_sum = shifted.iter().sum::<f64>();
    if positive_sum <= epsilon {
        return None;
    }
    let central_sum = (radius - 1..=radius + 1)
        .flat_map(|y| (radius - 1..=radius + 1).map(move |x| y * side + x))
        .map(|index| shifted[index])
        .sum::<f64>();
    finite_samples.clear();
    Some([
        dot / denominator,
        residual,
        central_sum / positive_sum,
        amplitude,
    ])
}

fn metric(metrics: Option<[f64; 4]>, index: usize) -> Option<f64> {
    metrics.and_then(|values| values[index].is_finite().then_some(values[index]))
}

#[allow(clippy::too_many_arguments)]
fn build_result(
    image: &[f64],
    height: usize,
    width: usize,
    frame_path: String,
    catalog_path: Option<String>,
    sources: &[Detection],
    options: &Options,
    mut progress: Option<&mut dyn FnMut(usize, usize)>,
) -> Result<FeaturePsfLeaveOutResult> {
    if options.per_class < 1 || options.support_radius < 3 || options.max_template_sources < 3 {
        return Err("per_class must be positive, support_radius at least 3, and max_template_sources at least 3".into());
    }
    if !options.correlation_threshold.is_finite()
        || !(0.0..=1.0).contains(&options.correlation_threshold)
        || options.correlation_threshold == 0.0
    {
        return Err("correlation_threshold must be between 0 and 1".into());
    }
    if height == 0 || width == 0 || height.checked_mul(width) != Some(image.len()) {
        return Err("expected a non-empty 2-D image".into());
    }
    if sources.is_empty() {
        return Err("sources cannot be empty".into());
    }
    let aux_mask = fits::auxiliary_mask(height, width)?;
    let positions = sources
        .iter()
        .map(Detection::measurement_position)
        .collect::<Vec<_>>();
    let spatial = KdTree::new(&positions)?;
    let empty = HashSet::new();
    let global_psf = estimate_psf(
        image,
        height,
        width,
        sources,
        &spatial,
        &aux_mask,
        options.support_radius,
        options.max_template_sources,
        &empty,
    );
    let selected = select_samples(sources, options.per_class);
    let selected_items = FEATURE_CLASSES
        .iter()
        .flat_map(|&(feature_class, feature_label)| {
            selected[feature_class]
                .iter()
                .map(move |&index| (feature_class, feature_label, index))
        })
        .collect::<Vec<_>>();
    let total = selected_items.len();
    let mut source_rows = Vec::with_capacity(total);
    for (index, &(feature_class, feature_label, source_index)) in selected_items.iter().enumerate()
    {
        let source = &sources[source_index];
        let in_metrics = global_psf
            .as_ref()
            .and_then(|psf| psf_similarity_metrics(image, height, width, source, psf, &aux_mask));
        let excluded = HashSet::from([source.detection_id]);
        let leaveout_psf = estimate_psf(
            image,
            height,
            width,
            sources,
            &spatial,
            &aux_mask,
            options.support_radius,
            options.max_template_sources,
            &excluded,
        );
        let leaveout_metrics = leaveout_psf
            .as_ref()
            .and_then(|psf| psf_similarity_metrics(image, height, width, source, psf, &aux_mask));
        let in_corr = metric(in_metrics, 0);
        let leaveout_corr = metric(leaveout_metrics, 0);
        let in_residual = metric(in_metrics, 1);
        let leaveout_residual = metric(leaveout_metrics, 1);
        let template_source_count = leaveout_psf.as_ref().map_or(0, |psf| psf.source_count);
        let note = if leaveout_psf.is_none() {
            "留一模板不足三个合格隔离源，未强行填充 PSF"
        } else if leaveout_metrics.is_none() {
            "留一模板已建立，但该源裁剪无有效形状指标"
        } else {
            "留一经验 PSF 形状指标；不等于恒星身份"
        };
        source_rows.push(FeaturePsfLeaveOutSourceRow {
            detection_id: source.detection_id,
            feature_class: feature_class.to_owned(),
            feature_class_label: feature_label.to_owned(),
            quality_passed: source.quality_passed,
            in_sample_correlation: in_corr,
            leaveout_correlation: leaveout_corr,
            correlation_delta_leaveout_minus_in_sample: leaveout_corr
                .zip(in_corr)
                .map(|(a, b)| a - b),
            in_sample_residual: in_residual,
            leaveout_residual,
            residual_delta_leaveout_minus_in_sample: leaveout_residual
                .zip(in_residual)
                .map(|(a, b)| a - b),
            in_sample_central_energy_fraction: metric(in_metrics, 2),
            leaveout_central_energy_fraction: metric(leaveout_metrics, 2),
            leaveout_amplitude_adu: metric(leaveout_metrics, 3),
            leaveout_template_source_count: template_source_count,
            leaveout_template_median_fwhm_px: leaveout_psf
                .as_ref()
                .and_then(|psf| psf.median_fwhm_px),
            leaveout_template_median_ellipticity: leaveout_psf
                .as_ref()
                .and_then(|psf| psf.median_ellipticity),
            in_sample_valid: in_metrics.is_some(),
            leaveout_valid: leaveout_metrics.is_some(),
            note: note.to_owned(),
        });
        if let Some(callback) = progress.as_deref_mut() {
            callback(index + 1, total);
        }
    }
    let global_count = global_psf.as_ref().map_or(0, |psf| psf.source_count);
    let mut class_rows = Vec::with_capacity(FEATURE_CLASSES.len());
    for (feature_class, feature_label) in FEATURE_CLASSES {
        let members = source_rows
            .iter()
            .filter(|row| row.feature_class == feature_class)
            .collect::<Vec<_>>();
        let in_corr = members
            .iter()
            .filter_map(|row| row.in_sample_correlation)
            .collect::<Vec<_>>();
        let leave_corr = members
            .iter()
            .filter_map(|row| row.leaveout_correlation)
            .collect::<Vec<_>>();
        let in_residual = members
            .iter()
            .filter_map(|row| row.in_sample_residual)
            .collect::<Vec<_>>();
        let leave_residual = members
            .iter()
            .filter_map(|row| row.leaveout_residual)
            .collect::<Vec<_>>();
        let paired_corr = members
            .iter()
            .filter_map(|row| row.correlation_delta_leaveout_minus_in_sample)
            .collect::<Vec<_>>();
        let paired_residual = members
            .iter()
            .filter_map(|row| row.residual_delta_leaveout_minus_in_sample)
            .collect::<Vec<_>>();
        let template_counts = members
            .iter()
            .filter(|row| row.leaveout_template_source_count > 0)
            .map(|row| row.leaveout_template_source_count as f64)
            .collect::<Vec<_>>();
        let at_or_above = leave_corr
            .iter()
            .filter(|&&value| value >= options.correlation_threshold)
            .count();
        class_rows.push(FeaturePsfLeaveOutClassRow {
            feature_class: feature_class.to_owned(),
            feature_class_label: feature_label.to_owned(),
            global_psf_source_count: global_count,
            sample_count: members.len(),
            in_sample_valid_count: in_corr.len(),
            leaveout_valid_count: leave_corr.len(),
            paired_count: paired_corr.len(),
            leaveout_template_built_count: members
                .iter()
                .filter(|row| row.leaveout_template_source_count > 0)
                .count(),
            leaveout_template_failed_count: members
                .iter()
                .filter(|row| row.leaveout_template_source_count == 0)
                .count(),
            leaveout_template_source_count_median: median(&template_counts),
            correlation_threshold: options.correlation_threshold,
            in_sample_correlation_median: median(&in_corr),
            leaveout_correlation_p10: percentile(&leave_corr, 10.0),
            leaveout_correlation_median: median(&leave_corr),
            leaveout_correlation_p90: percentile(&leave_corr, 90.0),
            leaveout_correlation_ge_threshold_count: at_or_above,
            leaveout_correlation_ge_threshold_fraction: (!leave_corr.is_empty())
                .then_some(at_or_above as f64 / leave_corr.len() as f64),
            in_sample_residual_median: median(&in_residual),
            leaveout_residual_p10: percentile(&leave_residual, 10.0),
            leaveout_residual_median: median(&leave_residual),
            leaveout_residual_p90: percentile(&leave_residual, 90.0),
            paired_correlation_delta_median: median(&paired_corr),
            paired_residual_delta_median: median(&paired_residual),
            sample_detection_ids: members
                .iter()
                .map(|row| row.detection_id.to_string())
                .collect::<Vec<_>>()
                .join("|"),
        });
    }
    Ok(FeaturePsfLeaveOutResult {
        frame_path,
        catalog_path,
        image_shape: [height, width],
        per_class: options.per_class,
        support_radius: options.support_radius,
        max_template_sources: options.max_template_sources,
        correlation_threshold: options.correlation_threshold,
        global_psf: global_psf.as_ref().map(EmpiricalPsf::summary),
        class_rows,
        source_rows,
        parameters: json!({
            "sampling":"各互斥首要类别按 flux_snr 升序后等距抽样，最多 per_class 个",
            "template":"质量通过、非边缘、非掩膜、非线状、非溢出、形状合格且全幅隔离的源",
            "leaveout":"每个样本只从模板候选池排除自身；邻峰隔离仍检查完整候选表",
            "metric":"局部背景扣除后的正残差与经验核余弦相关、相对残差和中心 3x3 能量占比",
            "correlation_threshold_is_not_probability":true
        }),
        conclusion: CONCLUSION.to_owned(),
    })
}

pub fn run(
    fits_path: &Path,
    catalog_path: &Path,
    options: &Options,
) -> Result<FeaturePsfLeaveOutResult> {
    run_with_progress(fits_path, catalog_path, options, None)
}

pub fn run_with_progress(
    fits_path: &Path,
    catalog_path: &Path,
    options: &Options,
    progress: Option<&mut dyn FnMut(usize, usize)>,
) -> Result<FeaturePsfLeaveOutResult> {
    let frame = fits::read_fits(fits_path, false)?;
    let sources = load_detection_catalog(catalog_path)?;
    let image = frame.data.to_f64();
    build_result(
        &image,
        frame.height,
        frame.width,
        fits_path.to_string_lossy().into_owned(),
        Some(catalog_path.to_string_lossy().into_owned()),
        &sources,
        options,
        progress,
    )
}

pub fn handle_request(request: &Value) -> Result<Value> {
    let fits_path = request["fits_path"].as_str().ok_or("missing FITS path")?;
    let catalog_path = request
        .get("source_catalog_path")
        .or_else(|| request.get("source_catalog"))
        .and_then(Value::as_str)
        .ok_or("missing source catalog path")?;
    let out_dir = request["out_dir"]
        .as_str()
        .ok_or("missing output directory")?;
    let mut options = Options {
        out_dir: PathBuf::from(out_dir),
        ..Options::default()
    };
    if let Some(value) = request["per_class"].as_u64() {
        options.per_class = value as usize;
    }
    if let Some(value) = request["support_radius"].as_u64() {
        options.support_radius = value as usize;
    }
    if let Some(value) = request["max_template_sources"].as_u64() {
        options.max_template_sources = value as usize;
    }
    if let Some(value) = request["correlation_threshold"].as_f64() {
        options.correlation_threshold = value;
    }
    let result = run(Path::new(fits_path), Path::new(catalog_path), &options)?;
    let output = write_artifacts(&result, &options.out_dir)?;
    let mut value = result.as_dict();
    value["out_dir"] = output.to_string_lossy().into_owned().into();
    Ok(value)
}

fn parse_optional_float(value: Option<&str>, field: &str, line: usize) -> Result<Option<f64>> {
    let value = value.unwrap_or("").trim();
    if value.is_empty() {
        return Ok(None);
    }
    let number = value.parse::<f64>().map_err(|_| {
        format!("invalid source catalog row {line}: {field} is not numeric: {value:?}")
    })?;
    Ok(number.is_finite().then_some(number))
}

fn required_float(
    record: &csv::StringRecord,
    columns: &HashMap<String, usize>,
    field: &str,
    line: usize,
) -> Result<f64> {
    parse_optional_float(
        columns.get(field).and_then(|&index| record.get(index)),
        field,
        line,
    )?
    .ok_or_else(|| format!("invalid source catalog row {line}: {field} is required"))
}

fn parse_optional_int(value: Option<&str>, field: &str, line: usize) -> Result<Option<i64>> {
    let value = value.unwrap_or("").trim();
    if value.is_empty() {
        return Ok(None);
    }
    let number = value.parse::<f64>().map_err(|_| {
        format!("invalid source catalog row {line}: {field} is not an integer: {value:?}")
    })?;
    if !number.is_finite() || number < i64::MIN as f64 || number > i64::MAX as f64 {
        return Err(format!(
            "invalid source catalog row {line}: {field} is not an integer: {value:?}"
        ));
    }
    Ok(Some(number.trunc() as i64))
}

fn parse_bool(value: Option<&str>, line: usize) -> Result<bool> {
    match value.unwrap_or("").trim().to_lowercase().as_str() {
        "" => Ok(true),
        "true" | "1" | "yes" | "y" | "是" => Ok(true),
        "false" | "0" | "no" | "n" | "否" => Ok(false),
        value => Err(format!(
            "invalid source catalog row {line}: invalid quality_passed value {value:?}"
        )),
    }
}

fn load_detection_catalog(path: &Path) -> Result<Vec<Detection>> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read source catalog {}: {error}", path.display()))?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let columns = headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.to_owned(), index))
        .collect::<HashMap<_, _>>();
    let required = [
        "detection_id",
        "x",
        "y",
        "peak",
        "flux",
        "background",
        "noise",
        "snr",
    ];
    let missing = required
        .iter()
        .filter(|name| !columns.contains_key(**name))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let mut sources = Vec::new();
    for (row_index, item) in reader.records().enumerate() {
        let line = row_index + 2;
        let row = item.map_err(|error| error.to_string())?;
        let detection_id = parse_optional_int(
            columns
                .get("detection_id")
                .and_then(|&index| row.get(index)),
            "detection_id",
            line,
        )?
        .ok_or_else(|| format!("invalid source catalog row {line}: detection_id is required"))?;
        let flags = columns
            .get("flags")
            .and_then(|&index| row.get(index))
            .unwrap_or("")
            .split('|')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect();
        let quality_passed = parse_bool(
            columns
                .get("quality_passed")
                .and_then(|&index| row.get(index)),
            line,
        )?;
        let optional = |field: &str| {
            parse_optional_float(
                columns.get(field).and_then(|&index| row.get(index)),
                field,
                line,
            )
        };
        sources.push(Detection {
            detection_id,
            x: required_float(&row, &columns, "x", line)?,
            y: required_float(&row, &columns, "y", line)?,
            peak: required_float(&row, &columns, "peak", line)?,
            flux: required_float(&row, &columns, "flux", line)?,
            background: required_float(&row, &columns, "background", line)?,
            noise: required_float(&row, &columns, "noise", line)?,
            snr: required_float(&row, &columns, "snr", line)?,
            fwhm: optional("fwhm")?,
            flags,
            flux_error: optional("flux_error")?,
            flux_snr: optional("flux_snr")?,
            filter_snr: optional("filter_snr")?,
            fwhm_x: optional("fwhm_x")?,
            fwhm_y: optional("fwhm_y")?,
            ellipticity: optional("ellipticity")?,
            sharpness: optional("sharpness")?,
            footprint_pixels: parse_optional_int(
                columns
                    .get("footprint_pixels")
                    .and_then(|&index| row.get(index)),
                "footprint_pixels",
                line,
            )?
            .map(|value| value.max(0) as usize),
            psf_support_pixels: parse_optional_int(
                columns
                    .get("psf_support_pixels")
                    .and_then(|&index| row.get(index)),
                "psf_support_pixels",
                line,
            )?
            .map(|value| value.max(0) as usize),
            quality_passed,
            peak_x: optional("peak_x")?,
            peak_y: optional("peak_y")?,
            centroid_shift_px: optional("centroid_shift_px")?,
            proposal_methods: Vec::new(),
            proposal_scales: Vec::new(),
            proposal_snr: optional("proposal_snr")?,
            nearest_gaussian_px: optional("nearest_gaussian_px")?,
            deblend_delta_bic: optional("deblend_delta_bic")?,
            deblend_component_snr: optional("deblend_component_snr")?,
            repeated_code_count: None,
            range_anomaly_pixel_count: None,
            repeated_code_values: Vec::new(),
        });
    }
    Ok(sources)
}

fn option_float(value: Option<f64>) -> String {
    value.map_or_else(String::new, |number| format!("{number:?}"))
}

fn write_csv(
    path: &Path,
    headers: &[&str],
    rows: impl IntoIterator<Item = Vec<String>>,
) -> Result<()> {
    let mut file = File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .terminator(Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(row)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    Ok(())
}

const SOURCE_HEADERS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "in_sample_correlation",
    "leaveout_correlation",
    "correlation_delta_leaveout_minus_in_sample",
    "in_sample_residual",
    "leaveout_residual",
    "residual_delta_leaveout_minus_in_sample",
    "in_sample_central_energy_fraction",
    "leaveout_central_energy_fraction",
    "leaveout_amplitude_adu",
    "leaveout_template_source_count",
    "leaveout_template_median_fwhm_px",
    "leaveout_template_median_ellipticity",
    "in_sample_valid",
    "leaveout_valid",
    "note",
];
const CLASS_HEADERS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "global_psf_source_count",
    "sample_count",
    "in_sample_valid_count",
    "leaveout_valid_count",
    "paired_count",
    "leaveout_template_built_count",
    "leaveout_template_failed_count",
    "leaveout_template_source_count_median",
    "correlation_threshold",
    "in_sample_correlation_median",
    "leaveout_correlation_p10",
    "leaveout_correlation_median",
    "leaveout_correlation_p90",
    "leaveout_correlation_ge_threshold_count",
    "leaveout_correlation_ge_threshold_fraction",
    "in_sample_residual_median",
    "leaveout_residual_p10",
    "leaveout_residual_median",
    "leaveout_residual_p90",
    "paired_correlation_delta_median",
    "paired_residual_delta_median",
    "sample_detection_ids",
];

fn source_csv_row(row: &FeaturePsfLeaveOutSourceRow) -> Vec<String> {
    vec![
        row.detection_id.to_string(),
        row.feature_class.clone(),
        row.feature_class_label.clone(),
        if row.quality_passed { "True" } else { "False" }.to_owned(),
        option_float(row.in_sample_correlation),
        option_float(row.leaveout_correlation),
        option_float(row.correlation_delta_leaveout_minus_in_sample),
        option_float(row.in_sample_residual),
        option_float(row.leaveout_residual),
        option_float(row.residual_delta_leaveout_minus_in_sample),
        option_float(row.in_sample_central_energy_fraction),
        option_float(row.leaveout_central_energy_fraction),
        option_float(row.leaveout_amplitude_adu),
        row.leaveout_template_source_count.to_string(),
        option_float(row.leaveout_template_median_fwhm_px),
        option_float(row.leaveout_template_median_ellipticity),
        if row.in_sample_valid { "True" } else { "False" }.to_owned(),
        if row.leaveout_valid { "True" } else { "False" }.to_owned(),
        row.note.clone(),
    ]
}

fn class_csv_row(row: &FeaturePsfLeaveOutClassRow) -> Vec<String> {
    vec![
        row.feature_class.clone(),
        row.feature_class_label.clone(),
        row.global_psf_source_count.to_string(),
        row.sample_count.to_string(),
        row.in_sample_valid_count.to_string(),
        row.leaveout_valid_count.to_string(),
        row.paired_count.to_string(),
        row.leaveout_template_built_count.to_string(),
        row.leaveout_template_failed_count.to_string(),
        option_float(row.leaveout_template_source_count_median),
        format!("{:?}", row.correlation_threshold),
        option_float(row.in_sample_correlation_median),
        option_float(row.leaveout_correlation_p10),
        option_float(row.leaveout_correlation_median),
        option_float(row.leaveout_correlation_p90),
        row.leaveout_correlation_ge_threshold_count.to_string(),
        option_float(row.leaveout_correlation_ge_threshold_fraction),
        option_float(row.in_sample_residual_median),
        option_float(row.leaveout_residual_p10),
        option_float(row.leaveout_residual_median),
        option_float(row.leaveout_residual_p90),
        option_float(row.paired_correlation_delta_median),
        option_float(row.paired_residual_delta_median),
        row.sample_detection_ids.clone(),
    ]
}

pub fn write_artifacts(result: &FeaturePsfLeaveOutResult, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let output = fs::canonicalize(out_dir).map_err(|error| error.to_string())?;
    write_csv(
        &output.join("feature_psf_leaveout.csv"),
        CLASS_HEADERS,
        result.class_rows.iter().map(class_csv_row),
    )?;
    write_csv(
        &output.join("feature_psf_leaveout_source.csv"),
        SOURCE_HEADERS,
        result.source_rows.iter().map(source_csv_row),
    )?;
    fs::write(
        output.join("feature_psf_leaveout.json"),
        serde_json::to_vec_pretty(result).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output)
}

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf, Options)> {
    let mut fits_path = None;
    let mut catalog_path = None;
    let mut options = Options::default();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let value = |index: &mut usize| -> Result<&str> {
            *index += 1;
            args.get(*index)
                .map(String::as_str)
                .ok_or_else(|| format!("{arg} needs a value"))
        };
        match arg.as_str() {
            "--source-catalog" => catalog_path = Some(PathBuf::from(value(&mut index)?)),
            "--per-class" => {
                options.per_class = value(&mut index)?
                    .parse()
                    .map_err(|_| "invalid --per-class".to_owned())?
            }
            "--support-radius" => {
                options.support_radius = value(&mut index)?
                    .parse()
                    .map_err(|_| "invalid --support-radius".to_owned())?
            }
            "--max-template-sources" => {
                options.max_template_sources = value(&mut index)?
                    .parse()
                    .map_err(|_| "invalid --max-template-sources".to_owned())?
            }
            "--correlation-threshold" => {
                options.correlation_threshold = value(&mut index)?
                    .parse()
                    .map_err(|_| "invalid --correlation-threshold".to_owned())?
            }
            "--out-dir" => options.out_dir = PathBuf::from(value(&mut index)?),
            "--help" | "-h" => return Err("help".into()),
            value if value.starts_with('-') => return Err(format!("unknown option: {value}")),
            value if fits_path.is_none() => fits_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unexpected argument: {arg}")),
        }
        index += 1;
    }
    Ok((
        fits_path.ok_or("FITS path is required")?,
        catalog_path.ok_or("--source-catalog is required")?,
        options,
    ))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let (fits_path, catalog_path, options) = parse_args(args)?;
    let result = run(&fits_path, &catalog_path, &options)?;
    let output = write_artifacts(&result, &options.out_dir)?;
    Ok(json!({
        "tool":"rst19-feature-psf-leaveout",
        "result_path":output.join("feature_psf_leaveout.json").to_string_lossy(),
        "out_dir":output.to_string_lossy(),
        "result":result
    }))
}

pub fn run_cli(args: &[String]) -> Result<i32> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("按特征类别逐源执行经验 PSF 留一审计\n\nUsage: rst19-feature-psf-leaveout FITS --source-catalog CSV [--per-class N] [--support-radius N] [--max-template-sources N] [--correlation-threshold F] [--out-dir DIR]");
        return Ok(0);
    }
    let (fits_path, catalog_path, options) = parse_args(args)?;
    let mut progress = |index, total| {
        eprintln!("feature PSF leave-one-out: sample {index}/{total}");
    };
    let result = run_with_progress(&fits_path, &catalog_path, &options, Some(&mut progress))?;
    let output = write_artifacts(&result, &options.out_dir)?;
    println!(
        "rst19-feature-psf-leaveout: 审计已写入 {}",
        output.to_string_lossy()
    );
    println!("{}", result.conclusion);
    for row in &result.class_rows {
        println!(
            "{}: sample={} leaveout_valid={} corr_median={} residual_median={}",
            row.feature_class,
            row.sample_count,
            row.leaveout_valid_count,
            row.leaveout_correlation_median
                .map_or_else(|| "null".to_owned(), |value| value.to_string()),
            row.leaveout_residual_median
                .map_or_else(|| "null".to_owned(), |value| value.to_string())
        );
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare(actual: &Value, expected: &Value, path: &str) {
        match (actual, expected) {
            (Value::Number(actual), Value::Number(expected)) => {
                let actual = actual.as_f64().unwrap();
                let expected = expected.as_f64().unwrap();
                assert!(
                    (actual - expected).abs() <= 5e-6 + 2e-6 * expected.abs(),
                    "{path}: {actual} != {expected}"
                );
            }
            (Value::Array(actual), Value::Array(expected)) => {
                assert_eq!(actual.len(), expected.len(), "{path}");
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    compare(actual, expected, &format!("{path}[{index}]"));
                }
            }
            (Value::Object(actual), Value::Object(expected)) => {
                assert_eq!(
                    actual.keys().collect::<Vec<_>>(),
                    expected.keys().collect::<Vec<_>>(),
                    "{path} keys"
                );
                for (key, expected) in expected {
                    compare(&actual[key], expected, &format!("{path}.{key}"));
                }
            }
            _ => assert_eq!(actual, expected, "{path}"),
        }
    }

    fn source(id: i64, x: f64, y: f64) -> Detection {
        Detection {
            detection_id: id,
            x,
            y,
            peak: 140.0,
            flux: 900.0,
            background: 20.0,
            noise: 5.0,
            snr: 28.0,
            fwhm: Some(3.0),
            flags: vec![],
            flux_error: None,
            flux_snr: Some(45.0),
            filter_snr: None,
            fwhm_x: None,
            fwhm_y: None,
            ellipticity: Some(0.1),
            sharpness: None,
            footprint_pixels: None,
            psf_support_pixels: None,
            quality_passed: true,
            peak_x: None,
            peak_y: None,
            centroid_shift_px: None,
            proposal_methods: vec![],
            proposal_scales: vec![],
            proposal_snr: None,
            nearest_gaussian_px: None,
            deblend_delta_bic: None,
            deblend_component_snr: None,
            repeated_code_count: None,
            range_anomaly_pixel_count: None,
            repeated_code_values: vec![],
        }
    }

    fn gaussian_image() -> (Vec<f64>, Vec<Detection>) {
        let mut image = vec![20.0f32; 96 * 96];
        let positions = [(16.0, 16.0), (48.0, 16.0), (16.0, 48.0), (48.0, 48.0)];
        let sigma: f64 = 3.0 / 2.354820045;
        for &(cx, cy) in &positions {
            for y in 0..96 {
                for x in 0..96 {
                    let r2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                    let value = 120.0 * (-0.5 * r2 / sigma.powi(2)).exp();
                    let index = y * 96 + x;
                    image[index] = (image[index] as f64 + value) as f32;
                }
            }
        }
        (
            image.into_iter().map(f64::from).collect(),
            positions
                .into_iter()
                .enumerate()
                .map(|(index, (x, y))| source(index as i64, x, y))
                .collect(),
        )
    }

    #[test]
    fn leaveout_templates_exclude_evaluated_source_and_preserve_progress_order() {
        let (image, sources) = gaussian_image();
        let mut progress = Vec::new();
        let mut callback = |index, total| progress.push((index, total));
        let options = Options {
            per_class: 4,
            support_radius: 3,
            max_template_sources: 4,
            ..Options::default()
        };
        let result = build_result(
            &image,
            96,
            96,
            "<array>".into(),
            None,
            &sources,
            &options,
            Some(&mut callback),
        )
        .unwrap();
        let compact = result
            .class_rows
            .iter()
            .find(|row| row.feature_class == "compact_quality")
            .unwrap();
        assert_eq!(compact.sample_count, 4);
        assert_eq!(compact.in_sample_valid_count, 4);
        assert_eq!(compact.leaveout_valid_count, 4);
        assert_eq!(compact.paired_count, 4);
        assert_eq!(compact.leaveout_template_source_count_median, Some(3.0));
        assert!(compact.leaveout_correlation_median.unwrap() > 0.95);
        assert_eq!(progress, [(1, 4), (2, 4), (3, 4), (4, 4)]);
        assert!(result
            .source_rows
            .iter()
            .all(|row| row.leaveout_template_source_count == 3));
        assert_eq!(result.global_psf.as_ref().unwrap().source_count, 4);
    }

    #[test]
    fn output_matches_python_baseline_for_every_numeric_field() {
        let (image, sources) = gaussian_image();
        let options = Options {
            per_class: 4,
            support_radius: 3,
            max_template_sources: 4,
            ..Options::default()
        };
        let result = build_result(
            &image,
            96,
            96,
            "<array>".into(),
            None,
            &sources,
            &options,
            None,
        )
        .unwrap();
        let actual = serde_json::to_value(result).unwrap();
        let expected: Value =
            serde_json::from_str(include_str!("../tests/fixtures/feature_psf_leaveout.json"))
                .unwrap();
        compare(&actual, &expected, "result");
    }

    #[test]
    fn artifact_csv_has_python_compatible_bom_crlf_and_column_order() {
        let (image, sources) = gaussian_image();
        let options = Options {
            per_class: 4,
            support_radius: 3,
            max_template_sources: 4,
            ..Options::default()
        };
        let result = build_result(
            &image,
            96,
            96,
            "<array>".into(),
            None,
            &sources,
            &options,
            None,
        )
        .unwrap();
        let root = std::env::temp_dir().join(format!("rst19-psf-leaveout-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let output = write_artifacts(&result, &root).unwrap();
        let bytes = fs::read(output.join("feature_psf_leaveout.csv")).unwrap();
        assert!(bytes.starts_with(&[0xef, 0xbb, 0xbf]));
        assert!(bytes.windows(2).any(|window| window == b"\r\n"));
        assert!(String::from_utf8_lossy(&bytes)
            .contains("feature_class,feature_class_label,global_psf_source_count"));
        let payload: Value =
            serde_json::from_slice(&fs::read(output.join("feature_psf_leaveout.json")).unwrap())
                .unwrap();
        assert_eq!(payload["global_psf"]["source_count"], 4);
        assert_eq!(payload["source_rows"].as_array().unwrap().len(), 4);
        fs::remove_dir_all(root).unwrap();
    }
}
