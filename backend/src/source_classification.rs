//! Deterministic primary source classes and proposal-method summaries.
//!
//! The input quality flags remain overlapping. These summaries add one
//! mutually exclusive primary class for reporting without changing detector
//! quality or implying physical source identity.

use crate::measurement::Detection;
use serde::Serialize;
use std::collections::BTreeMap;

pub const SOURCE_FEATURE_CLASSES: [(&str, &str); 9] = [
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

const FLAG_GROUPS: [(&str, &[&str]); 7] = [
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

/// Return the first matching primary class in the Python audit's priority order.
pub fn classify_source_feature(source: &Detection) -> &'static str {
    for (feature_class, group) in FLAG_GROUPS {
        if source
            .flags
            .iter()
            .any(|flag| group.contains(&flag.as_str()))
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

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FeatureSummaryRow {
    pub feature_class: &'static str,
    pub feature_class_label: &'static str,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub rejected_count: usize,
    pub quality_fraction: Option<f64>,
    pub median_flux_snr: Option<f64>,
    pub max_flux_snr: Option<f64>,
    pub common_flags: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProposalMethodSummaryRow {
    pub feature_class: &'static str,
    pub feature_class_label: &'static str,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub gaussian_present_count: usize,
    pub dog_narrow_present_count: usize,
    pub dog_broad_present_count: usize,
    pub all_three_count: usize,
    pub gaussian_only_count: usize,
    pub dog_only_count: usize,
    pub partial_combination_count: usize,
    pub no_gaussian_count: usize,
    pub no_method_count: usize,
    pub no_gaussian_quality_count: usize,
    pub no_gaussian_median_flux_snr: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FeatureMorphologyRow {
    pub feature_class: &'static str,
    pub feature_class_label: &'static str,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub rejected_count: usize,
    pub quality_fraction: Option<f64>,
    pub high_flux_snr_threshold: f64,
    pub high_flux_snr_count: usize,
    pub high_flux_snr_rejected_count: usize,
    pub high_flux_snr_rejection_fraction: Option<f64>,
    pub median_peak_snr: Option<f64>,
    pub median_flux_snr: Option<f64>,
    pub common_flags: String,
    pub peak_snr_p10: Option<f64>,
    pub peak_snr_median: Option<f64>,
    pub peak_snr_p90: Option<f64>,
    pub flux_snr_p10: Option<f64>,
    pub flux_snr_median: Option<f64>,
    pub flux_snr_p90: Option<f64>,
    pub filter_snr_p10: Option<f64>,
    pub filter_snr_median: Option<f64>,
    pub filter_snr_p90: Option<f64>,
    pub fwhm_px_p10: Option<f64>,
    pub fwhm_px_median: Option<f64>,
    pub fwhm_px_p90: Option<f64>,
    pub ellipticity_p10: Option<f64>,
    pub ellipticity_median: Option<f64>,
    pub ellipticity_p90: Option<f64>,
    pub sharpness_p10: Option<f64>,
    pub sharpness_median: Option<f64>,
    pub sharpness_p90: Option<f64>,
    pub psf_support_pixels_p10: Option<f64>,
    pub psf_support_pixels_median: Option<f64>,
    pub psf_support_pixels_p90: Option<f64>,
    pub footprint_pixels_p10: Option<f64>,
    pub footprint_pixels_median: Option<f64>,
    pub footprint_pixels_p90: Option<f64>,
    pub centroid_shift_px_p10: Option<f64>,
    pub centroid_shift_px_median: Option<f64>,
    pub centroid_shift_px_p90: Option<f64>,
    pub x_px_p10: Option<f64>,
    pub x_px_median: Option<f64>,
    pub x_px_p90: Option<f64>,
    pub y_px_p10: Option<f64>,
    pub y_px_median: Option<f64>,
    pub y_px_p90: Option<f64>,
    pub range_anomaly_fraction: Option<f64>,
    pub linear_artifact_fraction: Option<f64>,
    pub masked_or_edge_fraction: Option<f64>,
    pub crowded_blend_fraction: Option<f64>,
    pub spike_or_support_fraction: Option<f64>,
    pub weak_or_background_fraction: Option<f64>,
    pub shape_outlier_fraction: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FeatureSubclassSummaryRow {
    pub subclass: &'static str,
    pub subclass_label: &'static str,
    pub definition: &'static str,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub rejected_count: usize,
    pub quality_fraction: Option<f64>,
    pub high_flux_snr_threshold: f64,
    pub high_snr_rejected_count: usize,
    pub median_flux_snr: Option<f64>,
    pub primary_feature_classes: String,
    pub common_flags: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FeatureSpatialRow {
    pub feature_class: &'static str,
    pub feature_class_label: &'static str,
    pub grid_size: usize,
    pub grid_row: usize,
    pub grid_column: usize,
    pub x_min_px: f64,
    pub x_max_px: f64,
    pub y_min_px: f64,
    pub y_max_px: f64,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub rejected_count: usize,
    pub quality_fraction: Option<f64>,
    pub high_flux_snr_threshold: f64,
    pub high_flux_snr_count: usize,
    pub high_flux_snr_rejected_count: usize,
    pub median_flux_snr: Option<f64>,
    pub flux_snr_p90: Option<f64>,
}

fn selected_snr(source: &Detection) -> f64 {
    source.flux_snr.unwrap_or(source.snr)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn finite_values<'a>(
    sources: impl Iterator<Item = &'a Detection>,
    value: impl Fn(&Detection) -> Option<f64>,
) -> Vec<f64> {
    sources
        .filter_map(value)
        .filter(|value| value.is_finite())
        .collect()
}

fn percentile(values: &[f64], q: f64) -> Option<f64> {
    crate::statistics::percentile(values, q).ok()
}

fn common_flags(sources: &[&Detection], limit: usize) -> String {
    let mut counts = BTreeMap::<&str, usize>::new();
    for source in sources {
        for flag in &source.flags {
            *counts.entry(flag).or_default() += 1;
        }
    }
    let mut rows = counts.into_iter().collect::<Vec<_>>();
    rows.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
    rows.into_iter()
        .take(limit)
        .map(|(flag, _)| flag)
        .collect::<Vec<_>>()
        .join("|")
}

fn flag_fraction(sources: &[&Detection], group: &[&str]) -> Option<f64> {
    (!sources.is_empty()).then(|| {
        sources
            .iter()
            .filter(|source| {
                source
                    .flags
                    .iter()
                    .any(|flag| group.contains(&flag.as_str()))
            })
            .count() as f64
            / sources.len() as f64
    })
}

/// Summarize source shape and significance by the fixed primary-class order.
/// The high-SNR slice remains a diagnostic; it does not alter quality flags.
pub fn summarize_feature_morphology(
    sources: &[Detection],
    high_flux_snr_threshold: f64,
) -> crate::Result<Vec<FeatureMorphologyRow>> {
    if !high_flux_snr_threshold.is_finite() || high_flux_snr_threshold <= 0.0 {
        return Err("high_flux_snr_threshold must be finite and positive".into());
    }
    let grouped = group_sources(sources);
    let mut rows = Vec::with_capacity(SOURCE_FEATURE_CLASSES.len());
    for (members, (feature_class, feature_class_label)) in
        grouped.into_iter().zip(SOURCE_FEATURE_CLASSES)
    {
        let quality_count = members
            .iter()
            .filter(|source| source.quality_passed)
            .count();
        let rejected_count = members.len() - quality_count;
        let flux_snr = finite_values(members.iter().copied(), |source| source.flux_snr);
        let peak_snr = finite_values(members.iter().copied(), |source| Some(source.snr));
        let high_flux_snr_count = flux_snr
            .iter()
            .filter(|value| **value >= high_flux_snr_threshold)
            .count();
        let high_flux_snr_rejected_count = members
            .iter()
            .filter(|source| {
                !source.quality_passed
                    && source
                        .flux_snr
                        .is_some_and(|value| value.is_finite() && value >= high_flux_snr_threshold)
            })
            .count();

        let values =
            |select: fn(&Detection) -> Option<f64>| finite_values(members.iter().copied(), select);
        let peak_snr_values = values(|source| Some(source.snr));
        let flux_snr_values = values(|source| source.flux_snr);
        let filter_snr_values = values(|source| source.filter_snr);
        let fwhm_values = values(|source| source.fwhm);
        let ellipticity_values = values(|source| source.ellipticity);
        let sharpness_values = values(|source| source.sharpness);
        let support_values = values(|source| source.psf_support_pixels.map(|value| value as f64));
        let footprint_values = values(|source| source.footprint_pixels.map(|value| value as f64));
        let shift_values = values(|source| source.centroid_shift_px);
        let x_values = values(|source| Some(source.x));
        let y_values = values(|source| Some(source.y));
        let quantiles = |data: &[f64]| {
            (
                percentile(data, 10.0),
                percentile(data, 50.0),
                percentile(data, 90.0),
            )
        };
        let (peak_snr_p10, peak_snr_median, peak_snr_p90) = quantiles(&peak_snr_values);
        let (flux_snr_p10, flux_snr_median, flux_snr_p90) = quantiles(&flux_snr_values);
        let (filter_snr_p10, filter_snr_median, filter_snr_p90) = quantiles(&filter_snr_values);
        let (fwhm_px_p10, fwhm_px_median, fwhm_px_p90) = quantiles(&fwhm_values);
        let (ellipticity_p10, ellipticity_median, ellipticity_p90) = quantiles(&ellipticity_values);
        let (sharpness_p10, sharpness_median, sharpness_p90) = quantiles(&sharpness_values);
        let (psf_support_pixels_p10, psf_support_pixels_median, psf_support_pixels_p90) =
            quantiles(&support_values);
        let (footprint_pixels_p10, footprint_pixels_median, footprint_pixels_p90) =
            quantiles(&footprint_values);
        let (centroid_shift_px_p10, centroid_shift_px_median, centroid_shift_px_p90) =
            quantiles(&shift_values);
        let (x_px_p10, x_px_median, x_px_p90) = quantiles(&x_values);
        let (y_px_p10, y_px_median, y_px_p90) = quantiles(&y_values);
        rows.push(FeatureMorphologyRow {
            feature_class,
            feature_class_label,
            candidate_count: members.len(),
            quality_count,
            rejected_count,
            quality_fraction: (!members.is_empty())
                .then_some(quality_count as f64 / members.len() as f64),
            high_flux_snr_threshold,
            high_flux_snr_count,
            high_flux_snr_rejected_count,
            high_flux_snr_rejection_fraction: (high_flux_snr_count != 0)
                .then_some(high_flux_snr_rejected_count as f64 / high_flux_snr_count as f64),
            median_peak_snr: percentile(&peak_snr, 50.0),
            median_flux_snr: percentile(&flux_snr, 50.0),
            common_flags: common_flags(&members, 6),
            peak_snr_p10,
            peak_snr_median,
            peak_snr_p90,
            flux_snr_p10,
            flux_snr_median,
            flux_snr_p90,
            filter_snr_p10,
            filter_snr_median,
            filter_snr_p90,
            fwhm_px_p10,
            fwhm_px_median,
            fwhm_px_p90,
            ellipticity_p10,
            ellipticity_median,
            ellipticity_p90,
            sharpness_p10,
            sharpness_median,
            sharpness_p90,
            psf_support_pixels_p10,
            psf_support_pixels_median,
            psf_support_pixels_p90,
            footprint_pixels_p10,
            footprint_pixels_median,
            footprint_pixels_p90,
            centroid_shift_px_p10,
            centroid_shift_px_median,
            centroid_shift_px_p90,
            x_px_p10,
            x_px_median,
            x_px_p90,
            y_px_p10,
            y_px_median,
            y_px_p90,
            range_anomaly_fraction: flag_fraction(
                &members,
                &["SATURATED", "NEGATIVE_OVERFLOW", "CODE_PATTERN"],
            ),
            linear_artifact_fraction: flag_fraction(&members, &["LINE_ARTIFACT"]),
            masked_or_edge_fraction: flag_fraction(&members, &["MASKED", "PARTIAL_MASKED", "EDGE"]),
            crowded_blend_fraction: flag_fraction(&members, &["UNRESOLVED_BLEND"]),
            spike_or_support_fraction: flag_fraction(
                &members,
                &[
                    "SPIKE",
                    "NARROW",
                    "SMALL_FOOTPRINT",
                    "INSUFFICIENT_PSF_SUPPORT",
                ],
            ),
            weak_or_background_fraction: flag_fraction(
                &members,
                &["LOW_FLUX_SNR", "NON_POSITIVE_FLUX", "BACKGROUND_UNCERTAIN"],
            ),
            shape_outlier_fraction: flag_fraction(
                &members,
                &["NO_SHAPE", "BROAD", "ELONGATED", "DIFFUSE"],
            ),
        });
    }
    Ok(rows)
}

const SUBCLASSES: [(&str, &str, &str); 11] = [
    (
        "range_code_pattern",
        "数据有效性/重复码峰值",
        "flags 含 CODE_PATTERN；与其他范围旗标可重叠",
    ),
    (
        "range_negative_overflow",
        "数据有效性/极端负值",
        "flags 含 NEGATIVE_OVERFLOW；与其他范围旗标可重叠",
    ),
    (
        "range_saturated",
        "数据有效性/饱和标志",
        "flags 含 SATURATED；不是设备满阱标定",
    ),
    (
        "edge_only",
        "边界截断",
        "含 EDGE，但不含 PARTIAL_MASKED 或 MASKED",
    ),
    (
        "partial_masked_only",
        "部分掩膜（可容错候选）",
        "含 PARTIAL_MASKED，但不含 EDGE 或 MASKED",
    ),
    (
        "edge_partial_masked",
        "边界 + 部分掩膜",
        "同时含 EDGE 和 PARTIAL_MASKED，但不含 MASKED",
    ),
    (
        "hard_masked",
        "硬掩膜",
        "含 MASKED，但不含 EDGE 或 PARTIAL_MASKED",
    ),
    (
        "edge_hard_masked",
        "边界 + 硬掩膜",
        "同时含 EDGE 和 MASKED，但不含 PARTIAL_MASKED",
    ),
    ("line_artifact", "线状/拖影", "flags 含 LINE_ARTIFACT"),
    (
        "unresolved_blend",
        "未分辨近邻",
        "flags 含 UNRESOLVED_BLEND",
    ),
    ("spike", "尖峰", "flags 含 SPIKE"),
];

fn has_flag(source: &Detection, flag: &str) -> bool {
    source.flags.iter().any(|candidate| candidate == flag)
}

fn matches_subclass(source: &Detection, subclass: &str) -> bool {
    let edge = has_flag(source, "EDGE");
    let partial = has_flag(source, "PARTIAL_MASKED");
    let masked = has_flag(source, "MASKED");
    match subclass {
        "range_code_pattern" => has_flag(source, "CODE_PATTERN"),
        "range_negative_overflow" => has_flag(source, "NEGATIVE_OVERFLOW"),
        "range_saturated" => has_flag(source, "SATURATED"),
        "edge_only" => edge && !partial && !masked,
        "partial_masked_only" => partial && !edge && !masked,
        "edge_partial_masked" => edge && partial && !masked,
        "hard_masked" => masked && !edge && !partial,
        "edge_hard_masked" => edge && masked && !partial,
        "line_artifact" => has_flag(source, "LINE_ARTIFACT"),
        "unresolved_blend" => has_flag(source, "UNRESOLVED_BLEND"),
        "spike" => has_flag(source, "SPIKE"),
        _ => false,
    }
}

pub fn summarize_source_feature_subclasses(
    sources: &[Detection],
    high_flux_snr_threshold: f64,
) -> crate::Result<Vec<FeatureSubclassSummaryRow>> {
    if !high_flux_snr_threshold.is_finite() || high_flux_snr_threshold <= 0.0 {
        return Err("high_flux_snr_threshold must be finite and positive".into());
    }
    Ok(SUBCLASSES
        .iter()
        .map(|(subclass, subclass_label, definition)| {
            let members = sources
                .iter()
                .filter(|source| matches_subclass(source, subclass))
                .collect::<Vec<_>>();
            let quality_count = members
                .iter()
                .filter(|source| source.quality_passed)
                .count();
            let flux_snr = finite_values(members.iter().copied(), |source| source.flux_snr);
            let high_snr_rejected_count = members
                .iter()
                .filter(|source| {
                    !source.quality_passed
                        && source.flux_snr.is_some_and(|value| {
                            value.is_finite() && value >= high_flux_snr_threshold
                        })
                })
                .count();
            let mut class_counts = BTreeMap::<&str, usize>::new();
            for source in &members {
                *class_counts
                    .entry(classify_source_feature(source))
                    .or_default() += 1;
            }
            let mut classes = class_counts.into_iter().collect::<Vec<_>>();
            classes.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
            FeatureSubclassSummaryRow {
                subclass,
                subclass_label,
                definition,
                candidate_count: members.len(),
                quality_count,
                rejected_count: members.len() - quality_count,
                quality_fraction: (!members.is_empty())
                    .then_some(quality_count as f64 / members.len() as f64),
                high_flux_snr_threshold,
                high_snr_rejected_count,
                median_flux_snr: percentile(&flux_snr, 50.0),
                primary_feature_classes: classes
                    .into_iter()
                    .map(|(name, count)| format!("{name}:{count}"))
                    .collect::<Vec<_>>()
                    .join("|"),
                common_flags: common_flags(&members, 4),
            }
        })
        .collect())
}

pub fn summarize_feature_spatial_distribution(
    sources: &[Detection],
    image_shape: [usize; 2],
    grid_size: usize,
    high_flux_snr_threshold: f64,
) -> crate::Result<Vec<FeatureSpatialRow>> {
    if grid_size == 0 {
        return Err("grid_size must be positive".into());
    }
    if !high_flux_snr_threshold.is_finite() || high_flux_snr_threshold <= 0.0 {
        return Err("high_flux_snr_threshold must be finite and positive".into());
    }
    let [height, width] = image_shape;
    if height == 0 || width == 0 {
        return Err("image_shape must contain two positive dimensions".into());
    }
    let class_count = SOURCE_FEATURE_CLASSES.len();
    let mut cells = vec![Vec::<&Detection>::new(); class_count * grid_size * grid_size];
    for source in sources {
        let class_index = SOURCE_FEATURE_CLASSES
            .iter()
            .position(|(name, _)| *name == classify_source_feature(source))
            .expect("every source class must have a summary row");
        let x = source.peak_x.unwrap_or(source.x);
        let y = source.peak_y.unwrap_or(source.y);
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let column = ((x * grid_size as f64 / width as f64) as isize)
            .clamp(0, grid_size as isize - 1) as usize;
        let row = ((y * grid_size as f64 / height as f64) as isize).clamp(0, grid_size as isize - 1)
            as usize;
        cells[(class_index * grid_size + row) * grid_size + column].push(source);
    }
    let mut rows = Vec::with_capacity(class_count * grid_size * grid_size);
    for (class_index, (feature_class, feature_class_label)) in
        SOURCE_FEATURE_CLASSES.into_iter().enumerate()
    {
        for grid_row in 0..grid_size {
            for grid_column in 0..grid_size {
                let members =
                    &cells[(class_index * grid_size + grid_row) * grid_size + grid_column];
                let values = finite_values(members.iter().copied(), |source| source.flux_snr);
                let quality_count = members
                    .iter()
                    .filter(|source| source.quality_passed)
                    .count();
                let high_flux_snr_count = values
                    .iter()
                    .filter(|value| **value >= high_flux_snr_threshold)
                    .count();
                let high_flux_snr_rejected_count = members
                    .iter()
                    .filter(|source| {
                        !source.quality_passed
                            && source.flux_snr.is_some_and(|value| {
                                value.is_finite() && value >= high_flux_snr_threshold
                            })
                    })
                    .count();
                rows.push(FeatureSpatialRow {
                    feature_class,
                    feature_class_label,
                    grid_size,
                    grid_row,
                    grid_column,
                    x_min_px: width as f64 * grid_column as f64 / grid_size as f64,
                    x_max_px: width as f64 * (grid_column + 1) as f64 / grid_size as f64,
                    y_min_px: height as f64 * grid_row as f64 / grid_size as f64,
                    y_max_px: height as f64 * (grid_row + 1) as f64 / grid_size as f64,
                    candidate_count: members.len(),
                    quality_count,
                    rejected_count: members.len() - quality_count,
                    quality_fraction: (!members.is_empty())
                        .then_some(quality_count as f64 / members.len() as f64),
                    high_flux_snr_threshold,
                    high_flux_snr_count,
                    high_flux_snr_rejected_count,
                    median_flux_snr: percentile(&values, 50.0),
                    flux_snr_p90: percentile(&values, 90.0),
                });
            }
        }
    }
    Ok(rows)
}

fn group_sources<'a>(sources: &'a [Detection]) -> [Vec<&'a Detection>; 9] {
    let mut grouped: [Vec<&Detection>; 9] = std::array::from_fn(|_| Vec::new());
    for source in sources {
        let class = classify_source_feature(source);
        let index = SOURCE_FEATURE_CLASSES
            .iter()
            .position(|(name, _)| *name == class)
            .expect("every source class must have a summary row");
        grouped[index].push(source);
    }
    grouped
}

pub fn summarize_source_features(sources: &[Detection]) -> Vec<FeatureSummaryRow> {
    group_sources(sources)
        .into_iter()
        .zip(SOURCE_FEATURE_CLASSES)
        .map(|(members, (feature_class, feature_class_label))| {
            let mut snr_values = members
                .iter()
                .map(|source| selected_snr(source))
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            let max_flux_snr = snr_values.iter().copied().max_by(f64::total_cmp);
            let median_flux_snr = median(&mut snr_values);
            let quality_count = members
                .iter()
                .filter(|source| source.quality_passed)
                .count();

            let mut flag_counts = BTreeMap::<&str, usize>::new();
            for source in &members {
                for flag in &source.flags {
                    *flag_counts.entry(flag).or_default() += 1;
                }
            }
            let mut common_flags = flag_counts.into_iter().collect::<Vec<_>>();
            common_flags
                .sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));

            FeatureSummaryRow {
                feature_class,
                feature_class_label,
                candidate_count: members.len(),
                quality_count,
                rejected_count: members.len() - quality_count,
                quality_fraction: (!members.is_empty())
                    .then_some(quality_count as f64 / members.len() as f64),
                median_flux_snr,
                max_flux_snr,
                common_flags: common_flags
                    .into_iter()
                    .take(4)
                    .map(|(flag, _)| flag)
                    .collect::<Vec<_>>()
                    .join("|"),
            }
        })
        .collect()
}

pub fn summarize_source_proposal_methods(sources: &[Detection]) -> Vec<ProposalMethodSummaryRow> {
    group_sources(sources)
        .into_iter()
        .zip(SOURCE_FEATURE_CLASSES)
        .map(|(members, (feature_class, feature_class_label))| {
            let method_sets = members
                .iter()
                .map(|source| {
                    source
                        .proposal_methods
                        .iter()
                        .map(String::as_str)
                        .collect::<std::collections::HashSet<_>>()
                })
                .collect::<Vec<_>>();
            let gaussian_present_count = method_sets
                .iter()
                .filter(|methods| methods.contains("gaussian"))
                .count();
            let dog_narrow_present_count = method_sets
                .iter()
                .filter(|methods| methods.contains("dog_narrow"))
                .count();
            let dog_broad_present_count = method_sets
                .iter()
                .filter(|methods| methods.contains("dog_broad"))
                .count();
            let all_three_count = method_sets
                .iter()
                .filter(|methods| {
                    ["gaussian", "dog_narrow", "dog_broad"]
                        .iter()
                        .all(|method| methods.contains(method))
                })
                .count();
            let gaussian_only_count = method_sets
                .iter()
                .filter(|methods| methods.len() == 1 && methods.contains("gaussian"))
                .count();
            let dog_only_count = method_sets
                .iter()
                .filter(|methods| !methods.contains("gaussian") && !methods.is_empty())
                .count();
            let no_gaussian_indices = method_sets
                .iter()
                .enumerate()
                .filter_map(|(index, methods)| (!methods.contains("gaussian")).then_some(index))
                .collect::<Vec<_>>();
            let no_method_count = method_sets
                .iter()
                .filter(|methods| methods.is_empty())
                .count();
            let partial_combination_count = members.len()
                - all_three_count
                - gaussian_only_count
                - dog_only_count
                - no_method_count;
            let mut no_gaussian_snr = no_gaussian_indices
                .iter()
                .map(|index| selected_snr(members[*index]))
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            let no_gaussian_quality_count = no_gaussian_indices
                .iter()
                .filter(|index| members[**index].quality_passed)
                .count();

            ProposalMethodSummaryRow {
                feature_class,
                feature_class_label,
                candidate_count: members.len(),
                quality_count: members
                    .iter()
                    .filter(|source| source.quality_passed)
                    .count(),
                gaussian_present_count,
                dog_narrow_present_count,
                dog_broad_present_count,
                all_three_count,
                gaussian_only_count,
                dog_only_count,
                partial_combination_count,
                no_gaussian_count: no_gaussian_indices.len(),
                no_method_count,
                no_gaussian_quality_count,
                no_gaussian_median_flux_snr: median(&mut no_gaussian_snr),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(
        flags: &[&str],
        quality_passed: bool,
        flux_snr: Option<f64>,
        snr: f64,
        proposal_methods: &[&str],
    ) -> Detection {
        Detection {
            detection_id: 1,
            x: 0.0,
            y: 0.0,
            peak: 0.0,
            flux: 0.0,
            background: 0.0,
            noise: 1.0,
            snr,
            fwhm: None,
            flags: flags.iter().map(|flag| (*flag).into()).collect(),
            flux_error: None,
            flux_snr,
            filter_snr: None,
            fwhm_x: None,
            fwhm_y: None,
            ellipticity: None,
            sharpness: None,
            footprint_pixels: None,
            psf_support_pixels: None,
            quality_passed,
            peak_x: None,
            peak_y: None,
            centroid_shift_px: None,
            proposal_methods: proposal_methods
                .iter()
                .map(|method| (*method).into())
                .collect(),
            proposal_scales: Vec::new(),
            proposal_snr: None,
            nearest_gaussian_px: None,
            deblend_delta_bic: None,
            deblend_component_snr: None,
            repeated_code_count: None,
            range_anomaly_pixel_count: None,
            repeated_code_values: Vec::new(),
        }
    }

    #[test]
    fn primary_classes_keep_python_priority_and_quality_fallback() {
        assert_eq!(
            classify_source_feature(&source(
                &["SATURATED", "LINE_ARTIFACT", "MASKED"],
                false,
                None,
                1.0,
                &[]
            )),
            "range_anomaly"
        );
        assert_eq!(
            classify_source_feature(&source(&["LINE_ARTIFACT", "MASKED"], true, None, 1.0, &[])),
            "linear_artifact"
        );
        assert_eq!(
            classify_source_feature(&source(&["PARTIAL_MASKED"], false, None, 1.0, &[])),
            "masked_or_edge"
        );
        assert_eq!(
            classify_source_feature(&source(&[], true, None, 1.0, &[])),
            "compact_quality"
        );
        assert_eq!(
            classify_source_feature(&source(&[], false, None, 1.0, &[])),
            "other_rejected"
        );
    }

    #[test]
    fn feature_and_proposal_summaries_match_frozen_python_reference() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/source_classification_parity.json"
        ))
        .unwrap();
        let sources: Vec<Detection> = serde_json::from_value(fixture["sources"].clone()).unwrap();
        assert_eq!(
            serde_json::to_value(summarize_source_features(&sources)).unwrap(),
            fixture["feature_summary"]
        );
        assert_eq!(
            serde_json::to_value(summarize_source_proposal_methods(&sources)).unwrap(),
            fixture["proposal_method_summary"]
        );
        assert_eq!(
            serde_json::to_value(summarize_feature_morphology(&sources, 10.0).unwrap()).unwrap(),
            fixture["morphology_summary"]
        );
        assert_eq!(
            serde_json::to_value(summarize_source_feature_subclasses(&sources, 10.0).unwrap())
                .unwrap(),
            fixture["subclass_summary"]
        );
        assert_eq!(
            serde_json::to_value(
                summarize_feature_spatial_distribution(&sources, [128, 128], 4, 10.0).unwrap()
            )
            .unwrap(),
            fixture["spatial_summary"]
        );
    }

    #[test]
    fn source_summary_thresholds_and_grid_dimensions_must_be_valid() {
        let empty = [];
        assert!(summarize_feature_morphology(&empty, f64::NAN).is_err());
        assert!(summarize_source_feature_subclasses(&empty, 0.0).is_err());
        assert!(summarize_feature_spatial_distribution(&empty, [128, 128], 0, 10.0).is_err());
        assert!(summarize_feature_spatial_distribution(&empty, [0, 128], 4, 10.0).is_err());
    }

    #[test]
    fn nonfinite_snr_is_excluded_from_summaries() {
        let source = source(&[], false, Some(f64::NAN), 1.0, &[]);
        let rows = summarize_source_features(&[source.clone()]);
        assert_eq!(rows[8].median_flux_snr, None);
        assert_eq!(rows[8].max_flux_snr, None);
        let method_rows = summarize_source_proposal_methods(&[source]);
        assert_eq!(method_rows[8].no_gaussian_median_flux_snr, None);
    }
}
