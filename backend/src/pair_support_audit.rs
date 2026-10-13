//! Local two-source aperture support and holdout-mask audit.
use crate::{
    assignment::linear_sum_assignment,
    detection::{self, DetectionOptions},
    fits,
    measurement::Detection,
    models::{FitsFrame, PixelData},
    Result,
};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub const MASK_MODES: [&str; 7] = [
    "raw",
    "range_masked",
    "repeated_code_masked",
    "local_anomaly_masked",
    "local_anomaly_repeated_code_masked",
    "shared_aperture_masked",
    "shared_local_anomaly_repeated_code_masked",
];

const ROW_FIELDS: &[&str] = &[
    "frame_index",
    "path",
    "frame_shift_x_px",
    "frame_shift_y_px",
    "mask_mode",
    "crop_x0",
    "crop_y0",
    "crop_width",
    "crop_height",
    "masked_pixel_count",
    "local_negative_limit_adu",
    "candidate_count",
    "quality_count",
    "nearby_candidate_count",
    "nearby_quality_count",
    "assigned_candidate_count",
    "primary_re_detected_id",
    "secondary_re_detected_id",
    "primary_match_distance_px",
    "secondary_match_distance_px",
    "primary_local_quality_passed",
    "secondary_local_quality_passed",
    "primary_local_flags",
    "secondary_local_flags",
    "primary_catalog_quality_passed",
    "secondary_catalog_quality_passed",
    "primary_catalog_flags",
    "secondary_catalog_flags",
    "primary_background_adu",
    "primary_noise_adu",
    "primary_total_valid_pixels",
    "primary_shared_valid_pixels",
    "primary_exclusive_valid_pixels",
    "primary_partition_valid_pixels",
    "primary_total_net_flux_adu",
    "primary_shared_net_flux_adu",
    "primary_exclusive_net_flux_adu",
    "primary_partition_net_flux_adu",
    "primary_total_positive_excess_adu",
    "primary_shared_positive_excess_adu",
    "primary_exclusive_positive_excess_adu",
    "primary_partition_positive_excess_adu",
    "primary_shared_positive_fraction",
    "primary_exclusive_snr",
    "primary_partition_snr",
    "secondary_background_adu",
    "secondary_noise_adu",
    "secondary_total_valid_pixels",
    "secondary_shared_valid_pixels",
    "secondary_exclusive_valid_pixels",
    "secondary_partition_valid_pixels",
    "secondary_total_net_flux_adu",
    "secondary_shared_net_flux_adu",
    "secondary_exclusive_net_flux_adu",
    "secondary_partition_net_flux_adu",
    "secondary_total_positive_excess_adu",
    "secondary_shared_positive_excess_adu",
    "secondary_exclusive_positive_excess_adu",
    "secondary_partition_positive_excess_adu",
    "secondary_shared_positive_fraction",
    "secondary_exclusive_snr",
    "secondary_partition_snr",
    "note",
];

const SUMMARY_FIELDS: &[&str] = &[
    "mask_mode",
    "frame_count",
    "assigned_two_count",
    "assigned_one_count",
    "assigned_zero_count",
    "quality_two_count",
    "assigned_two_fraction",
    "quality_two_fraction",
    "masked_pixel_median",
    "primary_shared_positive_fraction_median",
    "secondary_shared_positive_fraction_median",
    "primary_exclusive_snr_median",
    "secondary_exclusive_snr_median",
    "primary_partition_snr_median",
    "secondary_partition_snr_median",
];

#[derive(Debug, Clone, Serialize)]
struct SupportMetrics {
    background_adu: Option<f64>,
    noise_adu: Option<f64>,
    total_valid_pixels: usize,
    shared_valid_pixels: usize,
    exclusive_valid_pixels: usize,
    partition_valid_pixels: usize,
    total_net_flux_adu: Option<f64>,
    shared_net_flux_adu: Option<f64>,
    exclusive_net_flux_adu: Option<f64>,
    partition_net_flux_adu: Option<f64>,
    total_positive_excess_adu: Option<f64>,
    shared_positive_excess_adu: Option<f64>,
    exclusive_positive_excess_adu: Option<f64>,
    partition_positive_excess_adu: Option<f64>,
    shared_positive_fraction: Option<f64>,
    exclusive_snr: Option<f64>,
    partition_snr: Option<f64>,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
struct AuditRow {
    frame_index: usize,
    path: String,
    frame_shift_x_px: f64,
    frame_shift_y_px: f64,
    mask_mode: String,
    crop_x0: usize,
    crop_y0: usize,
    crop_width: usize,
    crop_height: usize,
    masked_pixel_count: usize,
    local_negative_limit_adu: Option<f64>,
    candidate_count: usize,
    quality_count: usize,
    nearby_candidate_count: usize,
    nearby_quality_count: usize,
    assigned_candidate_count: usize,
    primary_re_detected_id: Option<i64>,
    secondary_re_detected_id: Option<i64>,
    primary_match_distance_px: Option<f64>,
    secondary_match_distance_px: Option<f64>,
    primary_local_quality_passed: Option<bool>,
    secondary_local_quality_passed: Option<bool>,
    primary_local_flags: Vec<String>,
    secondary_local_flags: Vec<String>,
    primary_catalog_quality_passed: Option<bool>,
    secondary_catalog_quality_passed: Option<bool>,
    primary_catalog_flags: Vec<String>,
    secondary_catalog_flags: Vec<String>,
    primary_background_adu: Option<f64>,
    primary_noise_adu: Option<f64>,
    primary_total_valid_pixels: usize,
    primary_shared_valid_pixels: usize,
    primary_exclusive_valid_pixels: usize,
    primary_partition_valid_pixels: usize,
    primary_total_net_flux_adu: Option<f64>,
    primary_shared_net_flux_adu: Option<f64>,
    primary_exclusive_net_flux_adu: Option<f64>,
    primary_partition_net_flux_adu: Option<f64>,
    primary_total_positive_excess_adu: Option<f64>,
    primary_shared_positive_excess_adu: Option<f64>,
    primary_exclusive_positive_excess_adu: Option<f64>,
    primary_partition_positive_excess_adu: Option<f64>,
    primary_shared_positive_fraction: Option<f64>,
    primary_exclusive_snr: Option<f64>,
    primary_partition_snr: Option<f64>,
    secondary_background_adu: Option<f64>,
    secondary_noise_adu: Option<f64>,
    secondary_total_valid_pixels: usize,
    secondary_shared_valid_pixels: usize,
    secondary_exclusive_valid_pixels: usize,
    secondary_partition_valid_pixels: usize,
    secondary_total_net_flux_adu: Option<f64>,
    secondary_shared_net_flux_adu: Option<f64>,
    secondary_exclusive_net_flux_adu: Option<f64>,
    secondary_partition_net_flux_adu: Option<f64>,
    secondary_total_positive_excess_adu: Option<f64>,
    secondary_shared_positive_excess_adu: Option<f64>,
    secondary_exclusive_positive_excess_adu: Option<f64>,
    secondary_partition_positive_excess_adu: Option<f64>,
    secondary_shared_positive_fraction: Option<f64>,
    secondary_exclusive_snr: Option<f64>,
    secondary_partition_snr: Option<f64>,
    note: String,
}

impl AuditRow {
    fn as_value(&self) -> Value {
        serde_json::to_value(self).expect("audit row is JSON serializable")
    }
}

#[derive(Debug, Clone)]
struct CatalogSource {
    detection_id: i64,
    x: f64,
    y: f64,
    quality_passed: bool,
    flags: Vec<String>,
    repeated_code_values: Vec<i64>,
}

#[derive(Debug, Clone, Copy)]
struct Assignment {
    source_index: usize,
    distance: f64,
}

#[derive(Debug, Clone, Copy)]
struct PixelMetrics {
    count: usize,
    net: Option<f64>,
    positive: Option<f64>,
    snr: Option<f64>,
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

fn population_std(values: &[f64], center: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn robust_location_scale(values: &[f64], mask: &[bool]) -> Result<(f64, f64)> {
    if values.len() != mask.len() {
        return Err("pair crop and mask shapes differ".into());
    }
    let mut samples: Vec<_> = values
        .iter()
        .zip(mask)
        .filter_map(|(value, invalid)| (value.is_finite() && !invalid).then_some(*value))
        .collect();
    let center = median(&mut samples).ok_or("pair crop has no valid pixels")?;
    let mut deviations: Vec<_> = samples.iter().map(|value| (value - center).abs()).collect();
    let mad = median(&mut deviations).unwrap_or(0.0) * 1.4826;
    let epsilon = f64::EPSILON;
    let noise = if !mad.is_finite() || mad <= epsilon {
        population_std(&samples, center).max(epsilon)
    } else {
        mad
    };
    Ok((center, noise))
}

fn crop_data(
    data: &PixelData,
    width: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> PixelData {
    macro_rules! crop_variant {
        ($values:expr, $variant:ident) => {{
            let mut cropped = Vec::with_capacity((x1 - x0) * (y1 - y0));
            for y in y0..y1 {
                let start = y * width + x0;
                cropped.extend_from_slice(&$values[start..start + (x1 - x0)]);
            }
            PixelData::$variant(cropped)
        }};
    }
    match data {
        PixelData::U8(values) => crop_variant!(values, U8),
        PixelData::I16(values) => crop_variant!(values, I16),
        PixelData::I32(values) => crop_variant!(values, I32),
        PixelData::I64(values) => crop_variant!(values, I64),
        PixelData::F32(values) => crop_variant!(values, F32),
        PixelData::F64(values) => crop_variant!(values, F64),
    }
}

fn negative_overflow_limit(data: &PixelData) -> Option<f64> {
    match data {
        PixelData::I16(_) => Some(-0.9 * i16::MAX as f64),
        PixelData::I32(_) => Some(-0.9 * i32::MAX as f64),
        PixelData::I64(_) => Some(-0.9 * i64::MAX as f64),
        _ => None,
    }
}

fn repeated_code_mask(data: &PixelData, codes: &[i64]) -> Vec<bool> {
    if codes.is_empty() {
        return vec![false; data.len()];
    }
    (0..data.len())
        .map(|index| codes.iter().any(|code| data.value(index) == *code as f64))
        .collect()
}

fn mask_components(
    data: &PixelData,
    width: usize,
    height: usize,
    origin: (usize, usize),
    positions: [(f64, f64); 2],
    radius: usize,
    codes: &[i64],
    overflow: Option<f64>,
) -> Result<(BTreeMap<String, Vec<bool>>, Option<f64>)> {
    let values = data.to_f64();
    let base: Vec<_> = values.iter().map(|value| !value.is_finite()).collect();
    let (background, noise) = robust_location_scale(&values, &base)?;
    let negative_limit = (-1000.0f64).min(background - 50.0 * noise);
    if width.checked_mul(height) != Some(data.len()) || width == 0 || height == 0 {
        return Err("pair crop dimensions do not match pixel data".into());
    }
    let repeated = repeated_code_mask(data, codes);
    let mut masks = BTreeMap::new();
    for mode in MASK_MODES {
        let mut mask = base.clone();
        for index in 0..values.len() {
            let x = (index % width) as f64 + origin.0 as f64;
            let y = (index / width) as f64 + origin.1 as f64;
            let primary_distance = (x - positions[0].0).hypot(y - positions[0].1);
            let secondary_distance = (x - positions[1].0).hypot(y - positions[1].1);
            let shared_aperture =
                primary_distance <= radius as f64 && secondary_distance <= radius as f64;
            let local_anomaly = values[index].is_finite() && values[index] <= negative_limit;
            let range = overflow.is_some_and(|limit| {
                values[index].is_finite() && (values[index] <= limit || values[index] >= -limit)
            });
            let include = match mode {
                "raw" => false,
                "range_masked" => range,
                "repeated_code_masked" => repeated[index],
                "local_anomaly_masked" => local_anomaly,
                "local_anomaly_repeated_code_masked" => local_anomaly || repeated[index],
                "shared_aperture_masked" => shared_aperture,
                "shared_local_anomaly_repeated_code_masked" => {
                    shared_aperture || local_anomaly || repeated[index]
                }
                _ => false,
            };
            mask[index] |= include;
        }
        masks.insert(mode.to_owned(), mask);
    }
    Ok((masks, Some(negative_limit)))
}

fn support_region_metrics(
    values: &[f64],
    mask: &[bool],
    region: &[bool],
    background: f64,
    noise: f64,
    annulus_count: usize,
) -> PixelMetrics {
    let residuals: Vec<_> = values
        .iter()
        .zip(mask)
        .zip(region)
        .filter_map(|((value, invalid), inside)| {
            (*inside && value.is_finite() && !invalid).then_some(*value - background)
        })
        .collect();
    let count = residuals.len();
    if count == 0 {
        return PixelMetrics {
            count,
            net: None,
            positive: None,
            snr: None,
        };
    }
    let net = residuals.iter().sum::<f64>();
    let positive = residuals.iter().map(|value| value.max(0.0)).sum::<f64>();
    let error =
        noise * (count as f64 + (count * count) as f64 / annulus_count.max(1) as f64).sqrt();
    PixelMetrics {
        count,
        net: Some(net),
        positive: Some(positive),
        snr: Some(net / error.max(f64::EPSILON)),
    }
}

fn support_metrics(
    values: &[f64],
    mask: &[bool],
    width: usize,
    height: usize,
    origin: (usize, usize),
    target: (f64, f64),
    other: (f64, f64),
    radius: usize,
) -> Result<SupportMetrics> {
    if values.len() != width * height || mask.len() != values.len() {
        return Err("support image and mask shapes differ".into());
    }
    let mut aperture = vec![false; values.len()];
    let mut shared = vec![false; values.len()];
    let mut exclusive = vec![false; values.len()];
    let mut partition = vec![false; values.len()];
    let mut annulus = vec![false; values.len()];
    let mut base_mask = mask.to_vec();
    for i in 0..values.len() {
        base_mask[i] |= !values[i].is_finite();
        let x = (i % width) as f64 + origin.0 as f64;
        let y = (i / width) as f64 + origin.1 as f64;
        let dt = (x - target.0).hypot(y - target.1);
        let do_ = (x - other.0).hypot(y - other.1);
        aperture[i] = dt <= radius as f64;
        shared[i] = aperture[i] && do_ <= radius as f64;
        exclusive[i] = aperture[i] && do_ > radius as f64;
        partition[i] = (aperture[i] || do_ <= radius as f64) && dt <= do_;
        annulus[i] = dt >= (radius + 2) as f64 && dt <= (radius + 6) as f64 && !base_mask[i];
    }
    let mut annulus_values: Vec<_> = (0..values.len())
        .filter_map(|i| annulus[i].then_some(values[i]))
        .collect();
    let (background, noise, annulus_count) = if !annulus_values.is_empty() {
        let bg = median(&mut annulus_values).unwrap();
        let mut deviations: Vec<_> = annulus_values
            .iter()
            .map(|value| (value - bg).abs())
            .collect();
        let mad = median(&mut deviations).unwrap_or(0.0) * 1.4826;
        let sigma = if !mad.is_finite() || mad <= f64::EPSILON {
            population_std(&annulus_values, bg).max(f64::EPSILON)
        } else {
            mad
        };
        (bg, sigma, annulus_values.len())
    } else {
        let fallback: Vec<_> = (0..values.len())
            .filter_map(|i| (!base_mask[i]).then_some(values[i]))
            .collect();
        if fallback.is_empty() {
            return Ok(empty_support_metrics());
        }
        let mut copy = fallback.clone();
        let bg = median(&mut copy).unwrap();
        (bg, population_std(&fallback, bg).max(f64::EPSILON), 0)
    };
    let total = support_region_metrics(
        values,
        &base_mask,
        &aperture,
        background,
        noise,
        annulus_count,
    );
    let shared_metrics = support_region_metrics(
        values,
        &base_mask,
        &shared,
        background,
        noise,
        annulus_count,
    );
    let exclusive_metrics = support_region_metrics(
        values,
        &base_mask,
        &exclusive,
        background,
        noise,
        annulus_count,
    );
    let partition_metrics = support_region_metrics(
        values,
        &base_mask,
        &partition,
        background,
        noise,
        annulus_count,
    );
    let shared_fraction = match (shared_metrics.positive, total.positive) {
        (Some(shared), Some(total)) if total > 0.0 => Some(shared / total),
        _ => None,
    };
    Ok(SupportMetrics {
        background_adu: Some(background),
        noise_adu: Some(noise),
        total_valid_pixels: total.count,
        shared_valid_pixels: shared_metrics.count,
        exclusive_valid_pixels: exclusive_metrics.count,
        partition_valid_pixels: partition_metrics.count,
        total_net_flux_adu: total.net,
        shared_net_flux_adu: shared_metrics.net,
        exclusive_net_flux_adu: exclusive_metrics.net,
        partition_net_flux_adu: partition_metrics.net,
        total_positive_excess_adu: total.positive,
        shared_positive_excess_adu: shared_metrics.positive,
        exclusive_positive_excess_adu: exclusive_metrics.positive,
        partition_positive_excess_adu: partition_metrics.positive,
        shared_positive_fraction: shared_fraction,
        exclusive_snr: exclusive_metrics.snr,
        partition_snr: partition_metrics.snr,
    })
}

fn empty_support_metrics() -> SupportMetrics {
    SupportMetrics {
        background_adu: None,
        noise_adu: None,
        total_valid_pixels: 0,
        shared_valid_pixels: 0,
        exclusive_valid_pixels: 0,
        partition_valid_pixels: 0,
        total_net_flux_adu: None,
        shared_net_flux_adu: None,
        exclusive_net_flux_adu: None,
        partition_net_flux_adu: None,
        total_positive_excess_adu: None,
        shared_positive_excess_adu: None,
        exclusive_positive_excess_adu: None,
        partition_positive_excess_adu: None,
        shared_positive_fraction: None,
        exclusive_snr: None,
        partition_snr: None,
    }
}

fn add_metrics(map: &mut Map<String, Value>, prefix: &str, metrics: &SupportMetrics) {
    let value = serde_json::to_value(metrics).expect("support metrics are serializable");
    for (name, item) in value.as_object().unwrap() {
        map.insert(format!("{prefix}_{name}"), item.clone());
    }
}

fn assignment_source(sources: &[Detection], assignment: Option<Assignment>) -> Option<&Detection> {
    assignment.map(|item| &sources[item.source_index])
}

fn assignment_flags(sources: &[Detection], assignment: Option<Assignment>) -> Vec<String> {
    assignment_source(sources, assignment)
        .map(|source| source.flags.clone())
        .unwrap_or_default()
}

fn row_note(
    mode: &str,
    assignments: [Option<Assignment>; 2],
    metrics: [&SupportMetrics; 2],
) -> String {
    let assignment_note = if assignments.iter().all(Option::is_some) {
        "一对一局部重检同时分配了两个候选"
    } else {
        "一对一局部重检未同时分配两个目标"
    };
    let mut fractions: Vec<_> = metrics
        .iter()
        .filter_map(|metric| metric.shared_positive_fraction)
        .collect();
    let fraction_note = if let Some(value) = median(&mut fractions) {
        format!("共同孔径正残差占比中位约 {:.1}%", value * 100.0)
    } else {
        "共同孔径正残差占比不可计算".to_owned()
    };
    if mode == "raw" {
        format!("{assignment_note}；{fraction_note}；raw 只作候选层基线")
    } else {
        format!("{assignment_note}；{fraction_note}；掩膜敏感性是诊断证据，不是物理真值")
    }
}

fn frame_row(
    frame_index: usize,
    frame: &FitsFrame,
    shift: (f64, f64),
    mode: &str,
    bounds: (usize, usize, usize, usize),
    masked_pixel_count: usize,
    local_negative_limit: Option<f64>,
    candidate_count: usize,
    quality_count: usize,
    nearby: &[&Detection],
    assignments: [Option<Assignment>; 2],
    sources: &[Detection],
    primary_catalog: Option<&CatalogSource>,
    secondary_catalog: Option<&CatalogSource>,
    primary: &SupportMetrics,
    secondary: &SupportMetrics,
) -> Value {
    let (x0, y0, x1, y1) = bounds;
    let mut row = json!({
        "frame_index":frame_index,"path":frame.path.to_string_lossy(),
        "frame_shift_x_px":shift.0,"frame_shift_y_px":shift.1,"mask_mode":mode,
        "crop_x0":x0,"crop_y0":y0,"crop_width":x1-x0,"crop_height":y1-y0,
        "masked_pixel_count":masked_pixel_count,"local_negative_limit_adu":local_negative_limit,
        "candidate_count":candidate_count,"quality_count":quality_count,
        "nearby_candidate_count":nearby.len(),
        "nearby_quality_count":nearby.iter().filter(|source| source.quality_passed).count(),
        "assigned_candidate_count":assignments.iter().filter(|assignment| assignment.is_some()).count(),
        "primary_re_detected_id":assignment_source(sources, assignments[0]).map(|source| source.detection_id),
        "secondary_re_detected_id":assignment_source(sources, assignments[1]).map(|source| source.detection_id),
        "primary_match_distance_px":assignments[0].map(|assignment| assignment.distance),
        "secondary_match_distance_px":assignments[1].map(|assignment| assignment.distance),
        "primary_local_quality_passed":assignment_source(sources, assignments[0]).map(|source| source.quality_passed),
        "secondary_local_quality_passed":assignment_source(sources, assignments[1]).map(|source| source.quality_passed),
        "primary_local_flags":assignment_flags(sources, assignments[0]),
        "secondary_local_flags":assignment_flags(sources, assignments[1]),
        "primary_catalog_quality_passed":primary_catalog.map(|source| source.quality_passed),
        "secondary_catalog_quality_passed":secondary_catalog.map(|source| source.quality_passed),
        "primary_catalog_flags":primary_catalog.map(|source| source.flags.clone()).unwrap_or_default(),
        "secondary_catalog_flags":secondary_catalog.map(|source| source.flags.clone()).unwrap_or_default(),
        "note":row_note(mode, assignments, [primary, secondary]),
    });
    let map = row.as_object_mut().unwrap();
    add_metrics(map, "primary", primary);
    add_metrics(map, "secondary", secondary);
    let typed: AuditRow = serde_json::from_value(row).expect("pair support row matches its schema");
    typed.as_value()
}

fn run_audit(
    paths: &[PathBuf],
    primary: (f64, f64),
    secondary: (f64, f64),
    sources: &[CatalogSource],
    primary_id: i64,
    secondary_id: i64,
    shifts: &[(f64, f64)],
    modes: &[String],
    codes: &[i64],
    padding: f64,
    radius: usize,
    match_radius: f64,
    threshold: f64,
    min_distance: usize,
    psf_fwhm: f64,
    background_box: usize,
    min_flux_snr: f64,
    min_psf_support: usize,
    proposal_mode: &str,
    reject_linear: bool,
    report_progress: bool,
) -> Result<Value> {
    if paths.is_empty() {
        return Err("frame_paths must contain at least one frame".into());
    }
    if primary == secondary {
        return Err("primary and secondary target coordinates must differ".into());
    }
    if radius < 1 {
        return Err("aperture_radius must be positive".into());
    }
    if !psf_fwhm.is_finite() || psf_fwhm <= 0.0 {
        return Err("psf_fwhm must be positive and finite".into());
    }
    if shifts.len() != paths.len() {
        return Err("frame_shifts length must match frame paths".into());
    }
    if !match_radius.is_finite() || match_radius <= 0.0 {
        return Err("match_radius_px must be positive and finite".into());
    }
    if !threshold.is_finite()
        || threshold <= 0.0
        || min_distance == 0
        || background_box < 16
        || !min_flux_snr.is_finite()
        || min_flux_snr <= 0.0
        || min_psf_support == 0
        || min_psf_support > 9
    {
        return Err("invalid detector threshold, spacing, PSF or background parameters".into());
    }
    let mut rows = Vec::new();
    let source_by_id: BTreeMap<_, _> = sources
        .iter()
        .map(|source| (source.detection_id, source))
        .collect();
    for (offset, (path, shift)) in paths.iter().zip(shifts).enumerate() {
        let frame = fits::read_fits(path, false)?;
        let positions = [
            (primary.0 + shift.0, primary.1 + shift.1),
            (secondary.0 + shift.0, secondary.1 + shift.1),
        ];
        let bounds = crop_bounds(frame.width, frame.height, positions, padding)?;
        let (x0, y0, x1, y1) = bounds;
        let width = x1 - x0;
        let height = y1 - y0;
        let crop = crop_data(&frame.data, frame.width, x0, y0, x1, y1);
        let overflow = negative_overflow_limit(&frame.data);
        let (masks, local_negative_limit) = mask_components(
            &crop,
            width,
            height,
            (x0, y0),
            positions,
            radius,
            codes,
            overflow,
        )?;
        for mode in modes {
            let mask = &masks[mode];
            let mut options = DetectionOptions::default();
            options.threshold_sigma = threshold;
            options.min_distance = min_distance;
            options.measurement.aperture_radius = radius;
            options.measurement.psf_fwhm = psf_fwhm;
            options.background_box_size = background_box;
            options.measurement.min_flux_snr = min_flux_snr;
            options.measurement.min_psf_support_pixels = min_psf_support;
            options.measurement.negative_overflow_limit = overflow;
            options.mask_zero_pixels = Some(false);
            options.allow_partial_zero_mask = Some(false);
            options.reject_linear_artifacts = reject_linear;
            options.proposal_mode = proposal_mode.to_owned();
            options.enable_local_deblend = false;
            let detected =
                detection::detect_sources(&crop, Some(mask), height, width, &options, None)?;
            let global_sources: Vec<_> = detected
                .sources
                .into_iter()
                .map(|mut source| {
                    source.x += x0 as f64;
                    source.y += y0 as f64;
                    source.peak_x = source.peak_x.map(|value| value + x0 as f64);
                    source.peak_y = source.peak_y.map(|value| value + y0 as f64);
                    source
                })
                .collect();
            let assignments = assign_targets(&global_sources, positions, match_radius)?;
            let nearby_radius = match_radius.max(5.0);
            let nearby: Vec<_> = global_sources
                .iter()
                .filter(|source| {
                    let first = (source.x - positions[0].0).hypot(source.y - positions[0].1);
                    let second = (source.x - positions[1].0).hypot(source.y - positions[1].1);
                    first.min(second) <= nearby_radius
                })
                .collect();
            let values = crop.to_f64();
            let primary_metrics = support_metrics(
                &values,
                mask,
                width,
                height,
                (x0, y0),
                positions[0],
                positions[1],
                radius,
            )?;
            let secondary_metrics = support_metrics(
                &values,
                mask,
                width,
                height,
                (x0, y0),
                positions[1],
                positions[0],
                radius,
            )?;
            let row = frame_row(
                offset + 1,
                &frame,
                *shift,
                mode,
                bounds,
                mask.iter().filter(|value| **value).count(),
                local_negative_limit,
                detected.candidate_count,
                detected.quality_count,
                &nearby,
                assignments,
                &global_sources,
                source_by_id.get(&primary_id).copied(),
                source_by_id.get(&secondary_id).copied(),
                &primary_metrics,
                &secondary_metrics,
            );
            rows.push(row);
        }
        if report_progress {
            eprintln!("pair support audit: frame {}/{}", offset + 1, paths.len());
        }
    }
    let by_mode = |name: &str| {
        rows.iter()
            .filter(|row| row["mask_mode"] == name)
            .collect::<Vec<_>>()
    };
    let raw = by_mode("raw");
    let repeated = by_mode("repeated_code_masked");
    let local = by_mode("local_anomaly_masked");
    let two_count = |items: &[&Value]| {
        items
            .iter()
            .filter(|row| row["assigned_candidate_count"] == 2)
            .count()
    };
    let raw_two = two_count(&raw);
    let repeated_two = two_count(&repeated);
    let local_two = two_count(&local);
    let mut shared: Vec<_> = raw
        .iter()
        .flat_map(|row| {
            [
                row["primary_shared_positive_fraction"].as_f64(),
                row["secondary_shared_positive_fraction"].as_f64(),
            ]
        })
        .flatten()
        .collect();
    let shared_note = if let Some(value) = median(&mut shared) {
        format!("raw 共同孔径正残差占比中位 {:.1}%", value * 100.0)
    } else {
        "raw 共同孔径正残差占比不可计算".to_owned()
    };
    let conclusion = format!(
        "局部双候选留出审计完成：{} 帧；raw 模式有两个一对一候选的帧数 {}/{}，重复码屏蔽后 {}/{}，局部负异常屏蔽后 {}/{}；{}。这说明 raw 双计数对共同孔径和数据有效性控制的敏感性，不能直接解释为两颗恒星；共同孔径屏蔽会损伤真实双源核心，因此仍需用有效独占支持、PSF/星表和跨帧证据共同裁决。",
        paths.len(), raw_two, raw.len(), repeated_two, repeated.len(), local_two, local.len(), shared_note
    );
    let parameters = json!({
        "detector":"current detect_sources on local crop", "threshold_sigma":threshold,
        "min_distance":min_distance,"aperture_radius":radius,"psf_fwhm":psf_fwhm,
        "background_box_size":background_box,"min_flux_snr":min_flux_snr,
        "min_psf_support_pixels":min_psf_support,"proposal_mode":proposal_mode,
        "reject_linear_artifacts":reject_linear,
        "coordinate_note":"目标坐标为首帧 detector 质心，按 frame_shifts 平移；局部候选坐标已加回全图原点",
        "assignment_note":"每个候选最多分给一个目标；先最大化匹配数，再最小化距离",
        "support_note":"shared/exclusive 是几何分区；partition 在 union aperture 上按最近目标单计数",
        "raw_quality_boundary":"局部 raw 重检没有全幅重复码频次先验；catalog flags 单独保留",
        "interpretation_boundary":"掩膜后的候选数、通量分区和 SNR 是诊断量，不是恒星概率或物理真值",
    });
    Ok(json!({
        "frame_count":paths.len(),"primary_detection_id":primary_id,"secondary_detection_id":secondary_id,
        "primary_target_xy":[primary.0,primary.1],"secondary_target_xy":[secondary.0,secondary.1],
        "frame_shifts":shifts.iter().map(|(x,y)| json!([x,y])).collect::<Vec<_>>(),
        "mask_modes":modes,"repeated_code_values":codes,"crop_padding_px":padding,
        "aperture_radius":radius,"match_radius_px":match_radius,"rows":rows,
        "parameters":parameters,"conclusion":conclusion,
    }))
}

fn parse_float(values: &BTreeMap<String, String>, key: &str, default: f64) -> Result<f64> {
    values.get(key).map_or(Ok(default), |value| {
        value
            .parse::<f64>()
            .map_err(|_| format!("invalid value for --{key}: {value:?}"))
    })
}

fn parse_usize(values: &BTreeMap<String, String>, key: &str, default: usize) -> Result<usize> {
    values.get(key).map_or(Ok(default), |value| {
        value
            .parse::<usize>()
            .map_err(|_| format!("invalid value for --{key}: {value:?}"))
    })
}

fn parse_i64(values: &BTreeMap<String, String>, key: &str) -> Result<i64> {
    values
        .get(key)
        .ok_or_else(|| format!("--{key} is required"))?
        .parse::<i64>()
        .map_err(|_| format!("invalid value for --{key}"))
}

fn parse_cli(args: &[String]) -> Result<(String, BTreeMap<String, String>, bool)> {
    let mut positional = None;
    let mut values = BTreeMap::new();
    let mut keep_linear = false;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--keep-linear-artifacts" {
            keep_linear = true;
        } else if let Some(option) = arg.strip_prefix("--") {
            if let Some((key, value)) = option.split_once('=') {
                values.insert(key.replace('-', "_"), value.to_owned());
            } else {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| format!("--{option} requires a value"))?;
                if value.starts_with("--") {
                    return Err(format!("--{option} requires a value"));
                }
                values.insert(option.replace('-', "_"), value.clone());
                index += 1;
            }
        } else if positional.replace(arg.clone()).is_some() {
            return Err("unexpected additional positional argument".into());
        }
        index += 1;
    }
    Ok((
        positional.ok_or("data_dir is required")?,
        values,
        keep_linear,
    ))
}

fn output_path(args: &[String], out_dir: &Path) -> PathBuf {
    args.iter()
        .position(|arg| arg == "--json-out")
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
        .or_else(|| {
            args.iter()
                .find_map(|arg| arg.strip_prefix("--json-out=").map(PathBuf::from))
        })
        .unwrap_or_else(|| out_dir.join("result.json"))
}

fn write_artifacts(result: &Value, out_dir: &Path) -> Result<(PathBuf, Value)> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let output = fs::canonicalize(out_dir).map_err(|error| error.to_string())?;
    let rows = result["rows"]
        .as_array()
        .ok_or("audit result has no rows")?;
    write_csv(&output.join("pair_support_audit.csv"), ROW_FIELDS, rows)?;
    let modes: Vec<String> =
        serde_json::from_value(result["mask_modes"].clone()).map_err(|error| error.to_string())?;
    let summary = summarize(rows, &modes);
    write_csv(
        &output.join("pair_support_summary.csv"),
        SUMMARY_FIELDS,
        &summary,
    )?;
    let mut payload = result.clone();
    payload["summary"] = json!(summary);
    payload["outputs"] = json!([
        "pair_support_audit.csv",
        "pair_support_summary.csv",
        "pair_support_audit.json"
    ]);
    payload["note"] = json!("该审计把局部重检、共同/非共同孔径和留出掩膜作为相互独立的控制；它不修改默认候选表，也不把屏蔽后消失直接解释为噪点。");
    let json_path = output.join("pair_support_audit.json");
    let mut content = serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?;
    content.push('\n');
    fs::write(&json_path, content).map_err(|error| error.to_string())?;
    Ok((output, payload))
}

/// Run the Rust-only command-line contract used by the desktop native worker.
pub fn run_desktop(args: &[String]) -> Result<Value> {
    let (data_dir, values, keep_linear) = parse_cli(args)?;
    let data_dir = PathBuf::from(data_dir);
    let mut paths = fs::read_dir(&data_dir)
        .map_err(|error| error.to_string())?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("fits"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("数据目录没有 FITS 文件：{}", data_dir.display()));
    }
    let catalog_path = values
        .get("source_catalog")
        .ok_or("--source-catalog is required")?;
    let catalog = read_source_catalog(Path::new(catalog_path))?;
    let primary_id = parse_i64(&values, "primary_id")?;
    let secondary_id = parse_i64(&values, "secondary_id")?;
    let by_id: BTreeMap<_, _> = catalog
        .iter()
        .map(|source| (source.detection_id, source))
        .collect();
    let primary_source = by_id
        .get(&primary_id)
        .ok_or_else(|| format!("source catalog 中不存在 detection_id={primary_id}"))?;
    let secondary_source = by_id
        .get(&secondary_id)
        .ok_or_else(|| format!("source catalog 中不存在 detection_id={secondary_id}"))?;
    let modes = resolve_modes(
        values
            .get("mask_modes")
            .map(String::as_str)
            .unwrap_or(&MASK_MODES.join(",")),
    )?;
    let shifts = resolve_shifts(values.get("shifts_json").map(String::as_str), paths.len())?;
    let codes = if let Some(text) = values.get("repeated_code_values") {
        let mut codes: Vec<i64> = text
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(|item| {
                item.parse::<i64>()
                    .map_err(|_| format!("invalid repeated code value: {item:?}"))
            })
            .collect::<Result<_>>()?;
        codes.sort_unstable();
        codes.dedup();
        codes
    } else {
        let mut codes: Vec<_> = catalog
            .iter()
            .flat_map(|source| source.repeated_code_values.iter().copied())
            .collect();
        codes.sort_unstable();
        codes.dedup();
        codes
    };
    let padding = parse_float(&values, "crop_padding_px", 128.0)?;
    let radius = parse_usize(&values, "aperture_radius", 4)?;
    let match_radius = parse_float(&values, "match_radius_px", 2.0)?;
    let threshold = parse_float(&values, "threshold_sigma", 4.0)?;
    let min_distance = parse_usize(&values, "min_distance", 4)?;
    let psf_fwhm = parse_float(&values, "psf_fwhm", 2.0)?;
    let background_box = parse_usize(&values, "background_box_size", 128)?;
    let min_flux_snr = parse_float(&values, "min_flux_snr", 5.0)?;
    let min_psf_support = parse_usize(&values, "min_psf_support_pixels", 3)?;
    let proposal_mode = values
        .get("proposal_mode")
        .map(String::as_str)
        .unwrap_or("hybrid");
    if !["gaussian", "hybrid", "ensemble"].contains(&proposal_mode) {
        return Err("invalid proposal_mode".into());
    }
    let audit = run_audit(
        &paths,
        (primary_source.x, primary_source.y),
        (secondary_source.x, secondary_source.y),
        &catalog,
        primary_id,
        secondary_id,
        &shifts,
        &modes,
        &codes,
        padding,
        radius,
        match_radius,
        threshold,
        min_distance,
        psf_fwhm,
        background_box,
        min_flux_snr,
        min_psf_support,
        proposal_mode,
        !keep_linear,
        true,
    )?;
    let out_dir = PathBuf::from(
        values
            .get("out_dir")
            .map(String::as_str)
            .unwrap_or("tmp/pair-support-audit"),
    );
    let (output, result) = write_artifacts(&audit, &out_dir)?;
    let result_path = output_path(args, &output);
    if let Some(parent) = result_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut result_text =
        serde_json::to_string_pretty(&result).map_err(|error| error.to_string())?;
    result_text.push('\n');
    fs::write(&result_path, result_text).map_err(|error| error.to_string())?;
    Ok(
        json!({"tool":"rst19-pair-support-audit","out_dir":output,"result_path":result_path,"result":result}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(id: i64, x: f64, y: f64) -> Detection {
        Detection {
            detection_id: id,
            x,
            y,
            peak: 100.0,
            flux: 500.0,
            background: 20.0,
            noise: 3.0,
            snr: 30.0,
            fwhm: Some(2.0),
            flags: vec![],
            flux_error: None,
            flux_snr: Some(20.0),
            filter_snr: Some(30.0),
            fwhm_x: None,
            fwhm_y: None,
            ellipticity: None,
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

    #[test]
    fn assignment_does_not_reuse_one_candidate() {
        let result = assign_targets(&[source(1, 1.0, 0.0)], [(0.0, 0.0), (2.0, 0.0)], 2.0).unwrap();
        assert_eq!(result.iter().filter(|item| item.is_some()).count(), 1);
    }

    #[test]
    fn support_metrics_separate_shared_and_exclusive_aperture() {
        let mut values = vec![20.0; 41 * 41];
        values[20 * 41 + 20] = 120.0;
        values[20 * 41 + 23] = 80.0;
        values[20 * 41 + 16] = 60.0;
        let metrics = support_metrics(
            &values,
            &vec![false; values.len()],
            41,
            41,
            (0, 0),
            (20.0, 20.0),
            (23.0, 20.0),
            4,
        )
        .unwrap();
        assert!(
            metrics.total_valid_pixels > metrics.exclusive_valid_pixels
                && metrics.exclusive_valid_pixels > 0
        );
        assert!(metrics.partition_valid_pixels > 0);
        assert!(metrics
            .shared_positive_fraction
            .is_some_and(|value| value > 0.0 && value < 1.0));
    }

    #[test]
    fn holdout_masks_include_expected_shared_and_code_pixels() {
        let mut values: Vec<i16> = (0..41 * 41).map(|index| 17 + (index % 7) as i16).collect();
        values[20 * 41 + 20] = -2000;
        values[20 * 41 + 21] = 700;
        let data = PixelData::I16(values);
        let (masks, limit) = mask_components(
            &data,
            41,
            41,
            (0, 0),
            [(20.0, 20.0), (23.0, 20.0)],
            4,
            &[700],
            Some(-29490.0),
        )
        .unwrap();
        let center = 20 * 41 + 20;
        let code = 20 * 41 + 21;
        assert!(masks["shared_aperture_masked"][center]);
        assert!(masks["local_anomaly_masked"][center]);
        assert!(masks["repeated_code_masked"][code]);
        assert!(limit.unwrap() <= -1000.0);
        assert!(!masks["raw"][code]);
    }
}

fn float_text(value: f64) -> String {
    if value == 0.0 && value.is_sign_negative() {
        "-0.0".into()
    } else if value == 0.0 {
        "0.0".into()
    } else {
        let rendered = value.to_string();
        if rendered.contains(['.', 'e', 'E']) {
            rendered
        } else {
            format!("{rendered}.0")
        }
    }
}

fn optional_f64(
    row: &csv::StringRecord,
    indexes: &BTreeMap<String, usize>,
    name: &str,
) -> Result<Option<f64>> {
    let Some(index) = indexes.get(name) else {
        return Ok(None);
    };
    let text = row.get(*index).unwrap_or("").trim();
    if text.is_empty() {
        return Ok(None);
    }
    let value = text
        .parse::<f64>()
        .map_err(|_| format!("source catalog field {name:?} is not numeric: {text:?}"))?;
    Ok(value.is_finite().then_some(value))
}

fn required_f64(
    row: &csv::StringRecord,
    indexes: &BTreeMap<String, usize>,
    name: &str,
) -> Result<f64> {
    optional_f64(row, indexes, name)?
        .ok_or_else(|| format!("source catalog field {name:?} is required"))
}

fn parse_pipe(text: &str) -> Vec<String> {
    text.split('|')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect()
}

fn parse_pipe_ints(text: &str) -> Result<Vec<i64>> {
    text.split('|')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| {
            item.parse::<i64>()
                .map_err(|_| format!("source catalog list contains a non-integer value: {item:?}"))
        })
        .collect()
}

fn read_source_catalog(path: &Path) -> Result<Vec<CatalogSource>> {
    let text = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let text = text.trim_start_matches('\u{feff}');
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let indexes: BTreeMap<_, _> = headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.to_owned(), index))
        .collect();
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
    let missing: Vec<_> = required
        .iter()
        .filter(|name| !indexes.contains_key(**name))
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let mut sources = Vec::new();
    for (offset, item) in reader.records().enumerate() {
        let row = item.map_err(|error| error.to_string())?;
        let line = offset + 2;
        let parsed = (|| {
            let id = optional_f64(&row, &indexes, "detection_id")?
                .ok_or("detection_id is required")? as i64;
            let quality_text = indexes
                .get("quality_passed")
                .and_then(|index| row.get(*index))
                .unwrap_or("True")
                .trim()
                .to_ascii_lowercase();
            let quality_passed = match quality_text.as_str() {
                "" | "true" | "1" | "yes" | "y" | "是" => true,
                "false" | "0" | "no" | "n" | "否" => false,
                _ => {
                    return Err(format!(
                        "source catalog boolean value is invalid: {quality_text:?}"
                    ))
                }
            };
            Ok(CatalogSource {
                detection_id: id,
                x: required_f64(&row, &indexes, "x")?,
                y: required_f64(&row, &indexes, "y")?,
                quality_passed,
                flags: indexes
                    .get("flags")
                    .and_then(|index| row.get(*index))
                    .map(parse_pipe)
                    .unwrap_or_default(),
                repeated_code_values: indexes
                    .get("repeated_code_values")
                    .and_then(|index| row.get(*index))
                    .map(parse_pipe_ints)
                    .transpose()?
                    .unwrap_or_default(),
            })
        })();
        sources.push(
            parsed
                .map_err(|error: String| format!("invalid source catalog row {line}: {error}"))?,
        );
    }
    Ok(sources)
}

fn resolve_modes(text: &str) -> Result<Vec<String>> {
    let mut seen = BTreeSet::new();
    let modes: Vec<_> = text
        .split(',')
        .map(str::trim)
        .filter(|mode| !mode.is_empty())
        .filter(|mode| seen.insert((*mode).to_owned()))
        .map(str::to_owned)
        .collect();
    if modes.is_empty() {
        return Err("mask_modes must contain at least one mode".into());
    }
    let unknown: Vec<_> = modes
        .iter()
        .filter(|mode| !MASK_MODES.contains(&mode.as_str()))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        return Err(format!(
            "unknown pair support mask mode(s): {}",
            unknown.join(", ")
        ));
    }
    Ok(modes)
}

fn resolve_shifts(path: Option<&str>, frame_count: usize) -> Result<Vec<(f64, f64)>> {
    let Some(path) = path else {
        return Ok(vec![(0.0, 0.0); frame_count]);
    };
    let payload: Value =
        serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if !payload.is_object() {
        return Err("shifts JSON must contain an object".into());
    }
    let (field, values) = if let Some(values) = payload
        .get("cumulative_shifts")
        .filter(|value| !value.is_null())
    {
        ("cumulative_shifts", values)
    } else if let Some(values) = payload.get("frame_shifts").filter(|value| !value.is_null()) {
        ("frame_shifts", values)
    } else {
        return Err("shifts JSON must contain cumulative_shifts or frame_shifts".into());
    };
    let raw = values
        .as_array()
        .ok_or_else(|| format!("{field} must contain x/y pairs"))?;
    let mut shifts = Vec::with_capacity(raw.len());
    for item in raw {
        let pair = item
            .as_array()
            .filter(|pair| pair.len() >= 2)
            .ok_or_else(|| format!("{field} must contain x/y pairs"))?;
        let x = pair[0]
            .as_f64()
            .ok_or_else(|| format!("{field} must contain x/y pairs"))?;
        let y = pair[1]
            .as_f64()
            .ok_or_else(|| format!("{field} must contain x/y pairs"))?;
        if !x.is_finite() || !y.is_finite() {
            return Err("frame_shifts must contain finite values".into());
        }
        shifts.push((x, y));
    }
    if shifts.len() != frame_count {
        return Err("frame_shifts length must match frame paths".into());
    }
    Ok(shifts)
}

fn crop_bounds(
    width: usize,
    height: usize,
    positions: [(f64, f64); 2],
    padding: f64,
) -> Result<(usize, usize, usize, usize)> {
    if !padding.is_finite() || padding <= 0.0 {
        return Err("crop_padding_px must be positive and finite".into());
    }
    let x0 = ((positions[0].0.min(positions[1].0) - padding).floor() as isize).max(0) as usize;
    let x1 = (((positions[0].0.max(positions[1].0) + padding).ceil() as isize + 1).max(0) as usize)
        .min(width);
    let y0 = ((positions[0].1.min(positions[1].1) - padding).floor() as isize).max(0) as usize;
    let y1 = (((positions[0].1.max(positions[1].1) + padding).ceil() as isize + 1).max(0) as usize)
        .min(height);
    if x1 <= x0 || y1 <= y0 {
        return Err("pair crop is empty".into());
    }
    Ok((x0, y0, x1, y1))
}

fn assign_targets(
    sources: &[Detection],
    targets: [(f64, f64); 2],
    radius: f64,
) -> Result<[Option<Assignment>; 2]> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err("match_radius_px must be positive and finite".into());
    }
    let source_count = sources.len();
    if source_count == 0 {
        return Ok([None, None]);
    }
    let invalid = radius.max(1.0) * 1_000_000.0;
    let mut costs = vec![invalid; source_count * 2];
    let mut distances = vec![f64::INFINITY; source_count * 2];
    for (source_index, source) in sources.iter().enumerate() {
        for (target_index, target) in targets.iter().enumerate() {
            let distance = (source.x - target.0).hypot(source.y - target.1);
            distances[source_index * 2 + target_index] = distance;
            if distance <= radius {
                costs[source_index * 2 + target_index] =
                    distance + 1e-9 * (source_index * 2 + target_index) as f64;
            }
        }
    }
    let mut assignments = [None, None];
    for (source_index, target_index) in linear_sum_assignment(&costs, source_count, 2)? {
        let distance = distances[source_index * 2 + target_index];
        if distance <= radius + 1e-7 {
            assignments[target_index] = Some(Assignment {
                source_index,
                distance,
            });
        }
    }
    Ok(assignments)
}

fn csv_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::Array(items) => items
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join("|"),
        Value::String(value) => value.clone(),
        Value::Number(value) if value.is_f64() => {
            value.as_f64().map(float_text).unwrap_or_default()
        }
        _ => value.to_string(),
    }
}

fn write_csv(path: &Path, fields: &[&str], rows: &[Value]) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::new());
    writer
        .write_record(fields)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(fields.iter().map(|field| csv_value(&row[*field])))
            .map_err(|error| error.to_string())?;
    }
    let bytes = writer.into_inner().map_err(|error| error.to_string())?;
    let mut output = b"\xef\xbb\xbf".to_vec();
    output.extend(bytes);
    fs::write(path, output).map_err(|error| error.to_string())
}

fn optional_median<'a>(rows: impl Iterator<Item = &'a Value>) -> Option<f64> {
    let mut values: Vec<_> = rows
        .filter_map(Value::as_f64)
        .filter(|value| value.is_finite())
        .collect();
    median(&mut values)
}

fn summarize(rows: &[Value], modes: &[String]) -> Vec<Value> {
    modes.iter().map(|mode| {
        let current: Vec<_> = rows.iter().filter(|row| row["mask_mode"] == mode.as_str()).collect();
        let count = current.len();
        let assigned_two = current.iter().filter(|row| row["assigned_candidate_count"] == 2).count();
        let assigned_one = current.iter().filter(|row| row["assigned_candidate_count"] == 1).count();
        let assigned_zero = current.iter().filter(|row| row["assigned_candidate_count"] == 0).count();
        let quality_two = current.iter().filter(|row| row["primary_local_quality_passed"] == true && row["secondary_local_quality_passed"] == true).count();
        json!({
            "mask_mode":mode,"frame_count":count,"assigned_two_count":assigned_two,
            "assigned_one_count":assigned_one,"assigned_zero_count":assigned_zero,"quality_two_count":quality_two,
            "assigned_two_fraction":if count == 0 { 0.0 } else { assigned_two as f64 / count as f64 },
            "quality_two_fraction":if count == 0 { 0.0 } else { quality_two as f64 / count as f64 },
            "masked_pixel_median":optional_median(current.iter().map(|row| &row["masked_pixel_count"])),
            "primary_shared_positive_fraction_median":optional_median(current.iter().map(|row| &row["primary_shared_positive_fraction"])),
            "secondary_shared_positive_fraction_median":optional_median(current.iter().map(|row| &row["secondary_shared_positive_fraction"])),
            "primary_exclusive_snr_median":optional_median(current.iter().map(|row| &row["primary_exclusive_snr"])),
            "secondary_exclusive_snr_median":optional_median(current.iter().map(|row| &row["secondary_exclusive_snr"])),
            "primary_partition_snr_median":optional_median(current.iter().map(|row| &row["primary_partition_snr"])),
            "secondary_partition_snr_median":optional_median(current.iter().map(|row| &row["secondary_partition_snr"])),
        })
    }).collect()
}
