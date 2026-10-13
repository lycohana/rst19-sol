//! Native contaminated-background double-source injection audit.
//!
//! Candidate hits, quality hits, baseline overlap, and newly recovered injected
//! endpoints are kept separate. These are image-plane detector measurements,
//! not physical source identity or precision estimates.

use crate::{
    detection::{self, DetectionOptions, DetectionResult},
    fits,
    measurement::{Detection, MeasurementOptions},
    models::{FitsFrame, PixelData},
    Result,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

const DEFAULT_SEPARATIONS: &[f64] = &[2.738, 4.123];
const DEFAULT_RATIOS: &[f64] = &[0.143, 1.0];
const NOTE: &str = "baseline flags describe pre-existing detections at the injected truth positions; new flags are injected-minus-baseline diagnostics, not precision or physical truth";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContaminatedPairAnchor {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub direction_x: f64,
    pub direction_y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContaminatedPairInjectionRow {
    pub source_path: String,
    pub anchor_name: String,
    pub anchor_x: f64,
    pub anchor_y: f64,
    pub direction_x: f64,
    pub direction_y: f64,
    pub separation_px: f64,
    pub secondary_to_primary_ratio: f64,
    pub total_peak_excess_adu: f64,
    pub primary_signal_adu: f64,
    pub secondary_signal_adu: f64,
    pub primary_x: f64,
    pub primary_y: f64,
    pub secondary_x: f64,
    pub secondary_y: f64,
    pub baseline_candidate_primary: bool,
    pub baseline_candidate_secondary: bool,
    pub injected_candidate_primary: bool,
    pub injected_candidate_secondary: bool,
    pub new_candidate_primary: bool,
    pub new_candidate_secondary: bool,
    pub baseline_quality_primary: bool,
    pub baseline_quality_secondary: bool,
    pub injected_quality_primary: bool,
    pub injected_quality_secondary: bool,
    pub new_quality_primary: bool,
    pub new_quality_secondary: bool,
    pub candidate_pair_resolved: bool,
    pub quality_pair_resolved: bool,
    pub merged_candidate: bool,
    pub merged_quality: bool,
    pub baseline_candidate_count: usize,
    pub injected_candidate_count: usize,
    pub baseline_quality_count: usize,
    pub injected_quality_count: usize,
    pub new_candidate_hit_count: usize,
    pub new_quality_hit_count: usize,
    pub match_radius_px: f64,
    pub detector_psf_fwhm_px: f64,
    pub injected_psf_fwhm_px: f64,
    pub local_background_adu: Option<f64>,
    pub local_noise_adu: Option<f64>,
    pub note: String,
    pub analysis_scope: String,
    pub roi_bounds: Option<[usize; 4]>,
    pub injected_primary_match_id: Option<i64>,
    pub injected_secondary_match_id: Option<i64>,
    pub injected_primary_match_distance_px: Option<f64>,
    pub injected_secondary_match_distance_px: Option<f64>,
    pub injected_primary_quality_reason: String,
    pub injected_secondary_quality_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContaminatedPairInjectionResult {
    pub source_path: String,
    pub anchors: Vec<ContaminatedPairAnchor>,
    pub separations_px: Vec<f64>,
    pub ratios: Vec<f64>,
    pub baseline_candidate_count: usize,
    pub baseline_quality_count: usize,
    pub parameters: Value,
    pub rows: Vec<ContaminatedPairInjectionRow>,
    pub conclusion: String,
    pub interpretation_boundary: String,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub separations_px: Vec<f64>,
    pub secondary_to_primary_ratios: Vec<f64>,
    pub total_peak_excess_adu: f64,
    pub psf_fwhm: f64,
    pub injected_psf_fwhm: Option<f64>,
    pub proposal_mode: String,
    pub threshold_sigma: f64,
    pub min_distance: usize,
    pub aperture_radius: usize,
    pub background_box_size: usize,
    pub min_flux_snr: f64,
    pub min_psf_support_pixels: usize,
    pub max_sources: Option<usize>,
    pub match_radius_px: f64,
    pub reject_linear_artifacts: bool,
    pub analysis_scope: String,
    pub roi_half_size_px: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            separations_px: DEFAULT_SEPARATIONS.to_vec(),
            secondary_to_primary_ratios: DEFAULT_RATIOS.to_vec(),
            total_peak_excess_adu: 4096.0,
            psf_fwhm: 2.0,
            injected_psf_fwhm: None,
            proposal_mode: "hybrid".into(),
            threshold_sigma: 4.0,
            min_distance: 4,
            aperture_radius: 4,
            background_box_size: 128,
            min_flux_snr: 5.0,
            min_psf_support_pixels: 3,
            max_sources: None,
            match_radius_px: 2.0,
            reject_linear_artifacts: true,
            analysis_scope: "full_frame".into(),
            roi_half_size_px: 192,
        }
    }
}

fn finite_positive(value: f64, name: &str) -> Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{name} must be finite and positive"));
    }
    Ok(value)
}

fn validate_values(values: &[f64], name: &str, upper: Option<f64>) -> Result<Vec<f64>> {
    if values.is_empty() {
        return Err(format!("{name} cannot be empty"));
    }
    let mut resolved = Vec::with_capacity(values.len());
    for value in values {
        finite_positive(*value, name)?;
        if upper.is_some_and(|limit| *value > limit) {
            return Err(format!("{name} must be <= {}", upper.unwrap()));
        }
        resolved.push(*value);
    }
    Ok(resolved)
}

fn normalize_anchor(anchor: &ContaminatedPairAnchor) -> Result<ContaminatedPairAnchor> {
    if anchor.name.trim().is_empty() {
        return Err("anchor name must not be empty".into());
    }
    if [anchor.x, anchor.y, anchor.direction_x, anchor.direction_y]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(format!(
            "anchor {:?} contains a non-finite value",
            anchor.name
        ));
    }
    let norm = anchor.direction_x.hypot(anchor.direction_y);
    if norm <= 0.0 {
        return Err(format!(
            "anchor {:?} direction must be non-zero",
            anchor.name
        ));
    }
    Ok(ContaminatedPairAnchor {
        name: anchor.name.trim().to_owned(),
        x: anchor.x,
        y: anchor.y,
        direction_x: anchor.direction_x / norm,
        direction_y: anchor.direction_y / norm,
    })
}

fn pair_positions(anchor: &ContaminatedPairAnchor, separation: f64) -> [[f64; 2]; 2] {
    let half = 0.5 * separation;
    let dx = half * anchor.direction_x;
    let dy = half * anchor.direction_y;
    [
        [anchor.x + dx, anchor.y + dy],
        [anchor.x - dx, anchor.y - dy],
    ]
}

fn roi_bounds(
    frame: &FitsFrame,
    anchor: &ContaminatedPairAnchor,
    half: usize,
) -> Result<[usize; 4]> {
    let center_x = anchor.x.round_ties_even() as isize;
    let center_y = anchor.y.round_ties_even() as isize;
    let half = isize::try_from(half).map_err(|_| "roi_half_size_px is too large")?;
    let x0 = (center_x - half).max(0) as usize;
    let y0 = (center_y - half).max(0) as usize;
    let x1 = usize::try_from((center_x + half + 1).min(frame.width as isize)).unwrap_or(0);
    let y1 = usize::try_from((center_y + half + 1).min(frame.height as isize)).unwrap_or(0);
    if x1 <= x0 || y1 <= y0 {
        return Err(format!("anchor {:?} has an empty local ROI", anchor.name));
    }
    Ok([x0, y0, x1, y1])
}

fn match_sources(
    targets: &[[f64; 2]; 2],
    detections: &[Detection],
    radius: f64,
) -> [Option<Detection>; 2] {
    let mut remaining: Vec<usize> = (0..detections.len()).collect();
    let mut matched = [None, None];
    for (target_index, target) in targets.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        for (slot, source_index) in remaining.iter().enumerate() {
            let source = &detections[*source_index];
            let distance = (source.x - target[0]).hypot(source.y - target[1]);
            if distance <= best.map_or(radius, |(_, best_distance)| best_distance) {
                best = Some((slot, distance));
            }
        }
        if let Some((slot, _)) = best {
            let source_index = remaining.remove(slot);
            matched[target_index] = Some(detections[source_index].clone());
        }
    }
    matched
}

fn hit_flags(targets: &[[f64; 2]; 2], detections: &[Detection], radius: f64) -> [bool; 2] {
    let matched = match_sources(targets, detections, radius);
    [matched[0].is_some(), matched[1].is_some()]
}

fn quality_reason(source: Option<&Detection>) -> String {
    match source {
        None => "NO_CANDIDATE".into(),
        Some(source) if source.quality_passed => "QUALITY_PASS".into(),
        Some(source) if !source.flags.is_empty() => source.flags.join("|"),
        Some(_) => "QUALITY_REJECTED_UNKNOWN".into(),
    }
}

fn match_distance(target: [f64; 2], source: Option<&Detection>) -> Option<f64> {
    source.map(|source| (source.x - target[0]).hypot(source.y - target[1]))
}

fn merged_pair(targets: &[[f64; 2]; 2], detections: &[Detection], radius: f64) -> bool {
    detections.iter().any(|source| {
        (source.x - targets[0][0]).hypot(source.y - targets[0][1]) <= radius
            && (source.x - targets[1][0]).hypot(source.y - targets[1][1]) <= radius
    })
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    })
}

fn local_stats(
    frame: &FitsFrame,
    anchor: &ContaminatedPairAnchor,
    radius: usize,
) -> (Option<f64>, Option<f64>) {
    let center_x = anchor.x.round_ties_even() as isize;
    let center_y = anchor.y.round_ties_even() as isize;
    let radius = radius as isize;
    let x0 = (center_x - radius).max(0) as usize;
    let y0 = (center_y - radius).max(0) as usize;
    let x1 = (center_x + radius + 1).clamp(0, frame.width as isize) as usize;
    let y1 = (center_y + radius + 1).clamp(0, frame.height as isize) as usize;
    let mut finite = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let value = frame.data.value(y * frame.width + x);
            if value.is_finite() {
                finite.push(value);
            }
        }
    }
    let Some(background) = median(&mut finite) else {
        return (None, None);
    };
    let mut deviations: Vec<_> = finite
        .iter()
        .map(|value| (value - background).abs())
        .collect();
    let Some(mad) = median(&mut deviations) else {
        return (Some(background), None);
    };
    let mut scale = 1.4826 * mad;
    if !scale.is_finite() || scale <= 0.0 {
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        scale = (finite
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / finite.len() as f64)
            .sqrt();
    }
    if !scale.is_finite() || scale <= 0.0 {
        scale = f64::EPSILON;
    }
    let clipped: Vec<_> = finite
        .iter()
        .copied()
        .filter(|value| (value - background).abs() <= 5.0 * scale)
        .collect();
    if clipped.len() >= 3 {
        let mut clipped_values = clipped.clone();
        let Some(clipped_background) = median(&mut clipped_values) else {
            return (Some(background), Some(scale));
        };
        let mut clipped_deviations: Vec<_> = clipped
            .iter()
            .map(|value| (value - clipped_background).abs())
            .collect();
        let Some(clipped_mad) = median(&mut clipped_deviations) else {
            return (Some(clipped_background), Some(scale));
        };
        let clipped_mean = clipped.iter().sum::<f64>() / clipped.len() as f64;
        let std = (clipped
            .iter()
            .map(|value| (value - clipped_mean).powi(2))
            .sum::<f64>()
            / clipped.len() as f64)
            .sqrt();
        let noise = (1.4826 * clipped_mad).max(std).max(f64::EPSILON);
        (Some(clipped_background), Some(noise))
    } else {
        (Some(background), Some(scale))
    }
}

fn detector_options(options: &Options) -> DetectionOptions {
    let measurement = MeasurementOptions {
        aperture_radius: options.aperture_radius,
        psf_fwhm: options.psf_fwhm,
        min_flux_snr: options.min_flux_snr,
        min_psf_support_pixels: options.min_psf_support_pixels,
        ..Default::default()
    };
    DetectionOptions {
        measurement,
        threshold_sigma: options.threshold_sigma,
        min_distance: options.min_distance,
        max_sources: options.max_sources,
        background_box_size: options.background_box_size,
        reject_linear_artifacts: options.reject_linear_artifacts,
        proposal_mode: options.proposal_mode.clone(),
        ..Default::default()
    }
}

fn detect(
    frame_data: &PixelData,
    height: usize,
    width: usize,
    mask: &[bool],
    options: &DetectionOptions,
) -> Result<DetectionResult> {
    detection::detect_sources(frame_data, Some(mask), height, width, options, None)
}

fn float32_copy(frame: &FitsFrame, bounds: Option<[usize; 4]>) -> (PixelData, usize, usize) {
    let (x0, y0, x1, y1) = bounds.map_or((0, 0, frame.width, frame.height), |b| {
        (b[0], b[1], b[2], b[3])
    });
    let width = x1 - x0;
    let height = y1 - y0;
    let mut data = Vec::with_capacity(width * height);
    for y in y0..y1 {
        for x in x0..x1 {
            data.push(frame.data.value(y * frame.width + x) as f32);
        }
    }
    (PixelData::F32(data), height, width)
}

fn cropped_mask(mask: &[bool], frame_width: usize, bounds: [usize; 4]) -> Vec<bool> {
    let [x0, y0, x1, y1] = bounds;
    let mut output = Vec::with_capacity((x1 - x0) * (y1 - y0));
    for y in y0..y1 {
        output.extend_from_slice(&mask[y * frame_width + x0..y * frame_width + x1]);
    }
    output
}

fn inject_gaussian(
    data: &mut PixelData,
    height: usize,
    width: usize,
    x: f64,
    y: f64,
    peak: f64,
    fwhm: f64,
) -> Result<()> {
    let PixelData::F32(image) = data else {
        return Err("injection target must be float32".into());
    };
    let sigma = (fwhm / 2.354820045).max(f64::EPSILON);
    let radius = 4usize.max((4.0 * sigma).ceil() as usize);
    let center_x = x.floor() as isize;
    let center_y = y.floor() as isize;
    let x0 = (center_x - radius as isize).max(0) as usize;
    let x1 = (center_x + radius as isize + 1).min(width as isize).max(0) as usize;
    let y0 = (center_y - radius as isize).max(0) as usize;
    let y1 = (center_y + radius as isize + 1).min(height as isize).max(0) as usize;
    for row in y0..y1 {
        for column in x0..x1 {
            let dx = column as f64 - x;
            let dy = row as f64 - y;
            let kernel = (-((dx * dx + dy * dy) / (2.0 * sigma * sigma))).exp();
            let index = row * width + column;
            image[index] = (f64::from(image[index]) + peak * kernel) as f32;
        }
    }
    Ok(())
}

fn offset_detections(sources: &[Detection], x_offset: usize, y_offset: usize) -> Vec<Detection> {
    sources
        .iter()
        .cloned()
        .map(|mut source| {
            source.x += x_offset as f64;
            source.y += y_offset as f64;
            source.peak_x = source.peak_x.map(|value| value + x_offset as f64);
            source.peak_y = source.peak_y.map(|value| value + y_offset as f64);
            source
        })
        .collect()
}

pub fn run(
    frame: &FitsFrame,
    anchors: &[ContaminatedPairAnchor],
    options: &Options,
) -> Result<ContaminatedPairInjectionResult> {
    if anchors.is_empty() {
        return Err("anchors cannot be empty".into());
    }
    let anchors: Vec<_> = anchors
        .iter()
        .map(normalize_anchor)
        .collect::<Result<_>>()?;
    let names: BTreeSet<_> = anchors.iter().map(|anchor| anchor.name.as_str()).collect();
    if names.len() != anchors.len() {
        return Err("anchor names must be unique".into());
    }
    let separations = validate_values(&options.separations_px, "separations_px", None)?;
    let ratios = validate_values(
        &options.secondary_to_primary_ratios,
        "secondary_to_primary_ratios",
        Some(1.0),
    )?;
    let total_peak = finite_positive(options.total_peak_excess_adu, "total_peak_excess_adu")?;
    let detector_fwhm = finite_positive(options.psf_fwhm, "psf_fwhm")?;
    let injected_fwhm = finite_positive(
        options.injected_psf_fwhm.unwrap_or(detector_fwhm),
        "injected_psf_fwhm",
    )?;
    let match_radius = finite_positive(options.match_radius_px, "match_radius_px")?;
    let scope = options.analysis_scope.trim().to_ascii_lowercase();
    if scope != "full_frame" && scope != "local_roi" {
        return Err("analysis_scope must be 'full_frame' or 'local_roi'".into());
    }
    if options.min_distance == 0 || options.aperture_radius == 0 || options.background_box_size < 16
    {
        return Err("detector geometry and threshold parameters are invalid".into());
    }
    if options.threshold_sigma <= 0.0
        || options.min_flux_snr <= 0.0
        || !(1..=9).contains(&options.min_psf_support_pixels)
    {
        return Err("min_flux_snr/min_psf_support_pixels are invalid".into());
    }
    if options.max_sources.is_some_and(|value| value == 0) {
        return Err("max_sources must be positive when provided".into());
    }
    if scope == "local_roi" {
        let minimum = (separations.iter().copied().fold(0.0, f64::max) / 2.0).ceil() as usize
            + options.aperture_radius
            + 2;
        if options.roi_half_size_px < minimum {
            return Err(format!("roi_half_size_px is too small for the requested separation/aperture; need at least {minimum}"));
        }
    }
    if frame.width == 0
        || frame.height == 0
        || frame.width.checked_mul(frame.height) != Some(frame.data.len())
    {
        return Err("expected a valid 2-D FITS image".into());
    }

    let full_mask = fits::auxiliary_mask(frame.height, frame.width)?;
    let mut detector = detector_options(options);
    detector.use_float32 = matches!(frame.data, PixelData::F32(_));
    let baseline = detect(
        &frame.data,
        frame.height,
        frame.width,
        &full_mask,
        &detector,
    )?;
    let mut injected_detector = detector.clone();
    injected_detector.use_float32 = true;
    injected_detector.measurement.saturation_level = baseline.parameters["saturation_level"]
        .as_f64()
        .filter(|value| *value > 0.0);
    injected_detector.measurement.negative_overflow_limit =
        baseline.parameters["negative_overflow_limit"].as_f64();
    injected_detector.mask_zero_pixels = Some(
        baseline.parameters["mask_zero_pixels"]
            .as_u64()
            .unwrap_or(0)
            != 0,
    );

    let mut rows = Vec::with_capacity(anchors.len() * separations.len() * ratios.len());
    for anchor in &anchors {
        let (local_background, local_noise) = local_stats(frame, anchor, 5);
        let bounds = if scope == "local_roi" {
            Some(roi_bounds(frame, anchor, options.roi_half_size_px)?)
        } else {
            None
        };
        let local_mask = bounds.map(|roi| cropped_mask(&full_mask, frame.width, roi));
        let detection_mask = local_mask.as_deref().unwrap_or(&full_mask);
        for separation in &separations {
            let targets = pair_positions(anchor, *separation);
            let baseline_candidate = hit_flags(&targets, &baseline.sources, match_radius);
            let baseline_quality_sources: Vec<_> = baseline
                .sources
                .iter()
                .filter(|source| source.quality_passed)
                .cloned()
                .collect();
            let baseline_quality = hit_flags(&targets, &baseline_quality_sources, match_radius);
            for ratio in &ratios {
                let primary_signal = total_peak / (1.0 + ratio);
                let secondary_signal = total_peak - primary_signal;
                let (mut injected_data, height, width) = float32_copy(frame, bounds);
                let (x_offset, y_offset) = bounds.map_or((0, 0), |roi| (roi[0], roi[1]));
                inject_gaussian(
                    &mut injected_data,
                    height,
                    width,
                    targets[0][0] - x_offset as f64,
                    targets[0][1] - y_offset as f64,
                    primary_signal,
                    injected_fwhm,
                )?;
                inject_gaussian(
                    &mut injected_data,
                    height,
                    width,
                    targets[1][0] - x_offset as f64,
                    targets[1][1] - y_offset as f64,
                    secondary_signal,
                    injected_fwhm,
                )?;
                let injected_result = detect(
                    &injected_data,
                    height,
                    width,
                    detection_mask,
                    &injected_detector,
                )?;
                let injected_sources =
                    offset_detections(&injected_result.sources, x_offset, y_offset);
                let injected_quality_sources: Vec<_> = injected_sources
                    .iter()
                    .filter(|source| source.quality_passed)
                    .cloned()
                    .collect();
                let injected_candidate = hit_flags(&targets, &injected_sources, match_radius);
                let injected_quality = hit_flags(&targets, &injected_quality_sources, match_radius);
                let matches = match_sources(&targets, &injected_sources, match_radius);
                let new_candidate = [
                    injected_candidate[0] && !baseline_candidate[0],
                    injected_candidate[1] && !baseline_candidate[1],
                ];
                let new_quality = [
                    injected_quality[0] && !baseline_quality[0],
                    injected_quality[1] && !baseline_quality[1],
                ];
                let injected_primary = matches[0].as_ref();
                let injected_secondary = matches[1].as_ref();
                rows.push(ContaminatedPairInjectionRow {
                    source_path: frame.path.to_string_lossy().into_owned(),
                    anchor_name: anchor.name.clone(),
                    anchor_x: anchor.x,
                    anchor_y: anchor.y,
                    direction_x: anchor.direction_x,
                    direction_y: anchor.direction_y,
                    separation_px: *separation,
                    secondary_to_primary_ratio: *ratio,
                    total_peak_excess_adu: total_peak,
                    primary_signal_adu: primary_signal,
                    secondary_signal_adu: secondary_signal,
                    primary_x: targets[0][0],
                    primary_y: targets[0][1],
                    secondary_x: targets[1][0],
                    secondary_y: targets[1][1],
                    baseline_candidate_primary: baseline_candidate[0],
                    baseline_candidate_secondary: baseline_candidate[1],
                    injected_candidate_primary: injected_candidate[0],
                    injected_candidate_secondary: injected_candidate[1],
                    new_candidate_primary: new_candidate[0],
                    new_candidate_secondary: new_candidate[1],
                    baseline_quality_primary: baseline_quality[0],
                    baseline_quality_secondary: baseline_quality[1],
                    injected_quality_primary: injected_quality[0],
                    injected_quality_secondary: injected_quality[1],
                    new_quality_primary: new_quality[0],
                    new_quality_secondary: new_quality[1],
                    candidate_pair_resolved: injected_candidate.iter().all(|hit| *hit),
                    quality_pair_resolved: injected_quality.iter().all(|hit| *hit),
                    merged_candidate: merged_pair(&targets, &injected_sources, match_radius),
                    merged_quality: merged_pair(&targets, &injected_quality_sources, match_radius),
                    baseline_candidate_count: baseline.candidate_count,
                    injected_candidate_count: injected_result.candidate_count,
                    baseline_quality_count: baseline.quality_count,
                    injected_quality_count: injected_result.quality_count,
                    new_candidate_hit_count: new_candidate.iter().filter(|hit| **hit).count(),
                    new_quality_hit_count: new_quality.iter().filter(|hit| **hit).count(),
                    match_radius_px: match_radius,
                    detector_psf_fwhm_px: detector_fwhm,
                    injected_psf_fwhm_px: injected_fwhm,
                    local_background_adu: local_background,
                    local_noise_adu: local_noise,
                    note: format!(
                        "{NOTE}; analysis_scope={scope}; injected counts are {}",
                        if scope == "local_roi" {
                            "ROI-local"
                        } else {
                            "full-frame"
                        }
                    ),
                    analysis_scope: scope.clone(),
                    roi_bounds: bounds,
                    injected_primary_match_id: injected_primary.map(|source| source.detection_id),
                    injected_secondary_match_id: injected_secondary
                        .map(|source| source.detection_id),
                    injected_primary_match_distance_px: match_distance(
                        targets[0],
                        injected_primary,
                    ),
                    injected_secondary_match_distance_px: match_distance(
                        targets[1],
                        injected_secondary,
                    ),
                    injected_primary_quality_reason: quality_reason(injected_primary),
                    injected_secondary_quality_reason: quality_reason(injected_secondary),
                });
            }
        }
    }
    let candidate_resolved = rows
        .iter()
        .filter(|row| row.candidate_pair_resolved)
        .count();
    let quality_resolved = rows.iter().filter(|row| row.quality_pair_resolved).count();
    let merged = rows.iter().filter(|row| row.merged_candidate).count();
    let conclusion = format!(
        "污染结构双源注入 pilot：{} 个条件中，候选双源分别命中 {candidate_resolved}/{}，质量层分别通过 {quality_resolved}/{}，候选合并 {merged}/{}；baseline 命中必须与新增命中分开解释。",
        rows.len(), rows.len(), rows.len(), rows.len()
    );
    let parameters = json!({
        "proposal_mode": options.proposal_mode,
        "threshold_sigma": options.threshold_sigma,
        "min_distance": options.min_distance,
        "aperture_radius": options.aperture_radius,
        "background_box_size": options.background_box_size,
        "min_flux_snr": options.min_flux_snr,
        "min_psf_support_pixels": options.min_psf_support_pixels,
        "max_sources": options.max_sources,
        "reject_linear_artifacts": options.reject_linear_artifacts,
        "signal_normalization": "peak_excess",
        "analysis_scope": scope,
        "roi_half_size_px": if scope == "local_roi" { json!(options.roi_half_size_px) } else { Value::Null },
        "roi_bounds_order": if scope == "local_roi" { json!("x0,y0,x1,y1") } else { Value::Null },
        "baseline_scope": "full_frame",
        "injected_count_scope": if scope == "local_roi" { "local_roi" } else { "full_frame" },
        "detector_psf_fwhm_px": detector_fwhm,
        "injected_psf_fwhm_px": injected_fwhm,
        "match_radius_px": match_radius,
    });
    Ok(ContaminatedPairInjectionResult {
        source_path: frame.path.to_string_lossy().into_owned(),
        anchors, separations_px: separations, ratios,
        baseline_candidate_count: baseline.candidate_count,
        baseline_quality_count: baseline.quality_count,
        parameters, rows, conclusion,
        interpretation_boundary: "contaminated-background injection pilot; baseline overlap is not injection truth; candidate/quality recovery is not precision or physical star truth".into(),
    })
}

pub fn write_artifacts(
    result: &ContaminatedPairInjectionResult,
    output_dir: &Path,
) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    let csv_path = output_dir.join("contaminated_pair_injection.csv");
    const COLUMNS: &[&str] = &[
        "source_path",
        "anchor_name",
        "anchor_x",
        "anchor_y",
        "direction_x",
        "direction_y",
        "separation_px",
        "secondary_to_primary_ratio",
        "total_peak_excess_adu",
        "primary_signal_adu",
        "secondary_signal_adu",
        "primary_x",
        "primary_y",
        "secondary_x",
        "secondary_y",
        "baseline_candidate_primary",
        "baseline_candidate_secondary",
        "injected_candidate_primary",
        "injected_candidate_secondary",
        "new_candidate_primary",
        "new_candidate_secondary",
        "baseline_quality_primary",
        "baseline_quality_secondary",
        "injected_quality_primary",
        "injected_quality_secondary",
        "new_quality_primary",
        "new_quality_secondary",
        "candidate_pair_resolved",
        "quality_pair_resolved",
        "merged_candidate",
        "merged_quality",
        "baseline_candidate_count",
        "injected_candidate_count",
        "baseline_quality_count",
        "injected_quality_count",
        "new_candidate_hit_count",
        "new_quality_hit_count",
        "match_radius_px",
        "detector_psf_fwhm_px",
        "injected_psf_fwhm_px",
        "local_background_adu",
        "local_noise_adu",
        "note",
        "analysis_scope",
        "roi_bounds",
        "injected_primary_match_id",
        "injected_secondary_match_id",
        "injected_primary_match_distance_px",
        "injected_secondary_match_distance_px",
        "injected_primary_quality_reason",
        "injected_secondary_quality_reason",
    ];
    let mut writer = csv::WriterBuilder::new().from_writer(Vec::new());
    writer
        .write_record(COLUMNS)
        .map_err(|error| error.to_string())?;
    for row in &result.rows {
        let value = serde_json::to_value(row).map_err(|error| error.to_string())?;
        let object = value
            .as_object()
            .ok_or("contaminated-pair row must serialize as an object")?;
        let fields = COLUMNS
            .iter()
            .map(|name| match object.get(*name).unwrap_or(&Value::Null) {
                Value::Null => String::new(),
                Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
                Value::Number(value) => value.to_string(),
                Value::String(value) => value.clone(),
                Value::Array(values) if *name == "roi_bounds" => format!(
                    "({})",
                    values
                        .iter()
                        .map(Value::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                Value::Array(values) => values
                    .iter()
                    .map(Value::to_string)
                    .collect::<Vec<_>>()
                    .join("|"),
                Value::Object(_) => serde_json::to_string(&object[*name]).unwrap_or_default(),
            })
            .collect::<Vec<_>>();
        writer
            .write_record(fields)
            .map_err(|error| error.to_string())?;
    }
    let mut bytes = writer.into_inner().map_err(|error| error.to_string())?;
    let mut with_bom = Vec::with_capacity(bytes.len() + 3);
    with_bom.extend_from_slice(b"\xef\xbb\xbf");
    with_bom.append(&mut bytes);
    fs::write(&csv_path, with_bom).map_err(|error| error.to_string())?;
    let json_path = output_dir.join("contaminated_pair_injection.json");
    fs::write(
        &json_path,
        serde_json::to_vec_pretty(result).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output_dir.to_path_buf())
}

struct ParsedArgs {
    frame: PathBuf,
    anchors: Vec<ContaminatedPairAnchor>,
    options: Options,
    output_dir: PathBuf,
}

fn parse_anchor(raw: &str) -> Result<ContaminatedPairAnchor> {
    let fields: Vec<_> = raw.split(',').map(str::trim).collect();
    if fields.len() != 5 {
        return Err("--anchor must be NAME,X,Y,DIRECTION_X,DIRECTION_Y".into());
    }
    let number = |index: usize| {
        fields[index]
            .parse::<f64>()
            .map_err(|error| format!("invalid --anchor {raw:?}: {error}"))
    };
    Ok(ContaminatedPairAnchor {
        name: fields[0].to_owned(),
        x: number(1)?,
        y: number(2)?,
        direction_x: number(3)?,
        direction_y: number(4)?,
    })
}

fn take_value<'a>(
    args: &'a [String],
    index: &mut usize,
    inline: Option<&'a str>,
) -> Result<&'a str> {
    if let Some(value) = inline {
        return Ok(value);
    }
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| "option requires a value".into())
}

fn parse_args(args: &[String]) -> Result<ParsedArgs> {
    let mut frame = None;
    let mut anchors = Vec::new();
    let mut options = Options::default();
    let mut output_dir = PathBuf::from("tmp/contaminated-pair-injection");
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline) = arg
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((arg.as_str(), None));
        match name {
            "--anchor" => anchors.push(parse_anchor(take_value(args, &mut index, inline)?)?),
            "--separations-px" | "--ratios" => {
                let mut values = Vec::new();
                if let Some(value) = inline {
                    values.push(value.parse::<f64>().map_err(|error| error.to_string())?);
                }
                while let Some(value) = args
                    .get(index + 1)
                    .and_then(|value| value.parse::<f64>().ok())
                {
                    index += 1;
                    values.push(value);
                }
                if values.is_empty() {
                    return Err(format!("{name} requires one or more numbers"));
                }
                if name == "--separations-px" {
                    options.separations_px = values;
                } else {
                    options.secondary_to_primary_ratios = values;
                }
            }
            "--total-peak-adu" => {
                options.total_peak_excess_adu = take_value(args, &mut index, inline)?
                    .parse()
                    .map_err(|error: std::num::ParseFloatError| error.to_string())?;
            }
            "--psf-fwhm" => {
                options.psf_fwhm = take_value(args, &mut index, inline)?
                    .parse()
                    .map_err(|error: std::num::ParseFloatError| error.to_string())?;
            }
            "--injected-psf-fwhm" => {
                options.injected_psf_fwhm = Some(
                    take_value(args, &mut index, inline)?
                        .parse()
                        .map_err(|error: std::num::ParseFloatError| error.to_string())?,
                );
            }
            "--proposal-mode" => {
                options.proposal_mode = take_value(args, &mut index, inline)?.to_owned()
            }
            "--threshold-sigma" => {
                options.threshold_sigma = take_value(args, &mut index, inline)?
                    .parse()
                    .map_err(|error: std::num::ParseFloatError| error.to_string())?;
            }
            "--min-distance" => {
                options.min_distance = take_value(args, &mut index, inline)?
                    .parse()
                    .map_err(|error: std::num::ParseIntError| error.to_string())?;
            }
            "--aperture-radius" => {
                options.aperture_radius = take_value(args, &mut index, inline)?
                    .parse()
                    .map_err(|error: std::num::ParseIntError| error.to_string())?;
            }
            "--background-box-size" => {
                options.background_box_size = take_value(args, &mut index, inline)?
                    .parse()
                    .map_err(|error: std::num::ParseIntError| error.to_string())?;
            }
            "--min-flux-snr" => {
                options.min_flux_snr = take_value(args, &mut index, inline)?
                    .parse()
                    .map_err(|error: std::num::ParseFloatError| error.to_string())?;
            }
            "--min-psf-support-pixels" => {
                options.min_psf_support_pixels = take_value(args, &mut index, inline)?
                    .parse()
                    .map_err(|error: std::num::ParseIntError| error.to_string())?;
            }
            "--match-radius-px" => {
                options.match_radius_px = take_value(args, &mut index, inline)?
                    .parse()
                    .map_err(|error: std::num::ParseFloatError| error.to_string())?;
            }
            "--analysis-scope" => {
                options.analysis_scope = take_value(args, &mut index, inline)?.to_owned()
            }
            "--roi-half-size-px" => {
                options.roi_half_size_px = take_value(args, &mut index, inline)?
                    .parse()
                    .map_err(|error: std::num::ParseIntError| error.to_string())?;
            }
            "--out-dir" => output_dir = PathBuf::from(take_value(args, &mut index, inline)?),
            "--allow-linear-artifacts" => options.reject_linear_artifacts = false,
            "--" => {
                index += 1;
                if let Some(value) = args.get(index) {
                    if frame.replace(PathBuf::from(value)).is_some() {
                        return Err("expected exactly one FITS frame".into());
                    }
                }
                break;
            }
            _ if name.starts_with('-') => return Err(format!("unknown argument {name}")),
            _ => {
                if frame.replace(PathBuf::from(name)).is_some() {
                    return Err("expected exactly one FITS frame".into());
                }
            }
        }
        index += 1;
    }
    let frame = frame.ok_or("expected one FITS frame")?;
    if anchors.is_empty() {
        return Err("at least one --anchor is required".into());
    }
    Ok(ParsedArgs {
        frame,
        anchors,
        options,
        output_dir,
    })
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let parsed = parse_args(args)?;
    let frame = fits::read_fits(&parsed.frame, false)?;
    let result = run(&frame, &parsed.anchors, &parsed.options)?;
    let output = write_artifacts(&result, &parsed.output_dir)?;
    Ok(json!({
        "tool": "rst19-contaminated-pair-injection",
        "out_dir": output,
        "result_path": output.join("contaminated_pair_injection.json"),
        "result": result,
    }))
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        println!("在真实局部结构上注入双源，并分开记录 baseline/new 命中。\n\nUsage: rst19-contaminated-pair-injection FRAME --anchor NAME,X,Y,DX,DY [--anchor ...] [--separations-px PX ...] [--ratios R ...] [--analysis-scope full_frame|local_roi] [--out-dir DIR]");
        return Ok(());
    }
    let output = run_desktop(args)?;
    println!(
        "污染结构双源注入结果写入 {}",
        output["out_dir"].as_str().unwrap_or_default()
    );
    println!(
        "{}",
        output["result"]["conclusion"].as_str().unwrap_or_default()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors_are_normalized_and_endpoints_are_symmetric() {
        let anchor = normalize_anchor(&ContaminatedPairAnchor {
            name: "  source  ".into(),
            x: 20.0,
            y: 30.0,
            direction_x: 3.0,
            direction_y: 4.0,
        })
        .unwrap();
        assert_eq!(anchor.name, "source");
        assert_eq!(anchor.direction_x, 0.6);
        assert_eq!(anchor.direction_y, 0.8);
        assert_eq!(pair_positions(&anchor, 10.0), [[23.0, 34.0], [17.0, 26.0]]);
    }

    #[test]
    fn matching_is_one_to_one_and_last_equal_distance_candidate_wins() {
        let source = |id, x| Detection {
            detection_id: id,
            x,
            y: 0.0,
            peak: 1.0,
            flux: 1.0,
            background: 0.0,
            noise: 1.0,
            snr: 1.0,
            fwhm: None,
            flags: vec![],
            flux_error: None,
            flux_snr: None,
            filter_snr: None,
            fwhm_x: None,
            fwhm_y: None,
            ellipticity: None,
            sharpness: None,
            footprint_pixels: None,
            psf_support_pixels: None,
            quality_passed: false,
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
        };
        let targets = [[0.0, 0.0], [0.0, 0.0]];
        let detections = [source(1, -1.0), source(2, 1.0)];
        let matches = match_sources(&targets, &detections, 1.0);
        assert_eq!(matches[0].as_ref().map(|value| value.detection_id), Some(2));
        assert_eq!(matches[1].as_ref().map(|value| value.detection_id), Some(1));
        assert_eq!(hit_flags(&targets, &detections, 1.0), [true, true]);
    }

    #[test]
    fn parser_accepts_repeated_anchors_and_numeric_lists() {
        let args = [
            "frame.fits",
            "--anchor",
            "a,1,2,1,0",
            "--anchor=b,3,4,0,1",
            "--separations-px",
            "2",
            "4.5",
            "--ratios",
            "0.25",
            "1",
            "--allow-linear-artifacts",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        let parsed = parse_args(&args).unwrap();
        assert_eq!(parsed.anchors.len(), 2);
        assert_eq!(parsed.options.separations_px, [2.0, 4.5]);
        assert_eq!(parsed.options.secondary_to_primary_ratios, [0.25, 1.0]);
        assert!(!parsed.options.reject_linear_artifacts);
    }

    #[test]
    fn audit_runs_native_detection_on_private_float32_copies() {
        let (width, height) = (160usize, 160usize);
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let pattern = ((x * 17 + y * 31) % 13) as f32 * 0.2;
                let dx = x as f64 - 70.0;
                let dy = y as f64 - 130.0;
                let star = (500.0 * (-0.5 * (dx * dx + dy * dy) / 1.2_f64.powi(2)).exp()) as f32;
                pixels.push(100.0 + pattern + star);
            }
        }
        let original = pixels.clone();
        let frame = FitsFrame {
            path: PathBuf::from("synthetic-contaminated-pair.fits"),
            header: Default::default(),
            data: PixelData::F32(pixels),
            auxiliary: None,
            data_offset: 0,
            width,
            height,
        };
        let options = Options {
            separations_px: vec![6.0],
            secondary_to_primary_ratios: vec![1.0],
            total_peak_excess_adu: 2048.0,
            proposal_mode: "gaussian".into(),
            background_box_size: 32,
            min_flux_snr: 2.0,
            analysis_scope: "local_roi".into(),
            roi_half_size_px: 24,
            ..Default::default()
        };
        let anchor = ContaminatedPairAnchor {
            name: "synthetic".into(),
            x: 70.0,
            y: 130.0,
            direction_x: 1.0,
            direction_y: 0.0,
        };
        let result = run(&frame, &[anchor], &options).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].analysis_scope, "local_roi");
        assert_eq!(result.rows[0].roi_bounds, Some([46, 106, 95, 155]));
        assert_eq!(result.rows[0].primary_x, 73.0);
        assert_eq!(result.rows[0].secondary_x, 67.0);
        assert!(!result.rows[0].injected_primary_quality_reason.is_empty());
        let PixelData::F32(after) = &frame.data else {
            unreachable!()
        };
        assert_eq!(
            after, &original,
            "audit must not mutate the original FITS pixels"
        );
    }
}
