//! Native recovery of faint, persistent sources from registered sequence stacks.
use crate::{
    detection::{self, DetectionOptions},
    fits,
    measurement::MeasurementOptions,
    models::PixelData,
    pipeline::FrameAnalysis,
    sequence::TrackPoint,
    sequence_motion::{self, TrackObservation},
    sequence_photometry::{self, FramePixels, StackMeasurementOptions},
    sequence_temporal,
    spatial::KdTree,
    Result,
};
use serde_json::{json, Value};

const DEFAULT_STACK_WORKING_LIMIT: usize = 8_000;
const MAX_FRAME_CHUNK: usize = 8_192;

#[derive(Debug, Clone)]
pub struct StackFaintOptions {
    pub enabled: bool,
    pub reference_mode: String,
    pub threshold_sigma: f64,
    pub min_flux_snr: f64,
    pub frame_min_flux_snr: f64,
    pub min_presence: usize,
    pub max_reference_sources: Option<usize>,
    pub link_radius_px: f64,
    pub min_distance: usize,
    pub aperture_radius: usize,
    pub psf_fwhm: f64,
    pub min_psf_support_pixels: usize,
    pub proposal_mode: String,
    pub mask_zero_pixels: bool,
}

impl Default for StackFaintOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            reference_mode: "median".into(),
            threshold_sigma: 4.0,
            min_flux_snr: 5.0,
            frame_min_flux_snr: 3.0,
            min_presence: 3,
            max_reference_sources: Some(DEFAULT_STACK_WORKING_LIMIT),
            link_radius_px: 4.0,
            min_distance: 3,
            aperture_radius: 4,
            psf_fwhm: 3.0,
            min_psf_support_pixels: 3,
            proposal_mode: "gaussian".into(),
            mask_zero_pixels: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StackFaintResult {
    pub tracks: Vec<Value>,
    pub candidate_count: usize,
    pub working_count: usize,
    pub audit: Value,
}

fn empty_result() -> StackFaintResult {
    StackFaintResult {
        tracks: Vec::new(),
        candidate_count: 0,
        working_count: 0,
        audit: json!({
            "stack_reference_quality_sources": 0,
            "stack_reference_working_sources": 0,
            "reject_stack_explained": 0,
            "reject_stack_presence": 0,
            "reject_stack_flux_snr": 0,
            "accepted_stack_faint": 0
        }),
    }
}

fn track_positions(tracks: &[Value]) -> Vec<[f64; 2]> {
    tracks
        .iter()
        .filter_map(|track| track.get("points").and_then(Value::as_array))
        .flat_map(|points| points.iter())
        .filter_map(|point| {
            let x = point.get("aligned_x")?.as_f64()?;
            let y = point.get("aligned_y")?.as_f64()?;
            (x.is_finite() && y.is_finite()).then_some([x, y])
        })
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn candidate_snr(source: &crate::measurement::Detection) -> Value {
    source
        .filter_snr
        .filter(|value| value.is_finite())
        .map_or(Value::Null, |value| json!(value))
}

/// Recheck stack proposals against the original frames before emitting low-confidence tracks.
pub fn recover_stack_faint_tracks(
    analyses: &[FrameAnalysis],
    cumulative_shifts: &[[f64; 2]],
    frame_times: &[f64],
    existing_tracks: &[Value],
    options: &StackFaintOptions,
) -> Result<StackFaintResult> {
    if !options.threshold_sigma.is_finite()
        || options.threshold_sigma <= 0.0
        || !options.min_flux_snr.is_finite()
        || options.min_flux_snr <= 0.0
        || !options.frame_min_flux_snr.is_finite()
        || options.frame_min_flux_snr <= 0.0
        || !options.link_radius_px.is_finite()
        || options.link_radius_px <= 0.0
        || options.min_presence == 0
        || options
            .max_reference_sources
            .is_some_and(|limit| limit == 0)
    {
        return Err("stack-faint thresholds and source limits must be positive".into());
    }
    if !matches!(options.reference_mode.as_str(), "median" | "coadd") {
        return Err("stack_reference_mode must be 'median' or 'coadd'".into());
    }
    if !options.enabled || analyses.len() < 2 {
        return Ok(empty_result());
    }
    if analyses.len() != cumulative_shifts.len() || frame_times.len() != analyses.len() {
        return Err("stack-faint frame, shift, and time counts must match".into());
    }
    let reference = sequence_temporal::registered_reference_from_analyses(
        analyses,
        cumulative_shifts,
        options.reference_mode == "coadd",
        3.0,
    )?;
    let Some(reference) = reference else {
        return Ok(empty_result());
    };

    let reference_height = reference.height;
    let reference_width = reference.width;
    let reference_data = PixelData::F32(reference.values);
    let PixelData::F32(reference_values) = &reference_data else {
        unreachable!("registered reference data remains float32");
    };
    let mut reference_mask = fits::auxiliary_mask(reference_height, reference_width)?;
    for (index, value) in reference_values.iter().enumerate() {
        if !value.is_finite() {
            reference_mask[index] = true;
        }
    }
    let stack_detection_options = DetectionOptions {
        measurement: MeasurementOptions {
            aperture_radius: options.aperture_radius,
            min_flux_snr: options.min_flux_snr,
            min_psf_support_pixels: options.min_psf_support_pixels,
            psf_fwhm: options.psf_fwhm,
            refine_local_background: false,
            allow_partial_zero_mask: false,
            ..MeasurementOptions::default()
        },
        threshold_sigma: options.threshold_sigma,
        min_distance: options.min_distance,
        max_sources: None,
        proposal_mode: options.proposal_mode.clone(),
        mask_zero_pixels: Some(false),
        allow_partial_zero_mask: Some(false),
        use_float32: true,
        fast_sequence: true,
        ..DetectionOptions::default()
    };
    let detection = detection::detect_sources(
        &reference_data,
        Some(&reference_mask),
        reference_height,
        reference_width,
        &stack_detection_options,
        None,
    )?;
    let mut candidates = detection
        .sources
        .iter()
        .enumerate()
        .filter(|(_, source)| source.quality_passed)
        .collect::<Vec<_>>();
    candidates.sort_by(|(left_index, left), (right_index, right)| {
        right
            .filter_snr
            .unwrap_or(f64::NEG_INFINITY)
            .total_cmp(&left.filter_snr.unwrap_or(f64::NEG_INFINITY))
            .then_with(|| left_index.cmp(right_index))
    });
    let candidate_count = candidates.len();
    if let Some(limit) = options.max_reference_sources {
        candidates.truncate(limit);
    }
    let working_count = candidates.len();
    if candidates.is_empty() {
        let mut result = empty_result();
        result.candidate_count = candidate_count;
        result.working_count = working_count;
        result.audit["stack_reference_quality_sources"] = json!(candidate_count);
        result.audit["stack_reference_working_sources"] = json!(working_count);
        return Ok(result);
    }

    let positions = track_positions(existing_tracks);
    let explained_tree = KdTree::new(&positions)?;
    let mut candidate_rows = Vec::with_capacity(candidates.len());
    let mut reject_explained = 0usize;
    for (_, source) in candidates {
        let [x, y] = source.peak_position();
        if explained_tree
            .nearest([x, y])
            .is_some_and(|(_, distance)| distance <= options.link_radius_px)
        {
            reject_explained += 1;
            continue;
        }
        candidate_rows.push((x, y, candidate_snr(source)));
    }
    if candidate_rows.is_empty() {
        let mut result = empty_result();
        result.candidate_count = candidate_count;
        result.working_count = working_count;
        result.audit["stack_reference_quality_sources"] = json!(candidate_count);
        result.audit["stack_reference_working_sources"] = json!(working_count);
        result.audit["reject_stack_explained"] = json!(reject_explained);
        return Ok(result);
    }

    let mut measured_by_frame = Vec::with_capacity(analyses.len());
    for (frame_index, analysis) in analyses.iter().enumerate() {
        // Convert and mask one image at a time so a 15-frame 4096² sequence does not
        // allocate fifteen extra float64 copies while the original FITS buffers remain live.
        let frame_image = analysis.frame.data.to_f64();
        let mut frame_mask = fits::auxiliary_mask(analysis.frame.height, analysis.frame.width)?;
        let mask_zero = analysis
            .detection
            .parameters
            .get("mask_zero_pixels")
            .and_then(Value::as_bool)
            .unwrap_or(options.mask_zero_pixels);
        for (index, value) in frame_image.iter().enumerate() {
            if !value.is_finite() || (mask_zero && *value == 0.0) {
                frame_mask[index] = true;
            }
        }
        let [shift_x, shift_y] = cumulative_shifts[frame_index];
        let mut measured = Vec::with_capacity(candidate_rows.len());
        for start in (0..candidate_rows.len()).step_by(MAX_FRAME_CHUNK) {
            let end = (start + MAX_FRAME_CHUNK).min(candidate_rows.len());
            let xs = candidate_rows[start..end]
                .iter()
                .map(|row| row.0 + shift_x)
                .collect::<Vec<_>>();
            let ys = candidate_rows[start..end]
                .iter()
                .map(|row| row.1 + shift_y)
                .collect::<Vec<_>>();
            let frame_measurements = sequence_photometry::stack_forced_measure_batch(
                FramePixels {
                    image: &frame_image,
                    mask: &frame_mask,
                    height: analysis.frame.height,
                    width: analysis.frame.width,
                },
                &xs,
                &ys,
                StackMeasurementOptions {
                    aperture_radius: options.aperture_radius,
                    global_background: analysis.detection.background,
                    global_noise: analysis.detection.noise,
                },
            )?;
            measured.extend(frame_measurements);
        }
        measured_by_frame.push(measured);
    }

    let mut tracks = Vec::new();
    let mut reject_presence = 0usize;
    let mut reject_flux_snr = 0usize;
    for (candidate_index, (reference_x, reference_y, proposal_snr)) in
        candidate_rows.iter().enumerate()
    {
        let mut points = Vec::new();
        let mut flux_snrs = Vec::new();
        for frame_index in 0..analyses.len() {
            let (flux_snr, support, center_valid, fwhm, ellipticity) =
                measured_by_frame[frame_index][candidate_index];
            let valid_shape = fwhm
                .is_some_and(|value| value.is_finite() && (0.8..=12.0).contains(&value))
                && ellipticity.is_some_and(|value| value.is_finite() && value <= 0.65);
            if !center_valid
                || support < options.min_psf_support_pixels as i16
                || !flux_snr
                    .is_some_and(|value| value.is_finite() && value >= options.frame_min_flux_snr)
                || !valid_shape
            {
                continue;
            }
            let flux_snr = flux_snr.unwrap();
            let [shift_x, shift_y] = cumulative_shifts[frame_index];
            points.push(TrackPoint {
                frame_index,
                detection_id: -1,
                x: reference_x + shift_x,
                y: reference_y + shift_y,
                aligned_x: *reference_x,
                aligned_y: *reference_y,
                flux_snr: Some(flux_snr),
                quality_passed: false,
                candidate_snr: proposal_snr.as_f64().or(Some(flux_snr)),
            });
            flux_snrs.push(flux_snr);
        }
        if points.len() < options.min_presence {
            reject_presence += 1;
            continue;
        }
        if median(&mut flux_snrs) < options.frame_min_flux_snr {
            reject_flux_snr += 1;
            continue;
        }
        let observations = points
            .iter()
            .map(|point| TrackObservation {
                frame_index: point.frame_index as i64,
                aligned_x: point.aligned_x,
                aligned_y: point.aligned_y,
            })
            .collect::<Vec<_>>();
        let fit = sequence_motion::fit_track_motion(&observations, Some(frame_times))?;
        tracks.push(json!({
            "track_id": -1,
            "classification":"persistent",
            "presence":points.len(),
            "displacement_px":fit.displacement_px,
            "speed_px_per_frame":fit.speed_px_per_frame,
            "fit_rms_px":fit.fit_rms_px,
            "quality_presence":0,
            "evidence_level":"stack_faint",
            "points":points,
        }));
    }

    let accepted = tracks.len();
    Ok(StackFaintResult {
        tracks,
        candidate_count,
        working_count,
        audit: json!({
            "stack_reference_quality_sources":candidate_count,
            "stack_reference_working_sources":working_count,
            "reject_stack_explained":reject_explained,
            "reject_stack_presence":reject_presence,
            "reject_stack_flux_snr":reject_flux_snr,
            "accepted_stack_faint":accepted,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        detection::DetectionResult,
        models::{FitsFrame, PixelData},
    };
    use std::path::PathBuf;

    fn synthetic_analysis(frame_index: usize) -> FrameAnalysis {
        let (width, height) = (48, 48);
        let mut state = 0x9e37_79b9u32.wrapping_add((frame_index as u32).wrapping_mul(0x85eb_ca6b));
        let data = (0..width * height)
            .map(|index| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = ((state >> 8) as f64 / 16_777_215.0 - 0.5) * 1.2;
                let x = (index % width) as f64;
                let y = (index / width) as f64;
                let radius_squared = (x - 24.2).powi(2) + (y - 23.8).powi(2);
                (3.0 * (-radius_squared / 2.0).exp() + noise) as f32
            })
            .collect::<Vec<_>>();
        FrameAnalysis {
            frame: FitsFrame {
                path: PathBuf::from(format!("synthetic-{frame_index}.fits")),
                header: serde_json::Map::new(),
                data: PixelData::F32(data),
                auxiliary: None,
                data_offset: 0,
                width,
                height,
            },
            detection: DetectionResult {
                image_shape: [height, width],
                background: 0.0,
                noise: 1.0,
                threshold: 4.0,
                candidate_count: 0,
                sources: Vec::new(),
                quality_count: 0,
                parameters: json!({"mask_zero_pixels":false}),
                candidate_peaks: Vec::new(),
                line_artifact_coordinates: Vec::new(),
            },
            matching: None,
            faintest: None,
            photometric_calibration: None,
            source_photometry: Vec::new(),
        }
    }

    #[test]
    fn disabled_stack_recovery_does_not_require_frames_or_build_a_reference() {
        let mut options = StackFaintOptions::default();
        options.enabled = false;
        let result = recover_stack_faint_tracks(&[], &[], &[], &[], &options).unwrap();
        assert!(result.tracks.is_empty());
        assert_eq!(result.audit["accepted_stack_faint"], 0);
    }

    #[test]
    fn stack_recovery_rejects_non_positive_thresholds() {
        let mut options = StackFaintOptions::default();
        options.min_flux_snr = 0.0;
        let result = recover_stack_faint_tracks(&[], &[], &[], &[], &options);
        assert_eq!(
            result.unwrap_err(),
            "stack-faint thresholds and source limits must be positive"
        );
    }

    #[test]
    fn registered_stack_candidates_are_rechecked_and_kept_separate_from_quality_tracks() {
        let analyses = (0..5).map(synthetic_analysis).collect::<Vec<_>>();
        let shifts = vec![[0.0, 0.0]; analyses.len()];
        let frame_times = (0..analyses.len())
            .map(|index| index as f64)
            .collect::<Vec<_>>();
        let options = StackFaintOptions {
            threshold_sigma: 2.5,
            min_flux_snr: 4.0,
            frame_min_flux_snr: 3.0,
            min_presence: 3,
            aperture_radius: 3,
            psf_fwhm: 2.0,
            ..StackFaintOptions::default()
        };
        let recovered =
            recover_stack_faint_tracks(&analyses, &shifts, &frame_times, &[], &options).unwrap();
        assert!(recovered.candidate_count > 0);
        assert!(recovered.tracks.iter().any(|track| {
            track["evidence_level"] == "stack_faint"
                && track["classification"] == "persistent"
                && track["presence"].as_u64().unwrap_or_default() >= 3
        }));

        let explained = json!([{
            "points":[{"aligned_x":24.0,"aligned_y":24.0}]
        }]);
        let explained = recover_stack_faint_tracks(
            &analyses,
            &shifts,
            &frame_times,
            explained.as_array().unwrap(),
            &options,
        )
        .unwrap();
        assert!(explained.tracks.is_empty());
        assert!(
            explained.audit["reject_stack_explained"]
                .as_u64()
                .unwrap_or_default()
                > 0
        );
    }
}
