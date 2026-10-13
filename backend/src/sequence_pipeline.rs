//! Native FITS-to-tracks orchestration for the sequence core.
//!
//! This module exposes the migrated detection and quality-track stages
//! separately from temporal candidate recovery, trail analysis, and relative
//! photometry, which remain later sequence stages.
use crate::{
    detection::{self, BackgroundModel},
    fits,
    pipeline::{self, FrameAnalysis, FrameOptions},
    sequence::{self, TrackPoint, TrackSequenceOptions},
    sequence_fast::{self, FastPointOptions},
    sequence_fast_candidates::{self, FastCandidateOptions, FastCandidateRecord},
    sequence_motion::{self, TrackObservation},
    sequence_photometry::{self, FramePixels, StackMeasurement, StackMeasurementOptions},
    sequence_stack::{self, StackFaintOptions},
    sequence_trails,
    spatial::KdTree,
    Result,
};
use serde_json::{json, Map, Value};
use std::{collections::HashMap, path::PathBuf};

const DEFAULT_SEQUENCE_MAX_SOURCES: usize = 6_000;
const DEFAULT_SEQUENCE_BACKGROUND_BOX_SIZE: usize = 256;
const DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT: usize = 4_096;
const DEFAULT_FAST_POINT_CANDIDATE_WORKING_LIMIT: usize = 20_000;
const DEFAULT_SEQUENCE_STACK_WORKING_LIMIT: usize = 8_000;

#[derive(Debug, Clone, Copy)]
struct PreparedFastCandidate {
    row_index: usize,
    raw_x: f64,
    raw_y: f64,
    filter_snr: f64,
}

fn object_or_empty(value: Option<&Value>, name: &str) -> Result<Map<String, Value>> {
    match value {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(object)) => Ok(object.clone()),
        _ => Err(format!("{name} must be an object")),
    }
}

fn optional_request_f64(request: &Value, key: &str) -> Result<Option<f64>> {
    match request.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(value)) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("{key} must be a finite number")),
        Some(_) => Err(format!("{key} must be a number")),
    }
}

fn optional_request_bool(request: &Value, key: &str, default: bool) -> Result<bool> {
    match request.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(format!("{key} must be a boolean")),
    }
}

fn optional_request_usize(request: &Value, key: &str, default: usize) -> Result<usize> {
    match request.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|_| format!("{key} must be a non-negative integer")),
    }
}

fn optional_request_limit(request: &Value, key: &str, default: usize) -> Result<Option<usize>> {
    match request.get(key) {
        None => Ok(Some(default)),
        Some(Value::Null) => Ok(None),
        Some(value) => serde_json::from_value(value.clone())
            .map(Some)
            .map_err(|_| format!("{key} must be a non-negative integer or null")),
    }
}

fn sequence_track_counts(tracks: &[Value]) -> Value {
    let count =
        |predicate: &dyn Fn(&Value) -> bool| tracks.iter().filter(|track| predicate(track)).count();
    let class_is = |class: &'static str| {
        move |track: &Value| track.get("classification").and_then(Value::as_str) == Some(class)
    };
    let evidence_is = |level: &'static str| {
        move |track: &Value| track.get("evidence_level").and_then(Value::as_str) == Some(level)
    };
    let static_count = count(&class_is("static"));
    let persistent_count =
        count(&|track| class_is("persistent")(track) && !evidence_is("stack_faint")(track));
    json!({
        "stable_source_count":static_count,
        "persistent_source_count":persistent_count,
        "stable_field_candidate_count":static_count+persistent_count,
        "candidate_consensus_count":count(&|track| evidence_is("candidate_consensus")(track) || evidence_is("temporal_reference")(track)),
        "temporal_reference_count":count(&evidence_is("temporal_reference")),
        "stack_faint_count":count(&evidence_is("stack_faint")),
        "moving_track_count":count(&class_is("moving")),
        "transient_track_count":count(&class_is("transient")),
    })
}

fn sequence_frame_options(request: &Value) -> Result<FrameOptions> {
    let mut options = object_or_empty(request.get("frame_options"), "frame_options")?;
    options
        .entry("background_box_size")
        .or_insert(json!(DEFAULT_SEQUENCE_BACKGROUND_BOX_SIZE));
    options
        .entry("refine_local_background")
        .or_insert(json!(false));
    options.entry("use_float32").or_insert(json!(true));
    let use_float32 = options
        .get("use_float32")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    options
        .entry("background_sample_limit")
        .or_insert(json!(if use_float32 {
            DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT
        } else {
            100_000
        }));
    options.entry("fast_sequence").or_insert(json!(use_float32));
    if options.get("max_sources").is_none_or(Value::is_null) {
        let max_sources = match request.get("sequence_max_sources") {
            Some(value) => value.clone(),
            None => json!(DEFAULT_SEQUENCE_MAX_SOURCES),
        };
        options.insert("max_sources".into(), max_sources);
    }
    serde_json::from_value(Value::Object(options)).map_err(|error| error.to_string())
}

fn prefilter_fast_candidate_peaks(
    peaks: &[[f64; 3]],
    line_artifact_coordinates: &[[usize; 2]],
    image_shape: [usize; 2],
    min_filter_snr: f64,
    candidate_limit: usize,
    aperture_radius: usize,
) -> Result<(Vec<PreparedFastCandidate>, Value)> {
    if !min_filter_snr.is_finite() || min_filter_snr <= 0.0 || candidate_limit == 0 {
        return Err("candidate working limit and filter SNR must be positive".into());
    }
    let [height, width] = image_shape;
    let mut finite_count = 0usize;
    let mut above_threshold = Vec::new();
    for (row_index, peak) in peaks.iter().enumerate() {
        if peak.iter().all(|value| value.is_finite()) {
            finite_count += 1;
            if peak[2] >= min_filter_snr {
                above_threshold.push(PreparedFastCandidate {
                    row_index,
                    raw_x: peak[0],
                    raw_y: peak[1],
                    filter_snr: peak[2],
                });
            }
        }
    }
    above_threshold.sort_by(|left, right| {
        right
            .filter_snr
            .total_cmp(&left.filter_snr)
            .then_with(|| left.row_index.cmp(&right.row_index))
    });
    let threshold_count = above_threshold.len();
    above_threshold.truncate(candidate_limit);
    let working_set_count = above_threshold.len();

    let safe_margin = aperture_radius.saturating_add(4) as f64;
    let safe_max_x = width as f64 - safe_margin;
    let safe_max_y = height as f64 - safe_margin;
    let mut edge_rejected_count = 0usize;
    let mut candidates = Vec::with_capacity(working_set_count);
    let line_mask = if line_artifact_coordinates.is_empty() {
        None
    } else {
        let size = height
            .checked_mul(width)
            .ok_or("image dimensions overflow while indexing line artifacts")?;
        let mut mask = vec![false; size];
        for [x, y] in line_artifact_coordinates {
            if *x < width && *y < height {
                mask[*y * width + *x] = true;
            }
        }
        Some(mask)
    };
    let artifact_radius = aperture_radius as f64 + 0.5;
    let mut line_artifact_rejected_count = 0usize;
    for candidate in above_threshold {
        if candidate.raw_x < safe_margin
            || candidate.raw_x >= safe_max_x
            || candidate.raw_y < safe_margin
            || candidate.raw_y >= safe_max_y
        {
            edge_rejected_count += 1;
            continue;
        }
        if line_mask.as_ref().is_some_and(|mask| {
            near_line_artifact(
                candidate.raw_x,
                candidate.raw_y,
                mask,
                width,
                height,
                artifact_radius,
            )
        }) {
            line_artifact_rejected_count += 1;
            continue;
        }
        candidates.push(candidate);
    }

    let audit = json!({
        "raw_peak_count": peaks.len(),
        "finite_peak_count": finite_count,
        "min_filter_snr": min_filter_snr,
        "min_filter_snr_pass_count": threshold_count,
        "candidate_working_limit": candidate_limit,
        "working_set_count": working_set_count,
        "working_set_truncated_count": threshold_count.saturating_sub(working_set_count),
        "edge_rejected_count": edge_rejected_count,
        "line_artifact_rejected_count": line_artifact_rejected_count,
        "explained_rejected_count": 0,
        "association_input_count": candidates.len(),
    });
    Ok((candidates, audit))
}

fn near_line_artifact(
    x: f64,
    y: f64,
    line_mask: &[bool],
    width: usize,
    height: usize,
    radius: f64,
) -> bool {
    if width == 0 || height == 0 || !radius.is_finite() || radius < 0.0 {
        return false;
    }
    let x_min = (x - radius).floor().max(0.0) as usize;
    let y_min = (y - radius).floor().max(0.0) as usize;
    let x_max = (x + radius).ceil().min((width - 1) as f64) as usize;
    let y_max = (y + radius).ceil().min((height - 1) as f64) as usize;
    let radius_squared = radius * radius;
    for row in y_min..=y_max {
        for column in x_min..=x_max {
            if line_mask[row * width + column] {
                let dx = x - column as f64;
                let dy = y - row as f64;
                if dx * dx + dy * dy <= radius_squared {
                    return true;
                }
            }
        }
    }
    false
}

fn explained_candidate_trees(
    tracking: &Value,
    fast_point_tracks: &[Value],
    frame_count: usize,
) -> Result<Vec<KdTree>> {
    let mut points = vec![Vec::<[f64; 2]>::new(); frame_count];
    let quality_tracks = tracking["tracks"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    for track in quality_tracks.iter().chain(fast_point_tracks) {
        if track["presence"].as_u64().unwrap_or(0) < 2 {
            continue;
        }
        let Some(observations) = track["points"].as_array() else {
            continue;
        };
        for observation in observations {
            let Some(frame_index) = observation["frame_index"].as_u64().map(|v| v as usize) else {
                continue;
            };
            let (Some(x), Some(y)) = (
                observation["aligned_x"].as_f64(),
                observation["aligned_y"].as_f64(),
            ) else {
                continue;
            };
            if frame_index < frame_count {
                points[frame_index].push([x, y]);
            }
        }
    }
    points.iter().map(|rows| KdTree::new(rows)).collect()
}

fn fast_candidate_proposals(
    prepared_by_frame: &[Vec<PreparedFastCandidate>],
    mut frame_audits: Vec<Value>,
    shifts: &[[f64; 2]],
    frame_times: &[f64],
    options: &FastPointOptions,
    tracking: &Value,
    existing_fast_point_tracks: &[Value],
) -> Result<Value> {
    let frame_count = prepared_by_frame.len();
    let trees = explained_candidate_trees(tracking, existing_fast_point_tracks, frame_count)?;
    let mut records_by_frame = Vec::with_capacity(frame_count);
    let mut raw_candidates_by_frame = Vec::with_capacity(frame_count);
    for frame_index in 0..frame_count {
        let mut records = Vec::new();
        let mut raw_candidates = HashMap::new();
        let mut explained_rejected_count = 0usize;
        for candidate in &prepared_by_frame[frame_index] {
            let aligned_x = candidate.raw_x - shifts[frame_index][0];
            let aligned_y = candidate.raw_y - shifts[frame_index][1];
            if trees[frame_index]
                .nearest([aligned_x, aligned_y])
                .is_some_and(|(_, distance)| distance <= options.link_radius_px)
            {
                explained_rejected_count += 1;
                continue;
            }
            records.push(FastCandidateRecord {
                row_index: candidate.row_index,
                aligned_x,
                aligned_y,
                filter_snr: candidate.filter_snr,
            });
            raw_candidates.insert(candidate.row_index, *candidate);
        }
        frame_audits[frame_index]["explained_rejected_count"] = json!(explained_rejected_count);
        frame_audits[frame_index]["association_input_count"] = json!(records.len());
        records_by_frame.push(records);
        raw_candidates_by_frame.push(raw_candidates);
    }

    let tracking_min_presence = tracking["min_presence"].as_u64().unwrap_or(3) as usize;
    let association_options = FastCandidateOptions {
        link_radius_px: options.link_radius_px,
        min_presence: tracking_min_presence.max(3),
        motion_min_displacement_px: options.motion_min_displacement_px,
        max_step_px: options.max_step_px,
        gate_px: options.gate_px,
        max_fit_rms_px: options.max_fit_rms_px,
    };
    let associated = sequence_fast_candidates::associate_fast_candidates(
        &records_by_frame,
        Some(frame_times),
        &association_options,
    )?;
    let mut proposals = Vec::new();
    if let Some(tracks) = associated["tracks"].as_array() {
        for track in tracks {
            let mut observations = Vec::new();
            if let Some(rows) = track["observations"].as_array() {
                for row in rows {
                    let frame_index = row["frame_index"]
                        .as_u64()
                        .ok_or("fast candidate frame index is invalid")?
                        as usize;
                    let row_index = row["row_index"]
                        .as_u64()
                        .ok_or("fast candidate row index is invalid")?
                        as usize;
                    let candidate = raw_candidates_by_frame
                        .get(frame_index)
                        .and_then(|frame| frame.get(&row_index))
                        .ok_or("fast candidate association returned an unknown peak")?;
                    observations.push(json!({
                        "frame_index": frame_index,
                        "row_index": row_index,
                        "x": candidate.raw_x,
                        "y": candidate.raw_y,
                        "aligned_x": candidate.raw_x - shifts[frame_index][0],
                        "aligned_y": candidate.raw_y - shifts[frame_index][1],
                        "filter_snr": candidate.filter_snr,
                    }));
                }
            }
            proposals.push(json!({
                "proposal_id": proposals.len(),
                "verification_status":"requires_original_pixel_forced_photometry_and_quality_gates",
                "observations":observations,
            }));
        }
    }
    let status = if proposals.is_empty() {
        "no_candidate_tracks"
    } else {
        "proposals_available"
    };
    Ok(json!({
        "status":status,
        "scope":"matched_filter_coordinate_association_and_original_pixel_quality_gates",
        "evidence_level":"unverified_candidate_proposal",
        "frame_audits":frame_audits,
        "proposal_count":proposals.len(),
        "proposals":proposals,
    }))
}

fn verify_fast_candidate_proposals(
    candidate_stage: &mut Value,
    analyses: &[FrameAnalysis],
    frame_times: &[f64],
    tracking_min_presence: usize,
    first_track_id: usize,
    frame_options: &FrameOptions,
    fast_point_options: &FastPointOptions,
) -> Result<Vec<Value>> {
    let frame_count = analyses.len();
    let proposal_rows = candidate_stage["proposals"]
        .as_array()
        .cloned()
        .ok_or("fast candidate proposals must be an array")?;
    let mut coordinates_by_frame = vec![HashMap::<usize, (f64, f64)>::new(); frame_count];
    for proposal in &proposal_rows {
        let observations = proposal["observations"]
            .as_array()
            .ok_or("fast candidate observations must be an array")?;
        for observation in observations {
            let frame_index = observation["frame_index"]
                .as_u64()
                .ok_or("fast candidate frame index is invalid")?
                as usize;
            let row_index = observation["row_index"]
                .as_u64()
                .ok_or("fast candidate row index is invalid")? as usize;
            let x = observation["x"]
                .as_f64()
                .ok_or("fast candidate x coordinate is invalid")?;
            let y = observation["y"]
                .as_f64()
                .ok_or("fast candidate y coordinate is invalid")?;
            if frame_index >= frame_count {
                return Err("fast candidate frame index is out of range".into());
            }
            coordinates_by_frame[frame_index].insert(row_index, (x, y));
        }
    }

    let aperture_radius = frame_options.detection.measurement.aperture_radius;
    let mut measurements_by_frame = vec![HashMap::<usize, StackMeasurement>::new(); frame_count];
    for frame_index in 0..frame_count {
        let coordinates = &coordinates_by_frame[frame_index];
        if coordinates.is_empty() {
            continue;
        }
        let analysis = &analyses[frame_index];
        let height = analysis.frame.height;
        let width = analysis.frame.width;
        let provided_mask = fits::auxiliary_mask(height, width)?;
        let parameters = &analysis.detection.parameters;
        let mask_zero_pixels = parameters
            .get("mask_zero_pixels")
            .and_then(Value::as_u64)
            .map(|value| value != 0);
        let saturation_level = parameters
            .get("saturation_level")
            .and_then(Value::as_f64)
            .filter(|value| *value > 0.0);
        let (effective_mask, _, _) = detection::working_mask_float32(
            &analysis.frame.data,
            Some(&provided_mask),
            analysis.detection.background,
            mask_zero_pixels,
            saturation_level,
        )?;
        let image = analysis.frame.data.to_f64();
        let mut rows = coordinates.keys().copied().collect::<Vec<_>>();
        rows.sort_unstable();
        let xs = rows
            .iter()
            .map(|row| coordinates[row].0)
            .collect::<Vec<_>>();
        let ys = rows
            .iter()
            .map(|row| coordinates[row].1)
            .collect::<Vec<_>>();
        let results = sequence_photometry::stack_forced_measure_batch(
            FramePixels {
                image: &image,
                mask: &effective_mask,
                height,
                width,
            },
            &xs,
            &ys,
            StackMeasurementOptions {
                aperture_radius,
                global_background: analysis.detection.background,
                global_noise: analysis.detection.noise,
            },
        )?;
        measurements_by_frame[frame_index].extend(rows.into_iter().zip(results));
    }

    let measurement = &frame_options.detection.measurement;
    let min_presence = tracking_min_presence.max(3);
    let mut accepted_tracks = Vec::new();
    let mut accepted_keys = Vec::<std::collections::HashSet<(usize, i64)>>::new();
    let proposals = candidate_stage["proposals"]
        .as_array_mut()
        .ok_or("fast candidate proposals must be an array")?;
    for proposal in proposals {
        let rows = proposal["observations"]
            .as_array_mut()
            .ok_or("fast candidate observations must be an array")?;
        let mut points = Vec::<TrackPoint>::new();
        for observation in rows {
            let frame_index = observation["frame_index"]
                .as_u64()
                .ok_or("fast candidate frame index is invalid")?
                as usize;
            let row_index = observation["row_index"]
                .as_u64()
                .ok_or("fast candidate row index is invalid")? as usize;
            let x = observation["x"]
                .as_f64()
                .ok_or("fast candidate x coordinate is invalid")?;
            let y = observation["y"]
                .as_f64()
                .ok_or("fast candidate y coordinate is invalid")?;
            let aligned_x = observation["aligned_x"]
                .as_f64()
                .ok_or("fast candidate aligned x coordinate is invalid")?;
            let aligned_y = observation["aligned_y"]
                .as_f64()
                .ok_or("fast candidate aligned y coordinate is invalid")?;
            let filter_snr = observation["filter_snr"]
                .as_f64()
                .ok_or("fast candidate filter SNR is invalid")?;
            let (flux_snr, support, center_valid, fwhm, ellipticity) = measurements_by_frame
                .get(frame_index)
                .and_then(|frame| frame.get(&row_index))
                .copied()
                .ok_or("fast candidate original-pixel measurement is missing")?;
            let mut quality_reasons = Vec::new();
            if !center_valid {
                quality_reasons.push("invalid_center");
            }
            if support < 0 || (support as usize) < measurement.min_psf_support_pixels {
                quality_reasons.push("insufficient_psf_support");
            }
            if !flux_snr
                .is_some_and(|value| value.is_finite() && value >= fast_point_options.min_flux_snr)
            {
                quality_reasons.push("flux_snr_below_threshold");
            }
            if !fwhm.is_some_and(|value| {
                value.is_finite() && measurement.min_fwhm <= value && value <= measurement.max_fwhm
            }) {
                quality_reasons.push("fwhm_outside_range");
            }
            if !ellipticity
                .is_some_and(|value| value.is_finite() && value <= measurement.max_ellipticity)
            {
                quality_reasons.push("ellipticity_above_threshold");
            }
            let frame_quality_passed = quality_reasons.is_empty();
            observation["original_pixel_measurement"] = json!({
                "flux_snr":flux_snr,
                "psf_support_pixels":support,
                "center_valid":center_valid,
                "fwhm":fwhm,
                "ellipticity":ellipticity,
            });
            observation["quality_gate_passed"] = json!(frame_quality_passed);
            observation["quality_rejection_reasons"] = json!(quality_reasons);
            if frame_quality_passed {
                let detection_id = i64::try_from(row_index)
                    .map_err(|_| "fast candidate row index exceeds signed ID range")?
                    .checked_neg()
                    .and_then(|value| value.checked_sub(1))
                    .ok_or("fast candidate row index exceeds signed ID range")?;
                points.push(TrackPoint {
                    frame_index,
                    detection_id,
                    x,
                    y,
                    aligned_x,
                    aligned_y,
                    flux_snr,
                    quality_passed: true,
                    candidate_snr: Some(filter_snr),
                });
            }
        }

        let mut rejection_reasons = Vec::new();
        if points.len() < min_presence {
            rejection_reasons.push("insufficient_quality_observations");
        }
        if points
            .windows(2)
            .any(|pair| pair[1].frame_index - pair[0].frame_index > 2)
        {
            rejection_reasons.push("observation_gap_exceeded");
        }
        let motion_fit = if points.len() >= 2 {
            Some(sequence_motion::fit_track_motion(
                &points
                    .iter()
                    .map(|point| TrackObservation {
                        frame_index: point.frame_index as i64,
                        aligned_x: point.aligned_x,
                        aligned_y: point.aligned_y,
                    })
                    .collect::<Vec<_>>(),
                Some(frame_times),
            )?)
        } else {
            None
        };
        if motion_fit.is_none() {
            rejection_reasons.push("motion_fit_unavailable");
        }
        if let Some(fit) = motion_fit {
            let min_speed = (fast_point_options.link_radius_px * 1.01).max(
                fast_point_options.motion_min_displacement_px
                    / (frame_count.saturating_sub(1).max(1) as f64),
            );
            if fit
                .fit_rms_px
                .is_none_or(|rms| !rms.is_finite() || rms > fast_point_options.max_fit_rms_px)
            {
                rejection_reasons.push("motion_fit_rms_exceeded");
            }
            if fit.speed_px_per_frame < min_speed {
                rejection_reasons.push("speed_below_minimum");
            }
            if fit.speed_px_per_frame > fast_point_options.max_step_px {
                rejection_reasons.push("speed_above_maximum");
            }
            if fit.displacement_px < fast_point_options.motion_min_displacement_px {
                rejection_reasons.push("displacement_below_minimum");
            }
        }
        let keys = points
            .iter()
            .map(|point| (point.frame_index, point.detection_id))
            .collect::<std::collections::HashSet<_>>();
        let overlaps_accepted = accepted_keys.iter().any(|previous| {
            let shared = keys.intersection(previous).count();
            shared >= 2usize.max(keys.len().min(previous.len()).div_ceil(2))
        });
        if overlaps_accepted {
            rejection_reasons.push("overlaps_accepted_candidate");
        }

        let accepted = rejection_reasons.is_empty();
        proposal["verification_status"] = json!(if accepted {
            "original_pixel_quality_and_motion_gates_passed"
        } else {
            "rejected_by_original_pixel_or_motion_gates"
        });
        proposal["accepted_as_fast_point_track"] = json!(accepted);
        proposal["quality_observation_count"] = json!(points.len());
        proposal["rejection_reasons"] = json!(rejection_reasons);
        if accepted {
            let fit = motion_fit.ok_or("accepted candidate is missing its motion fit")?;
            let track_id = first_track_id + accepted_tracks.len();
            proposal["track_id"] = json!(track_id);
            accepted_keys.push(keys);
            accepted_tracks.push(json!({
                "track_id":track_id,
                "classification":"moving",
                "presence":points.len(),
                "displacement_px":fit.displacement_px,
                "speed_px_per_frame":fit.speed_px_per_frame,
                "fit_rms_px":fit.fit_rms_px,
                "quality_presence":points.iter().filter(|point|point.quality_passed).count(),
                "evidence_level":"fast_point_motion",
                "points":points,
            }));
        }
    }
    candidate_stage["original_pixel_measurement_applied"] = json!(true);
    candidate_stage["accepted_track_count"] = json!(accepted_tracks.len());
    candidate_stage["claim_boundary"] = json!(
        "algorithmic_candidate_quality_and_motion_evidence_does_not_establish_physical_identity"
    );
    Ok(accepted_tracks)
}

fn fit_sequence_relative_photometry(
    quality_frames: &[Vec<crate::measurement::Detection>],
    headers: &[Value],
    tracking: &Value,
    min_presence: usize,
    request: &Value,
) -> Result<Value> {
    if quality_frames.len() != headers.len() {
        return Err("relative photometry frame data is misaligned".into());
    }
    let max_sources = request
        .get("relative_photometry_max_sources")
        .map(|value| serde_json::from_value::<usize>(value.clone()))
        .transpose()
        .map_err(|_| "relative_photometry_max_sources must be a positive integer")?
        .unwrap_or(400);
    if max_sources == 0 {
        return Err("relative_photometry_max_sources must be a positive integer".into());
    }

    let mut eligible = Vec::<(f64, usize)>::new();
    let mut track_by_observation = HashMap::<(usize, i64), usize>::new();
    for track in tracking["tracks"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        if track["classification"].as_str() != Some("static")
            || track["presence"].as_u64().unwrap_or(0) < min_presence as u64
        {
            continue;
        }
        let Some(track_id) = track["track_id"].as_u64().map(|value| value as usize) else {
            continue;
        };
        let points = track["points"].as_array().map(Vec::as_slice).unwrap_or(&[]);
        let mut snr_values = points
            .iter()
            .filter_map(|point| point["flux_snr"].as_f64())
            .filter(|value| value.is_finite() && *value > 0.0)
            .collect::<Vec<_>>();
        if snr_values.is_empty() {
            continue;
        }
        snr_values.sort_by(f64::total_cmp);
        let middle = snr_values.len() / 2;
        let median_snr = if snr_values.len().is_multiple_of(2) {
            (snr_values[middle - 1] + snr_values[middle]) * 0.5
        } else {
            snr_values[middle]
        };
        eligible.push((median_snr, track_id));
        for point in points {
            let (Some(frame_index), Some(detection_id)) = (
                point["frame_index"].as_u64().map(|value| value as usize),
                point["detection_id"].as_i64(),
            ) else {
                continue;
            };
            track_by_observation.insert((frame_index, detection_id), track_id);
        }
    }
    eligible.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let selected = eligible
        .into_iter()
        .take(max_sources)
        .map(|(_, track_id)| track_id)
        .collect::<std::collections::HashSet<_>>();

    let mut observations = Vec::new();
    for (frame_index, detections) in quality_frames.iter().enumerate() {
        let header = headers[frame_index]
            .as_object()
            .ok_or("relative photometry frame header is invalid")?;
        let exposure = fits::exposure_seconds(header, 1.0)?;
        for source in detections {
            let Some(track_id) = track_by_observation
                .get(&(frame_index, source.detection_id))
                .filter(|track_id| selected.contains(track_id))
            else {
                continue;
            };
            let Some(magnitude) = crate::photometry::instrumental_magnitude(source.flux, exposure)?
            else {
                continue;
            };
            let signal_snr = source.flux_snr.unwrap_or(source.snr);
            if !signal_snr.is_finite() || signal_snr <= 0.0 {
                continue;
            }
            observations.push(json!({
                "frame_id":frame_index,
                "source_id":format!("track:{track_id}"),
                "instrumental_magnitude":magnitude,
                "magnitude_error":1.0857362047581296 / signal_snr,
                "x":source.x,
                "y":source.y,
                "quality_passed":true,
                "is_moving":false,
            }));
        }
    }
    crate::relative_photometry::fit(
        &json!(observations),
        &json!({
            "spatial_order":request.get("relative_photometry_spatial_order").cloned().unwrap_or(json!(0)),
            "validation_fraction":request.get("relative_photometry_validation_fraction").cloned().unwrap_or(json!(0.2)),
            "random_state":0,
            "min_sources":2,
            "min_frames":2,
        }),
    )
}

/// Handle `sequence.analyze_core` JSONL requests.
pub fn handle_request(request: &Value) -> Result<Value> {
    let paths: Vec<PathBuf> = serde_json::from_value(
        request
            .get("paths")
            .cloned()
            .ok_or("missing sequence paths")?,
    )
    .map_err(|error| error.to_string())?;
    if paths.is_empty() {
        return Err("sequence paths cannot be empty".into());
    }
    let fast_point_motion = match request.get("fast_point_motion") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(enabled)) => *enabled,
        Some(_) => return Err("fast_point_motion must be a boolean".into()),
    };
    let relative_photometry = match request.get("relative_photometry") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(enabled)) => *enabled,
        Some(_) => return Err("relative_photometry must be a boolean".into()),
    };
    let candidate_limit = match request.get("fast_point_candidate_limit") {
        None | Some(Value::Null) => DEFAULT_FAST_POINT_CANDIDATE_WORKING_LIMIT,
        Some(value) => serde_json::from_value::<usize>(value.clone())
            .map_err(|_| "fast_point_candidate_limit must be a positive integer")?,
    };
    if candidate_limit == 0 {
        return Err("fast_point_candidate_limit must be positive".into());
    }
    let fast_point_options_input =
        object_or_empty(request.get("fast_point_options"), "fast_point_options")?;
    let candidate_min_filter_snr = fast_point_options_input
        .get("min_flux_snr")
        .and_then(Value::as_f64)
        .unwrap_or_else(|| FastPointOptions::default().min_flux_snr);
    let frame_options = sequence_frame_options(request)?;
    let mut stack_options = StackFaintOptions::default();
    stack_options.enabled = optional_request_bool(request, "stack_faint_recovery", true)?;
    stack_options.reference_mode = request
        .get("stack_reference_mode")
        .and_then(Value::as_str)
        .unwrap_or("median")
        .trim()
        .to_ascii_lowercase();
    if !matches!(stack_options.reference_mode.as_str(), "median" | "coadd") {
        return Err("stack_reference_mode must be 'median' or 'coadd'".into());
    }
    stack_options.threshold_sigma = optional_request_f64(request, "stack_threshold_sigma")?
        .unwrap_or(stack_options.threshold_sigma);
    stack_options.min_flux_snr =
        optional_request_f64(request, "stack_min_flux_snr")?.unwrap_or(stack_options.min_flux_snr);
    stack_options.frame_min_flux_snr = optional_request_f64(request, "stack_frame_min_flux_snr")?
        .unwrap_or(stack_options.frame_min_flux_snr);
    stack_options.min_presence = optional_request_usize(request, "stack_min_presence", 0)?;
    stack_options.max_reference_sources = optional_request_limit(
        request,
        "stack_max_reference_sources",
        DEFAULT_SEQUENCE_STACK_WORKING_LIMIT,
    )?;
    stack_options.min_distance = frame_options.detection.min_distance;
    stack_options.aperture_radius = frame_options.detection.measurement.aperture_radius;
    stack_options.psf_fwhm = frame_options.detection.measurement.psf_fwhm;
    stack_options.min_psf_support_pixels =
        frame_options.detection.measurement.min_psf_support_pixels;
    stack_options.proposal_mode = frame_options.detection.proposal_mode.clone();
    stack_options.mask_zero_pixels = frame_options.detection.mask_zero_pixels.unwrap_or(false);
    let temporal_proposal_mode = request
        .get("temporal_proposal_mode")
        .and_then(Value::as_str)
        .unwrap_or("median")
        .trim()
        .to_ascii_lowercase();
    if !matches!(temporal_proposal_mode.as_str(), "median" | "coadd" | "both") {
        return Err("temporal_proposal_mode must be 'median', 'coadd', or 'both'".into());
    }
    let candidate_consensus_min_snr =
        optional_request_f64(request, "candidate_consensus_min_snr")?.unwrap_or(15.0);
    let temporal_reference_min_snr = optional_request_f64(request, "temporal_reference_min_snr")?
        .unwrap_or(candidate_consensus_min_snr);
    let temporal_candidate_min_snr = optional_request_f64(request, "temporal_candidate_min_snr")?
        .unwrap_or((0.5 * candidate_consensus_min_snr).max(5.0));
    if candidate_consensus_min_snr <= 0.0 {
        return Err("candidate_consensus_min_snr must be positive".into());
    }
    if temporal_candidate_min_snr <= 0.0 || temporal_candidate_min_snr > candidate_consensus_min_snr
    {
        return Err(
            "temporal_candidate_min_snr must be positive and no greater than candidate_consensus_min_snr"
                .into(),
        );
    }
    if temporal_reference_min_snr <= 0.0 || temporal_reference_min_snr > candidate_consensus_min_snr
    {
        return Err(
            "temporal_reference_min_snr must be positive and no greater than candidate_consensus_min_snr"
                .into(),
        );
    }
    let temporal_multiscale = match request.get("temporal_multiscale") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(enabled)) => *enabled,
        Some(_) => return Err("temporal_multiscale must be a boolean".into()),
    };
    let temporal_min_psf_correlation =
        optional_request_f64(request, "temporal_min_psf_correlation")?.unwrap_or(0.8);
    if !(0.0 < temporal_min_psf_correlation && temporal_min_psf_correlation <= 1.0) {
        return Err("temporal_min_psf_correlation must be between 0 and 1".into());
    }
    let mut frames = Vec::with_capacity(paths.len());
    let mut frame_analyses = Vec::with_capacity(paths.len());
    let mut frame_analysis_objects = Vec::with_capacity(paths.len());
    let mut quality_frames = Vec::with_capacity(paths.len());
    let mut prepared_candidates_by_frame = Vec::with_capacity(paths.len());
    let mut candidate_frame_audits = Vec::with_capacity(paths.len());

    let first_frame = fits::read_fits(&paths[0], false)?;
    let shared_background = if paths.len() >= 2
        && frame_options.detection.fast_sequence
        && frame_options.detection.use_float32
    {
        let mask = fits::auxiliary_mask(first_frame.height, first_frame.width)?;
        let (background, noise) = detection::build_background_model(
            &first_frame.data,
            Some(&mask),
            first_frame.height,
            first_frame.width,
            &frame_options.detection,
        )?;
        Some((background, noise, first_frame.height, first_frame.width))
    } else {
        None
    };
    frames.push(first_frame);
    for path in paths.iter().skip(1) {
        frames.push(fits::read_fits(path, false)?);
    }

    let headers = frames
        .iter()
        .map(|frame| Value::Object(frame.header.clone()))
        .collect::<Vec<_>>();
    let fixed_sentinel_audit = crate::sequence_sentinel::audit_fixed_sentinel(&frames, -1, 10_000)?;
    let mut fixed_sentinel_impact_audit =
        crate::sequence_sentinel::FixedSentinelImpactAccumulator::new(
            paths.len(),
            &fixed_sentinel_audit,
            5_000,
        )?;
    let time_axis = sequence_trails::observation_time_axis(&json!({"headers":headers}))?;
    let frame_times: Vec<f64> = serde_json::from_value(time_axis["frame_times"].clone())
        .map_err(|error| error.to_string())?;

    for (frame_index, frame) in frames.into_iter().enumerate() {
        let background = shared_background
            .as_ref()
            .map(|(background, noise, height, width)| BackgroundModel {
                background,
                noise,
                height: *height,
                width: *width,
            });
        let analysis =
            pipeline::analyze_frame(frame, None, None, &frame_options, background.as_ref())?;
        fixed_sentinel_impact_audit.add_frame(frame_index, &analysis)?;
        if fast_point_motion && paths.len() >= 3 {
            let (prepared, audit) = prefilter_fast_candidate_peaks(
                &analysis.detection.candidate_peaks,
                &analysis.detection.line_artifact_coordinates,
                analysis.detection.image_shape,
                candidate_min_filter_snr,
                candidate_limit,
                frame_options.detection.measurement.aperture_radius,
            )?;
            prepared_candidates_by_frame.push(prepared);
            candidate_frame_audits.push(audit);
        }
        let detections = analysis
            .detection
            .sources
            .iter()
            .filter(|source| source.quality_passed)
            .cloned()
            .collect::<Vec<_>>();
        quality_frames.push(detections);
        frame_analyses.push(analysis.as_dict());
        frame_analysis_objects.push(analysis);
    }

    let mut tracking_options =
        object_or_empty(request.get("tracking_options"), "tracking_options")?;
    if tracking_options
        .get("frame_names")
        .is_none_or(Value::is_null)
    {
        tracking_options.insert(
            "frame_names".into(),
            json!(paths
                .iter()
                .map(|path| path.to_string_lossy())
                .collect::<Vec<_>>()),
        );
    }
    if tracking_options
        .get("frame_times")
        .is_none_or(Value::is_null)
    {
        tracking_options.insert("frame_times".into(), json!(frame_times));
    }
    if tracking_options
        .get("persistent_min_presence")
        .is_none_or(Value::is_null)
    {
        if let Some(value) = request
            .get("persistent_min_presence")
            .filter(|v| !v.is_null())
        {
            tracking_options.insert("persistent_min_presence".into(), value.clone());
        }
    }
    let tracking_options: TrackSequenceOptions =
        serde_json::from_value(Value::Object(tracking_options))
            .map_err(|error| error.to_string())?;
    let mut tracking = sequence::track_detections(&quality_frames, &tracking_options)?;
    let mut fast_point_options = fast_point_options_input;
    if fast_point_options
        .get("link_radius_px")
        .is_none_or(Value::is_null)
    {
        fast_point_options.insert(
            "link_radius_px".into(),
            json!(tracking_options.link_radius_px),
        );
    }
    if fast_point_options
        .get("min_presence")
        .is_none_or(Value::is_null)
    {
        fast_point_options.insert("min_presence".into(), tracking["min_presence"].clone());
    }
    if fast_point_options
        .get("motion_min_displacement_px")
        .is_none_or(Value::is_null)
    {
        fast_point_options.insert(
            "motion_min_displacement_px".into(),
            tracking["effective_motion_min_displacement_px"].clone(),
        );
    }
    let fast_point_options: FastPointOptions =
        serde_json::from_value(Value::Object(fast_point_options))
            .map_err(|error| error.to_string())?;
    let mut fast_point_tracks = if fast_point_motion && quality_frames.len() >= 2 {
        let shifts: Vec<[f64; 2]> = serde_json::from_value(tracking["cumulative_shifts"].clone())
            .map_err(|error| error.to_string())?;
        let existing_tracks: Vec<Value> = serde_json::from_value(tracking["tracks"].clone())
            .map_err(|error| error.to_string())?;
        let tracks = sequence_fast::track_fast_point_movers(
            &quality_frames,
            &shifts,
            Some(&frame_times),
            &fast_point_options,
            &existing_tracks,
        )?;
        serde_json::from_value::<Vec<Value>>(tracks["tracks"].clone())
            .map_err(|error| error.to_string())?
    } else {
        Vec::new()
    };
    let base_track_count = tracking["tracks"].as_array().map_or(0, Vec::len);
    for (index, track) in fast_point_tracks.iter_mut().enumerate() {
        track["track_id"] = json!(base_track_count + index);
    }

    let mut fast_candidate_proposals = if !fast_point_motion {
        json!({
            "status":"disabled",
            "scope":"matched_filter_coordinate_association_and_original_pixel_quality_gates",
            "evidence_level":"unverified_candidate_proposal",
            "frame_audits":[],
            "proposal_count":0,
            "proposals":[],
        })
    } else if paths.len() < 3 {
        json!({
            "status":"insufficient_frames",
            "scope":"matched_filter_coordinate_association_and_original_pixel_quality_gates",
            "evidence_level":"unverified_candidate_proposal",
            "frame_audits":[],
            "proposal_count":0,
            "proposals":[],
        })
    } else {
        let shifts: Vec<[f64; 2]> = serde_json::from_value(tracking["cumulative_shifts"].clone())
            .map_err(|error| error.to_string())?;
        fast_candidate_proposals(
            &prepared_candidates_by_frame,
            candidate_frame_audits,
            &shifts,
            &frame_times,
            &fast_point_options,
            &tracking,
            &fast_point_tracks,
        )?
    };
    if fast_point_motion && paths.len() >= 3 {
        let tracking_min_presence = tracking["min_presence"].as_u64().unwrap_or(3) as usize;
        let mut measured_candidate_tracks = verify_fast_candidate_proposals(
            &mut fast_candidate_proposals,
            &frame_analysis_objects,
            &frame_times,
            tracking_min_presence,
            base_track_count + fast_point_tracks.len(),
            &frame_options,
            &fast_point_options,
        )?;
        fast_point_tracks.append(&mut measured_candidate_tracks);
    }
    let shifts: Vec<[f64; 2]> = serde_json::from_value(tracking["cumulative_shifts"].clone())
        .map_err(|error| error.to_string())?;
    let mut temporal_candidate_frames = vec![Vec::<[f32; 5]>::new(); paths.len()];
    let mut temporal_reference_candidate_count = 0usize;
    let mut temporal_coadd_candidate_count = 0usize;
    let mut motion_frame_components = json!({
        "status":"insufficient_frames",
        "scope":"registered_temporal_residual_components_only",
        "frames":[]
    });
    if paths.len() >= 2 {
        let (median_result, median_reference) =
            crate::sequence_temporal::temporal_reference_candidates_and_reference_from_analyses(
                &frame_analysis_objects,
                &shifts,
                false,
                3.0,
                frame_options.detection.measurement.psf_fwhm,
                temporal_reference_min_snr,
                temporal_multiscale,
            )?;
        temporal_reference_candidate_count =
            median_result["candidate_count"].as_u64().unwrap_or(0) as usize;
        temporal_candidate_frames =
            serde_json::from_value(median_result["candidate_peak_frames"].clone())
                .map_err(|error| error.to_string())?;
        motion_frame_components =
            crate::sequence_motion_features::motion_frame_components_from_analyses(
                &frame_analysis_objects,
                median_reference.as_ref(),
                &shifts,
                frame_options.detection.measurement.psf_fwhm,
                15.0,
                100.0,
                40,
                4.0,
                None,
                true,
            )?;
        if matches!(temporal_proposal_mode.as_str(), "coadd" | "both") {
            let coadd_result =
                crate::sequence_temporal::temporal_reference_candidates_from_analyses(
                    &frame_analysis_objects,
                    &shifts,
                    true,
                    3.0,
                    frame_options.detection.measurement.psf_fwhm,
                    temporal_reference_min_snr,
                    temporal_multiscale,
                )?;
            temporal_coadd_candidate_count =
                coadd_result["candidate_count"].as_u64().unwrap_or(0) as usize;
            let coadd_frames: Vec<Vec<[f32; 5]>> =
                serde_json::from_value(coadd_result["candidate_peak_frames"].clone())
                    .map_err(|error| error.to_string())?;
            if temporal_proposal_mode == "coadd" {
                temporal_candidate_frames = coadd_frames;
            } else {
                for (median, coadd) in temporal_candidate_frames.iter_mut().zip(coadd_frames) {
                    median.extend(coadd);
                }
            }
        }
    }
    let candidate_peak_frames = frame_analysis_objects
        .iter()
        .zip(&temporal_candidate_frames)
        .map(|(analysis, temporal_rows)| {
            let raw_rows = analysis
                .detection
                .candidate_peaks
                .iter()
                .map(|[x, y, snr]| json!([x, y, snr, 0.0, null]))
                .collect::<Vec<_>>();
            let temporal_rows = temporal_rows
                .iter()
                .map(|row| json!(row))
                .collect::<Vec<_>>();
            let rows: Vec<Value> = raw_rows.into_iter().chain(temporal_rows).collect();
            json!(rows)
        })
        .collect::<Vec<_>>();
    let mut candidate_consensus = if paths.len() >= 3 {
        crate::sequence_consensus::candidate_consensus_from_analyses(
            &candidate_peak_frames,
            &shifts,
            tracking["tracks"]
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            Some(&frame_times),
            tracking["link_radius_px"].as_f64().unwrap_or(4.0),
            tracking["min_presence"].as_u64().unwrap_or(3) as usize,
            tracking["persistent_min_presence"].as_u64().unwrap_or(3) as usize,
            tracking["effective_motion_min_displacement_px"]
                .as_f64()
                .unwrap_or_else(|| {
                    tracking["motion_min_displacement_px"]
                        .as_f64()
                        .unwrap_or(2.0)
                }),
            tracking["max_motion_fit_rms_px"].as_f64().unwrap_or(0.75),
            candidate_consensus_min_snr,
            Some(temporal_candidate_min_snr),
            Some(temporal_reference_min_snr),
            temporal_min_psf_correlation,
            &frame_analysis_objects,
        )?
    } else {
        json!({"tracks":[],"audit":{}})
    };
    let candidate_consensus_count = {
        let tracks = candidate_consensus["tracks"]
            .as_array_mut()
            .ok_or("candidate consensus tracks must be an array")?;
        for (index, track) in tracks.iter_mut().enumerate() {
            track["track_id"] = json!(base_track_count + fast_point_tracks.len() + index);
        }
        tracks.len()
    };
    tracking["candidate_consensus_count"] = json!(candidate_consensus_count);
    tracking["candidate_consensus_audit"] = candidate_consensus["audit"].clone();
    tracking["fast_point_motion_count"] = json!(fast_point_tracks.len());
    tracking["temporal_reference_candidate_count"] = json!(temporal_reference_candidate_count);
    tracking["temporal_coadd_candidate_count"] = json!(temporal_coadd_candidate_count);
    if stack_options.min_presence == 0 {
        stack_options.min_presence = tracking["persistent_min_presence"]
            .as_u64()
            .map(|value| value as usize)
            .unwrap_or_else(|| 3usize.max(paths.len().div_ceil(2)));
    }
    stack_options.link_radius_px = tracking["link_radius_px"].as_f64().unwrap_or(4.0);
    let mut existing_tracks = tracking["tracks"].as_array().cloned().unwrap_or_default();
    existing_tracks.extend(fast_point_tracks.iter().cloned());
    existing_tracks.extend(
        candidate_consensus["tracks"]
            .as_array()
            .cloned()
            .unwrap_or_default(),
    );
    let mut stack_faint = sequence_stack::recover_stack_faint_tracks(
        &frame_analysis_objects,
        &shifts,
        &frame_times,
        &existing_tracks,
        &stack_options,
    )?;
    for (index, track) in stack_faint.tracks.iter_mut().enumerate() {
        track["track_id"] =
            json!(base_track_count + fast_point_tracks.len() + candidate_consensus_count + index);
    }
    let mut all_tracks = tracking["tracks"].as_array().cloned().unwrap_or_default();
    all_tracks.extend(fast_point_tracks.iter().cloned());
    all_tracks.extend(
        candidate_consensus["tracks"]
            .as_array()
            .cloned()
            .unwrap_or_default(),
    );
    all_tracks.extend(stack_faint.tracks.iter().cloned());
    let track_counts = sequence_track_counts(&all_tracks);
    for key in [
        "candidate_consensus_count",
        "temporal_reference_count",
        "stack_faint_count",
    ] {
        tracking[key] = track_counts[key].clone();
    }
    tracking["stack_faint_candidate_count"] = json!(stack_faint.candidate_count);
    tracking["stack_reference_mode"] = json!(stack_options.reference_mode);
    tracking["stack_threshold_sigma"] = json!(stack_options.threshold_sigma);
    tracking["stack_min_flux_snr"] = json!(stack_options.min_flux_snr);
    tracking["stack_frame_min_flux_snr"] = json!(stack_options.frame_min_flux_snr);
    tracking["stack_min_presence"] = json!(stack_options.min_presence);
    tracking["stack_faint_audit"] = stack_faint.audit.clone();
    let relative_photometry_result = if relative_photometry {
        Some(fit_sequence_relative_photometry(
            &quality_frames,
            &headers,
            &tracking,
            tracking["min_presence"].as_u64().unwrap_or(3) as usize,
            request,
        )?)
    } else {
        None
    };
    let fixed_sentinel_impact_audit = fixed_sentinel_impact_audit.finish();

    Ok(json!({
        "schema":"rst19.sequence.core.v1",
        "scope":"frame_detection_quality_tracks_temporal_reference_candidates_fast_point_motion_candidate_proposals_and_candidate_consensus",
        "frame_analyses":frame_analyses,
        "tracking":tracking,
        "fast_point_tracks":fast_point_tracks,
        "fast_candidate_proposals":fast_candidate_proposals,
        "candidate_consensus":candidate_consensus,
        "tracks":all_tracks,
        "track_counts":track_counts,
        "stack_faint_tracks":stack_faint.tracks,
        "stack_faint_candidate_count":stack_faint.candidate_count,
        "stack_faint_audit":stack_faint.audit,
        "temporal_reference_candidate_count":temporal_reference_candidate_count,
        "temporal_coadd_candidate_count":temporal_coadd_candidate_count,
        "temporal_proposal_mode":temporal_proposal_mode,
        "relative_photometry":relative_photometry_result,
        "motion_frame_components":motion_frame_components,
        "fixed_sentinel_audit":fixed_sentinel_audit,
        "fixed_sentinel_impact_audit":fixed_sentinel_impact_audit,
        "observation_time_axis":time_axis,
        "detector_options":serde_json::to_value(&frame_options.detection).map_err(|error| error.to_string())?,
        "shared_background_model":shared_background.is_some(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembled_track_counts_keep_candidate_layers_separate_from_stack_faint_sources() {
        let tracks = json!([
            {"classification":"static","evidence_level":"quality"},
            {"classification":"persistent","evidence_level":"quality"},
            {"classification":"persistent","evidence_level":"stack_faint"},
            {"classification":"transient","evidence_level":"candidate_consensus"},
            {"classification":"persistent","evidence_level":"temporal_reference"},
            {"classification":"moving","evidence_level":"motion_feature"}
        ]);
        let counts = sequence_track_counts(tracks.as_array().unwrap());
        assert_eq!(counts["stable_source_count"], 1);
        assert_eq!(counts["persistent_source_count"], 2);
        assert_eq!(counts["stable_field_candidate_count"], 3);
        assert_eq!(counts["candidate_consensus_count"], 2);
        assert_eq!(counts["temporal_reference_count"], 1);
        assert_eq!(counts["stack_faint_count"], 1);
        assert_eq!(counts["moving_track_count"], 1);
        assert_eq!(counts["transient_track_count"], 1);
    }

    #[test]
    fn sequence_detector_defaults_match_the_python_fast_sequence_contract() {
        let options = sequence_frame_options(&json!({})).unwrap();
        assert_eq!(options.detection.max_sources, Some(6_000));
        assert_eq!(options.detection.background_box_size, 256);
        assert_eq!(options.detection.background_sample_limit, 4_096);
        assert!(options.detection.use_float32);
        assert!(options.detection.fast_sequence);
        assert!(!options.detection.measurement.refine_local_background);
    }

    #[test]
    fn sequence_detector_can_disable_the_working_limit_and_fast_float32_path() {
        let options = sequence_frame_options(&json!({
            "sequence_max_sources": null,
            "frame_options": {"use_float32": false}
        }))
        .unwrap();
        assert_eq!(options.detection.max_sources, None);
        assert_eq!(options.detection.background_sample_limit, 100_000);
        assert!(!options.detection.use_float32);
        assert!(!options.detection.fast_sequence);
    }

    #[test]
    fn explicit_detector_source_limit_takes_precedence() {
        let options = sequence_frame_options(&json!({
            "sequence_max_sources": 8,
            "frame_options": {"max_sources": 12}
        }))
        .unwrap();
        assert_eq!(options.detection.max_sources, Some(12));
    }

    #[test]
    fn fast_candidate_prefilter_applies_snr_limit_edges_and_line_artifacts() {
        let peaks = [
            [15.0, 15.0, 8.0],
            [2.0, 20.0, 20.0],
            [50.0, 50.0, 15.0],
            [20.0, 20.0, 7.0],
            [30.0, 30.0, 12.0],
            [f64::NAN, 5.0, 50.0],
        ];
        let (candidates, audit) =
            prefilter_fast_candidate_peaks(&peaks, &[[54, 50]], [100, 100], 8.0, 3, 4).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].row_index, 4);
        assert_eq!(audit["finite_peak_count"], 5);
        assert_eq!(audit["min_filter_snr_pass_count"], 4);
        assert_eq!(audit["working_set_truncated_count"], 1);
        assert_eq!(audit["edge_rejected_count"], 1);
        assert_eq!(audit["line_artifact_rejected_count"], 1);
        assert_eq!(audit["association_input_count"], 1);
    }

    #[test]
    fn candidate_measurement_mask_uses_float32_values_like_python_sequence_gates() {
        let pixels = crate::models::PixelData::F64(vec![f64::from_bits(1), 1.0]);
        let (mask, _, _) =
            detection::working_mask_float32(&pixels, None, 0.0, Some(true), None).unwrap();
        assert_eq!(mask, vec![true, false]);
    }

    #[test]
    fn fast_candidate_prefilter_rejects_peaks_already_explained_by_quality_tracks() {
        let prepared = (0..3)
            .map(|frame| {
                vec![PreparedFastCandidate {
                    row_index: frame,
                    raw_x: 31.0 + frame as f64 * 10.0,
                    raw_y: 40.0,
                    filter_snr: 12.0,
                }]
            })
            .collect::<Vec<_>>();
        let audit = (0..3).map(|_| json!({})).collect();
        let tracking = json!({
            "min_presence":3,
            "tracks":[{
                "presence":3,
                "points":[
                    {"frame_index":0,"aligned_x":30.0,"aligned_y":40.0},
                    {"frame_index":1,"aligned_x":30.0,"aligned_y":40.0},
                    {"frame_index":2,"aligned_x":30.0,"aligned_y":40.0}
                ]
            }]
        });
        let result = fast_candidate_proposals(
            &prepared,
            audit,
            &[[1.0, 0.0], [11.0, 0.0], [21.0, 0.0]],
            &[0.0, 1.0, 2.0],
            &FastPointOptions::default(),
            &tracking,
            &[],
        )
        .unwrap();
        assert_eq!(result["proposal_count"], 0);
        assert_eq!(result["frame_audits"][0]["explained_rejected_count"], 1);
        assert_eq!(result["frame_audits"][2]["association_input_count"], 0);
    }

    #[test]
    fn empty_sequence_is_rejected_before_any_files_are_opened() {
        assert_eq!(
            handle_request(&json!({"paths": []})).unwrap_err(),
            "sequence paths cannot be empty"
        );
    }
}
