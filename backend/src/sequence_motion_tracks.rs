//! Candidate-only association of temporal residual trail components.
//!
//! This is one stage of Python's `detect_motion_features` pipeline. It links
//! registered per-frame components with the same bounded-gap prediction,
//! exposure-speed gate, sparse one-to-one assignment, and trail-geometry cost.
//! The result deliberately stays at the unverified candidate level: raw-frame
//! line-model review and the final motion classification are separate stages.
use crate::{
    sequence_assignment::{global_frame_assignments, TrackPosition},
    sequence_trails, Result,
};
use serde_json::{json, Value};
use std::collections::HashMap;

const MAX_FRAME_GAP: usize = 3;
const EXPOSURE_LINK_RATIO_MIN: f64 = 0.25;
const EXPOSURE_LINK_RATIO_MAX: f64 = 2.50;

fn number(point: &Value, key: &str) -> Result<f64> {
    point
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("motion component point missing numeric {key}"))
}

fn point_frame_index(point: &Value) -> Result<usize> {
    point
        .get("frame_index")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| "motion component point missing non-negative frame_index".into())
}

fn deconvolved_motion_length(measured: f64, psf_fwhm: f64) -> Option<f64> {
    if !measured.is_finite() || !psf_fwhm.is_finite() || measured <= 0.0 || psf_fwhm <= 0.0 {
        return None;
    }
    let psf_sigma = psf_fwhm / 2.354_820_045_030_949_3;
    let motion_variance = 0.75 * (measured * measured - 16.0 * psf_sigma * psf_sigma);
    (motion_variance.is_finite() && motion_variance > 0.0).then(|| motion_variance.sqrt())
}

fn point_xy(point: &Value) -> Result<[f64; 2]> {
    let x = number(point, "aligned_x")?;
    let y = number(point, "aligned_y")?;
    if !x.is_finite() || !y.is_finite() {
        return Err("motion component aligned coordinates must be finite".into());
    }
    Ok([x, y])
}

fn finite_number(point: &Value, key: &str) -> Option<f64> {
    point
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
}

fn model_is_supported(point: &Value) -> bool {
    let model = point.get("single_frame_model").and_then(Value::as_str);
    let line_snr = finite_number(point, "line_template_snr");
    match model {
        Some("long_line_profile") => {
            line_snr.is_some_and(|value| value >= 5.0)
                && finite_number(point, "line_vs_background_delta_bic")
                    .is_some_and(|value| value >= 6.0)
        }
        Some("line") => {
            line_snr.is_some_and(|value| value >= 5.0)
                && finite_number(point, "line_vs_point_delta_bic").is_some_and(|value| value >= 6.0)
                && finite_number(point, "line_vs_double_delta_bic")
                    .is_some_and(|value| value >= 6.0)
                && finite_number(point, "line_vs_elliptical_delta_bic")
                    .is_some_and(|value| value >= 6.0)
        }
        _ => false,
    }
}

/// Port of Python's `_single_frame_sequence_line_model_is_review_supported`.
pub fn sequence_line_model_review_supported(point: &Value, psf_fwhm: f64) -> Result<bool> {
    if !psf_fwhm.is_finite() || psf_fwhm <= 0.0 {
        return Err("psf_fwhm must be finite and positive".into());
    }
    if model_is_supported(point) {
        return Ok(true);
    }
    let model = point.get("single_frame_model").and_then(Value::as_str);
    if !matches!(
        model,
        Some("line" | "long_line_profile" | "elliptical_point")
    ) {
        return Ok(false);
    }
    let (Some(line_snr), Some(background_delta), Some(continuity), Some(fill), Some(length)) = (
        finite_number(point, "line_template_snr"),
        finite_number(point, "line_vs_background_delta_bic"),
        finite_number(point, "continuity_ratio"),
        finite_number(point, "profile_fill_ratio"),
        finite_number(point, "length_px"),
    ) else {
        return Ok(false);
    };
    if length < 6.0_f64.max(2.5 * psf_fwhm)
        || line_snr < 5.0
        || background_delta < 2.0
        || continuity < 0.45
        || fill < 0.45
    {
        return Ok(false);
    }
    if matches!(model, Some("line" | "elliptical_point")) {
        if [
            finite_number(point, "line_vs_point_delta_bic"),
            finite_number(point, "line_vs_double_delta_bic"),
            finite_number(point, "line_vs_elliptical_delta_bic"),
        ]
        .into_iter()
        .any(|value| value.is_none_or(|delta| delta < 2.0))
        {
            return Ok(false);
        }
    }
    if model == Some("elliptical_point")
        && finite_number(point, "width_px").is_none_or(|width| width > 0.75_f64.max(0.5 * psf_fwhm))
    {
        return Ok(false);
    }
    Ok(true)
}

fn line_geometries_overlap(first: &Value, second: &Value) -> bool {
    let (
        Some(mut first_x),
        Some(mut first_y),
        Some(mut first_length),
        Some(mut first_width),
        Some(mut first_angle),
        Some(mut second_x),
        Some(mut second_y),
        Some(mut second_length),
        Some(mut second_width),
        Some(mut second_angle),
    ) = (
        finite_number(first, "x"),
        finite_number(first, "y"),
        finite_number(first, "length_px"),
        finite_number(first, "width_px"),
        finite_number(first, "angle_deg"),
        finite_number(second, "x"),
        finite_number(second, "y"),
        finite_number(second, "length_px"),
        finite_number(second, "width_px"),
        finite_number(second, "angle_deg"),
    )
    else {
        return false;
    };
    if first_length <= 0.0 || second_length <= 0.0 {
        return false;
    }
    if second_length > first_length {
        std::mem::swap(&mut first_x, &mut second_x);
        std::mem::swap(&mut first_y, &mut second_y);
        std::mem::swap(&mut first_length, &mut second_length);
        std::mem::swap(&mut first_width, &mut second_width);
        std::mem::swap(&mut first_angle, &mut second_angle);
    }
    let angle_difference = ((first_angle - second_angle + 90.0).rem_euclid(180.0) - 90.0).abs();
    if angle_difference > 15.0 {
        return false;
    }
    let angle = first_angle.to_radians();
    let direction_x = angle.cos();
    let direction_y = angle.sin();
    let delta_x = second_x - first_x;
    let delta_y = second_y - first_y;
    let perpendicular_distance = (delta_x * direction_y - delta_y * direction_x).abs();
    let perpendicular_gate = 2.5_f64.max(0.5 * (first_width + second_width) + 1.5);
    if perpendicular_distance > perpendicular_gate {
        return false;
    }
    let parallel_distance = (delta_x * direction_x + delta_y * direction_y).abs();
    let overlap = 0.5 * (first_length + second_length) - parallel_distance;
    overlap / second_length >= 0.35
}

/// Attach raw-pixel line-model evidence to temporal residual proposals, then
/// reject temporal-only structures that lack a matching review-supported line.
pub fn review_temporal_components_against_raw_frames(
    temporal_components: &Value,
    raw_components_by_frame: &[Value],
    shifts: &[[f64; 2]],
    psf_fwhm: f64,
) -> Result<Value> {
    let frames = temporal_components
        .get("frames")
        .and_then(Value::as_array)
        .ok_or_else(|| "temporal components must contain frames".to_string())?;
    if frames.is_empty() || temporal_components["status"] == "insufficient_frames" {
        return Ok(json!({
            "status":"insufficient_frames",
            "scope":"temporal_components_reviewed_against_raw_pixel_line_models",
            "frames":[],
            "review_audit":[],
        }));
    }
    if frames.len() != raw_components_by_frame.len() || frames.len() != shifts.len() {
        return Err("temporal, raw, and shift frame counts must match".into());
    }
    if !psf_fwhm.is_finite() || psf_fwhm <= 0.0 {
        return Err("psf_fwhm must be finite and positive".into());
    }

    let model_fields = [
        "line_template_snr",
        "line_vs_point_delta_bic",
        "line_vs_double_delta_bic",
        "line_vs_background_delta_bic",
        "line_vs_elliptical_delta_bic",
        "single_frame_model",
        "single_frame_model_margin_bic",
        "continuity_ratio",
        "profile_fill_ratio",
        "line_artifact_overlap_fraction",
    ];
    let mut reviewed_frames = Vec::with_capacity(frames.len());
    let mut review_audit = Vec::with_capacity(frames.len());
    for (frame_index, (temporal_frame, raw_frame)) in
        frames.iter().zip(raw_components_by_frame).enumerate()
    {
        let temporal_points = temporal_frame
            .get("points")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let raw_points = raw_frame
            .get("points")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut reviewed_points = Vec::new();
        let mut geometry_match_count = 0usize;
        let mut model_rejected_count = 0usize;
        for mut temporal_point in temporal_points.iter().cloned() {
            let temporal_geometry = [
                finite_number(&temporal_point, "x").unwrap_or(f64::NAN),
                finite_number(&temporal_point, "y").unwrap_or(f64::NAN),
                finite_number(&temporal_point, "length_px").unwrap_or(f64::NAN),
                finite_number(&temporal_point, "width_px").unwrap_or(f64::NAN),
                finite_number(&temporal_point, "angle_deg").unwrap_or(f64::NAN),
            ];
            let mut matches = raw_points
                .iter()
                .filter(|raw_point| line_geometries_overlap(&temporal_point, raw_point))
                .map(|raw_point| {
                    let center_distance = (temporal_geometry[0]
                        - finite_number(raw_point, "x").unwrap_or(f64::INFINITY))
                    .hypot(
                        temporal_geometry[1]
                            - finite_number(raw_point, "y").unwrap_or(f64::INFINITY),
                    );
                    let length_difference = (temporal_geometry[2]
                        - finite_number(raw_point, "length_px").unwrap_or(f64::INFINITY))
                    .abs();
                    (center_distance, length_difference, raw_point)
                })
                .collect::<Vec<_>>();
            matches
                .sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.total_cmp(&right.1)));
            let Some((_, _, raw_point)) = matches.first() else {
                continue;
            };
            geometry_match_count += 1;
            if let Some(temporal_object) = temporal_point.as_object_mut() {
                for key in model_fields {
                    temporal_object.insert(
                        key.to_owned(),
                        raw_point.get(key).cloned().unwrap_or(Value::Null),
                    );
                }
            }
            if sequence_line_model_review_supported(&temporal_point, psf_fwhm)? {
                reviewed_points.push(temporal_point);
            } else {
                model_rejected_count += 1;
            }
        }
        let mut audit = temporal_frame
            .get("audit")
            .cloned()
            .unwrap_or_else(|| json!({}));
        if let Some(audit_object) = audit.as_object_mut() {
            audit_object.insert("feature_count".into(), json!(reviewed_points.len()));
            audit_object.insert(
                "max_feature_residual_snr".into(),
                reviewed_points
                    .iter()
                    .filter_map(|point| finite_number(point, "residual_snr"))
                    .max_by(f64::total_cmp)
                    .map_or(Value::Null, |value| json!(value)),
            );
        }
        review_audit.push(json!({
            "frame_index":frame_index,
            "temporal_component_count":temporal_points.len(),
            "raw_component_count":raw_points.len(),
            "raw_geometry_match_count":geometry_match_count,
            "model_rejected_count":model_rejected_count,
            "reviewed_component_count":reviewed_points.len(),
        }));
        reviewed_frames.push(json!({"points":reviewed_points,"audit":audit}));
    }
    Ok(json!({
        "status":"analyzed",
        "scope":"temporal_components_reviewed_against_raw_pixel_line_models",
        "min_length_px":temporal_components.get("min_length_px").cloned().unwrap_or(Value::Null),
        "thresholds_adu":temporal_components.get("thresholds_adu").cloned().unwrap_or(Value::Null),
        "metadata_supported":temporal_components.get("metadata_supported").cloned().unwrap_or(Value::Bool(false)),
        "frames":reviewed_frames,
        "review_audit":review_audit,
    }))
}

/// Add review-supported raw-frame proposals that are anchored to temporal
/// candidates or form the same coherent multi-frame chain used by Python.
pub fn supplement_with_coherent_raw_frame_candidates(
    reviewed_temporal_components: &Value,
    raw_components_by_frame: &[Value],
    frame_times: &[f64],
    exposure_seconds: &[Option<f64>],
    psf_fwhm: f64,
) -> Result<Value> {
    let mut result = reviewed_temporal_components.clone();
    let frames = result
        .get("frames")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| "reviewed temporal components must contain frames".to_string())?;
    if result["status"] == "insufficient_frames" || frames.len() < 2 {
        if let Some(object) = result.as_object_mut() {
            object.insert("raw_fallback_audit".into(), json!([]));
            object.insert("unassociated_single_frame_points".into(), json!([]));
        }
        return Ok(result);
    }
    if frames.len() != raw_components_by_frame.len()
        || frames.len() != frame_times.len()
        || frames.len() != exposure_seconds.len()
    {
        return Err("review, raw, time, and exposure frame counts must match".into());
    }
    if !psf_fwhm.is_finite() || psf_fwhm <= 0.0 {
        return Err("motion feature psf_fwhm must be finite and positive".into());
    }
    if frame_times.iter().any(|time| !time.is_finite())
        || frame_times.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err("motion feature frame_times must be finite and strictly increasing".into());
    }

    let mut intervals = frame_times
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    intervals.sort_by(f64::total_cmp);
    let reference_interval = if intervals.len() % 2 == 0 {
        let middle = intervals.len() / 2;
        0.5 * (intervals[middle - 1] + intervals[middle])
    } else {
        intervals[intervals.len() / 2]
    };
    let exposures = exposure_seconds
        .iter()
        .map(|exposure| match exposure {
            Some(value) if value.is_finite() && *value > 0.0 => Ok(*value),
            Some(_) => Err("exposure times must be finite and positive".to_string()),
            None => Ok(reference_interval),
        })
        .collect::<Result<Vec<_>>>()?;

    let mut supported_raw_frames = Vec::with_capacity(raw_components_by_frame.len());
    let mut supported_raw_indices = Vec::with_capacity(raw_components_by_frame.len());
    for raw_frame in raw_components_by_frame {
        let raw_points = raw_frame
            .get("points")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut supported_points = Vec::new();
        let mut original_indices = Vec::new();
        for (raw_index, point) in raw_points.iter().enumerate() {
            if sequence_line_model_review_supported(point, psf_fwhm)? {
                supported_points.push(point.clone());
                original_indices.push(raw_index);
            }
        }
        supported_raw_frames.push(supported_points);
        supported_raw_indices.push(original_indices);
    }
    let coherent = crate::sequence_trails::coherent_raw_motion_point_indices(&json!({
        "frames":supported_raw_frames,
        "psf_fwhm":psf_fwhm,
        "min_presence":3,
        "frame_times":frame_times,
        "exposure_times_s":exposures,
    }))?;
    let mut coherent_indices = std::collections::BTreeSet::<(usize, usize)>::new();
    if let Some(indices) = coherent.get("selected_indices").and_then(Value::as_array) {
        for pair in indices {
            let Some(pair) = pair.as_array() else {
                continue;
            };
            let (Some(frame_index), Some(supported_index)) = (
                pair.first().and_then(Value::as_u64),
                pair.get(1).and_then(Value::as_u64),
            ) else {
                continue;
            };
            let (Ok(frame_index), Ok(supported_index)) = (
                usize::try_from(frame_index),
                usize::try_from(supported_index),
            ) else {
                continue;
            };
            let Some(raw_index) = supported_raw_indices
                .get(frame_index)
                .and_then(|indices| indices.get(supported_index))
                .copied()
            else {
                continue;
            };
            coherent_indices.insert((frame_index, raw_index));
        }
    }

    let temporal_anchors = frames
        .iter()
        .flat_map(|frame| {
            frame
                .get("points")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|point| {
            Some((
                finite_number(point, "aligned_x")?,
                finite_number(point, "aligned_y")?,
            ))
        })
        .collect::<Vec<_>>();
    let mut updated_frames = frames;
    let mut fallback_audit = Vec::with_capacity(updated_frames.len());
    let mut unassociated_points = Vec::new();
    for (frame_index, frame) in updated_frames.iter_mut().enumerate() {
        let raw_points = raw_components_by_frame[frame_index]
            .get("points")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut frame_audit = frame.get("audit").cloned().unwrap_or_else(|| json!({}));
        let frame_points = frame
            .get_mut("points")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| "reviewed motion frame must contain a points array".to_string())?;
        let temporal_count = frame_points.len();
        let mut reviewed_raw_count = 0usize;
        let mut coherent_count = 0usize;
        let mut anchored_count = 0usize;
        let mut added_count = 0usize;
        let mut duplicate_count = 0usize;
        let mut unassociated_count = 0usize;
        for (raw_index, mut raw_point) in raw_points.into_iter().enumerate() {
            if !sequence_line_model_review_supported(&raw_point, psf_fwhm)? {
                continue;
            }
            reviewed_raw_count += 1;
            let coherent = coherent_indices.contains(&(frame_index, raw_index));
            if coherent {
                coherent_count += 1;
            }
            let anchor_supported = temporal_anchors.iter().any(|(anchor_x, anchor_y)| {
                let Some(x) = finite_number(&raw_point, "aligned_x") else {
                    return false;
                };
                let Some(y) = finite_number(&raw_point, "aligned_y") else {
                    return false;
                };
                (x - anchor_x).hypot(y - anchor_y) <= 24.0_f64.max(6.0 * psf_fwhm)
            });
            if anchor_supported {
                anchored_count += 1;
            }
            if !anchor_supported && !coherent {
                unassociated_count += 1;
                raw_point["sequence_candidate_source"] = json!("single_frame_trail_review_only");
                unassociated_points.push(json!({
                    "frame_index":frame_index,
                    "point":raw_point,
                }));
                continue;
            }

            let raw_x = finite_number(&raw_point, "aligned_x").unwrap_or(f64::NAN);
            let raw_y = finite_number(&raw_point, "aligned_y").unwrap_or(f64::NAN);
            let raw_length = finite_number(&raw_point, "length_px").unwrap_or(f64::NAN);
            let is_duplicate = frame_points.iter().any(|existing| {
                let Some(existing_x) = finite_number(existing, "aligned_x") else {
                    return false;
                };
                let Some(existing_y) = finite_number(existing, "aligned_y") else {
                    return false;
                };
                let Some(existing_length) = finite_number(existing, "length_px") else {
                    return false;
                };
                (raw_x - existing_x).hypot(raw_y - existing_y)
                    <= 6.0_f64.max(0.25 * raw_length.min(existing_length))
            });
            if is_duplicate {
                duplicate_count += 1;
                continue;
            }
            raw_point["sequence_candidate_source"] = json!("single_frame_shape");
            frame_points.push(raw_point);
            added_count += 1;
        }
        if let Some(audit) = frame_audit.as_object_mut() {
            audit.insert("feature_count".into(), json!(frame_points.len()));
            audit.insert(
                "max_feature_residual_snr".into(),
                frame_points
                    .iter()
                    .filter_map(|point| finite_number(point, "residual_snr"))
                    .max_by(f64::total_cmp)
                    .map_or(Value::Null, |value| json!(value)),
            );
            audit.insert("raw_fallback_added_count".into(), json!(added_count));
        }
        frame["audit"] = frame_audit;
        fallback_audit.push(json!({
            "frame_index":frame_index,
            "temporal_candidate_count":temporal_count,
            "review_supported_raw_count":reviewed_raw_count,
            "coherent_raw_chain_count":coherent_count,
            "temporal_anchor_count":anchored_count,
            "raw_fallback_added_count":added_count,
            "duplicate_raw_candidate_count":duplicate_count,
            "unassociated_single_frame_count":unassociated_count,
        }));
    }
    if let Some(object) = result.as_object_mut() {
        object.insert(
            "scope".into(),
            json!("temporal_components_reviewed_with_coherent_raw_shape_fallback"),
        );
        object.insert("frames".into(), json!(updated_frames));
        object.insert("raw_fallback_audit".into(), json!(fallback_audit));
        object.insert(
            "unassociated_single_frame_points".into(),
            json!(unassociated_points),
        );
    }
    Ok(result)
}

/// Associate Rust temporal-residual component proposals without promoting
/// them to confirmed motion features.
pub fn associate_temporal_component_tracks(
    component_result: &Value,
    frame_times: &[f64],
    exposure_seconds: &[Option<f64>],
    exposure_metadata_supported: bool,
    link_radius_px: f64,
    psf_fwhm: f64,
) -> Result<Value> {
    if !link_radius_px.is_finite() || link_radius_px <= 0.0 {
        return Err("motion line link radius must be finite and positive".into());
    }
    if !psf_fwhm.is_finite() || psf_fwhm <= 0.0 {
        return Err("motion feature psf_fwhm must be finite and positive".into());
    }
    let frames = component_result
        .get("frames")
        .and_then(Value::as_array)
        .ok_or_else(|| "motion component result must contain frames".to_string())?;
    if frames.is_empty() || component_result["status"] == "insufficient_frames" {
        return Ok(json!({
            "status":"insufficient_frames",
            "scope":"temporal_residual_component_association_only",
            "evidence_level":"unverified_candidate_proposal",
            "classification_performed":false,
            "tracks":[],
            "frame_audits":[],
        }));
    }
    if frames.len() != frame_times.len() || frames.len() != exposure_seconds.len() {
        return Err("motion component frames, times, and exposures must have equal lengths".into());
    }
    if frame_times.iter().any(|time| !time.is_finite())
        || frame_times.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err("motion feature frame_times must be finite and strictly increasing".into());
    }
    if exposure_metadata_supported
        && exposure_seconds
            .iter()
            .any(|value| value.is_none_or(|seconds| !seconds.is_finite() || seconds <= 0.0))
    {
        return Err("supported exposure metadata must contain positive exposure times".into());
    }

    if frames.len() < 2 {
        return Ok(json!({
            "status":"insufficient_frames",
            "scope":"temporal_residual_component_association_only",
            "evidence_level":"unverified_candidate_proposal",
            "classification_performed":false,
            "tracks":[],
            "frame_audits":[],
        }));
    }

    let mut tracks: Vec<Vec<Value>> = Vec::new();
    let mut frame_audits = Vec::with_capacity(frames.len());
    for (frame_index, frame) in frames.iter().enumerate() {
        let current_points = frame
            .get("points")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for point in &current_points {
            if point_frame_index(point)? != frame_index {
                return Err("motion component point frame_index does not match its frame".into());
            }
            let _ = point_xy(point)?;
            for key in ["length_px", "angle_deg"] {
                if !number(point, key)?.is_finite() {
                    return Err(format!("motion component {key} must be finite").into());
                }
            }
        }

        let mut predicted_tracks = Vec::new();
        let mut active_track_ids = Vec::new();
        let mut expected_steps = HashMap::<i64, f64>::new();
        for (track_id, history) in tracks.iter().enumerate() {
            let Some(last) = history.last() else {
                continue;
            };
            let last_index = point_frame_index(last)?;
            let frame_gap = frame_index.saturating_sub(last_index);
            if !(1..=MAX_FRAME_GAP).contains(&frame_gap) {
                continue;
            }
            let elapsed = frame_times[frame_index] - frame_times[last_index];
            if !elapsed.is_finite() || elapsed <= 0.0 {
                continue;
            }
            let last_xy = point_xy(last)?;
            let predicted = if history.len() >= 2 {
                let previous = &history[history.len() - 2];
                let previous_index = point_frame_index(previous)?;
                let previous_elapsed = frame_times[last_index] - frame_times[previous_index];
                if previous_index < last_index && previous_elapsed > 0.0 {
                    let previous_xy = point_xy(previous)?;
                    [
                        last_xy[0] + (last_xy[0] - previous_xy[0]) / previous_elapsed * elapsed,
                        last_xy[1] + (last_xy[1] - previous_xy[1]) / previous_elapsed * elapsed,
                    ]
                } else {
                    last_xy
                }
            } else {
                last_xy
            };
            let track_id = track_id as i64;
            active_track_ids.push(track_id);
            predicted_tracks.push(TrackPosition {
                track_id,
                x: predicted[0],
                y: predicted[1],
            });

            if exposure_metadata_supported {
                let exposure = exposure_seconds[last_index].expect("validated exposure metadata");
                if let Some(motion_length) =
                    deconvolved_motion_length(number(last, "length_px")?, psf_fwhm)
                {
                    let expected_step = motion_length * elapsed / exposure;
                    if expected_step.is_finite() && expected_step > 0.0 {
                        expected_steps.insert(track_id, expected_step);
                    }
                }
            }
        }

        let current_xy = current_points
            .iter()
            .map(point_xy)
            .collect::<Result<Vec<_>>>()?;
        let mut geometry_costs = HashMap::<(i64, usize), f64>::new();
        for position in &predicted_tracks {
            for (source_index, point) in current_points.iter().enumerate() {
                let predicted_distance = (position.x - current_xy[source_index][0])
                    .hypot(position.y - current_xy[source_index][1]);
                if predicted_distance > link_radius_px {
                    continue;
                }
                let expected_step = expected_steps.get(&position.track_id).copied();
                let cost = sequence_trails::motion_feature_pair_geometry_evidence(&json!({
                    "previous":tracks[position.track_id as usize].last().expect("active track has a point"),
                    "current":point,
                    "psf_fwhm":psf_fwhm,
                    "expected_step_px":expected_step,
                }))?;
                geometry_costs.insert(
                    (position.track_id, source_index),
                    cost["penalty"].as_f64().unwrap_or(0.0),
                );
            }
        }
        let pair_allowed = |track_id: i64, source_index: usize, _distance: f64| {
            let Some(expected_step) = expected_steps.get(&track_id) else {
                return true;
            };
            let Some(last) = tracks
                .get(track_id as usize)
                .and_then(|history| history.last())
            else {
                return false;
            };
            let Ok(last_xy) = point_xy(last) else {
                return false;
            };
            let observed_step = (current_xy[source_index][0] - last_xy[0])
                .hypot(current_xy[source_index][1] - last_xy[1]);
            let ratio = observed_step / expected_step.max(f64::EPSILON);
            ratio.is_finite()
                && (EXPOSURE_LINK_RATIO_MIN..=EXPOSURE_LINK_RATIO_MAX).contains(&ratio)
        };
        let pair_cost = |track_id: i64, source_index: usize, _distance: f64| {
            geometry_costs
                .get(&(track_id, source_index))
                .copied()
                .unwrap_or(0.0)
        };
        let assignments = global_frame_assignments(
            &predicted_tracks,
            &current_xy,
            link_radius_px,
            Some(&pair_allowed),
            Some(&pair_cost),
        )?;

        let mut assigned_tracks = vec![false; tracks.len()];
        let mut assigned_sources = vec![false; current_points.len()];
        for assignment in assignments {
            let track_id = usize::try_from(assignment.track_id)
                .map_err(|_| "motion assignment returned a negative track id")?;
            if track_id >= tracks.len()
                || assignment.source_index >= current_points.len()
                || assigned_tracks[track_id]
                || assigned_sources[assignment.source_index]
            {
                continue;
            }
            tracks[track_id].push(current_points[assignment.source_index].clone());
            assigned_tracks[track_id] = true;
            assigned_sources[assignment.source_index] = true;
        }
        let associated_count = assigned_sources
            .iter()
            .filter(|assigned| **assigned)
            .count();
        let mut new_track_count = 0usize;
        for (source_index, point) in current_points.iter().enumerate() {
            if !assigned_sources[source_index] {
                tracks.push(vec![point.clone()]);
                new_track_count += 1;
            }
        }
        frame_audits.push(json!({
            "frame_index":frame_index,
            "component_count":current_points.len(),
            "eligible_track_count":active_track_ids.len(),
            "associated_count":associated_count,
            "new_track_count":new_track_count,
        }));
    }

    let rendered_tracks = tracks
        .into_iter()
        .enumerate()
        .map(|(track_id, points)| {
            json!({
                "track_id":track_id,
                "point_count":points.len(),
                "points":points,
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "status":"analyzed",
        "scope":"temporal_residual_component_association_only",
        "evidence_level":"unverified_candidate_proposal",
        "classification_performed":false,
        "link_radius_px":link_radius_px,
        "max_frame_gap":MAX_FRAME_GAP,
        "tracks":rendered_tracks,
        "frame_audits":frame_audits,
    }))
}

#[cfg(test)]
mod tests {
    use super::associate_temporal_component_tracks;
    use serde_json::{json, Value};

    fn point(frame_index: usize, x: f64, length_px: f64) -> Value {
        json!({
            "frame_index":frame_index,
            "aligned_x":x,
            "aligned_y":20.0,
            "length_px":length_px,
            "width_px":2.0,
            "angle_deg":0.0,
            "touches_edge":false,
        })
    }

    fn components(frames: Vec<Vec<Value>>) -> Value {
        json!({
            "status":"analyzed",
            "frames":frames.into_iter().map(|points| json!({"points":points})).collect::<Vec<_>>(),
        })
    }

    #[test]
    fn links_constant_velocity_components_but_keeps_them_unclassified() {
        let result = associate_temporal_component_tracks(
            &components(vec![
                vec![point(0, 10.0, 12.0)],
                vec![point(1, 15.0, 12.0)],
                vec![point(2, 20.0, 12.0)],
            ]),
            &[0.0, 1.0, 2.0],
            &[None, None, None],
            false,
            8.0,
            2.0,
        )
        .unwrap();

        assert_eq!(result["tracks"].as_array().unwrap().len(), 1);
        assert_eq!(result["tracks"][0]["point_count"], 3);
        assert_eq!(result["classification_performed"], false);
        assert_eq!(result["evidence_level"], "unverified_candidate_proposal");
        assert_eq!(result["frame_audits"][1]["associated_count"], 1);
    }

    #[test]
    fn predicts_across_two_missing_frames_but_not_three() {
        let bridged = associate_temporal_component_tracks(
            &components(vec![
                vec![point(0, 10.0, 12.0)],
                vec![],
                vec![],
                vec![point(3, 25.0, 12.0)],
            ]),
            &[0.0, 1.0, 2.0, 3.0],
            &[None; 4],
            false,
            20.0,
            2.0,
        )
        .unwrap();
        assert_eq!(bridged["tracks"].as_array().unwrap().len(), 1);
        assert_eq!(bridged["tracks"][0]["point_count"], 2);

        let split = associate_temporal_component_tracks(
            &components(vec![
                vec![point(0, 10.0, 12.0)],
                vec![],
                vec![],
                vec![],
                vec![point(4, 30.0, 12.0)],
            ]),
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            &[None; 5],
            false,
            32.0,
            2.0,
        )
        .unwrap();
        assert_eq!(split["tracks"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn uses_one_to_one_assignment_when_nearest_edges_compete() {
        let result = associate_temporal_component_tracks(
            &components(vec![
                vec![point(0, 0.0, 12.0), point(0, 4.0, 12.0)],
                vec![point(1, 1.0, 12.0), point(1, -2.0, 12.0)],
            ]),
            &[0.0, 1.0],
            &[None, None],
            false,
            6.0,
            2.0,
        )
        .unwrap();

        let tracks = result["tracks"].as_array().unwrap();
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0]["points"].as_array().unwrap().len(), 2);
        assert_eq!(tracks[1]["points"].as_array().unwrap().len(), 2);
        assert_eq!(tracks[0]["points"][1]["aligned_x"], -2.0);
        assert_eq!(tracks[1]["points"][1]["aligned_x"], 1.0);
    }

    #[test]
    fn exposure_velocity_gate_rejects_inconsistent_component_step() {
        let result = associate_temporal_component_tracks(
            &components(vec![vec![point(0, 10.0, 10.0)], vec![point(1, 35.0, 10.0)]]),
            &[0.0, 1.0],
            &[Some(1.0), Some(1.0)],
            true,
            40.0,
            2.0,
        )
        .unwrap();

        assert_eq!(result["tracks"].as_array().unwrap().len(), 2);
        assert_eq!(result["frame_audits"][1]["associated_count"], 0);
        assert_eq!(result["frame_audits"][1]["new_track_count"], 1);
    }
}
