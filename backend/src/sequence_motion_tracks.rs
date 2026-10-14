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
