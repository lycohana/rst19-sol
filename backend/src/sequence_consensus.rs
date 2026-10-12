//! Repeated-candidate consensus before image-level quality promotion.
use crate::{
    sequence::TrackPoint,
    sequence_motion::{fit_track_motion, TrackObservation},
    spatial::KdTree,
    statistics, Result,
};
use serde_json::{json, Map, Value};

#[derive(Debug, Clone)]
struct CandidatePoint {
    x: f64,
    y: f64,
    snr: f64,
    temporal: bool,
    reference_snr: f64,
}

fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn normalize_frames(
    raw_frames: &[Value],
    min_snr: f64,
    temporal_min_snr: f64,
    reference_min_snr: f64,
) -> Option<(Vec<Vec<CandidatePoint>>, usize, usize)> {
    let mut frames = Vec::with_capacity(raw_frames.len());
    let mut input_rows = 0;
    let mut gated_rows = 0;
    for raw_frame in raw_frames {
        let rows = raw_frame.as_array()?;
        let mut frame = Vec::new();
        for raw_row in rows {
            let row = raw_row.as_array()?;
            if row.is_empty() {
                continue;
            }
            input_rows += 1;
            if row.len() < 3 {
                return None;
            }
            let (Some(x), Some(y), Some(snr)) = (number(&row[0]), number(&row[1]), number(&row[2]))
            else {
                continue;
            };
            if !x.is_finite() || !y.is_finite() || !snr.is_finite() {
                continue;
            }
            let temporal = row
                .get(3)
                .and_then(number)
                .is_some_and(|value| value.is_finite() && value >= 0.5);
            let reference_snr = row
                .get(4)
                .and_then(number)
                .filter(|value| value.is_finite())
                .unwrap_or(f64::NAN);
            if snr < if temporal { temporal_min_snr } else { min_snr } {
                continue;
            }
            if temporal && (!reference_snr.is_finite() || reference_snr < reference_min_snr) {
                continue;
            }
            gated_rows += 1;
            frame.push(CandidatePoint {
                x,
                y,
                snr,
                temporal,
                reference_snr,
            });
        }
        frames.push(frame);
    }
    Some((frames, input_rows, gated_rows))
}

fn quality_by_frame(
    tracks: &[Value],
    frame_count: usize,
    persistent_min_presence: usize,
) -> Vec<Vec<[f64; 2]>> {
    let mut points = vec![Vec::new(); frame_count];
    for track in tracks {
        let rows = track["points"].as_array().map(Vec::as_slice).unwrap_or(&[]);
        let presence = track["presence"]
            .as_u64()
            .map_or(rows.len(), |value| value as usize);
        if presence < persistent_min_presence {
            continue;
        }
        for point in rows {
            let Some(index) = point["frame_index"].as_u64().map(|value| value as usize) else {
                continue;
            };
            let (Some(x), Some(y)) = (point["aligned_x"].as_f64(), point["aligned_y"].as_f64())
            else {
                continue;
            };
            if index < frame_count && x.is_finite() && y.is_finite() {
                points[index].push([x, y]);
            }
        }
    }
    points
}

fn normalize_times(times: Option<&[f64]>, frame_count: usize) -> Option<Vec<f64>> {
    let values = times?;
    if values.len() != frame_count
        || values.iter().any(|value| !value.is_finite())
        || values.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Some((0..frame_count).map(|index| index as f64).collect());
    }
    Some(values.iter().map(|value| value - values[0]).collect())
}

/// Link repeated matched-filter candidate peaks into conservative persistent tracks.
/// This geometry-only layer intentionally does not apply image-pixel PSF or line-mask gates.
#[allow(clippy::too_many_arguments)]
pub fn candidate_consensus(
    candidate_peak_frames: &[Value],
    cumulative_shifts: &[[f64; 2]],
    quality_tracks: &[Value],
    frame_times: Option<&[f64]>,
    link_radius_px: f64,
    min_presence: usize,
    persistent_min_presence: usize,
    motion_min_displacement_px: f64,
    max_motion_fit_rms_px: f64,
    min_candidate_snr: f64,
    temporal_candidate_min_snr: Option<f64>,
    temporal_reference_min_snr: Option<f64>,
    temporal_min_psf_correlation: f64,
) -> Result<Value> {
    if candidate_peak_frames.is_empty() || candidate_peak_frames.len() != cumulative_shifts.len() {
        return Ok(json!({"tracks":[],"audit":{}}));
    }
    let temporal_reference_gate = temporal_reference_min_snr.unwrap_or(min_candidate_snr);
    if link_radius_px <= 0.0 || min_presence < 1 || persistent_min_presence < 1 {
        return Err("candidate consensus thresholds are invalid".into());
    }
    if persistent_min_presence > min_presence || min_candidate_snr <= 0.0 {
        return Err("candidate consensus presence/SNR thresholds are invalid".into());
    }
    if temporal_reference_gate <= 0.0 || temporal_reference_gate > min_candidate_snr {
        return Err("temporal reference SNR threshold is invalid".into());
    }
    let temporal_min_snr = temporal_candidate_min_snr.unwrap_or((0.5 * min_candidate_snr).max(5.0));
    if temporal_min_snr <= 0.0 || temporal_min_snr > min_candidate_snr {
        return Err("temporal candidate SNR threshold is invalid".into());
    }
    if !(0.0 < temporal_min_psf_correlation && temporal_min_psf_correlation <= 1.0) {
        return Err("temporal PSF correlation threshold is invalid".into());
    }
    let Some((frames, input_rows, gated_rows)) = normalize_frames(
        candidate_peak_frames,
        min_candidate_snr,
        temporal_min_snr,
        temporal_reference_gate,
    ) else {
        return Ok(json!({"tracks":[],"audit":{}}));
    };
    if frames.iter().all(Vec::is_empty) {
        return Ok(
            json!({"tracks":[],"audit":{"input_rows":input_rows,"rows_after_snr_gate":gated_rows}}),
        );
    }

    let frame_count = frames.len();
    let mut anchors = Vec::<[f64; 5]>::new();
    for (frame_index, current) in frames.iter().enumerate() {
        let aligned = current
            .iter()
            .map(|point| {
                [
                    point.x - cumulative_shifts[frame_index][0],
                    point.y - cumulative_shifts[frame_index][1],
                ]
            })
            .collect::<Vec<_>>();
        let old_anchor_count = anchors.len();
        let tree = if old_anchor_count > 0 {
            Some(KdTree::new(
                &anchors
                    .iter()
                    .map(|anchor| [anchor[0], anchor[1]])
                    .collect::<Vec<_>>(),
            )?)
        } else {
            None
        };
        for (index, point) in current.iter().enumerate() {
            let is_new = match tree.as_ref().and_then(|tree| tree.nearest(aligned[index])) {
                Some((_closest, distance)) => distance > link_radius_px,
                None => true,
            };
            if is_new {
                anchors.push([
                    aligned[index][0],
                    aligned[index][1],
                    point.snr,
                    if point.temporal { 1.0 } else { 0.0 },
                    point.reference_snr,
                ]);
            }
        }
    }
    let anchor_points = anchors
        .iter()
        .map(|row| [row[0], row[1]])
        .collect::<Vec<_>>();
    let anchor_tree = KdTree::new(&anchor_points)?;
    let mut matches = vec![vec![None::<usize>; frame_count]; anchors.len()];
    let mut scores = vec![vec![None::<f64>; frame_count]; anchors.len()];
    for frame_index in 0..frame_count {
        let current = &frames[frame_index];
        if current.is_empty() {
            continue;
        }
        let aligned = current
            .iter()
            .map(|point| {
                [
                    point.x - cumulative_shifts[frame_index][0],
                    point.y - cumulative_shifts[frame_index][1],
                ]
            })
            .collect::<Vec<_>>();
        let current_tree = KdTree::new(&aligned)?;
        let mut reciprocal = Vec::<(usize, usize, f64)>::new();
        for (anchor_index, anchor) in anchor_points.iter().enumerate() {
            let Some((current_index, distance)) = current_tree.nearest(*anchor) else {
                continue;
            };
            if distance > link_radius_px {
                continue;
            }
            if anchor_tree
                .nearest(aligned[current_index])
                .is_some_and(|(nearest_anchor, _)| nearest_anchor == anchor_index)
            {
                reciprocal.push((anchor_index, current_index, distance));
            }
        }
        reciprocal.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)));
        let mut used_current = vec![false; current.len()];
        for (anchor_index, current_index, _) in reciprocal {
            if used_current[current_index] {
                continue;
            }
            used_current[current_index] = true;
            matches[anchor_index][frame_index] = Some(current_index);
            scores[anchor_index][frame_index] = Some(current[current_index].snr);
        }
    }
    let eligible = matches
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            (row.iter().filter(|value| value.is_some()).count() >= persistent_min_presence)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        return Ok(json!({
            "tracks":[],
            "audit":{
                "input_rows":input_rows,
                "rows_after_snr_gate":gated_rows,
                "anchor_total":anchors.len(),
                "anchor_presence_ge_persistent":0
            }
        }));
    }
    let quality = quality_by_frame(quality_tracks, frame_count, persistent_min_presence);
    let quality_trees = quality
        .iter()
        .map(|points| KdTree::new(points))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let merge_radius = 1.25f64.max((link_radius_px * 0.5).min(2.0));
    let times = normalize_times(frame_times, frame_count);
    let mut rendered = Vec::new();
    let mut audit = Map::new();
    audit.insert("input_rows".into(), json!(input_rows));
    audit.insert("rows_after_snr_gate".into(), json!(gated_rows));
    audit.insert("anchor_total".into(), json!(anchors.len()));
    audit.insert(
        "anchor_presence_ge_persistent".into(),
        json!(eligible.len()),
    );
    let mut reject_quality_overlap = 0;
    let mut reject_candidate_snr = 0;
    let mut reject_reference_snr = 0;
    let mut reject_motion_fit = 0;
    for anchor_index in &eligible {
        let mut points = Vec::<TrackPoint>::new();
        let mut overlaps_quality = false;
        for frame_index in 0..frame_count {
            let Some(current_index) = matches[*anchor_index][frame_index] else {
                continue;
            };
            let candidate = &frames[frame_index][current_index];
            let shift = cumulative_shifts[frame_index];
            points.push(TrackPoint {
                frame_index,
                detection_id: -1 - *anchor_index as i64,
                x: candidate.x,
                y: candidate.y,
                aligned_x: candidate.x - shift[0],
                aligned_y: candidate.y - shift[1],
                flux_snr: None,
                quality_passed: false,
                candidate_snr: Some(candidate.snr),
            });
            if quality_trees[frame_index]
                .nearest([candidate.x - shift[0], candidate.y - shift[1]])
                .is_some_and(|(_, distance)| distance <= merge_radius)
            {
                overlaps_quality = true;
            }
        }
        if overlaps_quality {
            reject_quality_overlap += 1;
            continue;
        }
        let matched_scores = scores[*anchor_index]
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let median_snr = statistics::median(&matched_scores)?;
        let temporal = anchors[*anchor_index][3] >= 0.5;
        let score_threshold = if temporal {
            temporal_min_snr
        } else {
            min_candidate_snr
        };
        if !median_snr.is_finite() || median_snr < score_threshold {
            reject_candidate_snr += 1;
            continue;
        }
        if temporal
            && (!anchors[*anchor_index][4].is_finite()
                || anchors[*anchor_index][4] < temporal_reference_gate)
        {
            reject_reference_snr += 1;
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
        let fit = fit_track_motion(&observations, times.as_deref())?;
        if fit.fit_rms_px.is_none_or(|rms| rms > max_motion_fit_rms_px)
            || fit.displacement_px >= motion_min_displacement_px
        {
            reject_motion_fit += 1;
            continue;
        }
        rendered.push(json!({
            "track_id":-1,
            "classification":"persistent",
            "presence":points.len(),
            "displacement_px":fit.displacement_px,
            "speed_px_per_frame":fit.speed_px_per_frame,
            "fit_rms_px":fit.fit_rms_px,
            "quality_presence":0,
            "evidence_level":if temporal { "temporal_reference" } else { "candidate_consensus" },
            "points":points,
        }));
    }
    audit.insert("anchors_evaluated".into(), json!(eligible.len()));
    for (key, value) in [
        ("reject_quality_overlap", reject_quality_overlap),
        ("reject_median_candidate_snr", reject_candidate_snr),
        ("reject_temporal_reference_snr", reject_reference_snr),
        ("reject_motion_fit", reject_motion_fit),
    ] {
        if value > 0 {
            audit.insert(key.into(), json!(value));
        }
    }
    if !rendered.is_empty() {
        audit.insert("accepted".into(), json!(rendered.len()));
        let temporal_count = rendered
            .iter()
            .filter(|track| track["evidence_level"] == "temporal_reference")
            .count();
        let normal_count = rendered.len() - temporal_count;
        if temporal_count > 0 {
            audit.insert("accepted_temporal_reference".into(), json!(temporal_count));
        }
        if normal_count > 0 {
            audit.insert("accepted_candidate_consensus".into(), json!(normal_count));
        }
    }
    Ok(json!({"tracks":rendered,"audit":audit}))
}
