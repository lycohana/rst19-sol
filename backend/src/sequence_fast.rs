//! Supplemental constant-velocity association for quality point detections.
use crate::{
    measurement::Detection,
    sequence::TrackPoint,
    sequence_motion::{fit_track_motion, TrackObservation},
    spatial::KdTree,
    statistics, Result,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FastPointOptions {
    pub link_radius_px: f64,
    pub min_presence: Option<usize>,
    pub motion_min_displacement_px: f64,
    pub min_flux_snr: f64,
    pub max_step_px: f64,
    pub gate_px: f64,
    pub max_fit_rms_px: f64,
}

impl Default for FastPointOptions {
    fn default() -> Self {
        Self {
            link_radius_px: 4.0,
            min_presence: None,
            motion_min_displacement_px: 2.0,
            min_flux_snr: 7.0,
            max_step_px: 20.0,
            gate_px: 4.5,
            max_fit_rms_px: 1.5,
        }
    }
}

#[derive(Debug, Clone)]
struct Record {
    source: Detection,
    aligned_x: f64,
    aligned_y: f64,
    signal_snr: f64,
}

#[derive(Debug)]
struct Candidate {
    points: Vec<TrackPoint>,
    displacement: f64,
    speed: f64,
    fit_rms: f64,
    median_snr: f64,
}

fn normalized_times(times: Option<&[f64]>, frame_count: usize) -> Vec<f64> {
    let fallback = (0..frame_count)
        .map(|index| index as f64)
        .collect::<Vec<_>>();
    let Some(values) = times else {
        return fallback;
    };
    if values.len() != frame_count
        || values.iter().any(|value| !value.is_finite())
        || values.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return fallback;
    }
    values.iter().map(|value| value - values[0]).collect()
}

fn reference_interval(times: &[f64]) -> f64 {
    let differences = times
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    statistics::median(&differences)
        .ok()
        .filter(|value| *value > 0.0)
        .unwrap_or(1.0)
}

fn existing_reserved(existing_tracks: &[Value]) -> HashSet<(usize, i64)> {
    let mut reserved = HashSet::new();
    for track in existing_tracks {
        let points = track["points"].as_array().map(Vec::as_slice).unwrap_or(&[]);
        let presence = track["presence"]
            .as_u64()
            .map_or(points.len(), |n| n as usize);
        if presence < 2 {
            continue;
        }
        for point in points {
            if let (Some(frame), Some(detection_id)) = (
                point["frame_index"].as_u64(),
                point["detection_id"].as_i64(),
            ) {
                reserved.insert((frame as usize, detection_id));
            }
        }
    }
    reserved
}

fn signal_snr(source: &Detection) -> f64 {
    let value = source.flux_snr.unwrap_or(source.snr);
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn point(frame_index: usize, record: &Record) -> TrackPoint {
    TrackPoint {
        frame_index,
        detection_id: record.source.detection_id,
        x: record.source.x,
        y: record.source.y,
        aligned_x: record.aligned_x,
        aligned_y: record.aligned_y,
        flux_snr: Some(record.signal_snr),
        quality_passed: record.source.quality_passed,
        candidate_snr: None,
    }
}

fn fit_prediction(
    observations: &[(usize, usize)],
    records: &[Vec<Record>],
    times: &[f64],
) -> ([f64; 2], [f64; 2]) {
    let count = observations.len() as f64;
    let mean_t = observations
        .iter()
        .map(|(frame, _)| times[*frame])
        .sum::<f64>()
        / count;
    let mean_x = observations
        .iter()
        .map(|(frame, index)| records[*frame][*index].aligned_x)
        .sum::<f64>()
        / count;
    let mean_y = observations
        .iter()
        .map(|(frame, index)| records[*frame][*index].aligned_y)
        .sum::<f64>()
        / count;
    let denominator = observations
        .iter()
        .map(|(frame, _)| (times[*frame] - mean_t).powi(2))
        .sum::<f64>();
    let (mut numerator_x, mut numerator_y) = (0.0, 0.0);
    for (frame, index) in observations {
        let delta_t = times[*frame] - mean_t;
        numerator_x += delta_t * (records[*frame][*index].aligned_x - mean_x);
        numerator_y += delta_t * (records[*frame][*index].aligned_y - mean_y);
    }
    let slope_x = numerator_x / denominator;
    let slope_y = numerator_y / denominator;
    (
        [mean_x - slope_x * mean_t, slope_x],
        [mean_y - slope_y * mean_t, slope_y],
    )
}

fn choose_nearby(
    frame: usize,
    predicted: [f64; 2],
    used: &HashSet<(usize, i64)>,
    records: &[Vec<Record>],
    trees: &[KdTree],
    gate: f64,
) -> Result<Option<usize>> {
    let candidates = trees[frame].within_radius(predicted, gate)?;
    let mut choices = candidates
        .into_iter()
        .filter(|(index, _)| !used.contains(&(frame, records[frame][*index].source.detection_id)))
        .map(|(index, distance)| (distance, -records[frame][index].signal_snr, index))
        .collect::<Vec<_>>();
    choices.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.total_cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    Ok(choices.first().map(|choice| choice.2))
}

fn follow_seed(
    start_frame: usize,
    first: usize,
    second: usize,
    records: &[Vec<Record>],
    trees: &[KdTree],
    times: &[f64],
    gate: f64,
) -> Result<Vec<(usize, usize)>> {
    let mut observations = vec![(start_frame, first), (start_frame + 1, second)];
    let mut used = HashSet::from([
        (start_frame, records[start_frame][first].source.detection_id),
        (
            start_frame + 1,
            records[start_frame + 1][second].source.detection_id,
        ),
    ]);
    let mut missed = 0;
    for frame in start_frame + 2..records.len() {
        let (x, y) = fit_prediction(&observations, records, times);
        let predicted = [x[0] + x[1] * times[frame], y[0] + y[1] * times[frame]];
        if let Some(index) = choose_nearby(frame, predicted, &used, records, trees, gate)? {
            missed = 0;
            used.insert((frame, records[frame][index].source.detection_id));
            observations.push((frame, index));
        } else {
            missed += 1;
            if missed > 1 {
                break;
            }
        }
    }
    missed = 0;
    for frame in (0..start_frame).rev() {
        observations.sort_by_key(|item| item.0);
        let (x, y) = fit_prediction(&observations, records, times);
        let predicted = [x[0] + x[1] * times[frame], y[0] + y[1] * times[frame]];
        if let Some(index) = choose_nearby(frame, predicted, &used, records, trees, gate)? {
            missed = 0;
            used.insert((frame, records[frame][index].source.detection_id));
            observations.push((frame, index));
        } else {
            missed += 1;
            if missed > 1 {
                break;
            }
        }
    }
    observations.sort_by_key(|item| item.0);
    Ok(observations)
}

fn metrics(points: &[TrackPoint], frame_times: Option<&[f64]>) -> Result<(f64, f64, f64)> {
    let observations = points
        .iter()
        .map(|point| TrackObservation {
            frame_index: point.frame_index as i64,
            aligned_x: point.aligned_x,
            aligned_y: point.aligned_y,
        })
        .collect::<Vec<_>>();
    let metric = fit_track_motion(&observations, frame_times)?;
    Ok((
        metric.displacement_px,
        metric.speed_px_per_frame,
        metric.fit_rms_px.unwrap_or(f64::INFINITY),
    ))
}

/// Supplement ordinary short-radius tracks with high-SNR constant-velocity quality detections.
pub fn track_fast_point_movers(
    frames: &[Vec<Detection>],
    cumulative_shifts: &[[f64; 2]],
    frame_times: Option<&[f64]>,
    options: &FastPointOptions,
    existing_tracks: &[Value],
) -> Result<Value> {
    let frame_count = frames.len();
    if frame_count < 2 {
        return Ok(json!({"tracks":[]}));
    }
    if cumulative_shifts.len() != frame_count {
        return Err("cumulative_shifts length must match frame_sources".into());
    }
    if options.link_radius_px <= 0.0 || options.motion_min_displacement_px <= 0.0 {
        return Err("link radius and minimum displacement must be positive".into());
    }
    if options.min_flux_snr <= 0.0
        || options.max_step_px <= 0.0
        || options.gate_px <= 0.0
        || options.max_fit_rms_px <= 0.0
    {
        return Err("fast point motion thresholds must be positive".into());
    }
    let required = options
        .min_presence
        .unwrap_or_else(|| 3usize.max((4 * frame_count).div_ceil(5)));
    if required < 2 || required > frame_count {
        return Err("min_presence must be within 2..frame count".into());
    }
    if options.max_step_px <= options.link_radius_px {
        return Err("max_step_px must exceed link_radius_px".into());
    }

    let times = normalized_times(frame_times, frame_count);
    let interval_s = reference_interval(&times);
    let reserved = existing_reserved(existing_tracks);
    let mut records = Vec::with_capacity(frame_count);
    for (frame_index, sources) in frames.iter().enumerate() {
        let mut frame_records = Vec::new();
        for source in sources {
            if !source.quality_passed || reserved.contains(&(frame_index, source.detection_id)) {
                continue;
            }
            let snr = signal_snr(source);
            if snr < options.min_flux_snr {
                continue;
            }
            let aligned_x = source.x - cumulative_shifts[frame_index][0];
            let aligned_y = source.y - cumulative_shifts[frame_index][1];
            if !aligned_x.is_finite() || !aligned_y.is_finite() {
                return Err("point coordinates must be finite for fast motion association".into());
            }
            frame_records.push(Record {
                source: source.clone(),
                aligned_x,
                aligned_y,
                signal_snr: snr,
            });
        }
        records.push(frame_records);
    }
    if records.iter().all(Vec::is_empty) {
        return Ok(json!({"tracks":[]}));
    }
    let trees = records
        .iter()
        .map(|frame| {
            let points = frame
                .iter()
                .map(|record| [record.aligned_x, record.aligned_y])
                .collect::<Vec<_>>();
            KdTree::new(&points)
        })
        .collect::<Result<Vec<_>>>()?;

    let min_step = (options.link_radius_px * 1.01)
        .max(options.motion_min_displacement_px / (frame_count - 1).max(1) as f64);
    let min_speed_s = min_step / interval_s;
    let max_speed_s = options.max_step_px / interval_s;
    let mut candidates = Vec::new();
    for start in 0..frame_count - 1 {
        let elapsed = times[start + 1] - times[start];
        if elapsed <= 0.0 {
            continue;
        }
        let max_step = max_speed_s * elapsed;
        for first in 0..records[start].len() {
            let a = &records[start][first];
            for (second, step) in
                trees[start + 1].within_radius([a.aligned_x, a.aligned_y], max_step)?
            {
                if step < min_speed_s * elapsed {
                    continue;
                }
                let observations = follow_seed(
                    start,
                    first,
                    second,
                    &records,
                    &trees,
                    &times,
                    options.gate_px,
                )?;
                if observations.len() < required {
                    continue;
                }
                let points = observations
                    .iter()
                    .map(|(frame, index)| point(*frame, &records[*frame][*index]))
                    .collect::<Vec<_>>();
                let (displacement, speed, fit_rms) = metrics(&points, Some(&times))?;
                if fit_rms > options.max_fit_rms_px
                    || speed < min_step
                    || speed > options.max_step_px
                    || displacement < options.motion_min_displacement_px
                {
                    continue;
                }
                let (x, y) = fit_prediction(&observations, &records, &times);
                let max_residual = observations
                    .iter()
                    .map(|(frame, index)| {
                        let record = &records[*frame][*index];
                        (record.aligned_x - (x[0] + x[1] * times[*frame]))
                            .hypot(record.aligned_y - (y[0] + y[1] * times[*frame]))
                    })
                    .fold(0.0, f64::max);
                if max_residual > options.gate_px {
                    continue;
                }
                if observations
                    .windows(2)
                    .any(|pair| pair[1].0 - pair[0].0 > 2)
                {
                    continue;
                }
                let snrs = points
                    .iter()
                    .filter_map(|point| point.flux_snr)
                    .collect::<Vec<_>>();
                let median_snr = statistics::median(&snrs).unwrap_or(0.0);
                candidates.push(Candidate {
                    points,
                    displacement,
                    speed,
                    fit_rms,
                    median_snr,
                });
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.points
            .len()
            .cmp(&a.points.len())
            .then_with(|| (-b.fit_rms).total_cmp(&-a.fit_rms))
            .then_with(|| b.median_snr.total_cmp(&a.median_snr))
            .then_with(|| (-b.speed).total_cmp(&-a.speed))
    });
    let mut accepted: Vec<Candidate> = Vec::new();
    let mut accepted_keys: Vec<HashSet<(usize, i64)>> = Vec::new();
    for candidate in candidates {
        let keys = candidate
            .points
            .iter()
            .map(|point| (point.frame_index, point.detection_id))
            .collect::<HashSet<_>>();
        let duplicate = accepted_keys.iter().any(|previous| {
            let shared = keys.intersection(previous).count();
            shared >= 2usize.max(keys.len().min(previous.len()).div_ceil(2))
        });
        if duplicate {
            continue;
        }
        accepted_keys.push(keys);
        accepted.push(candidate);
    }
    let tracks = accepted.iter().map(|candidate| json!({
        "track_id":-1,
        "classification":"moving",
        "presence":candidate.points.len(),
        "displacement_px":candidate.displacement,
        "speed_px_per_frame":candidate.speed,
        "fit_rms_px":candidate.fit_rms,
        "quality_presence":candidate.points.iter().filter(|point| point.quality_passed).count(),
        "evidence_level":"fast_point_motion",
        "points":candidate.points,
    })).collect::<Vec<_>>();
    Ok(json!({"tracks":tracks}))
}
