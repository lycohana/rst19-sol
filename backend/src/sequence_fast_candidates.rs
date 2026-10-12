//! Coordinate-only association of raw matched-filter peaks into fast candidates.
//!
//! The operation deliberately stops before original-image forced photometry.
//! Candidate peaks are proposals; the caller must still measure and quality-gate
//! them against the original pixels before creating sequence tracks.

use crate::{spatial::KdTree, statistics, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;

#[derive(Debug, Clone, Deserialize)]
pub struct FastCandidateRecord {
    pub row_index: usize,
    pub aligned_x: f64,
    pub aligned_y: f64,
    pub filter_snr: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FastCandidateOptions {
    pub link_radius_px: f64,
    pub min_presence: usize,
    pub motion_min_displacement_px: f64,
    pub max_step_px: f64,
    pub gate_px: f64,
    pub max_fit_rms_px: f64,
}

impl Default for FastCandidateOptions {
    fn default() -> Self {
        Self {
            link_radius_px: 4.0,
            min_presence: 3,
            motion_min_displacement_px: 2.0,
            max_step_px: 20.0,
            gate_px: 4.5,
            max_fit_rms_px: 1.5,
        }
    }
}

#[derive(Debug)]
struct Candidate {
    observations: Vec<(usize, usize)>,
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

fn fit_prediction(
    observations: &[(usize, usize)],
    frames: &[Vec<FastCandidateRecord>],
    times: &[f64],
) -> Result<([f64; 2], [f64; 2])> {
    if observations.len() < 2 {
        return Err("fast candidate fit requires at least two observations".into());
    }
    let count = observations.len() as f64;
    let mean_t = observations
        .iter()
        .map(|(frame, _)| times[*frame])
        .sum::<f64>()
        / count;
    let mean_x = observations
        .iter()
        .map(|(frame, index)| frames[*frame][*index].aligned_x)
        .sum::<f64>()
        / count;
    let mean_y = observations
        .iter()
        .map(|(frame, index)| frames[*frame][*index].aligned_y)
        .sum::<f64>()
        / count;
    let denominator = observations
        .iter()
        .map(|(frame, _)| (times[*frame] - mean_t).powi(2))
        .sum::<f64>();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("fast candidate observation times are degenerate".into());
    }
    let (mut numerator_x, mut numerator_y) = (0.0, 0.0);
    for (frame, index) in observations {
        let delta_t = times[*frame] - mean_t;
        numerator_x += delta_t * (frames[*frame][*index].aligned_x - mean_x);
        numerator_y += delta_t * (frames[*frame][*index].aligned_y - mean_y);
    }
    let slope_x = numerator_x / denominator;
    let slope_y = numerator_y / denominator;
    Ok((
        [mean_x - slope_x * mean_t, slope_x],
        [mean_y - slope_y * mean_t, slope_y],
    ))
}

fn choose_nearby(
    frame: usize,
    predicted: [f64; 2],
    used: &HashSet<(usize, usize)>,
    frames: &[Vec<FastCandidateRecord>],
    trees: &[KdTree],
    gate: f64,
) -> Result<Option<usize>> {
    let mut choices = trees[frame]
        .within_radius(predicted, gate)?
        .into_iter()
        .filter(|(index, _)| !used.contains(&(frame, frames[frame][*index].row_index)))
        .map(|(index, distance)| (distance, -frames[frame][index].filter_snr, index))
        .collect::<Vec<_>>();
    choices.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.total_cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    Ok(choices.first().map(|choice| choice.2))
}

fn follow_seed(
    seed: &[(usize, usize)],
    frames: &[Vec<FastCandidateRecord>],
    trees: &[KdTree],
    times: &[f64],
    gate: f64,
) -> Result<Vec<(usize, usize)>> {
    let mut observations = seed.to_vec();
    let mut used = seed
        .iter()
        .map(|(frame, index)| (*frame, frames[*frame][*index].row_index))
        .collect::<HashSet<_>>();
    let first_frame = seed.first().map(|item| item.0).unwrap_or(0);
    let last_frame = seed.last().map(|item| item.0).unwrap_or(first_frame);

    let mut missed = 0;
    for frame in last_frame + 1..frames.len() {
        let (x, y) = fit_prediction(&observations, frames, times)?;
        let predicted = [x[0] + x[1] * times[frame], y[0] + y[1] * times[frame]];
        if let Some(index) = choose_nearby(frame, predicted, &used, frames, trees, gate)? {
            missed = 0;
            used.insert((frame, frames[frame][index].row_index));
            observations.push((frame, index));
        } else {
            missed += 1;
            if missed > 1 {
                break;
            }
        }
    }

    missed = 0;
    for frame in (0..first_frame).rev() {
        observations.sort_by_key(|item| item.0);
        let (x, y) = fit_prediction(&observations, frames, times)?;
        let predicted = [x[0] + x[1] * times[frame], y[0] + y[1] * times[frame]];
        if let Some(index) = choose_nearby(frame, predicted, &used, frames, trees, gate)? {
            missed = 0;
            used.insert((frame, frames[frame][index].row_index));
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

fn track_metrics(
    observations: &[(usize, usize)],
    frames: &[Vec<FastCandidateRecord>],
    times: &[f64],
    reference_interval_s: f64,
) -> Result<(f64, f64, f64, f64)> {
    let (x, y) = fit_prediction(observations, frames, times)?;
    let first = observations
        .first()
        .ok_or("fast candidate track is empty")?
        .0;
    let last = observations
        .last()
        .ok_or("fast candidate track is empty")?
        .0;
    let duration = times[last] - times[first];
    let speed_per_second = x[1].hypot(y[1]);
    let speed = speed_per_second * reference_interval_s;
    let displacement = speed_per_second * duration;
    let mut residual_square_sum = 0.0;
    let mut max_residual: f64 = 0.0;
    for (frame, index) in observations {
        let record = &frames[*frame][*index];
        let residual = (record.aligned_x - (x[0] + x[1] * times[*frame]))
            .hypot(record.aligned_y - (y[0] + y[1] * times[*frame]));
        residual_square_sum += residual * residual;
        max_residual = max_residual.max(residual);
    }
    let fit_rms = (residual_square_sum / observations.len() as f64).sqrt();
    Ok((displacement, speed, fit_rms, max_residual))
}

fn overlaps(previous: &HashSet<(usize, usize)>, candidate: &HashSet<(usize, usize)>) -> bool {
    let shared = candidate.intersection(previous).count();
    shared >= 2usize.max(candidate.len().min(previous.len()).div_ceil(2))
}

/// Associate prefiltered peak records into constant-velocity candidate tracks.
pub fn associate_fast_candidates(
    frames: &[Vec<FastCandidateRecord>],
    frame_times: Option<&[f64]>,
    options: &FastCandidateOptions,
) -> Result<Value> {
    let frame_count = frames.len();
    if frame_count < 3 {
        return Ok(json!({"tracks": []}));
    }
    if options.min_presence < 3 || options.min_presence > frame_count {
        return Err("min_presence must be within 3..frame count".into());
    }
    if options.link_radius_px <= 0.0
        || options.max_step_px <= options.link_radius_px
        || options.gate_px <= 0.0
        || options.max_fit_rms_px <= 0.0
    {
        return Err("invalid fast candidate association thresholds".into());
    }
    if frames.iter().flatten().any(|record| {
        !record.aligned_x.is_finite()
            || !record.aligned_y.is_finite()
            || !record.filter_snr.is_finite()
    }) {
        return Err("fast candidate records must be finite".into());
    }

    let times = normalized_times(frame_times, frame_count);
    let reference_interval_s = reference_interval(&times);
    let min_step_px = options
        .link_radius_px
        .mul_add(1.01, 0.0)
        .max(options.motion_min_displacement_px / (frame_count - 1).max(1) as f64);
    let min_speed_px_s = min_step_px / reference_interval_s;
    let max_speed_px_s = options.max_step_px / reference_interval_s;
    let trees = frames
        .iter()
        .map(|records| {
            KdTree::new(
                &records
                    .iter()
                    .map(|record| [record.aligned_x, record.aligned_y])
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    if frames.iter().all(Vec::is_empty) {
        return Ok(json!({"tracks": []}));
    }

    let mut candidates = Vec::new();
    for start_frame in 0..frame_count - 2 {
        if frames[start_frame].is_empty()
            || frames[start_frame + 1].is_empty()
            || frames[start_frame + 2].is_empty()
        {
            continue;
        }
        let first_elapsed_s = times[start_frame + 1] - times[start_frame];
        let second_elapsed_s = times[start_frame + 2] - times[start_frame + 1];
        if first_elapsed_s <= 0.0 || second_elapsed_s <= 0.0 {
            continue;
        }
        let min_first_step = min_speed_px_s * first_elapsed_s;
        let max_first_step = max_speed_px_s * first_elapsed_s;
        let min_second_step = min_speed_px_s * second_elapsed_s;
        let max_second_step = max_speed_px_s * second_elapsed_s;
        for (first_index, first) in frames[start_frame].iter().enumerate() {
            let mut second_neighbors = trees[start_frame + 1]
                .within_radius([first.aligned_x, first.aligned_y], max_first_step)?;
            // SciPy's batched query_ball_point sorts neighbor indexes by default.
            second_neighbors.sort_by_key(|(index, _)| *index);
            for (second_index, step) in second_neighbors {
                if step < min_first_step {
                    continue;
                }
                let second = &frames[start_frame + 1][second_index];
                let step_x = second.aligned_x - first.aligned_x;
                let step_y = second.aligned_y - first.aligned_y;
                let velocity_x = step_x / first_elapsed_s;
                let velocity_y = step_y / first_elapsed_s;
                let predicted_third = [
                    second.aligned_x + velocity_x * second_elapsed_s,
                    second.aligned_y + velocity_y * second_elapsed_s,
                ];
                let mut third_choices = Vec::new();
                for (third_index, distance) in
                    trees[start_frame + 2].within_radius(predicted_third, options.gate_px)?
                {
                    let third = &frames[start_frame + 2][third_index];
                    let step_two = (third.aligned_x - second.aligned_x)
                        .hypot(third.aligned_y - second.aligned_y);
                    if step_two < 0.55 * min_second_step || step_two > max_second_step {
                        continue;
                    }
                    third_choices.push((distance, -third.filter_snr, third_index));
                }
                third_choices.sort_by(|a, b| {
                    a.0.total_cmp(&b.0)
                        .then_with(|| a.1.total_cmp(&b.1))
                        .then_with(|| a.2.cmp(&b.2))
                });
                let Some((_, _, third_index)) = third_choices.first().copied() else {
                    continue;
                };
                let observations = follow_seed(
                    &[
                        (start_frame, first_index),
                        (start_frame + 1, second_index),
                        (start_frame + 2, third_index),
                    ],
                    frames,
                    &trees,
                    &times,
                    options.gate_px,
                )?;
                if observations.len() < options.min_presence
                    || observations
                        .windows(2)
                        .any(|pair| pair[1].0 - pair[0].0 > 2)
                {
                    continue;
                }
                let (displacement, speed, fit_rms, max_residual) =
                    track_metrics(&observations, frames, &times, reference_interval_s)?;
                if speed < min_step_px
                    || speed > options.max_step_px
                    || displacement < options.motion_min_displacement_px
                    || fit_rms > options.max_fit_rms_px
                    || max_residual > options.gate_px
                {
                    continue;
                }
                let snrs = observations
                    .iter()
                    .map(|(frame, index)| frames[*frame][*index].filter_snr)
                    .collect::<Vec<_>>();
                let median_snr = statistics::median(&snrs)?;
                candidates.push(Candidate {
                    observations,
                    speed,
                    fit_rms,
                    median_snr,
                });
            }
        }
    }

    candidates.sort_by(|a, b| {
        b.observations
            .len()
            .cmp(&a.observations.len())
            .then_with(|| a.fit_rms.total_cmp(&b.fit_rms))
            .then_with(|| b.median_snr.total_cmp(&a.median_snr))
            .then_with(|| a.speed.total_cmp(&b.speed))
    });
    let mut accepted: Vec<Candidate> = Vec::new();
    let mut accepted_keys: Vec<HashSet<(usize, usize)>> = Vec::new();
    for candidate in candidates {
        let keys = candidate
            .observations
            .iter()
            .map(|(frame, index)| (*frame, frames[*frame][*index].row_index))
            .collect::<HashSet<_>>();
        if accepted_keys
            .iter()
            .any(|previous| overlaps(previous, &keys))
        {
            continue;
        }
        accepted_keys.push(keys);
        accepted.push(candidate);
    }

    let tracks = accepted
        .into_iter()
        .map(|candidate| {
            json!({
                "observations": candidate.observations.iter().map(|(frame, index)| {
                    json!({"frame_index": frame, "row_index": frames[*frame][*index].row_index})
                }).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({"tracks": tracks}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(row_index: usize, x: f64, y: f64, snr: f64) -> FastCandidateRecord {
        FastCandidateRecord {
            row_index,
            aligned_x: x,
            aligned_y: y,
            filter_snr: snr,
        }
    }

    #[test]
    fn associates_a_linear_three_frame_candidate() {
        let frames = vec![
            vec![row(4, 10.0, 3.0, 12.0)],
            vec![row(8, 20.0, 3.0, 13.0)],
            vec![row(9, 30.0, 3.0, 14.0)],
        ];
        let result =
            associate_fast_candidates(&frames, None, &FastCandidateOptions::default()).unwrap();
        assert_eq!(result["tracks"][0]["observations"][0]["row_index"], 4);
        assert_eq!(result["tracks"][0]["observations"][2]["row_index"], 9);
    }

    #[test]
    fn elapsed_time_fit_uses_median_frame_interval() {
        let frames = vec![
            vec![row(0, 0.0, 0.0, 10.0)],
            vec![row(1, 10.0, 0.0, 10.0)],
            vec![row(2, 30.0, 0.0, 10.0)],
        ];
        let result = associate_fast_candidates(
            &frames,
            Some(&[0.0, 1.0, 3.0]),
            &FastCandidateOptions {
                link_radius_px: 1.0,
                min_presence: 3,
                motion_min_displacement_px: 2.0,
                max_step_px: 40.0,
                gate_px: 2.0,
                max_fit_rms_px: 0.1,
            },
        )
        .unwrap();
        let track = &result["tracks"][0];
        assert_eq!(track["observations"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn invalid_presence_and_velocity_controls_are_rejected() {
        let frames = vec![vec![row(0, 0.0, 0.0, 10.0)]; 3];
        let error = associate_fast_candidates(
            &frames,
            None,
            &FastCandidateOptions {
                min_presence: 2,
                ..FastCandidateOptions::default()
            },
        )
        .unwrap_err();
        assert_eq!(error, "min_presence must be within 3..frame count");

        let error = associate_fast_candidates(
            &frames,
            None,
            &FastCandidateOptions {
                gate_px: 0.0,
                ..FastCandidateOptions::default()
            },
        )
        .unwrap_err();
        assert_eq!(error, "invalid fast candidate association thresholds");
    }
}
