//! Time-aware motion metrics shared by sequence track association.
use crate::{statistics, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const MOTION_FLOOR_MAD_FACTOR: f64 = 5.0;
const MOTION_FLOOR_MIN_TRACKS: usize = 30;

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct TrackObservation {
    pub frame_index: i64,
    pub aligned_x: f64,
    pub aligned_y: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct MotionMetric {
    pub presence: usize,
    pub displacement_px: f64,
    pub speed_px_per_frame: f64,
    pub fit_rms_px: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct TrackMotionFit {
    pub displacement_px: f64,
    pub speed_px_per_frame: f64,
    pub fit_rms_px: Option<f64>,
}
impl TrackMotionFit {
    pub fn as_dict(self) -> Value {
        json!({
            "displacement_px":self.displacement_px,
            "speed_px_per_frame":self.speed_px_per_frame,
            "fit_rms_px":self.fit_rms_px
        })
    }
}

fn resolve_times(
    points: &[TrackObservation],
    frame_times: Option<&[f64]>,
) -> (Vec<f64>, Option<f64>) {
    let fallback: Vec<_> = points
        .iter()
        .map(|point| point.frame_index as f64)
        .collect();
    let Some(values) = frame_times else {
        return (fallback, None);
    };
    if values.iter().any(|value| !value.is_finite()) {
        return (fallback, None);
    }
    let selected: Option<Vec<_>> = if values.len() == points.len() {
        Some(values.to_vec())
    } else if points.iter().all(|point| {
        let index = point.frame_index;
        if index >= 0 {
            (index as usize) < values.len()
        } else {
            index >= -(values.len() as i64)
        }
    }) {
        Some(
            points
                .iter()
                .map(|point| {
                    let index = if point.frame_index >= 0 {
                        point.frame_index as usize
                    } else {
                        values.len() - point.frame_index.unsigned_abs() as usize
                    };
                    values[index]
                })
                .collect(),
        )
    } else {
        None
    };
    let Some(selected) = selected else {
        return (fallback, None);
    };
    if selected.iter().any(|value| !value.is_finite())
        || selected.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return (fallback, None);
    }
    let interval = if selected.len() >= 2 {
        let differences: Vec<_> = selected.windows(2).map(|pair| pair[1] - pair[0]).collect();
        Some(statistics::median(&differences).unwrap_or(0.0))
    } else {
        None
    };
    (
        selected.iter().map(|value| value - selected[0]).collect(),
        interval,
    )
}

pub fn fit_track_motion(
    points: &[TrackObservation],
    frame_times: Option<&[f64]>,
) -> Result<TrackMotionFit> {
    if points.len() < 2 {
        return Ok(TrackMotionFit {
            displacement_px: 0.0,
            speed_px_per_frame: 0.0,
            fit_rms_px: None,
        });
    }
    let (time, median_observation_interval) = resolve_times(points, frame_times);
    let mean_time = time.iter().sum::<f64>() / time.len() as f64;
    let mean_x = points.iter().map(|point| point.aligned_x).sum::<f64>() / points.len() as f64;
    let mean_y = points.iter().map(|point| point.aligned_y).sum::<f64>() / points.len() as f64;
    let denominator = time
        .iter()
        .map(|value| (value - mean_time).powi(2))
        .sum::<f64>();
    let slope_x = if denominator == 0.0 {
        0.0
    } else {
        time.iter()
            .zip(points)
            .map(|(value, point)| (value - mean_time) * (point.aligned_x - mean_x))
            .sum::<f64>()
            / denominator
    };
    let slope_y = if denominator == 0.0 {
        0.0
    } else {
        time.iter()
            .zip(points)
            .map(|(value, point)| (value - mean_time) * (point.aligned_y - mean_y))
            .sum::<f64>()
            / denominator
    };
    let intercept_x = mean_x - slope_x * mean_time;
    let intercept_y = mean_y - slope_y * mean_time;
    let sum_squared_error = time
        .iter()
        .zip(points)
        .map(|(value, point)| {
            let dx = point.aligned_x - (slope_x * value + intercept_x);
            let dy = point.aligned_y - (slope_y * value + intercept_y);
            dx * dx + dy * dy
        })
        .sum::<f64>();
    let duration = time[time.len() - 1] - time[0];
    let mut speed = slope_x.hypot(slope_y);
    if let Some(interval) = median_observation_interval {
        speed *= interval;
    }
    Ok(TrackMotionFit {
        displacement_px: (slope_x * duration).hypot(slope_y * duration),
        speed_px_per_frame: speed,
        fit_rms_px: Some((sum_squared_error / points.len() as f64).sqrt()),
    })
}

pub fn derive_motion_displacement_floor(
    metrics: &[MotionMetric],
    required_presence: usize,
    user_floor: f64,
    max_motion_fit_rms_px: f64,
    affine_frame_count: usize,
) -> Result<f64> {
    if affine_frame_count == 0 {
        return Ok(user_floor);
    }
    let displacements: Vec<_> = metrics
        .iter()
        .filter(|metric| {
            metric.presence >= required_presence
                && metric
                    .fit_rms_px
                    .is_some_and(|rms| rms.is_finite() && rms <= max_motion_fit_rms_px)
                && metric.displacement_px.is_finite()
        })
        .map(|metric| metric.displacement_px)
        .collect();
    if displacements.len() < MOTION_FLOOR_MIN_TRACKS {
        return Ok(user_floor);
    }
    let median = statistics::median(&displacements)?;
    let deviations: Vec<_> = displacements
        .iter()
        .map(|value| (value - median).abs())
        .collect();
    let mad = statistics::median(&deviations)?;
    let robust_floor = median + MOTION_FLOOR_MAD_FACTOR * 1.4826 * mad;
    if !robust_floor.is_finite() {
        return Ok(user_floor);
    }
    Ok(user_floor.max(robust_floor))
}
