//! Robust frame-to-frame translation estimation for the sequence pipeline.
use crate::{compatible_spatial::CompatibleKdTree, measurement::Detection, statistics, Result};

const DEFAULT_SEARCH_RADIUS_PX: f64 = 64.0;
const CONSENSUS_RADIUS_PX: f64 = 2.5;
const REFERENCE_LIMIT: usize = 1500;

fn signal_snr(source: &Detection) -> f64 {
    let value = source.flux_snr.unwrap_or(source.snr);
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn median_delta(deltas: &[[f64; 2]]) -> Result<[f64; 2]> {
    let x: Vec<_> = deltas.iter().map(|point| point[0]).collect();
    let y: Vec<_> = deltas.iter().map(|point| point[1]).collect();
    Ok([statistics::median(&x)?, statistics::median(&y)?])
}

fn residuals(points: &[[f64; 2]], center: [f64; 2]) -> Vec<f64> {
    points
        .iter()
        .map(|point| (point[0] - center[0]).hypot(point[1] - center[1]))
        .collect()
}

/// Estimate current-minus-previous shift from mutual nearest high-SNR sources.
/// The old radius argument remains the local scale; search expands independently
/// and the final answer comes from a consensus cluster in displacement space.
pub fn estimate_translation(
    previous: &[Detection],
    current: &[Detection],
    radius_px: f64,
) -> Result<[f64; 2]> {
    if !radius_px.is_finite() || radius_px <= 0.0 {
        return Err("registration radius must be positive".into());
    }
    if previous.is_empty() || current.is_empty() {
        return Ok([0.0, 0.0]);
    }
    let mut previous_top: Vec<_> = previous.iter().collect();
    let mut current_top: Vec<_> = current.iter().collect();
    previous_top.sort_by(|a, b| signal_snr(b).total_cmp(&signal_snr(a)));
    current_top.sort_by(|a, b| signal_snr(b).total_cmp(&signal_snr(a)));
    previous_top.truncate(REFERENCE_LIMIT);
    current_top.truncate(REFERENCE_LIMIT);

    let previous_points: Vec<_> = previous_top.iter().map(|s| [s.x, s.y]).collect();
    let current_points: Vec<_> = current_top.iter().map(|s| [s.x, s.y]).collect();
    let search_radius = radius_px.max(DEFAULT_SEARCH_RADIUS_PX.min(32.0f64.max(8.0 * radius_px)));
    let current_tree = CompatibleKdTree::new(&current_points)?;
    let previous_tree = CompatibleKdTree::new(&previous_points)?;
    let previous_queries: Vec<_> = previous_points
        .iter()
        .map(|point| current_tree.nearest(*point))
        .collect();
    let current_queries: Vec<_> = current_points
        .iter()
        .map(|point| previous_tree.nearest(*point))
        .collect();
    let mut deltas = Vec::new();
    for (previous_index, query) in previous_queries.iter().enumerate() {
        let Some((current_index, distance)) = query else {
            continue;
        };
        // cKDTree's bounded nearest query excludes the exact upper-bound edge.
        if *distance >= search_radius {
            continue;
        }
        if current_queries[*current_index].is_some_and(|(reciprocal, reverse_distance)| {
            reciprocal == previous_index && reverse_distance < search_radius
        }) {
            deltas.push([
                current_points[*current_index][0] - previous_points[previous_index][0],
                current_points[*current_index][1] - previous_points[previous_index][1],
            ]);
        }
    }
    if deltas.is_empty() {
        return Ok([0.0, 0.0]);
    }
    if deltas.len() >= 3 {
        let counts: Vec<_> = deltas
            .iter()
            .map(|center| {
                deltas
                    .iter()
                    .filter(|point| {
                        (point[0] - center[0]).hypot(point[1] - center[1]) <= CONSENSUS_RADIUS_PX
                    })
                    .count()
            })
            .collect();
        let mut seed_index = 0usize;
        for index in 1..counts.len() {
            if counts[index] > counts[seed_index] {
                seed_index = index;
            }
        }
        let consensus: Vec<_> = deltas
            .iter()
            .filter(|point| {
                (point[0] - deltas[seed_index][0]).hypot(point[1] - deltas[seed_index][1])
                    <= CONSENSUS_RADIUS_PX
            })
            .copied()
            .collect();
        if consensus.len() >= 3 {
            let center = median_delta(&consensus)?;
            let refined: Vec<_> = deltas
                .iter()
                .filter(|point| {
                    (point[0] - center[0]).hypot(point[1] - center[1]) <= CONSENSUS_RADIUS_PX
                })
                .copied()
                .collect();
            return if refined.len() >= 3 {
                median_delta(&refined)
            } else {
                Ok(center)
            };
        }
    }

    let center = median_delta(&deltas)?;
    let distances = residuals(&deltas, center);
    let median_distance = statistics::median(&distances)?;
    let deviations: Vec<_> = distances
        .iter()
        .map(|value| (value - median_distance).abs())
        .collect();
    let mad = statistics::median(&deviations)?;
    let cutoff = 1.0f64.max(3.0 * 1.4826 * mad);
    let inliers: Vec<_> = deltas
        .iter()
        .zip(distances)
        .filter_map(|(point, distance)| (distance <= cutoff).then_some(*point))
        .collect();
    if inliers.len() < 3 {
        Ok(center)
    } else {
        median_delta(&inliers)
    }
}
