//! Sparse connected-component assignment between active tracks and frame sources.
use crate::{assignment, spatial::KdTree, Result};

#[derive(Debug, Clone, Copy)]
pub struct TrackPosition {
    pub track_id: i64,
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameAssignment {
    pub distance_px: f64,
    pub track_id: i64,
    pub source_index: usize,
}

/// Build radius-gated bipartite components and run a rectangular minimum-cost
/// assignment in each component. Callback costs affect selection, while the
/// returned distance always remains the geometric separation.
pub fn global_frame_assignments(
    track_positions: &[TrackPosition],
    current_points: &[[f64; 2]],
    link_radius_px: f64,
    pair_allowed: Option<&dyn Fn(i64, usize, f64) -> bool>,
    pair_cost: Option<&dyn Fn(i64, usize, f64) -> f64>,
) -> Result<Vec<FrameAssignment>> {
    if !link_radius_px.is_finite() || link_radius_px <= 0.0 {
        return Err("link_radius_px must be positive".into());
    }
    if track_positions.is_empty() || current_points.is_empty() {
        return Ok(Vec::new());
    }
    let points = KdTree::new(current_points)?;
    let positions: Vec<_> = track_positions
        .iter()
        .map(|item| [item.x, item.y])
        .collect();
    let mut neighbors_by_row = Vec::with_capacity(track_positions.len());
    for (row_index, position) in positions.iter().enumerate() {
        let mut accepted = Vec::new();
        for (source_index, _) in points.within_radius(*position, link_radius_px)? {
            let distance = (position[0] - current_points[source_index][0])
                .hypot(position[1] - current_points[source_index][1]);
            if pair_allowed.is_some_and(|allowed| {
                !allowed(track_positions[row_index].track_id, source_index, distance)
            }) {
                continue;
            }
            accepted.push(source_index);
        }
        accepted.sort_unstable();
        neighbors_by_row.push(accepted);
    }
    let mut source_to_rows = vec![Vec::new(); current_points.len()];
    for (row_index, source_indices) in neighbors_by_row.iter().enumerate() {
        for &source_index in source_indices {
            source_to_rows[source_index].push(row_index);
        }
    }
    if source_to_rows.iter().all(Vec::is_empty) {
        return Ok(Vec::new());
    }

    let mut visited_rows = vec![false; track_positions.len()];
    let mut components = Vec::new();
    for (start_row, start_neighbors) in neighbors_by_row.iter().enumerate() {
        if start_neighbors.is_empty() || visited_rows[start_row] {
            continue;
        }
        let mut component_rows = Vec::new();
        let mut component_sources = Vec::new();
        let mut seen_rows = vec![false; track_positions.len()];
        let mut seen_sources = vec![false; current_points.len()];
        let mut pending_rows = vec![start_row];
        while let Some(row_index) = pending_rows.pop() {
            if seen_rows[row_index] {
                continue;
            }
            seen_rows[row_index] = true;
            visited_rows[row_index] = true;
            component_rows.push(row_index);
            for &source_index in &neighbors_by_row[row_index] {
                if seen_sources[source_index] {
                    continue;
                }
                seen_sources[source_index] = true;
                component_sources.push(source_index);
                pending_rows.extend(
                    source_to_rows[source_index]
                        .iter()
                        .copied()
                        .filter(|linked_row| !seen_rows[*linked_row]),
                );
            }
        }
        component_rows.sort_unstable();
        component_sources.sort_unstable();
        components.push((component_rows, component_sources));
    }

    let invalid_cost = link_radius_px.max(1.0) * 1_000_000.0;
    let mut assignments = Vec::new();
    for (component_rows, component_sources) in components {
        let row_count = component_rows.len();
        let source_count = component_sources.len();
        let mut costs = vec![invalid_cost; row_count * source_count];
        for (row, &global_row) in component_rows.iter().enumerate() {
            for &source_index in &neighbors_by_row[global_row] {
                let column = component_sources
                    .binary_search(&source_index)
                    .expect("component source must contain each linked source");
                let distance = (positions[global_row][0] - current_points[source_index][0])
                    .hypot(positions[global_row][1] - current_points[source_index][1]);
                let mut extra_cost = pair_cost
                    .map(|cost| cost(track_positions[global_row].track_id, source_index, distance))
                    .unwrap_or(0.0);
                if !extra_cost.is_finite() || extra_cost < 0.0 {
                    extra_cost = 0.0;
                }
                costs[row * source_count + column] =
                    distance + extra_cost + 1e-9 * (row * source_count + column) as f64;
            }
        }
        for (row, column) in assignment::linear_sum_assignment(&costs, row_count, source_count)? {
            let global_row = component_rows[row];
            let source_index = component_sources[column];
            let distance = (positions[global_row][0] - current_points[source_index][0])
                .hypot(positions[global_row][1] - current_points[source_index][1]);
            if distance > link_radius_px + 1e-7 {
                continue;
            }
            assignments.push(FrameAssignment {
                distance_px: distance,
                track_id: track_positions[global_row].track_id,
                source_index,
            });
        }
    }
    assignments.sort_by(|a, b| {
        a.distance_px
            .total_cmp(&b.distance_px)
            .then(a.track_id.cmp(&b.track_id))
            .then(a.source_index.cmp(&b.source_index))
    });
    Ok(assignments)
}
