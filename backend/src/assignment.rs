//! Native shortest augmenting paths and sparse catalog assignment.
//! Tie traversal follows SciPy's rectangular_lsap.cpp (PM Larsen).
//! See THIRD_PARTY_NOTICES.md for the upstream BSD license.
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Return the minimum-cost rectangular one-to-one assignment in row order.
pub fn linear_sum_assignment(
    cost: &[f64],
    rows: usize,
    columns: usize,
) -> Result<Vec<(usize, usize)>> {
    if rows.checked_mul(columns) != Some(cost.len())
        || cost.iter().any(|v| v.is_nan() || *v == f64::NEG_INFINITY)
    {
        return Err("invalid assignment cost matrix".into());
    }
    if rows == 0 || columns == 0 {
        return Ok(Vec::new());
    }
    let transposed = rows > columns;
    let (nr, nc) = if transposed {
        (columns, rows)
    } else {
        (rows, columns)
    };
    let at = |i: usize, j: usize| {
        if transposed {
            cost[j * columns + i]
        } else {
            cost[i * columns + j]
        }
    };
    let mut u = vec![0.; nr];
    let mut v = vec![0.; nc];
    let mut row_for_column = vec![None; nc];
    let mut column_for_row = vec![None; nr];
    for current in 0..nr {
        let mut remaining: Vec<_> = (0..nc).rev().collect();
        let mut row_seen = vec![false; nr];
        let mut column_seen = vec![false; nc];
        let mut shortest = vec![f64::INFINITY; nc];
        let mut path = vec![0; nc];
        let mut i = current;
        let mut minimum = 0.;
        let sink = loop {
            row_seen[i] = true;
            let mut index = None;
            let mut lowest = f64::INFINITY;
            for (position, &j) in remaining.iter().enumerate() {
                let reduced = minimum + at(i, j) - u[i] - v[j];
                if reduced < shortest[j] {
                    shortest[j] = reduced;
                    path[j] = i;
                }
                if shortest[j] < lowest || (shortest[j] == lowest && row_for_column[j].is_none()) {
                    lowest = shortest[j];
                    index = Some(position);
                }
            }
            minimum = lowest;
            if !minimum.is_finite() {
                return Err("assignment cost matrix is infeasible".into());
            }
            let position = index.ok_or("assignment has no augmenting path")?;
            let j = remaining.swap_remove(position);
            column_seen[j] = true;
            if let Some(row) = row_for_column[j] {
                i = row;
            } else {
                break j;
            }
        };
        u[current] += minimum;
        for row in 0..nr {
            if row_seen[row] && row != current {
                u[row] += minimum - shortest[column_for_row[row].unwrap()];
            }
        }
        for j in 0..nc {
            if column_seen[j] {
                v[j] -= minimum - shortest[j];
            }
        }
        let mut j = sink;
        loop {
            let row = path[j];
            row_for_column[j] = Some(row);
            let previous = column_for_row[row].replace(j);
            if row == current {
                break;
            }
            j = previous.unwrap();
        }
    }
    let mut result: Vec<_> = column_for_row
        .into_iter()
        .enumerate()
        .map(|(i, j)| {
            let j = j.unwrap();
            if transposed {
                (j, i)
            } else {
                (i, j)
            }
        })
        .collect();
    result.sort_unstable();
    Ok(result)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CandidateEdge(pub f64, pub usize, pub usize);
fn order(a: &CandidateEdge, b: &CandidateEdge) -> std::cmp::Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2))
}
/// Original catalog matching cost and connected-component policy.
/// This preserves the baseline objective even where its finite dummy-node
/// penalty differs from strict lexicographic maximum-cardinality matching.
pub fn assign_candidates(
    candidates: &[CandidateEdge],
    detection_count: usize,
    catalog_count: usize,
    radius: f64,
    mode: &str,
) -> Result<Vec<CandidateEdge>> {
    if !radius.is_finite() || radius <= 0. || !["global", "greedy"].contains(&mode) {
        return Err("invalid matching radius or assignment mode".into());
    }
    if candidates
        .iter()
        .any(|e| !e.0.is_finite() || e.0 < 0. || e.1 >= detection_count || e.2 >= catalog_count)
    {
        return Err("invalid candidate edge".into());
    }
    let mut result = Vec::new();
    if mode == "greedy" {
        let mut edges = candidates.to_vec();
        edges.sort_by(order);
        let mut detections = BTreeSet::new();
        let mut catalogs = BTreeSet::new();
        for edge in edges {
            if !detections.contains(&edge.1) && !catalogs.contains(&edge.2) {
                detections.insert(edge.1);
                catalogs.insert(edge.2);
                result.push(edge);
            }
        }
        return Ok(result);
    }
    let node_count = detection_count
        .checked_add(catalog_count)
        .ok_or("node count overflow")?;
    let mut parent: Vec<_> = (0..node_count).collect();
    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    for edge in candidates {
        let a = root(&mut parent, edge.1);
        let b = root(&mut parent, detection_count + edge.2);
        parent[b] = a;
    }
    let mut components: BTreeMap<usize, Vec<CandidateEdge>> = BTreeMap::new();
    for edge in candidates {
        components
            .entry(root(&mut parent, edge.1))
            .or_default()
            .push(*edge);
    }
    let penalty = radius + radius.max(1.);
    for edges in components.values() {
        let detections: Vec<_> = edges
            .iter()
            .map(|e| e.1)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let catalogs: Vec<_> = edges
            .iter()
            .map(|e| e.2)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let d = detections.len();
        let c = catalogs.len();
        let size = d + c;
        let mut cost = vec![4. * penalty; size.checked_mul(size).ok_or("cost matrix overflow")?];
        for i in 0..size {
            for j in 0..size {
                if i >= d && j >= c {
                    cost[i * size + j] = 0.;
                } else if i >= d || j >= c {
                    cost[i * size + j] = penalty;
                }
            }
        }
        let mut residuals = BTreeMap::new();
        for edge in edges {
            let value = residuals.entry((edge.1, edge.2)).or_insert(f64::INFINITY);
            *value = value.min(edge.0);
            let i = detections.binary_search(&edge.1).unwrap();
            let j = catalogs.binary_search(&edge.2).unwrap();
            cost[i * size + j] = *value;
        }
        for (i, j) in linear_sum_assignment(&cost, size, size)? {
            if i < d && j < c {
                if let Some(&residual) = residuals.get(&(detections[i], catalogs[j])) {
                    result.push(CandidateEdge(residual, detections[i], catalogs[j]));
                }
            }
        }
    }
    result.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)).then(a.0.total_cmp(&b.0)));
    Ok(result)
}
