//! Robust relative photometric fitting for multi-frame observation tables.
//!
//! This operation fits only relative frame zero points, source magnitudes, and
//! optional spatial nuisance terms. It does not perform catalogue or absolute
//! magnitude calibration.
use crate::{source_geometry_audit, Result};
use nalgebra::{DMatrix, DVector};
use serde_json::{json, Map, Value};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
struct Identifier {
    key: String,
    display: String,
    type_name: &'static str,
    repr: String,
}

#[derive(Clone, Debug)]
struct Observation {
    input_index: usize,
    frame: Identifier,
    source: Identifier,
    magnitude: f64,
    error: f64,
    x: f64,
    y: f64,
}

struct FitOptions {
    spatial_order: usize,
    validation_fraction: f64,
    random_state: u128,
    mad_threshold: f64,
    max_iterations: usize,
    min_sources: usize,
    min_frames: usize,
}

struct WlsSolution {
    beta: Vec<f64>,
    residuals: Vec<f64>,
    covariance: Option<DMatrix<f64>>,
    parameter_count: usize,
    condition_number: f64,
}

struct FittedModel {
    frame_zero_points: Vec<(Identifier, f64)>,
    source_magnitudes: Vec<(Identifier, f64)>,
    frame_uncertainties: Vec<(Identifier, Option<f64>)>,
    source_uncertainties: Vec<(Identifier, Option<f64>)>,
    spatial_coefficients: Vec<f64>,
    inlier_local_indices: Vec<usize>,
    residuals: Vec<f64>,
    rejected_input_indices: Vec<usize>,
    flags: Vec<String>,
}

struct FitAttempt {
    model: Option<FittedModel>,
    status: String,
    flags: Vec<String>,
}

fn identifier(value: &Value) -> Option<Identifier> {
    let (key, display, type_name, repr) = match value {
        Value::Null => ("none".into(), "None".into(), "NoneType", "None".into()),
        Value::Bool(value) => {
            let display = if *value { "True" } else { "False" }.to_owned();
            (format!("bool:{value}"), display.clone(), "bool", display)
        }
        Value::Number(value) => {
            let display = value.to_string();
            let is_integer = value.is_i64() || value.is_u64();
            if is_integer {
                (format!("int:{display}"), display.clone(), "int", display)
            } else {
                (
                    format!("float:{display}"),
                    display.clone(),
                    "float",
                    display,
                )
            }
        }
        Value::String(value) => (
            format!("str:{}", serde_json::to_string(value).ok()?),
            value.clone(),
            "str",
            format!("{value:?}"),
        ),
        Value::Array(_) | Value::Object(_) => return None,
    };
    Some(Identifier {
        key,
        display,
        type_name,
        repr,
    })
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse::<f64>().ok(),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

fn field<'a>(row: &'a Value, name: &str, position: usize) -> Option<&'a Value> {
    match row {
        Value::Object(map) => map.get(name),
        Value::Array(items) => items.get(position),
        _ => None,
    }
}

fn optional_variable(row: &Value) -> Option<&Value> {
    match row {
        Value::Object(map) => {
            if let Some(value) = map.get("is_variable_optional") {
                Some(value)
            } else {
                map.get("is_variable").or_else(|| map.get("variable"))
            }
        }
        Value::Array(items) => items.get(8),
        _ => None,
    }
}

fn coerce_observation(
    row: &Value,
    input_index: usize,
    require_coordinates: bool,
) -> std::result::Result<Observation, &'static str> {
    let Some(frame_value) = field(row, "frame_id", 0) else {
        return Err("INVALID_OBSERVATION");
    };
    let Some(source_value) = field(row, "source_id", 1) else {
        return Err("INVALID_OBSERVATION");
    };
    let Some(quality_value) = field(row, "quality_passed", 6) else {
        return Err("INVALID_OBSERVATION");
    };
    let Some(moving_value) = field(row, "is_moving", 7) else {
        return Err("INVALID_OBSERVATION");
    };
    let Some(magnitude_value) = field(row, "instrumental_magnitude", 2) else {
        return Err("INVALID_OBSERVATION");
    };
    let Some(error_value) = field(row, "magnitude_error", 3) else {
        return Err("INVALID_OBSERVATION");
    };
    let Some(x_value) = field(row, "x", 4) else {
        return Err("INVALID_OBSERVATION");
    };
    let Some(y_value) = field(row, "y", 5) else {
        return Err("INVALID_OBSERVATION");
    };
    let frame = identifier(frame_value).ok_or("INVALID_OBSERVATION")?;
    let source = identifier(source_value).ok_or("INVALID_OBSERVATION")?;
    let magnitude = number(magnitude_value).ok_or("INVALID_OBSERVATION")?;
    let error = number(error_value).ok_or("INVALID_OBSERVATION")?;
    let x = number(x_value).ok_or("INVALID_OBSERVATION")?;
    let y = number(y_value).ok_or("INVALID_OBSERVATION")?;
    if !truthy(quality_value) {
        return Err("QUALITY_REJECTED");
    }
    if truthy(moving_value) {
        return Err("MOVING_REJECTED");
    }
    if optional_variable(row).is_some_and(truthy) {
        return Err("VARIABLE_REJECTED");
    }
    if !magnitude.is_finite() || !error.is_finite() || error <= 0.0 {
        return Err("INVALID_MAGNITUDE_ERROR");
    }
    if require_coordinates && (!x.is_finite() || !y.is_finite()) {
        return Err("INVALID_COORDINATES");
    }
    Ok(Observation {
        input_index,
        frame,
        source,
        magnitude,
        error,
        x: if x.is_finite() { x } else { 0.0 },
        y: if y.is_finite() { y } else { 0.0 },
    })
}

fn unique_ids(records: &[Observation], frame: bool) -> Vec<Identifier> {
    let mut ids = Vec::new();
    let mut seen = HashMap::<String, ()>::new();
    for row in records {
        let id = if frame { &row.frame } else { &row.source };
        if seen.insert(id.key.clone(), ()).is_none() {
            ids.push(id.clone());
        }
    }
    ids
}

fn stable_ids(ids: &[Identifier]) -> Vec<Identifier> {
    let mut ordered = ids.to_vec();
    ordered.sort_by(|a, b| {
        a.type_name
            .cmp(b.type_name)
            .then_with(|| a.repr.cmp(&b.repr))
    });
    ordered
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    Some(if ordered.len().is_multiple_of(2) {
        (ordered[middle - 1] + ordered[middle]) * 0.5
    } else {
        ordered[middle]
    })
}

fn dedupe_flags(flags: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = HashMap::new();
    let mut output = Vec::new();
    for flag in flags {
        if seen.insert(flag.clone(), ()).is_none() {
            output.push(flag);
        }
    }
    output
}

fn spatial_feature_count(order: usize) -> usize {
    match order {
        0 => 0,
        1 => 2,
        _ => 5,
    }
}

fn spatial_normalization(records: &[Observation]) -> ((f64, f64), (f64, f64)) {
    let count = records.len() as f64;
    let center_x = records.iter().map(|row| row.x).sum::<f64>() / count;
    let center_y = records.iter().map(|row| row.y).sum::<f64>() / count;
    let scale_x = (records
        .iter()
        .map(|row| (row.x - center_x).powi(2))
        .sum::<f64>()
        / count)
        .sqrt();
    let scale_y = (records
        .iter()
        .map(|row| (row.y - center_y).powi(2))
        .sum::<f64>()
        / count)
        .sqrt();
    (
        (center_x, center_y),
        (
            if scale_x > f64::EPSILON { scale_x } else { 1.0 },
            if scale_y > f64::EPSILON { scale_y } else { 1.0 },
        ),
    )
}

fn spatial_terms(
    row: &Observation,
    order: usize,
    center: (f64, f64),
    scale: (f64, f64),
) -> Vec<f64> {
    if order == 0 {
        return Vec::new();
    }
    let x = (row.x - center.0) / scale.0;
    let y = (row.y - center.1) / scale.1;
    if order == 1 {
        vec![x, y]
    } else {
        vec![x, y, x * x, x * y, y * y]
    }
}

// The design matrix and its lookup tables are produced and consumed as one fit input.
#[allow(clippy::type_complexity)]
fn build_design(
    records: &[Observation],
    source_ids: &[Identifier],
    frame_ids: &[Identifier],
    spatial_order: usize,
    center: (f64, f64),
    scale: (f64, f64),
) -> (
    DMatrix<f64>,
    DVector<f64>,
    DVector<f64>,
    HashMap<String, usize>,
    HashMap<String, usize>,
    usize,
) {
    let source_index: HashMap<_, _> = source_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.key.clone(), i))
        .collect();
    let reference_frame_key = &frame_ids[0].key;
    let mut frame_index = HashMap::new();
    let mut next_column = source_ids.len();
    for id in frame_ids {
        if &id.key != reference_frame_key {
            frame_index.insert(id.key.clone(), next_column);
            next_column += 1;
        }
    }
    let spatial_start = next_column;
    let parameter_count = spatial_start + spatial_feature_count(spatial_order);
    let mut design = DMatrix::zeros(records.len(), parameter_count);
    let mut magnitudes = DVector::zeros(records.len());
    let mut errors = DVector::zeros(records.len());
    for (row_index, row) in records.iter().enumerate() {
        design[(row_index, source_index[&row.source.key])] = 1.0;
        if let Some(column) = frame_index.get(&row.frame.key) {
            design[(row_index, *column)] = 1.0;
        }
        for (offset, value) in spatial_terms(row, spatial_order, center, scale)
            .into_iter()
            .enumerate()
        {
            design[(row_index, spatial_start + offset)] = value;
        }
        magnitudes[row_index] = row.magnitude;
        errors[row_index] = row.error;
    }
    (
        design,
        magnitudes,
        errors,
        source_index,
        frame_index,
        spatial_start,
    )
}

fn weighted_least_squares(
    design: &DMatrix<f64>,
    magnitudes: &DVector<f64>,
    errors: &DVector<f64>,
) -> std::result::Result<WlsSolution, String> {
    let rows = design.nrows();
    let parameters = design.ncols();
    if rows < parameters || rows == 0 || parameters == 0 {
        return Err("INSUFFICIENT_SAMPLES".into());
    }
    let weights: Vec<_> = errors.iter().map(|error| 1.0 / (error * error)).collect();
    let maximum_weight = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum_weight.is_finite() || maximum_weight <= 0.0 {
        return Err("INVALID_MAGNITUDE_ERROR".into());
    }
    let sqrt_weights: Vec<_> = weights
        .iter()
        .map(|weight| (weight / maximum_weight).sqrt())
        .collect();
    let weighted_design =
        DMatrix::from_fn(rows, parameters, |i, j| design[(i, j)] * sqrt_weights[i]);
    let weighted_values = DVector::from_fn(rows, |i, _| magnitudes[i] * sqrt_weights[i]);
    let svd = weighted_design.clone().svd(true, true);
    let singular_values = svd.singular_values.clone();
    if singular_values.is_empty() || singular_values.iter().any(|value| !value.is_finite()) {
        return Err("DEGENERATE_DESIGN".into());
    }
    let tolerance = (rows.max(parameters) as f64) * f64::EPSILON * singular_values[0];
    let rank = singular_values
        .iter()
        .filter(|value| **value > tolerance)
        .count();
    if rank < parameters {
        return Err("DEGENERATE_DESIGN".into());
    }
    let beta_vector = svd
        .solve(&weighted_values, tolerance)
        .map_err(|_| "DEGENERATE_DESIGN".to_owned())?;
    let residual_vector = magnitudes - design * &beta_vector;
    let degrees_of_freedom = rows.saturating_sub(rank);
    let covariance = if degrees_of_freedom > 0 {
        let standardized_sum = residual_vector
            .iter()
            .zip(errors.iter())
            .map(|(residual, error)| (residual / error).powi(2))
            .sum::<f64>();
        let variance_scale = standardized_sum / degrees_of_freedom as f64;
        let normal = weighted_design.transpose() * &weighted_design;
        normal
            .clone()
            .svd(true, true)
            .pseudo_inverse(1e-15)
            .ok()
            .map(|inverse| inverse * (variance_scale / maximum_weight))
    } else {
        None
    };
    let smallest = singular_values[singular_values.len() - 1];
    let condition_number = if smallest > 0.0 {
        singular_values[0] / smallest
    } else {
        f64::INFINITY
    };
    Ok(WlsSolution {
        beta: beta_vector.iter().copied().collect(),
        residuals: residual_vector.iter().copied().collect(),
        covariance,
        parameter_count: parameters,
        condition_number,
    })
}

fn mad(values: &[f64]) -> Option<f64> {
    let center = median(values)?;
    let deviations: Vec<_> = values.iter().map(|value| (value - center).abs()).collect();
    median(&deviations).filter(|value| value.is_finite())
}

fn uncertainty(gradient: &[f64], covariance: Option<&DMatrix<f64>>) -> Option<f64> {
    let covariance = covariance?;
    let vector = DVector::from_column_slice(gradient);
    let variance = (vector.transpose() * covariance * vector)[(0, 0)];
    variance.is_finite().then(|| variance.max(0.0).sqrt())
}

fn fit_model(
    records: &[Observation],
    spatial_order: usize,
    mad_threshold: f64,
    max_iterations: usize,
) -> FitAttempt {
    let source_ids = unique_ids(records, false);
    let frame_ids = unique_ids(records, true);
    if records.is_empty() || source_ids.is_empty() || frame_ids.is_empty() {
        return FitAttempt {
            model: None,
            status: "NO_VALID_REFERENCES".into(),
            flags: vec!["NO_VALID_REFERENCES".into()],
        };
    }
    let (center, scale) = spatial_normalization(records);
    let mut active = (0..records.len()).collect::<Vec<_>>();
    let mut rejected_input_indices = Vec::new();
    let mut flags = Vec::<String>::new();
    let mut has_successful_fit = false;
    for iteration in 0..max_iterations {
        let active_records: Vec<_> = active.iter().map(|index| records[*index].clone()).collect();
        let (design, magnitudes, errors, _, _, _) = build_design(
            &active_records,
            &source_ids,
            &frame_ids,
            spatial_order,
            center,
            scale,
        );
        let solution = match weighted_least_squares(&design, &magnitudes, &errors) {
            Ok(solution) => solution,
            Err(status) => {
                if !has_successful_fit {
                    return FitAttempt {
                        model: None,
                        status: status.clone(),
                        flags: vec![status],
                    };
                }
                flags.push("MAD_REJECTION_STOPPED_BY_RANK".into());
                break;
            }
        };
        has_successful_fit = true;
        let residual_median = median(&solution.residuals).unwrap_or(0.0);
        let raw_mad = mad(&solution.residuals).unwrap_or(0.0);
        let error_values: Vec<_> = errors.iter().copied().collect();
        let error_floor = median(&error_values).unwrap_or(0.0);
        let robust_scale = (1.4826 * raw_mad).max(error_floor).max(1e-9);
        let mut outlier_positions: Vec<_> = solution
            .residuals
            .iter()
            .enumerate()
            .filter(|(_, residual)| {
                (**residual - residual_median).abs() > mad_threshold * robust_scale
            })
            .map(|(index, _)| index)
            .collect();
        if outlier_positions.is_empty() {
            flags.push("MAD_REJECTION_CONVERGED".into());
            break;
        }
        let mut candidate_indices: Vec<_> = active
            .iter()
            .enumerate()
            .filter(|(position, _)| !outlier_positions.contains(position))
            .map(|(_, index)| *index)
            .collect();
        let mut candidate_ok = false;
        if candidate_indices.len() >= solution.parameter_count {
            let candidate_records: Vec<_> = candidate_indices
                .iter()
                .map(|index| records[*index].clone())
                .collect();
            let (candidate_design, candidate_values, candidate_errors, _, _, _) = build_design(
                &candidate_records,
                &source_ids,
                &frame_ids,
                spatial_order,
                center,
                scale,
            );
            candidate_ok =
                weighted_least_squares(&candidate_design, &candidate_values, &candidate_errors)
                    .is_ok();
        }
        if !candidate_ok {
            outlier_positions.sort_by(|a, b| {
                let da = (solution.residuals[*a] - residual_median).abs();
                let db = (solution.residuals[*b] - residual_median).abs();
                db.total_cmp(&da).then_with(|| a.cmp(b))
            });
            candidate_ok = false;
            for position in outlier_positions.iter().copied() {
                let trial_indices: Vec<_> = active
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != position)
                    .map(|(_, index)| *index)
                    .collect();
                if trial_indices.len() < solution.parameter_count {
                    continue;
                }
                let trial_records: Vec<_> = trial_indices
                    .iter()
                    .map(|index| records[*index].clone())
                    .collect();
                let (trial_design, trial_values, trial_errors, _, _, _) = build_design(
                    &trial_records,
                    &source_ids,
                    &frame_ids,
                    spatial_order,
                    center,
                    scale,
                );
                if weighted_least_squares(&trial_design, &trial_values, &trial_errors).is_ok() {
                    candidate_indices = trial_indices;
                    outlier_positions = vec![position];
                    flags.push("MAD_REJECTION_SINGLE_STEP".into());
                    candidate_ok = true;
                    break;
                }
            }
        }
        if !candidate_ok {
            flags.push("MAD_REJECTION_STOPPED_BY_RANK".into());
            break;
        }
        for position in &outlier_positions {
            rejected_input_indices.push(records[active[*position]].input_index);
        }
        active = candidate_indices;
        flags.push("MAD_REJECTED_OBSERVATIONS".into());
        if iteration + 1 == max_iterations {
            flags.push("MAD_REJECTION_MAX_ITERATIONS".into());
        }
    }

    if active.is_empty() {
        return FitAttempt {
            model: None,
            status: "DEGENERATE_DESIGN".into(),
            flags: vec!["DEGENERATE_DESIGN".into()],
        };
    }
    let final_records: Vec<_> = active.iter().map(|index| records[*index].clone()).collect();
    let (design, magnitudes, errors, source_index, frame_index, spatial_start) = build_design(
        &final_records,
        &source_ids,
        &frame_ids,
        spatial_order,
        center,
        scale,
    );
    let final_solution = match weighted_least_squares(&design, &magnitudes, &errors) {
        Ok(solution) => solution,
        Err(status) => {
            return FitAttempt {
                model: None,
                status: status.clone(),
                flags: vec![status],
            }
        }
    };
    let reference_frame = &frame_ids[0];
    let parameter_count = final_solution.parameter_count;
    let mut raw_frame_values = HashMap::<String, f64>::new();
    let mut frame_gradients = HashMap::<String, Vec<f64>>::new();
    raw_frame_values.insert(reference_frame.key.clone(), 0.0);
    frame_gradients.insert(reference_frame.key.clone(), vec![0.0; parameter_count]);
    for frame in &frame_ids {
        if let Some(column) = frame_index.get(&frame.key) {
            raw_frame_values.insert(frame.key.clone(), final_solution.beta[*column]);
            let mut gradient = vec![0.0; parameter_count];
            gradient[*column] = 1.0;
            frame_gradients.insert(frame.key.clone(), gradient);
        }
    }
    let mut order: Vec<_> = frame_ids.iter().enumerate().collect();
    order.sort_by(|(ia, a), (ib, b)| {
        raw_frame_values[&a.key]
            .total_cmp(&raw_frame_values[&b.key])
            .then_with(|| ia.cmp(ib))
    });
    let mid = order.len() / 2;
    let median_positions: Vec<usize> = if order.len().is_multiple_of(2) {
        vec![mid - 1, mid]
    } else {
        vec![mid]
    };
    let median_shift = median_positions
        .iter()
        .map(|position| raw_frame_values[&order[*position].1.key])
        .sum::<f64>()
        / median_positions.len() as f64;
    let mut shift_gradient = vec![0.0; parameter_count];
    for position in &median_positions {
        let gradient = &frame_gradients[&order[*position].1.key];
        for (index, value) in gradient.iter().enumerate() {
            shift_gradient[index] += value / median_positions.len() as f64;
        }
    }
    let frame_zero_points: Vec<_> = frame_ids
        .iter()
        .map(|id| (id.clone(), raw_frame_values[&id.key] - median_shift))
        .collect();
    let source_magnitudes: Vec<_> = source_ids
        .iter()
        .map(|id| {
            (
                id.clone(),
                final_solution.beta[source_index[&id.key]] + median_shift,
            )
        })
        .collect();
    let frame_uncertainties = frame_ids
        .iter()
        .map(|id| {
            let mut gradient = frame_gradients[&id.key].clone();
            for (value, shift) in gradient.iter_mut().zip(&shift_gradient) {
                *value -= shift;
            }
            (
                id.clone(),
                uncertainty(&gradient, final_solution.covariance.as_ref()),
            )
        })
        .collect();
    let source_uncertainties = source_ids
        .iter()
        .map(|id| {
            let mut gradient = shift_gradient.clone();
            gradient[source_index[&id.key]] += 1.0;
            (
                id.clone(),
                uncertainty(&gradient, final_solution.covariance.as_ref()),
            )
        })
        .collect();
    if final_solution.covariance.is_some() {
        flags.push("UNCERTAINTY_ESTIMATED".into());
    } else {
        flags.push("UNCERTAINTY_UNAVAILABLE_NO_DOF".into());
    }
    if final_solution.condition_number > 1e12 {
        flags.push("ILL_CONDITIONED_DESIGN".into());
    }
    flags.push("MEDIAN_ZERO_POINT_FIXED".into());
    if spatial_order > 0 {
        flags.push("SPATIAL_TERMS_ENABLED".into());
    }
    let spatial_coefficients = final_solution.beta[spatial_start..].to_vec();
    let model = FittedModel {
        frame_zero_points,
        source_magnitudes,
        frame_uncertainties,
        source_uncertainties,
        spatial_coefficients,
        inlier_local_indices: active,
        residuals: final_solution.residuals,
        rejected_input_indices,
        flags: dedupe_flags(flags),
    };
    FitAttempt {
        flags: model.flags.clone(),
        model: Some(model),
        status: "VALID".into(),
    }
}

fn split_source_block(
    ids: &[Identifier],
    fraction: f64,
    seed: u128,
    minimum_training: usize,
) -> Vec<Identifier> {
    if fraction <= 0.0 || ids.len() <= minimum_training {
        return Vec::new();
    }
    let requested = ((ids.len() as f64 * fraction).round_ties_even() as usize).max(1);
    let count = requested.min(ids.len().saturating_sub(minimum_training));
    if count == 0 {
        return Vec::new();
    }
    let ordered = stable_ids(ids);
    let permutation = source_geometry_audit::numpy_default_rng_permutation(seed, ordered.len());
    let mut selected: Vec<_> = permutation
        .into_iter()
        .take(count)
        .map(|index| ordered[index].clone())
        .collect();
    selected.sort_by(|a, b| {
        a.type_name
            .cmp(b.type_name)
            .then_with(|| a.repr.cmp(&b.repr))
    });
    selected
}

fn source_block_validation(
    records: &[Observation],
    model: &FittedModel,
    spatial_order: usize,
    center: (f64, f64),
    scale: (f64, f64),
) -> (Option<f64>, Option<f64>, usize, Vec<String>) {
    if records.is_empty() {
        return (None, None, 0, vec!["NO_VALIDATION_SAMPLES".into()]);
    }
    let frame_values: HashMap<_, _> = model
        .frame_zero_points
        .iter()
        .map(|(id, value)| (id.key.clone(), *value))
        .collect();
    let coefficients = &model.spatial_coefficients;
    let mut values_by_source = Vec::<(Identifier, Vec<(f64, f64)>)>::new();
    let mut source_positions = HashMap::<String, usize>::new();
    let mut flags = Vec::new();
    for record in records {
        let Some(frame_offset) = frame_values.get(&record.frame.key) else {
            flags.push("VALIDATION_FRAME_NOT_IN_TRAINING".into());
            continue;
        };
        let spatial = spatial_terms(record, spatial_order, center, scale);
        let predicted: f64 = spatial.iter().zip(coefficients).map(|(a, b)| a * b).sum();
        let position = *source_positions
            .entry(record.source.key.clone())
            .or_insert_with(|| {
                values_by_source.push((record.source.clone(), Vec::new()));
                values_by_source.len() - 1
            });
        values_by_source[position]
            .1
            .push((record.magnitude - frame_offset - predicted, record.error));
    }
    let mut residuals = Vec::new();
    for (_, values) in &values_by_source {
        let weights: Vec<_> = values
            .iter()
            .map(|(_, error)| 1.0 / (error * error))
            .collect();
        let denominator = weights.iter().sum::<f64>();
        if denominator <= 0.0 || !denominator.is_finite() {
            continue;
        }
        let source_mean = values
            .iter()
            .zip(&weights)
            .map(|((value, _), weight)| value * weight)
            .sum::<f64>()
            / denominator;
        residuals.extend(values.iter().map(|(value, _)| value - source_mean));
    }
    if residuals.is_empty() {
        flags.push("NO_VALIDATION_SAMPLES".into());
        return (None, None, 0, dedupe_flags(flags));
    }
    let rms =
        (residuals.iter().map(|value| value * value).sum::<f64>() / residuals.len() as f64).sqrt();
    (
        Some(rms),
        mad(&residuals),
        residuals.len(),
        dedupe_flags(flags),
    )
}

fn mapping(entries: &[(Identifier, Value)]) -> Value {
    let mut sorted = entries.to_vec();
    sorted.sort_by(|a, b| a.0.display.cmp(&b.0.display));
    let mut result = Map::new();
    for (id, value) in sorted {
        result.insert(id.display, value);
    }
    Value::Object(result)
}

// Keep the serialization boundary explicit: every field mirrors the public result schema.
#[allow(clippy::too_many_arguments)]
fn result_json(
    status: &str,
    flags: &[String],
    frame_zero_points: &[(Identifier, f64)],
    source_magnitudes: &[(Identifier, f64)],
    frame_uncertainties: &[(Identifier, Option<f64>)],
    source_uncertainties: &[(Identifier, Option<f64>)],
    reference_count: usize,
    frame_counts: &[(Identifier, usize)],
    source_counts: &[(Identifier, usize)],
    train_rms: Option<f64>,
    train_mad: Option<f64>,
    validation_rms: Option<f64>,
    validation_mad: Option<f64>,
    validation_count: usize,
    validation_ids: &[Identifier],
    rejected: &[usize],
    excluded: usize,
    spatial_coefficients: &[f64],
    spatial_order: usize,
) -> Value {
    let float_map = |values: &[(Identifier, f64)]| {
        mapping(
            &values
                .iter()
                .map(|(id, value)| (id.clone(), json!(value)))
                .collect::<Vec<_>>(),
        )
    };
    let optional_map = |values: &[(Identifier, Option<f64>)]| {
        mapping(
            &values
                .iter()
                .map(|(id, value)| (id.clone(), value.map_or(Value::Null, |v| json!(v))))
                .collect::<Vec<_>>(),
        )
    };
    let count_map = |values: &[(Identifier, usize)]| {
        mapping(
            &values
                .iter()
                .map(|(id, value)| (id.clone(), json!(value)))
                .collect::<Vec<_>>(),
        )
    };
    json!({
        "status": status,
        "flags": flags,
        "frame_zero_points": float_map(frame_zero_points),
        "relative_magnitudes": float_map(source_magnitudes),
        "frame_zero_point_uncertainties": optional_map(frame_uncertainties),
        "relative_magnitude_uncertainties": optional_map(source_uncertainties),
        "reference_sample_count": reference_count,
        "reference_sample_count_by_frame": count_map(frame_counts),
        "reference_sample_count_by_source": count_map(source_counts),
        "training_residual_rms": train_rms,
        "training_residual_mad": train_mad,
        "validation_residual_rms": validation_rms,
        "validation_residual_mad": validation_mad,
        "validation_sample_count": validation_count,
        "validation_source_ids": validation_ids.iter().map(|id| id.display.clone()).collect::<Vec<_>>(),
        "rejected_observation_indices": rejected,
        "excluded_observation_count": excluded,
        "spatial_coefficients": spatial_coefficients,
        "spatial_order": spatial_order,
    })
}

fn empty_json(
    status: &str,
    flags: &[String],
    excluded: usize,
    spatial_order: usize,
    validation_ids: &[Identifier],
) -> Value {
    result_json(
        status,
        flags,
        &[],
        &[],
        &[],
        &[],
        0,
        &[],
        &[],
        None,
        None,
        None,
        None,
        0,
        validation_ids,
        &[],
        excluded,
        &[],
        spatial_order,
    )
}

fn selected_observations(payload: &Value) -> &Value {
    if let Value::Object(map) = payload {
        for key in ["observations", "relative_observations", "rows", "items"] {
            if let Some(value) = map.get(key) {
                return value;
            }
        }
    }
    payload
}

fn observation_rows(value: &Value) -> Result<Vec<Value>> {
    match value {
        Value::Array(rows) => Ok(rows.clone()),
        Value::Object(map) => Ok(map.keys().cloned().map(Value::String).collect()),
        Value::String(text) => Ok(text
            .chars()
            .map(|character| Value::String(character.to_string()))
            .collect()),
        _ => Err("observations must be iterable JSON data".into()),
    }
}

fn parse_options(value: &Value) -> Result<FitOptions> {
    let get = |key: &str| value.get(key);
    let mut spatial_order = get("spatial_order").and_then(Value::as_i64).unwrap_or(0);
    if let Some(alias) = get("spatial_terms").filter(|value| !value.is_null()) {
        let requested = if alias.is_boolean() {
            if truthy(alias) {
                1
            } else {
                0
            }
        } else if let Some(number) = alias.as_i64() {
            number
        } else if let Some(text) = alias.as_str() {
            text.parse::<i64>()
                .map_err(|_| "spatial_order must be 0, 1, or 2")?
        } else {
            return Err("spatial_order must be 0, 1, or 2".into());
        };
        if spatial_order != 0 && spatial_order != requested {
            return Err("spatial_order and spatial_terms disagree".into());
        }
        spatial_order = requested;
    }
    if let Some(alias) = get("include_spatial_terms").filter(|value| !value.is_null()) {
        let requested = if truthy(alias) { 1 } else { 0 };
        if spatial_order != 0 && spatial_order != requested {
            return Err("spatial_order and include_spatial_terms disagree".into());
        }
        spatial_order = requested;
    }
    if !(0..=2).contains(&spatial_order) {
        return Err("spatial_order must be 0, 1, or 2".into());
    }
    let validation_fraction = get("validation_fraction").and_then(number).unwrap_or(0.2);
    if !validation_fraction.is_finite() || !(0.0..1.0).contains(&validation_fraction) {
        return Err("validation_fraction must be in [0, 1)".into());
    }
    let mad_threshold = get("mad_threshold").and_then(number).unwrap_or(5.0);
    if !mad_threshold.is_finite() || mad_threshold <= 0.0 {
        return Err("mad_threshold must be positive".into());
    }
    let max_iterations = get("max_iterations").and_then(Value::as_u64).unwrap_or(8) as usize;
    if max_iterations < 1 {
        return Err("max_iterations must be at least 1".into());
    }
    let min_sources = get("min_sources").and_then(Value::as_u64).unwrap_or(2) as usize;
    let min_frames = get("min_frames").and_then(Value::as_u64).unwrap_or(2) as usize;
    if min_sources < 1 || min_frames < 1 {
        return Err("min_sources and min_frames must be positive".into());
    }
    let random_state = match get("random_state") {
        None | Some(Value::Null) => 0,
        Some(Value::Number(number)) => number
            .as_u64()
            .ok_or("random_state must be a non-negative integer")?
            as u128,
        Some(Value::String(text)) => text
            .parse::<u128>()
            .map_err(|_| "random_state must be a non-negative integer")?,
        _ => return Err("random_state must be a non-negative integer".into()),
    };
    Ok(FitOptions {
        spatial_order: spatial_order as usize,
        validation_fraction,
        random_state,
        mad_threshold,
        max_iterations,
        min_sources,
        min_frames,
    })
}

/// Fit a relative photometric scale from an observation array and option object.
pub fn fit(observations: &Value, options: &Value) -> Result<Value> {
    let opts = parse_options(options)?;
    let observations = selected_observations(observations);
    let rows = observation_rows(observations)?;
    let mut prepared = Vec::new();
    let mut rejected = HashMap::<&'static str, usize>::new();
    for (index, row) in rows.iter().enumerate() {
        match coerce_observation(row, index, opts.spatial_order > 0) {
            Ok(row) => prepared.push(row),
            Err(reason) => *rejected.entry(reason).or_default() += 1,
        }
    }
    let mut global_flags = Vec::new();
    for (reason, flag) in [
        ("QUALITY_REJECTED", "QUALITY_FAILED_EXCLUDED"),
        ("MOVING_REJECTED", "MOVING_EXCLUDED"),
        ("VARIABLE_REJECTED", "VARIABLE_EXCLUDED"),
        ("INVALID_OBSERVATION", "INVALID_OBSERVATION_EXCLUDED"),
        (
            "INVALID_MAGNITUDE_ERROR",
            "INVALID_MAGNITUDE_ERROR_EXCLUDED",
        ),
        ("INVALID_COORDINATES", "INVALID_COORDINATES_EXCLUDED"),
    ] {
        if rejected.get(reason).copied().unwrap_or(0) > 0 {
            global_flags.push(flag.to_owned());
        }
    }
    let excluded_count = rows.len() - prepared.len();
    if prepared.is_empty() {
        global_flags.push("NO_VALID_REFERENCES".into());
        return Ok(empty_json(
            "NO_VALID_REFERENCES",
            &dedupe_flags(global_flags),
            excluded_count,
            opts.spatial_order,
            &[],
        ));
    }
    let source_ids = unique_ids(&prepared, false);
    let frame_ids = unique_ids(&prepared, true);
    if source_ids.len() < opts.min_sources || frame_ids.len() < opts.min_frames {
        global_flags.push("INSUFFICIENT_SOURCES_OR_FRAMES".into());
        return Ok(empty_json(
            "INSUFFICIENT_SAMPLES",
            &dedupe_flags(global_flags),
            excluded_count,
            opts.spatial_order,
            &[],
        ));
    }
    let validation_ids = split_source_block(
        &source_ids,
        opts.validation_fraction,
        opts.random_state,
        opts.min_sources.max(2),
    );
    let validation_keys: HashMap<_, _> = validation_ids
        .iter()
        .map(|id| (id.key.clone(), ()))
        .collect();
    let training: Vec<_> = prepared
        .iter()
        .filter(|row| !validation_keys.contains_key(&row.source.key))
        .cloned()
        .collect();
    let validation: Vec<_> = prepared
        .iter()
        .filter(|row| validation_keys.contains_key(&row.source.key))
        .cloned()
        .collect();
    if validation_ids.is_empty() {
        global_flags.push("NO_HOLDOUT_VALIDATION".into());
    } else {
        global_flags.push("HOLDOUT_SOURCE_BLOCK".into());
    }
    let final_attempt = fit_model(
        &prepared,
        opts.spatial_order,
        opts.mad_threshold,
        opts.max_iterations,
    );
    let Some(model) = final_attempt.model else {
        global_flags.extend(final_attempt.flags);
        return Ok(empty_json(
            &final_attempt.status,
            &dedupe_flags(global_flags),
            excluded_count,
            opts.spatial_order,
            &validation_ids,
        ));
    };
    global_flags.extend(model.flags.iter().cloned());
    if !model.rejected_input_indices.is_empty() {
        global_flags.push("MAD_REJECTION_APPLIED".into());
    }
    let training_rms = if model.residuals.is_empty() {
        None
    } else {
        Some(
            (model.residuals.iter().map(|v| v * v).sum::<f64>() / model.residuals.len() as f64)
                .sqrt(),
        )
    };
    let training_mad = mad(&model.residuals);
    let mut frame_counts = HashMap::<String, (Identifier, usize)>::new();
    let mut source_counts = HashMap::<String, (Identifier, usize)>::new();
    for local_index in &model.inlier_local_indices {
        let row = &prepared[*local_index];
        frame_counts
            .entry(row.frame.key.clone())
            .or_insert_with(|| (row.frame.clone(), 0))
            .1 += 1;
        source_counts
            .entry(row.source.key.clone())
            .or_insert_with(|| (row.source.clone(), 0))
            .1 += 1;
    }
    let mut validation_rms = None;
    let mut validation_mad = None;
    let mut validation_count = 0;
    if !validation.is_empty() {
        let training_attempt = fit_model(
            &training,
            opts.spatial_order,
            opts.mad_threshold,
            opts.max_iterations,
        );
        if let Some(training_model) = training_attempt.model {
            let (center, scale) = spatial_normalization(&training);
            let (rms, residual_mad, count, flags) = source_block_validation(
                &validation,
                &training_model,
                opts.spatial_order,
                center,
                scale,
            );
            validation_rms = rms;
            validation_mad = residual_mad;
            validation_count = count;
            global_flags.extend(flags);
        } else {
            global_flags.push("VALIDATION_MODEL_UNAVAILABLE".into());
            global_flags.extend(training_attempt.flags);
        }
    }
    let mut rejected_input_indices = model.rejected_input_indices.clone();
    let mut seen_rejected = HashMap::new();
    rejected_input_indices.retain(|index| seen_rejected.insert(*index, ()).is_none());
    Ok(result_json(
        "VALID",
        &dedupe_flags(global_flags),
        &model.frame_zero_points,
        &model.source_magnitudes,
        &model.frame_uncertainties,
        &model.source_uncertainties,
        model.inlier_local_indices.len(),
        &frame_counts.into_values().collect::<Vec<_>>(),
        &source_counts.into_values().collect::<Vec<_>>(),
        training_rms,
        training_mad,
        validation_rms,
        validation_mad,
        validation_count,
        &validation_ids,
        &rejected_input_indices,
        excluded_count,
        &model.spatial_coefficients,
        opts.spatial_order,
    ))
}

/// JSONL operation wrapper: `photometry.relative.fit`.
pub fn handle_request(request: &Value) -> Result<Value> {
    let observations = request
        .get("observations")
        .or_else(|| request.get("relative_observations"))
        .or_else(|| request.get("rows"))
        .or_else(|| request.get("items"))
        .unwrap_or(request);
    let options = request.get("options").unwrap_or(request);
    fit(observations, options)
}

fn read_input(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    serde_json::from_slice(&bytes).map_err(|error| error.to_string())
}

fn parse_cli(args: &[String]) -> Result<(PathBuf, Option<PathBuf>, Value)> {
    let mut input = None;
    let mut output = None;
    let mut options = Map::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let mut take_value = |name: &str| -> Result<String> {
            if let Some((_, value)) = arg.split_once('=') {
                return Ok(value.to_owned());
            }
            index += 1;
            args.get(index)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        if arg == "--help" || arg == "-h" {
            return Err("__HELP__".into());
        }
        if arg == "--out" || arg.starts_with("--out=") {
            output = Some(PathBuf::from(take_value("--out")?));
        } else if arg == "--spatial-order" || arg.starts_with("--spatial-order=") {
            let value = take_value("--spatial-order")?
                .parse::<i64>()
                .map_err(|_| "spatial_order must be 0, 1, or 2")?;
            options.insert("spatial_order".into(), json!(value));
        } else if arg == "--validation-fraction" || arg.starts_with("--validation-fraction=") {
            let value = take_value("--validation-fraction")?
                .parse::<f64>()
                .map_err(|_| "validation_fraction must be in [0, 1)")?;
            options.insert("validation_fraction".into(), json!(value));
        } else if arg == "--random-state" || arg.starts_with("--random-state=") {
            let value = take_value("--random-state")?
                .parse::<u64>()
                .map_err(|_| "random_state must be a non-negative integer")?;
            options.insert("random_state".into(), json!(value));
        } else if arg == "--mad-threshold" || arg.starts_with("--mad-threshold=") {
            let value = take_value("--mad-threshold")?
                .parse::<f64>()
                .map_err(|_| "mad_threshold must be positive")?;
            options.insert("mad_threshold".into(), json!(value));
        } else if arg.starts_with('-') {
            return Err(format!("unrecognized argument: {arg}"));
        } else if input.replace(PathBuf::from(arg)).is_some() {
            return Err("too many input paths".into());
        }
        index += 1;
    }
    let input = input.ok_or("missing input JSON path")?;
    Ok((input, output, Value::Object(options)))
}

const HELP: &str = "Usage: rst19-relative-photometry INPUT [--out PATH] [--spatial-order 0|1|2] [--validation-fraction VALUE] [--random-state INTEGER] [--mad-threshold VALUE]\n";

/// Standalone CLI entry point; returns the Python command's 0/2 exit contract.
pub fn run_cli(args: &[String]) -> i32 {
    let (input, output, options) = match parse_cli(args) {
        Ok(value) => value,
        Err(error) if error == "__HELP__" => {
            print!("{HELP}");
            return 0;
        }
        Err(error) => {
            eprintln!("relative-photometry: {error}");
            return 2;
        }
    };
    let payload = match read_input(&input) {
        Ok(payload) => payload,
        Err(error) => {
            eprintln!("relative-photometry: {error}");
            return 2;
        }
    };
    let result = match fit(&payload, &options) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("relative-photometry: {error}");
            return 2;
        }
    };
    let rendered = match serde_json::to_string_pretty(&result) {
        Ok(value) => format!("{value}\n"),
        Err(error) => {
            eprintln!("relative-photometry: {error}");
            return 2;
        }
    };
    if let Some(output) = output {
        if let Some(parent) = output.parent() {
            if let Err(error) = fs::create_dir_all(parent) {
                eprintln!("relative-photometry: {error}");
                return 2;
            }
        }
        if let Err(error) = fs::write(output, rendered) {
            eprintln!("relative-photometry: {error}");
            return 2;
        }
    } else {
        print!("{rendered}");
    }
    0
}

/// Desktop tool route using the same argument surface as the standalone CLI.
pub fn run_desktop(args: &[String], out_dir: &Path) -> Result<Value> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let (input, output, options) = parse_cli(args)?;
    let payload = read_input(&input)?;
    let result = fit(&payload, &options)?;
    let result_path = output.unwrap_or_else(|| out_dir.join("relative_photometry.json"));
    if let Some(parent) = result_path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(
        &result_path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&result).map_err(|error| error.to_string())?
        ),
    )
    .map_err(|error| error.to_string())?;
    Ok(
        json!({"tool":"rst19-relative-photometry","result":result,"result_path":result_path,"out_dir":out_dir}),
    )
}
