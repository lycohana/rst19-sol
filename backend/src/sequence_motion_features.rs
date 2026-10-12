//! Per-frame connected-component proposals for the sequence motion-feature path.
//!
//! This module preserves the first geometric proposal stage from Python's
//! `detect_motion_features`. It returns image-plane candidates and audit counts
//! only; model review and cross-frame motion gates remain separate operations.
use crate::{sequence_trails::line_profile_metrics_for_plane, Result};
use serde_json::{json, Value};

const SHORT_TRAIL_LENGTH: f64 = 16.0;
const SHORT_TRAIL_MIN_AXIS_RATIO: f64 = 2.5;

fn next_down_positive(value: f64) -> f64 {
    if value > 0.0 && value.is_finite() {
        f64::from_bits(value.to_bits() - 1)
    } else {
        value
    }
}

#[derive(Clone, Copy)]
struct Options {
    frame_index: i64,
    shift_x: f64,
    shift_y: f64,
    threshold: f64,
    residual_noise: f64,
    psf_fwhm: f64,
    min_length: f64,
    min_feature_area: i64,
    min_axis_ratio: f64,
    bridge_gaps: bool,
    short_min_pixels: usize,
}

fn field_number(request: &Value, key: &str) -> Result<f64> {
    request
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric {key}"))
}

fn field_usize(request: &Value, key: &str) -> Result<usize> {
    request
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("missing non-negative integer {key}"))
}

fn field_i64(request: &Value, key: &str) -> Result<i64> {
    request
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("missing integer {key}"))
}

fn matrix_f32(request: &Value, key: &str) -> Result<Vec<Vec<f32>>> {
    let rows = request
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing matrix {key}"))?;
    if rows.is_empty() {
        return Err(format!("{key} must not be empty"));
    }
    let mut result = Vec::with_capacity(rows.len());
    let mut width = None;
    for row in rows {
        let values = row
            .as_array()
            .ok_or_else(|| format!("{key} rows must be arrays"))?;
        if values.is_empty() {
            return Err(format!("{key} rows must not be empty"));
        }
        if width.is_some_and(|expected| expected != values.len()) {
            return Err(format!("{key} must be rectangular"));
        }
        width = Some(values.len());
        result.push(
            values
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .map(|number| number as f32)
                        .ok_or_else(|| format!("{key} values must be numeric"))
                })
                .collect::<Result<Vec<_>>>()?,
        );
    }
    Ok(result)
}

fn matrix_bool(request: &Value, key: &str) -> Result<Vec<Vec<bool>>> {
    let rows = request
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing matrix {key}"))?;
    if rows.is_empty() {
        return Err(format!("{key} must not be empty"));
    }
    let mut result = Vec::with_capacity(rows.len());
    let mut width = None;
    for row in rows {
        let values = row
            .as_array()
            .ok_or_else(|| format!("{key} rows must be arrays"))?;
        if values.is_empty() {
            return Err(format!("{key} rows must not be empty"));
        }
        if width.is_some_and(|expected| expected != values.len()) {
            return Err(format!("{key} must be rectangular"));
        }
        width = Some(values.len());
        result.push(
            values
                .iter()
                .map(|value| {
                    value
                        .as_bool()
                        .ok_or_else(|| format!("{key} values must be booleans"))
                })
                .collect::<Result<Vec<_>>>()?,
        );
    }
    Ok(result)
}

fn options(request: &Value) -> Result<Options> {
    let frame_index = request
        .get("frame_index")
        .and_then(Value::as_i64)
        .ok_or_else(|| "missing integer frame_index".to_string())?;
    let shift = request
        .get("shift")
        .and_then(Value::as_array)
        .filter(|values| values.len() == 2)
        .ok_or_else(|| "shift must contain [x, y]".to_string())?;
    let shift_x = shift[0]
        .as_f64()
        .ok_or_else(|| "shift x must be numeric".to_string())?;
    let shift_y = shift[1]
        .as_f64()
        .ok_or_else(|| "shift y must be numeric".to_string())?;
    let bridge_gaps = request
        .get("bridge_gaps")
        .and_then(Value::as_bool)
        .ok_or_else(|| "missing boolean bridge_gaps".to_string())?;
    Ok(Options {
        frame_index,
        shift_x,
        shift_y,
        threshold: field_number(request, "threshold_adu")?,
        residual_noise: field_number(request, "residual_noise_adu")?,
        psf_fwhm: field_number(request, "psf_fwhm")?,
        min_length: field_number(request, "min_length_px")?,
        min_feature_area: field_i64(request, "min_feature_area")?,
        min_axis_ratio: field_number(request, "min_axis_ratio")?,
        bridge_gaps,
        short_min_pixels: field_usize(request, "short_min_pixels")?,
    })
}

fn validate_shapes(
    residual: &[Vec<f32>],
    valid: &[Vec<bool>],
    artifact: &[Vec<bool>],
) -> Result<()> {
    let height = residual.len();
    let width = residual.first().map_or(0, Vec::len);
    if valid.len() != height || artifact.len() != height {
        return Err("residual, valid_mask, and line_artifact_mask shapes must match".into());
    }
    if valid.iter().chain(artifact).any(|row| row.len() != width) {
        return Err("residual, valid_mask, and line_artifact_mask shapes must match".into());
    }
    Ok(())
}

fn label_components(geometry_support: &[Vec<bool>]) -> (Vec<Vec<usize>>, usize) {
    let height = geometry_support.len();
    let width = geometry_support.first().map_or(0, Vec::len);
    let mut labels = vec![vec![0usize; width]; height];
    let mut component_count = 0usize;
    let mut stack = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if !geometry_support[y][x] || labels[y][x] != 0 {
                continue;
            }
            component_count += 1;
            labels[y][x] = component_count;
            stack.push((x, y));
            while let Some((current_x, current_y)) = stack.pop() {
                let min_y = current_y.saturating_sub(1);
                let max_y = (current_y + 1).min(height - 1);
                let min_x = current_x.saturating_sub(1);
                let max_x = (current_x + 1).min(width - 1);
                for neighbor_y in min_y..=max_y {
                    for neighbor_x in min_x..=max_x {
                        if geometry_support[neighbor_y][neighbor_x]
                            && labels[neighbor_y][neighbor_x] == 0
                        {
                            labels[neighbor_y][neighbor_x] = component_count;
                            stack.push((neighbor_x, neighbor_y));
                        }
                    }
                }
            }
        }
    }
    (labels, component_count)
}

fn motion_profile_point(
    residual: &[Vec<f32>],
    valid: &[Vec<bool>],
    line_artifact_mask: &[Vec<bool>],
    component_pixels: &[(usize, usize)],
    size: usize,
    options: Options,
) -> Result<(Option<Value>, ComponentOutcome)> {
    let height = residual.len();
    let width = residual[0].len();
    let mut raw_x = Vec::with_capacity(size);
    let mut raw_y = Vec::with_capacity(size);
    let mut component_values = Vec::with_capacity(size);
    let mut artifact_count = 0usize;
    for &(x, y) in component_pixels {
        if valid[y][x] && f64::from(residual[y][x]) >= options.threshold {
            raw_x.push(x as f64);
            raw_y.push(y as f64);
            component_values.push(residual[y][x]);
            artifact_count += usize::from(line_artifact_mask[y][x]);
        }
    }
    if raw_x.len() < 3 {
        return Ok((None, ComponentOutcome::default()));
    }
    let count = raw_x.len() as f64;
    let mean_x = raw_x.iter().sum::<f64>() / count;
    let mean_y = raw_y.iter().sum::<f64>() / count;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    let mut covariance_xy = 0.0;
    for (&x, &y) in raw_x.iter().zip(&raw_y) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        variance_x += dx * dx;
        variance_y += dy * dy;
        covariance_xy += dx * dy;
    }
    variance_x /= count;
    variance_y /= count;
    covariance_xy /= count;
    let discriminant = (variance_x - variance_y).hypot(2.0 * covariance_xy);
    let mut major_variance = (0.5 * (variance_x + variance_y + discriminant)).max(0.0);
    let determinant = (variance_x * variance_y - covariance_xy * covariance_xy).max(0.0);
    let mut minor_variance = if major_variance > 0.0 {
        (determinant / major_variance).max(0.0)
    } else {
        0.0
    };
    // NumPy's LAPACK `eigh` returns the next representable value below an
    // exactly repeated positive eigenvalue for this 2x2 diagonal case. Keep
    // the same strict length-gate behavior at exact pixel-scale boundaries.
    if discriminant == 0.0 && major_variance > 0.0 {
        major_variance = next_down_positive(major_variance);
        minor_variance = next_down_positive(minor_variance);
    }
    let length_px = 4.0 * major_variance.sqrt();
    let width_px = 4.0 * minor_variance.sqrt();
    let axis_ratio = (major_variance / minor_variance.max(f64::EPSILON)).sqrt();
    let axis_ratio_gate = if length_px < SHORT_TRAIL_LENGTH {
        options.min_axis_ratio.min(SHORT_TRAIL_MIN_AXIS_RATIO)
    } else {
        options.min_axis_ratio
    };
    if length_px < options.min_length {
        return Ok((
            None,
            ComponentOutcome {
                length_rejected: 1,
                ..ComponentOutcome::default()
            },
        ));
    }
    if axis_ratio < axis_ratio_gate {
        return Ok((
            None,
            ComponentOutcome {
                axis_ratio_rejected: 1,
                ..ComponentOutcome::default()
            },
        ));
    }
    let min_pixels = options
        .min_feature_area
        .max((4.0 * options.psf_fwhm * options.psf_fwhm).ceil() as i64)
        as usize;
    if length_px >= SHORT_TRAIL_LENGTH && size < min_pixels {
        return Ok((
            None,
            ComponentOutcome {
                large_area_rejected: 1,
                ..ComponentOutcome::default()
            },
        ));
    }
    let x_min = *raw_x.iter().min_by(|a, b| a.total_cmp(b)).unwrap() as usize;
    let x_max = *raw_x.iter().max_by(|a, b| a.total_cmp(b)).unwrap() as usize;
    let y_min = *raw_y.iter().min_by(|a, b| a.total_cmp(b)).unwrap() as usize;
    let y_max = *raw_y.iter().max_by(|a, b| a.total_cmp(b)).unwrap() as usize;
    let touches_edge = x_min == 0 || y_min == 0 || x_max + 1 == width || y_max + 1 == height;
    if touches_edge && length_px < options.min_length.max(8.0 * options.psf_fwhm) {
        return Ok((
            None,
            ComponentOutcome {
                edge_rejected: 1,
                geometry_pass: 1,
                ..ComponentOutcome::default()
            },
        ));
    }
    let mut weight_sum = 0.0;
    let mut weighted_x = 0.0;
    let mut weighted_y = 0.0;
    for ((&x, &y), &value) in raw_x.iter().zip(&raw_y).zip(&component_values) {
        let weight = f64::from(value.max(0.0));
        weight_sum += weight;
        weighted_x += weight * x;
        weighted_y += weight * y;
    }
    let (center_x, center_y) = if weight_sum > 0.0 {
        (weighted_x / weight_sum, weighted_y / weight_sum)
    } else {
        (mean_x, mean_y)
    };
    // PCA axes are unoriented: this half-angle form is the same major axis as
    // the symmetric 2x2 eigensolver, with a deterministic [-90, 90] encoding.
    let mut angle_deg = (0.5 * (2.0 * covariance_xy).atan2(variance_x - variance_y)).to_degrees();
    if angle_deg >= 90.0 {
        angle_deg -= 180.0;
    } else if angle_deg < -90.0 {
        angle_deg += 180.0;
    }
    let residual_peak = component_values
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .unwrap_or(f32::NEG_INFINITY);
    let (continuity_ratio, profile_fill_ratio) = line_profile_metrics_for_plane(
        residual,
        center_x,
        center_y,
        length_px,
        width_px,
        angle_deg,
        options.threshold,
    );
    let point = json!({
        "frame_index": options.frame_index,
        "x": center_x,
        "y": center_y,
        "aligned_x": center_x - options.shift_x,
        "aligned_y": center_y - options.shift_y,
        "residual_snr": f64::from(residual_peak) / options.residual_noise,
        "area_pixels": size,
        "length_px": length_px,
        "width_px": width_px,
        "angle_deg": angle_deg,
        "bbox": [x_min, y_min, x_max, y_max],
        "touches_edge": touches_edge,
        "continuity_ratio": continuity_ratio,
        "profile_fill_ratio": profile_fill_ratio,
        "line_artifact_overlap_fraction": artifact_count as f64 / raw_x.len() as f64,
        "line_template_snr": null,
        "line_vs_point_delta_bic": null,
        "line_vs_double_delta_bic": null,
        "line_vs_background_delta_bic": null,
        "line_vs_elliptical_delta_bic": null,
        "single_frame_model": null,
        "single_frame_model_margin_bic": null,
        "observation_source": "detected"
    });
    Ok((
        Some(point),
        ComponentOutcome {
            geometry_pass: 1,
            ..ComponentOutcome::default()
        },
    ))
}

#[derive(Clone, Copy, Default)]
struct ComponentOutcome {
    length_rejected: usize,
    axis_ratio_rejected: usize,
    large_area_rejected: usize,
    edge_rejected: usize,
    geometry_pass: usize,
}

/// Extract one frame's connected line candidates from residual support pixels.
pub fn motion_feature_frame_components(request: &Value) -> Result<Value> {
    let residual = matrix_f32(request, "residual")?;
    let valid = matrix_bool(request, "valid_mask")?;
    let line_artifact_mask = matrix_bool(request, "line_artifact_mask")?;
    validate_shapes(&residual, &valid, &line_artifact_mask)?;
    let options = options(request)?;
    if !options.threshold.is_finite()
        || !options.residual_noise.is_finite()
        || options.residual_noise <= 0.0
        || !options.psf_fwhm.is_finite()
        || options.psf_fwhm <= 0.0
        || !options.min_length.is_finite()
        || options.min_length <= 0.0
        || !options.min_axis_ratio.is_finite()
        || !options.shift_x.is_finite()
        || !options.shift_y.is_finite()
        || options.short_min_pixels == 0
    {
        return Err("motion feature component thresholds must be finite and positive".into());
    }
    let height = residual.len();
    let width = residual[0].len();
    let mut support = vec![vec![false; width]; height];
    let mut valid_pixel_count = 0usize;
    let mut support_pixel_count = 0usize;
    for y in 0..height {
        for x in 0..width {
            valid_pixel_count += usize::from(valid[y][x]);
            support[y][x] = valid[y][x] && f64::from(residual[y][x]) >= options.threshold;
            support_pixel_count += usize::from(support[y][x]);
        }
    }
    let geometry_support = if options.bridge_gaps {
        let mut dilated = vec![vec![false; width]; height];
        for (y, row) in support.iter().enumerate() {
            for (x, is_support) in row.iter().copied().enumerate() {
                if !is_support {
                    continue;
                }
                for target_row in dilated
                    .iter_mut()
                    .take((y + 1).min(height - 1) + 1)
                    .skip(y.saturating_sub(1))
                {
                    for target in target_row
                        .iter_mut()
                        .take((x + 1).min(width - 1) + 1)
                        .skip(x.saturating_sub(1))
                    {
                        *target = true;
                    }
                }
            }
        }
        dilated
    } else {
        support.clone()
    };
    let (labels, component_count) = label_components(&geometry_support);
    let mut sizes = vec![0usize; component_count];
    let mut component_pixels = vec![Vec::new(); component_count];
    for (y, label_row) in labels.iter().enumerate() {
        for (x, label) in label_row.iter().copied().enumerate() {
            if label != 0 {
                let component = label - 1;
                component_pixels[component].push((x, y));
                sizes[component] += usize::from(support[y][x]);
            }
        }
    }
    let mut points = Vec::new();
    let mut area_pass_count = 0usize;
    let mut geometry_pass_count = 0usize;
    let mut edge_rejected_count = 0usize;
    let mut short_area_rejected_count = 0usize;
    let mut length_rejected_count = 0usize;
    let mut axis_ratio_rejected_count = 0usize;
    let mut large_area_rejected_count = 0usize;
    for (component_offset, size) in sizes.iter().copied().enumerate() {
        if size < options.short_min_pixels {
            short_area_rejected_count += 1;
            continue;
        }
        area_pass_count += 1;
        let (point, outcome) = motion_profile_point(
            &residual,
            &valid,
            &line_artifact_mask,
            &component_pixels[component_offset],
            size,
            options,
        )?;
        geometry_pass_count += outcome.geometry_pass;
        edge_rejected_count += outcome.edge_rejected;
        length_rejected_count += outcome.length_rejected;
        axis_ratio_rejected_count += outcome.axis_ratio_rejected;
        large_area_rejected_count += outcome.large_area_rejected;
        if let Some(point) = point {
            points.push(point);
        }
    }
    let max_feature_residual_snr = points
        .iter()
        .filter_map(|point| point.get("residual_snr").and_then(Value::as_f64))
        .max_by(f64::total_cmp);
    let feature_count = points.len();
    Ok(json!({
        "points": points,
        "audit": {
            "frame_index": options.frame_index,
            "threshold_adu": options.threshold,
            "residual_noise_adu": options.residual_noise,
            "valid_pixel_count": valid_pixel_count,
            "support_pixel_count": support_pixel_count,
            "component_count": component_count,
            "area_pass_count": area_pass_count,
            "geometry_pass_count": geometry_pass_count,
            "edge_rejected_count": edge_rejected_count,
            "feature_count": feature_count,
            "max_feature_residual_snr": max_feature_residual_snr,
            "short_area_rejected_count": short_area_rejected_count,
            "length_rejected_count": length_rejected_count,
            "axis_ratio_rejected_count": axis_ratio_rejected_count,
            "large_area_rejected_count": large_area_rejected_count,
            "short_min_pixels": options.short_min_pixels
        }
    }))
}
