//! Per-frame connected-component proposals for the sequence motion-feature path.
//!
//! This module preserves the first geometric proposal stage from Python's
//! `detect_motion_features`. It returns image-plane candidates and audit counts
//! only; model review and cross-frame motion gates remain separate operations.
use crate::{
    fits,
    pipeline::FrameAnalysis,
    sequence_temporal::RegisteredReference,
    sequence_trails::{self, line_profile_metrics_for_plane},
    Result,
};
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
    fit_line_model: bool,
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
        fit_line_model: request
            .get("fit_line_model")
            .and_then(Value::as_bool)
            .unwrap_or(false),
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
    let mut point = json!({
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
    if options.fit_line_model {
        let evidence = crate::single_frame_model::evaluate(
            residual,
            valid,
            center_x,
            center_y,
            length_px,
            width_px,
            angle_deg,
            options.psf_fwhm,
            options.residual_noise,
        );
        if let (Some(point), Some(fields)) = (point.as_object_mut(), evidence.as_object()) {
            point.extend(
                fields
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
        }
    }
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
    let options = options(request)?;
    motion_feature_frame_components_for_planes(&residual, &valid, &line_artifact_mask, options)
}

fn motion_feature_frame_components_for_planes(
    residual: &[Vec<f32>],
    valid: &[Vec<bool>],
    line_artifact_mask: &[Vec<bool>],
    options: Options,
) -> Result<Value> {
    validate_shapes(residual, valid, line_artifact_mask)?;
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
            residual,
            valid,
            line_artifact_mask,
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

/// Analyze one already-detected raw frame with the same connected-component
/// and local-model review stages used by the Python single-frame trail audit.
/// This typed entry point avoids expanding a full FITS image into JSON.
pub fn single_frame_components_from_analysis(
    analysis: &FrameAnalysis,
    psf_fwhm: f64,
    residual_sigma: f64,
    min_residual_adu: f64,
    min_feature_area: usize,
    min_axis_ratio: f64,
) -> Result<Value> {
    single_frame_components_from_analysis_with_limits(
        analysis,
        psf_fwhm,
        residual_sigma,
        min_residual_adu,
        min_feature_area,
        min_axis_ratio,
        8.0_f64.max(3.0 * psf_fwhm),
        20,
    )
}

/// Extract and model-review a raw frame using the metadata-aware proposal
/// floors used by the multi-frame sequence detector.
#[allow(clippy::too_many_arguments)]
pub fn sequence_single_frame_components_from_analysis(
    analysis: &FrameAnalysis,
    psf_fwhm: f64,
    residual_sigma: f64,
    min_residual_adu: f64,
    min_feature_area: usize,
    min_axis_ratio: f64,
    min_length_px: f64,
    short_min_feature_area: usize,
) -> Result<Value> {
    single_frame_components_from_analysis_with_limits(
        analysis,
        psf_fwhm,
        residual_sigma,
        min_residual_adu,
        min_feature_area,
        min_axis_ratio,
        min_length_px,
        short_min_feature_area,
    )
}

#[allow(clippy::too_many_arguments)]
fn single_frame_components_from_analysis_with_limits(
    analysis: &FrameAnalysis,
    psf_fwhm: f64,
    residual_sigma: f64,
    min_residual_adu: f64,
    min_feature_area: usize,
    min_axis_ratio: f64,
    min_length_px: f64,
    short_min_feature_area: usize,
) -> Result<Value> {
    if !psf_fwhm.is_finite()
        || psf_fwhm <= 0.0
        || !residual_sigma.is_finite()
        || residual_sigma <= 0.0
        || !min_residual_adu.is_finite()
        || min_residual_adu <= 0.0
        || min_feature_area < 3
        || !min_axis_ratio.is_finite()
        || min_axis_ratio <= 0.0
        || !min_length_px.is_finite()
        || min_length_px <= 0.0
        || short_min_feature_area < 3
    {
        return Err("single-frame trail parameters must be finite and positive".into());
    }
    let height = analysis.frame.height;
    let width = analysis.frame.width;
    if height == 0 || width == 0 || analysis.frame.data.len() != height.saturating_mul(width) {
        return Err("single-frame trail pixels do not match image shape".into());
    }
    let auxiliary = fits::auxiliary_mask(height, width)?;
    let mask_zero = enabled_parameter(&analysis.detection.parameters, "mask_zero_pixels");
    let saturation_level =
        numeric_parameter(&analysis.detection.parameters, "saturation_level", -1.0);
    let noise = analysis.detection.noise.max(f64::from(f32::EPSILON));
    let threshold = min_residual_adu.max(residual_sigma * noise);
    let background = analysis.detection.background;
    let mut residual = vec![vec![0.0_f32; width]; height];
    let mut valid = vec![vec![false; width]; height];
    let mut line_artifact_mask = vec![vec![false; width]; height];
    for &[x, y] in &analysis.detection.line_artifact_coordinates {
        if x < width && y < height {
            line_artifact_mask[y][x] = true;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let value = analysis.frame.data.value(index) as f32;
            let is_valid = value.is_finite()
                && !auxiliary[index]
                && (!mask_zero || value != 0.0)
                && (saturation_level <= 0.0 || f64::from(value) < saturation_level);
            valid[y][x] = is_valid;
            if is_valid {
                residual[y][x] = (f64::from(value) - background) as f32;
            }
        }
    }
    let short_min_pixels = short_min_feature_area.max((2.0 * psf_fwhm * psf_fwhm).ceil() as usize);
    motion_feature_frame_components_for_planes(
        &residual,
        &valid,
        &line_artifact_mask,
        Options {
            frame_index: 0,
            shift_x: 0.0,
            shift_y: 0.0,
            threshold,
            residual_noise: noise,
            psf_fwhm,
            min_length: min_length_px,
            min_feature_area: min_feature_area as i64,
            min_axis_ratio,
            bridge_gaps: false,
            short_min_pixels,
            fit_line_model: true,
        },
    )
}

fn sample_registered_reference(reference: &RegisteredReference, x: f64, y: f64) -> Option<f32> {
    if reference.height == 0
        || reference.width == 0
        || !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x > (reference.width - 1) as f64
        || y > (reference.height - 1) as f64
    {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(reference.width - 1);
    let y1 = (y0 + 1).min(reference.height - 1);
    let top_left = reference.values[y0 * reference.width + x0];
    let top_right = reference.values[y0 * reference.width + x1];
    let bottom_left = reference.values[y1 * reference.width + x0];
    let bottom_right = reference.values[y1 * reference.width + x1];
    if [top_left, top_right, bottom_left, bottom_right]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Some(f32::NAN);
    }
    let dx = x - x0 as f64;
    let dy = y - y0 as f64;
    let top = f64::from(top_left) * (1.0 - dx) + f64::from(top_right) * dx;
    let bottom = f64::from(bottom_left) * (1.0 - dx) + f64::from(bottom_right) * dx;
    Some((top * (1.0 - dy) + bottom * dy) as f32)
}

fn numeric_parameter(parameters: &Value, key: &str, default: f64) -> f64 {
    parameters
        .get(key)
        .and_then(Value::as_f64)
        .unwrap_or(default)
}

fn enabled_parameter(parameters: &Value, key: &str) -> bool {
    parameters.get(key).is_some_and(|value| {
        value
            .as_bool()
            .unwrap_or_else(|| value.as_i64().is_some_and(|number| number != 0))
    })
}

/// Run the first temporal-residual line-proposal stage directly on Rust frame
/// buffers. The JSON request adapter remains available for parity fixtures;
/// production sequence orchestration uses this typed path to avoid expanding
/// full detector images into JSON values.
pub fn motion_frame_components_from_analyses(
    analyses: &[FrameAnalysis],
    reference: Option<&RegisteredReference>,
    shifts: &[[f64; 2]],
    psf_fwhm: f64,
    residual_sigma: f64,
    min_residual_adu: f64,
    min_feature_area: usize,
    min_axis_ratio: f64,
    min_length_px: Option<f64>,
    bridge_gaps: bool,
) -> Result<Value> {
    if analyses.len() < 2 {
        return Ok(json!({
            "status":"insufficient_frames",
            "scope":"registered_temporal_residual_components_only",
            "frames":[]
        }));
    }
    if shifts.len() != analyses.len() {
        return Err("cumulative_shifts length must match frame_analyses".into());
    }
    if !psf_fwhm.is_finite()
        || psf_fwhm <= 0.0
        || !residual_sigma.is_finite()
        || residual_sigma <= 0.0
        || !min_residual_adu.is_finite()
        || min_residual_adu <= 0.0
        || !min_axis_ratio.is_finite()
        || min_axis_ratio <= 0.0
        || min_length_px.is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("motion feature component thresholds must be finite and positive".into());
    }

    let headers = analyses
        .iter()
        .map(|analysis| Value::Object(analysis.frame.header.clone()))
        .collect::<Vec<_>>();
    let declared_exposures_s = analyses
        .iter()
        .map(|analysis| {
            fits::exposure_milliseconds(&analysis.frame.header).map(|value| {
                value.and_then(|milliseconds| {
                    (milliseconds.is_finite() && milliseconds > 0.0)
                        .then_some(milliseconds / 1_000.0)
                })
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata_supported = sequence_trails::observation_time_metadata_supported(&headers)
        && declared_exposures_s.iter().all(Option::is_some);
    let proposal_thresholds = sequence_trails::motion_proposal_thresholds(&json!({
        "psf_fwhm":psf_fwhm,
        "min_length_px":min_length_px,
        "exposure_s":declared_exposures_s,
        "metadata_supported":metadata_supported,
    }))?;
    let min_length = proposal_thresholds["min_length_px"]
        .as_f64()
        .ok_or("motion proposal threshold did not return min_length_px")?;
    let reference_is_usable = reference.filter(|plane| {
        analyses[0].frame.height == plane.height
            && analyses[0].frame.width == plane.width
            && plane.values.len() == plane.height.saturating_mul(plane.width)
    });
    let difference_noise_factor = if reference_is_usable.is_some() {
        std::f64::consts::SQRT_2
    } else {
        1.0
    };

    let mut frame_results = Vec::with_capacity(analyses.len());
    let mut thresholds = Vec::with_capacity(analyses.len());
    for (frame_index, (analysis, shift)) in analyses.iter().zip(shifts).enumerate() {
        let height = analysis.frame.height;
        let width = analysis.frame.width;
        if height == 0 || width == 0 || analysis.frame.data.len() != height * width {
            return Err("motion feature frame pixels do not match image shape".into());
        }
        if !shift[0].is_finite() || !shift[1].is_finite() {
            return Err("cumulative shifts must be finite".into());
        }
        let auxiliary = fits::auxiliary_mask(height, width)?;
        let mask_zero = enabled_parameter(&analysis.detection.parameters, "mask_zero_pixels");
        let saturation_level =
            numeric_parameter(&analysis.detection.parameters, "saturation_level", -1.0);
        let background = analysis.detection.background as f32;
        let residual_noise =
            (analysis.detection.noise * difference_noise_factor).max(f64::from(f32::EPSILON));
        let threshold = min_residual_adu.max(residual_sigma * residual_noise);
        thresholds.push(threshold);

        let mut residual = vec![vec![0.0_f32; width]; height];
        let mut valid = vec![vec![false; width]; height];
        let mut line_artifact_mask = vec![vec![false; width]; height];
        for &[x, y] in &analysis.detection.line_artifact_coordinates {
            if x < width && y < height {
                line_artifact_mask[y][x] = true;
            }
        }
        let reference = reference_is_usable;
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                let value = analysis.frame.data.value(index) as f32;
                let is_valid = value.is_finite()
                    && !auxiliary[index]
                    && (!mask_zero || value != 0.0)
                    && (saturation_level <= 0.0 || f64::from(value) < saturation_level);
                valid[y][x] = is_valid;
                if !is_valid {
                    continue;
                }
                residual[y][x] = if let Some(reference) = reference {
                    let reference_value = sample_registered_reference(
                        reference,
                        x as f64 - shift[0],
                        y as f64 - shift[1],
                    );
                    reference_value
                        .filter(|sample| sample.is_finite())
                        .map_or(0.0, |sample| value - sample)
                } else {
                    value - background
                };
            }
        }

        let exposure_thresholds = sequence_trails::motion_proposal_thresholds(&json!({
            "psf_fwhm":psf_fwhm,
            "min_length_px":min_length_px,
            "exposure_s":declared_exposures_s,
            "metadata_supported":metadata_supported,
            "short_exposure_s":declared_exposures_s[frame_index],
        }))?;
        let exposure_area = exposure_thresholds["short_min_area"]
            .as_u64()
            .ok_or("motion proposal threshold did not return short_min_area")?
            as usize;
        let short_min_pixels = exposure_area.max((2.0 * psf_fwhm * psf_fwhm).ceil() as usize);
        let result = motion_feature_frame_components_for_planes(
            &residual,
            &valid,
            &line_artifact_mask,
            Options {
                frame_index: frame_index as i64,
                shift_x: shift[0],
                shift_y: shift[1],
                threshold,
                residual_noise,
                psf_fwhm,
                min_length,
                min_feature_area: min_feature_area as i64,
                min_axis_ratio,
                bridge_gaps,
                short_min_pixels,
                fit_line_model: false,
            },
        )?;
        frame_results.push(result);
    }

    Ok(json!({
        "status":"analyzed",
        "scope":"registered_temporal_residual_components_only",
        "metadata_supported":metadata_supported,
        "min_length_px":min_length,
        "thresholds_adu":thresholds,
        "frames":frame_results,
    }))
}
