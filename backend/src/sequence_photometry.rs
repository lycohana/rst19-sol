//! Native forced photometry for sources proposed on registered sequence stacks.
//!
//! This is deliberately a measurement layer: a stack proposal remains a
//! candidate until the caller confirms it across original frames.
use crate::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
struct Measurement {
    flux_snr: Option<f64>,
    support: i16,
    center_valid: bool,
    fwhm: Option<f64>,
    ellipticity: Option<f64>,
}

pub type StackMeasurement = (Option<f64>, i16, bool, Option<f64>, Option<f64>);

#[derive(Clone, Copy)]
pub struct FramePixels<'a> {
    pub image: &'a [f64],
    pub mask: &'a [bool],
    pub height: usize,
    pub width: usize,
}

#[derive(Clone, Copy)]
pub struct StackMeasurementOptions {
    pub aperture_radius: usize,
    pub global_background: f64,
    pub global_noise: f64,
}

fn number(value: &Value, name: &str) -> Result<f64> {
    if value.is_null() {
        return Ok(f64::NAN);
    }
    value
        .as_f64()
        .ok_or_else(|| format!("{name} must contain numeric values or null"))
}

fn matrix(value: &Value, name: &str) -> Result<(Vec<f64>, usize, usize)> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{name} must be a two-dimensional array"))?;
    let mut width = None;
    let mut output = Vec::new();
    for row in rows {
        let row = row
            .as_array()
            .ok_or_else(|| format!("{name} must be a rectangular two-dimensional array"))?;
        if width.is_some_and(|expected| expected != row.len()) {
            return Err(format!(
                "{name} must be a rectangular two-dimensional array"
            ));
        }
        width = Some(row.len());
        for value in row {
            output.push(number(value, name)?);
        }
    }
    let height = rows.len();
    let width = width.unwrap_or(0);
    if height == 0 || width == 0 {
        return Err(format!("{name} cannot be empty"));
    }
    Ok((output, height, width))
}

fn mask_matrix(value: &Value, height: usize, width: usize) -> Result<Vec<bool>> {
    let rows = value
        .as_array()
        .ok_or_else(|| "mask must be a two-dimensional boolean array".to_string())?;
    let mut found_width = None;
    let mut output = Vec::with_capacity(height * width);
    for row in rows {
        let row = row.as_array().ok_or_else(|| {
            "mask must be a rectangular two-dimensional boolean array".to_string()
        })?;
        if found_width.is_some_and(|expected| expected != row.len()) {
            return Err("mask must be a rectangular two-dimensional boolean array".into());
        }
        found_width = Some(row.len());
        for value in row {
            if let Some(value) = value.as_bool() {
                output.push(value);
            } else if let Some(value) = value.as_u64() {
                if value > 1 {
                    return Err("mask values must be boolean or 0/1".into());
                }
                output.push(value == 1);
            } else if let Some(value) = value.as_f64() {
                if value != 0.0 && value != 1.0 {
                    return Err("mask values must be boolean or 0/1".into());
                }
                output.push(value == 1.0);
            } else {
                return Err("mask values must be boolean or 0/1".into());
            }
        }
    }
    if rows.len() != height || found_width.unwrap_or(0) != width {
        return Err(format!(
            "mask shape ({}, {}) does not match image shape ({height}, {width})",
            rows.len(),
            found_width.unwrap_or(0)
        ));
    }
    Ok(output)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn clipped_annulus(values: &[f64]) -> Option<(f64, f64, f64)> {
    let mut working = values.to_vec();
    let mut summary = None;
    for _ in 0..3 {
        if working.is_empty() {
            break;
        }
        let mut center_values = working.clone();
        let center = median(&mut center_values);
        let mut deviations: Vec<_> = working.iter().map(|value| (value - center).abs()).collect();
        let scale = 1.4826 * median(&mut deviations);
        let variance = working
            .iter()
            .map(|value| (value - center) * (value - center))
            .sum::<f64>()
            / working.len() as f64;
        summary = Some((center, scale, variance.sqrt()));
        let clipped: Vec<_> = working
            .iter()
            .copied()
            .filter(|value| (value - center).abs() <= 3.0 * scale)
            .collect();
        if clipped.len() == working.len() {
            break;
        }
        working = clipped;
    }
    summary
}

fn measure_one(
    pixels: &FramePixels<'_>,
    options: StackMeasurementOptions,
    x: f64,
    y: f64,
) -> Measurement {
    let FramePixels {
        image,
        mask,
        height,
        width,
    } = *pixels;
    let StackMeasurementOptions {
        aperture_radius,
        global_background,
        global_noise,
    } = options;
    let no_measurement = |flux_snr, support, center_valid, fwhm, ellipticity| Measurement {
        flux_snr,
        support,
        center_valid,
        fwhm,
        ellipticity,
    };
    let in_frame = x.is_finite()
        && y.is_finite()
        && x >= 0.0
        && y >= 0.0
        && x <= (width - 1) as f64
        && y <= (height - 1) as f64;
    if !in_frame {
        return no_measurement(None, 0, false, None, None);
    }

    let outer = aperture_radius + 4;
    let floor_x = x.floor() as isize;
    let floor_y = y.floor() as isize;
    let dx = x - floor_x as f64;
    let dy = y - floor_y as f64;
    let mut aperture = Vec::new();
    let mut annulus = Vec::new();
    let mut annulus_count = 0usize;
    let radius = aperture_radius as f64;
    let outer_f = outer as f64;

    for offset_y in -(outer as isize)..=(outer as isize) {
        for offset_x in -(outer as isize)..=(outer as isize) {
            let distance = ((offset_x as f64 - dx).powi(2) + (offset_y as f64 - dy).powi(2)).sqrt();
            let is_aperture = distance <= radius;
            let is_annulus = distance > radius + 1.0 && distance <= outer_f;
            if !is_aperture && !is_annulus {
                continue;
            }
            let image_x = floor_x + offset_x;
            let image_y = floor_y + offset_y;
            let inside = image_x >= 0
                && image_y >= 0
                && image_x < width as isize
                && image_y < height as isize;
            let index = if inside {
                image_y as usize * width + image_x as usize
            } else {
                0
            };
            let valid = inside && !mask[index] && image[index].is_finite();
            if is_aperture && valid {
                aperture.push((image_x as f64, image_y as f64, image[index]));
            }
            if is_annulus && valid {
                annulus_count += 1;
                annulus.push(image[index]);
            }
        }
    }

    let epsilon = f64::EPSILON;
    let (local_background, mut local_noise) = if annulus_count >= 8 {
        if let Some((background, mad, stddev)) = clipped_annulus(&annulus) {
            (background, mad.max(stddev).max(epsilon))
        } else {
            (global_background, global_noise.max(epsilon))
        }
    } else {
        (global_background, global_noise.max(epsilon))
    };
    if local_noise > 3.0 * global_noise.max(epsilon) {
        local_noise = global_noise.max(epsilon);
    }
    local_noise = local_noise.max(epsilon);

    let aperture_count = aperture.len();
    let net_flux = aperture
        .iter()
        .map(|(_, _, value)| value - local_background)
        .sum::<f64>();
    let background_pixels = annulus_count.max(1) as f64;
    let background_variance = aperture_count as f64 * local_noise.powi(2)
        + (aperture_count as f64).powi(2) / background_pixels * local_noise.powi(2);
    let flux_error = background_variance.max(epsilon).sqrt();
    let flux_snr = net_flux / flux_error;

    let positive: Vec<_> = aperture
        .iter()
        .filter_map(|(px, py, value)| {
            let residual = (value - local_background).max(0.0);
            (residual > 0.0).then_some((*px, *py, residual))
        })
        .collect();
    let positive_sum = positive.iter().map(|(_, _, value)| value).sum::<f64>();
    let (fwhm, ellipticity) = if positive_sum > 0.0 {
        let shape_x = positive
            .iter()
            .map(|(px, _, value)| px * value)
            .sum::<f64>()
            / positive_sum;
        let shape_y = positive
            .iter()
            .map(|(_, py, value)| py * value)
            .sum::<f64>()
            / positive_sum;
        let variance_x = positive
            .iter()
            .map(|(px, _, value)| value * (px - shape_x).powi(2))
            .sum::<f64>()
            / positive_sum;
        let variance_y = positive
            .iter()
            .map(|(_, py, value)| value * (py - shape_y).powi(2))
            .sum::<f64>()
            / positive_sum;
        let fwhm_x = 2.35482 * variance_x.max(0.0).sqrt();
        let fwhm_y = 2.35482 * variance_y.max(0.0).sqrt();
        let fwhm = 2.35482 * ((variance_x + variance_y) * 0.5).max(0.0).sqrt();
        let ellipticity = (fwhm_x - fwhm_y).abs() / fwhm_x.max(fwhm_y).max(epsilon);
        if fwhm.is_finite() && fwhm > 0.0 && ellipticity.is_finite() {
            (Some(fwhm), Some(ellipticity))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    let center_x = x.round_ties_even().clamp(0.0, (width - 1) as f64) as usize;
    let center_y = y.round_ties_even().clamp(0.0, (height - 1) as f64) as usize;
    let center_index = center_y * width + center_x;
    let center_valid = !mask[center_index] && image[center_index].is_finite();
    let center_residual = image[center_index] - local_background;
    let peak_excess = if center_residual.is_nan() {
        f64::NAN
    } else {
        center_residual.max(0.0)
    };
    let support_threshold = if peak_excess.is_nan() {
        f64::NAN
    } else {
        (2.0 * local_noise).max(0.1 * peak_excess)
    };
    let mut support = 0i16;
    for core_y in center_y.saturating_sub(1)..=(center_y + 1).min(height - 1) {
        for core_x in center_x.saturating_sub(1)..=(center_x + 1).min(width - 1) {
            let index = core_y * width + core_x;
            if !mask[index]
                && image[index].is_finite()
                && image[index] - local_background >= support_threshold
            {
                support += 1;
            }
        }
    }
    no_measurement(Some(flux_snr), support, center_valid, fwhm, ellipticity)
}

/// Recheck registered-stack coordinates in original-frame pixels.
pub fn stack_forced_measure_batch(
    pixels: FramePixels<'_>,
    xs: &[f64],
    ys: &[f64],
    options: StackMeasurementOptions,
) -> Result<Vec<StackMeasurement>> {
    let FramePixels {
        image,
        mask,
        height,
        width,
    } = pixels;
    if height == 0 || width == 0 || image.len() != height * width {
        return Err("image shape does not match its data".into());
    }
    if mask.len() != image.len() {
        return Err("mask shape does not match image shape".into());
    }
    if xs.len() != ys.len() {
        return Err("xs and ys lengths must match".into());
    }
    let results = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| {
            let value = measure_one(&pixels, options, *x, *y);
            (
                value.flux_snr,
                value.support,
                value.center_valid,
                value.fwhm,
                value.ellipticity,
            )
        })
        .collect();
    Ok(results)
}

/// JSONL/desktop request wrapper for `sequence.stack_forced_measure_batch`.
pub fn handle_request(request: &Value) -> Result<Value> {
    let (image, height, width) = matrix(request.get("image").ok_or("missing image")?, "image")?;
    let mask = mask_matrix(request.get("mask").ok_or("missing mask")?, height, width)?;
    let coordinates = |key: &str| -> Result<Vec<f64>> {
        request
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{key} must be an array"))?
            .iter()
            .map(|value| number(value, key))
            .collect()
    };
    let xs = coordinates("xs")?;
    let ys = coordinates("ys")?;
    let aperture_radius = request
        .get("aperture_radius")
        .and_then(Value::as_u64)
        .ok_or("aperture_radius must be a non-negative integer")?
        as usize;
    let global_background = request
        .get("global_background")
        .and_then(Value::as_f64)
        .ok_or("missing global_background")?;
    let global_noise = request
        .get("global_noise")
        .and_then(Value::as_f64)
        .ok_or("missing global_noise")?;
    let results = stack_forced_measure_batch(
        FramePixels {
            image: &image,
            mask: &mask,
            height,
            width,
        },
        &xs,
        &ys,
        StackMeasurementOptions {
            aperture_radius,
            global_background,
            global_noise,
        },
    )?;
    let mut flux_snr = Vec::with_capacity(results.len());
    let mut support = Vec::with_capacity(results.len());
    let mut center_valid = Vec::with_capacity(results.len());
    let mut fwhm = Vec::with_capacity(results.len());
    let mut ellipticity = Vec::with_capacity(results.len());
    for (snr, pixels, valid, width, shape) in results {
        flux_snr.push(snr);
        support.push(pixels);
        center_valid.push(valid);
        fwhm.push(width);
        ellipticity.push(shape);
    }
    Ok(json!({
        "flux_snr": flux_snr,
        "support": support,
        "center_valid": center_valid,
        "fwhm": fwhm,
        "ellipticity": ellipticity,
    }))
}
