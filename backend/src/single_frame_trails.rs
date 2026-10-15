//! Native single-frame local-pixel trail proposals.
//!
//! The proposals are image-plane shape evidence only. They do not use sequence
//! state and cannot by themselves classify an object as moving.
use crate::Result;
use serde_json::{json, Value};

const DEFAULT_LENGTHS: &[f64] = &[8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0];
const TILE_SIZE: usize = 256;
const SEEDS_PER_TILE: usize = 2;
const MAX_SEEDS: usize = 512;
const TOP_PER_LENGTH: usize = 96;
const CENTER_FRACTIONS: [f64; 3] = [-0.25, 0.0, 0.25];
const COARSE_RESPONSE_MIN: f64 = 3.5;
const FINAL_RESPONSE_MIN: f64 = 5.0;
const MIN_LINE_COVERAGE: f64 = 0.90;
const MAX_PEAKS_PER_TEMPLATE: usize = 24;
const MAX_REFINEMENT_CANDIDATES: usize = 2048;
const MAX_OUTPUT_PROPOSALS: usize = 256;
const MIN_ACCEPTED_LENGTH_IN_FWHM: f64 = 3.0;
const GAUSSIAN_TRUNCATE: f64 = 4.0;

#[derive(Clone, Copy)]
struct Candidate {
    x: f64,
    y: f64,
    length: f64,
    angle: f64,
    response: f64,
}

#[derive(Clone, Copy)]
struct Proposal {
    x: f64,
    y: f64,
    length: f64,
    angle: f64,
    response: f64,
}

type ValidatedInput = (Vec<f64>, Vec<bool>, Vec<f64>, Vec<f64>, usize, usize, f64);

fn python_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(value) => value.parse().ok(),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn matrix(value: &Value, name: &str) -> Result<(Vec<f64>, usize, usize)> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{name} 必须是二维数组，实际维度为 0"))?;
    let mut width = None;
    let mut output = Vec::new();
    for row in rows {
        let values = row
            .as_array()
            .ok_or_else(|| format!("{name} 必须是二维数组"))?;
        if width.is_some_and(|expected| expected != values.len()) {
            return Err(format!("{name} 必须是矩形二维数组"));
        }
        width = Some(values.len());
        for value in values {
            output.push(
                python_float(value)
                    .ok_or_else(|| format!("{name} 必须是可以转换为数值数组的二维输入"))?,
            );
        }
    }
    let height = rows.len();
    let width = width.unwrap_or(0);
    if height == 0 || width == 0 {
        return Err(format!("{name} 不能是空数组"));
    }
    Ok((output, height, width))
}

fn field(value: &Value, name: &str, height: usize, width: usize) -> Result<Vec<f64>> {
    if value.is_array() {
        let (values, field_height, field_width) = matrix(value, name)?;
        if field_height != height || field_width != width {
            return Err(format!(
                "{name} 必须是标量或形状为 ({height}, {width}) 的二维数组，实际形状为 ({field_height}, {field_width})"
            ));
        }
        return Ok(values);
    }
    let scalar = python_float(value)
        .ok_or_else(|| format!("{name} 必须是标量或与 image 同形状的数值数组"))?;
    Ok(vec![scalar; height * width])
}

fn valid_mask(value: &Value, height: usize, width: usize) -> Result<Vec<bool>> {
    let rows = value
        .as_array()
        .ok_or_else(|| "valid_mask 必须是二维布尔/0-1 数组".to_string())?;
    let mut output = Vec::with_capacity(height * width);
    let mut found_width = None;
    for row in rows {
        let values = row
            .as_array()
            .ok_or_else(|| "valid_mask 必须是二维布尔/0-1 数组".to_string())?;
        if found_width.is_some_and(|expected| expected != values.len()) {
            return Err("valid_mask 必须是矩形二维数组".into());
        }
        found_width = Some(values.len());
        for value in values {
            if let Some(value) = value.as_bool() {
                output.push(value);
                continue;
            }
            let number = value
                .as_f64()
                .ok_or_else(|| "valid_mask 必须是布尔、整数或 0/1 浮点数组".to_string())?;
            if number != 0.0 && number != 1.0 {
                return Err("valid_mask 的浮点值只能是有限的 0 或 1".into());
            }
            output.push(number == 1.0);
        }
    }
    if rows.len() != height || found_width.unwrap_or(0) != width {
        return Err(format!(
            "valid_mask 必须是形状为 ({height}, {width}) 的二维数组，实际形状为 ({}, {})",
            rows.len(),
            found_width.unwrap_or(0)
        ));
    }
    Ok(output)
}

fn validate(request: &Value) -> Result<ValidatedInput> {
    let (image, height, width) = matrix(request.get("image").ok_or("missing image")?, "image")?;
    if height.min(width) < 8 {
        return Err(format!(
            "image 的最短边至少需要 8 像素，实际形状为 ({height}, {width})"
        ));
    }
    let mask = valid_mask(
        request.get("valid_mask").ok_or("missing valid_mask")?,
        height,
        width,
    )?;
    let background = field(
        request.get("background").ok_or("missing background")?,
        "background",
        height,
        width,
    )?;
    let noise = field(
        request.get("noise").ok_or("missing noise")?,
        "noise",
        height,
        width,
    )?;
    let fwhm = request
        .get("psf_fwhm")
        .and_then(python_float)
        .ok_or_else(|| "psf_fwhm 必须是有限的正数".to_string())?;
    if !fwhm.is_finite() || fwhm <= 0.0 {
        return Err("psf_fwhm 必须是有限的正数".into());
    }
    if fwhm < 0.5 {
        return Err("psf_fwhm 不能小于 0.5 像素；否则当前离散 PSF 近似不适用".into());
    }
    if fwhm > height.min(width) as f64 / 2.0 {
        return Err("psf_fwhm 不能大于图像最短边的一半".into());
    }
    let finite: Vec<_> = image
        .iter()
        .zip(&background)
        .zip(&noise)
        .map(|((image, background), noise)| {
            image.is_finite() && background.is_finite() && noise.is_finite()
        })
        .collect();
    if !mask.iter().any(|value| *value) {
        return Err("valid_mask 没有任何有效像素".into());
    }
    let bad_count = mask
        .iter()
        .zip(&finite)
        .filter(|(mask, finite)| **mask && !**finite)
        .count();
    if bad_count > 0 {
        return Err(format!(
            "valid_mask 标记的区域中有 {bad_count} 个非有限像素"
        ));
    }
    let usable: Vec<_> = mask
        .iter()
        .zip(&finite)
        .map(|(mask, finite)| *mask && *finite)
        .collect();
    if usable
        .iter()
        .zip(&noise)
        .any(|(usable, noise)| *usable && *noise <= 0.0)
    {
        return Err("valid_mask 标记的区域内 noise 必须全部大于 0".into());
    }
    if !usable.iter().any(|value| *value) {
        return Err("没有同时满足 valid、有限且 noise>0 的像素".into());
    }
    Ok((image, usable, background, noise, height, width, fwhm))
}

fn gaussian_constant(input: &[f64], height: usize, width: usize, sigma: f64) -> Vec<f64> {
    let radius = (GAUSSIAN_TRUNCATE * sigma + 0.5).floor() as isize;
    let mut kernel: Vec<_> = (-radius..=radius)
        .map(|offset| (-0.5 / (sigma * sigma) * (offset * offset) as f64).exp())
        .collect();
    let kernel_sum: f64 = kernel.iter().sum();
    for value in &mut kernel {
        *value /= kernel_sum;
    }
    let mut vertical = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (index, weight) in kernel.iter().enumerate() {
                let offset = index as isize - radius;
                let yy = y as isize + offset;
                if (0..height as isize).contains(&yy) {
                    sum += input[yy as usize * width + x] * weight;
                }
            }
            vertical[y * width + x] = sum;
        }
    }
    let mut output = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (index, weight) in kernel.iter().enumerate() {
                let offset = index as isize - radius;
                let xx = x as isize + offset;
                if (0..width as isize).contains(&xx) {
                    sum += vertical[y * width + xx as usize] * weight;
                }
            }
            output[y * width + x] = sum;
        }
    }
    output
}

fn maximum_filter(image: &[f64], height: usize, width: usize, radius: usize) -> Result<Vec<f64>> {
    // Edge replication is equivalent to SciPy's constant -infinity mode for
    // a maximum filter: out-of-image samples can never increase the maximum.
    crate::filtering::maximum_filter(image, height, width, radius)
}

fn bilinear(image: &[f64], height: usize, width: usize, x: f64, y: f64) -> f64 {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x > (width - 1) as f64
        || y > (height - 1) as f64
    {
        return 0.0;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let dx = x - x0 as f64;
    let dy = y - y0 as f64;
    let top = image[y0 * width + x0] * (1.0 - dx) + image[y0 * width + x1] * dx;
    let bottom = image[y1 * width + x0] * (1.0 - dx) + image[y1 * width + x1] * dx;
    top * (1.0 - dy) + bottom * dy
}

fn line_noise_std(sigma: f64, length: usize) -> f64 {
    let radius = (4.0 * sigma).ceil().max(1.0) as isize;
    let mut kernel = Vec::new();
    let mut kernel_sum = 0.0;
    for y in -radius..=radius {
        for x in -radius..=radius {
            let value = (-((x * x + y * y) as f64) / (2.0 * sigma * sigma)).exp();
            kernel.push(value);
            kernel_sum += value;
        }
    }
    let gaussian_variance: f64 = kernel
        .iter()
        .map(|value| (value / kernel_sum).powi(2))
        .sum();
    let mut weighted_sum = length as f64;
    for offset in 1..length {
        let autocorrelation = (-((offset * offset) as f64) / (4.0 * sigma * sigma)).exp();
        weighted_sum += 2.0 * (length - offset) as f64 * autocorrelation;
    }
    let variance = gaussian_variance * weighted_sum / (length * length) as f64;
    variance.max(f64::MIN_POSITIVE).sqrt()
}

fn ordered_indices(values: &[f64], indices: &mut [usize]) {
    indices.sort_by(|left, right| values[*right].total_cmp(&values[*left]));
}

#[allow(clippy::too_many_arguments)]
fn sample_line_response(
    smoothed: &[f64],
    valid: &[f64],
    height: usize,
    width: usize,
    x: f64,
    y: f64,
    length: f64,
    angle_deg: f64,
    sigma: f64,
) -> f64 {
    let integer_length = (length.round_ties_even() as usize).max(2);
    let sample_count = integer_length;
    let angle = angle_deg.to_radians();
    let (cosine, sine) = (angle.cos(), angle.sin());
    let mut signal_sum = 0.0;
    let mut support_sum = 0.0;
    for index in 0..sample_count {
        let fraction = index as f64 / (sample_count - 1) as f64;
        let offset = (fraction - 0.5) * length;
        let sample_x = x + offset * cosine;
        let sample_y = y + offset * sine;
        let support = bilinear(valid, height, width, sample_x, sample_y).clamp(0.0, 1.0);
        signal_sum += bilinear(smoothed, height, width, sample_x, sample_y) * support;
        support_sum += support;
    }
    let coverage = support_sum / sample_count as f64;
    if coverage < 0.90 || support_sum <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let mean_signal = signal_sum / support_sum;
    mean_signal * coverage.sqrt() / line_noise_std(sigma, integer_length)
}

fn refine_candidate(
    candidate: Candidate,
    smoothed: &[f64],
    valid: &[f64],
    height: usize,
    width: usize,
    sigma: f64,
    angle_step: f64,
) -> Candidate {
    let mut best = candidate;
    for angle_offset in [-angle_step / 2.0, 0.0, angle_step / 2.0] {
        let angle = (candidate.angle + angle_offset).rem_euclid(180.0);
        for factor in [0.80, 1.0, 1.25] {
            let length = (candidate.length * factor).max(2.0);
            let response = sample_line_response(
                smoothed,
                valid,
                height,
                width,
                candidate.x,
                candidate.y,
                length,
                angle,
                sigma,
            );
            if response > best.response {
                best.length = length;
                best.angle = angle;
                best.response = response;
            }
        }
    }
    best
}

fn erf_approx(value: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26; maximum absolute error is about 1.5e-7.
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736)
        * t
        + 0.254829592)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

fn positive_matched_score(data: &[f64], template: &[f64]) -> f64 {
    if data.is_empty() || data.len() != template.len() {
        return 0.0;
    }
    let data_mean = data.iter().sum::<f64>() / data.len() as f64;
    let template_mean = template.iter().sum::<f64>() / template.len() as f64;
    let mut numerator = 0.0;
    let mut norm_squared = 0.0;
    for (value, basis) in data.iter().zip(template) {
        let centered = basis - template_mean;
        numerator += (value - data_mean) * centered;
        norm_squared += centered * centered;
    }
    let norm = norm_squared.sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return 0.0;
    }
    (numerator / norm).max(0.0)
}

fn positive_two_template_score(data: &[f64], first: &[f64], second: &[f64]) -> f64 {
    if data.is_empty() || first.len() != data.len() || second.len() != data.len() {
        return 0.0;
    }
    let data_mean = data.iter().sum::<f64>() / data.len() as f64;
    let first_mean = first.iter().sum::<f64>() / first.len() as f64;
    let second_mean = second.iter().sum::<f64>() / second.len() as f64;
    let mut g00 = 0.0;
    let mut g01 = 0.0;
    let mut g11 = 0.0;
    let mut r0 = 0.0;
    let mut r1 = 0.0;
    for index in 0..data.len() {
        let a = first[index] - first_mean;
        let b = second[index] - second_mean;
        let y = data[index] - data_mean;
        g00 += a * a;
        g01 += a * b;
        g11 += b * b;
        r0 += a * y;
        r1 += b * y;
    }
    let determinant = g00 * g11 - g01 * g01;
    let (coefficient0, coefficient1) = if determinant == 0.0 || !determinant.is_finite() {
        (0.0, 0.0)
    } else {
        (
            (g11 * r0 - g01 * r1) / determinant,
            (g00 * r1 - g01 * r0) / determinant,
        )
    };
    if coefficient0 < 0.0 || coefficient1 < 0.0 {
        return positive_matched_score(data, first).max(positive_matched_score(data, second));
    }
    (coefficient0 * r0 + coefficient1 * r1).max(0.0).sqrt()
}

fn compact_source_control(
    whitened: &[f64],
    valid: &[bool],
    height: usize,
    width: usize,
    candidate: Candidate,
    fwhm: f64,
) -> bool {
    let sigma = (fwhm / 2.354820045).max(0.5);
    let half_length = 0.5 * candidate.length.max(2.0);
    let radius = 7.0_f64.max(half_length + 4.0 * sigma + 2.0);
    let x0 = ((candidate.x - radius).floor() as isize).max(0) as usize;
    let x1 = ((candidate.x + radius + 1.0).ceil() as usize).min(width);
    let y0 = ((candidate.y - radius).floor() as isize).max(0) as usize;
    let y1 = ((candidate.y + radius + 1.0).ceil() as usize).min(height);
    if x1.saturating_sub(x0) < 7 || y1.saturating_sub(y0) < 7 {
        return false;
    }
    let mut local_x = Vec::new();
    let mut local_y = Vec::new();
    let mut local_data = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let index = y * width + x;
            if valid[index] {
                local_x.push(x as f64);
                local_y.push(y as f64);
                local_data.push(whitened[index]);
            }
        }
    }
    if local_data.len() < 25 {
        return false;
    }
    let angle = candidate.angle.to_radians();
    let (cosine, sine) = (angle.cos(), angle.sin());
    let edge_scale = 2.0_f64.sqrt() * sigma.max(0.45);
    let half = 0.5 * candidate.length.max(0.5);
    let mut line_template = Vec::with_capacity(local_data.len());
    for index in 0..local_data.len() {
        let dx = local_x[index] - candidate.x;
        let dy = local_y[index] - candidate.y;
        let along = dx * cosine + dy * sine;
        let across = -dx * sine + dy * cosine;
        let finite = 0.5
            * (erf_approx((along + half) / edge_scale) - erf_approx((along - half) / edge_scale));
        line_template.push(finite * (-0.5 * (across / sigma.max(0.45)).powi(2)).exp());
    }
    let peak = line_template
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if peak > 0.0 && peak.is_finite() {
        for value in &mut line_template {
            *value /= peak;
        }
    }

    let mut peak_index = 0;
    for index in 1..local_data.len() {
        if local_data[index] > local_data[peak_index] {
            peak_index = index;
        }
    }
    let mut point_templates = Vec::with_capacity(2);
    for (center_x, center_y) in [
        (candidate.x, candidate.y),
        (local_x[peak_index], local_y[peak_index]),
    ] {
        point_templates.push(
            local_x
                .iter()
                .zip(&local_y)
                .map(|(x, y)| {
                    (-0.5 * ((x - center_x).powi(2) + (y - center_y).powi(2)) / (sigma * sigma))
                        .exp()
                })
                .collect::<Vec<_>>(),
        );
    }
    let point_score = positive_matched_score(&local_data, &point_templates[0])
        .max(positive_matched_score(&local_data, &point_templates[1]));
    let line_score = positive_matched_score(&local_data, &line_template);
    if line_score <= point_score + 1.0 {
        return false;
    }

    let mut double_score = 0.0_f64;
    for factor in [0.75, 1.0, 1.25] {
        let separation = (candidate.length * factor).max(1.0);
        let half_separation = 0.5 * separation;
        for test_degrees in (0..180).step_by(30) {
            let test_angle = (test_degrees as f64).to_radians();
            let test_cosine = test_angle.cos();
            let test_sine = test_angle.sin();
            let mut first = Vec::with_capacity(local_data.len());
            let mut second = Vec::with_capacity(local_data.len());
            for index in 0..local_data.len() {
                let dx0 = local_x[index] - (candidate.x - half_separation * test_cosine);
                let dy0 = local_y[index] - (candidate.y - half_separation * test_sine);
                let dx1 = local_x[index] - (candidate.x + half_separation * test_cosine);
                let dy1 = local_y[index] - (candidate.y + half_separation * test_sine);
                first.push((-0.5 * (dx0 * dx0 + dy0 * dy0) / (sigma * sigma)).exp());
                second.push((-0.5 * (dx1 * dx1 + dy1 * dy1) / (sigma * sigma)).exp());
            }
            double_score =
                double_score.max(positive_two_template_score(&local_data, &first, &second));
        }
    }
    line_score > (double_score - 0.5).max(0.0) + 1.0
}

fn angle_distance(first: f64, second: f64) -> f64 {
    let difference = (first - second).rem_euclid(180.0).abs();
    difference.min(180.0 - difference)
}

fn deduplicate(candidates: &[Candidate], fwhm: f64, max_output: usize) -> Vec<Proposal> {
    let mut candidates = candidates.to_vec();
    candidates.sort_by(|left, right| right.response.total_cmp(&left.response));
    let mut kept: Vec<Proposal> = Vec::new();
    let center_radius = (1.5 * fwhm).max(2.5);
    for candidate in candidates {
        let mut duplicate = false;
        for existing in &kept {
            let shorter = candidate.length.min(existing.length);
            let angle_tolerance = 12.0_f64.max((240.0 * fwhm / shorter.max(1.0)).min(30.0));
            if angle_distance(candidate.angle, existing.angle) > angle_tolerance {
                continue;
            }
            let angle = existing.angle.to_radians();
            let (direction_x, direction_y) = (angle.cos(), angle.sin());
            let (dx, dy) = (candidate.x - existing.x, candidate.y - existing.y);
            let perpendicular = (dx * direction_y - dy * direction_x).abs();
            if perpendicular > center_radius {
                continue;
            }
            let parallel = (dx * direction_x + dy * direction_y).abs();
            let overlap = (candidate.length + existing.length) / 2.0 - parallel;
            let overlap_fraction = if shorter > 0.0 {
                overlap / shorter
            } else {
                0.0
            };
            let ratio = if existing.length > 0.0 {
                candidate.length / existing.length
            } else {
                f64::INFINITY
            };
            let centers_close = dx.hypot(dy) <= center_radius;
            let same_scale = (0.50..=2.00).contains(&ratio);
            if (centers_close && same_scale) || overlap_fraction >= 0.25 {
                duplicate = true;
                break;
            }
        }
        if duplicate {
            continue;
        }
        kept.push(Proposal {
            x: candidate.x,
            y: candidate.y,
            length: candidate.length,
            angle: candidate.angle.rem_euclid(180.0),
            response: candidate.response,
        });
        if kept.len() >= max_output {
            break;
        }
    }
    kept
}

fn parse_positive_integer(value: Option<&Value>, default: usize, name: &str) -> Result<usize> {
    let Some(value) = value else {
        return Ok(default);
    };
    if value.as_bool().is_some() {
        return Err(format!("{name} 必须是正整数"));
    }
    let number = python_float(value).ok_or_else(|| format!("{name} 必须是正整数"))?;
    if !number.is_finite() || number <= 0.0 || number.fract() != 0.0 || number > usize::MAX as f64 {
        return Err(format!("{name} 必须是正整数"));
    }
    Ok(number as usize)
}

fn length_bank(request: &Value, height: usize, width: usize) -> Result<Vec<usize>> {
    let max_allowed = height.min(width).saturating_sub(1) as f64;
    let raw_lengths =
        match request
            .get("length_bank_px")
            .filter(|value| !value.is_null())
        {
            None => DEFAULT_LENGTHS.to_vec(),
            Some(Value::Array(values)) => values
                .iter()
                .map(python_float)
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| "length_bank_px 必须是一维正数序列".to_string())?,
            Some(value) => vec![python_float(value)
                .ok_or_else(|| "length_bank_px 必须是一维正数序列".to_string())?],
        };
    if raw_lengths.is_empty() {
        return Err("length_bank_px 必须是非空的一维正数序列".into());
    }
    if raw_lengths
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("length_bank_px 必须全部是有限的正数".into());
    }
    let mut output = Vec::new();
    for value in raw_lengths
        .into_iter()
        .filter(|value| *value <= max_allowed)
    {
        let length = (value.round_ties_even() as usize).max(2);
        if !output.contains(&length) {
            output.push(length);
        }
    }
    if output.is_empty() {
        return Err(format!(
            "length_bank_px 中没有不超过图像尺寸的长度（上限 {max_allowed} px）"
        ));
    }
    Ok(output)
}

fn single_frame_length_bank(
    request: &Value,
    height: usize,
    width: usize,
    fwhm: f64,
) -> Result<Vec<usize>> {
    let shortest = height.min(width) as f64;
    let max_allowed = shortest - 1.0;
    let Some(raw) = request
        .get("length_bank_px")
        .filter(|value| !value.is_null())
    else {
        let min_length = (1.5 * fwhm).max(2.0);
        let max_length = max_allowed.min((1024.0_f64).max(256.0 * fwhm));
        let mut min_length = min_length;
        if max_length < min_length {
            min_length = min_length.min(max_length);
        }
        min_length = min_length.max(2.0);
        let max_length = min_length.max(max_length);
        let mut values = vec![min_length];
        while *values.last().unwrap() < max_length {
            let next = max_length.min(values.last().unwrap() * std::f64::consts::SQRT_2);
            if next <= *values.last().unwrap() {
                break;
            }
            values.push(next);
        }
        let mut lengths = Vec::new();
        for value in values {
            let rounded = value.round_ties_even().max(2.0) as usize;
            if rounded <= max_allowed as usize && !lengths.contains(&rounded) {
                lengths.push(rounded);
            }
        }
        if lengths.is_empty() {
            return Err("图像尺寸不足以建立有效的线段长度库".into());
        }
        return Ok(lengths);
    };

    let raw_lengths =
        match raw {
            Value::Array(values) => values
                .iter()
                .map(python_float)
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| "length_bank_px 必须是一维正数序列".to_string())?,
            value => vec![python_float(value)
                .ok_or_else(|| "length_bank_px 必须是一维正数序列".to_string())?],
        };
    if raw_lengths.is_empty() {
        return Err("length_bank_px 必须是非空的一维正数序列".into());
    }
    if raw_lengths
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("length_bank_px 必须全部是有限的正数".into());
    }
    if raw_lengths.iter().any(|value| *value > max_allowed) {
        return Err(format!(
            "length_bank_px 不能大于图像最短边减一（{max_allowed} px）"
        ));
    }
    let mut lengths = Vec::new();
    for value in raw_lengths {
        let rounded = value.round_ties_even().max(2.0) as usize;
        if rounded as f64 > max_allowed {
            return Err("length_bank_px 中存在取整后超出图像尺寸的长度".into());
        }
        if !lengths.contains(&rounded) {
            lengths.push(rounded);
        }
    }
    if lengths.is_empty() {
        return Err("length_bank_px 取整后没有有效长度".into());
    }
    Ok(lengths)
}

fn rotate_constant(input: &[f64], height: usize, width: usize, angle_deg: f64) -> Vec<f64> {
    let angle = angle_deg.to_radians();
    let (cosine, sine) = (angle.cos(), angle.sin());
    let center_x = (width - 1) as f64 / 2.0;
    let center_y = (height - 1) as f64 / 2.0;
    let mut output = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 - center_x;
            let dy = y as f64 - center_y;
            let source_x = center_x + cosine * dx - sine * dy;
            let source_y = center_y + sine * dx + cosine * dy;
            output[y * width + x] = bilinear(input, height, width, source_x, source_y);
        }
    }
    output
}

fn uniform_filter_rows(input: &[f64], height: usize, width: usize, size: usize) -> Vec<f64> {
    let mut output = vec![0.0; input.len()];
    let left = size / 2;
    let mut prefix = vec![0.0; width + 1];
    for y in 0..height {
        prefix[0] = 0.0;
        for x in 0..width {
            prefix[x + 1] = prefix[x] + input[y * width + x];
        }
        for x in 0..width {
            let start = (x as isize - left as isize).max(0) as usize;
            let end = (x + (size - left)).min(width);
            output[y * width + x] = (prefix[end] - prefix[start]) / size as f64;
        }
    }
    output
}

fn top_local_peaks(
    response: &[f64],
    coverage: &[f64],
    height: usize,
    width: usize,
    threshold: f64,
    max_peaks: usize,
    peak_size: usize,
) -> Vec<usize> {
    let Ok(local_maximum) = maximum_filter(response, height, width, peak_size / 2) else {
        return Vec::new();
    };
    let mut indices: Vec<_> = (0..response.len())
        .filter(|index| {
            response[*index].is_finite()
                && response[*index] >= threshold
                && response[*index] >= local_maximum[*index]
                && coverage[*index] >= MIN_LINE_COVERAGE
        })
        .collect();
    ordered_indices(response, &mut indices);
    indices.truncate(max_peaks);
    indices
}

/// Return local-pixel finite-line proposals for one image.
pub fn local_pixel_trail_proposals(request: &Value) -> Result<Value> {
    local_pixel_trail_proposals_inner(request, validate(request)?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn local_pixel_trail_proposals_for_plane(
    image: Vec<f64>,
    valid: Vec<bool>,
    height: usize,
    width: usize,
    background: f64,
    noise: f64,
    psf_fwhm: f64,
    options: &Value,
) -> Result<Value> {
    let count = height.saturating_mul(width);
    if height.min(width) < 8
        || image.len() != count
        || valid.len() != count
        || !noise.is_finite()
        || noise <= 0.0
        || !background.is_finite()
        || !psf_fwhm.is_finite()
        || psf_fwhm < 0.5
        || psf_fwhm > height.min(width) as f64 / 2.0
    {
        return Err("invalid local trail image plane or parameters".into());
    }
    let usable = image
        .iter()
        .zip(valid)
        .map(|(value, valid)| valid && value.is_finite())
        .collect();
    local_pixel_trail_proposals_inner(
        options,
        (
            image,
            usable,
            vec![background; count],
            vec![noise; count],
            height,
            width,
            psf_fwhm,
        ),
    )
}

fn local_pixel_trail_proposals_inner(request: &Value, input: ValidatedInput) -> Result<Value> {
    let (image, usable, background, noise, height, width, fwhm) = input;
    let angle_step = match request.get("angle_step_deg") {
        None => 10.0,
        Some(value) => {
            python_float(value).ok_or_else(|| "angle_step_deg 必须是有限的正数".to_string())?
        }
    };
    if !angle_step.is_finite() || angle_step <= 0.0 || angle_step > 180.0 {
        return Err("angle_step_deg 必须满足 0 < angle_step_deg <= 180".into());
    }
    let max_output = parse_positive_integer(request.get("max_proposals"), 128, "max_proposals")?;
    let max_seeds =
        parse_positive_integer(request.get("max_seeds"), MAX_SEEDS, "max_seeds")?.min(MAX_SEEDS);
    let seed_threshold = match request.get("seed_threshold") {
        None => 2.25,
        Some(value) => {
            python_float(value).ok_or_else(|| "seed_threshold 必须是有限的正数".to_string())?
        }
    };
    if !seed_threshold.is_finite() || seed_threshold <= 0.0 {
        return Err("seed_threshold 必须是有限的正数".into());
    }
    let sample_step = match request.get("sample_step_px") {
        None => 1.0,
        Some(value) => {
            python_float(value).ok_or_else(|| "sample_step_px 必须是有限的正数".to_string())?
        }
    };
    if !sample_step.is_finite() || sample_step <= 0.0 {
        return Err("sample_step_px 必须是有限的正数".into());
    }
    let lengths = length_bank(request, height, width)?;

    let mut whitened = vec![0.0; image.len()];
    let mut valid_float = vec![0.0; image.len()];
    for index in 0..image.len() {
        if usable[index] {
            whitened[index] = (image[index] - background[index]) / noise[index];
            valid_float[index] = 1.0;
        }
    }
    let sigma = (fwhm / 2.354820045).max(0.5);
    let smoothed_signal = gaussian_constant(&whitened, height, width, sigma);
    let smoothed_support = gaussian_constant(&valid_float, height, width, sigma);
    let smoothed: Vec<_> = smoothed_signal
        .iter()
        .zip(&smoothed_support)
        .map(|(signal, support)| {
            if *support > 1e-6 {
                signal / support
            } else {
                0.0
            }
        })
        .collect();
    let peak_size = (2.0 * fwhm.ceil() + 1.0).max(5.0) as usize;
    let peak_maximum = maximum_filter(&smoothed, height, width, peak_size / 2)?;
    let mut seed_indices: Vec<_> = (0..smoothed.len())
        .filter(|index| {
            usable[*index]
                && smoothed[*index].is_finite()
                && smoothed[*index] >= seed_threshold
                && smoothed[*index] >= peak_maximum[*index]
        })
        .collect();
    if seed_indices.is_empty() {
        return Ok(json!({"proposals":[]}));
    }
    if seed_indices.len() > max_seeds {
        let mut tiled = Vec::new();
        for y0 in (0..height).step_by(TILE_SIZE) {
            let y1 = (y0 + TILE_SIZE).min(height);
            for x0 in (0..width).step_by(TILE_SIZE) {
                let x1 = (x0 + TILE_SIZE).min(width);
                let mut tile_indices: Vec<_> = seed_indices
                    .iter()
                    .copied()
                    .filter(|index| {
                        let y = *index / width;
                        let x = *index % width;
                        y0 <= y && y < y1 && x0 <= x && x < x1
                    })
                    .collect();
                ordered_indices(&smoothed, &mut tile_indices);
                tile_indices.truncate(SEEDS_PER_TILE);
                tiled.extend(tile_indices);
            }
        }
        seed_indices = tiled;
        if seed_indices.len() > max_seeds {
            ordered_indices(&smoothed, &mut seed_indices);
            seed_indices.truncate(max_seeds);
        }
    }
    ordered_indices(&smoothed, &mut seed_indices);
    let seed_xy: Vec<_> = seed_indices
        .iter()
        .map(|index| ((*index % width) as f64, (*index / width) as f64))
        .collect();

    let mut angles = Vec::new();
    let mut angle_index = 0usize;
    loop {
        let angle = angle_index as f64 * angle_step;
        if angle >= 180.0 {
            break;
        }
        angles.push(angle);
        angle_index += 1;
    }
    if angles.is_empty() {
        angles.push(0.0);
    }
    let mut noise_by_length = std::collections::BTreeMap::new();
    for length in &lengths {
        noise_by_length.insert(*length, line_noise_std(sigma, *length));
    }
    let mut coarse_candidates = Vec::new();
    let top_per_length = TOP_PER_LENGTH.min(16.max(max_output.saturating_mul(2)));
    for length in &lengths {
        let sample_count = ((*length as f64 / sample_step).ceil() as usize + 1).max(5);
        let mut offsets = Vec::with_capacity(sample_count);
        for index in 0..sample_count {
            let fraction = index as f64 / (sample_count - 1) as f64;
            offsets.push((fraction - 0.5) * *length as f64);
        }
        let mut responses =
            Vec::with_capacity(seed_xy.len() * CENTER_FRACTIONS.len() * angles.len());
        let mut candidates = Vec::with_capacity(responses.capacity());
        let noise_std = noise_by_length[length];
        for &(seed_x, seed_y) in &seed_xy {
            for fraction in CENTER_FRACTIONS {
                for &angle in &angles {
                    let radians = angle.to_radians();
                    let (cosine, sine) = (radians.cos(), radians.sin());
                    let center_x = seed_x + *length as f64 * fraction * cosine;
                    let center_y = seed_y + *length as f64 * fraction * sine;
                    let mut support_sum = 0.0;
                    let mut signal_sum = 0.0;
                    for offset in &offsets {
                        let x = center_x + *offset * cosine;
                        let y = center_y + *offset * sine;
                        let support = bilinear(&valid_float, height, width, x, y).clamp(0.0, 1.0);
                        let signal = bilinear(&smoothed, height, width, x, y);
                        support_sum += support;
                        signal_sum += signal * support;
                    }
                    let cov = support_sum / sample_count as f64;
                    let response = if support_sum > 1e-6 {
                        (signal_sum / support_sum) * cov.clamp(0.0, 1.0).sqrt() / noise_std
                    } else {
                        f64::NEG_INFINITY
                    };
                    let response = if cov < 0.90 {
                        f64::NEG_INFINITY
                    } else {
                        response
                    };
                    responses.push(response);
                    candidates.push(Candidate {
                        x: center_x,
                        y: center_y,
                        length: *length as f64,
                        angle,
                        response,
                    });
                }
            }
        }
        let mut eligible: Vec<_> = (0..responses.len())
            .filter(|index| responses[*index].is_finite() && responses[*index] >= 4.0)
            .collect();
        ordered_indices(&responses, &mut eligible);
        eligible.truncate(top_per_length);
        for index in eligible {
            coarse_candidates.push(candidates[index]);
        }
    }
    if coarse_candidates.is_empty() {
        return Ok(json!({"proposals":[]}));
    }
    coarse_candidates.sort_by(|left, right| right.response.total_cmp(&left.response));
    let refinement_limit = coarse_candidates
        .len()
        .min(64.max(max_output.saturating_mul(4)));
    let mut refined = Vec::new();
    for candidate in coarse_candidates.into_iter().take(refinement_limit) {
        let candidate = refine_candidate(
            candidate,
            &smoothed,
            &valid_float,
            height,
            width,
            sigma,
            angle_step,
        );
        if ![
            candidate.x,
            candidate.y,
            candidate.length,
            candidate.angle,
            candidate.response,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            continue;
        }
        if candidate.length < (4.0_f64).max(3.0 * fwhm) {
            continue;
        }
        if !compact_source_control(&whitened, &usable, height, width, candidate, fwhm) {
            continue;
        }
        if candidate.response >= 4.0 {
            refined.push(candidate);
        }
    }
    let proposals = deduplicate(&refined, fwhm, max_output);
    Ok(json!({"proposals":proposals.iter().map(|proposal| json!({
        "x":proposal.x,
        "y":proposal.y,
        "length":proposal.length,
        "width":fwhm,
        "angle":proposal.angle,
        "response":proposal.response,
    })).collect::<Vec<_>>() }))
}

/// Return the native-resolution, multi-angle single-frame line proposals.
/// The operation is restricted to one image and carries no sequence evidence.
pub fn single_frame_trail_proposals(request: &Value) -> Result<Value> {
    let (image, usable, background, noise, height, width, fwhm) = validate(request)?;
    let angle_step = match request.get("angle_step_deg") {
        None => 10.0,
        Some(value) => {
            python_float(value).ok_or_else(|| "angle_step_deg 必须是有限的正数".to_string())?
        }
    };
    if !angle_step.is_finite() || angle_step <= 0.0 || angle_step > 180.0 {
        return Err("angle_step_deg 必须满足 0 < angle_step_deg <= 180".into());
    }
    let max_output = parse_positive_integer(
        request.get("max_proposals"),
        MAX_OUTPUT_PROPOSALS,
        "max_proposals",
    )?;
    if max_output > 100_000 {
        return Err("max_proposals 不能大于 100000".into());
    }
    let lengths = single_frame_length_bank(request, height, width, fwhm)?;

    let mut whitened = vec![0.0; image.len()];
    let mut valid_float = vec![0.0; image.len()];
    for index in 0..image.len() {
        if usable[index] {
            whitened[index] = (image[index] - background[index]) / noise[index];
            valid_float[index] = 1.0;
        }
    }
    let sigma = (fwhm / 2.354820045).max(0.5);
    let smoothed_signal = gaussian_constant(&whitened, height, width, sigma);
    let smoothed_support = gaussian_constant(&valid_float, height, width, sigma);
    let smoothed: Vec<_> = smoothed_signal
        .iter()
        .zip(&smoothed_support)
        .map(|(signal, support)| {
            if *support > 1e-6 {
                signal / support
            } else {
                0.0
            }
        })
        .collect();

    let peak_size = (2.0 * fwhm.ceil() + 1.0).max(3.0) as usize;
    let peaks_per_template = MAX_PEAKS_PER_TEMPLATE.min(8.max(max_output));
    let center_x = (width - 1) as f64 / 2.0;
    let center_y = (height - 1) as f64 / 2.0;
    let mut raw_candidates = Vec::new();
    let mut angle_index = 0usize;
    loop {
        let angle_deg = angle_index as f64 * angle_step;
        if angle_deg >= 180.0 {
            break;
        }
        let rotated_signal = rotate_constant(&smoothed, height, width, angle_deg);
        let rotated_valid = rotate_constant(&valid_float, height, width, angle_deg);
        let mut best_response = vec![f64::NEG_INFINITY; image.len()];
        let mut best_coverage = vec![0.0; image.len()];
        let mut best_length = vec![0.0; image.len()];
        for length in &lengths {
            let line_mean = uniform_filter_rows(&rotated_signal, height, width, *length);
            let coverage = uniform_filter_rows(&rotated_valid, height, width, *length);
            let noise_std = line_noise_std(sigma, *length);
            for index in 0..image.len() {
                let cov = coverage[index];
                let corrected_mean = if cov > 1e-6 {
                    line_mean[index] / cov
                } else {
                    0.0
                };
                let response = if cov < MIN_LINE_COVERAGE {
                    f64::NEG_INFINITY
                } else {
                    corrected_mean * cov.clamp(0.0, 1.0).sqrt() / noise_std
                };
                if response > best_response[index] {
                    best_response[index] = response;
                    best_coverage[index] = cov;
                    best_length[index] = *length as f64;
                }
            }
        }
        let peak_indices = top_local_peaks(
            &best_response,
            &best_coverage,
            height,
            width,
            COARSE_RESPONSE_MIN,
            peaks_per_template,
            peak_size,
        );
        let angle = angle_deg.to_radians();
        let (cosine, sine) = (angle.cos(), angle.sin());
        for index in peak_indices {
            let x_rotated = (index % width) as f64;
            let y_rotated = (index / width) as f64;
            let dx = x_rotated - center_x;
            let dy = y_rotated - center_y;
            let candidate = Candidate {
                x: center_x + cosine * dx - sine * dy,
                y: center_y + sine * dx + cosine * dy,
                length: best_length[index],
                angle: angle_deg,
                response: best_response[index],
            };
            if [
                candidate.x,
                candidate.y,
                candidate.length,
                candidate.response,
            ]
            .iter()
            .all(|value| value.is_finite())
                && candidate.length > 0.0
            {
                raw_candidates.push(candidate);
            }
        }
        angle_index += 1;
    }

    raw_candidates.sort_by(|left, right| right.response.total_cmp(&left.response));
    raw_candidates.truncate(MAX_REFINEMENT_CANDIDATES.min(64.max(max_output.saturating_mul(4))));
    let mut refined = Vec::new();
    for candidate in raw_candidates {
        let candidate = refine_candidate(
            candidate,
            &smoothed,
            &valid_float,
            height,
            width,
            sigma,
            angle_step,
        );
        if ![
            candidate.x,
            candidate.y,
            candidate.length,
            candidate.angle,
            candidate.response,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            continue;
        }
        if candidate.length < 4.0_f64.max(MIN_ACCEPTED_LENGTH_IN_FWHM * fwhm) {
            continue;
        }
        if !compact_source_control(&whitened, &usable, height, width, candidate, fwhm) {
            continue;
        }
        if candidate.x < -0.5
            || candidate.x > width as f64 - 0.5
            || candidate.y < -0.5
            || candidate.y > height as f64 - 0.5
        {
            continue;
        }
        if candidate.response >= FINAL_RESPONSE_MIN {
            refined.push(candidate);
        }
    }
    let proposals = deduplicate(&refined, fwhm, max_output);
    Ok(json!({
        "proposals": proposals
            .iter()
            .map(|proposal| json!({
                "x":proposal.x,
                "y":proposal.y,
                "length":proposal.length,
                "width":fwhm,
                "angle":proposal.angle,
                "response":proposal.response,
            }))
            .collect::<Vec<_>>(),
    }))
}
