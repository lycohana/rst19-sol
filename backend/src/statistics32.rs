//! Float32 statistics with the baseline's reduction and scalar-rounding stages.
use crate::{
    statistics::{self, BackgroundOptions},
    Result,
};

pub fn pairwise_sum(values: &[f32]) -> f32 {
    // NumPy's default buffered reduction calls the pairwise inner loop once
    // per 8192 elements and adds each block to the reduction accumulator.
    // Recursing over the entire slice changes rounding at block boundaries.
    values
        .chunks(8192)
        .fold(0.0, |sum, block| sum + pairwise_block(block))
}

fn pairwise_block(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 8 {
        let mut sum = -0.0f32;
        for v in values {
            sum += v;
        }
        return sum;
    }
    if n <= 128 {
        let mut sums: [f32; 8] = values[..8].try_into().unwrap();
        let mut i = 8;
        while i + 7 < n {
            for k in 0..8 {
                sums[k] += values[i + k];
            }
            i += 8;
        }
        let mut result = ((sums[0] + sums[1]) + (sums[2] + sums[3]))
            + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
        for v in &values[i..] {
            result += v;
        }
        return result;
    }
    let mut middle = n / 2;
    middle -= middle % 8;
    pairwise_block(&values[..middle]) + pairwise_block(&values[middle..])
}
pub fn median(values: &[f32]) -> Result<f32> {
    if values.is_empty() {
        return Err("median requires non-empty values".into());
    }
    let mut sorted = values.to_vec();
    let high = sorted.len() / 2;
    sorted.select_nth_unstable_by(high, f32::total_cmp);
    let b = sorted[high];
    if sorted.len() % 2 == 1 {
        return Ok(b);
    }
    let a = sorted[..high]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    Ok((a + b) / 2.0)
}
pub fn stddev(values: &[f32]) -> f32 {
    let mean = pairwise_sum(values) / values.len() as f32;
    let squares: Vec<_> = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .collect();
    (pairwise_sum(&squares) / values.len() as f32).sqrt()
}
pub fn finite_values(image: &[f32], mask: Option<&[bool]>, limit: usize) -> Result<Vec<f32>> {
    if mask.is_some_and(|m| m.len() != image.len()) || limit == 0 {
        return Err("invalid mask or sample limit".into());
    }
    let filtered: Vec<_> = image
        .iter()
        .enumerate()
        .filter(|(i, v)| v.is_finite() && !mask.is_some_and(|m| m[*i]))
        .map(|(_, v)| *v)
        .collect();
    let stride = if filtered.len() > limit {
        (filtered.len() / limit).max(1)
    } else {
        1
    };
    Ok(filtered.into_iter().step_by(stride).collect())
}
pub fn sigma_clipped_stats(
    image: &[f32],
    mask: Option<&[bool]>,
    sigma: f64,
    iterations: usize,
    limit: usize,
) -> Result<(f64, f64)> {
    let mut selected = finite_values(image, mask, limit)?;
    if selected.is_empty() {
        return Err("image has no valid pixels for background estimation".into());
    }
    for _ in 0..iterations.max(1) {
        let center = median(&selected)?;
        let deviations: Vec<_> = selected.iter().map(|v| (v - center).abs()).collect();
        let mut scale = 1.4826 * median(&deviations)? as f64;
        if !scale.is_finite() || scale <= 0.0 {
            scale = stddev(&selected) as f64;
        }
        if !scale.is_finite() || scale <= 0.0 {
            scale = f64::EPSILON;
        }
        let threshold = (sigma * scale) as f32;
        let keep: Vec<_> = selected
            .iter()
            .copied()
            .filter(|v| (v - center).abs() <= threshold)
            .collect();
        if keep.len() == selected.len() {
            return Ok((
                center as f64,
                scale.max(stddev(&selected) as f64).max(f64::EPSILON),
            ));
        }
        selected = keep;
        if selected.is_empty() {
            return Ok((f64::NAN, f64::NAN));
        }
    }
    let center = median(&selected)?;
    let deviations: Vec<_> = selected.iter().map(|v| (v - center).abs()).collect();
    Ok((
        center as f64,
        (1.4826 * median(&deviations)? as f64)
            .max(stddev(&selected) as f64)
            .max(f64::EPSILON),
    ))
}
pub fn local_background_rms(
    image: &[f32],
    mask: &[bool],
    height: usize,
    width: usize,
    options: BackgroundOptions,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if options.box_size < 16
        || options.iterations < 1
        || options.sample_limit < 1
        || height.checked_mul(width) != Some(image.len())
        || mask.len() != image.len()
        || height == 0
        || width == 0
    {
        return Err("invalid background dimensions or parameters".into());
    }
    let global = match options.fallback {
        Some(v) => v,
        None => sigma_clipped_stats(image, Some(mask), 3.0, options.iterations, 1_000_000)?,
    };
    let gh = height.div_ceil(options.box_size);
    let gw = width.div_ceil(options.box_size);
    let mut backgrounds = Vec::new();
    let mut noises = Vec::new();
    for y0 in (0..height).step_by(options.box_size) {
        for x0 in (0..width).step_by(options.box_size) {
            let mut values = Vec::new();
            let mut invalid = Vec::new();
            for y in y0..(y0 + options.box_size).min(height) {
                for x in x0..(x0 + options.box_size).min(width) {
                    values.push(image[y * width + x]);
                    invalid.push(mask[y * width + x]);
                }
            }
            let mut stats = sigma_clipped_stats(
                &values,
                Some(&invalid),
                3.0,
                options.iterations,
                options.sample_limit,
            )
            .unwrap_or(global);
            if !stats.0.is_finite() || !stats.1.is_finite() || stats.1 <= 0.0 {
                stats = global;
            }
            backgrounds.push(stats.0);
            noises.push(stats.1);
        }
    }
    Ok((
        statistics::resize_grid(&backgrounds, gh, gw, height, width)?
            .into_iter()
            .map(|v| v as f32)
            .collect(),
        statistics::resize_grid(&noises, gh, gw, height, width)?
            .into_iter()
            .map(|v| (v as f32).max(f64::EPSILON as f32))
            .collect(),
    ))
}
