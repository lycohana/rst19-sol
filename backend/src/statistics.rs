//! Robust statistics and the legacy background-grid interpolation contract.
use crate::Result;

pub fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        let mut sum = -0.0;
        for v in values {
            sum += v;
        }
        return sum;
    }
    if n <= 128 {
        let mut sums: [f64; 8] = values[..8].try_into().unwrap();
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
    pairwise_sum(&values[..middle]) + pairwise_sum(&values[middle..])
}

pub fn percentile(values: &[f64], q: f64) -> Result<f64> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) || !(0.0..=100.0).contains(&q) {
        return Err("percentile requires finite data and q in [0,100]".into());
    }
    let mut data = values.to_vec();
    let location = (data.len() - 1) as f64 * q / 100.0;
    let lo = location.floor() as usize;
    let hi = location.ceil() as usize;
    data.select_nth_unstable_by(lo, f64::total_cmp);
    let a = data[lo];
    if hi == lo {
        return Ok(a);
    }
    data.select_nth_unstable_by(hi, f64::total_cmp);
    Ok(a + (data[hi] - a) * (location - lo as f64))
}
pub fn median(values: &[f64]) -> Result<f64> {
    percentile(values, 50.0)
}
pub fn stddev(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}
pub fn finite_values(image: &[f64], mask: Option<&[bool]>, limit: usize) -> Result<Vec<f64>> {
    if mask.is_some_and(|m| m.len() != image.len()) {
        return Err("mask shape does not match image shape".into());
    }
    if limit == 0 {
        return Err("sample_limit must be positive".into());
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
    image: &[f64],
    mask: Option<&[bool]>,
    sigma: f64,
    iterations: usize,
    sample_limit: usize,
) -> Result<(f64, f64)> {
    let mut selected = finite_values(image, mask, sample_limit)?;
    if selected.is_empty() {
        return Err("image has no valid pixels for background estimation".into());
    }
    for _ in 0..iterations.max(1) {
        let center = median(&selected)?;
        let deviations: Vec<_> = selected.iter().map(|v| (v - center).abs()).collect();
        let mut scale = 1.4826 * median(&deviations)?;
        if !scale.is_finite() || scale <= 0.0 {
            scale = stddev(&selected);
        }
        if !scale.is_finite() || scale <= 0.0 {
            scale = f64::EPSILON;
        }
        let keep: Vec<_> = selected
            .iter()
            .copied()
            .filter(|v| (v - center).abs() <= sigma * scale)
            .collect();
        if keep.len() == selected.len() {
            return Ok((center, scale.max(stddev(&selected)).max(f64::EPSILON)));
        }
        selected = keep;
        if selected.is_empty() {
            return Ok((f64::NAN, f64::NAN));
        }
    }
    let center = median(&selected)?;
    let deviations: Vec<_> = selected.iter().map(|v| (v - center).abs()).collect();
    Ok((
        center,
        (1.4826 * median(&deviations)?)
            .max(stddev(&selected))
            .max(f64::EPSILON),
    ))
}
pub fn resize_grid(
    grid: &[f64],
    gh: usize,
    gw: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f64>> {
    if gh == 0 || gw == 0 || height == 0 || width == 0 || grid.len() != gh * gw {
        return Err("invalid interpolation grid shape".into());
    }
    if gh == 1 && gw == 1 {
        return Ok(vec![grid[0]; height * width]);
    }
    let zy = if height > 1 {
        (height - 1) as f64 / gh.saturating_sub(1).max(1) as f64
    } else {
        1.0
    };
    let zx = if width > 1 {
        (width - 1) as f64 / gw.saturating_sub(1).max(1) as f64
    } else {
        1.0
    };
    // Python/NumPy ties-to-even rounding is part of the existing zoom/crop convention.
    let rh = (gh as f64 * zy).round_ties_even().max(1.0) as usize;
    let rw = (gw as f64 * zx).round_ties_even().max(1.0) as usize;
    let oh = height.min(rh);
    let ow = width.min(rw);
    let mut sampled = vec![0.0; oh * ow];
    for y in 0..oh {
        for x in 0..ow {
            let fy = if rh == 1 {
                0.0
            } else {
                y as f64 * (gh - 1) as f64 / (rh - 1) as f64
            };
            let fx = if rw == 1 {
                0.0
            } else {
                x as f64 * (gw - 1) as f64 / (rw - 1) as f64
            };
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(gh - 1);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(gw - 1);
            let a =
                grid[y0 * gw + x0] + (grid[y1 * gw + x0] - grid[y0 * gw + x0]) * (fy - y0 as f64);
            let b =
                grid[y0 * gw + x1] + (grid[y1 * gw + x1] - grid[y0 * gw + x1]) * (fy - y0 as f64);
            sampled[y * ow + x] = a + (b - a) * (fx - x0 as f64);
        }
    }
    let mut result = vec![sampled[sampled.len() - 1]; height * width];
    for y in 0..oh {
        result[y * width..y * width + ow].copy_from_slice(&sampled[y * ow..(y + 1) * ow]);
    }
    Ok(result)
}
#[derive(Debug, Clone, Copy)]
pub struct BackgroundOptions {
    pub box_size: usize,
    pub fallback: Option<(f64, f64)>,
    pub iterations: usize,
    pub sample_limit: usize,
}

pub fn local_background_rms(
    image: &[f64],
    mask: &[bool],
    height: usize,
    width: usize,
    options: BackgroundOptions,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let BackgroundOptions {
        box_size,
        fallback,
        iterations,
        sample_limit,
    } = options;
    if box_size < 16
        || iterations < 1
        || sample_limit < 1
        || image.len() != height * width
        || mask.len() != image.len()
        || height == 0
        || width == 0
    {
        return Err("invalid local background dimensions or parameters".into());
    }
    let global = match fallback {
        Some(v) => v,
        None => sigma_clipped_stats(image, Some(mask), 3.0, iterations, 1_000_000)?,
    };
    let gh = height.div_ceil(box_size);
    let gw = width.div_ceil(box_size);
    let mut centers = Vec::new();
    let mut noises = Vec::new();
    for y0 in (0..height).step_by(box_size) {
        for x0 in (0..width).step_by(box_size) {
            let mut block = Vec::new();
            let mut invalid = Vec::new();
            for y in y0..(y0 + box_size).min(height) {
                for x in x0..(x0 + box_size).min(width) {
                    block.push(image[y * width + x]);
                    invalid.push(mask[y * width + x]);
                }
            }
            let mut stats =
                sigma_clipped_stats(&block, Some(&invalid), 3.0, iterations, sample_limit)
                    .unwrap_or(global);
            if !stats.0.is_finite() || !stats.1.is_finite() || stats.1 <= 0.0 {
                stats = global;
            }
            centers.push(stats.0);
            noises.push(stats.1);
        }
    }
    Ok((
        resize_grid(&centers, gh, gw, height, width)?,
        resize_grid(&noises, gh, gw, height, width)?
            .into_iter()
            .map(|v| v.max(f64::EPSILON))
            .collect(),
    ))
}
