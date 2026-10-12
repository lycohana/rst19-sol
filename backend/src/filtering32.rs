//! Typed float32 planes: convolution accumulates in double and rounds at each axis output.
use crate::{statistics32, Result};
use std::collections::VecDeque;

fn shape(image: &[f32], h: usize, w: usize) -> Result<()> {
    if h == 0 || w == 0 || h.checked_mul(w) != Some(image.len()) {
        Err("invalid float32 image shape".into())
    } else {
        Ok(())
    }
}
fn symmetric_axis(
    image: &[f32],
    height: usize,
    width: usize,
    kernel: &[f64],
    vertical: bool,
) -> Result<Vec<f32>> {
    shape(image, height, width)?;
    if kernel.is_empty() || kernel.len().is_multiple_of(2) {
        return Err("invalid symmetric kernel".into());
    }
    let radius = kernel.len() / 2;
    let mut result = vec![0f32; image.len()];
    for y in 0..height {
        for x in 0..width {
            let at = |offset: isize| -> f64 {
                let yy = if vertical {
                    (y as isize + offset).clamp(0, height as isize - 1) as usize
                } else {
                    y
                };
                let xx = if vertical {
                    x
                } else {
                    (x as isize + offset).clamp(0, width as isize - 1) as usize
                };
                image[yy * width + xx] as f64
            };
            let mut sum = at(0) * kernel[radius];
            for (i, weight) in kernel.iter().enumerate().take(radius) {
                let offset = radius - i;
                sum += (at(-(offset as isize)) + at(offset as isize)) * weight;
            }
            result[y * width + x] = sum as f32;
        }
    }
    Ok(result)
}
pub fn gaussian_filter(image: &[f32], height: usize, width: usize, sigma: f64) -> Result<Vec<f32>> {
    shape(image, height, width)?;
    if !sigma.is_finite() || sigma < 0.0 {
        return Err("invalid Gaussian sigma".into());
    }
    if sigma <= 1e-15 {
        return Ok(image.to_vec());
    }
    let radius = (4.0 * sigma + 0.5).floor() as isize;
    let mut kernel: Vec<_> = (-radius..=radius)
        .map(|x| (-0.5 / (sigma * sigma) * (x * x) as f64).exp())
        .collect();
    let sum = crate::statistics::pairwise_sum(&kernel);
    for v in &mut kernel {
        *v /= sum;
    }
    let vertical = symmetric_axis(image, height, width, &kernel, true)?;
    symmetric_axis(&vertical, height, width, &kernel, false)
}
pub fn masked_gaussian_response(
    values: &[f32],
    valid: &[bool],
    height: usize,
    width: usize,
    sigma: f64,
    fast: bool,
) -> Result<(Vec<f32>, &'static str)> {
    shape(values, height, width)?;
    if valid.len() != values.len() {
        return Err("mask shape mismatch".into());
    }
    let sparse = fast && valid.iter().filter(|v| !**v).count() as f64 / valid.len() as f64 <= 0.02;
    if sparse {
        let mut result = gaussian_filter(values, height, width, sigma)?;
        for (v, ok) in result.iter_mut().zip(valid) {
            if !ok {
                *v = 0.0;
            }
        }
        return Ok((result, "sparse_mask_single_pass"));
    }
    let weighted: Vec<_> = values
        .iter()
        .zip(valid)
        .map(|(v, ok)| *v * if *ok { 1.0 } else { 0.0 })
        .collect();
    let weights: Vec<_> = valid.iter().map(|ok| if *ok { 1.0 } else { 0.0 }).collect();
    let sums = gaussian_filter(&weighted, height, width, sigma)?;
    let weights = gaussian_filter(&weights, height, width, sigma)?;
    Ok((
        sums.into_iter()
            .zip(weights)
            .map(|(v, w)| if w > 0.5 { v / w } else { 0.0 })
            .collect(),
        "normalized_two_pass",
    ))
}
pub fn masked_starlet_smooth(
    values: &[f32],
    valid: &[bool],
    height: usize,
    width: usize,
    step: usize,
    fast: bool,
) -> Result<(Vec<f32>, &'static str)> {
    shape(values, height, width)?;
    if step < 1 || valid.len() != values.len() {
        return Err("invalid starlet step or mask".into());
    }
    let mut kernel = vec![0.0; 4 * step + 1];
    for (i, v) in [1., 4., 6., 4., 1.].iter().enumerate() {
        kernel[i * step] = v / 16.;
    }
    let smooth = |v: &[f32]| -> Result<Vec<f32>> {
        let h = symmetric_axis(v, height, width, &kernel, false)?;
        symmetric_axis(&h, height, width, &kernel, true)
    };
    let weighted: Vec<_> = values
        .iter()
        .zip(valid)
        .map(|(v, ok)| if *ok { *v } else { 0.0 })
        .collect();
    let mut result = smooth(&weighted)?;
    let sparse = fast && valid.iter().filter(|v| !**v).count() as f64 / valid.len() as f64 <= 0.02;
    if !sparse {
        let weights = smooth(
            &valid
                .iter()
                .map(|v| if *v { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
        )?;
        for (v, w) in result.iter_mut().zip(weights) {
            *v = if w > 0.5 { *v / w } else { 0.0 };
        }
    }
    for (v, ok) in result.iter_mut().zip(valid) {
        if !ok {
            *v = 0.0;
        }
    }
    Ok((
        result,
        if sparse {
            "sparse_mask_single_pass"
        } else {
            "normalized_two_pass"
        },
    ))
}
pub fn standardize_response(
    response: &[f32],
    mask: &[bool],
    noise: &[f32],
    valid: &[bool],
    limit: usize,
) -> Result<(Vec<f32>, f64, f64)> {
    if [mask.len(), noise.len(), valid.len()]
        .iter()
        .any(|n| *n != response.len())
    {
        return Err("response shape mismatch".into());
    }
    let (background, scale) =
        statistics32::sigma_clipped_stats(response, Some(mask), 3.0, 4, limit)?;
    let scale = scale.max(f64::EPSILON);
    let median = statistics32::median(
        &noise
            .iter()
            .zip(valid)
            .filter(|(_, ok)| **ok)
            .map(|(v, _)| *v)
            .collect::<Vec<_>>(),
    )?
    .max(f64::EPSILON as f32);
    let result = response
        .iter()
        .zip(noise)
        .map(|(r, n)| {
            let local = (n / median).max(0.25);
            let denominator = (scale as f32) * local;
            (r - background as f32) / denominator
        })
        .collect();
    Ok((result, background, scale))
}

fn max_line(values: &[f32], radius: usize) -> Vec<f32> {
    let mut output = vec![0.; values.len()];
    let mut queue: VecDeque<(usize, f32)> = VecDeque::new();
    for i in 0..values.len() + 2 * radius {
        let value =
            values[(i as isize - radius as isize).clamp(0, values.len() as isize - 1) as usize];
        while queue.back().is_some_and(|(_, v)| *v <= value) {
            queue.pop_back();
        }
        queue.push_back((i, value));
        let start = i.saturating_sub(2 * radius);
        while queue.front().is_some_and(|(k, _)| *k < start) {
            queue.pop_front();
        }
        if i >= 2 * radius {
            output[i - 2 * radius] = queue.front().unwrap().1;
        }
    }
    output
}
pub fn maximum_filter(
    image: &[f32],
    height: usize,
    width: usize,
    radius: usize,
) -> Result<Vec<f32>> {
    shape(image, height, width)?;
    let mut horizontal = Vec::with_capacity(image.len());
    for row in image.chunks_exact(width) {
        horizontal.extend(max_line(row, radius));
    }
    let mut result = vec![0.; image.len()];
    for x in 0..width {
        let column: Vec<_> = (0..height).map(|y| horizontal[y * width + x]).collect();
        let values = max_line(&column, radius);
        for (y, v) in values.iter().enumerate() {
            result[y * width + x] = *v;
        }
    }
    Ok(result)
}
pub fn response_proposals(
    response: &[f32],
    valid: &[bool],
    height: usize,
    width: usize,
    options: &crate::filtering::ProposalOptions,
    gate: Option<&[bool]>,
) -> Result<Vec<crate::filtering::Proposal>> {
    shape(response, height, width)?;
    if valid.len() != response.len() || gate.is_some_and(|g| g.len() != response.len()) {
        return Err("candidate gate shape mismatch".into());
    }
    let radius = (2.0 * (options.scale_fwhm / 2.35482).max(0.5))
        .ceil()
        .max(1.0) as usize;
    let maxima = maximum_filter(response, height, width, radius)?;
    let mut ranked = Vec::new();
    for (i, r) in response.iter().enumerate() {
        if r.is_finite()
            && valid[i]
            && *r >= options.threshold_sigma as f32
            && gate.is_none_or(|g| g[i])
            && *r == maxima[i]
        {
            ranked.push((i % width, i / width, *r as f64));
        }
    }
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2));
    Ok(
        crate::filtering::suppress_close_candidates(&ranked, options.min_distance)
            .into_iter()
            .map(|(x, y, score)| crate::filtering::Proposal {
                x,
                y,
                score,
                method: options.method.clone(),
                scale_fwhm: options.scale_fwhm,
            })
            .collect(),
    )
}
pub fn linear_artifact_mask(
    residual: &[f32],
    noise: &[f32],
    valid: &[bool],
    height: usize,
    width: usize,
    detection_sigma: f64,
    psf_fwhm: f64,
) -> Result<Vec<bool>> {
    shape(residual, height, width)?;
    if noise.len() != residual.len() || valid.len() != residual.len() {
        return Err("line support shape mismatch".into());
    }
    let threshold = detection_sigma.max(4.0) as f32;
    let support: Vec<_> = residual
        .iter()
        .zip(noise)
        .zip(valid)
        .map(|((r, n), v)| *v && r.is_finite() && n.is_finite() && r / n >= threshold)
        .collect();
    crate::filtering::linear_support_mask(&support, height, width, psf_fwhm)
}
