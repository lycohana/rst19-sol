//! Scientific image filters and point-proposal gates used by all detection modes.
use crate::{statistics, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

fn shape(image: &[f64], height: usize, width: usize) -> Result<()> {
    if height == 0 || width == 0 || height.checked_mul(width) != Some(image.len()) {
        Err("invalid 2-D image shape".into())
    } else {
        Ok(())
    }
}
pub fn convolve_axis_nearest(
    image: &[f64],
    height: usize,
    width: usize,
    kernel: &[f64],
    vertical: bool,
) -> Result<Vec<f64>> {
    shape(image, height, width)?;
    if kernel.is_empty() || kernel.len() % 2 != 1 {
        return Err("expected odd non-empty convolution kernel".into());
    }
    let radius = (kernel.len() / 2) as isize;
    let mut out = vec![0.0; image.len()];
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let delta = k as isize - radius;
                let yy = if vertical {
                    (y as isize + delta).clamp(0, height as isize - 1) as usize
                } else {
                    y
                };
                let xx = if vertical {
                    x
                } else {
                    (x as isize + delta).clamp(0, width as isize - 1) as usize
                };
                value += image[yy * width + xx] * weight;
            }
            out[y * width + x] = value;
        }
    }
    Ok(out)
}
pub fn gaussian_filter(image: &[f64], height: usize, width: usize, sigma: f64) -> Result<Vec<f64>> {
    shape(image, height, width)?;
    if !sigma.is_finite() || sigma < 0.0 {
        return Err("Gaussian sigma must be finite and nonnegative".into());
    }
    if sigma <= 1e-15 {
        return Ok(image.to_vec());
    }
    let radius = (4.0 * sigma + 0.5).floor() as isize;
    let mut kernel: Vec<_> = (-radius..=radius)
        .map(|x| (-0.5 / (sigma * sigma) * (x * x) as f64).exp())
        .collect();
    let sum = kernel.iter().sum::<f64>();
    for k in &mut kernel {
        *k /= sum;
    }
    let rows = convolve_axis_nearest(image, height, width, &kernel, true)?;
    convolve_axis_nearest(&rows, height, width, &kernel, false)
}
pub fn masked_gaussian_response(
    residual: &[f64],
    valid: &[bool],
    height: usize,
    width: usize,
    sigma: f64,
    fast: bool,
) -> Result<(Vec<f64>, &'static str)> {
    shape(residual, height, width)?;
    if valid.len() != residual.len() {
        return Err("mask shape mismatch".into());
    }
    if fast && valid.iter().filter(|b| !**b).count() as f64 / valid.len() as f64 <= 0.02 {
        let mut result = gaussian_filter(residual, height, width, sigma)?;
        for (value, v) in result.iter_mut().zip(valid) {
            if !v {
                *value = 0.0;
            }
        }
        return Ok((result, "sparse_mask_single_pass"));
    }
    let weighted: Vec<_> = residual
        .iter()
        .zip(valid)
        .map(|(r, v)| *r * if *v { 1.0 } else { 0.0 })
        .collect();
    let weights: Vec<_> = valid.iter().map(|v| if *v { 1.0 } else { 0.0 }).collect();
    let sums = gaussian_filter(&weighted, height, width, sigma)?;
    let weights = gaussian_filter(&weights, height, width, sigma)?;
    Ok((
        sums.into_iter()
            .zip(weights)
            .map(|(sum, weight)| if weight > 0.5 { sum / weight } else { 0.0 })
            .collect(),
        "normalized_two_pass",
    ))
}
pub fn masked_starlet_smooth(
    values: &[f64],
    valid: &[bool],
    height: usize,
    width: usize,
    step: usize,
    fast: bool,
) -> Result<(Vec<f64>, &'static str)> {
    shape(values, height, width)?;
    if step < 1 || valid.len() != values.len() {
        return Err("invalid starlet step or mask".into());
    }
    let mut kernel = vec![0.0; 4 * step + 1];
    for (i, v) in [1.0, 4.0, 6.0, 4.0, 1.0].iter().enumerate() {
        kernel[i * step] = v / 16.0;
    }
    let smooth = |image: &[f64]| -> Result<Vec<f64>> {
        let h = convolve_axis_nearest(image, height, width, &kernel, false)?;
        convolve_axis_nearest(&h, height, width, &kernel, true)
    };
    let weighted: Vec<_> = values
        .iter()
        .zip(valid)
        .map(|(r, v)| if *v { *r } else { 0.0 })
        .collect();
    let sparse = fast && valid.iter().filter(|b| !**b).count() as f64 / valid.len() as f64 <= 0.02;
    let mut result = smooth(&weighted)?;
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
    response: &[f64],
    mask: &[bool],
    noise: &[f64],
    valid: &[bool],
    sample_limit: usize,
) -> Result<(Vec<f64>, f64, f64)> {
    if [mask.len(), noise.len(), valid.len()]
        .iter()
        .any(|n| *n != response.len())
    {
        return Err("response mask/noise shape mismatch".into());
    }
    let (background, scale) =
        statistics::sigma_clipped_stats(response, Some(mask), 3.0, 4, sample_limit)?;
    let scale = scale.max(f64::EPSILON);
    let median = statistics::median(
        &noise
            .iter()
            .zip(valid)
            .filter(|(_, v)| **v)
            .map(|(n, _)| *n)
            .collect::<Vec<_>>(),
    )?
    .max(f64::EPSILON);
    Ok((
        response
            .iter()
            .zip(noise)
            .map(|(r, n)| (r - background) / (scale * (n / median).max(0.25)))
            .collect(),
        background,
        scale,
    ))
}

fn max_line(values: &[f64], radius: usize) -> Vec<f64> {
    let mut output = vec![0.0; values.len()];
    let mut queue: VecDeque<(usize, f64)> = VecDeque::new();
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
    image: &[f64],
    height: usize,
    width: usize,
    radius: usize,
) -> Result<Vec<f64>> {
    shape(image, height, width)?;
    let mut horizontal = Vec::with_capacity(image.len());
    for row in image.chunks_exact(width) {
        horizontal.extend(max_line(row, radius));
    }
    let mut output = vec![0.0; image.len()];
    for x in 0..width {
        let line: Vec<_> = (0..height).map(|y| horizontal[y * width + x]).collect();
        let values = max_line(&line, radius);
        for (y, v) in values.iter().enumerate() {
            output[y * width + x] = *v;
        }
    }
    Ok(output)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub x: usize,
    pub y: usize,
    pub score: f64,
    pub method: String,
    pub scale_fwhm: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergedProposal {
    pub x: usize,
    pub y: usize,
    pub score: f64,
    pub methods: Vec<String>,
    pub scales: Vec<f64>,
    pub nearest_gaussian_px: Option<f64>,
}
pub fn suppress_close_candidates(
    candidates: &[(usize, usize, f64)],
    distance: usize,
) -> Vec<(usize, usize, f64)> {
    if distance < 1 {
        return candidates.to_vec();
    }
    let mut accepted: Vec<(usize, usize, f64)> = Vec::new();
    let mut grid: HashMap<(isize, isize), Vec<usize>> = HashMap::new();
    for &(x, y, score) in candidates {
        let cell = ((x / distance) as isize, (y / distance) as isize);
        let mut close = false;
        for gx in cell.0 - 1..=cell.0 + 1 {
            for gy in cell.1 - 1..=cell.1 + 1 {
                if let Some(indices) = grid.get(&(gx, gy)) {
                    for index in indices {
                        let (ax, ay, _) = accepted[*index];
                        let dx = x as f64 - ax as f64;
                        let dy = y as f64 - ay as f64;
                        if dx * dx + dy * dy < (distance * distance) as f64 {
                            close = true;
                            break;
                        }
                    }
                }
            }
        }
        if !close {
            grid.entry(cell).or_default().push(accepted.len());
            accepted.push((x, y, score));
        }
    }
    accepted
}
#[derive(Debug, Clone)]
pub struct ProposalOptions {
    pub threshold_sigma: f64,
    pub min_distance: usize,
    pub scale_fwhm: f64,
    pub method: String,
}
pub fn response_proposals(
    response: &[f64],
    valid: &[bool],
    height: usize,
    width: usize,
    options: &ProposalOptions,
    gate: Option<&[bool]>,
) -> Result<Vec<Proposal>> {
    shape(response, height, width)?;
    if valid.len() != response.len() || gate.is_some_and(|g| g.len() != response.len()) {
        return Err("candidate_gate shape must match response_snr".into());
    }
    let radius = (2.0 * (options.scale_fwhm / 2.35482).max(0.5))
        .ceil()
        .max(1.0) as usize;
    let maxima = maximum_filter(response, height, width, radius)?;
    let mut ranked = Vec::new();
    for (i, r) in response.iter().enumerate() {
        if r.is_finite()
            && valid[i]
            && *r >= options.threshold_sigma
            && gate.is_none_or(|g| g[i])
            && *r == maxima[i]
        {
            ranked.push((i % width, i / width, *r));
        }
    }
    // Stable order preserves row-major tie handling in the Python implementation.
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2));
    Ok(suppress_close_candidates(&ranked, options.min_distance)
        .into_iter()
        .map(|(x, y, score)| Proposal {
            x,
            y,
            score,
            method: options.method.clone(),
            scale_fwhm: options.scale_fwhm,
        })
        .collect())
}
pub fn merge_proposals(
    primary: &[Proposal],
    extras: &[Proposal],
    distance: usize,
) -> Vec<MergedProposal> {
    let tree = crate::spatial::KdTree::new(
        &primary
            .iter()
            .map(|p| [p.x as f64, p.y as f64])
            .collect::<Vec<_>>(),
    )
    .expect("integer proposal coordinates are finite");
    let mut extras = extras.to_vec();
    extras.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut accepted: Vec<MergedProposal> = Vec::new();
    let mut grid: HashMap<(isize, isize), Vec<usize>> = HashMap::new();
    let cell_size = distance.max(1);
    for p in primary.iter().chain(&extras) {
        let cell = ((p.x / cell_size) as isize, (p.y / cell_size) as isize);
        let mut nearest = None;
        let mut best = f64::INFINITY;
        for gx in cell.0 - 1..=cell.0 + 1 {
            for gy in cell.1 - 1..=cell.1 + 1 {
                if let Some(indices) = grid.get(&(gx, gy)) {
                    for i in indices {
                        let c = &accepted[*i];
                        let d =
                            (p.x as f64 - c.x as f64).powi(2) + (p.y as f64 - c.y as f64).powi(2);
                        if d < (distance * distance) as f64 && d < best {
                            nearest = Some(*i);
                            best = d;
                        }
                    }
                }
            }
        }
        if let Some(index) = nearest {
            let c = &mut accepted[index];
            c.score = c.score.max(p.score);
            if !c.methods.contains(&p.method) {
                c.methods.push(p.method.clone());
            }
            if !c.scales.iter().any(|s| (*s - p.scale_fwhm).abs() < 1e-6) {
                c.scales.push(p.scale_fwhm);
            }
        } else {
            grid.entry(cell).or_default().push(accepted.len());
            accepted.push(MergedProposal {
                x: p.x,
                y: p.y,
                score: p.score,
                methods: vec![p.method.clone()],
                scales: vec![p.scale_fwhm],
                nearest_gaussian_px: None,
            });
        }
    }
    for c in &mut accepted {
        c.scales.sort_by(f64::total_cmp);
        if !c.methods.iter().any(|m| m == "gaussian") {
            c.nearest_gaussian_px = tree
                .nearest([c.x as f64, c.y as f64])
                .map(|(_, distance)| distance);
        }
    }
    accepted.sort_by(|a, b| b.score.total_cmp(&a.score));
    accepted
}
pub fn linear_artifact_mask(
    residual: &[f64],
    noise: &[f64],
    valid: &[bool],
    height: usize,
    width: usize,
    detection_sigma: f64,
    psf_fwhm: f64,
) -> Result<Vec<bool>> {
    shape(residual, height, width)?;
    if noise.len() != residual.len() || valid.len() != residual.len() {
        return Err("line mask shape mismatch".into());
    }
    let threshold = detection_sigma.max(4.0);
    let support: Vec<_> = residual
        .iter()
        .zip(noise)
        .zip(valid)
        .map(|((r, n), v)| *v && r.is_finite() && n.is_finite() && r / n >= threshold)
        .collect();
    linear_support_mask(&support, height, width, psf_fwhm)
}
pub fn linear_support_mask(
    support: &[bool],
    height: usize,
    width: usize,
    psf_fwhm: f64,
) -> Result<Vec<bool>> {
    if height == 0 || width == 0 || height.checked_mul(width) != Some(support.len()) {
        return Err("invalid line support dimensions".into());
    }
    let mut visited = vec![false; support.len()];
    let mut line_mask = vec![false; support.len()];
    let min_pixels = (2.5 * psf_fwhm.powi(2)).round_ties_even().max(24.0) as usize;
    for seed in 0..support.len() {
        if !support[seed] || visited[seed] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([seed]);
        visited[seed] = true;
        while let Some(index) = queue.pop_front() {
            component.push(index);
            let y = index / width;
            let x = index % width;
            for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let n = yy * width + xx;
                    if support[n] && !visited[n] {
                        visited[n] = true;
                        queue.push_back(n);
                    }
                }
            }
        }
        if component.len() < min_pixels {
            continue;
        }
        let count = component.len() as f64;
        let mx = component.iter().map(|i| (i % width) as f64).sum::<f64>() / count;
        let my = component.iter().map(|i| (i / width) as f64).sum::<f64>() / count;
        let xx = component
            .iter()
            .map(|i| (*i % width) as f64 - mx)
            .map(|v| v * v)
            .sum::<f64>()
            / count;
        let yy = component
            .iter()
            .map(|i| (*i / width) as f64 - my)
            .map(|v| v * v)
            .sum::<f64>()
            / count;
        let xy = component
            .iter()
            .map(|i| ((*i % width) as f64 - mx) * ((*i / width) as f64 - my))
            .sum::<f64>()
            / count;
        let disc = (xx - yy).hypot(2.0 * xy);
        let major = (xx + yy + disc) / 2.0;
        let minor = ((xx + yy - disc) / 2.0).max(0.0);
        if 4.0 * major.max(0.0).sqrt() >= 16.0f64.max(5.0 * psf_fwhm)
            && (major / minor.max(f64::EPSILON)).sqrt() >= 4.0
        {
            for index in component {
                line_mask[index] = true;
            }
        }
    }
    let dilation = (psf_fwhm / 2.0).round_ties_even().max(1.0) as usize;
    for _ in 0..dilation {
        let mut expanded = line_mask.clone();
        for i in 0..line_mask.len() {
            if line_mask[i] {
                let y = i / width;
                let x = i % width;
                if x > 0 {
                    expanded[i - 1] = true;
                }
                if x + 1 < width {
                    expanded[i + 1] = true;
                }
                if y > 0 {
                    expanded[i - width] = true;
                }
                if y + 1 < height {
                    expanded[i + width] = true;
                }
            }
        }
        line_mask = expanded;
    }
    Ok(line_mask)
}
