//! Fixed-position single/pair PSF evidence; quality flags remain independent.
use crate::{
    measurement::{quality_passes, Detection},
    models::NumericPlane,
    spatial::KdTree,
    Result,
};
use nalgebra::{DMatrix, DVector};
use std::collections::BTreeMap;

fn least_squares(design: &DMatrix<f64>, values: &DVector<f64>) -> Option<DVector<f64>> {
    let normal = design.transpose() * design;
    let rhs = design.transpose() * values;
    if let Some(beta) = normal
        .lu()
        .solve(&rhs)
        .filter(|b| b.iter().all(|v| v.is_finite()))
    {
        return Some(beta);
    }
    let svd = design.clone().svd(true, true);
    let limit = f64::EPSILON
        * design.nrows().max(design.ncols()) as f64
        * svd.singular_values.iter().copied().fold(0.0, f64::max);
    svd.solve(values, limit).ok()
}
pub fn pair_psf_evidence(
    image: &[f64],
    mask: &[bool],
    height: usize,
    width: usize,
    primary: [f64; 2],
    secondary: [f64; 2],
    fwhm: f64,
) -> Result<(Option<f64>, Option<f64>)> {
    pair_psf_evidence_typed(image.into(), mask, height, width, primary, secondary, fwhm)
}
pub fn pair_psf_evidence_typed(
    image: NumericPlane<'_>,
    mask: &[bool],
    height: usize,
    width: usize,
    primary: [f64; 2],
    secondary: [f64; 2],
    fwhm: f64,
) -> Result<(Option<f64>, Option<f64>)> {
    if height.checked_mul(width) != Some(image.len())
        || mask.len() != image.len()
        || height == 0
        || width == 0
    {
        return Err("invalid PSF image shape".into());
    }
    if primary.iter().chain(&secondary).any(|v| !v.is_finite()) || !fwhm.is_finite() || fwhm <= 0.0
    {
        return Err("PSF coordinates and width must be finite".into());
    }
    // Identical template columns leave the two amplitudes unidentifiable.
    // Return no pair evidence instead of exposing solver-dependent splits.
    if primary[0] == secondary[0] && primary[1] == secondary[1] {
        return Ok((None, None));
    }
    let sigma = (fwhm / 2.35482).max(0.5);
    let margin = (3.5 * sigma).ceil().max(4.0) as i64;
    let [px, py] = primary;
    let [sx, sy] = secondary;
    let x0 = (px.min(sx).floor() as i64 - margin).max(0) as usize;
    let x1 = (px.max(sx).ceil() as i64 + margin + 1)
        .max(0)
        .min(width as i64) as usize;
    let y0 = (py.min(sy).floor() as i64 - margin).max(0) as usize;
    let y1 = (py.max(sy).ceil() as i64 + margin + 1)
        .max(0)
        .min(height as i64) as usize;
    if x1.saturating_sub(x0) < 5 || y1.saturating_sub(y0) < 5 {
        return Ok((None, None));
    }
    let center_x = 0.5 * (x0 + x1 - 1) as f64;
    let center_y = 0.5 * (y0 + y1 - 1) as f64;
    let scale = (x1 - x0).max(y1 - y0).max(1) as f64;
    let mut rows_single = Vec::new();
    let mut rows_pair = Vec::new();
    let mut ys = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let index = y * width + x;
            if mask[index] || !image.value(index).is_finite() {
                continue;
            }
            let gx = x as f64;
            let gy = y as f64;
            let p = (-0.5 * (((gx - px) / sigma).powi(2) + ((gy - py) / sigma).powi(2))).exp();
            let s = (-0.5 * (((gx - sx) / sigma).powi(2) + ((gy - sy) / sigma).powi(2))).exp();
            let background = [1.0, (gx - center_x) / scale, (gy - center_y) / scale];
            rows_single.extend(background);
            rows_single.push(p);
            rows_pair.extend(background);
            rows_pair.extend([p, s]);
            ys.push(image.value(index));
        }
    }
    if ys.len() < 24 {
        return Ok((None, None));
    }
    let n = ys.len();
    let design_single = DMatrix::from_row_slice(n, 4, &rows_single);
    let design_pair = DMatrix::from_row_slice(n, 5, &rows_pair);
    let values = DVector::from_vec(ys);
    let Some(beta_single) = least_squares(&design_single, &values) else {
        return Ok((None, None));
    };
    let Some(beta_pair) = least_squares(&design_pair, &values) else {
        return Ok((None, None));
    };
    if beta_pair[3] <= 0.0 || beta_pair[4] <= 0.0 {
        return Ok((None, None));
    }
    let residual_single = &values - design_single * beta_single;
    let residual_pair = &values - &design_pair * &beta_pair;
    let rss_single = residual_single.dot(&residual_single).max(f64::EPSILON);
    let rss_pair = residual_pair.dot(&residual_pair).max(f64::EPSILON);
    let n_f = n as f64;
    let delta = n_f * (rss_single / n_f).ln() + 4.0 * n_f.ln()
        - (n_f * (rss_pair / n_f).ln() + 5.0 * n_f.ln());
    let variance = rss_pair / n.saturating_sub(5).max(1) as f64;
    let normal = design_pair.transpose() * design_pair;
    let inverse = normal
        .clone()
        .lu()
        .solve(&DMatrix::identity(5, 5))
        .or_else(|| {
            let svd = normal.svd(true, true);
            let cutoff = 1e-15 * svd.singular_values.iter().copied().fold(0.0, f64::max);
            svd.pseudo_inverse(cutoff).ok()
        });
    let Some(inverse) = inverse else {
        return Ok((Some(delta), None));
    };
    let error = (variance * inverse[(4, 4)]).max(f64::EPSILON).sqrt();
    Ok((Some(delta), Some(beta_pair[4] / error)))
}

#[derive(Debug, Clone, Copy)]
pub struct PairGuardOptions {
    pub psf_fwhm: f64,
    pub delta_bic_min: f64,
    pub component_snr_min: f64,
}
pub fn guard_unresolved_gaussian_pairs(
    sources: &[Detection],
    image: &[f64],
    mask: &[bool],
    height: usize,
    width: usize,
    options: PairGuardOptions,
) -> Result<(Vec<Detection>, usize, usize, f64)> {
    guard_unresolved_gaussian_pairs_typed(sources, image.into(), mask, height, width, options)
}
pub fn guard_unresolved_gaussian_pairs_typed(
    sources: &[Detection],
    image: NumericPlane<'_>,
    mask: &[bool],
    height: usize,
    width: usize,
    options: PairGuardOptions,
) -> Result<(Vec<Detection>, usize, usize, f64)> {
    if options.psf_fwhm <= 0.0 || options.delta_bic_min <= 0.0 || options.component_snr_min <= 0.0 {
        return Err("PSF pair guard parameters must be positive".into());
    }
    let radius = 2.5f64.max(1.5 * options.psf_fwhm);
    let eligible: Vec<_> = sources
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            s.proposal_methods.iter().any(|m| m == "gaussian")
                && (s.quality_passed
                    || s.flags.iter().any(|f| {
                        ["CODE_PATTERN", "NEGATIVE_OVERFLOW", "MASKED", "SATURATED"]
                            .contains(&f.as_str())
                    }))
        })
        .map(|(i, _)| i)
        .collect();
    if eligible.len() < 2 {
        return Ok((sources.to_vec(), 0, 0, radius));
    }
    let positions: Vec<_> = eligible
        .iter()
        .map(|i| sources[*i].measurement_position())
        .collect();
    let tree = KdTree::new(&positions)?;
    let mut pairs = Vec::new();
    for (i, p) in positions.iter().enumerate() {
        for (j, _) in tree.within_radius(*p, radius)? {
            if i < j {
                pairs.push((i, j));
            }
        }
    }
    pairs.sort();
    let strength = |s: &Detection| [s.peak, s.filter_snr.unwrap_or(s.snr), s.flux];
    let mut replacements = BTreeMap::new();
    for (i, j) in &pairs {
        let (mut primary, mut secondary) = (eligible[*i], eligible[*j]);
        let a = strength(&sources[primary]);
        let b = strength(&sources[secondary]);
        if b.iter()
            .zip(a)
            .map(|(b, a)| b.total_cmp(&a))
            .find(|order| !order.is_eq())
            .is_some_and(|order| order.is_gt())
        {
            std::mem::swap(&mut primary, &mut secondary);
        }
        let evidence = pair_psf_evidence_typed(
            image,
            mask,
            height,
            width,
            sources[primary].peak_position(),
            sources[secondary].peak_position(),
            options.psf_fwhm,
        )?;
        if !(evidence.0.is_some_and(|v| v >= options.delta_bic_min)
            && evidence.1.is_some_and(|v| v >= options.component_snr_min))
        {
            replacements.insert(secondary, evidence);
        }
    }
    let mut output = sources.to_vec();
    for (index, (delta, snr)) in &replacements {
        let s = &mut output[*index];
        if !s.flags.iter().any(|f| f == "UNRESOLVED_BLEND") {
            s.flags.push("UNRESOLVED_BLEND".into());
        }
        s.quality_passed = false;
        s.deblend_delta_bic = *delta;
        s.deblend_component_snr = *snr;
    }
    Ok((output, pairs.len(), replacements.len(), radius))
}

pub fn resolve_supplemental_blends(
    sources: &[Detection],
    image: &[f64],
    mask: &[bool],
    height: usize,
    width: usize,
    options: PairGuardOptions,
) -> Result<Vec<Detection>> {
    resolve_supplemental_blends_typed(sources, image.into(), mask, height, width, options)
}
pub fn resolve_supplemental_blends_typed(
    sources: &[Detection],
    image: NumericPlane<'_>,
    mask: &[bool],
    height: usize,
    width: usize,
    options: PairGuardOptions,
) -> Result<Vec<Detection>> {
    let gaussian: Vec<_> = sources
        .iter()
        .filter(|s| s.proposal_methods.iter().any(|m| m == "gaussian"))
        .collect();
    if gaussian.is_empty() {
        return Ok(sources.to_vec());
    }
    let tree = crate::compatible_spatial::CompatibleKdTree::new(
        &gaussian
            .iter()
            .map(|s| s.peak_position())
            .collect::<Vec<_>>(),
    )?;
    let mut output = sources.to_vec();
    for source in &mut output {
        if !source.flags.iter().any(|f| f == "UNRESOLVED_BLEND") {
            continue;
        }
        let (index, _) = tree
            .nearest(source.peak_position())
            .ok_or("missing Gaussian neighbor")?;
        let (delta, snr) = pair_psf_evidence_typed(
            image,
            mask,
            height,
            width,
            gaussian[index].peak_position(),
            source.peak_position(),
            options.psf_fwhm,
        )?;
        if delta.is_some_and(|v| v >= options.delta_bic_min)
            && snr.is_some_and(|v| v >= options.component_snr_min)
        {
            source.flags.retain(|f| f != "UNRESOLVED_BLEND");
        }
        source.quality_passed = quality_passes(&source.flags);
        source.deblend_delta_bic = delta;
        source.deblend_component_snr = snr;
    }
    Ok(output)
}
