//! Residual proposal search preserves the shared fixed-PSF projection and masked fallback.
use crate::{
    filtering::{self, MergedProposal, Proposal},
    models::NumericPlane,
    spatial::KdTree,
    Result,
};
use nalgebra::{DMatrix, DVector};

pub struct DeblendContext<'a> {
    pub image: NumericPlane<'a>,
    pub valid: &'a [bool],
    pub line_mask: Option<&'a [bool]>,
    pub background: NumericPlane<'a>,
    pub noise: NumericPlane<'a>,
    pub height: usize,
    pub width: usize,
}
#[derive(Debug, Clone, Copy)]
pub struct DeblendOptions {
    pub psf_fwhm: f64,
    pub min_primary_snr: f64,
    pub min_residual_sigma: f64,
    pub search_radius_factor: f64,
}
pub fn local_deblend_proposals(
    ctx: &DeblendContext<'_>,
    primary: &[Proposal],
    options: DeblendOptions,
) -> Result<Vec<Proposal>> {
    if options.min_primary_snr <= 0.0
        || options.min_residual_sigma <= 0.0
        || options.search_radius_factor <= 0.0
    {
        return Err("local deblend thresholds must be positive".into());
    }
    let size = ctx
        .height
        .checked_mul(ctx.width)
        .ok_or("local deblend shape overflow")?;
    if size != ctx.image.len()
        || [ctx.valid.len(), ctx.background.len(), ctx.noise.len()]
            .iter()
            .any(|n| *n != size)
        || ctx.line_mask.is_some_and(|m| m.len() != size)
    {
        return Err("local deblend arrays must have the same shape".into());
    }
    let sigma = (options.psf_fwhm / 2.35482).max(0.5);
    let search_radius = (options.search_radius_factor * options.psf_fwhm).max(3.0);
    let search_pixels = search_radius.ceil().max(4.0) as usize;
    let minimum = 1.5f64.max(0.8 * options.psf_fwhm);
    let patch_size = 2 * search_pixels + 1;
    let mut design_values = Vec::new();
    for y in 0..patch_size {
        for x in 0..patch_size {
            let gx = x as f64 - search_pixels as f64;
            let gy = y as f64 - search_pixels as f64;
            design_values.extend([
                1.0,
                gx / patch_size as f64,
                gy / patch_size as f64,
                (-0.5 * ((gx / sigma).powi(2) + (gy / sigma).powi(2))).exp(),
            ]);
        }
    }
    let shared_design = DMatrix::from_row_slice(patch_size * patch_size, 4, &design_values);
    let shared_svd = shared_design.clone().svd(true, true);
    let cutoff = 1e-10
        * shared_svd
            .singular_values
            .iter()
            .copied()
            .fold(0.0, f64::max);
    let projection = shared_svd
        .pseudo_inverse(cutoff)
        .map_err(|e| e.to_string())?;
    let eligible: Vec<_> = primary
        .iter()
        .filter(|p| p.score >= options.min_primary_snr)
        .collect();
    let mut interior = Vec::new();
    let mut fallback = Vec::new();
    for p in eligible {
        if p.x >= search_pixels
            && p.y >= search_pixels
            && p.x < ctx.width.saturating_sub(search_pixels)
            && p.y < ctx.height.saturating_sub(search_pixels)
        {
            interior.push(p);
        } else {
            fallback.push(p);
        }
    }
    let mut proposals = Vec::new();
    for p in interior {
        let x0 = p.x - search_pixels;
        let y0 = p.y - search_pixels;
        let mut indices = Vec::new();
        for y in y0..y0 + patch_size {
            for x in x0..x0 + patch_size {
                indices.push(y * ctx.width + x);
            }
        }
        if indices.iter().any(|i| {
            !ctx.valid[*i]
                || !ctx.image.value(*i).is_finite()
                || ctx.line_mask.is_some_and(|m| m[*i])
        }) {
            fallback.push(p);
            continue;
        }
        let values =
            DVector::from_iterator(indices.len(), indices.iter().map(|i| ctx.image.value(*i)));
        let beta = &projection * &values;
        if !beta[3].is_finite() || beta[3] <= 0.0 {
            continue;
        }
        let residual = values - &shared_design * beta;
        if let Some(candidate) = best_candidate(
            ctx,
            &indices,
            residual.as_slice(),
            SearchGate {
                height: patch_size,
                width: patch_size,
                peak: [p.x, p.y],
                radius: search_radius,
                minimum,
                threshold: options.min_residual_sigma,
            },
        )? {
            proposals.push(Proposal {
                x: candidate.0,
                y: candidate.1,
                score: candidate.2,
                method: "deblend_local".into(),
                scale_fwhm: options.psf_fwhm,
            });
        }
    }
    for p in fallback {
        let x0 = p.x.saturating_sub(search_pixels);
        let x1 = p.x.saturating_add(search_pixels + 1).min(ctx.width);
        let y0 = p.y.saturating_sub(search_pixels);
        let y1 = p.y.saturating_add(search_pixels + 1).min(ctx.height);
        let (w, h) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
        if w == 0 || h == 0 {
            continue;
        }
        let scale = w.max(h).max(1) as f64;
        let mut indices = Vec::new();
        let mut full_design = Vec::new();
        let mut usable_design = Vec::new();
        let mut usable_values = Vec::new();
        for y in y0..y1 {
            for x in x0..x1 {
                let index = y * ctx.width + x;
                let gx = x as f64 - p.x as f64;
                let gy = y as f64 - p.y as f64;
                let terms = [
                    1.0,
                    gx / scale,
                    gy / scale,
                    (-0.5 * ((gx / sigma).powi(2) + (gy / sigma).powi(2))).exp(),
                ];
                indices.push(index);
                full_design.extend(terms);
                if ctx.valid[index]
                    && ctx.image.value(index).is_finite()
                    && !ctx.line_mask.is_some_and(|m| m[index])
                {
                    usable_design.extend(terms);
                    usable_values.push(ctx.image.value(index));
                }
            }
        }
        if usable_values.len() < 24 {
            continue;
        }
        let design = DMatrix::from_row_slice(usable_values.len(), 4, &usable_design);
        let svd = design.svd(true, true);
        let cutoff = f64::EPSILON
            * usable_values.len().max(4) as f64
            * svd.singular_values.iter().copied().fold(0.0, f64::max);
        let Ok(beta) = svd.solve(&DVector::from_vec(usable_values), cutoff) else {
            continue;
        };
        if !beta[3].is_finite() || beta[3] <= 0.0 {
            continue;
        }
        let model = DMatrix::from_row_slice(indices.len(), 4, &full_design) * beta;
        let residual: Vec<_> = indices
            .iter()
            .enumerate()
            .map(|(i, index)| ctx.image.value(*index) - model[i])
            .collect();
        if let Some(candidate) = best_candidate(
            ctx,
            &indices,
            &residual,
            SearchGate {
                height: h,
                width: w,
                peak: [p.x, p.y],
                radius: search_radius,
                minimum,
                threshold: options.min_residual_sigma,
            },
        )? {
            proposals.push(Proposal {
                x: candidate.0,
                y: candidate.1,
                score: candidate.2,
                method: "deblend_local".into(),
                scale_fwhm: options.psf_fwhm,
            });
        }
    }
    Ok(proposals)
}

struct SearchGate {
    height: usize,
    width: usize,
    peak: [usize; 2],
    radius: f64,
    minimum: f64,
    threshold: f64,
}
fn best_candidate(
    ctx: &DeblendContext<'_>,
    indices: &[usize],
    residual: &[f64],
    gate: SearchGate,
) -> Result<Option<(usize, usize, f64)>> {
    let maxima = filtering::maximum_filter(residual, gate.height, gate.width, 1)?;
    let mut best = None;
    let mut best_score = f64::NEG_INFINITY;
    for (i, index) in indices.iter().enumerate() {
        let x = index % ctx.width;
        let y = index / ctx.width;
        let distance = (x as f64 - gate.peak[0] as f64).hypot(y as f64 - gate.peak[1] as f64);
        let noise = ctx.noise.value(*index).max(f64::EPSILON);
        let value = residual[i];
        if ctx.valid[*index]
            && ctx.image.value(*index).is_finite()
            && !ctx.line_mask.is_some_and(|m| m[*index])
            && value == maxima[i]
            && distance >= gate.minimum
            && distance <= gate.radius
            && value >= gate.threshold * noise
            && ctx.image.value(*index) - ctx.background.value(*index) >= 2.0 * noise
        {
            let score = value / noise;
            if score > best_score && score.is_finite() {
                best_score = score;
                best = Some((x, y, score));
            }
        }
    }
    Ok(best)
}
pub fn append_deblend_proposals(
    merged: &[MergedProposal],
    deblend: &[Proposal],
    primary: &[Proposal],
    min_distance: usize,
) -> Vec<MergedProposal> {
    let mut accepted = merged.to_vec();
    let mut ranked = deblend.to_vec();
    ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
    let radius = (0.5 * min_distance as f64).max(1.5).floor().max(1.0);
    let mut unique: Vec<Proposal> = Vec::new();
    for p in ranked {
        if unique.iter().any(|c| {
            (p.x as f64 - c.x as f64).powi(2) + (p.y as f64 - c.y as f64).powi(2) < radius * radius
        }) || accepted
            .iter()
            .any(|c| (p.x as f64 - c.x as f64).powi(2) + (p.y as f64 - c.y as f64).powi(2) < 2.25)
        {
            continue;
        }
        unique.push(p);
    }
    let tree = KdTree::new(
        &primary
            .iter()
            .map(|p| [p.x as f64, p.y as f64])
            .collect::<Vec<_>>(),
    )
    .expect("finite integer proposal coordinates");
    for p in unique {
        accepted.push(MergedProposal {
            x: p.x,
            y: p.y,
            score: p.score,
            methods: vec![p.method],
            scales: vec![p.scale_fwhm],
            nearest_gaussian_px: tree.nearest([p.x as f64, p.y as f64]).map(|(_, d)| d),
        });
    }
    accepted.sort_by(|a, b| b.score.total_cmp(&a.score));
    accepted
}
