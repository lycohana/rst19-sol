//! Source-level aperture photometry, shape, position and independent quality evidence.
use crate::{filtering::MergedProposal, models::NumericPlane, statistics, statistics32, Result};
use serde::{Deserialize, Serialize};

pub const QUALITY_REJECT_FLAGS: [&str; 18] = [
    "EDGE",
    "MASKED",
    "LINE_ARTIFACT",
    "BACKGROUND_UNCERTAIN",
    "SATURATED",
    "NON_POSITIVE_FLUX",
    "LOW_FLUX_SNR",
    "NO_SHAPE",
    "NARROW",
    "BROAD",
    "ELONGATED",
    "DIFFUSE",
    "SPIKE",
    "SMALL_FOOTPRINT",
    "INSUFFICIENT_PSF_SUPPORT",
    "UNRESOLVED_BLEND",
    "NEGATIVE_OVERFLOW",
    "CODE_PATTERN",
];
pub fn quality_passes(flags: &[String]) -> bool {
    !flags
        .iter()
        .any(|flag| QUALITY_REJECT_FLAGS.contains(&flag.as_str()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub detection_id: i64,
    pub x: f64,
    pub y: f64,
    pub peak: f64,
    pub flux: f64,
    pub background: f64,
    pub noise: f64,
    pub snr: f64,
    pub fwhm: Option<f64>,
    pub flags: Vec<String>,
    pub flux_error: Option<f64>,
    pub flux_snr: Option<f64>,
    pub filter_snr: Option<f64>,
    pub fwhm_x: Option<f64>,
    pub fwhm_y: Option<f64>,
    pub ellipticity: Option<f64>,
    pub sharpness: Option<f64>,
    pub footprint_pixels: Option<usize>,
    pub psf_support_pixels: Option<usize>,
    pub quality_passed: bool,
    pub peak_x: Option<f64>,
    pub peak_y: Option<f64>,
    pub centroid_shift_px: Option<f64>,
    pub proposal_methods: Vec<String>,
    pub proposal_scales: Vec<f64>,
    pub proposal_snr: Option<f64>,
    pub nearest_gaussian_px: Option<f64>,
    pub deblend_delta_bic: Option<f64>,
    pub deblend_component_snr: Option<f64>,
    pub repeated_code_count: Option<usize>,
    pub range_anomaly_pixel_count: Option<usize>,
    pub repeated_code_values: Vec<i64>,
}
impl Detection {
    pub fn peak_position(&self) -> [f64; 2] {
        [self.peak_x.unwrap_or(self.x), self.peak_y.unwrap_or(self.y)]
    }
    pub fn measurement_position(&self) -> [f64; 2] {
        [self.x, self.y]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MeasurementOptions {
    pub aperture_radius: usize,
    pub saturation_level: Option<f64>,
    pub min_flux_snr: f64,
    pub min_fwhm: f64,
    pub max_fwhm: f64,
    pub max_ellipticity: f64,
    pub min_sharpness: f64,
    pub max_sharpness: f64,
    pub min_footprint_pixels: usize,
    pub min_psf_support_pixels: usize,
    pub gain_e_per_adu: Option<f64>,
    pub read_noise_adu: f64,
    pub refine_local_background: bool,
    pub psf_fwhm: f64,
    pub dog_blend_radius_px: f64,
    pub allow_partial_zero_mask: bool,
    pub negative_overflow_limit: Option<f64>,
    pub repeated_high_code_values: Vec<i64>,
}
impl Default for MeasurementOptions {
    fn default() -> Self {
        Self {
            aperture_radius: 4,
            saturation_level: None,
            min_flux_snr: 5.0,
            min_fwhm: 0.8,
            max_fwhm: 12.0,
            max_ellipticity: 0.65,
            min_sharpness: 0.005,
            max_sharpness: 0.85,
            min_footprint_pixels: 2,
            min_psf_support_pixels: 3,
            gain_e_per_adu: None,
            read_noise_adu: 0.0,
            refine_local_background: true,
            psf_fwhm: 3.0,
            dog_blend_radius_px: 7.5,
            allow_partial_zero_mask: false,
            negative_overflow_limit: None,
            repeated_high_code_values: vec![],
        }
    }
}
pub struct MeasurementContext<'a> {
    pub image: NumericPlane<'a>,
    pub mask: &'a [bool],
    pub line_mask: Option<&'a [bool]>,
    pub background_map: NumericPlane<'a>,
    pub noise_map: NumericPlane<'a>,
    pub height: usize,
    pub width: usize,
}
pub fn source_from_peak(
    ctx: &MeasurementContext<'_>,
    peak: [usize; 2],
    options: &MeasurementOptions,
    proposal: &MergedProposal,
    filter_snr: f64,
) -> Result<Detection> {
    let image = ctx.image;
    let float32 = image.is_float32();
    let reduce = |values: &[f64]| -> f64 {
        if float32 {
            statistics32::pairwise_sum(&values.iter().map(|v| *v as f32).collect::<Vec<_>>()) as f64
        } else {
            statistics::pairwise_sum(values)
        }
    };
    let scalar = |value: f64| -> f64 {
        if float32 {
            value as f32 as f64
        } else {
            value
        }
    };
    let (height, width) = (ctx.height, ctx.width);
    let [xp, yp] = peak;
    let size = height.checked_mul(width).ok_or("image shape overflow")?;
    if size != image.len()
        || [
            ctx.mask.len(),
            ctx.background_map.len(),
            ctx.noise_map.len(),
        ]
        .iter()
        .any(|n| *n != size)
        || ctx.line_mask.is_some_and(|m| m.len() != size)
        || xp >= width
        || yp >= height
    {
        return Err("source measurement shape or peak is invalid".into());
    }
    let radius = options.aperture_radius;
    let outer = radius.checked_add(4).ok_or("aperture radius overflow")?;
    let y0 = yp.saturating_sub(outer);
    let y1 = yp.saturating_add(outer + 1).min(height);
    let x0 = xp.saturating_sub(outer);
    let x1 = xp.saturating_add(outer + 1).min(width);
    let mut patch = Vec::new();
    let mut positions = Vec::new();
    let mut distances = Vec::new();
    let mut aperture = Vec::new();
    let mut geometry = Vec::new();
    let mut patch_mask = Vec::new();
    let mut annulus_values = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let index = y * width + x;
            let value = image.value(index);
            let distance = (x as f64 - xp as f64).hypot(y as f64 - yp as f64);
            let masked = ctx.mask[index] || !value.is_finite();
            let geometric = distance <= radius as f64;
            patch.push(value);
            positions.push([x as f64, y as f64]);
            distances.push(distance);
            geometry.push(geometric);
            patch_mask.push(masked);
            aperture.push(geometric && !masked);
            if distance >= (radius + 2) as f64 && distance <= outer as f64 && !masked {
                annulus_values.push(value);
            }
        }
    }
    let annulus_usable = annulus_values.len() >= 16;
    let (background, local_noise) = if annulus_usable && options.refine_local_background {
        if float32 {
            statistics32::sigma_clipped_stats(
                &annulus_values.iter().map(|v| *v as f32).collect::<Vec<_>>(),
                None,
                3.0,
                4,
                100_000,
            )?
        } else {
            statistics::sigma_clipped_stats(&annulus_values, None, 3.0, 4, 100_000)?
        }
    } else {
        (
            ctx.background_map.value(yp * width + xp),
            ctx.noise_map.value(yp * width + xp),
        )
    };
    let noise = local_noise.max(f64::EPSILON);
    let residual: Vec<_> = patch
        .iter()
        .map(|p| scalar(p - scalar(background)))
        .collect();
    let valid_flux: Vec<_> = residual
        .iter()
        .zip(&aperture)
        .filter(|(_, ok)| **ok)
        .map(|(v, _)| *v)
        .collect();
    let flux = reduce(&valid_flux);
    let aperture_pixels = aperture.iter().filter(|v| **v).count();
    let background_pixels = annulus_values.len().max(1);
    let source_variance = options
        .gain_e_per_adu
        .map(|g| flux.max(0.0) / g)
        .unwrap_or(0.0)
        + aperture_pixels as f64 * options.read_noise_adu.powi(2);
    let background_variance = aperture_pixels as f64 * noise.powi(2)
        + (aperture_pixels as f64).powi(2) / background_pixels as f64 * noise.powi(2);
    let flux_error = (source_variance + background_variance)
        .max(f64::EPSILON)
        .sqrt();
    let flux_snr = flux / flux_error;
    let positive: Vec<_> = residual
        .iter()
        .zip(&aperture)
        .map(|(v, ok)| if *ok { v.max(0.0) } else { 0.0 })
        .collect();
    let positive_sum = reduce(&positive);
    let (mut x, mut y) = (xp as f64, yp as f64);
    let (mut fwhm, mut fwhm_x, mut fwhm_y, mut ellipticity, mut sharpness) =
        (None, None, None, None, None);
    if positive_sum > 0.0 {
        let sigma = (options.psf_fwhm / 2.35482).max(0.5);
        let centroid_radius = (radius as f64).min((2.5 * sigma).max(2.0));
        let mut weights: Vec<_> = residual
            .iter()
            .zip(&aperture)
            .zip(&distances)
            .map(|((r, ok), d)| {
                if *ok && *d <= centroid_radius {
                    scalar(
                        scalar(r - scalar(0.5 * noise)).max(0.0)
                            * (-0.5 * (d / sigma).powi(2)).exp(),
                    )
                } else {
                    0.0
                }
            })
            .collect();
        let mut sum = reduce(&weights);
        if sum <= f64::EPSILON {
            weights = positive.clone();
            sum = positive_sum;
        }
        let centroid_x: Vec<_> = weights
            .iter()
            .zip(&positions)
            .map(|(w, p)| w * p[0])
            .collect();
        let centroid_y: Vec<_> = weights
            .iter()
            .zip(&positions)
            .map(|(w, p)| w * p[1])
            .collect();
        x = statistics::pairwise_sum(&centroid_x) / sum;
        y = statistics::pairwise_sum(&centroid_y) / sum;
        let shape_x_terms: Vec<_> = positive
            .iter()
            .zip(&positions)
            .map(|(w, p)| w * p[0])
            .collect();
        let shape_y_terms: Vec<_> = positive
            .iter()
            .zip(&positions)
            .map(|(w, p)| w * p[1])
            .collect();
        let shape_x = statistics::pairwise_sum(&shape_x_terms) / positive_sum;
        let shape_y = statistics::pairwise_sum(&shape_y_terms) / positive_sum;
        let variance_x_terms: Vec<_> = positive
            .iter()
            .zip(&positions)
            .map(|(w, p)| w * (p[0] - shape_x).powi(2))
            .collect();
        let variance_y_terms: Vec<_> = positive
            .iter()
            .zip(&positions)
            .map(|(w, p)| w * (p[1] - shape_y).powi(2))
            .collect();
        let vx = statistics::pairwise_sum(&variance_x_terms) / positive_sum;
        let vy = statistics::pairwise_sum(&variance_y_terms) / positive_sum;
        let fx = 2.35482 * vx.max(0.0).sqrt();
        let fy = 2.35482 * vy.max(0.0).sqrt();
        let fm = 2.35482 * ((vx + vy) / 2.0).max(0.0).sqrt();
        fwhm = fm.is_finite().then_some(fm).filter(|v| *v > 0.0);
        fwhm_x = Some(fx);
        fwhm_y = Some(fy);
        ellipticity = Some((fx - fy).abs() / fx.max(fy).max(f64::EPSILON));
        sharpness = Some(
            residual
                .iter()
                .zip(&aperture)
                .filter(|(_, v)| **v)
                .map(|(r, _)| *r)
                .fold(f64::NEG_INFINITY, f64::max)
                .max(0.0)
                / positive_sum.max(f64::EPSILON),
        );
    }
    let peak = image.value(yp * width + xp);
    let snr = (peak - background) / noise;
    let footprint = residual
        .iter()
        .zip(&aperture)
        .filter(|(r, ok)| **ok && **r >= scalar(noise))
        .count();
    let support_threshold = (2.0 * noise).max(0.1 * (peak - background).max(0.0));
    let support = positions
        .iter()
        .zip(&patch_mask)
        .zip(&residual)
        .filter(|((p, m), r)| {
            !**m && (p[0] - xp as f64).abs() <= 1.0
                && (p[1] - yp as f64).abs() <= 1.0
                && r.is_finite()
                && **r >= scalar(support_threshold)
        })
        .count();
    let repeated_count = patch
        .iter()
        .zip(&aperture)
        .filter(|(v, ok)| {
            **ok && options
                .repeated_high_code_values
                .iter()
                .any(|code| **v == *code as f64)
        })
        .count();
    let range_limit = (-1000.0f64).min(background - 50.0 * noise);
    let range_count = patch
        .iter()
        .zip(&aperture)
        .filter(|(v, ok)| **ok && **v <= scalar(range_limit))
        .count();
    let peak_repeated = options
        .repeated_high_code_values
        .contains(&(peak.round_ties_even() as i64));
    let mut flags: Vec<String> = Vec::new();
    if x0 == 0 || y0 == 0 || x1 == width || y1 == height {
        flags.push("EDGE".into());
    }
    let masked_indices: Vec<_> = patch_mask
        .iter()
        .zip(&geometry)
        .enumerate()
        .filter(|(_, (m, g))| **m && **g)
        .map(|(i, _)| i)
        .collect();
    if !masked_indices.is_empty() {
        let geometric_pixels = geometry.iter().filter(|v| **v).count();
        let minimum = (0.8 * geometric_pixels as f64).ceil().max(9.0) as usize;
        let center_valid = !ctx.mask[yp * width + xp] && peak.is_finite();
        let zero_only = masked_indices
            .iter()
            .all(|i| patch[*i].is_finite() && patch[*i] == 0.0);
        flags.push(
            if options.allow_partial_zero_mask
                && center_valid
                && zero_only
                && aperture_pixels >= minimum
            {
                "PARTIAL_MASKED"
            } else {
                "MASKED"
            }
            .into(),
        );
    }
    if options.saturation_level.is_some_and(|s| {
        patch
            .iter()
            .zip(&geometry)
            .any(|(v, g)| *g && v.is_finite() && *v >= scalar(s))
    }) {
        flags.push("SATURATED".into());
    }
    if options.negative_overflow_limit.is_some_and(|limit| {
        patch
            .iter()
            .zip(&geometry)
            .any(|(v, g)| *g && v.is_finite() && *v <= scalar(limit))
    }) {
        flags.push("NEGATIVE_OVERFLOW".into());
    }
    if peak_repeated && range_count > 0 {
        flags.push("CODE_PATTERN".into());
    }
    if ctx.line_mask.is_some_and(|line| {
        positions
            .iter()
            .zip(&geometry)
            .any(|(p, g)| *g && line[p[1] as usize * width + p[0] as usize])
    }) {
        flags.push("LINE_ARTIFACT".into());
    }
    if !annulus_usable {
        flags.push("BACKGROUND_UNCERTAIN".into());
    }
    if flux <= 0.0 {
        flags.push("NON_POSITIVE_FLUX".into());
    }
    if !flux_snr.is_finite() || flux_snr < options.min_flux_snr {
        flags.push("LOW_FLUX_SNR".into());
    }
    if let (Some(fm), Some(_), Some(_)) = (fwhm, fwhm_x, fwhm_y) {
        if fm < options.min_fwhm {
            flags.push("NARROW".into());
        }
        if fm > options.max_fwhm {
            flags.push("BROAD".into());
        }
        if ellipticity.is_some_and(|e| e > options.max_ellipticity) {
            flags.push("ELONGATED".into());
        }
    } else {
        flags.push("NO_SHAPE".into());
    }
    if sharpness.is_none_or(|s| s < options.min_sharpness) {
        flags.push("DIFFUSE".into());
    } else if sharpness.is_some_and(|s| s > options.max_sharpness) {
        flags.push("SPIKE".into());
    }
    if footprint < options.min_footprint_pixels {
        flags.push("SMALL_FOOTPRINT".into());
    }
    if support < options.min_psf_support_pixels {
        flags.push("INSUFFICIENT_PSF_SUPPORT".into());
    }
    if !proposal.methods.iter().any(|m| m == "gaussian")
        && proposal
            .nearest_gaussian_px
            .is_some_and(|d| d < options.dog_blend_radius_px)
    {
        flags.push("UNRESOLVED_BLEND".into());
    }
    let passed = quality_passes(&flags);
    Ok(Detection {
        detection_id: -1,
        x,
        y,
        peak,
        flux,
        background,
        noise,
        snr,
        fwhm,
        flags,
        flux_error: Some(flux_error),
        flux_snr: Some(flux_snr),
        filter_snr: Some(filter_snr),
        fwhm_x,
        fwhm_y,
        ellipticity,
        sharpness,
        footprint_pixels: Some(footprint),
        psf_support_pixels: Some(support),
        quality_passed: passed,
        peak_x: Some(xp as f64),
        peak_y: Some(yp as f64),
        centroid_shift_px: Some((x - xp as f64).hypot(y - yp as f64)),
        proposal_methods: proposal.methods.clone(),
        proposal_scales: proposal.scales.clone(),
        proposal_snr: Some(proposal.score),
        nearest_gaussian_px: proposal.nearest_gaussian_px,
        deblend_delta_bic: None,
        deblend_component_snr: None,
        repeated_code_count: Some(repeated_count),
        range_anomaly_pixel_count: Some(range_count),
        repeated_code_values: options.repeated_high_code_values.clone(),
    })
}
