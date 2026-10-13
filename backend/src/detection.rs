//! End-to-end detector: proposals remain separate from source quality and PSF evidence.
use crate::{
    filtering::{self, ProposalOptions},
    measurement::{self, Detection, MeasurementContext, MeasurementOptions},
    models::PixelData,
    precision::Calculation,
    psf::{self, PairGuardOptions},
    statistics::BackgroundOptions,
    Result,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectionOptions {
    #[serde(flatten)]
    pub measurement: MeasurementOptions,
    pub threshold_sigma: f64,
    pub min_distance: usize,
    pub max_sources: Option<usize>,
    pub background_box_size: usize,
    pub background_sample_limit: usize,
    pub mask_zero_pixels: Option<bool>,
    pub allow_partial_zero_mask: Option<bool>,
    pub reject_linear_artifacts: bool,
    pub proposal_mode: String,
    pub dog_threshold_sigma: Option<f64>,
    pub dog_min_peak_sigma: f64,
    pub dog_blend_radius_factor: f64,
    pub dog_scale_factors: [f64; 2],
    pub starlet_threshold_sigma: Option<f64>,
    pub starlet_min_peak_sigma: f64,
    pub deblend_delta_bic_min: f64,
    pub deblend_component_snr_min: f64,
    pub deblend_primary_snr_min: f64,
    pub deblend_min_residual_sigma: f64,
    pub deblend_search_radius_factor: f64,
    pub enable_local_deblend: bool,
    pub use_float32: bool,
    pub fast_sequence: bool,
}
impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            measurement: MeasurementOptions::default(),
            threshold_sigma: 4.0,
            min_distance: 3,
            max_sources: None,
            background_box_size: 128,
            background_sample_limit: 100_000,
            mask_zero_pixels: None,
            allow_partial_zero_mask: None,
            reject_linear_artifacts: true,
            proposal_mode: "gaussian".into(),
            dog_threshold_sigma: None,
            dog_min_peak_sigma: 2.0,
            dog_blend_radius_factor: 2.5,
            dog_scale_factors: [0.75, 1.60],
            starlet_threshold_sigma: None,
            starlet_min_peak_sigma: 2.5,
            deblend_delta_bic_min: 10.0,
            deblend_component_snr_min: 5.0,
            deblend_primary_snr_min: 12.0,
            deblend_min_residual_sigma: 4.0,
            deblend_search_radius_factor: 2.0,
            enable_local_deblend: false,
            use_float32: false,
            fast_sequence: false,
        }
    }
}
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub image_shape: [usize; 2],
    pub background: f64,
    pub noise: f64,
    pub threshold: f64,
    pub candidate_count: usize,
    pub sources: Vec<Detection>,
    pub quality_count: usize,
    pub parameters: Value,
    pub candidate_peaks: Vec<[f64; 3]>,
    pub line_artifact_coordinates: Vec<[usize; 2]>,
}
pub struct BackgroundModel<'a> {
    pub background: &'a [f64],
    pub noise: &'a [f64],
    pub height: usize,
    pub width: usize,
}
impl DetectionResult {
    pub fn as_dict(&self) -> Value {
        json!({"image_shape":self.image_shape,"background":self.background,"noise":self.noise,"threshold":self.threshold,"candidate_count":self.candidate_count,"returned_count":self.sources.len(),"quality_count":self.quality_count,"rejected_count":self.sources.len()-self.quality_count,"truncated":self.candidate_count>self.sources.len(),"parameters":self.parameters,"sources":self.sources})
    }
}
fn integer_limits(data: &PixelData) -> Option<(f64, f64, bool)> {
    match data {
        PixelData::U8(_) => Some((0., 255., false)),
        PixelData::I16(_) => Some((i16::MIN as f64, i16::MAX as f64, true)),
        PixelData::I32(_) => Some((i32::MIN as f64, i32::MAX as f64, true)),
        PixelData::I64(_) => Some((i64::MIN as f64, i64::MAX as f64, true)),
        _ => None,
    }
}
pub fn repeated_high_code_values(
    data: &PixelData,
    mask: &[bool],
    background: f64,
    noise: f64,
    minimum_count: usize,
) -> Result<Vec<i64>> {
    if integer_limits(data).is_none_or(|(_, _, signed)| !signed) {
        return Ok(vec![]);
    }
    if mask.len() != data.len() {
        return Err("mask shape does not match image for repeated-code audit".into());
    }
    if minimum_count == 0 || !background.is_finite() || !noise.is_finite() {
        return Ok(vec![]);
    }
    let floor = 500.0f64.max(background + 20.0 * noise.max(1e-12));
    let mut counts = BTreeMap::new();
    for (i, invalid) in mask.iter().enumerate() {
        if *invalid || data.value(i) < floor {
            continue;
        }
        let value = match data {
            PixelData::I16(v) => v[i] as i64,
            PixelData::I32(v) => v[i] as i64,
            PixelData::I64(v) => v[i],
            _ => unreachable!(),
        };
        *counts.entry(value).or_insert(0usize) += 1;
    }
    Ok(counts
        .into_iter()
        .filter(|(_, n)| *n >= minimum_count)
        .map(|(v, _)| v)
        .collect())
}
pub fn working_mask(
    data: &PixelData,
    provided: Option<&[bool]>,
    background: f64,
    mask_zero_pixels: Option<bool>,
    saturation: Option<f64>,
) -> Result<(Vec<bool>, Option<f64>, bool)> {
    if provided.is_some_and(|m| m.len() != data.len()) {
        return Err("mask shape does not match image shape".into());
    }
    let limits = integer_limits(data);
    let zeros = mask_zero_pixels
        .unwrap_or(limits.is_some() && background.is_finite() && background.abs() > 1.0);
    let inferred = saturation.or_else(|| limits.map(|(_, max, _)| max - 32.0));
    let mask = (0..data.len())
        .map(|i| {
            let value = data.value(i);
            provided.is_some_and(|m| m[i])
                || !value.is_finite()
                || (zeros && value == 0.0)
                || if saturation.is_none() {
                    limits.is_some_and(|(min, max, _)| value >= max - 32.0 || value <= min + 32.0)
                } else {
                    saturation.is_some_and(|s| value >= s)
                }
        })
        .collect();
    Ok((mask, inferred, zeros))
}

/// Build the original-frame mask with the float32 comparison values used by
/// Python sequence candidate forced-photometry gates.
pub fn working_mask_float32(
    data: &PixelData,
    provided: Option<&[bool]>,
    background: f64,
    mask_zero_pixels: Option<bool>,
    saturation: Option<f64>,
) -> Result<(Vec<bool>, Option<f64>, bool)> {
    let numeric = (0..data.len())
        .map(|index| data.value(index) as f32)
        .collect::<Vec<_>>();
    working_mask_typed(
        data,
        &numeric,
        provided,
        background,
        mask_zero_pixels,
        saturation,
    )
}

fn validate(options: &DetectionOptions) -> Result<()> {
    let m = &options.measurement;
    if options.threshold_sigma <= 0.0
        || options.min_distance < 1
        || m.aperture_radius < 1
        || m.psf_fwhm <= 0.0
        || m.min_flux_snr <= 0.0
        || options.background_box_size < 16
        || options.background_sample_limit < 1
    {
        return Err("invalid detector threshold, spacing, PSF or background parameters".into());
    }
    if m.min_fwhm <= 0.0
        || m.max_fwhm < m.min_fwhm
        || !(0.0..=1.0).contains(&m.max_ellipticity)
        || m.min_sharpness < 0.0
        || m.max_sharpness <= m.min_sharpness
        || m.min_footprint_pixels < 1
        || !(1..=9).contains(&m.min_psf_support_pixels)
    {
        return Err("invalid source quality parameters".into());
    }
    if m.gain_e_per_adu.is_some_and(|g| g <= 0.0)
        || m.read_noise_adu < 0.0
        || m.negative_overflow_limit.is_some_and(|v| !v.is_finite())
    {
        return Err("invalid gain, read noise or negative-overflow limit".into());
    }
    if !["gaussian", "hybrid", "ensemble"]
        .contains(&options.proposal_mode.trim().to_lowercase().as_str())
    {
        return Err("invalid proposal_mode".into());
    }
    if options.dog_threshold_sigma.is_some_and(|v| v <= 0.0)
        || options.starlet_threshold_sigma.is_some_and(|v| v <= 0.0)
        || options.dog_min_peak_sigma <= 0.0
        || options.starlet_min_peak_sigma <= 0.0
        || options.dog_blend_radius_factor <= 0.0
        || !(options.dog_scale_factors[0] > 0.0
            && options.dog_scale_factors[0] < 1.0
            && options.dog_scale_factors[1] > 1.0)
    {
        return Err("invalid multiscale proposal parameters".into());
    }
    if [
        options.deblend_delta_bic_min,
        options.deblend_component_snr_min,
        options.deblend_primary_snr_min,
        options.deblend_min_residual_sigma,
        options.deblend_search_radius_factor,
    ]
    .iter()
    .any(|v| *v <= 0.0)
    {
        return Err("local deblend thresholds must be positive".into());
    }
    if options.max_sources == Some(0) {
        return Err("max_sources must be positive when provided".into());
    }
    // Explicitly reject pending execution paths; never silently use a different algorithm.
    Ok(())
}
fn working_mask_typed<T: Calculation>(
    data: &PixelData,
    numeric: &[T],
    provided: Option<&[bool]>,
    background: f64,
    mask_zero: Option<bool>,
    saturation: Option<f64>,
) -> Result<(Vec<bool>, Option<f64>, bool)> {
    if provided.is_some_and(|m| m.len() != data.len()) {
        return Err("mask shape mismatch".into());
    }
    let limits = integer_limits(data);
    let zeros =
        mask_zero.unwrap_or(limits.is_some() && background.is_finite() && background.abs() > 1.0);
    let inferred = saturation.or_else(|| limits.map(|(_, max, _)| max - 32.0));
    let mask = numeric
        .iter()
        .enumerate()
        .map(|(i, value)| {
            provided.is_some_and(|m| m[i])
                || !value.finite()
                || (zeros && *value == T::from_f64(0.0))
                || if saturation.is_none() {
                    limits.is_some_and(|(min, max, _)| {
                        *value >= T::from_f64(max - 32.0) || *value <= T::from_f64(min + 32.0)
                    })
                } else {
                    saturation.is_some_and(|limit| *value >= T::from_f64(limit))
                }
        })
        .collect();
    Ok((mask, inferred, zeros))
}

pub fn detect_sources(
    data: &PixelData,
    provided: Option<&[bool]>,
    height: usize,
    width: usize,
    options: &DetectionOptions,
    background_model: Option<&BackgroundModel<'_>>,
) -> Result<DetectionResult> {
    if options.use_float32 && !matches!(data, PixelData::F64(_)) {
        detect_typed::<f32>(data, provided, height, width, options, background_model)
    } else {
        detect_typed::<f64>(data, provided, height, width, options, background_model)
    }
}

/// Build the same reusable local background model used by the sequence fast
/// path without running detection. The pilot model is intentionally based on
/// the first frame only and is consumed by later per-frame analyses.
pub fn build_background_model(
    data: &PixelData,
    provided: Option<&[bool]>,
    height: usize,
    width: usize,
    options: &DetectionOptions,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if height.checked_mul(width) != Some(data.len())
        || height == 0
        || width == 0
        || provided.is_some_and(|mask| mask.len() != data.len())
    {
        return Err("invalid background-model image or mask shape".into());
    }
    if options.use_float32 && !matches!(data, PixelData::F64(_)) {
        build_background_typed::<f32>(data, provided, height, width, options)
    } else {
        build_background_typed::<f64>(data, provided, height, width, options)
    }
}

fn build_background_typed<T: Calculation>(
    data: &PixelData,
    provided: Option<&[bool]>,
    height: usize,
    width: usize,
    options: &DetectionOptions,
) -> Result<(Vec<f64>, Vec<f64>)> {
    validate(options)?;
    let numeric: Vec<T> = (0..data.len())
        .map(|index| T::from_f64(data.value(index)))
        .collect();
    let base_mask: Vec<_> = numeric
        .iter()
        .enumerate()
        .map(|(index, value)| !value.finite() || provided.is_some_and(|mask| mask[index]))
        .collect();
    let (global_background, _) = T::stats(&numeric, Some(&base_mask), 3.0, 4, 1_000_000)?;
    let (effective, _, _) = working_mask_typed(
        data,
        &numeric,
        provided,
        global_background,
        options.mask_zero_pixels,
        options.measurement.saturation_level,
    )?;
    let (background, noise) = T::stats(&numeric, Some(&effective), 3.0, 4, 1_000_000)?;
    let (background_map, noise_map) = T::background(
        &numeric,
        &effective,
        height,
        width,
        BackgroundOptions {
            box_size: options.background_box_size,
            fallback: Some((background, noise)),
            iterations: if options.measurement.refine_local_background {
                4
            } else {
                2
            },
            sample_limit: options.background_sample_limit,
        },
    )?;
    Ok((
        background_map.into_iter().map(T::to_f64).collect(),
        noise_map.into_iter().map(T::to_f64).collect(),
    ))
}

fn detect_typed<T: Calculation>(
    data: &PixelData,
    provided: Option<&[bool]>,
    height: usize,
    width: usize,
    options: &DetectionOptions,
    background_model: Option<&BackgroundModel<'_>>,
) -> Result<DetectionResult> {
    validate(options)?;
    if height.checked_mul(width) != Some(data.len())
        || height == 0
        || width == 0
        || provided.is_some_and(|m| m.len() != data.len())
    {
        return Err("invalid detection image or mask shape".into());
    }
    let numeric: Vec<T> = (0..data.len())
        .map(|i| T::from_f64(data.value(i)))
        .collect();
    let base_mask: Vec<_> = numeric
        .iter()
        .enumerate()
        .map(|(i, v)| !v.finite() || provided.is_some_and(|m| m[i]))
        .collect();
    let (global_background, global_noise) =
        T::stats(&numeric, Some(&base_mask), 3.0, 4, 1_000_000)?;
    let (effective, saturation, used_zero) = working_mask_typed(
        data,
        &numeric,
        provided,
        global_background,
        options.mask_zero_pixels,
        options.measurement.saturation_level,
    )?;
    let (background, noise) = T::stats(&numeric, Some(&effective), 3.0, 4, 1_000_000)?;
    let repeated = repeated_high_code_values(data, &effective, background, noise, 128)?;
    let mut measurement = options.measurement.clone();
    measurement.saturation_level = saturation;
    measurement.allow_partial_zero_mask = options.allow_partial_zero_mask.unwrap_or(used_zero);
    measurement.negative_overflow_limit = measurement.negative_overflow_limit.or_else(|| {
        integer_limits(data)
            .filter(|(_, _, signed)| *signed)
            .map(|(_, max, _)| -0.9 * max)
    });
    measurement.repeated_high_code_values = repeated.clone();
    measurement.dog_blend_radius_px = options.dog_blend_radius_factor * measurement.psf_fwhm;
    let mut background_mode = "per_frame_local";
    let (background_map, noise_map) = if let Some(model) = background_model.filter(|m| {
        m.height == height
            && m.width == width
            && m.background.len() == numeric.len()
            && m.noise.len() == numeric.len()
    }) {
        background_mode = "shared_sequence_pilot";
        (
            model.background.iter().map(|v| T::from_f64(*v)).collect(),
            model.noise.iter().map(|v| T::from_f64(*v)).collect(),
        )
    } else {
        if background_model.is_some() {
            background_mode = "per_frame_local_shape_fallback";
        }
        T::background(
            &numeric,
            &effective,
            height,
            width,
            BackgroundOptions {
                box_size: options.background_box_size,
                fallback: Some((background, noise)),
                iterations: if measurement.refine_local_background {
                    4
                } else {
                    2
                },
                sample_limit: options.background_sample_limit,
            },
        )?
    };
    let valid: Vec<_> = effective.iter().map(|v| !*v).collect();
    let residual: Vec<_> = numeric
        .iter()
        .zip(&background_map)
        .zip(&valid)
        .map(|((v, b), ok)| if *ok { *v - *b } else { T::from_f64(0.0) })
        .collect();
    let line = if options.reject_linear_artifacts {
        Some(T::lines(
            &residual,
            &noise_map,
            &valid,
            height,
            width,
            options.threshold_sigma,
            measurement.psf_fwhm,
        )?)
    } else {
        None
    };
    let sigma = measurement.psf_fwhm / 2.35482;
    let (filtered, filter_mode) = T::gaussian(
        &residual,
        &valid,
        height,
        width,
        sigma,
        options.fast_sequence,
    )?;
    let (filter_snr, _, filter_noise) =
        T::standardize(&filtered, &effective, &noise_map, &valid, 1_000_000)?;
    let proposal =
        |values: &[T], threshold: f64, scale: f64, method: &str, gate: Option<&[bool]>| {
            T::proposals(
                values,
                &valid,
                height,
                width,
                &ProposalOptions {
                    threshold_sigma: threshold,
                    min_distance: options.min_distance,
                    scale_fwhm: scale,
                    method: method.into(),
                },
                gate,
            )
        };
    let gaussian = proposal(
        &filter_snr,
        options.threshold_sigma,
        measurement.psf_fwhm,
        "gaussian",
        None,
    )?;
    let dog_threshold = options
        .dog_threshold_sigma
        .unwrap_or(6.0f64.max(options.threshold_sigma + 2.0));
    let starlet_threshold = options
        .starlet_threshold_sigma
        .unwrap_or(7.0f64.max(options.threshold_sigma + 3.0));
    let mode = options.proposal_mode.trim().to_lowercase();
    let mut extras = Vec::new();
    let (mut dog_narrow, mut dog_broad, mut starlet1, mut starlet2) = (0, 0, 0, 0);
    let maxima = if mode != "gaussian" {
        Some(T::maximum(&residual, height, width, 1)?)
    } else {
        None
    };
    if mode == "hybrid" || mode == "ensemble" {
        let gate: Vec<_> = residual
            .iter()
            .enumerate()
            .map(|(i, r)| {
                valid[i]
                    && *r == maxima.as_ref().unwrap()[i]
                    && *r >= T::from_f64(options.dog_min_peak_sigma) * noise_map[i]
            })
            .collect();
        for (factor, method) in [
            (options.dog_scale_factors[0], "dog_narrow"),
            (options.dog_scale_factors[1], "dog_broad"),
        ] {
            let (other, _) = T::gaussian(
                &residual,
                &valid,
                height,
                width,
                sigma * factor,
                options.fast_sequence,
            )?;
            let difference: Vec<_> = other
                .iter()
                .zip(&filtered)
                .map(|(a, b)| if factor < 1.0 { *a - *b } else { *b - *a })
                .collect();
            let (snr, _, _) =
                T::standardize(&difference, &effective, &noise_map, &valid, 1_000_000)?;
            let candidates = proposal(
                &snr,
                dog_threshold,
                measurement.psf_fwhm * factor.sqrt(),
                method,
                Some(&gate),
            )?;
            if factor < 1.0 {
                dog_narrow = candidates.len();
            } else {
                dog_broad = candidates.len();
            }
            extras.extend(candidates);
        }
    }
    if mode == "ensemble" {
        let gate: Vec<_> = residual
            .iter()
            .enumerate()
            .map(|(i, r)| {
                valid[i]
                    && *r == maxima.as_ref().unwrap()[i]
                    && *r >= T::from_f64(options.starlet_min_peak_sigma) * noise_map[i]
            })
            .collect();
        let (c1, _) = T::starlet(&residual, &valid, height, width, 1, options.fast_sequence)?;
        let (c2, _) = T::starlet(&c1, &valid, height, width, 2, options.fast_sequence)?;
        for (base, smooth, scale, method) in [
            (&residual, &c1, 1.7, "starlet_s1"),
            (&c1, &c2, 3.5, "starlet_s2"),
        ] {
            let difference: Vec<_> = base.iter().zip(smooth).map(|(a, b)| *a - *b).collect();
            let (snr, _, _) =
                T::standardize(&difference, &effective, &noise_map, &valid, 1_000_000)?;
            let candidates = proposal(&snr, starlet_threshold, scale, method, Some(&gate))?;
            if method == "starlet_s1" {
                starlet1 = candidates.len();
            } else {
                starlet2 = candidates.len();
            }
            extras.extend(candidates);
        }
    }
    let mut selected = filtering::merge_proposals(&gaussian, &extras, options.min_distance);
    let deblend_proposals = if options.enable_local_deblend && !options.fast_sequence {
        let ctx = crate::deblend::DeblendContext {
            image: T::plane(&numeric),
            valid: &valid,
            line_mask: line.as_deref(),
            background: T::plane(&background_map),
            noise: T::plane(&noise_map),
            height,
            width,
        };
        let candidates = crate::deblend::local_deblend_proposals(
            &ctx,
            &gaussian,
            crate::deblend::DeblendOptions {
                psf_fwhm: measurement.psf_fwhm,
                min_primary_snr: options.deblend_primary_snr_min,
                min_residual_sigma: options.deblend_min_residual_sigma,
                search_radius_factor: options.deblend_search_radius_factor,
            },
        )?;
        selected = crate::deblend::append_deblend_proposals(
            &selected,
            &candidates,
            &gaussian,
            options.min_distance,
        );
        candidates
    } else {
        vec![]
    };
    let candidate_count = selected.len();
    let candidate_peaks = if options.fast_sequence {
        selected
            .iter()
            .map(|p| [p.x as f32 as f64, p.y as f32 as f64, p.score as f32 as f64])
            .collect()
    } else {
        vec![]
    };
    if let Some(limit) = options.max_sources {
        selected.truncate(limit);
    }
    let ctx = MeasurementContext {
        image: T::plane(&numeric),
        mask: &effective,
        line_mask: line.as_deref(),
        background_map: T::plane(&background_map),
        noise_map: T::plane(&noise_map),
        height,
        width,
    };
    let sources: Result<Vec<_>> = selected
        .iter()
        .map(|p| {
            measurement::source_from_peak(
                &ctx,
                [p.x, p.y],
                &measurement,
                p,
                filter_snr[p.y * width + p.x].to_f64(),
            )
        })
        .collect();
    let pair_options = PairGuardOptions {
        psf_fwhm: measurement.psf_fwhm,
        delta_bic_min: options.deblend_delta_bic_min,
        component_snr_min: options.deblend_component_snr_min,
    };
    let sources = psf::resolve_supplemental_blends_typed(
        &sources?,
        T::plane(&numeric),
        &effective,
        height,
        width,
        pair_options,
    )?;
    let (mut sources, pair_tested, pair_guarded, pair_radius) =
        psf::guard_unresolved_gaussian_pairs_typed(
            &sources,
            T::plane(&numeric),
            &effective,
            height,
            width,
            pair_options,
        )?;
    sources.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));
    for (i, s) in sources.iter_mut().enumerate() {
        s.detection_id = i as i64;
    }
    let quality_count = sources.iter().filter(|s| s.quality_passed).count();
    let mut parameters = json!({"threshold_sigma":options.threshold_sigma,"min_distance":options.min_distance,"aperture_radius":measurement.aperture_radius,"max_sources":options.max_sources.map(|v|v as i64).unwrap_or(-1),"psf_fwhm":measurement.psf_fwhm,"background_box_size":options.background_box_size,"background_sample_limit":options.background_sample_limit,"statistics_sample_limit":1_000_000,"min_flux_snr":measurement.min_flux_snr,"min_fwhm":measurement.min_fwhm,"max_fwhm":measurement.max_fwhm,"max_ellipticity":measurement.max_ellipticity,"min_sharpness":measurement.min_sharpness,"max_sharpness":measurement.max_sharpness,"min_footprint_pixels":measurement.min_footprint_pixels,"min_psf_support_pixels":measurement.min_psf_support_pixels,"gain_e_per_adu":measurement.gain_e_per_adu.unwrap_or(-1.0),"read_noise_adu":measurement.read_noise_adu,"mask_zero_pixels":used_zero as u8,"allow_partial_zero_mask":measurement.allow_partial_zero_mask as u8,"partial_mask_policy":if measurement.allow_partial_zero_mask{"exact_zero_center_valid_aperture_ge_80pct"}else{"disabled"},"reject_linear_artifacts":options.reject_linear_artifacts as u8,"proposal_mode":mode,"dog_threshold_sigma":dog_threshold,"dog_min_peak_sigma":options.dog_min_peak_sigma,"dog_blend_radius_factor":options.dog_blend_radius_factor,"dog_blend_radius_px":measurement.dog_blend_radius_px,"dog_narrow_factor":options.dog_scale_factors[0],"dog_broad_factor":options.dog_scale_factors[1],"starlet_threshold_sigma":starlet_threshold,"starlet_min_peak_sigma":options.starlet_min_peak_sigma,"deblend_delta_bic_min":options.deblend_delta_bic_min,"deblend_component_snr_min":options.deblend_component_snr_min,"deblend_primary_snr_min":options.deblend_primary_snr_min,"deblend_min_residual_sigma":options.deblend_min_residual_sigma,"deblend_search_radius_factor":options.deblend_search_radius_factor,"enable_local_deblend":options.enable_local_deblend as u8,"repeated_high_code_values":repeated,"repeated_high_code_min_count":128,"code_pattern_negative_limit_adu":-1000.0,"close_pair_guard_radius_px":pair_radius,"close_pair_guard_tested_count":pair_tested,"close_pair_guarded_source_count":pair_guarded,"proposal_count_deblend_local":deblend_proposals.len(),"proposal_count_gaussian":gaussian.len(),"proposal_count_dog_narrow":dog_narrow,"proposal_count_dog_broad":dog_broad,"proposal_count_starlet_s1":starlet1,"proposal_count_starlet_s2":starlet2,"proposal_count_merged":candidate_count,"refine_local_background":measurement.refine_local_background as u8,"background_clip_iterations":if measurement.refine_local_background{4}else{2},"fast_sequence":options.fast_sequence as u8,"masked_filter_mode":filter_mode,"background_stats_mode":if T::FLOAT32&&!measurement.refine_local_background{"sampled_grid_4096"}else{"full_grid"},"background_model_mode":background_mode,"use_float32":T::FLOAT32 as u8,"line_artifact_pixels":line.as_ref().map(|m|m.iter().filter(|v|**v).count()).unwrap_or(0)});
    let metadata = parameters.as_object_mut().unwrap();
    metadata.insert("saturation_level".into(), json!(saturation.unwrap_or(-1.0)));
    metadata.insert(
        "negative_overflow_limit".into(),
        json!(measurement.negative_overflow_limit),
    );
    metadata.insert("global_background".into(), json!(global_background));
    metadata.insert("global_noise".into(), json!(global_noise));
    metadata.insert("filter_noise".into(), json!(filter_noise));
    metadata.insert("centroid_method".into(), json!("psf_weighted_noise_floor"));
    let line_artifact_coordinates = line
        .as_ref()
        .map(|m| {
            m.iter()
                .enumerate()
                .filter(|(_, v)| **v)
                .map(|(i, _)| [i % width, i / width])
                .collect()
        })
        .unwrap_or_default();
    Ok(DetectionResult {
        image_shape: [height, width],
        background,
        noise,
        threshold: background + options.threshold_sigma * noise,
        candidate_count,
        sources,
        quality_count,
        parameters,
        candidate_peaks,
        line_artifact_coordinates,
    })
}
