//! Temporal-reference candidate proposals before ordinary frame consensus.
use crate::{filtering32, statistics32, Result};
use crate::{models::PixelData, pipeline::FrameAnalysis};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
struct Plane {
    values: Vec<f32>,
    height: usize,
    width: usize,
}

#[derive(Debug, Clone)]
enum FrameImage<'a> {
    Owned(Plane),
    Borrowed {
        pixels: &'a PixelData,
        height: usize,
        width: usize,
    },
}

impl FrameImage<'_> {
    fn height(&self) -> usize {
        match self {
            Self::Owned(plane) => plane.height,
            Self::Borrowed { height, .. } => *height,
        }
    }

    fn width(&self) -> usize {
        match self {
            Self::Owned(plane) => plane.width,
            Self::Borrowed { width, .. } => *width,
        }
    }

    fn value(&self, index: usize) -> Option<f32> {
        match self {
            Self::Owned(plane) => plane.values.get(index).copied(),
            Self::Borrowed { pixels, .. } => {
                (index < pixels.len()).then(|| pixels.value(index) as f32)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Frame<'a> {
    image: FrameImage<'a>,
    background: f32,
    noise: f32,
    mask_zero_pixels: bool,
    saturation_level: f32,
    min_psf_support_pixels: usize,
}

fn numeric(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse().ok())
}

fn numeric_flag(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Bool(value)) => *value,
        Some(value) => numeric(value).is_some_and(|value| value != 0.0),
        None => false,
    }
}

fn parse_plane(value: &Value, label: &str) -> Result<Plane> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{label} must be a 2D array"))?;
    if rows.is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    let mut width = None;
    let mut values = Vec::new();
    for row in rows {
        let row = row
            .as_array()
            .ok_or_else(|| format!("{label} rows must be arrays"))?;
        if row.is_empty() || width.is_some_and(|expected| expected != row.len()) {
            return Err(format!("{label} rows must have equal, nonzero lengths"));
        }
        width = Some(row.len());
        for value in row {
            let pixel = if value.is_null() {
                f32::NAN
            } else {
                numeric(value).ok_or_else(|| format!("{label} pixels must be numeric or null"))?
                    as f32
            };
            values.push(pixel);
        }
    }
    Ok(Plane {
        values,
        height: rows.len(),
        width: width.unwrap_or(0),
    })
}

fn parse_frame(value: &Value) -> Result<Frame<'static>> {
    let image = parse_plane(
        value
            .get("image")
            .ok_or_else(|| "frame image is required".to_owned())?,
        "frame image",
    )?;
    let background = value
        .get("background")
        .and_then(numeric)
        .ok_or_else(|| "frame background must be numeric".to_owned())? as f32;
    let noise = value
        .get("noise")
        .and_then(numeric)
        .ok_or_else(|| "frame noise must be numeric".to_owned())? as f32;
    if !background.is_finite() || !noise.is_finite() {
        return Err("frame background and noise must be finite".into());
    }
    let parameters = value.get("parameters").unwrap_or(&Value::Null);
    Ok(Frame {
        image: FrameImage::Owned(image),
        background,
        noise: noise.max(f32::EPSILON),
        mask_zero_pixels: numeric_flag(parameters.get("mask_zero_pixels")),
        saturation_level: parameters
            .get("saturation_level")
            .and_then(numeric)
            .unwrap_or(-1.0) as f32,
        min_psf_support_pixels: parameters
            .get("min_psf_support_pixels")
            .and_then(Value::as_u64)
            .unwrap_or(3)
            .max(3) as usize,
    })
}

fn frames_from_analyses(analyses: &[FrameAnalysis]) -> Result<Vec<Frame<'_>>> {
    analyses
        .iter()
        .map(|analysis| {
            let height = analysis.frame.height;
            let width = analysis.frame.width;
            if width.checked_mul(height) != Some(analysis.frame.data.len()) {
                return Err("FITS image dimensions do not match pixel data".into());
            }
            let parameters = &analysis.detection.parameters;
            Ok(Frame {
                image: FrameImage::Borrowed {
                    pixels: &analysis.frame.data,
                    height,
                    width,
                },
                background: analysis.detection.background as f32,
                noise: (analysis.detection.noise as f32).max(f32::EPSILON),
                mask_zero_pixels: numeric_flag(parameters.get("mask_zero_pixels")),
                saturation_level: parameters
                    .get("saturation_level")
                    .and_then(numeric)
                    .unwrap_or(-1.0) as f32,
                min_psf_support_pixels: parameters
                    .get("min_psf_support_pixels")
                    .and_then(Value::as_u64)
                    .unwrap_or(3)
                    .max(3) as usize,
            })
        })
        .collect()
}

fn empty_result(frame_count: usize) -> Value {
    json!({
        "candidate_peak_frames": (0..frame_count).map(|_| Vec::<[f32; 5]>::new()).collect::<Vec<_>>(),
        "candidate_count": 0
    })
}

fn temporal_fwhms(psf_fwhm: f64, multiscale: bool) -> Vec<f64> {
    if !multiscale {
        return vec![psf_fwhm];
    }
    let mut values = [0.65, 1.0, 1.35]
        .into_iter()
        .map(|factor| ((psf_fwhm * factor).max(0.8) * 1_000_000.0).round_ties_even() / 1_000_000.0)
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values.dedup_by(|a, b| a.to_bits() == b.to_bits());
    values
}

fn pixel_valid(frame: &Frame<'_>, x: isize, y: isize) -> bool {
    if x < 0
        || y < 0
        || x >= frame.image.width() as isize
        || y >= frame.image.height() as isize
        || (y == 0 && x < 104)
    {
        return false;
    }
    let Some(value) = frame
        .image
        .value(y as usize * frame.image.width() + x as usize)
    else {
        return false;
    };
    value.is_finite()
        && (!frame.mask_zero_pixels || value != 0.0)
        && (frame.saturation_level <= 0.0 || value < frame.saturation_level)
}

fn pixel(frame: &Frame<'_>, x: isize, y: isize) -> f32 {
    if x < 0 || y < 0 || x >= frame.image.width() as isize || y >= frame.image.height() as isize {
        return f32::NAN;
    }
    frame
        .image
        .value(y as usize * frame.image.width() + x as usize)
        .unwrap_or(f32::NAN)
}

fn reference_shape_supported(
    reference: &Plane,
    background: f64,
    noise: f64,
    x: usize,
    y: usize,
    shape_radius: usize,
    required_support: usize,
) -> bool {
    if x < shape_radius
        || y < shape_radius
        || x >= reference.width.saturating_sub(shape_radius)
        || y >= reference.height.saturating_sub(shape_radius)
    {
        return false;
    }
    let side = 2 * shape_radius + 1;
    let mut patch = Vec::with_capacity(side * side);
    for py in y - shape_radius..=y + shape_radius {
        for px in x - shape_radius..=x + shape_radius {
            let value = reference.values[py * reference.width + px];
            if !value.is_finite() {
                // NumPy's positive-weight sum becomes NaN for a patch containing
                // an invalid reference pixel, so the Python gate rejects it.
                return false;
            }
            patch.push(value as f64 - background);
        }
    }
    let peak_excess = (reference.values[y * reference.width + x] as f64 - background).max(0.0);
    let support_threshold = (2.0 * noise).max(0.1 * peak_excess);
    let mut support = 0;
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            let index = (shape_radius as isize + dy) as usize * side
                + (shape_radius as isize + dx) as usize;
            support += usize::from(patch[index] >= support_threshold);
        }
    }
    if support < required_support {
        return false;
    }
    let weights = patch
        .iter()
        .map(|value| (value - 0.5 * noise).max(0.0))
        .collect::<Vec<_>>();
    let weight_sum = weights.iter().sum::<f64>();
    if !weight_sum.is_finite() || weight_sum <= f64::EPSILON {
        return false;
    }
    let mut center_x = 0.0;
    let mut center_y = 0.0;
    for py in 0..side {
        for px in 0..side {
            let weight = weights[py * side + px];
            center_x += weight * px as f64;
            center_y += weight * py as f64;
        }
    }
    center_x /= weight_sum;
    center_y /= weight_sum;
    let (mut variance_x, mut variance_y, mut covariance) = (0.0, 0.0, 0.0);
    for py in 0..side {
        for px in 0..side {
            let weight = weights[py * side + px];
            let dx = px as f64 - center_x;
            let dy = py as f64 - center_y;
            variance_x += weight * dx * dx;
            variance_y += weight * dy * dy;
            covariance += weight * dx * dy;
        }
    }
    variance_x /= weight_sum;
    variance_y /= weight_sum;
    covariance /= weight_sum;
    let half_trace = 0.5 * (variance_x + variance_y);
    let half_difference = 0.5 * (variance_x - variance_y);
    let eigen_radius = (half_difference * half_difference + covariance * covariance).sqrt();
    let minor = (half_trace - eigen_radius).max(0.0);
    let major = (half_trace + eigen_radius).max(0.0);
    let fwhm = 2.35482 * (0.5 * (major + minor)).max(0.0).sqrt();
    let ellipticity = 1.0 - minor.sqrt() / major.sqrt().max(f64::EPSILON);
    fwhm.is_finite()
        && (0.5..=12.0).contains(&fwhm)
        && ellipticity.is_finite()
        && ellipticity <= 0.75
}

struct ForcedResponse<'a> {
    reference_points: &'a [[f32; 3]],
    reference_snr_map: &'a [f32],
    reference_width: usize,
    kernels: &'a [Vec<f32>],
    kernel_radius: isize,
    required_support: usize,
    min_candidate_snr: f64,
}

fn propose_frame_rows(
    frame: &Frame<'_>,
    shifts: [f64; 2],
    config: &ForcedResponse<'_>,
) -> Vec<[f32; 5]> {
    let patch_radius = config.kernel_radius + 1;
    let frame_gate = 4.0f64.max(0.5 * config.min_candidate_snr) as f32;
    let mut output = Vec::new();
    for reference_point in config.reference_points {
        // NumPy arrays remain float32 when a Python scalar shift is added.
        let x_base = (reference_point[0] + shifts[0] as f32).round_ties_even() as isize;
        let y_base = (reference_point[1] + shifts[1] as f32).round_ties_even() as isize;
        if x_base < patch_radius
            || x_base >= frame.image.width() as isize - patch_radius
            || y_base < patch_radius
            || y_base >= frame.image.height() as isize - patch_radius
        {
            continue;
        }
        let mut scores = [f32::NEG_INFINITY; 9];
        for kernel in config.kernels {
            let kernel_side = 2 * config.kernel_radius + 1;
            for search_y in -1isize..=1 {
                for search_x in -1isize..=1 {
                    let search_index = ((search_y + 1) * 3 + search_x + 1) as usize;
                    let mut numerator_terms = Vec::with_capacity(kernel.len());
                    let mut denominator_terms = Vec::with_capacity(kernel.len());
                    let mut kernel_index = 0;
                    for ky in -config.kernel_radius..=config.kernel_radius {
                        for kx in -config.kernel_radius..=config.kernel_radius {
                            let x = x_base + search_x + kx;
                            let y = y_base + search_y + ky;
                            let weight = kernel[kernel_index];
                            kernel_index += 1;
                            if pixel_valid(frame, x, y) {
                                numerator_terms
                                    .push((pixel(frame, x, y) - frame.background) * weight);
                                denominator_terms.push(weight * weight);
                            } else {
                                numerator_terms.push(0.0);
                                denominator_terms.push(0.0);
                            }
                        }
                    }
                    debug_assert_eq!(kernel_index, (kernel_side * kernel_side) as usize);
                    let denominator = frame.noise
                        * statistics32::pairwise_sum(&denominator_terms)
                            .max(f32::EPSILON)
                            .sqrt();
                    let score = statistics32::pairwise_sum(&numerator_terms) / denominator;
                    scores[search_index] = scores[search_index].max(score);
                }
            }
        }
        let mut best_search = 0usize;
        for search_index in 1..scores.len() {
            if scores[search_index] > scores[best_search] {
                best_search = search_index;
            }
        }
        let best_score = scores[best_search];
        let search_y = best_search as isize / 3 - 1;
        let search_x = best_search as isize % 3 - 1;
        let center_x = x_base + search_x;
        let center_y = y_base + search_y;
        if !best_score.is_finite()
            || best_score < frame_gate
            || !pixel_valid(frame, center_x, center_y)
        {
            continue;
        }
        let center_value = pixel(frame, center_x, center_y);
        let peak_excess = (center_value - frame.background).max(0.0);
        let support_threshold = (2.0 * frame.noise).max(0.1 * peak_excess);
        let mut support = 0;
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let x = center_x + dx;
                let y = center_y + dy;
                if pixel_valid(frame, x, y)
                    && pixel(frame, x, y) - frame.background >= support_threshold
                {
                    support += 1;
                }
            }
        }
        if support < config.required_support {
            continue;
        }
        let base_index = y_base as usize * config.reference_width + x_base as usize;
        output.push([
            (x_base + search_x) as f32,
            (y_base + search_y) as f32,
            best_score,
            1.0,
            config.reference_snr_map[base_index],
        ]);
    }
    output
}

/// Propose temporal-reference candidates and recheck them against every raw frame.
pub fn temporal_reference_candidates(request: &Value) -> Result<Value> {
    let raw_frames = request["frames"]
        .as_array()
        .ok_or_else(|| "frames must be an array".to_owned())?;
    let frame_count = raw_frames.len();
    let empty = empty_result(frame_count);
    let shifts = request["cumulative_shifts"].as_array();
    let Some(shifts) = shifts else {
        return Err("cumulative_shifts must be an array".into());
    };
    if request["reference"].is_null() || frame_count < 2 || shifts.len() != frame_count {
        return Ok(empty);
    }
    let frames = raw_frames
        .iter()
        .map(parse_frame)
        .collect::<Result<Vec<_>>>()?;
    let cumulative_shifts = shifts
        .iter()
        .map(|row| {
            let pair = row
                .as_array()
                .ok_or_else(|| "each shift must have two values".to_owned())?;
            if pair.len() != 2 {
                return Err("each shift must have two values".into());
            }
            let x = numeric(&pair[0]).ok_or_else(|| "shift values must be numeric".to_owned())?;
            let y = numeric(&pair[1]).ok_or_else(|| "shift values must be numeric".to_owned())?;
            Ok([x, y])
        })
        .collect::<Result<Vec<_>>>()?;
    if cumulative_shifts
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err("shift values must be finite".into());
    }
    let reference = parse_plane(&request["reference"], "reference")?;
    let options = request.get("options").unwrap_or(&Value::Null);
    temporal_candidates_from_frames(&reference, &frames, &cumulative_shifts, options)
}

fn temporal_candidates_from_frames(
    reference: &Plane,
    frames: &[Frame<'_>],
    cumulative_shifts: &[[f64; 2]],
    options: &Value,
) -> Result<Value> {
    let empty = empty_result(frames.len());
    if frames.len() < 2 || cumulative_shifts.len() != frames.len() {
        return Ok(empty);
    }
    let psf_fwhm = options.get("psf_fwhm").and_then(numeric).unwrap_or(3.0);
    let min_candidate_snr = options
        .get("min_candidate_snr")
        .and_then(numeric)
        .unwrap_or(15.0);
    if psf_fwhm <= 0.0 || min_candidate_snr <= 0.0 {
        return Err("temporal reference candidate thresholds must be positive".into());
    }
    let multiscale = numeric_flag(options.get("temporal_multiscale"));
    let temporal_fwhm = temporal_fwhms(psf_fwhm, multiscale);
    let temporal_sigmas = temporal_fwhm
        .iter()
        .map(|fwhm| (fwhm / 2.35482).max(0.5))
        .collect::<Vec<_>>();
    let mut valid = reference
        .values
        .iter()
        .map(|v| v.is_finite())
        .collect::<Vec<_>>();
    for value in valid.iter_mut().take(reference.width.min(104)) {
        *value = false;
    }
    if !valid.iter().any(|value| *value) {
        return Ok(empty);
    }
    let invalid = valid.iter().map(|value| !*value).collect::<Vec<_>>();
    let (background, noise) =
        statistics32::sigma_clipped_stats(&reference.values, Some(&invalid), 3.0, 4, 1_000_000)?;
    let noise = noise.max(f32::EPSILON as f64);
    let residual = reference
        .values
        .iter()
        .zip(&valid)
        .map(
            |(value, ok)| {
                if *ok {
                    *value - background as f32
                } else {
                    0.0
                }
            },
        )
        .collect::<Vec<_>>();
    let mut response_snr_bank = Vec::with_capacity(temporal_sigmas.len());
    let mut response_local_max_bank = Vec::with_capacity(temporal_sigmas.len());
    for sigma in &temporal_sigmas {
        let response =
            filtering32::gaussian_filter(&residual, reference.height, reference.width, *sigma)?;
        let response_values = response
            .iter()
            .zip(&valid)
            .filter_map(|(value, ok)| ok.then_some(*value))
            .collect::<Vec<_>>();
        let (response_background, response_noise) =
            statistics32::sigma_clipped_stats(&response_values, None, 3.0, 4, 1_000_000)?;
        let response_noise = response_noise.max(f32::EPSILON as f64);
        let response_snr = response
            .iter()
            .map(|value| (*value - response_background as f32) / response_noise as f32)
            .collect::<Vec<_>>();
        let peak_radius = (2.0 * sigma).ceil().max(1.0) as usize;
        let maxima = filtering32::maximum_filter(
            &response_snr,
            reference.height,
            reference.width,
            peak_radius,
        )?;
        response_local_max_bank.push(
            response_snr
                .iter()
                .zip(maxima)
                .map(|(value, maximum)| *value == maximum)
                .collect::<Vec<_>>(),
        );
        response_snr_bank.push(response_snr);
    }
    let seed_threshold = (0.5 * min_candidate_snr).max(5.0) as f32;
    let raw_maxima =
        filtering32::maximum_filter(&reference.values, reference.height, reference.width, 1)?;
    let mut candidates = Vec::<[f32; 3]>::new();
    let mut response_snr_map = vec![f32::NEG_INFINITY; reference.values.len()];
    for response in &response_snr_bank {
        for (target, value) in response_snr_map.iter_mut().zip(response) {
            *target = target.max(*value);
        }
    }
    for index in 0..reference.values.len() {
        let mut response_peak = false;
        for (response, local_max) in response_snr_bank.iter().zip(&response_local_max_bank) {
            if local_max[index] && response[index] >= seed_threshold {
                response_peak = true;
                break;
            }
        }
        if valid[index]
            && response_peak
            && reference.values[index] == raw_maxima[index]
            && reference.values[index] - background as f32 >= (2.0 * noise) as f32
        {
            let y = index / reference.width;
            let x = index % reference.width;
            candidates.push([x as f32, y as f32, response_snr_map[index]]);
        }
    }
    if candidates.is_empty() {
        return Ok(empty);
    }
    let required_support = frames
        .first()
        .map_or(3, |frame| frame.min_psf_support_pixels);
    let shape_radius = (3.5 * temporal_sigmas.iter().copied().fold(0.0, f64::max))
        .ceil()
        .max(4.0) as usize;
    let reference_points = candidates
        .iter()
        .filter(|point| {
            reference_shape_supported(
                reference,
                background,
                noise,
                point[0] as usize,
                point[1] as usize,
                shape_radius,
                required_support,
            )
        })
        .copied()
        .collect::<Vec<_>>();
    if reference_points.is_empty() {
        return Ok(empty);
    }
    let kernel_radius = (2.75 * temporal_sigmas.iter().copied().fold(0.0, f64::max))
        .ceil()
        .max(3.0) as isize;
    let mut kernels = Vec::with_capacity(temporal_sigmas.len());
    for sigma in &temporal_sigmas {
        let mut kernel = Vec::new();
        for dy in -kernel_radius..=kernel_radius {
            for dx in -kernel_radius..=kernel_radius {
                kernel.push((-0.5 * ((dx * dx + dy * dy) as f64) / sigma.powi(2)).exp() as f32);
            }
        }
        kernels.push(kernel);
    }
    let forced_response = ForcedResponse {
        reference_points: &reference_points,
        reference_snr_map: &response_snr_map,
        reference_width: reference.width,
        kernels: &kernels,
        kernel_radius,
        required_support,
        min_candidate_snr,
    };
    let candidate_peak_frames = frames
        .iter()
        .zip(cumulative_shifts.iter())
        .map(|(frame, shift)| propose_frame_rows(frame, *shift, &forced_response))
        .collect::<Vec<_>>();
    Ok(json!({
        "candidate_peak_frames":candidate_peak_frames,
        "candidate_count":reference_points.len()
    }))
}

/// Build temporal-reference candidate proposals directly from analyzed FITS frames.
/// Pixel buffers remain borrowed and are never expanded into JSON arrays.
pub fn temporal_reference_candidates_from_analyses(
    analyses: &[FrameAnalysis],
    cumulative_shifts: &[[f64; 2]],
    coadd: bool,
    clip_sigma: f64,
    psf_fwhm: f64,
    min_candidate_snr: f64,
    multiscale: bool,
) -> Result<Value> {
    let frames = frames_from_analyses(analyses)?;
    if frames.len() < 2 {
        return Ok(empty_result(frames.len()));
    }
    if coadd && clip_sigma <= 0.0 {
        return Err("clip_sigma must be positive".into());
    }
    let Some(reference) =
        build_registered_reference(&frames, cumulative_shifts, coadd, clip_sigma)?
    else {
        return Ok(empty_result(frames.len()));
    };
    let options = json!({
        "psf_fwhm":psf_fwhm,
        "min_candidate_snr":min_candidate_snr,
        "temporal_multiscale":multiscale,
    });
    temporal_candidates_from_frames(&reference, &frames, cumulative_shifts, &options)
}

fn parse_shifts(value: &Value, frame_count: usize) -> Result<Vec<[f64; 2]>> {
    let rows = value
        .as_array()
        .ok_or_else(|| "cumulative_shifts must be an array".to_owned())?;
    if rows.len() != frame_count {
        return Err("cumulative_shifts length must match frame_analyses".into());
    }
    rows.iter()
        .map(|row| {
            let pair = row
                .as_array()
                .ok_or_else(|| "each shift must have two values".to_owned())?;
            if pair.len() != 2 {
                return Err("each shift must have two values".into());
            }
            let x = numeric(&pair[0]).ok_or_else(|| "shift values must be numeric".to_owned())?;
            let y = numeric(&pair[1]).ok_or_else(|| "shift values must be numeric".to_owned())?;
            if !x.is_finite() || !y.is_finite() {
                return Err("shift values must be finite".into());
            }
            Ok([x, y])
        })
        .collect()
}

fn valid_reference_pixel(frame: &Frame<'_>, x: usize, y: usize) -> Option<f32> {
    if y == 0 && x < 104 {
        return None;
    }
    let value = frame.image.value(y * frame.image.width() + x)?;
    (value.is_finite()
        && (!frame.mask_zero_pixels || value != 0.0)
        && (frame.saturation_level <= 0.0 || value < frame.saturation_level))
        .then_some(value)
}

fn bilinear_sample_frame(frame: &Frame<'_>, x: f32, y: f32) -> f32 {
    let width = frame.image.width();
    let height = frame.image.height();
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x > (width - 1) as f32
        || y > (height - 1) as f32
    {
        return f32::NAN;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let (Some(a), Some(b), Some(c), Some(d)) = (
        valid_reference_pixel(frame, x0, y0),
        valid_reference_pixel(frame, x1, y0),
        valid_reference_pixel(frame, x0, y1),
        valid_reference_pixel(frame, x1, y1),
    ) else {
        return f32::NAN;
    };
    let dx = (x - x0 as f32) as f64;
    let dy = (y - y0 as f32) as f64;
    let top = a as f64 + (b as f64 - a as f64) * dx;
    let bottom = c as f64 + (d as f64 - c as f64) * dx;
    (top + (bottom - top) * dy) as f32
}

fn median_f32(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn build_registered_reference(
    frames: &[Frame<'_>],
    shifts: &[[f64; 2]],
    coadd: bool,
    clip_sigma: f64,
) -> Result<Option<Plane>> {
    if frames.len() < 2 || shifts.len() != frames.len() {
        return Ok(None);
    }
    let height = frames[0].image.height();
    let width = frames[0].image.width();
    if width == 0
        || height == 0
        || frames
            .iter()
            .any(|frame| frame.image.height() != height || frame.image.width() != width)
    {
        return Ok(None);
    }
    let direct_reference = shifts
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0f64, f64::max)
        <= 0.75;
    let mut output = vec![f32::NAN; height * width];
    let mut samples = Vec::with_capacity(frames.len());
    for y in 0..height {
        for x in 0..width {
            samples.clear();
            for (frame_index, frame) in frames.iter().enumerate() {
                let value = if direct_reference {
                    valid_reference_pixel(frame, x, y).unwrap_or(f32::NAN)
                } else {
                    bilinear_sample_frame(
                        frame,
                        x as f32 + shifts[frame_index][0] as f32,
                        y as f32 + shifts[frame_index][1] as f32,
                    )
                };
                if value.is_finite() {
                    samples.push(value);
                }
            }
            if samples.is_empty() {
                continue;
            }
            let mut ordered = samples.clone();
            let center = median_f32(&mut ordered);
            if !coadd {
                output[y * width + x] = center;
                continue;
            }
            let mut deviations = samples
                .iter()
                .map(|value| (*value - center).abs())
                .collect::<Vec<_>>();
            let mad = median_f32(&mut deviations);
            let scale = (1.4826f32 * mad).max(f32::EPSILON);
            let threshold = clip_sigma as f32 * scale;
            let mut count = 0usize;
            let mut sum = 0.0f32;
            for value in &samples {
                if (*value - center).abs() <= threshold {
                    count += 1;
                    sum += *value;
                }
            }
            if count > 0 {
                output[y * width + x] = sum / count as f32;
            }
        }
    }
    Ok(Some(Plane {
        values: output,
        height,
        width,
    }))
}

fn reference_value(reference: &Plane) -> Value {
    let rows = reference
        .values
        .chunks_exact(reference.width)
        .map(|row| {
            row.iter()
                .map(|value| {
                    if value.is_finite() {
                        json!(value)
                    } else {
                        Value::Null
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    json!(rows)
}

fn reference_rows(request: &Value, coadd: bool) -> Result<Value> {
    let raw_frames = request["frames"]
        .as_array()
        .ok_or_else(|| "frames must be an array".to_owned())?;
    if raw_frames.len() < 2 {
        return Ok(json!({"reference":null}));
    }
    let shifts = parse_shifts(&request["cumulative_shifts"], raw_frames.len())?;
    let options = request.get("options").unwrap_or(&Value::Null);
    let clip_sigma = options.get("clip_sigma").and_then(numeric).unwrap_or(3.0);
    if coadd && clip_sigma <= 0.0 {
        return Err("clip_sigma must be positive".into());
    }
    let frames = raw_frames
        .iter()
        .map(parse_frame)
        .collect::<Result<Vec<_>>>()?;
    let reference = build_registered_reference(&frames, &shifts, coadd, clip_sigma)?;
    Ok(json!({"reference":reference.as_ref().map(reference_value)}))
}

/// Build the registered, pixel-wise temporal median reference.
pub fn registered_median_reference(request: &Value) -> Result<Value> {
    reference_rows(request, false)
}

/// Build the robustly clipped registered temporal mean reference.
pub fn registered_coadd_reference(request: &Value) -> Result<Value> {
    reference_rows(request, true)
}
