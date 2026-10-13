//! Repeated-candidate consensus before image-level quality promotion.
use crate::{
    models::PixelData,
    pipeline::FrameAnalysis,
    sequence::TrackPoint,
    sequence_motion::{fit_track_motion, TrackObservation},
    spatial::KdTree,
    statistics, Result,
};
use serde_json::{json, Map, Value};

#[derive(Debug, Clone)]
struct CandidatePoint {
    x: f64,
    y: f64,
    snr: f64,
    temporal: bool,
    reference_snr: f64,
}

fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn normalize_frames(
    raw_frames: &[Value],
    min_snr: f64,
    temporal_min_snr: f64,
    reference_min_snr: f64,
) -> Option<(Vec<Vec<CandidatePoint>>, usize, usize)> {
    let mut frames = Vec::with_capacity(raw_frames.len());
    let mut input_rows = 0;
    let mut gated_rows = 0;
    for raw_frame in raw_frames {
        let rows = raw_frame.as_array()?;
        let mut frame = Vec::new();
        for raw_row in rows {
            let row = raw_row.as_array()?;
            if row.is_empty() {
                continue;
            }
            input_rows += 1;
            if row.len() < 3 {
                return None;
            }
            let (Some(x), Some(y), Some(snr)) = (number(&row[0]), number(&row[1]), number(&row[2]))
            else {
                continue;
            };
            if !x.is_finite() || !y.is_finite() || !snr.is_finite() {
                continue;
            }
            let temporal = row
                .get(3)
                .and_then(number)
                .is_some_and(|value| value.is_finite() && value >= 0.5);
            let reference_snr = row
                .get(4)
                .and_then(number)
                .filter(|value| value.is_finite())
                .unwrap_or(f64::NAN);
            if snr < if temporal { temporal_min_snr } else { min_snr } {
                continue;
            }
            if temporal && (!reference_snr.is_finite() || reference_snr < reference_min_snr) {
                continue;
            }
            gated_rows += 1;
            frame.push(CandidatePoint {
                x,
                y,
                snr,
                temporal,
                reference_snr,
            });
        }
        frames.push(frame);
    }
    Some((frames, input_rows, gated_rows))
}

fn quality_by_frame(
    tracks: &[Value],
    frame_count: usize,
    persistent_min_presence: usize,
) -> Vec<Vec<[f64; 2]>> {
    let mut points = vec![Vec::new(); frame_count];
    for track in tracks {
        let rows = track["points"].as_array().map(Vec::as_slice).unwrap_or(&[]);
        let presence = track["presence"]
            .as_u64()
            .map_or(rows.len(), |value| value as usize);
        if presence < persistent_min_presence {
            continue;
        }
        for point in rows {
            let Some(index) = point["frame_index"].as_u64().map(|value| value as usize) else {
                continue;
            };
            let (Some(x), Some(y)) = (point["aligned_x"].as_f64(), point["aligned_y"].as_f64())
            else {
                continue;
            };
            if index < frame_count && x.is_finite() && y.is_finite() {
                points[index].push([x, y]);
            }
        }
    }
    points
}

fn normalize_times(times: Option<&[f64]>, frame_count: usize) -> Option<Vec<f64>> {
    let values = times?;
    if values.len() != frame_count
        || values.iter().any(|value| !value.is_finite())
        || values.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Some((0..frame_count).map(|index| index as f64).collect());
    }
    Some(values.iter().map(|value| value - values[0]).collect())
}

#[derive(Debug)]
enum ImagePixels<'a> {
    Rows(Vec<Vec<f64>>),
    Fits(&'a PixelData),
}

impl ImagePixels<'_> {
    fn value(&self, width: usize, x: usize, y: usize) -> Option<f64> {
        match self {
            Self::Rows(rows) => rows.get(y)?.get(x).copied(),
            Self::Fits(data) => {
                let index = y.checked_mul(width)?.checked_add(x)?;
                (index < data.len()).then(|| data.value(index))
            }
        }
    }
}

#[derive(Debug)]
struct ImageFrame<'a> {
    pixels: ImagePixels<'a>,
    width: usize,
    height: usize,
    background: f64,
    noise: f64,
    mask_zero_pixels: bool,
    saturation_level: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ImageGateCounts {
    valid: usize,
    support: usize,
    line_overlap: usize,
    shape: usize,
    psf_correlation: usize,
}

fn json_rows(value: &Value, label: &str) -> Result<Vec<Vec<f64>>> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{label} must be a 2D array"))?;
    let mut output = Vec::with_capacity(rows.len());
    let mut width = None;
    for row in rows {
        let values = row
            .as_array()
            .ok_or_else(|| format!("{label} rows must be arrays"))?;
        if width.is_some_and(|expected| expected != values.len()) {
            return Err(format!("{label} rows must have equal length"));
        }
        width = Some(values.len());
        output.push(
            values
                .iter()
                .map(|value| number(value).unwrap_or(f64::NAN))
                .collect(),
        );
    }
    if output.is_empty() || width.unwrap_or(0) == 0 {
        return Err(format!("{label} must not be empty"));
    }
    Ok(output)
}

fn numeric_flag(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Bool(value)) => *value,
        Some(value) => number(value).is_some_and(|value| value != 0.0),
        None => false,
    }
}

fn image_frames(
    value: Option<&Value>,
    frame_count: usize,
) -> Result<Option<Vec<ImageFrame<'static>>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let raw_frames = value["frames"]
        .as_array()
        .ok_or_else(|| "image_support.frames must be an array".to_owned())?;
    let backgrounds = value["backgrounds"]
        .as_array()
        .ok_or_else(|| "image_support.backgrounds must be an array".to_owned())?;
    let noises = value["noises"]
        .as_array()
        .ok_or_else(|| "image_support.noises must be an array".to_owned())?;
    let parameters = value["parameters"]
        .as_array()
        .ok_or_else(|| "image_support.parameters must be an array".to_owned())?;
    if [
        raw_frames.len(),
        backgrounds.len(),
        noises.len(),
        parameters.len(),
    ]
    .iter()
    .any(|length| *length != frame_count)
    {
        return Err("image_support frame arrays must match candidate_peak_frames".into());
    }
    let mut frames = Vec::with_capacity(frame_count);
    for index in 0..frame_count {
        let pixels = json_rows(&raw_frames[index], "image_support frame")?;
        let background = number(&backgrounds[index])
            .ok_or_else(|| "image_support background must be numeric".to_owned())?;
        let noise = number(&noises[index])
            .ok_or_else(|| "image_support noise must be numeric".to_owned())?;
        if !background.is_finite() || !noise.is_finite() {
            return Err("image_support backgrounds and noises must be finite".into());
        }
        let parameter = &parameters[index];
        let height = pixels.len();
        let width = pixels.first().map_or(0, Vec::len);
        frames.push(ImageFrame {
            pixels: ImagePixels::Rows(pixels),
            width,
            height,
            background,
            noise: noise.max(f64::EPSILON),
            mask_zero_pixels: numeric_flag(parameter.get("mask_zero_pixels")),
            saturation_level: parameter
                .get("saturation_level")
                .and_then(number)
                .unwrap_or(-1.0),
        });
    }
    Ok(Some(frames))
}

#[derive(Debug)]
struct ImageSupport<'a> {
    frames: Vec<ImageFrame<'a>>,
    line_frames: Vec<Vec<[f64; 2]>>,
    aperture_radius: usize,
    required_support: usize,
    psf_sigmas: Vec<f64>,
}

fn parse_image_support(
    value: Option<&Value>,
    frame_count: usize,
) -> Result<Option<ImageSupport<'static>>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let frames = image_frames(Some(value), frame_count)?
        .ok_or_else(|| "image_support.frames are required".to_owned())?;
    let parameters = value["parameters"]
        .as_array()
        .ok_or_else(|| "image_support.parameters must be an array".to_owned())?;
    let first = parameters
        .first()
        .ok_or_else(|| "image_support.parameters must not be empty".to_owned())?;
    let aperture_radius = first
        .get("aperture_radius")
        .and_then(Value::as_u64)
        .unwrap_or(4)
        .max(1) as usize;
    let required_support = first
        .get("min_psf_support_pixels")
        .and_then(Value::as_u64)
        .unwrap_or(3)
        .max(3) as usize;
    let psf_fwhm = first.get("psf_fwhm").and_then(number).unwrap_or(3.0);
    if !psf_fwhm.is_finite() || psf_fwhm <= 0.0 {
        return Err("image_support psf_fwhm must be positive".into());
    }
    let mut fwhm_bank = [0.65, 1.0, 1.35]
        .into_iter()
        .map(|factor| ((psf_fwhm * factor).max(0.8) * 1_000_000.0).round_ties_even() / 1_000_000.0)
        .collect::<Vec<_>>();
    fwhm_bank.sort_by(f64::total_cmp);
    fwhm_bank.dedup_by(|a, b| a.to_bits() == b.to_bits());
    let psf_sigmas = fwhm_bank
        .into_iter()
        .map(|fwhm| (fwhm / 2.35482).max(0.5))
        .collect();
    let mut line_frames = vec![Vec::new(); frame_count];
    if let Some(raw_line_frames) = value.get("line_artifact_frames").and_then(Value::as_array) {
        if raw_line_frames.len() != frame_count {
            return Err("line_artifact_frames length must match image_support.frames".into());
        }
        for (index, raw_frame) in raw_line_frames.iter().enumerate() {
            let rows = raw_frame
                .as_array()
                .ok_or_else(|| "line artifact frame must be an array".to_owned())?;
            for row in rows {
                let coordinates = row
                    .as_array()
                    .ok_or_else(|| "line artifact coordinates must be an array".to_owned())?;
                if coordinates.len() < 2 {
                    return Err("line artifact coordinates must contain x and y".into());
                }
                let (Some(x), Some(y)) = (number(&coordinates[0]), number(&coordinates[1])) else {
                    continue;
                };
                if x.is_finite() && y.is_finite() {
                    line_frames[index].push([x, y]);
                }
            }
        }
    }
    Ok(Some(ImageSupport {
        frames,
        line_frames,
        aperture_radius,
        required_support,
        psf_sigmas,
    }))
}

fn image_support_from_analyses(analyses: &[FrameAnalysis]) -> Result<Option<ImageSupport<'_>>> {
    if analyses.is_empty() {
        return Ok(None);
    }
    let first = &analyses[0];
    let aperture_radius = first.detection.parameters["aperture_radius"]
        .as_u64()
        .unwrap_or(4)
        .max(1) as usize;
    let required_support = first.detection.parameters["min_psf_support_pixels"]
        .as_u64()
        .unwrap_or(3)
        .max(3) as usize;
    let psf_fwhm = first.detection.parameters["psf_fwhm"]
        .as_f64()
        .unwrap_or(3.0);
    if !psf_fwhm.is_finite() || psf_fwhm <= 0.0 {
        return Err("image_support psf_fwhm must be positive".into());
    }
    let mut fwhm_bank = [0.65, 1.0, 1.35]
        .into_iter()
        .map(|factor| ((psf_fwhm * factor).max(0.8) * 1_000_000.0).round_ties_even() / 1_000_000.0)
        .collect::<Vec<_>>();
    fwhm_bank.sort_by(f64::total_cmp);
    fwhm_bank.dedup_by(|a, b| a.to_bits() == b.to_bits());
    let psf_sigmas = fwhm_bank
        .into_iter()
        .map(|fwhm| (fwhm / 2.35482).max(0.5))
        .collect();
    let mut frames = Vec::with_capacity(analyses.len());
    let mut line_frames = Vec::with_capacity(analyses.len());
    for analysis in analyses {
        let height = analysis.frame.height;
        let width = analysis.frame.width;
        if width.checked_mul(height) != Some(analysis.frame.data.len()) {
            return Err("FITS image dimensions do not match pixel data".into());
        }
        let parameters = &analysis.detection.parameters;
        frames.push(ImageFrame {
            pixels: ImagePixels::Fits(&analysis.frame.data),
            width,
            height,
            background: analysis.detection.background,
            noise: analysis.detection.noise.max(f64::EPSILON),
            mask_zero_pixels: numeric_flag(parameters.get("mask_zero_pixels")),
            saturation_level: parameters
                .get("saturation_level")
                .and_then(number)
                .unwrap_or(-1.0),
        });
        line_frames.push(
            analysis
                .detection
                .line_artifact_coordinates
                .iter()
                .map(|[x, y]| [*x as f64, *y as f64])
                .collect(),
        );
    }
    Ok(Some(ImageSupport {
        frames,
        line_frames,
        aperture_radius,
        required_support,
        psf_sigmas,
    }))
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn image_pixel_valid(frame: &ImageFrame<'_>, x: isize, y: isize) -> Option<f64> {
    if x < 0 || y < 0 || (y == 0 && x < 104) {
        return None;
    }
    let value = frame.pixels.value(frame.width, x as usize, y as usize)?;
    (value.is_finite()
        && (!frame.mask_zero_pixels || value != 0.0)
        && (frame.saturation_level <= 0.0 || value < frame.saturation_level))
        .then_some(value)
}

fn count_image_support(
    image_support: &ImageSupport<'_>,
    candidates: &[Vec<CandidatePoint>],
    matches: &[Vec<Option<usize>>],
    eligible: &[usize],
    min_psf_correlation: f64,
) -> Vec<ImageGateCounts> {
    let frames = &image_support.frames;
    let line_frames = &image_support.line_frames;
    let aperture_radius = image_support.aperture_radius;
    let required_support = image_support.required_support;
    let psf_sigmas = &image_support.psf_sigmas;
    let aperture_radius_i = aperture_radius as isize;
    let aperture_geometry = (-aperture_radius_i..=aperture_radius_i)
        .flat_map(|dy| {
            (-aperture_radius_i..=aperture_radius_i).filter_map(move |dx| {
                ((dx * dx + dy * dy) <= aperture_radius_i * aperture_radius_i).then_some((dx, dy))
            })
        })
        .collect::<Vec<_>>();
    let annulus_inner_radius = 1.5f64.max(0.75 * aperture_radius as f64);
    let annulus_geometry = aperture_geometry
        .iter()
        .copied()
        .filter(|(dx, dy)| (dx * dx + dy * dy) as f64 >= annulus_inner_radius.powi(2))
        .collect::<Vec<_>>();
    let minimum_annulus_pixels =
        8usize.max((0.25 * aperture_geometry.len() as f64).ceil() as usize);
    let minimum_aperture_valid = 9usize.max((0.8 * aperture_geometry.len() as f64).ceil() as usize);
    let core_offsets = (-1isize..=1)
        .flat_map(|dy| (-1isize..=1).map(move |dx| (dx, dy)))
        .collect::<Vec<_>>();
    let mut counts = vec![ImageGateCounts::default(); matches.len()];
    for anchor_index in eligible {
        for frame_index in 0..frames.len() {
            let Some(candidate_index) = matches[*anchor_index][frame_index] else {
                continue;
            };
            let frame = &frames[frame_index];
            let candidate = &candidates[frame_index][candidate_index];
            let center_x = candidate.x.round_ties_even() as isize;
            let center_y = candidate.y.round_ties_even() as isize;
            let height = frame.height as isize;
            let width = frame.width as isize;
            let edge_margin = aperture_radius as isize + 4;
            if center_x < edge_margin
                || center_x >= width - edge_margin
                || center_y < edge_margin
                || center_y >= height - edge_margin
            {
                continue;
            }
            let Some(center_value) = image_pixel_valid(frame, center_x, center_y) else {
                continue;
            };
            let aperture_values = aperture_geometry
                .iter()
                .map(|(dx, dy)| image_pixel_valid(frame, center_x + dx, center_y + dy))
                .collect::<Vec<_>>();
            let valid_aperture_count = aperture_values
                .iter()
                .filter(|value| value.is_some())
                .count();
            if valid_aperture_count < minimum_aperture_valid {
                continue;
            }
            let mut annulus_values = Vec::new();
            for (offset, value) in aperture_geometry.iter().zip(&aperture_values) {
                if annulus_geometry.contains(offset) {
                    if let Some(value) = value {
                        annulus_values.push(*value);
                    }
                }
            }
            let mut local_background = frame.background;
            let mut local_noise = frame.noise;
            if annulus_values.len() >= minimum_annulus_pixels {
                let mut sorted = annulus_values.clone();
                let annulus_median = median(&mut sorted);
                let mut deviations = annulus_values
                    .iter()
                    .map(|value| (value - annulus_median).abs())
                    .collect::<Vec<_>>();
                let annulus_noise = 1.4826 * median(&mut deviations);
                if annulus_median.is_finite()
                    && annulus_noise.is_finite()
                    && annulus_noise > f64::EPSILON
                {
                    local_background = annulus_median;
                    local_noise = annulus_noise;
                }
            }
            local_noise = local_noise.max(0.5 * frame.noise);

            if line_frames.get(frame_index).is_some_and(|line_points| {
                line_points.iter().any(|point| {
                    let dx = point[0] - center_x as f64;
                    let dy = point[1] - center_y as f64;
                    dx * dx + dy * dy <= (aperture_radius as f64).powi(2)
                })
            }) {
                counts[*anchor_index].line_overlap += 1;
                continue;
            }
            counts[*anchor_index].valid += 1;

            let core_values = core_offsets
                .iter()
                .map(|(dx, dy)| {
                    image_pixel_valid(frame, center_x + dx, center_y + dy)
                        .map(|value| (value, true))
                        .unwrap_or((0.0, false))
                })
                .collect::<Vec<_>>();
            let peak_excess = (center_value - local_background).max(0.0);
            let support_threshold = (2.0 * local_noise).max(0.1 * peak_excess);
            if core_values
                .iter()
                .filter(|(value, valid)| *valid && *value - local_background >= support_threshold)
                .count()
                >= required_support
            {
                counts[*anchor_index].support += 1;
            }

            let mut positive = Vec::with_capacity(aperture_geometry.len());
            let mut shape_peak = f64::NEG_INFINITY;
            for value in &aperture_values {
                if let Some(value) = value {
                    let residual = *value - local_background;
                    shape_peak = shape_peak.max(residual);
                    positive.push(residual.max(0.0));
                } else {
                    positive.push(0.0);
                }
            }
            let positive_sum = positive.iter().sum::<f64>();
            let safe_sum = positive_sum.max(f64::EPSILON);
            let center_offset_x = positive
                .iter()
                .zip(&aperture_geometry)
                .map(|(value, (dx, _))| value * *dx as f64)
                .sum::<f64>()
                / safe_sum;
            let center_offset_y = positive
                .iter()
                .zip(&aperture_geometry)
                .map(|(value, (_, dy))| value * *dy as f64)
                .sum::<f64>()
                / safe_sum;
            let variance_x = positive
                .iter()
                .zip(&aperture_geometry)
                .map(|(value, (dx, _))| value * (*dx as f64 - center_offset_x).powi(2))
                .sum::<f64>()
                / safe_sum;
            let variance_y = positive
                .iter()
                .zip(&aperture_geometry)
                .map(|(value, (_, dy))| value * (*dy as f64 - center_offset_y).powi(2))
                .sum::<f64>()
                / safe_sum;
            let sigma_x = variance_x.max(0.0).sqrt();
            let sigma_y = variance_y.max(0.0).sqrt();
            let fwhm = 2.35482 * (((variance_x + variance_y) * 0.5).max(0.0)).sqrt();
            let ellipticity = (sigma_x - sigma_y).abs() / sigma_x.max(sigma_y).max(f64::EPSILON);
            let sharpness = shape_peak.max(0.0) / safe_sum;
            if fwhm.is_finite()
                && (0.8..=4.5).contains(&fwhm)
                && ellipticity.is_finite()
                && ellipticity <= 0.65
                && sharpness.is_finite()
                && sharpness >= 0.08
            {
                counts[*anchor_index].shape += 1;
            }

            let mut best_correlation = f64::NEG_INFINITY;
            for sigma in psf_sigmas {
                let mut dot = 0.0;
                let mut template_norm_sq = 0.0;
                let mut residual_norm_sq = 0.0;
                for (index, (dx, dy)) in aperture_geometry.iter().enumerate() {
                    let Some(pixel) = aperture_values[index] else {
                        continue;
                    };
                    let residual = pixel - local_background;
                    let template = (-0.5 * ((*dx * *dx + *dy * *dy) as f64) / sigma.powi(2)).exp();
                    dot += residual * template;
                    template_norm_sq += template * template;
                    residual_norm_sq += residual * residual;
                }
                let denominator = residual_norm_sq.sqrt() * template_norm_sq.sqrt();
                let correlation = if denominator > f64::EPSILON {
                    dot / denominator
                } else {
                    f64::NEG_INFINITY
                };
                best_correlation = best_correlation.max(correlation);
            }
            if best_correlation >= min_psf_correlation {
                counts[*anchor_index].psf_correlation += 1;
            }
        }
    }
    counts
}

/// Link repeated matched-filter candidate peaks and apply optional raw-image gates.
#[allow(clippy::too_many_arguments)]
pub fn candidate_consensus(
    candidate_peak_frames: &[Value],
    cumulative_shifts: &[[f64; 2]],
    quality_tracks: &[Value],
    frame_times: Option<&[f64]>,
    link_radius_px: f64,
    min_presence: usize,
    persistent_min_presence: usize,
    motion_min_displacement_px: f64,
    max_motion_fit_rms_px: f64,
    min_candidate_snr: f64,
    temporal_candidate_min_snr: Option<f64>,
    temporal_reference_min_snr: Option<f64>,
    temporal_min_psf_correlation: f64,
    image_support_data: Option<&Value>,
) -> Result<Value> {
    let image_support = parse_image_support(image_support_data, candidate_peak_frames.len())?;
    candidate_consensus_with_image_support(
        candidate_peak_frames,
        cumulative_shifts,
        quality_tracks,
        frame_times,
        link_radius_px,
        min_presence,
        persistent_min_presence,
        motion_min_displacement_px,
        max_motion_fit_rms_px,
        min_candidate_snr,
        temporal_candidate_min_snr,
        temporal_reference_min_snr,
        temporal_min_psf_correlation,
        image_support.as_ref(),
    )
}

/// Run candidate consensus against FITS pixels already held by native analysis.
#[allow(clippy::too_many_arguments)]
pub fn candidate_consensus_from_analyses(
    candidate_peak_frames: &[Value],
    cumulative_shifts: &[[f64; 2]],
    quality_tracks: &[Value],
    frame_times: Option<&[f64]>,
    link_radius_px: f64,
    min_presence: usize,
    persistent_min_presence: usize,
    motion_min_displacement_px: f64,
    max_motion_fit_rms_px: f64,
    min_candidate_snr: f64,
    temporal_candidate_min_snr: Option<f64>,
    temporal_reference_min_snr: Option<f64>,
    temporal_min_psf_correlation: f64,
    analyses: &[FrameAnalysis],
) -> Result<Value> {
    let image_support = image_support_from_analyses(analyses)?;
    candidate_consensus_with_image_support(
        candidate_peak_frames,
        cumulative_shifts,
        quality_tracks,
        frame_times,
        link_radius_px,
        min_presence,
        persistent_min_presence,
        motion_min_displacement_px,
        max_motion_fit_rms_px,
        min_candidate_snr,
        temporal_candidate_min_snr,
        temporal_reference_min_snr,
        temporal_min_psf_correlation,
        image_support.as_ref(),
    )
}

#[allow(clippy::too_many_arguments)]
fn candidate_consensus_with_image_support(
    candidate_peak_frames: &[Value],
    cumulative_shifts: &[[f64; 2]],
    quality_tracks: &[Value],
    frame_times: Option<&[f64]>,
    link_radius_px: f64,
    min_presence: usize,
    persistent_min_presence: usize,
    motion_min_displacement_px: f64,
    max_motion_fit_rms_px: f64,
    min_candidate_snr: f64,
    temporal_candidate_min_snr: Option<f64>,
    temporal_reference_min_snr: Option<f64>,
    temporal_min_psf_correlation: f64,
    image_support: Option<&ImageSupport<'_>>,
) -> Result<Value> {
    if candidate_peak_frames.is_empty() || candidate_peak_frames.len() != cumulative_shifts.len() {
        return Ok(json!({"tracks":[],"audit":{}}));
    }
    let temporal_reference_gate = temporal_reference_min_snr.unwrap_or(min_candidate_snr);
    if link_radius_px <= 0.0 || min_presence < 1 || persistent_min_presence < 1 {
        return Err("candidate consensus thresholds are invalid".into());
    }
    if persistent_min_presence > min_presence || min_candidate_snr <= 0.0 {
        return Err("candidate consensus presence/SNR thresholds are invalid".into());
    }
    if temporal_reference_gate <= 0.0 || temporal_reference_gate > min_candidate_snr {
        return Err("temporal reference SNR threshold is invalid".into());
    }
    let temporal_min_snr = temporal_candidate_min_snr.unwrap_or((0.5 * min_candidate_snr).max(5.0));
    if temporal_min_snr <= 0.0 || temporal_min_snr > min_candidate_snr {
        return Err("temporal candidate SNR threshold is invalid".into());
    }
    if !(0.0 < temporal_min_psf_correlation && temporal_min_psf_correlation <= 1.0) {
        return Err("temporal PSF correlation threshold is invalid".into());
    }
    let Some((frames, input_rows, gated_rows)) = normalize_frames(
        candidate_peak_frames,
        min_candidate_snr,
        temporal_min_snr,
        temporal_reference_gate,
    ) else {
        return Ok(json!({"tracks":[],"audit":{}}));
    };
    if frames.iter().all(Vec::is_empty) {
        return Ok(
            json!({"tracks":[],"audit":{"input_rows":input_rows,"rows_after_snr_gate":gated_rows}}),
        );
    }

    let frame_count = frames.len();
    if image_support.is_some_and(|support| support.frames.len() != frame_count) {
        return Err("image_support frame count must match candidate_peak_frames".into());
    }
    let mut anchors = Vec::<[f64; 5]>::new();
    for (frame_index, current) in frames.iter().enumerate() {
        let aligned = current
            .iter()
            .map(|point| {
                [
                    point.x - cumulative_shifts[frame_index][0],
                    point.y - cumulative_shifts[frame_index][1],
                ]
            })
            .collect::<Vec<_>>();
        let old_anchor_count = anchors.len();
        let tree = if old_anchor_count > 0 {
            Some(KdTree::new(
                &anchors
                    .iter()
                    .map(|anchor| [anchor[0], anchor[1]])
                    .collect::<Vec<_>>(),
            )?)
        } else {
            None
        };
        for (index, point) in current.iter().enumerate() {
            let is_new = match tree.as_ref().and_then(|tree| tree.nearest(aligned[index])) {
                Some((_closest, distance)) => distance > link_radius_px,
                None => true,
            };
            if is_new {
                anchors.push([
                    aligned[index][0],
                    aligned[index][1],
                    point.snr,
                    if point.temporal { 1.0 } else { 0.0 },
                    point.reference_snr,
                ]);
            }
        }
    }
    let anchor_points = anchors
        .iter()
        .map(|row| [row[0], row[1]])
        .collect::<Vec<_>>();
    let anchor_tree = KdTree::new(&anchor_points)?;
    let mut matches = vec![vec![None::<usize>; frame_count]; anchors.len()];
    let mut scores = vec![vec![None::<f64>; frame_count]; anchors.len()];
    for frame_index in 0..frame_count {
        let current = &frames[frame_index];
        if current.is_empty() {
            continue;
        }
        let aligned = current
            .iter()
            .map(|point| {
                [
                    point.x - cumulative_shifts[frame_index][0],
                    point.y - cumulative_shifts[frame_index][1],
                ]
            })
            .collect::<Vec<_>>();
        let current_tree = KdTree::new(&aligned)?;
        let mut reciprocal = Vec::<(usize, usize, f64)>::new();
        for (anchor_index, anchor) in anchor_points.iter().enumerate() {
            let Some((current_index, distance)) = current_tree.nearest(*anchor) else {
                continue;
            };
            if distance > link_radius_px {
                continue;
            }
            if anchor_tree
                .nearest(aligned[current_index])
                .is_some_and(|(nearest_anchor, _)| nearest_anchor == anchor_index)
            {
                reciprocal.push((anchor_index, current_index, distance));
            }
        }
        reciprocal.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)));
        let mut used_current = vec![false; current.len()];
        for (anchor_index, current_index, _) in reciprocal {
            if used_current[current_index] {
                continue;
            }
            used_current[current_index] = true;
            matches[anchor_index][frame_index] = Some(current_index);
            scores[anchor_index][frame_index] = Some(current[current_index].snr);
        }
    }
    let eligible = matches
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            (row.iter().filter(|value| value.is_some()).count() >= persistent_min_presence)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        return Ok(json!({
            "tracks":[],
            "audit":{
                "input_rows":input_rows,
                "rows_after_snr_gate":gated_rows,
                "anchor_total":anchors.len(),
                "anchor_presence_ge_persistent":0
            }
        }));
    }
    let image_gate_counts = image_support.map(|support| {
        count_image_support(
            support,
            &frames,
            &matches,
            &eligible,
            temporal_min_psf_correlation,
        )
    });
    let quality = quality_by_frame(quality_tracks, frame_count, persistent_min_presence);
    let quality_trees = quality
        .iter()
        .map(|points| KdTree::new(points))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let merge_radius = 1.25f64.max((link_radius_px * 0.5).min(2.0));
    let times = normalize_times(frame_times, frame_count);
    let mut rendered = Vec::new();
    let mut audit = Map::new();
    audit.insert("input_rows".into(), json!(input_rows));
    audit.insert("rows_after_snr_gate".into(), json!(gated_rows));
    audit.insert("anchor_total".into(), json!(anchors.len()));
    audit.insert(
        "anchor_presence_ge_persistent".into(),
        json!(eligible.len()),
    );
    let mut reject_quality_overlap = 0;
    let mut reject_candidate_snr = 0;
    let mut reject_reference_snr = 0;
    let mut reject_motion_fit = 0;
    let mut reject_temporal_valid = 0;
    let mut reject_temporal_support = 0;
    let mut reject_line_overlap = 0;
    let mut reject_temporal_shape = 0;
    let mut reject_temporal_psf_correlation = 0;
    for anchor_index in &eligible {
        let mut points = Vec::<TrackPoint>::new();
        let mut overlaps_quality = false;
        for frame_index in 0..frame_count {
            let Some(current_index) = matches[*anchor_index][frame_index] else {
                continue;
            };
            let candidate = &frames[frame_index][current_index];
            let shift = cumulative_shifts[frame_index];
            points.push(TrackPoint {
                frame_index,
                detection_id: -1 - *anchor_index as i64,
                x: candidate.x,
                y: candidate.y,
                aligned_x: candidate.x - shift[0],
                aligned_y: candidate.y - shift[1],
                flux_snr: None,
                quality_passed: false,
                candidate_snr: Some(candidate.snr),
            });
            if quality_trees[frame_index]
                .nearest([candidate.x - shift[0], candidate.y - shift[1]])
                .is_some_and(|(_, distance)| distance <= merge_radius)
            {
                overlaps_quality = true;
            }
        }
        if overlaps_quality {
            reject_quality_overlap += 1;
            continue;
        }
        let matched_scores = scores[*anchor_index]
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let median_snr = statistics::median(&matched_scores)?;
        let temporal = anchors[*anchor_index][3] >= 0.5;
        let score_threshold = if temporal {
            temporal_min_snr
        } else {
            min_candidate_snr
        };
        if !median_snr.is_finite() || median_snr < score_threshold {
            reject_candidate_snr += 1;
            continue;
        }
        if temporal
            && (!anchors[*anchor_index][4].is_finite()
                || anchors[*anchor_index][4] < temporal_reference_gate)
        {
            reject_reference_snr += 1;
            continue;
        }
        if let Some(counts) = image_gate_counts
            .as_ref()
            .map(|counts| counts[*anchor_index])
        {
            for (name, value) in [
                ("presence", points.len()),
                ("temporal_valid", counts.valid),
                ("temporal_support", counts.support),
                ("temporal_shape", counts.shape),
                ("temporal_psf_correlation", counts.psf_correlation),
                ("line_overlap", counts.line_overlap),
            ] {
                let key = format!("{name}_{value}");
                let old = audit.get(&key).and_then(Value::as_u64).unwrap_or(0);
                audit.insert(key, json!(old + 1));
            }
            let minimum_valid = (0.8 * points.len() as f64).ceil().max(1.0) as usize;
            if counts.valid < minimum_valid {
                reject_temporal_valid += 1;
                continue;
            }
            let minimum_support = (0.6 * counts.valid as f64).ceil().max(1.0) as usize;
            if counts.support < minimum_support {
                reject_temporal_support += 1;
                let key = format!(
                    "support_reject_valid_{}_support_{}",
                    counts.valid, counts.support
                );
                let old = audit.get(&key).and_then(Value::as_u64).unwrap_or(0);
                audit.insert(key, json!(old + 1));
                continue;
            }
            let allowed_line_overlap = if points.len() >= 5 {
                (0.2 * points.len() as f64).floor().max(1.0) as usize
            } else {
                0
            };
            if counts.line_overlap > allowed_line_overlap {
                reject_line_overlap += 1;
                continue;
            }
            if counts.shape < minimum_support {
                reject_temporal_shape += 1;
                continue;
            }
            if temporal && counts.psf_correlation < minimum_support {
                reject_temporal_psf_correlation += 1;
                continue;
            }
        }
        let observations = points
            .iter()
            .map(|point| TrackObservation {
                frame_index: point.frame_index as i64,
                aligned_x: point.aligned_x,
                aligned_y: point.aligned_y,
            })
            .collect::<Vec<_>>();
        let fit = fit_track_motion(&observations, times.as_deref())?;
        if fit.fit_rms_px.is_none_or(|rms| rms > max_motion_fit_rms_px)
            || fit.displacement_px >= motion_min_displacement_px
        {
            reject_motion_fit += 1;
            continue;
        }
        rendered.push(json!({
            "track_id":-1,
            "classification":"persistent",
            "presence":points.len(),
            "displacement_px":fit.displacement_px,
            "speed_px_per_frame":fit.speed_px_per_frame,
            "fit_rms_px":fit.fit_rms_px,
            "quality_presence":0,
            "evidence_level":if temporal { "temporal_reference" } else { "candidate_consensus" },
            "points":points,
        }));
    }
    audit.insert("anchors_evaluated".into(), json!(eligible.len()));
    for (key, value) in [
        ("reject_quality_overlap", reject_quality_overlap),
        ("reject_median_candidate_snr", reject_candidate_snr),
        ("reject_temporal_reference_snr", reject_reference_snr),
        ("reject_temporal_valid", reject_temporal_valid),
        ("reject_temporal_support", reject_temporal_support),
        ("reject_line_overlap", reject_line_overlap),
        ("reject_temporal_shape", reject_temporal_shape),
        (
            "reject_temporal_psf_correlation",
            reject_temporal_psf_correlation,
        ),
        ("reject_motion_fit", reject_motion_fit),
    ] {
        if value > 0 {
            audit.insert(key.into(), json!(value));
        }
    }
    if !rendered.is_empty() {
        audit.insert("accepted".into(), json!(rendered.len()));
        let temporal_count = rendered
            .iter()
            .filter(|track| track["evidence_level"] == "temporal_reference")
            .count();
        let normal_count = rendered.len() - temporal_count;
        if temporal_count > 0 {
            audit.insert("accepted_temporal_reference".into(), json!(temporal_count));
        }
        if normal_count > 0 {
            audit.insert("accepted_candidate_consensus".into(), json!(normal_count));
        }
    }
    Ok(json!({"tracks":rendered,"audit":audit}))
}
