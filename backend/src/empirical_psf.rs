//! Empirical PSF estimation shared by injection and source-shape audits.
//!
//! The estimator intentionally uses only isolated, unsaturated quality detections.
//! It does not silently synthesize a fallback PSF when the input lacks enough
//! reference stars.
use crate::{
    fits,
    measurement::Detection,
    models::{NumericPlane, PixelData},
    Result,
};
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EmpiricalPsf {
    pub kernel: Vec<f32>,
    pub support_radius: usize,
    pub source_count: usize,
    pub median_fwhm_px: Option<f64>,
    pub median_ellipticity: Option<f64>,
}

impl EmpiricalPsf {
    pub fn side(&self) -> usize {
        self.support_radius * 2 + 1
    }

    pub fn as_dict(&self) -> Value {
        let kernel_sum: f64 = self.kernel.iter().map(|value| f64::from(*value)).sum();
        json!({
            "model": "empirical",
            "support_radius": self.support_radius,
            "source_count": self.source_count,
            "median_fwhm_px": self.median_fwhm_px,
            "median_ellipticity": self.median_ellipticity,
            "kernel_shape": [self.side(), self.side()],
            "kernel_sum": kernel_sum,
        })
    }
}

/// Native JSON request handler used by the JSONL engine and desktop worker.
///
/// The request accepts a FITS path and the source detections from the Rust
/// detection pipeline. The returned PSF includes its kernel so callers can
/// pass it directly to downstream fitting or injection code.
pub fn handle_request(request: &Value) -> Result<Value> {
    let fits_path = request["fits_path"].as_str().ok_or("missing FITS path")?;
    let sources: Vec<Detection> =
        serde_json::from_value(request.get("sources").cloned().ok_or("missing sources")?)
            .map_err(|error| format!("invalid sources: {error}"))?;
    let isolation_sources: Option<Vec<Detection>> = request
        .get("isolation_sources")
        .filter(|value| !value.is_null())
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| format!("invalid isolation_sources: {error}"))?;
    let support_radius = request
        .get("support_radius")
        .and_then(Value::as_u64)
        .map(usize::try_from)
        .transpose()
        .map_err(|error| error.to_string())?
        .unwrap_or(7);
    let max_sources = request
        .get("max_sources")
        .and_then(Value::as_u64)
        .map(usize::try_from)
        .transpose()
        .map_err(|error| error.to_string())?
        .unwrap_or(64);
    let excluded_ids: Vec<i64> = request
        .get("exclude_detection_ids")
        .filter(|value| !value.is_null())
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| format!("invalid exclude_detection_ids: {error}"))?
        .unwrap_or_default();
    let frame = fits::read_fits(fits_path, false)?;
    let psf = match &frame.data {
        PixelData::F32(values) => estimate(
            values.as_slice().into(),
            frame.height,
            frame.width,
            &sources,
            isolation_sources.as_deref(),
            support_radius,
            max_sources,
            &excluded_ids,
        )?,
        data => {
            let values = (0..data.len())
                .map(|index| data.value(index))
                .collect::<Vec<_>>();
            estimate(
                values.as_slice().into(),
                frame.height,
                frame.width,
                &sources,
                isolation_sources.as_deref(),
                support_radius,
                max_sources,
                &excluded_ids,
            )?
        }
    };
    let Some(psf) = psf else {
        return Ok(json!({"psf": Value::Null}));
    };
    let side = psf.side();
    let kernel = psf
        .kernel
        .chunks_exact(side)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let mut value = psf.as_dict();
    value["kernel"] = json!(kernel);
    Ok(json!({"psf":value}))
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

fn shifted_bilinear(
    input: &[f64],
    width: usize,
    height: usize,
    shift_x: f64,
    shift_y: f64,
) -> Vec<f64> {
    let mut output = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            // scipy.ndimage.shift samples input at output_coordinate - shift.
            let source_x = x as f64 - shift_x;
            let source_y = y as f64 - shift_y;
            if source_x < 0.0
                || source_y < 0.0
                || source_x > (width - 1) as f64
                || source_y > (height - 1) as f64
            {
                continue;
            }
            let x0 = source_x.floor() as usize;
            let y0 = source_y.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            let dx = source_x - x0 as f64;
            let dy = source_y - y0 as f64;
            let top = input[y0 * width + x0] * (1.0 - dx) + input[y0 * width + x1] * dx;
            let bottom = input[y1 * width + x0] * (1.0 - dx) + input[y1 * width + x1] * dx;
            output[y * width + x] = top * (1.0 - dy) + bottom * dy;
        }
    }
    output
}

/// Estimate a peak-normalized median PSF from isolated quality detections.
///
/// `sources` is the template pool. `isolation_sources`, when supplied, remains
/// the full-frame candidate set so a local template pool cannot hide a nearby
/// rejected or crowded source. Detection IDs in `excluded_ids` are omitted
/// from template extraction but remain in the isolation pool.
#[allow(clippy::too_many_arguments)]
pub fn estimate(
    image: NumericPlane<'_>,
    height: usize,
    width: usize,
    sources: &[Detection],
    isolation_sources: Option<&[Detection]>,
    support_radius: usize,
    max_sources: usize,
    excluded_ids: &[i64],
) -> Result<Option<EmpiricalPsf>> {
    if height == 0 || width == 0 || height.checked_mul(width) != Some(image.len()) {
        return Err("invalid empirical PSF image shape".into());
    }
    if support_radius < 3 || max_sources == 0 {
        return Err("support_radius must be at least 3 and max_sources must be positive".into());
    }
    let radius = support_radius;
    if radius > isize::MAX as usize {
        return Err("empirical PSF support radius is too large".into());
    }
    let side = radius
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or("empirical PSF support radius is too large")?;
    let _stamp_area = side
        .checked_mul(side)
        .ok_or("empirical PSF support area is too large")?;
    let margin = radius
        .checked_add(1)
        .ok_or("empirical PSF support radius is too large")?;
    if side > width || side > height {
        return Ok(None);
    }
    let aux_mask = fits::auxiliary_mask(height, width)?;
    let excluded: std::collections::HashSet<i64> = excluded_ids.iter().copied().collect();

    let mut eligible: Vec<(usize, f64)> = sources
        .iter()
        .enumerate()
        .filter_map(|(index, source)| {
            let metric = source.flux_snr.unwrap_or(source.snr);
            (source.x.is_finite()
                && source.y.is_finite()
                && source.x >= margin as f64
                && source.x < (width.saturating_sub(margin)) as f64
                && source.y >= margin as f64
                && source.y < (height.saturating_sub(margin)) as f64
                && !excluded.contains(&source.detection_id)
                && source.quality_passed
                && !source.flags.iter().any(|flag| {
                    matches!(
                        flag.as_str(),
                        "EDGE"
                            | "MASKED"
                            | "PARTIAL_MASKED"
                            | "SATURATED"
                            | "LINE_ARTIFACT"
                            | "NEGATIVE_OVERFLOW"
                    )
                })
                && source
                    .ellipticity
                    .is_some_and(|value| value.is_finite() && value <= 0.45)
                && source
                    .fwhm
                    .is_some_and(|value| value.is_finite() && (1.0..=8.0).contains(&value)))
            .then_some((
                index,
                if metric.is_finite() {
                    metric
                } else {
                    f64::NEG_INFINITY
                },
            ))
        })
        .collect();
    eligible.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    let isolation_pool = isolation_sources.unwrap_or(sources);
    let isolation_points: Vec<[f64; 2]> = isolation_pool
        .iter()
        .map(|source| [source.x, source.y])
        .collect();
    if isolation_points
        .iter()
        .flatten()
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err("empirical PSF isolation coordinates must be finite".into());
    }

    let mut stamps = Vec::<Vec<f32>>::new();
    let mut fwhm_values = Vec::new();
    let mut ellipticity_values = Vec::new();
    for (source_index, _) in eligible {
        if stamps.len() >= max_sources {
            break;
        }
        let source = &sources[source_index];
        if isolation_points.len() > 1 {
            let mut distances: Vec<f64> = isolation_points
                .iter()
                .map(|point| (point[0] - source.x).hypot(point[1] - source.y))
                .collect();
            distances.sort_by(f64::total_cmp);
            let nearest_other = if distances[0] <= 1.0e-8 {
                distances[1]
            } else {
                distances[0]
            };
            if nearest_other < radius.saturating_mul(3).max(18) as f64 {
                continue;
            }
        }

        let center_x = source.x.round_ties_even() as isize;
        let center_y = source.y.round_ties_even() as isize;
        let x0 = center_x - radius as isize;
        let y0 = center_y - radius as isize;
        if x0 < 0
            || y0 < 0
            || x0 + side as isize > width as isize
            || y0 + side as isize > height as isize
        {
            continue;
        }
        let x0 = x0 as usize;
        let y0 = y0 as usize;
        let mut stamp = Vec::with_capacity(side * side);
        let mut masked = false;
        let mut finite = true;
        for y in y0..y0 + side {
            for x in x0..x0 + side {
                let index = y * width + x;
                masked |= aux_mask[index];
                let value = image.value(index);
                finite &= value.is_finite();
                stamp.push(value);
            }
        }
        if masked || !finite {
            continue;
        }
        let local_background = if source.background.is_finite() {
            source.background
        } else {
            let mut background_sample = stamp.clone();
            median(&mut background_sample).unwrap_or(0.0)
        };
        let positive: Vec<f64> = stamp
            .into_iter()
            .map(|value| (value - local_background).max(0.0))
            .collect();
        let shift_x = center_x as f64 - source.x;
        let shift_y = center_y as f64 - source.y;
        let mut aligned = shifted_bilinear(&positive, side, side, shift_x, shift_y);
        let peak = aligned.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !peak.is_finite() || peak <= 0.0 {
            continue;
        }
        stamps.push(
            aligned
                .drain(..)
                .map(|value| (value / peak) as f32)
                .collect(),
        );
        fwhm_values.push(source.fwhm.unwrap());
        if let Some(value) = source.ellipticity.filter(|value| value.is_finite()) {
            ellipticity_values.push(value);
        }
    }
    if stamps.len() < 3 {
        return Ok(None);
    }

    let mut kernel = vec![0.0f32; side * side];
    let mut samples = Vec::with_capacity(stamps.len());
    for index in 0..kernel.len() {
        samples.clear();
        samples.extend(stamps.iter().map(|stamp| f64::from(stamp[index])));
        kernel[index] = median(&mut samples).unwrap_or(0.0) as f32;
    }
    for value in &mut kernel {
        *value = value.max(0.0);
    }
    let kernel_peak = kernel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !kernel_peak.is_finite() || kernel_peak <= 0.0 {
        return Ok(None);
    }
    for value in &mut kernel {
        *value /= kernel_peak;
    }
    Ok(Some(EmpiricalPsf {
        kernel,
        support_radius: radius,
        source_count: stamps.len(),
        median_fwhm_px: median(&mut fwhm_values),
        median_ellipticity: median(&mut ellipticity_values),
    }))
}

/// Compare one raw source cutout with an empirical PSF template.
///
/// Returns cosine correlation, normalized residual, central 3x3 energy
/// fraction, and least-squares amplitude. Invalid or under-supported cutouts
/// return `None` just like the Python reference.
pub fn similarity_metrics(
    image: NumericPlane<'_>,
    height: usize,
    width: usize,
    source: &Detection,
    psf: &EmpiricalPsf,
) -> Result<Option<(f64, f64, f64, f64)>> {
    let aux_mask = fits::auxiliary_mask(height, width)?;
    similarity_metrics_with_mask(image, height, width, source, psf, &aux_mask)
}

/// Compare one source cutout with an empirical PSF using a reusable image mask.
pub fn similarity_metrics_with_mask(
    image: NumericPlane<'_>,
    height: usize,
    width: usize,
    source: &Detection,
    psf: &EmpiricalPsf,
    aux_mask: &[bool],
) -> Result<Option<(f64, f64, f64, f64)>> {
    if height == 0 || width == 0 || height.checked_mul(width) != Some(image.len()) {
        return Err("invalid empirical PSF image shape".into());
    }
    if aux_mask.len() != image.len() {
        return Err("auxiliary mask shape does not match empirical PSF image".into());
    }
    let radius = psf.support_radius;
    let side = radius * 2 + 1;
    if psf.kernel.len() != side * side {
        return Err("empirical PSF kernel shape does not match support radius".into());
    }
    if psf.kernel.iter().any(|value| !value.is_finite())
        || psf.kernel.iter().copied().fold(f32::NEG_INFINITY, f32::max) <= 0.0
    {
        return Err("empirical PSF kernel must be finite and positive".into());
    }
    if !source.x.is_finite() || !source.y.is_finite() {
        return Ok(None);
    }
    let center_x = source.x.round_ties_even() as isize;
    let center_y = source.y.round_ties_even() as isize;
    let x0 = center_x - radius as isize;
    let y0 = center_y - radius as isize;
    if x0 < 0
        || y0 < 0
        || x0 + side as isize > width as isize
        || y0 + side as isize > height as isize
    {
        return Ok(None);
    }
    let mut valid = vec![false; side * side];
    let mut stamp = vec![0.0; side * side];
    for y in 0..side {
        for x in 0..side {
            let local = y * side + x;
            let global = (y0 as usize + y) * width + x0 as usize + x;
            let value = image.value(global);
            valid[local] = value.is_finite() && !aux_mask[global];
            stamp[local] = value;
        }
    }
    if valid.iter().filter(|value| **value).count() < (0.8 * stamp.len() as f64).ceil() as usize {
        return Ok(None);
    }
    let background = if source.background.is_finite() {
        source.background
    } else {
        let mut finite: Vec<f64> = stamp
            .iter()
            .zip(&valid)
            .filter_map(|(value, is_valid)| is_valid.then_some(*value))
            .collect();
        median(&mut finite).unwrap_or(0.0)
    };
    let observed: Vec<f64> = stamp
        .iter()
        .zip(&valid)
        .map(|(value, is_valid)| {
            if *is_valid {
                (value - background).max(0.0)
            } else {
                0.0
            }
        })
        .collect();
    let shift_x = center_x as f64 - source.x;
    let shift_y = center_y as f64 - source.y;
    let observed = shifted_bilinear(&observed, side, side, shift_x, shift_y);
    let template: Vec<f64> = psf
        .kernel
        .iter()
        .zip(&valid)
        .map(|(value, is_valid)| if *is_valid { f64::from(*value) } else { 0.0 })
        .collect();
    let data_norm = observed
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let template_norm = template
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let denominator = data_norm * template_norm;
    let epsilon = f64::EPSILON;
    if denominator <= epsilon {
        return Ok(None);
    }
    let dot = observed
        .iter()
        .zip(&template)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let template_energy = template.iter().map(|value| value * value).sum::<f64>();
    let amplitude = (dot / template_energy.max(epsilon)).max(0.0);
    let residual = observed
        .iter()
        .zip(&template)
        .map(|(value, template)| (value - amplitude * template).powi(2))
        .sum::<f64>()
        .sqrt()
        / data_norm.max(epsilon);
    let positive_sum = observed.iter().sum::<f64>();
    if positive_sum <= epsilon {
        return Ok(None);
    }
    let mut central_sum = 0.0;
    for y in radius - 1..=radius + 1 {
        for x in radius - 1..=radius + 1 {
            central_sum += observed[y * side + x];
        }
    }
    Ok(Some((
        dot / denominator,
        residual,
        central_sum / positive_sum,
        amplitude,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);

    fn fits_card(key: &str, value: &str) -> [u8; 80] {
        let mut result = [b' '; 80];
        let text = format!("{key:<8}= {value:>20}");
        result[..text.len()].copy_from_slice(text.as_bytes());
        result
    }

    fn temporary_f64_fits(width: usize, height: usize, pixels: &[f64]) -> PathBuf {
        let mut bytes = Vec::new();
        for (key, value) in [
            ("SIMPLE", "T".to_owned()),
            ("BITPIX", "-64".to_owned()),
            ("NAXIS", "2".to_owned()),
            ("NAXIS1", width.to_string()),
            ("NAXIS2", height.to_string()),
        ] {
            bytes.extend_from_slice(&fits_card(key, &value));
        }
        let mut end = [b' '; 80];
        end[..3].copy_from_slice(b"END");
        bytes.extend_from_slice(&end);
        bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
        for pixel in pixels {
            bytes.extend_from_slice(&pixel.to_be_bytes());
        }
        bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
        let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rst19-empirical-psf-{}-{sequence}.fits",
            std::process::id()
        ));
        fs::write(&path, bytes).unwrap();
        path
    }

    fn point(detection_id: i64, x: f64, y: f64, flux_snr: f64) -> Detection {
        Detection {
            detection_id,
            x,
            y,
            peak: 100.0,
            flux: 500.0,
            background: 100.0,
            noise: 2.0,
            snr: flux_snr,
            fwhm: Some(2.8),
            flags: Vec::new(),
            flux_error: Some(1.0),
            flux_snr: Some(flux_snr),
            filter_snr: Some(flux_snr),
            fwhm_x: Some(2.8),
            fwhm_y: Some(2.8),
            ellipticity: Some(0.1),
            sharpness: Some(0.5),
            footprint_pixels: Some(8),
            psf_support_pixels: Some(7),
            quality_passed: true,
            peak_x: Some(x.round_ties_even()),
            peak_y: Some(y.round_ties_even()),
            centroid_shift_px: Some(0.0),
            proposal_methods: vec!["gaussian".into()],
            proposal_scales: vec![1.0],
            proposal_snr: Some(flux_snr),
            nearest_gaussian_px: None,
            deblend_delta_bic: None,
            deblend_component_snr: None,
            repeated_code_count: None,
            range_anomaly_pixel_count: None,
            repeated_code_values: Vec::new(),
        }
    }

    fn gaussian_image(width: usize, height: usize, sources: &[Detection]) -> Vec<f64> {
        let mut image = vec![100.0; width * height];
        for source in sources {
            for y in 0..height {
                for x in 0..width {
                    let dx = x as f64 - source.x;
                    let dy = y as f64 - source.y;
                    image[y * width + x] +=
                        80.0 * (-0.5 * (dx * dx + dy * dy) / 1.2f64.powi(2)).exp();
                }
            }
        }
        image
    }

    #[test]
    fn estimates_peak_normalized_psf_from_isolated_quality_sources() {
        let sources = vec![
            point(1, 30.2, 130.1, 30.0),
            point(2, 90.4, 137.7, 60.0),
            point(3, 150.1, 132.8, 45.0),
        ];
        let image = gaussian_image(190, 210, &sources);
        let psf = estimate(
            image.as_slice().into(),
            210,
            190,
            &sources,
            None,
            3,
            64,
            &[],
        )
        .unwrap()
        .unwrap();
        assert_eq!(psf.source_count, 3);
        assert_eq!(psf.kernel.len(), 49);
        assert_eq!(psf.kernel.iter().copied().fold(0.0f32, f32::max), 1.0);
        assert!(psf
            .kernel
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
        assert!(psf.as_dict()["kernel_sum"].as_f64().unwrap() > 0.0);
        assert_eq!(psf.median_fwhm_px, Some(2.8));
        assert_eq!(psf.median_ellipticity, Some(0.1));

        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/empirical_psf.json")).unwrap();
        assert_eq!(
            fixture["source_count"].as_u64().unwrap() as usize,
            psf.source_count
        );
        let expected_kernel = fixture["kernel"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap())
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        assert_eq!(expected_kernel.len(), psf.kernel.len());
        for (index, (actual, expected)) in psf.kernel.iter().zip(&expected_kernel).enumerate() {
            assert!(
                (f64::from(*actual) - f64::from(*expected)).abs() <= 2.0e-6,
                "kernel[{index}] actual={actual:.9} expected={expected:.9}"
            );
        }
        let expected_similarity: Vec<f64> = fixture["similarity"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect();
        let actual_similarity =
            similarity_metrics(image.as_slice().into(), 210, 190, &sources[0], &psf)
                .unwrap()
                .unwrap();
        let actual_similarity = [
            actual_similarity.0,
            actual_similarity.1,
            actual_similarity.2,
            actual_similarity.3,
        ];
        for (index, (actual, expected)) in actual_similarity
            .iter()
            .zip(&expected_similarity)
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 2.0e-6 + 2.0e-7 * expected.abs(),
                "similarity[{index}] actual={actual:.12} expected={expected:.12}"
            );
        }
    }

    #[test]
    fn excludes_nearby_sources_and_explicit_template_ids_but_keeps_isolation_pool() {
        let sources = vec![
            point(1, 30.2, 130.1, 30.0),
            point(2, 90.4, 137.7, 60.0),
            point(3, 150.1, 132.8, 45.0),
            point(4, 94.0, 138.0, 100.0),
            point(5, 205.1, 175.2, 55.0),
            point(6, 210.3, 130.8, 50.0),
        ];
        let image = gaussian_image(240, 210, &sources);
        let candidates = [
            &sources[0],
            &sources[1],
            &sources[2],
            &sources[4],
            &sources[5],
        ]
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
        let psf = estimate(
            image.as_slice().into(),
            210,
            240,
            &candidates,
            Some(&sources),
            3,
            64,
            &[1],
        )
        .unwrap()
        .unwrap();
        assert_eq!(psf.source_count, 3);
    }

    #[test]
    fn similarity_metrics_are_finite_for_the_reference_template_source() {
        let sources = vec![
            point(1, 30.2, 130.1, 30.0),
            point(2, 90.4, 137.7, 60.0),
            point(3, 150.1, 132.8, 45.0),
        ];
        let image = gaussian_image(190, 210, &sources);
        let psf = estimate(
            image.as_slice().into(),
            210,
            190,
            &sources,
            None,
            3,
            64,
            &[],
        )
        .unwrap()
        .unwrap();
        let metrics = similarity_metrics(image.as_slice().into(), 210, 190, &sources[0], &psf)
            .unwrap()
            .unwrap();
        assert!(metrics.0.is_finite());
        assert!(metrics.1.is_finite());
        assert!(metrics.2.is_finite());
        assert!(metrics.3.is_finite());
        assert!((0.0..=1.0).contains(&metrics.0));
        assert!(metrics.1 >= 0.0);
        assert!((0.0..=1.0).contains(&metrics.2));
        assert!(metrics.3 >= 0.0);
    }

    #[test]
    fn json_request_reads_fits_and_returns_the_full_kernel() {
        let sources = vec![
            point(1, 30.2, 130.1, 30.0),
            point(2, 90.4, 137.7, 60.0),
            point(3, 150.1, 132.8, 45.0),
        ];
        let image = gaussian_image(190, 210, &sources);
        let path = temporary_f64_fits(190, 210, &image);
        let result = handle_request(&json!({
            "fits_path":path,
            "sources":sources,
            "support_radius":3
        }))
        .unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(result["psf"]["source_count"], 3);
        assert_eq!(result["psf"]["kernel_shape"], json!([7, 7]));
        assert_eq!(result["psf"]["kernel"].as_array().unwrap().len(), 7);
        assert_eq!(result["psf"]["kernel"][3][3], 1.0);
    }
}
