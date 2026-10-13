//! Local single-frame line-template review used to distinguish exposure trails
//! from compact point, double-point, and static elliptical profiles.
//!
//! Geometry is supplied by the connected-component proposal stage. This
//! module only compares fixed local templates; it does not infer motion.
use nalgebra::{DMatrix, DVector};
use serde_json::{json, Value};

const MODEL_MAX_LENGTH_PX: f64 = 128.0;
pub const MODEL_MIN_SNR: f64 = 5.0;
pub const MODEL_MIN_DELTA_BIC: f64 = 6.0;
pub const LONG_PROFILE_MIN_SNR: f64 = 5.0;
pub const LONG_PROFILE_MIN_DELTA_BIC: f64 = 6.0;

#[derive(Clone, Copy, Debug)]
struct Sample {
    x: f64,
    y: f64,
    value: f64,
    background_x: f64,
    background_y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Fit {
    rss: f64,
    first_template_snr: f64,
}

#[derive(Clone, Copy, Debug)]
struct Evidence {
    line_template_snr: Option<f64>,
    line_vs_point_delta_bic: Option<f64>,
    line_vs_double_delta_bic: Option<f64>,
    line_vs_background_delta_bic: Option<f64>,
    line_vs_elliptical_delta_bic: Option<f64>,
    single_frame_model: &'static str,
    single_frame_model_margin_bic: Option<f64>,
}

impl Evidence {
    fn not_fit() -> Self {
        Self {
            line_template_snr: None,
            line_vs_point_delta_bic: None,
            line_vs_double_delta_bic: None,
            line_vs_background_delta_bic: None,
            line_vs_elliptical_delta_bic: None,
            single_frame_model: "not_fit",
            single_frame_model_margin_bic: None,
        }
    }

    fn as_fields(self) -> Value {
        json!({
            "line_template_snr": self.line_template_snr,
            "line_vs_point_delta_bic": self.line_vs_point_delta_bic,
            "line_vs_double_delta_bic": self.line_vs_double_delta_bic,
            "line_vs_background_delta_bic": self.line_vs_background_delta_bic,
            "line_vs_elliptical_delta_bic": self.line_vs_elliptical_delta_bic,
            "single_frame_model": self.single_frame_model,
            "single_frame_model_margin_bic": self.single_frame_model_margin_bic,
        })
    }
}

fn finite_plane_shape<T>(plane: &[Vec<T>]) -> Option<(usize, usize)> {
    let height = plane.len();
    let width = plane.first()?.len();
    (height > 0 && width > 0 && plane.iter().all(|row| row.len() == width))
        .then_some((height, width))
}

fn rounded_candidates(length: f64, factors: &[f64], floor: f64) -> Vec<f64> {
    let mut values = factors
        .iter()
        .map(|factor| ((length * factor).max(floor) * 10_000.0).round() / 10_000.0)
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values.dedup_by(|left, right| *left == *right);
    values
}

fn bic(rss: f64, sample_count: usize, parameter_count: usize) -> f64 {
    if sample_count == 0 || !rss.is_finite() {
        return f64::INFINITY;
    }
    let n = sample_count as f64;
    n * (rss.max(f64::MIN_POSITIVE) / n).ln() + parameter_count as f64 * n.ln()
}

/// Numerically stable approximation to the error function, with maximum
/// absolute error below 1.6e-7. The approximation is far smaller than the
/// pixel/noise uncertainty of the fitted profiles.
fn erf(value: f64) -> f64 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial =
        (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

fn fit_template(samples: &[Sample], templates: &[Vec<f64>], noise: f64) -> Option<Fit> {
    if samples.is_empty()
        || templates
            .iter()
            .any(|template| template.len() != samples.len())
    {
        return None;
    }
    let safe_noise = noise.max(f64::EPSILON);
    let template_count = templates.len();
    let mut best: Option<(f64, Vec<usize>, Vec<f64>, DMatrix<f64>)> = None;

    // Enumerate active sets so negative template amplitudes are rejected and
    // the remaining background/template coefficients are re-fitted.
    for active_mask in 0usize..(1usize << template_count) {
        let active = (0..template_count)
            .filter(|index| active_mask & (1 << index) != 0)
            .collect::<Vec<_>>();
        let column_count = 3 + active.len();
        let mut gram = DMatrix::<f64>::zeros(column_count, column_count);
        let mut rhs = DVector::<f64>::zeros(column_count);
        for sample_index in 0..samples.len() {
            let sample = samples[sample_index];
            let mut columns = Vec::with_capacity(column_count);
            columns.extend([1.0, sample.background_x, sample.background_y]);
            columns.extend(active.iter().map(|index| templates[*index][sample_index]));
            for row in 0..column_count {
                rhs[row] += columns[row] * sample.value;
                for column in 0..column_count {
                    gram[(row, column)] += columns[row] * columns[column];
                }
            }
        }
        let Some(mut coefficients) = gram.clone().lu().solve(&rhs) else {
            continue;
        };
        let tolerance = 1e-9 * coefficients.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        if (3..column_count).any(|index| coefficients[index] < -tolerance) {
            continue;
        }
        for index in 3..column_count {
            coefficients[index] = coefficients[index].max(0.0);
        }
        let mut rss = 0.0;
        for (sample_index, sample) in samples.iter().enumerate() {
            let mut prediction = coefficients[0]
                + coefficients[1] * sample.background_x
                + coefficients[2] * sample.background_y;
            for (active_position, template_index) in active.iter().enumerate() {
                prediction +=
                    coefficients[3 + active_position] * templates[*template_index][sample_index];
            }
            let residual = (sample.value - prediction) / safe_noise;
            rss += residual * residual;
        }
        if best.as_ref().is_none_or(|current| rss < current.0) {
            best = Some((rss, active, coefficients.as_slice().to_vec(), gram));
        }
    }

    let (rss, active, coefficients, gram) = best?;
    let Some(active_position) = active.iter().position(|index| *index == 0) else {
        return Some(Fit {
            rss,
            first_template_snr: 0.0,
        });
    };
    let coefficient_index = 3 + active_position;
    let variance = gram
        .try_inverse()
        .map(|inverse| inverse[(coefficient_index, coefficient_index)] * safe_noise.powi(2))
        .unwrap_or(f64::INFINITY);
    let snr = if variance > 0.0 && variance.is_finite() {
        coefficients[coefficient_index] / variance.sqrt()
    } else {
        0.0
    };
    Some(Fit {
        rss,
        first_template_snr: snr,
    })
}

fn line_template(
    samples: &[Sample],
    length: f64,
    cx: f64,
    cy: f64,
    angle: f64,
    sigma: f64,
) -> Vec<f64> {
    let cosine = angle.to_radians().cos();
    let sine = angle.to_radians().sin();
    let mut values = Vec::with_capacity(samples.len());
    if length <= 32.0 {
        let sample_count = ((length * 2.0).ceil() as usize + 1).clamp(9, 257);
        for sample in samples {
            let mut total = 0.0;
            for index in 0..sample_count {
                let offset = -0.5 * length
                    + length * index as f64 / (sample_count.saturating_sub(1).max(1) as f64);
                let dx = sample.x - (cx + offset * cosine);
                let dy = sample.y - (cy + offset * sine);
                total += (-0.5 * (dx * dx + dy * dy) / sigma.powi(2)).exp();
            }
            values.push(total / sample_count as f64);
        }
    } else {
        let edge_scale = std::f64::consts::SQRT_2 * sigma;
        for sample in samples {
            let dx = sample.x - cx;
            let dy = sample.y - cy;
            let along = dx * cosine + dy * sine;
            let across = -dx * sine + dy * cosine;
            let segment = 0.5
                * (erf((along + 0.5 * length) / edge_scale)
                    - erf((along - 0.5 * length) / edge_scale));
            values.push(segment * (-0.5 * (across / sigma).powi(2)).exp());
        }
    }
    normalize_template(values)
}

fn normalize_template(mut values: Vec<f64>) -> Vec<f64> {
    let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if peak > 0.0 && peak.is_finite() {
        for value in &mut values {
            *value /= peak;
        }
    }
    values
}

fn samples_in_patch(
    residual: &[Vec<f32>],
    valid: &[Vec<bool>],
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    cx: f64,
    cy: f64,
    background_x_scale: f64,
    background_y_scale: f64,
) -> Vec<Sample> {
    let mut samples = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let value = f64::from(residual[y][x]);
            if valid[y][x] && value.is_finite() {
                samples.push(Sample {
                    x: x as f64,
                    y: y as f64,
                    value,
                    background_x: (x as f64 - cx) / background_x_scale.max(1.0),
                    background_y: (y as f64 - cy) / background_y_scale.max(1.0),
                });
            }
        }
    }
    samples
}

fn empty_patch() -> Evidence {
    Evidence::not_fit()
}

fn long_line_profile(
    residual: &[Vec<f32>],
    valid: &[Vec<bool>],
    cx: f64,
    cy: f64,
    length: f64,
    width: f64,
    angle_deg: f64,
    psf_fwhm: f64,
    noise: f64,
) -> Evidence {
    let Some((height, image_width)) = finite_plane_shape(residual) else {
        return empty_patch();
    };
    let psf_sigma = (psf_fwhm / 2.354_820_045).max(0.45);
    let transverse_sigma = psf_sigma.max((width / 4.0).max(0.45).min(8.0));
    let half_length = 0.5 * length;
    let angle = angle_deg.to_radians();
    let cosine = angle.cos();
    let sine = angle.sin();
    let along_margin = 4.0f64.max(4.0 * psf_sigma);
    let across_radius = 6.0f64.max(4.0 * transverse_sigma + 2.0);
    let x_radius = half_length * cosine.abs() + across_radius * sine.abs() + along_margin;
    let y_radius = half_length * sine.abs() + across_radius * cosine.abs() + along_margin;
    let x0 = (cx - x_radius).floor().max(0.0) as usize;
    let x1 = ((cx + x_radius + 1.0).ceil() as usize).min(image_width);
    let y0 = (cy - y_radius).floor().max(0.0) as usize;
    let y1 = ((cy + y_radius + 1.0).ceil() as usize).min(height);
    if x1.saturating_sub(x0) < 15 || y1.saturating_sub(y0) < 15 || valid.len() != height {
        return empty_patch();
    }
    let samples = samples_in_patch(residual, valid, x0, x1, y0, y1, cx, cy, x_radius, y_radius);
    if samples.len() < 100 {
        return empty_patch();
    }
    let edge_scale = std::f64::consts::SQRT_2 * psf_sigma;
    let line = normalize_template(
        samples
            .iter()
            .map(|sample| {
                let dx = sample.x - cx;
                let dy = sample.y - cy;
                let along = dx * cosine + dy * sine;
                let across = -dx * sine + dy * cosine;
                0.5 * (erf((along + half_length) / edge_scale)
                    - erf((along - half_length) / edge_scale))
                    * (-0.5 * (across / transverse_sigma).powi(2)).exp()
            })
            .collect(),
    );
    let Some(line_fit) = fit_template(&samples, &[line], noise) else {
        return empty_patch();
    };
    let Some(background_fit) = fit_template(&samples, &[], noise) else {
        return empty_patch();
    };
    let line_bic = bic(line_fit.rss, samples.len(), 5);
    let background_bic = bic(background_fit.rss, samples.len(), 3);
    let delta = background_bic - line_bic;
    if !line_fit.first_template_snr.is_finite() || !delta.is_finite() {
        return empty_patch();
    }
    let supported =
        line_fit.first_template_snr >= LONG_PROFILE_MIN_SNR && delta >= LONG_PROFILE_MIN_DELTA_BIC;
    Evidence {
        line_template_snr: Some(line_fit.first_template_snr),
        line_vs_point_delta_bic: None,
        line_vs_double_delta_bic: None,
        line_vs_background_delta_bic: Some(delta),
        line_vs_elliptical_delta_bic: None,
        single_frame_model: if supported {
            "long_line_profile"
        } else {
            "background"
        },
        single_frame_model_margin_bic: supported.then_some(delta),
    }
}

fn local_model(
    residual: &[Vec<f32>],
    valid: &[Vec<bool>],
    cx: f64,
    cy: f64,
    length: f64,
    width: f64,
    angle_deg: f64,
    psf_fwhm: f64,
    noise: f64,
) -> Evidence {
    let Some((height, image_width)) = finite_plane_shape(residual) else {
        return empty_patch();
    };
    if valid.len() != height || valid.iter().any(|row| row.len() != image_width) {
        return empty_patch();
    }
    let sigma = (psf_fwhm / 2.354_820_045).max(0.45);
    let radius = (0.5 * length + 4.0 * sigma + 3.0).clamp(6.0, 96.0);
    let x0 = (cx - radius).floor().max(0.0) as usize;
    let x1 = ((cx + radius + 1.0).ceil() as usize).min(image_width);
    let y0 = (cy - radius).floor().max(0.0) as usize;
    let y1 = ((cy + radius + 1.0).ceil() as usize).min(height);
    if x1.saturating_sub(x0) < 7 || y1.saturating_sub(y0) < 7 {
        return empty_patch();
    }
    let samples = samples_in_patch(residual, valid, x0, x1, y0, y1, cx, cy, radius, radius);
    if samples.len() < 25 {
        return empty_patch();
    }

    let point = normalize_template(
        samples
            .iter()
            .map(|sample| {
                (-0.5 * ((sample.x - cx).powi(2) + (sample.y - cy).powi(2)) / sigma.powi(2)).exp()
            })
            .collect(),
    );
    let point_fit = fit_template(&samples, &[point], noise);
    let Some(point_fit) = point_fit else {
        return empty_patch();
    };

    let angle = angle_deg.to_radians();
    let (cosine, sine) = (angle.cos(), angle.sin());
    let (normal_x, normal_y) = (-sine, cosine);
    let line_lengths = rounded_candidates(length, &[0.75, 1.0, 1.25], 0.5);
    let mut line_fit: Option<(Fit, f64)> = None;
    for candidate_length in line_lengths {
        let template = line_template(&samples, candidate_length, cx, cy, angle_deg, sigma);
        if let Some(candidate) = fit_template(&samples, &[template], noise) {
            if line_fit.is_none_or(|(current, _)| candidate.rss < current.rss) {
                line_fit = Some((candidate, candidate_length));
            }
        }
    }
    let Some((line_fit, _line_length)) = line_fit else {
        return empty_patch();
    };

    let ellipse_major_base = sigma.max(length / 4.0);
    let ellipse_minor_base = 0.45f64.max(ellipse_major_base.min(width / 4.0));
    let major_candidates = rounded_candidates(ellipse_major_base, &[0.75, 1.0, 1.25], 0.45);
    let minor_candidates = rounded_candidates(ellipse_minor_base, &[0.75, 1.0, 1.25], 0.45);
    let mut ellipse_fit: Option<(Fit, f64, f64)> = None;
    for major_sigma in &major_candidates {
        for minor_sigma in &minor_candidates {
            let template = normalize_template(
                samples
                    .iter()
                    .map(|sample| {
                        let dx = sample.x - cx;
                        let dy = sample.y - cy;
                        let along = dx * cosine + dy * sine;
                        let across = dx * normal_x + dy * normal_y;
                        (-0.5 * ((along / major_sigma).powi(2) + (across / minor_sigma).powi(2)))
                            .exp()
                    })
                    .collect(),
            );
            if let Some(candidate) = fit_template(&samples, &[template], noise) {
                if ellipse_fit.is_none_or(|(current, _, _)| candidate.rss < current.rss) {
                    ellipse_fit = Some((candidate, *major_sigma, *minor_sigma));
                }
            }
        }
    }

    let double_separations = rounded_candidates(length, &[0.5, 0.75, 1.0, 1.25], 0.5);
    let (along_offsets, across_offsets): (&[f64], &[f64]) = if length <= 48.0 {
        (&[-2.0, -1.0, 0.0, 1.0, 2.0], &[-1.0, 0.0, 1.0])
    } else {
        (&[0.0], &[0.0])
    };
    let mut double_fit: Option<Fit> = None;
    for separation in double_separations {
        for along_offset in along_offsets {
            for across_offset in across_offsets {
                let midpoint_x = cx + along_offset * cosine + across_offset * normal_x;
                let midpoint_y = cy + along_offset * sine + across_offset * normal_y;
                let half = 0.5 * separation;
                let mut first = Vec::with_capacity(samples.len());
                let mut second = Vec::with_capacity(samples.len());
                for sample in &samples {
                    let x1 = midpoint_x - half * cosine;
                    let y1 = midpoint_y - half * sine;
                    let x2 = midpoint_x + half * cosine;
                    let y2 = midpoint_y + half * sine;
                    first.push(
                        (-0.5 * ((sample.x - x1).powi(2) + (sample.y - y1).powi(2))
                            / sigma.powi(2))
                        .exp(),
                    );
                    second.push(
                        (-0.5 * ((sample.x - x2).powi(2) + (sample.y - y2).powi(2))
                            / sigma.powi(2))
                        .exp(),
                    );
                }
                let templates = [normalize_template(first), normalize_template(second)];
                if let Some(candidate) = fit_template(&samples, &templates, noise) {
                    if double_fit.is_none_or(|current| candidate.rss < current.rss) {
                        double_fit = Some(candidate);
                    }
                }
            }
        }
    }

    let background_fit = fit_template(&samples, &[], noise);
    let (Some(background_fit), Some(double_fit), Some((ellipse_fit, _, _))) =
        (background_fit, double_fit, ellipse_fit)
    else {
        return empty_patch();
    };
    let bics = [
        ("background", bic(background_fit.rss, samples.len(), 3)),
        ("point", bic(point_fit.rss, samples.len(), 4)),
        ("double", bic(double_fit.rss, samples.len(), 8)),
        ("line", bic(line_fit.rss, samples.len(), 5)),
        ("elliptical_point", bic(ellipse_fit.rss, samples.len(), 6)),
    ];
    let mut ranked = bics.to_vec();
    ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
    let best_model = ranked[0].0;
    let margin = ranked.get(1).map(|next| next.1 - ranked[0].1);
    let _ = (cx, cy, width); // Keep the review inputs visible for parity auditing.
    Evidence {
        line_template_snr: Some(line_fit.first_template_snr),
        line_vs_point_delta_bic: Some(bics[1].1 - bics[3].1),
        line_vs_double_delta_bic: Some(bics[2].1 - bics[3].1),
        line_vs_background_delta_bic: Some(bics[0].1 - bics[3].1),
        line_vs_elliptical_delta_bic: Some(bics[4].1 - bics[3].1),
        single_frame_model: best_model,
        single_frame_model_margin_bic: margin,
    }
}

/// Compare the current-frame residual cutout against the same fixed local
/// line, single PSF, double PSF, static ellipse, and background alternatives.
pub fn evaluate(
    residual: &[Vec<f32>],
    valid: &[Vec<bool>],
    center_x: f64,
    center_y: f64,
    length_px: f64,
    width_px: f64,
    angle_deg: f64,
    psf_fwhm: f64,
    noise: f64,
) -> Value {
    let evidence = if !length_px.is_finite()
        || length_px <= 0.0
        || !psf_fwhm.is_finite()
        || psf_fwhm <= 0.0
        || !noise.is_finite()
        || noise <= 0.0
        || !center_x.is_finite()
        || !center_y.is_finite()
        || !angle_deg.is_finite()
    {
        Evidence::not_fit()
    } else if length_px > MODEL_MAX_LENGTH_PX {
        long_line_profile(
            residual, valid, center_x, center_y, length_px, width_px, angle_deg, psf_fwhm, noise,
        )
    } else {
        local_model(
            residual, valid, center_x, center_y, length_px, width_px, angle_deg, psf_fwhm, noise,
        )
    };
    evidence.as_fields()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_frame(sources: &[(f64, f64, f64)], sigma: f64) -> Vec<Vec<f32>> {
        let size = 64usize;
        (0..size)
            .map(|y| {
                (0..size)
                    .map(|x| {
                        sources
                            .iter()
                            .map(|(sx, sy, amplitude)| {
                                let distance = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                                amplitude * (-0.5 * distance / sigma.powi(2)).exp()
                            })
                            .sum::<f64>() as f32
                    })
                    .collect()
            })
            .collect()
    }

    fn all_valid(height: usize, width: usize) -> Vec<Vec<bool>> {
        vec![vec![true; width]; height]
    }

    #[test]
    fn compact_double_is_preferred_over_a_line_for_two_close_psfs() {
        let image = gaussian_frame(&[(26.0, 32.0, 100.0), (38.0, 32.0, 75.0)], 1.27);
        let result = evaluate(
            &image,
            &all_valid(64, 64),
            32.0,
            32.0,
            12.0,
            3.0,
            0.0,
            3.0,
            1.0,
        );
        assert_eq!(result["single_frame_model"], "double");
        assert!(
            (result["line_template_snr"].as_f64().unwrap() - 177.207_573_853_938_3).abs() < 1e-5
        );
        assert!(
            (result["line_vs_double_delta_bic"].as_f64().unwrap() + 10_550.697_977_232_487).abs()
                < 1e-3
        );
    }

    #[test]
    fn finite_exposure_line_is_preferred_for_uniform_streak() {
        let image = (0..64)
            .map(|y| {
                (0..64)
                    .map(|x| {
                        if (18..=46).contains(&x) {
                            (90.0 * (-0.5 * ((y as f64 - 32.0) / 1.25).powi(2)).exp()) as f32
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect::<Vec<Vec<f32>>>();
        let result = evaluate(
            &image,
            &all_valid(64, 64),
            32.0,
            32.0,
            28.0,
            3.0,
            0.0,
            3.0,
            1.0,
        );
        assert_eq!(result["single_frame_model"], "line");
        assert!(result["line_template_snr"].as_f64().unwrap() >= MODEL_MIN_SNR);
        assert!(result["line_vs_double_delta_bic"].as_f64().unwrap() >= MODEL_MIN_DELTA_BIC);
        assert!(
            (result["line_vs_double_delta_bic"].as_f64().unwrap() - 7_802.131_916_488_355).abs()
                < 1e-3
        );
    }

    #[test]
    fn very_long_streak_uses_full_length_profile_model() {
        let image = (0..256)
            .map(|y| {
                (0..256)
                    .map(|x| {
                        if (28..=228).contains(&x) {
                            (90.0 * (-0.5 * ((y as f64 - 128.0) / 1.25).powi(2)).exp()) as f32
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect::<Vec<Vec<f32>>>();
        let result = evaluate(
            &image,
            &all_valid(256, 256),
            128.0,
            128.0,
            200.0,
            3.0,
            0.0,
            3.0,
            1.0,
        );
        println!("long-line model evidence: {result}");
        assert_eq!(result["single_frame_model"], "long_line_profile");
        assert!(result["line_template_snr"].as_f64().unwrap() >= LONG_PROFILE_MIN_SNR);
        assert!(
            result["line_vs_background_delta_bic"].as_f64().unwrap() >= LONG_PROFILE_MIN_DELTA_BIC
        );
    }
}
