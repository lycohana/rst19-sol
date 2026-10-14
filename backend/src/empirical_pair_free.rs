//! Bounded free-position empirical PSF comparison for K=1 and K=2.
//!
//! This is a local model diagnostic. It is not a physical-source identity or
//! probability estimate.

use crate::{
    empirical_pair_fit::{
        self, mask_modes, ordered_fits_paths, patch_bounds, resolve_repeated_codes, shifted_basis,
        MASK_MODES,
    },
    empirical_psf::EmpiricalPsf,
    feature_psf_leaveout, fits,
    measurement::Detection,
    models::FitsFrame,
    Result,
};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

const CSV_FIELDS: &[&str] = &[
    "frame_index",
    "path",
    "mask_mode",
    "patch_x0",
    "patch_y0",
    "patch_width",
    "patch_height",
    "sample_count",
    "masked_pixel_count",
    "single_rss",
    "double_rss",
    "single_bic",
    "double_bic",
    "delta_bic_single_minus_double",
    "single_initial_position",
    "single_fitted_position",
    "single_position_offset",
    "single_position_at_bound",
    "single_component_snr",
    "single_optimizer_success",
    "single_optimizer_nfev",
    "double_initial_positions",
    "double_fitted_positions",
    "double_position_offsets",
    "double_positions_at_bound",
    "double_amplitudes",
    "double_component_snr",
    "double_second_component_snr",
    "double_active_component_count",
    "double_optimizer_success",
    "double_optimizer_nfev",
    "fitted_separation_px",
    "confirmation_line",
    "note",
];

pub const HELP: &str = "在同一经验 PSF 和掩膜下联合优化 K=1/K=2 的源位置、幅度和背景\n\nUsage: rst19-empirical-pair-free DATA_DIR --source-catalog CSV --primary-id ID --secondary-id ID [OPTIONS]\n\nOptions: --support-radius N --max-template-sources N --patch-padding-px N\n         --single-position-radius-px N --double-position-radius-px N\n         --minimum-fitted-separation-px N --mask-modes LIST --repeated-code-values LIST\n         --shifts-json JSON --out-dir DIR";

#[derive(Clone, Debug, Serialize)]
pub struct EmpiricalFreePairRow {
    pub frame_index: usize,
    pub path: String,
    pub mask_mode: String,
    pub patch_x0: usize,
    pub patch_y0: usize,
    pub patch_width: usize,
    pub patch_height: usize,
    pub sample_count: usize,
    pub masked_pixel_count: usize,
    pub single_rss: Option<f64>,
    pub double_rss: Option<f64>,
    pub single_bic: Option<f64>,
    pub double_bic: Option<f64>,
    pub delta_bic_single_minus_double: Option<f64>,
    pub single_initial_position: Option<[f64; 2]>,
    pub single_fitted_position: Option<[f64; 2]>,
    pub single_position_offset: Option<[f64; 2]>,
    pub single_position_at_bound: bool,
    pub single_component_snr: Option<f64>,
    pub single_optimizer_success: bool,
    pub single_optimizer_nfev: Option<usize>,
    pub double_initial_positions: Vec<[f64; 2]>,
    pub double_fitted_positions: Vec<[f64; 2]>,
    pub double_position_offsets: Vec<[f64; 2]>,
    pub double_positions_at_bound: Vec<bool>,
    pub double_amplitudes: Vec<f64>,
    pub double_component_snr: Vec<f64>,
    pub double_second_component_snr: Option<f64>,
    pub double_active_component_count: usize,
    pub double_optimizer_success: bool,
    pub double_optimizer_nfev: Option<usize>,
    pub fitted_separation_px: Option<f64>,
    pub confirmation_line: bool,
    pub note: String,
}

#[derive(Clone, Debug)]
pub struct EmpiricalFreePairResult {
    pub frame_count: usize,
    pub primary_detection_id: Option<i64>,
    pub secondary_detection_id: Option<i64>,
    pub primary_target_xy: [f64; 2],
    pub secondary_target_xy: [f64; 2],
    pub frame_shifts: Vec<[f64; 2]>,
    pub template_frame_path: String,
    pub template_catalog_path: Option<String>,
    pub mask_modes: Vec<String>,
    pub support_radius: usize,
    pub patch_padding_px: f64,
    pub single_position_radius_px: f64,
    pub double_position_radius_px: f64,
    pub confirmation_delta_bic: f64,
    pub confirmation_component_snr: f64,
    pub minimum_fitted_separation_px: f64,
    pub empirical_psf: EmpiricalPsf,
    pub parameters: Value,
    pub rows: Vec<EmpiricalFreePairRow>,
    pub conclusion: String,
}

impl EmpiricalFreePairResult {
    fn as_json(&self) -> Value {
        json!({
            "frame_count": self.frame_count,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "primary_target_xy": self.primary_target_xy,
            "secondary_target_xy": self.secondary_target_xy,
            "frame_shifts": self.frame_shifts,
            "template_frame_path": self.template_frame_path,
            "template_catalog_path": self.template_catalog_path,
            "mask_modes": self.mask_modes,
            "support_radius": self.support_radius,
            "patch_padding_px": self.patch_padding_px,
            "single_position_radius_px": self.single_position_radius_px,
            "double_position_radius_px": self.double_position_radius_px,
            "confirmation_delta_bic": self.confirmation_delta_bic,
            "confirmation_component_snr": self.confirmation_component_snr,
            "minimum_fitted_separation_px": self.minimum_fitted_separation_px,
            "empirical_psf": self.empirical_psf.as_dict(),
            "parameters": self.parameters,
            "rows": self.rows,
            "conclusion": self.conclusion,
            "note": "K=1/K=2 在同一窗口和掩膜下联合优化位置、幅度与平面背景；位置有边界，因此 BIC 只表示局部研究模型证据，不是星表身份、恒星概率或物理真值。"
        })
    }
}

#[derive(Clone, Debug)]
struct Options {
    data_dir: PathBuf,
    source_catalog: PathBuf,
    primary_id: i64,
    secondary_id: i64,
    support_radius: usize,
    max_template_sources: usize,
    patch_padding_px: Option<f64>,
    single_position_radius_px: f64,
    double_position_radius_px: f64,
    minimum_fitted_separation_px: f64,
    mask_modes: Vec<String>,
    repeated_code_values: Option<Vec<i64>>,
    shifts_json: Option<PathBuf>,
    out_dir: PathBuf,
}

fn argument_value(
    args: &[String],
    index: &mut usize,
    name: &str,
    inline: Option<&str>,
) -> Result<String> {
    if let Some(value) = inline {
        return Ok(value.to_owned());
    }
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("--{name} requires a value"))
}

fn parse_codes(value: &str) -> Result<Vec<i64>> {
    value
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .map(|item| {
            item.trim()
                .parse::<i64>()
                .map_err(|_| format!("invalid repeated-code value: {item}"))
        })
        .collect()
}

fn parse_args(args: &[String]) -> Result<Options> {
    let (mut data_dir, mut source_catalog, mut primary_id, mut secondary_id) =
        (None, None, None, None);
    let mut support_radius = 7usize;
    let mut max_template_sources = 64usize;
    let mut patch_padding_px = None;
    let mut single_position_radius_px = 2.0;
    let mut double_position_radius_px = 1.25;
    let mut minimum_fitted_separation_px = 1.0;
    let mut mask_modes = "raw,range_masked,sentinel_masked".to_owned();
    let mut repeated_code_values = None;
    let mut shifts_json = None;
    let mut out_dir = PathBuf::from("tmp/empirical-pair-free-audit");
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if !arg.starts_with('-') {
            if data_dir.replace(PathBuf::from(arg)).is_some() {
                return Err("only one data directory is allowed".into());
            }
            i += 1;
            continue;
        }
        let (name, inline) = arg[2..]
            .split_once('=')
            .map(|(n, v)| (n, Some(v)))
            .unwrap_or((&arg[2..], None));
        let value = argument_value(args, &mut i, name, inline)?;
        match name {
            "source-catalog" => source_catalog = Some(PathBuf::from(value)),
            "primary-id" => primary_id = Some(value.parse().map_err(|_| "invalid --primary-id")?),
            "secondary-id" => {
                secondary_id = Some(value.parse().map_err(|_| "invalid --secondary-id")?)
            }
            "support-radius" => {
                support_radius = value.parse().map_err(|_| "invalid --support-radius")?
            }
            "max-template-sources" => {
                max_template_sources = value
                    .parse()
                    .map_err(|_| "invalid --max-template-sources")?
            }
            "patch-padding-px" => {
                patch_padding_px = Some(value.parse().map_err(|_| "invalid --patch-padding-px")?)
            }
            "single-position-radius-px" => {
                single_position_radius_px = value
                    .parse()
                    .map_err(|_| "invalid --single-position-radius-px")?
            }
            "double-position-radius-px" => {
                double_position_radius_px = value
                    .parse()
                    .map_err(|_| "invalid --double-position-radius-px")?
            }
            "minimum-fitted-separation-px" => {
                minimum_fitted_separation_px = value
                    .parse()
                    .map_err(|_| "invalid --minimum-fitted-separation-px")?
            }
            "mask-modes" => mask_modes = value,
            "repeated-code-values" => repeated_code_values = Some(parse_codes(&value)?),
            "shifts-json" => shifts_json = Some(PathBuf::from(value)),
            "out-dir" => out_dir = PathBuf::from(value),
            "help" => return Err(HELP.into()),
            _ => return Err(format!("unknown argument --{name}")),
        }
        i += 1;
    }
    Ok(Options {
        data_dir: data_dir.ok_or("missing data directory")?,
        source_catalog: source_catalog.ok_or("missing --source-catalog")?,
        primary_id: primary_id.ok_or("missing --primary-id")?,
        secondary_id: secondary_id.ok_or("missing --secondary-id")?,
        support_radius,
        max_template_sources,
        patch_padding_px,
        single_position_radius_px,
        double_position_radius_px,
        minimum_fitted_separation_px,
        mask_modes: mask_modes
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect(),
        repeated_code_values,
        shifts_json,
        out_dir,
    })
}

fn read_shifts(path: &Path) -> Result<Vec<[f64; 2]>> {
    let value: Value = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("invalid shifts JSON: {e}"))?;
    value
        .get("cumulative_shifts")
        .and_then(Value::as_array)
        .ok_or("shifts JSON requires cumulative_shifts")?
        .iter()
        .map(|item| {
            let pair = item
                .as_array()
                .ok_or("each cumulative shift must be [x,y]")?;
            if pair.len() != 2 {
                return Err("each cumulative shift must be [x,y]".into());
            }
            let shift = [
                pair[0].as_f64().ok_or("shift x must be numeric")?,
                pair[1].as_f64().ok_or("shift y must be numeric")?,
            ];
            if shift.iter().any(|v| !v.is_finite()) {
                return Err("frame shifts must be finite".into());
            }
            Ok(shift)
        })
        .collect()
}

#[derive(Clone)]
struct Sample {
    value: f64,
    background: Vector3<f64>,
    local_index: usize,
}

#[derive(Clone, Debug)]
struct Fit {
    rss: f64,
    bic: f64,
    sample_count: usize,
    masked_pixel_count: usize,
    starts: Vec<[f64; 2]>,
    positions: Vec<[f64; 2]>,
    offsets: Vec<[f64; 2]>,
    at_bound: Vec<bool>,
    amplitudes: Vec<f64>,
    snr: Vec<f64>,
    active_count: usize,
    success: bool,
    nfev: usize,
}

fn residuals(
    parameters: &[f64],
    samples: &[Sample],
    component_count: usize,
    bounds: [usize; 4],
    origin: [usize; 2],
    psf: &EmpiricalPsf,
) -> Result<Vec<f64>> {
    let [x0, y0, x1, y1] = bounds;
    let positions = (0..component_count)
        .map(|i| [parameters[4 + 3 * i], parameters[5 + 3 * i]])
        .collect::<Vec<_>>();
    let source = positions
        .iter()
        .map(|p| shifted_basis(x1 - x0, y1 - y0, origin, *p, psf))
        .collect::<Result<Vec<_>>>()?;
    Ok(samples
        .iter()
        .map(|s| {
            let mut value = s.background[0] * parameters[0]
                + s.background[1] * parameters[1]
                + s.background[2] * parameters[2];
            for i in 0..component_count {
                value += parameters[3 + 3 * i] * source[i][s.local_index];
            }
            value - s.value
        })
        .collect())
}

fn objective(residual: &[f64]) -> f64 {
    residual.iter().map(|v| v * v).sum::<f64>() * 0.5
}

fn jacobian(
    parameters: &[f64],
    current: &[f64],
    lower: &[f64],
    upper: &[f64],
    samples: &[Sample],
    component_count: usize,
    bounds: [usize; 4],
    origin: [usize; 2],
    psf: &EmpiricalPsf,
) -> Result<DMatrix<f64>> {
    let mut result = DMatrix::zeros(current.len(), parameters.len());
    for column in 0..parameters.len() {
        let step = f64::EPSILON.sqrt() * parameters[column].abs().max(1.0);
        let probe = if parameters[column] + step <= upper[column] {
            parameters[column] + step
        } else if parameters[column] - step >= lower[column] {
            parameters[column] - step
        } else {
            continue;
        };
        let mut p = parameters.to_vec();
        p[column] = probe;
        let shifted = residuals(&p, samples, component_count, bounds, origin, psf)?;
        let scale = 1.0 / (probe - parameters[column]);
        for row in 0..current.len() {
            result[(row, column)] = (shifted[row] - current[row]) * scale;
        }
    }
    Ok(result)
}

// Bounded damped Gauss-Newton with central finite differences. The linear
// amplitudes and planar background are initialized by the exact fixed-position
// projection; all parameters then move together within the SciPy contract's
// box constraints.
fn optimize(
    start_positions: &[[f64; 2]],
    centers: &[[f64; 2]],
    radius: f64,
    frame: &FitsFrame,
    mask: &[bool],
    bounds: [usize; 4],
    psf: &EmpiricalPsf,
) -> Result<Option<Fit>> {
    let [x0, y0, x1, y1] = bounds;
    let width = x1 - x0;
    let height = y1 - y0;
    let count = start_positions.len();
    if count == 0
        || centers.len() != count
        || start_positions.iter().zip(centers).any(|(p, c)| {
            (p[0] - c[0]).abs() > radius + 1e-9 || (p[1] - c[1]).abs() > radius + 1e-9
        })
    {
        return Ok(None);
    }
    let center_x = 0.5 * (x0 + x1 - 1) as f64;
    let center_y = 0.5 * (y0 + y1 - 1) as f64;
    let scale = width.max(height).max(1) as f64;
    let mut samples = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let index = y * frame.width + x;
            let value = frame.data.value(index);
            if value.is_finite() && !mask[index] {
                samples.push(Sample {
                    value,
                    background: Vector3::new(
                        1.0,
                        (x as f64 - center_x) / scale,
                        (y as f64 - center_y) / scale,
                    ),
                    local_index: (y - y0) * width + (x - x0),
                });
            }
        }
    }
    let sample_count = samples.len();
    let parameter_count = 3 + 3 * count;
    if sample_count < 24.max(parameter_count + 8) {
        return Ok(None);
    }
    let bases = start_positions
        .iter()
        .map(|p| shifted_basis(width, height, [x0, y0], *p, psf))
        .collect::<Result<Vec<_>>>()?;
    let mut gram = Matrix3::zeros();
    let mut bg_y = Vector3::zeros();
    let mut bg_sources = vec![Vector3::zeros(); count];
    for sample in &samples {
        gram += sample.background * sample.background.transpose();
        bg_y += sample.background * sample.value;
        for i in 0..count {
            bg_sources[i] += sample.background * bases[i][sample.local_index];
        }
    }
    let Some(gram_inv) = gram.try_inverse() else {
        return Ok(None);
    };
    let projected_y = gram_inv * bg_y;
    let projected_sources = bg_sources.iter().map(|v| gram_inv * v).collect::<Vec<_>>();
    let data_residual = samples
        .iter()
        .map(|s| s.value - s.background.dot(&projected_y))
        .collect::<Vec<_>>();
    let source_residual = (0..count)
        .map(|i| {
            samples
                .iter()
                .map(|s| bases[i][s.local_index] - s.background.dot(&projected_sources[i]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let amplitudes = empirical_pair_fit::solve_nnls(&source_residual, &data_residual);
    let mut bg_rhs = bg_y;
    for i in 0..count {
        bg_rhs -= bg_sources[i] * amplitudes[i];
    }
    let bg = gram_inv * bg_rhs;
    let mut parameters = vec![bg[0], bg[1], bg[2]];
    let mut lower = vec![f64::NEG_INFINITY; parameter_count];
    let mut upper = vec![f64::INFINITY; parameter_count];
    for i in 0..count {
        parameters.extend([amplitudes[i], start_positions[i][0], start_positions[i][1]]);
        lower[3 + 3 * i] = 0.0;
        lower[4 + 3 * i] = centers[i][0] - radius;
        lower[5 + 3 * i] = centers[i][1] - radius;
        upper[4 + 3 * i] = centers[i][0] + radius;
        upper[5 + 3 * i] = centers[i][1] + radius;
    }
    let mut nfev = 0usize;
    let mut current = residuals(&parameters, &samples, count, bounds, [x0, y0], psf)?;
    nfev += 1;
    let mut cost = objective(&current);
    let mut lambda = 1e-3;
    let mut success = false;
    for _ in 0..80 {
        if nfev >= 300 {
            break;
        }
        let jac = jacobian(
            &parameters,
            &current,
            &lower,
            &upper,
            &samples,
            count,
            bounds,
            [x0, y0],
            psf,
        )?;
        let jt = jac.transpose();
        let normal = &jt * &jac;
        let gradient = &jt * DVector::from_column_slice(&current);
        let projected_max = (0..parameter_count)
            .map(|i| {
                let at_lo = parameters[i] <= lower[i] + 1e-9;
                let at_hi = parameters[i] >= upper[i] - 1e-9;
                if (at_lo && gradient[i] > 0.0) || (at_hi && gradient[i] < 0.0) {
                    0.0
                } else {
                    gradient[i].abs()
                }
            })
            .fold(0.0, f64::max);
        if projected_max < 1e-7 {
            success = true;
            break;
        }
        let mut accepted = false;
        for _ in 0..12 {
            let mut damped = normal.clone();
            for i in 0..parameter_count {
                damped[(i, i)] += lambda * normal[(i, i)].abs().max(1.0);
            }
            let Some(inv) = damped.try_inverse() else {
                lambda *= 10.0;
                continue;
            };
            let delta = -(inv * gradient.clone());
            let mut trial = parameters.clone();
            for i in 0..parameter_count {
                trial[i] = (trial[i] + delta[i]).max(lower[i]).min(upper[i]);
            }
            if trial
                .iter()
                .zip(&parameters)
                .all(|(a, b)| (a - b).abs() < 1e-12)
            {
                lambda *= 10.0;
                continue;
            }
            let trial_res = residuals(&trial, &samples, count, bounds, [x0, y0], psf)?;
            nfev += 1;
            let trial_cost = objective(&trial_res);
            if trial_cost < cost {
                let reduction = cost - trial_cost;
                let step = trial
                    .iter()
                    .zip(&parameters)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm = parameters.iter().map(|v| v * v).sum::<f64>().sqrt();
                parameters = trial;
                current = trial_res;
                cost = trial_cost;
                lambda = (lambda * 0.3).max(1e-12);
                accepted = true;
                if step <= 1e-9 * (norm + 1e-9) || reduction <= 1e-9 * cost.max(1.0) {
                    success = true;
                }
                break;
            } else {
                lambda *= 10.0;
            }
        }
        if success {
            break;
        }
        if !accepted && lambda > 1e14 {
            // No feasible trial lowers the objective beyond floating-point
            // resolution; treat this stationary bounded solution as converged.
            success = true;
            break;
        }
    }
    let rss = (2.0 * cost).max(f64::EPSILON);
    let bic = sample_count as f64 * (rss / sample_count as f64).ln()
        + parameter_count as f64 * (sample_count as f64).ln();
    let positions = (0..count)
        .map(|i| [parameters[4 + 3 * i], parameters[5 + 3 * i]])
        .collect::<Vec<_>>();
    let offsets = positions
        .iter()
        .zip(centers)
        .map(|(p, c)| [p[0] - c[0], p[1] - c[1]])
        .collect::<Vec<_>>();
    let tolerance = 1e-5f64.max(radius * 1e-5);
    let at_bound = offsets
        .iter()
        .map(|o| o[0].abs() >= radius - tolerance || o[1].abs() >= radius - tolerance)
        .collect::<Vec<_>>();
    let amps = (0..count)
        .map(|i| parameters[3 + 3 * i])
        .collect::<Vec<_>>();
    let max_amp = amps.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let active = amps
        .iter()
        .filter(|a| **a > 1e-12f64.max(f64::EPSILON * max_amp.max(1.0)))
        .count();
    let jac = jacobian(
        &parameters,
        &current,
        &lower,
        &upper,
        &samples,
        count,
        bounds,
        [x0, y0],
        psf,
    )?;
    let normal = jac.transpose() * jac;
    let dof = sample_count.saturating_sub(parameter_count).max(1) as f64;
    let variance = rss / dof;
    let covariance = normal
        .svd(true, true)
        .pseudo_inverse(1e-15)
        .ok()
        .map(|m| m * variance);
    let snr = (0..count)
        .map(|i| {
            let idx = 3 + 3 * i;
            let error = covariance
                .as_ref()
                .map_or(f64::EPSILON, |c| c[(idx, idx)].max(f64::EPSILON).sqrt());
            if amps[i] > 1e-12 {
                amps[i] / error
            } else {
                0.0
            }
        })
        .collect();
    Ok(Some(Fit {
        rss,
        bic,
        sample_count,
        masked_pixel_count: width * height - sample_count,
        starts: start_positions.to_vec(),
        positions,
        offsets,
        at_bound,
        amplitudes: amps,
        snr,
        active_count: active,
        success,
        nfev,
    }))
}

fn single_starts(midpoint: [f64; 2], pair: [[f64; 2]; 2], radius: f64) -> Vec<[f64; 2]> {
    let offsets = [
        [0.0, 0.0],
        [radius * 0.5, 0.0],
        [-radius * 0.5, 0.0],
        [0.0, radius * 0.5],
        [0.0, -radius * 0.5],
        [radius * 0.5, radius * 0.5],
        [radius * 0.5, -radius * 0.5],
        [-radius * 0.5, radius * 0.5],
        [-radius * 0.5, -radius * 0.5],
    ];
    let mut points = Vec::new();
    for [dx, dy] in offsets {
        let p = [midpoint[0] + dx, midpoint[1] + dy];
        if (dx.abs() <= radius + 1e-9) && (dy.abs() <= radius + 1e-9) {
            points.push(p);
        }
    }
    for p in pair {
        if (p[0] - midpoint[0]).abs() <= radius + 1e-9
            && (p[1] - midpoint[1]).abs() <= radius + 1e-9
        {
            points.push(p);
        }
    }
    let mut unique = Vec::new();
    for p in points {
        if !unique
            .iter()
            .any(|q: &[f64; 2]| (p[0] - q[0]).hypot(p[1] - q[1]) < 1e-8)
        {
            unique.push(p);
        }
    }
    unique
}

fn best_fit(
    starts: &[Vec<[f64; 2]>],
    centers: &[[f64; 2]],
    radius: f64,
    frame: &FitsFrame,
    mask: &[bool],
    bounds: [usize; 4],
    psf: &EmpiricalPsf,
) -> Result<Option<Fit>> {
    let mut best: Option<Fit> = None;
    for start in starts {
        let candidate = optimize(start, centers, radius, frame, mask, bounds, psf)?;
        if candidate
            .as_ref()
            .is_some_and(|f| best.as_ref().is_none_or(|b| f.rss < b.rss))
        {
            best = candidate;
        }
    }
    Ok(best)
}

#[allow(clippy::too_many_arguments)]
fn run_audit(
    paths: &[PathBuf],
    template: &FitsFrame,
    sources: &[Detection],
    primary: &Detection,
    secondary: &Detection,
    pxy: [f64; 2],
    sxy: [f64; 2],
    catalog: &Path,
    shifts: Option<&[[f64; 2]]>,
    modes: &[String],
    codes: &[i64],
    radius: usize,
    max_sources: usize,
    padding: Option<f64>,
    single_radius: f64,
    double_radius: f64,
    min_separation: f64,
    confirmation_bic: f64,
    confirmation_snr: f64,
) -> Result<EmpiricalFreePairResult> {
    if paths.is_empty() {
        return Err("paths cannot be empty".into());
    }
    if (pxy[0] - sxy[0]).hypot(pxy[1] - sxy[1]) <= 1e-6 {
        return Err("primary and secondary target coordinates must differ".into());
    }
    if modes.is_empty()
        || modes.iter().collect::<std::collections::HashSet<_>>().len() != modes.len()
        || modes.iter().any(|m| !MASK_MODES.contains(&m.as_str()))
    {
        return Err("mask_modes must be supported, non-empty, and unique".into());
    }
    if modes.iter().any(|m| m.contains("repeated_code")) && codes.is_empty() {
        return Err("repeated-code mask mode requires repeated_code_values".into());
    }
    if radius < 3 || max_sources == 0 {
        return Err("support_radius must be at least 3 and source count must be positive".into());
    }
    for (name, value) in [
        ("single_position_radius_px", single_radius),
        ("double_position_radius_px", double_radius),
        ("minimum_fitted_separation_px", min_separation),
        ("confirmation_delta_bic", confirmation_bic),
        ("confirmation_component_snr", confirmation_snr),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{name} must be finite and positive"));
        }
    }
    if sources.is_empty() {
        return Err("template_sources cannot be empty".into());
    }
    let psf = empirical_pair_fit::estimate_psf(template, sources, radius, max_sources)?
        .ok_or("not enough eligible isolated sources to build an empirical PSF")?;
    let pad = padding.unwrap_or((psf.support_radius as f64 + 3.0).max(10.0));
    if !pad.is_finite() || pad < psf.support_radius as f64 + 1.0 {
        return Err("patch_padding_px must be at least support_radius + 1".into());
    }
    let shifts = if let Some(v) = shifts {
        if v.len() != paths.len() || v.iter().flatten().any(|x| !x.is_finite()) {
            return Err("frame_shifts must match paths and be finite".into());
        }
        v.to_vec()
    } else {
        vec![[0.0, 0.0]; paths.len()]
    };
    let midpoint = [(pxy[0] + sxy[0]) * 0.5, (pxy[1] + sxy[1]) * 0.5];
    let pairs = [[pxy[0], pxy[1]], [sxy[0], sxy[1]]];
    let mut rows = Vec::new();
    for (fi, path) in paths.iter().enumerate() {
        let frame = fits::read_fits(path, false)?;
        let shift = shifts[fi];
        let pair = [
            [pairs[0][0] + shift[0], pairs[0][1] + shift[1]],
            [pairs[1][0] + shift[0], pairs[1][1] + shift[1]],
        ];
        let mid = [midpoint[0] + shift[0], midpoint[1] + shift[1]];
        let bounds = patch_bounds(frame.width, frame.height, &pair, pad);
        let [x0, y0, x1, y1] = bounds;
        if x1 - x0 < 2 * psf.support_radius + 3 || y1 - y0 < 2 * psf.support_radius + 3 {
            return Err(format!("pair patch is too small in frame {}", fi + 1));
        }
        let singles = single_starts(mid, pair, single_radius)
            .into_iter()
            .map(|p| vec![p])
            .collect::<Vec<_>>();
        let single_centers = [mid];
        let double_starts = vec![
            pair.to_vec(),
            vec![mid, mid],
            vec![pair[0], mid],
            vec![mid, pair[1]],
        ];
        for (mode, mask) in mask_modes(&frame, modes, codes)? {
            let single = best_fit(
                &singles,
                &single_centers,
                single_radius,
                &frame,
                &mask,
                bounds,
                &psf,
            )?;
            let single_pos = single
                .as_ref()
                .and_then(|f| f.positions.first().copied())
                .unwrap_or(mid);
            let mut starts = double_starts.clone();
            starts.push(vec![single_pos, mid]);
            starts.push(vec![single_pos, single_pos]);
            let double = best_fit(&starts, &pair, double_radius, &frame, &mask, bounds, &psf)?;
            let delta = single
                .as_ref()
                .zip(double.as_ref())
                .map(|(a, b)| a.bic - b.bic);
            let fitted_sep = double.as_ref().and_then(|f| {
                (f.positions.len() == 2).then(|| {
                    (f.positions[0][0] - f.positions[1][0])
                        .hypot(f.positions[0][1] - f.positions[1][1])
                })
            });
            let second_snr = double.as_ref().and_then(|f| f.snr.get(1).copied());
            let confirmation = delta.is_some_and(|v| v >= confirmation_bic)
                && second_snr.is_some_and(|v| v >= confirmation_snr)
                && double.as_ref().is_some_and(|f| {
                    f.active_count >= 2 && f.success && !f.at_bound.iter().any(|v| *v)
                })
                && fitted_sep.is_some_and(|v| v >= min_separation);
            let note = if single.is_none() || double.is_none() {
                "自由位置拟合无完整 K=1/K=2 结果"
            } else if confirmation {
                "自由位置模型达到当前工程证据线；仍需空间 PSF、注入和星表核验"
            } else if double
                .as_ref()
                .is_some_and(|f| f.at_bound.iter().any(|v| *v))
            {
                "双源位置触碰搜索边界；局部模型比较，不等于物理恒星身份"
            } else if double.as_ref().is_some_and(|f| !f.success) {
                "双源优化器未报告收敛；局部模型比较，不等于物理恒星身份"
            } else {
                "自由位置模型未达到当前双源证据线；不等于排除真实近邻"
            };
            rows.push(EmpiricalFreePairRow {
                frame_index: fi + 1,
                path: path.to_string_lossy().into_owned(),
                mask_mode: mode,
                patch_x0: x0,
                patch_y0: y0,
                patch_width: x1 - x0,
                patch_height: y1 - y0,
                sample_count: double.as_ref().map_or_else(
                    || single.as_ref().map_or(0, |f| f.sample_count),
                    |f| f.sample_count,
                ),
                masked_pixel_count: double.as_ref().map_or_else(
                    || single.as_ref().map_or(0, |f| f.masked_pixel_count),
                    |f| f.masked_pixel_count,
                ),
                single_rss: single.as_ref().map(|f| f.rss),
                double_rss: double.as_ref().map(|f| f.rss),
                single_bic: single.as_ref().map(|f| f.bic),
                double_bic: double.as_ref().map(|f| f.bic),
                delta_bic_single_minus_double: delta,
                single_initial_position: single.as_ref().and_then(|f| f.starts.first().copied()),
                single_fitted_position: single.as_ref().and_then(|f| f.positions.first().copied()),
                single_position_offset: single.as_ref().and_then(|f| f.offsets.first().copied()),
                single_position_at_bound: single
                    .as_ref()
                    .and_then(|f| f.at_bound.first().copied())
                    .unwrap_or(false),
                single_component_snr: single.as_ref().and_then(|f| f.snr.first().copied()),
                single_optimizer_success: single.as_ref().is_some_and(|f| f.success),
                single_optimizer_nfev: single.as_ref().map(|f| f.nfev),
                double_initial_positions: double
                    .as_ref()
                    .map_or_else(Vec::new, |f| f.starts.clone()),
                double_fitted_positions: double
                    .as_ref()
                    .map_or_else(Vec::new, |f| f.positions.clone()),
                double_position_offsets: double
                    .as_ref()
                    .map_or_else(Vec::new, |f| f.offsets.clone()),
                double_positions_at_bound: double
                    .as_ref()
                    .map_or_else(Vec::new, |f| f.at_bound.clone()),
                double_amplitudes: double
                    .as_ref()
                    .map_or_else(Vec::new, |f| f.amplitudes.clone()),
                double_component_snr: double.as_ref().map_or_else(Vec::new, |f| f.snr.clone()),
                double_second_component_snr: second_snr,
                double_active_component_count: double.as_ref().map_or(0, |f| f.active_count),
                double_optimizer_success: double.as_ref().is_some_and(|f| f.success),
                double_optimizer_nfev: double.as_ref().map(|f| f.nfev),
                fitted_separation_px: fitted_sep,
                confirmation_line: confirmation,
                note: note.into(),
            });
        }
    }
    let raw = rows
        .iter()
        .filter(|r| r.mask_mode == "raw")
        .collect::<Vec<_>>();
    let confirmed = raw.iter().filter(|r| r.confirmation_line).count();
    let bounded = raw
        .iter()
        .filter(|r| r.double_positions_at_bound.iter().any(|v| *v))
        .count();
    let conclusion=format!("自由位置经验 PSF 审计完成：{} 帧；raw K=2 达到当前证据线 {}/{} 帧，双源位置触碰边界 {}/{} 帧。该模型仍是有边界的局部研究模型；未通过不能排除真实近邻，达到证据线也不能替代空间 PSF、注入真值或星表/WCS 身份。",paths.len(),confirmed,raw.len(),bounded,raw.len());
    let parameters = json!({"model":"shared planar background + free-position empirical PSF + nonnegative amplitudes","mask_modes":modes,"repeated_code_values":codes,"support_radius":radius,"max_template_sources":max_sources,"patch_padding_px":pad,"single_position_radius_px":single_radius,"double_position_radius_px":double_radius,"confirmation_delta_bic":confirmation_bic,"confirmation_component_snr":confirmation_snr,"minimum_fitted_separation_px":min_separation,"frame_shifts":shifts,"interpretation_boundary":"有边界自由位置 BIC 是局部模型证据；不是星表身份、恒星概率或物理真值"});
    Ok(EmpiricalFreePairResult {
        frame_count: paths.len(),
        primary_detection_id: Some(primary.detection_id),
        secondary_detection_id: Some(secondary.detection_id),
        primary_target_xy: pxy,
        secondary_target_xy: sxy,
        frame_shifts: shifts,
        template_frame_path: template.path.to_string_lossy().into_owned(),
        template_catalog_path: Some(catalog.to_string_lossy().into_owned()),
        mask_modes: modes.to_vec(),
        support_radius: radius,
        patch_padding_px: pad,
        single_position_radius_px: single_radius,
        double_position_radius_px: double_radius,
        confirmation_delta_bic: confirmation_bic,
        confirmation_component_snr: confirmation_snr,
        minimum_fitted_separation_px: min_separation,
        empirical_psf: psf,
        parameters,
        rows,
        conclusion,
    })
}

fn write_csv(path: &Path, rows: &[EmpiricalFreePairRow]) -> Result<()> {
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|e| e.to_string())?;
    let mut w = csv::WriterBuilder::new()
        .has_headers(false)
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    w.write_record(CSV_FIELDS).map_err(|e| e.to_string())?;
    for row in rows {
        let v = serde_json::to_value(row).map_err(|e| e.to_string())?;
        let obj = v.as_object().ok_or("row serialization failed")?;
        w.write_record(
            CSV_FIELDS
                .iter()
                .map(|key| obj.get(*key).map(csv_value).unwrap_or_default()),
        )
        .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}
fn csv_value(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::String(value) => value.clone(),
        Value::Array(_) => serde_json::to_string(v).unwrap_or_default(),
        _ => v.to_string(),
    }
}

fn draw_line(px: &mut [u8], w: usize, h: usize, a: [usize; 2], b: [usize; 2], c: [u8; 3]) {
    let (mut x, mut y) = (a[0] as isize, a[1] as isize);
    let dx = (b[0] as isize - x).abs();
    let sx = if x < b[0] as isize { 1 } else { -1 };
    let dy = -((b[1] as isize - y).abs());
    let sy = if y < b[1] as isize { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            let i = (y as usize * w + x as usize) * 3;
            px[i..i + 3].copy_from_slice(&c);
        }
        if x == b[0] as isize && y == b[1] as isize {
            break;
        }
        let e = 2 * err;
        if e >= dy {
            err += dy;
            x += sx;
        }
        if e <= dx {
            err += dx;
            y += sy;
        }
    }
}
fn write_chart(path: &Path, rows: &[EmpiricalFreePairRow]) -> Result<()> {
    let (w, h) = (1200usize, 680usize);
    let raw = rows
        .iter()
        .filter(|r| r.mask_mode == "raw")
        .collect::<Vec<_>>();
    let vals = raw
        .iter()
        .filter_map(|r| r.delta_bic_single_minus_double)
        .collect::<Vec<_>>();
    let (mut lo, mut hi) = if vals.is_empty() {
        (0.0, 1.0)
    } else {
        (
            vals.iter().copied().fold(f64::INFINITY, f64::min),
            vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    if hi <= lo {
        hi = lo + 1.0;
    }
    let pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;
    let plot = [70usize, 40usize, w - 35, h - 60];
    let mut pixels = vec![255u8; w * h * 3];
    for k in 0..=5 {
        let y = plot[1] + (plot[3] - plot[1]) * k / 5;
        draw_line(
            &mut pixels,
            w,
            h,
            [plot[0], y],
            [plot[2], y],
            [224, 230, 236],
        );
    }
    draw_line(
        &mut pixels,
        w,
        h,
        [plot[0], plot[1]],
        [plot[0], plot[3]],
        [70, 83, 96],
    );
    draw_line(
        &mut pixels,
        w,
        h,
        [plot[0], plot[3]],
        [plot[2], plot[3]],
        [70, 83, 96],
    );
    let points = vals
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let x = if vals.len() <= 1 {
                (plot[0] + plot[2]) / 2
            } else {
                plot[0] + (plot[2] - plot[0]) * i / (vals.len() - 1)
            };
            let y = plot[3]
                - (((v - lo) / (hi - lo)).clamp(0.0, 1.0) * (plot[3] - plot[1]) as f64).round()
                    as usize;
            [x, y]
        })
        .collect::<Vec<_>>();
    for pair in points.windows(2) {
        draw_line(&mut pixels, w, h, pair[0], pair[1], [79, 155, 131]);
    }
    let mut e = png::Encoder::new(
        fs::File::create(path).map_err(|e| e.to_string())?,
        w as u32,
        h as u32,
    );
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    e.write_header()
        .map_err(|e| e.to_string())?
        .write_image_data(&pixels)
        .map_err(|e| e.to_string())
}

pub fn write_artifacts(result: &EmpiricalFreePairResult, out: &Path) -> Result<Value> {
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    write_csv(&out.join("empirical_pair_free.csv"), &result.rows)?;
    fs::write(
        out.join("empirical_pair_free.json"),
        serde_json::to_vec_pretty(&result.as_json()).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if result.rows.iter().any(|r| r.mask_mode == "raw") {
        write_chart(&out.join("empirical_pair_free_bic.png"), &result.rows)?;
    }
    Ok(json!({"output":out,"frame_count":result.frame_count,"conclusion":result.conclusion}))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let opt = parse_args(args)?;
    let paths = ordered_fits_paths(&opt.data_dir)?;
    let sources = feature_psf_leaveout::load_detection_catalog(&opt.source_catalog)?;
    let primary = sources
        .iter()
        .find(|s| s.detection_id == opt.primary_id)
        .ok_or_else(|| format!("source catalog has no detection_id={}", opt.primary_id))?;
    let secondary = sources
        .iter()
        .find(|s| s.detection_id == opt.secondary_id)
        .ok_or_else(|| format!("source catalog has no detection_id={}", opt.secondary_id))?;
    let shifts = opt.shifts_json.as_deref().map(read_shifts).transpose()?;
    let codes = resolve_repeated_codes(&sources, opt.repeated_code_values.as_deref());
    if opt.mask_modes.iter().any(|m| m.contains("repeated_code"))
        && opt.repeated_code_values.is_none()
        && codes.is_empty()
    {
        return Err("repeated-code mask mode requires repeated_code_values".into());
    }
    let template = fits::read_fits(&paths[0], false)?;
    let target = |s: &Detection| [s.x, s.y];
    let result = run_audit(
        &paths,
        &template,
        &sources,
        primary,
        secondary,
        target(primary),
        target(secondary),
        &opt.source_catalog,
        shifts.as_deref(),
        &opt.mask_modes,
        &codes,
        opt.support_radius,
        opt.max_template_sources,
        opt.patch_padding_px,
        opt.single_position_radius_px,
        opt.double_position_radius_px,
        opt.minimum_fitted_separation_px,
        10.0,
        5.0,
    )?;
    write_artifacts(&result, &opt.out_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs::File,
        io::BufReader,
        time::{SystemTime, UNIX_EPOCH},
    };

    const REFERENCE_JSON: &str =
        include_str!("../tests/fixtures/empirical_pair_free_parity/reference-output.json");
    const REFERENCE_CSV: &[u8] =
        include_bytes!("../tests/fixtures/empirical_pair_free_parity/reference.csv");

    fn tolerance(field: &str, expected: f64) -> f64 {
        if field.contains("component_snr") {
            0.5f64.max(expected.abs() * 0.05)
        } else if field.contains("rss") {
            2.0
        } else if field.contains("bic") {
            0.6
        } else if field.contains("amplitude") {
            0.5
        } else if field.contains("position") || field.contains("separation") {
            0.005
        } else if field == "kernel_sum" {
            1e-6
        } else {
            1e-8f64.max(expected.abs() * 1e-10)
        }
    }

    fn compare_json(actual: &Value, expected: &Value, field: &str) {
        match (actual, expected) {
            (Value::Object(a), Value::Object(e)) => {
                let mut ak = a.keys().collect::<Vec<_>>();
                let mut ek = e.keys().collect::<Vec<_>>();
                ak.sort();
                ek.sort();
                assert_eq!(ak, ek, "{field} keys");
                for key in e.keys() {
                    if matches!(
                        key.as_str(),
                        "path" | "template_frame_path" | "template_catalog_path"
                    ) || key.ends_with("optimizer_nfev")
                    {
                        continue;
                    }
                    compare_json(&a[key], &e[key], key);
                }
            }
            (Value::Array(a), Value::Array(e)) => {
                assert_eq!(a.len(), e.len(), "{field} length");
                for (a, e) in a.iter().zip(e) {
                    compare_json(a, e, field);
                }
            }
            (Value::Number(a), Value::Number(e)) => {
                if let (Some(a), Some(e)) = (a.as_i64(), e.as_i64()) {
                    assert_eq!(a, e, "{field}");
                } else {
                    let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
                    let tol = tolerance(field, e);
                    assert!((a - e).abs() <= tol, "{field}: {a} != {e} ± {tol}");
                }
            }
            _ => assert_eq!(actual, expected, "{field}"),
        }
    }

    #[test]
    fn parser_defaults_match_python_cli() {
        let args = vec![
            "frames".into(),
            "--source-catalog".into(),
            "catalog.csv".into(),
            "--primary-id".into(),
            "1".into(),
            "--secondary-id".into(),
            "2".into(),
        ];
        let o = parse_args(&args).unwrap();
        assert_eq!(o.single_position_radius_px, 2.0);
        assert_eq!(o.double_position_radius_px, 1.25);
        assert_eq!(o.mask_modes, ["raw", "range_masked", "sentinel_masked"]);
        assert!(o.repeated_code_values.is_none());
    }

    #[test]
    fn frozen_python_free_fit_matches_rust_outputs_and_artifacts() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let frames = manifest.join("tests/fixtures/empirical_pair_audit_parity/frames");
        let catalog = manifest.join("tests/fixtures/source_audit_parity/source_catalog.csv");
        let output = std::env::temp_dir().join(format!(
            "rst19-empirical-pair-free-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&output).unwrap();
        let args = [
            frames.to_string_lossy().into_owned(),
            "--source-catalog".into(),
            catalog.to_string_lossy().into_owned(),
            "--primary-id".into(),
            "0".into(),
            "--secondary-id".into(),
            "1".into(),
            "--out-dir".into(),
            output.to_string_lossy().into_owned(),
        ];
        run_desktop(&args).unwrap();
        let actual_json: Value =
            serde_json::from_slice(&fs::read(output.join("empirical_pair_free.json")).unwrap())
                .unwrap();
        let expected_json: Value = serde_json::from_str(REFERENCE_JSON).unwrap();
        compare_json(&actual_json, &expected_json, "result");

        let actual_csv = fs::read(output.join("empirical_pair_free.csv")).unwrap();
        let actual_csv = actual_csv
            .strip_prefix(&[0xef, 0xbb, 0xbf])
            .unwrap_or(&actual_csv);
        let expected_csv = REFERENCE_CSV
            .strip_prefix(&[0xef, 0xbb, 0xbf])
            .unwrap_or(REFERENCE_CSV);
        let mut actual_reader = csv::Reader::from_reader(actual_csv);
        let mut expected_reader = csv::Reader::from_reader(expected_csv);
        let actual_headers = actual_reader.headers().unwrap().clone();
        let expected_headers = expected_reader.headers().unwrap().clone();
        assert_eq!(actual_headers, expected_headers, "CSV headers");
        let actual_rows = actual_reader
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let expected_rows = expected_reader
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(actual_rows.len(), expected_rows.len());
        for (row, (actual, expected)) in actual_rows.iter().zip(&expected_rows).enumerate() {
            for column in 0..actual.len() {
                let field = &actual_headers[column];
                let (a, e) = (&actual[column], &expected[column]);
                if field == "path" || field.ends_with("optimizer_nfev") || a == e {
                    continue;
                }
                if let (Ok(a), Ok(e)) = (
                    serde_json::from_str::<Value>(a),
                    serde_json::from_str::<Value>(e),
                ) {
                    compare_json(&a, &e, field);
                } else if let (Ok(a), Ok(e)) = (a.parse::<f64>(), e.parse::<f64>()) {
                    let tol = tolerance(field, e);
                    assert!(
                        (a - e).abs() <= tol,
                        "CSV row {row} {field}: {a} != {e} ± {tol}"
                    );
                } else {
                    assert_eq!(a, e, "CSV row {row} {field}");
                }
            }
        }

        let png = File::open(output.join("empirical_pair_free_bic.png")).unwrap();
        let reader = png::Decoder::new(BufReader::new(png)).read_info().unwrap();
        assert_eq!((reader.info().width, reader.info().height), (1200, 680));
        fs::remove_dir_all(output).unwrap();
    }

    #[test]
    fn bounded_free_fit_recovers_a_well_separated_injected_pair() {
        fn inject(image: &mut [f32], width: usize, height: usize, x: f64, y: f64, peak: f64) {
            let sigma = 3.0 / 2.354_820_045_f64;
            let radius = (4.0 * sigma).ceil() as isize;
            let x0 = (x.floor() as isize - radius).max(0);
            let x1 = (x.floor() as isize + radius + 1).min(width as isize);
            let y0 = (y.floor() as isize - radius).max(0);
            let y1 = (y.floor() as isize + radius + 1).min(height as isize);
            for py in y0..y1 {
                for px in x0..x1 {
                    let dx = px as f64 - x;
                    let dy = py as f64 - y;
                    let add = peak * (-0.5 * (dx * dx + dy * dy) / sigma.powi(2)).exp();
                    let index = py as usize * width + px as usize;
                    image[index] = (f64::from(image[index]) + add) as f32;
                }
            }
        }

        let (width, height) = (96usize, 96usize);
        let mut image = vec![20.0f32; width * height];
        inject(&mut image, width, height, 42.0, 48.0, 1200.0);
        inject(&mut image, width, height, 48.0, 48.0, 900.0);
        let kernel = (0..7)
            .flat_map(|ky| {
                (0..7).map(move |kx| {
                    let x = kx as f64 - 3.0;
                    let y = ky as f64 - 3.0;
                    (-0.5 * (x * x + y * y) / 1.3f64.powi(2)).exp() as f32
                })
            })
            .collect::<Vec<_>>();
        let max_kernel = kernel.iter().copied().fold(0.0f32, f32::max);
        let psf = EmpiricalPsf {
            kernel: kernel.into_iter().map(|v| v / max_kernel).collect(),
            support_radius: 3,
            source_count: 4,
            median_fwhm_px: Some(3.0),
            median_ellipticity: Some(0.1),
        };
        let frame = FitsFrame {
            path: PathBuf::from("injected-pair.fits"),
            header: Default::default(),
            data: crate::models::PixelData::F32(image),
            auxiliary: None,
            data_offset: 0,
            width,
            height,
        };
        let pair = [[42.0, 48.0], [48.0, 48.0]];
        let midpoint = [45.0, 48.0];
        let bounds = patch_bounds(width, height, &pair, 10.0);
        let mask = vec![false; width * height];
        let singles = single_starts(midpoint, pair, 3.0)
            .into_iter()
            .map(|position| vec![position])
            .collect::<Vec<_>>();
        let single = best_fit(&singles, &[midpoint], 3.0, &frame, &mask, bounds, &psf)
            .unwrap()
            .unwrap();
        let single_best = single.positions[0];
        let double_starts = vec![
            pair.to_vec(),
            vec![midpoint, midpoint],
            vec![pair[0], midpoint],
            vec![midpoint, pair[1]],
            vec![single_best, midpoint],
            vec![single_best, single_best],
        ];
        let double = best_fit(&double_starts, &pair, 3.0, &frame, &mask, bounds, &psf)
            .unwrap()
            .unwrap();
        let separation = (double.positions[0][0] - double.positions[1][0])
            .hypot(double.positions[0][1] - double.positions[1][1]);
        assert!(double.success, "{double:?}");
        assert!(double.active_count >= 2);
        assert!(double.snr[1] >= 5.0);
        assert!(double.bic < single.bic - 10.0);
        assert!(separation >= 2.0);
        assert!(!double.at_bound.iter().any(|value| *value));
    }
}
