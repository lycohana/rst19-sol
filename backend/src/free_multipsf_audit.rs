//! Bounded free-position Gaussian K=1..N PSF model comparison.
//!
//! Positions move only inside a per-coordinate box around their detector
//! candidate. This is a local sensitivity audit, not a physical identity test.

use crate::{empirical_pair_fit, fits, models::FitsFrame, multipsf_audit, Result};
use nalgebra::{DMatrix, DVector};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const MASK_MODES: [&str; 4] = [
    "raw",
    "range_masked",
    "sentinel_masked",
    "range_sentinel_masked",
];
const AUDIT_FIELDS: &[&str] = &[
    "frame_index",
    "path",
    "mask_mode",
    "component_count",
    "initial_positions",
    "fitted_positions",
    "position_offsets",
    "positions_at_bound",
    "sample_count",
    "masked_pixel_count",
    "rss",
    "bic",
    "delta_bic_from_single",
    "delta_bic_from_previous",
    "amplitudes",
    "component_snr",
    "active_component_count",
    "max_template_correlation",
    "design_condition_number",
    "optimizer_success",
    "optimizer_status",
    "optimizer_nfev",
    "confirmation_line",
    "note",
];
const SWEEP_FIELDS: &[&str] = &[
    "position_radius_px",
    "frame_index",
    "path",
    "mask_mode",
    "component_count",
    "delta_bic_from_single",
    "component_snr",
    "fitted_positions",
    "positions_at_bound",
    "optimizer_success",
    "confirmation_line",
    "note",
];

pub const HELP: &str = "比较同一局部的有限自由位置 K=1..N Gaussian PSF 模型\n\nUsage: rst19-free-multipsf-audit DATA_DIR [OPTIONS]\n\nOptions: --positions \"x,y;x,y;...\" --psf-fwhm N --position-radius N\n         --radius-sweep --position-radii N [N ...]\n         --mask-modes MODE [MODE ...] --shifts-json FILE --out-dir DIR";

#[derive(Clone, Debug, Serialize)]
pub struct FreeMultiPSFAuditRow {
    pub frame_index: usize,
    pub path: String,
    pub mask_mode: String,
    pub component_count: usize,
    pub initial_positions: Vec<[f64; 2]>,
    pub fitted_positions: Vec<[f64; 2]>,
    pub position_offsets: Vec<[f64; 2]>,
    pub positions_at_bound: Vec<bool>,
    pub sample_count: usize,
    pub masked_pixel_count: usize,
    pub rss: Option<f64>,
    pub bic: Option<f64>,
    pub delta_bic_from_single: Option<f64>,
    pub delta_bic_from_previous: Option<f64>,
    pub amplitudes: Vec<f64>,
    pub component_snr: Vec<f64>,
    pub active_component_count: usize,
    pub max_template_correlation: Option<f64>,
    pub design_condition_number: Option<f64>,
    pub optimizer_success: bool,
    pub optimizer_status: Option<i32>,
    pub optimizer_nfev: Option<usize>,
    pub confirmation_line: bool,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct FreeMultiPSFAuditResult {
    pub frame_count: usize,
    pub target_positions: Vec<[f64; 2]>,
    pub frame_shifts: Vec<[f64; 2]>,
    pub mask_modes: Vec<String>,
    pub psf_fwhm_px: f64,
    pub position_radius_px: f64,
    pub parameters: Value,
    pub rows: Vec<FreeMultiPSFAuditRow>,
    pub conclusion: String,
}

impl FreeMultiPSFAuditResult {
    fn as_json(&self) -> Value {
        json!({
            "frame_count": self.frame_count,
            "target_positions": self.target_positions,
            "frame_shifts": self.frame_shifts,
            "mask_modes": self.mask_modes,
            "psf_fwhm_px": self.psf_fwhm_px,
            "position_radius_px": self.position_radius_px,
            "parameters": self.parameters,
            "rows": self.rows,
            "conclusion": self.conclusion,
            "note": "中心只允许在初始 detector 候选周围的有限半径内优化；K=1..N 共享局部背景和像素窗口，源幅度用非负约束。结果是局部模型敏感性证据，不是物理恒星身份。"
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FreeMultiPSFSweepRow {
    pub position_radius_px: f64,
    pub frame_index: usize,
    pub path: String,
    pub mask_mode: String,
    pub component_count: usize,
    pub delta_bic_from_single: Option<f64>,
    pub component_snr: Vec<f64>,
    pub fitted_positions: Vec<[f64; 2]>,
    pub positions_at_bound: Vec<bool>,
    pub optimizer_success: bool,
    pub confirmation_line: bool,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct FreeMultiPSFSweepResult {
    pub frame_count: usize,
    pub target_positions: Vec<[f64; 2]>,
    pub position_radii_px: Vec<f64>,
    pub mask_modes: Vec<String>,
    pub psf_fwhm_px: f64,
    pub parameters: Value,
    pub rows: Vec<FreeMultiPSFSweepRow>,
    pub conclusion: String,
}

impl FreeMultiPSFSweepResult {
    fn as_json(&self) -> Value {
        json!({
            "frame_count": self.frame_count,
            "target_positions": self.target_positions,
            "position_radii_px": self.position_radii_px,
            "mask_modes": self.mask_modes,
            "psf_fwhm_px": self.psf_fwhm_px,
            "parameters": self.parameters,
            "rows": self.rows,
            "conclusion": self.conclusion,
            "note": "位置搜索半径敏感性只用于检查局部模型是否稳定；位置触边、优化未收敛和跨半径不稳定都不能作为物理恒星身份。"
        })
    }
}

#[derive(Clone, Debug)]
struct FreeOptions {
    data_dir: PathBuf,
    positions: Vec<[f64; 2]>,
    psf_fwhm: f64,
    position_radius: f64,
    radius_sweep: bool,
    position_radii: Vec<f64>,
    mask_modes: Vec<String>,
    shifts_json: Option<PathBuf>,
    out_dir: PathBuf,
}

fn parse_positions(value: &str) -> Result<Vec<[f64; 2]>> {
    let mut positions = Vec::new();
    for item in value.split(';') {
        let pair = item.split(',').map(str::trim).collect::<Vec<_>>();
        if pair.len() != 2 {
            return Err(format!("position must use x,y;x,y format: {item}"));
        }
        positions.push([
            pair[0]
                .parse::<f64>()
                .map_err(|_| format!("invalid x coordinate in position: {item}"))?,
            pair[1]
                .parse::<f64>()
                .map_err(|_| format!("invalid y coordinate in position: {item}"))?,
        ]);
    }
    if positions.len() < 2 {
        return Err("at least two positions are required".into());
    }
    Ok(positions)
}

fn parse_args(args: &[String]) -> Result<FreeOptions> {
    let mut data_dir = None;
    let mut positions = "1269,3465;1274,3467;1272,3453".to_owned();
    let mut psf_fwhm = 2.0f64;
    let mut position_radius = 1.25f64;
    let mut radius_sweep = false;
    let mut position_radii: Vec<f64> = vec![0.25, 0.5, 0.75, 1.0, 1.25];
    let mut mask_modes = vec![
        "raw".into(),
        "range_masked".into(),
        "sentinel_masked".into(),
    ];
    let mut shifts_json = None;
    let mut out_dir = PathBuf::from("tmp/local-free-multipsf-audit");
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if !arg.starts_with('-') {
            if data_dir.replace(PathBuf::from(arg)).is_some() {
                return Err("only one data directory is allowed".into());
            }
            index += 1;
            continue;
        }
        if arg == "--radius-sweep" {
            radius_sweep = true;
            index += 1;
            continue;
        }
        let (name, inline) = arg[2..]
            .split_once('=')
            .map(|(name, value)| (name, Some(value.to_owned())))
            .unwrap_or((&arg[2..], None));
        index += 1;
        let mut one_value = || -> Result<String> {
            if let Some(value) = inline.as_ref() {
                return Ok(value.clone());
            }
            let value = args
                .get(index)
                .filter(|value| !value.starts_with('-'))
                .cloned()
                .ok_or_else(|| format!("--{name} requires a value"))?;
            index += 1;
            Ok(value)
        };
        match name {
            "positions" => positions = one_value()?,
            "psf-fwhm" => psf_fwhm = one_value()?.parse().map_err(|_| "invalid --psf-fwhm")?,
            "position-radius" => {
                position_radius = one_value()?
                    .parse()
                    .map_err(|_| "invalid --position-radius")?
            }
            "position-radii" | "mask-modes" => {
                let mut values = Vec::new();
                if let Some(value) = inline {
                    values.push(value);
                }
                while let Some(value) = args.get(index).filter(|value| !value.starts_with('-')) {
                    values.push(value.clone());
                    index += 1;
                }
                if values.is_empty() {
                    return Err(format!("--{name} requires at least one value"));
                }
                if name == "position-radii" {
                    position_radii = values
                        .iter()
                        .map(|value| {
                            value
                                .parse()
                                .map_err(|_| "invalid --position-radii".to_owned())
                        })
                        .collect::<Result<Vec<f64>>>()?;
                } else {
                    mask_modes = values;
                }
            }
            "shifts-json" => shifts_json = Some(PathBuf::from(one_value()?)),
            "out-dir" => out_dir = PathBuf::from(one_value()?),
            "help" => return Err(HELP.into()),
            _ => return Err(format!("unknown argument --{name}")),
        }
    }
    let positions = parse_positions(&positions)?;
    if positions.iter().flatten().any(|value| !value.is_finite()) {
        return Err("target_positions must contain finite coordinates".into());
    }
    if !psf_fwhm.is_finite() || psf_fwhm <= 0.0 {
        return Err("psf_fwhm must be finite and positive".into());
    }
    if !position_radius.is_finite() || position_radius <= 0.0 {
        return Err("position_radius_px must be finite and positive".into());
    }
    if position_radii.is_empty()
        || position_radii
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || position_radii.iter().enumerate().any(|(index, radius)| {
            position_radii[..index]
                .iter()
                .any(|previous| previous == radius)
        })
    {
        return Err("position_radii_px must be finite, positive, and unique".into());
    }
    if mask_modes.is_empty()
        || mask_modes
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != mask_modes.len()
        || mask_modes
            .iter()
            .any(|mode| !MASK_MODES.contains(&mode.as_str()))
    {
        return Err("mask_modes must be non-empty, unique, and supported".into());
    }
    Ok(FreeOptions {
        data_dir: data_dir.ok_or("missing data directory")?,
        positions,
        psf_fwhm,
        position_radius,
        radius_sweep,
        position_radii,
        mask_modes,
        shifts_json,
        out_dir,
    })
}

#[derive(Clone)]
struct Sample {
    x: f64,
    y: f64,
    value: f64,
    background: [f64; 3],
}

#[derive(Clone, Debug)]
struct FreeFit {
    sample_count: usize,
    masked_pixel_count: usize,
    rss: f64,
    bic: f64,
    initial_positions: Vec<[f64; 2]>,
    fitted_positions: Vec<[f64; 2]>,
    position_offsets: Vec<[f64; 2]>,
    positions_at_bound: Vec<bool>,
    amplitudes: Vec<f64>,
    component_snr: Vec<f64>,
    active_component_count: usize,
    max_template_correlation: Option<f64>,
    design_condition_number: Option<f64>,
    optimizer_success: bool,
    optimizer_status: i32,
    optimizer_nfev: usize,
}

fn gaussian_terms(samples: &[Sample], positions: &[[f64; 2]], sigma: f64) -> DMatrix<f64> {
    DMatrix::from_fn(samples.len(), positions.len(), |row, column| {
        let dx = (samples[row].x - positions[column][0]) / sigma;
        let dy = (samples[row].y - positions[column][1]) / sigma;
        (-0.5 * (dx * dx + dy * dy)).exp()
    })
}

fn residuals(parameters: &[f64], samples: &[Sample], sigma: f64) -> Vec<f64> {
    let count = (parameters.len() - 3) / 3;
    let positions = (0..count)
        .map(|index| [parameters[4 + 3 * index], parameters[5 + 3 * index]])
        .collect::<Vec<_>>();
    let terms = gaussian_terms(samples, &positions, sigma);
    samples
        .iter()
        .enumerate()
        .map(|(row, sample)| {
            let background = sample.background[0] * parameters[0]
                + sample.background[1] * parameters[1]
                + sample.background[2] * parameters[2];
            let source = (0..count)
                .map(|column| parameters[3 + 3 * column] * terms[(row, column)])
                .sum::<f64>();
            background + source - sample.value
        })
        .collect()
}

fn jacobian(
    parameters: &[f64],
    current: &[f64],
    lower: &[f64],
    upper: &[f64],
    samples: &[Sample],
    sigma: f64,
) -> DMatrix<f64> {
    let mut matrix = DMatrix::zeros(current.len(), parameters.len());
    for column in 0..parameters.len() {
        let step = f64::EPSILON.sqrt() * parameters[column].abs().max(1.0);
        let probe = if parameters[column] + step <= upper[column] {
            parameters[column] + step
        } else if parameters[column] - step >= lower[column] {
            parameters[column] - step
        } else {
            continue;
        };
        let mut changed = parameters.to_vec();
        changed[column] = probe;
        let shifted = residuals(&changed, samples, sigma);
        for row in 0..current.len() {
            matrix[(row, column)] = (shifted[row] - current[row]) / (probe - parameters[column]);
        }
    }
    matrix
}

fn fit_free(
    frame: &FitsFrame,
    mask: &[bool],
    model_positions: &[[f64; 2]],
    patch_positions: &[[f64; 2]],
    psf_fwhm: f64,
    position_radius: f64,
) -> Result<Option<FreeFit>> {
    if model_positions.is_empty()
        || patch_positions.is_empty()
        || mask.len() != frame.data.len()
        || model_positions
            .iter()
            .chain(patch_positions)
            .flatten()
            .any(|value| !value.is_finite())
        || !psf_fwhm.is_finite()
        || psf_fwhm <= 0.0
        || !position_radius.is_finite()
        || position_radius <= 0.0
    {
        return Ok(None);
    }
    let sigma = (psf_fwhm / 2.35482).max(0.5);
    let margin = 4usize.max((3.5 * sigma).ceil() as usize);
    let bounds = multipsf_audit::patch_bounds(frame.width, frame.height, patch_positions, margin);
    let [x0, y0, x1, y1] = bounds;
    let (width, height) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    if width < 5 || height < 5 {
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
            if !mask[index] && value.is_finite() {
                samples.push(Sample {
                    x: x as f64,
                    y: y as f64,
                    value,
                    background: [
                        1.0,
                        (x as f64 - center_x) / scale,
                        (y as f64 - center_y) / scale,
                    ],
                });
            }
        }
    }
    let count = model_positions.len();
    let parameter_count = 3 + 3 * count;
    if samples.len() < 24.max(parameter_count + 8) {
        return Ok(None);
    }

    let fixed = multipsf_audit::fit_fixed(frame, mask, model_positions, patch_positions, psf_fwhm);
    let Some(fixed) = fixed else {
        return Ok(None);
    };
    let source_terms = gaussian_terms(&samples, model_positions, sigma);
    let background_terms = DMatrix::from_fn(samples.len(), 3, |row, column| {
        samples[row].background[column]
    });
    let values = DVector::from_iterator(samples.len(), samples.iter().map(|sample| sample.value));
    let amplitudes = DVector::from_column_slice(&fixed.amplitudes);
    let initial_residual = &values - &source_terms * amplitudes;
    let Some(initial_background) = multipsf_audit::least_squares(
        &background_terms,
        &DMatrix::from_column_slice(samples.len(), 1, initial_residual.as_slice()),
    ) else {
        return Ok(None);
    };
    let mut parameters = vec![
        initial_background[(0, 0)],
        initial_background[(1, 0)],
        initial_background[(2, 0)],
    ];
    let mut lower = vec![f64::NEG_INFINITY; parameter_count];
    let mut upper = vec![f64::INFINITY; parameter_count];
    for (index, position) in model_positions.iter().enumerate() {
        parameters.extend([fixed.amplitudes[index].max(0.0), position[0], position[1]]);
        lower[3 + 3 * index] = 0.0;
        lower[4 + 3 * index] = position[0] - position_radius;
        lower[5 + 3 * index] = position[1] - position_radius;
        upper[4 + 3 * index] = position[0] + position_radius;
        upper[5 + 3 * index] = position[1] + position_radius;
    }
    let mut current = residuals(&parameters, &samples, sigma);
    let mut cost = 0.5 * current.iter().map(|value| value * value).sum::<f64>();
    let mut lambda = 1e-3;
    let mut success = false;
    let mut status = 0;
    let mut nfev = 1usize;
    for _ in 0..80 {
        if nfev >= 300 {
            break;
        }
        let jac = jacobian(&parameters, &current, &lower, &upper, &samples, sigma);
        let jt = jac.transpose();
        let normal = &jt * &jac;
        let gradient = &jt * DVector::from_column_slice(&current);
        let projected_max = (0..parameter_count)
            .map(|index| {
                let at_lower = parameters[index] <= lower[index] + 1e-9;
                let at_upper = parameters[index] >= upper[index] - 1e-9;
                if (at_lower && gradient[index] > 0.0) || (at_upper && gradient[index] < 0.0) {
                    0.0
                } else {
                    gradient[index].abs()
                }
            })
            .fold(0.0, f64::max);
        if projected_max < 1e-7 {
            success = true;
            status = 1;
            break;
        }
        let mut accepted = false;
        for _ in 0..12 {
            let mut damped = normal.clone();
            for index in 0..parameter_count {
                damped[(index, index)] += lambda * normal[(index, index)].abs().max(1.0);
            }
            let Some(inverse) = damped.try_inverse() else {
                lambda *= 10.0;
                continue;
            };
            let delta = -(inverse * gradient.clone());
            let mut trial = parameters.clone();
            for index in 0..parameter_count {
                trial[index] = (trial[index] + delta[index])
                    .max(lower[index])
                    .min(upper[index]);
            }
            if trial
                .iter()
                .zip(&parameters)
                .all(|(a, b)| (a - b).abs() < 1e-12)
            {
                lambda *= 10.0;
                continue;
            }
            let trial_residual = residuals(&trial, &samples, sigma);
            nfev += 1;
            let trial_cost = 0.5
                * trial_residual
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>();
            if trial_cost < cost {
                let reduction = cost - trial_cost;
                let step = trial
                    .iter()
                    .zip(&parameters)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm = parameters
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                parameters = trial;
                current = trial_residual;
                cost = trial_cost;
                lambda = (lambda * 0.3).max(1e-12);
                accepted = true;
                if step <= 1e-9 * (norm + 1e-9) {
                    success = true;
                    status = 2;
                } else if reduction <= 1e-9 * cost.max(1.0) {
                    success = true;
                    status = 3;
                }
                break;
            }
            lambda *= 10.0;
        }
        if success {
            break;
        }
        if !accepted && lambda > 1e14 {
            success = true;
            status = 4;
            break;
        }
    }

    let rss = current
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .max(f64::EPSILON);
    let bic = samples.len() as f64 * (rss / samples.len() as f64).ln()
        + parameter_count as f64 * (samples.len() as f64).ln();
    let fitted_positions = (0..count)
        .map(|index| [parameters[4 + 3 * index], parameters[5 + 3 * index]])
        .collect::<Vec<_>>();
    let position_offsets = fitted_positions
        .iter()
        .zip(model_positions)
        .map(|(fitted, initial)| [fitted[0] - initial[0], fitted[1] - initial[1]])
        .collect::<Vec<_>>();
    let position_tolerance = 1e-5f64.max(position_radius * 1e-5);
    let positions_at_bound = position_offsets
        .iter()
        .map(|offset| {
            offset[0].abs() >= position_radius - position_tolerance
                || offset[1].abs() >= position_radius - position_tolerance
        })
        .collect::<Vec<_>>();
    let fitted_terms = gaussian_terms(&samples, &fitted_positions, sigma);
    let Some(background_from_terms) =
        multipsf_audit::least_squares(&background_terms, &fitted_terms)
    else {
        return Ok(None);
    };
    let source_residual = &fitted_terms - &background_terms * background_from_terms;
    let gram = source_residual.transpose() * &source_residual;
    let max_template_correlation = (count >= 2).then(|| {
        let mut maximum = 0.0f64;
        for left in 0..count {
            for right in left + 1..count {
                let norm = (gram[(left, left)].max(f64::EPSILON)
                    * gram[(right, right)].max(f64::EPSILON))
                .sqrt();
                maximum = maximum.max((gram[(left, right)] / norm).abs());
            }
        }
        maximum
    });
    let singular = source_residual.clone().svd(false, false).singular_values;
    let design_condition_number = if singular.is_empty() {
        None
    } else {
        let maximum = singular.iter().copied().fold(0.0, f64::max);
        let minimum = singular.iter().copied().fold(f64::INFINITY, f64::min);
        (minimum > f64::EPSILON * maximum.max(1.0))
            .then_some(maximum / minimum)
            .filter(|value| value.is_finite())
    };
    let final_jacobian = jacobian(&parameters, &current, &lower, &upper, &samples, sigma);
    let normal = final_jacobian.transpose() * final_jacobian;
    let dof = samples.len().saturating_sub(parameter_count).max(1) as f64;
    let variance = rss / dof;
    let covariance = normal
        .svd(true, true)
        .pseudo_inverse(1e-15)
        .ok()
        .map(|matrix| matrix * variance);
    let amplitudes = (0..count)
        .map(|index| parameters[3 + 3 * index])
        .collect::<Vec<_>>();
    let max_amplitude = amplitudes
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let active_tolerance = 1e-12f64.max(f64::EPSILON * max_amplitude.max(1.0));
    let active_component_count = amplitudes
        .iter()
        .filter(|value| **value > active_tolerance)
        .count();
    let component_snr = (0..count)
        .map(|index| {
            let amplitude = amplitudes[index];
            let error = covariance.as_ref().map_or(f64::EPSILON, |matrix| {
                matrix[(3 + 3 * index, 3 + 3 * index)]
                    .max(f64::EPSILON)
                    .sqrt()
            });
            if amplitude > active_tolerance {
                amplitude / error
            } else {
                0.0
            }
        })
        .collect();
    Ok(Some(FreeFit {
        sample_count: samples.len(),
        masked_pixel_count: width * height - samples.len(),
        rss,
        bic,
        initial_positions: model_positions.to_vec(),
        fitted_positions,
        position_offsets,
        positions_at_bound,
        amplitudes,
        component_snr,
        active_component_count,
        max_template_correlation,
        design_condition_number,
        optimizer_success: success,
        optimizer_status: status,
        optimizer_nfev: nfev,
    }))
}

pub fn run_audit(
    paths: &[PathBuf],
    positions: &[[f64; 2]],
    psf_fwhm: f64,
    position_radius: f64,
    mask_modes: &[String],
    frame_shifts: Option<&[[f64; 2]]>,
) -> Result<FreeMultiPSFAuditResult> {
    if paths.is_empty() {
        return Err("paths cannot be empty".into());
    }
    if positions.len() < 2 || positions.iter().flatten().any(|value| !value.is_finite()) {
        return Err("target_positions must contain at least two finite positions".into());
    }
    for (index, left) in positions.iter().enumerate() {
        if positions[index + 1..]
            .iter()
            .any(|right| (left[0] - right[0]).hypot(left[1] - right[1]) <= 1e-6)
        {
            return Err("target_positions must be distinct".into());
        }
    }
    if !psf_fwhm.is_finite() || psf_fwhm <= 0.0 {
        return Err("psf_fwhm must be finite and positive".into());
    }
    if !position_radius.is_finite() || position_radius <= 0.0 {
        return Err("position_radius_px must be finite and positive".into());
    }
    if mask_modes.is_empty()
        || mask_modes
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != mask_modes.len()
        || mask_modes
            .iter()
            .any(|mode| !MASK_MODES.contains(&mode.as_str()))
    {
        return Err("mask_modes must be non-empty, unique, and supported".into());
    }
    let shifts = if let Some(shifts) = frame_shifts {
        if shifts.len() != paths.len() || shifts.iter().flatten().any(|value| !value.is_finite()) {
            return Err("frame_shifts must match paths and contain finite values".into());
        }
        shifts.to_vec()
    } else {
        vec![[0.0, 0.0]; paths.len()]
    };
    let mut rows = Vec::new();
    for (frame_index, path) in paths.iter().enumerate() {
        let frame = fits::read_fits(path, false)?;
        let shift = shifts[frame_index];
        let frame_positions = positions
            .iter()
            .map(|position| [position[0] + shift[0], position[1] + shift[1]])
            .collect::<Vec<_>>();
        let masks = empirical_pair_fit::mask_modes(&frame, mask_modes, &[])?;
        for (mask_mode, mask) in masks {
            let fits = (1..=frame_positions.len())
                .map(|count| {
                    fit_free(
                        &frame,
                        &mask,
                        &frame_positions[..count],
                        &frame_positions,
                        psf_fwhm,
                        position_radius,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let single_bic = fits.first().and_then(Option::as_ref).map(|fit| fit.bic);
            for (index, fit) in fits.iter().enumerate() {
                let component_count = index + 1;
                let previous_bic = index
                    .checked_sub(1)
                    .and_then(|previous| fits[previous].as_ref().map(|fit| fit.bic));
                let bic = fit.as_ref().map(|fit| fit.bic);
                let delta_single = single_bic
                    .zip(bic)
                    .map(|(single, current)| single - current);
                let delta_previous = previous_bic
                    .zip(bic)
                    .map(|(previous, current)| previous - current);
                let note;
                let (
                    initial_positions,
                    fitted_positions,
                    position_offsets,
                    positions_at_bound,
                    sample_count,
                    masked_pixel_count,
                    rss,
                    bic_value,
                    delta_single_value,
                    delta_previous_value,
                    amplitudes,
                    component_snr,
                    active_component_count,
                    max_template_correlation,
                    design_condition_number,
                    optimizer_success,
                    optimizer_status,
                    optimizer_nfev,
                    confirmation_line,
                ) = if let Some(fit) = fit {
                    let added_snr = if component_count >= 2 {
                        &fit.component_snr[1..]
                    } else {
                        &[]
                    };
                    let added_positions_interior =
                        !fit.positions_at_bound.iter().skip(1).any(|value| *value);
                    let confirmation = component_count >= 2
                        && delta_single.is_some_and(|value| value >= 10.0)
                        && !added_snr.is_empty()
                        && added_snr.iter().all(|value| *value >= 5.0)
                        && added_positions_interior
                        && fit.optimizer_success;
                    note = if confirmation {
                        "达到当前工程双/多源证据线；仍需实测 PSF、注入和星表核验"
                    } else if component_count >= 2 && !added_positions_interior {
                        "新增分量触碰位置搜索边界；局部模型比较，不等于物理恒星身份"
                    } else if !fit.optimizer_success {
                        "优化器未报告收敛；局部模型比较，不等于物理恒星身份"
                    } else {
                        "局部模型比较；不等于物理恒星身份"
                    };
                    (
                        fit.initial_positions.clone(),
                        fit.fitted_positions.clone(),
                        fit.position_offsets.clone(),
                        fit.positions_at_bound.clone(),
                        fit.sample_count,
                        fit.masked_pixel_count,
                        Some(fit.rss),
                        bic,
                        delta_single,
                        delta_previous,
                        fit.amplitudes.clone(),
                        fit.component_snr.clone(),
                        fit.active_component_count,
                        fit.max_template_correlation,
                        fit.design_condition_number,
                        fit.optimizer_success,
                        Some(fit.optimizer_status),
                        Some(fit.optimizer_nfev),
                        confirmation,
                    )
                } else {
                    note = "局部有效像素不足或自由位置拟合数值不稳定";
                    let initial = frame_positions[..component_count].to_vec();
                    let n = initial.len();
                    (
                        initial.clone(),
                        initial,
                        vec![[0.0, 0.0]; n],
                        vec![false; n],
                        0,
                        0,
                        None,
                        None,
                        None,
                        None,
                        Vec::new(),
                        Vec::new(),
                        0,
                        None,
                        None,
                        false,
                        None,
                        None,
                        false,
                    )
                };
                rows.push(FreeMultiPSFAuditRow {
                    frame_index: frame_index + 1,
                    path: multipsf_audit::display_input_path(&frame.path),
                    mask_mode: mask_mode.clone(),
                    component_count,
                    initial_positions,
                    fitted_positions,
                    position_offsets,
                    positions_at_bound,
                    sample_count,
                    masked_pixel_count,
                    rss,
                    bic: bic_value,
                    delta_bic_from_single: delta_single_value,
                    delta_bic_from_previous: delta_previous_value,
                    amplitudes,
                    component_snr,
                    active_component_count,
                    max_template_correlation,
                    design_condition_number,
                    optimizer_success,
                    optimizer_status,
                    optimizer_nfev,
                    confirmation_line,
                    note: note.into(),
                });
            }
        }
    }
    let raw_rows = rows
        .iter()
        .filter(|row| row.mask_mode == "raw")
        .collect::<Vec<_>>();
    let confirmations = (2..=positions.len())
        .map(|count| {
            let passed = raw_rows
                .iter()
                .filter(|row| row.component_count == count && row.confirmation_line)
                .count();
            format!("K={count}: {passed}/{} 帧", paths.len())
        })
        .collect::<Vec<_>>()
        .join(", ");
    let mut best_by_frame = Vec::new();
    for frame_index in 1..=paths.len() {
        let mut candidates = raw_rows
            .iter()
            .copied()
            .filter(|row| row.frame_index == frame_index && row.bic.is_some())
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            best_by_frame.push(format!("F{frame_index:02}: 无有效局部拟合"));
            continue;
        }
        candidates.sort_by(|left, right| left.bic.unwrap().total_cmp(&right.bic.unwrap()));
        let best = candidates[0];
        best_by_frame.push(format!(
            "F{frame_index:02}: BIC 最优 K={}{}",
            best.component_count,
            best.delta_bic_from_single
                .map(|value| format!(" (ΔBIC={value:.2})"))
                .unwrap_or_default()
        ));
    }
    let conclusion = format!(
        "自由位置 K 源 PSF 审计完成：{} 帧、{} 个候选位置，raw 证据线下确认统计为 {}。位置仅允许在初始候选周围 ±{} px 移动；该模型仍不能替代空间变 PSF、完整噪声似然、人工注入或 WCS/星表身份。{}",
        paths.len(),
        positions.len(),
        confirmations,
        position_radius,
        if best_by_frame.is_empty() {
            String::new()
        } else {
            format!(" 各帧 raw 最优模型：{}。", best_by_frame.join("; "))
        }
    );
    Ok(FreeMultiPSFAuditResult {
        frame_count: paths.len(),
        target_positions: positions.to_vec(),
        frame_shifts: shifts.clone(),
        mask_modes: mask_modes.to_vec(),
        psf_fwhm_px: psf_fwhm,
        position_radius_px: position_radius,
        parameters: json!({
            "psf_fwhm_px": psf_fwhm,
            "position_radius_px": position_radius,
            "mask_modes": mask_modes,
            "component_counts": (1..=positions.len()).collect::<Vec<_>>(),
            "confirmation_delta_bic": 10.0,
            "confirmation_component_snr": 5.0,
            "require_added_positions_interior": true,
            "frame_shifts": shifts,
            "model": "shared linear background + bounded free-position Gaussian PSF + nonnegative amplitudes",
            "interpretation_boundary": "local model evidence, not WCS/catalog identity"
        }),
        rows,
        conclusion,
    })
}

pub fn run_radius_sweep(
    paths: &[PathBuf],
    positions: &[[f64; 2]],
    position_radii: &[f64],
    psf_fwhm: f64,
    mask_modes: &[String],
    frame_shifts: Option<&[[f64; 2]]>,
) -> Result<FreeMultiPSFSweepResult> {
    if paths.is_empty() {
        return Err("paths cannot be empty".into());
    }
    if position_radii.is_empty()
        || position_radii
            .iter()
            .any(|radius| !radius.is_finite() || *radius <= 0.0)
        || position_radii.iter().enumerate().any(|(index, radius)| {
            position_radii[..index]
                .iter()
                .any(|previous| previous == radius)
        })
    {
        return Err("position_radii_px must be finite, positive, and unique".into());
    }
    let mut rows = Vec::new();
    for radius in position_radii {
        let audit = run_audit(
            paths,
            positions,
            psf_fwhm,
            *radius,
            mask_modes,
            frame_shifts,
        )?;
        rows.extend(audit.rows.into_iter().map(|row| FreeMultiPSFSweepRow {
            position_radius_px: *radius,
            frame_index: row.frame_index,
            path: row.path,
            mask_mode: row.mask_mode,
            component_count: row.component_count,
            delta_bic_from_single: row.delta_bic_from_single,
            component_snr: row.component_snr,
            fitted_positions: row.fitted_positions,
            positions_at_bound: row.positions_at_bound,
            optimizer_success: row.optimizer_success,
            confirmation_line: row.confirmation_line,
            note: row.note,
        }));
    }
    let mut summaries = Vec::new();
    for radius in position_radii {
        let pair_rows = rows
            .iter()
            .filter(|row| {
                row.position_radius_px == *radius
                    && row.mask_mode == "raw"
                    && row.component_count == 2
            })
            .collect::<Vec<_>>();
        let confirmed = pair_rows.iter().filter(|row| row.confirmation_line).count();
        let at_bound = pair_rows
            .iter()
            .filter(|row| row.positions_at_bound.iter().skip(1).any(|value| *value))
            .count();
        let not_converged = pair_rows
            .iter()
            .filter(|row| !row.optimizer_success)
            .count();
        summaries.push(format!(
            "r={radius}: raw K=2 确认 {confirmed}/{}，新增位置触边 {at_bound}/{}，未收敛 {not_converged}/{}",
            paths.len(),
            paths.len(),
            paths.len()
        ));
    }
    Ok(FreeMultiPSFSweepResult {
        frame_count: paths.len(),
        target_positions: positions.to_vec(),
        position_radii_px: position_radii.to_vec(),
        mask_modes: mask_modes.to_vec(),
        psf_fwhm_px: psf_fwhm,
        parameters: json!({
            "position_radii_px": position_radii,
            "psf_fwhm_px": psf_fwhm,
            "mask_modes": mask_modes,
            "confirmation_delta_bic": 10.0,
            "confirmation_component_snr": 5.0,
            "frame_shifts": frame_shifts,
            "interpretation_boundary": "radius stability is a necessary local check, not catalog identity"
        }),
        rows,
        conclusion: format!(
            "自由位置半径敏感性审计完成：{} 个半径、{} 帧。{}。半径改变会改变可吸收的局部结构，因此稳定性是必要的审计条件，不等于物理恒星身份。",
            position_radii.len(),
            paths.len(),
            summaries.join("；")
        ),
    })
}

fn csv_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::String(value) => value.clone(),
        Value::Array(_) => serde_json::to_string(value).unwrap_or_default(),
        _ => value.to_string(),
    }
}

fn write_csv<T: Serialize>(path: &Path, fields: &[&str], rows: &[T]) -> Result<()> {
    let mut file = File::create(path).map_err(|error| error.to_string())?;
    file.write_all(b"\xef\xbb\xbf")
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);
    writer
        .write_record(fields)
        .map_err(|error| error.to_string())?;
    for row in rows {
        let value = serde_json::to_value(row).map_err(|error| error.to_string())?;
        let object = value.as_object().ok_or("row serialization failed")?;
        writer
            .write_record(
                fields
                    .iter()
                    .map(|field| object.get(*field).map(csv_value).unwrap_or_default()),
            )
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn chart_proxy(
    target_positions: Vec<[f64; 2]>,
    frame_count: usize,
    psf_fwhm: f64,
    mask_modes: Vec<String>,
    rows: Vec<multipsf_audit::LocalMultiPSFAuditRow>,
    conclusion: String,
) -> multipsf_audit::LocalMultiPSFAuditResult {
    multipsf_audit::LocalMultiPSFAuditResult {
        frame_count,
        target_positions,
        frame_shifts: Vec::new(),
        mask_modes,
        psf_fwhm_px: psf_fwhm,
        parameters: Value::Null,
        rows,
        conclusion,
    }
}

pub fn write_audit_artifacts(result: &FreeMultiPSFAuditResult, output: &Path) -> Result<Value> {
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    let csv_path = output.join("local_free_multipsf_audit.csv");
    write_csv(&csv_path, AUDIT_FIELDS, &result.rows)?;
    fs::write(
        output.join("local_free_multipsf_audit.json"),
        serde_json::to_vec_pretty(&result.as_json()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let proxy_rows = result
        .rows
        .iter()
        .map(|row| multipsf_audit::LocalMultiPSFAuditRow {
            frame_index: row.frame_index,
            path: row.path.clone(),
            mask_mode: row.mask_mode.clone(),
            component_count: row.component_count,
            target_positions: row.fitted_positions.clone(),
            sample_count: row.sample_count,
            masked_pixel_count: row.masked_pixel_count,
            rss: row.rss,
            bic: row.bic,
            delta_bic_from_single: row.delta_bic_from_single,
            delta_bic_from_previous: row.delta_bic_from_previous,
            amplitudes: row.amplitudes.clone(),
            component_snr: row.component_snr.clone(),
            active_component_count: row.active_component_count,
            max_template_correlation: row.max_template_correlation,
            design_condition_number: row.design_condition_number,
            confirmation_line: row.confirmation_line,
            note: row.note.clone(),
        })
        .collect();
    let chart = chart_proxy(
        result.target_positions.clone(),
        result.frame_count,
        result.psf_fwhm_px,
        result.mask_modes.clone(),
        proxy_rows,
        result.conclusion.clone(),
    );
    multipsf_audit::write_chart(&output.join("local_free_multipsf_bic.png"), &chart)?;
    Ok(json!({"output":output,"frame_count":result.frame_count,"conclusion":result.conclusion}))
}

pub fn write_sweep_artifacts(result: &FreeMultiPSFSweepResult, output: &Path) -> Result<Value> {
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    let csv_path = output.join("local_free_multipsf_radius_sweep.csv");
    write_csv(&csv_path, SWEEP_FIELDS, &result.rows)?;
    fs::write(
        output.join("local_free_multipsf_radius_sweep.json"),
        serde_json::to_vec_pretty(&result.as_json()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let mut proxy_rows = Vec::new();
    for (radius_index, radius) in result.position_radii_px.iter().enumerate() {
        let count = radius_index + 1;
        for mode in &result.mask_modes {
            let mut values = result
                .rows
                .iter()
                .filter(|row| {
                    row.position_radius_px == *radius
                        && row.mask_mode == *mode
                        && row.component_count == 2
                })
                .filter_map(|row| row.delta_bic_from_single)
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            let median = if values.is_empty() {
                None
            } else {
                let middle = values.len() / 2;
                Some(if values.len().is_multiple_of(2) {
                    (values[middle - 1] + values[middle]) / 2.0
                } else {
                    values[middle]
                })
            };
            proxy_rows.push(multipsf_audit::LocalMultiPSFAuditRow {
                frame_index: 1,
                path: String::new(),
                mask_mode: mode.clone(),
                component_count: count,
                target_positions: Vec::new(),
                sample_count: 0,
                masked_pixel_count: 0,
                rss: None,
                bic: None,
                delta_bic_from_single: median,
                delta_bic_from_previous: None,
                amplitudes: Vec::new(),
                component_snr: Vec::new(),
                active_component_count: 0,
                max_template_correlation: None,
                design_condition_number: None,
                confirmation_line: false,
                note: String::new(),
            });
        }
    }
    let targets = result
        .position_radii_px
        .iter()
        .map(|radius| [*radius, 0.0])
        .collect();
    let chart = chart_proxy(
        targets,
        result.frame_count,
        result.psf_fwhm_px,
        result.mask_modes.clone(),
        proxy_rows,
        result.conclusion.clone(),
    );
    multipsf_audit::write_chart(&output.join("local_free_multipsf_radius_sweep.png"), &chart)?;
    Ok(json!({"output":output,"frame_count":result.frame_count,"conclusion":result.conclusion}))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let options = parse_args(args)?;
    let paths = empirical_pair_fit::ordered_fits_paths(&options.data_dir)?;
    let shifts = options
        .shifts_json
        .as_deref()
        .map(multipsf_audit::parse_shifts)
        .transpose()?;
    if options.radius_sweep {
        let result = run_radius_sweep(
            &paths,
            &options.positions,
            &options.position_radii,
            options.psf_fwhm,
            &options.mask_modes,
            shifts.as_deref(),
        )?;
        write_sweep_artifacts(&result, &options.out_dir)
    } else {
        let result = run_audit(
            &paths,
            &options.positions,
            options.psf_fwhm,
            options.position_radius,
            &options.mask_modes,
            shifts.as_deref(),
        )?;
        write_audit_artifacts(&result, &options.out_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AuxiliaryData, PixelData};

    fn synthetic_frame() -> FitsFrame {
        let (width, height) = (96usize, 96usize);
        let mut image = vec![21.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                image[y * width + x] += ((x * 17 + y * 31) % 9) as f32 * 0.08 - 0.32;
            }
        }
        let sigma = 2.0 / 2.35482;
        for (x0, y0, amplitude) in [(48.35, 48.25, 150.0), (53.15, 50.20, 90.0)] {
            for y in 0..height {
                for x in 0..width {
                    let dx = (x as f64 - x0) / sigma;
                    let dy = (y as f64 - y0) / sigma;
                    image[y * width + x] += (amplitude * (-0.5 * (dx * dx + dy * dy)).exp()) as f32;
                }
            }
        }
        FitsFrame {
            path: PathBuf::from("synthetic-free-multipsf.fits"),
            header: Default::default(),
            data: PixelData::F32(image),
            auxiliary: Some(AuxiliaryData { values: [0.0; 26] }),
            data_offset: 0,
            width,
            height,
        }
    }

    #[test]
    fn parser_preserves_free_position_defaults_and_radius_sweep_controls() {
        let defaults = parse_args(&["frames".into()]).unwrap();
        assert_eq!(defaults.psf_fwhm, 2.0);
        assert_eq!(defaults.position_radius, 1.25);
        assert!(!defaults.radius_sweep);
        assert_eq!(defaults.position_radii, [0.25, 0.5, 0.75, 1.0, 1.25]);
        assert_eq!(defaults.positions.len(), 3);
        let parsed = parse_args(&[
            "frames".into(),
            "--positions".into(),
            "10,12;14,16".into(),
            "--radius-sweep".into(),
            "--position-radii".into(),
            "0.5".into(),
            "1".into(),
            "--mask-modes".into(),
            "raw".into(),
        ])
        .unwrap();
        assert!(parsed.radius_sweep);
        assert_eq!(parsed.position_radii, [0.5, 1.0]);
        assert_eq!(parsed.positions, [[10.0, 12.0], [14.0, 16.0]]);
        assert_eq!(parsed.mask_modes, ["raw"]);
    }

    #[test]
    fn bounded_free_fit_refines_injected_gaussian_centers() {
        let frame = synthetic_frame();
        let mask = vec![false; frame.data.len()];
        let initial = [[48.0, 48.0], [53.0, 50.0]];
        let fit = fit_free(&frame, &mask, &initial, &initial, 2.0, 1.25)
            .unwrap()
            .unwrap();
        assert!(fit.optimizer_success, "status={}", fit.optimizer_status);
        assert_eq!(fit.active_component_count, 2);
        assert!(
            (fit.fitted_positions[0][0] - 48.35).hypot(fit.fitted_positions[0][1] - 48.25) < 0.4
        );
        assert!(
            (fit.fitted_positions[1][0] - 53.15).hypot(fit.fitted_positions[1][1] - 50.20) < 0.4
        );
        assert!(fit.component_snr[1] > 5.0);
        assert!(fit.design_condition_number.unwrap().is_finite());
        assert!(fit.rss < 10.0);
    }

    #[test]
    fn audit_and_radius_sweep_write_native_artifacts() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/empirical_pair_audit_parity/frames");
        let paths = empirical_pair_fit::ordered_fits_paths(&fixture_dir).unwrap();
        let positions = [[46.0, 48.0], [48.5, 48.0], [50.0, 45.0]];
        let modes = vec!["raw".to_owned()];
        let audit = run_audit(&paths, &positions, 2.0, 0.75, &modes, None).unwrap();
        assert_eq!(audit.rows.len(), 3);
        assert!(audit.rows.iter().all(|row| {
            row.position_offsets
                .iter()
                .flatten()
                .all(|offset| offset.abs() <= 0.75 + 1e-8)
        }));
        let sweep = run_radius_sweep(&paths, &positions, &[0.5, 1.0], 2.0, &modes, None).unwrap();
        assert_eq!(sweep.rows.len(), 6);

        let root = std::env::temp_dir().join(format!(
            "rst19-free-multipsf-artifacts-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_audit_artifacts(&audit, &root.join("audit")).unwrap();
        write_sweep_artifacts(&sweep, &root.join("sweep")).unwrap();
        for (directory, artifacts) in [
            (
                root.join("audit"),
                [
                    "local_free_multipsf_audit.csv",
                    "local_free_multipsf_audit.json",
                    "local_free_multipsf_bic.png",
                ],
            ),
            (
                root.join("sweep"),
                [
                    "local_free_multipsf_radius_sweep.csv",
                    "local_free_multipsf_radius_sweep.json",
                    "local_free_multipsf_radius_sweep.png",
                ],
            ),
        ] {
            for artifact in artifacts {
                assert!(directory.join(artifact).is_file(), "missing {artifact}");
            }
        }
        let _ = fs::remove_dir_all(root);
    }
}
