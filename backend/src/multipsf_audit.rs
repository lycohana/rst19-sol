//! Fixed-position K=1..N Gaussian-PSF model comparison.
//!
//! This is a local model diagnostic. It does not establish catalog identity,
//! physical multiplicity, or stellar probability.

use crate::{empirical_pair_fit, fits, models::FitsFrame, Result};
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
const CSV_FIELDS: &[&str] = &[
    "frame_index",
    "path",
    "mask_mode",
    "component_count",
    "target_positions",
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
    "confirmation_line",
    "note",
];

pub const HELP: &str = "比较同一局部的固定位置 K=1..N Gaussian PSF 模型\n\nUsage: rst19-multipsf-audit DATA_DIR [OPTIONS]\n\nOptions: --positions \"x,y;x,y;...\" --psf-fwhm N\n         --mask-modes raw range_masked sentinel_masked range_sentinel_masked\n         --shifts-json FILE --out-dir DIR";

#[derive(Clone, Debug, Serialize)]
pub struct LocalMultiPSFAuditRow {
    pub frame_index: usize,
    pub path: String,
    pub mask_mode: String,
    pub component_count: usize,
    pub target_positions: Vec<[f64; 2]>,
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
    pub confirmation_line: bool,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalMultiPSFAuditResult {
    pub frame_count: usize,
    pub target_positions: Vec<[f64; 2]>,
    pub frame_shifts: Vec<[f64; 2]>,
    pub mask_modes: Vec<String>,
    pub psf_fwhm_px: f64,
    pub parameters: Value,
    pub rows: Vec<LocalMultiPSFAuditRow>,
    pub conclusion: String,
}

impl LocalMultiPSFAuditResult {
    fn as_json(&self) -> Value {
        json!({
            "frame_count": self.frame_count,
            "target_positions": self.target_positions,
            "frame_shifts": self.frame_shifts,
            "mask_modes": self.mask_modes,
            "psf_fwhm_px": self.psf_fwhm_px,
            "parameters": self.parameters,
            "rows": self.rows,
            "conclusion": self.conclusion,
            "note": "位置由 detector 候选/人工审计坐标固定给出；K=1..N 共享局部背景和像素窗口，非负幅度用 NNLS 拟合。结果是局部模型敏感性证据，不是物理恒星身份。"
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ModelFit {
    pub(crate) sample_count: usize,
    pub(crate) masked_pixel_count: usize,
    pub(crate) rss: f64,
    pub(crate) bic: f64,
    pub(crate) amplitudes: Vec<f64>,
    pub(crate) component_snr: Vec<f64>,
    pub(crate) active_component_count: usize,
    pub(crate) max_template_correlation: Option<f64>,
    pub(crate) design_condition_number: Option<f64>,
}

#[derive(Clone, Debug)]
struct Options {
    data_dir: PathBuf,
    positions: Vec<[f64; 2]>,
    psf_fwhm: f64,
    mask_modes: Vec<String>,
    shifts_json: Option<PathBuf>,
    out_dir: PathBuf,
}

fn parse_positions(value: &str) -> Result<Vec<[f64; 2]>> {
    let mut positions = Vec::new();
    for item in value.split(';') {
        let parts = item.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() != 2 {
            return Err(format!("position must use x,y;... format: {item}"));
        }
        let x = parts[0]
            .parse::<f64>()
            .map_err(|_| format!("invalid x coordinate in position: {item}"))?;
        let y = parts[1]
            .parse::<f64>()
            .map_err(|_| format!("invalid y coordinate in position: {item}"))?;
        positions.push([x, y]);
    }
    if positions.len() < 2 {
        return Err("at least two positions are required".into());
    }
    Ok(positions)
}

fn required_value(args: &[String], index: &mut usize, name: &str) -> Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("--{name} requires a value"))
}

fn parse_args(args: &[String]) -> Result<Options> {
    let mut data_dir = None;
    let mut positions = "1269,3465;1274,3467;1272,3453".to_owned();
    let mut psf_fwhm = 2.0;
    let mut mask_modes = vec![
        "raw".to_owned(),
        "range_masked".to_owned(),
        "sentinel_masked".to_owned(),
    ];
    let mut shifts_json = None;
    let mut out_dir = PathBuf::from("tmp/local-multipsf-audit");
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--help" || arg == "-h" {
            return Err(HELP.into());
        }
        if !arg.starts_with('-') {
            if data_dir.replace(PathBuf::from(arg)).is_some() {
                return Err("only one data directory is allowed".into());
            }
            index += 1;
            continue;
        }
        let (name, inline) = arg[2..]
            .split_once('=')
            .map(|(name, value)| (name, Some(value.to_owned())))
            .unwrap_or((&arg[2..], None));
        match name {
            "positions" => {
                positions = match inline {
                    Some(value) => value,
                    None => required_value(args, &mut index, name)?,
                };
            }
            "psf-fwhm" => {
                let value = match inline {
                    Some(value) => value,
                    None => required_value(args, &mut index, name)?,
                };
                psf_fwhm = value
                    .parse()
                    .map_err(|_| "invalid --psf-fwhm: expected a number")?;
            }
            "mask-modes" => {
                mask_modes.clear();
                if let Some(value) = inline {
                    mask_modes.extend(value.split(',').map(str::to_owned));
                } else {
                    while index + 1 < args.len() && !args[index + 1].starts_with('-') {
                        index += 1;
                        mask_modes.extend(args[index].split(',').map(str::to_owned));
                    }
                    if mask_modes.is_empty() {
                        return Err("--mask-modes requires at least one mode".into());
                    }
                }
            }
            "shifts-json" => {
                shifts_json = Some(PathBuf::from(match inline {
                    Some(value) => value,
                    None => required_value(args, &mut index, name)?,
                }));
            }
            "out-dir" => {
                out_dir = PathBuf::from(match inline {
                    Some(value) => value,
                    None => required_value(args, &mut index, name)?,
                });
            }
            _ => return Err(format!("unknown argument --{name}")),
        }
        index += 1;
    }
    let positions = parse_positions(&positions)?;
    Ok(Options {
        data_dir: data_dir.ok_or("missing data directory")?,
        positions,
        psf_fwhm,
        mask_modes,
        shifts_json,
        out_dir,
    })
}

pub(crate) fn patch_bounds(
    width: usize,
    height: usize,
    positions: &[[f64; 2]],
    margin: usize,
) -> [usize; 4] {
    let min_x = positions.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
    let max_x = positions
        .iter()
        .map(|p| p[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = positions.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let max_y = positions
        .iter()
        .map(|p| p[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let x0 = ((min_x.floor() as isize) - margin as isize).max(0) as usize;
    let y0 = ((min_y.floor() as isize) - margin as isize).max(0) as usize;
    let x1 = ((max_x.ceil() as isize + margin as isize + 1).max(0) as usize).min(width);
    let y1 = ((max_y.ceil() as isize + margin as isize + 1).max(0) as usize).min(height);
    [x0, y0, x1, y1]
}

pub(crate) fn display_input_path(path: &Path) -> String {
    let value = path.to_string_lossy();
    if path.is_relative() {
        value
            .strip_prefix(".\\")
            .or_else(|| value.strip_prefix("./"))
            .unwrap_or(&value)
            .to_owned()
    } else {
        value.into_owned()
    }
}

pub(crate) fn least_squares(matrix: &DMatrix<f64>, rhs: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if matrix.nrows() != rhs.nrows() || matrix.ncols() == 0 {
        return None;
    }
    let tolerance = f64::EPSILON * matrix.nrows().max(matrix.ncols()) as f64;
    let inverse = matrix
        .clone()
        .svd(true, true)
        .pseudo_inverse(tolerance)
        .ok()?;
    Some(inverse * rhs)
}

fn solve_nnls(matrix: &DMatrix<f64>, data: &DVector<f64>) -> Option<DVector<f64>> {
    let components = matrix.ncols();
    if components == 0 || matrix.nrows() != data.len() {
        return None;
    }
    let mut solution = DVector::zeros(components);
    let mut passive = vec![false; components];
    let tolerance =
        10.0 * f64::EPSILON * matrix.norm().max(1.0) * matrix.nrows().max(components) as f64;
    for _ in 0..(3 * components * components + 1) {
        let residual = data - matrix * &solution;
        let gradient = matrix.transpose() * residual;
        let entering = (0..components)
            .filter(|&i| !passive[i] && gradient[i] > tolerance)
            .max_by(|&a, &b| gradient[a].total_cmp(&gradient[b]));
        let Some(entering) = entering else {
            return Some(solution);
        };
        passive[entering] = true;
        let mut inner_converged = false;
        for _ in 0..(3 * components + 1) {
            let active = (0..components).filter(|&i| passive[i]).collect::<Vec<_>>();
            let active_matrix = DMatrix::from_columns(
                &active
                    .iter()
                    .map(|&column| matrix.column(column).into_owned())
                    .collect::<Vec<_>>(),
            );
            let rhs = DMatrix::from_column_slice(data.len(), 1, data.as_slice());
            let active_solution = least_squares(&active_matrix, &rhs)?;
            let mut trial = DVector::zeros(components);
            for (index, &component) in active.iter().enumerate() {
                trial[component] = active_solution[(index, 0)];
            }
            if active.iter().all(|&component| trial[component] > tolerance) {
                solution = trial;
                inner_converged = true;
                break;
            }
            let alpha = active
                .iter()
                .filter(|&&component| trial[component] <= tolerance)
                .filter_map(|&component| {
                    let denominator = solution[component] - trial[component];
                    (denominator > 0.0).then_some(solution[component] / denominator)
                })
                .fold(f64::INFINITY, f64::min);
            if !alpha.is_finite() {
                passive[entering] = false;
                inner_converged = true;
                break;
            }
            solution += (trial - &solution) * alpha.clamp(0.0, 1.0);
            for &component in &active {
                if solution[component] <= tolerance {
                    solution[component] = 0.0;
                    passive[component] = false;
                }
            }
        }
        if !inner_converged {
            return None;
        }
    }
    None
}

pub(crate) fn fit_fixed(
    frame: &FitsFrame,
    mask: &[bool],
    model_positions: &[[f64; 2]],
    patch_positions: &[[f64; 2]],
    psf_fwhm: f64,
) -> Option<ModelFit> {
    if model_positions.is_empty()
        || patch_positions.is_empty()
        || !psf_fwhm.is_finite()
        || psf_fwhm <= 0.0
        || model_positions
            .iter()
            .chain(patch_positions)
            .flatten()
            .any(|v| !v.is_finite())
    {
        return None;
    }
    let sigma = (psf_fwhm / 2.35482).max(0.5);
    let margin = 4usize.max((3.5 * sigma).ceil() as usize);
    let [x0, y0, x1, y1] = patch_bounds(frame.width, frame.height, patch_positions, margin);
    let (patch_width, patch_height) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    if patch_width < 5 || patch_height < 5 || mask.len() != frame.data.len() {
        return None;
    }
    let mut samples = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let index = y * frame.width + x;
            let value = frame.data.value(index);
            if !mask[index] && value.is_finite() {
                samples.push((x, y, value));
            }
        }
    }
    let sample_count = samples.len();
    let component_count = model_positions.len();
    let parameter_count = 3 + component_count;
    if sample_count < 24.max(parameter_count + 8) {
        return None;
    }
    let center_x = 0.5 * (x0 + x1 - 1) as f64;
    let center_y = 0.5 * (y0 + y1 - 1) as f64;
    let scale = patch_width.max(patch_height).max(1) as f64;
    let mut background = DMatrix::zeros(sample_count, 3);
    let mut source = DMatrix::zeros(sample_count, component_count);
    let mut values = DVector::zeros(sample_count);
    for (row, &(x, y, value)) in samples.iter().enumerate() {
        let gx = x as f64;
        let gy = y as f64;
        background[(row, 0)] = 1.0;
        background[(row, 1)] = (gx - center_x) / scale;
        background[(row, 2)] = (gy - center_y) / scale;
        values[row] = value;
        for (component, position) in model_positions.iter().enumerate() {
            let dx = (gx - position[0]) / sigma;
            let dy = (gy - position[1]) / sigma;
            source[(row, component)] = (-0.5 * (dx * dx + dy * dy)).exp();
        }
    }
    let values_matrix = DMatrix::from_column_slice(sample_count, 1, values.as_slice());
    let background_from_values = least_squares(&background, &values_matrix)?;
    let background_from_source = least_squares(&background, &source)?;
    let source_residual = &source - &background * background_from_source;
    let data_residual = &values - &background * background_from_values.column(0);
    let amplitudes = solve_nnls(&source_residual, &data_residual)?;
    let source_model = &source * &amplitudes;
    let background_rhs = DMatrix::from_column_slice(
        sample_count,
        1,
        (0..sample_count)
            .map(|i| values[i] - source_model[i])
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let background_coefficients = least_squares(&background, &background_rhs)?;
    let residual = values - background * background_coefficients.column(0) - source_model;
    let rss = residual.dot(&residual).max(f64::EPSILON);
    let bic = sample_count as f64 * (rss / sample_count as f64).ln()
        + parameter_count as f64 * (sample_count as f64).ln();
    let degrees_of_freedom = sample_count.saturating_sub(parameter_count).max(1) as f64;
    let variance = rss / degrees_of_freedom;
    let active_tolerance = 1e-12f64.max(f64::EPSILON * amplitudes.amax().max(1.0));
    let active = (0..component_count)
        .filter(|&i| amplitudes[i] > active_tolerance)
        .collect::<Vec<_>>();
    let mut component_snr = vec![0.0; component_count];
    if !active.is_empty() {
        let active_design = DMatrix::from_columns(
            &active
                .iter()
                .map(|&column| source_residual.column(column).into_owned())
                .collect::<Vec<_>>(),
        );
        let information = active_design.transpose() * &active_design;
        let tolerance = f64::EPSILON * information.nrows().max(information.ncols()) as f64;
        if let Ok(inverse) = information
            .clone()
            .svd(true, true)
            .pseudo_inverse(tolerance)
        {
            for (index, &component) in active.iter().enumerate() {
                let standard_error = (variance * inverse[(index, index)])
                    .max(f64::EPSILON)
                    .sqrt();
                component_snr[component] = amplitudes[component] / standard_error;
            }
        }
    }
    let gram = source_residual.transpose() * &source_residual;
    let max_template_correlation = if component_count >= 2 {
        let mut maximum: f64 = 0.0;
        for left in 0..component_count {
            for right in left + 1..component_count {
                let norm = (gram[(left, left)].max(f64::EPSILON)
                    * gram[(right, right)].max(f64::EPSILON))
                .sqrt();
                maximum = maximum.max((gram[(left, right)] / norm).abs());
            }
        }
        Some(maximum)
    } else {
        None
    };
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
    Some(ModelFit {
        sample_count,
        masked_pixel_count: patch_width * patch_height - sample_count,
        rss,
        bic,
        amplitudes: amplitudes.iter().copied().collect(),
        component_snr,
        active_component_count: active.len(),
        max_template_correlation,
        design_condition_number,
    })
}

pub(crate) fn parse_shifts(path: &Path) -> Result<Vec<[f64; 2]>> {
    let payload: Value = serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("cannot read shifts JSON: {error}"))?,
    )
    .map_err(|error| format!("invalid shifts JSON: {error}"))?;
    let raw = payload
        .get("cumulative_shifts")
        .and_then(Value::as_array)
        .ok_or("shifts JSON is missing cumulative_shifts")?;
    raw.iter()
        .map(|entry| {
            let values = entry
                .as_array()
                .filter(|values| values.len() == 2)
                .ok_or("each cumulative shift must contain x and y")?;
            let shift = [
                values[0].as_f64().ok_or("invalid x frame shift")?,
                values[1].as_f64().ok_or("invalid y frame shift")?,
            ];
            if shift.iter().any(|value| !value.is_finite()) {
                return Err("frame shifts must be finite".into());
            }
            Ok(shift)
        })
        .collect()
}

pub fn run_audit(
    paths: &[PathBuf],
    positions: &[[f64; 2]],
    psf_fwhm: f64,
    mask_modes: &[String],
    frame_shifts: Option<&[[f64; 2]]>,
) -> Result<LocalMultiPSFAuditResult> {
    if paths.is_empty() {
        return Err("paths cannot be empty".into());
    }
    if positions.len() < 2 {
        return Err("target_positions must contain at least two positions".into());
    }
    if positions.iter().flatten().any(|value| !value.is_finite()) {
        return Err("target_positions must contain finite coordinates".into());
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
        if shifts.len() != paths.len() {
            return Err("frame_shifts length must match paths".into());
        }
        if shifts.iter().flatten().any(|value| !value.is_finite()) {
            return Err("frame_shifts must contain finite values".into());
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
                    fit_fixed(
                        &frame,
                        &mask,
                        &frame_positions[..count],
                        &frame_positions,
                        psf_fwhm,
                    )
                })
                .collect::<Vec<_>>();
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
                let added_snr = fit
                    .as_ref()
                    .filter(|_| component_count >= 2)
                    .map(|fit| &fit.component_snr[1..]);
                let confirmation = component_count >= 2
                    && delta_single.is_some_and(|delta| delta >= 10.0)
                    && added_snr.is_some_and(|snr| {
                        !snr.is_empty() && snr.iter().all(|value| *value >= 5.0)
                    });
                let note = if fit.is_none() {
                    "局部有效像素不足或线性模型数值不稳定"
                } else if confirmation {
                    "达到当前工程双/多源证据线；仍需实测 PSF、注入和星表核验"
                } else {
                    "局部模型比较；不等于物理恒星身份"
                };
                rows.push(LocalMultiPSFAuditRow {
                    frame_index: frame_index + 1,
                    path: display_input_path(&frame.path),
                    mask_mode: mask_mode.clone(),
                    component_count,
                    target_positions: frame_positions[..component_count].to_vec(),
                    sample_count: fit.as_ref().map_or(0, |fit| fit.sample_count),
                    masked_pixel_count: fit.as_ref().map_or(0, |fit| fit.masked_pixel_count),
                    rss: fit.as_ref().map(|fit| fit.rss),
                    bic,
                    delta_bic_from_single: delta_single,
                    delta_bic_from_previous: delta_previous,
                    amplitudes: fit
                        .as_ref()
                        .map_or_else(Vec::new, |fit| fit.amplitudes.clone()),
                    component_snr: fit
                        .as_ref()
                        .map_or_else(Vec::new, |fit| fit.component_snr.clone()),
                    active_component_count: fit
                        .as_ref()
                        .map_or(0, |fit| fit.active_component_count),
                    max_template_correlation: fit
                        .as_ref()
                        .and_then(|fit| fit.max_template_correlation),
                    design_condition_number: fit
                        .as_ref()
                        .and_then(|fit| fit.design_condition_number),
                    confirmation_line: confirmation,
                    note: note.into(),
                });
            }
        }
    }
    let raw_rows = rows
        .iter()
        .filter(|row| row.mask_mode == "raw")
        .collect::<Vec<_>>();
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
        candidates.sort_by(|a, b| a.bic.unwrap().total_cmp(&b.bic.unwrap()));
        let best = candidates[0];
        best_by_frame.push(format!(
            "F{frame_index:02}: BIC 最优 K={}{}",
            best.component_count,
            best.delta_bic_from_single
                .map(|delta| format!(" (ΔBIC={delta:.2})"))
                .unwrap_or_default()
        ));
    }
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
    let conclusion = format!(
        "固定位置 K 源 PSF 审计完成：{} 帧、{} 个候选位置，当前证据线下 raw 掩膜确认统计为 {}。K 源模型只说明局部像素对某个受限模型的相对支持；它不能替代自由位置、空间变 PSF、完整噪声似然、人工注入或 WCS/星表身份。 {}",
        paths.len(),
        positions.len(),
        confirmations,
        if best_by_frame.is_empty() {
            String::new()
        } else {
            format!("首帧/各帧 raw 最优模型：{}。", best_by_frame.join("; "))
        }
    );
    Ok(LocalMultiPSFAuditResult {
        frame_count: paths.len(),
        target_positions: positions.to_vec(),
        frame_shifts: shifts.clone(),
        mask_modes: mask_modes.to_vec(),
        psf_fwhm_px: psf_fwhm,
        parameters: json!({
            "psf_fwhm_px": psf_fwhm,
            "mask_modes": mask_modes,
            "component_counts": (1..=positions.len()).collect::<Vec<_>>(),
            "confirmation_delta_bic": 10.0,
            "confirmation_component_snr": 5.0,
            "frame_shifts": shifts,
            "model": "shared linear background + fixed-position Gaussian PSF + nonnegative amplitudes",
            "interpretation_boundary": "local model evidence, not WCS/catalog identity"
        }),
        rows,
        conclusion,
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

fn draw_line(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    a: [usize; 2],
    b: [usize; 2],
    color: [u8; 3],
) {
    let (mut x, mut y) = (a[0] as isize, a[1] as isize);
    let dx = (b[0] as isize - x).abs();
    let sx = if x < b[0] as isize { 1 } else { -1 };
    let dy = -((b[1] as isize - y).abs());
    let sy = if y < b[1] as isize { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
            let index = (y as usize * width + x as usize) * 3;
            pixels[index..index + 3].copy_from_slice(&color);
        }
        if x == b[0] as isize && y == b[1] as isize {
            break;
        }
        let doubled = 2 * error;
        if doubled >= dy {
            error += dy;
            x += sx;
        }
        if doubled <= dx {
            error += dx;
            y += sy;
        }
    }
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

pub(crate) fn write_chart(path: &Path, result: &LocalMultiPSFAuditResult) -> Result<()> {
    let (width, height) = (1200usize, 680usize);
    let plot = [70usize, 40usize, width - 35, height - 60];
    let mut series = Vec::new();
    let mut all_values = vec![10.0];
    let colors = [
        ("raw", [215, 148, 50]),
        ("range_masked", [79, 155, 131]),
        ("sentinel_masked", [77, 145, 184]),
        ("range_sentinel_masked", [199, 53, 158]),
    ];
    for mode in &result.mask_modes {
        let color = colors
            .iter()
            .find(|(name, _)| *name == mode)
            .map_or([98, 107, 125], |(_, color)| *color);
        let mut values = Vec::new();
        for count in 1..=result.target_positions.len() {
            let mut entries = result
                .rows
                .iter()
                .filter(|row| row.mask_mode == *mode && row.component_count == count)
                .filter_map(|row| row.delta_bic_from_single)
                .collect::<Vec<_>>();
            values.push(median(&mut entries));
        }
        all_values.extend(values.iter().flatten().copied());
        series.push((color, values));
    }
    let low = all_values.iter().copied().fold(0.0f64, f64::min);
    let high = all_values.iter().copied().fold(10.0f64, f64::max);
    let span = (high - low).max(1.0);
    let low = low - 0.08 * span;
    let high = high + 0.08 * span;
    let mut pixels = vec![255u8; width * height * 3];
    for index in 0..=5 {
        let y = plot[1] + (plot[3] - plot[1]) * index / 5;
        draw_line(
            &mut pixels,
            width,
            height,
            [plot[0], y],
            [plot[2], y],
            [224, 230, 236],
        );
    }
    draw_line(
        &mut pixels,
        width,
        height,
        [plot[0], plot[1]],
        [plot[0], plot[3]],
        [70, 83, 96],
    );
    draw_line(
        &mut pixels,
        width,
        height,
        [plot[0], plot[3]],
        [plot[2], plot[3]],
        [70, 83, 96],
    );
    let threshold_y = plot[3]
        - (((10.0 - low) / (high - low)).clamp(0.0, 1.0) * (plot[3] - plot[1]) as f64).round()
            as usize;
    draw_line(
        &mut pixels,
        width,
        height,
        [plot[0], threshold_y],
        [plot[2], threshold_y],
        [32, 41, 61],
    );
    for (color, values) in series {
        let points = values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                value.map(|value| {
                    let x = if values.len() <= 1 {
                        (plot[0] + plot[2]) / 2
                    } else {
                        plot[0] + (plot[2] - plot[0]) * index / (values.len() - 1)
                    };
                    let y = plot[3]
                        - (((value - low) / (high - low)).clamp(0.0, 1.0)
                            * (plot[3] - plot[1]) as f64)
                            .round() as usize;
                    [x, y]
                })
            })
            .collect::<Vec<_>>();
        for pair in points.windows(2) {
            draw_line(&mut pixels, width, height, pair[0], pair[1], color);
        }
        for point in points {
            draw_line(&mut pixels, width, height, point, point, color);
        }
    }
    let mut encoder = png::Encoder::new(
        File::create(path).map_err(|error| error.to_string())?,
        width as u32,
        height as u32,
    );
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .map_err(|error| error.to_string())?
        .write_image_data(&pixels)
        .map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &LocalMultiPSFAuditResult, output: &Path) -> Result<Value> {
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    let csv_path = output.join("local_multipsf_audit.csv");
    let mut file = File::create(&csv_path).map_err(|error| error.to_string())?;
    file.write_all(b"\xef\xbb\xbf")
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);
    writer
        .write_record(CSV_FIELDS)
        .map_err(|error| error.to_string())?;
    for row in &result.rows {
        let value = serde_json::to_value(row).map_err(|error| error.to_string())?;
        let object = value.as_object().ok_or("row serialization failed")?;
        writer
            .write_record(
                CSV_FIELDS
                    .iter()
                    .map(|field| object.get(*field).map(csv_value).unwrap_or_default()),
            )
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    fs::write(
        output.join("local_multipsf_audit.json"),
        serde_json::to_vec_pretty(&result.as_json()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    write_chart(&output.join("local_multipsf_bic.png"), result)?;
    Ok(json!({"output":output,"frame_count":result.frame_count,"conclusion":result.conclusion}))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let options = parse_args(args)?;
    let paths = empirical_pair_fit::ordered_fits_paths(&options.data_dir)?;
    let shifts = options
        .shifts_json
        .as_deref()
        .map(parse_shifts)
        .transpose()?;
    let result = run_audit(
        &paths,
        &options.positions,
        options.psf_fwhm,
        &options.mask_modes,
        shifts.as_deref(),
    )?;
    write_artifacts(&result, &options.out_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AuxiliaryData, PixelData};

    const REFERENCE_JSON: &str =
        include_str!("../tests/fixtures/multipsf_audit_parity/reference-output.json");
    const REFERENCE_CSV: &[u8] =
        include_bytes!("../tests/fixtures/multipsf_audit_parity/reference-output.csv");

    fn assert_json_parity(actual: &Value, expected: &Value, location: &str) {
        match (actual, expected) {
            (Value::Object(actual), Value::Object(expected)) => {
                assert_eq!(actual.len(), expected.len(), "object keys at {location}");
                for (key, expected_value) in expected {
                    if key == "path" {
                        continue;
                    }
                    let actual_value = actual
                        .get(key)
                        .unwrap_or_else(|| panic!("missing key {key} at {location}"));
                    assert_json_parity(actual_value, expected_value, &format!("{location}.{key}"));
                }
            }
            (Value::Array(actual), Value::Array(expected)) => {
                assert_eq!(actual.len(), expected.len(), "array length at {location}");
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert_json_parity(actual, expected, &format!("{location}[{index}]"));
                }
            }
            (Value::Number(actual), Value::Number(expected)) => {
                if let (Some(actual), Some(expected)) = (actual.as_f64(), expected.as_f64()) {
                    let tolerance = 1e-8 + 1e-10 * expected.abs();
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "numeric mismatch at {location}: {actual} vs {expected} (tol {tolerance})"
                    );
                } else {
                    assert_eq!(actual, expected, "number at {location}");
                }
            }
            _ => assert_eq!(actual, expected, "value at {location}"),
        }
    }

    fn parse_csv(bytes: &[u8]) -> Vec<csv::StringRecord> {
        let text = String::from_utf8_lossy(bytes);
        let text = text.trim_start_matches('\u{feff}');
        csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(text.as_bytes())
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap()
    }

    fn synthetic_frame() -> FitsFrame {
        let (width, height) = (96usize, 96usize);
        let mut image = vec![21.0f32; width * height];
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
            path: PathBuf::from("synthetic-multipsf.fits"),
            header: Default::default(),
            data: PixelData::F32(image),
            auxiliary: Some(AuxiliaryData { values: [0.0; 26] }),
            data_offset: 0,
            width,
            height,
        }
    }

    #[test]
    fn parser_preserves_defaults_and_reads_multi_position_arguments() {
        let defaults = parse_args(&["frames".into()]).unwrap();
        assert_eq!(defaults.psf_fwhm, 2.0);
        assert_eq!(
            defaults.mask_modes,
            ["raw", "range_masked", "sentinel_masked"]
        );
        assert_eq!(defaults.positions.len(), 3);
        let parsed = parse_args(&[
            "frames".into(),
            "--positions".into(),
            "10,12;14,16".into(),
            "--mask-modes".into(),
            "raw".into(),
            "range_masked".into(),
        ])
        .unwrap();
        assert_eq!(parsed.positions, [[10.0, 12.0], [14.0, 16.0]]);
        assert_eq!(parsed.mask_modes, ["raw", "range_masked"]);
    }

    #[test]
    fn fixed_multi_psf_fit_recovers_two_injected_components() {
        let frame = synthetic_frame();
        let mask = vec![false; frame.data.len()];
        let positions = [[48.35, 48.25], [53.15, 50.20]];
        let single = fit_fixed(&frame, &mask, &positions[..1], &positions, 2.0).unwrap();
        let pair = fit_fixed(&frame, &mask, &positions, &positions, 2.0).unwrap();
        assert!(pair.bic < single.bic);
        assert_eq!(pair.active_component_count, 2);
        assert!(pair.amplitudes.iter().all(|amplitude| *amplitude > 0.0));
        assert!(pair.component_snr[1] > 5.0);
        assert!(pair.design_condition_number.unwrap().is_finite());
    }

    #[test]
    fn nnls_handles_three_components_and_inactive_sources() {
        let matrix = DMatrix::from_row_slice(
            5,
            3,
            &[
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0,
            ],
        );
        let data = DVector::from_vec(vec![3.0, 0.0, 2.0, 5.0, 8.0]);
        let solution = solve_nnls(&matrix, &data).unwrap();
        assert!((solution[0] - 3.0).abs() < 1e-8);
        assert!(solution[1].abs() < 1e-8);
        assert!((solution[2] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn fixed_multipsf_output_matches_frozen_legacy_reference_without_python() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/empirical_pair_audit_parity/frames");
        let paths = empirical_pair_fit::ordered_fits_paths(&fixture_dir).unwrap();
        let positions = [[46.0, 48.0], [48.5, 48.0], [50.0, 45.0]];
        let result = run_audit(&paths, &positions, 2.0, &["raw".into()], None).unwrap();
        let output = std::env::temp_dir().join(format!(
            "rst19-multipsf-parity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_artifacts(&result, &output).unwrap();

        let expected: Value = serde_json::from_str(REFERENCE_JSON).unwrap();
        let actual: Value =
            serde_json::from_slice(&fs::read(output.join("local_multipsf_audit.json")).unwrap())
                .unwrap();
        assert_json_parity(&actual, &expected, "root");

        let actual_csv = parse_csv(&fs::read(output.join("local_multipsf_audit.csv")).unwrap());
        let expected_csv = parse_csv(REFERENCE_CSV);
        assert_eq!(actual_csv.len(), expected_csv.len(), "CSV row count");
        assert_eq!(actual_csv[0], expected_csv[0], "CSV header");
        for (row_index, (actual_row, expected_row)) in
            actual_csv.iter().zip(&expected_csv).enumerate().skip(1)
        {
            for (column_index, (actual, expected)) in
                actual_row.iter().zip(expected_row).enumerate()
            {
                let column = &actual_csv[0][column_index];
                if column == "path" {
                    continue;
                }
                let location = format!("CSV row {row_index}, column {column}");
                if ["target_positions", "amplitudes", "component_snr"].contains(&column) {
                    let actual: Value = serde_json::from_str(actual).unwrap();
                    let expected: Value = serde_json::from_str(expected).unwrap();
                    assert_json_parity(&actual, &expected, &location);
                } else if let (Ok(actual_number), Ok(expected_number)) =
                    (actual.parse::<f64>(), expected.parse::<f64>())
                {
                    let tolerance = 1e-8 + 1e-10 * expected_number.abs();
                    assert!(
                        (actual_number - expected_number).abs() <= tolerance,
                        "numeric mismatch at {location}: {actual_number} vs {expected_number}"
                    );
                } else {
                    assert_eq!(actual, expected, "{location}");
                }
            }
        }

        let png = fs::read(output.join("local_multipsf_bic.png")).unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), 1200);
        assert_eq!(u32::from_be_bytes(png[20..24].try_into().unwrap()), 680);
        let _ = fs::remove_dir_all(output);
    }
}
