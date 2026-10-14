//! Fixed empirical-PSF single/double fits with a best-single-position control.
//!
//! The BIC and component SNR values are local model diagnostics. They do not
//! establish catalog identity, physical multiplicity, or stellar probability.

use crate::{
    empirical_psf::{self, EmpiricalPsf},
    feature_psf_leaveout, fits,
    measurement::Detection,
    models::{FitsFrame, NumericPlane, PixelData},
    Result,
};
use nalgebra::{Matrix2, Matrix3, Vector3};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

const MASK_MODES: [&str; 8] = [
    "raw",
    "range_masked",
    "sentinel_masked",
    "range_sentinel_masked",
    "repeated_code_masked",
    "range_repeated_code_masked",
    "sentinel_repeated_code_masked",
    "range_sentinel_repeated_code_masked",
];

const CSV_FIELDS: &[&str] = &[
    "frame_index",
    "path",
    "coordinate_mode",
    "mask_mode",
    "patch_x0",
    "patch_y0",
    "patch_width",
    "patch_height",
    "sample_count",
    "masked_pixel_count",
    "fixed_single_rss",
    "fixed_double_rss",
    "fixed_single_bic",
    "fixed_double_bic",
    "fixed_delta_bic",
    "best_single_grid_rss",
    "best_single_grid_bic",
    "best_single_grid_x",
    "best_single_grid_y",
    "best_single_grid_evaluated_count",
    "best_single_grid_delta_bic",
    "double_amplitudes",
    "double_component_snr",
    "double_second_component_snr",
    "double_amplitude_correlation",
    "double_design_condition_number",
    "double_active_component_count",
    "fixed_confirmation_line",
    "best_single_grid_control_line",
    "note",
];

pub const HELP: &str = "用经验 PSF 比较近邻候选的固定双源模型与最佳单源位置控制\n\nUsage: rst19-empirical-pair-fit DATA_DIR --source-catalog CSV --primary-id ID --secondary-id ID [OPTIONS]\n\nOptions: --coordinate-mode centroid|peak --support-radius N --max-template-sources N\n         --patch-padding-px N --best-single-search-radius-px N --best-single-grid-step-px N\n         --mask-modes LIST --repeated-code-values LIST --shifts-json JSON --out-dir DIR";

#[derive(Clone, Debug, Serialize)]
pub struct EmpiricalPairFitRow {
    pub frame_index: usize,
    pub path: String,
    pub coordinate_mode: String,
    pub mask_mode: String,
    pub patch_x0: usize,
    pub patch_y0: usize,
    pub patch_width: usize,
    pub patch_height: usize,
    pub sample_count: usize,
    pub masked_pixel_count: usize,
    pub fixed_single_rss: Option<f64>,
    pub fixed_double_rss: Option<f64>,
    pub fixed_single_bic: Option<f64>,
    pub fixed_double_bic: Option<f64>,
    pub fixed_delta_bic: Option<f64>,
    pub best_single_grid_rss: Option<f64>,
    pub best_single_grid_bic: Option<f64>,
    pub best_single_grid_x: Option<f64>,
    pub best_single_grid_y: Option<f64>,
    pub best_single_grid_evaluated_count: usize,
    pub best_single_grid_delta_bic: Option<f64>,
    pub double_amplitudes: Vec<f64>,
    pub double_component_snr: Vec<f64>,
    pub double_second_component_snr: Option<f64>,
    pub double_amplitude_correlation: Option<f64>,
    pub double_design_condition_number: Option<f64>,
    pub double_active_component_count: usize,
    pub fixed_confirmation_line: bool,
    pub best_single_grid_control_line: bool,
    pub note: String,
}

#[derive(Clone, Debug)]
pub struct EmpiricalPairFitResult {
    pub frame_count: usize,
    pub coordinate_mode: String,
    pub primary_detection_id: Option<i64>,
    pub secondary_detection_id: Option<i64>,
    pub primary_target_xy: [f64; 2],
    pub secondary_target_xy: [f64; 2],
    pub frame_shifts: Vec<[f64; 2]>,
    pub template_frame_path: String,
    pub template_catalog_path: Option<String>,
    pub mask_modes: Vec<String>,
    pub support_radius: usize,
    pub max_template_sources: usize,
    pub patch_padding_px: f64,
    pub best_single_search_radius_px: f64,
    pub best_single_grid_step_px: f64,
    pub confirmation_delta_bic: f64,
    pub confirmation_component_snr: f64,
    pub empirical_psf: EmpiricalPsf,
    pub parameters: Value,
    pub rows: Vec<EmpiricalPairFitRow>,
    pub conclusion: String,
}

impl EmpiricalPairFitResult {
    fn as_json(&self) -> Value {
        json!({
            "frame_count": self.frame_count,
            "coordinate_mode": self.coordinate_mode,
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
            "best_single_search_radius_px": self.best_single_search_radius_px,
            "best_single_grid_step_px": self.best_single_grid_step_px,
            "confirmation_delta_bic": self.confirmation_delta_bic,
            "confirmation_component_snr": self.confirmation_component_snr,
            "empirical_psf": self.empirical_psf.as_dict(),
            "parameters": self.parameters,
            "rows": self.rows,
            "conclusion": self.conclusion,
            "note": "固定位置 K=1/K=2 使用相同局部窗口和掩膜；最佳单源网格为位置锚定控制，不能直接当作自由位置模型的正式 BIC。经验 PSF 只表示模板帧的形状样本，所有结果均不是星表身份或物理真值。"
        })
    }
}

#[derive(Clone, Debug)]
struct Options {
    data_dir: PathBuf,
    source_catalog: PathBuf,
    primary_id: i64,
    secondary_id: i64,
    coordinate_mode: String,
    support_radius: usize,
    max_template_sources: usize,
    patch_padding_px: Option<f64>,
    best_single_search_radius_px: f64,
    best_single_grid_step_px: f64,
    mask_modes: Vec<String>,
    repeated_code_values: Option<Vec<i64>>,
    shifts_json: Option<PathBuf>,
    out_dir: PathBuf,
}

fn required_value(
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
    let mut data_dir = None;
    let mut source_catalog = None;
    let mut primary_id = None;
    let mut secondary_id = None;
    let mut coordinate_mode = "centroid".to_owned();
    let mut support_radius = 7usize;
    let mut max_template_sources = 64usize;
    let mut patch_padding_px = None;
    let mut best_single_search_radius_px = 2.0;
    let mut best_single_grid_step_px = 0.5;
    let mut mask_modes = "raw,range_masked,sentinel_masked".to_owned();
    let mut repeated_code_values = None;
    let mut shifts_json = None;
    let mut out_dir = PathBuf::from("tmp/empirical-pair-fit-audit");
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
        let (name, inline) = arg[2..]
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((&arg[2..], None));
        let value = required_value(args, &mut index, name, inline.as_deref())?;
        match name {
            "source-catalog" => source_catalog = Some(PathBuf::from(value)),
            "primary-id" => primary_id = Some(value.parse().map_err(|_| "invalid --primary-id")?),
            "secondary-id" => {
                secondary_id = Some(value.parse().map_err(|_| "invalid --secondary-id")?)
            }
            "coordinate-mode" => coordinate_mode = value,
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
            "best-single-search-radius-px" => {
                best_single_search_radius_px = value
                    .parse()
                    .map_err(|_| "invalid --best-single-search-radius-px")?
            }
            "best-single-grid-step-px" => {
                best_single_grid_step_px = value
                    .parse()
                    .map_err(|_| "invalid --best-single-grid-step-px")?
            }
            "mask-modes" => mask_modes = value,
            "repeated-code-values" => repeated_code_values = Some(parse_codes(&value)?),
            "shifts-json" => shifts_json = Some(PathBuf::from(value)),
            "out-dir" => out_dir = PathBuf::from(value),
            "help" => return Err(HELP.into()),
            _ => return Err(format!("unknown argument --{name}")),
        }
        index += 1;
    }
    let data_dir = data_dir.ok_or("missing data directory")?;
    let source_catalog = source_catalog.ok_or("missing --source-catalog")?;
    let primary_id = primary_id.ok_or("missing --primary-id")?;
    let secondary_id = secondary_id.ok_or("missing --secondary-id")?;
    let mask_modes = mask_modes
        .split(',')
        .map(str::trim)
        .filter(|mode| !mode.is_empty())
        .map(str::to_owned)
        .collect();
    Ok(Options {
        data_dir,
        source_catalog,
        primary_id,
        secondary_id,
        coordinate_mode,
        support_radius,
        max_template_sources,
        patch_padding_px,
        best_single_search_radius_px,
        best_single_grid_step_px,
        mask_modes,
        repeated_code_values,
        shifts_json,
        out_dir,
    })
}

fn ordered_fits_paths(data_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(data_dir)
        .map_err(|error| format!("cannot read data directory {}: {error}", data_dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "fits"))
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "data directory has no FITS files: {}",
            data_dir.display()
        ));
    }
    Ok(paths)
}

fn numeric_plane<T>(
    frame: &FitsFrame,
    operation: impl FnOnce(NumericPlane<'_>) -> Result<T>,
) -> Result<T> {
    match &frame.data {
        PixelData::F32(values) => operation(values.as_slice().into()),
        PixelData::F64(values) => operation(values.as_slice().into()),
        _ => {
            let values = frame.data.to_f64();
            operation(values.as_slice().into())
        }
    }
}

fn estimate_psf(
    frame: &FitsFrame,
    sources: &[Detection],
    radius: usize,
    max_sources: usize,
) -> Result<Option<EmpiricalPsf>> {
    numeric_plane(frame, |plane| {
        empirical_psf::estimate(
            plane,
            frame.height,
            frame.width,
            sources,
            Some(sources),
            radius,
            max_sources,
            &[],
        )
    })
}

fn negative_overflow_limit(data: &PixelData) -> Option<f64> {
    match data {
        PixelData::I16(_) => Some(-0.9 * f64::from(i16::MAX)),
        PixelData::I32(_) => Some(-0.9 * f64::from(i32::MAX)),
        PixelData::I64(_) => Some(-0.9 * i64::MAX as f64),
        _ => None,
    }
}

fn resolve_repeated_codes(sources: &[Detection], explicit: Option<&[i64]>) -> Vec<i64> {
    let mut codes = explicit.map_or_else(
        || {
            sources
                .iter()
                .flat_map(|source| source.repeated_code_values.iter().copied())
                .collect::<Vec<_>>()
        },
        |values| values.to_vec(),
    );
    codes.sort_unstable();
    codes.dedup();
    codes
}

fn mask_modes(
    frame: &FitsFrame,
    modes: &[String],
    codes: &[i64],
) -> Result<Vec<(String, Vec<bool>)>> {
    let base = fits::auxiliary_mask(frame.height, frame.width)?;
    let overflow = negative_overflow_limit(&frame.data);
    let mut output = Vec::with_capacity(modes.len());
    for mode in modes {
        let mut mask = base.clone();
        if matches!(
            mode.as_str(),
            "range_masked"
                | "range_sentinel_masked"
                | "range_repeated_code_masked"
                | "range_sentinel_repeated_code_masked"
        ) {
            if let Some(limit) = overflow {
                for (index, value) in mask.iter_mut().enumerate() {
                    let pixel = frame.data.value(index);
                    *value |= pixel <= limit || pixel >= -limit;
                }
            }
        }
        if matches!(
            mode.as_str(),
            "sentinel_masked"
                | "range_sentinel_masked"
                | "sentinel_repeated_code_masked"
                | "range_sentinel_repeated_code_masked"
        ) {
            for (index, value) in mask.iter_mut().enumerate() {
                *value |= frame.data.value(index) == -1.0;
            }
        }
        if mode.contains("repeated_code") && !codes.is_empty() {
            for (index, value) in mask.iter_mut().enumerate() {
                let pixel = frame.data.value(index);
                *value |= codes.iter().any(|code| pixel == *code as f64);
            }
        }
        output.push((mode.clone(), mask));
    }
    Ok(output)
}

fn patch_bounds(width: usize, height: usize, positions: &[[f64; 2]], padding: f64) -> [usize; 4] {
    let x0 = positions
        .iter()
        .map(|point| point[0])
        .fold(f64::INFINITY, f64::min);
    let x1 = positions
        .iter()
        .map(|point| point[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let y0 = positions
        .iter()
        .map(|point| point[1])
        .fold(f64::INFINITY, f64::min);
    let y1 = positions
        .iter()
        .map(|point| point[1])
        .fold(f64::NEG_INFINITY, f64::max);
    [
        ((x0 - padding).floor() as isize).max(0) as usize,
        ((y0 - padding).floor() as isize).max(0) as usize,
        ((x1 + padding).ceil() as usize + 1).min(width),
        ((y1 + padding).ceil() as usize + 1).min(height),
    ]
}

fn shifted_basis(
    patch_width: usize,
    patch_height: usize,
    patch_origin: [usize; 2],
    position: [f64; 2],
    psf: &EmpiricalPsf,
) -> Result<Vec<f64>> {
    let radius = psf.support_radius;
    let side = psf.side();
    if psf.kernel.len() != side * side {
        return Err("empirical PSF shape does not match support radius".into());
    }
    let center_x = position[0].round_ties_even() as isize;
    let center_y = position[1].round_ties_even() as isize;
    let shift_x = center_x as f64 - position[0];
    let shift_y = center_y as f64 - position[1];
    let mut shifted = vec![0.0; side * side];
    for y in 0..side {
        for x in 0..side {
            let source_y = y as f64 - shift_y;
            let source_x = x as f64 - shift_x;
            if source_y < 0.0
                || source_x < 0.0
                || source_y > (side - 1) as f64
                || source_x > (side - 1) as f64
            {
                continue;
            }
            let y0 = source_y.floor() as usize;
            let x0 = source_x.floor() as usize;
            let y1 = (y0 + 1).min(side - 1);
            let x1 = (x0 + 1).min(side - 1);
            let dy = source_y - y0 as f64;
            let dx = source_x - x0 as f64;
            let top = f64::from(psf.kernel[y0 * side + x0]) * (1.0 - dx)
                + f64::from(psf.kernel[y0 * side + x1]) * dx;
            let bottom = f64::from(psf.kernel[y1 * side + x0]) * (1.0 - dx)
                + f64::from(psf.kernel[y1 * side + x1]) * dx;
            shifted[y * side + x] = top * (1.0 - dy) + bottom * dy;
        }
    }
    let mut basis = vec![0.0; patch_width * patch_height];
    for ky in 0..side {
        for kx in 0..side {
            let gx = center_x - radius as isize + kx as isize;
            let gy = center_y - radius as isize + ky as isize;
            let lx = gx - patch_origin[0] as isize;
            let ly = gy - patch_origin[1] as isize;
            if lx >= 0 && ly >= 0 && (lx as usize) < patch_width && (ly as usize) < patch_height {
                basis[ly as usize * patch_width + lx as usize] = shifted[ky * side + kx];
            }
        }
    }
    Ok(basis)
}

#[derive(Clone)]
struct Sample {
    value: f64,
    background: Vector3<f64>,
    sources: Vec<f64>,
}

#[derive(Clone, Debug)]
struct ModelFit {
    sample_count: usize,
    masked_pixel_count: usize,
    rss: f64,
    bic: f64,
    amplitudes: Vec<f64>,
    component_snr: Vec<f64>,
    active_component_count: usize,
    amplitude_correlation: Option<f64>,
    design_condition_number: Option<f64>,
}

fn vector_dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn solve_nnls(source_residual: &[Vec<f64>], data_residual: &[f64]) -> Vec<f64> {
    let count = source_residual.len();
    if count == 1 {
        let column = &source_residual[0];
        let denominator = vector_dot(column, column);
        let amplitude = if denominator > 0.0 {
            (vector_dot(column, data_residual) / denominator).max(0.0)
        } else {
            0.0
        };
        return vec![amplitude];
    }
    let s0 = &source_residual[0];
    let s1 = &source_residual[1];
    let h00 = vector_dot(s0, s0);
    let h11 = vector_dot(s1, s1);
    let h01 = vector_dot(s0, s1);
    let g0 = vector_dot(s0, data_residual);
    let g1 = vector_dot(s1, data_residual);
    let mut best = [0.0, 0.0];
    let base_rss = vector_dot(data_residual, data_residual);
    let mut best_rss = base_rss;
    for (index, (h, g)) in [(h00, g0), (h11, g1)].into_iter().enumerate() {
        if h <= 0.0 {
            continue;
        }
        let amplitude = (g / h).max(0.0);
        let rss = base_rss - 2.0 * amplitude * g + amplitude * amplitude * h;
        if rss < best_rss {
            best_rss = rss;
            best = if index == 0 {
                [amplitude, 0.0]
            } else {
                [0.0, amplitude]
            };
        }
    }
    let determinant = h00 * h11 - h01 * h01;
    if determinant > f64::EPSILON * h00.max(h11).max(1.0) {
        let a0 = (g0 * h11 - g1 * h01) / determinant;
        let a1 = (g1 * h00 - g0 * h01) / determinant;
        if a0 > 0.0 && a1 > 0.0 {
            let rss = vector_dot(data_residual, data_residual) - 2.0 * (a0 * g0 + a1 * g1)
                + a0 * a0 * h00
                + 2.0 * a0 * a1 * h01
                + a1 * a1 * h11;
            if rss < best_rss {
                best = [a0, a1];
            }
        }
    }
    best.to_vec()
}

fn pseudoinverse_2(matrix: Matrix2<f64>) -> Matrix2<f64> {
    let eigen = matrix.symmetric_eigen();
    let largest = eigen.eigenvalues[0].max(eigen.eigenvalues[1]).max(0.0);
    if largest == 0.0 {
        return Matrix2::zeros();
    }
    let inverse = Matrix2::new(
        if eigen.eigenvalues[0] > largest * 1e-15 {
            1.0 / eigen.eigenvalues[0]
        } else {
            0.0
        },
        0.0,
        0.0,
        if eigen.eigenvalues[1] > largest * 1e-15 {
            1.0 / eigen.eigenvalues[1]
        } else {
            0.0
        },
    );
    eigen.eigenvectors * inverse * eigen.eigenvectors.transpose()
}

fn fit_fixed(
    frame: &FitsFrame,
    mask: &[bool],
    positions: &[[f64; 2]],
    bounds: [usize; 4],
    psf: &EmpiricalPsf,
) -> Result<Option<ModelFit>> {
    let [x0, y0, x1, y1] = bounds;
    let patch_width = x1.saturating_sub(x0);
    let patch_height = y1.saturating_sub(y0);
    let parameter_count = 3 + positions.len();
    let bases = positions
        .iter()
        .map(|position| shifted_basis(patch_width, patch_height, [x0, y0], *position, psf))
        .collect::<Result<Vec<_>>>()?;
    let center_x = 0.5 * (x0 + x1 - 1) as f64;
    let center_y = 0.5 * (y0 + y1 - 1) as f64;
    let scale = patch_width.max(patch_height).max(1) as f64;
    let mut samples = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let image_index = y * frame.width + x;
            let value = frame.data.value(image_index);
            if !value.is_finite() || mask[image_index] {
                continue;
            }
            let local_index = (y - y0) * patch_width + (x - x0);
            samples.push(Sample {
                value,
                background: Vector3::new(
                    1.0,
                    (x as f64 - center_x) / scale,
                    (y as f64 - center_y) / scale,
                ),
                sources: bases.iter().map(|basis| basis[local_index]).collect(),
            });
        }
    }
    let sample_count = samples.len();
    if sample_count < 24.max(parameter_count + 8) {
        return Ok(None);
    }
    let mut gram = Matrix3::zeros();
    let mut background_data = Vector3::zeros();
    let mut background_sources = vec![Vector3::zeros(); positions.len()];
    for sample in &samples {
        gram += sample.background * sample.background.transpose();
        background_data += sample.background * sample.value;
        for (index, source_value) in sample.sources.iter().enumerate() {
            background_sources[index] += sample.background * *source_value;
        }
    }
    let Some(gram_inverse) = gram.try_inverse() else {
        return Ok(None);
    };
    let background_from_data = gram_inverse * background_data;
    let background_from_sources = background_sources
        .iter()
        .map(|value| gram_inverse * value)
        .collect::<Vec<_>>();
    let mut source_residual = vec![Vec::with_capacity(sample_count); positions.len()];
    let mut data_residual = Vec::with_capacity(sample_count);
    for sample in &samples {
        data_residual.push(sample.value - sample.background.dot(&background_from_data));
        for component in 0..positions.len() {
            source_residual[component].push(
                sample.sources[component]
                    - sample.background.dot(&background_from_sources[component]),
            );
        }
    }
    let amplitudes = solve_nnls(&source_residual, &data_residual);
    let mut refit_background = background_data;
    for (basis, amplitude) in background_sources.iter().zip(&amplitudes) {
        refit_background -= basis * *amplitude;
    }
    let background_coefficients = gram_inverse * refit_background;
    let mut rss = 0.0;
    for sample in &samples {
        let model = sample.background.dot(&background_coefficients)
            + sample
                .sources
                .iter()
                .zip(&amplitudes)
                .map(|(basis, amplitude)| basis * amplitude)
                .sum::<f64>();
        let residual = sample.value - model;
        rss += residual * residual;
    }
    rss = rss.max(f64::EPSILON);
    let bic = sample_count as f64 * (rss / sample_count as f64).ln()
        + parameter_count as f64 * (sample_count as f64).ln();
    let max_amplitude = amplitudes
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let active_threshold = 1e-12f64.max(f64::EPSILON * max_amplitude.max(1.0));
    let active = amplitudes
        .iter()
        .enumerate()
        .filter_map(|(index, amplitude)| (*amplitude > active_threshold).then_some(index))
        .collect::<Vec<_>>();
    let degrees_of_freedom = sample_count.saturating_sub(parameter_count).max(1);
    let variance = rss / degrees_of_freedom as f64;
    let mut component_snr = vec![0.0; positions.len()];
    let mut covariance = None;
    if active.len() == 1 {
        let index = active[0];
        let information = vector_dot(&source_residual[index], &source_residual[index]);
        if information > 0.0 {
            let variance_amplitude = variance / information;
            component_snr[index] = amplitudes[index] / variance_amplitude.max(f64::EPSILON).sqrt();
            covariance = Some(Matrix2::new(variance_amplitude, 0.0, 0.0, 0.0));
        }
    } else if active.len() == 2 {
        let left = active[0];
        let right = active[1];
        let h00 = vector_dot(&source_residual[left], &source_residual[left]);
        let h11 = vector_dot(&source_residual[right], &source_residual[right]);
        let h01 = vector_dot(&source_residual[left], &source_residual[right]);
        let inverse = pseudoinverse_2(Matrix2::new(h00, h01, h01, h11));
        let cov = inverse * variance;
        component_snr[left] = amplitudes[left] / cov[(0, 0)].max(f64::EPSILON).sqrt();
        component_snr[right] = amplitudes[right] / cov[(1, 1)].max(f64::EPSILON).sqrt();
        covariance = Some(cov);
    }
    let design_condition_number = if positions.len() == 1 {
        let energy = vector_dot(&source_residual[0], &source_residual[0]);
        (energy > 0.0).then_some(1.0)
    } else {
        let h00 = vector_dot(&source_residual[0], &source_residual[0]);
        let h11 = vector_dot(&source_residual[1], &source_residual[1]);
        let h01 = vector_dot(&source_residual[0], &source_residual[1]);
        let eigen = Matrix2::new(h00, h01, h01, h11).symmetric_eigen();
        let largest = eigen.eigenvalues[0].max(eigen.eigenvalues[1]);
        let smallest = eigen.eigenvalues[0].min(eigen.eigenvalues[1]);
        (smallest > 0.0 && largest.is_finite()).then_some((largest / smallest).sqrt())
    };
    let amplitude_correlation = if positions.len() == 2 && active.len() == 2 {
        covariance.and_then(|cov| {
            let denominator = (cov[(0, 0)].max(0.0) * cov[(1, 1)].max(0.0)).sqrt();
            (denominator > f64::EPSILON).then_some(cov[(0, 1)] / denominator)
        })
    } else {
        None
    };
    Ok(Some(ModelFit {
        sample_count,
        masked_pixel_count: patch_width * patch_height - sample_count,
        rss,
        bic,
        amplitudes,
        component_snr,
        active_component_count: active.len(),
        amplitude_correlation: amplitude_correlation.filter(|value| value.is_finite()),
        design_condition_number: design_condition_number.filter(|value| value.is_finite()),
    }))
}

fn grid_positions(center: [f64; 2], radius: f64, step: f64) -> Vec<[f64; 2]> {
    let upper = radius + 0.5 * step;
    let mut offsets = Vec::new();
    let mut index = 0usize;
    loop {
        let offset = -radius + index as f64 * step;
        if offset >= upper {
            break;
        }
        offsets.push(offset);
        index += 1;
    }
    offsets
        .iter()
        .flat_map(|dy| {
            offsets
                .iter()
                .map(move |dx| [center[0] + dx, center[1] + dy])
        })
        .collect()
}

fn finite_option(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

#[allow(clippy::too_many_arguments)]
fn run_audit(
    paths: &[PathBuf],
    template_frame: &FitsFrame,
    sources: &[Detection],
    primary: &Detection,
    secondary: &Detection,
    primary_xy: [f64; 2],
    secondary_xy: [f64; 2],
    catalog_path: &Path,
    frame_shifts: Option<&[[f64; 2]]>,
    coordinate_mode: &str,
    resolved_modes: &[String],
    repeated_codes: &[i64],
    support_radius: usize,
    max_template_sources: usize,
    patch_padding_px: Option<f64>,
    search_radius: f64,
    grid_step: f64,
    confirmation_delta_bic: f64,
    confirmation_component_snr: f64,
) -> Result<EmpiricalPairFitResult> {
    if paths.is_empty() {
        return Err("paths cannot be empty".into());
    }
    if (primary_xy[0] - secondary_xy[0]).hypot(primary_xy[1] - secondary_xy[1]) <= 1e-6 {
        return Err("primary and secondary target coordinates must differ".into());
    }
    if !matches!(coordinate_mode, "centroid" | "peak") {
        return Err("coordinate_mode must be centroid or peak".into());
    }
    if resolved_modes.is_empty()
        || resolved_modes
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != resolved_modes.len()
    {
        return Err("mask_modes must be a non-empty sequence without duplicates".into());
    }
    if resolved_modes
        .iter()
        .any(|mode| !MASK_MODES.contains(&mode.as_str()))
    {
        return Err("mask_modes contains an unsupported mode".into());
    }
    if resolved_modes
        .iter()
        .any(|mode| mode.contains("repeated_code"))
        && repeated_codes.is_empty()
    {
        return Err("repeated-code mask mode requires repeated_code_values".into());
    }
    if support_radius < 3 || max_template_sources == 0 {
        return Err("support_radius must be at least 3 and source count must be positive".into());
    }
    if !search_radius.is_finite() || search_radius <= 0.0 {
        return Err("best_single_search_radius_px must be finite and positive".into());
    }
    if !grid_step.is_finite() || grid_step <= 0.0 {
        return Err("best_single_grid_step_px must be finite and positive".into());
    }
    if !confirmation_delta_bic.is_finite()
        || !confirmation_component_snr.is_finite()
        || confirmation_delta_bic <= 0.0
        || confirmation_component_snr <= 0.0
    {
        return Err("confirmation thresholds must be finite and positive".into());
    }
    if sources.is_empty() {
        return Err("template_sources cannot be empty".into());
    }
    let psf = estimate_psf(
        template_frame,
        sources,
        support_radius,
        max_template_sources,
    )?
    .ok_or("not enough eligible isolated sources to build an empirical PSF")?;
    let radius = psf.support_radius as f64;
    let patch_padding = patch_padding_px.unwrap_or((radius + 3.0).max(10.0));
    if !patch_padding.is_finite() || patch_padding < radius + 1.0 {
        return Err("patch_padding_px must be at least support_radius + 1".into());
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
    let midpoint = [
        0.5 * (primary_xy[0] + secondary_xy[0]),
        0.5 * (primary_xy[1] + secondary_xy[1]),
    ];
    let mut rows = Vec::new();
    for (frame_index, path) in paths.iter().enumerate() {
        let frame = fits::read_fits(path, false)?;
        let [shift_x, shift_y] = shifts[frame_index];
        let pair_positions = [
            [primary_xy[0] + shift_x, primary_xy[1] + shift_y],
            [secondary_xy[0] + shift_x, secondary_xy[1] + shift_y],
        ];
        let frame_midpoint = [midpoint[0] + shift_x, midpoint[1] + shift_y];
        let search_positions = grid_positions(frame_midpoint, search_radius, grid_step);
        let bounds = patch_bounds(frame.width, frame.height, &pair_positions, patch_padding);
        let [x0, y0, x1, y1] = bounds;
        let patch_width = x1.saturating_sub(x0);
        let patch_height = y1.saturating_sub(y0);
        if patch_height < 2 * psf.support_radius + 3 || patch_width < 2 * psf.support_radius + 3 {
            return Err(format!(
                "pair patch is too small in frame {}",
                frame_index + 1
            ));
        }
        let masks = mask_modes(&frame, resolved_modes, repeated_codes)?;
        for (mask_mode, mask) in masks {
            let single_fit = fit_fixed(&frame, &mask, &pair_positions[..1], bounds, &psf)?;
            let double_fit = fit_fixed(&frame, &mask, &pair_positions, bounds, &psf)?;
            let mut best_grid_fit: Option<ModelFit> = None;
            let mut best_grid_position = None;
            let mut evaluated_count = 0;
            for grid_position in &search_positions {
                if let Some(candidate) = fit_fixed(&frame, &mask, &[*grid_position], bounds, &psf)?
                {
                    evaluated_count += 1;
                    if best_grid_fit
                        .as_ref()
                        .is_none_or(|best| candidate.rss < best.rss)
                    {
                        best_grid_position = Some(*grid_position);
                        best_grid_fit = Some(candidate);
                    }
                }
            }
            let fixed_single_bic = single_fit.as_ref().map(|fit| fit.bic);
            let fixed_double_bic = double_fit.as_ref().map(|fit| fit.bic);
            let fixed_delta = fixed_single_bic
                .zip(fixed_double_bic)
                .map(|(single, double)| single - double);
            let grid_bic = best_grid_fit.as_ref().map(|fit| fit.bic);
            let grid_delta = grid_bic
                .zip(fixed_double_bic)
                .map(|(single, double)| single - double);
            let amplitudes = double_fit
                .as_ref()
                .map_or_else(Vec::new, |fit| fit.amplitudes.clone());
            let component_snr = double_fit
                .as_ref()
                .map_or_else(Vec::new, |fit| fit.component_snr.clone());
            let second_snr = component_snr.get(1).copied();
            let fixed_line = fixed_delta.is_some_and(|delta| delta >= confirmation_delta_bic)
                && second_snr.is_some_and(|snr| snr >= confirmation_component_snr);
            let grid_line = grid_delta.is_some_and(|delta| delta >= confirmation_delta_bic)
                && second_snr.is_some_and(|snr| snr >= confirmation_component_snr);
            let note = if double_fit.is_none() {
                "双源经验 PSF 拟合无有效结果"
            } else if fixed_line && grid_line {
                "固定锚点和最佳单源位置控制均显示双源改善；仍非星表身份"
            } else if fixed_line {
                "固定锚点双源改善，但最佳单源位置控制未通过；不作双星结论"
            } else {
                "固定位置双源未形成当前证据线；不等于排除混叠或弱源"
            };
            rows.push(EmpiricalPairFitRow {
                frame_index: frame_index + 1,
                path: path.to_string_lossy().into_owned(),
                coordinate_mode: coordinate_mode.to_owned(),
                mask_mode,
                patch_x0: x0,
                patch_y0: y0,
                patch_width,
                patch_height,
                sample_count: double_fit.as_ref().map_or(0, |fit| fit.sample_count),
                masked_pixel_count: double_fit.as_ref().map_or(0, |fit| fit.masked_pixel_count),
                fixed_single_rss: single_fit.as_ref().map(|fit| fit.rss),
                fixed_double_rss: double_fit.as_ref().map(|fit| fit.rss),
                fixed_single_bic,
                fixed_double_bic,
                fixed_delta_bic: fixed_delta,
                best_single_grid_rss: best_grid_fit.as_ref().map(|fit| fit.rss),
                best_single_grid_bic: grid_bic,
                best_single_grid_x: best_grid_position.map(|point: [f64; 2]| point[0]),
                best_single_grid_y: best_grid_position.map(|point: [f64; 2]| point[1]),
                best_single_grid_evaluated_count: evaluated_count,
                best_single_grid_delta_bic: grid_delta,
                double_amplitudes: amplitudes,
                double_component_snr: component_snr,
                double_second_component_snr: second_snr,
                double_amplitude_correlation: double_fit
                    .as_ref()
                    .and_then(|fit| finite_option(fit.amplitude_correlation)),
                double_design_condition_number: double_fit
                    .as_ref()
                    .and_then(|fit| finite_option(fit.design_condition_number)),
                double_active_component_count: double_fit
                    .as_ref()
                    .map_or(0, |fit| fit.active_component_count),
                fixed_confirmation_line: fixed_line,
                best_single_grid_control_line: grid_line,
                note: note.to_owned(),
            });
        }
    }
    let raw_rows = rows
        .iter()
        .filter(|row| row.mask_mode == "raw")
        .collect::<Vec<_>>();
    let fixed_count = raw_rows
        .iter()
        .filter(|row| row.fixed_confirmation_line)
        .count();
    let grid_count = raw_rows
        .iter()
        .filter(|row| row.best_single_grid_control_line)
        .count();
    let conclusion = format!(
        "经验 PSF 锚点敏感性审计完成：{} 帧，坐标锚点={}；raw 固定位置 K=2 证据线 {}/{} 帧，加入局部最佳单源位置控制后 {}/{} 帧。后一个数不是形式化的自由位置 BIC，而是同窗网格搜索的保守诊断；若双源优势在该控制下消失，优先解释为单源位置锚定或非高斯残差的模型自由度，不能据此确认两颗真实恒星。",
        paths.len(), coordinate_mode, fixed_count, raw_rows.len(), grid_count, raw_rows.len()
    );
    let parameters = json!({
        "model":"shared planar background + fixed empirical PSF + nonnegative amplitudes",
        "coordinate_mode":coordinate_mode,
        "mask_modes":resolved_modes,
        "repeated_code_values":repeated_codes,
        "support_radius":support_radius,
        "max_template_sources":max_template_sources,
        "patch_padding_px":patch_padding,
        "best_single_search_radius_px":search_radius,
        "best_single_grid_step_px":grid_step,
        "best_single_grid_center":midpoint,
        "confirmation_delta_bic":confirmation_delta_bic,
        "confirmation_component_snr":confirmation_component_snr,
        "frame_shifts":shifts,
        "interpretation_boundary":"固定位置 BIC 是局部模型证据；最佳单源网格搜索是位置锚定控制；二者均不是星表身份、恒星概率或物理真值"
    });
    Ok(EmpiricalPairFitResult {
        frame_count: paths.len(),
        coordinate_mode: coordinate_mode.to_owned(),
        primary_detection_id: Some(primary.detection_id),
        secondary_detection_id: Some(secondary.detection_id),
        primary_target_xy: primary_xy,
        secondary_target_xy: secondary_xy,
        frame_shifts: shifts,
        template_frame_path: template_frame.path.to_string_lossy().into_owned(),
        template_catalog_path: Some(catalog_path.to_string_lossy().into_owned()),
        mask_modes: resolved_modes.to_vec(),
        support_radius,
        max_template_sources,
        patch_padding_px: patch_padding,
        best_single_search_radius_px: search_radius,
        best_single_grid_step_px: grid_step,
        confirmation_delta_bic,
        confirmation_component_snr,
        empirical_psf: psf,
        parameters,
        rows,
        conclusion,
    })
}

fn read_shifts(path: &Path) -> Result<Vec<[f64; 2]>> {
    let payload: Value =
        serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("invalid shifts JSON: {error}"))?;
    let values = payload
        .get("cumulative_shifts")
        .and_then(Value::as_array)
        .ok_or("shifts JSON requires cumulative_shifts")?;
    values
        .iter()
        .map(|item| {
            let pair = item
                .as_array()
                .ok_or("each cumulative shift must be [x,y]")?;
            if pair.len() != 2 {
                return Err("each cumulative shift must be [x,y]".into());
            }
            let x = pair[0].as_f64().ok_or("shift x must be numeric")?;
            let y = pair[1].as_f64().ok_or("shift y must be numeric")?;
            if !x.is_finite() || !y.is_finite() {
                return Err("frame shifts must be finite".into());
            }
            Ok([x, y])
        })
        .collect()
}

fn csv_optional(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn csv_row(row: &EmpiricalPairFitRow) -> Vec<String> {
    vec![
        row.frame_index.to_string(),
        row.path.clone(),
        row.coordinate_mode.clone(),
        row.mask_mode.clone(),
        row.patch_x0.to_string(),
        row.patch_y0.to_string(),
        row.patch_width.to_string(),
        row.patch_height.to_string(),
        row.sample_count.to_string(),
        row.masked_pixel_count.to_string(),
        csv_optional(row.fixed_single_rss),
        csv_optional(row.fixed_double_rss),
        csv_optional(row.fixed_single_bic),
        csv_optional(row.fixed_double_bic),
        csv_optional(row.fixed_delta_bic),
        csv_optional(row.best_single_grid_rss),
        csv_optional(row.best_single_grid_bic),
        csv_optional(row.best_single_grid_x),
        csv_optional(row.best_single_grid_y),
        row.best_single_grid_evaluated_count.to_string(),
        csv_optional(row.best_single_grid_delta_bic),
        serde_json::to_string(&row.double_amplitudes).unwrap_or_default(),
        serde_json::to_string(&row.double_component_snr).unwrap_or_default(),
        csv_optional(row.double_second_component_snr),
        csv_optional(row.double_amplitude_correlation),
        csv_optional(row.double_design_condition_number),
        row.double_active_component_count.to_string(),
        if row.fixed_confirmation_line {
            "True"
        } else {
            "False"
        }
        .to_owned(),
        if row.best_single_grid_control_line {
            "True"
        } else {
            "False"
        }
        .to_owned(),
        row.note.clone(),
    ]
}

fn write_csv(path: &Path, rows: &[EmpiricalPairFitRow]) -> Result<()> {
    let mut file = fs::File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(CSV_FIELDS)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(csv_row(row))
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn draw_line(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    a: [usize; 2],
    b: [usize; 2],
    color: [u8; 3],
) {
    let dx = (b[0] as isize - a[0] as isize).abs();
    let sx = if a[0] < b[0] { 1isize } else { -1 };
    let dy = -(b[1] as isize - a[1] as isize).abs();
    let sy = if a[1] < b[1] { 1isize } else { -1 };
    let mut error = dx + dy;
    let (mut x, mut y) = (a[0] as isize, a[1] as isize);
    loop {
        if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
            let index = (y as usize * width + x as usize) * 3;
            pixels[index..index + 3].copy_from_slice(&color);
        }
        if x == b[0] as isize && y == b[1] as isize {
            break;
        }
        let twice = 2 * error;
        if twice >= dy {
            error += dy;
            x += sx;
        }
        if twice <= dx {
            error += dx;
            y += sy;
        }
    }
}

fn write_bic_chart(path: &Path, rows: &[EmpiricalPairFitRow]) -> Result<()> {
    const WIDTH: usize = 1200;
    const HEIGHT: usize = 680;
    let raw = rows
        .iter()
        .filter(|row| row.mask_mode == "raw")
        .collect::<Vec<_>>();
    let values = raw
        .iter()
        .flat_map(|row| [row.fixed_delta_bic, row.best_single_grid_delta_bic])
        .flatten()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let (mut low, mut high) = if values.is_empty() {
        (0.0, 1.0)
    } else {
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (low, high)
    };
    if high <= low {
        high = low + 1.0;
    }
    let padding = (high - low) * 0.08;
    low -= padding;
    high += padding;
    let mut pixels = vec![255u8; WIDTH * HEIGHT * 3];
    let plot = [70usize, 40usize, WIDTH - 35, HEIGHT - 60];
    for step in 0..=5 {
        let y = plot[1] + (plot[3] - plot[1]) * step / 5;
        draw_line(
            &mut pixels,
            WIDTH,
            HEIGHT,
            [plot[0], y],
            [plot[2], y],
            [224, 230, 236],
        );
    }
    draw_line(
        &mut pixels,
        WIDTH,
        HEIGHT,
        [plot[0], plot[1]],
        [plot[0], plot[3]],
        [70, 83, 96],
    );
    draw_line(
        &mut pixels,
        WIDTH,
        HEIGHT,
        [plot[0], plot[3]],
        [plot[2], plot[3]],
        [70, 83, 96],
    );
    for (column, color) in [(0usize, [215, 148, 50]), (1usize, [79, 155, 131])] {
        let points = raw
            .iter()
            .enumerate()
            .filter_map(|(index, row)| {
                let value = if column == 0 {
                    row.fixed_delta_bic
                } else {
                    row.best_single_grid_delta_bic
                }?;
                let x = if raw.len() <= 1 {
                    (plot[0] + plot[2]) / 2
                } else {
                    plot[0] + (plot[2] - plot[0]) * index / (raw.len() - 1)
                };
                let ratio = ((value - low) / (high - low)).clamp(0.0, 1.0);
                let y = plot[3] - (ratio * (plot[3] - plot[1]) as f64).round() as usize;
                Some([x, y])
            })
            .collect::<Vec<_>>();
        for pair in points.windows(2) {
            draw_line(&mut pixels, WIDTH, HEIGHT, pair[0], pair[1], color);
        }
        for point in points {
            draw_line(
                &mut pixels,
                WIDTH,
                HEIGHT,
                [point[0].saturating_sub(3), point[1]],
                [(point[0] + 3).min(WIDTH - 1), point[1]],
                color,
            );
            draw_line(
                &mut pixels,
                WIDTH,
                HEIGHT,
                [point[0], point[1].saturating_sub(3)],
                [point[0], (point[1] + 3).min(HEIGHT - 1)],
                color,
            );
        }
    }
    let file = fs::File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = png::Encoder::new(file, WIDTH as u32, HEIGHT as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .map_err(|error| error.to_string())?
        .write_image_data(&pixels)
        .map_err(|error| error.to_string())
}

pub fn write_artifacts(result: &EmpiricalPairFitResult, out_dir: &Path) -> Result<Value> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    write_csv(&out_dir.join("empirical_pair_fit.csv"), &result.rows)?;
    fs::write(
        out_dir.join("empirical_pair_fit.json"),
        serde_json::to_vec_pretty(&result.as_json()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if result.rows.iter().any(|row| row.mask_mode == "raw") {
        write_bic_chart(&out_dir.join("empirical_pair_fit_bic.png"), &result.rows)?;
    }
    Ok(json!({"output":out_dir,"frame_count":result.frame_count,"conclusion":result.conclusion}))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let options = parse_args(args)?;
    let paths = ordered_fits_paths(&options.data_dir)?;
    let sources = feature_psf_leaveout::load_detection_catalog(&options.source_catalog)?;
    let primary = sources
        .iter()
        .find(|source| source.detection_id == options.primary_id)
        .ok_or_else(|| format!("source catalog has no detection_id={}", options.primary_id))?;
    let secondary = sources
        .iter()
        .find(|source| source.detection_id == options.secondary_id)
        .ok_or_else(|| {
            format!(
                "source catalog has no detection_id={}",
                options.secondary_id
            )
        })?;
    let target = |source: &Detection| {
        if options.coordinate_mode == "peak" {
            match (source.peak_x, source.peak_y) {
                (Some(x), Some(y)) => [x, y],
                _ => [source.x, source.y],
            }
        } else {
            [source.x, source.y]
        }
    };
    let shifts = options
        .shifts_json
        .as_deref()
        .map(read_shifts)
        .transpose()?;
    let repeated_codes = resolve_repeated_codes(&sources, options.repeated_code_values.as_deref());
    let explicit_modes = options.mask_modes.clone();
    let raw_codes = options.repeated_code_values.as_deref();
    if explicit_modes
        .iter()
        .any(|mode| mode.contains("repeated_code"))
        && raw_codes.is_none()
        && repeated_codes.is_empty()
    {
        return Err("repeated-code mask mode requires repeated_code_values".into());
    }
    let template_frame = fits::read_fits(&paths[0], false)?;
    let result = run_audit(
        &paths,
        &template_frame,
        &sources,
        primary,
        secondary,
        target(primary),
        target(secondary),
        &options.source_catalog,
        shifts.as_deref(),
        &options.coordinate_mode,
        &explicit_modes,
        &repeated_codes,
        options.support_radius,
        options.max_template_sources,
        options.patch_padding_px,
        options.best_single_search_radius_px,
        options.best_single_grid_step_px,
        10.0,
        5.0,
    )?;
    write_artifacts(&result, &options.out_dir)
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
        include_str!("../tests/fixtures/empirical_pair_fit_parity/reference-output.json");
    const REFERENCE_CSV: &[u8] =
        include_bytes!("../tests/fixtures/empirical_pair_fit_parity/reference.csv");

    fn close(actual: f64, expected: f64, label: &str) {
        let tolerance = 1e-5f64.max(expected.abs() * 1e-8);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: {actual:?} != {expected:?} ± {tolerance:?}"
        );
    }

    fn compare_json(actual: &Value, expected: &Value, label: &str) {
        match (actual, expected) {
            (Value::Object(a), Value::Object(b)) => {
                let mut ak = a.keys().collect::<Vec<_>>();
                let mut bk = b.keys().collect::<Vec<_>>();
                ak.sort();
                bk.sort();
                assert_eq!(ak, bk, "{label} keys");
                for key in b.keys() {
                    if matches!(
                        key.as_str(),
                        "path" | "template_frame_path" | "template_catalog_path"
                    ) {
                        continue;
                    }
                    compare_json(&a[key], &b[key], &format!("{label}.{key}"));
                }
            }
            (Value::Array(a), Value::Array(b)) => {
                assert_eq!(a.len(), b.len(), "{label} length");
                for (index, (a, b)) in a.iter().zip(b).enumerate() {
                    compare_json(a, b, &format!("{label}[{index}]"));
                }
            }
            (Value::Number(a), Value::Number(b)) => {
                close(a.as_f64().unwrap(), b.as_f64().unwrap(), label)
            }
            _ => assert_eq!(actual, expected, "{label}"),
        }
    }

    #[test]
    fn frozen_python_fit_matches_rust_json_csv_and_png_dimensions() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let frames = manifest.join("tests/fixtures/empirical_pair_audit_parity/frames");
        let catalog = manifest.join("tests/fixtures/source_audit_parity/source_catalog.csv");
        let output = std::env::temp_dir().join(format!(
            "rst19-empirical-pair-fit-{}-{}",
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
            serde_json::from_slice(&fs::read(output.join("empirical_pair_fit.json")).unwrap())
                .unwrap();
        let expected_json: Value = serde_json::from_str(REFERENCE_JSON).unwrap();
        compare_json(&actual_json, &expected_json, "result");

        let actual_csv = fs::read(output.join("empirical_pair_fit.csv")).unwrap();
        let actual_csv = actual_csv
            .strip_prefix(&[0xef, 0xbb, 0xbf])
            .unwrap_or(&actual_csv);
        let reference_csv = REFERENCE_CSV
            .strip_prefix(&[0xef, 0xbb, 0xbf])
            .unwrap_or(REFERENCE_CSV);
        let mut ar = csv::Reader::from_reader(actual_csv);
        let mut er = csv::Reader::from_reader(reference_csv);
        let ah = ar.headers().unwrap().clone();
        let eh = er.headers().unwrap().clone();
        assert_eq!(ah, eh, "CSV headers");
        let actual_rows = ar
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let expected_rows = er
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(actual_rows.len(), expected_rows.len());
        for (row_index, (a, e)) in actual_rows.iter().zip(&expected_rows).enumerate() {
            for column in 0..a.len() {
                let field = &ah[column];
                let av = &a[column];
                let ev = &e[column];
                if field == "path" || av == ev {
                    continue;
                }
                if matches!(field, "double_amplitudes" | "double_component_snr") {
                    let aa: Vec<f64> = serde_json::from_str(av).unwrap();
                    let ee: Vec<f64> = serde_json::from_str(ev).unwrap();
                    assert_eq!(aa.len(), ee.len());
                    for (i, (x, y)) in aa.iter().zip(ee).enumerate() {
                        close(*x, y, &format!("CSV row {row_index} {field}[{i}]"));
                    }
                    continue;
                }
                match (av.parse::<f64>(), ev.parse::<f64>()) {
                    (Ok(x), Ok(y)) => close(x, y, &format!("CSV row {row_index} {field}")),
                    _ => assert_eq!(av, ev, "CSV row {row_index} {field}"),
                }
            }
        }
        let png = File::open(output.join("empirical_pair_fit_bic.png")).unwrap();
        let reader = png::Decoder::new(BufReader::new(png)).read_info().unwrap();
        assert_eq!(reader.info().width, 1200);
        assert_eq!(reader.info().height, 680);
        fs::remove_dir_all(output).unwrap();
    }
}
