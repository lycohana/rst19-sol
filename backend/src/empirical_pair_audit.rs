//! Global/local empirical-PSF shape checks for a fixed pair across FITS frames.
//!
//! Correlation and residual values are shape diagnostics only; this module
//! does not infer catalog identity or physical source truth.

use crate::{
    empirical_psf::{self, EmpiricalPsf},
    feature_psf_leaveout, fits,
    measurement::Detection,
    models::{FitsFrame, NumericPlane, PixelData},
    statistics, Result,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize)]
pub struct PairAuditRow {
    pub frame_index: usize,
    pub path: String,
    pub frame_shift_x_px: f64,
    pub frame_shift_y_px: f64,
    pub primary_x: f64,
    pub primary_y: f64,
    pub secondary_x: f64,
    pub secondary_y: f64,
    pub primary_background_adu: Option<f64>,
    pub secondary_background_adu: Option<f64>,
    pub global_psf_source_count: usize,
    pub local_psf_source_count: usize,
    pub local_template_available: bool,
    pub primary_global_correlation: Option<f64>,
    pub primary_global_relative_residual: Option<f64>,
    pub primary_global_central_energy_fraction: Option<f64>,
    pub primary_global_fitted_amplitude_adu: Option<f64>,
    pub secondary_global_correlation: Option<f64>,
    pub secondary_global_relative_residual: Option<f64>,
    pub secondary_global_central_energy_fraction: Option<f64>,
    pub secondary_global_fitted_amplitude_adu: Option<f64>,
    pub primary_local_correlation: Option<f64>,
    pub primary_local_relative_residual: Option<f64>,
    pub primary_local_central_energy_fraction: Option<f64>,
    pub primary_local_fitted_amplitude_adu: Option<f64>,
    pub secondary_local_correlation: Option<f64>,
    pub secondary_local_relative_residual: Option<f64>,
    pub secondary_local_central_energy_fraction: Option<f64>,
    pub secondary_local_fitted_amplitude_adu: Option<f64>,
    pub note: String,
}

#[derive(Clone, Debug)]
pub struct PairAuditResult {
    pub frame_count: usize,
    pub primary_detection_id: Option<i64>,
    pub secondary_detection_id: Option<i64>,
    pub primary_target_xy: [f64; 2],
    pub secondary_target_xy: [f64; 2],
    pub frame_shifts: Vec<[f64; 2]>,
    pub template_frame_path: String,
    pub template_catalog_path: Option<String>,
    pub support_radius: usize,
    pub max_template_sources: usize,
    pub grid_size: usize,
    pub pair_cell: [usize; 2],
    pub local_padding_px: f64,
    pub global_psf: Option<EmpiricalPsf>,
    pub local_psf: Option<EmpiricalPsf>,
    pub rows: Vec<PairAuditRow>,
    pub conclusion: String,
}

#[derive(Clone, Debug)]
pub struct PairAuditOptions {
    pub data_dir: PathBuf,
    pub source_catalog: PathBuf,
    pub primary_id: i64,
    pub secondary_id: i64,
    pub coordinate_mode: String,
    pub support_radius: usize,
    pub max_template_sources: usize,
    pub grid_size: usize,
    pub shifts_json: Option<PathBuf>,
    pub out_dir: PathBuf,
}

#[derive(Clone, Copy)]
enum CorrelationSeries {
    PrimaryGlobal,
    SecondaryGlobal,
    PrimaryLocal,
    SecondaryLocal,
}

impl CorrelationSeries {
    fn value(self, row: &PairAuditRow) -> Option<f64> {
        match self {
            Self::PrimaryGlobal => row.primary_global_correlation,
            Self::SecondaryGlobal => row.secondary_global_correlation,
            Self::PrimaryLocal => row.primary_local_correlation,
            Self::SecondaryLocal => row.secondary_local_correlation,
        }
    }
}

const CSV_FIELDS: &[&str] = &[
    "frame_index",
    "path",
    "frame_shift_x_px",
    "frame_shift_y_px",
    "primary_x",
    "primary_y",
    "secondary_x",
    "secondary_y",
    "primary_background_adu",
    "secondary_background_adu",
    "global_psf_source_count",
    "local_psf_source_count",
    "local_template_available",
    "primary_global_correlation",
    "primary_global_relative_residual",
    "primary_global_central_energy_fraction",
    "primary_global_fitted_amplitude_adu",
    "secondary_global_correlation",
    "secondary_global_relative_residual",
    "secondary_global_central_energy_fraction",
    "secondary_global_fitted_amplitude_adu",
    "primary_local_correlation",
    "primary_local_relative_residual",
    "primary_local_central_energy_fraction",
    "primary_local_fitted_amplitude_adu",
    "secondary_local_correlation",
    "secondary_local_relative_residual",
    "secondary_local_central_energy_fraction",
    "secondary_local_fitted_amplitude_adu",
    "note",
];

pub const HELP: &str = "用源表构造全局/局部经验 PSF，逐帧复核两个近邻候选的形状\n\nUsage: rst19-empirical-pair-audit DATA_DIR --source-catalog CSV --primary-id ID --secondary-id ID [OPTIONS]\n\nOptions: --coordinate-mode centroid|peak --support-radius N --max-template-sources N\n         --grid-size N --shifts-json JSON --out-dir DIR";

fn integer(value: &str, name: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid non-negative integer for --{name}"))
}

fn parse_args(args: &[String]) -> Result<PairAuditOptions> {
    let mut data_dir = None;
    let mut source_catalog = None;
    let mut primary_id = None;
    let mut secondary_id = None;
    let mut coordinate_mode = "centroid".to_owned();
    let mut support_radius = 7;
    let mut max_template_sources = 64;
    let mut grid_size = 4;
    let mut shifts_json = None;
    let mut out_dir = PathBuf::from("tmp/empirical-pair-psf-audit");
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
        let value = if let Some(value) = inline {
            value.to_owned()
        } else {
            index += 1;
            args.get(index)
                .ok_or_else(|| format!("--{name} requires a value"))?
                .clone()
        };
        match name {
            "source-catalog" => source_catalog = Some(PathBuf::from(value)),
            "primary-id" => primary_id = Some(value.parse().map_err(|_| "invalid --primary-id")?),
            "secondary-id" => {
                secondary_id = Some(value.parse().map_err(|_| "invalid --secondary-id")?)
            }
            "coordinate-mode" => {
                if !matches!(value.as_str(), "centroid" | "peak") {
                    return Err("--coordinate-mode must be centroid or peak".into());
                }
                coordinate_mode = value;
            }
            "support-radius" => support_radius = integer(&value, name)?,
            "max-template-sources" => max_template_sources = integer(&value, name)?,
            "grid-size" => grid_size = integer(&value, name)?,
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
    if support_radius < 3 || max_template_sources == 0 || grid_size == 0 {
        return Err("support radius must be at least 3; template source count and grid size must be positive".into());
    }
    Ok(PairAuditOptions {
        data_dir,
        source_catalog,
        primary_id,
        secondary_id,
        coordinate_mode,
        support_radius,
        max_template_sources,
        grid_size,
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
    isolation_sources: &[Detection],
    excluded_ids: &[i64],
    support_radius: usize,
    max_sources: usize,
) -> Result<Option<EmpiricalPsf>> {
    numeric_plane(frame, |plane| {
        empirical_psf::estimate(
            plane,
            frame.height,
            frame.width,
            sources,
            Some(isolation_sources),
            support_radius,
            max_sources,
            excluded_ids,
        )
    })
}

fn similarity(
    frame: &FitsFrame,
    source: &Detection,
    psf: Option<&EmpiricalPsf>,
    mask: &[bool],
) -> Result<Option<(f64, f64, f64, f64)>> {
    let Some(psf) = psf else {
        return Ok(None);
    };
    numeric_plane(frame, |plane| {
        empirical_psf::similarity_metrics_with_mask(
            plane,
            frame.height,
            frame.width,
            source,
            psf,
            mask,
        )
    })
}

fn probe_background(
    frame: &FitsFrame,
    mask: &[bool],
    x: f64,
    y: f64,
    support_radius: usize,
) -> Option<f64> {
    let radius = (support_radius + 4).max(8) as isize;
    let center_x = x.round_ties_even() as isize;
    let center_y = y.round_ties_even() as isize;
    let x0 = (center_x - radius).max(0) as usize;
    let y0 = (center_y - radius).max(0) as usize;
    let x1 = (center_x + radius + 1).min(frame.width as isize).max(0) as usize;
    let y1 = (center_y + radius + 1).min(frame.height as isize).max(0) as usize;
    let mut samples = Vec::new();
    for row in y0..y1 {
        for col in x0..x1 {
            let index = row * frame.width + col;
            let value = frame.data.value(index);
            let distance = (col as f64 - x).hypot(row as f64 - y);
            if value.is_finite() && !mask[index] && distance >= support_radius as f64 {
                samples.push(value);
            }
        }
    }
    statistics::median(&samples).ok()
}

fn probe_source(
    template: &Detection,
    id: i64,
    x: f64,
    y: f64,
    background: Option<f64>,
) -> Detection {
    let mut source = template.clone();
    source.detection_id = id;
    source.x = x;
    source.y = y;
    source.peak = 0.0;
    source.flux = 0.0;
    source.background = background.unwrap_or(0.0);
    source.noise = 1.0;
    source.snr = 0.0;
    source
}

fn grid_cell(x: f64, y: f64, height: usize, width: usize, grid_size: usize) -> (usize, usize) {
    let column =
        ((x * grid_size as f64 / width as f64) as isize).clamp(0, grid_size as isize - 1) as usize;
    let row =
        ((y * grid_size as f64 / height as f64) as isize).clamp(0, grid_size as isize - 1) as usize;
    (row, column)
}

fn median_optional(values: impl IntoIterator<Item = Option<f64>>) -> Option<f64> {
    let finite = values
        .into_iter()
        .flatten()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    statistics::median(&finite).ok()
}

fn metric(metrics: Option<(f64, f64, f64, f64)>, index: usize) -> Option<f64> {
    metrics.and_then(|values| {
        let value = [values.0, values.1, values.2, values.3][index];
        value.is_finite().then_some(value)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    frame_paths: &[PathBuf],
    template_sources: &[Detection],
    primary: &Detection,
    secondary: &Detection,
    primary_xy: [f64; 2],
    secondary_xy: [f64; 2],
    catalog_path: &Path,
    frame_shifts: Option<&[[f64; 2]]>,
    support_radius: usize,
    max_template_sources: usize,
    grid_size: usize,
) -> Result<PairAuditResult> {
    if frame_paths.is_empty() {
        return Err("paths cannot be empty".into());
    }
    if !primary_xy
        .iter()
        .chain(&secondary_xy)
        .all(|value| value.is_finite())
    {
        return Err("target coordinates must be finite".into());
    }
    if (secondary_xy[0] - primary_xy[0]).hypot(secondary_xy[1] - primary_xy[1]) <= 1e-6 {
        return Err("primary and secondary target coordinates must differ".into());
    }
    if support_radius < 3 || max_template_sources == 0 || grid_size == 0 {
        return Err(
            "support radius must be at least 3; source count and grid size must be positive".into(),
        );
    }
    if template_sources.is_empty() {
        return Err("template_sources cannot be empty".into());
    }
    let shifts = if let Some(shifts) = frame_shifts {
        if shifts.len() != frame_paths.len() {
            return Err("frame_shifts length must match paths".into());
        }
        if shifts.iter().flatten().any(|value| !value.is_finite()) {
            return Err("frame_shifts must contain finite values".into());
        }
        shifts.to_vec()
    } else {
        vec![[0.0, 0.0]; frame_paths.len()]
    };
    let template_frame = fits::read_fits(&frame_paths[0], false)?;
    let excluded_ids = [primary.detection_id, secondary.detection_id];
    let global_psf = estimate_psf(
        &template_frame,
        template_sources,
        template_sources,
        &[],
        support_radius,
        max_template_sources,
    )?;
    let pair_midpoint = [
        0.5 * (primary_xy[0] + secondary_xy[0]),
        0.5 * (primary_xy[1] + secondary_xy[1]),
    ];
    let pair_cell = grid_cell(
        pair_midpoint[0],
        pair_midpoint[1],
        template_frame.height,
        template_frame.width,
        grid_size,
    );
    let local_padding = (2 * support_radius + 4).max(18) as f64;
    let x_min = template_frame.width as f64 * pair_cell.1 as f64 / grid_size as f64 - local_padding;
    let x_max =
        template_frame.width as f64 * (pair_cell.1 + 1) as f64 / grid_size as f64 + local_padding;
    let y_min =
        template_frame.height as f64 * pair_cell.0 as f64 / grid_size as f64 - local_padding;
    let y_max =
        template_frame.height as f64 * (pair_cell.0 + 1) as f64 / grid_size as f64 + local_padding;
    let local_pool = template_sources
        .iter()
        .filter(|source| {
            source.detection_id != primary.detection_id
                && source.detection_id != secondary.detection_id
                && source.x.is_finite()
                && source.y.is_finite()
                && x_min <= source.x
                && source.x < x_max
                && y_min <= source.y
                && source.y < y_max
        })
        .cloned()
        .collect::<Vec<_>>();
    let local_psf = estimate_psf(
        &template_frame,
        &local_pool,
        template_sources,
        &excluded_ids,
        support_radius,
        max_template_sources,
    )?;

    let mut rows = Vec::with_capacity(frame_paths.len());
    for (frame_index, path) in frame_paths.iter().enumerate() {
        let frame = fits::read_fits(path, false)?;
        let [shift_x, shift_y] = shifts[frame_index];
        let primary_x = primary_xy[0] + shift_x;
        let primary_y = primary_xy[1] + shift_y;
        let secondary_x = secondary_xy[0] + shift_x;
        let secondary_y = secondary_xy[1] + shift_y;
        let mask = fits::auxiliary_mask(frame.height, frame.width)?;
        let primary_background =
            probe_background(&frame, &mask, primary_x, primary_y, support_radius);
        let secondary_background =
            probe_background(&frame, &mask, secondary_x, secondary_y, support_radius);
        let primary_probe = probe_source(
            primary,
            primary.detection_id,
            primary_x,
            primary_y,
            primary_background,
        );
        let secondary_probe = probe_source(
            secondary,
            secondary.detection_id,
            secondary_x,
            secondary_y,
            secondary_background,
        );
        let primary_global = similarity(&frame, &primary_probe, global_psf.as_ref(), &mask)?;
        let secondary_global = similarity(&frame, &secondary_probe, global_psf.as_ref(), &mask)?;
        let primary_local = similarity(&frame, &primary_probe, local_psf.as_ref(), &mask)?;
        let secondary_local = similarity(&frame, &secondary_probe, local_psf.as_ref(), &mask)?;
        let note = if global_psf.is_none() {
            "全局经验 PSF 模板不足；形状诊断不可用"
        } else if local_psf.is_none() {
            "局部隔离模板不足；仅报告全局 PSF 形状诊断，不等于星表身份"
        } else {
            "全局/局部经验 PSF 仅为形状诊断，不等于星表身份"
        };
        rows.push(PairAuditRow {
            frame_index: frame_index + 1,
            path: path.to_string_lossy().into_owned(),
            frame_shift_x_px: shift_x,
            frame_shift_y_px: shift_y,
            primary_x,
            primary_y,
            secondary_x,
            secondary_y,
            primary_background_adu: primary_background,
            secondary_background_adu: secondary_background,
            global_psf_source_count: global_psf.as_ref().map_or(0, |psf| psf.source_count),
            local_psf_source_count: local_psf.as_ref().map_or(0, |psf| psf.source_count),
            local_template_available: local_psf.is_some(),
            primary_global_correlation: metric(primary_global, 0),
            primary_global_relative_residual: metric(primary_global, 1),
            primary_global_central_energy_fraction: metric(primary_global, 2),
            primary_global_fitted_amplitude_adu: metric(primary_global, 3),
            secondary_global_correlation: metric(secondary_global, 0),
            secondary_global_relative_residual: metric(secondary_global, 1),
            secondary_global_central_energy_fraction: metric(secondary_global, 2),
            secondary_global_fitted_amplitude_adu: metric(secondary_global, 3),
            primary_local_correlation: metric(primary_local, 0),
            primary_local_relative_residual: metric(primary_local, 1),
            primary_local_central_energy_fraction: metric(primary_local, 2),
            primary_local_fitted_amplitude_adu: metric(primary_local, 3),
            secondary_local_correlation: metric(secondary_local, 0),
            secondary_local_relative_residual: metric(secondary_local, 1),
            secondary_local_central_energy_fraction: metric(secondary_local, 2),
            secondary_local_fitted_amplitude_adu: metric(secondary_local, 3),
            note: note.to_owned(),
        });
    }
    let primary_global_median =
        median_optional(rows.iter().map(|row| row.primary_global_correlation));
    let secondary_global_median =
        median_optional(rows.iter().map(|row| row.secondary_global_correlation));
    let primary_global_residual =
        median_optional(rows.iter().map(|row| row.primary_global_relative_residual));
    let secondary_global_residual = median_optional(
        rows.iter()
            .map(|row| row.secondary_global_relative_residual),
    );
    let valid_global = rows
        .iter()
        .filter(|row| {
            row.primary_global_correlation.is_some() && row.secondary_global_correlation.is_some()
        })
        .count();
    let local_available = rows
        .iter()
        .filter(|row| row.local_template_available)
        .count();
    let conclusion = format!(
        "经验 PSF 对照完成：{} 帧；全局模板 {} 个，局部模板 {} 个。两个目标均得到全局形状指标的帧数为 {}/{}，局部模板可用帧数为 {}/{}。全局相关中位数 primary={}, secondary={}；相对残差中位数 primary={}, secondary={}。这些指标受混叠、背景估计、空间 PSF 变化和模板选择影响，只能支持形状复核，不能单独确认或否定星表身份。",
        rows.len(),
        global_psf.as_ref().map_or(0, |psf| psf.source_count),
        local_psf.as_ref().map_or(0, |psf| psf.source_count),
        valid_global,
        rows.len(),
        local_available,
        rows.len(),
        display_option(primary_global_median),
        display_option(secondary_global_median),
        display_option(primary_global_residual),
        display_option(secondary_global_residual),
    );
    Ok(PairAuditResult {
        frame_count: rows.len(),
        primary_detection_id: Some(primary.detection_id),
        secondary_detection_id: Some(secondary.detection_id),
        primary_target_xy: primary_xy,
        secondary_target_xy: secondary_xy,
        frame_shifts: shifts,
        template_frame_path: frame_paths[0].to_string_lossy().into_owned(),
        template_catalog_path: Some(catalog_path.to_string_lossy().into_owned()),
        support_radius,
        max_template_sources,
        grid_size,
        pair_cell: [pair_cell.0, pair_cell.1],
        local_padding_px: local_padding,
        global_psf,
        local_psf,
        rows,
        conclusion,
    })
}

fn display_option(value: Option<f64>) -> String {
    value.map_or_else(|| "None".to_owned(), |value| format!("{value:?}"))
}

fn psf_json(psf: &Option<EmpiricalPsf>) -> Value {
    let Some(psf) = psf else {
        return Value::Null;
    };
    psf.as_dict()
}

impl PairAuditResult {
    pub fn as_json(&self) -> Value {
        json!({
            "frame_count": self.frame_count,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "primary_target_xy": self.primary_target_xy,
            "secondary_target_xy": self.secondary_target_xy,
            "frame_shifts": self.frame_shifts,
            "template_frame_path": self.template_frame_path,
            "template_catalog_path": self.template_catalog_path,
            "support_radius": self.support_radius,
            "grid_size": self.grid_size,
            "local_padding_px": self.local_padding_px,
            "global_psf": psf_json(&self.global_psf),
            "local_psf": psf_json(&self.local_psf),
            "parameters": {
                "support_radius": self.support_radius,
                "max_template_sources": self.max_template_sources,
                "grid_size": self.grid_size,
                "pair_cell": self.pair_cell,
                "local_padding_px": self.local_padding_px,
                "background_estimator": "median of valid pixels outside the PSF support radius",
                "frame_shifts": self.frame_shifts,
                "interpretation_boundary": "empirical PSF shape diagnostic, not catalog identity or probability"
            },
            "rows": self.rows,
            "conclusion": self.conclusion,
            "note": "经验 PSF 由模板帧的质量通过、未饱和、未掩膜且隔离源构造；局部模板不足时保持 None。所有 correlation/residual/central_fraction 均为形状诊断，不是星表身份、恒星概率或独立去混叠通量。"
        })
    }
}

fn csv_float(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:?}"))
}

fn write_csv(path: &Path, rows: &[PairAuditRow]) -> Result<()> {
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
        let values = vec![
            row.frame_index.to_string(),
            row.path.clone(),
            csv_float(Some(row.frame_shift_x_px)),
            csv_float(Some(row.frame_shift_y_px)),
            csv_float(Some(row.primary_x)),
            csv_float(Some(row.primary_y)),
            csv_float(Some(row.secondary_x)),
            csv_float(Some(row.secondary_y)),
            csv_float(row.primary_background_adu),
            csv_float(row.secondary_background_adu),
            row.global_psf_source_count.to_string(),
            row.local_psf_source_count.to_string(),
            if row.local_template_available {
                "True"
            } else {
                "False"
            }
            .to_owned(),
            csv_float(row.primary_global_correlation),
            csv_float(row.primary_global_relative_residual),
            csv_float(row.primary_global_central_energy_fraction),
            csv_float(row.primary_global_fitted_amplitude_adu),
            csv_float(row.secondary_global_correlation),
            csv_float(row.secondary_global_relative_residual),
            csv_float(row.secondary_global_central_energy_fraction),
            csv_float(row.secondary_global_fitted_amplitude_adu),
            csv_float(row.primary_local_correlation),
            csv_float(row.primary_local_relative_residual),
            csv_float(row.primary_local_central_energy_fraction),
            csv_float(row.primary_local_fitted_amplitude_adu),
            csv_float(row.secondary_local_correlation),
            csv_float(row.secondary_local_relative_residual),
            csv_float(row.secondary_local_central_energy_fraction),
            csv_float(row.secondary_local_fitted_amplitude_adu),
            row.note.clone(),
        ];
        writer
            .write_record(values)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

pub fn write_artifacts(
    result: &PairAuditResult,
    out_dir: &Path,
    max_template_sources: usize,
) -> Result<Value> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    write_csv(&out_dir.join("empirical_pair_psf_audit.csv"), &result.rows)?;
    let mut payload = result.as_json();
    payload["parameters"]["max_template_sources"] = json!(max_template_sources);
    fs::write(
        out_dir.join("empirical_pair_psf_audit.json"),
        serde_json::to_vec_pretty(&payload).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    write_similarity_chart(
        &out_dir.join("empirical_pair_psf_similarity.png"),
        &result.rows,
    )?;
    Ok(json!({"output":out_dir,"frame_count":result.frame_count,"conclusion":result.conclusion}))
}

fn write_similarity_chart(path: &Path, rows: &[PairAuditRow]) -> Result<()> {
    const WIDTH: usize = 1200;
    const HEIGHT: usize = 680;
    let mut pixels = vec![255u8; WIDTH * HEIGHT * 3];
    let plot = (70usize, 40usize, WIDTH - 35, HEIGHT - 60);
    let mut line = |x0: usize, y0: usize, x1: usize, y1: usize, color: [u8; 3]| {
        let dx = (x1 as isize - x0 as isize).abs();
        let sx = if x0 < x1 { 1isize } else { -1 };
        let dy = -(y1 as isize - y0 as isize).abs();
        let sy = if y0 < y1 { 1isize } else { -1 };
        let mut error = dx + dy;
        let (mut x, mut y) = (x0 as isize, y0 as isize);
        loop {
            if x >= 0 && y >= 0 && (x as usize) < WIDTH && (y as usize) < HEIGHT {
                let index = (y as usize * WIDTH + x as usize) * 3;
                pixels[index..index + 3].copy_from_slice(&color);
            }
            if x == x1 as isize && y == y1 as isize {
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
    };
    let grid = [224, 230, 236];
    for step in 0..=5 {
        let y = plot.1 + (plot.3 - plot.1) * step / 5;
        line(plot.0, y, plot.2, y, grid);
    }
    line(plot.0, plot.1, plot.0, plot.3, [70, 83, 96]);
    line(plot.0, plot.3, plot.2, plot.3, [70, 83, 96]);
    let series = [
        (CorrelationSeries::PrimaryGlobal, [215, 148, 50]),
        (CorrelationSeries::SecondaryGlobal, [199, 53, 158]),
        (CorrelationSeries::PrimaryLocal, [79, 155, 131]),
        (CorrelationSeries::SecondaryLocal, [114, 185, 212]),
    ];
    let values = rows
        .iter()
        .flat_map(|row| series.iter().filter_map(|(kind, _)| kind.value(row)))
        .collect::<Vec<_>>();
    let (mut min_y, mut max_y) = values
        .iter()
        .copied()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| {
            (lo.min(value), hi.max(value))
        });
    if !min_y.is_finite() {
        min_y = 0.0;
        max_y = 1.0;
    }
    if max_y <= min_y {
        max_y = min_y + 1.0;
    }
    let padding = (max_y - min_y) * 0.08;
    min_y -= padding;
    max_y += padding;
    for (get, color) in series {
        let points = rows
            .iter()
            .enumerate()
            .filter_map(|(index, row)| {
                get.value(row)
                    .filter(|value| value.is_finite())
                    .map(|value| {
                        let x = if rows.len() <= 1 {
                            (plot.0 + plot.2) / 2
                        } else {
                            plot.0 + (plot.2 - plot.0) * index / (rows.len() - 1)
                        };
                        let ratio = ((value - min_y) / (max_y - min_y)).clamp(0.0, 1.0);
                        let y = plot.3 - (ratio * (plot.3 - plot.1) as f64).round() as usize;
                        (x, y)
                    })
            })
            .collect::<Vec<_>>();
        for pair in points.windows(2) {
            line(pair[0].0, pair[0].1, pair[1].0, pair[1].1, color);
        }
        for (x, y) in points {
            line(x.saturating_sub(3), y, (x + 3).min(WIDTH - 1), y, color);
            line(x, y.saturating_sub(3), x, (y + 3).min(HEIGHT - 1), color);
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
            let values = item
                .as_array()
                .ok_or("each cumulative shift must be [x,y]")?;
            if values.len() != 2 {
                return Err("each cumulative shift must be [x,y]".into());
            }
            let x = values[0].as_f64().ok_or("shift x must be numeric")?;
            let y = values[1].as_f64().ok_or("shift y must be numeric")?;
            if !x.is_finite() || !y.is_finite() {
                return Err("frame shifts must be finite".into());
            }
            Ok([x, y])
        })
        .collect()
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
    let position = |source: &Detection| {
        if options.coordinate_mode == "peak" {
            match (source.peak_x, source.peak_y) {
                (Some(x), Some(y)) => [x, y],
                _ => [source.x, source.y],
            }
        } else {
            [source.x, source.y]
        }
    };
    let primary_xy = position(primary);
    let secondary_xy = position(secondary);
    let shifts = options
        .shifts_json
        .as_deref()
        .map(read_shifts)
        .transpose()?;
    let result = run(
        &paths,
        &sources,
        primary,
        secondary,
        primary_xy,
        secondary_xy,
        &options.source_catalog,
        shifts.as_deref(),
        options.support_radius,
        options.max_template_sources,
        options.grid_size,
    )?;
    write_artifacts(&result, &options.out_dir, options.max_template_sources)
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
        include_str!("../tests/fixtures/empirical_pair_audit_parity/reference-output.json");
    const REFERENCE_CSV: &[u8] =
        include_bytes!("../tests/fixtures/empirical_pair_audit_parity/reference.csv");

    fn assert_close(actual: f64, expected: f64, label: &str) {
        let tolerance = 1e-8f64.max(expected.abs() * 1e-12);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
        );
    }

    fn assert_json_matches(actual: &Value, expected: &Value, label: &str) {
        match (actual, expected) {
            (Value::Object(actual), Value::Object(expected)) => {
                let mut actual_keys = actual.keys().collect::<Vec<_>>();
                let mut expected_keys = expected.keys().collect::<Vec<_>>();
                actual_keys.sort();
                expected_keys.sort();
                assert_eq!(actual_keys, expected_keys, "{label} keys");
                for key in expected.keys() {
                    let child_label = format!("{label}.{key}");
                    if matches!(
                        key.as_str(),
                        "path" | "template_frame_path" | "template_catalog_path"
                    ) {
                        continue;
                    }
                    assert_json_matches(&actual[key], &expected[key], &child_label);
                }
            }
            (Value::Array(actual), Value::Array(expected)) => {
                assert_eq!(actual.len(), expected.len(), "{label} length");
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert_json_matches(actual, expected, &format!("{label}[{index}]"));
                }
            }
            (Value::Number(actual), Value::Number(expected)) => {
                let actual = actual.as_f64().expect("actual JSON number");
                let expected = expected.as_f64().expect("reference JSON number");
                if label.ends_with(".kernel_sum") {
                    assert!(
                        (actual - expected).abs() <= 1e-6,
                        "{label}: actual={actual:?}, expected={expected:?}, tolerance=1e-6"
                    );
                } else {
                    assert_close(actual, expected, label);
                }
            }
            (Value::String(actual), Value::String(expected)) if label == "result.conclusion" => {
                let (actual_text, actual_numbers) = normalize_numbers(actual);
                let (expected_text, expected_numbers) = normalize_numbers(expected);
                assert_eq!(actual_text, expected_text, "{label} wording");
                assert_eq!(
                    actual_numbers.len(),
                    expected_numbers.len(),
                    "{label} values"
                );
                for (index, (actual, expected)) in
                    actual_numbers.iter().zip(&expected_numbers).enumerate()
                {
                    assert_close(*actual, *expected, &format!("{label} number {index}"));
                }
            }
            _ => assert_eq!(actual, expected, "{label}"),
        }
    }

    fn normalize_numbers(text: &str) -> (String, Vec<f64>) {
        let characters = text.chars().collect::<Vec<_>>();
        let mut normalized = String::with_capacity(text.len());
        let mut numbers = Vec::new();
        let mut index = 0;
        while index < characters.len() {
            let starts_number = characters[index].is_ascii_digit()
                || (characters[index] == '-'
                    && characters.get(index + 1).is_some_and(char::is_ascii_digit));
            if !starts_number {
                normalized.push(characters[index]);
                index += 1;
                continue;
            }
            let start = index;
            index += 1;
            while index < characters.len()
                && (characters[index].is_ascii_digit()
                    || matches!(characters[index], '.' | 'e' | 'E' | '+' | '-'))
            {
                index += 1;
            }
            let number = characters[start..index].iter().collect::<String>();
            numbers.push(number.parse().expect("numeric text in conclusion"));
            normalized.push('#');
        }
        (normalized, numbers)
    }

    fn test_output_dir() -> PathBuf {
        let root = std::env::temp_dir();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after Unix epoch")
            .as_nanos();
        let path = root.join(format!(
            "rst19-empirical-pair-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create isolated empirical-pair test output directory");
        path
    }

    #[test]
    fn parser_keeps_public_defaults_and_validates_modes() {
        let options = parse_args(&[
            "frames".into(),
            "--source-catalog".into(),
            "sources.csv".into(),
            "--primary-id".into(),
            "1".into(),
            "--secondary-id".into(),
            "2".into(),
        ])
        .unwrap();
        assert_eq!(options.coordinate_mode, "centroid");
        assert_eq!(options.support_radius, 7);
        assert_eq!(options.grid_size, 4);
        assert!(parse_args(&[
            "frames".into(),
            "--source-catalog".into(),
            "sources.csv".into(),
            "--primary-id".into(),
            "1".into(),
            "--secondary-id".into(),
            "2".into(),
            "--coordinate-mode".into(),
            "peakish".into(),
        ])
        .is_err());
    }

    #[test]
    fn pair_positions_and_radius_are_validated() {
        let sources = vec![Detection {
            detection_id: 1,
            x: 20.0,
            y: 20.0,
            peak: 1.0,
            flux: 1.0,
            background: 0.0,
            noise: 1.0,
            snr: 1.0,
            fwhm: Some(2.0),
            flags: vec![],
            flux_error: None,
            flux_snr: Some(10.0),
            filter_snr: None,
            fwhm_x: None,
            fwhm_y: None,
            ellipticity: Some(0.1),
            sharpness: None,
            footprint_pixels: Some(9),
            psf_support_pixels: Some(9),
            quality_passed: true,
            peak_x: Some(20.0),
            peak_y: Some(20.0),
            centroid_shift_px: None,
            proposal_methods: vec![],
            proposal_scales: vec![],
            proposal_snr: None,
            nearest_gaussian_px: None,
            deblend_delta_bic: None,
            deblend_component_snr: None,
            repeated_code_count: None,
            range_anomaly_pixel_count: None,
            repeated_code_values: vec![],
        }];
        let result = run(
            &[PathBuf::from("missing.fits")],
            &sources,
            &sources[0],
            &sources[0],
            [20.0, 20.0],
            [20.0, 20.0],
            Path::new("sources.csv"),
            None,
            7,
            64,
            4,
        );
        assert!(result.unwrap_err().contains("coordinates must differ"));
    }

    #[test]
    fn frozen_python_reference_matches_rust_json_csv_and_png_dimensions() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let data_dir = manifest.join("tests/fixtures/empirical_pair_audit_parity/frames");
        let catalog = manifest.join("tests/fixtures/source_audit_parity/source_catalog.csv");
        let output = test_output_dir();
        let args = [
            data_dir.to_string_lossy().into_owned(),
            "--source-catalog".into(),
            catalog.to_string_lossy().into_owned(),
            "--primary-id".into(),
            "0".into(),
            "--secondary-id".into(),
            "1".into(),
            "--out-dir".into(),
            output.to_string_lossy().into_owned(),
        ];

        run_desktop(&args).expect("run native empirical pair audit");

        let actual_json: Value = serde_json::from_slice(
            &fs::read(output.join("empirical_pair_psf_audit.json")).expect("read Rust JSON"),
        )
        .expect("parse Rust JSON");
        let expected_json: Value = serde_json::from_str(REFERENCE_JSON).expect("parse Python JSON");
        assert_json_matches(&actual_json, &expected_json, "result");

        let actual_csv =
            fs::read(output.join("empirical_pair_psf_audit.csv")).expect("read Rust CSV");
        let actual_csv = actual_csv
            .strip_prefix(&[0xef, 0xbb, 0xbf])
            .unwrap_or(&actual_csv);
        let reference_csv = REFERENCE_CSV
            .strip_prefix(&[0xef, 0xbb, 0xbf])
            .unwrap_or(REFERENCE_CSV);
        let mut actual_reader = csv::Reader::from_reader(actual_csv);
        let mut expected_reader = csv::Reader::from_reader(reference_csv);
        let actual_headers = actual_reader.headers().expect("Rust CSV headers").clone();
        let expected_headers = expected_reader
            .headers()
            .expect("Python CSV headers")
            .clone();
        assert_eq!(actual_headers, expected_headers, "CSV headers");
        let actual_rows = actual_reader
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("Rust CSV rows");
        let expected_rows = expected_reader
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("Python CSV rows");
        assert_eq!(actual_rows.len(), expected_rows.len(), "CSV row count");
        for (row_index, (actual, expected)) in actual_rows.iter().zip(&expected_rows).enumerate() {
            for column_index in 0..actual.len() {
                let actual_value = &actual[column_index];
                let expected_value = &expected[column_index];
                let field = &actual_headers[column_index];
                let label = format!("CSV row {row_index} field {field}");
                if field == "path" || actual_value == expected_value {
                    continue;
                }
                match (actual_value.parse::<f64>(), expected_value.parse::<f64>()) {
                    (Ok(actual), Ok(expected)) => assert_close(actual, expected, &label),
                    _ => assert_eq!(actual_value, expected_value, "{label}"),
                }
            }
        }

        let chart = File::open(output.join("empirical_pair_psf_similarity.png"))
            .expect("open Rust similarity chart");
        let decoder = png::Decoder::new(BufReader::new(chart));
        let reader = decoder.read_info().expect("read Rust PNG header");
        assert_eq!(reader.info().width, 1200);
        assert_eq!(reader.info().height, 680);
        fs::remove_dir_all(output).expect("remove isolated empirical-pair test output");
    }
}
