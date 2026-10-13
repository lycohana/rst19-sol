//! Native empirical-PSF aperture sensitivity audit.
//!
//! This keeps the original experiment boundary: one real background, fixed
//! injection coordinates and signal, varying only the aperture used by the
//! detector. Recovery rates are conditional injection measurements, not the
//! completeness or physical-star fraction of the source image.

use crate::{
    detection::DetectionOptions,
    empirical_psf::{self, EmpiricalPsf},
    fits,
    models::{FitsFrame, PixelData},
    pipeline::{self, FrameOptions},
    source_geometry_audit::NumpyPcg64,
    Result,
};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

const ROW_FIELDS: [&str; 32] = [
    "source_path",
    "grid_size",
    "cell_id",
    "x",
    "y",
    "local_background_adu",
    "local_noise_adu",
    "aperture_radius",
    "reference_aperture_radius",
    "control_signal_adu",
    "injected_peak_excess_adu",
    "paired_control_candidate_count",
    "paired_control_quality_count",
    "paired_control_filter_noise",
    "candidate_count",
    "quality_count",
    "candidate_delta_vs_control",
    "quality_delta_vs_control",
    "filter_noise",
    "filter_noise_delta_vs_control",
    "candidate_recovered",
    "quality_recovered",
    "nearest_candidate_distance_px",
    "detection_id",
    "peak_snr",
    "flux_snr",
    "filter_snr",
    "fwhm",
    "psf_support_pixels",
    "quality_passed",
    "flags",
    "note",
];

const NOTE: &str = "同一经验 PSF、同一离散积分注入和同一空间位置；只改变测光孔径。candidate/quality 是注入真值回收，不是当前 FITS 的恒星完备率。";
const CONCLUSION_PREFIX: &str = "经验 PSF 孔径敏感性完成：";
const CONCLUSION_MIDDLE: &str = " 个固定位置、";
const CONCLUSION_SUFFIX: &str = " 个孔径；质量回收率最高的本次 pilot 孔径为 r=";
const CONCLUSION_TAIL: &str = "，该选择只描述当前信号、背景和模板条件，不是普适最优孔径。扩大孔径会同时改变通量截取、背景方差、形状测量和质量源计数，因此不能把质量回收变化直接解释为真实恒星数变化。";

#[derive(Clone, Debug)]
pub struct Options {
    pub aperture_radii: Vec<usize>,
    pub reference_aperture_radius: usize,
    pub control_signal_adu: f64,
    pub grid_size: usize,
    pub psf_fwhm: f64,
    pub proposal_mode: String,
    pub threshold_sigma: f64,
    pub min_distance: usize,
    pub background_box_size: usize,
    pub min_flux_snr: f64,
    pub min_psf_support_pixels: usize,
    pub max_sources: Option<usize>,
    pub empirical_psf_radius: usize,
    pub empirical_psf_sources: usize,
    pub seed: u128,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            aperture_radii: vec![3, 4, 5, 6],
            reference_aperture_radius: 4,
            control_signal_adu: 512.0,
            grid_size: 2,
            psf_fwhm: 2.0,
            proposal_mode: "hybrid".into(),
            threshold_sigma: 4.0,
            min_distance: 4,
            background_box_size: 128,
            min_flux_snr: 5.0,
            min_psf_support_pixels: 3,
            max_sources: None,
            empirical_psf_radius: 7,
            empirical_psf_sources: 64,
            seed: 19_019,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Position {
    pub cell_id: String,
    pub x: f64,
    pub y: f64,
    pub local_background_adu: Option<f64>,
    pub local_noise_adu: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Row {
    pub source_path: String,
    pub grid_size: usize,
    pub cell_id: String,
    pub x: f64,
    pub y: f64,
    pub local_background_adu: Option<f64>,
    pub local_noise_adu: Option<f64>,
    pub aperture_radius: usize,
    pub reference_aperture_radius: usize,
    pub control_signal_adu: f64,
    pub injected_peak_excess_adu: f64,
    pub paired_control_candidate_count: usize,
    pub paired_control_quality_count: usize,
    pub paired_control_filter_noise: f64,
    pub candidate_count: usize,
    pub quality_count: usize,
    pub candidate_delta_vs_control: i64,
    pub quality_delta_vs_control: i64,
    pub filter_noise: f64,
    pub filter_noise_delta_vs_control: f64,
    pub candidate_recovered: bool,
    pub quality_recovered: bool,
    pub nearest_candidate_distance_px: Option<f64>,
    pub detection_id: Option<i64>,
    pub peak_snr: Option<f64>,
    pub flux_snr: Option<f64>,
    pub filter_snr: Option<f64>,
    pub fwhm: Option<f64>,
    pub psf_support_pixels: Option<usize>,
    pub quality_passed: Option<bool>,
    pub flags: Vec<String>,
    pub note: String,
}

impl Row {
    fn csv_cells(&self) -> Vec<String> {
        vec![
            self.source_path.clone(),
            self.grid_size.to_string(),
            self.cell_id.clone(),
            csv_float(self.x),
            csv_float(self.y),
            csv_option_float(self.local_background_adu),
            csv_option_float(self.local_noise_adu),
            self.aperture_radius.to_string(),
            self.reference_aperture_radius.to_string(),
            csv_float(self.control_signal_adu),
            csv_float(self.injected_peak_excess_adu),
            self.paired_control_candidate_count.to_string(),
            self.paired_control_quality_count.to_string(),
            csv_float(self.paired_control_filter_noise),
            self.candidate_count.to_string(),
            self.quality_count.to_string(),
            self.candidate_delta_vs_control.to_string(),
            self.quality_delta_vs_control.to_string(),
            csv_float(self.filter_noise),
            csv_float(self.filter_noise_delta_vs_control),
            python_bool(self.candidate_recovered).into(),
            python_bool(self.quality_recovered).into(),
            csv_option_float(self.nearest_candidate_distance_px),
            self.detection_id
                .map_or_else(String::new, |value| value.to_string()),
            csv_option_float(self.peak_snr),
            csv_option_float(self.flux_snr),
            csv_option_float(self.filter_snr),
            csv_option_float(self.fwhm),
            self.psf_support_pixels
                .map_or_else(String::new, |value| value.to_string()),
            self.quality_passed
                .map_or_else(String::new, |value| python_bool(value).into()),
            python_list(&self.flags),
            self.note.clone(),
        ]
    }

    fn as_value(&self) -> Value {
        serde_json::to_value(self).expect("serializing an aperture sensitivity row cannot fail")
    }
}

#[derive(Clone, Debug)]
pub struct ResultData {
    pub source_path: String,
    pub image_shape: [usize; 2],
    pub grid_size: usize,
    pub aperture_radii: Vec<usize>,
    pub reference_aperture_radius: usize,
    pub control_signal_adu: f64,
    pub baseline_candidate_count: usize,
    pub baseline_quality_count: usize,
    pub psf_model: String,
    pub psf_source_count: usize,
    pub psf_support_radius: usize,
    pub psf_median_fwhm_px: Option<f64>,
    pub psf_kernel_sum: f64,
    pub psf_encircled_energy_by_radius: Map<String, Value>,
    pub parameters: Map<String, Value>,
    pub positions: Vec<Position>,
    pub rows: Vec<Row>,
    pub summary_by_aperture: Vec<Value>,
    pub conclusion: String,
}

impl ResultData {
    pub fn as_value(&self) -> Value {
        json!({
            "source_path": self.source_path,
            "image_shape": self.image_shape,
            "grid_size": self.grid_size,
            "aperture_radii": self.aperture_radii,
            "reference_aperture_radius": self.reference_aperture_radius,
            "control_signal_adu": self.control_signal_adu,
            "baseline_candidate_count": self.baseline_candidate_count,
            "baseline_quality_count": self.baseline_quality_count,
            "psf_model": self.psf_model,
            "psf_source_count": self.psf_source_count,
            "psf_support_radius": self.psf_support_radius,
            "psf_median_fwhm_px": self.psf_median_fwhm_px,
            "psf_kernel_sum": self.psf_kernel_sum,
            "psf_encircled_energy_by_radius": self.psf_encircled_energy_by_radius,
            "parameters": self.parameters,
            "positions": self.positions.iter().map(position_value).collect::<Vec<_>>(),
            "rows": self.rows.iter().map(Row::as_value).collect::<Vec<_>>(),
            "summary_by_aperture": self.summary_by_aperture,
            "conclusion": self.conclusion,
        })
    }
}

fn position_value(position: &Position) -> Value {
    json!({
        "cell_id": position.cell_id,
        "x": position.x,
        "y": position.y,
        "local_background_adu": position.local_background_adu,
        "local_noise_adu": position.local_noise_adu,
    })
}

fn finite_option(value: Option<f64>) -> Option<f64> {
    value.filter(|number| number.is_finite())
}

fn numeric_parameter(parameters: &Value, key: &str) -> Option<f64> {
    parameters
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
}

fn image_dtype(data: &PixelData) -> &'static str {
    match data {
        PixelData::U8(_) => "uint8",
        PixelData::I16(_) => ">i2",
        PixelData::I32(_) => ">i4",
        PixelData::I64(_) => ">i8",
        PixelData::F32(_) => ">f4",
        PixelData::F64(_) => ">f8",
    }
}

fn negative_overflow_limit(data: &PixelData) -> Option<f64> {
    match data {
        PixelData::I16(_) => Some(-0.9 * f64::from(i16::MAX)),
        PixelData::I32(_) => Some(-0.9 * f64::from(i32::MAX)),
        PixelData::I64(_) => Some(-0.9 * i64::MAX as f64),
        _ => None,
    }
}

fn special_value_mask(data: &PixelData) -> Vec<bool> {
    let negative_limit = negative_overflow_limit(data);
    (0..data.len())
        .map(|index| {
            let value = data.value(index);
            if !value.is_finite() {
                return false;
            }
            value == -1.0
                || negative_limit.is_some_and(|limit| value <= limit)
                || match data {
                    PixelData::I16(_) => value >= f64::from(i16::MAX - 32),
                    PixelData::I32(_) => value >= f64::from(i32::MAX - 32),
                    PixelData::I64(_) => value >= (i64::MAX - 32) as f64,
                    _ => false,
                }
        })
        .collect()
}

fn sorted_median(values: &mut [f64]) -> Option<f64> {
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

fn linear_quantile(values: &mut [f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let position = quantile * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    Some(values[lower] + (values[upper] - values[lower]) * (position - lower as f64))
}

fn local_patch_stats(
    data: &PixelData,
    height: usize,
    width: usize,
    x: f64,
    y: f64,
) -> Option<(f64, f64, f64)> {
    let center_x = x.round_ties_even() as isize;
    let center_y = y.round_ties_even() as isize;
    let x0 = (center_x - 5).max(0) as usize;
    let y0 = (center_y - 5).max(0) as usize;
    let x1 = (center_x + 6).min(width as isize).max(0) as usize;
    let y1 = (center_y + 6).min(height as isize).max(0) as usize;
    let mut finite = Vec::with_capacity((x1 - x0) * (y1 - y0));
    for row in y0..y1 {
        for column in x0..x1 {
            let value = data.value(row * width + column);
            if value.is_finite() {
                finite.push(value);
            }
        }
    }
    if finite.len() < 16 {
        return None;
    }
    let median = sorted_median(&mut finite.clone())?;
    let mut deviations = finite
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    let mad_noise = (1.4826 * sorted_median(&mut deviations)?).max(f64::EPSILON);
    let upper_excess = linear_quantile(&mut finite, 0.98)? - median;
    Some((median, mad_noise, upper_excess))
}

#[derive(Clone, Debug)]
struct Site {
    position: Position,
}

fn cell_bounds(
    height: usize,
    width: usize,
    grid: usize,
    cell_x: usize,
    cell_y: usize,
) -> [usize; 4] {
    [
        width * cell_x / grid,
        height * cell_y / grid,
        width * (cell_x + 1) / grid,
        height * (cell_y + 1) / grid,
    ]
}

fn select_positions(
    frame: &FitsFrame,
    baseline: &pipeline::FrameAnalysis,
    options: &Options,
) -> Result<Vec<Position>> {
    let height = frame.height;
    let width = frame.width;
    let image = &frame.data;
    let baseline_sources = &baseline.detection.sources;
    let neighborhood_radius = 8usize.max((2.0 * options.psf_fwhm).ceil() as usize);
    let source_exclusion_radius = 12.0_f64.max(2.0 * options.psf_fwhm);
    let minimum_distance = 10.0_f64.max(2.5 * options.psf_fwhm);
    let interior_margin = (options.reference_aperture_radius + 4)
        .max((4.0 * options.psf_fwhm / 2.354820045).ceil() as usize + 2);
    let special_mask = special_value_mask(image);
    let mut rng = NumpyPcg64::new(options.seed);
    let mut output = Vec::with_capacity(options.grid_size * options.grid_size);

    for cell_y in 0..options.grid_size {
        for cell_x in 0..options.grid_size {
            let [x0, y0, x1, y1] = cell_bounds(height, width, options.grid_size, cell_x, cell_y);
            if x1.saturating_sub(x0) <= 2 * interior_margin
                || y1.saturating_sub(y0) <= 2 * interior_margin
            {
                return Err("spatial cell is too small for the requested injection margins".into());
            }
            // Python requests one site here and samples max(64, 1 * 80) candidates.
            let target_pool = 80usize;
            let attempts_limit = 20_000usize.max(target_pool * 160);
            let mut pool = Vec::<Site>::with_capacity(target_pool);
            for _ in 0..attempts_limit {
                if pool.len() >= target_pool {
                    break;
                }
                let x = rng.uniform((x0 + interior_margin) as f64, (x1 - interior_margin) as f64);
                let y = rng.uniform((y0 + interior_margin) as f64, (y1 - interior_margin) as f64);
                if baseline_sources
                    .iter()
                    .any(|source| (source.x - x).hypot(source.y - y) <= source_exclusion_radius)
                {
                    continue;
                }
                let Some((local_background, local_noise, upper_excess)) =
                    local_patch_stats(image, height, width, x, y)
                else {
                    continue;
                };
                let center_x = x.round_ties_even() as usize;
                let center_y = y.round_ties_even() as usize;
                let sx0 = center_x.saturating_sub(neighborhood_radius);
                let sy0 = center_y.saturating_sub(neighborhood_radius);
                let sx1 = (center_x + neighborhood_radius + 1).min(width);
                let sy1 = (center_y + neighborhood_radius + 1).min(height);
                let local_area = (sx1 - sx0) * (sy1 - sy0);
                let special_count = (sy0..sy1)
                    .flat_map(|row| (sx0..sx1).map(move |column| row * width + column))
                    .filter(|index| special_mask[*index])
                    .count();
                let special_fraction = if local_area == 0 {
                    0.0
                } else {
                    special_count as f64 / local_area as f64
                };
                if special_fraction > 0.0
                    || upper_excess > (3.0 * local_noise).max(3.0 * baseline.detection.noise)
                {
                    continue;
                }
                pool.push(Site {
                    position: Position {
                        cell_id: format!("r{cell_y}c{cell_x}"),
                        x,
                        y,
                        local_background_adu: Some(local_background),
                        local_noise_adu: Some(local_noise),
                    },
                });
            }
            if pool.is_empty() {
                return Err(format!(
                    "空间单元 ({x0},{y0}) 合规位置不足：仅找到 0 个，需要 1 个"
                ));
            }
            let order = rng.permutation(pool.len());
            let mut selected = Vec::<Site>::new();
            for index in order {
                let site = pool[index].clone();
                if selected.iter().all(|other| {
                    (site.position.x - other.position.x).hypot(site.position.y - other.position.y)
                        >= minimum_distance
                }) {
                    selected.push(site);
                    if selected.len() == 1 {
                        break;
                    }
                }
            }
            if selected.len() != 1 {
                return Err(format!(
                    "空间单元 ({x0},{y0}) 无法得到 1 个互相分离的注入位置，仅找到 {} 个",
                    selected.len()
                ));
            }
            output.push(selected.remove(0).position);
        }
    }
    Ok(output)
}

fn normalize_positions(positions: &[Position]) -> Result<Vec<Position>> {
    if positions.is_empty() {
        return Err("at least one aperture sensitivity position is required".into());
    }
    positions
        .iter()
        .map(|position| {
            if !position.x.is_finite() || !position.y.is_finite() {
                return Err("aperture sensitivity positions must be finite".into());
            }
            Ok(Position {
                cell_id: position.cell_id.clone(),
                x: position.x,
                y: position.y,
                local_background_adu: finite_option(position.local_background_adu),
                local_noise_adu: finite_option(position.local_noise_adu),
            })
        })
        .collect()
}

fn empirical_kernel_patch(
    image: &mut [f32],
    height: usize,
    width: usize,
    x: f64,
    y: f64,
    psf: &EmpiricalPsf,
    integrated_signal: f64,
) -> Result<f64> {
    if !x.is_finite()
        || !y.is_finite()
        || !integrated_signal.is_finite()
        || integrated_signal <= 0.0
    {
        return Err("injection position and signal must be finite, with a positive signal".into());
    }
    let radius = psf.support_radius;
    let side = psf.side();
    let center_x = x.round_ties_even() as isize;
    let center_y = y.round_ties_even() as isize;
    let shift_x = x - center_x as f64;
    let shift_y = y - center_y as f64;
    let mut shifted = vec![0.0_f32; psf.kernel.len()];
    for out_y in 0..side {
        for out_x in 0..side {
            let source_x = out_x as f64 - shift_x;
            let source_y = out_y as f64 - shift_y;
            if source_x < 0.0
                || source_y < 0.0
                || source_x > (side - 1) as f64
                || source_y > (side - 1) as f64
            {
                continue;
            }
            let x0 = source_x.floor() as usize;
            let y0 = source_y.floor() as usize;
            let x1 = (x0 + 1).min(side - 1);
            let y1 = (y0 + 1).min(side - 1);
            let dx = source_x - x0 as f64;
            let dy = source_y - y0 as f64;
            let top = f64::from(psf.kernel[y0 * side + x0]) * (1.0 - dx)
                + f64::from(psf.kernel[y0 * side + x1]) * dx;
            let bottom = f64::from(psf.kernel[y1 * side + x0]) * (1.0 - dx)
                + f64::from(psf.kernel[y1 * side + x1]) * dx;
            shifted[out_y * side + out_x] = (top * (1.0 - dy) + bottom * dy) as f32;
        }
    }
    let y0 = (center_y - radius as isize).max(0) as usize;
    let x0 = (center_x - radius as isize).max(0) as usize;
    let y1 = (center_y + radius as isize + 1).clamp(0, height as isize) as usize;
    let x1 = (center_x + radius as isize + 1).clamp(0, width as isize) as usize;
    let ky0 = y0 as isize - (center_y - radius as isize);
    let kx0 = x0 as isize - (center_x - radius as isize);
    let mut kernel_sum = 0.0_f64;
    for row in y0..y1 {
        for column in x0..x1 {
            let kernel_y = (ky0 + (row - y0) as isize) as usize;
            let kernel_x = (kx0 + (column - x0) as isize) as usize;
            kernel_sum += f64::from(shifted[kernel_y * side + kernel_x]);
        }
    }
    if !kernel_sum.is_finite() || kernel_sum <= 0.0 {
        return Err("injection PSF kernel must have a finite positive sum".into());
    }
    let peak_excess = integrated_signal / kernel_sum;
    for row in y0..y1 {
        for column in x0..x1 {
            let kernel_y = (ky0 + (row - y0) as isize) as usize;
            let kernel_x = (kx0 + (column - x0) as isize) as usize;
            let index = row * width + column;
            image[index] += peak_excess as f32 * shifted[kernel_y * side + kernel_x];
        }
    }
    Ok(peak_excess)
}

#[allow(clippy::field_reassign_with_default, clippy::too_many_arguments)]
fn run_detection(
    frame: &FitsFrame,
    data: PixelData,
    options: &Options,
    aperture_radius: usize,
    control_saturation: Option<f64>,
    control_negative_limit: Option<f64>,
    mask_zero_pixels: Option<bool>,
    use_float32: bool,
) -> Result<pipeline::FrameAnalysis> {
    let mut detector = DetectionOptions::default();
    detector.threshold_sigma = options.threshold_sigma;
    detector.min_distance = options.min_distance;
    detector.max_sources = options.max_sources;
    detector.background_box_size = options.background_box_size;
    detector.proposal_mode = options.proposal_mode.clone();
    detector.mask_zero_pixels = mask_zero_pixels;
    detector.measurement.aperture_radius = aperture_radius;
    detector.measurement.psf_fwhm = options.psf_fwhm;
    detector.measurement.min_flux_snr = options.min_flux_snr;
    detector.measurement.min_psf_support_pixels = options.min_psf_support_pixels;
    detector.measurement.saturation_level = control_saturation;
    detector.measurement.negative_overflow_limit = control_negative_limit;
    detector.use_float32 = use_float32;
    let mut frame = frame.clone();
    frame.data = data;
    pipeline::analyze_frame(
        frame,
        None,
        None,
        &FrameOptions {
            detection: detector,
            ..FrameOptions::default()
        },
        None,
    )
}

fn closest_source(
    sources: &[crate::measurement::Detection],
    x: f64,
    y: f64,
    radius: f64,
) -> (Option<&crate::measurement::Detection>, Option<f64>) {
    let closest = sources
        .iter()
        .enumerate()
        .filter_map(|(index, source)| {
            let distance = (source.x - x).hypot(source.y - y);
            distance.is_finite().then_some((distance, index, source))
        })
        .min_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
    match closest {
        Some((distance, _, source)) if distance <= radius => (Some(source), Some(distance)),
        Some((distance, _, _)) => (None, Some(distance)),
        None => (None, None),
    }
}

fn validate(options: &Options) -> Result<Vec<usize>> {
    if options.grid_size < 1 {
        return Err("grid_size must be positive".into());
    }
    if options.reference_aperture_radius < 1 {
        return Err("reference_aperture_radius must be positive".into());
    }
    let mut radii = options.aperture_radii.clone();
    radii.sort_unstable();
    radii.dedup();
    if radii.is_empty() || radii.iter().any(|radius| *radius < 1) {
        return Err("aperture_radii must contain positive integers".into());
    }
    if !options.control_signal_adu.is_finite() || options.control_signal_adu <= 0.0 {
        return Err("control_signal_adu must be finite and positive".into());
    }
    if !options.psf_fwhm.is_finite()
        || options.psf_fwhm <= 0.0
        || !options.threshold_sigma.is_finite()
        || options.threshold_sigma <= 0.0
        || options.min_distance < 1
    {
        return Err("detector parameters must be positive".into());
    }
    if options.background_box_size < 16
        || !options.min_flux_snr.is_finite()
        || options.min_flux_snr <= 0.0
    {
        return Err("background_box_size/min_flux_snr are invalid".into());
    }
    if !(1..=9).contains(&options.min_psf_support_pixels) {
        return Err("min_psf_support_pixels must be between 1 and 9".into());
    }
    if options.empirical_psf_radius < 3 || options.empirical_psf_sources < 1 {
        return Err(
            "empirical PSF radius must be at least 3 and source count must be positive".into(),
        );
    }
    Ok(radii)
}

/// Run the empirical-PSF aperture audit. When `fixed_positions` is `None`,
/// the original NumPy-PCG64 blank-site selection is reproduced.
pub fn run(
    frame_path: &Path,
    options: &Options,
    fixed_positions: Option<&[Position]>,
) -> Result<ResultData> {
    let radii = validate(options)?;
    let frame = fits::read_fits(frame_path, false)?;
    if frame.width == 0 || frame.height == 0 {
        return Err("expected a non-empty 2-D FITS image".into());
    }
    let baseline = run_detection(
        &frame,
        frame.data.clone(),
        options,
        options.reference_aperture_radius,
        None,
        None,
        None,
        false,
    )?;
    let empirical_psf = match &frame.data {
        PixelData::F32(values) => empirical_psf::estimate(
            values.as_slice().into(),
            frame.height,
            frame.width,
            &baseline.detection.sources,
            None,
            options.empirical_psf_radius,
            options.empirical_psf_sources,
            &[],
        )?,
        PixelData::F64(values) => empirical_psf::estimate(
            values.as_slice().into(),
            frame.height,
            frame.width,
            &baseline.detection.sources,
            None,
            options.empirical_psf_radius,
            options.empirical_psf_sources,
            &[],
        )?,
        _ => {
            let values = frame.data.to_f64();
            empirical_psf::estimate(
                values.as_slice().into(),
                frame.height,
                frame.width,
                &baseline.detection.sources,
                None,
                options.empirical_psf_radius,
                options.empirical_psf_sources,
                &[],
            )?
        }
    }
    .ok_or("真实质量源不足，无法建立实测 PSF")?;
    let positions = if let Some(positions) = fixed_positions {
        normalize_positions(positions)?
    } else {
        select_positions(&frame, &baseline, options)?
    };
    for position in &positions {
        if position.x < 0.0
            || position.y < 0.0
            || position.x > (frame.width - 1) as f64
            || position.y > (frame.height - 1) as f64
        {
            return Err(format!(
                "aperture sensitivity position {} is outside the image",
                position.cell_id
            ));
        }
    }
    let saturation = numeric_parameter(&baseline.detection.parameters, "saturation_level")
        .filter(|value| *value > 0.0);
    let negative_limit =
        numeric_parameter(&baseline.detection.parameters, "negative_overflow_limit")
            .or_else(|| negative_overflow_limit(&frame.data));
    let mask_zero_pixels = baseline.detection.parameters["mask_zero_pixels"]
        .as_bool()
        .or_else(|| {
            baseline.detection.parameters["mask_zero_pixels"]
                .as_i64()
                .map(|n| n != 0)
        })
        .unwrap_or(false);
    let energy = encircled_energy(&empirical_psf)?;
    let mut rows = Vec::with_capacity(radii.len() * positions.len());
    let mut summaries = Vec::with_capacity(radii.len());
    let recovery_radius = 3.0_f64.max(options.psf_fwhm);
    let original_path = frame.path.to_string_lossy().into_owned();
    let baseline_dtype = image_dtype(&frame.data);
    let baseline_negative_limit = negative_overflow_limit(&frame.data);
    for aperture_radius in &radii {
        let control = run_detection(
            &frame,
            PixelData::F32(
                frame
                    .data
                    .to_f64()
                    .into_iter()
                    .map(|value| value as f32)
                    .collect(),
            ),
            options,
            *aperture_radius,
            saturation,
            negative_limit,
            Some(mask_zero_pixels),
            true,
        )?;
        let control_filter_noise = numeric_parameter(&control.detection.parameters, "filter_noise")
            .ok_or("检测结果缺少有限的 filter_noise")?;
        let mut aperture_rows = Vec::with_capacity(positions.len());
        for position in &positions {
            let mut image = frame
                .data
                .to_f64()
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            let injected_peak = empirical_kernel_patch(
                &mut image,
                frame.height,
                frame.width,
                position.x,
                position.y,
                &empirical_psf,
                options.control_signal_adu,
            )?;
            let detected = run_detection(
                &frame,
                PixelData::F32(image),
                options,
                *aperture_radius,
                saturation,
                negative_limit,
                Some(mask_zero_pixels),
                true,
            )?;
            let (candidate, distance) = closest_source(
                &detected.detection.sources,
                position.x,
                position.y,
                recovery_radius,
            );
            let quality_sources = detected
                .detection
                .sources
                .iter()
                .filter(|source| source.quality_passed)
                .cloned()
                .collect::<Vec<_>>();
            let (quality, _) =
                closest_source(&quality_sources, position.x, position.y, recovery_radius);
            let filter_noise = numeric_parameter(&detected.detection.parameters, "filter_noise")
                .ok_or("注入检测结果缺少有限的 filter_noise")?;
            let row = Row {
                source_path: original_path.clone(),
                grid_size: options.grid_size,
                cell_id: position.cell_id.clone(),
                x: position.x,
                y: position.y,
                local_background_adu: position.local_background_adu,
                local_noise_adu: position.local_noise_adu,
                aperture_radius: *aperture_radius,
                reference_aperture_radius: options.reference_aperture_radius,
                control_signal_adu: options.control_signal_adu,
                injected_peak_excess_adu: injected_peak,
                paired_control_candidate_count: control.detection.candidate_count,
                paired_control_quality_count: control.detection.quality_count,
                paired_control_filter_noise: control_filter_noise,
                candidate_count: detected.detection.candidate_count,
                quality_count: detected.detection.quality_count,
                candidate_delta_vs_control: detected.detection.candidate_count as i64
                    - control.detection.candidate_count as i64,
                quality_delta_vs_control: detected.detection.quality_count as i64
                    - control.detection.quality_count as i64,
                filter_noise,
                filter_noise_delta_vs_control: filter_noise - control_filter_noise,
                candidate_recovered: candidate.is_some(),
                quality_recovered: quality.is_some(),
                nearest_candidate_distance_px: distance,
                detection_id: candidate.map(|source| source.detection_id),
                peak_snr: candidate.and_then(|source| finite_option(Some(source.snr))),
                flux_snr: candidate.and_then(|source| finite_option(source.flux_snr)),
                filter_snr: candidate.and_then(|source| finite_option(source.filter_snr)),
                fwhm: candidate.and_then(|source| finite_option(source.fwhm)),
                psf_support_pixels: candidate.and_then(|source| source.psf_support_pixels),
                quality_passed: candidate.map(|source| source.quality_passed),
                flags: candidate.map_or_else(Vec::new, |source| source.flags.clone()),
                note: NOTE.into(),
            };
            aperture_rows.push(row.clone());
            rows.push(row);
        }
        let flux_values = aperture_rows
            .iter()
            .filter_map(|row| row.flux_snr)
            .collect::<Vec<_>>();
        let fwhm_values = aperture_rows
            .iter()
            .filter_map(|row| row.fwhm)
            .collect::<Vec<_>>();
        let candidate_recovered_count = aperture_rows
            .iter()
            .filter(|row| row.candidate_recovered)
            .count();
        let quality_recovered_count = aperture_rows
            .iter()
            .filter(|row| row.quality_recovered)
            .count();
        let mut flags_by_position = Map::new();
        for row in &aperture_rows {
            flags_by_position.insert(row.cell_id.clone(), json!(row.flags));
        }
        summaries.push(json!({
            "aperture_radius": aperture_radius,
            "paired_control_candidate_count": control.detection.candidate_count,
            "paired_control_quality_count": control.detection.quality_count,
            "paired_control_filter_noise": control_filter_noise,
            "candidate_recovered_count": candidate_recovered_count,
            "quality_recovered_count": quality_recovered_count,
            "injected_count": aperture_rows.len(),
            "candidate_recall": candidate_recovered_count as f64 / aperture_rows.len() as f64,
            "quality_recall": quality_recovered_count as f64 / aperture_rows.len() as f64,
            "median_candidate_flux_snr": median_value(flux_values),
            "median_candidate_fwhm_px": median_value(fwhm_values),
            "quality_flags_by_position": flags_by_position,
        }));
    }
    let best_radius = summaries
        .iter()
        .max_by(|left, right| {
            left["quality_recall"]
                .as_f64()
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(
                    &right["quality_recall"]
                        .as_f64()
                        .unwrap_or(f64::NEG_INFINITY),
                )
                .then_with(|| {
                    right["aperture_radius"]
                        .as_u64()
                        .cmp(&left["aperture_radius"].as_u64())
                })
        })
        .and_then(|item| item["aperture_radius"].as_u64())
        .unwrap_or(radii[0] as u64);
    let radius_count = radii.len();
    let position_count = positions.len();
    let parameters = json!({
        "reference_detector_aperture_radius": options.reference_aperture_radius,
        "aperture_radii": radii,
        "control_signal_adu": options.control_signal_adu,
        "signal_normalization": "integrated_excess",
        "psf_model": "empirical",
        "psf_fwhm_for_detector": options.psf_fwhm,
        "proposal_mode": options.proposal_mode,
        "threshold_sigma": options.threshold_sigma,
        "min_distance": options.min_distance,
        "background_box_size": options.background_box_size,
        "min_flux_snr": options.min_flux_snr,
        "min_psf_support_pixels": options.min_psf_support_pixels,
        "empirical_psf_radius": options.empirical_psf_radius,
        "empirical_psf_sources": options.empirical_psf_sources,
        "seed": options.seed,
        "position_count": position_count,
        "position_source": if fixed_positions.is_some() { "provided" } else { "generated_with_spatial_blank_selector" },
        "paired_control_dtype": "float32",
        "baseline_dtype": baseline_dtype,
        "baseline_negative_overflow_limit": baseline_negative_limit,
        "summary_json": serde_json::to_string(&summaries).map_err(|error| error.to_string())?,
    })
    .as_object()
    .cloned()
    .unwrap_or_default();
    Ok(ResultData {
        source_path: original_path,
        image_shape: [frame.height, frame.width],
        grid_size: options.grid_size,
        aperture_radii: radii,
        reference_aperture_radius: options.reference_aperture_radius,
        control_signal_adu: options.control_signal_adu,
        baseline_candidate_count: baseline.detection.candidate_count,
        baseline_quality_count: baseline.detection.quality_count,
        psf_model: "empirical".into(),
        psf_source_count: empirical_psf.source_count,
        psf_support_radius: empirical_psf.support_radius,
        psf_median_fwhm_px: empirical_psf.median_fwhm_px,
        psf_kernel_sum: empirical_psf
            .kernel
            .iter()
            .map(|value| f64::from(*value))
            .sum(),
        psf_encircled_energy_by_radius: energy,
        parameters,
        positions,
        rows,
        summary_by_aperture: summaries,
        conclusion: format!(
            "{CONCLUSION_PREFIX}{}{CONCLUSION_MIDDLE}{}{CONCLUSION_SUFFIX}{}{CONCLUSION_TAIL}",
            position_count, radius_count, best_radius
        ),
    })
}

fn median_value(mut values: Vec<f64>) -> Option<f64> {
    sorted_median(&mut values)
}

fn encircled_energy(psf: &EmpiricalPsf) -> Result<Map<String, Value>> {
    let side = psf.side();
    if psf.kernel.len() != side * side {
        return Err("empirical PSF kernel shape is invalid".into());
    }
    let sum: f64 = psf.kernel.iter().map(|value| f64::from(*value)).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("empirical PSF kernel must have a finite positive sum".into());
    }
    let mut output = Map::new();
    let center = psf.support_radius as isize;
    for radius in 1..=psf.support_radius {
        let energy = psf
            .kernel
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                let y = *index / side;
                let x = *index % side;
                ((x as isize - center) as f64).hypot((y as isize - center) as f64) <= radius as f64
            })
            .map(|(_, value)| f64::from(*value))
            .sum::<f64>();
        output.insert(radius.to_string(), json!(energy / sum));
    }
    Ok(output)
}

fn csv_float(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1.0e16 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}

fn csv_option_float(value: Option<f64>) -> String {
    value.map_or_else(String::new, csv_float)
}

fn python_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

fn python_list(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("'{}'", value.replace('\'', "\\'")))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub fn write_artifacts(result: &ResultData, out_dir: &Path) -> Result<PathBuf> {
    let output = out_dir
        .canonicalize()
        .or_else(|_| {
            fs::create_dir_all(out_dir)?;
            out_dir.canonicalize()
        })
        .map_err(|error| error.to_string())?;
    let csv_path = output.join("aperture_sensitivity_audit.csv");
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(ROW_FIELDS)
        .map_err(|error| error.to_string())?;
    for row in &result.rows {
        writer
            .write_record(row.csv_cells())
            .map_err(|error| error.to_string())?;
    }
    let mut csv_bytes = vec![0xef, 0xbb, 0xbf];
    csv_bytes.extend(writer.into_inner().map_err(|error| error.to_string())?);
    fs::write(csv_path, csv_bytes).map_err(|error| error.to_string())?;
    fs::write(
        output.join("aperture_sensitivity_audit.json"),
        serde_json::to_vec_pretty(&result.as_value()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output)
}

fn parse_number<T: std::str::FromStr>(value: &str, name: &str) -> Result<T> {
    value
        .parse::<T>()
        .map_err(|_| format!("invalid value for {name}: {value}"))
}

fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_position(raw: &str) -> Result<Position> {
    let fields = raw.split(',').collect::<Vec<_>>();
    if fields.len() != 5 {
        return Err("--position must be CELL_ID,X,Y,LOCAL_BACKGROUND,LOCAL_NOISE".into());
    }
    let optional = |value: &str| -> Result<Option<f64>> {
        if value.is_empty()
            || value.eq_ignore_ascii_case("none")
            || value.eq_ignore_ascii_case("null")
        {
            Ok(None)
        } else {
            Ok(Some(parse_number(value, "--position")?))
        }
    };
    Ok(Position {
        cell_id: fields[0].to_owned(),
        x: parse_number(fields[1], "--position X")?,
        y: parse_number(fields[2], "--position Y")?,
        local_background_adu: optional(fields[3])?,
        local_noise_adu: optional(fields[4])?,
    })
}

const HELP: &str = "经验 PSF 孔径敏感性审计\n\nUsage: rst19-aperture-sensitivity FITS [OPTIONS]\n\nOptions:\n  --aperture-radii PX [PX ...]\n  --reference-aperture-radius PX\n  --control-signal-adu ADU\n  --grid-size N\n  --psf-fwhm PX\n  --proposal-mode gaussian|hybrid|ensemble\n  --threshold-sigma N\n  --min-distance PX\n  --background-box-size PX\n  --min-flux-snr N\n  --min-psf-support-pixels N\n  --max-sources N\n  --empirical-psf-radius PX\n  --empirical-psf-sources N\n  --seed N\n  --position CELL_ID,X,Y,LOCAL_BACKGROUND,LOCAL_NOISE (repeatable)\n  --out-dir DIR\n  -h, --help\n";

#[derive(Debug)]
struct ParsedArgs {
    input: PathBuf,
    out_dir: PathBuf,
    options: Options,
    positions: Vec<Position>,
}

fn parse_args(args: &[String]) -> Result<ParsedArgs> {
    let mut parsed = ParsedArgs {
        input: PathBuf::new(),
        out_dir: PathBuf::from("tmp/aperture-sensitivity-audit"),
        options: Options::default(),
        positions: Vec::new(),
    };
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if arg == "-h" || arg == "--help" {
            return Ok(parsed);
        }
        if arg == "--out-dir" {
            parsed.out_dir = take_value(args, &mut index, arg)?.into();
        } else if arg == "--aperture-radii" {
            parsed.options.aperture_radii.clear();
            index += 1;
            while index < args.len() && !args[index].starts_with('-') {
                parsed
                    .options
                    .aperture_radii
                    .push(parse_number(&args[index], "--aperture-radii")?);
                index += 1;
            }
            if parsed.options.aperture_radii.is_empty() {
                return Err("--aperture-radii requires at least one positive integer".into());
            }
            continue;
        } else if arg == "--reference-aperture-radius" {
            parsed.options.reference_aperture_radius =
                parse_number(&take_value(args, &mut index, arg)?, arg)?;
        } else if arg == "--control-signal-adu" {
            parsed.options.control_signal_adu =
                parse_number(&take_value(args, &mut index, arg)?, arg)?;
        } else if arg == "--grid-size" {
            parsed.options.grid_size = parse_number(&take_value(args, &mut index, arg)?, arg)?;
        } else if arg == "--psf-fwhm" {
            parsed.options.psf_fwhm = parse_number(&take_value(args, &mut index, arg)?, arg)?;
        } else if arg == "--proposal-mode" {
            parsed.options.proposal_mode = take_value(args, &mut index, arg)?;
        } else if arg == "--threshold-sigma" {
            parsed.options.threshold_sigma =
                parse_number(&take_value(args, &mut index, arg)?, arg)?;
        } else if arg == "--min-distance" {
            parsed.options.min_distance = parse_number(&take_value(args, &mut index, arg)?, arg)?;
        } else if arg == "--background-box-size" {
            parsed.options.background_box_size =
                parse_number(&take_value(args, &mut index, arg)?, arg)?;
        } else if arg == "--min-flux-snr" {
            parsed.options.min_flux_snr = parse_number(&take_value(args, &mut index, arg)?, arg)?;
        } else if arg == "--min-psf-support-pixels" {
            parsed.options.min_psf_support_pixels =
                parse_number(&take_value(args, &mut index, arg)?, arg)?;
        } else if arg == "--max-sources" {
            parsed.options.max_sources =
                Some(parse_number(&take_value(args, &mut index, arg)?, arg)?);
        } else if arg == "--empirical-psf-radius" {
            parsed.options.empirical_psf_radius =
                parse_number(&take_value(args, &mut index, arg)?, arg)?;
        } else if arg == "--empirical-psf-sources" {
            parsed.options.empirical_psf_sources =
                parse_number(&take_value(args, &mut index, arg)?, arg)?;
        } else if arg == "--seed" {
            parsed.options.seed = parse_number(&take_value(args, &mut index, arg)?, arg)?;
        } else if arg == "--position" {
            parsed
                .positions
                .push(parse_position(&take_value(args, &mut index, arg)?)?);
        } else if arg.starts_with('-') {
            return Err(format!("unknown option: {arg}"));
        } else if parsed.input.as_os_str().is_empty() {
            parsed.input = arg.into();
        } else {
            return Err(format!("unexpected positional argument: {arg}"));
        }
        index += 1;
    }
    if parsed.input.as_os_str().is_empty() {
        return Err("a FITS input path is required".into());
    }
    if !parsed.input.is_file() {
        return Err(format!("FITS 不存在：{}", parsed.input.display()));
    }
    Ok(parsed)
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let parsed = parse_args(args)?;
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        return Ok(json!({"help": HELP}));
    }
    let positions = (!parsed.positions.is_empty()).then_some(parsed.positions.as_slice());
    let result = run(&parsed.input, &parsed.options, positions)?;
    let output = write_artifacts(&result, &parsed.out_dir)?;
    Ok(json!({
        "tool": "rst19-aperture-sensitivity",
        "out_dir": output,
        "result_path": output.join("aperture_sensitivity_audit.json"),
        "result": result.as_value(),
    }))
}

pub fn run_cli(args: &[String]) -> i32 {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print!("{HELP}");
        return 0;
    }
    match run_desktop(args) {
        Ok(value) => {
            let out_dir = value["out_dir"].as_str().unwrap_or_default();
            println!("rst19-aperture-sensitivity: artifacts written to {out_dir}");
            println!(
                "{}",
                value["result"]["conclusion"].as_str().unwrap_or_default()
            );
            for row in value["result"]["summary_by_aperture"]
                .as_array()
                .into_iter()
                .flatten()
            {
                println!(
                    "r={} candidate={:.3} quality={:.3} median_flux_snr={} control={}/{}",
                    row["aperture_radius"].as_u64().unwrap_or_default(),
                    row["candidate_recall"].as_f64().unwrap_or_default(),
                    row["quality_recall"].as_f64().unwrap_or_default(),
                    row["median_candidate_flux_snr"],
                    row["paired_control_candidate_count"]
                        .as_u64()
                        .unwrap_or_default(),
                    row["paired_control_quality_count"]
                        .as_u64()
                        .unwrap_or_default(),
                );
            }
            0
        }
        Err(error) => {
            eprintln!("rst19-aperture-sensitivity: 审计失败：{error}");
            2
        }
    }
}
