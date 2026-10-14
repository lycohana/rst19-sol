//! Rust spatially stratified injection/recovery audit for real FITS backgrounds.
//!
//! Recovery against known injected positions is not physical-star completeness or precision.

use crate::{
    detection::DetectionOptions,
    empirical_psf::{self, EmpiricalPsf},
    fits,
    measurement::Detection,
    models::{FitsFrame, PixelData},
    pipeline::{self, FrameOptions},
    source_geometry_audit::NumpyPcg64,
    Result,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    str::FromStr,
};

const ROW_FIELDS: &[&str] = &[
    "source_path",
    "grid_size",
    "cell_x",
    "cell_y",
    "cell_id",
    "cell_x0",
    "cell_y0",
    "cell_x1",
    "cell_y1",
    "proposal_mode",
    "psf_model",
    "psf_source_count",
    "psf_median_fwhm_px",
    "signal_normalization",
    "control_signal_adu",
    "mean_injected_peak_excess_adu",
    "trial_count",
    "injected_count",
    "candidate_recovered_count",
    "quality_recovered_count",
    "candidate_recall",
    "quality_recall",
    "mean_candidate_count",
    "mean_quality_count",
    "mean_filter_noise",
    "mean_background_candidate_count",
    "mean_background_quality_count",
    "paired_control_candidate_count",
    "paired_control_quality_count",
    "paired_control_filter_noise",
    "candidate_delta_vs_paired_control",
    "quality_delta_vs_paired_control",
    "filter_noise_delta_vs_paired_control",
    "background_candidate_delta_vs_paired_control",
    "background_quality_delta_vs_paired_control",
    "local_background_adu",
    "local_noise_adu",
    "special_pixel_fraction",
    "nearest_baseline_source_px",
    "site_condition_json",
    "note",
];

const NOTE: &str = "每个空间单元在注入前远离当前候选和特殊值；候选/质量回收是已知注入真值召回，不是当前真实 FITS 的 precision、误检率或物理恒星完备率。";

#[derive(Clone, Debug)]
pub struct Options {
    pub grid_size: usize,
    pub peak_levels: Vec<f64>,
    pub trials_per_level: usize,
    pub sources_per_cell: usize,
    pub psf_fwhm: f64,
    pub injected_psf_fwhm: Option<f64>,
    pub proposal_mode: String,
    pub threshold_sigma: f64,
    pub min_distance: usize,
    pub aperture_radius: usize,
    pub background_box_size: usize,
    pub min_flux_snr: f64,
    pub min_psf_support_pixels: usize,
    pub max_sources: Option<usize>,
    pub match_radius_px: Option<f64>,
    pub seed: u64,
    pub reject_linear_artifacts: bool,
    pub psf_model: String,
    pub signal_normalization: String,
    pub empirical_psf_radius: usize,
    pub empirical_psf_sources: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            grid_size: 2,
            peak_levels: vec![56.0, 128.0],
            trials_per_level: 1,
            sources_per_cell: 1,
            psf_fwhm: 2.0,
            injected_psf_fwhm: None,
            proposal_mode: "hybrid".into(),
            threshold_sigma: 4.0,
            min_distance: 4,
            aperture_radius: 4,
            background_box_size: 128,
            min_flux_snr: 5.0,
            min_psf_support_pixels: 3,
            max_sources: None,
            match_radius_px: None,
            seed: 19_019,
            reject_linear_artifacts: true,
            psf_model: "gaussian".into(),
            signal_normalization: "integrated_excess".into(),
            empirical_psf_radius: 7,
            empirical_psf_sources: 64,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct Site {
    x: f64,
    y: f64,
    local_background_adu: f64,
    local_noise_adu: f64,
    upper_excess_adu: f64,
    special_pixel_fraction: f64,
    baseline_neighbor_count: usize,
    nearest_baseline_source_px: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct Row {
    source_path: String,
    grid_size: usize,
    cell_x: usize,
    cell_y: usize,
    cell_id: String,
    cell_x0: usize,
    cell_y0: usize,
    cell_x1: usize,
    cell_y1: usize,
    proposal_mode: String,
    psf_model: String,
    psf_source_count: usize,
    psf_median_fwhm_px: Option<f64>,
    signal_normalization: String,
    control_signal_adu: f64,
    mean_injected_peak_excess_adu: f64,
    trial_count: usize,
    injected_count: usize,
    candidate_recovered_count: usize,
    quality_recovered_count: usize,
    candidate_recall: f64,
    quality_recall: f64,
    mean_candidate_count: f64,
    mean_quality_count: f64,
    mean_filter_noise: f64,
    mean_background_candidate_count: f64,
    mean_background_quality_count: f64,
    paired_control_candidate_count: usize,
    paired_control_quality_count: usize,
    paired_control_filter_noise: f64,
    candidate_delta_vs_paired_control: f64,
    quality_delta_vs_paired_control: f64,
    filter_noise_delta_vs_paired_control: f64,
    background_candidate_delta_vs_paired_control: f64,
    background_quality_delta_vs_paired_control: f64,
    local_background_adu: f64,
    local_noise_adu: f64,
    special_pixel_fraction: f64,
    nearest_baseline_source_px: Option<f64>,
    site_condition_json: String,
    note: String,
}

#[derive(Clone, Debug, Serialize)]
struct ResultData {
    source_path: String,
    image_shape: [usize; 2],
    grid_size: usize,
    peak_levels: Vec<f64>,
    trials_per_level: usize,
    sources_per_cell: usize,
    baseline_candidate_count: usize,
    baseline_quality_count: usize,
    paired_control_candidate_count: usize,
    paired_control_quality_count: usize,
    paired_control_filter_noise: f64,
    psf_model: String,
    psf_source_count: usize,
    psf_median_fwhm_px: Option<f64>,
    parameters: Value,
    rows: Vec<Row>,
    conclusion: String,
}

fn validate(options: &Options) -> Result<()> {
    if options.grid_size == 0 {
        return Err("grid_size must be positive".into());
    }
    if options.trials_per_level == 0 || options.sources_per_cell == 0 {
        return Err("trials_per_level and sources_per_cell must be positive".into());
    }
    if !options.psf_fwhm.is_finite()
        || options.psf_fwhm <= 0.0
        || !options.threshold_sigma.is_finite()
        || options.threshold_sigma <= 0.0
        || options.min_distance == 0
        || options.aperture_radius == 0
    {
        return Err("spatial injection detector parameters must be positive".into());
    }
    if options.background_box_size < 16
        || !options.min_flux_snr.is_finite()
        || options.min_flux_snr <= 0.0
        || !(1..=9).contains(&options.min_psf_support_pixels)
    {
        return Err("background_box_size/min_flux_snr/min_psf_support_pixels are invalid".into());
    }
    if options.max_sources == Some(0) {
        return Err("max_sources must be positive when provided".into());
    }
    if options
        .match_radius_px
        .is_some_and(|v| !v.is_finite() || v <= 0.0)
    {
        return Err("match_radius_px must be positive when provided".into());
    }
    if options.psf_model != "gaussian" && options.psf_model != "empirical" {
        return Err("psf_model must be gaussian or empirical".into());
    }
    if options.signal_normalization != "peak_excess"
        && options.signal_normalization != "integrated_excess"
    {
        return Err("signal_normalization must be peak_excess or integrated_excess".into());
    }
    if options.empirical_psf_radius < 3 || options.empirical_psf_sources == 0 {
        return Err(
            "empirical PSF radius must be at least 3 and source count must be positive".into(),
        );
    }
    if options.peak_levels.is_empty()
        || options
            .peak_levels
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err("peak_levels must contain positive values".into());
    }
    if !["gaussian", "hybrid", "ensemble"].contains(&options.proposal_mode.as_str()) {
        return Err("proposal_mode must be gaussian, hybrid or ensemble".into());
    }
    if options
        .injected_psf_fwhm
        .is_some_and(|v| !v.is_finite() || v <= 0.0)
    {
        return Err("injected_psf_fwhm must be positive when provided".into());
    }
    Ok(())
}

fn analyze(
    frame: &FitsFrame,
    data: PixelData,
    options: &Options,
    baseline: Option<&pipeline::FrameAnalysis>,
) -> Result<pipeline::FrameAnalysis> {
    let mut detection = DetectionOptions::default();
    detection.threshold_sigma = options.threshold_sigma;
    detection.min_distance = options.min_distance;
    detection.max_sources = options.max_sources;
    detection.background_box_size = options.background_box_size;
    detection.proposal_mode = options.proposal_mode.clone();
    detection.reject_linear_artifacts = options.reject_linear_artifacts;
    detection.measurement.aperture_radius = options.aperture_radius;
    detection.measurement.psf_fwhm = options.psf_fwhm;
    detection.measurement.min_flux_snr = options.min_flux_snr;
    detection.measurement.min_psf_support_pixels = options.min_psf_support_pixels;
    if let Some(baseline) = baseline {
        let parameters = &baseline.detection.parameters;
        detection.measurement.saturation_level =
            parameters["saturation_level"].as_f64().filter(|v| *v > 0.0);
        detection.measurement.negative_overflow_limit =
            parameters["negative_overflow_limit"].as_f64();
        detection.mask_zero_pixels = Some(
            parameters["mask_zero_pixels"].as_u64().unwrap_or(0) != 0
                || parameters["mask_zero_pixels"].as_bool().unwrap_or(false),
        );
        detection.use_float32 = true;
    }
    let mut copy = frame.clone();
    copy.data = data;
    pipeline::analyze_frame(
        copy,
        None,
        None,
        &FrameOptions {
            detection,
            ..FrameOptions::default()
        },
        None,
    )
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

fn quantile(values: &mut [f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let position = q * (values.len() - 1) as f64;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    Some(values[lo] + (values[hi] - values[lo]) * (position - lo as f64))
}

fn local_patch_stats(
    image: &PixelData,
    height: usize,
    width: usize,
    x: f64,
    y: f64,
) -> Option<(f64, f64, f64)> {
    let cx = x.round_ties_even() as isize;
    let cy = y.round_ties_even() as isize;
    let x0 = (cx - 5).max(0) as usize;
    let y0 = (cy - 5).max(0) as usize;
    let x1 = (cx + 6).clamp(0, width as isize) as usize;
    let y1 = (cy + 6).clamp(0, height as isize) as usize;
    let mut finite = Vec::with_capacity((x1 - x0) * (y1 - y0));
    for row in y0..y1 {
        for col in x0..x1 {
            let value = image.value(row * width + col);
            if value.is_finite() {
                finite.push(value);
            }
        }
    }
    if finite.len() < 16 {
        return None;
    }
    let center = median(&mut finite.clone())?;
    let mut deviations = finite
        .iter()
        .map(|value| (value - center).abs())
        .collect::<Vec<_>>();
    let mad = (1.4826 * median(&mut deviations)?).max(f64::EPSILON);
    let upper = quantile(&mut finite, 0.98)? - center;
    Some((center, mad, upper))
}

fn special_mask(image: &PixelData) -> Vec<bool> {
    (0..image.len())
        .map(|index| {
            let value = image.value(index);
            if !value.is_finite() {
                return false;
            }
            if value == -1.0 {
                return true;
            }
            match image {
                PixelData::I16(_) => {
                    value <= -0.9 * i16::MAX as f64 || value >= (i16::MAX - 32) as f64
                }
                PixelData::I32(_) => {
                    value <= -0.9 * i32::MAX as f64 || value >= (i32::MAX - 32) as f64
                }
                PixelData::I64(_) => {
                    value <= -0.9 * i64::MAX as f64 || value >= (i64::MAX - 32) as f64
                }
                _ => false,
            }
        })
        .collect()
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

fn select_sites(
    image: &PixelData,
    height: usize,
    width: usize,
    sources: &[Detection],
    baseline_noise: f64,
    bounds: [usize; 4],
    count: usize,
    psf_fwhm: f64,
    aperture_radius: usize,
    rng: &mut NumpyPcg64,
) -> Result<Vec<Site>> {
    let special = special_mask(image);
    let neighborhood = 8usize.max((2.0 * psf_fwhm).ceil() as usize);
    let exclusion = 12.0_f64.max(2.0 * psf_fwhm);
    let minimum_distance = 10.0_f64.max(2.5 * psf_fwhm);
    let margin = (aperture_radius + 4).max((4.0 * psf_fwhm / 2.354820045).ceil() as usize + 2);
    let target_pool = 64usize.max(count.saturating_mul(80));
    let attempts_limit = 20_000usize.max(target_pool.saturating_mul(160));
    let [x0, y0, x1, y1] = bounds;
    if x1.saturating_sub(x0) <= 2 * margin || y1.saturating_sub(y0) <= 2 * margin {
        return Err("spatial cell is too small for the requested injection margins".into());
    }
    let mut pool = Vec::with_capacity(target_pool);
    for _ in 0..attempts_limit {
        if pool.len() >= target_pool {
            break;
        }
        let x = rng.uniform((x0 + margin) as f64, (x1 - margin) as f64);
        let y = rng.uniform((y0 + margin) as f64, (y1 - margin) as f64);
        if sources
            .iter()
            .any(|source| (source.x - x).hypot(source.y - y) <= exclusion)
        {
            continue;
        }
        let Some((background, noise, upper)) = local_patch_stats(image, height, width, x, y) else {
            continue;
        };
        let center_x = x.round_ties_even() as usize;
        let center_y = y.round_ties_even() as usize;
        let sx0 = center_x.saturating_sub(neighborhood);
        let sy0 = center_y.saturating_sub(neighborhood);
        let sx1 = (center_x + neighborhood + 1).min(width);
        let sy1 = (center_y + neighborhood + 1).min(height);
        let area = (sx1 - sx0) * (sy1 - sy0);
        let special_count = (sy0..sy1)
            .flat_map(|row| (sx0..sx1).map(move |col| row * width + col))
            .filter(|index| special[*index])
            .count();
        let special_fraction = if area == 0 {
            0.0
        } else {
            special_count as f64 / area as f64
        };
        if special_fraction > 0.0 || upper > (3.0 * noise).max(3.0 * baseline_noise) {
            continue;
        }
        let nearby = sources
            .iter()
            .filter_map(|source| {
                let distance = (source.x - x).hypot(source.y - y);
                (distance <= neighborhood as f64).then_some(distance)
            })
            .collect::<Vec<_>>();
        pool.push(Site {
            x,
            y,
            local_background_adu: background,
            local_noise_adu: noise,
            upper_excess_adu: upper,
            special_pixel_fraction: special_fraction,
            baseline_neighbor_count: nearby.len(),
            nearest_baseline_source_px: nearby.into_iter().min_by(f64::total_cmp),
        });
    }
    if pool.len() < count {
        return Err(format!(
            "空间单元 ({x0},{y0}) 合规位置不足：仅找到 {} 个，需要 {count} 个",
            pool.len()
        ));
    }
    let mut chosen = Vec::with_capacity(count);
    for index in rng.permutation(pool.len()) {
        let site = pool[index].clone();
        if chosen
            .iter()
            .all(|other: &Site| (site.x - other.x).hypot(site.y - other.y) >= minimum_distance)
        {
            chosen.push(site);
            if chosen.len() == count {
                break;
            }
        }
    }
    if chosen.len() != count {
        return Err(format!(
            "空间单元 ({x0},{y0}) 无法得到 {count} 个互相分离的注入位置，仅找到 {} 个",
            chosen.len()
        ));
    }
    Ok(chosen)
}

fn gaussian_patch(
    height: usize,
    width: usize,
    x: f64,
    y: f64,
    fwhm: f64,
) -> (usize, usize, usize, usize, Vec<f64>) {
    let sigma = (fwhm / 2.354820045).max(f64::EPSILON);
    let radius = 4usize.max((4.0 * sigma).ceil() as usize);
    let cx = x.floor() as isize;
    let cy = y.floor() as isize;
    let x0 = (cx - radius as isize).max(0) as usize;
    let y0 = (cy - radius as isize).max(0) as usize;
    let x1 = (cx + radius as isize + 1).min(width as isize) as usize;
    let y1 = (cy + radius as isize + 1).min(height as isize) as usize;
    let mut kernel = Vec::with_capacity((x1 - x0) * (y1 - y0));
    for row in y0..y1 {
        for col in x0..x1 {
            let dx = col as f64 - x;
            let dy = row as f64 - y;
            kernel.push((-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp());
        }
    }
    (y0, y1, x0, x1, kernel)
}

fn empirical_patch(
    height: usize,
    width: usize,
    x: f64,
    y: f64,
    psf: &EmpiricalPsf,
) -> Result<(usize, usize, usize, usize, Vec<f64>)> {
    let radius = psf.support_radius;
    let side = psf.side();
    if psf.kernel.len() != side * side {
        return Err("empirical PSF kernel has an invalid shape".into());
    }
    let center_x = x.round_ties_even() as isize;
    let center_y = y.round_ties_even() as isize;
    let shift_x = x - center_x as f64;
    let shift_y = y - center_y as f64;
    let mut shifted = vec![0.0_f64; psf.kernel.len()];
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
            shifted[out_y * side + out_x] = (top * (1.0 - dy) + bottom * dy) as f32 as f64;
        }
    }
    let y0 = (center_y - radius as isize).max(0) as usize;
    let x0 = (center_x - radius as isize).max(0) as usize;
    let y1 = (center_y + radius as isize + 1).min(height as isize) as usize;
    let x1 = (center_x + radius as isize + 1).min(width as isize) as usize;
    let ky0 = y0 as isize - (center_y - radius as isize);
    let kx0 = x0 as isize - (center_x - radius as isize);
    let mut patch = Vec::with_capacity((y1 - y0) * (x1 - x0));
    for row in y0..y1 {
        for col in x0..x1 {
            let ky = (ky0 + (row - y0) as isize) as usize;
            let kx = (kx0 + (col - x0) as isize) as usize;
            patch.push(shifted[ky * side + kx]);
        }
    }
    Ok((y0, y1, x0, x1, patch))
}

fn inject_signal(
    image: &mut [f32],
    height: usize,
    width: usize,
    x: f64,
    y: f64,
    signal: f64,
    options: &Options,
    empirical: Option<&EmpiricalPsf>,
) -> Result<f64> {
    if !signal.is_finite() || signal <= 0.0 {
        return Err("signal_adu must be finite and positive".into());
    }
    let (y0, _y1, x0, x1, kernel) = if options.psf_model == "empirical" {
        empirical_patch(
            height,
            width,
            x,
            y,
            empirical.ok_or("empirical_psf is required for empirical injection")?,
        )?
    } else {
        gaussian_patch(
            height,
            width,
            x,
            y,
            options.injected_psf_fwhm.unwrap_or(options.psf_fwhm),
        )
    };
    let peak = if options.signal_normalization == "peak_excess" {
        signal
    } else {
        let sum: f64 = kernel.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err("injection PSF kernel must have a finite positive sum".into());
        }
        signal / sum
    };
    for (offset, value) in kernel.iter().enumerate() {
        let row = y0 + offset / (x1 - x0);
        let col = x0 + offset % (x1 - x0);
        let index = row * width + col;
        image[index] = (f64::from(image[index]) + peak * value) as f32;
    }
    Ok(peak)
}

fn matched_count(targets: &[(f64, f64)], detections: &[Detection], radius: f64) -> usize {
    let mut remaining = detections.to_vec();
    let mut matched = 0;
    for &(x, y) in targets {
        let mut best_index = None;
        let mut best_distance = radius;
        for (index, source) in remaining.iter().enumerate() {
            let distance = (source.x - x).hypot(source.y - y);
            if distance <= best_distance {
                best_index = Some(index);
                best_distance = distance;
            }
        }
        if let Some(index) = best_index {
            remaining.remove(index);
            matched += 1;
        }
    }
    matched
}

fn count_outside_injection_regions(
    detections: &[Detection],
    targets: &[(f64, f64)],
    radius: f64,
) -> usize {
    detections
        .iter()
        .filter(|source| {
            targets
                .iter()
                .all(|(x, y)| (source.x - x).hypot(source.y - y) > radius)
        })
        .count()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn median_value(values: &[f64]) -> Option<f64> {
    let mut copy = values.to_vec();
    median(&mut copy)
}

fn run(input: &Path, options: &Options) -> Result<ResultData> {
    validate(options)?;
    let frame = fits::read_fits(input, false)?;
    let (height, width) = (frame.height, frame.width);
    if height < 2 * (options.aperture_radius + 8) || width < 2 * (options.aperture_radius + 8) {
        return Err("image is too small for spatial injection margins".into());
    }
    let baseline = analyze(&frame, frame.data.clone(), options, None)?;
    let mut empirical = None;
    if options.psf_model == "empirical" {
        let values = frame.data.to_f64();
        empirical = empirical_psf::estimate(
            values.as_slice().into(),
            height,
            width,
            &baseline.detection.sources,
            None,
            options.empirical_psf_radius,
            options.empirical_psf_sources,
            &[],
        )?;
        if empirical.is_none() {
            return Err("真实质量源不足，无法建立实测 PSF；请改用 gaussian 或增加模板源".into());
        }
    }
    let float32_background = frame
        .data
        .to_f64()
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    let paired = analyze(
        &frame,
        PixelData::F32(float32_background.clone()),
        options,
        Some(&baseline),
    )?;
    let paired_filter_noise = paired.detection.parameters["filter_noise"]
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or("检测结果缺少有限的 filter_noise，无法进行空间注入配对控制")?;

    let mut rng = NumpyPcg64::new(u128::from(options.seed));
    let mut layouts = Vec::with_capacity(options.grid_size * options.grid_size);
    let baseline_noise = baseline.detection.noise;
    for cell_y in 0..options.grid_size {
        for cell_x in 0..options.grid_size {
            let bounds = cell_bounds(height, width, options.grid_size, cell_x, cell_y);
            let mut trials = Vec::with_capacity(options.trials_per_level);
            for _ in 0..options.trials_per_level {
                trials.push(select_sites(
                    &frame.data,
                    height,
                    width,
                    &baseline.detection.sources,
                    baseline_noise,
                    bounds,
                    options.sources_per_cell,
                    options.psf_fwhm,
                    options.aperture_radius,
                    &mut rng,
                )?);
            }
            layouts.push((cell_x, cell_y, bounds, trials));
        }
    }

    let recovery_radius = options.match_radius_px.unwrap_or(options.psf_fwhm).max(3.0);
    let protection_radius = (2.0 * options.psf_fwhm).max(8.0);
    let mut rows = Vec::with_capacity(layouts.len() * options.peak_levels.len());
    for (cell_x, cell_y, bounds, trials) in &layouts {
        let site_condition_json = serde_json::to_string(
            &trials
                .iter()
                .enumerate()
                .flat_map(|(trial_index, sites)| {
                    sites.iter().enumerate().map(move |(site_index, site)| json!({
                "trial_index": trial_index + 1, "site_index": site_index + 1,
                "x": site.x, "y": site.y, "local_background_adu": site.local_background_adu,
                "local_noise_adu": site.local_noise_adu, "upper_excess_adu": site.upper_excess_adu,
                "special_pixel_fraction": site.special_pixel_fraction,
                "baseline_neighbor_count": site.baseline_neighbor_count,
                "nearest_baseline_source_px": site.nearest_baseline_source_px,
            }))
                })
                .collect::<Vec<_>>(),
        )
        .map_err(|error| error.to_string())?;
        for &control_signal in &options.peak_levels {
            let mut candidate_recovered = 0;
            let mut quality_recovered = 0;
            let mut candidate_counts = Vec::with_capacity(trials.len());
            let mut quality_counts = Vec::with_capacity(trials.len());
            let mut filter_noises = Vec::with_capacity(trials.len());
            let mut background_candidate_counts = Vec::with_capacity(trials.len());
            let mut background_quality_counts = Vec::with_capacity(trials.len());
            let mut actual_peaks = Vec::with_capacity(trials.len() * options.sources_per_cell);
            let mut local_backgrounds = Vec::with_capacity(trials.len() * options.sources_per_cell);
            let mut local_noises = Vec::with_capacity(trials.len() * options.sources_per_cell);
            let mut special_fractions = Vec::with_capacity(trials.len() * options.sources_per_cell);
            let mut nearest_distances = Vec::new();
            for sites in trials {
                let mut image = float32_background.clone();
                let positions = sites
                    .iter()
                    .map(|site| (site.x, site.y))
                    .collect::<Vec<_>>();
                for site in sites {
                    actual_peaks.push(inject_signal(
                        &mut image,
                        height,
                        width,
                        site.x,
                        site.y,
                        control_signal,
                        options,
                        empirical.as_ref(),
                    )?);
                    local_backgrounds.push(site.local_background_adu);
                    local_noises.push(site.local_noise_adu);
                    special_fractions.push(site.special_pixel_fraction);
                    if let Some(distance) = site.nearest_baseline_source_px {
                        nearest_distances.push(distance);
                    }
                }
                let detection = analyze(&frame, PixelData::F32(image), options, Some(&baseline))?;
                let filter_noise = detection.detection.parameters["filter_noise"]
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or("检测结果缺少有限的 filter_noise，无法进行空间注入配对控制")?;
                candidate_recovered +=
                    matched_count(&positions, &detection.detection.sources, recovery_radius);
                let quality_sources = detection
                    .detection
                    .sources
                    .iter()
                    .filter(|source| source.quality_passed)
                    .cloned()
                    .collect::<Vec<_>>();
                quality_recovered += matched_count(&positions, &quality_sources, recovery_radius);
                candidate_counts.push(detection.detection.candidate_count as f64);
                quality_counts.push(detection.detection.quality_count as f64);
                filter_noises.push(filter_noise);
                background_candidate_counts.push(count_outside_injection_regions(
                    &detection.detection.sources,
                    &positions,
                    protection_radius,
                ) as f64);
                background_quality_counts.push(count_outside_injection_regions(
                    &quality_sources,
                    &positions,
                    protection_radius,
                ) as f64);
            }
            let injected_count = options.trials_per_level * options.sources_per_cell;
            let mean_candidate_count = mean(&candidate_counts);
            let mean_quality_count = mean(&quality_counts);
            let mean_filter_noise = mean(&filter_noises);
            let mean_background_candidate_count = mean(&background_candidate_counts);
            let mean_background_quality_count = mean(&background_quality_counts);
            let paired_filter = paired.detection.parameters["filter_noise"]
                .as_f64()
                .unwrap_or_default();
            let nearest_median = median_value(&nearest_distances);
            rows.push(Row {
                source_path: frame.path.to_string_lossy().into_owned(),
                grid_size: options.grid_size,
                cell_x: *cell_x,
                cell_y: *cell_y,
                cell_id: format!("r{cell_y}c{cell_x}"),
                cell_x0: bounds[0],
                cell_y0: bounds[1],
                cell_x1: bounds[2],
                cell_y1: bounds[3],
                proposal_mode: options.proposal_mode.clone(),
                psf_model: options.psf_model.clone(),
                psf_source_count: empirical.as_ref().map_or(0, |psf| psf.source_count),
                psf_median_fwhm_px: empirical.as_ref().and_then(|psf| psf.median_fwhm_px),
                signal_normalization: options.signal_normalization.clone(),
                control_signal_adu: control_signal,
                mean_injected_peak_excess_adu: mean(&actual_peaks),
                trial_count: options.trials_per_level,
                injected_count,
                candidate_recovered_count: candidate_recovered,
                quality_recovered_count: quality_recovered,
                candidate_recall: candidate_recovered as f64 / injected_count as f64,
                quality_recall: quality_recovered as f64 / injected_count as f64,
                mean_candidate_count,
                mean_quality_count,
                mean_filter_noise,
                mean_background_candidate_count,
                mean_background_quality_count,
                paired_control_candidate_count: paired.detection.candidate_count,
                paired_control_quality_count: paired.detection.quality_count,
                paired_control_filter_noise: paired_filter,
                candidate_delta_vs_paired_control: mean_candidate_count
                    - paired.detection.candidate_count as f64,
                quality_delta_vs_paired_control: mean_quality_count
                    - paired.detection.quality_count as f64,
                filter_noise_delta_vs_paired_control: mean_filter_noise - paired_filter,
                background_candidate_delta_vs_paired_control: mean_background_candidate_count
                    - paired.detection.candidate_count as f64,
                background_quality_delta_vs_paired_control: mean_background_quality_count
                    - paired.detection.quality_count as f64,
                local_background_adu: mean(&local_backgrounds),
                local_noise_adu: mean(&local_noises),
                special_pixel_fraction: mean(&special_fractions),
                nearest_baseline_source_px: nearest_median,
                site_condition_json: site_condition_json.clone(),
                note: NOTE.into(),
            });
        }
    }

    let source_path = frame.path.to_string_lossy().into_owned();
    let psf = empirical.as_ref();
    let result_parameters = json!({
        "grid_size":options.grid_size, "peak_levels":options.peak_levels,
        "trials_per_level":options.trials_per_level, "sources_per_cell":options.sources_per_cell,
        "psf_fwhm":options.psf_fwhm, "injected_psf_fwhm":options.injected_psf_fwhm.unwrap_or(options.psf_fwhm),
        "proposal_mode":options.proposal_mode, "threshold_sigma":options.threshold_sigma,
        "min_distance":options.min_distance, "aperture_radius":options.aperture_radius,
        "background_box_size":options.background_box_size, "min_flux_snr":options.min_flux_snr,
        "min_psf_support_pixels":options.min_psf_support_pixels, "max_sources":options.max_sources,
        "match_radius_px":recovery_radius, "seed":options.seed,
        "reject_linear_artifacts":options.reject_linear_artifacts,
        "signal_normalization":options.signal_normalization, "empirical_psf_radius":options.empirical_psf_radius,
        "empirical_psf_sources":options.empirical_psf_sources, "baseline_noise_adu":baseline.detection.noise,
        "paired_control_dtype":"float32", "paired_control_candidate_count":paired.detection.candidate_count,
        "paired_control_quality_count":paired.detection.quality_count, "paired_control_filter_noise":paired_filter_noise,
        "negative_overflow_limit":baseline.detection.parameters["negative_overflow_limit"],
    });
    Ok(ResultData {
        source_path, image_shape:[height, width], grid_size:options.grid_size,
        peak_levels:options.peak_levels.clone(), trials_per_level:options.trials_per_level,
        sources_per_cell:options.sources_per_cell, baseline_candidate_count:baseline.detection.candidate_count,
        baseline_quality_count:baseline.detection.quality_count,
        paired_control_candidate_count:paired.detection.candidate_count,
        paired_control_quality_count:paired.detection.quality_count,
        paired_control_filter_noise:paired_filter_noise, psf_model:options.psf_model.clone(),
        psf_source_count:psf.map_or(0, |value| value.source_count),
        psf_median_fwhm_px:psf.and_then(|value| value.median_fwhm_px),
        parameters:result_parameters, rows,
        conclusion:format!(
            "空间分区留出注入完成：{}×{} 单元、{} 个信号档位、每档 {} 次；候选/质量回收只表示已知注入源召回。候选总数变化以同 dtype 配对控制为参照；各单元差异用于定位噪声、PSF 或边界校准方向，不直接解释为恒星密度或物理完备率。",
            options.grid_size, options.grid_size, options.peak_levels.len(), options.trials_per_level
        ),
    })
}

fn write_artifacts(result: &ResultData, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir)
        .map_err(|error| format!("cannot create output directory: {error}"))?;
    let output = out_dir
        .canonicalize()
        .map_err(|error| format!("cannot resolve output directory: {error}"))?;
    let csv_path = output.join("spatial_injection_audit.csv");
    let mut file = fs::File::create(&csv_path).map_err(|error| error.to_string())?;
    use std::io::Write;
    file.write_all(&[0xEF, 0xBB, 0xBF])
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    if result.rows.is_empty() {
        writer
            .write_record(["cell_id", "control_signal_adu"])
            .map_err(|error| error.to_string())?;
    } else {
        writer
            .write_record(ROW_FIELDS)
            .map_err(|error| error.to_string())?;
        for row in &result.rows {
            writer.serialize(row).map_err(|error| error.to_string())?;
        }
    }
    writer.flush().map_err(|error| error.to_string())?;
    let json_path = output.join("spatial_injection_audit.json");
    let json_file = fs::File::create(&json_path).map_err(|error| error.to_string())?;
    serde_json::to_writer_pretty(json_file, result).map_err(|error| error.to_string())?;
    Ok(output)
}

const HELP: &str = "在真实 FITS 背景的 detector 空间单元中做留出注入-回收\n\
用法: rst19-spatial-injection <fits_path> [选项]\n\
  --grid-size N --peak-levels ADU [ADU ...] --trials N --sources-per-cell N\n\
  --threshold-sigma X --min-distance N --aperture-radius N --psf-fwhm X\n\
  --injected-psf-fwhm X --background-box-size N --min-flux-snr X\n\
  --min-psf-support-pixels N --max-sources N --match-radius-px X --seed N\n\
  --proposal-mode gaussian|hybrid|ensemble --psf-model gaussian|empirical\n\
  --signal-normalization peak_excess|integrated_excess --empirical-psf-radius N\n\
  --empirical-psf-sources N --keep-linear-artifacts --out-dir DIR\n";

fn take_value(args: &[String], index: &mut usize, option: &str) -> Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn parse_number<T>(args: &[String], index: &mut usize, option: &str) -> Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    let value = take_value(args, index, option)?;
    value
        .parse()
        .map_err(|error| format!("invalid {option} value {value:?}: {error}"))
}

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf, Options)> {
    let mut input = None;
    let mut out_dir = PathBuf::from("tmp/spatial-injection-audit");
    let mut options = Options::default();
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        match arg {
            "--grid-size" => options.grid_size = parse_number(args, &mut index, arg)?,
            "--peak-levels" => {
                let mut levels = Vec::new();
                while index + 1 < args.len() && !args[index + 1].starts_with('-') {
                    index += 1;
                    levels.push(args[index].parse().map_err(|error| {
                        format!("invalid --peak-levels value {:?}: {error}", args[index])
                    })?);
                }
                if levels.is_empty() {
                    return Err("--peak-levels requires at least one value".into());
                }
                options.peak_levels = levels;
            }
            "--trials" => options.trials_per_level = parse_number(args, &mut index, arg)?,
            "--sources-per-cell" => options.sources_per_cell = parse_number(args, &mut index, arg)?,
            "--threshold-sigma" => options.threshold_sigma = parse_number(args, &mut index, arg)?,
            "--min-distance" => options.min_distance = parse_number(args, &mut index, arg)?,
            "--aperture-radius" => options.aperture_radius = parse_number(args, &mut index, arg)?,
            "--psf-fwhm" => options.psf_fwhm = parse_number(args, &mut index, arg)?,
            "--injected-psf-fwhm" => {
                options.injected_psf_fwhm = Some(parse_number(args, &mut index, arg)?)
            }
            "--background-box-size" => {
                options.background_box_size = parse_number(args, &mut index, arg)?
            }
            "--min-flux-snr" => options.min_flux_snr = parse_number(args, &mut index, arg)?,
            "--min-psf-support-pixels" => {
                options.min_psf_support_pixels = parse_number(args, &mut index, arg)?
            }
            "--max-sources" => options.max_sources = Some(parse_number(args, &mut index, arg)?),
            "--match-radius-px" => {
                options.match_radius_px = Some(parse_number(args, &mut index, arg)?)
            }
            "--seed" => options.seed = parse_number(args, &mut index, arg)?,
            "--proposal-mode" => options.proposal_mode = take_value(args, &mut index, arg)?,
            "--psf-model" => options.psf_model = take_value(args, &mut index, arg)?,
            "--signal-normalization" => {
                options.signal_normalization = take_value(args, &mut index, arg)?
            }
            "--empirical-psf-radius" => {
                options.empirical_psf_radius = parse_number(args, &mut index, arg)?
            }
            "--empirical-psf-sources" => {
                options.empirical_psf_sources = parse_number(args, &mut index, arg)?
            }
            "--keep-linear-artifacts" => options.reject_linear_artifacts = false,
            "--out-dir" => out_dir = take_value(args, &mut index, arg)?.into(),
            "-h" | "--help" => {}
            option if option.starts_with('-') => return Err(format!("unknown option: {option}")),
            _ if input.is_none() => input = Some(PathBuf::from(arg)),
            _ => return Err(format!("unexpected positional argument: {arg}")),
        }
        index += 1;
    }
    let input = input.ok_or("a FITS input path is required")?;
    if !input.is_file() {
        return Err(format!("FITS 不存在：{}", input.display()));
    }
    validate(&options)?;
    Ok((input, out_dir, options))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        return Ok(json!({"help":HELP}));
    }
    let (input, out_dir, options) = parse_args(args)?;
    let result = run(&input, &options)?;
    let output = write_artifacts(&result, &out_dir)?;
    Ok(json!({
        "tool":"rst19-spatial-injection", "out_dir":output,
        "result_path":output.join("spatial_injection_audit.json"),
        "result":result,
    }))
}

pub fn run_cli(args: &[String]) -> i32 {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print!("{HELP}");
        return 0;
    }
    match run_desktop(args) {
        Ok(value) => {
            println!(
                "rst19-spatial-injection: artifacts written to {}",
                value["out_dir"].as_str().unwrap_or_default()
            );
            println!(
                "{}",
                value["result"]["conclusion"].as_str().unwrap_or_default()
            );
            for row in value["result"]["rows"].as_array().into_iter().flatten() {
                println!(
                    "{} signal={} candidate={:.3} quality={:.3} noise={:.3}",
                    row["cell_id"].as_str().unwrap_or_default(),
                    row["control_signal_adu"].as_f64().unwrap_or_default(),
                    row["candidate_recall"].as_f64().unwrap_or_default(),
                    row["quality_recall"].as_f64().unwrap_or_default(),
                    row["local_noise_adu"].as_f64().unwrap_or_default()
                );
            }
            0
        }
        Err(error) => {
            eprintln!("rst19-spatial-injection: 审计失败：{error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrated_gaussian_injection_preserves_requested_discrete_signal() {
        let mut image = vec![100.0_f32; 64 * 64];
        let mut options = Options::default();
        options.psf_fwhm = 2.4;
        options.signal_normalization = "integrated_excess".into();
        let peak = inject_signal(&mut image, 64, 64, 31.2, 28.7, 128.0, &options, None).unwrap();
        let actual_sum = image
            .iter()
            .map(|value| f64::from(*value) - 100.0)
            .sum::<f64>();
        assert!(peak > 0.0);
        assert!((actual_sum - 128.0).abs() < 1e-3);
    }

    #[test]
    fn greedy_truth_matching_does_not_reuse_a_detection() {
        let source = Detection {
            detection_id: 1,
            x: 10.0,
            y: 10.0,
            peak: 10.0,
            flux: 10.0,
            background: 0.0,
            noise: 1.0,
            snr: 10.0,
            fwhm: None,
            flags: vec![],
            flux_error: None,
            flux_snr: None,
            filter_snr: None,
            fwhm_x: None,
            fwhm_y: None,
            ellipticity: None,
            sharpness: None,
            footprint_pixels: None,
            psf_support_pixels: None,
            quality_passed: true,
            peak_x: None,
            peak_y: None,
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
        };
        let targets = [(9.0, 10.0), (11.0, 10.0)];
        assert_eq!(matched_count(&targets, &[source], 2.0), 1);
    }
}
