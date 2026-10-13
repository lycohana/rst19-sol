//! Local pixel-repair counterfactual for a nearby pair of detections.
//!
//! The image is copied to memory and each repair variant is re-detected with
//! the native detector. Repaired values are diagnostic controls; they are not
//! written back to the original FITS file or presented as physical truth.

use crate::{
    detection::{self, DetectionOptions},
    fits,
    measurement::MeasurementOptions,
    models::{FitsFrame, PixelData},
    statistics, Result,
};
use csv::{Terminator, WriterBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

pub const VARIANTS: [&str; 4] = [
    "raw",
    "repair_repeated_code",
    "repair_negative_anomaly",
    "repair_both",
];

const GUARDRAILS: [&str; 4] = [
    "邻域补值是反事实敏感性控制，不是坏像素校准或物理真值重建。",
    "候选数量变化不等于物理星数变化；修复后的 detector 仍需 PSF/WCS/注入核验。",
    "局部窗口重跑的 detection_id 只在变体内部有意义，不能与首帧全图 ID 直接比较。",
    "补值结果只能说明当前响应对该类值域的敏感性，不提供噪点概率或伪影率。",
];

#[derive(Debug, Clone, Deserialize)]
pub struct TargetSource {
    pub detection_id: i64,
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub peak_x: Option<f64>,
    #[serde(default)]
    pub peak_y: Option<f64>,
    #[serde(default)]
    pub repeated_code_values: Vec<i64>,
    #[serde(default)]
    pub background: Option<f64>,
    #[serde(default)]
    pub noise: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplacementRow {
    pub variant: String,
    pub pixel_class: String,
    pub x: usize,
    pub y: usize,
    pub original_value_adu: f64,
    pub replacement_value_adu: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateRow {
    pub variant: String,
    pub detection_id: i64,
    pub x: f64,
    pub y: f64,
    pub peak_x: Option<f64>,
    pub peak_y: Option<f64>,
    pub peak: f64,
    pub flux_snr: Option<f64>,
    pub filter_snr: Option<f64>,
    pub quality_passed: bool,
    pub flags: String,
    pub distance_to_primary_px: f64,
    pub distance_to_secondary_px: f64,
    pub nearest_target: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SummaryRow {
    pub variant: String,
    pub repaired_pixel_count: usize,
    pub candidate_count: usize,
    pub returned_count: usize,
    pub quality_count: usize,
    pub target_window_candidate_count: usize,
    pub target_window_quality_count: usize,
    pub primary_window_candidate_count: usize,
    pub secondary_window_candidate_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairRepairResult {
    pub source_path: String,
    pub primary_detection_id: i64,
    pub secondary_detection_id: i64,
    pub primary_target: [f64; 2],
    pub secondary_target: [f64; 2],
    pub patch_bounds: [usize; 4],
    pub repeated_code_values: Vec<i64>,
    pub negative_anomaly_threshold_adu: f64,
    pub replacement_radius_px: usize,
    pub target_match_radius_px: f64,
    pub summaries: Vec<SummaryRow>,
    pub candidates: Vec<CandidateRow>,
    pub replacements: Vec<ReplacementRow>,
    pub parameters: Value,
    pub conclusion: String,
}

#[derive(Debug, Clone, Copy)]
struct PatchGeometry {
    height: usize,
    width: usize,
    x0: usize,
    y0: usize,
}

#[derive(Debug, Clone, Copy)]
struct RepairMetadata<'a> {
    radius: usize,
    variant: &'a str,
    pixel_class: &'a str,
}

impl PairRepairResult {
    pub fn as_dict(&self) -> Value {
        json!({
            "source_path": self.source_path,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "primary_target": self.primary_target,
            "secondary_target": self.secondary_target,
            "patch_bounds": self.patch_bounds,
            "repeated_code_values": self.repeated_code_values,
            "negative_anomaly_threshold_adu": self.negative_anomaly_threshold_adu,
            "replacement_radius_px": self.replacement_radius_px,
            "target_match_radius_px": self.target_match_radius_px,
            "summaries": self.summaries,
            "candidates": self.candidates,
            "replacements": self.replacements,
            "parameters": self.parameters,
            "conclusion": self.conclusion,
            "interpretation_guardrails": GUARDRAILS,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Options {
    pub patch_padding_px: usize,
    pub replacement_radius_px: usize,
    pub target_match_radius_px: f64,
    pub negative_anomaly_threshold_adu: f64,
    pub threshold_sigma: f64,
    pub min_distance: usize,
    pub aperture_radius: usize,
    pub psf_fwhm: f64,
    pub background_box_size: usize,
    pub min_flux_snr: f64,
    pub min_psf_support_pixels: usize,
    pub proposal_mode: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            patch_padding_px: 128,
            replacement_radius_px: 1,
            target_match_radius_px: 4.0,
            negative_anomaly_threshold_adu: -1000.0,
            threshold_sigma: 4.0,
            min_distance: 4,
            aperture_radius: 4,
            psf_fwhm: 2.0,
            background_box_size: 128,
            min_flux_snr: 5.0,
            min_psf_support_pixels: 3,
            proposal_mode: "hybrid".into(),
        }
    }
}

fn validate_options(options: &Options) -> Result<()> {
    if options.patch_padding_px == 0 {
        return Err("patch_padding_px must be positive".into());
    }
    if options.replacement_radius_px == 0 {
        return Err("replacement_radius_px must be positive".into());
    }
    finite_positive(options.target_match_radius_px, "target_match_radius_px")?;
    if !options.negative_anomaly_threshold_adu.is_finite() {
        return Err("negative_anomaly_threshold_adu must be finite".into());
    }
    if !options.threshold_sigma.is_finite() || options.threshold_sigma <= 0.0 {
        return Err("threshold_sigma must be finite and positive".into());
    }
    if options.min_distance == 0 {
        return Err("min_distance must be positive".into());
    }
    if options.aperture_radius == 0 {
        return Err("aperture_radius must be positive".into());
    }
    finite_positive(options.psf_fwhm, "psf_fwhm")?;
    if options.background_box_size == 0 {
        return Err("background_box_size must be positive".into());
    }
    if !options.min_flux_snr.is_finite() || options.min_flux_snr < 0.0 {
        return Err("min_flux_snr must be finite and non-negative".into());
    }
    if options.min_psf_support_pixels == 0 {
        return Err("min_psf_support_pixels must be positive".into());
    }
    if !["gaussian", "hybrid", "ensemble"].contains(&options.proposal_mode.as_str()) {
        return Err("proposal_mode must be gaussian, hybrid, or ensemble".into());
    }
    Ok(())
}

fn finite_positive(value: f64, name: &str) -> Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{name} must be finite and positive"));
    }
    Ok(value)
}

fn peak_coordinate(source: &TargetSource, x_axis: bool) -> Result<usize> {
    let value = if x_axis {
        source.peak_x.unwrap_or(source.x)
    } else {
        source.peak_y.unwrap_or(source.y)
    };
    if !value.is_finite() || value < 0.0 || value > usize::MAX as f64 {
        return Err("target peak coordinate must be finite and inside the image".into());
    }
    Ok(value.round_ties_even() as usize)
}

fn median_or_fallback(values: &[f64], fallback: f64) -> Result<f64> {
    if values.is_empty() {
        Ok(fallback)
    } else {
        statistics::median(values)
    }
}

fn repair_patch(
    patch: &[f64],
    geometry: PatchGeometry,
    selector: &[bool],
    exclusion: &[bool],
    metadata: RepairMetadata<'_>,
) -> Result<(Vec<f64>, Vec<ReplacementRow>)> {
    let PatchGeometry {
        height,
        width,
        x0: patch_x0,
        y0: patch_y0,
    } = geometry;
    let RepairMetadata {
        radius,
        variant,
        pixel_class,
    } = metadata;
    if patch.len() != height * width
        || selector.len() != patch.len()
        || exclusion.len() != patch.len()
    {
        return Err("selector and exclusion must match patch shape".into());
    }
    if radius == 0 {
        return Err("radius must be positive".into());
    }
    let ordinary: Vec<_> = patch
        .iter()
        .enumerate()
        .filter(|(index, value)| value.is_finite() && !exclusion[*index])
        .map(|(_, value)| *value)
        .collect();
    let fallback = median_or_fallback(&ordinary, 0.0)?;
    let mut output = patch.to_vec();
    let mut rows = Vec::new();
    for index in 0..patch.len() {
        if !selector[index] {
            continue;
        }
        let y = index / width;
        let x = index % width;
        let y0 = y.saturating_sub(radius);
        let y1 = y.saturating_add(radius).saturating_add(1).min(height);
        let x0 = x.saturating_sub(radius);
        let x1 = x.saturating_add(radius).saturating_add(1).min(width);
        let mut neighborhood = Vec::new();
        for ny in y0..y1 {
            for nx in x0..x1 {
                let local_index = ny * width + nx;
                if patch[local_index].is_finite() && !exclusion[local_index] {
                    neighborhood.push(patch[local_index]);
                }
            }
        }
        let replacement = median_or_fallback(&neighborhood, fallback)?;
        output[index] = replacement;
        rows.push(ReplacementRow {
            variant: variant.into(),
            pixel_class: pixel_class.into(),
            x: x + patch_x0,
            y: y + patch_y0,
            original_value_adu: patch[index],
            replacement_value_adu: replacement,
        });
    }
    Ok((output, rows))
}

fn detector_options(options: &Options) -> DetectionOptions {
    let measurement = MeasurementOptions {
        aperture_radius: options.aperture_radius,
        psf_fwhm: options.psf_fwhm,
        min_flux_snr: options.min_flux_snr,
        min_psf_support_pixels: options.min_psf_support_pixels,
        ..MeasurementOptions::default()
    };
    DetectionOptions {
        measurement,
        threshold_sigma: options.threshold_sigma,
        min_distance: options.min_distance,
        background_box_size: options.background_box_size,
        proposal_mode: options.proposal_mode.clone(),
        reject_linear_artifacts: true,
        ..DetectionOptions::default()
    }
}

fn detect_patch(
    patch: Vec<f64>,
    height: usize,
    width: usize,
    options: &Options,
) -> Result<detection::DetectionResult> {
    let data = PixelData::F64(patch);
    detection::detect_sources(&data, None, height, width, &detector_options(options), None)
}

fn candidate_rows(
    variant: &str,
    detection: &detection::DetectionResult,
    patch_x0: usize,
    patch_y0: usize,
    primary_target: [f64; 2],
    secondary_target: [f64; 2],
    target_radius: f64,
) -> Vec<CandidateRow> {
    let mut rows = Vec::new();
    for source in &detection.sources {
        let x = source.x + patch_x0 as f64;
        let y = source.y + patch_y0 as f64;
        let primary_distance = (x - primary_target[0]).hypot(y - primary_target[1]);
        let secondary_distance = (x - secondary_target[0]).hypot(y - secondary_target[1]);
        if primary_distance.min(secondary_distance) > target_radius {
            continue;
        }
        let nearest_target = if primary_distance <= secondary_distance {
            "primary"
        } else {
            "secondary"
        };
        rows.push(CandidateRow {
            variant: variant.into(),
            detection_id: source.detection_id,
            x,
            y,
            peak_x: source.peak_x.map(|value| value + patch_x0 as f64),
            peak_y: source.peak_y.map(|value| value + patch_y0 as f64),
            peak: source.peak,
            flux_snr: source.flux_snr,
            filter_snr: source.filter_snr,
            quality_passed: source.quality_passed,
            flags: source.flags.join("|"),
            distance_to_primary_px: primary_distance,
            distance_to_secondary_px: secondary_distance,
            nearest_target: nearest_target.into(),
        });
    }
    rows.sort_by(|left, right| {
        left.nearest_target
            .cmp(&right.nearest_target)
            .then(left.detection_id.cmp(&right.detection_id))
    });
    rows
}

pub fn run(
    frame: &FitsFrame,
    primary: &TargetSource,
    secondary: &TargetSource,
    repeated_code_values: Option<&[i64]>,
    options: &Options,
) -> Result<PairRepairResult> {
    validate_options(options)?;
    if primary.detection_id == secondary.detection_id {
        return Err("primary and secondary detections must differ".into());
    }
    if ![primary.x, primary.y, secondary.x, secondary.y]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err("target centroids must be finite".into());
    }
    let height = frame.height;
    let width = frame.width;
    if height == 0 || width == 0 || frame.data.len() != height * width {
        return Err("expected a 2-D FITS image".into());
    }
    let primary_peak_x = peak_coordinate(primary, true)?;
    let primary_peak_y = peak_coordinate(primary, false)?;
    let secondary_peak_x = peak_coordinate(secondary, true)?;
    let secondary_peak_y = peak_coordinate(secondary, false)?;
    if primary_peak_x >= width
        || secondary_peak_x >= width
        || primary_peak_y >= height
        || secondary_peak_y >= height
    {
        return Err("target peak coordinate must be finite and inside the image".into());
    }
    let patch_x0 = primary_peak_x
        .min(secondary_peak_x)
        .saturating_sub(options.patch_padding_px);
    let patch_y0 = primary_peak_y
        .min(secondary_peak_y)
        .saturating_sub(options.patch_padding_px);
    let patch_x1 = primary_peak_x
        .max(secondary_peak_x)
        .saturating_add(options.patch_padding_px)
        .saturating_add(1)
        .min(width);
    let patch_y1 = primary_peak_y
        .max(secondary_peak_y)
        .saturating_add(options.patch_padding_px)
        .saturating_add(1)
        .min(height);
    let patch_width = patch_x1 - patch_x0;
    let patch_height = patch_y1 - patch_y0;
    let mut patch = Vec::with_capacity(patch_width * patch_height);
    for y in patch_y0..patch_y1 {
        for x in patch_x0..patch_x1 {
            patch.push(frame.data.value(y * width + x));
        }
    }
    if !patch.iter().any(|value| value.is_finite()) {
        return Err("pair patch contains no finite pixels".into());
    }
    let mut repeated = BTreeSet::new();
    if let Some(values) = repeated_code_values {
        repeated.extend(values.iter().copied());
    }
    repeated.extend(primary.repeated_code_values.iter().copied());
    repeated.extend(secondary.repeated_code_values.iter().copied());
    let repeated_code_values: Vec<_> = repeated.into_iter().collect();
    let repeated_mask: Vec<_> = patch
        .iter()
        .map(|value| {
            repeated_code_values
                .iter()
                .any(|code| *value == *code as f64)
        })
        .collect();
    let negative_mask: Vec<_> = patch
        .iter()
        .map(|value| value.is_finite() && *value <= options.negative_anomaly_threshold_adu)
        .collect();
    let special_mask: Vec<_> = patch
        .iter()
        .enumerate()
        .map(|(index, value)| !value.is_finite() || repeated_mask[index] || negative_mask[index])
        .collect();
    let primary_target = [primary.x, primary.y];
    let secondary_target = [secondary.x, secondary.y];
    let mut summaries = Vec::with_capacity(VARIANTS.len());
    let mut candidates = Vec::new();
    let mut replacements = Vec::new();

    for variant in VARIANTS {
        let (selector, pixel_class): (Vec<bool>, &str) = match variant {
            "raw" => (vec![false; patch.len()], ""),
            "repair_repeated_code" => (repeated_mask.clone(), "repeated_code"),
            "repair_negative_anomaly" => (negative_mask.clone(), "negative_anomaly"),
            "repair_both" => (
                repeated_mask
                    .iter()
                    .zip(&negative_mask)
                    .map(|(repeated, negative)| *repeated || *negative)
                    .collect(),
                "both",
            ),
            _ => unreachable!(),
        };
        let (variant_image, current_replacements) = if variant == "raw" {
            (patch.clone(), Vec::new())
        } else {
            repair_patch(
                &patch,
                PatchGeometry {
                    height: patch_height,
                    width: patch_width,
                    x0: patch_x0,
                    y0: patch_y0,
                },
                &selector,
                &special_mask,
                RepairMetadata {
                    radius: options.replacement_radius_px,
                    variant,
                    pixel_class,
                },
            )?
        };
        let detection = detect_patch(variant_image, patch_height, patch_width, options)?;
        let current_candidates = candidate_rows(
            variant,
            &detection,
            patch_x0,
            patch_y0,
            primary_target,
            secondary_target,
            options.target_match_radius_px,
        );
        let primary_count = current_candidates
            .iter()
            .filter(|row| row.nearest_target == "primary")
            .count();
        let secondary_count = current_candidates.len() - primary_count;
        let target_quality_count = current_candidates
            .iter()
            .filter(|row| row.quality_passed)
            .count();
        summaries.push(SummaryRow {
            variant: variant.into(),
            repaired_pixel_count: current_replacements.len(),
            candidate_count: detection.candidate_count,
            returned_count: detection.sources.len(),
            quality_count: detection.quality_count,
            target_window_candidate_count: current_candidates.len(),
            target_window_quality_count: target_quality_count,
            primary_window_candidate_count: primary_count,
            secondary_window_candidate_count: secondary_count,
        });
        candidates.extend(current_candidates);
        replacements.extend(current_replacements);
    }

    let raw = &summaries[0];
    let negative = &summaries[2];
    let both = &summaries[3];
    let conclusion = format!(
        "局部异常值修复反事实完成：raw 目标窗口候选 {} 个，修复负异常后 {} 个，同时修复两类后 {} 个。若只修复重复码仍保留双响应、而修复负异常使目标窗口收缩为单主响应，则负值缺口是当前双峰分裂的主要形状触发，重复码更像值域有效性/锚点问题；该方向仍是局部 detector-level 反事实，不是硬件机理或物理星数结论。",
        raw.target_window_candidate_count,
        negative.target_window_candidate_count,
        both.target_window_candidate_count,
    );
    Ok(PairRepairResult {
        source_path: frame.path.to_string_lossy().into_owned(),
        primary_detection_id: primary.detection_id,
        secondary_detection_id: secondary.detection_id,
        primary_target,
        secondary_target,
        patch_bounds: [patch_x0, patch_y0, patch_x1, patch_y1],
        repeated_code_values,
        negative_anomaly_threshold_adu: options.negative_anomaly_threshold_adu,
        replacement_radius_px: options.replacement_radius_px,
        target_match_radius_px: options.target_match_radius_px,
        summaries,
        candidates,
        replacements,
        parameters: json!({
            "threshold_sigma": options.threshold_sigma,
            "min_distance": options.min_distance,
            "aperture_radius": options.aperture_radius,
            "psf_fwhm": options.psf_fwhm,
            "background_box_size": options.background_box_size,
            "min_flux_snr": options.min_flux_snr,
            "min_psf_support_pixels": options.min_psf_support_pixels,
            "proposal_mode": options.proposal_mode,
            "replacement_rule": "simultaneous local median within square neighborhood after excluding all known special pixels",
            "target_window_rule": "measured centroid distance to either target <= target_match_radius_px",
            "interpretation_boundary": "local repair counterfactual; no calibration, truth, FDR, precision, or physical artifact rate",
        }),
        conclusion,
    })
}

pub fn run_path(
    frame_path: &Path,
    catalog_path: &Path,
    primary_id: i64,
    secondary_id: i64,
    repeated_code_values: Option<&[i64]>,
    options: &Options,
) -> Result<PairRepairResult> {
    let (primary, secondary) = load_targets_csv(catalog_path, primary_id, secondary_id)?;
    let frame = fits::read_fits(frame_path, true)?;
    run(&frame, &primary, &secondary, repeated_code_values, options)
}

fn optional_float(raw: Option<&str>, field: &str) -> Result<Option<f64>> {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let value = raw
        .parse::<f64>()
        .map_err(|_| format!("source catalog field {field:?} is not numeric: {raw:?}"))?;
    if !value.is_finite() {
        return Err(format!(
            "source catalog field {field:?} is not finite: {raw:?}"
        ));
    }
    Ok(Some(value))
}

fn parse_pipe_ints(raw: Option<&str>) -> Result<Vec<i64>> {
    raw.unwrap_or_default()
        .split('|')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<i64>()
                .map_err(|_| format!("source catalog list contains a non-integer value: {value:?}"))
        })
        .collect()
}

pub fn load_targets_csv(
    path: &Path,
    primary_id: i64,
    secondary_id: i64,
) -> Result<(TargetSource, TargetSource)> {
    let csv_bytes = fs::read(path).map_err(|error| error.to_string())?;
    let csv_bytes = csv_bytes
        .strip_prefix(&[0xef, 0xbb, 0xbf])
        .unwrap_or(&csv_bytes);
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(csv_bytes);
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let index = |name: &str| headers.iter().position(|header| header == name);
    let required = ["detection_id", "x", "y"];
    let missing: Vec<_> = required
        .iter()
        .filter(|name| index(name).is_none())
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "source catalog is missing required columns: {}",
            missing.join(", ")
        ));
    }
    let mut selected: [Option<TargetSource>; 2] = [None, None];
    for (row_index, row) in reader.records().enumerate() {
        let row = row.map_err(|error| error.to_string())?;
        let get = |name: &str| index(name).and_then(|column| row.get(column));
        let raw_id = get("detection_id").unwrap_or_default().trim();
        let id = raw_id
            .parse::<i64>()
            .or_else(|_| raw_id.parse::<f64>().map(|value| value as i64))
            .map_err(|_| {
                format!(
                    "invalid source catalog row {}: detection_id is required",
                    row_index + 2
                )
            })?;
        let slot = if id == primary_id {
            Some(0)
        } else if id == secondary_id {
            Some(1)
        } else {
            None
        };
        let Some(slot) = slot else { continue };
        let parse_required = |field: &str| -> Result<f64> {
            optional_float(get(field), field)?.ok_or_else(|| {
                format!(
                    "invalid source catalog row {}: {field} is required",
                    row_index + 2
                )
            })
        };
        let source = TargetSource {
            detection_id: id,
            x: parse_required("x")?,
            y: parse_required("y")?,
            peak_x: optional_float(get("peak_x"), "peak_x")?,
            peak_y: optional_float(get("peak_y"), "peak_y")?,
            repeated_code_values: parse_pipe_ints(get("repeated_code_values"))?,
            background: optional_float(get("background"), "background")?,
            noise: optional_float(get("noise"), "noise")?,
        };
        selected[slot] = Some(source);
    }
    let primary = selected[0]
        .take()
        .ok_or_else(|| format!("source catalog does not contain detection_id={primary_id}"))?;
    let secondary = selected[1]
        .take()
        .ok_or_else(|| format!("source catalog does not contain detection_id={secondary_id}"))?;
    Ok((primary, secondary))
}

fn write_rows<T: Serialize>(path: &Path, headers: &[&str], rows: &[T]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut file = File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .terminator(Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer.serialize(row).map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    Ok(())
}

pub fn write_artifacts(result: &PairRepairResult, output_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    write_rows(
        &output_dir.join("pair_repair_summaries.csv"),
        &[
            "variant",
            "repaired_pixel_count",
            "candidate_count",
            "returned_count",
            "quality_count",
            "target_window_candidate_count",
            "target_window_quality_count",
            "primary_window_candidate_count",
            "secondary_window_candidate_count",
        ],
        &result.summaries,
    )?;
    write_rows(
        &output_dir.join("pair_repair_candidates.csv"),
        &[
            "variant",
            "detection_id",
            "x",
            "y",
            "peak_x",
            "peak_y",
            "peak",
            "flux_snr",
            "filter_snr",
            "quality_passed",
            "flags",
            "distance_to_primary_px",
            "distance_to_secondary_px",
            "nearest_target",
        ],
        &result.candidates,
    )?;
    write_rows(
        &output_dir.join("pair_repair_replacements.csv"),
        &[
            "variant",
            "pixel_class",
            "x",
            "y",
            "original_value_adu",
            "replacement_value_adu",
        ],
        &result.replacements,
    )?;
    fs::write(
        output_dir.join("pair_repair_counterfactual.json"),
        serde_json::to_vec_pretty(&result.as_dict()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    fs::canonicalize(output_dir).map_err(|error| error.to_string())
}

fn option_value<'a>(
    args: &'a [String],
    index: &mut usize,
    name: &str,
    inline: Option<&'a str>,
) -> Result<&'a str> {
    if let Some(value) = inline {
        Ok(value)
    } else {
        *index += 1;
        args.get(*index)
            .map(String::as_str)
            .ok_or_else(|| format!("{name} requires a value"))
    }
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut frame = None;
    let mut catalog = None;
    let mut primary_id = None;
    let mut secondary_id = None;
    let mut repeated_codes: Option<Vec<i64>> = None;
    let mut output_dir = PathBuf::from("tmp/pair-repair-counterfactual");
    let mut options = Options::default();
    let mut index = 0;
    while index < args.len() {
        let (name, inline) = args[index]
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((args[index].as_str(), None));
        match name {
            "--source-catalog"
            | "--primary-id"
            | "--secondary-id"
            | "--patch-padding-px"
            | "--replacement-radius-px"
            | "--target-match-radius-px"
            | "--repeated-code-values"
            | "--negative-anomaly-threshold-adu"
            | "--threshold-sigma"
            | "--min-distance"
            | "--aperture-radius"
            | "--psf-fwhm"
            | "--background-box-size"
            | "--min-flux-snr"
            | "--min-psf-support-pixels"
            | "--proposal-mode"
            | "--out-dir" => {
                let raw = option_value(args, &mut index, name, inline)?;
                let invalid =
                    |error: &dyn std::fmt::Display| format!("invalid value for {name}: {error}");
                match name {
                    "--source-catalog" => catalog = Some(PathBuf::from(raw)),
                    "--primary-id" => {
                        primary_id = Some(raw.parse::<i64>().map_err(|error| invalid(&error))?)
                    }
                    "--secondary-id" => {
                        secondary_id = Some(raw.parse::<i64>().map_err(|error| invalid(&error))?)
                    }
                    "--patch-padding-px" => {
                        options.patch_padding_px = raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--replacement-radius-px" => {
                        options.replacement_radius_px =
                            raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--target-match-radius-px" => {
                        options.target_match_radius_px =
                            raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--repeated-code-values" => {
                        repeated_codes = Some(
                            raw.split(',')
                                .map(str::trim)
                                .filter(|part| !part.is_empty())
                                .map(|part| part.parse::<i64>().map_err(|error| invalid(&error)))
                                .collect::<Result<Vec<_>>>()?,
                        );
                    }
                    "--negative-anomaly-threshold-adu" => {
                        options.negative_anomaly_threshold_adu =
                            raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--threshold-sigma" => {
                        options.threshold_sigma = raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--min-distance" => {
                        options.min_distance = raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--aperture-radius" => {
                        options.aperture_radius = raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--psf-fwhm" => {
                        options.psf_fwhm = raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--background-box-size" => {
                        options.background_box_size =
                            raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--min-flux-snr" => {
                        options.min_flux_snr = raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--min-psf-support-pixels" => {
                        options.min_psf_support_pixels =
                            raw.parse().map_err(|error| invalid(&error))?
                    }
                    "--proposal-mode" => options.proposal_mode = raw.to_owned(),
                    _ => output_dir = PathBuf::from(raw),
                }
            }
            _ if name.starts_with('-') => return Err(format!("unknown argument {name}")),
            _ => {
                if frame.replace(PathBuf::from(name)).is_some() {
                    return Err("only one FITS frame is allowed".into());
                }
            }
        }
        index += 1;
    }
    let frame_path = frame.ok_or("an input FITS frame is required")?;
    let catalog_path = catalog.ok_or("--source-catalog is required")?;
    let primary_id = primary_id.ok_or("--primary-id is required")?;
    let secondary_id = secondary_id.ok_or("--secondary-id is required")?;
    let result = run_path(
        &frame_path,
        &catalog_path,
        primary_id,
        secondary_id,
        repeated_codes.as_deref(),
        &options,
    )?;
    let output_dir = write_artifacts(&result, &output_dir)?;
    Ok(json!({
        "tool": "rst19-pair-repair-counterfactual",
        "out_dir": output_dir,
        "result_path": output_dir.join("pair_repair_counterfactual.json"),
        "result": result.as_dict(),
    }))
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!(
            "rst19-pair-repair-counterfactual FRAME --source-catalog CSV --primary-id ID --secondary-id ID [options]\n\
             Options: --patch-padding-px N --replacement-radius-px N --target-match-radius-px PX\n\
             --repeated-code-values A,B --negative-anomaly-threshold-adu ADU --threshold-sigma N\n\
             --min-distance N --aperture-radius N --psf-fwhm PX --background-box-size N\n\
             --min-flux-snr N --min-psf-support-pixels N --proposal-mode gaussian|hybrid|ensemble --out-dir DIR"
        );
        return Ok(());
    }
    let result = run_desktop(args)?;
    println!(
        "pair repair counterfactual: {}",
        result["out_dir"].as_str().unwrap_or_default()
    );
    println!(
        "{}",
        result["result"]["conclusion"].as_str().unwrap_or_default()
    );
    if let Some(rows) = result["result"]["summaries"].as_array() {
        for row in rows {
            println!(
                "{}: target_candidates={} target_quality={} repaired={}",
                row["variant"].as_str().unwrap_or_default(),
                row["target_window_candidate_count"]
                    .as_u64()
                    .unwrap_or_default(),
                row["target_window_quality_count"]
                    .as_u64()
                    .unwrap_or_default(),
                row["repaired_pixel_count"].as_u64().unwrap_or_default(),
            );
        }
    }
    Ok(())
}

pub fn handle_request(request: &Value) -> Result<Value> {
    let frame_path = request["frame_path"].as_str().ok_or("missing frame_path")?;
    let catalog_path = request["source_catalog_path"]
        .as_str()
        .ok_or("missing source_catalog_path")?;
    let primary_id = request["primary_id"].as_i64().ok_or("missing primary_id")?;
    let secondary_id = request["secondary_id"]
        .as_i64()
        .ok_or("missing secondary_id")?;
    let repeated = request
        .get("repeated_code_values")
        .filter(|value| !value.is_null())
        .cloned()
        .map(serde_json::from_value::<Vec<i64>>)
        .transpose()
        .map_err(|error| error.to_string())?;
    let mut options = Options::default();
    if let Some(value) = request.get("options") {
        let mut merged = serde_json::to_value(&options).map_err(|error| error.to_string())?;
        if let (Some(destination), Some(source)) = (merged.as_object_mut(), value.as_object()) {
            for (key, value) in source {
                destination.insert(key.clone(), value.clone());
            }
        }
        options = serde_json::from_value(merged).map_err(|error| error.to_string())?;
    }
    let result = run_path(
        Path::new(frame_path),
        Path::new(catalog_path),
        primary_id,
        secondary_id,
        repeated.as_deref(),
        &options,
    )?;
    let output = if let Some(path) = request.get("output_dir").and_then(Value::as_str) {
        let output = write_artifacts(&result, Path::new(path))?;
        json!({"out_dir":output, "result_path":output.join("pair_repair_counterfactual.json")})
    } else {
        Value::Null
    };
    Ok(json!({"result":result.as_dict(), "output":output}))
}
