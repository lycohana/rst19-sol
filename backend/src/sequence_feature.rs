//! Native port of the sequence feature-persistence audit.
//!
//! Candidate and quality layers intentionally remain separate. Matches describe
//! detector response in a registered neighborhood; they are not source identity.

use crate::measurement::MeasurementOptions;
use crate::{
    compatible_spatial::CompatibleKdTree,
    detection::DetectionOptions,
    fits,
    measurement::Detection,
    pipeline::{self, FrameOptions},
    sequence_registration,
    source_classification::{self, SOURCE_FEATURE_CLASSES},
    Result,
};
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

pub const HELP: &str = "15 帧特征类别持久性审计（Rust）\n\nUsage: rst19-feature-sequence DATA_DIR [options]\n\nOptions: --threshold-sigma N --min-distance N --aperture-radius N --psf-fwhm N --proposal-mode gaussian|hybrid|ensemble --background-box-size N --background-sample-limit N --min-flux-snr N --min-psf-support-pixels N --max-sources N --registration-radius-px N --association-radius-px N --out-dir PATH";

const PERSISTENCE_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "anchor_candidate_count",
    "anchor_quality_count",
    "frame_count",
    "required_presence",
    "association_radius_px",
    "candidate_median_presence",
    "candidate_mean_presence",
    "candidate_presence_ge_required_count",
    "candidate_presence_all_frames_count",
    "quality_median_presence",
    "quality_mean_presence",
    "quality_presence_ge_required_count",
    "quality_presence_all_frames_count",
];
const FRAME_FIELDS: &[&str] = &[
    "frame_index",
    "path",
    "candidate_count",
    "returned_count",
    "quality_count",
    "background_adu",
    "noise_adu",
    "feature_class",
    "feature_class_label",
    "feature_candidate_count",
    "feature_quality_count",
];
const METHOD_FRAME_FIELDS: &[&str] = &[
    "frame_index",
    "path",
    "feature_class",
    "feature_class_label",
    "feature_candidate_count",
    "feature_quality_count",
    "all_three_count",
    "gaussian_only_count",
    "dog_only_count",
    "partial_combination_count",
    "no_gaussian_count",
    "no_method_count",
    "no_gaussian_quality_count",
];
const METHOD_PROFILE_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "frame_count",
    "active_frame_count",
    "candidate_count_total",
    "quality_count_total",
    "all_three_count",
    "gaussian_only_count",
    "dog_only_count",
    "partial_combination_count",
    "no_gaussian_count",
    "no_method_count",
    "no_gaussian_quality_count",
    "all_three_fraction",
    "dog_only_fraction",
    "no_gaussian_fraction",
    "no_gaussian_quality_fraction",
];
const SOURCE_SUBGROUP_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "proposal_subgroup",
    "proposal_subgroup_label",
    "anchor_count",
    "anchor_quality_count",
    "frame_count",
    "required_presence",
    "association_radius_px",
    "candidate_median_presence",
    "candidate_mean_presence",
    "candidate_presence_ge_required_count",
    "candidate_presence_all_frames_count",
    "quality_median_presence",
    "quality_mean_presence",
    "quality_presence_ge_required_count",
    "quality_presence_all_frames_count",
];
const DIAGNOSTIC_SUBGROUP_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "diagnostic_subgroup",
    "diagnostic_subgroup_label",
    "diagnostic_subgroup_definition",
    "anchor_count",
    "anchor_quality_count",
    "frame_count",
    "required_presence",
    "association_radius_px",
    "candidate_median_presence",
    "candidate_mean_presence",
    "candidate_presence_ge_required_count",
    "candidate_presence_all_frames_count",
    "candidate_same_subgroup_median_presence",
    "candidate_same_subgroup_mean_presence",
    "candidate_same_subgroup_presence_ge_required_count",
    "candidate_same_subgroup_presence_all_frames_count",
    "quality_median_presence",
    "quality_mean_presence",
    "quality_presence_ge_required_count",
    "quality_presence_all_frames_count",
    "quality_same_subgroup_median_presence",
    "quality_same_subgroup_mean_presence",
    "quality_same_subgroup_presence_ge_required_count",
    "quality_same_subgroup_presence_all_frames_count",
];
const DIAGNOSTIC_SOURCE_FIELDS: &[&str] = &[
    "detection_id",
    "feature_class",
    "feature_class_label",
    "diagnostic_subgroup",
    "diagnostic_subgroup_label",
    "diagnostic_subgroup_definition",
    "quality_passed",
    "peak_x",
    "peak_y",
    "x",
    "y",
    "flux_snr",
    "filter_snr",
    "fwhm",
    "ellipticity",
    "sharpness",
    "psf_support_pixels",
    "footprint_pixels",
    "centroid_shift_px",
    "flags",
    "proposal_methods",
    "candidate_presence",
    "candidate_same_subgroup_presence",
    "quality_presence",
    "quality_same_subgroup_presence",
];
const TEMPORAL_FIELDS: &[&str] = &[
    "feature_class",
    "feature_class_label",
    "frame_count",
    "active_frame_count",
    "quality_active_frame_count",
    "candidate_count_mean",
    "candidate_count_min",
    "candidate_count_max",
    "candidate_count_cv",
    "candidate_fraction_mean",
    "candidate_fraction_cv",
    "quality_count_mean",
    "quality_count_min",
    "quality_count_max",
    "quality_count_cv",
    "quality_fraction_mean",
    "quality_fraction_weighted",
];
const TRANSITION_FIELDS: &[&str] = &[
    "layer",
    "anchor_feature_class",
    "anchor_feature_class_label",
    "response_feature_class",
    "response_feature_class_label",
    "anchor_count",
    "frame_count",
    "possible_match_count",
    "matched_count",
    "match_rate",
    "mean_matches_per_anchor",
];

#[derive(Clone)]
struct Options {
    data_dir: PathBuf,
    out_dir: PathBuf,
    threshold_sigma: f64,
    min_distance: usize,
    aperture_radius: usize,
    psf_fwhm: f64,
    proposal_mode: String,
    background_box_size: usize,
    background_sample_limit: usize,
    min_flux_snr: f64,
    min_psf_support_pixels: usize,
    max_sources: Option<usize>,
    registration_radius_px: f64,
    association_radius_px: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::new(),
            out_dir: PathBuf::from("tmp/sequence-feature-audit"),
            threshold_sigma: 4.0,
            min_distance: 4,
            aperture_radius: 4,
            psf_fwhm: 2.0,
            proposal_mode: "hybrid".into(),
            background_box_size: 128,
            background_sample_limit: 100_000,
            min_flux_snr: 5.0,
            min_psf_support_pixels: 3,
            max_sources: None,
            registration_radius_px: 8.0,
            association_radius_px: 1.0,
        }
    }
}

fn parse_options(args: &[String]) -> Result<Options> {
    let mut options = Options::default();
    let mut positional = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        let arg = &args[index];
        if !arg.starts_with('-') {
            positional.push(PathBuf::from(arg));
            index += 1;
            continue;
        }
        if arg == "-h" || arg == "--help" {
            return Err(HELP.into());
        }
        let (key, inline) = arg[2..]
            .split_once('=')
            .map(|(k, v)| (k, Some(v.to_owned())))
            .unwrap_or((&arg[2..], None));
        index += 1;
        let raw = if let Some(value) = inline {
            value
        } else {
            let value = args
                .get(index)
                .filter(|value| !value.starts_with('-'))
                .ok_or_else(|| format!("--{key} requires a value"))?
                .clone();
            index += 1;
            value
        };
        macro_rules! number {
            ($ty:ty) => {
                raw.parse::<$ty>()
                    .map_err(|_| format!("invalid --{key}: {raw}"))?
            };
        }
        match key {
            "out-dir" => options.out_dir = PathBuf::from(raw),
            "threshold-sigma" => options.threshold_sigma = number!(f64),
            "min-distance" => options.min_distance = number!(usize),
            "aperture-radius" => options.aperture_radius = number!(usize),
            "psf-fwhm" => options.psf_fwhm = number!(f64),
            "proposal-mode" => options.proposal_mode = raw,
            "background-box-size" => options.background_box_size = number!(usize),
            "background-sample-limit" => options.background_sample_limit = number!(usize),
            "min-flux-snr" => options.min_flux_snr = number!(f64),
            "min-psf-support-pixels" => options.min_psf_support_pixels = number!(usize),
            "max-sources" => {
                let value = number!(usize);
                options.max_sources = (value > 0).then_some(value);
            }
            "registration-radius-px" => options.registration_radius_px = number!(f64),
            "association-radius-px" => options.association_radius_px = number!(f64),
            _ => return Err(format!("unknown option --{key}")),
        }
    }
    if positional.len() != 1 {
        return Err("expected exactly one data directory".into());
    }
    options.data_dir = positional.remove(0);
    if !options.threshold_sigma.is_finite()
        || options.threshold_sigma <= 0.0
        || options.min_distance == 0
        || options.aperture_radius == 0
        || !options.psf_fwhm.is_finite()
        || options.psf_fwhm <= 0.0
        || options.background_box_size < 16
        || options.background_sample_limit == 0
        || !options.min_flux_snr.is_finite()
        || options.min_flux_snr <= 0.0
        || !(1..=9).contains(&options.min_psf_support_pixels)
        || !options.registration_radius_px.is_finite()
        || options.registration_radius_px <= 0.0
        || !options.association_radius_px.is_finite()
        || options.association_radius_px <= 0.0
        || !["gaussian", "hybrid", "ensemble"].contains(&options.proposal_mode.as_str())
    {
        return Err("invalid detector, background, or association options".into());
    }
    Ok(options)
}

#[derive(Clone)]
struct FrameData {
    candidates: Vec<Detection>,
    candidate_points: Vec<[f64; 2]>,
    centroid_points: Vec<[f64; 2]>,
    quality_points: Vec<[f64; 2]>,
    candidate_classes: Vec<&'static str>,
    quality_classes: Vec<&'static str>,
    candidate_diagnostics: Vec<Option<&'static str>>,
    quality_diagnostics: Vec<Option<&'static str>>,
    candidate_proposals: Vec<Option<&'static str>>,
}

fn has(source: &Detection, flag: &str) -> bool {
    source.flags.iter().any(|value| value == flag)
}

fn diagnostic_subgroup(source: &Detection, class: &str) -> Option<&'static str> {
    let code = has(source, "CODE_PATTERN");
    let overflow = has(source, "NEGATIVE_OVERFLOW");
    let sat = has(source, "SATURATED");
    let edge = has(source, "EDGE");
    let partial = has(source, "PARTIAL_MASKED");
    let masked = has(source, "MASKED");
    let spike = has(source, "SPIKE");
    let support = has(source, "INSUFFICIENT_PSF_SUPPORT");
    Some(match class {
        "range_anomaly" => {
            if code {
                "range_code_pattern"
            } else if overflow {
                "range_negative_overflow"
            } else if sat {
                "range_saturated"
            } else {
                "range_other"
            }
        }
        "linear_artifact" => "line_artifact",
        "masked_or_edge" => {
            if masked {
                "masked_hard"
            } else if edge && partial {
                "masked_partial_edge"
            } else if partial {
                "masked_partial"
            } else if edge {
                "masked_edge_only"
            } else {
                "masked_other"
            }
        }
        "crowded_blend" => {
            if has(source, "UNRESOLVED_BLEND") {
                "blend_unresolved"
            } else {
                "blend_other"
            }
        }
        "spike_or_support" => {
            if spike && support {
                "spike_and_psf_support"
            } else if spike {
                "spike_only"
            } else if support {
                "psf_support_only"
            } else if has(source, "NARROW") || has(source, "SMALL_FOOTPRINT") {
                "narrow_or_small_footprint"
            } else {
                "spike_other"
            }
        }
        "weak_or_background" => {
            if has(source, "NON_POSITIVE_FLUX") {
                "weak_non_positive_flux"
            } else if has(source, "BACKGROUND_UNCERTAIN") {
                "weak_background_uncertain"
            } else if has(source, "LOW_FLUX_SNR") {
                "weak_low_flux_snr"
            } else {
                "weak_other"
            }
        }
        "shape_outlier" => {
            if has(source, "NO_SHAPE") {
                "shape_no_shape"
            } else if has(source, "BROAD") {
                "shape_broad"
            } else if has(source, "ELONGATED") {
                "shape_elongated"
            } else if has(source, "DIFFUSE") {
                "shape_diffuse"
            } else {
                "shape_other"
            }
        }
        "other_rejected" => "other_rejected",
        _ => return None,
    })
}

fn proposal_subgroup(source: &Detection, class: &str) -> Option<&'static str> {
    if class != "compact_quality" {
        return None;
    }
    let gaussian = source.proposal_methods.iter().any(|v| v == "gaussian");
    let narrow = source.proposal_methods.iter().any(|v| v == "dog_narrow");
    let broad = source.proposal_methods.iter().any(|v| v == "dog_broad");
    if !gaussian {
        return Some(
            if source.deblend_delta_bic.is_some() && source.deblend_component_snr.is_some() {
                "dog_only_deblend"
            } else {
                "dog_only_no_deblend"
            },
        );
    }
    if narrow && broad {
        Some("all_three")
    } else if source.proposal_methods.len() == 1 {
        Some("gaussian_only")
    } else {
        Some("compact_other")
    }
}

fn shifted(point: [f64; 2], shift: [f64; 2]) -> [f64; 2] {
    [point[0] - shift[0], point[1] - shift[1]]
}

fn reciprocal_pairs(
    anchor: &[[f64; 2]],
    frame: &[[f64; 2]],
    radius: f64,
) -> Result<Vec<(usize, usize)>> {
    if anchor.is_empty() || frame.is_empty() {
        return Ok(Vec::new());
    }
    let anchor_tree = CompatibleKdTree::new(anchor)?;
    let frame_tree = CompatibleKdTree::new(frame)?;
    let forward: Vec<_> = anchor
        .iter()
        .map(|point| frame_tree.nearest(*point))
        .collect();
    let reverse: Vec<_> = frame
        .iter()
        .map(|point| anchor_tree.nearest(*point))
        .collect();
    Ok(forward
        .iter()
        .enumerate()
        .filter_map(|(a, result)| {
            let (f, distance) = (*result)?;
            if distance >= radius {
                return None;
            }
            reverse
                .get(f)
                .and_then(|r| *r)
                .filter(|(back, reverse_distance)| *back == a && *reverse_distance < radius)
                .map(|_| (a, f))
        })
        .collect())
}

fn median(values: &[usize]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<_> = values.iter().map(|v| *v as f64).collect();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    Some(if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    })
}
fn mean(values: &[usize]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<usize>() as f64 / values.len() as f64)
}
fn count_at_least(values: &[usize], threshold: usize) -> usize {
    values.iter().filter(|v| **v >= threshold).count()
}
fn ratio(n: usize, d: usize) -> Value {
    if d == 0 {
        Value::Null
    } else {
        json!(n as f64 / d as f64)
    }
}

fn class_index(name: &str) -> usize {
    SOURCE_FEATURE_CLASSES
        .iter()
        .position(|(class, _)| *class == name)
        .unwrap_or(0)
}

fn diagnostic_spec(id: &str) -> (&'static str, &'static str) {
    match id {
        "range_code_pattern" => ("范围异常·重复码峰值", "首要类别为范围异常且含 CODE_PATTERN；这是值域/编码审计子组，不是恒星类型。"),
        "range_negative_overflow" => ("范围异常·极端负值", "首要类别为范围异常且含 NEGATIVE_OVERFLOW；优先于饱和子组显示。"),
        "range_saturated" => ("范围异常·饱和标志", "首要类别为范围异常且含 SATURATED；当前只表示检测器标志，不等于已完成满阱标定。"),
        "range_other" => ("范围异常·其他组合", "首要类别为范围异常，但未命中当前三个值域子组。"),
        "line_artifact" => ("线状/拖影", "首要类别为线状候选且含 LINE_ARTIFACT；需要结合长度、宽度、方向和跨帧一致性复核。"),
        "masked_hard" => ("边缘/掩膜·硬掩膜", "首要类别为边缘/掩膜且含 MASKED；即使同时有 EDGE 或 PARTIAL_MASKED，也按硬掩膜优先归组。"),
        "masked_partial_edge" => ("边缘/掩膜·边界+部分掩膜", "首要类别为边缘/掩膜，同时含 EDGE 和 PARTIAL_MASKED，但不含 MASKED。"),
        "masked_partial" => ("边缘/掩膜·部分掩膜", "首要类别为边缘/掩膜，含 PARTIAL_MASKED 但不含 EDGE 或 MASKED。"),
        "masked_edge_only" => ("边缘/掩膜·纯边界", "首要类别为边缘/掩膜，含 EDGE 但不含 PARTIAL_MASKED 或 MASKED。"),
        "masked_other" => ("边缘/掩膜·其他组合", "首要类别为边缘/掩膜，但未命中当前边缘/掩膜组合。"),
        "blend_unresolved" => ("拥挤/未分辨近邻", "首要类别为拥挤近邻且含 UNRESOLVED_BLEND；需以双 PSF 证据、分离度和跨帧位置共同判断。"),
        "blend_other" => ("拥挤/未分辨近邻·其他", "首要类别为拥挤近邻，但未保留 UNRESOLVED_BLEND 标志。"),
        "spike_and_psf_support" => ("尖峰/支持·尖峰+PSF支持不足", "首要类别为尖峰/PSF支持不足，同时含 SPIKE 和 INSUFFICIENT_PSF_SUPPORT。"),
        "spike_only" => ("尖峰/支持·尖峰", "首要类别为尖峰/PSF支持不足，含 SPIKE 但不含 INSUFFICIENT_PSF_SUPPORT。"),
        "psf_support_only" => ("尖峰/支持·支持不足", "首要类别为尖峰/PSF支持不足，含 INSUFFICIENT_PSF_SUPPORT 但不含 SPIKE。"),
        "narrow_or_small_footprint" => ("尖峰/支持·窄峰或小足迹", "首要类别为尖峰/PSF支持不足，命中 NARROW 或 SMALL_FOOTPRINT，但未命中上述子组。"),
        "spike_other" => ("尖峰/支持·其他组合", "首要类别为尖峰/PSF支持不足，但未命中当前尖峰/支持子组。"),
        "weak_non_positive_flux" => ("弱通量/背景·非正通量", "首要类别为弱通量/背景且含 NON_POSITIVE_FLUX；它优先于一般低 SNR。"),
        "weak_background_uncertain" => ("弱通量/背景·背景不确定", "首要类别为弱通量/背景且含 BACKGROUND_UNCERTAIN。"),
        "weak_low_flux_snr" => ("弱通量/背景·通量SNR不足", "首要类别为弱通量/背景且含 LOW_FLUX_SNR。"),
        "weak_other" => ("弱通量/背景·其他组合", "首要类别为弱通量/背景，但未命中当前弱源子组。"),
        "shape_no_shape" => ("形状异常·无可用形状", "首要类别为形状异常且含 NO_SHAPE。"),
        "shape_broad" => ("形状异常·宽峰", "首要类别为形状异常且含 BROAD。"),
        "shape_elongated" => ("形状异常·拉长", "首要类别为形状异常且含 ELONGATED。"),
        "shape_diffuse" => ("形状异常·弥散", "首要类别为形状异常且含 DIFFUSE。"),
        "shape_other" => ("形状异常·其他组合", "首要类别为形状异常，但未命中当前形状子组。"),
        _ => ("其他拒绝", "未命中已定义特征旗标且未通过质量层。"),
    }
}

const PROPOSAL_SUBGROUPS: [(&str, &str); 5] = [
    ("all_three", "紧凑质量·Gaussian+DoG窄/宽三路共同支持"),
    ("gaussian_only", "紧凑质量·Gaussian-only"),
    ("dog_only_deblend", "紧凑质量·DoG-only·有近邻双PSF记录"),
    ("dog_only_no_deblend", "紧凑质量·DoG-only·无近邻双PSF记录"),
    ("compact_other", "紧凑质量·其他来源组合"),
];

#[derive(Default)]
struct Audit {
    frame_count: usize,
    required: usize,
    options: Option<Options>,
    shifts: Vec<[f64; 2]>,
    frames: Vec<FrameData>,
    frame_rows: Vec<Value>,
    method_rows: Vec<Value>,
}

fn analyze(options: Options) -> Result<Audit> {
    let mut paths = fs::read_dir(&options.data_dir)
        .map_err(|error| format!("cannot read {}: {error}", options.data_dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("fits"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "数据目录没有 FITS 文件：{}",
            options.data_dir.display()
        ));
    }
    let frame_count = paths.len();
    let required = frame_count.min(((frame_count as f64 * 0.8).ceil() as usize).max(1));
    let mut audit = Audit {
        frame_count,
        required,
        options: Some(options.clone()),
        ..Default::default()
    };
    let mut previous_quality: Option<Vec<Detection>> = None;
    let mut cumulative = [0.0, 0.0];

    for (frame_index, path) in paths.iter().enumerate() {
        let frame = fits::read_fits(path, false)?;
        let mut measurement = MeasurementOptions::default();
        measurement.aperture_radius = options.aperture_radius;
        measurement.psf_fwhm = options.psf_fwhm;
        measurement.min_flux_snr = options.min_flux_snr;
        measurement.min_psf_support_pixels = options.min_psf_support_pixels;
        measurement.refine_local_background = true;
        let detection = DetectionOptions {
            measurement,
            threshold_sigma: options.threshold_sigma,
            min_distance: options.min_distance,
            max_sources: options.max_sources,
            background_box_size: options.background_box_size,
            background_sample_limit: options.background_sample_limit,
            reject_linear_artifacts: true,
            proposal_mode: options.proposal_mode.clone(),
            ..DetectionOptions::default()
        };
        let analysis = pipeline::analyze_frame(
            frame,
            None,
            None,
            &FrameOptions {
                detection,
                ..FrameOptions::default()
            },
            None,
        )?;
        let candidate_count = analysis.detection.candidate_count;
        let returned_count = analysis.detection.sources.len();
        let quality_count = analysis.detection.quality_count;
        let background = analysis.detection.background;
        let noise = analysis.detection.noise;
        let candidates = analysis.detection.sources;
        let quality = candidates
            .iter()
            .filter(|source| source.quality_passed)
            .cloned()
            .collect::<Vec<_>>();
        if let Some(previous) = &previous_quality {
            let delta = sequence_registration::estimate_translation(
                previous,
                &quality,
                options.registration_radius_px,
            )?;
            cumulative[0] += delta[0];
            cumulative[1] += delta[1];
        }
        let candidate_classes = candidates
            .iter()
            .map(source_classification::classify_source_feature)
            .collect::<Vec<_>>();
        let quality_classes = quality
            .iter()
            .map(source_classification::classify_source_feature)
            .collect::<Vec<_>>();
        let candidate_diagnostics = candidates
            .iter()
            .zip(&candidate_classes)
            .map(|(s, c)| diagnostic_subgroup(s, c))
            .collect::<Vec<_>>();
        let quality_diagnostics = quality
            .iter()
            .zip(&quality_classes)
            .map(|(s, c)| diagnostic_subgroup(s, c))
            .collect::<Vec<_>>();
        let candidate_proposals = candidates
            .iter()
            .zip(&candidate_classes)
            .map(|(s, c)| proposal_subgroup(s, c))
            .collect::<Vec<_>>();
        let shift = cumulative;
        let candidate_points = candidates
            .iter()
            .map(|s| shifted(s.peak_position(), shift))
            .collect();
        let centroid_points = candidates
            .iter()
            .map(|s| shifted(s.measurement_position(), shift))
            .collect();
        let quality_points = quality
            .iter()
            .map(|s| shifted(s.measurement_position(), shift))
            .collect();

        let features = source_classification::summarize_source_features(&candidates);
        for summary in features {
            audit.frame_rows.push(json!({
                "frame_index":frame_index+1, "path":path.to_string_lossy(),
                "candidate_count":candidate_count,
                "returned_count":returned_count,
                "quality_count":quality_count,
                "background_adu":background, "noise_adu":noise,
                "feature_class":summary.feature_class, "feature_class_label":summary.feature_class_label,
                "feature_candidate_count":summary.candidate_count, "feature_quality_count":summary.quality_count,
            }));
        }
        for summary in source_classification::summarize_source_proposal_methods(&candidates) {
            audit.method_rows.push(json!({
                "frame_index":frame_index+1, "path":path.to_string_lossy(),
                "feature_class":summary.feature_class, "feature_class_label":summary.feature_class_label,
                "feature_candidate_count":summary.candidate_count, "feature_quality_count":summary.quality_count,
                "all_three_count":summary.all_three_count, "gaussian_only_count":summary.gaussian_only_count,
                "dog_only_count":summary.dog_only_count, "partial_combination_count":summary.partial_combination_count,
                "no_gaussian_count":summary.no_gaussian_count, "no_method_count":summary.no_method_count,
                "no_gaussian_quality_count":summary.no_gaussian_quality_count,
            }));
        }
        audit.shifts.push(shift);
        audit.frames.push(FrameData {
            candidates,
            candidate_points,
            centroid_points,
            quality_points,
            candidate_classes,
            quality_classes,
            candidate_diagnostics,
            quality_diagnostics,
            candidate_proposals,
        });
        previous_quality = Some(quality);
        eprintln!(
            "feature persistence: frame {}/{}",
            frame_index + 1,
            frame_count
        );
    }
    Ok(audit)
}

const DIAGNOSTIC_GROUPS: &[(&str, &str)] = &[
    ("range_anomaly", "range_code_pattern"),
    ("range_anomaly", "range_negative_overflow"),
    ("range_anomaly", "range_saturated"),
    ("range_anomaly", "range_other"),
    ("linear_artifact", "line_artifact"),
    ("masked_or_edge", "masked_hard"),
    ("masked_or_edge", "masked_partial_edge"),
    ("masked_or_edge", "masked_partial"),
    ("masked_or_edge", "masked_edge_only"),
    ("masked_or_edge", "masked_other"),
    ("crowded_blend", "blend_unresolved"),
    ("crowded_blend", "blend_other"),
    ("spike_or_support", "spike_and_psf_support"),
    ("spike_or_support", "spike_only"),
    ("spike_or_support", "psf_support_only"),
    ("spike_or_support", "narrow_or_small_footprint"),
    ("spike_or_support", "spike_other"),
    ("weak_or_background", "weak_non_positive_flux"),
    ("weak_or_background", "weak_background_uncertain"),
    ("weak_or_background", "weak_low_flux_snr"),
    ("weak_or_background", "weak_other"),
    ("shape_outlier", "shape_no_shape"),
    ("shape_outlier", "shape_broad"),
    ("shape_outlier", "shape_elongated"),
    ("shape_outlier", "shape_diffuse"),
    ("shape_outlier", "shape_other"),
    ("other_rejected", "other_rejected"),
];

fn source_row(
    source: &Detection,
    class: &str,
    subgroup: &str,
    candidate_presence: usize,
    candidate_same: usize,
    quality_presence: usize,
    quality_same: usize,
) -> Value {
    let (subgroup_label, definition) = diagnostic_spec(subgroup);
    json!({
        "detection_id":source.detection_id, "feature_class":class,
        "feature_class_label":SOURCE_FEATURE_CLASSES[class_index(class)].1,
        "diagnostic_subgroup":subgroup, "diagnostic_subgroup_label":subgroup_label,
        "diagnostic_subgroup_definition":definition, "quality_passed":source.quality_passed,
        "peak_x":source.peak_x.unwrap_or(source.x), "peak_y":source.peak_y.unwrap_or(source.y),
        "x":source.x,"y":source.y,"flux_snr":source.flux_snr,"filter_snr":source.filter_snr,
        "fwhm":source.fwhm,"ellipticity":source.ellipticity,"sharpness":source.sharpness,
        "psf_support_pixels":source.psf_support_pixels,"footprint_pixels":source.footprint_pixels,
        "centroid_shift_px":source.centroid_shift_px,"flags":source.flags.join("|"),
        "proposal_methods":source.proposal_methods.join("|"),"candidate_presence":candidate_presence,
        "candidate_same_subgroup_presence":candidate_same,"quality_presence":quality_presence,
        "quality_same_subgroup_presence":quality_same,
    })
}

fn build_rows(audit: &Audit) -> Result<Value> {
    let options = audit.options.as_ref().ok_or("missing audit options")?;
    let anchor = audit.frames.first().ok_or("missing anchor frame")?;
    let candidate_pairs = audit
        .frames
        .iter()
        .map(|frame| {
            reciprocal_pairs(
                &anchor.candidate_points,
                &frame.candidate_points,
                options.association_radius_px,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let quality_pairs_all = audit
        .frames
        .iter()
        .map(|frame| {
            reciprocal_pairs(
                &anchor.centroid_points,
                &frame.quality_points,
                options.association_radius_px,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut candidate_presence = vec![0usize; anchor.candidates.len()];
    let mut quality_presence = vec![0usize; anchor.candidates.len()];
    let mut candidate_same = vec![0usize; anchor.candidates.len()];
    let mut quality_same = vec![0usize; anchor.candidates.len()];
    for (frame_index, pairs) in candidate_pairs.iter().enumerate() {
        for &(a, b) in pairs {
            candidate_presence[a] += 1;
            if anchor.candidate_diagnostics[a].is_some()
                && anchor.candidate_diagnostics[a]
                    == audit.frames[frame_index].candidate_diagnostics[b]
            {
                candidate_same[a] += 1;
            }
        }
    }
    for (frame_index, pairs) in quality_pairs_all.iter().enumerate() {
        for &(a, b) in pairs {
            quality_presence[a] += 1;
            if anchor.candidate_diagnostics[a].is_some()
                && anchor.candidate_diagnostics[a]
                    == audit.frames[frame_index].quality_diagnostics[b]
            {
                quality_same[a] += 1;
            }
        }
    }

    let mut persistence = Vec::new();
    for &(class, label) in &SOURCE_FEATURE_CLASSES {
        let indices = anchor
            .candidate_classes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| (*c == class).then_some(i))
            .collect::<Vec<_>>();
        let q_indices = indices
            .iter()
            .copied()
            .filter(|i| anchor.candidates[*i].quality_passed)
            .collect::<Vec<_>>();
        let cp = indices
            .iter()
            .map(|i| candidate_presence[*i])
            .collect::<Vec<_>>();
        let qp = indices
            .iter()
            .map(|i| quality_presence[*i])
            .collect::<Vec<_>>();
        persistence.push(json!({
            "feature_class":class,"feature_class_label":label,"anchor_candidate_count":indices.len(),
            "anchor_quality_count":q_indices.len(),"frame_count":audit.frame_count,"required_presence":audit.required,
            "association_radius_px":options.association_radius_px,
            "candidate_median_presence":median(&cp),"candidate_mean_presence":mean(&cp),
            "candidate_presence_ge_required_count":count_at_least(&cp,audit.required),
            "candidate_presence_all_frames_count":count_at_least(&cp,audit.frame_count),
            "quality_median_presence":median(&qp),"quality_mean_presence":mean(&qp),
            "quality_presence_ge_required_count":count_at_least(&qp,audit.required),
            "quality_presence_all_frames_count":count_at_least(&qp,audit.frame_count),
        }));
    }

    let mut source_subgroups = Vec::new();
    for &(subgroup, label) in &PROPOSAL_SUBGROUPS {
        let indices = anchor
            .candidate_proposals
            .iter()
            .enumerate()
            .filter_map(|(i, v)| (*v == Some(subgroup)).then_some(i))
            .collect::<Vec<_>>();
        if indices.is_empty() {
            continue;
        }
        let cp = indices
            .iter()
            .map(|i| candidate_presence[*i])
            .collect::<Vec<_>>();
        let qp = indices
            .iter()
            .map(|i| quality_presence[*i])
            .collect::<Vec<_>>();
        source_subgroups.push(json!({
            "feature_class":"compact_quality","feature_class_label":SOURCE_FEATURE_CLASSES[class_index("compact_quality")].1,
            "proposal_subgroup":subgroup,"proposal_subgroup_label":label,"anchor_count":indices.len(),
            "anchor_quality_count":indices.iter().filter(|i|anchor.candidates[**i].quality_passed).count(),
            "frame_count":audit.frame_count,"required_presence":audit.required,"association_radius_px":options.association_radius_px,
            "candidate_median_presence":median(&cp),"candidate_mean_presence":mean(&cp),
            "candidate_presence_ge_required_count":count_at_least(&cp,audit.required),
            "candidate_presence_all_frames_count":count_at_least(&cp,audit.frame_count),
            "quality_median_presence":median(&qp),"quality_mean_presence":mean(&qp),
            "quality_presence_ge_required_count":count_at_least(&qp,audit.required),
            "quality_presence_all_frames_count":count_at_least(&qp,audit.frame_count),
        }));
    }

    let mut diagnostic_rows = Vec::new();
    for &(class, subgroup) in DIAGNOSTIC_GROUPS {
        let indices = anchor
            .candidate_classes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                (*c == class && anchor.candidate_diagnostics[i] == Some(subgroup)).then_some(i)
            })
            .collect::<Vec<_>>();
        if indices.is_empty() {
            continue;
        }
        let cp = indices
            .iter()
            .map(|i| candidate_presence[*i])
            .collect::<Vec<_>>();
        let cs = indices
            .iter()
            .map(|i| candidate_same[*i])
            .collect::<Vec<_>>();
        let qp = indices
            .iter()
            .map(|i| quality_presence[*i])
            .collect::<Vec<_>>();
        let qs = indices.iter().map(|i| quality_same[*i]).collect::<Vec<_>>();
        let (label, definition) = diagnostic_spec(subgroup);
        diagnostic_rows.push(json!({
            "feature_class":class,"feature_class_label":SOURCE_FEATURE_CLASSES[class_index(class)].1,
            "diagnostic_subgroup":subgroup,"diagnostic_subgroup_label":label,"diagnostic_subgroup_definition":definition,
            "anchor_count":indices.len(),"anchor_quality_count":indices.iter().filter(|i|anchor.candidates[**i].quality_passed).count(),
            "frame_count":audit.frame_count,"required_presence":audit.required,"association_radius_px":options.association_radius_px,
            "candidate_median_presence":median(&cp),"candidate_mean_presence":mean(&cp),
            "candidate_presence_ge_required_count":count_at_least(&cp,audit.required),"candidate_presence_all_frames_count":count_at_least(&cp,audit.frame_count),
            "candidate_same_subgroup_median_presence":median(&cs),"candidate_same_subgroup_mean_presence":mean(&cs),
            "candidate_same_subgroup_presence_ge_required_count":count_at_least(&cs,audit.required),"candidate_same_subgroup_presence_all_frames_count":count_at_least(&cs,audit.frame_count),
            "quality_median_presence":median(&qp),"quality_mean_presence":mean(&qp),
            "quality_presence_ge_required_count":count_at_least(&qp,audit.required),"quality_presence_all_frames_count":count_at_least(&qp,audit.frame_count),
            "quality_same_subgroup_median_presence":median(&qs),"quality_same_subgroup_mean_presence":mean(&qs),
            "quality_same_subgroup_presence_ge_required_count":count_at_least(&qs,audit.required),"quality_same_subgroup_presence_all_frames_count":count_at_least(&qs,audit.frame_count),
        }));
    }
    let diagnostic_sources = anchor
        .candidates
        .iter()
        .enumerate()
        .filter_map(|(i, source)| {
            let class = anchor.candidate_classes[i];
            anchor.candidate_diagnostics[i].map(|group| {
                source_row(
                    source,
                    class,
                    group,
                    candidate_presence[i],
                    candidate_same[i],
                    quality_presence[i],
                    quality_same[i],
                )
            })
        })
        .collect::<Vec<_>>();

    let mut transition_counts =
        vec![0usize; 2 * SOURCE_FEATURE_CLASSES.len() * SOURCE_FEATURE_CLASSES.len()];
    for (frame_index, pairs) in candidate_pairs.iter().enumerate() {
        for &(a, b) in pairs {
            let ai = class_index(anchor.candidate_classes[a]);
            let bi = class_index(audit.frames[frame_index].candidate_classes[b]);
            transition_counts[ai * 9 + bi] += 1;
        }
    }
    let quality_anchor_indices = anchor
        .candidates
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.quality_passed.then_some(i))
        .collect::<Vec<_>>();
    let quality_anchor_points = quality_anchor_indices
        .iter()
        .map(|i| anchor.centroid_points[*i])
        .collect::<Vec<_>>();
    let quality_pairs = audit
        .frames
        .iter()
        .map(|frame| {
            reciprocal_pairs(
                &quality_anchor_points,
                &frame.quality_points,
                options.association_radius_px,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    for (frame_index, pairs) in quality_pairs.iter().enumerate() {
        for &(a, b) in pairs {
            let ai = class_index(anchor.candidate_classes[quality_anchor_indices[a]]);
            let bi = class_index(audit.frames[frame_index].quality_classes[b]);
            transition_counts[81 + ai * 9 + bi] += 1;
        }
    }
    let mut transitions = Vec::new();
    for (layer, offset, anchor_classes) in [
        ("candidate", 0usize, anchor.candidate_classes.as_slice()),
        (
            "quality",
            81usize,
            &quality_anchor_indices
                .iter()
                .map(|i| anchor.candidate_classes[*i])
                .collect::<Vec<_>>(),
        ),
    ] {
        for (ai, (aclass, alabel)) in SOURCE_FEATURE_CLASSES.iter().enumerate() {
            let anchor_count = anchor_classes.iter().filter(|v| **v == *aclass).count();
            for (bi, (bclass, blabel)) in SOURCE_FEATURE_CLASSES.iter().enumerate() {
                let matched = transition_counts[offset + ai * 9 + bi];
                let possible = anchor_count * audit.frame_count;
                transitions.push(json!({
                    "layer":layer,"anchor_feature_class":aclass,"anchor_feature_class_label":alabel,
                    "response_feature_class":bclass,"response_feature_class_label":blabel,
                    "anchor_count":anchor_count,"frame_count":audit.frame_count,"possible_match_count":possible,
                    "matched_count":matched,"match_rate":if possible==0{Value::Null}else{json!(matched as f64/possible as f64)},
                    "mean_matches_per_anchor":if anchor_count==0{Value::Null}else{json!(matched as f64/anchor_count as f64)},
                }));
            }
        }
    }

    let mut temporal_profiles = Vec::new();
    for &(class, label) in &SOURCE_FEATURE_CLASSES {
        let rows = audit
            .frame_rows
            .iter()
            .filter(|row| row["feature_class"] == class)
            .collect::<Vec<_>>();
        let candidates = rows
            .iter()
            .map(|r| r["feature_candidate_count"].as_u64().unwrap_or(0) as usize)
            .collect::<Vec<_>>();
        let quality = rows
            .iter()
            .map(|r| r["feature_quality_count"].as_u64().unwrap_or(0) as usize)
            .collect::<Vec<_>>();
        let totals = rows
            .iter()
            .map(|r| r["candidate_count"].as_u64().unwrap_or(0) as usize)
            .collect::<Vec<_>>();
        let fractions = candidates
            .iter()
            .zip(&totals)
            .map(|(n, d)| if *d == 0 { 0.0 } else { *n as f64 / *d as f64 })
            .collect::<Vec<_>>();
        let quality_fractions = quality
            .iter()
            .zip(&candidates)
            .map(|(n, d)| if *d == 0 { 0.0 } else { *n as f64 / *d as f64 })
            .collect::<Vec<_>>();
        let cv = |values: &[usize]| -> Value {
            let m = values.iter().sum::<usize>() as f64 / values.len().max(1) as f64;
            if m <= 0.0 {
                Value::Null
            } else {
                let var = values.iter().map(|v| (*v as f64 - m).powi(2)).sum::<f64>()
                    / values.len() as f64;
                json!(var.sqrt() / m)
            }
        };
        let avg = |values: &[f64]| values.iter().sum::<f64>() / values.len().max(1) as f64;
        let total_c = candidates.iter().sum::<usize>();
        let total_q = quality.iter().sum::<usize>();
        let count_as_f64 = |values: &[usize]| values.iter().map(|v| *v as f64).collect::<Vec<_>>();
        let candidate_count_values = count_as_f64(&candidates);
        let quality_count_values = count_as_f64(&quality);
        let candidate_fraction_mean = avg(&fractions);
        let candidate_fraction_cv = if candidate_fraction_mean <= 0.0 {
            Value::Null
        } else {
            let variance = fractions
                .iter()
                .map(|value| (value - candidate_fraction_mean).powi(2))
                .sum::<f64>()
                / fractions.len().max(1) as f64;
            json!(variance.sqrt() / candidate_fraction_mean)
        };
        let quality_fraction_weighted = if total_c == 0 {
            Value::Null
        } else {
            json!(total_q as f64 / total_c as f64)
        };
        temporal_profiles.push(json!({
            "feature_class":class,"feature_class_label":label,"frame_count":rows.len(),
            "active_frame_count":candidates.iter().filter(|v|**v>0).count(),"quality_active_frame_count":quality.iter().filter(|v|**v>0).count(),
            "candidate_count_mean":avg(&candidate_count_values),"candidate_count_min":candidates.iter().min().copied().unwrap_or(0),"candidate_count_max":candidates.iter().max().copied().unwrap_or(0),"candidate_count_cv":cv(&candidates),
            "candidate_fraction_mean":candidate_fraction_mean,"candidate_fraction_cv":candidate_fraction_cv,
            "quality_count_mean":avg(&quality_count_values),"quality_count_min":quality.iter().min().copied().unwrap_or(0),"quality_count_max":quality.iter().max().copied().unwrap_or(0),"quality_count_cv":cv(&quality),
            "quality_fraction_mean":avg(&quality_fractions),"quality_fraction_weighted":quality_fraction_weighted,
        }));
    }
    let mut method_profiles = Vec::new();
    for &(class, label) in &SOURCE_FEATURE_CLASSES {
        let rows = audit
            .method_rows
            .iter()
            .filter(|r| r["feature_class"] == class)
            .collect::<Vec<_>>();
        let sum = |key: &str| {
            rows.iter()
                .map(|r| r[key].as_u64().unwrap_or(0) as usize)
                .sum::<usize>()
        };
        let candidates = sum("feature_candidate_count");
        let no_gaussian = sum("no_gaussian_count");
        let no_gaussian_quality = sum("no_gaussian_quality_count");
        method_profiles.push(json!({"feature_class":class,"feature_class_label":label,"frame_count":rows.len(),
            "active_frame_count":rows.iter().filter(|r|r["feature_candidate_count"].as_u64().unwrap_or(0)>0).count(),
            "candidate_count_total":candidates,"quality_count_total":sum("feature_quality_count"),
            "all_three_count":sum("all_three_count"),"gaussian_only_count":sum("gaussian_only_count"),"dog_only_count":sum("dog_only_count"),
            "partial_combination_count":sum("partial_combination_count"),"no_gaussian_count":no_gaussian,"no_method_count":sum("no_method_count"),
            "no_gaussian_quality_count":no_gaussian_quality,"all_three_fraction":ratio(sum("all_three_count"),candidates),
            "dog_only_fraction":ratio(sum("dog_only_count"),candidates),"no_gaussian_fraction":ratio(no_gaussian,candidates),
            "no_gaussian_quality_fraction":ratio(no_gaussian_quality,no_gaussian)}));
    }

    Ok(json!({
        "frame_count":audit.frame_count,"association_radius_px":options.association_radius_px,"required_presence":audit.required,
        "association_method":"reciprocal_nearest_one_to_one",
        "candidate_source_mode":if options.max_sources.is_none(){"all returned source measurements".into()}else{format!("returned source measurements capped at max_sources={}",options.max_sources.unwrap())},
        "quality_source_mode":if options.max_sources.is_none(){"all returned quality sources".into()}else{format!("quality sources within max_sources={} working set",options.max_sources.unwrap())},
        "cumulative_shifts":audit.shifts,"frame_rows":audit.frame_rows,"persistence_rows":persistence,
        "class_transition_rows":transitions,"proposal_method_rows":audit.method_rows,
        "source_subgroup_rows":source_subgroups,"diagnostic_subgroup_rows":diagnostic_rows,
        "diagnostic_source_row_count":diagnostic_sources.len(),"diagnostic_source_rows":diagnostic_sources,
        "temporal_profiles":temporal_profiles,"proposal_method_profiles":method_profiles,
        "note":"presence 是首帧坐标在固定小邻域内的互相最近一对一响应次数，不是星表身份；密集候选场中的邻近命中不能替代星表、注入和人工真值。",
        "class_transition_note":"class_transition_rows 只统计注册坐标中互相最近的一对一响应，并以首帧类别与响应帧类别分开计数；它仍不是星表身份匹配，也不代表物理类别转移。",
        "temporal_profile_note":"candidate_fraction_cv 描述类别构成的时间稳定性；quality_fraction_weighted 对 compact_quality 包含按 quality_passed 分类造成的定义性泄漏，不是真阳性率。",
        "source_subgroup_note":"source_subgroup_rows 只对首帧 compact_quality 来源子组做逐源持久性统计；dog_only_deblend 表示进入近邻双 PSF 审计适用域，不等于星表确认，无双 PSF 记录也不表示拟合失败。",
        "diagnostic_subgroup_note":"diagnostic_subgroup_rows 只对首帧非 compact_quality 首要特征做互斥子组审计；candidate/quality_presence 表示位置响应，same_subgroup_presence 还要求响应帧的首要诊断子组相同。它们都是 detector-level 统计，不是星表身份、物理伪影真值或真阳性率。",
        "diagnostic_source_note":"diagnostic_source_rows 已写入 sequence_feature_diagnostic_sources.csv；JSON 只保存 diagnostic_source_row_count，避免把逐源长表重复嵌入 JSON。",
    }))
}

fn csv_cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(value) => value.clone(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::Number(value) => value.to_string(),
        _ => value.to_string(),
    }
}

fn write_csv(path: &Path, fields: &[&str], rows: &[Value]) -> Result<()> {
    let mut file =
        File::create(path).map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    file.write_all(&[0xEF, 0xBB, 0xBF])
        .map_err(|e| e.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    writer.write_record(fields).map_err(|e| e.to_string())?;
    for row in rows {
        writer
            .write_record(fields.iter().map(|field| csv_cell(&row[*field])))
            .map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn pixel(pixels: &mut [u8], width: usize, height: usize, x: i32, y: i32, color: [u8; 3]) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }
    let index = (y as usize * width + x as usize) * 3;
    pixels[index..index + 3].copy_from_slice(&color);
}

fn line(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    from: (i32, i32),
    to: (i32, i32),
    color: [u8; 3],
) {
    let (mut x0, mut y0) = (from.0, from.1);
    let (x1, y1) = (to.0, to.1);
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        pixel(pixels, width, height, x0, y0, color);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let twice = 2 * error;
        if twice >= dy {
            error += dy;
            x0 += sx;
        }
        if twice <= dx {
            error += dx;
            y0 += sy;
        }
    }
}

fn glyph(character: char) -> [u8; 7] {
    match character.to_ascii_uppercase() {
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [14, 17, 16, 16, 16, 17, 14],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [14, 17, 16, 23, 17, 17, 15],
        'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [14, 4, 4, 4, 4, 4, 14],
        'J' => [7, 2, 2, 2, 18, 18, 12],
        'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 25, 25, 21, 19, 19, 17],
        'O' => [14, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'Q' => [14, 17, 17, 17, 21, 18, 13],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'X' => [17, 17, 10, 4, 10, 17, 17],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        'Z' => [31, 1, 2, 4, 8, 16, 31],
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        '.' => [0, 0, 0, 0, 0, 6, 6],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '/' => [1, 2, 2, 4, 8, 8, 16],
        ':' => [0, 6, 6, 0, 6, 6, 0],
        '>' => [16, 8, 4, 2, 4, 8, 16],
        '=' => [0, 31, 0, 31, 0, 0, 0],
        '(' => [2, 4, 8, 8, 8, 4, 2],
        ')' => [8, 4, 2, 2, 2, 4, 8],
        _ => [0; 7],
    }
}

fn draw_text(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    text: &str,
    color: [u8; 3],
) {
    for (character_index, character) in text.chars().enumerate() {
        for (row_index, row) in glyph(character).into_iter().enumerate() {
            for column in 0..5 {
                if row & (1 << (4 - column)) != 0 {
                    pixel(
                        pixels,
                        width,
                        height,
                        x + character_index as i32 * 6 + column,
                        y + row_index as i32,
                        color,
                    );
                }
            }
        }
    }
}

fn write_chart(path: &Path, rows: &[Value], required: usize, frame_count: usize) -> Result<()> {
    const W: usize = 960;
    const H: usize = 520;
    const LEFT: i32 = 72;
    const RIGHT_MARGIN: i32 = 24;
    const TOP: i32 = 78;
    const BOTTOM_MARGIN: i32 = 64;
    let mut pixels = vec![255u8; W * H * 3];
    let left = LEFT;
    let right = W as i32 - RIGHT_MARGIN;
    let top = TOP;
    let bottom = H as i32 - BOTTOM_MARGIN;
    let candidate_color = [215, 148, 50];
    let quality_color = [79, 155, 131];
    draw_text(
        &mut pixels,
        W,
        H,
        left,
        20,
        "SEQUENCE FEATURE PERSISTENCE - ANCHOR FRAME",
        [32, 41, 61],
    );
    draw_text(
        &mut pixels,
        W,
        H,
        left,
        46,
        "CANDIDATE NEIGHBORHOOD",
        candidate_color,
    );
    line(
        &mut pixels,
        W,
        H,
        (left + 216, 49),
        (left + 238, 49),
        candidate_color,
    );
    draw_text(
        &mut pixels,
        W,
        H,
        left + 252,
        46,
        "QUALITY NEIGHBORHOOD",
        quality_color,
    );
    line(
        &mut pixels,
        W,
        H,
        (left + 468, 49),
        (left + 490, 49),
        quality_color,
    );
    draw_text(
        &mut pixels,
        W,
        H,
        left,
        top - 10,
        "FRACTION PRESENT",
        [80, 88, 98],
    );
    draw_text(
        &mut pixels,
        W,
        H,
        left,
        H as i32 - 23,
        "FEATURE CLASS ORDER (SEE CSV)",
        [80, 88, 98],
    );
    draw_text(&mut pixels, W, H, 24, bottom - 3, "0.0", [80, 88, 98]);
    draw_text(&mut pixels, W, H, 24, top - 3, "1.0", [80, 88, 98]);
    for step in 0..=5 {
        let y = bottom - (bottom - top) * step / 5;
        line(&mut pixels, W, H, (left, y), (right, y), [224, 228, 232]);
        if step > 0 && step < 5 {
            draw_text(
                &mut pixels,
                W,
                H,
                24,
                y - 3,
                &format!("0.{}", step * 2),
                [80, 88, 98],
            );
        }
    }
    line(&mut pixels, W, H, (left, top), (left, bottom), [85, 91, 98]);
    line(
        &mut pixels,
        W,
        H,
        (left, bottom),
        (right, bottom),
        [85, 91, 98],
    );
    let mut candidate = Vec::new();
    let mut quality = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let count = row["anchor_candidate_count"].as_u64().unwrap_or(0) as usize;
        let denom = count.max(1) as f64;
        let x = left + (right - left) * index as i32 / (SOURCE_FEATURE_CLASSES.len() as i32 - 1);
        candidate.push((
            x,
            bottom
                - ((bottom - top) as f64
                    * (row["candidate_presence_ge_required_count"]
                        .as_u64()
                        .unwrap_or(0) as f64
                        / denom))
                    .round() as i32,
        ));
        quality.push((
            x,
            bottom
                - ((bottom - top) as f64
                    * (row["quality_presence_ge_required_count"]
                        .as_u64()
                        .unwrap_or(0) as f64
                        / denom))
                    .round() as i32,
        ));
        draw_text(
            &mut pixels,
            W,
            H,
            x - 2,
            bottom + 10,
            &(index + 1).to_string(),
            [80, 88, 98],
        );
    }
    for pair in candidate.windows(2) {
        line(&mut pixels, W, H, pair[0], pair[1], candidate_color);
    }
    for pair in quality.windows(2) {
        line(&mut pixels, W, H, pair[0], pair[1], quality_color);
    }
    for point in &candidate {
        for y in -3..=3 {
            for x in -3..=3 {
                pixel(&mut pixels, W, H, point.0 + x, point.1 + y, candidate_color);
            }
        }
    }
    for point in &quality {
        for y in -3..=3 {
            for x in -3..=3 {
                if x * x + y * y <= 9 {
                    pixel(&mut pixels, W, H, point.0 + x, point.1 + y, quality_color);
                }
            }
        }
    }
    draw_text(
        &mut pixels,
        W,
        H,
        left + 145,
        top - 20,
        &format!("REQUIRED PRESENCE >= {required}/{frame_count}"),
        [80, 88, 98],
    );
    let mut file =
        File::create(path).map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    let mut encoder = png::Encoder::new(&mut file, W as u32, H as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer
        .write_image_data(&pixels)
        .map_err(|e| e.to_string())?;
    drop(writer);
    Ok(())
}

fn write_artifacts(audit: &Audit, mut payload: Value) -> Result<(PathBuf, Value)> {
    let options = audit.options.as_ref().ok_or("missing audit options")?;
    fs::create_dir_all(&options.out_dir).map_err(|e| e.to_string())?;
    let output = options.out_dir.canonicalize().map_err(|e| e.to_string())?;
    let rows = |key: &str| payload[key].as_array().cloned().unwrap_or_default();
    let persistence = rows("persistence_rows");
    let frame = rows("frame_rows");
    let methods = rows("proposal_method_rows");
    let method_profiles = rows("proposal_method_profiles");
    let source_groups = rows("source_subgroup_rows");
    let diagnostic_groups = rows("diagnostic_subgroup_rows");
    let diagnostic_sources = rows("diagnostic_source_rows");
    let temporal = rows("temporal_profiles");
    let transitions = rows("class_transition_rows");
    write_csv(
        &output.join("sequence_feature_persistence.csv"),
        PERSISTENCE_FIELDS,
        &persistence,
    )?;
    write_csv(
        &output.join("sequence_feature_frame_summary.csv"),
        FRAME_FIELDS,
        &frame,
    )?;
    write_csv(
        &output.join("sequence_feature_method_frame_summary.csv"),
        METHOD_FRAME_FIELDS,
        &methods,
    )?;
    write_csv(
        &output.join("sequence_feature_method_profile.csv"),
        METHOD_PROFILE_FIELDS,
        &method_profiles,
    )?;
    write_csv(
        &output.join("sequence_feature_source_subgroup_persistence.csv"),
        SOURCE_SUBGROUP_FIELDS,
        &source_groups,
    )?;
    write_csv(
        &output.join("sequence_feature_diagnostic_subgroup_persistence.csv"),
        DIAGNOSTIC_SUBGROUP_FIELDS,
        &diagnostic_groups,
    )?;
    write_csv(
        &output.join("sequence_feature_diagnostic_sources.csv"),
        DIAGNOSTIC_SOURCE_FIELDS,
        &diagnostic_sources,
    )?;
    write_csv(
        &output.join("sequence_feature_temporal_profile.csv"),
        TEMPORAL_FIELDS,
        &temporal,
    )?;
    write_csv(
        &output.join("sequence_feature_class_transition.csv"),
        TRANSITION_FIELDS,
        &transitions,
    )?;
    payload
        .as_object_mut()
        .ok_or("sequence feature payload must be an object")?
        .remove("diagnostic_source_rows");
    payload
        .as_object_mut()
        .unwrap()
        .remove("proposal_method_profiles");
    payload["note"]=json!("presence 是首帧坐标在固定小邻域内的互相最近一对一响应次数，不是星表身份；密集候选场中的邻近命中不能替代星表、注入和人工真值。");
    fs::write(
        output.join("sequence_feature_persistence.json"),
        serde_json::to_vec_pretty(&payload).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    write_chart(
        &output.join("sequence_feature_persistence.png"),
        &persistence,
        audit.required,
        audit.frame_count,
    )?;
    Ok((output, payload))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let options = parse_options(args)?;
    let audit = analyze(options)?;
    let payload = build_rows(&audit)?;
    let (_output, payload) = write_artifacts(&audit, payload)?;
    Ok(payload)
}

pub fn run_cli(args: &[String]) -> i32 {
    let result = (|| -> Result<(PathBuf, Audit, Value)> {
        let options = parse_options(args)?;
        let audit = analyze(options)?;
        let payload = build_rows(&audit)?;
        let (output, payload) = write_artifacts(&audit, payload)?;
        Ok((output, audit, payload))
    })();
    match result {
        Ok((output, audit, payload)) => {
            println!("rst19-feature-sequence: 审计已写入 {}", output.display());
            println!(
                "frames={} radius={} required_presence={}",
                audit.frame_count,
                audit
                    .options
                    .as_ref()
                    .map(|v| v.association_radius_px)
                    .unwrap_or_default(),
                audit.required
            );
            for row in payload["persistence_rows"].as_array().into_iter().flatten() {
                println!(
                    "{}: anchor={} candidate>={}={} quality>={}={}",
                    row["feature_class"].as_str().unwrap_or_default(),
                    row["anchor_candidate_count"],
                    audit.required,
                    row["candidate_presence_ge_required_count"],
                    audit.required,
                    row["quality_presence_ge_required_count"]
                );
            }
            0
        }
        Err(error) => {
            eprintln!("rst19-feature-sequence: {error}");
            if error == HELP {
                0
            } else {
                2
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn association_radius_is_strictly_exclusive_at_the_boundary() {
        let pairs = reciprocal_pairs(&[[0.0, 0.0]], &[[1.0, 0.0]], 1.0).unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn diagnostic_priority_matches_primary_class_flags() {
        let mut source = Detection {
            detection_id: 0,
            x: 1.0,
            y: 1.0,
            peak: 10.0,
            flux: 30.0,
            background: 1.0,
            noise: 1.0,
            snr: 10.0,
            fwhm: None,
            flags: vec!["SPIKE".into(), "INSUFFICIENT_PSF_SUPPORT".into()],
            flux_error: None,
            flux_snr: None,
            filter_snr: None,
            fwhm_x: None,
            fwhm_y: None,
            ellipticity: None,
            sharpness: None,
            footprint_pixels: None,
            psf_support_pixels: None,
            quality_passed: false,
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
        assert_eq!(
            diagnostic_subgroup(
                &source,
                source_classification::classify_source_feature(&source)
            ),
            Some("spike_and_psf_support")
        );
        source.flags.push("SATURATED".into());
        assert_eq!(
            diagnostic_subgroup(
                &source,
                source_classification::classify_source_feature(&source)
            ),
            Some("range_saturated")
        );
    }

    #[test]
    fn single_frame_fits_generates_all_audit_artifacts() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let data_dir = workspace.join("backend/tests/fixtures/empirical_pair_audit_parity/frames");
        let root = std::env::temp_dir().join(format!(
            "rst19-sequence-feature-test-{}",
            std::process::id()
        ));
        let options = Options {
            data_dir,
            out_dir: root.join("out"),
            ..Options::default()
        };
        let audit = analyze(options).unwrap();
        let payload = build_rows(&audit).unwrap();
        assert_eq!(payload["frame_count"], 1);
        assert_eq!(
            payload["persistence_rows"].as_array().unwrap().len(),
            SOURCE_FEATURE_CLASSES.len()
        );
        assert_eq!(
            payload["class_transition_rows"].as_array().unwrap().len(),
            2 * SOURCE_FEATURE_CLASSES.len().pow(2)
        );
        let (output, payload) = write_artifacts(&audit, payload).unwrap();
        assert_eq!(
            payload["diagnostic_source_row_count"].as_u64().unwrap() as usize,
            audit.frames[0]
                .candidates
                .iter()
                .filter(|s| source_classification::classify_source_feature(s) != "compact_quality")
                .count()
        );
        for name in [
            "sequence_feature_persistence.csv",
            "sequence_feature_frame_summary.csv",
            "sequence_feature_method_frame_summary.csv",
            "sequence_feature_method_profile.csv",
            "sequence_feature_source_subgroup_persistence.csv",
            "sequence_feature_diagnostic_subgroup_persistence.csv",
            "sequence_feature_diagnostic_sources.csv",
            "sequence_feature_temporal_profile.csv",
            "sequence_feature_class_transition.csv",
            "sequence_feature_persistence.json",
            "sequence_feature_persistence.png",
        ] {
            assert!(output.join(name).is_file(), "missing {name}");
        }
        let bytes = fs::read(output.join("sequence_feature_persistence.csv")).unwrap();
        assert!(bytes.starts_with(&[0xEF, 0xBB, 0xBF]));
        assert!(bytes.windows(2).any(|value| value == b"\r\n"));
        fs::remove_dir_all(root).unwrap();
    }
}
