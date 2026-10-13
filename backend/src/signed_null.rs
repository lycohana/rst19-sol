//! Sign-flipped source-level negative control on a real FITS image.
use crate::{
    detection::{self, DetectionOptions},
    fits,
    measurement::{Detection, MeasurementOptions},
    models::{FitsFrame, PixelData},
    statistics, Result,
};
use csv::Terminator;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

pub const SNR_THRESHOLDS: [f64; 6] = [4.0, 5.0, 7.0, 10.0, 15.0, 20.0];

const FEATURE_CLASSES: [(&str, &str); 9] = [
    ("range_anomaly", "数据有效性/范围异常邻域"),
    ("linear_artifact", "线状/拖影候选"),
    ("masked_or_edge", "边缘/掩膜候选"),
    ("crowded_blend", "拥挤/未分辨近邻"),
    ("spike_or_support", "尖峰/PSF 支持不足"),
    ("weak_or_background", "弱通量/背景不确定"),
    ("shape_outlier", "形状异常"),
    ("compact_quality", "质量通过的紧凑候选"),
    ("other_rejected", "其他拒绝候选"),
];

#[derive(Debug, Clone)]
pub struct Options {
    pub threshold_sigma: f64,
    pub min_distance: usize,
    pub aperture_radius: usize,
    pub psf_fwhm: f64,
    pub background_box_size: usize,
    pub min_flux_snr: f64,
    pub min_psf_support_pixels: usize,
    pub proposal_mode: String,
    pub reject_linear_artifacts: bool,
    pub snr_thresholds: Vec<f64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            threshold_sigma: 4.0,
            min_distance: 4,
            aperture_radius: 4,
            psf_fwhm: 2.0,
            background_box_size: 128,
            min_flux_snr: 5.0,
            min_psf_support_pixels: 3,
            proposal_mode: "hybrid".into(),
            reject_linear_artifacts: true,
            snr_thresholds: SNR_THRESHOLDS.to_vec(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ThresholdRow {
    threshold: f64,
    positive_filter_count: usize,
    negative_filter_count: usize,
    positive_flux_count: usize,
    negative_flux_count: usize,
    filter_leakage_ratio: Option<f64>,
    flux_leakage_ratio: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct FeatureRow {
    control: String,
    feature_class: String,
    feature_class_label: String,
    candidate_count: usize,
    quality_count: usize,
    quality_fraction: Option<f64>,
    median_flux_snr: Option<f64>,
    max_flux_snr: Option<f64>,
    common_flags: String,
}

#[derive(Debug, Clone, Serialize)]
struct SourceRow {
    detection_id: i64,
    x: f64,
    y: f64,
    peak_x: f64,
    peak_y: f64,
    peak: f64,
    flux_snr: Option<f64>,
    filter_snr: Option<f64>,
    fwhm: Option<f64>,
    psf_support_pixels: Option<usize>,
    raw_peak_adu: f64,
    raw_negative_pixel_count: usize,
    raw_negative_anomaly_pixel_count: usize,
    raw_extreme_negative_pixel_count: usize,
    raw_minus_one_count: usize,
    raw_min_adu: Option<f64>,
    raw_max_adu: Option<f64>,
    range_transfer_overlap: bool,
    raw_evidence_layer: String,
    flags: String,
}

#[derive(Debug, Clone)]
struct AuditResult {
    frame_path: String,
    image_shape: [usize; 2],
    background_adu: f64,
    positive_noise_adu: f64,
    negative_noise_adu: f64,
    positive_candidate_count: usize,
    positive_quality_count: usize,
    negative_candidate_count: usize,
    negative_quality_count: usize,
    negative_quality_count_after_range_exclusion: usize,
    candidate_leakage_ratio: Option<f64>,
    quality_leakage_ratio: Option<f64>,
    quality_leakage_ratio_after_range_exclusion: Option<f64>,
    negative_range_transfer_candidate_count: usize,
    negative_range_transfer_quality_count: usize,
    thresholds: Vec<ThresholdRow>,
    feature_rows: Vec<FeatureRow>,
    negative_quality_sources: Vec<SourceRow>,
    parameters: Value,
    conclusion: String,
}

impl AuditResult {
    fn as_dict(&self) -> Value {
        json!({
            "frame_path": self.frame_path,
            "image_shape": self.image_shape,
            "background_adu": self.background_adu,
            "positive_noise_adu": self.positive_noise_adu,
            "negative_noise_adu": self.negative_noise_adu,
            "positive_candidate_count": self.positive_candidate_count,
            "positive_quality_count": self.positive_quality_count,
            "negative_candidate_count": self.negative_candidate_count,
            "negative_quality_count": self.negative_quality_count,
            "negative_quality_count_after_range_exclusion": self.negative_quality_count_after_range_exclusion,
            "candidate_leakage_ratio": self.candidate_leakage_ratio,
            "quality_leakage_ratio": self.quality_leakage_ratio,
            "quality_leakage_ratio_after_range_exclusion": self.quality_leakage_ratio_after_range_exclusion,
            "negative_range_transfer_candidate_count": self.negative_range_transfer_candidate_count,
            "negative_range_transfer_quality_count": self.negative_range_transfer_quality_count,
            "thresholds": self.thresholds,
            "feature_rows": self.feature_rows,
            "negative_quality_sources": self.negative_quality_sources,
            "parameters": self.parameters,
            "conclusion": self.conclusion,
        })
    }
}

#[derive(Debug)]
struct RangeTransfer {
    expanded: Vec<bool>,
    limit: Option<f64>,
    raw_pixel_count: usize,
    expanded_pixel_count: usize,
}

fn validate_thresholds(values: &[f64]) -> Result<()> {
    if values.is_empty() {
        return Err("snr_thresholds must not be empty".into());
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("snr_thresholds must contain finite positive values".into());
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("snr_thresholds must be strictly increasing".into());
    }
    Ok(())
}

fn round_ties_even(value: f64) -> Option<isize> {
    if !value.is_finite() || value < isize::MIN as f64 || value > isize::MAX as f64 {
        return None;
    }
    let lower = value.floor();
    let fraction = value - lower;
    let rounded = if fraction < 0.5 {
        lower
    } else if fraction > 0.5 || (lower as i128) % 2 != 0 {
        lower + 1.0
    } else {
        lower
    };
    Some(rounded as isize)
}

fn classify(source: &Detection) -> &'static str {
    const PRIORITY: [(&str, &[&str]); 7] = [
        (
            "range_anomaly",
            &["SATURATED", "NEGATIVE_OVERFLOW", "CODE_PATTERN"],
        ),
        ("linear_artifact", &["LINE_ARTIFACT"]),
        ("masked_or_edge", &["MASKED", "PARTIAL_MASKED", "EDGE"]),
        ("crowded_blend", &["UNRESOLVED_BLEND"]),
        (
            "spike_or_support",
            &[
                "SPIKE",
                "NARROW",
                "SMALL_FOOTPRINT",
                "INSUFFICIENT_PSF_SUPPORT",
            ],
        ),
        (
            "weak_or_background",
            &["LOW_FLUX_SNR", "NON_POSITIVE_FLUX", "BACKGROUND_UNCERTAIN"],
        ),
        (
            "shape_outlier",
            &["NO_SHAPE", "BROAD", "ELONGATED", "DIFFUSE"],
        ),
    ];
    for (feature_class, flags) in PRIORITY {
        if source
            .flags
            .iter()
            .any(|flag| flags.contains(&flag.as_str()))
        {
            return feature_class;
        }
    }
    if source.quality_passed {
        "compact_quality"
    } else {
        "other_rejected"
    }
}

fn feature_rows(control: &str, sources: &[Detection]) -> Result<Vec<FeatureRow>> {
    if control != "positive" && control != "sign_flipped" {
        return Err(format!("unsupported control: {control:?}"));
    }
    let mut output = Vec::with_capacity(FEATURE_CLASSES.len());
    for (class, label) in FEATURE_CLASSES {
        let members = sources
            .iter()
            .filter(|source| classify(source) == class)
            .collect::<Vec<_>>();
        let quality_count = members
            .iter()
            .filter(|source| source.quality_passed)
            .count();
        let snr_values = members
            .iter()
            .map(|source| source.flux_snr.unwrap_or(source.snr))
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let (median_snr, max_snr) = if snr_values.is_empty() {
            (None, None)
        } else {
            let maximum = snr_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (Some(statistics::median(&snr_values)?), Some(maximum))
        };
        let mut flags = BTreeMap::<String, usize>::new();
        for source in &members {
            for flag in &source.flags {
                *flags.entry(flag.clone()).or_default() += 1;
            }
        }
        let mut flags = flags.into_iter().collect::<Vec<_>>();
        flags.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
        let common_flags = flags
            .into_iter()
            .take(4)
            .map(|(flag, _)| flag)
            .collect::<Vec<_>>()
            .join("|");
        output.push(FeatureRow {
            control: control.to_owned(),
            feature_class: class.to_owned(),
            feature_class_label: label.to_owned(),
            candidate_count: members.len(),
            quality_count,
            quality_fraction: (!members.is_empty())
                .then_some(quality_count as f64 / members.len() as f64),
            median_flux_snr: median_snr,
            max_flux_snr: max_snr,
            common_flags,
        });
    }
    Ok(output)
}

fn range_transfer(frame: &FitsFrame, radius: usize) -> RangeTransfer {
    let (signed, maximum) = match &frame.data {
        PixelData::I16(_) => (true, i16::MAX as f64),
        PixelData::I32(_) => (true, i32::MAX as f64),
        PixelData::I64(_) => (true, i64::MAX as f64),
        _ => (false, 0.0),
    };
    let mut expanded = vec![false; frame.data.len()];
    if !signed {
        return RangeTransfer {
            expanded,
            limit: None,
            raw_pixel_count: 0,
            expanded_pixel_count: 0,
        };
    }
    let limit = -0.9 * maximum;
    let raw = (0..frame.data.len())
        .filter(|index| frame.data.value(*index) <= limit)
        .collect::<Vec<_>>();
    let radius_i = radius as isize;
    let radius_sq = (radius_i * radius_i) as i64;
    for index in &raw {
        let x = (*index % frame.width) as isize;
        let y = (*index / frame.width) as isize;
        for dy in -radius_i..=radius_i {
            for dx in -radius_i..=radius_i {
                if (dx * dx + dy * dy) as i64 > radius_sq {
                    continue;
                }
                let px = x + dx;
                let py = y + dy;
                if px >= 0 && py >= 0 && px < frame.width as isize && py < frame.height as isize {
                    expanded[py as usize * frame.width + px as usize] = true;
                }
            }
        }
    }
    let expanded_pixel_count = expanded.iter().filter(|value| **value).count();
    RangeTransfer {
        expanded,
        limit: Some(limit),
        raw_pixel_count: raw.len(),
        expanded_pixel_count,
    }
}

fn hits_range_transfer(
    source: &Detection,
    transfer: &RangeTransfer,
    width: usize,
    height: usize,
) -> bool {
    let Some(x) = round_ties_even(source.peak_x.unwrap_or(source.x)) else {
        return false;
    };
    let Some(y) = round_ties_even(source.peak_y.unwrap_or(source.y)) else {
        return false;
    };
    x >= 0
        && y >= 0
        && x < width as isize
        && y < height as isize
        && transfer.expanded[y as usize * width + x as usize]
}

fn raw_evidence_layer(
    range_transfer_overlap: bool,
    raw_negative_pixel_count: usize,
    raw_negative_anomaly_pixel_count: usize,
    raw_minus_one_count: usize,
) -> &'static str {
    if range_transfer_overlap {
        "extreme_range_transfer"
    } else if raw_negative_anomaly_pixel_count > 0 {
        "raw_negative_anomaly"
    } else if raw_minus_one_count > 0 {
        "fixed_sentinel_candidate"
    } else if raw_negative_pixel_count > 0 {
        "raw_negative_other"
    } else {
        "no_negative_evidence"
    }
}

fn raw_source_row(
    source: &Detection,
    frame: &FitsFrame,
    mask: &[bool],
    transfer: &RangeTransfer,
    background: f64,
    noise: f64,
    aperture_radius: usize,
) -> Result<SourceRow> {
    let peak_x = round_ties_even(source.peak_x.unwrap_or(source.x))
        .ok_or("signed-null source x coordinate is invalid")?;
    let peak_y = round_ties_even(source.peak_y.unwrap_or(source.y))
        .ok_or("signed-null source y coordinate is invalid")?;
    let width = frame.width as isize;
    let height = frame.height as isize;
    let radius = aperture_radius as isize;
    let negative_anomaly_limit = (-1000.0f64).min(background - 50.0 * noise.max(1e-12));
    let mut aperture = Vec::new();
    for y in (peak_y - radius).max(0)..=(peak_y + radius).min(height - 1) {
        for x in (peak_x - radius).max(0)..=(peak_x + radius).min(width - 1) {
            let dx = x - peak_x;
            let dy = y - peak_y;
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            let index = y as usize * frame.width + x as usize;
            let value = frame.data.value(index);
            if !mask[index] && value.is_finite() {
                aperture.push(value);
            }
        }
    }
    if aperture.is_empty() {
        return Err("signed-null quality source has no valid aperture pixels".into());
    }
    let raw_peak_adu = if peak_x >= 0 && peak_y >= 0 && peak_x < width && peak_y < height {
        frame
            .data
            .value(peak_y as usize * frame.width + peak_x as usize)
    } else {
        f64::NAN
    };
    let raw_negative_pixel_count = aperture.iter().filter(|value| **value < 0.0).count();
    let raw_negative_anomaly_pixel_count = aperture
        .iter()
        .filter(|value| **value <= negative_anomaly_limit)
        .count();
    let raw_extreme_negative_pixel_count = transfer
        .limit
        .map(|limit| aperture.iter().filter(|value| **value <= limit).count())
        .unwrap_or(0);
    let raw_minus_one_count = aperture.iter().filter(|value| **value == -1.0).count();
    let range_transfer_overlap = hits_range_transfer(source, transfer, frame.width, frame.height);
    Ok(SourceRow {
        detection_id: source.detection_id,
        x: source.x,
        y: source.y,
        peak_x: peak_x as f64,
        peak_y: peak_y as f64,
        peak: source.peak,
        flux_snr: source.flux_snr,
        filter_snr: source.filter_snr,
        fwhm: source.fwhm,
        psf_support_pixels: source.psf_support_pixels,
        raw_peak_adu,
        raw_negative_pixel_count,
        raw_negative_anomaly_pixel_count,
        raw_extreme_negative_pixel_count,
        raw_minus_one_count,
        raw_min_adu: aperture.iter().copied().reduce(f64::min),
        raw_max_adu: aperture.iter().copied().reduce(f64::max),
        range_transfer_overlap,
        raw_evidence_layer: raw_evidence_layer(
            range_transfer_overlap,
            raw_negative_pixel_count,
            raw_negative_anomaly_pixel_count,
            raw_minus_one_count,
        )
        .into(),
        flags: source.flags.join("|"),
    })
}

fn finite_snr(sources: &[Detection], filter: bool) -> Vec<f64> {
    sources
        .iter()
        .filter_map(|source| {
            if filter {
                source.filter_snr
            } else {
                source.flux_snr
            }
        })
        .filter(|value| value.is_finite())
        .collect()
}

fn count_at_least(values: &[f64], threshold: f64) -> usize {
    values.iter().filter(|value| **value >= threshold).count()
}

fn safe_ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn detector_options(options: &Options) -> DetectionOptions {
    DetectionOptions {
        measurement: MeasurementOptions {
            aperture_radius: options.aperture_radius,
            min_flux_snr: options.min_flux_snr,
            min_psf_support_pixels: options.min_psf_support_pixels,
            psf_fwhm: options.psf_fwhm,
            ..MeasurementOptions::default()
        },
        threshold_sigma: options.threshold_sigma,
        min_distance: options.min_distance,
        background_box_size: options.background_box_size,
        reject_linear_artifacts: options.reject_linear_artifacts,
        proposal_mode: options.proposal_mode.clone(),
        ..DetectionOptions::default()
    }
}

fn run(path: &Path, options: &Options) -> Result<AuditResult> {
    let frame = fits::read_fits(path, false)?;
    run_frame(frame, options)
}

fn run_frame(frame: FitsFrame, options: &Options) -> Result<AuditResult> {
    if options.threshold_sigma <= 0.0 {
        return Err("threshold_sigma must be positive".into());
    }
    if options.min_distance < 1 || options.aperture_radius < 1 {
        return Err("min_distance and aperture_radius must be positive".into());
    }
    if options.psf_fwhm <= 0.0 || options.background_box_size < 16 || options.min_flux_snr <= 0.0 {
        return Err(
            "psf_fwhm and min_flux_snr must be positive; background_box_size must be at least 16"
                .into(),
        );
    }
    if !(1..=9).contains(&options.min_psf_support_pixels) {
        return Err("min_psf_support_pixels must be between 1 and 9".into());
    }
    validate_thresholds(&options.snr_thresholds)?;
    if !["gaussian", "hybrid", "ensemble"].contains(&options.proposal_mode.as_str()) {
        return Err("proposal_mode must be gaussian, hybrid, or ensemble".into());
    }
    let width = frame.width;
    let height = frame.height;
    let mask = fits::auxiliary_mask(height, width)?;
    let numeric = frame.data.to_f64();
    let background = statistics::sigma_clipped_stats(&numeric, Some(&mask), 3.0, 4, 1_000_000)?.0;
    let transfer = range_transfer(&frame, options.aperture_radius);
    let detector = detector_options(options);
    let positive =
        detection::detect_sources(&frame.data, Some(&mask), height, width, &detector, None)?;
    let mirrored = PixelData::F64(
        numeric
            .iter()
            .map(|value| 2.0 * background - value)
            .collect(),
    );
    let negative =
        detection::detect_sources(&mirrored, Some(&mask), height, width, &detector, None)?;
    let positive_filter = finite_snr(&positive.sources, true);
    let negative_filter = finite_snr(&negative.sources, true);
    let positive_flux = finite_snr(&positive.sources, false);
    let negative_flux = finite_snr(&negative.sources, false);
    let thresholds = options
        .snr_thresholds
        .iter()
        .copied()
        .map(|threshold| {
            let positive_filter_count = count_at_least(&positive_filter, threshold);
            let negative_filter_count = count_at_least(&negative_filter, threshold);
            let positive_flux_count = count_at_least(&positive_flux, threshold);
            let negative_flux_count = count_at_least(&negative_flux, threshold);
            ThresholdRow {
                threshold,
                positive_filter_count,
                negative_filter_count,
                positive_flux_count,
                negative_flux_count,
                filter_leakage_ratio: safe_ratio(negative_filter_count, positive_filter_count),
                flux_leakage_ratio: safe_ratio(negative_flux_count, positive_flux_count),
            }
        })
        .collect::<Vec<_>>();
    let range_transfer_sources = negative
        .sources
        .iter()
        .filter(|source| hits_range_transfer(source, &transfer, width, height))
        .collect::<Vec<_>>();
    let range_transfer_quality_count = range_transfer_sources
        .iter()
        .filter(|source| source.quality_passed)
        .count();
    let effective_quality_count = negative
        .quality_count
        .saturating_sub(range_transfer_quality_count);
    let negative_quality_sources = negative
        .sources
        .iter()
        .filter(|source| source.quality_passed)
        .map(|source| {
            raw_source_row(
                source,
                &frame,
                &mask,
                &transfer,
                background,
                positive.noise,
                options.aperture_radius,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut feature_summary = feature_rows("positive", &positive.sources)?;
    feature_summary.extend(feature_rows("sign_flipped", &negative.sources)?);
    let detector_args = json!({
        "mask":{"type":"ndarray","shape":[height,width],"dtype":"bool"},
        "threshold_sigma":options.threshold_sigma,
        "min_distance":options.min_distance,
        "aperture_radius":options.aperture_radius,
        "psf_fwhm":options.psf_fwhm,
        "background_box_size":options.background_box_size,
        "min_flux_snr":options.min_flux_snr,
        "min_psf_support_pixels":options.min_psf_support_pixels,
        "proposal_mode":options.proposal_mode,
        "reject_linear_artifacts":options.reject_linear_artifacts
    });
    let parameters = json!({
        "thresholds":options.snr_thresholds,
        "detector":detector_args,
        "positive_detector_parameters":positive.parameters,
        "negative_detector_parameters":negative.parameters,
        "background_estimator":"sigma_clipped_stats(mask=auxiliary_mask, sample_limit=1000000)",
        "mirror_definition":"I_mirror = 2 * B_global - I",
        "mask_policy":"FITS first-row auxiliary bytes are masked; detector additionally masks non-finite values",
        "source_count_definition":"candidate_count is post-NMS proposal count; quality_count is source-level quality pass",
        "diagnostic_name":"signed-tail leakage",
        "not_formal_fdr":true,
        "negative_range_transfer":{
            "enabled":transfer.limit.is_some(),
            "original_negative_overflow_limit_adu":transfer.limit,
            "original_extreme_negative_pixel_count":transfer.raw_pixel_count,
            "expanded_mask_radius_px":options.aperture_radius,
            "expanded_mask_pixel_count":transfer.expanded_pixel_count,
            "candidate_overlap_count":range_transfer_sources.len(),
            "quality_overlap_count":range_transfer_quality_count,
            "effective_quality_count_after_exclusion":effective_quality_count,
            "label":"original signed-integer negative-range response transferred to positive by mirroring"
        },
        "limitations":[
            "符号反相只约束近似对称的背景/噪声尾部，不是 FDR、p 值、precision 或真实源概率。",
            "正向热像素、宇宙线、亮线、饱和和非对称值域结构不会被该对照完整模拟。",
            "反相图使用 float64，因此原始有符号整型的负侧满量程编码审计不等价于真实帧。"
        ]
    });
    let conclusion = "符号反相只约束可近似镜像的背景/噪声尾部；正负响应比是泄漏诊断，不是 FDR、p 值或真实源概率。正向热像素、宇宙线、亮线和非对称值域结构仍需通过类别、PSF、跨帧和星表/注入证据独立确认。";
    Ok(AuditResult {
        frame_path: frame.path.to_string_lossy().into_owned(),
        image_shape: [height, width],
        background_adu: background,
        positive_noise_adu: positive.noise,
        negative_noise_adu: negative.noise,
        positive_candidate_count: positive.candidate_count,
        positive_quality_count: positive.quality_count,
        negative_candidate_count: negative.candidate_count,
        negative_quality_count: negative.quality_count,
        negative_quality_count_after_range_exclusion: effective_quality_count,
        candidate_leakage_ratio: safe_ratio(negative.candidate_count, positive.candidate_count),
        quality_leakage_ratio: safe_ratio(negative.quality_count, positive.quality_count),
        quality_leakage_ratio_after_range_exclusion: safe_ratio(
            effective_quality_count,
            positive.quality_count,
        ),
        negative_range_transfer_candidate_count: range_transfer_sources.len(),
        negative_range_transfer_quality_count: range_transfer_quality_count,
        thresholds,
        feature_rows: feature_summary,
        negative_quality_sources,
        parameters,
        conclusion: conclusion.into(),
    })
}

pub(crate) fn run_frame_value(frame: FitsFrame, options: &Options) -> Result<Value> {
    Ok(run_frame(frame, options)?.as_dict())
}

fn write_rows<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut file = file;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    if rows.is_empty() {
        file.write_all(b"\r\n").map_err(|error| error.to_string())?;
    } else {
        let mut writer = csv::WriterBuilder::new()
            .terminator(Terminator::CRLF)
            .from_writer(file);
        for row in rows {
            writer.serialize(row).map_err(|error| error.to_string())?;
        }
        writer.flush().map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn write_artifacts(result: &AuditResult, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let output = out_dir.canonicalize().map_err(|error| error.to_string())?;
    let summary = output.join("signed_null_summary.json");
    let file = File::create(summary).map_err(|error| error.to_string())?;
    serde_json::to_writer_pretty(file, &result.as_dict()).map_err(|error| error.to_string())?;
    write_rows(
        &output.join("signed_null_thresholds.csv"),
        &result.thresholds,
    )?;
    write_rows(
        &output.join("signed_null_feature_counts.csv"),
        &result.feature_rows,
    )?;
    Ok(output)
}

#[derive(Debug)]
struct CliArgs {
    fits_path: PathBuf,
    out_dir: PathBuf,
    options: Options,
}

fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn option_value(args: &[String], index: &mut usize, arg: &str, flag: &str) -> Result<String> {
    if let Some(value) = arg.strip_prefix(&format!("{flag}=")) {
        Ok(value.to_owned())
    } else {
        take_value(args, index, flag)
    }
}

fn parse_args(args: &[String]) -> Result<CliArgs> {
    let mut fits_path = None;
    let mut out_dir = PathBuf::from("tmp/signed-null-audit");
    let mut options = Options::default();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--keep-linear-artifacts" {
            options.reject_linear_artifacts = false;
        } else if arg == "--out-dir" || arg.starts_with("--out-dir=") {
            out_dir = PathBuf::from(if let Some(value) = arg.strip_prefix("--out-dir=") {
                value.to_owned()
            } else {
                take_value(args, &mut index, arg)?
            });
        } else if arg == "--snr-thresholds" {
            options.snr_thresholds.clear();
            index += 1;
            while index < args.len() && !args[index].starts_with('-') {
                options.snr_thresholds.push(
                    args[index]
                        .parse()
                        .map_err(|_| "invalid --snr-thresholds value")?,
                );
                index += 1;
            }
            index -= 1;
        } else if let Some(value) = arg.strip_prefix("--snr-thresholds=") {
            options.snr_thresholds = value
                .split(',')
                .map(|part| {
                    part.parse()
                        .map_err(|_| "invalid --snr-thresholds value".to_owned())
                })
                .collect::<Result<Vec<f64>>>()?;
        } else if arg == "--threshold-sigma" || arg.starts_with("--threshold-sigma=") {
            let value = option_value(args, &mut index, arg, "--threshold-sigma")?;
            options.threshold_sigma = value.parse().map_err(|_| "invalid --threshold-sigma")?;
        } else if arg == "--min-distance" || arg.starts_with("--min-distance=") {
            let value = option_value(args, &mut index, arg, "--min-distance")?;
            options.min_distance = value.parse().map_err(|_| "invalid --min-distance")?;
        } else if arg == "--aperture-radius" || arg.starts_with("--aperture-radius=") {
            let value = option_value(args, &mut index, arg, "--aperture-radius")?;
            options.aperture_radius = value.parse().map_err(|_| "invalid --aperture-radius")?;
        } else if arg == "--psf-fwhm" || arg.starts_with("--psf-fwhm=") {
            let value = option_value(args, &mut index, arg, "--psf-fwhm")?;
            options.psf_fwhm = value.parse().map_err(|_| "invalid --psf-fwhm")?;
        } else if arg == "--background-box-size" || arg.starts_with("--background-box-size=") {
            let value = option_value(args, &mut index, arg, "--background-box-size")?;
            options.background_box_size =
                value.parse().map_err(|_| "invalid --background-box-size")?;
        } else if arg == "--min-flux-snr" || arg.starts_with("--min-flux-snr=") {
            let value = option_value(args, &mut index, arg, "--min-flux-snr")?;
            options.min_flux_snr = value.parse().map_err(|_| "invalid --min-flux-snr")?;
        } else if arg == "--min-psf-support-pixels" || arg.starts_with("--min-psf-support-pixels=")
        {
            let value = option_value(args, &mut index, arg, "--min-psf-support-pixels")?;
            options.min_psf_support_pixels = value
                .parse()
                .map_err(|_| "invalid --min-psf-support-pixels")?;
        } else if arg == "--proposal-mode" || arg.starts_with("--proposal-mode=") {
            options.proposal_mode = option_value(args, &mut index, arg, "--proposal-mode")?;
        } else if arg == "--" {
            index += 1;
            for input in &args[index..] {
                if fits_path.replace(PathBuf::from(input)).is_some() {
                    return Err("only one FITS input path is allowed".into());
                }
            }
            break;
        } else if arg.starts_with('-') {
            return Err(format!("unrecognized argument: {arg}"));
        } else if fits_path.replace(PathBuf::from(arg)).is_some() {
            return Err("only one FITS input path is allowed".into());
        }
        index += 1;
    }
    Ok(CliArgs {
        fits_path: fits_path.ok_or("missing FITS file")?,
        out_dir,
        options,
    })
}

fn execute(args: &[String]) -> Result<(AuditResult, PathBuf)> {
    let parsed = parse_args(args)?;
    let result = run(&parsed.fits_path, &parsed.options)?;
    let output = write_artifacts(&result, &parsed.out_dir)?;
    Ok((result, output))
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("用真实 FITS 的背景符号反相图做源级 signed-tail leakage 对照，不输出源目录\n\nUsage: rst19-signed-null-audit FITS [--threshold-sigma N] [--min-distance N] [--aperture-radius N] [--psf-fwhm N] [--proposal-mode gaussian|hybrid|ensemble] [--background-box-size N] [--min-flux-snr N] [--min-psf-support-pixels N] [--keep-linear-artifacts] [--snr-thresholds N...] [--out-dir DIR]");
        return Ok(());
    }
    let (result, output) = execute(args)?;
    println!("signed-null audit: {}", output.display());
    println!(
        "positive candidate={} quality={}; sign_flipped candidate={} quality={}",
        result.positive_candidate_count,
        result.positive_quality_count,
        result.negative_candidate_count,
        result.negative_quality_count
    );
    println!(
        "candidate_leakage={:?} quality_leakage={:?}",
        result.candidate_leakage_ratio, result.quality_leakage_ratio
    );
    Ok(())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let (result, output) = execute(args)?;
    Ok(json!({
        "tool":"rst19-signed-null-audit",
        "out_dir":output,
        "result_path":output.join("signed_null_summary.json"),
        "result":result.as_dict()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_thresholds_and_safe_ratio() {
        assert!(validate_thresholds(&[]).is_err());
        assert!(validate_thresholds(&[4.0, 4.0]).is_err());
        assert!(validate_thresholds(&[4.0, 5.0]).is_ok());
        assert_eq!(safe_ratio(1, 0), None);
        assert_eq!(safe_ratio(1, 2), Some(0.5));
    }

    #[test]
    fn source_feature_priority_matches_shared_audit_classes() {
        let source = Detection {
            detection_id: 1,
            x: 0.0,
            y: 0.0,
            peak: 1.0,
            flux: 1.0,
            background: 0.0,
            noise: 1.0,
            snr: 1.0,
            fwhm: None,
            flags: vec!["ELONGATED".into(), "SPIKE".into(), "LINE_ARTIFACT".into()],
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
        assert_eq!(classify(&source), "linear_artifact");
    }

    #[test]
    fn python_rounding_ties_to_even_for_mask_anchors() {
        assert_eq!(round_ties_even(2.5), Some(2));
        assert_eq!(round_ties_even(3.5), Some(4));
        assert_eq!(round_ties_even(-0.5), Some(0));
    }
}
