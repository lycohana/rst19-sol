//! Sensitivity of detector candidates to threshold and flux-SNR settings.
//!
//! Candidate persistence, quality admission, and same-coordinate association
//! remain separate diagnostic layers; none is a physical-source label.
use crate::{
    detection::{self, DetectionOptions, DetectionResult},
    fits,
    measurement::{Detection, MeasurementOptions},
    models::FitsFrame,
    spatial::KdTree,
    Result,
};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

pub const FEATURE_CLASS_ORDER: [&str; 9] = [
    "range_anomaly",
    "linear_artifact",
    "masked_or_edge",
    "crowded_blend",
    "spike_or_support",
    "weak_or_background",
    "shape_outlier",
    "compact_quality",
    "other_rejected",
];

const PROFILE_NOTE: &str = "baseline_match 是同一首帧坐标在参数变化后的互相最近一对一响应；feature_class 是算法首要类别，不是物理恒星类别。";
const INTERPRETATION: &str = "参数扫描只衡量同一坐标的算法响应稳定性；候选持续不等于物理恒星，质量通过也不替代星表/WCS、注入回收和跨帧身份确认。";

#[derive(Clone, Debug, Serialize)]
pub struct FeatureParameterRunRow {
    pub source_path: String,
    pub parameter_family: String,
    pub parameter_value: f64,
    pub threshold_sigma: f64,
    pub min_flux_snr: f64,
    pub candidate_count: usize,
    pub returned_count: usize,
    pub quality_count: usize,
    pub rejected_count: usize,
    pub background_adu: f64,
    pub noise_adu: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeatureParameterProfileRow {
    pub source_path: String,
    pub parameter_family: String,
    pub parameter_value: f64,
    pub threshold_sigma: f64,
    pub min_flux_snr: f64,
    pub feature_class: String,
    pub baseline_candidate_anchor_count: usize,
    pub current_candidate_count: usize,
    pub current_quality_count: usize,
    pub current_quality_fraction: Option<f64>,
    pub baseline_candidate_matched_count: usize,
    pub baseline_same_class_matched_count: usize,
    pub baseline_candidate_match_fraction: Option<f64>,
    pub baseline_same_class_match_fraction: Option<f64>,
    pub baseline_quality_anchor_count: usize,
    pub baseline_quality_matched_count: usize,
    pub baseline_quality_still_passed_count: usize,
    pub baseline_quality_match_fraction: Option<f64>,
    pub baseline_quality_still_passed_fraction: Option<f64>,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeatureParameterTransitionRow {
    pub source_path: String,
    pub parameter_family: String,
    pub parameter_value: f64,
    pub threshold_sigma: f64,
    pub min_flux_snr: f64,
    pub baseline_feature_class: String,
    pub current_feature_class: String,
    pub matched_count: usize,
    pub current_quality_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeatureParameterSensitivityResult {
    pub source_path: String,
    pub image_shape: [usize; 2],
    pub baseline_threshold_sigma: f64,
    pub baseline_min_flux_snr: f64,
    pub association_radius_px: f64,
    pub detector_parameters: Value,
    pub requested_threshold_levels: Vec<f64>,
    pub requested_flux_snr_levels: Vec<f64>,
    pub runs: Vec<FeatureParameterRunRow>,
    pub profiles: Vec<FeatureParameterProfileRow>,
    pub transitions: Vec<FeatureParameterTransitionRow>,
    pub observations: Value,
}

#[derive(Clone, Debug)]
pub struct FeatureParameterSensitivityOptions {
    pub threshold_levels: Vec<f64>,
    pub flux_snr_levels: Vec<f64>,
    pub baseline_threshold_sigma: f64,
    pub baseline_min_flux_snr: f64,
    pub min_distance: usize,
    pub aperture_radius: usize,
    pub psf_fwhm: f64,
    pub background_box_size: usize,
    pub background_sample_limit: usize,
    pub min_psf_support_pixels: usize,
    pub proposal_mode: String,
    pub max_sources: Option<usize>,
    pub association_radius_px: f64,
}

impl Default for FeatureParameterSensitivityOptions {
    fn default() -> Self {
        Self {
            threshold_levels: vec![4.0, 5.0, 6.0, 8.0],
            flux_snr_levels: vec![3.0, 5.0, 7.0, 9.0],
            baseline_threshold_sigma: 4.0,
            baseline_min_flux_snr: 5.0,
            min_distance: 4,
            aperture_radius: 4,
            psf_fwhm: 2.0,
            background_box_size: 128,
            background_sample_limit: 100_000,
            min_psf_support_pixels: 3,
            proposal_mode: "hybrid".into(),
            max_sources: None,
            association_radius_px: 1.5,
        }
    }
}

#[derive(Clone, Debug)]
struct RunSpec {
    family: String,
    value: f64,
    threshold_sigma: f64,
    min_flux_snr: f64,
}

#[derive(Clone, Debug)]
struct RunOutput {
    threshold_sigma: f64,
    min_flux_snr: f64,
    detection: DetectionResult,
}

type ProgressReporter<'a> = Option<&'a mut dyn FnMut(usize, usize, &str)>;

fn has_flag(source: &Detection, flags: &[&str]) -> bool {
    source
        .flags
        .iter()
        .any(|flag| flags.iter().any(|candidate| flag == candidate))
}

pub fn classify_source_feature(source: &Detection) -> &'static str {
    if has_flag(source, &["SATURATED", "NEGATIVE_OVERFLOW", "CODE_PATTERN"]) {
        "range_anomaly"
    } else if has_flag(source, &["LINE_ARTIFACT"]) {
        "linear_artifact"
    } else if has_flag(source, &["MASKED", "PARTIAL_MASKED", "EDGE"]) {
        "masked_or_edge"
    } else if has_flag(source, &["UNRESOLVED_BLEND"]) {
        "crowded_blend"
    } else if has_flag(
        source,
        &[
            "SPIKE",
            "NARROW",
            "SMALL_FOOTPRINT",
            "INSUFFICIENT_PSF_SUPPORT",
        ],
    ) {
        "spike_or_support"
    } else if has_flag(
        source,
        &["LOW_FLUX_SNR", "NON_POSITIVE_FLUX", "BACKGROUND_UNCERTAIN"],
    ) {
        "weak_or_background"
    } else if has_flag(source, &["NO_SHAPE", "BROAD", "ELONGATED", "DIFFUSE"]) {
        "shape_outlier"
    } else if source.quality_passed {
        "compact_quality"
    } else {
        "other_rejected"
    }
}

fn point(source: &Detection) -> Option<[f64; 2]> {
    let [x, y] = source.peak_position();
    (x.is_finite() && y.is_finite()).then_some([x, y])
}

/// Mutual nearest-neighbor matching, with original source order as the tie rule.
pub fn reciprocal_match_indices(
    anchor_sources: &[Detection],
    current_sources: &[Detection],
    radius_px: f64,
) -> Result<BTreeMap<usize, usize>> {
    if !radius_px.is_finite() || radius_px <= 0.0 {
        return Ok(BTreeMap::new());
    }
    let anchors: Vec<_> = anchor_sources
        .iter()
        .enumerate()
        .filter_map(|(index, source)| point(source).map(|position| (index, position)))
        .collect();
    let current: Vec<_> = current_sources
        .iter()
        .enumerate()
        .filter_map(|(index, source)| point(source).map(|position| (index, position)))
        .collect();
    if anchors.is_empty() || current.is_empty() {
        return Ok(BTreeMap::new());
    }
    let anchor_tree = KdTree::new(&anchors.iter().map(|row| row.1).collect::<Vec<_>>())?;
    let current_tree = KdTree::new(&current.iter().map(|row| row.1).collect::<Vec<_>>())?;
    let mut result = BTreeMap::new();
    for (anchor_position, (anchor_index, anchor_point)) in anchors.iter().enumerate() {
        let Some((current_position, distance)) = current_tree.nearest(*anchor_point) else {
            continue;
        };
        if distance > radius_px {
            continue;
        }
        let Some((nearest_anchor, reverse_distance)) =
            anchor_tree.nearest(current[current_position].1)
        else {
            continue;
        };
        if nearest_anchor == anchor_position && reverse_distance <= radius_px {
            result.insert(*anchor_index, current[current_position].0);
        }
    }
    Ok(result)
}

fn fraction(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn ordered_class_index(value: &str) -> usize {
    FEATURE_CLASS_ORDER
        .iter()
        .position(|candidate| *candidate == value)
        .unwrap_or(FEATURE_CLASS_ORDER.len())
}

fn profiles_for_run(
    source_path: &str,
    spec: &RunSpec,
    baseline_sources: &[Detection],
    current_sources: &[Detection],
    association_radius_px: f64,
) -> Result<Vec<FeatureParameterProfileRow>> {
    let baseline_classes: Vec<_> = baseline_sources
        .iter()
        .map(classify_source_feature)
        .collect();
    let current_classes: Vec<_> = current_sources
        .iter()
        .map(classify_source_feature)
        .collect();
    let matches =
        reciprocal_match_indices(baseline_sources, current_sources, association_radius_px)?;
    let mut rows = Vec::with_capacity(FEATURE_CLASS_ORDER.len());
    for feature_class in FEATURE_CLASS_ORDER {
        let anchor_indices: Vec<_> = baseline_classes
            .iter()
            .enumerate()
            .filter_map(|(index, class)| (*class == feature_class).then_some(index))
            .collect();
        let matched_indices: Vec<_> = anchor_indices
            .iter()
            .filter_map(|index| matches.get(index).map(|current| (*index, *current)))
            .collect();
        let same_class_count = matched_indices
            .iter()
            .filter(|(_, current)| current_classes[*current] == feature_class)
            .count();
        let quality_anchor_indices: Vec<_> = anchor_indices
            .iter()
            .copied()
            .filter(|index| baseline_sources[*index].quality_passed)
            .collect();
        let quality_matched_indices: Vec<_> = quality_anchor_indices
            .iter()
            .filter_map(|index| matches.get(index).map(|current| (*index, *current)))
            .collect();
        let quality_still_passed_count = quality_matched_indices
            .iter()
            .filter(|(_, current)| current_sources[*current].quality_passed)
            .count();
        let current_count = current_classes
            .iter()
            .filter(|class| **class == feature_class)
            .count();
        let current_quality_count = current_classes
            .iter()
            .zip(current_sources)
            .filter(|(class, source)| **class == feature_class && source.quality_passed)
            .count();
        rows.push(FeatureParameterProfileRow {
            source_path: source_path.into(),
            parameter_family: spec.family.clone(),
            parameter_value: spec.value,
            threshold_sigma: spec.threshold_sigma,
            min_flux_snr: spec.min_flux_snr,
            feature_class: feature_class.into(),
            baseline_candidate_anchor_count: anchor_indices.len(),
            current_candidate_count: current_count,
            current_quality_count,
            current_quality_fraction: fraction(current_quality_count, current_count),
            baseline_candidate_matched_count: matched_indices.len(),
            baseline_same_class_matched_count: same_class_count,
            baseline_candidate_match_fraction: fraction(
                matched_indices.len(),
                anchor_indices.len(),
            ),
            baseline_same_class_match_fraction: fraction(same_class_count, anchor_indices.len()),
            baseline_quality_anchor_count: quality_anchor_indices.len(),
            baseline_quality_matched_count: quality_matched_indices.len(),
            baseline_quality_still_passed_count: quality_still_passed_count,
            baseline_quality_match_fraction: fraction(
                quality_matched_indices.len(),
                quality_anchor_indices.len(),
            ),
            baseline_quality_still_passed_fraction: fraction(
                quality_still_passed_count,
                quality_anchor_indices.len(),
            ),
            note: PROFILE_NOTE.into(),
        });
    }
    Ok(rows)
}

fn transitions_for_run(
    source_path: &str,
    spec: &RunSpec,
    baseline_sources: &[Detection],
    current_sources: &[Detection],
    association_radius_px: f64,
) -> Result<Vec<FeatureParameterTransitionRow>> {
    let baseline_classes: Vec<_> = baseline_sources
        .iter()
        .map(classify_source_feature)
        .collect();
    let current_classes: Vec<_> = current_sources
        .iter()
        .map(classify_source_feature)
        .collect();
    let matches =
        reciprocal_match_indices(baseline_sources, current_sources, association_radius_px)?;
    let mut counts: BTreeMap<(usize, usize), (usize, usize)> = BTreeMap::new();
    for (baseline_index, current_index) in matches {
        let key = (
            ordered_class_index(baseline_classes[baseline_index]),
            ordered_class_index(current_classes[current_index]),
        );
        let counts = counts.entry(key).or_default();
        counts.0 += 1;
        counts.1 += usize::from(current_sources[current_index].quality_passed);
    }
    Ok(counts
        .into_iter()
        .map(
            |((baseline_class, current_class), (matched_count, current_quality_count))| {
                FeatureParameterTransitionRow {
                    source_path: source_path.into(),
                    parameter_family: spec.family.clone(),
                    parameter_value: spec.value,
                    threshold_sigma: spec.threshold_sigma,
                    min_flux_snr: spec.min_flux_snr,
                    baseline_feature_class: FEATURE_CLASS_ORDER[baseline_class].into(),
                    current_feature_class: FEATURE_CLASS_ORDER[current_class].into(),
                    matched_count,
                    current_quality_count,
                }
            },
        )
        .collect())
}

fn normalise_levels(values: &[f64], name: &str) -> Result<Vec<f64>> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(format!("{name} must contain finite positive values"));
    }
    let mut result = Vec::new();
    for value in values {
        if !result.contains(value) {
            result.push(*value);
        }
    }
    Ok(result)
}

fn run_detection(
    frame: &FitsFrame,
    mask: &[bool],
    threshold_sigma: f64,
    min_flux_snr: f64,
    options: &FeatureParameterSensitivityOptions,
) -> Result<DetectionResult> {
    let detection_options = DetectionOptions {
        threshold_sigma,
        min_distance: options.min_distance,
        max_sources: options.max_sources,
        background_box_size: options.background_box_size,
        background_sample_limit: options.background_sample_limit,
        reject_linear_artifacts: true,
        proposal_mode: options.proposal_mode.clone(),
        measurement: MeasurementOptions {
            aperture_radius: options.aperture_radius,
            psf_fwhm: options.psf_fwhm,
            min_flux_snr,
            min_psf_support_pixels: options.min_psf_support_pixels,
            refine_local_background: true,
            ..Default::default()
        },
        ..Default::default()
    };
    detection::detect_sources(
        &frame.data,
        Some(mask),
        frame.height,
        frame.width,
        &detection_options,
        None,
    )
}

pub fn run_feature_parameter_sensitivity(
    path: impl AsRef<Path>,
    options: &FeatureParameterSensitivityOptions,
    mut progress: ProgressReporter<'_>,
) -> Result<FeatureParameterSensitivityResult> {
    if options.baseline_threshold_sigma <= 0.0
        || !options.baseline_threshold_sigma.is_finite()
        || options.baseline_min_flux_snr <= 0.0
        || !options.baseline_min_flux_snr.is_finite()
    {
        return Err("baseline thresholds must be finite and positive".into());
    }
    if options.min_distance < 1 || options.aperture_radius < 1 || options.psf_fwhm <= 0.0 {
        return Err("detector geometry parameters must be positive".into());
    }
    if options.background_box_size < 16 || options.background_sample_limit == 0 {
        return Err("background parameters are invalid".into());
    }
    if !(1..=9).contains(&options.min_psf_support_pixels) {
        return Err("min_psf_support_pixels must be within 1..=9".into());
    }
    if options.association_radius_px <= 0.0 || !options.association_radius_px.is_finite() {
        return Err("association_radius_px must be finite and positive".into());
    }
    if !["gaussian", "hybrid", "ensemble"].contains(&options.proposal_mode.as_str()) {
        return Err("proposal_mode must be gaussian, hybrid, or ensemble".into());
    }
    let threshold_values = normalise_levels(&options.threshold_levels, "threshold_levels")?;
    let flux_values = normalise_levels(&options.flux_snr_levels, "flux_snr_levels")?;
    let frame = fits::read_fits(path.as_ref(), true)?;
    let source_path = if frame.path.is_absolute() {
        frame.path.to_string_lossy().into_owned()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(&frame.path)
            .to_string_lossy()
            .into_owned()
    };
    let mask = fits::auxiliary_mask(frame.height, frame.width)?;
    let mut specs = Vec::new();
    for value in &threshold_values {
        specs.push(RunSpec {
            family: "threshold_sigma".into(),
            value: *value,
            threshold_sigma: *value,
            min_flux_snr: options.baseline_min_flux_snr,
        });
    }
    for value in &flux_values {
        specs.push(RunSpec {
            family: "min_flux_snr".into(),
            value: *value,
            threshold_sigma: options.baseline_threshold_sigma,
            min_flux_snr: *value,
        });
    }
    let mut configs = Vec::<(f64, f64)>::new();
    for config in specs
        .iter()
        .map(|spec| (spec.threshold_sigma, spec.min_flux_snr))
        .chain(std::iter::once((
            options.baseline_threshold_sigma,
            options.baseline_min_flux_snr,
        )))
    {
        if !configs.contains(&config) {
            configs.push(config);
        }
    }
    let mut outputs = Vec::with_capacity(configs.len());
    for (index, (threshold_sigma, min_flux_snr)) in configs.iter().copied().enumerate() {
        let detection = run_detection(&frame, &mask, threshold_sigma, min_flux_snr, options)?;
        outputs.push(RunOutput {
            threshold_sigma,
            min_flux_snr,
            detection,
        });
        if let Some(callback) = progress.as_deref_mut() {
            let label = format!("threshold={threshold_sigma}, flux_snr={min_flux_snr}");
            callback(index + 1, configs.len(), &label);
        }
    }
    let output_for = |config: (f64, f64)| -> Result<&RunOutput> {
        outputs
            .iter()
            .find(|row| row.threshold_sigma == config.0 && row.min_flux_snr == config.1)
            .ok_or_else(|| "internal sensitivity configuration is missing".into())
    };
    let baseline_config = (
        options.baseline_threshold_sigma,
        options.baseline_min_flux_snr,
    );
    let baseline = output_for(baseline_config)?;
    let mut run_rows = Vec::with_capacity(specs.len());
    let mut profile_rows = Vec::with_capacity(specs.len() * FEATURE_CLASS_ORDER.len());
    let mut transition_rows = Vec::new();
    for spec in &specs {
        let detection = &output_for((spec.threshold_sigma, spec.min_flux_snr))?.detection;
        let quality_count = detection.quality_count;
        run_rows.push(FeatureParameterRunRow {
            source_path: source_path.clone(),
            parameter_family: spec.family.clone(),
            parameter_value: spec.value,
            threshold_sigma: spec.threshold_sigma,
            min_flux_snr: spec.min_flux_snr,
            candidate_count: detection.candidate_count,
            returned_count: detection.sources.len(),
            quality_count,
            rejected_count: detection.sources.len().saturating_sub(quality_count),
            background_adu: detection.background,
            noise_adu: detection.noise,
        });
        profile_rows.extend(profiles_for_run(
            &source_path,
            spec,
            &baseline.detection.sources,
            &detection.sources,
            options.association_radius_px,
        )?);
        transition_rows.extend(transitions_for_run(
            &source_path,
            spec,
            &baseline.detection.sources,
            &detection.sources,
            options.association_radius_px,
        )?);
    }
    let threshold_runs: Vec<_> = run_rows
        .iter()
        .filter(|row| row.parameter_family == "threshold_sigma")
        .collect();
    let mut flux_runs: Vec<_> = run_rows
        .iter()
        .filter(|row| row.parameter_family == "min_flux_snr")
        .collect();
    flux_runs.sort_by(|left, right| left.parameter_value.total_cmp(&right.parameter_value));
    let threshold_min = threshold_runs.iter().map(|row| row.candidate_count).min();
    let threshold_max = threshold_runs.iter().map(|row| row.candidate_count).max();
    let flux_candidate_pool_stable = flux_runs.first().is_some_and(|first| {
        flux_runs
            .iter()
            .all(|row| row.candidate_count == first.candidate_count)
    });
    let flux_quality_monotone = flux_runs
        .windows(2)
        .all(|pair| pair[0].quality_count >= pair[1].quality_count);
    let mut observations = Map::new();
    observations.insert(
        "baseline_candidate_count".into(),
        json!(baseline.detection.candidate_count),
    );
    observations.insert(
        "baseline_returned_count".into(),
        json!(baseline.detection.sources.len()),
    );
    observations.insert(
        "baseline_quality_count".into(),
        json!(baseline.detection.quality_count),
    );
    observations.insert(
        "threshold_candidate_count_range".into(),
        json!([threshold_min, threshold_max]),
    );
    observations.insert(
        "flux_candidate_pool_stable".into(),
        json!(flux_candidate_pool_stable),
    );
    observations.insert(
        "flux_quality_monotone_nonincreasing_with_threshold".into(),
        json!(flux_quality_monotone),
    );
    observations.insert("interpretation".into(), json!(INTERPRETATION));
    let mut detector_parameters = Map::new();
    detector_parameters.insert("min_distance".into(), json!(options.min_distance));
    detector_parameters.insert("aperture_radius".into(), json!(options.aperture_radius));
    detector_parameters.insert("psf_fwhm".into(), json!(options.psf_fwhm));
    detector_parameters.insert(
        "background_box_size".into(),
        json!(options.background_box_size),
    );
    detector_parameters.insert(
        "background_sample_limit".into(),
        json!(options.background_sample_limit),
    );
    detector_parameters.insert(
        "min_psf_support_pixels".into(),
        json!(options.min_psf_support_pixels),
    );
    detector_parameters.insert("proposal_mode".into(), json!(options.proposal_mode));
    detector_parameters.insert("max_sources".into(), json!(options.max_sources));
    detector_parameters.insert("refine_local_background".into(), json!(true));
    detector_parameters.insert("reject_linear_artifacts".into(), json!(true));
    Ok(FeatureParameterSensitivityResult {
        source_path,
        image_shape: [frame.height, frame.width],
        baseline_threshold_sigma: options.baseline_threshold_sigma,
        baseline_min_flux_snr: options.baseline_min_flux_snr,
        association_radius_px: options.association_radius_px,
        detector_parameters: Value::Object(detector_parameters),
        requested_threshold_levels: threshold_values,
        requested_flux_snr_levels: flux_values,
        runs: run_rows,
        profiles: profile_rows,
        transitions: transition_rows,
        observations: Value::Object(observations),
    })
}

const RUN_FIELDS: &[&str] = &[
    "source_path",
    "parameter_family",
    "parameter_value",
    "threshold_sigma",
    "min_flux_snr",
    "candidate_count",
    "returned_count",
    "quality_count",
    "rejected_count",
    "background_adu",
    "noise_adu",
];
const PROFILE_FIELDS: &[&str] = &[
    "source_path",
    "parameter_family",
    "parameter_value",
    "threshold_sigma",
    "min_flux_snr",
    "feature_class",
    "baseline_candidate_anchor_count",
    "current_candidate_count",
    "current_quality_count",
    "current_quality_fraction",
    "baseline_candidate_matched_count",
    "baseline_same_class_matched_count",
    "baseline_candidate_match_fraction",
    "baseline_same_class_match_fraction",
    "baseline_quality_anchor_count",
    "baseline_quality_matched_count",
    "baseline_quality_still_passed_count",
    "baseline_quality_match_fraction",
    "baseline_quality_still_passed_fraction",
    "note",
];
const TRANSITION_FIELDS: &[&str] = &[
    "source_path",
    "parameter_family",
    "parameter_value",
    "threshold_sigma",
    "min_flux_snr",
    "baseline_feature_class",
    "current_feature_class",
    "matched_count",
    "current_quality_count",
];

fn python_float(value: f64) -> String {
    let mut text = value.to_string();
    if value.is_finite() && !text.contains('.') && !text.contains('e') && !text.contains('E') {
        text.push_str(".0");
    }
    text
}

fn csv_cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(number) if number.is_f64() => {
            number.as_f64().map(python_float).unwrap_or_default()
        }
        Value::Number(number) => number.to_string(),
        Value::String(text) => text.clone(),
        _ => value.to_string(),
    }
}

fn write_table<T: Serialize>(path: &Path, fields: &[&str], rows: &[T]) -> Result<()> {
    let mut file = File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(fields)
        .map_err(|error| error.to_string())?;
    for row in rows {
        let value = serde_json::to_value(row).map_err(|error| error.to_string())?;
        let record = fields
            .iter()
            .map(|field| csv_cell(value.get(*field).unwrap_or(&Value::Null)));
        writer
            .write_record(record)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    Ok(())
}

pub fn write_artifacts(
    result: &FeatureParameterSensitivityResult,
    out_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    let output = out_dir.as_ref();
    let output = if output.is_absolute() {
        output.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(output)
    };
    fs::create_dir_all(&output).map_err(|error| error.to_string())?;
    write_table(
        &output.join("feature_parameter_runs.csv"),
        RUN_FIELDS,
        &result.runs,
    )?;
    write_table(
        &output.join("feature_parameter_profiles.csv"),
        PROFILE_FIELDS,
        &result.profiles,
    )?;
    let transition_fields = if result.transitions.is_empty() {
        &["baseline_feature_class"][..]
    } else {
        TRANSITION_FIELDS
    };
    write_table(
        &output.join("feature_parameter_transitions.csv"),
        transition_fields,
        &result.transitions,
    )?;
    let json = serde_json::to_string_pretty(result).map_err(|error| error.to_string())?;
    fs::write(output.join("feature_parameter_sensitivity.json"), json)
        .map_err(|error| error.to_string())?;
    Ok(output)
}

struct CliArgs {
    path: PathBuf,
    options: FeatureParameterSensitivityOptions,
    out_dir: PathBuf,
}

fn parse_levels(text: &str, name: &str) -> Result<Vec<f64>> {
    let values = text
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<f64>()
                .map_err(|_| format!("{name} must be comma-separated numbers"))
        })
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    Ok(values)
}

fn parse_args(args: &[String]) -> Result<CliArgs> {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        return Err("__help__".into());
    }
    let mut path = None;
    let mut out_dir = PathBuf::from("tmp/feature-parameter-sensitivity-current");
    let mut options = FeatureParameterSensitivityOptions::default();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline) = arg
            .strip_prefix("--")
            .map(|rest| {
                rest.split_once('=')
                    .map_or((rest, None), |(name, value)| (name, Some(value)))
            })
            .unwrap_or(("", None));
        if name.is_empty() {
            if path.replace(PathBuf::from(arg)).is_some() {
                return Err("only one FITS path is accepted".into());
            }
            index += 1;
            continue;
        }
        let value = if let Some(value) = inline {
            value.to_owned()
        } else {
            index += 1;
            args.get(index)
                .cloned()
                .ok_or_else(|| format!("--{name} requires a value"))?
        };
        match name {
            "threshold-levels" => {
                options.threshold_levels = parse_levels(&value, "threshold_levels")?
            }
            "flux-snr-levels" => options.flux_snr_levels = parse_levels(&value, "flux_snr_levels")?,
            "baseline-threshold-sigma" => {
                options.baseline_threshold_sigma =
                    value.parse().map_err(|_| "invalid baseline threshold")?
            }
            "baseline-min-flux-snr" => {
                options.baseline_min_flux_snr =
                    value.parse().map_err(|_| "invalid baseline flux SNR")?
            }
            "min-distance" => {
                options.min_distance = value.parse().map_err(|_| "invalid min distance")?
            }
            "aperture-radius" => {
                options.aperture_radius = value.parse().map_err(|_| "invalid aperture radius")?
            }
            "psf-fwhm" => options.psf_fwhm = value.parse().map_err(|_| "invalid PSF FWHM")?,
            "background-box-size" => {
                options.background_box_size =
                    value.parse().map_err(|_| "invalid background box size")?
            }
            "background-sample-limit" => {
                options.background_sample_limit = value
                    .parse()
                    .map_err(|_| "invalid background sample limit")?
            }
            "min-psf-support-pixels" => {
                options.min_psf_support_pixels =
                    value.parse().map_err(|_| "invalid PSF support count")?
            }
            "proposal-mode" => options.proposal_mode = value,
            "max-sources" => {
                let value = value.parse::<i64>().map_err(|_| "invalid max sources")?;
                options.max_sources = (value > 0).then_some(value as usize);
            }
            "association-radius-px" => {
                options.association_radius_px =
                    value.parse().map_err(|_| "invalid association radius")?
            }
            "out-dir" => out_dir = PathBuf::from(value),
            _ => return Err(format!("unknown option --{name}")),
        }
        index += 1;
    }
    Ok(CliArgs {
        path: path.ok_or("a FITS input path is required")?,
        options,
        out_dir,
    })
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let cli = parse_args(args)?;
    let result = run_feature_parameter_sensitivity(&cli.path, &cli.options, None)?;
    let output = write_artifacts(&result, &cli.out_dir)?;
    Ok(json!({
        "result": result,
        "result_path": output.join("feature_parameter_sensitivity.json"),
        "runs_path": output.join("feature_parameter_runs.csv"),
        "profiles_path": output.join("feature_parameter_profiles.csv"),
        "transitions_path": output.join("feature_parameter_transitions.csv")
    }))
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("Usage: rst19-feature-parameter-sensitivity FITS_PATH [OPTIONS]\n\n--threshold-levels CSV --flux-snr-levels CSV --baseline-threshold-sigma N --baseline-min-flux-snr N\n--min-distance N --aperture-radius N --psf-fwhm N --background-box-size N --background-sample-limit N\n--min-psf-support-pixels N --proposal-mode gaussian|hybrid|ensemble --max-sources N --association-radius-px N --out-dir PATH");
        return Ok(());
    }
    let cli = parse_args(args)?;
    let mut progress = |index: usize, total: usize, label: &str| {
        eprintln!("feature parameter sensitivity: {index}/{total} · {label}");
    };
    let result = run_feature_parameter_sensitivity(&cli.path, &cli.options, Some(&mut progress))?;
    let output = write_artifacts(&result, &cli.out_dir)?;
    println!(
        "rst19-feature-parameter-sensitivity: 审计已写入 {}",
        output.display()
    );
    println!(
        "baseline candidates={} quality={}",
        result.observations["baseline_candidate_count"],
        result.observations["baseline_quality_count"]
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(id: i64, x: f64, y: f64, quality_passed: bool, flags: &[&str]) -> Detection {
        Detection {
            detection_id: id,
            x,
            y,
            peak: 100.0,
            flux: 200.0,
            background: 10.0,
            noise: 2.0,
            snr: 20.0,
            fwhm: Some(2.0),
            flags: flags.iter().map(|flag| (*flag).into()).collect(),
            flux_error: Some(5.0),
            flux_snr: Some(10.0),
            filter_snr: Some(12.0),
            fwhm_x: Some(2.0),
            fwhm_y: Some(2.0),
            ellipticity: Some(0.0),
            sharpness: Some(0.2),
            footprint_pixels: Some(10),
            psf_support_pixels: Some(5),
            quality_passed,
            peak_x: Some(x),
            peak_y: Some(y),
            centroid_shift_px: Some(0.0),
            proposal_methods: vec!["gaussian".into()],
            proposal_scales: vec![1.0],
            proposal_snr: Some(12.0),
            nearest_gaussian_px: Some(0.0),
            deblend_delta_bic: None,
            deblend_component_snr: None,
            repeated_code_count: None,
            range_anomaly_pixel_count: None,
            repeated_code_values: vec![],
        }
    }

    #[test]
    fn feature_classes_follow_the_python_priority_order() {
        assert_eq!(
            classify_source_feature(&source(
                1,
                0.0,
                0.0,
                true,
                &["BROAD", "LINE_ARTIFACT", "SATURATED"]
            )),
            "range_anomaly"
        );
        assert_eq!(
            classify_source_feature(&source(2, 0.0, 0.0, false, &["LINE_ARTIFACT", "EDGE"])),
            "linear_artifact"
        );
        assert_eq!(
            classify_source_feature(&source(3, 0.0, 0.0, true, &[])),
            "compact_quality"
        );
        assert_eq!(
            classify_source_feature(&source(4, 0.0, 0.0, false, &[])),
            "other_rejected"
        );
    }

    #[test]
    fn reciprocal_matches_are_radius_limited_and_one_to_one() {
        let anchors = [
            source(1, 10.0, 10.0, true, &[]),
            source(2, 30.0, 30.0, true, &[]),
        ];
        let current = [
            source(3, 10.4, 10.2, true, &[]),
            source(4, 60.0, 60.0, true, &[]),
        ];
        assert_eq!(
            reciprocal_match_indices(&anchors, &current, 1.0).unwrap(),
            BTreeMap::from([(0, 0)])
        );
    }

    #[test]
    fn level_normalisation_preserves_first_seen_order() {
        assert_eq!(
            normalise_levels(&[5.0, 4.0, 5.0], "levels").unwrap(),
            [5.0, 4.0]
        );
        assert!(normalise_levels(&[0.0], "levels").is_err());
    }
}
