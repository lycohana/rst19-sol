//! Local cross-check of reverse-quality source locations against forward detections.
use crate::{
    detection::{self, DetectionOptions},
    fits,
    measurement::{Detection, MeasurementOptions},
    models::FitsFrame,
    spatial::KdTree,
    Result,
};
use csv::Terminator;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use unicode_casefold::UnicodeCaseFold;

const RADII: [f64; 3] = [1.0, 2.0, 4.0];

#[derive(Debug, Clone)]
struct ReverseSource {
    frame_index: usize,
    detection_id: i64,
    x: f64,
    y: f64,
    raw_evidence_layer: String,
    flux_snr: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct FrameRow {
    frame_index: usize,
    frame_path: String,
    candidate_count: usize,
    returned_count: usize,
    quality_count: usize,
    reverse_quality_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SourceRow {
    frame_index: usize,
    frame_path: String,
    reverse_detection_id: i64,
    reverse_x: f64,
    reverse_y: f64,
    raw_evidence_layer: String,
    reverse_flux_snr: Option<f64>,
    candidate_count_r1: usize,
    candidate_count_r2: usize,
    candidate_count_r4: usize,
    quality_count_r1: usize,
    quality_count_r2: usize,
    quality_count_r4: usize,
    overlap_class: String,
    nearest_candidate_detection_id: Option<i64>,
    nearest_candidate_x: Option<f64>,
    nearest_candidate_y: Option<f64>,
    nearest_candidate_peak_x: Option<f64>,
    nearest_candidate_peak_y: Option<f64>,
    nearest_candidate_peak: Option<f64>,
    nearest_candidate_flux_snr: Option<f64>,
    nearest_candidate_filter_snr: Option<f64>,
    nearest_candidate_fwhm: Option<f64>,
    nearest_candidate_feature_class: Option<String>,
    nearest_candidate_quality_passed: Option<bool>,
    nearest_candidate_flags: String,
    nearest_candidate_distance_px: Option<f64>,
    nearest_quality_detection_id: Option<i64>,
    nearest_quality_x: Option<f64>,
    nearest_quality_y: Option<f64>,
    nearest_quality_peak_x: Option<f64>,
    nearest_quality_peak_y: Option<f64>,
    nearest_quality_peak: Option<f64>,
    nearest_quality_flux_snr: Option<f64>,
    nearest_quality_filter_snr: Option<f64>,
    nearest_quality_fwhm: Option<f64>,
    nearest_quality_feature_class: Option<String>,
    nearest_quality_quality_passed: Option<bool>,
    nearest_quality_flags: String,
    nearest_quality_distance_px: Option<f64>,
}

#[derive(Debug, Clone)]
struct SourceSummary {
    detection_id: i64,
    x: f64,
    y: f64,
    peak_x: f64,
    peak_y: f64,
    peak: f64,
    flux_snr: Option<f64>,
    filter_snr: Option<f64>,
    fwhm: Option<f64>,
    feature_class: String,
    quality_passed: bool,
    flags: String,
}

impl SourceSummary {
    fn from_detection(source: &Detection) -> Self {
        Self {
            detection_id: source.detection_id,
            x: source.x,
            y: source.y,
            peak_x: source.peak_x.unwrap_or(source.x),
            peak_y: source.peak_y.unwrap_or(source.y),
            peak: source.peak,
            flux_snr: source.flux_snr,
            filter_snr: source.filter_snr,
            fwhm: source.fwhm,
            feature_class: classify_source_feature(source),
            quality_passed: source.quality_passed,
            flags: source.flags.join("|"),
        }
    }
}

#[derive(Debug)]
struct Nearest<'a> {
    source: &'a SourceSummary,
    distance: f64,
}

#[derive(Debug, Clone)]
struct AuditResult {
    frame_rows: Vec<FrameRow>,
    source_rows: Vec<SourceRow>,
    raw_evidence_layer_counts: BTreeMap<String, usize>,
    overlap_class_counts: BTreeMap<String, usize>,
    parameters: Value,
    conclusion: String,
}

impl AuditResult {
    fn as_dict(&self) -> Value {
        json!({
            "frame_count": self.frame_rows.len(),
            "reverse_quality_source_count": self.source_rows.len(),
            "source_row_count": self.source_rows.len(),
            "frame_rows": self.frame_rows,
            "frame_summaries": self.frame_rows,
            "raw_evidence_layer_counts": self.raw_evidence_layer_counts,
            "overlap_class_counts": self.overlap_class_counts,
            "parameters": self.parameters,
            "detector_parameters": self.parameters,
            "conclusion": self.conclusion,
            "interpretation": self.conclusion,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub threshold_sigma: f64,
    pub min_distance: usize,
    pub aperture_radius: usize,
    pub psf_fwhm: f64,
    pub background_box_size: usize,
    pub background_sample_limit: usize,
    pub min_flux_snr: f64,
    pub min_psf_support_pixels: usize,
    pub proposal_mode: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            threshold_sigma: 4.0,
            min_distance: 4,
            aperture_radius: 4,
            psf_fwhm: 2.0,
            background_box_size: 128,
            background_sample_limit: 100_000,
            min_flux_snr: 5.0,
            min_psf_support_pixels: 3,
            proposal_mode: "hybrid".into(),
        }
    }
}

#[derive(Debug, Clone)]
struct CliArgs {
    inputs: Vec<PathBuf>,
    reverse_source_csv: PathBuf,
    out_dir: PathBuf,
    options: Options,
}

fn classify_source_feature(source: &Detection) -> String {
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
    for (class, flags) in PRIORITY {
        if source
            .flags
            .iter()
            .any(|flag| flags.contains(&flag.as_str()))
        {
            return class.into();
        }
    }
    if source.quality_passed {
        "compact_quality".into()
    } else {
        "other_rejected".into()
    }
}

fn load_reverse_sources(path: &Path, frame_count: usize) -> Result<Vec<Vec<ReverseSource>>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|error| error.to_string())?;
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    let required = [
        "frame_index",
        "detection_id",
        "x",
        "y",
        "raw_evidence_layer",
        "flux_snr",
    ];
    let mut indices = BTreeMap::new();
    let mut missing = Vec::new();
    for field in required {
        if let Some(index) = headers.iter().position(|header| header == field) {
            indices.insert(field, index);
        } else {
            missing.push(field);
        }
    }
    if !missing.is_empty() {
        missing.sort_unstable();
        return Err(format!(
            "reverse source CSV is missing fields: {}",
            missing.join(", ")
        ));
    }
    let mut grouped = vec![Vec::new(); frame_count];
    for record in reader.records() {
        let row = record.map_err(|error| error.to_string())?;
        let text = |field: &str| -> Result<&str> {
            row.get(indices[field])
                .ok_or_else(|| "reverse source CSV contains an invalid numeric row".into())
        };
        let parse_row = || -> Result<ReverseSource> {
            let frame_index = text("frame_index")?
                .parse::<usize>()
                .map_err(|_| "reverse source CSV contains an invalid numeric row")?;
            let detection_id = text("detection_id")?
                .parse::<i64>()
                .map_err(|_| "reverse source CSV contains an invalid numeric row")?;
            let x = text("x")?
                .parse::<f64>()
                .map_err(|_| "reverse source CSV contains an invalid numeric row")?;
            let y = text("y")?
                .parse::<f64>()
                .map_err(|_| "reverse source CSV contains an invalid numeric row")?;
            let flux = text("flux_snr")?;
            let flux_snr = if flux.is_empty() {
                None
            } else {
                Some(
                    flux.parse::<f64>()
                        .map_err(|_| "reverse source CSV contains an invalid numeric row")?,
                )
            };
            if !x.is_finite() || !y.is_finite() || flux_snr.is_some_and(|value| !value.is_finite())
            {
                return Err("reverse source CSV contains an invalid numeric row".into());
            }
            Ok(ReverseSource {
                frame_index,
                detection_id,
                x,
                y,
                raw_evidence_layer: text("raw_evidence_layer")?.to_owned(),
                flux_snr,
            })
        };
        let reverse = parse_row()?;
        if !(1..=frame_count).contains(&reverse.frame_index) {
            return Err(format!(
                "reverse source frame_index is outside 1..{frame_count}"
            ));
        }
        grouped[reverse.frame_index - 1].push(reverse);
    }
    Ok(grouped)
}

fn summaries(sources: &[Detection]) -> (Vec<SourceSummary>, Option<KdTree>) {
    let rows = sources
        .iter()
        .map(SourceSummary::from_detection)
        .collect::<Vec<_>>();
    let points = rows.iter().map(|row| [row.x, row.y]).collect::<Vec<_>>();
    let tree = KdTree::new(&points).ok();
    (rows, tree)
}

fn nearest<'a>(
    point: [f64; 2],
    sources: &'a [SourceSummary],
    tree: Option<&KdTree>,
) -> Option<Nearest<'a>> {
    let (index, distance) = tree?.nearest(point)?;
    Some(Nearest {
        source: sources.get(index)?,
        distance,
    })
}

fn count_within(point: [f64; 2], tree: Option<&KdTree>, radius: f64) -> Result<usize> {
    tree.map(|tree| tree.within_radius(point, radius).map(|found| found.len()))
        .unwrap_or(Ok(0))
}

fn source_row(
    frame_index: usize,
    frame_path: &Path,
    reverse: &ReverseSource,
    candidates: &[SourceSummary],
    candidate_tree: Option<&KdTree>,
    quality: &[SourceSummary],
    quality_tree: Option<&KdTree>,
) -> Result<SourceRow> {
    let point = [reverse.x, reverse.y];
    let candidate_counts = [
        count_within(point, candidate_tree, RADII[0])?,
        count_within(point, candidate_tree, RADII[1])?,
        count_within(point, candidate_tree, RADII[2])?,
    ];
    let quality_counts = [
        count_within(point, quality_tree, RADII[0])?,
        count_within(point, quality_tree, RADII[1])?,
        count_within(point, quality_tree, RADII[2])?,
    ];
    let nearest_candidate = nearest(point, candidates, candidate_tree);
    let nearest_quality = nearest(point, quality, quality_tree);
    let overlap_class = if quality_counts[1] > 0 {
        "quality_counterpart_within_2px"
    } else if candidate_counts[1] > 0 {
        "candidate_only_within_2px"
    } else if candidate_counts[2] > 0 {
        "candidate_only_within_4px"
    } else {
        "no_forward_candidate_within_4px"
    };
    let candidate = nearest_candidate.as_ref().map(|nearest| nearest.source);
    let quality_source = nearest_quality.as_ref().map(|nearest| nearest.source);
    Ok(SourceRow {
        frame_index,
        frame_path: frame_path.to_string_lossy().into_owned(),
        reverse_detection_id: reverse.detection_id,
        reverse_x: reverse.x,
        reverse_y: reverse.y,
        raw_evidence_layer: reverse.raw_evidence_layer.clone(),
        reverse_flux_snr: reverse.flux_snr,
        candidate_count_r1: candidate_counts[0],
        candidate_count_r2: candidate_counts[1],
        candidate_count_r4: candidate_counts[2],
        quality_count_r1: quality_counts[0],
        quality_count_r2: quality_counts[1],
        quality_count_r4: quality_counts[2],
        overlap_class: overlap_class.into(),
        nearest_candidate_detection_id: candidate.map(|source| source.detection_id),
        nearest_candidate_x: candidate.map(|source| source.x),
        nearest_candidate_y: candidate.map(|source| source.y),
        nearest_candidate_peak_x: candidate.map(|source| source.peak_x),
        nearest_candidate_peak_y: candidate.map(|source| source.peak_y),
        nearest_candidate_peak: candidate.map(|source| source.peak),
        nearest_candidate_flux_snr: candidate.and_then(|source| source.flux_snr),
        nearest_candidate_filter_snr: candidate.and_then(|source| source.filter_snr),
        nearest_candidate_fwhm: candidate.and_then(|source| source.fwhm),
        nearest_candidate_feature_class: candidate.map(|source| source.feature_class.clone()),
        nearest_candidate_quality_passed: candidate.map(|source| source.quality_passed),
        nearest_candidate_flags: candidate.map_or_else(String::new, |source| source.flags.clone()),
        nearest_candidate_distance_px: nearest_candidate.map(|nearest| nearest.distance),
        nearest_quality_detection_id: quality_source.map(|source| source.detection_id),
        nearest_quality_x: quality_source.map(|source| source.x),
        nearest_quality_y: quality_source.map(|source| source.y),
        nearest_quality_peak_x: quality_source.map(|source| source.peak_x),
        nearest_quality_peak_y: quality_source.map(|source| source.peak_y),
        nearest_quality_peak: quality_source.map(|source| source.peak),
        nearest_quality_flux_snr: quality_source.and_then(|source| source.flux_snr),
        nearest_quality_filter_snr: quality_source.and_then(|source| source.filter_snr),
        nearest_quality_fwhm: quality_source.and_then(|source| source.fwhm),
        nearest_quality_feature_class: quality_source.map(|source| source.feature_class.clone()),
        nearest_quality_quality_passed: quality_source.map(|source| source.quality_passed),
        nearest_quality_flags: quality_source
            .map_or_else(String::new, |source| source.flags.clone()),
        nearest_quality_distance_px: nearest_quality.map(|nearest| nearest.distance),
    })
}

fn detect(frame: FitsFrame, options: &Options) -> Result<detection::DetectionResult> {
    let mask = fits::auxiliary_mask(frame.height, frame.width)?;
    let detector = DetectionOptions {
        measurement: MeasurementOptions {
            aperture_radius: options.aperture_radius,
            psf_fwhm: options.psf_fwhm,
            min_flux_snr: options.min_flux_snr,
            min_psf_support_pixels: options.min_psf_support_pixels,
            refine_local_background: true,
            ..MeasurementOptions::default()
        },
        threshold_sigma: options.threshold_sigma,
        min_distance: options.min_distance,
        background_box_size: options.background_box_size,
        background_sample_limit: options.background_sample_limit,
        reject_linear_artifacts: true,
        proposal_mode: options.proposal_mode.clone(),
        ..DetectionOptions::default()
    };
    detection::detect_sources(
        &frame.data,
        Some(&mask),
        frame.height,
        frame.width,
        &detector,
        None,
    )
}

fn run(paths: &[PathBuf], reverse_csv: &Path, options: &Options) -> Result<AuditResult> {
    if paths.is_empty() {
        return Err("frame_paths must not be empty".into());
    }
    if options.threshold_sigma <= 0.0
        || options.min_distance < 1
        || options.aperture_radius < 1
        || options.psf_fwhm <= 0.0
    {
        return Err("detector parameters must be positive".into());
    }
    if options.background_box_size < 16
        || options.background_sample_limit < 1
        || options.min_flux_snr <= 0.0
    {
        return Err("background and quality parameters are invalid".into());
    }
    if !(1..=9).contains(&options.min_psf_support_pixels) {
        return Err("min_psf_support_pixels must be within 1..9".into());
    }
    if !["gaussian", "hybrid", "ensemble"].contains(&options.proposal_mode.as_str()) {
        return Err("proposal_mode must be gaussian, hybrid, or ensemble".into());
    }
    let reverse_by_frame = load_reverse_sources(reverse_csv, paths.len())?;
    let mut frame_rows = Vec::with_capacity(paths.len());
    let mut source_rows = Vec::new();
    let mut raw_evidence_layer_counts = BTreeMap::<String, usize>::new();
    let mut overlap_class_counts = BTreeMap::<String, usize>::new();
    for (index, path) in paths.iter().enumerate() {
        let frame = fits::read_fits(path, true)?;
        let detections = detect(frame, options)?;
        let candidate_detections = detections.sources;
        let quality_detections = candidate_detections
            .iter()
            .filter(|source| source.quality_passed)
            .cloned()
            .collect::<Vec<_>>();
        let (candidate_sources, candidate_tree) = summaries(&candidate_detections);
        let (quality_sources, quality_tree) = summaries(&quality_detections);
        frame_rows.push(FrameRow {
            frame_index: index + 1,
            frame_path: path.to_string_lossy().into_owned(),
            candidate_count: detections.candidate_count,
            returned_count: candidate_detections.len(),
            quality_count: detections.quality_count,
            reverse_quality_count: reverse_by_frame[index].len(),
        });
        for reverse in &reverse_by_frame[index] {
            *raw_evidence_layer_counts
                .entry(reverse.raw_evidence_layer.clone())
                .or_default() += 1;
            let row = source_row(
                index + 1,
                path,
                reverse,
                &candidate_sources,
                candidate_tree.as_ref(),
                &quality_sources,
                quality_tree.as_ref(),
            )?;
            *overlap_class_counts
                .entry(row.overlap_class.clone())
                .or_default() += 1;
            source_rows.push(row);
        }
    }
    let no_quality = source_rows
        .iter()
        .filter(|row| row.quality_count_r4 == 0)
        .count();
    let no_candidate = source_rows
        .iter()
        .filter(|row| row.overlap_class == "no_forward_candidate_within_4px")
        .count();
    let conclusion = format!(
        "{no_quality}/{} reverse quality sources have no forward quality neighbor within 4 px; {no_candidate}/{} have no forward candidate within 4 px. This is detector-level local overlap, not a source-identity match.",
        source_rows.len(), source_rows.len()
    );
    let parameters = json!({
        "threshold_sigma": options.threshold_sigma,
        "min_distance": options.min_distance,
        "aperture_radius": options.aperture_radius,
        "psf_fwhm": options.psf_fwhm,
        "background_box_size": options.background_box_size,
        "background_sample_limit": options.background_sample_limit,
        "min_flux_snr": options.min_flux_snr,
        "min_psf_support_pixels": options.min_psf_support_pixels,
        "proposal_mode": options.proposal_mode,
        "reject_linear_artifacts": true,
        "refine_local_background": true,
        "max_sources": Value::Null,
        "inner_radius_px": RADII[0],
        "match_radius_px": RADII[1],
        "report_radius_px": RADII[2],
        "reverse_source_csv": reverse_csv.to_string_lossy(),
    });
    Ok(AuditResult {
        frame_rows,
        source_rows,
        raw_evidence_layer_counts,
        overlap_class_counts,
        parameters,
        conclusion,
    })
}

fn collect_paths(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let entries = fs::read_dir(input).map_err(|error| error.to_string())?;
            let mut lowercase_fits = Vec::new();
            let mut uppercase_fits = Vec::new();
            for entry in entries {
                let path = entry.map_err(|error| error.to_string())?.path();
                let extension = path.extension().and_then(|extension| extension.to_str());
                match extension {
                    Some("fits") => lowercase_fits.push(path),
                    Some("FITS") => uppercase_fits.push(path),
                    _ => {}
                }
            }
            let by_name = |left: &PathBuf, right: &PathBuf| {
                left.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .cmp(&right.file_name().unwrap_or_default().to_string_lossy())
            };
            lowercase_fits.sort_by(by_name);
            uppercase_fits.sort_by(by_name);
            paths.extend(lowercase_fits);
            paths.extend(uppercase_fits);
        } else {
            paths.push(input.clone());
        }
    }
    let mut unique = BTreeMap::<String, usize>::new();
    let mut deduplicated = Vec::new();
    for path in paths {
        let resolved = path.canonicalize().unwrap_or_else(|_| path.clone());
        let identity: String = resolved.to_string_lossy().case_fold().collect();
        if let Some(index) = unique.get(&identity).copied() {
            deduplicated[index] = path;
        } else {
            unique.insert(identity, deduplicated.len());
            deduplicated.push(path);
        }
    }
    deduplicated.sort_by(|left, right| {
        left.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .cmp(&right.file_name().unwrap_or_default().to_string_lossy())
    });
    Ok(deduplicated)
}

fn take_value(args: &[String], index: &mut usize, name: &str) -> Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{name} requires a value"))
}

fn parse_args(args: &[String]) -> Result<CliArgs> {
    let mut parsed = CliArgs {
        inputs: Vec::new(),
        reverse_source_csv: PathBuf::new(),
        out_dir: PathBuf::from("tmp/signed-null-positive-overlap"),
        options: Options::default(),
    };
    let mut reverse_csv_seen = false;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--reverse-source-csv" || arg.starts_with("--reverse-source-csv=") {
            let value = if let Some(value) = arg.strip_prefix("--reverse-source-csv=") {
                value.to_owned()
            } else {
                take_value(args, &mut index, "--reverse-source-csv")?
            };
            parsed.reverse_source_csv = PathBuf::from(value);
            reverse_csv_seen = true;
        } else if arg == "--out-dir" || arg.starts_with("--out-dir=") {
            let value = if let Some(value) = arg.strip_prefix("--out-dir=") {
                value.to_owned()
            } else {
                take_value(args, &mut index, "--out-dir")?
            };
            parsed.out_dir = PathBuf::from(value);
        } else if let Some(value) = arg.strip_prefix("--threshold-sigma=") {
            parsed.options.threshold_sigma =
                value.parse().map_err(|_| "invalid --threshold-sigma")?;
        } else if arg == "--threshold-sigma" {
            parsed.options.threshold_sigma = take_value(args, &mut index, arg)?
                .parse()
                .map_err(|_| "invalid --threshold-sigma")?;
        } else if let Some(value) = arg.strip_prefix("--min-distance=") {
            parsed.options.min_distance = value.parse().map_err(|_| "invalid --min-distance")?;
        } else if arg == "--min-distance" {
            parsed.options.min_distance = take_value(args, &mut index, arg)?
                .parse()
                .map_err(|_| "invalid --min-distance")?;
        } else if let Some(value) = arg.strip_prefix("--aperture-radius=") {
            parsed.options.aperture_radius =
                value.parse().map_err(|_| "invalid --aperture-radius")?;
        } else if arg == "--aperture-radius" {
            parsed.options.aperture_radius = take_value(args, &mut index, arg)?
                .parse()
                .map_err(|_| "invalid --aperture-radius")?;
        } else if let Some(value) = arg.strip_prefix("--psf-fwhm=") {
            parsed.options.psf_fwhm = value.parse().map_err(|_| "invalid --psf-fwhm")?;
        } else if arg == "--psf-fwhm" {
            parsed.options.psf_fwhm = take_value(args, &mut index, arg)?
                .parse()
                .map_err(|_| "invalid --psf-fwhm")?;
        } else if let Some(value) = arg.strip_prefix("--background-box-size=") {
            parsed.options.background_box_size =
                value.parse().map_err(|_| "invalid --background-box-size")?;
        } else if arg == "--background-box-size" {
            parsed.options.background_box_size = take_value(args, &mut index, arg)?
                .parse()
                .map_err(|_| "invalid --background-box-size")?;
        } else if let Some(value) = arg.strip_prefix("--background-sample-limit=") {
            parsed.options.background_sample_limit = value
                .parse()
                .map_err(|_| "invalid --background-sample-limit")?;
        } else if arg == "--background-sample-limit" {
            parsed.options.background_sample_limit = take_value(args, &mut index, arg)?
                .parse()
                .map_err(|_| "invalid --background-sample-limit")?;
        } else if let Some(value) = arg.strip_prefix("--min-flux-snr=") {
            parsed.options.min_flux_snr = value.parse().map_err(|_| "invalid --min-flux-snr")?;
        } else if arg == "--min-flux-snr" {
            parsed.options.min_flux_snr = take_value(args, &mut index, arg)?
                .parse()
                .map_err(|_| "invalid --min-flux-snr")?;
        } else if let Some(value) = arg.strip_prefix("--min-psf-support-pixels=") {
            parsed.options.min_psf_support_pixels = value
                .parse()
                .map_err(|_| "invalid --min-psf-support-pixels")?;
        } else if arg == "--min-psf-support-pixels" {
            parsed.options.min_psf_support_pixels = take_value(args, &mut index, arg)?
                .parse()
                .map_err(|_| "invalid --min-psf-support-pixels")?;
        } else if let Some(value) = arg.strip_prefix("--proposal-mode=") {
            parsed.options.proposal_mode = value.to_owned();
        } else if arg == "--proposal-mode" {
            parsed.options.proposal_mode = take_value(args, &mut index, arg)?;
        } else if arg.starts_with('-') {
            return Err(format!("unrecognized argument: {arg}"));
        } else {
            parsed.inputs.push(PathBuf::from(arg));
        }
        index += 1;
    }
    if !reverse_csv_seen {
        return Err("--reverse-source-csv is required".into());
    }
    if parsed.inputs.is_empty() {
        return Err("at least one FITS file or directory is required".into());
    }
    parsed.inputs = collect_paths(&parsed.inputs)?;
    if parsed.inputs.is_empty() {
        return Err("no FITS files found".into());
    }
    Ok(parsed)
}

fn execute(args: &[String]) -> Result<(AuditResult, PathBuf)> {
    let parsed = parse_args(args)?;
    let result = run(&parsed.inputs, &parsed.reverse_source_csv, &parsed.options)?;
    write_artifacts(&result, &parsed.out_dir)?;
    Ok((result, parsed.out_dir))
}

fn write_artifacts(result: &AuditResult, out_dir: &Path) -> Result<()> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let csv_path = out_dir.join("signed_null_forward_overlap.csv");
    if result.source_rows.is_empty() {
        fs::write(&csv_path, "\n").map_err(|error| error.to_string())?;
    } else {
        let mut writer = csv::WriterBuilder::new()
            .terminator(Terminator::CRLF)
            .from_path(&csv_path)
            .map_err(|error| error.to_string())?;
        for row in &result.source_rows {
            writer.serialize(row).map_err(|error| error.to_string())?;
        }
        writer.flush().map_err(|error| error.to_string())?;
    }
    let mut payload = result.as_dict();
    payload["outputs"] = json!([
        "signed_null_forward_overlap.csv",
        "signed_null_forward_overlap_summary.json"
    ]);
    let summary_path = out_dir.join("signed_null_forward_overlap_summary.json");
    let file = fs::File::create(&summary_path).map_err(|error| error.to_string())?;
    serde_json::to_writer_pretty(file, &payload).map_err(|error| error.to_string())?;
    Ok(())
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("逐帧查找符号反相质量源在原图正向检测中的局部近邻\n\nUsage: rst19-signed-null-overlap <FITS|DIR>... --reverse-source-csv CSV [--threshold-sigma N] [--min-distance N] [--aperture-radius N] [--psf-fwhm N] [--background-box-size N] [--background-sample-limit N] [--min-flux-snr N] [--min-psf-support-pixels N] [--proposal-mode gaussian|hybrid|ensemble] [--out-dir DIR]");
        return Ok(());
    }
    let (result, output) = execute(args)?;
    println!("signed-null forward overlap: {}", output.display());
    println!("{}", result.conclusion);
    println!(
        "overlap_classes={}",
        serde_json::to_string(&result.overlap_class_counts).map_err(|error| error.to_string())?
    );
    Ok(())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let (result, output) = execute(args)?;
    Ok(json!({
        "tool": "rst19-signed-null-overlap",
        "out_dir": output,
        "result_path": output.join("signed_null_forward_overlap_summary.json"),
        "result": result.as_dict(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(id: i64, x: f64, y: f64, flags: &[&str], quality_passed: bool) -> Detection {
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
            flux_error: None,
            flux_snr: Some(10.0),
            filter_snr: Some(20.0),
            fwhm_x: None,
            fwhm_y: None,
            ellipticity: None,
            sharpness: None,
            footprint_pixels: Some(8),
            psf_support_pixels: Some(5),
            quality_passed,
            peak_x: Some(x.round()),
            peak_y: Some(y.round()),
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
        }
    }

    #[test]
    fn feature_class_uses_the_same_flag_priority_as_the_python_baseline() {
        let source = detection(1, 0.0, 0.0, &["ELONGATED", "SPIKE", "LINE_ARTIFACT"], false);
        assert_eq!(classify_source_feature(&source), "linear_artifact");
        assert_eq!(
            classify_source_feature(&detection(2, 0.0, 0.0, &[], true)),
            "compact_quality"
        );
        assert_eq!(
            classify_source_feature(&detection(3, 0.0, 0.0, &[], false)),
            "other_rejected"
        );
    }

    #[test]
    fn local_overlap_keeps_candidates_distinct_from_quality_neighbors() {
        let candidate = SourceSummary::from_detection(&detection(1, 10.8, 10.0, &["SPIKE"], false));
        let candidates = vec![candidate];
        let points = candidates
            .iter()
            .map(|source| [source.x, source.y])
            .collect::<Vec<_>>();
        let tree = KdTree::new(&points).unwrap();
        let reverse = ReverseSource {
            frame_index: 1,
            detection_id: 101,
            x: 10.0,
            y: 10.0,
            raw_evidence_layer: "raw_negative_anomaly".into(),
            flux_snr: Some(20.0),
        };
        let row = source_row(
            1,
            Path::new("frame.fits"),
            &reverse,
            &candidates,
            Some(&tree),
            &[],
            None,
        )
        .unwrap();
        assert_eq!(row.candidate_count_r2, 1);
        assert_eq!(row.quality_count_r4, 0);
        assert_eq!(row.overlap_class, "candidate_only_within_2px");
        assert_eq!(
            row.nearest_candidate_feature_class.as_deref(),
            Some("spike_or_support")
        );
    }

    #[test]
    fn parser_preserves_default_detection_parameters_and_repeatable_inputs() {
        let args = vec![
            "frame-a.fits".into(),
            "frame-b.fits".into(),
            "--reverse-source-csv".into(),
            "reverse.csv".into(),
        ];
        let parsed = parse_args(&args).unwrap();
        assert_eq!(parsed.inputs.len(), 2);
        assert_eq!(parsed.options.min_distance, 4);
        assert_eq!(parsed.options.psf_fwhm, 2.0);
        assert_eq!(parsed.options.proposal_mode, "hybrid");
    }
}
