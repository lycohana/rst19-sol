//! Native FITS-to-tracks orchestration for the sequence core.
//!
//! This module exposes the migrated detection and quality-track stages
//! separately from temporal candidate recovery, trail analysis, and relative
//! photometry, which remain later sequence stages.
use crate::{
    detection::{self, BackgroundModel},
    fits,
    pipeline::{self, FrameOptions},
    sequence::{self, TrackSequenceOptions},
    sequence_trails, Result,
};
use serde_json::{json, Map, Value};
use std::path::PathBuf;

const DEFAULT_SEQUENCE_MAX_SOURCES: usize = 6_000;
const DEFAULT_SEQUENCE_BACKGROUND_BOX_SIZE: usize = 256;
const DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT: usize = 4_096;

fn object_or_empty(value: Option<&Value>, name: &str) -> Result<Map<String, Value>> {
    match value {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(object)) => Ok(object.clone()),
        _ => Err(format!("{name} must be an object")),
    }
}

fn sequence_frame_options(request: &Value) -> Result<FrameOptions> {
    let mut options = object_or_empty(request.get("frame_options"), "frame_options")?;
    options
        .entry("background_box_size")
        .or_insert(json!(DEFAULT_SEQUENCE_BACKGROUND_BOX_SIZE));
    options
        .entry("refine_local_background")
        .or_insert(json!(false));
    options.entry("use_float32").or_insert(json!(true));
    let use_float32 = options
        .get("use_float32")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    options
        .entry("background_sample_limit")
        .or_insert(json!(if use_float32 {
            DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT
        } else {
            100_000
        }));
    options.entry("fast_sequence").or_insert(json!(use_float32));
    if options.get("max_sources").is_none_or(Value::is_null) {
        let max_sources = match request.get("sequence_max_sources") {
            Some(value) => value.clone(),
            None => json!(DEFAULT_SEQUENCE_MAX_SOURCES),
        };
        options.insert("max_sources".into(), max_sources);
    }
    serde_json::from_value(Value::Object(options)).map_err(|error| error.to_string())
}

/// Handle `sequence.analyze_core` JSONL requests.
pub fn handle_request(request: &Value) -> Result<Value> {
    let paths: Vec<PathBuf> = serde_json::from_value(
        request
            .get("paths")
            .cloned()
            .ok_or("missing sequence paths")?,
    )
    .map_err(|error| error.to_string())?;
    if paths.is_empty() {
        return Err("sequence paths cannot be empty".into());
    }
    let frame_options = sequence_frame_options(request)?;
    let mut frames = Vec::with_capacity(paths.len());
    let mut frame_analyses = Vec::with_capacity(paths.len());
    let mut quality_frames = Vec::with_capacity(paths.len());

    let first_frame = fits::read_fits(&paths[0], false)?;
    let shared_background = if paths.len() >= 2
        && frame_options.detection.fast_sequence
        && frame_options.detection.use_float32
    {
        let mask = fits::auxiliary_mask(first_frame.height, first_frame.width)?;
        let (background, noise) = detection::build_background_model(
            &first_frame.data,
            Some(&mask),
            first_frame.height,
            first_frame.width,
            &frame_options.detection,
        )?;
        Some((background, noise, first_frame.height, first_frame.width))
    } else {
        None
    };
    frames.push(first_frame);
    for path in paths.iter().skip(1) {
        frames.push(fits::read_fits(path, false)?);
    }

    let headers = frames
        .iter()
        .map(|frame| Value::Object(frame.header.clone()))
        .collect::<Vec<_>>();
    let time_axis = sequence_trails::observation_time_axis(&json!({"headers":headers}))?;
    let frame_times: Vec<f64> = serde_json::from_value(time_axis["frame_times"].clone())
        .map_err(|error| error.to_string())?;

    for frame in frames {
        let background = shared_background
            .as_ref()
            .map(|(background, noise, height, width)| BackgroundModel {
                background,
                noise,
                height: *height,
                width: *width,
            });
        let analysis =
            pipeline::analyze_frame(frame, None, None, &frame_options, background.as_ref())?;
        let detections = analysis
            .detection
            .sources
            .iter()
            .filter(|source| source.quality_passed)
            .cloned()
            .collect::<Vec<_>>();
        quality_frames.push(detections);
        frame_analyses.push(analysis.as_dict());
    }

    let mut tracking_options =
        object_or_empty(request.get("tracking_options"), "tracking_options")?;
    if tracking_options
        .get("frame_names")
        .is_none_or(Value::is_null)
    {
        tracking_options.insert(
            "frame_names".into(),
            json!(paths
                .iter()
                .map(|path| path.to_string_lossy())
                .collect::<Vec<_>>()),
        );
    }
    if tracking_options
        .get("frame_times")
        .is_none_or(Value::is_null)
    {
        tracking_options.insert("frame_times".into(), json!(frame_times));
    }
    let tracking_options: TrackSequenceOptions =
        serde_json::from_value(Value::Object(tracking_options))
            .map_err(|error| error.to_string())?;
    let tracking = sequence::track_detections(&quality_frames, &tracking_options)?;

    Ok(json!({
        "schema":"rst19.sequence.core.v1",
        "scope":"frame_detection_and_quality_track_association",
        "frame_analyses":frame_analyses,
        "tracking":tracking,
        "observation_time_axis":time_axis,
        "detector_options":serde_json::to_value(&frame_options.detection).map_err(|error| error.to_string())?,
        "shared_background_model":shared_background.is_some(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_detector_defaults_match_the_python_fast_sequence_contract() {
        let options = sequence_frame_options(&json!({})).unwrap();
        assert_eq!(options.detection.max_sources, Some(6_000));
        assert_eq!(options.detection.background_box_size, 256);
        assert_eq!(options.detection.background_sample_limit, 4_096);
        assert!(options.detection.use_float32);
        assert!(options.detection.fast_sequence);
        assert!(!options.detection.measurement.refine_local_background);
    }

    #[test]
    fn sequence_detector_can_disable_the_working_limit_and_fast_float32_path() {
        let options = sequence_frame_options(&json!({
            "sequence_max_sources": null,
            "frame_options": {"use_float32": false}
        }))
        .unwrap();
        assert_eq!(options.detection.max_sources, None);
        assert_eq!(options.detection.background_sample_limit, 100_000);
        assert!(!options.detection.use_float32);
        assert!(!options.detection.fast_sequence);
    }

    #[test]
    fn explicit_detector_source_limit_takes_precedence() {
        let options = sequence_frame_options(&json!({
            "sequence_max_sources": 8,
            "frame_options": {"max_sources": 12}
        }))
        .unwrap();
        assert_eq!(options.detection.max_sources, Some(12));
    }

    #[test]
    fn empty_sequence_is_rejected_before_any_files_are_opened() {
        assert_eq!(
            handle_request(&json!({"paths": []})).unwrap_err(),
            "sequence paths cannot be empty"
        );
    }
}
