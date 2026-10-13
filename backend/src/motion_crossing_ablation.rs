//! Deterministic association-stage crossing experiment using frozen detections.
//!
//! This mirrors `rst19.motion_crossing_ablation` without image or Python
//! dependencies. The experiment compares the native ordinary tracker with its
//! additive high-speed point-motion supplement; truth remains synthetic input,
//! not an identity claim about physical objects.

use crate::{
    measurement::Detection,
    sequence::{self, TrackSequenceOptions},
    sequence_fast::{self, FastPointOptions},
    Result,
};
use flate2::{Compression, GzBuilder};
use png::{BitDepth, ColorType, Encoder};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const FRAME_COUNT: usize = 15;
const MIN_PRESENCE: usize = 12;
const LINK_RADIUS_PX: f64 = 4.0;
const REGISTRATION_RADIUS_PX: f64 = 8.0;
const MOTION_MIN_DISPLACEMENT_PX: f64 = 2.0;
const BASELINE_MAX_FIT_RMS_PX: f64 = 0.75;
const SUPPLEMENT_MIN_FLUX_SNR: f64 = 7.0;
const SUPPLEMENT_MAX_STEP_PX: f64 = 20.0;
const SUPPLEMENT_PREDICTION_GATE_PX: f64 = 4.5;
const SUPPLEMENT_MAX_FIT_RMS_PX: f64 = 1.5;
const SCENARIO_SLOW: &str = "slow_linked_identity_swap";
const SCENARIO_FAST: &str = "fast_orthogonal_supplement";

#[derive(Clone, Copy)]
struct TargetSpec {
    target_id: &'static str,
    detection_id: i64,
    start_x: f64,
    start_y: f64,
    velocity_x: f64,
    velocity_y: f64,
}

#[derive(Clone, Copy)]
struct ScenarioSpec {
    name: &'static str,
    targets: [TargetSpec; 2],
}

const SCENARIOS: [ScenarioSpec; 2] = [
    ScenarioSpec {
        name: SCENARIO_SLOW,
        targets: [
            TargetSpec {
                target_id: "SLOW-CROSS-A",
                detection_id: 1_000_001,
                start_x: 1500.0,
                start_y: 1800.0,
                velocity_x: 4.0,
                velocity_y: 0.0,
            },
            TargetSpec {
                target_id: "SLOW-CROSS-B",
                detection_id: 1_000_002,
                start_x: 1552.0,
                start_y: 1800.0,
                velocity_x: -4.0,
                velocity_y: 0.0,
            },
        ],
    },
    ScenarioSpec {
        name: SCENARIO_FAST,
        targets: [
            TargetSpec {
                target_id: "FAST-CROSS-A",
                detection_id: 1_000_011,
                start_x: 1948.0,
                start_y: 2000.0,
                velocity_x: 8.0,
                velocity_y: 0.0,
            },
            TargetSpec {
                target_id: "FAST-CROSS-B",
                detection_id: 1_000_012,
                start_x: 2000.0,
                start_y: 1948.0,
                velocity_x: 0.0,
                velocity_y: 8.0,
            },
        ],
    },
];

const FROZEN_FIELDS: [&str; 16] = [
    "frame_index",
    "detection_id",
    "x_px",
    "y_px",
    "peak_adu",
    "flux_adu",
    "background_adu",
    "noise_adu",
    "snr",
    "fwhm_px",
    "flags_json",
    "flux_error",
    "flux_snr",
    "filter_snr",
    "quality_passed",
    "truth_target_id",
];

const SUMMARY_FIELDS: [&str; 19] = [
    "scenario",
    "version",
    "input_sha256",
    "target_count",
    "truth_observations_total",
    "truth_observations_covered_unique",
    "truth_observation_assignments_in_tracks",
    "duplicated_truth_observation_assignments",
    "targets_with_all_observations_covered",
    "targets_with_full_clean_track_present",
    "targets_with_exclusive_full_track",
    "target_track_assignments_fragment_count_sum",
    "identity_switch_count",
    "mixed_truth_track_count",
    "background_contaminated_target_track_count",
    "predicted_track_count",
    "registration_max_shift_px",
    "runtime_seconds",
    "supplement_track_count",
];

const TRACK_FIELDS: [&str; 20] = [
    "scenario",
    "version",
    "output_track_id",
    "origin_arm",
    "source_track_id",
    "classification",
    "evidence_level",
    "point_count",
    "truth_target_ids_json",
    "full_clean_target_ids_json",
    "truth_observation_count",
    "background_observation_count",
    "identity_switch_count",
    "mixed_truth_ids",
    "background_contaminated",
    "frame_indices_csv",
    "detection_ids_csv",
    "displacement_px",
    "speed_px_per_frame",
    "fit_rms_px",
];

const TARGET_FIELDS: [&str; 11] = [
    "scenario",
    "version",
    "target_id",
    "expected_observations",
    "covered_observations",
    "observation_coverage_fraction",
    "predicted_tracks_containing_target",
    "full_clean_track_count",
    "full_clean_track_present",
    "exclusive_full_track",
    "identity_switches_on_containing_tracks",
];

const POINT_FIELDS: [&str; 11] = [
    "scenario",
    "version",
    "output_track_id",
    "origin_arm",
    "frame_index",
    "detection_id",
    "truth_target_id",
    "x_px",
    "y_px",
    "aligned_x_px",
    "aligned_y_px",
];

#[derive(Clone)]
struct TrackRecord {
    output_track_id: String,
    origin_arm: String,
    track: Value,
    points: Vec<Value>,
}

struct VariantRows {
    summary: Vec<String>,
    tracks: Vec<Vec<String>>,
    targets: Vec<Vec<String>>,
    points: Vec<Vec<String>>,
}

struct VariantContext<'a> {
    scenario: &'a str,
    version: &'a str,
    truth_rows: &'a [Value],
    input_sha256: &'a str,
    registration_max_shift_px: f64,
    runtime_seconds: f64,
    supplement_track_count: usize,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fmt9(value: f64) -> String {
    format!("{value:.9}")
}

fn json_string(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".into())
}

fn anchors() -> Vec<(f64, f64)> {
    [120.0, 320.0, 520.0, 720.0]
        .into_iter()
        .flat_map(|y| [120.0, 320.0, 520.0, 720.0, 920.0, 1120.0].map(move |x| (x, y)))
        .collect()
}

fn make_detection(detection_id: i64, x: f64, y: f64) -> Detection {
    Detection {
        detection_id,
        x,
        y,
        peak: 100.0,
        flux: 500.0,
        background: 10.0,
        noise: 2.0,
        snr: 20.0,
        fwhm: Some(2.0),
        flags: Vec::new(),
        flux_error: Some(20.0),
        flux_snr: Some(20.0),
        filter_snr: Some(20.0),
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
        proposal_methods: Vec::new(),
        proposal_scales: Vec::new(),
        proposal_snr: None,
        nearest_gaussian_px: None,
        deblend_delta_bic: None,
        deblend_component_snr: None,
        repeated_code_count: None,
        range_anomaly_pixel_count: None,
        repeated_code_values: Vec::new(),
    }
}

fn build_scenario(spec: ScenarioSpec) -> (Vec<Vec<Detection>>, Vec<Value>) {
    let anchor_points = anchors();
    let mut frames = Vec::with_capacity(FRAME_COUNT);
    let mut truth = Vec::with_capacity(FRAME_COUNT * spec.targets.len());
    for frame_index in 0..FRAME_COUNT {
        let mut sources = anchor_points
            .iter()
            .enumerate()
            .map(|(index, (x, y))| make_detection(index as i64, *x, *y))
            .collect::<Vec<_>>();
        for target in spec.targets {
            let x = target.start_x + frame_index as f64 * target.velocity_x;
            let y = target.start_y + frame_index as f64 * target.velocity_y;
            sources.push(make_detection(target.detection_id, x, y));
            truth.push(json!({
                "scenario":spec.name,
                "target_id":target.target_id,
                "detection_id":target.detection_id,
                "frame_index":frame_index,
                "x_px":fmt9(x),
                "y_px":fmt9(y),
                "role":"known synthetic crossing trajectory"
            }));
        }
        sources.sort_by_key(|source| source.detection_id);
        frames.push(sources);
    }
    (frames, truth)
}

fn row_for_detection(
    frame_index: usize,
    source: &Detection,
    truth_target: Option<&str>,
) -> Vec<String> {
    vec![
        frame_index.to_string(),
        source.detection_id.to_string(),
        fmt9(source.x),
        fmt9(source.y),
        fmt9(source.peak),
        fmt9(source.flux),
        fmt9(source.background),
        fmt9(source.noise),
        fmt9(source.snr),
        source.fwhm.map(fmt9).unwrap_or_default(),
        "[]".into(),
        source.flux_error.map(fmt9).unwrap_or_default(),
        source.flux_snr.map(fmt9).unwrap_or_default(),
        source.filter_snr.map(fmt9).unwrap_or_default(),
        "1".into(),
        truth_target.unwrap_or_default().to_owned(),
    ]
}

fn csv_bytes(headers: &[&str], rows: &[Vec<String>]) -> Result<Vec<u8>> {
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .terminator(csv::Terminator::Any(b'\n'))
        .from_writer(Vec::new());
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(row)
            .map_err(|error| error.to_string())?;
    }
    writer.into_inner().map_err(|error| error.to_string())
}

fn write_csv(path: &Path, headers: &[&str], rows: &[Vec<String>]) -> Result<()> {
    let mut bytes = b"\xef\xbb\xbf".to_vec();
    bytes.extend(csv_bytes(headers, rows)?);
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn frozen_csv(frames: &[Vec<Detection>], spec: ScenarioSpec) -> Result<Vec<u8>> {
    let targets: HashMap<i64, &str> = spec
        .targets
        .iter()
        .map(|target| (target.detection_id, target.target_id))
        .collect();
    let rows = frames
        .iter()
        .enumerate()
        .flat_map(|(frame_index, sources)| {
            let targets = &targets;
            sources.iter().map(move |source| {
                row_for_detection(
                    frame_index,
                    source,
                    targets.get(&source.detection_id).copied(),
                )
            })
        })
        .collect::<Vec<_>>();
    csv_bytes(&FROZEN_FIELDS, &rows)
}

fn gzip_bytes(payload: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = GzBuilder::new()
        .mtime(0)
        .write(Vec::new(), Compression::new(6));
    encoder
        .write_all(payload)
        .map_err(|error| error.to_string())?;
    encoder.finish().map_err(|error| error.to_string())
}

fn fingerprint_frames(frames: &[Vec<Detection>]) -> String {
    let rows = frames
        .iter()
        .enumerate()
        .flat_map(|(frame_index, sources)| {
            sources.iter().map(move |source| {
                json!([
                    frame_index,
                    source.detection_id,
                    source.x,
                    source.y,
                    source.peak,
                    source.flux,
                    source.background,
                    source.noise,
                    source.snr,
                    source.fwhm,
                    source.flags,
                    source.flux_error,
                    source.flux_snr,
                    source.filter_snr,
                    source.quality_passed
                ])
            })
        })
        .collect::<Vec<_>>();
    sha256(serde_json::to_string(&rows).unwrap_or_default().as_bytes())
}

fn text_field<'a>(value: &'a Value, name: &str) -> &'a str {
    value.get(name).and_then(Value::as_str).unwrap_or("")
}

fn integer_field(value: &Value, name: &str) -> i64 {
    value.get(name).and_then(Value::as_i64).unwrap_or_default()
}

fn float_field(value: &Value, name: &str) -> f64 {
    value.get(name).and_then(Value::as_f64).unwrap_or_default()
}

fn point_key(point: &Value) -> (usize, i64) {
    (
        point["frame_index"].as_u64().unwrap_or_default() as usize,
        integer_field(point, "detection_id"),
    )
}

fn point_label(point: &Value, truth: &HashMap<(usize, i64), String>) -> String {
    truth.get(&point_key(point)).cloned().unwrap_or_default()
}

fn track_records(tracks: &[Value], origin: &str) -> Vec<TrackRecord> {
    tracks
        .iter()
        .enumerate()
        .map(|(index, track)| {
            let mut points = track["points"].as_array().cloned().unwrap_or_default();
            points.sort_by_key(point_key);
            TrackRecord {
                output_track_id: format!("{origin}-{index:04}"),
                origin_arm: origin.to_owned(),
                track: track.clone(),
                points,
            }
        })
        .collect()
}

fn score_variant(records: &[TrackRecord], context: VariantContext<'_>) -> VariantRows {
    let VariantContext {
        scenario,
        version,
        truth_rows,
        input_sha256,
        registration_max_shift_px,
        runtime_seconds,
        supplement_track_count,
    } = context;
    let mut truth_by_key = HashMap::new();
    let mut expected_by_target: BTreeMap<String, HashSet<(usize, i64)>> = BTreeMap::new();
    for row in truth_rows {
        let target_id = text_field(row, "target_id").to_owned();
        let key = (
            row["frame_index"].as_u64().unwrap_or_default() as usize,
            integer_field(row, "detection_id"),
        );
        truth_by_key.insert(key, target_id.clone());
        expected_by_target.entry(target_id).or_default().insert(key);
    }

    let mut track_rows = Vec::new();
    let mut point_rows = Vec::new();
    let mut track_keys_by_target: BTreeMap<String, Vec<HashSet<(usize, i64)>>> = BTreeMap::new();
    let mut covered_by_target: BTreeMap<String, HashSet<(usize, i64)>> = BTreeMap::new();
    let mut switches_by_track = HashMap::new();
    let mut mixed_track_count = 0usize;
    let mut background_contamination_count = 0usize;
    let mut assigned_truth_observations = 0usize;
    let mut assigned_truth_keys = HashSet::new();

    for record in records {
        let labels = record
            .points
            .iter()
            .map(|point| point_label(point, &truth_by_key))
            .collect::<Vec<_>>();
        let target_labels = labels
            .iter()
            .filter(|label| !label.is_empty())
            .cloned()
            .collect::<Vec<_>>();
        let unique_targets = target_labels.iter().cloned().collect::<BTreeSet<_>>();
        let identity_switches = target_labels
            .windows(2)
            .filter(|pair| pair[0] != pair[1])
            .count();
        switches_by_track.insert(record.output_track_id.clone(), identity_switches);
        let is_mixed = unique_targets.len() > 1;
        let has_background = labels.iter().any(String::is_empty);
        mixed_track_count += usize::from(is_mixed);
        background_contamination_count += usize::from(!unique_targets.is_empty() && has_background);

        let mut track_keys = HashSet::new();
        for (point, label) in record.points.iter().zip(&labels) {
            let key = point_key(point);
            track_keys.insert(key);
            if !label.is_empty() {
                covered_by_target
                    .entry(label.clone())
                    .or_default()
                    .insert(key);
                assigned_truth_keys.insert(key);
                assigned_truth_observations += 1;
            }
            point_rows.push(vec![
                scenario.to_owned(),
                version.to_owned(),
                record.output_track_id.clone(),
                record.origin_arm.clone(),
                key.0.to_string(),
                key.1.to_string(),
                label.clone(),
                fmt9(float_field(point, "x")),
                fmt9(float_field(point, "y")),
                fmt9(float_field(point, "aligned_x")),
                fmt9(float_field(point, "aligned_y")),
            ]);
        }

        let mut full_clean_targets = Vec::new();
        for (target_id, expected_keys) in &expected_by_target {
            let target_keys = track_keys
                .iter()
                .filter(|key| truth_by_key.get(key) == Some(target_id))
                .copied()
                .collect::<HashSet<_>>();
            if !target_keys.is_empty() {
                track_keys_by_target
                    .entry(target_id.clone())
                    .or_default()
                    .push(target_keys.clone());
            }
            if &target_keys == expected_keys && record.points.len() == expected_keys.len() {
                full_clean_targets.push(target_id.clone());
            }
        }
        let frame_indices = record
            .points
            .iter()
            .map(|point| point_key(point).0.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let detection_ids = record
            .points
            .iter()
            .map(|point| point_key(point).1.to_string())
            .collect::<Vec<_>>()
            .join(",");
        track_rows.push(vec![
            scenario.to_owned(),
            version.to_owned(),
            record.output_track_id.clone(),
            record.origin_arm.clone(),
            integer_field(&record.track, "track_id").to_string(),
            text_field(&record.track, "classification").to_owned(),
            text_field(&record.track, "evidence_level").to_owned(),
            record.points.len().to_string(),
            json_string(&json!(unique_targets.iter().collect::<Vec<_>>())),
            json_string(&json!(full_clean_targets)),
            target_labels.len().to_string(),
            (labels.len() - target_labels.len()).to_string(),
            identity_switches.to_string(),
            usize::from(is_mixed).to_string(),
            usize::from(!unique_targets.is_empty() && has_background).to_string(),
            frame_indices,
            detection_ids,
            fmt9(float_field(&record.track, "displacement_px")),
            fmt9(float_field(&record.track, "speed_px_per_frame")),
            record.track["fit_rms_px"]
                .as_f64()
                .map(fmt9)
                .unwrap_or_default(),
        ]);
    }

    let expected_total = expected_by_target.values().map(HashSet::len).sum::<usize>();
    let mut target_rows = Vec::new();
    let mut total_fragments = 0usize;
    let mut full_clean_present_count = 0usize;
    let mut exclusive_full_count = 0usize;
    let mut covered_all_count = 0usize;
    for (target_id, expected_keys) in &expected_by_target {
        let fragments = track_keys_by_target.get(target_id).map_or(0, Vec::len);
        let clean_count = records
            .iter()
            .filter(|record| {
                let keys = record.points.iter().map(point_key).collect::<HashSet<_>>();
                keys == *expected_keys && record.points.len() == expected_keys.len()
            })
            .count();
        let covered = covered_by_target
            .get(target_id)
            .map_or(0, |keys| keys.intersection(expected_keys).count());
        let clean_present = clean_count > 0;
        let exclusive = clean_count == 1 && fragments == 1;
        total_fragments += fragments;
        full_clean_present_count += usize::from(clean_present);
        exclusive_full_count += usize::from(exclusive);
        covered_all_count += usize::from(covered == expected_keys.len());
        let target_switches = records
            .iter()
            .filter(|record| {
                record.points.iter().any(|point| {
                    truth_by_key
                        .get(&point_key(point))
                        .is_some_and(|label| label == target_id)
                })
            })
            .map(|record| switches_by_track[&record.output_track_id])
            .sum::<usize>();
        target_rows.push(vec![
            scenario.to_owned(),
            version.to_owned(),
            target_id.clone(),
            expected_keys.len().to_string(),
            covered.to_string(),
            format!("{:.6}", covered as f64 / expected_keys.len() as f64),
            fragments.to_string(),
            clean_count.to_string(),
            usize::from(clean_present).to_string(),
            usize::from(exclusive).to_string(),
            target_switches.to_string(),
        ]);
    }
    point_rows.sort_by(|left, right| {
        (
            left[3].as_str(),
            left[2].as_str(),
            left[4].parse::<usize>().unwrap_or(0),
        )
            .cmp(&(
                right[3].as_str(),
                right[2].as_str(),
                right[4].parse::<usize>().unwrap_or(0),
            ))
    });

    let summary = vec![
        scenario.to_owned(),
        version.to_owned(),
        input_sha256.to_owned(),
        expected_by_target.len().to_string(),
        expected_total.to_string(),
        assigned_truth_keys.len().to_string(),
        assigned_truth_observations.to_string(),
        (assigned_truth_observations - assigned_truth_keys.len()).to_string(),
        covered_all_count.to_string(),
        full_clean_present_count.to_string(),
        exclusive_full_count.to_string(),
        total_fragments.to_string(),
        switches_by_track.values().sum::<usize>().to_string(),
        mixed_track_count.to_string(),
        background_contamination_count.to_string(),
        records.len().to_string(),
        fmt9(registration_max_shift_px),
        fmt9(runtime_seconds),
        supplement_track_count.to_string(),
    ];
    VariantRows {
        summary,
        tracks: track_rows,
        targets: target_rows,
        points: point_rows,
    }
}

fn csv_objects(path: &Path, headers: &[&str], rows: &[Value]) -> Result<()> {
    let mut records = Vec::with_capacity(rows.len());
    for row in rows {
        let object = row.as_object().ok_or("CSV row must be an object")?;
        records.push(
            headers
                .iter()
                .map(|header| match object.get(*header).unwrap_or(&Value::Null) {
                    Value::String(text) => text.clone(),
                    Value::Null => String::new(),
                    value => value.to_string(),
                })
                .collect::<Vec<_>>(),
        );
    }
    write_csv(path, headers, &records)
}

fn set_pixel(image: &mut [u8], width: usize, height: usize, x: i32, y: i32, color: [u8; 3]) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }
    let offset = (y as usize * width + x as usize) * 3;
    image[offset..offset + 3].copy_from_slice(&color);
}

fn draw_line(
    image: &mut [u8],
    width: usize,
    height: usize,
    from: (i32, i32),
    to: (i32, i32),
    color: [u8; 3],
    thickness: i32,
) {
    let (mut x0, mut y0) = from;
    let (x1, y1) = to;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        for oy in -thickness / 2..=thickness / 2 {
            for ox in -thickness / 2..=thickness / 2 {
                set_pixel(image, width, height, x0 + ox, y0 + oy, color);
            }
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let doubled = 2 * error;
        if doubled >= dy {
            error += dy;
            x0 += sx;
        }
        if doubled <= dx {
            error += dx;
            y0 += sy;
        }
    }
}

fn draw_circle(
    image: &mut [u8],
    width: usize,
    height: usize,
    center: (i32, i32),
    radius: i32,
    color: [u8; 3],
) {
    for y in -radius..=radius {
        for x in -radius..=radius {
            if x * x + y * y <= radius * radius {
                set_pixel(image, width, height, center.0 + x, center.1 + y, color);
            }
        }
    }
}

fn draw_panel(
    image: &mut [u8],
    width: usize,
    height: usize,
    panel_index: usize,
    scenario: &str,
    truth_rows: &[Value],
    point_rows: &[Vec<String>],
) {
    let gap = 22usize;
    let panel_width = (width - 3 * gap) / 2;
    let left = gap + panel_index * (panel_width + gap);
    let top = gap;
    let right = left + panel_width;
    let bottom = height - gap;
    for y in top..bottom {
        for x in left..right {
            let edge = x == left || x + 1 == right || y == top || y + 1 == bottom;
            set_pixel(
                image,
                width,
                height,
                x as i32,
                y as i32,
                if edge {
                    [205, 215, 228]
                } else {
                    [255, 255, 255]
                },
            );
        }
    }
    let selected_truth = truth_rows
        .iter()
        .filter(|row| text_field(row, "scenario") == scenario)
        .collect::<Vec<_>>();
    let mut groups: BTreeMap<String, Vec<(usize, f64, f64)>> = BTreeMap::new();
    for row in selected_truth {
        groups
            .entry(text_field(row, "target_id").into())
            .or_default()
            .push((
                row["frame_index"].as_u64().unwrap_or_default() as usize,
                text_field(row, "x_px").parse().unwrap_or(0.0),
                text_field(row, "y_px").parse().unwrap_or(0.0),
            ));
    }
    for values in groups.values_mut() {
        values.sort_by_key(|value| value.0);
    }
    let all_coordinates = groups.values().flatten().collect::<Vec<_>>();
    if all_coordinates.is_empty() {
        return;
    }
    let min_x = all_coordinates
        .iter()
        .map(|value| value.1)
        .fold(f64::INFINITY, f64::min)
        - 12.0;
    let max_x = all_coordinates
        .iter()
        .map(|value| value.1)
        .fold(f64::NEG_INFINITY, f64::max)
        + 12.0;
    let min_y = all_coordinates
        .iter()
        .map(|value| value.2)
        .fold(f64::INFINITY, f64::min)
        - 12.0;
    let max_y = all_coordinates
        .iter()
        .map(|value| value.2)
        .fold(f64::NEG_INFINITY, f64::max)
        + 12.0;
    let plot_left = left as f64 + 38.0;
    let plot_top = top as f64 + 70.0;
    let plot_width = panel_width as f64 - 60.0;
    let plot_height = height as f64 - 130.0;
    let scale = (plot_width / (max_x - min_x).max(1.0)).min(plot_height / (max_y - min_y).max(1.0));
    let offset_x = plot_left + (plot_width - (max_x - min_x) * scale) / 2.0;
    let offset_y = plot_top + (plot_height - (max_y - min_y) * scale) / 2.0;
    let map_point = |x: f64, y: f64| {
        (
            (offset_x + (x - min_x) * scale).round() as i32,
            (offset_y + (y - min_y) * scale).round() as i32,
        )
    };
    let truth_colors = [[245, 174, 54], [171, 84, 199]];
    for (target_index, points) in groups.values().enumerate() {
        let color = truth_colors[target_index % truth_colors.len()];
        let pixels = points
            .iter()
            .map(|point| map_point(point.1, point.2))
            .collect::<Vec<_>>();
        for pair in pixels.windows(2) {
            draw_line(image, width, height, pair[0], pair[1], color, 3);
        }
        for point in pixels {
            draw_circle(image, width, height, point, 2, color);
        }
    }
    for arm in ["A", "B_supplement"] {
        let mut tracks: BTreeMap<String, Vec<(usize, f64, f64)>> = BTreeMap::new();
        for row in point_rows
            .iter()
            .filter(|row| row[0] == scenario && row[3] == arm && !row[6].is_empty())
        {
            tracks.entry(row[2].clone()).or_default().push((
                row[4].parse().unwrap_or(0),
                row[7].parse().unwrap_or(0.0),
                row[8].parse().unwrap_or(0.0),
            ));
        }
        let arm_color = if arm == "A" {
            [207, 77, 77]
        } else {
            [38, 150, 138]
        };
        for (index, track) in tracks.values_mut().enumerate() {
            track.sort_by_key(|point| point.0);
            let points = track
                .iter()
                .map(|point| map_point(point.1, point.2))
                .collect::<Vec<_>>();
            let color = if arm == "A" {
                [
                    [213, 72, 79],
                    [55, 113, 199],
                    [142, 79, 171],
                    [199, 119, 45],
                ][index % 4]
            } else {
                arm_color
            };
            for pair in points.windows(2) {
                draw_line(image, width, height, pair[0], pair[1], color, 2);
            }
            for point in points {
                draw_circle(
                    image,
                    width,
                    height,
                    point,
                    if arm == "A" { 3 } else { 2 },
                    color,
                );
            }
        }
    }
}

fn write_plot(path: &Path, truth_rows: &[Value], point_rows: &[Vec<String>]) -> Result<()> {
    let width = 1160usize;
    let height = 500usize;
    let mut image = vec![0u8; width * height * 3];
    for pixel in image.chunks_mut(3) {
        pixel.copy_from_slice(&[239, 243, 248]);
    }
    draw_panel(
        &mut image,
        width,
        height,
        0,
        SCENARIO_SLOW,
        truth_rows,
        point_rows,
    );
    draw_panel(
        &mut image,
        width,
        height,
        1,
        SCENARIO_FAST,
        truth_rows,
        point_rows,
    );
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = Encoder::new(BufWriter::new(file), width as u32, height as u32);
    encoder.set_color(ColorType::Rgb);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
    writer
        .write_image_data(&image)
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn utc_iso_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = (seconds / 86_400) as i64;
    let day_seconds = seconds % 86_400;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        day_seconds / 3_600,
        (day_seconds % 3_600) / 60,
        day_seconds % 60
    )
}

fn resolved_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(path))
    }
}

fn write_artifact_manifest(paths: &BTreeMap<String, PathBuf>) -> Result<Value> {
    let mut outputs = serde_json::Map::new();
    for (name, path) in paths {
        let bytes = fs::read(path).map_err(|error| error.to_string())?;
        outputs.insert(name.clone(), json!({"path":path,"sha256":sha256(&bytes)}));
    }
    Ok(Value::Object(outputs))
}

/// Run the frozen slow/fast crossing experiment and write its audit artifacts.
pub fn run(out_dir: &Path) -> Result<Value> {
    let output = resolved_path(out_dir)?;
    if output.is_dir()
        && fs::read_dir(&output)
            .map_err(|error| error.to_string())?
            .next()
            .is_some()
    {
        return Err(format!(
            "refusing to overwrite non-empty experiment directory: {}",
            output.display()
        ));
    }
    fs::create_dir_all(&output).map_err(|error| error.to_string())?;
    let frozen_dir = output.join("frozen_inputs");
    fs::create_dir_all(&frozen_dir).map_err(|error| error.to_string())?;

    let mut summary_rows = Vec::new();
    let mut track_rows = Vec::new();
    let mut target_rows = Vec::new();
    let mut point_rows = Vec::new();
    let mut truth_rows_all = Vec::new();
    let mut input_records = serde_json::Map::new();
    let mut runtime_by_scenario = serde_json::Map::new();
    let mut registration_by_scenario = serde_json::Map::new();
    let anchor_points = anchors();

    for scenario in SCENARIOS {
        let (frames, truth_rows) = build_scenario(scenario);
        truth_rows_all.extend(truth_rows.iter().cloned());
        let frozen_path = frozen_dir.join(format!("{}.csv.gz", scenario.name));
        let frozen_bytes = gzip_bytes(&frozen_csv(&frames, scenario)?)?;
        fs::write(&frozen_path, &frozen_bytes).map_err(|error| error.to_string())?;
        let frozen_hash = sha256(&frozen_bytes);
        let before_fingerprint = fingerprint_frames(&frames);
        let source_count_by_frame = frames.iter().map(Vec::len).collect::<Vec<_>>();
        let frame_names = (0..FRAME_COUNT)
            .map(|index| format!("synthetic:{}:{index:02}", scenario.name))
            .collect::<Vec<_>>();
        let frame_times = (0..FRAME_COUNT)
            .map(|index| index as f64)
            .collect::<Vec<_>>();

        let started = Instant::now();
        let baseline = sequence::track_detections(
            &frames,
            &TrackSequenceOptions {
                frame_names: Some(frame_names),
                frame_times: Some(frame_times.clone()),
                link_radius_px: LINK_RADIUS_PX,
                min_presence: Some(MIN_PRESENCE),
                motion_min_displacement_px: MOTION_MIN_DISPLACEMENT_PX,
                registration_radius_px: REGISTRATION_RADIUS_PX,
                max_motion_fit_rms_px: BASELINE_MAX_FIT_RMS_PX,
                persistent_min_presence: None,
            },
        )?;
        let baseline_seconds = started.elapsed().as_secs_f64();
        let fingerprint_before_supplement = fingerprint_frames(&frames);
        if fingerprint_before_supplement != before_fingerprint {
            return Err(format!(
                "baseline association mutated frozen Detection input for {}",
                scenario.name
            ));
        }
        let baseline_tracks = baseline["tracks"].as_array().cloned().unwrap_or_default();
        let shifts = baseline["cumulative_shifts"]
            .as_array()
            .ok_or("baseline tracker omitted cumulative_shifts")?
            .iter()
            .map(|shift| {
                let values = shift.as_array().ok_or("invalid cumulative shift")?;
                Ok::<[f64; 2], String>([
                    values[0].as_f64().ok_or("invalid cumulative shift x")?,
                    values[1].as_f64().ok_or("invalid cumulative shift y")?,
                ])
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let started = Instant::now();
        let supplement = sequence_fast::track_fast_point_movers(
            &frames,
            &shifts,
            Some(&frame_times),
            &FastPointOptions {
                link_radius_px: LINK_RADIUS_PX,
                min_presence: Some(MIN_PRESENCE),
                motion_min_displacement_px: MOTION_MIN_DISPLACEMENT_PX,
                min_flux_snr: SUPPLEMENT_MIN_FLUX_SNR,
                max_step_px: SUPPLEMENT_MAX_STEP_PX,
                gate_px: SUPPLEMENT_PREDICTION_GATE_PX,
                max_fit_rms_px: SUPPLEMENT_MAX_FIT_RMS_PX,
            },
            &baseline_tracks,
        )?;
        let supplement_seconds = started.elapsed().as_secs_f64();
        let fingerprint_after_supplement = fingerprint_frames(&frames);
        if fingerprint_after_supplement != before_fingerprint {
            return Err(format!(
                "supplement association mutated frozen Detection input for {}",
                scenario.name
            ));
        }
        let registration_max_shift = shifts
            .iter()
            .map(|shift| shift[0].hypot(shift[1]))
            .fold(0.0f64, f64::max);
        if registration_max_shift > 1.0e-9 {
            return Err(format!(
                "static anchors failed to hold registration at zero for {}",
                scenario.name
            ));
        }

        let baseline_records = track_records(&baseline_tracks, "A");
        let supplement_tracks = supplement["tracks"].as_array().cloned().unwrap_or_default();
        let supplement_records = track_records(&supplement_tracks, "B_supplement");
        let mut union_records = baseline_records.clone();
        union_records.extend(supplement_records.iter().cloned());
        let input_path = frozen_path.to_string_lossy().into_owned();
        input_records.insert(
            scenario.name.into(),
            json!({
                "path":input_path,
                "sha256":frozen_hash,
                "detection_count":source_count_by_frame.iter().sum::<usize>(),
                "detection_count_by_frame":source_count_by_frame,
                "same_input_sha256_for_A_and_B":true,
                "A_input_fingerprint":before_fingerprint,
                "B_input_fingerprint":fingerprint_before_supplement,
                "post_B_input_fingerprint":fingerprint_after_supplement,
                "A_and_B_same_Detection_list":true
            }),
        );
        registration_by_scenario.insert(
            scenario.name.into(),
            json!({
                "cumulative_shifts_px":shifts,
                "max_shift_px":registration_max_shift,
                "zero_verified":true,
                "static_anchor_count_per_frame":anchor_points.len()
            }),
        );
        runtime_by_scenario.insert(
            scenario.name.into(),
            json!({
                "A_baseline_seconds":baseline_seconds,
                "B_supplement_seconds":supplement_seconds
            }),
        );

        let variants = [
            ("A", baseline_records.as_slice(), baseline_seconds),
            (
                "B_supplement",
                supplement_records.as_slice(),
                supplement_seconds,
            ),
            (
                "B_union",
                union_records.as_slice(),
                baseline_seconds + supplement_seconds,
            ),
        ];
        for (version, records, seconds) in variants {
            let rows = score_variant(
                records,
                VariantContext {
                    scenario: scenario.name,
                    version,
                    truth_rows: &truth_rows,
                    input_sha256: &frozen_hash,
                    registration_max_shift_px: registration_max_shift,
                    runtime_seconds: seconds,
                    supplement_track_count: supplement_records.len(),
                },
            );
            summary_rows.push(rows.summary);
            track_rows.extend(rows.tracks);
            target_rows.extend(rows.targets);
            point_rows.extend(rows.points);
        }
    }

    let truth_path = output.join("truth.csv");
    let truth_headers = [
        "scenario",
        "target_id",
        "detection_id",
        "frame_index",
        "x_px",
        "y_px",
        "role",
    ];
    csv_objects(&truth_path, &truth_headers, &truth_rows_all)?;
    let summary_path = output.join("summary.csv");
    write_csv(&summary_path, &SUMMARY_FIELDS, &summary_rows)?;
    let track_path = output.join("track_level.csv");
    write_csv(&track_path, &TRACK_FIELDS, &track_rows)?;
    let target_path = output.join("target_summary.csv");
    write_csv(&target_path, &TARGET_FIELDS, &target_rows)?;
    let points_path = output.join("track_points.csv");
    write_csv(&points_path, &POINT_FIELDS, &point_rows)?;
    let plot_path = output.join("motion_crossing_tracks.png");
    write_plot(&plot_path, &truth_rows_all, &point_rows)?;

    let mut artifacts = BTreeMap::from([
        ("truth_csv".into(), truth_path),
        ("summary_csv".into(), summary_path.clone()),
        ("track_level_csv".into(), track_path),
        ("target_summary_csv".into(), target_path),
        ("track_points_csv".into(), points_path),
        ("motion_crossing_plot".into(), plot_path),
    ]);
    for scenario in SCENARIOS {
        artifacts.insert(
            format!("frozen_input_{}", scenario.name),
            frozen_dir.join(format!("{}.csv.gz", scenario.name)),
        );
    }
    let artifact_manifest = write_artifact_manifest(&artifacts)?;
    let scenarios_payload = SCENARIOS
        .iter()
        .map(|scenario| {
            let targets = scenario
                .targets
                .iter()
                .map(|target| {
                    json!({
                        "target_id":target.target_id,
                        "detection_id":target.detection_id,
                        "start_registered_xy_px":[target.start_x,target.start_y],
                        "velocity_px_per_frame":[target.velocity_x,target.velocity_y],
                        "expected_frames":(0..FRAME_COUNT).collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>();
            (
                scenario.name.to_owned(),
                json!({
                    "targets":targets,
                    "input":input_records[scenario.name],
                    "registration":registration_by_scenario[scenario.name],
                    "runtime_seconds":runtime_by_scenario[scenario.name]
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let manifest = json!({
        "schema_version":1,
        "experiment":"synthetic_crossing_track_association_ablation",
        "status":"completed",
        "created_local_time":utc_iso_timestamp(),
        "scope":"association-stage synthetic Detection experiment; no image detector or physical identity is tested",
        "frame_count":FRAME_COUNT,
        "frame_times_seconds":(0..FRAME_COUNT).map(|index| index as f64).collect::<Vec<_>>(),
        "scenarios":scenarios_payload,
        "static_anchors":{
            "count_per_frame":anchor_points.len(),
            "positions_px":anchor_points,
            "purpose":"hold the estimated global translation at exactly zero while crossing detections are associated"
        },
        "versions":[
            {"id":"A","definition":"track_detections output with fixed link/registration parameters"},
            {"id":"B_supplement","definition":"track_fast_point_movers on the same frozen Detection list; A tracks are passed as existing_tracks"},
            {"id":"B_union","definition":"A tracks concatenated with B_supplement tracks; duplicate observations are retained and counted"}
        ],
        "parameters":{
            "link_radius_px":LINK_RADIUS_PX,
            "registration_radius_px":REGISTRATION_RADIUS_PX,
            "min_presence_frames":MIN_PRESENCE,
            "motion_min_displacement_px":MOTION_MIN_DISPLACEMENT_PX,
            "baseline_max_motion_fit_rms_px":BASELINE_MAX_FIT_RMS_PX,
            "supplement_min_flux_snr":SUPPLEMENT_MIN_FLUX_SNR,
            "supplement_max_step_px_per_frame":SUPPLEMENT_MAX_STEP_PX,
            "supplement_prediction_gate_px":SUPPLEMENT_PREDICTION_GATE_PX,
            "supplement_max_fit_rms_px":SUPPLEMENT_MAX_FIT_RMS_PX,
            "detection_dtype":"not applicable; this is a frozen Detection-record experiment without pixel arrays",
            "anchor_detection_snr":20.0,
            "target_detection_snr":20.0
        },
        "metric_definitions":{
            "identity_switch":"adjacent truth-labelled observations in a predicted track have different target IDs",
            "mixed_truth_track":"one predicted track contains detections from more than one injected target ID",
            "fragment_count":"number of output tracks containing at least one observation for the target; B_union retains A and supplement outputs",
            "full_clean_track_present":"at least one output track contains exactly every expected target observation and no other observation",
            "exclusive_full_track":"exactly one target-containing track exists and it is a full clean track",
            "duplicate_observation_assignment":"the same (frame_index, detection_id) appears in multiple output tracks",
            "truth_observation_coverage":"unique expected (frame_index, detection_id) keys represented in output tracks divided by expected keys"
        },
        "limitations":[
            "All Detection records are synthetic and quality_passed; no FITS pixels, detector proposals, photometry uncertainty, or quality gate are exercised.",
            "The two crossing scenarios establish deterministic behavior of these association functions only; they do not establish physical identity or sky-wide precision.",
            "B is an additive supplement and cannot replace or correct any A track; B_union intentionally preserves duplicate baseline singleton assignments.",
            "The cases use two clean targets and static anchors; crowded backgrounds, noise, missing frames, and detector misses are not part of this crossing experiment."
        ],
        "outputs":artifact_manifest,
        "execution":{
            "runtime":"Rust",
            "version":env!("CARGO_PKG_VERSION"),
            "target_os":std::env::consts::OS,
            "target_arch":std::env::consts::ARCH,
            "source_revision":option_env!("RST19_SOURCE_REVISION")
        }
    });
    let manifest_path = output.join("manifest.json");
    fs::write(
        &manifest_path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&manifest).map_err(|error| error.to_string())?
        ),
    )
    .map_err(|error| error.to_string())?;
    Ok(json!({
        "status":"completed",
        "run_dir":output,
        "manifest":manifest_path,
        "summary":summary_path
    }))
}

fn parse_args(args: &[String]) -> Result<PathBuf> {
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline) = arg.strip_prefix("--").map_or(("", None), |rest| {
            rest.split_once('=')
                .map_or((rest, None), |(name, value)| (name, Some(value)))
        });
        if name != "out-dir" {
            return Err(if name.is_empty() {
                "unexpected positional argument".into()
            } else {
                format!("unknown option --{name}")
            });
        }
        let value = if let Some(value) = inline {
            value.to_owned()
        } else {
            index += 1;
            args.get(index)
                .cloned()
                .ok_or("--out-dir requires a value")?
        };
        if output.replace(PathBuf::from(value)).is_some() {
            return Err("--out-dir may be provided only once".into());
        }
        index += 1;
    }
    Ok(output.unwrap_or_else(|| PathBuf::from("tmp/ablation/motion-crossing-native")))
}

/// Entry point used by the Rust desktop worker.
pub fn run_desktop(args: &[String]) -> Result<Value> {
    let output = parse_args(args)?;
    run(&output)
}

/// Standalone CLI entry point. Returns process-style status for the binary wrapper.
pub fn run_cli(args: &[String]) -> i32 {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("冻结两种合成交叉轨迹 Detection 输入，比较基础关联与速度预测补充关联\n\nUsage: rst19-motion-crossing-ablation [--out-dir DIR]");
        return 0;
    }
    match run_desktop(args) {
        Ok(result) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&result).unwrap_or_default()
            );
            0
        }
        Err(error) => {
            eprintln!("rst19-motion-crossing-ablation: failed: {error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEST_DIR: AtomicUsize = AtomicUsize::new(0);

    fn read_csv_rows(path: &Path) -> Vec<Value> {
        let mut bytes = fs::read(path).unwrap();
        if bytes.starts_with(b"\xef\xbb\xbf") {
            bytes.drain(..3);
        }
        let mut reader = csv::ReaderBuilder::new().from_reader(bytes.as_slice());
        let headers = reader.headers().unwrap().clone();
        reader
            .records()
            .map(|record| {
                let record = record.unwrap();
                let mut row = serde_json::Map::new();
                for (header, value) in headers.iter().zip(record.iter()) {
                    row.insert(header.to_owned(), json!(value));
                }
                Value::Object(row)
            })
            .collect()
    }

    #[test]
    fn crossing_experiment_matches_frozen_python_track_and_summary_rows() {
        let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
        let output = std::env::temp_dir().join(format!(
            "rst19-crossing-native-{}-{sequence}",
            std::process::id()
        ));
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/motion_crossing_parity.json"
        ))
        .unwrap();
        let result = run(&output).unwrap();
        assert_eq!(result["status"], "completed");

        for name in [
            "truth.csv",
            "summary.csv",
            "track_level.csv",
            "target_summary.csv",
            "track_points.csv",
        ] {
            let mut rows = read_csv_rows(&output.join(name));
            if name == "summary.csv" {
                for row in &mut rows {
                    row.as_object_mut().unwrap().remove("runtime_seconds");
                    row.as_object_mut().unwrap().remove("input_sha256");
                }
            } else if name == "track_level.csv" {
                rows.retain(|row| row["truth_target_ids_json"] != "[]");
            } else if name == "track_points.csv" {
                rows.retain(|row| !row["truth_target_id"].as_str().unwrap().is_empty());
            }
            assert_eq!(
                rows,
                fixture["files"][name].as_array().unwrap().to_vec(),
                "frozen Python rows differ for {name}"
            );
        }

        let manifest: Value =
            serde_json::from_slice(&fs::read(output.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["execution"]["runtime"], "Rust");
        assert_eq!(
            manifest["scenarios"][SCENARIO_SLOW]["registration"]["zero_verified"],
            true
        );
        assert_eq!(
            manifest["scenarios"][SCENARIO_FAST]["registration"]["zero_verified"],
            true
        );
        assert_eq!(manifest["outputs"].as_object().unwrap().len(), 8);
        let compressed = fs::read(
            output
                .join("frozen_inputs")
                .join(format!("{SCENARIO_SLOW}.csv.gz")),
        )
        .unwrap();
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut frozen_csv = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut frozen_csv).unwrap();
        assert!(frozen_csv.starts_with(b"frame_index,detection_id,x_px,y_px"));
        let png = fs::read(output.join("motion_crossing_tracks.png")).unwrap();
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), 1160);
        assert_eq!(u32::from_be_bytes(png[20..24].try_into().unwrap()), 500);
        fs::remove_dir_all(output).unwrap();
    }
}
