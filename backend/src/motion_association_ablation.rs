//! Frozen-input association ablation, implemented without a Python runtime.
//!
//! The Python implementation remains the migration reference until full parity
//! fixtures are checked in; this module is the native production path.

use crate::{
    measurement::Detection,
    sequence::{self, TrackSequenceOptions},
    sequence_fast::{self, FastPointOptions},
    Result,
};
use flate2::{read::GzDecoder, Compression, GzBuilder};
use png::{BitDepth, ColorType, Encoder};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const TARGET_SPECS: [(&str, f64, f64, &[usize], &str); 4] = [
    ("FAST-15", 8.0, -4.0, &[], "fast positive, 15/15"),
    (
        "FAST-GAP-12",
        6.0,
        2.0,
        &[3, 7, 11],
        "fast positive with three separated gaps, 12/15",
    ),
    (
        "SLOW-BASE-15",
        2.0,
        0.5,
        &[],
        "slow positive control for ordinary association",
    ),
    (
        "BELOW-12",
        -7.0,
        -3.0,
        &[2, 5, 8, 11],
        "presence negative control, 11/15",
    ),
];

#[derive(Debug, Clone)]
pub struct MotionAblationOptions {
    pub seed: i64,
    pub min_presence: Option<usize>,
    pub link_radius_px: Option<f64>,
    pub registration_radius_px: f64,
    pub min_flux_snr: f64,
    pub max_step_px: f64,
    pub prediction_gate_px: f64,
    pub max_fit_rms_px: f64,
    pub argv: Vec<String>,
}

impl Default for MotionAblationOptions {
    fn default() -> Self {
        Self {
            seed: 19019,
            min_presence: None,
            link_radius_px: None,
            registration_radius_px: 8.0,
            min_flux_snr: 7.0,
            max_step_px: 20.0,
            prediction_gate_px: 4.5,
            max_fit_rms_px: 1.5,
            argv: Vec::new(),
        }
    }
}

struct FrozenCache {
    envelope: Value,
    sequence: Value,
    frames: Vec<Value>,
    frame_paths: Vec<PathBuf>,
    identities: Vec<Value>,
    background: Vec<Vec<Detection>>,
    frame_times: Vec<f64>,
    cached_shifts: Vec<[f64; 2]>,
    registered_background: Vec<Vec<[f64; 2]>>,
    quality_point_count: usize,
}

fn fail(message: impl Into<String>) -> String {
    message.into()
}

fn json_i64(value: &Value, field: &str) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| fail(format!("{field} must be an integer")))
}

fn json_f64(value: &Value, field: &str) -> Result<f64> {
    let number = value
        .as_f64()
        .ok_or_else(|| fail(format!("{field} must be a number")))?;
    if !number.is_finite() {
        return Err(fail(format!("{field} must be finite")));
    }
    Ok(number)
}

fn json_string<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .as_str()
        .ok_or_else(|| fail(format!("{field} must be a string")))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn display_path(path: &Path) -> String {
    let value = path.to_string_lossy();
    #[cfg(windows)]
    {
        if let Some(unc) = value.strip_prefix("\\\\?\\UNC\\") {
            return format!("\\\\{unc}");
        }
        if let Some(path) = value.strip_prefix("\\\\?\\") {
            return path.to_owned();
        }
    }
    value.into_owned()
}

fn point_source(detection_id: i64, x: f64, y: f64, flux_snr: f64) -> Detection {
    Detection {
        detection_id,
        x,
        y,
        peak: 100.0,
        flux: 500.0,
        background: 10.0,
        noise: 2.0,
        snr: flux_snr,
        fwhm: Some(2.0),
        flags: Vec::new(),
        flux_error: Some(20.0),
        flux_snr: Some(flux_snr),
        filter_snr: Some(flux_snr),
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

/// Python's `random.Random` uses MT19937 with `init_by_array` for integer seeds.
/// Keeping this small implementation local makes truth placement byte-for-byte
/// reproducible across Windows, macOS, and Linux.
#[derive(Clone)]
struct PythonMt19937 {
    state: [u32; 624],
    index: usize,
}

impl PythonMt19937 {
    fn seeded(seed: i64) -> Self {
        let mut key = Vec::new();
        let mut value = seed.unsigned_abs();
        loop {
            key.push(value as u32);
            value >>= 32;
            if value == 0 {
                break;
            }
        }
        let mut random = Self {
            state: [0; 624],
            index: 624,
        };
        random.init_genrand(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;
        for _ in 0..624usize.max(key.len()) {
            let previous = random.state[i - 1];
            random.state[i] = (random.state[i]
                ^ ((previous ^ (previous >> 30)).wrapping_mul(1_664_525)))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                random.state[0] = random.state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            let previous = random.state[i - 1];
            random.state[i] = (random.state[i]
                ^ ((previous ^ (previous >> 30)).wrapping_mul(1_566_083_941)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                random.state[0] = random.state[623];
                i = 1;
            }
        }
        random.state[0] = 0x8000_0000;
        random.index = 624;
        random
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..624 {
            self.state[i] = 1_812_433_253u32
                .wrapping_mul(self.state[i - 1] ^ (self.state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.index = 624;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn random(&mut self) -> f64 {
        let high = (self.next_u32() >> 5) as u64;
        let low = (self.next_u32() >> 6) as u64;
        ((high << 26) + low) as f64 / 9_007_199_254_740_992.0
    }

    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.random()
    }
}

fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = month as i64 + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day as i64 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn parse_timestamp_micros(value: &str) -> Result<i128> {
    let (date, time_zone) = value
        .split_once('T')
        .or_else(|| value.split_once(' '))
        .ok_or_else(|| fail(format!("invalid ISO timestamp: {value}")))?;
    let mut date_parts = date.split('-');
    let year = date_parts
        .next()
        .and_then(|v| v.parse::<i64>().ok())
        .ok_or("invalid timestamp year")?;
    let month = date_parts
        .next()
        .and_then(|v| v.parse::<u32>().ok())
        .ok_or("invalid timestamp month")?;
    let day = date_parts
        .next()
        .and_then(|v| v.parse::<u32>().ok())
        .ok_or("invalid timestamp day")?;
    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => 0,
    };
    if date_parts.next().is_some() || !(1..=9999).contains(&year) || day == 0 || day > days_in_month
    {
        return Err(fail(format!("invalid ISO timestamp date: {value}")));
    }
    let zone_at = time_zone
        .char_indices()
        .find(|(index, ch)| *index >= 8 && (*ch == '+' || *ch == '-' || *ch == 'Z' || *ch == 'z'))
        .map(|(index, _)| index);
    let (time, zone) = zone_at.map_or((time_zone, ""), |index| time_zone.split_at(index));
    let mut parts = time.split(':');
    let hour = parts
        .next()
        .and_then(|v| v.parse::<i64>().ok())
        .ok_or("invalid timestamp hour")?;
    let minute = parts
        .next()
        .and_then(|v| v.parse::<i64>().ok())
        .ok_or("invalid timestamp minute")?;
    let seconds = parts.next().ok_or("invalid timestamp seconds")?;
    if parts.next().is_some() || !(0..=23).contains(&hour) || !(0..=59).contains(&minute) {
        return Err(fail(format!("invalid ISO timestamp time: {value}")));
    }
    let (whole_seconds, fraction) = seconds.split_once('.').unwrap_or((seconds, ""));
    let second = whole_seconds
        .parse::<i64>()
        .map_err(|_| "invalid timestamp seconds")?;
    if !(0..=59).contains(&second) || !fraction.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(fail(format!("invalid ISO timestamp seconds: {value}")));
    }
    let fraction = format!("{fraction:0<6}");
    let micros = fraction
        .chars()
        .take(6)
        .collect::<String>()
        .parse::<i64>()
        .unwrap_or(0);
    let offset_seconds = if zone.is_empty() || zone == "Z" {
        0
    } else {
        let sign = if zone.starts_with('-') { -1 } else { 1 };
        let offset = zone.trim_start_matches(['+', '-']);
        let normalized = offset.replace(':', "");
        let (hour_text, minute_text) = match normalized.len() {
            2 => (&normalized[..2], "00"),
            4 => (&normalized[..2], &normalized[2..4]),
            _ => return Err(fail(format!("invalid ISO timestamp offset: {value}"))),
        };
        let hours = hour_text
            .parse::<i64>()
            .map_err(|_| "invalid timestamp offset hour")?;
        let minutes = minute_text
            .parse::<i64>()
            .map_err(|_| "invalid timestamp offset minute")?;
        if hours > 23 || minutes > 59 {
            return Err(fail(format!("invalid ISO timestamp offset: {value}")));
        }
        sign * (hours * 3600 + minutes * 60)
    };
    let days = days_from_civil(year, month, day) as i128;
    Ok(
        ((days * 86_400 + (hour * 3600 + minute * 60 + second - offset_seconds) as i128)
            * 1_000_000)
            + micros as i128,
    )
}

fn relative_frame_times(timestamps: &[Option<i128>]) -> Vec<f64> {
    let fallback = || {
        (0..timestamps.len())
            .map(|index| index as f64)
            .collect::<Vec<_>>()
    };
    let Some(parsed) = timestamps.iter().copied().collect::<Option<Vec<_>>>() else {
        return fallback();
    };
    if parsed.is_empty() || parsed.windows(2).any(|pair| pair[1] <= pair[0]) {
        return fallback();
    }
    let origin = parsed[0];
    parsed
        .iter()
        .map(|stamp| (*stamp - origin) as f64 / 1_000_000.0)
        .collect()
}

fn median(values: &[f64]) -> Result<f64> {
    if values.is_empty() {
        return Err("cannot take median of an empty list".into());
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        Ok((values[middle - 1] + values[middle]) / 2.0)
    } else {
        Ok(values[middle])
    }
}

fn load_frozen_cache(path: &Path) -> Result<FrozenCache> {
    eprintln!("motion-ablation: reading compressed sequence envelope");
    let file = File::open(path).map_err(|error| error.to_string())?;
    let envelope: Value =
        serde_json::from_reader(GzDecoder::new(file)).map_err(|error| error.to_string())?;
    eprintln!("motion-ablation: parsed sequence envelope");
    let sequence = envelope
        .get("sequence")
        .cloned()
        .ok_or("cache has no sequence summary")?;
    let frames = sequence["frames"]
        .as_array()
        .cloned()
        .ok_or("cache sequence has no frames")?;
    let identities = envelope["input_identity"]["frames"]
        .as_array()
        .cloned()
        .ok_or("cache has no input frame identities")?;
    if frames.len() < 3 || frames.len() != identities.len() {
        return Err(
            "sequence cache must contain matching frame summaries and input identities".into(),
        );
    }
    let mut frame_paths = Vec::with_capacity(frames.len());
    for identity in &identities {
        let raw_path = PathBuf::from(json_string(&identity["path"], "input identity path")?);
        let path = raw_path.canonicalize().map_err(|error| error.to_string())?;
        let expected = json_string(&identity["sha256"], "input identity sha256")?;
        let actual = sha256_file(&path)?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(format!("cached input SHA-256 mismatch: {}", path.display()));
        }
        frame_paths.push(path);
    }
    for (index, frame) in frames.iter().enumerate() {
        let path = PathBuf::from(json_string(&frame["path"], "frame path")?)
            .canonicalize()
            .map_err(|error| error.to_string())?;
        if path != frame_paths[index] {
            return Err("frame order differs between sequence cache and input identity".into());
        }
    }

    let mut background = vec![Vec::new(); frames.len()];
    let mut registered_background = vec![Vec::new(); frames.len()];
    let mut seen = vec![HashSet::new(); frames.len()];
    let mut quality_point_count = 0usize;
    let tracks = sequence["tracks"]
        .as_array()
        .ok_or("cache sequence has no tracks")?;
    for track in tracks {
        if track
            .get("evidence_level")
            .and_then(Value::as_str)
            .unwrap_or("quality")
            != "quality"
        {
            continue;
        }
        let points = track["points"]
            .as_array()
            .ok_or("cached track has no points")?;
        for point in points {
            let frame_index = json_i64(&point["frame_index"], "cached frame index")?;
            if frame_index < 0 || frame_index as usize >= frames.len() {
                return Err("cached track point has an invalid frame index".into());
            }
            let frame_index = frame_index as usize;
            let detection_id = json_i64(&point["detection_id"], "cached detection id")?;
            if !seen[frame_index].insert(detection_id) {
                return Err(format!(
                    "duplicate quality detection id in cached frame {frame_index}: {detection_id}"
                ));
            }
            if point.get("quality_passed").and_then(Value::as_bool) != Some(true) {
                return Err(
                    "cached quality track contains a point that failed its quality gate".into(),
                );
            }
            let flux_snr = json_f64(&point["flux_snr"], "cached flux SNR")?;
            let x = json_f64(&point["x"], "cached x")?;
            let y = json_f64(&point["y"], "cached y")?;
            background[frame_index].push(point_source(detection_id, x, y, flux_snr));
            registered_background[frame_index].push([
                json_f64(&point["aligned_x"], "cached aligned x")?,
                json_f64(&point["aligned_y"], "cached aligned y")?,
            ]);
            quality_point_count += 1;
        }
    }
    for (index, sources) in background.iter_mut().enumerate() {
        sources.sort_by_key(|source| source.detection_id);
        if sources.len()
            != json_i64(
                &frames[index]["quality_count"],
                "cached frame quality count",
            )? as usize
        {
            return Err(format!(
                "cached quality points for frame {index} ({}) do not match frame summary ({})",
                sources.len(),
                frames[index]["quality_count"]
            ));
        }
    }
    let shifts = sequence["cumulative_shifts"]
        .as_array()
        .ok_or("cache has no cumulative shifts")?;
    if shifts.len() != frames.len() {
        return Err("cached cumulative shift count differs from frame count".into());
    }
    let mut cached_shifts = Vec::with_capacity(frames.len());
    for shift in shifts {
        let values = shift.as_array().ok_or("cached shift must be a pair")?;
        if values.len() != 2 {
            return Err("cached shift must be a pair".into());
        }
        cached_shifts.push([
            json_f64(&values[0], "cached shift x")?,
            json_f64(&values[1], "cached shift y")?,
        ]);
    }
    let timestamps = frames
        .iter()
        .map(|frame| {
            frame["timestamp"]
                .as_str()
                .and_then(|stamp| parse_timestamp_micros(stamp).ok())
        })
        .collect::<Vec<_>>();
    let frame_times = relative_frame_times(&timestamps);
    Ok(FrozenCache {
        envelope,
        sequence,
        frames,
        frame_paths,
        identities,
        background,
        frame_times,
        cached_shifts,
        registered_background,
        quality_point_count,
    })
}

fn make_truth_paths(
    width: usize,
    height: usize,
    frame_times: &[f64],
    cached_shifts: &[[f64; 2]],
    background: &[Vec<[f64; 2]>],
    seed: i64,
) -> Result<Vec<Value>> {
    let frame_count = frame_times.len();
    if frame_count != cached_shifts.len() || frame_count != background.len() {
        return Err("truth trajectory inputs have inconsistent frame counts".into());
    }
    if frame_times.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err("truth trajectory times must be strictly increasing".into());
    }
    let deltas = frame_times
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let reference_interval = median(&deltas)?;
    let reference_steps = frame_times
        .iter()
        .map(|time| (time - frame_times[0]) / reference_interval)
        .collect::<Vec<_>>();
    let mut rng = PythonMt19937::seeded(seed);
    let mut accepted = Vec::<Value>::new();
    let mut positions_by_target = Vec::<Vec<[f64; 2]>>::new();
    for (target_index, (target_id, vx, vy, missing_frames, role)) in TARGET_SPECS.iter().enumerate()
    {
        let mut selected = None;
        for _attempt in 0..50_000 {
            let start_x = rng.uniform(180.0, width as f64 - 180.0);
            let start_y = rng.uniform(180.0, height as f64 - 180.0);
            let registered = reference_steps
                .iter()
                .map(|step| [start_x + vx * step, start_y + vy * step])
                .collect::<Vec<_>>();
            if registered.iter().enumerate().any(|(index, pos)| {
                !(60.0 <= pos[0] + cached_shifts[index][0]
                    && pos[0] + cached_shifts[index][0] <= width as f64 - 60.0
                    && 60.0 <= pos[1] + cached_shifts[index][1]
                    && pos[1] + cached_shifts[index][1] <= height as f64 - 60.0)
            }) {
                continue;
            }
            let contaminated = registered.iter().enumerate().any(|(index, pos)| {
                if missing_frames.contains(&index) {
                    return false;
                }
                background[index]
                    .iter()
                    .any(|point| (pos[0] - point[0]).hypot(pos[1] - point[1]) < 18.0)
            });
            if contaminated {
                continue;
            }
            let too_close = positions_by_target
                .iter()
                .enumerate()
                .any(|(prior_index, prior)| {
                    registered.iter().enumerate().any(|(index, pos)| {
                        !missing_frames.contains(&index)
                            && !TARGET_SPECS[prior_index].3.contains(&index)
                            && (pos[0] - prior[index][0]).hypot(pos[1] - prior[index][1]) < 50.0
                    })
                });
            if too_close {
                continue;
            }
            selected = Some(registered);
            break;
        }
        let registered =
            selected.ok_or_else(|| fail(format!("could not find clean path for {target_id}")))?;
        let detection_id = 1_000_000 + target_index as i64;
        let expected = registered
            .iter()
            .enumerate()
            .filter(|(frame, _)| !missing_frames.contains(frame))
            .map(|(frame_index, pos)| {
                json!({
                    "frame_index":frame_index,
                    "time_seconds":frame_times[frame_index],
                    "registered_x_px":pos[0],
                    "registered_y_px":pos[1],
                    "raw_x_px":pos[0] + cached_shifts[frame_index][0],
                    "raw_y_px":pos[1] + cached_shifts[frame_index][1],
                    "detection_id":detection_id
                })
            })
            .collect::<Vec<_>>();
        accepted.push(json!({
            "target_id":target_id,
            "vx_px_per_reference_interval":vx,
            "vy_px_per_reference_interval":vy,
            "missing_frames":missing_frames,
            "role":role,
            "detection_id":detection_id,
            "expected_presence":expected.len(),
            "registered_positions":registered,
            "expected_observations":expected,
            "flux_snr":20.0
        }));
        positions_by_target.push(registered);
    }
    Ok(accepted)
}

fn append_truth_sources(
    background: &[Vec<Detection>],
    truths: &[Value],
) -> Result<Vec<Vec<Detection>>> {
    let mut result = background.to_vec();
    for truth in truths {
        let detection_id = json_i64(&truth["detection_id"], "truth detection id")?;
        let flux_snr = json_f64(&truth["flux_snr"], "truth flux SNR")?;
        for point in truth["expected_observations"]
            .as_array()
            .ok_or("truth has no expected observations")?
        {
            let frame_index = json_i64(&point["frame_index"], "truth frame index")? as usize;
            let sources = result
                .get_mut(frame_index)
                .ok_or("truth observation frame is out of range")?;
            sources.push(point_source(
                detection_id,
                json_f64(&point["raw_x_px"], "truth raw x")?,
                json_f64(&point["raw_y_px"], "truth raw y")?,
                flux_snr,
            ));
        }
    }
    for sources in &mut result {
        sources.sort_by_key(|source| source.detection_id);
    }
    Ok(result)
}

#[derive(Default)]
struct TargetMetrics {
    track_count: usize,
    max_target_points: usize,
    covered_frames: usize,
    exact: bool,
    classification: String,
    mixed_target: bool,
    mixed_background: bool,
}

fn target_tracks(
    tracks: &[Value],
    target_id: i64,
    expected_frames: &HashSet<usize>,
) -> TargetMetrics {
    let containing = tracks
        .iter()
        .filter(|track| {
            track["points"].as_array().is_some_and(|points| {
                points
                    .iter()
                    .any(|point| point["detection_id"].as_i64() == Some(target_id))
            })
        })
        .collect::<Vec<_>>();
    let mut best: Option<(&Value, usize)> = None;
    let mut other_ids = HashSet::new();
    for track in &containing {
        let points = track["points"].as_array().map(Vec::as_slice).unwrap_or(&[]);
        let target_count = points
            .iter()
            .filter(|point| point["detection_id"].as_i64() == Some(target_id))
            .count();
        if best.is_none_or(|(_, previous)| target_count > previous) {
            best = Some((track, target_count));
        }
        for point in points {
            if let Some(id) = point["detection_id"].as_i64() {
                if id != target_id {
                    other_ids.insert(id);
                }
            }
        }
    }
    let Some((best_track, best_count)) = best else {
        return TargetMetrics::default();
    };
    let best_points = best_track["points"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut target_frames = HashSet::new();
    let mut target_point_count = 0usize;
    for point in best_points {
        if point["detection_id"].as_i64() == Some(target_id) {
            target_point_count += 1;
            if let Some(frame) = point["frame_index"].as_u64() {
                target_frames.insert(frame as usize);
            }
        }
    }
    let _ = best_count;
    let exact = target_frames == *expected_frames
        && target_point_count == expected_frames.len()
        && containing.len() == 1;
    TargetMetrics {
        track_count: containing.len(),
        max_target_points: target_point_count,
        covered_frames: target_frames.intersection(expected_frames).count(),
        exact,
        classification: best_track["classification"]
            .as_str()
            .unwrap_or("")
            .to_owned(),
        mixed_target: other_ids
            .iter()
            .any(|id| (1_000_000..1_000_100).contains(id)),
        mixed_background: other_ids.iter().any(|id| *id < 1_000_000),
    }
}

fn bool_text(value: bool) -> String {
    if value { "True" } else { "False" }.to_owned()
}

fn csv_bytes(headers: &[&str], rows: &[Vec<String>], bom: bool) -> Result<Vec<u8>> {
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(headers)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(row)
            .map_err(|error| error.to_string())?;
    }
    let mut bytes = writer.into_inner().map_err(|error| error.to_string())?;
    if bom {
        bytes.splice(0..0, [0xef, 0xbb, 0xbf]);
    }
    Ok(bytes)
}

fn write_gzip(path: &Path, bytes: &[u8], level: u32) -> Result<()> {
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = GzBuilder::new()
        .mtime(0)
        .write(file, Compression::new(level));
    encoder
        .write_all(bytes)
        .map_err(|error| error.to_string())?;
    encoder.finish().map_err(|error| error.to_string())?;
    Ok(())
}

fn track_array(value: &Value) -> Result<&[Value]> {
    value["tracks"]
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| "association result has no tracks".into())
}

fn make_summary_rows(
    truths: &[Value],
    baseline: &[Value],
    fast: &[Value],
    frames: usize,
) -> Result<(Vec<Value>, Vec<Value>)> {
    let mut summary = Vec::new();
    let mut pairing = Vec::new();
    for truth in truths {
        let target_id = json_i64(&truth["detection_id"], "truth detection id")?;
        let expected = truth["expected_observations"]
            .as_array()
            .ok_or("truth has no observations")?;
        let expected_frames = expected
            .iter()
            .map(|point| {
                json_i64(&point["frame_index"], "truth frame index").map(|value| value as usize)
            })
            .collect::<Result<HashSet<_>>>()?;
        let baseline_metrics = target_tracks(baseline, target_id, &expected_frames);
        let fast_metrics = target_tracks(fast, target_id, &expected_frames);
        let mut combined = baseline.to_vec();
        combined.extend_from_slice(fast);
        let combined_metrics = target_tracks(&combined, target_id, &expected_frames);
        let mut sorted_frames = expected_frames.iter().copied().collect::<Vec<_>>();
        sorted_frames.sort_unstable();
        summary.push(json!({
            "target_id":truth["target_id"],
            "role":truth["role"],
            "expected_presence":truth["expected_presence"],
            "sequence_frames":frames,
            "baseline_tracks_containing_target":baseline_metrics.track_count,
            "baseline_best_track_target_points":baseline_metrics.max_target_points,
            "baseline_target_frame_coverage":baseline_metrics.covered_frames,
            "baseline_exact_track_recovered":baseline_metrics.exact,
            "baseline_fragments":baseline_metrics.track_count,
            "baseline_mixed_with_other_injected_target":baseline_metrics.mixed_target,
            "baseline_mixed_with_background_source":baseline_metrics.mixed_background,
            "fast_supplement_tracks_containing_target":fast_metrics.track_count,
            "fast_best_track_target_points":fast_metrics.max_target_points,
            "fast_target_frame_coverage":fast_metrics.covered_frames,
            "fast_exact_track_recovered":fast_metrics.exact,
            "fast_track_classification":fast_metrics.classification,
            "fast_mixed_with_other_injected_target":fast_metrics.mixed_target,
            "fast_mixed_with_background_source":fast_metrics.mixed_background,
            "combined_single_track_only_exact":combined_metrics.exact,
            "expected_frame_indices":sorted_frames.iter().map(usize::to_string).collect::<Vec<_>>().join(","),
            "synthetic_detection_id":target_id
        }));
        pairing.push(json!({
            "target_id":truth["target_id"],
            "expected_presence":truth["expected_presence"],
            "baseline_exact_track_recovered":baseline_metrics.exact,
            "fast_exact_track_recovered":fast_metrics.exact,
            "combined_single_track_only_exact":combined_metrics.exact,
            "baseline_fragments":baseline_metrics.track_count,
            "fast_supplement_track_count":fast_metrics.track_count,
            "fast_mixed_with_background_source":fast_metrics.mixed_background
        }));
    }
    Ok((summary, pairing))
}

fn row_values(row: &Value, fields: &[&str]) -> Vec<String> {
    fields
        .iter()
        .map(|field| match &row[*field] {
            Value::String(value) => value.clone(),
            Value::Bool(value) => bool_text(*value),
            Value::Number(value) => value.to_string(),
            Value::Null => String::new(),
            other => other.to_string(),
        })
        .collect()
}

fn json_pretty(path: &Path, value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn sha256_json(path: &Path) -> Result<String> {
    sha256_file(path)
}

struct RgbCanvas {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl RgbCanvas {
    fn new(width: usize, height: usize, color: [u8; 3]) -> Self {
        let mut pixels = vec![0; width * height * 3];
        for pixel in pixels.chunks_exact_mut(3) {
            pixel.copy_from_slice(&color);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    fn pixel(&mut self, x: i32, y: i32, color: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let index = (y as usize * self.width + x as usize) * 3;
        self.pixels[index..index + 3].copy_from_slice(&color);
    }

    fn line(&mut self, from: [i32; 2], to: [i32; 2], color: [u8; 3], width: i32) {
        let (mut x0, mut y0) = (from[0], from[1]);
        let (x1, y1) = (to[0], to[1]);
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            let radius = width.saturating_sub(1) / 2;
            for oy in -radius..=radius {
                for ox in -radius..=radius {
                    self.pixel(x0 + ox, y0 + oy, color);
                }
            }
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

    fn circle(&mut self, center: [i32; 2], radius: i32, fill: [u8; 3], outline: [u8; 3]) {
        for y in -radius..=radius {
            for x in -radius..=radius {
                let distance = x * x + y * y;
                if distance <= radius * radius {
                    let color = if distance >= (radius - 1) * (radius - 1) {
                        outline
                    } else {
                        fill
                    };
                    self.pixel(center[0] + x, center[1] + y, color);
                }
            }
        }
    }

    fn text(&mut self, origin: [i32; 2], text: &str, color: [u8; 3], scale: i32) {
        let mut x = origin[0];
        for ch in text.chars() {
            let rows = glyph_3x5(ch);
            for (row, bits) in rows.into_iter().enumerate() {
                for column in 0..3 {
                    if bits & (1 << (2 - column)) != 0 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                self.pixel(
                                    x + column * scale + sx,
                                    origin[1] + row as i32 * scale + sy,
                                    color,
                                );
                            }
                        }
                    }
                }
            }
            x += 4 * scale;
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path).map_err(|error| error.to_string())?;
        let mut encoder = Encoder::new(file, self.width as u32, self.height as u32);
        encoder.set_color(ColorType::Rgb);
        encoder.set_depth(BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
        writer
            .write_image_data(&self.pixels)
            .map_err(|error| error.to_string())?;
        Ok(())
    }
}

fn glyph_3x5(ch: char) -> [u8; 5] {
    match ch.to_ascii_uppercase() {
        'A' => [2, 5, 7, 5, 5],
        'B' => [6, 5, 6, 5, 6],
        'C' => [3, 4, 4, 4, 3],
        'D' => [6, 5, 5, 5, 6],
        'E' => [7, 4, 6, 4, 7],
        'F' => [7, 4, 6, 4, 4],
        'G' => [3, 4, 5, 5, 3],
        'H' => [5, 5, 7, 5, 5],
        'I' => [7, 2, 2, 2, 7],
        'J' => [1, 1, 1, 5, 2],
        'K' => [5, 5, 6, 5, 5],
        'L' => [4, 4, 4, 4, 7],
        'M' => [5, 7, 7, 5, 5],
        'N' => [5, 7, 7, 7, 5],
        'O' => [2, 5, 5, 5, 2],
        'P' => [6, 5, 6, 4, 4],
        'Q' => [2, 5, 5, 3, 1],
        'R' => [6, 5, 6, 5, 5],
        'S' => [3, 4, 2, 1, 6],
        'T' => [7, 2, 2, 2, 2],
        'U' => [5, 5, 5, 5, 7],
        'V' => [5, 5, 5, 5, 2],
        'W' => [5, 5, 7, 7, 5],
        'X' => [5, 5, 2, 5, 5],
        'Y' => [5, 5, 2, 2, 2],
        'Z' => [7, 1, 2, 4, 7],
        '0' => [7, 5, 5, 5, 7],
        '1' => [2, 6, 2, 2, 7],
        '2' => [6, 1, 2, 4, 7],
        '3' => [6, 1, 2, 1, 6],
        '4' => [5, 5, 7, 1, 1],
        '5' => [7, 4, 6, 1, 6],
        '6' => [3, 4, 6, 5, 2],
        '7' => [7, 1, 2, 2, 2],
        '8' => [2, 5, 2, 5, 2],
        '9' => [2, 5, 3, 1, 6],
        '-' => [0, 0, 7, 0, 0],
        '/' => [1, 1, 2, 4, 4],
        ':' => [0, 2, 0, 2, 0],
        ',' => [0, 0, 0, 2, 4],
        '.' => [0, 0, 0, 0, 2],
        '(' => [1, 2, 2, 2, 1],
        ')' => [4, 2, 2, 2, 4],
        '|' => [2, 2, 2, 2, 2],
        '=' => [0, 7, 0, 7, 0],
        '+' => [0, 2, 7, 2, 0],
        '_' => [0, 0, 0, 0, 7],
        _ => [0; 5],
    }
}

fn plot_track_points(
    canvas: &mut RgbCanvas,
    tracks: &[Value],
    detection_id: i64,
    color: [u8; 3],
    connect: bool,
    start: [f64; 2],
    vx: f64,
    vy: f64,
    norm: f64,
    bottom: i32,
    left: i32,
    right: i32,
    frame_count: usize,
) {
    let mut points = tracks
        .iter()
        .flat_map(|track| track["points"].as_array().into_iter().flatten())
        .filter(|point| point["detection_id"].as_i64() == Some(detection_id))
        .filter_map(|point| {
            Some((
                point["frame_index"].as_u64()? as usize,
                [point["aligned_x"].as_f64()?, point["aligned_y"].as_f64()?],
            ))
        })
        .collect::<Vec<_>>();
    points.sort_by_key(|item| item.0);
    let mut drawn = Vec::new();
    for (frame, position) in points {
        if frame >= frame_count {
            continue;
        }
        let projected = ((position[0] - start[0]) * vx + (position[1] - start[1]) * vy) / norm;
        let px = left as f64
            + (right - left) as f64 * frame as f64 / frame_count.saturating_sub(1).max(1) as f64;
        let py = bottom as f64 - 15.0 - (projected + 12.0) / 160.0 * 92.0;
        let point = [px.round() as i32, py.round() as i32];
        drawn.push(point);
        canvas.circle(point, 4, color, [255, 255, 255]);
    }
    if connect {
        for pair in drawn.windows(2) {
            canvas.line(pair[0], pair[1], color, 2);
        }
    }
}

fn save_track_plot(
    path: &Path,
    truths: &[Value],
    baseline: &[Value],
    fast: &[Value],
    frame_count: usize,
) -> Result<()> {
    let width = 1050usize;
    let height = 640usize.max(180 * truths.len() + 70);
    let mut canvas = RgbCanvas::new(width, height, [248, 250, 252]);
    let left = 190i32;
    let right = width as i32 - 40;
    for (index, truth) in truths.iter().enumerate() {
        let top = 36 + index as i32 * 180;
        let bottom = top + 132;
        canvas.text(
            [22, top + 4],
            truth["target_id"].as_str().unwrap_or(""),
            [28, 39, 54],
            2,
        );
        canvas.text(
            [22, top + 28],
            truth["role"].as_str().unwrap_or(""),
            [85, 99, 117],
            1,
        );
        canvas.line([left, bottom], [right, bottom], [170, 180, 192], 1);
        canvas.line([left, top], [left, bottom], [170, 180, 192], 1);
        for frame in 0..frame_count {
            let px =
                left + (right - left) * frame as i32 / frame_count.saturating_sub(1).max(1) as i32;
            canvas.line([px, bottom - 3], [px, bottom + 3], [170, 180, 192], 1);
            if index + 1 == truths.len() {
                canvas.text([px - 2, bottom + 7], &frame.to_string(), [91, 102, 118], 1);
            }
        }
        let expected = truth["expected_observations"]
            .as_array()
            .ok_or("truth has no expected observations")?;
        let expected_frames = expected
            .iter()
            .filter_map(|point| point["frame_index"].as_u64().map(|value| value as usize))
            .collect::<HashSet<_>>();
        let positions = truth["registered_positions"]
            .as_array()
            .ok_or("truth has no registered positions")?;
        let start = [
            positions[0][0].as_f64().ok_or("truth start x invalid")?,
            positions[0][1].as_f64().ok_or("truth start y invalid")?,
        ];
        let vx = truth["vx_px_per_reference_interval"]
            .as_f64()
            .ok_or("truth vx invalid")?;
        let vy = truth["vy_px_per_reference_interval"]
            .as_f64()
            .ok_or("truth vy invalid")?;
        let norm = vx.hypot(vy).max(1e-9);
        let mut projected_truth = Vec::new();
        for (frame, position) in positions.iter().enumerate() {
            if !expected_frames.contains(&frame) {
                continue;
            }
            let projected = ((position[0].as_f64().unwrap_or(0.0) - start[0]) * vx
                + (position[1].as_f64().unwrap_or(0.0) - start[1]) * vy)
                / norm;
            let px = left as f64
                + (right - left) as f64 * frame as f64
                    / frame_count.saturating_sub(1).max(1) as f64;
            let py = bottom as f64 - 15.0 - (projected + 12.0) / 160.0 * 92.0;
            projected_truth.push([px.round() as i32, py.round() as i32]);
        }
        for pair in projected_truth.windows(2) {
            canvas.line(pair[0], pair[1], [250, 189, 45], 3);
        }
        let detection_id = truth["detection_id"]
            .as_i64()
            .ok_or("truth detection id invalid")?;
        plot_track_points(
            &mut canvas,
            baseline,
            detection_id,
            [223, 105, 62],
            false,
            start,
            vx,
            vy,
            norm,
            bottom,
            left,
            right,
            frame_count,
        );
        plot_track_points(
            &mut canvas,
            fast,
            detection_id,
            [24, 151, 108],
            true,
            start,
            vx,
            vy,
            norm,
            bottom,
            left,
            right,
            frame_count,
        );
        canvas.text(
            [left, top + 4],
            "ALONG-TRACK DISPLACEMENT (PX)",
            [75, 89, 107],
            1,
        );
    }
    canvas.text(
        [left, height as i32 - 24],
        "FRAME ORDER | YELLOW = TRUTH | ORANGE = BASELINE | GREEN = SUPPLEMENT",
        [65, 78, 96],
        1,
    );
    canvas.save(path)
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn csv_rows(path: &Path) -> Result<Vec<Value>> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let mut reader = csv::ReaderBuilder::new().from_reader(bytes);
    let headers = reader.headers().map_err(|error| error.to_string())?.clone();
    reader
        .records()
        .map(|record| {
            let record = record.map_err(|error| error.to_string())?;
            let mut row = Map::new();
            for (header, value) in headers.iter().zip(record.iter()) {
                row.insert(header.to_owned(), json!(value));
            }
            Ok(Value::Object(row))
        })
        .collect()
}

fn copy_file(source: &Path, destination: &Path) -> Result<()> {
    if source.canonicalize().ok() == destination.canonicalize().ok() {
        return Ok(());
    }
    fs::copy(source, destination).map_err(|error| error.to_string())?;
    Ok(())
}

fn render_report(
    point_run: &Path,
    report_path: &Path,
    output: &Path,
    manifest: &Value,
    rows: &[Value],
) -> Result<PathBuf> {
    let point_manifest_path = point_run.join("manifest.json");
    let point_manifest: Value =
        serde_json::from_slice(&fs::read(&point_manifest_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if point_manifest["schema_version"].as_i64() != Some(3)
        || point_manifest["experiment"].as_str() != Some("point_source_detection_ablation")
    {
        return Err(
            "point-source report requires schema_version 3 ablation output; rerun the experiment"
                .into(),
        );
    }
    let report = if report_path.is_absolute() {
        report_path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(report_path)
    };
    let report_parent = report
        .parent()
        .ok_or("report path has no parent directory")?;
    fs::create_dir_all(report_parent).map_err(|error| error.to_string())?;
    let point_evidence = report_parent.join("evidence").join("point");
    let motion_evidence = report_parent.join("evidence").join("motion");
    fs::create_dir_all(&point_evidence).map_err(|error| error.to_string())?;
    fs::create_dir_all(&motion_evidence).map_err(|error| error.to_string())?;

    for name in [
        "manifest.json",
        "summary.csv",
        "target_results.csv",
        "control_summary.csv",
    ] {
        copy_file(&point_run.join(name), &point_evidence.join(name))?;
    }
    let point_rows = csv_rows(&point_run.join("summary.csv"))?;
    if let Some(case_sheets) = point_manifest["outputs"]["case_sheets"].as_object() {
        for value in case_sheets.values() {
            let source = PathBuf::from(value.as_str().ok_or("point case sheet path invalid")?);
            let name = source
                .file_name()
                .ok_or("point case sheet has no file name")?;
            copy_file(&point_run.join(name), &point_evidence.join(name))?;
        }
    }
    for name in [
        "manifest.json",
        "trajectory_results.csv",
        "motion_pairing.csv",
        "truth_trajectories.json",
        "frozen_frame_sources.csv.gz",
        "motion_tracks.png",
    ] {
        copy_file(&output.join(name), &motion_evidence.join(name))?;
    }

    let mut table_rows = String::new();
    for row in rows {
        let target = row["target_id"].as_str().unwrap_or("");
        let role = match target {
            "FAST-15" => "快速正例，15/15",
            "FAST-GAP-12" => "快速正例，3 个分散缺帧，12/15",
            "SLOW-BASE-15" => "慢速普通关联正例，15/15",
            "BELOW-12" => "持续性负例，11/15",
            _ => row["role"].as_str().unwrap_or(""),
        };
        table_rows.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}/{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}/{}</td><td>{}</td></tr>",
            html_escape(target),
            html_escape(role),
            row["expected_presence"],
            row["sequence_frames"],
            row["baseline_exact_track_recovered"],
            row["baseline_fragments"],
            row["fast_exact_track_recovered"],
            row["fast_best_track_target_points"],
            row["expected_presence"],
            row["fast_mixed_with_background_source"]
        ));
    }
    let parameters = &manifest["parameters"];
    let actual_argv = manifest["execution"]["argv"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    let source_inputs = manifest["input_cache"]["source_hashes_verified"]
        .as_u64()
        .unwrap_or(0);
    let cache_path = manifest["input_cache"]["path"].as_str().unwrap_or("");
    let cache_hash = manifest["input_cache"]["sha256"].as_str().unwrap_or("");
    let section = format!(
        "<!-- RST19-MOTION-ABLATION-START --><section class=\"card\"><h2>8. 运动约束：独立的轨迹关联对照</h2><p>该实验冻结同一逐帧质量源列表，A 使用普通最近邻轨迹关联，B 在保留 A 多帧轨迹的同时，对未被 A 串联的质量点运行速度预测补充关联。轨迹真值按每帧实际观测时间生成，速度以中位相邻帧间隔归一化；图中横轴按帧序显示。注入对象是逐帧 <code>Detection</code> 记录，不重新经过图像像素检测；因此它验证的是关联阶段，不是运动目标的像素检出能力。</p><div class=\"note\"><b>比较口径：</b>A 的精确轨迹要求普通关联单独形成完整真值轨迹。B 的精确补充轨迹表示速度预测补充层形成完整轨迹；A 中遗留的单帧碎片仍可与补充轨迹并存。</div><div class=\"scroll\"><table><thead><tr><th>固定真值轨迹</th><th>角色</th><th>有效观测／总帧数</th><th>A 普通关联精确轨迹</th><th>A 关联碎片数</th><th>B 速度补充精确轨迹</th><th>B 最佳补充点数</th><th>B 是否关联背景源</th></tr></thead><tbody>{table_rows}</tbody></table></div><p>真值含 15/15 快速正例、带 3 个分散缺帧的 12/15 快速正例、15/15 慢速基线正例和 11/15 持续性负例。统一参数：普通关联半径 {link} px、最少 {presence} 帧；补充层 SNR ≥ {snr}、速度上限 {step} px/参考间隔、预测门 {gate} px、拟合 RMS ≤ {rms} px。缓存重算配准相对已保存平移的最大偏差为 {shift:.4} px；{quality} 个原始 FITS SHA-256 已核验。</p><img src=\"evidence/motion/motion_tracks.png\" alt=\"普通关联与速度补充关联的注入轨迹对比\"><p class=\"muted\">橙点表示普通关联中目标的点，绿线表示速度预测补充轨迹，黄线表示注入真值。背景污染仅检查轨迹是否混入冻结质量源列表中的其它检测 ID；11/15 负例用于验证 min_presence 门，不是全图误报率估计。</p><p>本次实际命令：</p><pre><code>{argv}</code></pre><ul><li><a href=\"evidence/motion/manifest.json\">运动关联参数与来源哈希 manifest.json</a></li><li><a href=\"evidence/motion/trajectory_results.csv\">逐轨迹结果 trajectory_results.csv</a></li><li><a href=\"evidence/motion/motion_pairing.csv\">A/B 配对摘要 motion_pairing.csv</a></li><li><a href=\"evidence/motion/truth_trajectories.json\">冻结真值轨迹 truth_trajectories.json</a></li><li><a href=\"evidence/motion/frozen_frame_sources.csv.gz\">冻结逐帧源列表</a></li></ul><p class=\"muted\">缓存：<code>{cache}</code> · SHA-256：<code>{cache_hash}</code> · 实验目录：<code>{run_dir}</code></p></section><!-- RST19-MOTION-ABLATION-END -->",
        link = parameters["link_radius_px"].as_f64().unwrap_or_default(),
        presence = parameters["min_presence_frames"].as_u64().unwrap_or_default(),
        snr = parameters["supplement_min_flux_snr"].as_f64().unwrap_or_default(),
        step = parameters["supplement_max_step_px_per_reference_interval"].as_f64().unwrap_or_default(),
        gate = parameters["supplement_prediction_gate_px"].as_f64().unwrap_or_default(),
        rms = parameters["supplement_max_fit_rms_px"].as_f64().unwrap_or_default(),
        shift = manifest["recomputed_registration_max_delta_px"].as_f64().unwrap_or_default(),
        quality = source_inputs,
        argv = html_escape(&actual_argv),
        cache = html_escape(cache_path),
        cache_hash = html_escape(cache_hash),
        run_dir = html_escape(&display_path(output)),
    );
    let point_section = format!(
        "<section class=\"card\"><h2>点源检测消融</h2><div class=\"scroll\"><table><thead><tr><th>注入峰值</th><th>版本</th><th>命中目标</th><th>召回</th><th>定位误差中位数 / P90</th><th>未命中</th></tr></thead><tbody>{}</tbody></table></div><p><a href=\"evidence/point/manifest.json\">点源实验清单</a> · <a href=\"evidence/point/target_results.csv\">逐目标数据</a></p></section>",
        point_rows
            .iter()
            .map(|row| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}/{}</td><td>{}</td><td>{} / {}</td><td>{}</td></tr>",
                    html_escape(row["gaussian_center_amplitude_adu"].as_str().unwrap_or("")),
                    html_escape(row["version"].as_str().unwrap_or("")),
                    html_escape(row["matched_targets"].as_str().unwrap_or("")),
                    html_escape(row["target_total"].as_str().unwrap_or("")),
                    html_escape(row["recall"].as_str().unwrap_or("")),
                    html_escape(row["median_position_error_px"].as_str().unwrap_or("")),
                    html_escape(row["p90_position_error_px"].as_str().unwrap_or("")),
                    html_escape(row["unmatched_targets"].as_str().unwrap_or("")),
                )
            })
            .collect::<String>()
    );
    let mut html = if report.is_file() {
        fs::read_to_string(&report).map_err(|error| error.to_string())?
    } else {
        format!("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>RST19 配对消融报告</title><style>body{{font:16px/1.7 system-ui,sans-serif;background:#f4f6f8;color:#1d2733;margin:0;padding:32px}}.card{{background:white;border-radius:12px;padding:24px;margin:20px auto;max-width:1200px;box-shadow:0 2px 14px #18283a16}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dfe5eb;text-align:left}}.muted{{color:#667585}}img{{max-width:100%;height:auto}}.note{{background:#fff8df;padding:14px}}</style></head><body><main><h1>RST19 配对消融报告</h1>{point_section}")
    };
    if let Some(start) = html.find("<!-- RST19-MOTION-ABLATION-START -->") {
        if let Some(end) = html[start..].find("<!-- RST19-MOTION-ABLATION-END -->") {
            html.replace_range(
                start..start + end + "<!-- RST19-MOTION-ABLATION-END -->".len(),
                &section,
            );
        }
    } else if let Some(heading) = html.find("运动约束：独立的轨迹关联对照") {
        let start = html[..heading].rfind("<section").unwrap_or(heading);
        let end = html[heading..]
            .find("</section>")
            .map(|offset| heading + offset + "</section>".len())
            .unwrap_or(heading);
        html.replace_range(start..end, &section);
    } else if let Some(end) = html.find("</body>") {
        html.insert_str(end, &section);
    } else if let Some(end) = html.find("</html>") {
        html.insert_str(end, &section);
    } else {
        html.push_str(&section);
        html.push_str("</main></body></html>");
    }
    fs::write(&report, html).map_err(|error| error.to_string())?;
    crate::ablation_report::decorate(&report)?;
    Ok(report)
}

fn source_hashes() -> Value {
    let backend = Path::new(env!("CARGO_MANIFEST_DIR"));
    let files = [
        "src/motion_association_ablation.rs",
        "src/sequence.rs",
        "src/sequence_fast.rs",
        "src/sequence_motion.rs",
        "src/sequence_registration.rs",
    ];
    let mut hashes = Map::new();
    for relative in files {
        let path = backend.join(relative);
        hashes.insert(
            relative.replace('\\', "/"),
            sha256_file(&path).map_or(Value::Null, |value| json!(value)),
        );
    }
    Value::Object(hashes)
}

fn utc_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let days = seconds.div_euclid(86_400);
    let day_seconds = seconds.rem_euclid(86_400);
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
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}+00:00",
        day_seconds / 3600,
        (day_seconds % 3600) / 60,
        day_seconds % 60
    )
}

pub fn run(
    cache_path: &Path,
    point_run_dir: &Path,
    out_dir: &Path,
    report_path: &Path,
    options: &MotionAblationOptions,
) -> Result<Value> {
    eprintln!("motion-ablation: preparing output and validating cached inputs");
    let cache_path = cache_path
        .canonicalize()
        .map_err(|error| error.to_string())?;
    eprintln!("motion-ablation: cache path resolved");
    if !cache_path.is_file() {
        return Err(format!(
            "sequence cache does not exist: {}",
            cache_path.display()
        ));
    }
    if out_dir.exists()
        && fs::read_dir(out_dir)
            .map_err(|error| error.to_string())?
            .next()
            .is_some()
    {
        return Err(format!(
            "refusing to overwrite non-empty experiment directory: {}",
            out_dir.display()
        ));
    }
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let output = out_dir.canonicalize().map_err(|error| error.to_string())?;
    eprintln!("motion-ablation: output directory ready");
    let point_run_dir = point_run_dir
        .canonicalize()
        .map_err(|error| error.to_string())?;
    eprintln!("motion-ablation: point-run path resolved");
    let cache_hash = sha256_file(&cache_path)?;
    eprintln!("motion-ablation: cache digest computed");
    let frozen = load_frozen_cache(&cache_path)?;
    eprintln!(
        "motion-ablation: loaded {} frames and {} cached quality points",
        frozen.frames.len(),
        frozen.quality_point_count
    );
    let actual_min_presence = options
        .min_presence
        .unwrap_or_else(|| frozen.sequence["min_presence"].as_u64().unwrap_or(0) as usize);
    let actual_link_radius = options.link_radius_px.unwrap_or_else(|| {
        frozen.sequence["link_radius_px"]
            .as_f64()
            .unwrap_or(f64::NAN)
    });
    if actual_min_presence < 3 || actual_min_presence > frozen.frames.len() {
        return Err("minimum presence must be within the number of cached frames".into());
    }
    if actual_min_presence != 12 {
        return Err("the fixed 12/15 and 11/15 trajectory roles require min_presence=12".into());
    }
    if !actual_link_radius.is_finite() || actual_link_radius <= 0.0 {
        return Err("link radius must be finite and positive".into());
    }
    let deltas = frozen
        .frame_times
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    if deltas.is_empty()
        || deltas
            .iter()
            .any(|delta| !delta.is_finite() || *delta <= 0.0)
    {
        return Err("cached frame times must be finite and strictly increasing".into());
    }
    let reference_interval = median(&deltas)?;
    let maximum_slow_step = 2.0f64.hypot(0.5)
        * deltas.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        / reference_interval;
    if actual_link_radius < maximum_slow_step {
        return Err(format!(
            "link radius is too small for the fixed slow-positive control: need at least {maximum_slow_step:.3} px for the largest timestamp interval"
        ));
    }
    let width = json_i64(&frozen.frames[0]["width_px"], "frame width")? as usize;
    let height = json_i64(&frozen.frames[0]["height_px"], "frame height")? as usize;
    let background_positions = frozen
        .registered_background
        .iter()
        .map(|frame| frame.to_vec())
        .collect::<Vec<_>>();
    let truths = make_truth_paths(
        width,
        height,
        &frozen.frame_times,
        &frozen.cached_shifts,
        &background_positions,
        options.seed,
    )?;
    eprintln!(
        "motion-ablation: placed {} deterministic truth paths",
        truths.len()
    );
    let injected = append_truth_sources(&frozen.background, &truths)?;

    let mut frozen_rows = Vec::new();
    let synthetic = truths
        .iter()
        .map(|truth| {
            Ok((
                json_i64(&truth["detection_id"], "truth detection id")?,
                json_string(&truth["target_id"], "truth target id")?.to_owned(),
            ))
        })
        .collect::<Result<std::collections::HashMap<_, _>>>()?;
    for (frame_index, sources) in injected.iter().enumerate() {
        for source in sources {
            frozen_rows.push(vec![
                frame_index.to_string(),
                source.detection_id.to_string(),
                format!("{:.9}", source.x),
                format!("{:.9}", source.y),
                format!("{:.6}", source.flux_snr.unwrap_or(0.0)),
                "1".to_owned(),
                synthetic
                    .get(&source.detection_id)
                    .cloned()
                    .unwrap_or_default(),
            ]);
        }
    }
    let frozen_csv = output.join("frozen_frame_sources.csv.gz");
    let frozen_csv_bytes = csv_bytes(
        &[
            "frame_index",
            "detection_id",
            "x_px",
            "y_px",
            "flux_snr",
            "quality_passed",
            "synthetic_target_id",
        ],
        &frozen_rows,
        false,
    )?;
    write_gzip(&frozen_csv, &frozen_csv_bytes, 6)?;
    let truth_file = output.join("truth_trajectories.json");
    json_pretty(&truth_file, &json!(truths))?;

    let sequence_options = TrackSequenceOptions {
        frame_names: Some(
            frozen
                .frame_paths
                .iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
        ),
        frame_times: Some(frozen.frame_times.clone()),
        link_radius_px: actual_link_radius,
        min_presence: Some(actual_min_presence),
        motion_min_displacement_px: frozen.sequence["motion_min_displacement_px"]
            .as_f64()
            .ok_or("cache is missing motion_min_displacement_px")?,
        registration_radius_px: options.registration_radius_px,
        max_motion_fit_rms_px: frozen.sequence["max_motion_fit_rms_px"]
            .as_f64()
            .ok_or("cache is missing max_motion_fit_rms_px")?,
        persistent_min_presence: frozen.sequence["persistent_min_presence"]
            .as_u64()
            .map(|value| value as usize),
    };
    eprintln!("motion-ablation: running ordinary Rust association");
    let started = Instant::now();
    let baseline_value = sequence::track_detections(&injected, &sequence_options)?;
    let baseline_seconds = started.elapsed().as_secs_f64();
    let baseline = track_array(&baseline_value)?.to_vec();
    eprintln!(
        "motion-ablation: ordinary association produced {} tracks",
        baseline.len()
    );
    let cumulative = baseline_value["cumulative_shifts"]
        .as_array()
        .ok_or("baseline has no cumulative shifts")?;
    let cumulative_shifts = cumulative
        .iter()
        .map(|shift| {
            let pair = shift.as_array().ok_or("baseline shift is not a pair")?;
            if pair.len() != 2 {
                return Err("baseline shift is not a pair".into());
            }
            Ok([
                json_f64(&pair[0], "baseline shift x")?,
                json_f64(&pair[1], "baseline shift y")?,
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    let started = Instant::now();
    eprintln!("motion-ablation: running Rust fast-point supplement");
    let fast_value = sequence_fast::track_fast_point_movers(
        &injected,
        &cumulative_shifts,
        Some(&frozen.frame_times),
        &FastPointOptions {
            link_radius_px: actual_link_radius,
            min_presence: Some(actual_min_presence),
            motion_min_displacement_px: sequence_options.motion_min_displacement_px,
            min_flux_snr: options.min_flux_snr,
            max_step_px: options.max_step_px,
            gate_px: options.prediction_gate_px,
            max_fit_rms_px: options.max_fit_rms_px,
        },
        &baseline,
    )?;
    let supplement_seconds = started.elapsed().as_secs_f64();
    let fast = track_array(&fast_value)?.to_vec();
    eprintln!("motion-ablation: supplement produced {} tracks", fast.len());
    let (summary, pairing) = make_summary_rows(&truths, &baseline, &fast, frozen.frames.len())?;

    let summary_fields = [
        "target_id",
        "role",
        "expected_presence",
        "sequence_frames",
        "baseline_tracks_containing_target",
        "baseline_best_track_target_points",
        "baseline_target_frame_coverage",
        "baseline_exact_track_recovered",
        "baseline_fragments",
        "baseline_mixed_with_other_injected_target",
        "baseline_mixed_with_background_source",
        "fast_supplement_tracks_containing_target",
        "fast_best_track_target_points",
        "fast_target_frame_coverage",
        "fast_exact_track_recovered",
        "fast_track_classification",
        "fast_mixed_with_other_injected_target",
        "fast_mixed_with_background_source",
        "combined_single_track_only_exact",
        "expected_frame_indices",
        "synthetic_detection_id",
    ];
    let pairing_fields = [
        "target_id",
        "expected_presence",
        "baseline_exact_track_recovered",
        "fast_exact_track_recovered",
        "combined_single_track_only_exact",
        "baseline_fragments",
        "fast_supplement_track_count",
        "fast_mixed_with_background_source",
    ];
    let summary_path = output.join("trajectory_results.csv");
    fs::write(
        &summary_path,
        csv_bytes(
            &summary_fields,
            &summary
                .iter()
                .map(|row| row_values(row, &summary_fields))
                .collect::<Vec<_>>(),
            true,
        )?,
    )
    .map_err(|error| error.to_string())?;
    let pairing_path = output.join("motion_pairing.csv");
    fs::write(
        &pairing_path,
        csv_bytes(
            &pairing_fields,
            &pairing
                .iter()
                .map(|row| row_values(row, &pairing_fields))
                .collect::<Vec<_>>(),
            true,
        )?,
    )
    .map_err(|error| error.to_string())?;
    let plot_path = output.join("motion_tracks.png");
    save_track_plot(&plot_path, &truths, &baseline, &fast, frozen.frames.len())?;
    eprintln!("motion-ablation: wrote association artifacts");

    let registration_deltas = cumulative_shifts
        .iter()
        .zip(&frozen.cached_shifts)
        .map(|(computed, cached)| (computed[0] - cached[0]).hypot(computed[1] - cached[1]))
        .collect::<Vec<_>>();
    let input_sources = frozen
        .frame_paths
        .iter()
        .zip(&frozen.identities)
        .map(|(path, identity)| {
            json!({
                "path":display_path(path),
                "sha256":identity["sha256"],
                "size_bytes":identity["size"]
            })
        })
        .collect::<Vec<_>>();
    let manifest_path = output.join("manifest.json");
    let mut manifest_truths = truths.clone();
    for truth in &mut manifest_truths {
        if let Some(object) = truth.as_object_mut() {
            object.remove("registered_positions");
        }
    }
    let manifest = json!({
        "schema_version":2,
        "experiment":"association_stage_motion_constraint_ablation",
        "status":"completed",
        "created_local_time":utc_timestamp(),
        "input_cache":{
            "path":display_path(&cache_path),
            "sha256":cache_hash,
            "cache_version":frozen.envelope.get("cache_version").cloned().unwrap_or(Value::Null),
            "sequence_cache_version":frozen.envelope.get("sequence_cache_version").cloned().unwrap_or(Value::Null),
            "quality_track_points_rebuilt":frozen.quality_point_count,
            "background_source_counts_by_frame":frozen.background.iter().map(Vec::len).collect::<Vec<_>>(),
            "source_hashes_verified":frozen.identities.len(),
            "source_inputs":input_sources
        },
        "frozen_association_input":{
            "path":display_path(&frozen_csv),
            "sha256":sha256_json(&frozen_csv)?,
            "truth_path_file":display_path(&truth_file),
            "truth_path_sha256":sha256_json(&truth_file)?,
            "dtype":"Detection objects reconstructed from cached quality points; synthetic points appended",
            "coordinate_rule":"synthetic raw position = frozen registered truth position + cached cumulative shift",
            "all_quality_source_flux_snr_verified":true
        },
        "frame_times_seconds":frozen.frame_times,
        "truth_time_model":{
            "reference_interval_seconds":reference_interval,
            "registered_position":"origin + velocity_px_per_reference_interval * (frame_time - first_frame_time) / median_adjacent_frame_interval",
            "coordinate_conversion":"raw position = registered truth position + cached cumulative registration shift",
            "plot_x_axis":"frame order (not uniformly spaced elapsed time)"
        },
        "cached_cumulative_shifts_px":frozen.cached_shifts,
        "recomputed_registration_max_delta_px":registration_deltas.iter().copied().fold(0.0, f64::max),
        "truth_trajectories":manifest_truths,
        "versions":[
            {"id":"A","name":"普通质量点源关联","function":"track_detections","link_radius_px":actual_link_radius},
            {"id":"B","name":"A + 速度预测补充关联","function":"track_fast_point_movers","existing_tracks":"A tracks reserved; no already-linked multiframe source reused"}
        ],
        "parameters":{
            "link_radius_px":actual_link_radius,
            "min_presence_frames":actual_min_presence,
            "motion_min_displacement_px_from_cache":sequence_options.motion_min_displacement_px,
            "registration_radius_px":options.registration_radius_px,
            "max_motion_fit_rms_px_baseline":sequence_options.max_motion_fit_rms_px,
            "supplement_min_flux_snr":options.min_flux_snr,
            "supplement_max_step_px_per_reference_interval":options.max_step_px,
            "supplement_prediction_gate_px":options.prediction_gate_px,
            "supplement_max_fit_rms_px":options.max_fit_rms_px,
            "truth_coordinate_avoid_radius_px":18.0,
            "truth_position_seed":options.seed,
            "truth_sample_types":"15/15 fast, 12/15 separated gaps, 15/15 slow baseline positive, 11/15 below-presence negative"
        },
        "association_metrics":{
            "exact_track":"one output track contains exactly the target's expected detection IDs on every expected frame, with no other frame for that target, and no additional track contains its ID",
            "fragment_count":"number of output tracks containing the target detection ID; baseline singletons count as fragments",
            "combined_single_track_only_exact":"the union of baseline and supplement has exactly one target-containing track and that track contains exactly the expected target observations; false when baseline fragments remain alongside a complete supplement track",
            "mixed_target_ids":"one track contains synthetic detection IDs from more than one known truth trajectory",
            "background_contamination":"one track containing a synthetic target ID also contains a cached background detection ID",
            "negative_control":"the 11/15 path should not produce a fast supplement because min_presence is 12; it is not a false-positive estimate"
        },
        "runtime":{"baseline_seconds":baseline_seconds,"supplement_seconds":supplement_seconds},
        "outputs":{
            "trajectory_csv":display_path(&summary_path),
            "pairing_csv":display_path(&pairing_path),
            "frozen_sources_csv_gz":display_path(&frozen_csv),
            "truth_json":display_path(&truth_file),
            "track_plot":display_path(&plot_path)
        },
        "limitations":[
            "This is an association-stage test: artificial Detection records are appended to a frozen quality-source list.",
            "It does not test whether the pixel detector can discover these targets or whether the target passes single-frame image quality measurement.",
            "The background quality detections are reconstructed from the saved SequenceResult cache and revalidated against all raw FITS SHA-256 identities.",
            "Track recovery applies only to the four injected paths and does not establish physical moving-object truth or full-sequence association precision.",
            "A crossing-track adversarial case is not included because the current supplement intentionally cannot replace already linked multiframe baseline tracks."
        ],
        "execution":{
            "argv":if options.argv.is_empty() { json!(["rst19-motion-ablation"]) } else { json!(options.argv) },
            "runtime":"Rust",
            "version":env!("CARGO_PKG_VERSION"),
            "target_os":std::env::consts::OS,
            "target_arch":std::env::consts::ARCH,
            "source_revision":option_env!("RST19_SOURCE_REVISION"),
            "source_file_sha256":source_hashes()
        }
    });
    json_pretty(&manifest_path, &manifest)?;
    let report = render_report(&point_run_dir, report_path, &output, &manifest, &summary)?;
    eprintln!("motion-ablation: completed run and report");
    Ok(json!({
        "status":"completed",
        "run_dir":display_path(&output),
        "manifest":display_path(&manifest_path),
        "report":display_path(&report),
        "baseline_seconds":baseline_seconds,
        "supplement_seconds":supplement_seconds
    }))
}

fn default_output_dir() -> PathBuf {
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    PathBuf::from(format!("tmp/ablation/motion-{epoch}"))
}

fn take_value(
    args: &[String],
    index: &mut usize,
    inline: Option<&str>,
    option: &str,
) -> Result<String> {
    if let Some(value) = inline {
        return Ok(value.to_owned());
    }
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("--{option} requires a value"))
}

fn parse_args(
    args: &[String],
) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf, MotionAblationOptions)> {
    let mut cache = None;
    let mut point_run_dir = None;
    let mut out_dir = None;
    let mut report = None;
    let mut options = MotionAblationOptions::default();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let (name, inline) = arg.strip_prefix("--").map_or(("", None), |rest| {
            rest.split_once('=')
                .map_or((rest, None), |(name, value)| (name, Some(value)))
        });
        let value = match name {
            "cache"
            | "point-run-dir"
            | "out-dir"
            | "report"
            | "seed"
            | "min-presence"
            | "link-radius-px"
            | "registration-radius-px"
            | "min-flux-snr"
            | "max-step-px"
            | "prediction-gate-px"
            | "max-fit-rms-px" => Some(take_value(args, &mut index, inline, name)?),
            "" => return Err("unexpected positional argument".into()),
            _ => return Err(format!("unknown option --{name}")),
        }
        .unwrap_or_default();
        match name {
            "cache" => cache = Some(PathBuf::from(value)),
            "point-run-dir" => point_run_dir = Some(PathBuf::from(value)),
            "out-dir" => out_dir = Some(PathBuf::from(value)),
            "report" => report = Some(PathBuf::from(value)),
            "seed" => {
                options.seed = value
                    .parse()
                    .map_err(|_| "--seed must be a signed integer")?
            }
            "min-presence" => {
                options.min_presence = Some(
                    value
                        .parse()
                        .map_err(|_| "--min-presence must be an integer")?,
                )
            }
            "link-radius-px" => {
                options.link_radius_px = Some(
                    value
                        .parse()
                        .map_err(|_| "--link-radius-px must be a number")?,
                )
            }
            "registration-radius-px" => {
                options.registration_radius_px = value
                    .parse()
                    .map_err(|_| "--registration-radius-px must be a number")?
            }
            "min-flux-snr" => {
                options.min_flux_snr = value
                    .parse()
                    .map_err(|_| "--min-flux-snr must be a number")?
            }
            "max-step-px" => {
                options.max_step_px = value
                    .parse()
                    .map_err(|_| "--max-step-px must be a number")?
            }
            "prediction-gate-px" => {
                options.prediction_gate_px = value
                    .parse()
                    .map_err(|_| "--prediction-gate-px must be a number")?
            }
            "max-fit-rms-px" => {
                options.max_fit_rms_px = value
                    .parse()
                    .map_err(|_| "--max-fit-rms-px must be a number")?
            }
            _ => unreachable!(),
        }
        index += 1;
    }
    options.argv = std::iter::once("rst19-motion-ablation".to_owned())
        .chain(args.iter().cloned())
        .collect();
    Ok((
        cache.ok_or("--cache is required")?,
        point_run_dir.ok_or("--point-run-dir is required")?,
        out_dir.unwrap_or_else(default_output_dir),
        report.unwrap_or_else(|| PathBuf::from("deliverables/ablation/index.html")),
        options,
    ))
}

/// Entry point used by the native desktop worker and command line wrapper.
pub fn run_desktop(args: &[String]) -> Result<Value> {
    let (cache, point_run_dir, out_dir, report, options) = parse_args(args)?;
    run(&cache, &point_run_dir, &out_dir, &report, &options)
}

pub fn run_cli(args: &[String]) -> i32 {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("在同一冻结质量源列表和真值轨迹下比较普通关联与速度预测补充关联\n\nUsage: rst19-motion-ablation --cache CACHE --point-run-dir DIR [--out-dir DIR] [--report FILE] [--seed N]");
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
            eprintln!("rst19-motion-ablation: failed: {error}");
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

    fn fixture_root() -> PathBuf {
        let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rst19-motion-association-ablation-{}-{sequence}",
            std::process::id()
        ))
    }

    fn create_cache(root: &Path) -> PathBuf {
        let template: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/motion_association_ablation/cache_template.json"
        ))
        .unwrap();
        let mut envelope = template;
        let frame_count = envelope["input_identity"]["frames"]
            .as_array()
            .unwrap()
            .len();
        for index in 0..frame_count {
            let path = root.join(format!("fixture-frame-{index:02}.fits"));
            let bytes = format!("motion-association-parity-frame-{index}\n").into_bytes();
            fs::write(&path, &bytes).unwrap();
            let path_text = path.to_string_lossy().into_owned();
            let identity = &mut envelope["input_identity"]["frames"][index];
            identity["path"] = json!(path_text);
            identity["sha256"] = json!(sha256_bytes_for_test(&bytes));
            identity["size"] = json!(bytes.len());
            envelope["sequence"]["frames"][index]["path"] = json!(path_text);
        }
        let cache = root.join("sequence-cache.json.gz");
        let bytes = serde_json::to_vec(&envelope).unwrap();
        write_gzip(&cache, &bytes, 6).unwrap();
        cache
    }

    fn sha256_bytes_for_test(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn create_point_run(root: &Path) -> PathBuf {
        let run = root.join("point-run");
        fs::create_dir_all(&run).unwrap();
        fs::write(
            run.join("manifest.json"),
            br#"{"schema_version":3,"experiment":"point_source_detection_ablation","outputs":{"case_sheets":{}}}"#,
        )
        .unwrap();
        fs::write(
            run.join("summary.csv"),
            "gaussian_center_amplitude_adu,version,matched_targets,target_total,recall,median_position_error_px,p90_position_error_px,unmatched_targets,detector_seconds\r\n24,A Gaussian,1,1,1.0,,,0,0.01\r\n",
        )
        .unwrap();
        fs::write(run.join("target_results.csv"), "target_id\r\n").unwrap();
        fs::write(run.join("control_summary.csv"), "version\r\n").unwrap();
        run
    }

    fn read_gzip(path: &Path) -> Vec<u8> {
        let compressed = fs::read(path).unwrap();
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut bytes = Vec::new();
        decoder.read_to_end(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn frozen_cache_run_matches_python_truth_and_association_rows() {
        let root = fixture_root();
        fs::create_dir_all(&root).unwrap();
        let cache = create_cache(&root);
        let point_run = create_point_run(&root);
        let output = root.join("motion-run");
        let report = root.join("report").join("index.html");
        let result = run(
            &cache,
            &point_run,
            &output,
            &report,
            &MotionAblationOptions {
                argv: vec![
                    "rst19-motion-ablation".into(),
                    "--seed".into(),
                    "19019".into(),
                ],
                ..MotionAblationOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result["status"], "completed");

        let reference: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/motion_association_ablation/python_reference.json"
        ))
        .unwrap();
        assert_eq!(
            csv_rows(&output.join("trajectory_results.csv")).unwrap(),
            reference["trajectory_results"].as_array().unwrap().to_vec()
        );
        assert_eq!(
            csv_rows(&output.join("motion_pairing.csv")).unwrap(),
            reference["motion_pairing"].as_array().unwrap().to_vec()
        );
        let truth: Value =
            serde_json::from_slice(&fs::read(output.join("truth_trajectories.json")).unwrap())
                .unwrap();
        assert_eq!(truth, reference["truth_trajectories"]);
        assert_eq!(
            String::from_utf8(read_gzip(&output.join("frozen_frame_sources.csv.gz"))).unwrap(),
            reference["frozen_sources_csv"].as_str().unwrap()
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&fs::read(output.join("manifest.json")).unwrap())
                .unwrap()["recomputed_registration_max_delta_px"],
            reference["recomputed_registration_max_delta_px"]
        );
        let png = fs::read(output.join("motion_tracks.png")).unwrap();
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(
            (
                u32::from_be_bytes(png[16..20].try_into().unwrap()),
                u32::from_be_bytes(png[20..24].try_into().unwrap())
            ),
            (
                reference["png_size"][0].as_u64().unwrap() as u32,
                reference["png_size"][1].as_u64().unwrap() as u32
            )
        );
        assert!(fs::read_to_string(&report)
            .unwrap()
            .contains("运动约束：独立的轨迹关联对照"));
        assert!(root
            .join("report/evidence/motion/motion_tracks.png")
            .is_file());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn seeded_python_mt19937_matches_fixed_truth_coordinate() {
        let frame_times = (0..15).map(|index| index as f64).collect::<Vec<_>>();
        let shifts = vec![[0.0, 0.0]; 15];
        let background = vec![Vec::new(); 15];
        let paths =
            make_truth_paths(4096, 4096, &frame_times, &shifts, &background, 19019).unwrap();
        let first = &paths[0]["registered_positions"][0];
        assert!((first[0].as_f64().unwrap() - 1288.0031881322316).abs() < 1e-10);
        assert!((first[1].as_f64().unwrap() - 3719.0465759391645).abs() < 1e-10);
    }

    #[test]
    fn parses_iso_timestamps_with_fraction_and_offset() {
        let start = parse_timestamp_micros("2026-03-30T16:32:05.4132+08:00").unwrap();
        let next = parse_timestamp_micros("2026-03-30T08:32:06.913200Z").unwrap();
        assert_eq!(next - start, 1_500_000);
        assert_eq!(
            parse_timestamp_micros("2026-03-30T16:32:05+08").unwrap(),
            parse_timestamp_micros("2026-03-30T08:32:05Z").unwrap()
        );
        assert_eq!(
            parse_timestamp_micros("2026-03-30T16:32:05+0800").unwrap(),
            parse_timestamp_micros("2026-03-30T08:32:05Z").unwrap()
        );
        assert!(parse_timestamp_micros("2026-02-30T08:32:05Z").is_err());
    }

    #[test]
    fn invalid_or_non_monotonic_timestamps_fall_back_to_frame_order() {
        assert_eq!(
            relative_frame_times(&[Some(5_000_000), None, Some(9_000_000)]),
            vec![0.0, 1.0, 2.0]
        );
        assert_eq!(
            relative_frame_times(&[Some(5_000_000), Some(4_000_000)]),
            vec![0.0, 1.0]
        );
    }
}
