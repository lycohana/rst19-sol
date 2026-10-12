//! Pixel-profile measurements used by sequence motion-feature analysis.
use crate::Result;
use serde_json::{json, Value};

fn plane(request: &Value) -> Option<Vec<Vec<f32>>> {
    let rows = request.get("residual")?.as_array()?;
    let mut result = Vec::with_capacity(rows.len());
    let mut width = None;
    for row in rows {
        let values = row.as_array()?;
        if width.is_some_and(|expected| expected != values.len()) {
            return None;
        }
        width = Some(values.len());
        result.push(
            values
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .map(|number| number as f32)
                        .unwrap_or(f32::NAN)
                })
                .collect(),
        );
    }
    Some(result)
}

fn number(request: &Value, key: &str) -> Result<f64> {
    request
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric {key}"))
}

fn optional_number(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn model_supported(point: &Value) -> bool {
    let model = point.get("single_frame_model").and_then(Value::as_str);
    let snr = optional_number(point, "line_template_snr");
    if model == Some("long_line_profile") {
        return snr.is_some_and(|value| value >= 5.0)
            && optional_number(point, "line_vs_background_delta_bic")
                .is_some_and(|value| value >= 6.0);
    }
    model == Some("line")
        && snr.is_some_and(|value| value >= 5.0)
        && optional_number(point, "line_vs_point_delta_bic").is_some_and(|value| value >= 6.0)
        && optional_number(point, "line_vs_double_delta_bic").is_some_and(|value| value >= 6.0)
        && optional_number(point, "line_vs_elliptical_delta_bic").is_some_and(|value| value >= 6.0)
}

fn shape_supported(point: &Value, fill_threshold: f64, continuity_threshold: f64) -> bool {
    optional_number(point, "profile_fill_ratio").is_some_and(|value| value >= fill_threshold)
        && optional_number(point, "continuity_ratio")
            .is_some_and(|value| value >= continuity_threshold)
}

fn percent(value: f64) -> String {
    format!("{:.0}%", value * 100.0)
}

fn trail_shape_reason(
    point: &Value,
    min_length: f64,
    min_fill: f64,
    min_continuity: f64,
) -> Result<Option<String>> {
    if min_length <= 0.0 {
        return Err("single-frame short-trail length gate must be positive".into());
    }
    let length = number(point, "length_px")?;
    if length < min_length {
        return Ok(Some(format!("长度不足（{length:.1} < {min_length:.1}px）")));
    }
    let Some(continuity) = optional_number(point, "continuity_ratio") else {
        return Ok(Some("缺少连续支撑证据".into()));
    };
    let Some(fill) = optional_number(point, "profile_fill_ratio") else {
        return Ok(Some("缺少中段填充证据".into()));
    };
    if optional_number(point, "line_artifact_overlap_fraction").is_some_and(|value| value >= 0.75) {
        let overlap = optional_number(point, "line_artifact_overlap_fraction").unwrap();
        return Ok(Some(format!(
            "线状伪影候选重叠过高，保留待复核（{}）",
            percent(overlap)
        )));
    }
    let model = point.get("single_frame_model").and_then(Value::as_str);
    if model == Some("not_fit") {
        return Ok(Some("超出局部有限线段模型范围，保留几何候选".into()));
    }
    if model_supported(point) {
        let rescue_continuity = if length >= 16.0 {
            0.60_f64.max(0.45)
        } else {
            0.60
        };
        let rescue_fill = if length >= 16.0 {
            0.60_f64.max(0.45)
        } else {
            0.60
        };
        if continuity >= rescue_continuity && fill >= rescue_fill {
            return Ok(None);
        }
        return Ok(Some(format!(
            "有限线段模型较优，但像素连续支撑不足（连续 {} / 填充 {}）",
            percent(continuity),
            percent(fill)
        )));
    }
    let model_reason = match model {
        Some("elliptical_point") => Some("局部模型更像静态椭圆星像"),
        Some("point") => Some("局部模型更像单星"),
        Some("double") => Some("局部模型更像双星"),
        Some("background") => Some("局部模型更像背景起伏"),
        Some("long_line_profile") => Some("超长线段剖面证据不足，保留待复核"),
        _ => None,
    };
    if let Some(reason) = model_reason {
        return Ok(Some(reason.into()));
    }
    let (required_continuity, required_fill) = if length >= 16.0 {
        (0.45, 0.45)
    } else {
        (min_continuity, min_fill)
    };
    if continuity < required_continuity {
        return Ok(Some(format!(
            "连续支撑不足（{} < {}）",
            percent(continuity),
            percent(required_continuity)
        )));
    }
    if fill < required_fill {
        return Ok(Some(format!(
            "中段填充不足（{} < {}）",
            percent(fill),
            percent(required_fill)
        )));
    }
    Ok(None)
}

/// Evaluate pixel-shape evidence gates for a single-frame line candidate.
pub fn trail_shape_evidence(request: &Value) -> Result<Value> {
    let point = request
        .get("point")
        .ok_or_else(|| "missing point".to_string())?;
    let options = request.get("options").unwrap_or(&Value::Null);
    let min_length = optional_number(options, "min_length_px").unwrap_or(8.0);
    let min_fill = optional_number(options, "min_profile_fill_ratio").unwrap_or(0.80);
    let min_continuity = optional_number(options, "min_continuity_ratio").unwrap_or(0.80);
    let motion_min_fill = optional_number(options, "motion_min_profile_fill_ratio").unwrap_or(0.45);
    let motion_min_continuity =
        optional_number(options, "motion_min_continuity_ratio").unwrap_or(0.45);
    let reason = trail_shape_reason(point, min_length, min_fill, min_continuity)?;
    let single_frame_supported = reason.is_none();
    let motion_shape_supported = shape_supported(point, motion_min_fill, motion_min_continuity);
    let sequence_motion_shape_supported = shape_supported(point, 0.45, 0.45);
    let model = point.get("single_frame_model").and_then(Value::as_str);
    let length = number(point, "length_px")?;
    let sequence_single_frame_supported = trail_shape_reason(point, 8.0, 0.80, 0.80)?.is_none();
    let sequence_line_supported = sequence_single_frame_supported
        || (matches!(model, Some("line" | "long_line_profile"))
            && model_supported(point)
            && length >= 8.0
            && sequence_motion_shape_supported);
    Ok(json!({
        "single_frame_model_supported": model_supported(point),
        "motion_feature_shape_supported": motion_shape_supported,
        "single_frame_supported": single_frame_supported,
        "single_frame_reason": reason,
        "sequence_line_supported": sequence_line_supported,
    }))
}

fn array<'a>(value: &'a Value, key: &str) -> Result<&'a [Value]> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("missing array {key}"))
}

fn field_number(point: &Value, key: &str) -> Result<f64> {
    number(point, key)
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

fn circular_line_angle_difference(first: f64, second: f64) -> f64 {
    ((first - second + 90.0).rem_euclid(180.0) - 90.0).abs()
}

fn motion_feature_track_shape_supported(
    points: &[Value],
    min_profile_fill: f64,
    min_continuity: f64,
    max_angle_deviation: f64,
    max_length_deviation: f64,
) -> Result<bool> {
    if points.len() < 3 {
        return Ok(false);
    }
    let mut supported = Vec::new();
    for point in points {
        let length = field_number(point, "length_px")?;
        let short = length < 16.0;
        let fill_threshold = if short {
            min_profile_fill.max(0.80)
        } else {
            min_profile_fill
        };
        let continuity_threshold = if short {
            min_continuity.max(0.80)
        } else {
            min_continuity
        };
        if shape_supported(point, fill_threshold, continuity_threshold) {
            supported.push(point);
        }
    }
    let required_count = ((points.len() as f64 * 0.5).ceil() as usize).max(2);
    if supported.len() < required_count {
        return Ok(false);
    }
    let mut profile_values: Vec<f64> = supported
        .iter()
        .filter_map(|point| optional_number(point, "profile_fill_ratio"))
        .collect();
    if median(&mut profile_values).is_none_or(|value| value < min_profile_fill) {
        return Ok(false);
    }

    let angles: Vec<f64> = supported
        .iter()
        .filter_map(|point| optional_number(point, "angle_deg"))
        .filter(|value| value.is_finite())
        .collect();
    if angles.len() >= 2 {
        let doubled: Vec<f64> = angles
            .iter()
            .map(|angle| (2.0 * angle).to_radians())
            .collect();
        let mean_sin = doubled.iter().map(|angle| angle.sin()).sum::<f64>() / doubled.len() as f64;
        let mean_cos = doubled.iter().map(|angle| angle.cos()).sum::<f64>() / doubled.len() as f64;
        let mean_doubled = mean_sin.atan2(mean_cos);
        let max_deviation = doubled
            .iter()
            .map(|angle| {
                0.5 * (angle - mean_doubled)
                    .sin()
                    .atan2((angle - mean_doubled).cos())
            })
            .map(f64::abs)
            .fold(0.0, f64::max)
            .to_degrees();
        if max_deviation > max_angle_deviation {
            return Ok(false);
        }
    }

    let lengths: Vec<f64> = points
        .iter()
        .filter_map(|point| optional_number(point, "length_px"))
        .filter(|value| value.is_finite())
        .collect();
    if lengths.len() >= 2 {
        let Some(median_length) = median(&mut lengths.clone()) else {
            return Ok(false);
        };
        if median_length <= 0.0 {
            return Ok(false);
        }
        let consistent_count = lengths
            .iter()
            .filter(|length| {
                ((*length - median_length).abs() / median_length) <= max_length_deviation
            })
            .count();
        if consistent_count < required_count {
            return Ok(false);
        }
    }
    Ok(true)
}

fn motion_feature_axis_supported(
    points: &[Value],
    max_deviation: f64,
    min_pass_fraction: f64,
) -> Result<bool> {
    if points.len() < 3 {
        return Ok(false);
    }
    if !max_deviation.is_finite() || max_deviation <= 0.0 {
        return Err("max_deviation_deg must be positive".into());
    }
    if !min_pass_fraction.is_finite() || min_pass_fraction <= 0.0 || min_pass_fraction > 1.0 {
        return Err("min_pass_fraction must be in (0, 1]".into());
    }
    let mut ordered = points.to_vec();
    ordered.sort_by_key(|point| point["frame_index"].as_i64().unwrap_or_default());
    let mut pass_count = 0usize;
    let mut transition_count = 0usize;
    for pair in ordered.windows(2) {
        let delta_x = field_number(&pair[1], "aligned_x")? - field_number(&pair[0], "aligned_x")?;
        let delta_y = field_number(&pair[1], "aligned_y")? - field_number(&pair[0], "aligned_y")?;
        let step = delta_x.hypot(delta_y);
        if !step.is_finite() || step < 1.0 {
            continue;
        }
        transition_count += 1;
        let Some(line_angle) = optional_number(&pair[1], "angle_deg").filter(|v| v.is_finite())
        else {
            continue;
        };
        let motion_angle = delta_y.atan2(delta_x).to_degrees();
        if circular_line_angle_difference(line_angle, motion_angle) <= max_deviation {
            pass_count += 1;
        }
    }
    if transition_count == 0 {
        return Ok(false);
    }
    let required = ((transition_count as f64 * min_pass_fraction).ceil() as usize).max(2);
    Ok(pass_count >= required)
}

fn artifact_supported_as_moving(
    points: &[Value],
    overlap_fraction: f64,
    min_clear_fraction: f64,
) -> Result<bool> {
    if points.is_empty() {
        return Ok(false);
    }
    if !overlap_fraction.is_finite() || !(0.0..=1.0).contains(&overlap_fraction) {
        return Err("overlap_fraction must be in [0, 1]".into());
    }
    if !min_clear_fraction.is_finite() || min_clear_fraction <= 0.0 || min_clear_fraction > 1.0 {
        return Err("min_clear_fraction must be in (0, 1]".into());
    }
    let known: Vec<f64> = points
        .iter()
        .filter_map(|point| optional_number(point, "line_artifact_overlap_fraction"))
        .filter(|value| value.is_finite())
        .collect();
    if known.is_empty() {
        return Ok(true);
    }
    let clear_count = known
        .iter()
        .filter(|value| **value < overlap_fraction)
        .count();
    Ok(clear_count as f64 / known.len() as f64 >= min_clear_fraction)
}

fn artifact_overlap_compatible(
    points: &[Value],
    compatibility: &Value,
    overlap_fraction: f64,
    min_clear_fraction: f64,
) -> Result<bool> {
    if artifact_supported_as_moving(points, overlap_fraction, min_clear_fraction)? {
        return Ok(true);
    }
    if compatibility["presence_supported"].as_bool() != Some(true) {
        return Ok(false);
    }
    let displacement = number(compatibility, "displacement_px")?;
    let min_displacement = number(compatibility, "min_displacement_px")?;
    if !displacement.is_finite() || displacement < min_displacement {
        return Ok(false);
    }
    let Some(fit_rms) = optional_number(compatibility, "fit_rms_px") else {
        return Ok(false);
    };
    let max_fit_rms = number(compatibility, "max_fit_rms_px")?;
    if !fit_rms.is_finite() || !max_fit_rms.is_finite() || fit_rms > max_fit_rms {
        return Ok(false);
    }
    Ok([
        "shape_supported",
        "model_supported",
        "motion_axis_supported",
        "temporal_continuity_supported",
        "exposure_supported",
    ]
    .iter()
    .all(|key| compatibility[*key].as_bool() == Some(true)))
}

fn integer(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_f64().map(|number| number as i64))
}

fn edge_truncation_supported(points: &[Value], frame_shapes: &[Value], margin: f64) -> bool {
    if points.len() < 3 || frame_shapes.len() < 3 || !margin.is_finite() || margin < 0.0 {
        return false;
    }
    let mut indexed: Vec<(i64, &Value)> = points
        .iter()
        .filter_map(|point| integer(&point["frame_index"]).map(|index| (index, point)))
        .collect();
    if indexed.len() != points.len() {
        return false;
    }
    indexed.sort_by_key(|(index, _)| *index);
    let indices: Vec<i64> = indexed.iter().map(|(index, _)| *index).collect();
    if indices.windows(2).any(|pair| pair[0] == pair[1])
        || indices.first().is_none_or(|index| *index < 0)
        || indices
            .last()
            .is_none_or(|index| *index >= frame_shapes.len() as i64)
        || indices.last().unwrap() - indices.first().unwrap() + 1 != indices.len() as i64
    {
        return false;
    }
    let first = *indices.first().unwrap();
    let last = *indices.last().unwrap();
    let prefix = first == 0;
    let suffix = last == frame_shapes.len() as i64 - 1;
    if !prefix && !suffix {
        return false;
    }
    let mut endpoint_indices = Vec::new();
    if suffix {
        endpoint_indices.push(first);
    }
    if prefix {
        endpoint_indices.push(last);
    }
    for index in endpoint_indices {
        let Some((_, point)) = indexed
            .iter()
            .find(|(point_index, _)| *point_index == index)
        else {
            continue;
        };
        let Some(shape) = frame_shapes.get(index as usize).and_then(Value::as_array) else {
            continue;
        };
        if shape.len() != 2 {
            continue;
        }
        let (Some(height), Some(width)) = (integer(&shape[0]), integer(&shape[1])) else {
            continue;
        };
        if height <= 0 || width <= 0 {
            continue;
        }
        let Some(bbox) = point["bbox"].as_array() else {
            continue;
        };
        if bbox.len() != 4 {
            continue;
        }
        let Some((x_min, y_min, x_max, y_max)) = bbox
            .iter()
            .map(Value::as_f64)
            .collect::<Option<Vec<_>>>()
            .map(|values| (values[0], values[1], values[2], values[3]))
        else {
            continue;
        };
        let edge_distance = x_min
            .min(y_min)
            .min((width - 1) as f64 - x_max)
            .min((height - 1) as f64 - y_max);
        if edge_distance.is_finite() && edge_distance <= margin {
            return true;
        }
    }
    false
}

/// Evaluate the cross-frame shape, motion-axis, artifact, and edge gates.
pub fn motion_feature_track_gates(request: &Value) -> Result<Value> {
    let points: Vec<Value> = array(request, "points")?.to_vec();
    let options = request.get("options").unwrap_or(&Value::Null);
    let track_shape_supported = motion_feature_track_shape_supported(
        &points,
        optional_number(options, "min_profile_fill_ratio").unwrap_or(0.45),
        optional_number(options, "min_continuity_ratio").unwrap_or(0.45),
        optional_number(options, "max_angle_deviation_deg").unwrap_or(15.0),
        optional_number(options, "max_length_deviation_fraction").unwrap_or(0.75),
    )?;
    let axis_supported = motion_feature_axis_supported(
        &points,
        optional_number(options, "max_motion_axis_deviation_deg").unwrap_or(25.0),
        optional_number(options, "min_motion_axis_pass_fraction").unwrap_or(0.60),
    )?;
    let overlap_fraction = optional_number(options, "artifact_overlap_fraction").unwrap_or(0.90);
    let min_clear_fraction =
        optional_number(options, "min_artifact_clear_fraction").unwrap_or(0.50);
    let artifact_supported =
        artifact_supported_as_moving(&points, overlap_fraction, min_clear_fraction)?;
    let artifact_compatible = artifact_overlap_compatible(
        &points,
        request
            .get("artifact_compatibility")
            .unwrap_or(&Value::Null),
        overlap_fraction,
        min_clear_fraction,
    )?;
    let edge_supported = edge_truncation_supported(
        &points,
        request
            .get("frame_shapes")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]),
        optional_number(options, "edge_margin_px").unwrap_or(4.0),
    );
    Ok(json!({
        "track_shape_supported": track_shape_supported,
        "motion_axis_supported": axis_supported,
        "artifact_supported_as_moving": artifact_supported,
        "artifact_overlap_compatible_with_motion": artifact_compatible,
        "edge_truncation_supported": edge_supported,
    }))
}

fn unique_frame_indices(points: &[Value]) -> Option<Vec<i64>> {
    let mut indices: Vec<i64> = points
        .iter()
        .map(|point| integer(&point["frame_index"]))
        .collect::<Option<Vec<_>>>()?;
    indices.sort_unstable();
    indices.dedup();
    Some(indices)
}

fn frame_runs(indices: &[i64]) -> Vec<Vec<i64>> {
    let mut runs: Vec<Vec<i64>> = Vec::new();
    for index in indices {
        if runs
            .last()
            .is_some_and(|run| run.last().is_some_and(|last| *index == *last + 1))
        {
            runs.last_mut().unwrap().push(*index);
        } else {
            runs.push(vec![*index]);
        }
    }
    runs
}

fn temporal_continuity_supported(
    points: &[Value],
    min_presence: i64,
    edge_truncated: bool,
) -> Result<bool> {
    if points.is_empty() {
        return Ok(false);
    }
    if min_presence <= 0 {
        return Err("min_presence must be positive".into());
    }
    let Some(indices) = unique_frame_indices(points) else {
        return Ok(false);
    };
    let required = if edge_truncated {
        3_i64.max(min_presence - 1)
    } else {
        min_presence
    };
    Ok(frame_runs(&indices)
        .iter()
        .any(|run| run.len() as i64 >= required))
}

fn internal_gap_bridge_supported(
    points: &[Value],
    max_gap_frames: i64,
    min_run_length: i64,
) -> bool {
    if max_gap_frames <= 0 || min_run_length < 2 || points.is_empty() {
        return false;
    }
    let Some(indices) = unique_frame_indices(points) else {
        return false;
    };
    let runs = frame_runs(&indices);
    let strong_runs: Vec<&Vec<i64>> = runs
        .iter()
        .filter(|run| run.len() as i64 >= min_run_length)
        .collect();
    if strong_runs.len() != 2 {
        return false;
    }
    let left = strong_runs[0];
    let right = strong_runs[1];
    let gap = right[0] - left[left.len() - 1] - 1;
    if gap <= 0 || gap > max_gap_frames {
        return false;
    }
    let isolated_runs: Vec<&Vec<i64>> = runs
        .iter()
        .filter(|run| (run.len() as i64) < min_run_length)
        .collect();
    let isolated_count: usize = isolated_runs.iter().map(|run| run.len()).sum();
    if isolated_count > 2 {
        return false;
    }
    for run in isolated_runs {
        let distance = if run[run.len() - 1] < left[0] {
            left[0] - run[run.len() - 1] - 1
        } else if run[0] > right[right.len() - 1] {
            run[0] - right[right.len() - 1] - 1
        } else {
            return false;
        };
        if distance > max_gap_frames + 1 {
            return false;
        }
    }
    true
}

/// Evaluate the continuous-run confirmation and bounded interior-gap gates.
pub fn motion_feature_continuity_gates(request: &Value) -> Result<Value> {
    let points = array(request, "points")?;
    let options = request.get("options").unwrap_or(&Value::Null);
    let min_presence = request
        .get("min_presence")
        .and_then(Value::as_i64)
        .ok_or_else(|| "missing min_presence".to_string())?;
    let edge_truncated = request
        .get("edge_truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let temporal = temporal_continuity_supported(points, min_presence, edge_truncated)?;
    let bridge = internal_gap_bridge_supported(
        points,
        optional_number(options, "max_gap_frames").unwrap_or(2.0) as i64,
        optional_number(options, "min_run_length").unwrap_or(3.0) as i64,
    );
    Ok(json!({
        "temporal_continuity_supported": temporal,
        "internal_gap_bridge_supported": bridge,
    }))
}

fn normalized_motion_times(values: Option<&[Value]>, frame_count: usize) -> Vec<f64> {
    let fallback: Vec<f64> = (0..frame_count).map(|index| index as f64).collect();
    let Some(values) = values else {
        return fallback;
    };
    let Some(times) = values.iter().map(python_float).collect::<Option<Vec<_>>>() else {
        return fallback;
    };
    if times.len() != frame_count
        || times.iter().any(|value| !value.is_finite())
        || times.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return fallback;
    }
    let first = times.first().copied().unwrap_or(0.0);
    times.iter().map(|value| value - first).collect()
}

fn reference_time_interval_seconds(frame_times: &[f64]) -> f64 {
    let differences: Vec<f64> = frame_times
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|difference| difference.is_finite() && *difference > 0.0)
        .collect();
    let mut differences = differences;
    median(&mut differences).unwrap_or(1.0)
}

/// Validate a frame-time vector and return its normalized axis and reference step.
pub fn motion_time_axis(request: &Value) -> Result<Value> {
    let frame_count = request
        .get("frame_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| "frame_count must be a nonnegative integer".to_string())?
        as usize;
    let frame_times = request.get("frame_times").and_then(Value::as_array);
    let normalized = normalized_motion_times(frame_times.map(Vec::as_slice), frame_count);
    let interval = reference_time_interval_seconds(&normalized);
    Ok(json!({
        "frame_times": normalized,
        "reference_interval_s": interval,
    }))
}

fn days_from_civil(mut year: i64, month: i64, day: i64) -> i64 {
    year -= i64::from(month <= 2);
    let era = if year >= 0 {
        year / 400
    } else {
        (year - 399) / 400
    };
    let year_of_era = year - era * 400;
    let adjusted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era
}

fn parse_utc_microseconds(value: &str) -> Option<i128> {
    let text = value.trim().replace('Z', "+00:00");
    if text.len() < 10 || !text.is_ascii() {
        return None;
    }
    let bytes = text.as_bytes();
    if bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let parse = |start: usize, end: usize| -> Option<i64> { text.get(start..end)?.parse().ok() };
    let year = parse(0, 4)?;
    let month = parse(5, 7)?;
    let day = parse(8, 10)?;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || day < 1
        || day > month_days[(month - 1) as usize]
    {
        return None;
    }

    let offset_at = text
        .char_indices()
        .find(|(index, character)| *index >= 10 && matches!(character, '+' | '-'))
        .map(|(index, _)| index);
    let (date_time, offset_text) = match offset_at {
        Some(10) => (text.as_str(), None),
        Some(index) => (&text[..index], Some(&text[index..])),
        None => (text.as_str(), None),
    };
    let offset_seconds = if let Some(offset) = offset_text {
        let sign = if offset.starts_with('-') {
            -1_i64
        } else {
            1_i64
        };
        let digits: String = offset[1..]
            .chars()
            .filter(|character| *character != ':')
            .collect();
        if !digits.chars().all(|character| character.is_ascii_digit())
            || !matches!(digits.len(), 2 | 4 | 6)
        {
            return None;
        }
        let hours: i64 = digits[0..2].parse().ok()?;
        let minutes: i64 = if digits.len() >= 4 {
            digits[2..4].parse().ok()?
        } else {
            0
        };
        let seconds: i64 = if digits.len() == 6 {
            digits[4..6].parse().ok()?
        } else {
            0
        };
        if hours >= 24 || minutes >= 60 || seconds >= 60 {
            return None;
        }
        sign * (hours * 3600 + minutes * 60 + seconds)
    } else {
        0
    };

    let time_text = if date_time.len() == 10 {
        ""
    } else {
        &date_time[11..]
    };
    let (clock_text, fractional) = time_text
        .split_once('.')
        .map_or((time_text, None), |(clock, fraction)| {
            (clock, Some(fraction))
        });
    let clock_parts: Vec<&str> = clock_text.split(':').collect();
    if !(2..=3).contains(&clock_parts.len()) {
        if date_time.len() == 10 && time_text.is_empty() {
            let utc_seconds =
                days_from_civil(year, month, day) as i128 * 86_400 - offset_seconds as i128;
            return Some(utc_seconds * 1_000_000);
        }
        return None;
    }
    let hour: i64 = clock_parts[0].parse().ok()?;
    let minute: i64 = clock_parts[1].parse().ok()?;
    let second: i64 = if clock_parts.len() == 3 {
        clock_parts[2].parse().ok()?
    } else {
        0
    };
    if hour >= 24 || minute >= 60 || second >= 60 {
        return None;
    }
    let microseconds = if let Some(fraction) = fractional {
        if fraction.is_empty() || !fraction.chars().all(|character| character.is_ascii_digit()) {
            return None;
        }
        let significant = &fraction[..fraction.len().min(6)];
        let parsed: i64 = significant.parse().ok()?;
        parsed * 10_i64.pow((6 - significant.len()) as u32)
    } else {
        0
    };
    let utc_seconds = days_from_civil(year, month, day) as i128 * 86_400
        + (hour * 3600 + minute * 60 + second - offset_seconds) as i128;
    Some(utc_seconds * 1_000_000 + microseconds as i128)
}

/// Build the observation time axis from FITS DATE-OBS headers with frame fallback.
pub fn observation_time_axis(request: &Value) -> Result<Value> {
    let headers = request
        .get("headers")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing headers array".to_string())?;
    let fallback: Vec<f64> = (0..headers.len()).map(|index| index as f64).collect();
    let parsed: Option<Vec<i128>> = headers
        .iter()
        .map(|header| {
            header
                .get("DATE-OBS")
                .and_then(Value::as_str)
                .and_then(parse_utc_microseconds)
        })
        .collect();
    let Some(timestamps) = parsed else {
        return Ok(json!({"frame_times": fallback}));
    };
    let Some(first) = timestamps.first().copied() else {
        return Ok(json!({"frame_times": fallback}));
    };
    let mut times = Vec::with_capacity(timestamps.len());
    let mut previous = None;
    for timestamp in timestamps {
        if previous.is_some_and(|value| timestamp <= value) {
            return Ok(json!({"frame_times": fallback}));
        }
        times.push((timestamp - first) as f64 / 1_000_000.0);
        previous = Some(timestamp);
    }
    Ok(json!({"frame_times": times}))
}

fn line_angle_difference(first: f64, second: f64) -> f64 {
    ((first - second + 90.0).rem_euclid(180.0) - 90.0).abs()
}

fn point_distance(first: [f64; 2], second: [f64; 2]) -> f64 {
    (first[0] - second[0]).hypot(first[1] - second[1])
}

fn motion_feature_pair_geometry_penalty(
    previous: &Value,
    current: &Value,
    psf_fwhm: f64,
    expected_step_px: Option<f64>,
) -> Result<f64> {
    if psf_fwhm <= 0.0 {
        return Err("psf_fwhm must be positive".into());
    }
    let previous_center = [
        field_number(previous, "aligned_x")?,
        field_number(previous, "aligned_y")?,
    ];
    let current_center = [
        field_number(current, "aligned_x")?,
        field_number(current, "aligned_y")?,
    ];
    let delta = [
        current_center[0] - previous_center[0],
        current_center[1] - previous_center[1],
    ];
    let movement = delta[0].hypot(delta[1]);
    if !movement.is_finite() || movement <= 1.0_f64.max(0.75 * psf_fwhm) {
        return Ok(0.0);
    }
    let movement_angle = delta[1].atan2(delta[0]).to_degrees();
    let edge = previous
        .get("touches_edge")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || current
            .get("touches_edge")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let edge_weight = if edge { 0.30 } else { 1.0 };
    let angles = [
        optional_number(previous, "angle_deg"),
        optional_number(current, "angle_deg"),
    ];
    let mut penalty = 0.0;
    if let [Some(previous_angle), Some(current_angle)] = angles {
        if previous_angle.is_finite() && current_angle.is_finite() {
            let angle_error = 0.5
                * (line_angle_difference(previous_angle, movement_angle)
                    + line_angle_difference(current_angle, movement_angle));
            penalty += edge_weight * 5.0_f64.min(2.0 * (angle_error / 35.0).powi(2));
            if let Some(expected_step) =
                expected_step_px.filter(|value| value.is_finite() && *value > 0.0)
            {
                let radians = previous_angle.to_radians();
                let previous_unit = [radians.cos(), radians.sin()];
                let expected_forward = [
                    previous_center[0] + expected_step * previous_unit[0],
                    previous_center[1] + expected_step * previous_unit[1],
                ];
                let expected_backward = [
                    previous_center[0] - expected_step * previous_unit[0],
                    previous_center[1] - expected_step * previous_unit[1],
                ];
                let expected_center_error = point_distance(expected_forward, current_center)
                    .min(point_distance(expected_backward, current_center));
                let expected_step_scale = (2.0 * psf_fwhm).max(0.20 * expected_step);
                if expected_center_error.is_finite() && expected_step_scale > 0.0 {
                    penalty += edge_weight
                        * 5.0_f64.min(1.25 * (expected_center_error / expected_step_scale).powi(2));
                }
            }
        }
    }

    let previous_length = field_number(previous, "length_px")?;
    let current_length = field_number(current, "length_px")?;
    if [previous_length, current_length]
        .iter()
        .all(|length| length.is_finite() && *length > 0.0)
    {
        if let [Some(previous_angle), Some(current_angle)] = angles {
            let previous_radians = previous_angle.to_radians();
            let current_radians = current_angle.to_radians();
            if previous_radians.is_finite() && current_radians.is_finite() {
                let previous_half = [
                    0.5 * previous_length * previous_radians.cos(),
                    0.5 * previous_length * previous_radians.sin(),
                ];
                let current_half = [
                    0.5 * current_length * current_radians.cos(),
                    0.5 * current_length * current_radians.sin(),
                ];
                let previous_endpoints = [
                    [
                        previous_center[0] - previous_half[0],
                        previous_center[1] - previous_half[1],
                    ],
                    [
                        previous_center[0] + previous_half[0],
                        previous_center[1] + previous_half[1],
                    ],
                ];
                let current_endpoints = [
                    [
                        current_center[0] - current_half[0],
                        current_center[1] - current_half[1],
                    ],
                    [
                        current_center[0] + current_half[0],
                        current_center[1] + current_half[1],
                    ],
                ];
                let translated_previous = [
                    [
                        previous_endpoints[0][0] + delta[0],
                        previous_endpoints[0][1] + delta[1],
                    ],
                    [
                        previous_endpoints[1][0] + delta[0],
                        previous_endpoints[1][1] + delta[1],
                    ],
                ];
                let direct_error = 0.5
                    * (point_distance(translated_previous[0], current_endpoints[0])
                        + point_distance(translated_previous[1], current_endpoints[1]));
                let swapped_error = 0.5
                    * (point_distance(translated_previous[0], current_endpoints[1])
                        + point_distance(translated_previous[1], current_endpoints[0]));
                let endpoint_error = direct_error.min(swapped_error);
                let endpoint_scale =
                    (1.5 * psf_fwhm).max(0.12 * previous_length.min(current_length));
                if endpoint_error.is_finite() && endpoint_scale > 0.0 {
                    penalty +=
                        edge_weight * 3.0_f64.min(0.75 * (endpoint_error / endpoint_scale).powi(2));
                }
            }
        }
    }
    Ok(if penalty.is_finite() { penalty } else { 0.0 })
}

/// Return the soft geometry cost for linking two single-frame trail candidates.
pub fn motion_feature_pair_geometry_evidence(request: &Value) -> Result<Value> {
    let previous = request
        .get("previous")
        .ok_or_else(|| "missing previous point".to_string())?;
    let current = request
        .get("current")
        .ok_or_else(|| "missing current point".to_string())?;
    let psf_fwhm = number(request, "psf_fwhm")?;
    let expected_step_px = optional_number(request, "expected_step_px");
    let penalty =
        motion_feature_pair_geometry_penalty(previous, current, psf_fwhm, expected_step_px)?;
    Ok(json!({"penalty": penalty}))
}

fn point_motion_times(points: &[Value], frame_times: Option<&[Value]>) -> Result<Vec<f64>> {
    if points.is_empty() {
        return Ok(Vec::new());
    }
    let frame_indices: Vec<i64> = points
        .iter()
        .map(|point| {
            point
                .get("frame_index")
                .and_then(Value::as_i64)
                .ok_or_else(|| "motion feature point missing integer frame_index".to_string())
        })
        .collect::<Result<_>>()?;
    let max_frame = frame_indices.iter().copied().max().unwrap_or(0);
    let frame_count = usize::try_from(max_frame + 1)
        .map_err(|_| "motion feature frame_index must be nonnegative".to_string())?;
    let resolved = normalized_motion_times(frame_times, frame_count);
    let selected: Option<Vec<f64>> = frame_indices
        .iter()
        .map(|index| {
            let position = if *index < 0 {
                frame_count.checked_sub(index.unsigned_abs() as usize)?
            } else {
                *index as usize
            };
            resolved.get(position).copied()
        })
        .collect();
    let selected = selected.ok_or_else(|| "motion feature frame_index out of range".to_string())?;
    let valid = selected.iter().all(|value| value.is_finite())
        && selected.windows(2).all(|pair| pair[1] > pair[0]);
    let values = if valid {
        selected
    } else {
        frame_indices.iter().map(|index| *index as f64).collect()
    };
    let first = values.first().copied().unwrap_or(0.0);
    Ok(values.into_iter().map(|value| value - first).collect())
}

/// Fit the spatial line and elapsed-time speed of a sequence of trail centers.
pub fn fit_collinear_motion_geometry(request: &Value) -> Result<Value> {
    let points = array(request, "points")?;
    if points.len() < 2 {
        return Ok(json!({
            "projected_span": 0.0,
            "mean_speed_px_per_frame": 0.0,
            "rms_px": null,
            "angle_deg": null,
        }));
    }
    let coordinates: Vec<[f64; 2]> = points
        .iter()
        .map(|point| {
            Ok([
                field_number(point, "aligned_x")?,
                field_number(point, "aligned_y")?,
            ])
        })
        .collect::<Result<_>>()?;
    let count = coordinates.len() as f64;
    let center = [
        coordinates.iter().map(|point| point[0]).sum::<f64>() / count,
        coordinates.iter().map(|point| point[1]).sum::<f64>() / count,
    ];
    let centered: Vec<[f64; 2]> = coordinates
        .iter()
        .map(|point| [point[0] - center[0], point[1] - center[1]])
        .collect();
    let xx = centered
        .iter()
        .map(|point| point[0] * point[0])
        .sum::<f64>()
        / count;
    let yy = centered
        .iter()
        .map(|point| point[1] * point[1])
        .sum::<f64>()
        / count;
    let xy = centered
        .iter()
        .map(|point| point[0] * point[1])
        .sum::<f64>()
        / count;
    if ![xx, yy, xy].iter().all(|value| value.is_finite()) {
        return Ok(json!({
            "projected_span": 0.0,
            "mean_speed_px_per_frame": 0.0,
            "rms_px": null,
            "angle_deg": null,
        }));
    }
    let anisotropy = (xx - yy).hypot(2.0 * xy);
    if !anisotropy.is_finite() || anisotropy <= f64::EPSILON {
        return Ok(json!({
            "projected_span": 0.0,
            "mean_speed_px_per_frame": 0.0,
            "rms_px": null,
            "angle_deg": null,
        }));
    }
    let principal_angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let direction = [principal_angle.cos(), principal_angle.sin()];
    let normal = [-direction[1], direction[0]];
    let projections: Vec<f64> = centered
        .iter()
        .map(|point| point[0] * direction[0] + point[1] * direction[1])
        .collect();
    let residuals: Vec<f64> = centered
        .iter()
        .map(|point| (point[0] * normal[0] + point[1] * normal[1]).abs())
        .collect();
    let span = projections
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - projections.iter().copied().fold(f64::INFINITY, f64::min);
    let rms = (residuals.iter().map(|value| value * value).sum::<f64>() / count).sqrt();
    let frame_times = request.get("frame_times").and_then(Value::as_array);
    let times = point_motion_times(points, frame_times.map(Vec::as_slice))?;
    let time_span = times.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - times.iter().copied().fold(f64::INFINITY, f64::min);
    let speed_per_second = if time_span > 0.0 {
        span / time_span
    } else {
        0.0
    };
    let interval = request
        .get("reference_interval_s")
        .and_then(python_float)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0);
    Ok(json!({
        "projected_span": span,
        "mean_speed_px_per_frame": speed_per_second * interval,
        "rms_px": rms,
        "angle_deg": principal_angle.to_degrees(),
    }))
}

/// Select a monotonic, cross-frame collinear group from candidate trail points.
pub fn best_collinear_motion_feature_group(request: &Value) -> Result<Value> {
    let points = array(request, "points")?;
    let candidates = request
        .get("candidate_indices")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing candidate_indices array".to_string())?;
    let min_length = number(request, "min_length_px")?;
    let angle_tolerance = number(request, "angle_tolerance_deg")?;
    let line_tolerance = number(request, "line_tolerance_px")?;
    let min_span = number(request, "min_span_px")?;
    if candidates.len() < 3 {
        return Ok(json!({"group_indices": null}));
    }
    let mut valid_indices = Vec::new();
    for candidate in candidates {
        let raw_index = candidate
            .as_i64()
            .ok_or_else(|| "candidate index must be an integer".to_string())?;
        let index = if raw_index < 0 {
            points
                .len()
                .checked_sub(raw_index.unsigned_abs() as usize)
                .ok_or_else(|| "candidate index out of range".to_string())?
        } else {
            raw_index as usize
        };
        let point = points
            .get(index)
            .ok_or_else(|| "candidate index out of range".to_string())?;
        if field_number(point, "length_px")? >= min_length {
            valid_indices.push(index);
        }
    }
    if valid_indices.len() < 3 {
        return Ok(json!({"group_indices": null}));
    }
    let mut coordinates = Vec::with_capacity(valid_indices.len());
    let mut angles = Vec::with_capacity(valid_indices.len());
    let mut frame_numbers = Vec::with_capacity(valid_indices.len());
    for index in &valid_indices {
        let point = &points[*index];
        coordinates.push([
            field_number(point, "aligned_x")?,
            field_number(point, "aligned_y")?,
        ]);
        angles.push(field_number(point, "angle_deg")?);
        frame_numbers.push(
            point
                .get("frame_index")
                .and_then(Value::as_i64)
                .ok_or_else(|| "motion feature point missing integer frame_index".to_string())?,
        );
    }
    let monotonic_tolerance = 2.0_f64.max(0.25 * line_tolerance);
    let mut best_score: Option<(usize, f64, f64, f64)> = None;
    let mut best_group: Option<Vec<usize>> = None;
    for left in 0..valid_indices.len().saturating_sub(1) {
        for right in left + 1..valid_indices.len() {
            if frame_numbers[left] == frame_numbers[right] {
                continue;
            }
            let vector = [
                coordinates[right][0] - coordinates[left][0],
                coordinates[right][1] - coordinates[left][1],
            ];
            let separation = vector[0].hypot(vector[1]);
            if !separation.is_finite() || separation < min_span {
                continue;
            }
            let direction = [vector[0] / separation, vector[1] / separation];
            let line_angle = direction[1].atan2(direction[0]).to_degrees();
            let angle_mask: Vec<bool> = angles
                .iter()
                .map(|angle| {
                    angle.is_finite()
                        && line_angle_difference(*angle, line_angle) <= angle_tolerance
                })
                .collect();
            if angle_mask.iter().filter(|value| **value).count() < 3 {
                continue;
            }
            let normal = [-direction[1], direction[0]];
            let perpendicular: Vec<f64> = coordinates
                .iter()
                .map(|coordinate| {
                    ((coordinate[0] - coordinates[left][0]) * normal[0]
                        + (coordinate[1] - coordinates[left][1]) * normal[1])
                        .abs()
                })
                .collect();
            let member_positions: Vec<usize> = (0..valid_indices.len())
                .filter(|index| angle_mask[*index] && perpendicular[*index] <= line_tolerance)
                .collect();
            if member_positions.len() < 3 {
                continue;
            }
            let mut member_frames: Vec<i64> = member_positions
                .iter()
                .map(|position| frame_numbers[*position])
                .collect();
            member_frames.sort_unstable();
            member_frames.dedup();
            let mut selected_positions = Vec::with_capacity(member_frames.len());
            for frame_index in member_frames {
                let selected = member_positions
                    .iter()
                    .copied()
                    .filter(|position| frame_numbers[*position] == frame_index)
                    .min_by(|first, second| {
                        perpendicular[*first]
                            .total_cmp(&perpendicular[*second])
                            .then_with(|| first.cmp(second))
                    })
                    .unwrap();
                selected_positions.push(selected);
            }
            selected_positions.sort_by_key(|position| frame_numbers[*position]);
            if selected_positions.len() < 3 {
                continue;
            }
            let selected_projection: Vec<f64> = selected_positions
                .iter()
                .map(|position| {
                    (coordinates[*position][0] - coordinates[left][0]) * direction[0]
                        + (coordinates[*position][1] - coordinates[left][1]) * direction[1]
                })
                .collect();
            let differences: Vec<f64> = selected_projection
                .windows(2)
                .map(|pair| pair[1] - pair[0])
                .collect();
            let increasing = differences
                .iter()
                .all(|difference| *difference >= -monotonic_tolerance);
            let decreasing = differences
                .iter()
                .all(|difference| *difference <= monotonic_tolerance);
            if !increasing && !decreasing {
                continue;
            }
            let selected_span = selected_projection
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                - selected_projection
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
            if selected_span < min_span {
                continue;
            }
            let selected_points: Vec<Value> = selected_positions
                .iter()
                .map(|position| points[valid_indices[*position]].clone())
                .collect();
            let geometry = fit_collinear_motion_geometry(&json!({
                "points": selected_points,
                "frame_times": request.get("frame_times").cloned().unwrap_or(Value::Null),
                "reference_interval_s": request.get("reference_interval_s").cloned().unwrap_or(Value::Null),
            }))?;
            let fitted_span = optional_number(&geometry, "projected_span").unwrap_or(0.0);
            let fitted_rms = optional_number(&geometry, "rms_px");
            let Some(fitted_rms) = fitted_rms else {
                continue;
            };
            if fitted_span < min_span || fitted_rms > line_tolerance {
                continue;
            }
            let perpendicular_sum = selected_positions
                .iter()
                .map(|position| perpendicular[*position])
                .sum::<f64>();
            let score = (
                selected_positions.len(),
                fitted_span,
                -fitted_rms,
                -perpendicular_sum,
            );
            let greater = best_score.is_none_or(|best| {
                score.0 > best.0
                    || (score.0 == best.0
                        && (score.1 > best.1
                            || (score.1 == best.1
                                && (score.2 > best.2 || (score.2 == best.2 && score.3 > best.3)))))
            });
            if greater {
                best_score = Some(score);
                best_group = Some(
                    selected_positions
                        .iter()
                        .map(|position| valid_indices[*position])
                        .collect(),
                );
            }
        }
    }
    Ok(json!({"group_indices": best_group}))
}

fn deconvolved_motion_length(measured: f64, psf_fwhm: f64) -> Option<f64> {
    if !measured.is_finite() || !psf_fwhm.is_finite() || measured <= 0.0 || psf_fwhm <= 0.0 {
        return None;
    }
    let psf_sigma = psf_fwhm / 2.354_820_045_030_949_3;
    let motion_variance = 0.75 * (measured * measured - 16.0 * psf_sigma * psf_sigma);
    (motion_variance.is_finite() && motion_variance > 0.0).then(|| motion_variance.sqrt())
}

fn median_absolute_deviation(values: &[f64]) -> Option<(f64, f64)> {
    let mut ordered = values.to_vec();
    let median_value = median(&mut ordered)?;
    let mut deviations: Vec<f64> = values
        .iter()
        .map(|value| (value - median_value).abs())
        .collect();
    Some((median_value, median(&mut deviations)?))
}

fn motion_feature_exposure_metrics(
    points: &[Value],
    headers: &[Value],
    frame_times: Option<&[Value]>,
    psf_fwhm: f64,
) -> Result<(Option<f64>, Option<f64>, Option<f64>)> {
    if points.len() < 2 || headers.len() < 2 || psf_fwhm <= 0.0 {
        return Ok((None, None, None));
    }
    let mut exposures = Vec::with_capacity(headers.len());
    for header in headers {
        let Some(map) = header.as_object() else {
            return Ok((None, None, None));
        };
        if header.get("DATE-OBS").is_none_or(Value::is_null) {
            return Ok((None, None, None));
        }
        let exposure_ms = crate::fits::exposure_milliseconds(map)?;
        let Some(exposure_ms) = exposure_ms.filter(|value| value.is_finite() && *value > 0.0)
        else {
            return Ok((None, None, None));
        };
        exposures.push(exposure_ms / 1000.0);
    }
    let times = normalized_motion_times(frame_times, headers.len());
    let mut ordered = points.to_vec();
    ordered.sort_by_key(|point| integer(&point["frame_index"]).unwrap_or_default());
    let mut ratios = Vec::new();
    let minimum_step = 2.0_f64.max(psf_fwhm);
    for pair in ordered.windows(2) {
        let Some(previous_index) = integer(&pair[0]["frame_index"]) else {
            continue;
        };
        let Some(current_index) = integer(&pair[1]["frame_index"]) else {
            continue;
        };
        if previous_index < 0
            || current_index <= previous_index
            || current_index >= headers.len() as i64
            || previous_index >= headers.len() as i64
        {
            continue;
        }
        if pair[1]["touches_edge"].as_bool() == Some(true) {
            continue;
        }
        let previous_index = previous_index as usize;
        let current_index = current_index as usize;
        let elapsed = times[current_index] - times[previous_index];
        let distance = (field_number(&pair[1], "aligned_x")?
            - field_number(&pair[0], "aligned_x")?)
        .hypot(field_number(&pair[1], "aligned_y")? - field_number(&pair[0], "aligned_y")?);
        if !elapsed.is_finite()
            || elapsed <= 0.0
            || !distance.is_finite()
            || distance < minimum_step
        {
            continue;
        }
        let Some(motion_length) =
            deconvolved_motion_length(field_number(&pair[1], "length_px")?, psf_fwhm)
        else {
            continue;
        };
        let denominator = (distance * exposures[current_index]).max(f64::EPSILON);
        let ratio = motion_length * elapsed / denominator;
        if ratio.is_finite() && ratio > 0.0 {
            ratios.push(ratio);
        }
    }
    if ratios.len() < 2 {
        return Ok((None, None, None));
    }
    let logs: Vec<f64> = ratios.iter().map(|ratio| ratio.ln()).collect();
    let Some((median_log, log_mad)) = median_absolute_deviation(&logs) else {
        return Ok((None, None, None));
    };
    let ratio = median_log.exp();
    let in_range_count = ratios
        .iter()
        .filter(|value| (0.35..=2.50).contains(*value))
        .count();
    Ok((
        Some(ratio),
        Some(log_mad),
        Some(in_range_count as f64 / ratios.len() as f64),
    ))
}

fn motion_feature_exposure_supported(
    ratio: Option<f64>,
    log_mad: Option<f64>,
    consistency_fraction: Option<f64>,
) -> bool {
    let (Some(ratio), Some(log_mad), Some(consistency_fraction)) =
        (ratio, log_mad, consistency_fraction)
    else {
        return true;
    };
    (0.35..=2.50).contains(&ratio) && log_mad <= 0.45 && consistency_fraction >= 0.75
}

/// Measure line-length and inter-frame speed agreement from exposure metadata.
pub fn motion_feature_exposure_evidence(request: &Value) -> Result<Value> {
    let points = array(request, "points")?;
    let headers = array(request, "frame_headers")?;
    let frame_times = request
        .get("frame_times")
        .filter(|value| !value.is_null())
        .and_then(Value::as_array)
        .map(Vec::as_slice);
    let psf_fwhm = number(request, "psf_fwhm")?;
    let (ratio, log_mad, in_range_fraction) =
        motion_feature_exposure_metrics(points, headers, frame_times, psf_fwhm)?;
    Ok(json!({
        "ratio": ratio,
        "log_mad": log_mad,
        "in_range_fraction": in_range_fraction,
        "supported": motion_feature_exposure_supported(ratio, log_mad, in_range_fraction),
    }))
}

fn motion_feature_confirmation_presence(frame_count: i64) -> Result<i64> {
    if frame_count <= 0 {
        return Err("frame_count must be positive".into());
    }
    if frame_count < 8 {
        Ok(3_i64.min(frame_count))
    } else {
        Ok(3_i64.max((frame_count as f64 * 0.5).ceil() as i64))
    }
}

fn optional_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn motion_feature_rejection_reasons(track: &Value, points: &[Value]) -> Vec<&'static str> {
    if track.get("classification").and_then(Value::as_str) != Some("candidate") {
        return Vec::new();
    }
    if track.get("evidence_level").and_then(Value::as_str) == Some("single_frame_trail") {
        let mut reasons = vec!["SINGLE_FRAME_ONLY", "NO_CROSS_FRAME_ASSOCIATION"];
        if points.is_empty() {
            reasons.push("NO_GEOMETRIC_POINT");
        }
        return reasons;
    }

    let mut reasons = Vec::new();
    if points.len() < 2 {
        reasons.extend(["NO_CROSS_FRAME_ASSOCIATION", "INSUFFICIENT_FRAME_PRESENCE"]);
    } else if optional_bool(track, "presence_supported") == Some(false) {
        reasons.push(if optional_bool(track, "edge_truncated") == Some(true) {
            "EDGE_TRUNCATION_INSUFFICIENT_COVERAGE"
        } else {
            "INSUFFICIENT_FRAME_PRESENCE"
        });
    }
    if let Some(minimum) = optional_number(track, "min_displacement_px") {
        let displacement = optional_number(track, "displacement_px").unwrap_or(0.0);
        if !displacement.is_finite() || displacement < minimum {
            reasons.push("INSUFFICIENT_DISPLACEMENT");
        }
    }
    let fit_rms = optional_number(track, "fit_rms_px");
    if fit_rms.is_none_or(|value| !value.is_finite()) {
        reasons.push("NO_MOTION_FIT");
    } else if optional_number(track, "max_fit_rms_px")
        .is_some_and(|maximum| fit_rms.is_some_and(|value| value > maximum))
    {
        reasons.push("MOTION_FIT_RMS_TOO_HIGH");
    }
    if optional_bool(track, "shape_supported") == Some(false) {
        reasons.push("FRAME_SHAPE_INCONSISTENT");
    }
    if optional_bool(track, "model_supported") == Some(false) {
        let models: std::collections::BTreeSet<&str> = points
            .iter()
            .filter_map(|point| point.get("single_frame_model").and_then(Value::as_str))
            .collect();
        if models.contains("double") {
            reasons.push("COMPACT_DOUBLE_MODEL_PRESENT");
        }
        if models.contains("point") {
            reasons.push("COMPACT_POINT_MODEL_PRESENT");
        }
        if models.contains("elliptical_point") {
            reasons.push("COMPACT_ELLIPTICAL_MODEL_PRESENT");
        }
        reasons.push("RAW_LINE_MODEL_MAJORITY_NOT_MET");
    }
    if optional_bool(track, "motion_axis_supported") == Some(false) {
        reasons.push("MOTION_AXIS_MISMATCH");
    }
    if optional_bool(track, "temporal_continuity_supported") == Some(false) {
        reasons.push("FRAME_GAP_OR_NONCONTIGUOUS");
    }
    if optional_bool(track, "exposure_supported") == Some(false) {
        reasons.push("EXPOSURE_SPEED_INCONSISTENT");
    }
    if optional_bool(track, "artifact_supported") == Some(false) {
        reasons.push("LINE_ARTIFACT_OVERLAP");
    }
    if reasons.is_empty() {
        reasons.push("CONFIRMATION_GATES_NOT_SATISFIED");
    }
    let mut seen = std::collections::BTreeSet::new();
    reasons.retain(|reason| seen.insert(*reason));
    reasons
}

/// Return the required confirmation coverage and stable failure reason codes.
pub fn motion_feature_confirmation_evidence(request: &Value) -> Result<Value> {
    let frame_count = request
        .get("frame_count")
        .and_then(Value::as_i64)
        .ok_or_else(|| "missing frame_count".to_string())?;
    let required_presence = motion_feature_confirmation_presence(frame_count)?;
    let track = request
        .get("track")
        .ok_or_else(|| "missing track".to_string())?;
    let points = array(track, "points")?;
    let reasons = motion_feature_rejection_reasons(track, points);
    Ok(json!({
        "required_presence": required_presence,
        "rejection_reasons": reasons,
    }))
}

fn python_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(value) => value.parse().ok(),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn python_max(first: f64, second: f64) -> f64 {
    if second > first {
        second
    } else {
        first
    }
}

fn python_min(first: f64, second: f64) -> f64 {
    if second < first {
        second
    } else {
        first
    }
}

fn median_f64(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[middle - 1] + values[middle]) / 2.0)
    } else {
        Some(values[middle])
    }
}

fn sequence_motion_candidate_min_length(request: &Value) -> Result<f64> {
    if let Some(value) = request
        .get("min_length_px")
        .filter(|value| !value.is_null())
    {
        return python_float(value).ok_or_else(|| "min_length_px must be numeric or null".into());
    }
    let psf_fwhm = request
        .get("psf_fwhm")
        .and_then(python_float)
        .ok_or_else(|| "missing numeric psf_fwhm".to_string())?;
    let base = python_max(8.0, 3.0 * psf_fwhm);
    let metadata_supported = request
        .get("metadata_supported")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !metadata_supported || psf_fwhm <= 0.0 {
        return Ok(base);
    }
    let exposure_values = request
        .get("exposure_s")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing exposure_s array".to_string())?;
    let mut values = Vec::with_capacity(exposure_values.len());
    for value in exposure_values {
        let Some(number) = python_float(value) else {
            return Ok(base);
        };
        values.push(number);
    }
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Ok(base);
    }
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sorted = values.clone();
    let median = median_f64(&mut sorted).unwrap_or(0.0);
    if maximum - minimum > 0.05_f64.max(0.10 * median) || !(1.35..=1.65).contains(&median) {
        return Ok(base);
    }
    Ok(python_min(base, python_max(6.0, 2.5 * psf_fwhm)))
}

fn exposure_aware_short_trail_min_area(request: &Value) -> i64 {
    let Some(psf_fwhm) = request.get("psf_fwhm").and_then(python_float) else {
        return 20;
    };
    let Some(exposure_s) = request
        .get("short_exposure_s")
        .filter(|value| !value.is_null())
        .and_then(python_float)
    else {
        return 20;
    };
    if !psf_fwhm.is_finite()
        || psf_fwhm <= 0.0
        || !exposure_s.is_finite()
        || !(1.35..=1.65).contains(&exposure_s)
    {
        return 20;
    }
    let exposure_floor = if 1.5 * psf_fwhm * psf_fwhm >= 20.0 {
        20
    } else {
        12.max((1.5 * psf_fwhm * psf_fwhm).ceil() as i64)
    };
    20.min(exposure_floor)
}

/// Return metadata-aware raw sequence trail proposal thresholds.
pub fn motion_proposal_thresholds(request: &Value) -> Result<Value> {
    Ok(json!({
        "min_length_px": sequence_motion_candidate_min_length(request)?,
        "short_min_area": exposure_aware_short_trail_min_area(request),
    }))
}

fn bilinear(image: &[Vec<f32>], x: f64, y: f64) -> Option<f32> {
    let height = image.len();
    let width = image.first().map_or(0, Vec::len);
    if height == 0 || width == 0 || !x.is_finite() || !y.is_finite() {
        return None;
    }
    // scipy.ndimage.map_coordinates(mode="constant") returns cval for a
    // coordinate outside the center-to-center image domain.
    if x < 0.0 || y < 0.0 || x > (width - 1) as f64 || y > (height - 1) as f64 {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let dx = (x - x0 as f64) as f32;
    let dy = (y - y0 as f64) as f32;
    let top_left = image[y0][x0];
    let top_right = image[y0][x1];
    let bottom_left = image[y1][x0];
    let bottom_right = image[y1][x1];
    if [top_left, top_right, bottom_left, bottom_right]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Some(f32::NAN);
    }
    let top = f64::from(top_left) * (1.0 - f64::from(dx)) + f64::from(top_right) * f64::from(dx);
    let bottom =
        f64::from(bottom_left) * (1.0 - f64::from(dx)) + f64::from(bottom_right) * f64::from(dx);
    Some((top * (1.0 - f64::from(dy)) + bottom * f64::from(dy)) as f32)
}

/// Return the absolute continuous support ratio and relative profile fill.
/// These are morphological diagnostics, not motion confirmation.
pub fn line_profile_metrics(request: &Value) -> Result<Value> {
    let Some(image) = plane(request) else {
        return Ok(json!({"continuity_ratio":null,"profile_fill_ratio":null}));
    };
    let center_x = number(request, "center_x")?;
    let center_y = number(request, "center_y")?;
    let length = number(request, "length_px")?;
    let width = number(request, "width_px")?;
    let angle_deg = number(request, "angle_deg")?;
    let threshold = number(request, "threshold_adu")?;
    if image.is_empty() || image[0].is_empty() || length <= 0.0 || threshold <= 0.0 {
        return Ok(json!({"continuity_ratio":null,"profile_fill_ratio":null}));
    }
    if !length.is_finite()
        || !center_x.is_finite()
        || !center_y.is_finite()
        || !angle_deg.is_finite()
    {
        return Ok(json!({"continuity_ratio":null,"profile_fill_ratio":null}));
    }

    let sample_count = ((length * 2.0).ceil() as usize + 1).max(25);
    let half_width = (0.5 * width + 1.0).clamp(1.5, 4.0);
    let angle = angle_deg.to_radians();
    let unit_x = angle.cos();
    let unit_y = angle.sin();
    let normal_x = -unit_y;
    let normal_y = unit_x;
    let mut profile = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let fraction = index as f64 / (sample_count - 1) as f64;
        let along = (fraction - 0.5) * length;
        let mut best = f32::NEG_INFINITY;
        for across_index in 0..9 {
            let across = -half_width + (2.0 * half_width) * across_index as f64 / 8.0;
            let x = center_x + along * unit_x + across * normal_x;
            let y = center_y + along * unit_y + across * normal_y;
            if let Some(value) = bilinear(&image, x, y).filter(|value| value.is_finite()) {
                best = best.max(value);
            }
        }
        profile.push(best);
    }
    let valid: Vec<bool> = profile.iter().map(|value| value.is_finite()).collect();
    let valid_count = valid.iter().filter(|value| **value).count();
    if valid_count == 0 {
        return Ok(json!({"continuity_ratio":null,"profile_fill_ratio":null}));
    }
    let support: Vec<bool> = profile
        .iter()
        .zip(&valid)
        .map(|(value, is_valid)| *is_valid && f64::from(*value) >= threshold)
        .collect();
    let mut longest = 0usize;
    let mut current = 0usize;
    for index in 0..support.len() {
        let bridged = support[index]
            || (index > 0 && index + 1 < support.len() && support[index - 1] && support[index + 1]);
        if bridged {
            current += 1;
            longest = longest.max(current);
        } else {
            current = 0;
        }
    }
    let peak = profile
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(f32::total_cmp)
        .unwrap_or(0.0);
    let relative_threshold = threshold.max(0.25 * f64::from(peak));
    let relative_count = profile
        .iter()
        .zip(&valid)
        .filter(|(value, is_valid)| **is_valid && f64::from(**value) >= relative_threshold)
        .count();
    Ok(json!({
        "continuity_ratio": longest as f64 / valid_count as f64,
        "profile_fill_ratio": relative_count as f64 / valid_count as f64,
    }))
}
