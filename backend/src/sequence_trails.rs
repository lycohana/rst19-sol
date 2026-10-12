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
    let Some(times) = values.iter().map(Value::as_f64).collect::<Option<Vec<_>>>() else {
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
