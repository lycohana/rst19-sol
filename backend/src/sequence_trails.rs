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
