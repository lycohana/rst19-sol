//! Auditable image-plane kinematics and innovation-report measurements.
use crate::Result;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn number(value: &Value) -> Option<f64> {
    let number = match value {
        Value::Bool(value) => Some(u8::from(*value) as f64),
        Value::String(value) => value.trim().parse().ok(),
        _ => value.as_f64(),
    }?;
    number.is_finite().then_some(number)
}

fn integer(value: &Value) -> Option<i64> {
    match value {
        Value::Bool(value) => Some(i64::from(*value)),
        Value::String(value) => value.trim().parse().ok(),
        _ => value.as_i64().or_else(|| {
            value
                .as_f64()
                .filter(|value| {
                    value.is_finite() && *value >= i64::MIN as f64 && *value < i64::MAX as f64
                })
                .map(|value| value.trunc() as i64)
        }),
    }
}

fn timestamp(value: &Value) -> Option<i128> {
    value
        .as_str()
        .and_then(crate::sequence_trails::parse_utc_microseconds)
}

fn timestamp_text(value: &Value) -> Option<String> {
    let Some(micros) = timestamp(value) else {
        return match value {
            Value::Null => None,
            Value::String(text) => Some(text.clone()),
            Value::Bool(value) => Some(if *value { "True" } else { "False" }.into()),
            _ => Some(value.to_string()),
        };
    };
    // Invert the Gregorian day count used by the shared timestamp parser.
    let days = micros.div_euclid(86_400_000_000);
    let era = days.div_euclid(146_097);
    let doe = days - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i128::from(month <= 2);
    let clock = micros.rem_euclid(86_400_000_000);
    let second = clock / 1_000_000;
    let fraction = clock % 1_000_000;
    let mut text = format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}",
        second / 3600,
        second % 3600 / 60,
        second % 60
    );
    if fraction != 0 {
        text.push_str(&format!(".{fraction:06}"));
    }
    Some(text)
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn resolve_frame_path(raw: &Path, sequence_json: &Path) -> Result<PathBuf> {
    if raw.is_absolute() && raw.is_file() {
        return Ok(raw.to_path_buf());
    }
    let current = std::env::current_dir().map_err(|error| error.to_string())?;
    let mut bases = vec![current.clone()];
    let mut parent = sequence_json.parent();
    for _ in 0..3 {
        if let Some(path) = parent {
            bases.push(path.to_path_buf());
            parent = path.parent();
        }
    }
    for base in bases {
        let candidate = base.join(raw);
        if candidate.is_file() {
            return std::fs::canonicalize(candidate).map_err(|error| error.to_string());
        }
    }
    Ok(current.join(raw))
}

pub fn robust_image_stats(frame: &crate::models::FitsFrame) -> Result<Value> {
    let valid = (0..frame.data.len())
        .filter(|index| !(*index < frame.width && *index < 104))
        .map(|index| frame.data.value(index))
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return Ok(
            json!({"valid_pixels":0,"background_median_adu":null,"background_mad_adu":null,"background_rms_adu":null,"p01_adu":null,"p50_adu":null,"p99_adu":null,"p998_adu":null,"raw_max_adu":null,"zero_fraction":null}),
        );
    }
    let median = crate::statistics::median(&valid)?;
    let mad = crate::statistics::median(
        &valid
            .iter()
            .map(|value| (value - median).abs())
            .collect::<Vec<_>>(),
    )?;
    Ok(
        json!({"valid_pixels":valid.len(),"background_median_adu":median,"background_mad_adu":mad,"background_rms_adu":1.4826*mad,
        "p01_adu":crate::statistics::percentile(&valid,1.0)?,"p50_adu":median,"p99_adu":crate::statistics::percentile(&valid,99.0)?,"p998_adu":crate::statistics::percentile(&valid,99.8)?,
        "raw_max_adu":valid.iter().copied().max_by(f64::total_cmp),"zero_fraction":valid.iter().filter(|value|**value == 0.0).count() as f64/valid.len() as f64}),
    )
}

fn vector(row: &Value, prefix: &str, velocity: bool) -> Option<[f64; 3]> {
    let suffix = if velocity { "v" } else { "" };
    Some([
        number(&row[format!("{prefix}_x{suffix}")])?,
        number(&row[format!("{prefix}_y{suffix}")])?,
        number(&row[format!("{prefix}_z{suffix}")])?,
    ])
}

fn norm(vector: [f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

pub fn vector_metrics(row: &Value) -> Value {
    let mut metrics = serde_json::Map::new();
    for prefix in ["j2000", "wgs84"] {
        for (kind, velocity) in [("position", false), ("velocity", true)] {
            let value = vector(row, prefix, velocity);
            metrics.insert(format!("{prefix}_{kind}_norm_raw"), json!(value.map(norm)));
            metrics.insert(
                format!("{prefix}_{kind}_azimuth_deg"),
                json!(value.map(|v| v[1].atan2(v[0]).to_degrees())),
            );
            metrics.insert(
                format!("{prefix}_{kind}_elevation_deg"),
                json!(value.map(|v| v[2].atan2(v[0].hypot(v[1])).to_degrees())),
            );
        }
    }
    Value::Object(metrics)
}

pub fn frame_rows(payload: &Value, sequence_json: &Path) -> Result<Vec<Value>> {
    let summaries = payload["frames"]
        .as_array()
        .ok_or("sequence JSON 缺少 frames 列表")?;
    let mut rows = Vec::with_capacity(summaries.len());
    for (index, summary) in summaries.iter().enumerate() {
        if !summary.is_object() {
            return Err(format!("frames[{index}] 不是对象"));
        }
        let path = resolve_frame_path(
            Path::new(summary["path"].as_str().unwrap_or("")),
            sequence_json,
        )?;
        let frame = crate::fits::read_fits(&path, false)?;
        let raw_time = summary
            .get("timestamp")
            .filter(|value| truthy(value))
            .or_else(|| frame.header.get("DATE-OBS"))
            .unwrap_or(&Value::Null);
        let auxiliary = summary["auxiliary"]
            .as_object()
            .cloned()
            .unwrap_or_else(|| {
                frame
                    .auxiliary
                    .as_ref()
                    .map(|value| value.as_dict())
                    .unwrap_or_default()
            });
        let shift = payload["cumulative_shifts"][index].as_array();
        let sx = shift
            .and_then(|shift| shift.first())
            .map_or(Some(0.0), number);
        let sy = shift
            .and_then(|shift| shift.get(1))
            .map_or(Some(0.0), number);
        let exposure = match number(&summary["exposure_ms"]) {
            Some(value) => Some(value),
            None => crate::fits::exposure_milliseconds(&frame.header)?,
        };
        let mut row = json!({"frame_index":summary["frame_index"].as_i64().unwrap_or(index as i64),"path":path.to_string_lossy(),"timestamp":timestamp_text(raw_time),"exposure_ms":exposure,
            "width_px":frame.width,"height_px":frame.height,"candidate_count":summary["candidate_count"].as_u64().unwrap_or(0),"returned_count":summary["returned_count"].as_u64().unwrap_or(0),"quality_count":summary["quality_count"].as_u64().unwrap_or(0),
            "cumulative_shift_x_px":sx,"cumulative_shift_y_px":sy,"cumulative_shift_norm_px":sx.unwrap_or(0.0).hypot(sy.unwrap_or(0.0))});
        row.as_object_mut()
            .unwrap()
            .extend(robust_image_stats(&frame)?.as_object().unwrap().clone());
        row.as_object_mut().unwrap().extend(auxiliary);
        let metrics = vector_metrics(&row);
        row.as_object_mut()
            .unwrap()
            .extend(metrics.as_object().unwrap().clone());
        rows.push(row);
    }
    for prefix in ["j2000", "wgs84"] {
        let first = rows.first().and_then(|row| vector(row, prefix, false));
        for row in &mut rows {
            let delta = vector(row, prefix, false)
                .zip(first)
                .map(|(current, first)| {
                    [
                        current[0] - first[0],
                        current[1] - first[1],
                        current[2] - first[2],
                    ]
                });
            row[format!("{prefix}_position_delta_norm_raw")] = json!(delta.map(norm));
        }
    }
    let times = rows
        .iter()
        .map(|row| timestamp(&row["timestamp"]))
        .collect::<Vec<_>>();
    let first = times.iter().find_map(|time| *time);
    for (index, row) in rows.iter_mut().enumerate() {
        row["elapsed_s"] = json!(times[index]
            .zip(first)
            .map(|(time, first)| (time - first) as f64 / 1_000_000.0));
        row["interval_s"] = json!(if index == 0 {
            None
        } else {
            times[index]
                .zip(times[index - 1])
                .map(|(time, last)| (time - last) as f64 / 1_000_000.0)
        });
    }
    Ok(rows)
}

fn summary(values: &[f64]) -> Result<(Option<f64>, Option<f64>, Option<f64>)> {
    if values.is_empty() {
        Ok((None, None, None))
    } else {
        Ok((
            values.iter().copied().min_by(f64::total_cmp),
            Some(crate::statistics::median(values)?),
            values.iter().copied().max_by(f64::total_cmp),
        ))
    }
}

pub fn relative_rates(rows: &[Value], fields: &[&str]) -> Result<Value> {
    let times = rows
        .iter()
        .map(|row| timestamp(&row["timestamp"]))
        .collect::<Vec<_>>();
    let mut result = serde_json::Map::new();
    for field in fields {
        let mut rates = Vec::new();
        for index in 1..rows.len() {
            if let Some(((left, right), (before, after))) = times[index - 1]
                .zip(times[index])
                .zip(number(&rows[index - 1][*field]).zip(number(&rows[index][*field])))
            {
                if right > left {
                    rates.push((after - before) / ((right - left) as f64 / 1_000_000.0));
                }
            }
        }
        let (min, median, max) = summary(&rates)?;
        result.insert(
            (*field).into(),
            json!({"min_per_s":min,"max_per_s":max,"median_per_s":median}),
        );
    }
    Ok(Value::Object(result))
}

pub fn telemetry_consistency(rows: &mut [Value], prefix: &str) -> Result<Value> {
    let times = rows
        .iter()
        .map(|row| timestamp(&row["timestamp"]))
        .collect::<Vec<_>>();
    let positions = rows
        .iter()
        .map(|row| vector(row, prefix, false))
        .collect::<Vec<_>>();
    let velocities = rows
        .iter()
        .map(|row| vector(row, prefix, true))
        .collect::<Vec<_>>();
    let rate_key = format!("{prefix}_position_rate_norm_m_per_s");
    let residual_key = format!("{prefix}_velocity_consistency_residual_norm_assuming_m_per_s");
    let error_key = format!("{prefix}_velocity_consistency_relative_error_assuming_m_per_s");
    for row in rows.iter_mut() {
        for key in [&rate_key, &residual_key, &error_key] {
            row[key] = Value::Null;
        }
    }
    let mut rates = Vec::new();
    let mut provided = Vec::new();
    let mut residuals = Vec::new();
    let mut errors = Vec::new();
    let mut interior = Vec::new();
    for index in 0..rows.len() {
        if rows.len() < 2 {
            continue;
        }
        let (left, right) = if index == 0 {
            (0, 1)
        } else if index + 1 == rows.len() {
            (index - 1, index)
        } else {
            (index - 1, index + 1)
        };
        let Some((tl, tr)) = times[left].zip(times[right]) else {
            continue;
        };
        if tr <= tl || positions[index].is_none() {
            continue;
        }
        let Some(((before, after), velocity)) =
            positions[left].zip(positions[right]).zip(velocities[index])
        else {
            continue;
        };
        let dt = (tr - tl) as f64 / 1_000_000.0;
        let rate = [
            (after[0] - before[0]) / dt,
            (after[1] - before[1]) / dt,
            (after[2] - before[2]) / dt,
        ];
        let rate_norm = norm(rate);
        let provided_norm = norm(velocity);
        let residual_norm = norm([
            rate[0] - velocity[0],
            rate[1] - velocity[1],
            rate[2] - velocity[2],
        ]);
        let error = (provided_norm > 1e-12).then(|| residual_norm / provided_norm);
        rates.push(rate_norm);
        provided.push(provided_norm);
        residuals.push(residual_norm);
        if let Some(error) = error {
            errors.push(error);
            if index > 0 && index + 1 < rows.len() {
                interior.push(error);
            }
        }
        rows[index][&rate_key] = json!(rate_norm);
        rows[index][&residual_key] = json!(residual_norm);
        rows[index][&error_key] = json!(error);
    }
    let (rmin, rmed, rmax) = summary(&rates)?;
    let (vmin, vmed, vmax) = summary(&provided)?;
    let (emin, emed, emax) = summary(&residuals)?;
    let (_, relative_median, relative_max) = summary(&errors)?;
    let (_, interior_median, interior_max) = summary(&interior)?;
    Ok(
        json!({"sample_count":rows.len(),"valid_sample_count":rates.len(),"interior_sample_count":interior.len(),
        "difference_method":"首帧/末帧使用一侧差分，中间帧使用中心差分；position(m)/time(s) 得到 m/s",
        "position_rate_norm_m_per_s_min":rmin,"position_rate_norm_m_per_s_median":rmed,"position_rate_norm_m_per_s_max":rmax,
        "provided_velocity_norm_doc_unit_min":vmin,"provided_velocity_norm_doc_unit_median":vmed,"provided_velocity_norm_doc_unit_max":vmax,
        "residual_norm_assuming_m_per_s_min":emin,"residual_norm_assuming_m_per_s_median":emed,"residual_norm_assuming_m_per_s_max":emax,
        "relative_error_assuming_m_per_s_median":relative_median,"relative_error_assuming_m_per_s_max":relative_max,
        "interior_relative_error_assuming_m_per_s_median":interior_median,"interior_relative_error_assuming_m_per_s_max":interior_max,
        "velocity_unit_hypothesis":"m/s","velocity_unit_status":"由位置(m)/DATE-OBS(s) 差分高度支持；格式说明速度字段原文写 m，待主办方确认"}),
    )
}

pub fn telemetry_prediction(rows: &[Value], prefix: &str, horizon_s: Option<f64>) -> Value {
    let mut result = json!({"prefix":prefix,"method":"last_state_constant_velocity","horizon_s":horizon_s,"frame_index":null,"status":"unavailable",
        "last_position_m":null,"last_velocity_m_per_s_assumed":null,"predicted_position_m_assuming_m_per_s":null,
        "predicted_position_norm_m_assuming_m_per_s":null,"velocity_azimuth_deg":null,"velocity_elevation_deg":null,"unit_status":"速度单位待主办方确认"});
    let Some(horizon) = horizon_s.filter(|horizon| horizon.is_finite() && *horizon >= 0.0) else {
        return result;
    };
    let Some(last) = rows.last() else {
        return result;
    };
    let Some((position, velocity)) = vector(last, prefix, false).zip(vector(last, prefix, true))
    else {
        return result;
    };
    let predicted = [
        position[0] + velocity[0] * horizon,
        position[1] + velocity[1] * horizon,
        position[2] + velocity[2] * horizon,
    ];
    result["frame_index"] = json!(integer(&last["frame_index"]).unwrap_or(rows.len() as i64 - 1));
    result["status"] = json!("available");
    result["last_position_m"] = json!(position);
    result["last_velocity_m_per_s_assumed"] = json!(velocity);
    result["predicted_position_m_assuming_m_per_s"] = json!(predicted);
    result["predicted_position_norm_m_assuming_m_per_s"] = json!(norm(predicted));
    result["velocity_azimuth_deg"] = json!(velocity[1].atan2(velocity[0]).to_degrees());
    result["velocity_elevation_deg"] = json!(velocity[2]
        .atan2(velocity[0].hypot(velocity[1]))
        .to_degrees());
    result
}

pub fn motion_audit_rows(payload: &Value) -> Vec<Value> {
    let mut rows = Vec::new();
    for audit in payload["motion_frame_audits"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|value| value.is_object())
    {
        let mut row = audit.clone();
        let index = integer(&row["frame_index"]).unwrap_or(rows.len() as i64);
        row["frame_index"] = json!(index);
        row["frame_number"] = json!(index.saturating_add(1));
        rows.push(row);
    }
    rows
}

pub fn motion_evidence_link_rows(payload: &Value) -> Vec<Value> {
    let mut rows = Vec::new();
    for link in payload["motion_evidence_links"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|value| value.is_object())
    {
        let Some((point, line, count)) = integer(&link["point_track_id"])
            .zip(integer(&link["line_track_id"]))
            .zip(integer(&link["overlap_frame_count"]))
            .map(|((point, line), count)| (point, line, count))
        else {
            continue;
        };
        let Some((fraction, distance, direction)) = number(&link["overlap_fraction"])
            .zip(number(&link["median_perpendicular_distance_px"]))
            .zip(number(&link["direction_difference_deg"]))
            .map(|((fraction, distance), direction)| (fraction, distance, direction))
        else {
            continue;
        };
        if point < 0 || line < 0 || count < 0 {
            continue;
        }
        rows.push(json!({"point_track_id":point,"line_track_id":line,"overlap_frame_count":count,"overlap_fraction":fraction,"median_perpendicular_distance_px":distance,"direction_difference_deg":direction}));
    }
    rows
}

pub fn telemetry_prediction_rows(predictions: &Value) -> Vec<Value> {
    ["j2000","wgs84"].iter().map(|prefix| {
        let prediction = &predictions[*prefix];
        let position = &prediction["last_position_m"];
        let velocity = &prediction["last_velocity_m_per_s_assumed"];
        let forecast = &prediction["predicted_position_m_assuming_m_per_s"];
        json!({"reference_frame":prefix.to_uppercase(),"status":prediction["status"],"frame_index":prediction["frame_index"],"horizon_s":prediction["horizon_s"],
            "last_x_m":position[0],"last_y_m":position[1],"last_z_m":position[2],
            "velocity_x_m_per_s_assumed":velocity[0],"velocity_y_m_per_s_assumed":velocity[1],"velocity_z_m_per_s_assumed":velocity[2],
            "predicted_x_m_assuming_m_per_s":forecast[0],"predicted_y_m_assuming_m_per_s":forecast[1],"predicted_z_m_assuming_m_per_s":forecast[2],
            "predicted_position_norm_m_assuming_m_per_s":prediction["predicted_position_norm_m_assuming_m_per_s"],"velocity_azimuth_deg":prediction["velocity_azimuth_deg"],"velocity_elevation_deg":prediction["velocity_elevation_deg"],"unit_status":prediction["unit_status"]})
    }).collect()
}

fn motion_rows_impl(payload: &Value, frames: &[Value], point_mode: bool) -> Result<Vec<Value>> {
    let times = frames
        .iter()
        .map(|row| timestamp(&row["timestamp"]))
        .collect::<Vec<_>>();
    let intervals = frames
        .iter()
        .filter_map(|row| number(&row["interval_s"]))
        .filter(|interval| *interval > 0.0)
        .collect::<Vec<_>>();
    let horizon = if intervals.is_empty() {
        None
    } else {
        Some(5.0 * crate::statistics::median(&intervals)?)
    };
    let mut result = Vec::new();
    let key = if point_mode {
        "tracks"
    } else {
        "motion_features"
    };
    for track in payload[key]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|track| track.is_object())
    {
        if point_mode && track["classification"] != "moving" {
            continue;
        }
        let mut points = track["points"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|point| point.is_object())
            .cloned()
            .collect::<Vec<_>>();
        if points.is_empty() {
            continue;
        }
        points.sort_by_key(|point| integer(&point["frame_index"]).unwrap_or(0));
        let indices = points
            .iter()
            .map(|point| integer(&point["frame_index"]).unwrap_or(0))
            .collect::<Vec<_>>();
        let at_time = |index: i64| {
            usize::try_from(index)
                .ok()
                .and_then(|index| times.get(index).copied().flatten())
        };
        let first = indices[0];
        let last = *indices.last().unwrap();
        let endpoints = at_time(first).zip(at_time(last));
        let duration = endpoints.map(|(start, end)| (end - start) as f64 / 1_000_000.0);
        let coordinates = |aligned: &str, raw: &str| {
            points
                .iter()
                .map(|point| {
                    let value = point.get(aligned).or_else(|| point.get(raw));
                    value.map_or(Ok(0.0), |value| {
                        number(value)
                            .ok_or_else(|| format!("motion point requires finite {aligned}/{raw}"))
                    })
                })
                .collect::<Result<Vec<_>>>()
        };
        let x = coordinates("aligned_x", "x")?;
        let y = coordinates("aligned_y", "y")?;
        let all_times = indices.iter().all(|index| at_time(*index).is_some());
        let fit = if points.len() >= 2
            && duration.is_some_and(|duration| duration > 0.0)
            && (!point_mode || all_times)
        {
            let start = endpoints.unwrap().0;
            let elapsed = indices
                .iter()
                .enumerate()
                .map(|(index, frame)| {
                    at_time(*frame).map_or(index as f64, |time| (time - start) as f64 / 1_000_000.0)
                })
                .collect::<Vec<_>>();
            Some(fit_constant_velocity(&elapsed, &x, &y, horizon)?)
        } else {
            None
        };
        let fitted = fit.as_ref().unwrap_or(&Value::Null);
        let speed = number(&fitted["speed_px_per_s"]);
        let displacement = speed
            .zip(duration)
            .map(|(speed, duration)| speed * duration)
            .or_else(|| number(&track["displacement_px"]));
        let mut point_records = points.clone();
        for (index, point) in point_records.iter_mut().enumerate() {
            point["fit_residual_px"] = fitted["fit_residual_px"][index].clone();
        }
        let mut row = json!({"track_id":integer(&track["track_id"]).unwrap_or(result.len() as i64),"classification":track["classification"].as_str().unwrap_or("candidate"),
            "presence":points.len(),"first_frame":first,"last_frame":last,"duration_s":duration,"displacement_px":displacement,"speed_px_per_frame":number(&track["speed_px_per_frame"]),
            "speed_px_per_s":speed,"velocity_x_px_per_s":fitted["velocity_x_px_per_s"],"velocity_y_px_per_s":fitted["velocity_y_px_per_s"],
            "speed_ci95_low_px_per_s":fitted["speed_ci95_low_px_per_s"],"speed_ci95_high_px_per_s":fitted["speed_ci95_high_px_per_s"],
            "direction_deg_image":fitted["direction_deg_image"],"direction_ci95_half_width_deg":fitted["direction_ci95_half_width_deg"],
            "fit_rms_px":fit.as_ref().map_or_else(||json!(number(&track["fit_rms_px"])),|fit|fit["fit_rms_px"].clone()),
            "kinematic_model":fit.as_ref().map(|_|"constant_velocity_ols_date_obs"),"uncertainty_method":fit.as_ref().and_then(|fit|(!fit["t95_critical"].is_null()).then_some("student_t_95_delta_method_independent_xy")),
            "fit_degrees_of_freedom":fitted["degrees_of_freedom"],"start_x_px":x[0],"start_y_px":y[0],"end_x_px":x[x.len()-1],"end_y_px":y[y.len()-1],
            "fitted_start_x_px":fitted["fitted_x_px"][0],"fitted_start_y_px":fitted["fitted_y_px"][0],"fitted_end_x_px":fitted["fitted_x_px"][points.len()-1],"fitted_end_y_px":fitted["fitted_y_px"][points.len()-1],
            "prediction_horizon_s":fit.as_ref().and(horizon),"predicted_x_px":fitted["predicted_x_px"],"predicted_y_px":fitted["predicted_y_px"],"predicted_x_ci95_half_width_px":fitted["predicted_x_ci95_half_width_px"],"predicted_y_ci95_half_width_px":fitted["predicted_y_ci95_half_width_px"],
            "frame_indices":indices.iter().map(|index|(index+1).to_string()).collect::<Vec<_>>().join(","),"points":point_records});
        if point_mode {
            row["evidence_level"] =
                json!(track["evidence_level"].as_str().unwrap_or("quality_track"));
            for (field, source) in [
                ("median_flux_snr", "flux_snr"),
                ("median_candidate_snr", "candidate_snr"),
            ] {
                let values = points
                    .iter()
                    .filter_map(|point| number(&point[source]))
                    .collect::<Vec<_>>();
                row[field] = json!(if values.is_empty() {
                    None
                } else {
                    Some(crate::statistics::median(&values)?)
                });
            }
        } else {
            for (median_field, max_field, source) in [
                ("median_length_px", "max_length_px", "length_px"),
                ("median_width_px", "", "width_px"),
            ] {
                let values = points
                    .iter()
                    .map(|point| {
                        point.get(source).map_or(Ok(0.0), |value| {
                            number(value).ok_or_else(|| format!("invalid {source}"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                row[median_field] = json!(crate::statistics::median(&values)?);
                if !max_field.is_empty() {
                    row[max_field] = json!(values.iter().copied().max_by(f64::total_cmp));
                }
            }
            let snrs = points
                .iter()
                .map(|point| {
                    point.get("residual_snr").map_or(Ok(0.0), |value| {
                        number(value).ok_or_else(|| "invalid residual_snr".to_string())
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            row["max_residual_snr"] = json!(snrs.iter().copied().max_by(f64::total_cmp));
            row["velocity_x_se_px_per_s"] = fitted["velocity_x_se_px_per_s"].clone();
            row["velocity_y_se_px_per_s"] = fitted["velocity_y_se_px_per_s"].clone();
            row["speed_se_px_per_s"] = fitted["speed_se_px_per_s"].clone();
            row["fit_t95_critical"] = fitted["t95_critical"].clone();
            row["uncertainty_note"] = fit.as_ref().map_or_else(
                || json!("单帧或两点轨迹不能估计小样本拟合不确定度。"),
                |fit| fit["uncertainty_note"].clone(),
            );
            if fit.is_none() {
                row["direction_deg_image"] = if points.len() == 1 {
                    json!(number(&points[0]["angle_deg"]))
                } else {
                    json!((y[y.len() - 1] - y[0])
                        .atan2(x[x.len() - 1] - x[0])
                        .to_degrees())
                };
            }
        }
        result.push(row);
    }
    Ok(result)
}

pub fn motion_rows(payload: &Value, frames: &[Value]) -> Result<Vec<Value>> {
    motion_rows_impl(payload, frames, false)
}
pub fn point_motion_rows(payload: &Value, frames: &[Value]) -> Result<Vec<Value>> {
    motion_rows_impl(payload, frames, true)
}

pub fn build_innovation_report_from_payload(payload: &Value, source_hint: &Path) -> Result<Value> {
    if !payload.is_object() {
        return Err("sequence JSON 根节点必须是对象".into());
    }
    let mut frames = frame_rows(payload, source_hint)?;
    let lines = motion_rows(payload, &frames)?;
    let points = point_motion_rows(payload, &frames)?;
    let audits = motion_audit_rows(payload);
    let links = motion_evidence_link_rows(payload);
    let moving_lines = lines
        .iter()
        .filter(|row| row["classification"] == "moving")
        .count();
    let groups = (moving_lines + points.len()).saturating_sub(links.len());
    let intervals = frames
        .iter()
        .filter_map(|row| number(&row["interval_s"]))
        .filter(|interval| *interval > 0.0)
        .collect::<Vec<_>>();
    let interval_median = if intervals.is_empty() {
        None
    } else {
        Some(crate::statistics::median(&intervals)?)
    };
    let interval_std = if intervals.is_empty() {
        None
    } else {
        let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
        Some(
            (intervals
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / intervals.len() as f64)
                .sqrt(),
        )
    };
    let horizon = interval_median.map(|interval| 5.0 * interval);
    let mut attitude = serde_json::Map::new();
    for field in ["roll", "pitch", "yaw"] {
        let values = frames
            .iter()
            .filter_map(|row| number(&row[field]))
            .collect::<Vec<_>>();
        attitude.insert(field.into(),json!({"start":frames.first().and_then(|row|number(&row[field])),"end":frames.last().and_then(|row|number(&row[field])),"min":values.iter().copied().min_by(f64::total_cmp),"max":values.iter().copied().max_by(f64::total_cmp)}));
    }
    let quaternion = frames
        .iter()
        .map(|row| {
            ["q1", "q2", "q3", "q4"]
                .iter()
                .map(|key| number(&row[*key]).unwrap_or(0.0).powi(2))
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let mut telemetry = serde_json::Map::new();
    let mut consistency = serde_json::Map::new();
    let mut predictions = serde_json::Map::new();
    for prefix in ["j2000", "wgs84"] {
        let velocities = frames
            .iter()
            .filter_map(|row| number(&row[format!("{prefix}_velocity_norm_raw")]))
            .collect::<Vec<_>>();
        let azimuths = frames
            .iter()
            .filter_map(|row| number(&row[format!("{prefix}_velocity_azimuth_deg")]))
            .collect::<Vec<_>>();
        let position_key = format!("{prefix}_position_norm_raw");
        let delta_key = format!("{prefix}_position_delta_norm_raw");
        telemetry.insert(prefix.into(),json!({"velocity_norm_raw_min":velocities.iter().copied().min_by(f64::total_cmp),"velocity_norm_raw_max":velocities.iter().copied().max_by(f64::total_cmp),"velocity_azimuth_deg_min":azimuths.iter().copied().min_by(f64::total_cmp),"velocity_azimuth_deg_max":azimuths.iter().copied().max_by(f64::total_cmp),"start_position_norm_raw":frames.first().map(|row|row[&position_key].clone()),"end_position_norm_raw":frames.last().map(|row|row[&position_key].clone()),"end_position_delta_norm_raw":frames.last().map(|row|row[&delta_key].clone())}));
        consistency.insert(prefix.into(), telemetry_consistency(&mut frames, prefix)?);
        predictions.insert(
            prefix.into(),
            telemetry_prediction(&frames, prefix, horizon),
        );
    }
    let reference = payload["motion_reference_mode"]
        .as_str()
        .unwrap_or("per_frame_background");
    let reference_note = match reference {
        "registered_median" => "插值注册后的时间中值参考差分；差分噪声按 sqrt(2) 倍单帧噪声估计。",
        "temporal_median_small_shift" => {
            "累计平移小于 0.75 px 时的原坐标时间中值近似差分；差分噪声按 sqrt(2) 倍单帧噪声估计。"
        }
        _ => "单帧背景差分；没有跨帧参考，因此线状结果只能作为形状候选。",
    };
    let shapes = frames
        .iter()
        .map(|row| format!("{}x{}", row["width_px"], row["height_px"]))
        .collect::<std::collections::BTreeSet<_>>();
    let exposures = frames
        .iter()
        .filter_map(|row| number(&row["exposure_ms"]))
        .collect::<Vec<_>>();
    let mut exposures = exposures;
    exposures.sort_by(f64::total_cmp);
    exposures.dedup();
    let mut quality = json!({"source_working_limit":payload["source_working_limit"],"calculation_dtype":payload["calculation_dtype"].as_str().unwrap_or("float64"),"background_model_mode":payload["background_model_mode"].as_str().unwrap_or("per_frame_local"),
        "note":if !payload["source_working_limit"].is_null() { "candidate_count 是全量候选审计；returned_count、quality_count 和点轨迹只在每帧最高 SNR 工作集内统计。" } else { "当前序列使用全量源级工作集。统计排除了首行 208 字节辅助区；未设置饱和阈值，不报告饱和率。" }});
    for field in ["candidate_count", "returned_count", "quality_count"] {
        let values = frames
            .iter()
            .filter_map(|row| number(&row[field]))
            .collect::<Vec<_>>();
        quality[format!("{field}_min")] = json!(values.iter().copied().min_by(f64::total_cmp));
        quality[format!("{field}_max")] = json!(values.iter().copied().max_by(f64::total_cmp));
    }
    let noise = frames
        .iter()
        .filter_map(|row| number(&row["background_rms_adu"]))
        .collect::<Vec<_>>();
    quality["background_rms_median_adu"] = json!(if noise.is_empty() {
        None
    } else {
        Some(crate::statistics::median(&noise)?)
    });
    let motion_note = format!("{reference_note} 点源 moving 与线状 moving 分开检测；evidence_group_count = 点源 moving + 线状 moving − 跨通道重复关联。关联只用于审计去重，不删除原始轨迹，也不等于物理目标真值。speed_px_per_s 是图像平面速度；三点及以上轨迹附带小样本 OLS 速度、方向和轨迹均值外推的 95% 区间。区间不包含配准系统误差、模型失配或未来单次观测噪声；没有像元尺度/WCS 时不解释为真实天体速度。");
    let point_fast = points
        .iter()
        .filter(|row| row["evidence_level"] == "fast_point_motion")
        .count();
    Ok(
        json!({"schema_version":2,"source_sequence_json":source_hint.to_string_lossy(),
        "relation":{"frame_count":frames.len(),"same_shape":!frames.is_empty() && shapes.len() == 1,"shapes":shapes,"exposure_ms":exposures,"start_timestamp":frames.first().map(|row|row["timestamp"].clone()),"end_timestamp":frames.last().map(|row|row["timestamp"].clone()),"duration_s":frames.iter().rev().find_map(|row|number(&row["elapsed_s"])),"interval_median_s":interval_median,"interval_std_s":interval_std,"intervals_s":intervals},
        "registration":{"final_cumulative_shift_x_px":frames.last().map(|row|row["cumulative_shift_x_px"].clone()),"final_cumulative_shift_y_px":frames.last().map(|row|row["cumulative_shift_y_px"].clone()),"max_cumulative_shift_norm_px":frames.iter().filter_map(|row|number(&row["cumulative_shift_norm_px"])).max_by(f64::total_cmp),"note":"当前为平移配准基线；无标准 WCS 时不换算为角秒或姿态角。"},
        "attitude_auxiliary":{"ranges":attitude,"rates":relative_rates(&frames,&["roll","pitch","yaw"])?,"quaternion_norm_range":{"min":quaternion.iter().copied().min_by(f64::total_cmp),"max":quaternion.iter().copied().max_by(f64::total_cmp)},"note":"辅助字段已按项目格式解码；单位、参考系和字段语义仍以主办方定义为准。"},
        "telemetry_vectors":{"vectors":telemetry,"units":{"position":"m（格式说明）","provided_velocity":"m/s?（格式说明速度行原文写 m；由位置/时间差分高度支持，待主办方确认）","angular_velocity":"deg/s（格式说明）"},"consistency":consistency,"note":"保留 *_raw 字段以兼容旧报告；位置差分自洽检查仅验证辅助速度字段的数值一致性，不等同于图像平面运动目标速度，也不构成轨道解算。"},
        "telemetry_predictions":{"horizon_s":horizon,"method":"末帧位置 + 末帧速度 × 5 个中位帧间隔","predictions":predictions,"note":"常速度外推基线；速度单位和参考系仍以主办方定义为准，不是轨道传播或未来真实位置保证。"},
        "quality":quality,"photometric_evidence":crate::innovation_photometry::photometric_evidence_section(payload),
        "motion":{"feature_count":lines.len(),"moving_count":moving_lines,"candidate_count":lines.iter().filter(|row|row["classification"] == "candidate").count(),"moving_point_count":points.len(),"evidence_link_count":links.len(),"evidence_group_count":groups,"evidence_links":links,"reference_mode":reference,"tracks":lines,"frame_audits":audits,"note":motion_note},
        "point_motion":{"track_count":points.len(),"moving_count":points.len(),"fast_point_motion_count":point_fast,"tracks":points,"note":"这是逐帧点源关联层的运动证据，与差分图线状 motion 分开统计。fast_point_motion 由宽筛候选回到原始 FITS 做通量 SNR、PSF 支持、形状、持续性和常速度拟合复核；它是算法候选，不等于人工真值或星表身份。"},
        "frames":frames}),
    )
}

pub fn build_innovation_report(path: &Path) -> Result<Value> {
    let path = std::fs::canonicalize(path).map_err(|error| error.to_string())?;
    let payload = serde_json::from_slice(&std::fs::read(&path).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())?;
    build_innovation_report_from_payload(&payload, &path)
}

fn t95_critical(degrees_of_freedom: usize) -> Result<f64> {
    const TABLE: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    if degrees_of_freedom == 0 {
        return Err("degrees_of_freedom must be positive".into());
    }
    Ok(TABLE.get(degrees_of_freedom - 1).copied().unwrap_or(1.96))
}

/// Independent X/Y OLS and small-sample confidence intervals for the fitted mean.
/// Forecast intervals exclude future observation noise and registration systematics.
pub fn fit_constant_velocity(
    elapsed: &[f64],
    x: &[f64],
    y: &[f64],
    forecast_horizon_s: Option<f64>,
) -> Result<Value> {
    if elapsed.len() < 2
        || elapsed.len() != x.len()
        || x.len() != y.len()
        || elapsed
            .iter()
            .chain(x)
            .chain(y)
            .any(|value| !value.is_finite())
    {
        return Err(
            "constant-velocity fit requires matching arrays and at least two finite observations"
                .into(),
        );
    }
    let n = elapsed.len() as f64;
    let mean_time = elapsed.iter().sum::<f64>() / n;
    let spread = elapsed
        .iter()
        .map(|time| (time - mean_time).powi(2))
        .sum::<f64>();
    if spread <= 0.0 || !spread.is_finite() {
        return Err("constant-velocity fit requires distinct observation times".into());
    }
    let fit = |values: &[f64]| {
        let mean = values.iter().sum::<f64>() / n;
        let slope = elapsed
            .iter()
            .zip(values)
            .map(|(time, value)| (time - mean_time) * (value - mean))
            .sum::<f64>()
            / spread;
        let fitted = elapsed
            .iter()
            .map(|time| mean + slope * (time - mean_time))
            .collect::<Vec<_>>();
        let squared_error = values
            .iter()
            .zip(&fitted)
            .map(|(value, fitted)| (value - fitted).powi(2))
            .sum::<f64>();
        (mean, slope, fitted, squared_error)
    };
    let (mean_x, vx, fitted_x, error_x) = fit(x);
    let (mean_y, vy, fitted_y, error_y) = fit(y);
    let residual = x
        .iter()
        .zip(&fitted_x)
        .zip(y.iter().zip(&fitted_y))
        .map(|((x, fx), (y, fy))| (x - fx).hypot(y - fy))
        .collect::<Vec<_>>();
    let speed = vx.hypot(vy);
    let df = elapsed.len() - 2;
    let critical = if df > 0 {
        Some(t95_critical(df)?)
    } else {
        None
    };
    let variance_x = critical.map(|_| error_x / df as f64);
    let variance_y = critical.map(|_| error_y / df as f64);
    let vx_se = variance_x.map(|variance| (variance / spread).max(0.0).sqrt());
    let vy_se = variance_y.map(|variance| (variance / spread).max(0.0).sqrt());
    let speed_se = vx_se.zip(vy_se).and_then(|(sx, sy)| {
        (speed > 0.0)
            .then(|| ((vx / speed).powi(2) * sx * sx + (vy / speed).powi(2) * sy * sy).sqrt())
    });
    let low = critical
        .zip(speed_se)
        .map(|(t, se)| (speed - t * se).max(0.0));
    let high = critical.zip(speed_se).map(|(t, se)| speed + t * se);
    let direction_half = critical.zip(vx_se.zip(vy_se)).and_then(|(t, (sx, sy))| {
        (speed > 0.0).then(|| {
            (t * (vy * vy * sx * sx + vx * vx * sy * sy).sqrt() / (speed * speed)).to_degrees()
        })
    });
    let horizon = forecast_horizon_s.filter(|value| value.is_finite() && *value >= 0.0);
    let forecast_time = horizon.map(|horizon| elapsed[elapsed.len() - 1] + horizon);
    let predicted_x = forecast_time.map(|time| mean_x + vx * (time - mean_time));
    let predicted_y = forecast_time.map(|time| mean_y + vy * (time - mean_time));
    let forecast_half = |variance: Option<f64>| {
        forecast_time
            .zip(variance)
            .zip(critical)
            .map(|((time, variance), critical)| {
                critical
                    * (variance * (1.0 / n + (time - mean_time).powi(2) / spread))
                        .max(0.0)
                        .sqrt()
            })
    };
    Ok(
        json!({"velocity_x_px_per_s":vx,"velocity_y_px_per_s":vy,"velocity_x_se_px_per_s":vx_se,"velocity_y_se_px_per_s":vy_se,
        "speed_px_per_s":speed,"speed_se_px_per_s":speed_se,"speed_ci95_low_px_per_s":low,"speed_ci95_high_px_per_s":high,
        "direction_deg_image":vy.atan2(vx).to_degrees(),"direction_ci95_half_width_deg":direction_half,
        "fit_rms_px":((error_x+error_y)/n).sqrt(),"degrees_of_freedom":df,"t95_critical":critical,
        "fitted_x_px":fitted_x,"fitted_y_px":fitted_y,"fit_residual_px":residual,
        "prediction_horizon_s":horizon,"forecast_elapsed_s":forecast_time,"predicted_x_px":predicted_x,"predicted_y_px":predicted_y,
        "predicted_x_ci95_half_width_px":forecast_half(variance_x),"predicted_y_ci95_half_width_px":forecast_half(variance_y),
        "uncertainty_note":"逐轴 OLS 轨迹均值 95% 置信区间；未包含模型失配、配准系统误差或未来单次观测噪声。"}),
    )
}
