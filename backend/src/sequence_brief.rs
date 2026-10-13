//! Native sequence-evidence brief construction.
//!
//! This module only summarizes already-produced image-plane evidence. It does
//! not rerun detection, infer probabilities, or promote an algorithmic track to
//! a physical identity.

use serde_json::{json, Value};

#[derive(Clone, Debug)]
struct Target {
    value: Value,
    classification: String,
    kind: String,
    track_id: i64,
    presence: usize,
    first_frame: i64,
    last_frame: i64,
    speed_px_per_s: Option<f64>,
    direction_deg: Option<f64>,
    fit_rms_px: Option<f64>,
}

fn number(value: Option<&Value>) -> Option<f64> {
    value.and_then(Value::as_f64)
}

fn integer(value: Option<&Value>) -> i64 {
    value
        .and_then(Value::as_i64)
        .or_else(|| {
            value
                .and_then(Value::as_u64)
                .map(|number| number.min(i64::MAX as u64) as i64)
        })
        .unwrap_or(0)
}

fn string(value: Option<&Value>, default: &str) -> String {
    value.and_then(Value::as_str).unwrap_or(default).to_owned()
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

fn digits(value: &str) -> Option<i64> {
    (!value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| value.parse().ok())
        .flatten()
}

fn valid_utc_offset(value: &str) -> bool {
    if value == "Z" {
        return true;
    }
    let Some(signless) = value.strip_prefix('+').or_else(|| value.strip_prefix('-')) else {
        return false;
    };
    let compact: String = signless
        .chars()
        .filter(|character| *character != ':')
        .collect();
    if !matches!(compact.len(), 2 | 4 | 6) || !compact.bytes().all(|byte| byte.is_ascii_digit()) {
        return false;
    }
    let hour = digits(&compact[0..2]).unwrap_or(24);
    let minute = if compact.len() >= 4 {
        digits(&compact[2..4]).unwrap_or(60)
    } else {
        0
    };
    let second = if compact.len() == 6 {
        digits(&compact[4..6]).unwrap_or(60)
    } else {
        0
    };
    hour < 24 && minute < 60 && second < 60
}

/// Match `datetime.fromisoformat(...).replace(tzinfo=None)` semantics: offsets
/// are validated but intentionally ignored, as the Python brief does not
/// convert timestamps to UTC before subtraction.
fn naive_iso_seconds(value: &str) -> Option<f64> {
    let text = value.trim();
    let bytes = text.as_bytes();
    if bytes.len() < 10 || bytes.get(4) != Some(&b'-') || bytes.get(7) != Some(&b'-') {
        return None;
    }
    let year = digits(text.get(0..4)?)?;
    let month = digits(text.get(5..7)?)?;
    let day = digits(text.get(8..10)?)?;
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

    let mut seconds_of_day = 0_i64;
    let mut fraction = 0.0;
    if bytes.len() > 10 {
        if text.len() < 12 || !text.is_char_boundary(11) {
            return None;
        }
        let rest = &text[11..];
        let offset_index = rest
            .char_indices()
            .find(|(index, character)| *index > 0 && matches!(character, '+' | '-'))
            .map(|(index, _)| index);
        let (time_text, offset) = if let Some(index) = offset_index {
            (&rest[..index], Some(&rest[index..]))
        } else if let Some(time) = rest.strip_suffix('Z') {
            (time, Some("Z"))
        } else {
            (rest, None)
        };
        if offset.is_some_and(|offset| !valid_utc_offset(offset)) {
            return None;
        }

        let mut fields = time_text.split(':');
        let hour = digits(fields.next()?)?;
        let minute_text = fields.next();
        let second_text = fields.next();
        if fields.next().is_some() || !(0..=23).contains(&hour) {
            return None;
        }
        let minute = match minute_text {
            Some(value) => digits(value)?,
            None => 0,
        };
        if !(0..=59).contains(&minute) {
            return None;
        }
        let second_and_fraction = second_text.unwrap_or("0");
        let (second_text, fraction_text) = second_and_fraction
            .split_once('.')
            .or_else(|| second_and_fraction.split_once(','))
            .map_or((second_and_fraction, None), |(seconds, fraction)| {
                (seconds, Some(fraction))
            });
        let second = digits(second_text)?;
        if !(0..=59).contains(&second) {
            return None;
        }
        if let Some(fraction_text) = fraction_text {
            if fraction_text.is_empty() || !fraction_text.bytes().all(|byte| byte.is_ascii_digit())
            {
                return None;
            }
            let micros = format!("{fraction_text:0<6}");
            let micros = &micros[..6.min(micros.len())];
            fraction = digits(micros)? as f64 / 1_000_000.0;
        }
        seconds_of_day = hour * 3600 + minute * 60 + second;
    }

    let days = days_from_civil(year, month, day) - days_from_civil(1970, 1, 1);
    Some((days * 86_400 + seconds_of_day) as f64 + fraction)
}

fn timestamp(frame: &Value) -> Option<f64> {
    let value = frame.get("timestamp")?;
    let text = value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string());
    if text.is_empty() {
        None
    } else {
        naive_iso_seconds(&text)
    }
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[middle]
    } else {
        0.5 * (values[middle - 1] + values[middle])
    })
}

fn point_index(point: &Value) -> i64 {
    integer(point.get("frame_index"))
}

fn in_bounds_index(index: i64, len: usize) -> Option<usize> {
    usize::try_from(index).ok().filter(|index| *index < len)
}

fn fit_motion(
    points: &[Value],
    frames: &[Value],
    fallback_speed_per_frame: f64,
) -> (Option<f64>, Option<f64>, Option<f64>) {
    if points.len() < 2 {
        return (None, None, Some(0.0));
    }
    let timestamps: Vec<_> = frames.iter().map(timestamp).collect();
    let first_time = in_bounds_index(point_index(&points[0]), timestamps.len())
        .and_then(|index| timestamps[index]);
    let mut elapsed = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for point in points {
        let current = in_bounds_index(point_index(point), timestamps.len())
            .and_then(|index| timestamps[index]);
        if let (Some(first), Some(current)) = (first_time, current) {
            elapsed.push(current - first);
            xs.push(number(point.get("aligned_x")).unwrap_or(0.0));
            ys.push(number(point.get("aligned_y")).unwrap_or(0.0));
        }
    }
    if elapsed.len() >= 2 && elapsed[elapsed.len() - 1] > elapsed[0] {
        let mean_t = elapsed.iter().sum::<f64>() / elapsed.len() as f64;
        let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
        let mean_y = ys.iter().sum::<f64>() / ys.len() as f64;
        let denominator = elapsed
            .iter()
            .map(|time| (time - mean_t).powi(2))
            .sum::<f64>();
        if denominator > 0.0 {
            let vx = elapsed
                .iter()
                .zip(&xs)
                .map(|(time, x)| (time - mean_t) * (x - mean_x))
                .sum::<f64>()
                / denominator;
            let vy = elapsed
                .iter()
                .zip(&ys)
                .map(|(time, y)| (time - mean_t) * (y - mean_y))
                .sum::<f64>()
                / denominator;
            return (
                Some(vx.hypot(vy)),
                Some(vy.atan2(vx).to_degrees()),
                Some(elapsed[elapsed.len() - 1] - elapsed[0]),
            );
        }
    }

    let first = &points[0];
    let last = &points[points.len() - 1];
    let dx = number(last.get("aligned_x")).unwrap_or(0.0)
        - number(first.get("aligned_x")).unwrap_or(0.0);
    let dy = number(last.get("aligned_y")).unwrap_or(0.0)
        - number(first.get("aligned_y")).unwrap_or(0.0);
    let frame_span = (point_index(last) - point_index(first)).max(1);
    (
        Some(fallback_speed_per_frame),
        Some(dy.atan2(dx).to_degrees()),
        Some(frame_span as f64),
    )
}

fn target_from_track(track: &Value, frames: &[Value], kind: &str) -> Option<Target> {
    let points = track.get("points").and_then(Value::as_array)?;
    if points.is_empty() {
        return None;
    }
    let first = point_index(&points[0]);
    let last = point_index(points.last()?);
    let (speed, direction, duration) = fit_motion(
        points,
        frames,
        number(track.get("speed_px_per_frame")).unwrap_or(0.0),
    );
    let classification = string(track.get("classification"), "candidate");
    let evidence_level = string(track.get("evidence_level"), "quality");
    let evidence_text = if classification == "moving" {
        "连续帧位置/形状通过当前 moving 门槛，仍待星表或人工真值确认"
    } else if evidence_level == "single_frame_trail" {
        "单帧曝光线形候选；只能说明本次曝光出现线状形态，不能估计跨帧速度"
    } else if points.len() == 1 {
        "15 帧线层只保留 1 个时序候选，不能据此称为单帧曝光候选或估计速度"
    } else {
        "多帧出现但未达到严格 moving 门槛，保留为待复核候选"
    }
    .to_owned();
    let track_id = integer(track.get("track_id"));
    let fit_rms = number(track.get("fit_rms_px"));
    Some(Target {
        value: json!({
            "track_id":track_id,
            "kind":kind,
            "classification":classification,
            "first_frame":first + 1,
            "last_frame":last + 1,
            "presence":points.len(),
            "duration_s":duration,
            "speed_px_per_s":speed,
            "direction_deg":direction,
            "fit_rms_px":fit_rms,
            "evidence_text":evidence_text,
        }),
        classification,
        kind: kind.to_owned(),
        track_id,
        presence: points.len(),
        first_frame: first + 1,
        last_frame: last + 1,
        speed_px_per_s: speed,
        direction_deg: direction,
        fit_rms_px: fit_rms,
    })
}

fn comma_group(value: i64) -> String {
    let digits = value.unsigned_abs().to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    if value < 0 {
        grouped.push('-');
    }
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

fn frame_text(target: &Target) -> String {
    if target.first_frame == target.last_frame {
        format!("F{:02}", target.first_frame)
    } else {
        format!("F{:02}–F{:02}", target.first_frame, target.last_frame)
    }
}

fn brief_text(
    headline: &str,
    findings: &[String],
    boundaries: &[String],
    mosaic_note: &str,
) -> String {
    let mut lines = vec![headline.to_owned()];
    lines.extend(findings.iter().map(|line| format!("• {line}")));
    lines.push("结论边界：".to_owned());
    lines.extend(boundaries.iter().map(|line| format!("• {line}")));
    lines.push(format!("• {mosaic_note}"));
    lines.join("\n")
}

pub fn handle_request(request: &Value) -> Result<Value, String> {
    let result = request
        .get("result")
        .ok_or_else(|| "missing sequence result".to_owned())?;
    let frames = result
        .get("frames")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut intervals = Vec::new();
    let datetimes: Vec<_> = frames.iter().map(timestamp).collect();
    for pair in datetimes.windows(2) {
        if let (Some(left), Some(right)) = (pair[0], pair[1]) {
            if right - left > 0.0 {
                intervals.push(right - left);
            }
        }
    }
    let interval_median = median(&mut intervals);
    let duration = if datetimes.len() >= 2 {
        datetimes[0]
            .zip(datetimes[datetimes.len() - 1])
            .map(|(first, last)| last - first)
    } else {
        None
    };
    let max_shift = result
        .get("cumulative_shifts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_array)
        .filter_map(|shift| {
            let dx = number(shift.first())?;
            let dy = number(shift.get(1)).unwrap_or(0.0);
            Some(dx.hypot(dy))
        })
        .fold(0.0_f64, f64::max);

    let mut targets = Vec::new();
    if let Some(tracks) = result.get("motion_features").and_then(Value::as_array) {
        for track in tracks {
            if let Some(target) = target_from_track(track, frames, "line") {
                targets.push(target);
            }
        }
    }
    if let Some(tracks) = result.get("tracks").and_then(Value::as_array) {
        for track in tracks {
            if string(track.get("classification"), "") != "moving" {
                continue;
            }
            if let Some(target) = target_from_track(track, frames, "point") {
                targets.push(target);
            }
        }
    }
    targets.sort_by(|left, right| {
        (left.classification != "moving")
            .cmp(&(right.classification != "moving"))
            .then_with(|| right.presence.cmp(&left.presence))
            .then_with(|| left.track_id.cmp(&right.track_id))
    });
    let moving: Vec<_> = targets
        .iter()
        .filter(|target| target.classification == "moving")
        .collect();
    let pending: Vec<_> = targets
        .iter()
        .filter(|target| target.classification != "moving")
        .collect();
    let headline = if !moving.is_empty() {
        format!("先说结论：15 帧中形成 {} 条连续运动候选。", moving.len())
    } else if !pending.is_empty() {
        "先说结论：当前门槛下没有形成严格 moving，只有待复核候选。".to_owned()
    } else {
        "先说结论：当前门槛下没有形成运动候选。".to_owned()
    };

    let mut timing = format!("{} 帧连续观测", frames.len());
    if let Some(duration) = duration {
        timing.push_str(&format!("，总时长约 {duration:.3} s"));
    }
    if let Some(interval) = interval_median {
        timing.push_str(&format!("，相邻帧约每 {interval:.3} s 一张"));
    }
    let mut findings = vec![
        format!("{timing}。"),
        format!("固定星场配准后的最大共同位移约 {max_shift:.3} px，说明本次序列的共同星场变化很小，适合在配准坐标里找异常运动。"),
    ];
    for target in targets.iter().take(3) {
        let kind_label = if target.kind == "line" {
            "线状"
        } else {
            "点状"
        };
        if target.classification == "moving"
            && target.speed_px_per_s.is_some()
            && target.direction_deg.is_some()
        {
            let fit = target
                .fit_rms_px
                .map(|value| format!("，拟合 RMS {value:.2} px"))
                .unwrap_or_default();
            findings.push(format!(
                "{kind_label}目标 {:04} 出现在 {}（{} 帧），图像平面速度约 {:.2} px/s，方向约 {:.1}°{fit}。",
                target.track_id,
                frame_text(target),
                target.presence,
                target.speed_px_per_s.unwrap_or_default(),
                target.direction_deg.unwrap_or_default(),
            ));
        } else if target.first_frame == target.last_frame {
            findings.push(format!(
                "{kind_label}候选 {:04} 只在 {} 出现；没有把它和其它时间段的轨迹硬合并。",
                target.track_id,
                frame_text(target),
            ));
        } else {
            findings.push(format!(
                "候选 {:04} 出现在 {}，但当前证据不足以升级为严格 moving。",
                target.track_id,
                frame_text(target),
            ));
        }
    }
    if targets.is_empty() {
        findings.push("没有生成可展示的点状或线状运动轨迹；这表示当前算法层未找到，不等于物理上绝对没有目标。".to_owned());
    }
    let point_moving = moving
        .iter()
        .filter(|target| target.kind == "point")
        .count();
    if point_moving >= 2 {
        findings.push("多条点状候选的方向/速度相近，先按同向运动候选报告；还要排除共同配准误差或错误关联，不能直接按数量认定为多个真实天体。".to_owned());
    }

    let stable_count = integer(result.get("stable_source_count"));
    let persistent_count = integer(result.get("persistent_source_count"));
    let mut boundaries = vec![
        "moving 是通过当前图像几何、跨帧关联和拟合门槛的算法候选，不等于已经完成星表身份或人工真值确认。".to_owned(),
        "速度和方向目前是图像平面 px/s 与图像坐标角度；没有像元角尺度和完整 WCS 时，不能写成真实角速度或轨道速度。".to_owned(),
        format!(
            "严格静态 {}、持续候选 {} 等是证据层计数，不是物理恒星总数。",
            comma_group(stable_count),
            comma_group(persistent_count),
        ),
    ];
    if !pending.is_empty() {
        boundaries.push("单帧长线可作为曝光运动候选，说明形态上可能在曝光内发生位移；但仍需结合其它帧、差分图和原始像素确认真实身份。".to_owned());
    }

    let mosaic_note = if let Some(mosaic) = request.get("mosaic").filter(|value| !value.is_null()) {
        let shape = mosaic.get("output_shape").and_then(Value::as_array);
        let height = integer(shape.and_then(|shape| shape.first()));
        let width = integer(shape.and_then(|shape| shape.get(1)));
        let covered = integer(mosaic.get("covered_pixel_count"));
        let overlap = integer(mosaic.get("overlap_pixel_count"));
        format!(
            "合成图为 {}×{} px，覆盖 {} 个像素，重叠 {} 个像素；它说明多帧如何融合，不会自动增加恒星数或确认运动目标。",
            width, height, comma_group(covered), comma_group(overlap),
        )
    } else {
        "注册合成图尚未生成；生成后只用于查看联合覆盖、重叠和像素稳定性。".to_owned()
    };
    let brief = json!({
        "headline":headline,
        "findings":findings,
        "boundaries":boundaries,
        "targets":targets.iter().map(|target| target.value.clone()).collect::<Vec<_>>(),
        "mosaic_note":mosaic_note,
        "evidence_refs":["innovation_report.json","motion_evidence.csv","motion_evidence_links.csv","motion_frame_audit.csv","registered_mosaic_coverage.npy"],
        "frame_count":frames.len(),
        "duration_s":duration,
        "interval_median_s":interval_median,
        "registration_max_shift_px":max_shift,
    });
    let findings_text: Vec<String> = brief["findings"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[])
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let boundaries_text: Vec<String> = brief["boundaries"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[])
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let text = brief_text(
        brief["headline"].as_str().unwrap_or_default(),
        &findings_text,
        &boundaries_text,
        brief["mosaic_note"].as_str().unwrap_or_default(),
    );
    Ok(json!({"brief":brief,"text":text}))
}

#[cfg(test)]
mod tests {
    use super::{handle_request, naive_iso_seconds};
    use serde_json::json;

    #[test]
    fn timestamp_parser_ignores_offset_like_python_brief() {
        assert_eq!(
            naive_iso_seconds("2026-03-30T16:32:05.250+03:00"),
            naive_iso_seconds("2026-03-30T16:32:05.250Z"),
        );
        assert_eq!(
            naive_iso_seconds("2026-03-30T16:32:05.250")
                .zip(naive_iso_seconds("2026-03-30T16:32:07.750"))
                .map(|(left, right)| right - left),
            Some(2.5),
        );
        assert!(naive_iso_seconds("2026-02-30T16:32:05").is_none());
    }

    #[test]
    fn builds_empty_sequence_brief_with_scientific_boundaries() {
        let output = handle_request(&json!({
            "result": {
                "frames": [
                    {"timestamp":"2026-03-30T16:32:05.000"},
                    {"timestamp":"2026-03-30T16:32:07.000"}
                ],
                "cumulative_shifts":[[0.0,0.0],[0.1,0.2]],
                "tracks":[],"motion_features":[],
                "stable_source_count":1234,"persistent_source_count":12
            }
        }))
        .unwrap();
        assert!(output["brief"]["headline"]
            .as_str()
            .unwrap()
            .contains("没有形成运动候选"));
        assert_eq!(output["brief"]["duration_s"], 2.0);
        assert!(output["text"].as_str().unwrap().contains("1,234"));
    }
}
