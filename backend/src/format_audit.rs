//! Read-only audit of FITS storage facts, pixel ranges, and auxiliary velocity units.
use crate::{fits, models::FitsFrame, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub const STANDARD_WCS_HEADER_CARDS: [&str; 17] = [
    "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "CDELT1", "CDELT2", "CROTA2", "PV1_0", "PV2_0", "A_ORDER", "B_ORDER",
];
const WCS_MINIMAL_CORE_CARDS: [&str; 6] =
    ["CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2"];
const INTERPRETATION_BOUNDARY: &str = "actual FITS storage and internal telemetry consistency; not a calibration of ADC signedness, BLANK semantics, saturation level, complete WCS, or physical source truth";
const INTERPRETATION_NOTES: [&str; 5] = [
    "以合法 FITS 头、数据长度和 BITPIX 作为实际存储事实，不把格式说明中的尺寸文字写死。",
    "负值、精确 -1 和正负极值只做诊断统计；没有 BLANK、ADC、满阱或坏像素定义时不自动掩膜。",
    "uint16 视图只用于显示同一字节的替代解释，不改变默认有符号 >i2 读取。",
    "首行前 208 字节辅助数据已从像素值域统计中排除。",
    "标准 WCS 卡片只做存在性记录；辅助光轴/姿态字段位于首行数据区时，不把它们混同为完整 WCS。",
];

#[derive(Clone)]
struct Record {
    path: PathBuf,
    frame: FitsFrame,
    time_us: Option<i64>,
    display_time: Option<String>,
}

fn optional_i64(value: Option<&Value>) -> Option<i64> {
    let value = value?;
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|v| i64::try_from(v).ok()))
        .or_else(|| {
            value.as_f64().and_then(|v| {
                (v.is_finite() && v.fract() == 0.0 && v >= i64::MIN as f64 && v <= i64::MAX as f64)
                    .then_some(v as i64)
            })
        })
        .or_else(|| value.as_str().and_then(|v| v.parse().ok()))
}

fn optional_f64(value: Option<&Value>) -> Option<f64> {
    let value = value?;
    let number = value
        .as_f64()
        .or_else(|| value.as_str().and_then(|v| v.parse().ok()))
        .or_else(|| value.as_bool().map(|v| if v { 1.0 } else { 0.0 }))?;
    number.is_finite().then_some(number)
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

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

fn digits(text: &str) -> Option<i64> {
    (!text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| text.parse().ok())
        .flatten()
}

fn parse_iso_timestamp(value: &str) -> Option<i64> {
    let text = value.trim();
    let (date, rest) =
        if let Some((date, rest)) = text.split_once('T').or_else(|| text.split_once('t')) {
            (date, Some(rest))
        } else if let Some((date, rest)) = text.split_once(' ') {
            (date, Some(rest))
        } else {
            (text, None)
        };
    let mut date_parts = date.split('-');
    let year = digits(date_parts.next()?)?;
    let month = digits(date_parts.next()?)?;
    let day = digits(date_parts.next()?)?;
    if date_parts.next().is_some() || !(1..=9999).contains(&year) || !(1..=12).contains(&month) {
        return None;
    }
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
    if day < 1 || day > month_days[(month - 1) as usize] {
        return None;
    }

    let mut seconds_of_day = 0_i64;
    let mut fraction_us = 0_i64;
    let mut offset_seconds = 0_i64;
    if let Some(rest) = rest {
        let offset_index = rest
            .char_indices()
            .find(|(index, character)| *index > 0 && matches!(character, '+' | '-'))
            .map(|(index, _)| index);
        let (time_text, zone) = if let Some(index) = offset_index {
            (&rest[..index], Some(&rest[index..]))
        } else if rest.ends_with('Z') || rest.ends_with('z') {
            (&rest[..rest.len() - 1], Some("Z"))
        } else {
            (rest, None)
        };
        if let Some(zone) = zone.filter(|zone| *zone != "Z") {
            let sign = if zone.starts_with('-') {
                -1
            } else if zone.starts_with('+') {
                1
            } else {
                return None;
            };
            let compact: String = zone[1..]
                .chars()
                .filter(|character| *character != ':')
                .collect();
            if !matches!(compact.len(), 2 | 4 | 6) {
                return None;
            }
            let hour = digits(&compact[0..2])?;
            let minute = if compact.len() >= 4 {
                digits(&compact[2..4])?
            } else {
                0
            };
            let second = if compact.len() == 6 {
                digits(&compact[4..6])?
            } else {
                0
            };
            if hour >= 24 || minute >= 60 || second >= 60 {
                return None;
            }
            offset_seconds = sign * (hour * 3600 + minute * 60 + second);
        }
        let mut time_parts = time_text.split(':');
        let hour = digits(time_parts.next()?)?;
        let minute = match time_parts.next() {
            Some(value) => digits(value)?,
            None => 0,
        };
        let seconds_text = time_parts.next().unwrap_or("0");
        if time_parts.next().is_some() || !(0..=23).contains(&hour) || !(0..=59).contains(&minute) {
            return None;
        }
        let (second_text, fraction_text) = seconds_text
            .split_once('.')
            .or_else(|| seconds_text.split_once(','))
            .map_or((seconds_text, None), |(seconds, fraction)| {
                (seconds, Some(fraction))
            });
        let second = digits(second_text)?;
        if !(0..=59).contains(&second) {
            return None;
        }
        if let Some(fraction) = fraction_text {
            if fraction.is_empty() || !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            let micros = format!("{fraction:0<6}");
            fraction_us = digits(&micros[..6.min(micros.len())])?;
        }
        seconds_of_day = hour * 3600 + minute * 60 + second;
    }
    let days = days_from_civil(year, month, day) - days_from_civil(1970, 1, 1);
    Some(((days * 86_400 + seconds_of_day - offset_seconds) * 1_000_000) + fraction_us)
}

fn timestamp_text(value: Option<&Value>) -> (Option<i64>, Option<String>) {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return (None, None);
    };
    let raw = value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string());
    let Some(micros) = parse_iso_timestamp(&raw) else {
        return (None, Some(raw));
    };
    let seconds = micros.div_euclid(1_000_000);
    let fraction = micros.rem_euclid(1_000_000);
    let days = seconds.div_euclid(86_400);
    let second_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = second_of_day / 3_600;
    let minute = second_of_day % 3_600 / 60;
    let second = second_of_day % 60;
    let fraction = if fraction == 0 {
        String::new()
    } else {
        format!(".{fraction:06}")
    };
    (
        Some(micros),
        Some(format!(
            "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}{fraction}+00:00"
        )),
    )
}

fn resolve_paths(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        if input
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("fits"))
        {
            return Ok(vec![input.to_owned()]);
        }
        return Err(format!("expected a .fits file, got {input:?}"));
    }
    if input.is_dir() {
        let mut paths = fs::read_dir(input)
            .map_err(|error| error.to_string())?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.is_file() && path.extension().and_then(|value| value.to_str()) == Some("fits")
            })
            .collect::<Vec<_>>();
        paths.sort();
        if paths.is_empty() {
            return Err(format!("directory contains no FITS files: {input:?}"));
        }
        return Ok(paths);
    }
    Err(format!("input path does not exist: {input:?}"))
}

fn resolved_display(path: &Path) -> String {
    let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_owned());
    let text = canonical.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else {
        text.strip_prefix(r"\\?\").unwrap_or(&text).to_owned()
    }
}

fn integer_sample(frame: &FitsFrame, index: usize) -> Option<i64> {
    Some(match &frame.data {
        crate::models::PixelData::U8(values) => values[index] as i64,
        crate::models::PixelData::I16(values) => values[index] as i64,
        crate::models::PixelData::I32(values) => values[index] as i64,
        crate::models::PixelData::I64(values) => values[index],
        crate::models::PixelData::F32(values) => values[index] as i64,
        crate::models::PixelData::F64(values) => values[index] as i64,
    })
}

fn frame_row(index: usize, record: &Record, extreme_fraction: f64) -> Result<Value> {
    let frame = &record.frame;
    let width = optional_i64(frame.header.get("NAXIS1"));
    let height = optional_i64(frame.header.get("NAXIS2"));
    let bitpix = optional_i64(frame.header.get("BITPIX"));
    let expected_size = width
        .zip(height)
        .zip(bitpix.filter(|value| *value != 0))
        .and_then(|((width, height), bitpix)| {
            width
                .checked_mul(height)?
                .checked_mul(bitpix.checked_abs()? / 8)
        })
        .and_then(|bytes| usize::try_from(bytes).ok())
        .and_then(|bytes| frame.data_offset.checked_add(bytes));
    let file_size = fs::metadata(&record.path)
        .map_err(|error| error.to_string())?
        .len();
    let masked = frame.width.min(104).min(frame.data.len());
    let mut valid_count = 0usize;
    let mut signed_min: Option<i64> = None;
    let mut signed_max: Option<i64> = None;
    let mut negative = 0usize;
    let mut zero = 0usize;
    let mut minus_one = 0usize;
    let mut negative_extreme = 0usize;
    let mut positive_extreme = 0usize;
    let mut unsigned_min: Option<u16> = None;
    let mut unsigned_max: Option<u16> = None;
    let negative_threshold = ((i16::MIN as f64) * extreme_fraction).floor() as i64;
    let positive_threshold = ((i16::MAX as f64) * extreme_fraction).ceil() as i64;
    for pixel in masked..frame.data.len() {
        let value = integer_sample(frame, pixel).ok_or("invalid FITS sample")?;
        valid_count += 1;
        signed_min = Some(signed_min.map_or(value, |min| min.min(value)));
        signed_max = Some(signed_max.map_or(value, |max| max.max(value)));
        negative += usize::from(value < 0);
        zero += usize::from(value == 0);
        minus_one += usize::from(value == -1);
        negative_extreme += usize::from(value <= negative_threshold);
        positive_extreme += usize::from(value >= positive_threshold);
        if let crate::models::PixelData::I16(values) = &frame.data {
            let unsigned = values[pixel] as u16;
            unsigned_min = Some(unsigned_min.map_or(unsigned, |min| min.min(unsigned)));
            unsigned_max = Some(
                unsigned_max.map_or(unsigned_max.unwrap_or(unsigned), |max| max.max(unsigned)),
            );
        }
    }
    let present: Vec<&str> = STANDARD_WCS_HEADER_CARDS
        .iter()
        .copied()
        .filter(|key| frame.header.contains_key(*key))
        .collect();
    let missing: Vec<&str> = STANDARD_WCS_HEADER_CARDS
        .iter()
        .copied()
        .filter(|key| !frame.header.contains_key(*key))
        .collect();
    let exposure_ms = fits::exposure_milliseconds(&frame.header)?;
    let shape_matches = width
        .zip(height)
        .map(|(w, h)| frame.width as i64 == w && frame.height as i64 == h);
    Ok(json!({
        "frame_index": index,
        "frame_number": index + 1,
        "path": resolved_display(&record.path),
        "file_size_bytes": file_size,
        "data_offset_bytes": frame.data_offset,
        "expected_file_size_bytes": expected_size,
        "file_size_matches": expected_size.map(|size| file_size == size as u64),
        "bitpix": bitpix,
        "naxis1": width,
        "naxis2": height,
        "shape": format!("{}x{}", frame.height, frame.width),
        "shape_matches_header": shape_matches,
        "dtype": frame.data.dtype(),
        "bscale": optional_f64(frame.header.get("BSCALE")),
        "bzero": optional_f64(frame.header.get("BZERO")),
        "blank": optional_f64(frame.header.get("BLANK")),
        "has_bscale": frame.header.contains_key("BSCALE"),
        "has_bzero": frame.header.contains_key("BZERO"),
        "has_blank": frame.header.contains_key("BLANK"),
        "timestamp": record.display_time,
        "exposure_ms": exposure_ms,
        "auxiliary_complete": frame.auxiliary.is_some(),
        "auxiliary_masked_pixel_count": if frame.height > 0 { frame.width.min(104) } else { 0 },
        "wcs_standard_card_count": present.len(),
        "wcs_standard_cards_present": present.join("|"),
        "wcs_standard_cards_missing": missing.join("|"),
        "wcs_minimal_core_present": WCS_MINIMAL_CORE_CARDS.iter().all(|key| frame.header.contains_key(*key)),
        "valid_pixel_count": valid_count,
        "signed_min": signed_min,
        "signed_max": signed_max,
        "negative_count": negative,
        "zero_count": zero,
        "exact_minus_one_count": minus_one,
        "negative_extreme_count": negative_extreme,
        "positive_extreme_count": positive_extreme,
        "unsigned_diagnostic_min": unsigned_min,
        "unsigned_diagnostic_max": unsigned_max
    }))
}

fn vector_error(estimate: [f64; 3], reference: [f64; 3]) -> (f64, Option<f64>) {
    let error = ((estimate[0] - reference[0]).powi(2)
        + (estimate[1] - reference[1]).powi(2)
        + (estimate[2] - reference[2]).powi(2))
    .sqrt();
    let norm = (reference[0].powi(2) + reference[1].powi(2) + reference[2].powi(2)).sqrt();
    (error, (norm > f64::EPSILON).then_some(error / norm))
}

fn velocity_row(index: usize, records: &[Record]) -> Result<Value> {
    let record = &records[index];
    let path = resolved_display(&record.path);
    let timestamp = record.time_us.and_then(|_| record.display_time.clone());
    let Some(current_time) = record.time_us else {
        return Ok(
            json!({"frame_index":index,"frame_number":index+1,"path":path,"timestamp":null,"dt_seconds":null,"derivative_method":"unavailable","j2000_position_velocity_error":null,"j2000_position_velocity_relative_error":null,"wgs84_position_velocity_error":null,"wgs84_position_velocity_relative_error":null}),
        );
    };
    let Some(current_aux) = record.frame.auxiliary.as_ref() else {
        return Ok(
            json!({"frame_index":index,"frame_number":index+1,"path":path,"timestamp":timestamp,"dt_seconds":null,"derivative_method":"unavailable","j2000_position_velocity_error":null,"j2000_position_velocity_relative_error":null,"wgs84_position_velocity_error":null,"wgs84_position_velocity_relative_error":null}),
        );
    };
    let (left, right, dt_us, method) = if index == 0 && records.len() > 1 {
        let next = &records[1];
        let dt = next.time_us.map(|time| time - current_time);
        let right = next
            .frame
            .auxiliary
            .as_ref()
            .map(|aux| &aux.values)
            .ok_or("missing auxiliary data for forward velocity difference")?;
        (&current_aux.values, right, dt, "forward_difference")
    } else if index + 1 == records.len() && records.len() > 1 {
        let previous = &records[index - 1];
        let dt = previous.time_us.map(|time| current_time - time);
        let left = previous
            .frame
            .auxiliary
            .as_ref()
            .map(|aux| &aux.values)
            .ok_or("missing auxiliary data for backward velocity difference")?;
        (left, &current_aux.values, dt, "backward_difference")
    } else if index > 0
        && index + 1 < records.len()
        && records[index - 1].time_us.is_some()
        && records[index + 1].time_us.is_some()
    {
        let previous = &records[index - 1];
        let next = &records[index + 1];
        let dt = next.time_us.unwrap() - previous.time_us.unwrap();
        let left = previous
            .frame
            .auxiliary
            .as_ref()
            .map(|aux| &aux.values)
            .ok_or("missing auxiliary data for centered velocity difference")?;
        let right = next
            .frame
            .auxiliary
            .as_ref()
            .map(|aux| &aux.values)
            .ok_or("missing auxiliary data for centered velocity difference")?;
        (left, right, Some(dt), "centered_difference")
    } else {
        return Ok(
            json!({"frame_index":index,"frame_number":index+1,"path":path,"timestamp":timestamp,"dt_seconds":null,"derivative_method":"unavailable","j2000_position_velocity_error":null,"j2000_position_velocity_relative_error":null,"wgs84_position_velocity_error":null,"wgs84_position_velocity_relative_error":null}),
        );
    };
    let Some(dt_us) = dt_us.filter(|dt| dt.abs() > 0) else {
        return Ok(
            json!({"frame_index":index,"frame_number":index+1,"path":path,"timestamp":timestamp,"dt_seconds":null,"derivative_method":"unavailable","j2000_position_velocity_error":null,"j2000_position_velocity_relative_error":null,"wgs84_position_velocity_error":null,"wgs84_position_velocity_relative_error":null}),
        );
    };
    let dt_seconds = dt_us as f64 / 1_000_000.0;
    let j2000_estimate =
        std::array::from_fn(|axis| (right[4 + axis] - left[4 + axis]) / dt_seconds);
    let j2000_reference = [
        current_aux.values[7],
        current_aux.values[8],
        current_aux.values[9],
    ];
    let wgs84_estimate =
        std::array::from_fn(|axis| (right[20 + axis] - left[20 + axis]) / dt_seconds);
    let wgs84_reference = [
        current_aux.values[23],
        current_aux.values[24],
        current_aux.values[25],
    ];
    let (j2000_error, j2000_relative) = vector_error(j2000_estimate, j2000_reference);
    let (wgs84_error, wgs84_relative) = vector_error(wgs84_estimate, wgs84_reference);
    Ok(
        json!({"frame_index":index,"frame_number":index+1,"path":path,"timestamp":timestamp,"dt_seconds":dt_seconds.abs(),"derivative_method":method,"j2000_position_velocity_error":j2000_error,"j2000_position_velocity_relative_error":j2000_relative,"wgs84_position_velocity_error":wgs84_error,"wgs84_position_velocity_relative_error":wgs84_relative}),
    )
}

fn number_array(values: impl Iterator<Item = Option<f64>>) -> Vec<f64> {
    values.flatten().filter(|value| value.is_finite()).collect()
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn summarize(
    frames: &[Value],
    velocities: &[Value],
    times: &[Option<i64>],
    extreme_fraction: f64,
) -> Value {
    let count_by = |key: &str| -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for row in frames {
            if let Some(value) = row.get(key).filter(|value| !value.is_null()) {
                *counts
                    .entry(value.to_string().trim_matches('"').to_owned())
                    .or_default() += 1;
            }
        }
        counts
    };
    let mut file_sizes = frames
        .iter()
        .filter_map(|row| row["file_size_bytes"].as_u64())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    file_sizes.sort_unstable();
    let mut parsed_times = times.iter().flatten().copied().collect::<Vec<_>>();
    parsed_times.sort_unstable();
    let mut j2000 = number_array(
        velocities
            .iter()
            .map(|row| row["j2000_position_velocity_relative_error"].as_f64()),
    );
    let mut wgs84 = number_array(
        velocities
            .iter()
            .map(|row| row["wgs84_position_velocity_relative_error"].as_f64()),
    );
    let max = |values: &[f64]| values.iter().copied().reduce(f64::max);
    let card_counts: BTreeMap<&str, usize> = STANDARD_WCS_HEADER_CARDS
        .iter()
        .map(|card| {
            (
                *card,
                frames
                    .iter()
                    .filter(|row| {
                        row["wcs_standard_cards_present"]
                            .as_str()
                            .unwrap_or("")
                            .split('|')
                            .any(|present| present == *card)
                    })
                    .count(),
            )
        })
        .collect();
    let shape_counts = count_by("shape");
    let bitpix_counts = count_by("bitpix");
    let dtype_counts = count_by("dtype");
    let timestamp_iso = |time: Option<i64>| {
        time.and_then(|time| timestamp_text(Some(&Value::String(format_unix_for_parse(time)))).1)
    };
    let span = (parsed_times.len() >= 2)
        .then(|| (parsed_times[parsed_times.len() - 1] - parsed_times[0]) as f64 / 1_000_000.0);
    json!({
        "frame_count":frames.len(),"file_size_bytes":file_sizes,"shape_counts":shape_counts,"bitpix_counts":bitpix_counts,"dtype_counts":dtype_counts,
        "all_file_sizes_match_header":frames.iter().all(|row| row["file_size_matches"] == true),
        "all_shapes_match_header":frames.iter().all(|row| row["shape_matches_header"] == true),
        "complete_auxiliary_frames":frames.iter().filter(|row| row["auxiliary_complete"] == true).count(),
        "wcs_standard_card_counts":card_counts,
        "frames_with_any_standard_wcs_card":frames.iter().filter(|row| row["wcs_standard_card_count"].as_u64().unwrap_or(0)>0).count(),
        "frames_with_minimal_wcs_core":frames.iter().filter(|row| row["wcs_minimal_core_present"]==true).count(),
        "bscale_cards_present":frames.iter().filter(|row| row["has_bscale"]==true).count(),"bzero_cards_present":frames.iter().filter(|row| row["has_bzero"]==true).count(),"blank_cards_present":frames.iter().filter(|row| row["has_blank"]==true).count(),
        "total_valid_pixels":frames.iter().map(|row|row["valid_pixel_count"].as_u64().unwrap_or(0)).sum::<u64>(),
        "total_negative_pixels":frames.iter().map(|row|row["negative_count"].as_u64().unwrap_or(0)).sum::<u64>(),
        "total_zero_pixels":frames.iter().map(|row|row["zero_count"].as_u64().unwrap_or(0)).sum::<u64>(),
        "total_exact_minus_one_pixels":frames.iter().map(|row|row["exact_minus_one_count"].as_u64().unwrap_or(0)).sum::<u64>(),
        "total_negative_extreme_pixels":frames.iter().map(|row|row["negative_extreme_count"].as_u64().unwrap_or(0)).sum::<u64>(),
        "total_positive_extreme_pixels":frames.iter().map(|row|row["positive_extreme_count"].as_u64().unwrap_or(0)).sum::<u64>(),
        "extreme_fraction":extreme_fraction,
        "timestamp_start":timestamp_iso(parsed_times.first().copied()),"timestamp_end":timestamp_iso(parsed_times.last().copied()),"timestamp_span_seconds":span,
        "velocity_consistency":{"usable_frame_count":velocities.iter().filter(|row|!row["dt_seconds"].is_null()).count(),"j2000_relative_error_median":median(&mut j2000),"j2000_relative_error_max":max(&j2000),"wgs84_relative_error_median":median(&mut wgs84),"wgs84_relative_error_max":max(&wgs84),"unit_interpretation":"位置字段标注为 m 且与 DATE-OBS 差分高度自洽时，速度字段的 m/s 是内部一致性推断；格式说明原文仍需主办方确认。"},
        "interpretation_notes":INTERPRETATION_NOTES
    })
}

// Convert canonical timestamp text back through the single parser used above.
fn format_unix_for_parse(micros: i64) -> String {
    let seconds = micros.div_euclid(1_000_000);
    let fraction = micros.rem_euclid(1_000_000);
    let days = seconds.div_euclid(86_400);
    let second_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{fraction:06}Z",
        second_of_day / 3600,
        second_of_day % 3600 / 60,
        second_of_day % 60
    )
}

pub fn run_format_audit(input_path: &Path, extreme_fraction: f64) -> Result<Value> {
    if !extreme_fraction.is_finite() || !(0.0 < extreme_fraction && extreme_fraction <= 1.0) {
        return Err("extreme_fraction must be in (0, 1]".into());
    }
    let mut records = resolve_paths(input_path)?
        .into_iter()
        .map(|path| {
            let frame = fits::read_fits(&path, false)?;
            let (time_us, display_time) = timestamp_text(frame.header.get("DATE-OBS"));
            Ok(Record {
                path,
                frame,
                time_us,
                display_time,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    records.sort_by(|left, right| match (left.time_us, right.time_us) {
        (Some(left_time), Some(right_time)) => left_time
            .cmp(&right_time)
            .then_with(|| left.path.cmp(&right.path)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => left.path.cmp(&right.path),
    });
    let frames = records
        .iter()
        .enumerate()
        .map(|(index, record)| frame_row(index, record, extreme_fraction))
        .collect::<Result<Vec<_>>>()?;
    let velocities = (0..records.len())
        .map(|index| velocity_row(index, &records))
        .collect::<Result<Vec<_>>>()?;
    let times = records
        .iter()
        .map(|record| record.time_us)
        .collect::<Vec<_>>();
    let summary = summarize(&frames, &velocities, &times, extreme_fraction);
    Ok(json!({
        "input_path":resolved_display(input_path),
        "extreme_fraction":extreme_fraction,
        "frames":frames,
        "velocity":velocities,
        "summary":summary,
        "interpretation_boundary":INTERPRETATION_BOUNDARY
    }))
}

const FRAME_COLUMNS: [&str; 37] = [
    "frame_number",
    "frame_index",
    "path",
    "file_size_bytes",
    "data_offset_bytes",
    "expected_file_size_bytes",
    "file_size_matches",
    "bitpix",
    "naxis1",
    "naxis2",
    "shape",
    "shape_matches_header",
    "dtype",
    "bscale",
    "bzero",
    "blank",
    "has_bscale",
    "has_bzero",
    "has_blank",
    "timestamp",
    "exposure_ms",
    "auxiliary_complete",
    "auxiliary_masked_pixel_count",
    "wcs_standard_card_count",
    "wcs_standard_cards_present",
    "wcs_standard_cards_missing",
    "wcs_minimal_core_present",
    "valid_pixel_count",
    "signed_min",
    "signed_max",
    "negative_count",
    "zero_count",
    "exact_minus_one_count",
    "negative_extreme_count",
    "positive_extreme_count",
    "unsigned_diagnostic_min",
    "unsigned_diagnostic_max",
];
const VELOCITY_COLUMNS: [&str; 10] = [
    "frame_number",
    "frame_index",
    "path",
    "timestamp",
    "dt_seconds",
    "derivative_method",
    "j2000_position_velocity_error",
    "j2000_position_velocity_relative_error",
    "wgs84_position_velocity_error",
    "wgs84_position_velocity_relative_error",
];

fn cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        _ => value.to_string(),
    }
}

fn write_csv(path: &Path, columns: &[&str], rows: &[Value]) -> Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(columns)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(columns.iter().map(|column| {
                if *column == "frame_number" {
                    row.get("frame_number").map(cell).unwrap_or_default()
                } else {
                    row.get(*column).map(cell).unwrap_or_default()
                }
            }))
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    let mut file = writer.into_inner().map_err(|error| error.to_string())?;
    file.flush().map_err(|error| error.to_string())?;
    Ok(())
}

pub fn write_format_audit_artifacts(result: &Value, output_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    let frames = result["frames"]
        .as_array()
        .ok_or("format result frames missing")?;
    let velocity = result["velocity"]
        .as_array()
        .ok_or("format result velocity missing")?;
    write_csv(
        &output_dir.join("format_audit_frames.csv"),
        &FRAME_COLUMNS,
        frames,
    )?;
    write_csv(
        &output_dir.join("format_audit_velocity.csv"),
        &VELOCITY_COLUMNS,
        velocity,
    )?;
    fs::write(
        output_dir.join("format_audit.json"),
        serde_json::to_vec_pretty(result).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(output_dir.to_owned())
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        println!("审计 FITS 实际尺寸、字节序、特殊值和辅助位置-速度自洽性，不修改检测器\n\nUsage: rst19-format-audit INPUT_PATH [--extreme-fraction VALUE] [--out-dir PATH]");
        return Ok(());
    }
    let mut input = None;
    let mut extreme_fraction = 0.9;
    let mut output = PathBuf::from("tmp/format-audit");
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--extreme-fraction" => {
                index += 1;
                extreme_fraction = args
                    .get(index)
                    .ok_or("--extreme-fraction requires a value")?
                    .parse::<f64>()
                    .map_err(|_| "invalid --extreme-fraction")?;
            }
            "--out-dir" => {
                index += 1;
                output = PathBuf::from(args.get(index).ok_or("--out-dir requires a path")?);
            }
            "--" => {
                index += 1;
                while index < args.len() {
                    if input.replace(PathBuf::from(&args[index])).is_some() {
                        return Err("only one input path is allowed".into());
                    }
                    index += 1;
                }
                break;
            }
            value if value.starts_with('-') => return Err(format!("unknown argument {value}")),
            value => {
                if input.replace(PathBuf::from(value)).is_some() {
                    return Err("only one input path is allowed".into());
                }
            }
        }
        index += 1;
    }
    let input = input.ok_or("missing FITS file or directory")?;
    let result = run_format_audit(&input, extreme_fraction)?;
    let output = write_format_audit_artifacts(&result, &output)?;
    let summary = &result["summary"];
    println!("format audit: {}", output.display());
    println!(
        "frames={} shape_counts={} file_size_ok={} auxiliary={} velocity_j2000_median={}",
        summary["frame_count"],
        summary["shape_counts"],
        summary["all_file_sizes_match_header"],
        summary["complete_auxiliary_frames"],
        summary["velocity_consistency"]["j2000_relative_error_median"]
    );
    Ok(())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut input = None;
    let mut extreme_fraction = 0.9;
    let mut output = PathBuf::from("tmp/format-audit");
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--extreme-fraction" => {
                index += 1;
                extreme_fraction = args
                    .get(index)
                    .ok_or("--extreme-fraction requires a value")?
                    .parse::<f64>()
                    .map_err(|_| "invalid --extreme-fraction")?;
            }
            "--out-dir" => {
                index += 1;
                output = PathBuf::from(args.get(index).ok_or("--out-dir requires a path")?);
            }
            "--" => {
                index += 1;
                while index < args.len() {
                    if input.replace(PathBuf::from(&args[index])).is_some() {
                        return Err("only one input path is allowed".into());
                    }
                    index += 1;
                }
                break;
            }
            value if value.starts_with('-') => return Err(format!("unknown argument {value}")),
            value => {
                if input.replace(PathBuf::from(value)).is_some() {
                    return Err("only one input path is allowed".into());
                }
            }
        }
        index += 1;
    }
    let input = input.ok_or("missing FITS file or directory")?;
    let result = run_format_audit(&input, extreme_fraction)?;
    let output = write_format_audit_artifacts(&result, &output)?;
    Ok(json!({
        "tool": "rst19-format-audit",
        "out_dir": output,
        "result_path": output.join("format_audit.json"),
        "result": result
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_timestamp_offsets_and_microseconds_normalize_to_utc() {
        assert_eq!(
            parse_iso_timestamp("2026-03-30T16:32:05+02:00"),
            parse_iso_timestamp("2026-03-30T14:32:05Z")
        );
        let (_, text) = timestamp_text(Some(&Value::String(
            "2026-03-30T16:32:05.120000+02:00".into(),
        )));
        assert_eq!(text.as_deref(), Some("2026-03-30T14:32:05.120000+00:00"));
        assert!(parse_iso_timestamp("2026-02-30T00:00:00Z").is_none());
    }
}
