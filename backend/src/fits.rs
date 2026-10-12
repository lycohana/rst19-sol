//! Port of rst19.fits: primary HDU, raw ADU, auxiliary telemetry, exposure units.
use crate::{
    models::{AuxiliaryData, FitsFrame, PixelData},
    Result,
};
use regex::Regex;
use serde_json::{Map, Value};
use std::{fs::File, io::Read, path::Path, sync::LazyLock};

pub const BLOCK_SIZE: usize = 2880;
pub const AUXILIARY_BYTES: usize = 208;
static NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?").unwrap());
static EXPOSURE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)\s*([A-Za-zµμ]*)\s*$")
        .unwrap()
});

pub fn parse_card_value(raw: &str) -> Value {
    let value = raw.split('/').next().unwrap_or("").trim();
    if value.is_empty() {
        return Value::Null;
    }
    if let Some(start) = value.strip_prefix('\'') {
        return Value::from(match value.rfind('\'') {
            Some(end) if end > 0 => value[1..end].replace("''", "'"),
            _ => start.into(),
        });
    }
    match value {
        "T" => return Value::Bool(true),
        "F" => return Value::Bool(false),
        _ => (),
    }
    let numeric = value.replace('D', "E").replace('d', "e");
    fn number(text: &str) -> Option<Value> {
        if text.contains(['.', 'e', 'E']) {
            text.parse::<f64>().ok().map(Value::from)
        } else {
            text.parse::<i64>().ok().map(Value::from)
        }
    }
    if let Some(parsed) = number(&numeric) {
        return parsed;
    }
    let tokens: Vec<_> = NUMBER.find_iter(&numeric).map(|m| m.as_str()).collect();
    if tokens.is_empty() {
        return Value::from(value);
    }
    let selected = tokens
        .iter()
        .find(|s| s.parse::<f64>().is_ok_and(|v| v != 0.0))
        .unwrap_or(&tokens[0]);
    number(selected).unwrap_or_else(|| Value::from(value))
}

pub fn decode_auxiliary(raw: &[u8]) -> Result<AuxiliaryData> {
    if raw.len() != AUXILIARY_BYTES {
        return Err(format!(
            "expected {AUXILIARY_BYTES} auxiliary bytes, got {}",
            raw.len()
        ));
    }
    let mut swapped = raw.to_vec();
    for pair in swapped.as_chunks_mut::<2>().0 {
        pair.swap(0, 1);
    }
    let mut values = [0.0; 26];
    for (index, bytes) in swapped.as_chunks::<8>().0.iter().enumerate() {
        values[index] = f64::from_le_bytes(*bytes);
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err("auxiliary data contains NaN or infinity".into());
    }
    Ok(AuxiliaryData { values })
}

pub fn finite_header_float(header: &Map<String, Value>, key: &str) -> Option<f64> {
    let value = header.get(key)?;
    let number = value
        .as_f64()
        .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
        .or_else(|| value.as_bool().map(|b| if b { 1.0 } else { 0.0 }))?;
    number.is_finite().then_some(number)
}
pub fn wrapped_degrees_delta(value: f64, reference: f64) -> f64 {
    (value - reference + 180.0).rem_euclid(360.0) - 180.0
}
pub fn auxiliary_is_plausible(aux: &AuxiliaryData, header: &Map<String, Value>) -> bool {
    let v = &aux.values;
    if !(0.0..360.0).contains(&v[0])
        || !(-90.0..=90.0).contains(&v[1])
        || !(0.0..360.0).contains(&v[2])
        || !(-90.0..=90.0).contains(&v[3])
    {
        return false;
    }
    let norm = v[10..14].iter().map(|q| q * q).sum::<f64>().sqrt();
    if !norm.is_finite() || !(1e-6..=1e6).contains(&norm) {
        return false;
    }
    if finite_header_float(header, "AZIMUTH")
        .is_some_and(|az| wrapped_degrees_delta(v[0], az).abs() > 0.01)
    {
        return false;
    }
    if finite_header_float(header, "ELEVATIO").is_some_and(|el| (v[1] - el).abs() > 0.01) {
        return false;
    }
    true
}
pub fn auxiliary_mask(height: usize, width: usize) -> Result<Vec<bool>> {
    if height == 0 || width == 0 {
        return Err("expected a non-empty 2-D image shape".into());
    }
    let size = height.checked_mul(width).ok_or("image shape overflow")?;
    let mut mask = vec![false; size];
    mask[..width.min(104)].fill(true);
    Ok(mask)
}
fn normalise_unit(value: &str) -> Option<&'static str> {
    match value
        .trim()
        .to_lowercase()
        .replace('μ', "µ")
        .replace(' ', "")
        .as_str()
    {
        "s" | "sec" | "secs" | "second" | "seconds" => Some("s"),
        "ms" | "msec" | "msecs" | "millisecond" | "milliseconds" => Some("ms"),
        _ => None,
    }
}
pub fn exposure_seconds(header: &Map<String, Value>, default: f64) -> Result<f64> {
    if !default.is_finite() || default <= 0.0 {
        return Err("default exposure must be positive".into());
    }
    for (key, default_unit, unit_keys) in [
        ("EXPTIME", "s", ["EXPTIME_UNIT", "TIMEUNIT"]),
        ("EXPOSURE", "ms", ["EXPOSURE_UNIT", "TIMEUNIT"]),
    ] {
        let Some(value) = header.get(key) else {
            continue;
        };
        let (number, inline) = if let Some(n) = value.as_f64() {
            (Some(n), None)
        } else if let Some(s) = value.as_str() {
            if let Some(c) = EXPOSURE.captures(s) {
                (
                    c[1].replace('D', "E").replace('d', "e").parse::<f64>().ok(),
                    c.get(2).and_then(|s| normalise_unit(s.as_str())),
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        let Some(number) = number.filter(|n| n.is_finite() && *n > 0.0) else {
            continue;
        };
        let unit = inline
            .or_else(|| {
                unit_keys.iter().find_map(|key| {
                    header
                        .get(*key)
                        .and_then(Value::as_str)
                        .and_then(normalise_unit)
                })
            })
            .unwrap_or(default_unit);
        let seconds = if unit == "ms" {
            number / 1000.0
        } else {
            number
        };
        if seconds.is_finite() && seconds > 0.0 {
            return Ok(seconds);
        }
    }
    Ok(default)
}
pub fn exposure_milliseconds(header: &Map<String, Value>) -> Result<Option<f64>> {
    if !header.contains_key("EXPTIME") && !header.contains_key("EXPOSURE") {
        return Ok(None);
    }
    Ok(Some(exposure_seconds(header, 1.0)? * 1000.0))
}

fn read_header(stream: &mut impl Read) -> Result<(Map<String, Value>, usize)> {
    let mut header = Map::new();
    let mut size = 0usize;
    loop {
        let mut block = [0u8; BLOCK_SIZE];
        stream
            .read_exact(&mut block)
            .map_err(|_| "FITS header is truncated before a complete 2880-byte block")?;
        size = size
            .checked_add(BLOCK_SIZE)
            .ok_or("FITS header size overflow")?;
        for bytes in block.as_chunks::<80>().0 {
            let card: String = bytes
                .iter()
                .map(|b| if b.is_ascii() { *b as char } else { '?' })
                .collect();
            let key = card[..8].trim();
            if key == "END" {
                return Ok((header, size));
            }
            if !key.is_empty() && bytes[8] == b'=' {
                header.insert(key.into(), parse_card_value(&card[10..]));
            }
        }
    }
}
fn dimension(header: &Map<String, Value>, key: &str) -> Result<usize> {
    let value = header
        .get(key)
        .ok_or("FITS header must contain integer NAXIS1 and NAXIS2")?;
    let integer = value
        .as_i64()
        .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
        .or_else(|| value.as_f64().filter(|v| v.is_finite()).map(|v| v as i64));
    match integer {
        Some(v) if v > 0 => usize::try_from(v).map_err(|e| e.to_string()),
        _ => Err("invalid image shape".into()),
    }
}
pub fn read_fits(path: impl AsRef<Path>, apply_scaling: bool) -> Result<FitsFrame> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|e| {
        format!(
            "FITS file does not exist or cannot be read: {}: {e}",
            path.display()
        )
    })?;
    let (header, offset) = read_header(&mut file)?;
    if header.get("SIMPLE") != Some(&Value::Bool(true)) {
        return Err("primary HDU is missing SIMPLE=T".into());
    }
    if header.get("NAXIS").and_then(Value::as_f64) != Some(2.0) {
        return Err(format!(
            "expected a 2-D primary image, got NAXIS={:?}",
            header.get("NAXIS")
        ));
    }
    let bitpix = header
        .get("BITPIX")
        .and_then(Value::as_i64)
        .ok_or("unsupported BITPIX")?;
    let bytes_per_pixel = match bitpix {
        8 => 1,
        16 => 2,
        32 | -32 => 4,
        64 | -64 => 8,
        _ => return Err(format!("unsupported BITPIX={bitpix}")),
    };
    let width = dimension(&header, "NAXIS1")?;
    let height = dimension(&header, "NAXIS2")?;
    let count = width
        .checked_mul(height)
        .ok_or("FITS dimensions overflow")?;
    let expected = count
        .checked_mul(bytes_per_pixel)
        .ok_or("FITS image byte count overflow")?;
    let remaining = file
        .metadata()
        .map_err(|e| e.to_string())?
        .len()
        .saturating_sub(offset as u64);
    if remaining < expected as u64 {
        return Err(format!(
            "FITS image data is truncated: expected {expected}, got {remaining}"
        ));
    }
    let mut raw = Vec::new();
    raw.try_reserve_exact(expected).map_err(|e| e.to_string())?;
    raw.resize(expected, 0);
    file.read_exact(&mut raw).map_err(|e| e.to_string())?;
    let auxiliary = if raw.len() >= AUXILIARY_BYTES {
        decode_auxiliary(&raw[..AUXILIARY_BYTES])
            .ok()
            .filter(|a| auxiliary_is_plausible(a, &header))
    } else {
        None
    };
    macro_rules! pixels {
        ($size:expr,$type:ty,$variant:ident) => {
            PixelData::$variant(
                raw.chunks_exact($size)
                    .map(|b| <$type>::from_be_bytes(b.try_into().unwrap()))
                    .collect(),
            )
        };
    }
    let mut data = match bitpix {
        8 => PixelData::U8(raw),
        16 => pixels!(2, i16, I16),
        32 => pixels!(4, i32, I32),
        64 => pixels!(8, i64, I64),
        -32 => pixels!(4, f32, F32),
        -64 => pixels!(8, f64, F64),
        _ => unreachable!(),
    };
    if apply_scaling {
        let scaling_value = |key: &str, default: f64| -> Result<f64> {
            match header.get(key) {
                None => Ok(default),
                Some(value) => value
                    .as_f64()
                    .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
                    .or_else(|| value.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                    .ok_or_else(|| format!("invalid FITS scaling value {key}")),
            }
        };
        let scale = scaling_value("BSCALE", 1.0)?;
        let zero = scaling_value("BZERO", 0.0)?;
        data = PixelData::F64(
            data.to_f64()
                .into_iter()
                .map(|v| v * scale + zero)
                .collect(),
        );
    }
    Ok(FitsFrame {
        path: path.to_owned(),
        header,
        data,
        auxiliary,
        data_offset: offset,
        width,
        height,
    })
}
