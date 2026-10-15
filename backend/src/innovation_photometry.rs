//! Consumption of photometric evidence without fitting or promoting diagnostics.
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use unicode_casefold::UnicodeCaseFold;

type Obj = Map<String, Value>;
const ABS_VALUES: &[&str] = &[
    "M_G",
    "M_V",
    "M",
    "absolute_magnitude_value",
    "absolute_mag",
    "m_abs",
];
const ABS_STRICT: &[&str] = &["strict_absolute_magnitude", "strict_M"];
const BANDS: &[&str] = &[
    "absolute_magnitude_band",
    "absolute_band",
    "M_band",
    "absolute_passband",
    "magnitude_band",
    "photometric_band",
    "band",
    "passband",
];
const EXT_BANDS: &[&str] = &[
    "absolute_magnitude_extinction_band",
    "extinction_band",
    "ext_band",
    "extinction_passband",
];
const SYSTEMS: &[&str] = &[
    "absolute_magnitude_system",
    "absolute_system",
    "photometric_system",
    "system",
];
const EXT_SYSTEMS: &[&str] = &[
    "absolute_magnitude_extinction_system",
    "extinction_system",
    "extinction_photometric_system",
    "ext_system",
];
const ABS_STATUS: &[&str] = &["absolute_status", "M_status", "absolute_magnitude_status"];
const DIST_SOURCE: &[&str] = &[
    "absolute_magnitude_source",
    "absolute_magnitude_distance_source",
    "distance_source",
    "distance_method",
    "distance_origin",
    "absolute_source",
    "absolute_method",
];
const INTERVAL: &[&str] = &[
    "absolute_magnitude_distance_interval_status",
    "distance_interval_status",
    "distance_interval_state",
];
const LOWER: &[&str] = &[
    "absolute_magnitude_distance_lower_pc",
    "distance_lower_pc",
    "distance_lower",
];
const UPPER: &[&str] = &[
    "absolute_magnitude_distance_upper_pc",
    "distance_upper_pc",
    "distance_upper",
];
const EXT_VALUES: &[&str] = &[
    "absolute_magnitude_extinction_mag",
    "extinction_mag",
    "extinction",
    "A_V",
    "av",
];
const VALID_CALIBRATION: &[&str] = &[
    "VALID",
    "VALID_NO_HOLDOUT",
    "CALIBRATED",
    "CALIBRATION_VALID",
    "RELATIVE_CALIBRATED",
    "APPARENT_CALIBRATED",
    "ABSOLUTE_ELIGIBLE",
];
const REJECTED: &[&str] = &[
    "CATALOG_INCONSISTENT",
    "PHOTOMETRIC_OUTLIER",
    "PHOTOMETRICALLY_INCONSISTENT",
    "INSTRUMENTAL",
    "INSTRUMENTAL_ONLY",
    "CATALOG_MATCH_NO_CALIBRATION",
    "CALIBRATION_MISSING_COLOR",
    "CALIBRATION_INVALID",
    "CALIBRATION_NOT_APPLIED",
    "CALIBRATION_METADATA_MISMATCH",
    "CALIBRATION_COLOR_METADATA_MISMATCH",
    "NO_POSITIVE_FLUX",
    "NO_PHOTOMETRY_ROW",
    "NO_APPARENT_MAGNITUDE",
    "PHOTOMETRY_REJECTED_QUALITY_GATE",
    "PHOTOMETRIC_METADATA_REQUIRED",
    "PHOTOMETRIC_METADATA_MISMATCH",
    "QUALITY_REJECTED",
    "REJECTED_QUALITY",
    "INSUFFICIENT_CALIBRATORS",
    "INSUFFICIENT_COLOR_RANGE",
    "INSUFFICIENT_INLIERS",
    "DUPLICATE_MATCHES",
];

fn text(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::Null => "None".into(),
        _ => value.to_string(),
    }
}
fn key(value: &str) -> String {
    value
        .trim()
        .to_lowercase()
        .chars()
        .map(|character| {
            if character.is_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .into()
}
fn lookup<'a, 'b>(
    mapping: Option<&'a Obj>,
    aliases: &'b [&'b str],
) -> Option<(&'b str, &'a Value)> {
    let mapping = mapping?;
    let normalized = mapping
        .iter()
        .map(|(name, value)| (key(name), value))
        .collect::<BTreeMap<_, _>>();
    aliases
        .iter()
        .find_map(|alias| normalized.get(&key(alias)).map(|value| (*alias, *value)))
}
fn value<'a>(mapping: Option<&'a Obj>, aliases: &[&str]) -> &'a Value {
    lookup(mapping, aliases)
        .map(|(_, value)| value)
        .unwrap_or(&Value::Null)
}
fn number(value: &Value) -> Option<f64> {
    let value = match value {
        Value::Bool(value) => Some(u8::from(*value) as f64),
        Value::String(value) => value.trim().parse().ok(),
        _ => value.as_f64(),
    }?;
    value.is_finite().then_some(value)
}
fn label(value: &Value) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let text = text(value)
        .split(|character: char| {
            character.is_whitespace() || ('\u{001c}'..='\u{001f}').contains(&character)
        })
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let folded = text.as_str().case_fold().collect::<String>();
    (!matches!(
        folded.as_str(),
        "" | "unknown" | "none" | "null" | "n/a" | "na" | "undefined" | "?" | "-" | "—"
    ))
    .then_some(text)
}
fn band(value: &Value) -> Option<String> {
    let text = label(value)?;
    let folded = text
        .as_str()
        .case_fold()
        .filter(|character| character.is_alphanumeric())
        .collect::<String>();
    Some(match folded.as_str() {
        "g" | "gaiag" | "gaiadr3g" => "G".into(),
        "v" | "johnsonv" | "johnsoncousinsv" => "V".into(),
        _ => text,
    })
}
fn system(value: &Value) -> Option<String> {
    let text = label(value)?;
    let folded = text
        .as_str()
        .case_fold()
        .filter(|character| character.is_alphanumeric())
        .collect::<String>();
    Some(match folded.as_str() {
        "gaia" | "gaiavega" | "gaiadr3" | "gaiadr3vega" => "Gaia".into(),
        "johnson" | "johnsoncousins" | "johnsonv" | "johnsoncousinsv" => "Johnson".into(),
        "monochromatic" | "monochromatic5414nm" | "a0" => "monochromatic".into(),
        _ => text,
    })
}
fn status(value: &Value) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let text = text(value).trim().to_uppercase().replace(['-', ' '], "_");
    (!text.is_empty()).then_some(text)
}
fn boolean(value: &Value) -> Option<bool> {
    if let Some(value) = value.as_bool() {
        return Some(value);
    }
    if value.is_number() {
        return value.as_f64().and_then(|value| {
            if value == 1.0 {
                Some(true)
            } else if value == 0.0 {
                Some(false)
            } else {
                None
            }
        });
    }
    match value.as_str()?.trim().to_lowercase().as_str() {
        "true" | "yes" | "y" | "1" | "pass" | "passed" | "ok" => Some(true),
        "false" | "no" | "n" | "0" | "fail" | "failed" | "invalid" => Some(false),
        _ => None,
    }
}
fn strings(value: &Value) -> Vec<String> {
    match value {
        Value::String(value) => vec![value.clone()],
        Value::Array(values) => values
            .iter()
            .take(64)
            .filter(|value| !value.is_null())
            .map(text)
            .collect(),
        _ => vec![],
    }
}
fn first_field(sources: &[(&str, &Obj)], aliases: &[&str]) -> (Value, Option<String>) {
    for (name, mapping) in sources {
        if let Some((alias, value)) = lookup(Some(mapping), aliases) {
            if label(value).is_some() {
                return (value.clone(), Some(format!("{name}.{alias}")));
            }
        }
    }
    (Value::Null, None)
}
fn first_number(sources: &[(&str, &Obj)], aliases: &[&str]) -> (bool, Option<f64>, Option<String>) {
    for (name, mapping) in sources {
        if let Some((alias, value)) = lookup(Some(mapping), aliases) {
            if !value.is_null() && *value != "" {
                return (true, number(value), Some(format!("{name}.{alias}")));
            }
        }
    }
    (false, None, None)
}
fn folded_equal(first: &str, second: &str) -> bool {
    first.case_fold().collect::<String>() == second.case_fold().collect::<String>()
}

#[derive(Clone)]
struct Candidate {
    raw: Value,
    value: Option<f64>,
    source: String,
    named_band: Option<String>,
}
fn candidates(mapping: &Obj, aliases: &[&str], source: &str) -> Vec<Candidate> {
    aliases
        .iter()
        .filter_map(|alias| {
            let (_, raw) = lookup(Some(mapping), &[*alias])?;
            if raw.is_object() {
                return None;
            }
            Some(Candidate {
                raw: raw.clone(),
                value: number(raw),
                source: format!("{source}.{alias}"),
                named_band: match alias.to_uppercase().as_str() {
                    "M_G" => Some("G".into()),
                    "M_V" => Some("V".into()),
                    _ => None,
                },
            })
        })
        .collect()
}

fn reason_text(code: &str) -> &str {
    match code {
        "ABSOLUTE_RESULT_MISSING" => "没有绝对星等结果",
        "ABSOLUTE_VALUE_INVALID" => "绝对星等数值缺失或不是有限数",
        "IS_STRICT_MISSING" => "旧结果缺少 is_strict，不能确认严格绝对星等",
        "IS_STRICT_FALSE" => "绝对星等未通过 is_strict 严格质量门",
        "ABSOLUTE_STATUS_MISSING" => "缺少绝对星等状态",
        "ABSOLUTE_STATUS_NOT_STRICT" => "绝对星等状态不属于严格有效状态",
        "MODEL_DISTANCE_INTERVAL_MISSING" => "缺少完整距离区间数值（中心值、下界和上界）",
        "DISTANCE_INTERVAL_STATUS_MISSING" => "缺少完整距离区间状态",
        "DISTANCE_INTERVAL_STATUS_INVALID" => "距离区间状态不允许严格绝对星等",
        "DISTANCE_SOURCE_REQUIRED" => "缺少距离来源",
        "DISTANCE_SOURCE_INVALID" => "距离来源为空或不可用于严格结果",
        "EXTINCTION_NOT_PROVIDED" => "未提供消光修正",
        "EXTINCTION_INVALID" => "消光值无效",
        "EXTINCTION_SEMANTICS_REQUIRED" => "缺少消光系统/波段语义",
        "EXTINCTION_BAND_MISMATCH" => "绝对星等波段与消光波段不一致",
        "EXTINCTION_SYSTEM_MISMATCH" => "绝对星等系统与消光系统不一致",
        "ABSOLUTE_BAND_REQUIRED" => "未声明绝对星等波段",
        "ABSOLUTE_BAND_MISMATCH" => "M_G/M_V 的波段身份与输入波段不一致",
        "ABSOLUTE_BAND_AMBIGUOUS" => "同一结果同时包含不同波段的绝对星等，不能合并",
        "ROW_PHOTOMETRY_REJECTED" => "逐源光度状态已拒绝该结果",
        _ => code,
    }
}

pub fn absolute_magnitude_evidence(row: &Value) -> Value {
    let empty = Obj::new();
    let row = row.as_object().unwrap_or(&empty);
    let absolute = lookup(
        Some(row),
        &["absolute_magnitude", "absolute", "absolute_result"],
    );
    let nested = absolute.and_then(|(_, value)| value.as_object());
    let mut sources = Vec::new();
    if let Some(nested) = nested {
        sources.push((absolute.unwrap().0, nested));
    }
    sources.push(("row", row));
    let mut records = if let Some(nested) = nested {
        let name = absolute.unwrap().0;
        let named = candidates(nested, &ABS_VALUES[..2], name);
        if named.is_empty() {
            candidates(
                nested,
                &[
                    "M",
                    "absolute_magnitude_value",
                    "absolute_mag",
                    "m_abs",
                    "value",
                    "absolute_magnitude",
                ],
                name,
            )
        } else {
            named
        }
    } else if let Some((alias, raw)) =
        absolute.filter(|(_, value)| !value.is_null() && **value != "")
    {
        vec![Candidate {
            raw: raw.clone(),
            value: number(raw),
            source: alias.into(),
            named_band: None,
        }]
    } else {
        vec![]
    };
    if records.is_empty() {
        records = candidates(row, &ABS_VALUES[..2], "row");
        if records.is_empty() {
            records = candidates(row, &ABS_VALUES[2..], "row");
        }
    }
    let strict_records = sources
        .iter()
        .flat_map(|(name, mapping)| candidates(mapping, ABS_STRICT, name))
        .collect::<Vec<_>>();
    let value_found = !records.is_empty() || !strict_records.is_empty();
    let selected = records.iter().find(|record| record.value.is_some());
    let strict_record = strict_records.iter().find(|record| record.value.is_some());
    let diagnostic_value = selected.and_then(|record| record.value);
    let value_source = selected.map(|record| record.source.clone());
    let strict_candidate_value = strict_record.and_then(|record| record.value);
    let strict_value_source = strict_record.map(|record| record.source.clone());
    let strict_candidate_invalid = strict_records.iter().any(|record| {
        record.value.is_none()
            && !record.raw.is_null()
            && !record
                .raw
                .as_str()
                .is_some_and(|value| value.trim().is_empty())
    });
    let named_bands = records
        .iter()
        .filter(|record| record.value.is_some())
        .filter_map(|record| record.named_band.clone())
        .collect::<BTreeSet<_>>();
    let selected_band = selected.and_then(|record| record.named_band.clone());
    let mut diagnostics = Obj::new();
    for record in &records {
        if let Some(value) = record.value {
            diagnostics
                .entry(
                    record
                        .named_band
                        .as_ref()
                        .map_or("M".into(), |band| format!("M_{band}")),
                )
                .or_insert(json!(value));
        }
    }
    let (explicit_band_raw, explicit_band_source) = first_field(&sources, BANDS);
    let (extinction_band_raw, extinction_band_source) = first_field(&sources, EXT_BANDS);
    let explicit_band = band(&explicit_band_raw);
    let extinction_band = band(&extinction_band_raw);
    let magnitude_band = selected_band
        .clone()
        .or_else(|| explicit_band.clone())
        .or_else(|| extinction_band.clone());
    let band_source = if selected_band.is_some() {
        value_source.clone()
    } else if explicit_band.is_some() {
        explicit_band_source.clone()
    } else {
        extinction_band_source.clone()
    };
    if diagnostic_value.is_some() && named_bands.is_empty() {
        diagnostics.clear();
        diagnostics.insert(
            magnitude_band
                .as_ref()
                .map_or("M".into(), |band| format!("M_{band}")),
            json!(diagnostic_value),
        );
    }
    let (explicit_system_raw, explicit_system_source) = first_field(&sources, SYSTEMS);
    let (extinction_system_raw, extinction_system_source) = first_field(&sources, EXT_SYSTEMS);
    let explicit_system = system(&explicit_system_raw);
    let extinction_system = system(&extinction_system_raw);
    let magnitude_system = explicit_system
        .clone()
        .or_else(|| extinction_system.clone());
    let system_source = if explicit_system.is_some() {
        explicit_system_source
    } else {
        extinction_system_source.clone()
    };
    let (mut status_raw, mut status_source) = first_field(&sources, ABS_STATUS);
    if status_raw.is_null() {
        (status_raw, status_source) = first_field(&sources, &["status", "state"]);
    }
    let absolute_status = status(&status_raw);
    let row_status = status(value(Some(row), &["status", "state"]));
    let calibration_status = status(value(Some(row), &["calibration_status", "m_cal_status"]));
    let consistent = boolean(value(Some(row), &["photometric_consistent"]));
    let quality = boolean(value(Some(row), &["quality_passed", "quality_ok"]));
    let (strict_raw, strict_marker_source) =
        first_field(&sources, &["is_strict", "absolute_magnitude_is_strict"]);
    let strict_marker = strict_marker_source
        .as_ref()
        .and_then(|_| boolean(&strict_raw));
    let (distance_source_raw, distance_source_field) = first_field(&sources, DIST_SOURCE);
    let distance_source = label(&distance_source_raw);
    let (interval_raw, interval_source) = first_field(&sources, INTERVAL);
    let mut interval = status(&interval_raw);
    let mut flags = Vec::new();
    for (_, mapping) in &sources {
        for flag in strings(value(Some(mapping), &["flags", "flag"])) {
            flags.push(status(&json!(flag)).unwrap_or(flag));
        }
    }
    if interval.is_none() {
        for flag in &flags {
            if let Some(value) = match flag.as_str() {
                "DISTANCE_INTERVAL_INVALID" => Some("INVALID"),
                "DISTANCE_INTERVAL_DERIVED_FROM_PARALLAX_ERROR" => {
                    Some("DERIVED_FROM_PARALLAX_ERROR")
                }
                "DISTANCE_INTERVAL_USED" => Some("PROVIDED"),
                "DISTANCE_INTERVAL_ONE_SIDED" => Some("ONE_SIDED"),
                "DISTANCE_INTERVAL_REQUIRED" => Some("REQUIRED"),
                "DISTANCE_ERROR_NOT_PROVIDED" => Some("NOT_PROVIDED"),
                _ => None,
            } {
                interval = Some(value.into());
                break;
            }
        }
    }
    let (lower_found, lower, _) = first_number(&sources, LOWER);
    let (upper_found, upper, _) = first_number(&sources, UPPER);
    if interval.is_none() && (lower_found || upper_found) {
        interval = Some("PRESENT_UNCLASSIFIED".into());
    }
    let (extinction_found, extinction, extinction_value_source) =
        first_number(&sources, EXT_VALUES);
    let (_, distance, _) = first_number(
        &sources,
        &["absolute_magnitude_distance_pc", "distance_pc", "distance"],
    );
    let has_evidence = value_found || nested.is_some();
    let gate_value = strict_candidate_value.or(diagnostic_value);
    let mut reasons = Vec::<String>::new();
    let mut reason = |code: &str| {
        if !reasons.iter().any(|value| value == code) {
            reasons.push(code.into());
        }
    };
    if gate_value.is_none() {
        reason(if value_found {
            "ABSOLUTE_VALUE_INVALID"
        } else {
            "ABSOLUTE_RESULT_MISSING"
        });
    }
    if has_evidence {
        if strict_candidate_invalid {
            reason("ABSOLUTE_VALUE_INVALID");
        }
        if strict_marker_source.is_none() {
            reason("IS_STRICT_MISSING");
        } else if strict_marker != Some(true) {
            reason("IS_STRICT_FALSE");
        } else if !strict_records.is_empty() && strict_candidate_value.is_none() {
            reason("ABSOLUTE_VALUE_INVALID");
        }
        match absolute_status.as_deref() {
            None => reason("ABSOLUTE_STATUS_MISSING"),
            Some("VALID" | "VALID_MODEL_DISTANCE") => {}
            Some("VALID_MODEL_DISTANCE_NO_INTERVAL" | "MODEL_DISTANCE_INTERVAL_REQUIRED") => {
                reason("MODEL_DISTANCE_INTERVAL_MISSING")
            }
            Some("VALID_NO_EXTINCTION" | "MODEL_DISTANCE_NO_EXTINCTION") => {
                reason("EXTINCTION_NOT_PROVIDED")
            }
            Some("MODEL_DISTANCE_SOURCE_REQUIRED" | "NO_PARALLAX" | "NO_DISTANCE") => {
                reason("DISTANCE_SOURCE_REQUIRED")
            }
            Some("INVALID_DISTANCE_INTERVAL") => reason("DISTANCE_INTERVAL_STATUS_INVALID"),
            _ => reason("ABSOLUTE_STATUS_NOT_STRICT"),
        }
        let strict_interval = matches!(
            interval.as_deref(),
            Some("PROVIDED" | "DERIVED_FROM_PARALLAX_ERROR")
        );
        if interval.is_none() {
            reason(
                if matches!(
                    absolute_status.as_deref(),
                    Some("VALID_MODEL_DISTANCE" | "VALID_MODEL_DISTANCE_NO_INTERVAL")
                ) {
                    "MODEL_DISTANCE_INTERVAL_MISSING"
                } else {
                    "DISTANCE_INTERVAL_STATUS_MISSING"
                },
            );
        } else if !strict_interval {
            reason(
                if interval.as_deref() == Some("INVALID")
                    || flags.iter().any(|flag| flag == "DISTANCE_INTERVAL_INVALID")
                {
                    "DISTANCE_INTERVAL_STATUS_INVALID"
                } else {
                    "MODEL_DISTANCE_INTERVAL_MISSING"
                },
            );
        }
        if flags.iter().any(|flag| flag == "DISTANCE_INTERVAL_INVALID")
            || (strict_interval
                && flags.iter().any(|flag| {
                    matches!(
                        flag.as_str(),
                        "DISTANCE_INTERVAL_ONE_SIDED"
                            | "DISTANCE_INTERVAL_REQUIRED"
                            | "DISTANCE_ERROR_NOT_PROVIDED"
                    )
                }))
        {
            reason("DISTANCE_INTERVAL_STATUS_INVALID");
        }
        if strict_interval {
            match distance.zip(lower).zip(upper) {
                None => reason("MODEL_DISTANCE_INTERVAL_MISSING"),
                Some(((distance, lower), upper))
                    if distance <= 0.0
                        || lower <= 0.0
                        || upper <= 0.0
                        || lower > upper
                        || distance < lower
                        || distance > upper =>
                {
                    reason("DISTANCE_INTERVAL_STATUS_INVALID")
                }
                _ => {}
            }
        }
        if distance_source.is_none() {
            reason("DISTANCE_SOURCE_REQUIRED");
        }
        if !extinction_found {
            reason("EXTINCTION_NOT_PROVIDED");
        } else if extinction.is_none_or(|value| value < 0.0) {
            reason("EXTINCTION_INVALID");
        } else {
            if extinction_band.is_none() || extinction_system.is_none() {
                reason("EXTINCTION_SEMANTICS_REQUIRED");
            }
            if magnitude_band
                .as_ref()
                .zip(extinction_band.as_ref())
                .is_some_and(|(a, b)| !folded_equal(a, b))
            {
                reason("EXTINCTION_BAND_MISMATCH");
            }
            if magnitude_system
                .as_ref()
                .zip(extinction_system.as_ref())
                .is_some_and(|(a, b)| !folded_equal(a, b))
            {
                reason("EXTINCTION_SYSTEM_MISMATCH");
            }
        }
        if magnitude_band.is_none() {
            reason("ABSOLUTE_BAND_REQUIRED");
        }
        if named_bands.len() > 1 {
            reason("ABSOLUTE_BAND_AMBIGUOUS");
        }
        if selected_band
            .as_ref()
            .zip(explicit_band.as_ref())
            .is_some_and(|(a, b)| !folded_equal(a, b))
        {
            reason("ABSOLUTE_BAND_MISMATCH");
        }
        if row_status
            .as_ref()
            .is_some_and(|value| REJECTED.contains(&value.as_str()))
            || calibration_status
                .as_ref()
                .is_some_and(|value| !VALID_CALIBRATION.contains(&value.as_str()))
            || consistent == Some(false)
            || quality == Some(false)
            || flags.iter().any(|flag| REJECTED.contains(&flag.as_str()))
        {
            reason("ROW_PHOTOMETRY_REJECTED");
        }
    }
    let strict_value = gate_value.filter(|_| reasons.is_empty());
    let gate_status = if strict_value.is_some() {
        "STRICT"
    } else if diagnostic_value.is_some() || strict_candidate_value.is_some() {
        "DIAGNOSTIC_ONLY"
    } else {
        "NOT_AVAILABLE"
    };
    let magnitude_label = if named_bands.len() > 1 {
        "M_G/M_V".into()
    } else {
        magnitude_band
            .as_ref()
            .map_or("M".into(), |band| format!("M_{band}"))
    };
    let reason_text = reasons
        .iter()
        .map(|reason| reason_text(reason))
        .collect::<Vec<_>>()
        .join("；");
    let mut unique_flags = Vec::new();
    for flag in flags {
        if !flag.is_empty() && !unique_flags.contains(&flag) {
            unique_flags.push(flag);
        }
    }
    json!({"diagnostic_value":diagnostic_value,"strict_value":strict_value,"strict":strict_value.is_some(),"gate_status":gate_status,"strict_reasons":reasons,"strict_reason":reason_text,
        "value_source":value_source,"strict_value_source":strict_value_source,"value_found":value_found,"has_evidence":has_evidence,"strict_candidate_value":strict_candidate_value,"strict_candidate_invalid":strict_candidate_invalid,"numeric_value":gate_value,"diagnostic_values":diagnostics,
        "band":magnitude_band,"band_source":band_source,"label":magnitude_label,"system":magnitude_system,"system_source":system_source,"status":absolute_status,"status_source":status_source,"is_strict":strict_marker,"is_strict_source":strict_marker_source,
        "distance_source":distance_source,"distance_source_field":distance_source_field,"distance_interval_status":interval,"distance_interval_source":interval_source,"distance_pc":distance,"distance_lower_pc":lower,"distance_upper_pc":upper,
        "extinction_mag":extinction,"extinction_value_source":extinction_value_source,"extinction_band":extinction_band,"extinction_system":extinction_system,"extinction_band_source":extinction_band_source,"extinction_system_source":extinction_system_source,"flags":unique_flags})
}

const LEVELS: &[&str] = &[
    "INSTRUMENTAL_ONLY",
    "RELATIVE_CALIBRATED",
    "APPARENT_CALIBRATED",
    "ABSOLUTE_ELIGIBLE",
    "GEOMETRY_UNAVAILABLE",
];
fn count(value: &Value) -> Option<u64> {
    number(value)
        .filter(|value| *value >= 0.0 && *value < u64::MAX as f64)
        .map(|value| value.trunc() as u64)
}
fn count_map(input: &Value) -> BTreeMap<String, u64> {
    input
        .as_object()
        .into_iter()
        .flat_map(|mapping| mapping.iter())
        .filter_map(|(key, value)| count(value).map(|value| (key.clone(), value)))
        .collect()
}
fn finite_map(input: &Value) -> Obj {
    input
        .as_object()
        .into_iter()
        .flat_map(|mapping| mapping.iter())
        .filter_map(|(key, value)| number(value).map(|value| (key.clone(), json!(value))))
        .collect()
}
fn evidence_rows(input: &Value) -> Vec<Value> {
    if let Some(rows) = input.as_array() {
        return rows.iter().filter(|row| row.is_object()).cloned().collect();
    }
    for alias in [
        "rows",
        "source_rows",
        "source_photometry",
        "photometry_rows",
        "photometry_results",
        "results",
        "items",
    ] {
        if let Some(rows) = value(input.as_object(), &[alias]).as_array() {
            return rows.iter().filter(|row| row.is_object()).cloned().collect();
        }
    }
    vec![]
}
fn explicit_level(row: &Value) -> Option<String> {
    status(value(
        row.as_object(),
        &["observability_level", "observability", "level"],
    ))
    .filter(|level| LEVELS.contains(&level.as_str()))
}

pub fn row_evidence_level(row: &Value) -> Option<String> {
    let mut explicit = explicit_level(row);
    let absolute = absolute_magnitude_evidence(row);
    let mut row_status = status(value(
        row.as_object(),
        &["status", "state", "calibration_status"],
    ));
    if row_status.is_none() {
        row_status = absolute["status"].as_str().map(str::to_owned);
    }
    let calibration = status(value(
        row.as_object(),
        &["calibration_status", "m_cal_status"],
    ));
    let quality = boolean(value(row.as_object(), &["quality_passed", "quality_ok"]));
    let flags = strings(value(row.as_object(), &["flags", "flag"]));
    if row_status
        .as_ref()
        .is_some_and(|status| REJECTED.contains(&status.as_str()))
        || flags
            .iter()
            .filter_map(|flag| status(&json!(flag)))
            .any(|flag| REJECTED.contains(&flag.as_str()))
        || quality == Some(false)
        || calibration
            .as_ref()
            .is_some_and(|status| !VALID_CALIBRATION.contains(&status.as_str()))
    {
        return Some("INSTRUMENTAL_ONLY".into());
    }
    if absolute["strict"] == true {
        return Some("ABSOLUTE_ELIGIBLE".into());
    }
    match row_status.as_deref() {
        Some("RELATIVE_CALIBRATED" | "RELATIVE" | "RELATIVE_CALIBRATION") => {
            return Some("RELATIVE_CALIBRATED".into())
        }
        Some(
            "CALIBRATED"
            | "APPARENT_CALIBRATED"
            | "APPARENT"
            | "VALID"
            | "VALID_NO_HOLDOUT"
            | "VALID_MODEL_DISTANCE"
            | "VALID_MODEL_DISTANCE_NO_INTERVAL"
            | "ABSOLUTE_VALID"
            | "ABSOLUTE_ELIGIBLE"
            | "CALIBRATED_ABSOLUTE",
        ) => return Some("APPARENT_CALIBRATED".into()),
        _ => {}
    }
    if explicit.as_deref() == Some("ABSOLUTE_ELIGIBLE") {
        if number(value(
            row.as_object(),
            &[
                "m_cal",
                "calibrated_magnitude",
                "apparent_magnitude",
                "m_std",
            ],
        ))
        .is_some()
        {
            return Some("APPARENT_CALIBRATED".into());
        }
        explicit = None;
    }
    explicit.or_else(|| {
        number(value(
            row.as_object(),
            &["m_inst", "instrumental_magnitude", "instrumental_mag"],
        ))
        .map(|_| "INSTRUMENTAL_ONLY".into())
    })
}

pub fn compact_photometric_row(row: &Value) -> Value {
    let mapping = row.as_object();
    let absolute = absolute_magnitude_evidence(row);
    let mut result = json!({"row_key":value(mapping,&["row_key","key"]),"frame_id":value(mapping,&["frame_id","frame"]),"source_id":value(mapping,&["source_id","detection_id","id"]),"observability_level":row_evidence_level(row),
        "status":value(mapping,&["status","state"]),"m_inst":number(value(mapping,&["m_inst","instrumental_magnitude","instrumental_mag"])),"m_cal":number(value(mapping,&["m_cal","calibrated_magnitude","apparent_magnitude","m_std"])),
        "absolute_magnitude":absolute["diagnostic_value"],"absolute_magnitude_diagnostic":absolute["diagnostic_value"],"absolute_magnitude_strict":absolute["strict_value"],"strict_absolute_magnitude":absolute["strict_value"],"strict_M":absolute["strict_value"],
        "absolute_magnitude_value_role":if absolute["strict"] == true { "STRICT" } else if !absolute["numeric_value"].is_null() { "DIAGNOSTIC_ONLY" } else { "ABSENT" },
        "absolute_magnitude_diagnostics":absolute["diagnostic_values"],"absolute_magnitude_gate_status":absolute["gate_status"],"absolute_magnitude_strict_reasons":absolute["strict_reasons"],"absolute_magnitude_strict_reason":absolute["strict_reason"],
        "absolute_magnitude_status":absolute["status"],"absolute_magnitude_is_strict":absolute["is_strict"],"absolute_magnitude_band":absolute["band"],"absolute_magnitude_label":absolute["label"],"absolute_magnitude_system":absolute["system"],"absolute_magnitude_source":absolute["distance_source"],"absolute_magnitude_band_source":absolute["band_source"],"absolute_magnitude_distance_source":absolute["distance_source"],
        "photometric_residual_mag":number(value(mapping,&["photometric_residual_mag","photometric_residual","catalog_residual_mag"])),"photometric_consistent":boolean(value(mapping,&["photometric_consistent"])),"photometric_outlier_reason":value(mapping,&["photometric_outlier_reason","outlier_reason"]),"calibration_sample_role":value(mapping,&["calibration_sample_role","sample_role"]),
        "flags":strings(value(mapping,&["flags"])),"rejection_reasons":strings(value(mapping,&["rejection_reasons","reasons"])),"missing_inputs":strings(value(mapping,&["missing_inputs"]))});
    for field in [
        "distance_pc",
        "distance_lower_pc",
        "distance_upper_pc",
        "distance_interval_status",
        "extinction_band",
        "extinction_system",
    ] {
        result[format!("absolute_magnitude_{field}")] = absolute[field].clone();
    }
    result
}

pub fn relative_photometry_summary(input: &Value) -> Value {
    let Some(mapping) = input.as_object() else {
        return json!({"present":false,"status":null,"usable":false,"flags":[],"relative_magnitudes":{},"frame_zero_points":{}});
    };
    let relative = finite_map(value(
        Some(mapping),
        &["relative_magnitudes", "source_relative_magnitudes"],
    ));
    let offsets = finite_map(value(Some(mapping), &["frame_zero_points", "zero_points"]));
    let row_status = status(value(Some(mapping), &["status", "state"]));
    let reference = count(value(
        Some(mapping),
        &["reference_sample_count", "reference_count"],
    ));
    let usable = matches!(row_status.as_deref(), Some("VALID" | "VALID_NO_HOLDOUT"))
        && reference.is_some_and(|reference| reference > 0)
        && !relative.is_empty()
        && !offsets.is_empty();
    let values = relative.values().filter_map(number).collect::<Vec<_>>();
    let mut result = json!({"present":true,"status":row_status,"usable":usable,"flags":strings(value(Some(mapping),&["flags"])),"reference_sample_count":reference,
        "reference_sample_count_by_frame":count_map(value(Some(mapping),&["reference_sample_count_by_frame"])),"reference_sample_count_by_source":count_map(value(Some(mapping),&["reference_sample_count_by_source"])),
        "frame_count":offsets.len(),"source_count":relative.len(),"frame_zero_point_count":offsets.len(),"relative_magnitude_count":relative.len(),"frame_zero_points":offsets,"relative_magnitudes":relative,
        "relative_magnitude_range":{"min":values.iter().copied().min_by(f64::total_cmp),"max":values.iter().copied().max_by(f64::total_cmp)},
        "validation_source_count":strings(value(Some(mapping),&["validation_source_ids"])).len(),"absolute_magnitude_claim":"NOT_DERIVED_FROM_RELATIVE_SCALE","note":"这些数值是相对光度尺度和帧零点；创新报告不把它们改名为表观或绝对星等。"});
    for (destination, aliases) in [
        (
            "training_residual_rms_mag",
            vec!["training_residual_rms", "train_rms"],
        ),
        (
            "training_residual_mad_mag",
            vec!["training_residual_mad", "train_mad"],
        ),
        (
            "validation_residual_rms_mag",
            vec!["validation_residual_rms", "validation_rms"],
        ),
        (
            "validation_residual_mad_mag",
            vec!["validation_residual_mad", "validation_mad"],
        ),
    ] {
        result[destination] = json!(number(value(Some(mapping), &aliases)));
    }
    for field in [
        "validation_sample_count",
        "excluded_observation_count",
        "spatial_order",
    ] {
        result[field] = json!(count(value(Some(mapping), &[field])));
    }
    result
}

pub fn photometric_calibration_summary(input: &Value) -> Value {
    let Some(mapping) = input.as_object() else {
        return json!({"present":false});
    };
    let raw_coefficients = value(Some(mapping), &["coefficients"]);
    let coefficients = if let Some(values) = raw_coefficients.as_array() {
        values.iter().filter_map(number).collect::<Vec<_>>()
    } else {
        strings(raw_coefficients)
            .iter()
            .filter_map(|value| number(&json!(value)))
            .collect()
    };
    let mut result = json!({"present":true,"status":status(value(Some(mapping),&["status","state"])),"photometric_system":value(Some(mapping),&["photometric_system","system"]),"photometric_band":value(Some(mapping),&["photometric_band","band"]),"color_name":value(Some(mapping),&["color_name","color"]),"coefficients":coefficients,"flags":strings(value(Some(mapping),&["flags"])),"note":"这是输入中已有的同设备/同条件标定声明；创新报告不重新拟合零点。"});
    for field in [
        "color_order",
        "calibrator_count",
        "inlier_count",
        "validation_count",
    ] {
        result[field] = json!(count(value(Some(mapping), &[field])));
    }
    for field in ["fit_rms_mag", "validation_rms_mag"] {
        result[field] = json!(number(value(Some(mapping), &[field])));
    }
    result
}

fn increment(counts: &mut BTreeMap<String, u64>, key: &str) {
    *counts.entry(key.into()).or_default() += 1;
}

pub fn photometric_quality_summary(input: &Value, source_name: &str) -> Value {
    let mapping = input.as_object();
    let rows = evidence_rows(input);
    if mapping.is_none() && rows.is_empty() {
        return json!({"present":false,"source":source_name,"row_count":0,"sample_rows":[],"sample_truncated":false});
    }
    let declared = |input: &Value| {
        count_map(input)
            .into_iter()
            .filter_map(|(key, count)| {
                status(&json!(key))
                    .filter(|level| LEVELS.contains(&level.as_str()))
                    .map(|level| (level, count))
            })
            .collect::<BTreeMap<_, _>>()
    };
    let mut declared_counts = BTreeMap::new();
    for alias in [
        "observability_counts",
        "level_counts",
        "observability_levels",
    ] {
        declared_counts = declared(value(mapping, &[alias]));
        if !declared_counts.is_empty() {
            break;
        }
    }
    if declared_counts.is_empty() {
        declared_counts = declared(value(
            value(mapping, &["counts"]).as_object(),
            &["observability"],
        ));
    }
    let mut row_counts = BTreeMap::new();
    let (
        mut instrumental,
        mut apparent,
        mut calibrated,
        mut numeric,
        mut results,
        mut diagnostic_only,
        mut eligible,
    ) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut marker_true, mut marker_false, mut marker_missing) = (0u64, 0u64, 0u64);
    let (mut consistent_count, mut inconsistent_count, mut unknown_count) = (0u64, 0u64, 0u64);
    let mut strict_reasons = BTreeMap::new();
    let mut diagnostic_bands = BTreeMap::new();
    let mut strict_bands = BTreeMap::new();
    let mut derived_reasons = BTreeMap::new();
    let mut derived_missing = BTreeMap::new();
    let mut status_counts = BTreeMap::new();
    for row in &rows {
        let level = row_evidence_level(row);
        if let Some(level) = &level {
            increment(&mut row_counts, level);
        }
        if number(value(
            row.as_object(),
            &["m_inst", "instrumental_magnitude", "instrumental_mag"],
        ))
        .is_some()
        {
            instrumental += 1;
        }
        let m_cal = number(value(
            row.as_object(),
            &[
                "m_cal",
                "calibrated_magnitude",
                "apparent_magnitude",
                "m_std",
            ],
        ));
        if m_cal.is_some() {
            apparent += 1;
        }
        if m_cal.is_some()
            && matches!(
                level.as_deref(),
                Some("APPARENT_CALIBRATED" | "ABSOLUTE_ELIGIBLE")
            )
        {
            calibrated += 1;
        }
        let absolute = absolute_magnitude_evidence(row);
        let absolute_label = absolute["label"].as_str().unwrap_or("M");
        if !absolute["numeric_value"].is_null() {
            numeric += 1;
            increment(&mut diagnostic_bands, absolute_label);
        }
        if absolute["has_evidence"] == true {
            results += 1;
            for reason in strings(&absolute["strict_reasons"]) {
                increment(&mut strict_reasons, &reason);
            }
            match absolute["is_strict"].as_bool() {
                Some(true) => marker_true += 1,
                Some(false) => marker_false += 1,
                None => marker_missing += 1,
            }
        }
        if !absolute["numeric_value"].is_null() && absolute["strict"] != true {
            diagnostic_only += 1;
        }
        if absolute["strict"] == true {
            eligible += 1;
            increment(&mut strict_bands, absolute_label);
        }
        if let Some(status) = status(value(row.as_object(), &["status", "state"])) {
            increment(&mut status_counts, &status);
        }
        match boolean(value(row.as_object(), &["photometric_consistent"])) {
            Some(true) => consistent_count += 1,
            Some(false) => inconsistent_count += 1,
            None if number(value(
                row.as_object(),
                &[
                    "photometric_residual_mag",
                    "photometric_residual",
                    "catalog_residual_mag",
                ],
            ))
            .is_some() =>
            {
                unknown_count += 1
            }
            _ => {}
        }
        for reason in strings(value(row.as_object(), &["rejection_reasons", "reasons"])) {
            increment(&mut derived_reasons, &reason);
        }
        for missing in strings(value(row.as_object(), &["missing_inputs"])) {
            increment(&mut derived_missing, &missing);
        }
    }
    let level_counts = if declared_counts.is_empty() {
        row_counts.clone()
    } else {
        declared_counts.clone()
    };
    let explicit_claims = rows
        .iter()
        .filter(|row| explicit_level(row).as_deref() == Some("ABSOLUTE_ELIGIBLE"))
        .count() as u64;
    let mut reported_eligible = declared_counts
        .get("ABSOLUTE_ELIGIBLE")
        .copied()
        .unwrap_or(explicit_claims);
    if rows.is_empty() {
        if let Some(count) = count(value(
            mapping,
            &[
                "absolute_magnitude_diagnostic_count",
                "absolute_magnitude_value_count",
            ],
        )) {
            numeric = numeric.max(count);
            diagnostic_only = diagnostic_only.max(count);
        }
        if let Some(count) = count(value(
            mapping,
            &[
                "reported_absolute_eligible_count",
                "absolute_eligible_count",
            ],
        )) {
            reported_eligible = reported_eligible.max(count);
        }
        results = results.max(numeric);
    }
    let mut false_valid = boolean(value(mapping, &["false_valid"]));
    let mut gate_passed = boolean(value(
        value(mapping, &["false_valid_gate"]).as_object(),
        &["passed"],
    ));
    if false_valid.is_none() {
        false_valid = gate_passed.map(|passed| !passed);
    }
    if gate_passed.is_none() {
        gate_passed = false_valid.map(|value| !value);
    }
    let absolute_gate = if false_valid == Some(true) {
        "REJECTED"
    } else if eligible > 0 {
        "STRICT_AVAILABLE"
    } else if numeric > 0 || reported_eligible > 0 {
        "DIAGNOSTIC_ONLY"
    } else {
        "NOT_AVAILABLE"
    };
    let mut missing = count_map(value(mapping, &["missing_inputs", "missing_input_counts"]));
    if missing.is_empty() {
        missing = derived_missing;
    }
    let mut rejections = count_map(value(mapping, &["rejection_reasons", "reason_counts"]));
    if rejections.is_empty() {
        rejections = derived_reasons;
    }
    let mut provenance = Obj::new();
    if let Some(input) = value(mapping, &["provenance"]).as_object() {
        for (key, raw) in input {
            if raw.is_string()
                || raw.is_boolean()
                || raw.as_i64().is_some()
                || raw.as_u64().is_some()
            {
                provenance.insert(key.clone(), raw.clone());
            } else if let Some(number) = number(raw) {
                provenance.insert(key.clone(), json!(number));
            } else if let Some(values) = raw.as_array() {
                provenance.insert(
                    key.clone(),
                    json!(values
                        .iter()
                        .take(16)
                        .filter(|value| !value.is_null())
                        .map(text)
                        .collect::<Vec<_>>()),
                );
            }
        }
    }
    let samples = rows
        .iter()
        .take(20)
        .map(compact_photometric_row)
        .collect::<Vec<_>>();
    json!({"present":true,"source":source_name,"status":status(value(mapping,&["status","state"])),"frame_count":count(value(mapping,&["frame_count","frames"])),"source_count":count(value(mapping,&["source_count"])).unwrap_or(rows.len() as u64),"row_count":rows.len(),
        "level_counts":level_counts,"derived_level_counts":row_counts,"status_counts":status_counts,"photometric_consistent_count":consistent_count,"photometric_inconsistent_count":inconsistent_count,"photometric_consistency_unknown_count":unknown_count,
        "instrumental_magnitude_count":instrumental,"apparent_magnitude_count":apparent,"calibrated_apparent_magnitude_count":calibrated,
        "absolute_magnitude_value_count":numeric,"absolute_magnitude_diagnostic_count":numeric,"absolute_magnitude_result_count":results,"absolute_magnitude_diagnostic_only_count":diagnostic_only,
        "absolute_magnitude_strict_count":eligible,"strict_absolute_magnitude_count":eligible,"absolute_eligible_count":eligible,"reported_absolute_eligible_count":reported_eligible,"absolute_magnitude_gate_status":absolute_gate,
        "absolute_magnitude_strict_reason_counts":strict_reasons,"absolute_magnitude_diagnostic_band_counts":diagnostic_bands,"absolute_magnitude_strict_band_counts":strict_bands,"absolute_magnitude_is_strict_counts":{"true":marker_true,"false":marker_false,"missing":marker_missing},
        "false_valid":false_valid,"false_valid_gate_passed":gate_passed,"flags":strings(value(mapping,&["flags"])),"errors":strings(value(mapping,&["errors"])),"rejection_reasons":rejections,"missing_inputs":missing,"provenance":provenance,"sample_rows":samples,"sample_truncated":rows.len()>samples.len()})
}

pub fn photometric_evidence_section(payload: &Value) -> Value {
    let mapping = payload.as_object();
    let relative = relative_photometry_summary(value(mapping, &["relative_photometry"]));
    let mut quality_raw = Value::Null;
    let mut quality_source = None;
    for alias in [
        "photometric_quality_report",
        "photometric_quality",
        "photometric_report",
        "photometry_report",
    ] {
        if let Some((_, input)) = lookup(mapping, &[alias]).filter(|(_, input)| !input.is_null()) {
            quality_raw = input.clone();
            quality_source = Some(alias.to_string());
            break;
        }
    }
    let container = value(mapping, &["photometry"]).as_object();
    if quality_source.is_none() {
        for alias in ["quality_report", "quality", "report"] {
            if let Some((_, input)) =
                lookup(container, &[alias]).filter(|(_, input)| !input.is_null())
            {
                quality_raw = input.clone();
                quality_source = Some(format!("photometry.{alias}"));
                break;
            }
        }
    }
    if quality_source.is_none() {
        if let Some((_, input)) =
            lookup(mapping, &["photometry"]).filter(|(_, input)| !input.is_null())
        {
            quality_raw = input.clone();
            quality_source = Some("photometry".into());
        }
    }
    if quality_source.is_none() {
        if let Some((_, input)) =
            lookup(mapping, &["source_photometry"]).filter(|(_, input)| !input.is_null())
        {
            quality_raw = json!({"rows":input});
            quality_source = Some("source_photometry".into());
        }
    }
    let quality = photometric_quality_summary(
        &quality_raw,
        quality_source
            .as_deref()
            .unwrap_or("photometric_quality_report"),
    );
    let calibration = photometric_calibration_summary(value(
        mapping,
        &["photometric_calibration", "calibration"],
    ));
    let mut sources = Vec::new();
    if relative["present"] == true {
        sources.push("relative_photometry".to_string());
    }
    if quality["present"] == true {
        sources.push(quality["source"].as_str().unwrap_or("").to_string());
    }
    if calibration["present"] == true {
        sources.push("photometric_calibration".to_string());
    }
    let rejected = quality["false_valid"] == true;
    let get_count = |field: &str| quality[field].as_u64().unwrap_or(0);
    let level_count = quality["level_counts"]
        .as_object()
        .into_iter()
        .flat_map(|mapping| mapping.values())
        .filter_map(Value::as_u64)
        .sum::<u64>();
    let mut calibrated_apparent = get_count("calibrated_apparent_magnitude_count");
    if calibrated_apparent == 0 && get_count("row_count") == 0 {
        calibrated_apparent = quality["level_counts"]["APPARENT_CALIBRATED"]
            .as_u64()
            .unwrap_or(0);
    }
    let diagnostic = quality
        .get("absolute_magnitude_diagnostic_count")
        .or_else(|| quality.get("absolute_magnitude_value_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let raw_strict = quality
        .get("strict_absolute_magnitude_count")
        .or_else(|| quality.get("absolute_magnitude_strict_count"))
        .or_else(|| quality.get("absolute_eligible_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let strict = if rejected { 0 } else { raw_strict };
    let strict_bands = if rejected {
        json!({})
    } else {
        quality
            .get("absolute_magnitude_strict_band_counts")
            .cloned()
            .unwrap_or_else(|| json!({}))
    };
    let reported = quality
        .get("reported_absolute_eligible_count")
        .or_else(|| quality.get("absolute_eligible_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let available = relative["usable"] == true
        || calibrated_apparent + strict > 0
        || matches!(
            calibration["status"].as_str(),
            Some("VALID" | "VALID_NO_HOLDOUT")
        );
    let section_status = if sources.is_empty() {
        "NOT_PRESENT"
    } else if rejected {
        "PRESENT_BUT_REJECTED"
    } else if available {
        "AVAILABLE"
    } else if quality["present"] == true
        && (get_count("row_count") > 0 || level_count > 0)
        && get_count("instrumental_magnitude_count") > 0
    {
        "INSTRUMENTAL_ONLY"
    } else {
        "PRESENT_BUT_INCOMPLETE"
    };
    let mut apparent = get_count("apparent_magnitude_count");
    if apparent == 0 {
        apparent = quality["level_counts"]["APPARENT_CALIBRATED"]
            .as_u64()
            .unwrap_or(0)
            + quality["level_counts"]["ABSOLUTE_ELIGIBLE"]
                .as_u64()
                .unwrap_or(0);
    }
    let absolute_gate = quality["absolute_magnitude_gate_status"]
        .as_str()
        .unwrap_or("NOT_AVAILABLE");
    let boundary_status = if absolute_gate == "NOT_AVAILABLE" {
        if get_count("absolute_magnitude_result_count") > 0 {
            "PRESENT_BUT_INCOMPLETE"
        } else {
            "NOT_INFERRED"
        }
    } else {
        absolute_gate
    };
    let reason_counts = quality
        .get("absolute_magnitude_strict_reason_counts")
        .cloned()
        .unwrap_or_else(|| json!({}));
    json!({"status":section_status,"evidence_sources":sources,"relative_photometry":relative,"photometric_quality":quality,"photometric_calibration":calibration,
        "strict_absolute_magnitude":{"status":if strict > 0 { "AVAILABLE" } else { "NOT_AVAILABLE" },"count":strict,"bands":strict_bands,"reason_counts":reason_counts,"note":"只有逐源绝对结果显式 is_strict=True 且通过状态、距离区间、距离来源和消光语义门时才计入。"},
        "summary":{"relative_scale_usable":relative["usable"].as_bool().unwrap_or(false),"relative_source_count":relative["relative_magnitude_count"].as_u64().unwrap_or(0),"relative_frame_count":relative["frame_zero_point_count"].as_u64().unwrap_or(0),"reported_apparent_magnitude_count":apparent,
            "reported_absolute_magnitude_value_count":diagnostic,"reported_absolute_magnitude_diagnostic_count":diagnostic,"reported_absolute_magnitude_diagnostic_only_count":get_count("absolute_magnitude_diagnostic_only_count"),"reported_strict_absolute_magnitude_count":strict,"reported_strict_absolute_magnitude_evidence_count":raw_strict,"reported_absolute_eligible_count":reported,"absolute_magnitude_status":absolute_gate},
        "boundary":{"instrumental_magnitude":{"status":"INPUT_MEASUREMENT_ONLY","note":"m_inst/仪器星等可作为观测量消费，但不含同设备绝对零点。"},
            "relative_magnitude":{"status":if relative["present"] == true { "CONSUMED_WITH_RELATIVE_LABEL" } else { "NOT_PRESENT" },"note":"relative_photometry 只提供跨帧相对尺度、帧零点和残差证据。"},
            "apparent_magnitude":{"status":if apparent > 0 { "REPORTED_BY_INPUT_ONLY" } else { "NOT_INFERRED" },"count":apparent,"note":"只有输入质量报告明确提供并通过其自身状态门时才保留为表观星等证据；本模块不重新标定。"},
            "absolute_magnitude":{"status":boundary_status,"value_count":diagnostic,"diagnostic_count":diagnostic,"diagnostic_only_count":get_count("absolute_magnitude_diagnostic_only_count"),"strict_count":strict,"eligible_count":strict,"reported_eligible_count":reported,"strict_reason_counts":reason_counts,
                "diagnostic_band_counts":quality.get("absolute_magnitude_diagnostic_band_counts").cloned().unwrap_or_else(||json!({})),"strict_band_counts":strict_bands,"note":"absolute_magnitude 是兼容旧输入的诊断值；正式绝对星等只使用 strict_absolute_magnitude。需要目录身份、距离/视差、完整距离区间、消光语义和质量门。"}},
        "notes":["星等 section 是创新报告对已有光度证据的审计摘要，不是新的星表匹配或重新测光结果。","运动目标没有被自动当作 Gaia 恒星；相对光度证据也不会改变运动/静态分类。","旧 payload 中的 numeric M、M_G 或 M_V 只作为带波段标识的诊断值保留；没有 is_strict=True 时不计入正式绝对星等。"]})
}
