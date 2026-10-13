//! JSON photometry observability and false-validity audit.
//!
//! This module mirrors the standalone report contract in
//! `rst19.photometric_report`: it consumes serialized JSON mappings, preserves
//! diagnostic values, and promotes a magnitude only when its evidence gates
//! pass. The report is descriptive evidence, not a stellar identity claim.
use crate::Result;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const LEVELS: [&str; 5] = [
    "INSTRUMENTAL_ONLY",
    "RELATIVE_CALIBRATED",
    "APPARENT_CALIBRATED",
    "ABSOLUTE_ELIGIBLE",
    "GEOMETRY_UNAVAILABLE",
];
const CALIBRATION_VALID: [&str; 7] = [
    "VALID",
    "VALID_NO_HOLDOUT",
    "CALIBRATED",
    "CALIBRATION_VALID",
    "RELATIVE_CALIBRATED",
    "APPARENT_CALIBRATED",
    "ABSOLUTE_ELIGIBLE",
];
const ABSOLUTE_VALID: [&str; 7] = [
    "VALID",
    "VALID_NO_HOLDOUT",
    "VALID_MODEL_DISTANCE",
    "VALID_MODEL_DISTANCE_NO_INTERVAL",
    "ABSOLUTE_VALID",
    "ABSOLUTE_ELIGIBLE",
    "CALIBRATED_ABSOLUTE",
];
const INVALID_STATUS: [&str; 16] = [
    "INVALID",
    "REJECTED",
    "REJECT",
    "MISSING",
    "NONE",
    "UNKNOWN",
    "UNAVAILABLE",
    "NO_CALIBRATION",
    "CALIBRATION_INVALID",
    "CALIBRATION_MISSING_COLOR",
    "CALIBRATION_NOT_APPLIED",
    "LOW_PARALLAX_SNR",
    "NO_PARALLAX",
    "NO_EXTINCTION",
    "GEOMETRY_UNAVAILABLE",
    "NO_WCS",
];
const ROW_COLLECTIONS: [&str; 9] = [
    "sources",
    "source_rows",
    "source_photometry",
    "source_results",
    "photometry",
    "photometry_rows",
    "photometry_results",
    "rows",
    "items",
];
const RESULT_COLLECTIONS: [&str; 7] = [
    "results",
    "rows",
    "items",
    "calibrations",
    "calibration_results",
    "absolute_results",
    "absolute_magnitudes",
];
const FRAME_COLLECTIONS: [&str; 3] = ["frames", "frame_rows", "frame_results"];
const KNOWN_ROW_KEYS: [&str; 37] = [
    "frame_id",
    "frame",
    "source_id",
    "detection_id",
    "id",
    "m_inst",
    "instrumental_magnitude",
    "m_cal",
    "calibrated_magnitude",
    "apparent_magnitude",
    "m",
    "M",
    "absolute_magnitude",
    "status",
    "observability_level",
    "level",
    "wcs",
    "wcs_available",
    "has_wcs",
    "geometry",
    "geometry_status",
    "catalog",
    "catalog_name",
    "parallax_mas",
    "extinction_mag",
    "calibration",
    "calibration_result",
    "absolute",
    "absolute_result",
    "error_budget",
    "provenance",
    "catalog_id",
    "photometric_system",
    "photometric_band",
    "distance_pc",
    "absolute_status",
    "quality_passed",
];
const PROVENANCE_KEYS: [(&str, &[&str]); 7] = [
    (
        "catalog",
        &["catalog", "catalog_name", "catalog_id", "catalog_version"],
    ),
    ("photometric_system", &["photometric_system", "system"]),
    ("photometric_band", &["photometric_band", "band"]),
    (
        "calibration",
        &[
            "calibration_id",
            "calibration_version",
            "calibration_source",
        ],
    ),
    ("wcs", &["wcs_source", "wcs_version", "wcs_id"]),
    (
        "absolute",
        &["absolute_method", "distance_source", "absolute_source"],
    ),
    (
        "input",
        &["source", "source_file", "input_file", "data_source"],
    ),
];
const ERROR_KEYS: [(&str, &[&str]); 12] = [
    (
        "m_inst",
        &[
            "m_inst_error",
            "instrumental_magnitude_error",
            "instrumental_error",
            "sigma_m_inst",
        ],
    ),
    (
        "m_cal",
        &[
            "m_cal_error",
            "calibrated_magnitude_error",
            "apparent_magnitude_error",
            "sigma_m_cal",
        ],
    ),
    (
        "M",
        &[
            "M_error",
            "absolute_magnitude_error",
            "sigma_M",
            "sigma_absolute_magnitude",
        ],
    ),
    (
        "zero_point",
        &["zero_point_error", "zp_error", "zero_point_sigma"],
    ),
    (
        "color_term",
        &["color_term_error", "color_coefficient_error", "color_sigma"],
    ),
    (
        "calibration_fit",
        &["fit_rms_mag", "fit_error_mag", "calibration_fit_error"],
    ),
    (
        "calibration_validation",
        &["validation_rms_mag", "validation_error_mag"],
    ),
    (
        "calibration_rms",
        &["rms", "residual_rms", "training_residual_rms"],
    ),
    (
        "calibration_mad",
        &["mad", "residual_mad", "training_residual_mad"],
    ),
    ("parallax", &["parallax_error_mas", "sigma_parallax_mas"]),
    (
        "extinction",
        &["extinction_error_mag", "sigma_extinction_mag"],
    ),
    (
        "wcs",
        &[
            "wcs_error_arcsec",
            "wcs_residual_arcsec",
            "rms_residual_px",
            "wcs_rms_px",
        ],
    ),
];

type Obj = Map<String, Value>;

fn norm_key(value: &str) -> String {
    value
        .trim()
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .to_lowercase()
}

fn lookup<'a>(mapping: Option<&'a Obj>, aliases: &[&str]) -> Option<&'a Value> {
    let mapping = mapping?;
    for alias in aliases {
        let key = norm_key(alias);
        if let Some((_, value)) = mapping.iter().rev().find(|(name, _)| norm_key(name) == key) {
            return Some(value);
        }
    }
    None
}

fn value<'a>(mapping: Option<&'a Obj>, aliases: &[&str]) -> Option<&'a Value> {
    lookup(mapping, aliases)
}

fn python_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        _ => value.to_string(),
    }
}

fn nonempty(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::String(text)) => !text.trim().is_empty(),
        Some(Value::Object(mapping)) => !mapping.is_empty(),
        _ => true,
    }
}

fn number(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Null | Value::Bool(_) | Value::Array(_) | Value::Object(_) => None,
        Value::Number(n) => n.as_f64().filter(|value| value.is_finite()),
        Value::String(text) => text
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite()),
    }
}

fn json_scalar(value: Option<&Value>) -> Value {
    match value {
        None | Some(Value::Null) => Value::Null,
        Some(Value::String(_) | Value::Bool(_) | Value::Number(_)) => value.unwrap().clone(),
        _ => Value::Null,
    }
}

fn present_number(mapping: Option<&Obj>, aliases: &[&str]) -> (bool, Option<f64>) {
    let raw = lookup(mapping, aliases);
    let present = raw.is_some_and(|item| !item.is_null() && item != "");
    (present, number(raw))
}

fn explicit_field_present(mapping: Option<&Obj>, aliases: &[&str]) -> bool {
    lookup(mapping, aliases)
        .is_some_and(|item| !item.is_null() && !item.as_str().is_some_and(|s| s.trim().is_empty()))
}

fn norm_status(value: Option<&Value>) -> String {
    value
        .filter(|item| !item.is_null())
        .map(python_string)
        .unwrap_or_default()
        .trim()
        .to_uppercase()
        .replace(['-', ' '], "_")
}

fn status(mapping: Option<&Obj>, aliases: &[&str]) -> String {
    let item = value(mapping, aliases);
    if nonempty(item) {
        norm_status(item)
    } else {
        String::new()
    }
}

fn bool_value(value: Option<&Value>) -> Option<bool> {
    match value? {
        Value::Bool(value) => Some(*value),
        Value::Number(value) if value.as_f64() == Some(1.0) => Some(true),
        Value::Number(value) if value.as_f64() == Some(0.0) => Some(false),
        Value::String(_) => match norm_status(value).as_str() {
            "TRUE" | "YES" | "Y" | "AVAILABLE" | "VALID" | "OK" | "PRESENT" => Some(true),
            "FALSE" | "NO" | "N" | "MISSING" | "UNAVAILABLE" | "INVALID" | "ABSENT" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn level(value: Option<&Value>) -> Option<String> {
    let normalized = norm_status(value);
    LEVELS.contains(&normalized.as_str()).then_some(normalized)
}

fn looks_like_row(mapping: &Obj) -> bool {
    mapping.keys().any(|key| {
        let normalized = norm_key(key);
        KNOWN_ROW_KEYS
            .iter()
            .any(|item| norm_key(item) == normalized)
            || ROW_COLLECTIONS
                .iter()
                .any(|item| norm_key(item) == normalized)
    })
}

fn row_list(value: Option<&Value>, id_aliases: &[&str]) -> (Vec<Obj>, usize) {
    let Some(value) = value else {
        return (Vec::new(), 0);
    };
    if let Some(mapping) = value.as_object() {
        for alias in id_aliases {
            if let Some(nested) = lookup(Some(mapping), &[*alias]).filter(|item| item.is_array()) {
                return row_list(Some(nested), id_aliases);
            }
        }
        if looks_like_row(mapping) {
            return (vec![mapping.clone()], 0);
        }
        for alias in ROW_COLLECTIONS.iter().chain(RESULT_COLLECTIONS.iter()) {
            if let Some(nested) =
                lookup(Some(mapping), &[*alias]).filter(|item| item.is_array() || item.is_object())
            {
                return row_list(Some(nested), id_aliases);
            }
        }
        let mut rows = Vec::new();
        let mut malformed = 0;
        for (key, raw) in mapping {
            let mut item = if let Some(object) = raw.as_object() {
                object.clone()
            } else if raw.is_array() && id_aliases.iter().any(|alias| norm_key(alias) == "frame_id")
            {
                let mut frame = Map::new();
                frame.insert("frame_id".into(), Value::String(key.clone()));
                frame.insert("sources".into(), raw.clone());
                frame
            } else {
                malformed += 1;
                continue;
            };
            if !id_aliases.is_empty() && lookup(Some(&item), id_aliases).is_none() {
                let mut with_id = Map::new();
                with_id.insert(id_aliases[0].to_string(), Value::String(key.clone()));
                with_id.extend(item);
                item = with_id;
            }
            rows.push(item);
        }
        return (rows, malformed);
    }
    if let Some(items) = value.as_array() {
        let mut rows = Vec::new();
        let mut malformed = 0;
        for item in items {
            if let Some(object) = item.as_object() {
                rows.push(object.clone());
            } else {
                malformed += 1;
            }
        }
        return (rows, malformed);
    }
    (Vec::new(), 1)
}

fn collection(mapping: &Obj) -> Option<&Value> {
    lookup(Some(mapping), &ROW_COLLECTIONS)
}

fn geometry_state(mapping: Option<&Obj>, default: &str) -> String {
    let Some(mapping) = mapping else {
        return default.to_owned();
    };
    if let Some(parsed) = bool_value(lookup(
        Some(mapping),
        &["wcs_available", "has_wcs", "geometry_available"],
    )) {
        return if parsed { "available" } else { "unavailable" }.into();
    }
    if let Some(item) = lookup(Some(mapping), &["geometry_status", "wcs_status"]) {
        let state = norm_status(Some(item));
        if ["VALID", "AVAILABLE", "SOLVED", "OK", "CALIBRATED", "READY"].contains(&state.as_str()) {
            return "available".into();
        }
        if !state.is_empty() {
            return "unavailable".into();
        }
    }
    if let Some(item) = lookup(
        Some(mapping),
        &["wcs", "wcs_solution", "world_coordinate_system"],
    ) {
        if let Some(nested) = item.as_object() {
            let nested_status = status(Some(nested), &["status"]);
            if ["VALID", "AVAILABLE", "SOLVED", "OK", "CALIBRATED", "READY"]
                .contains(&nested_status.as_str())
            {
                return "available".into();
            }
            if !nested_status.is_empty() {
                return "unavailable".into();
            }
            return if nested.is_empty() {
                "unavailable"
            } else {
                "available"
            }
            .into();
        }
        if let Some(parsed) = bool_value(Some(item)) {
            return if parsed { "available" } else { "unavailable" }.into();
        }
        return if nonempty(Some(item)) {
            "available"
        } else {
            "unavailable"
        }
        .into();
    }
    if let Some(item) = lookup(Some(mapping), &["geometry", "sky_geometry"]) {
        if let Some(nested) = item.as_object() {
            return if nested.is_empty() {
                "unavailable"
            } else {
                "available"
            }
            .into();
        }
        if let Some(parsed) = bool_value(Some(item)) {
            return if parsed { "available" } else { "unavailable" }.into();
        }
    }
    default.to_owned()
}

fn merge_context(frame: &Obj, source: &Obj, frame_id: &str, geometry: &str) -> Obj {
    const CONTEXT: &[&str] = &[
        "wcs",
        "wcs_available",
        "has_wcs",
        "geometry",
        "geometry_status",
        "catalog",
        "catalog_name",
        "catalog_id",
        "photometric_system",
        "photometric_band",
        "band",
        "provenance",
        "calibration",
        "calibration_result",
        "absolute",
        "absolute_result",
        "absolute_magnitude",
        "error_budget",
    ];
    let mut merged = Obj::new();
    for alias in CONTEXT {
        if let Some(item) = lookup(Some(frame), &[*alias]) {
            merged.insert((*alias).into(), item.clone());
        }
        if let Some(item) = lookup(Some(source), &[*alias]) {
            merged.insert((*alias).into(), item.clone());
        }
    }
    merged.extend(source.clone());
    if lookup(Some(&merged), &["frame_id", "frame"]).is_none() {
        merged.insert("frame_id".into(), Value::String(frame_id.into()));
    }
    let source_has_geometry = lookup(
        Some(source),
        &[
            "wcs",
            "wcs_available",
            "has_wcs",
            "geometry",
            "geometry_status",
        ],
    )
    .is_some();
    let source_geometry = if geometry == "unavailable" && !source_has_geometry {
        "unavailable".to_owned()
    } else {
        geometry_state(Some(source), geometry)
    };
    merged.insert(
        "__report_geometry_state".into(),
        Value::String(source_geometry),
    );
    merged.insert("__report_frame_context".into(), Value::Bool(true));
    merged
}

fn normalise_frame_input(input_value: Option<&Value>) -> (Vec<Obj>, Vec<Obj>, usize) {
    let mut input = input_value;
    if let Some(mapping) = input.and_then(Value::as_object) {
        for alias in FRAME_COLLECTIONS {
            if let Some(nested) = lookup(Some(mapping), &[alias]) {
                input = Some(nested);
                break;
            }
        }
    }
    let (rows, mut malformed) = row_list(input, &["frame_id", "frame", "id"]);
    if rows.is_empty() {
        return (Vec::new(), Vec::new(), malformed);
    }
    let has_nested = rows.iter().any(|row| collection(row).is_some());
    if !has_nested && rows.iter().all(looks_like_row) {
        let direct = rows
            .into_iter()
            .map(|mut row| {
                let geometry = geometry_state(Some(&row), "unknown");
                row.insert("__report_geometry_state".into(), Value::String(geometry));
                row.insert("__report_frame_context".into(), Value::Bool(false));
                row
            })
            .collect();
        return (Vec::new(), direct, malformed);
    }
    let mut frames = Vec::new();
    let mut sources = Vec::new();
    for (index, frame) in rows.into_iter().enumerate() {
        let raw_id = value(Some(&frame), &["frame_id", "frame", "id"])
            .cloned()
            .unwrap_or(json!(index));
        let frame_id = if nonempty(Some(&raw_id)) {
            python_string(&raw_id)
        } else {
            index.to_string()
        };
        let raw_sources = collection(&frame);
        let frame_geometry = geometry_state(
            Some(&frame),
            if raw_sources.is_some() {
                "unavailable"
            } else {
                "unknown"
            },
        );
        let mut frame_record = frame.clone();
        frame_record.insert("frame_id".into(), Value::String(frame_id.clone()));
        frame_record.insert(
            "__report_geometry_state".into(),
            Value::String(frame_geometry.clone()),
        );
        frames.push(frame_record);
        let Some(raw_sources) = raw_sources else {
            if looks_like_row(&frame) {
                let mut source = frame;
                source.insert(
                    "__report_geometry_state".into(),
                    Value::String(frame_geometry),
                );
                source.insert("__report_frame_context".into(), Value::Bool(true));
                sources.push(source);
            }
            continue;
        };
        let (nested, bad) = row_list(Some(raw_sources), &["source_id", "detection_id", "id"]);
        malformed += bad;
        sources.extend(
            nested
                .iter()
                .map(|source| merge_context(&frame, source, &frame_id, &frame_geometry)),
        );
    }
    (frames, sources, malformed)
}

fn normalise_source_input(value: Option<&Value>) -> (Vec<Obj>, usize) {
    let (rows, malformed) = row_list(value, &["source_id", "detection_id", "id"]);
    let rows = rows
        .into_iter()
        .map(|mut row| {
            let geometry = geometry_state(Some(&row), "unknown");
            row.insert("__report_geometry_state".into(), Value::String(geometry));
            row.insert("__report_frame_context".into(), Value::Bool(false));
            row
        })
        .collect();
    (rows, malformed)
}

#[derive(Default)]
struct NormalisedResults {
    globals: Vec<Obj>,
    indexed: BTreeMap<(String, String), Obj>,
    malformed: usize,
}

fn normalise_results(input: Option<&Value>) -> NormalisedResults {
    let Some(input) = input else {
        return NormalisedResults::default();
    };
    if let Some(mapping) = input.as_object().filter(|mapping| !looks_like_row(mapping)) {
        for alias in RESULT_COLLECTIONS {
            if let Some(nested) = lookup(Some(mapping), &[alias]) {
                let mut result = normalise_results(Some(nested));
                let mut metadata = mapping.clone();
                for name in RESULT_COLLECTIONS {
                    metadata.remove(name);
                }
                if !metadata.is_empty() && looks_like_row(&metadata) {
                    result.globals.insert(0, metadata);
                }
                return result;
            }
        }
    }
    let (rows, malformed) = row_list(
        Some(input),
        &["source_id", "detection_id", "frame_id", "id"],
    );
    let mut result = NormalisedResults {
        malformed,
        ..NormalisedResults::default()
    };
    for row in rows {
        let source_id =
            value(Some(&row), &["source_id", "source", "catalog_id"]).filter(|v| !v.is_null());
        let detection_id = value(Some(&row), &["detection_id", "id"]).filter(|v| !v.is_null());
        let frame_id = value(Some(&row), &["frame_id", "frame"]).filter(|v| !v.is_null());
        if source_id.is_none() && detection_id.is_none() && frame_id.is_none() {
            result.globals.push(row);
            continue;
        }
        if let Some(id) = source_id {
            result
                .indexed
                .insert(("source_id".into(), python_string(id)), row.clone());
        }
        if let Some(id) = detection_id {
            result
                .indexed
                .insert(("detection_id".into(), python_string(id)), row.clone());
        }
        if let Some(id) = frame_id {
            result
                .indexed
                .insert(("frame_id".into(), python_string(id)), row);
        }
    }
    result
}

fn result_for(row: &Obj, results: &NormalisedResults, nested_aliases: &[&str]) -> Obj {
    let mut merged = Obj::new();
    for global in &results.globals {
        merged.extend(global.clone());
    }
    for (aliases, index_key) in [
        (&["frame_id", "frame"][..], "frame_id"),
        (&["source_id", "source", "catalog_id"][..], "source_id"),
        (&["detection_id", "id"][..], "detection_id"),
    ] {
        if let Some(id) = value(Some(row), aliases).filter(|v| !v.is_null()) {
            if let Some(specific) = results
                .indexed
                .get(&(index_key.to_owned(), python_string(id)))
            {
                merged.extend(specific.clone());
            }
        }
    }
    for alias in nested_aliases {
        if let Some(nested) = value(Some(row), &[*alias]).and_then(Value::as_object) {
            merged.extend(nested.clone());
        }
    }
    merged
}

fn unique_mappings<'a>(maps: impl IntoIterator<Item = &'a Obj>) -> Vec<&'a Obj> {
    let mut seen = BTreeSet::new();
    maps.into_iter()
        .filter(|map| seen.insert(*map as *const Obj as usize))
        .collect()
}

fn catalog_present(row: &Obj) -> bool {
    if let Some(parsed) = bool_value(lookup(
        Some(row),
        &["has_catalog", "catalog_available", "catalog_match"],
    )) {
        return parsed;
    }
    if let Some(catalog) = lookup(
        Some(row),
        &[
            "catalog",
            "catalog_source",
            "catalog_name",
            "catalog_version",
        ],
    ) {
        if let Some(mapping) = catalog.as_object() {
            let state = status(Some(mapping), &["status"]);
            if !state.is_empty() && contains_invalid_status(&state) {
                return false;
            }
            return !mapping.is_empty();
        }
    }
    [
        &["catalog_id", "matched_source_id"][..],
        &["catalog", "catalog_name", "catalog_version"][..],
        &["catalog_magnitude", "reference_magnitude", "catalog_mag"][..],
        &["match", "catalog_match_result"][..],
    ]
    .iter()
    .any(|aliases| lookup(Some(row), aliases).is_some_and(|item| nonempty(Some(item))))
}

fn calibration_mode(row: &Obj, calibration: &Obj) -> &'static str {
    let raw = value(
        Some(row),
        &[
            "calibration_scope",
            "calibration_type",
            "calibration_mode",
            "mode",
            "photometric_scope",
        ],
    )
    .or_else(|| {
        value(
            Some(calibration),
            &[
                "calibration_scope",
                "calibration_type",
                "calibration_mode",
                "mode",
            ],
        )
    });
    let normalized = norm_status(raw);
    if [
        "RELATIVE",
        "RELATIVE_CALIBRATION",
        "RELATIVE_CALIBRATED",
        "INSTRUMENTAL_RELATIVE",
    ]
    .contains(&normalized.as_str())
    {
        return "relative";
    }
    if [
        "APPARENT",
        "APPARENT_CALIBRATION",
        "APPARENT_CALIBRATED",
        "STANDARD",
        "ABSOLUTE_FLUX",
    ]
    .contains(&normalized.as_str())
    {
        return "apparent";
    }
    for mapping in [row, calibration] {
        if bool_value(lookup(Some(mapping), &["relative", "is_relative"])) == Some(true) {
            return "relative";
        }
        if bool_value(lookup(Some(mapping), &["apparent", "is_apparent"])) == Some(true) {
            return "apparent";
        }
    }
    if catalog_present(row)
        || catalog_present(calibration)
        || value(
            Some(calibration),
            &["photometric_system", "photometric_band", "band"],
        )
        .is_some_and(|v| nonempty(Some(v)))
    {
        "apparent"
    } else {
        "relative"
    }
}

fn contains_invalid_status(status: &str) -> bool {
    if status.is_empty() {
        return false;
    }
    if INVALID_STATUS.contains(&status) {
        return true;
    }
    if CALIBRATION_VALID.contains(&status) || ABSOLUTE_VALID.contains(&status) {
        return false;
    }
    status
        .split('_')
        .any(|token| ["INVALID", "REJECTED", "MISSING", "UNAVAILABLE", "NO"].contains(&token))
}

fn calibration_status_valid(
    statuses: &[String],
    explicit_level: Option<&str>,
    calibration: &Obj,
) -> bool {
    if explicit_level.is_some_and(|level| {
        [
            "RELATIVE_CALIBRATED",
            "APPARENT_CALIBRATED",
            "ABSOLUTE_ELIGIBLE",
        ]
        .contains(&level)
    }) && statuses.iter().all(String::is_empty)
    {
        return true;
    }
    let nonempty: Vec<_> = statuses
        .iter()
        .filter(|status| !status.is_empty())
        .collect();
    if nonempty.is_empty() {
        return false;
    }
    let mut has_valid = false;
    for status in nonempty {
        if contains_invalid_status(status) {
            return false;
        }
        if CALIBRATION_VALID.contains(&status.as_str()) {
            has_valid = true;
            continue;
        }
        if status.contains("CALIBRAT") && !status.contains("NOT") {
            has_valid = true;
            continue;
        }
        return false;
    }
    if has_valid {
        return true;
    }
    lookup(Some(calibration), &["valid", "is_valid"])
        .is_some_and(|value| bool_value(Some(value)) == Some(true))
}

fn absolute_status_valid(statuses: &[String]) -> bool {
    let nonempty: Vec<_> = statuses
        .iter()
        .filter(|status| !status.is_empty())
        .collect();
    if nonempty.is_empty() {
        return false;
    }
    let lower = [
        "INSTRUMENTAL_ONLY",
        "RELATIVE_CALIBRATED",
        "APPARENT_CALIBRATED",
        "CALIBRATED",
    ];
    let mut has_valid = false;
    for status in nonempty {
        if ABSOLUTE_VALID.contains(&status.as_str()) {
            has_valid = true;
            continue;
        }
        if lower.contains(&status.as_str()) {
            continue;
        }
        if contains_invalid_status(status) {
            return false;
        }
        if status == "VALID" || (status.contains("ABSOLUTE") && status.contains("ELIGIB")) {
            has_valid = true;
            continue;
        }
        return false;
    }
    has_valid
}

fn distance_source_kind(value: Option<&Value>) -> Option<&'static str> {
    let normalized = norm_status(value);
    if normalized.is_empty() {
        return None;
    }
    if (normalized.contains("GSP") && normalized.contains("PHOT"))
        || (normalized.contains("MODEL") && normalized.contains("DISTANCE"))
    {
        return Some("model");
    }
    if normalized.contains("PARALLAX") {
        return Some("parallax");
    }
    Some("unknown")
}

fn distance_evidence(
    row: &Obj,
    absolute: &Obj,
    statuses: &[String],
    parallax: Option<f64>,
    distance: Option<f64>,
) -> (Option<&'static str>, bool, Option<&'static str>) {
    let positive_parallax = parallax.is_some_and(|value| value > 0.0);
    let positive_distance = distance.is_some_and(|value| value > 0.0);
    let sources: Vec<_> = [row, absolute]
        .iter()
        .filter_map(|mapping| {
            lookup(
                Some(mapping),
                &["distance_source", "distance_method", "distance_origin"],
            )
            .filter(|value| nonempty(Some(value)))
        })
        .collect();
    let kinds: Vec<_> = sources
        .iter()
        .map(|value| distance_source_kind(Some(value)))
        .collect();
    let path = if !kinds.is_empty() {
        if kinds
            .iter()
            .any(|kind| !matches!(kind, Some("parallax" | "model")))
        {
            return (None, false, Some("DISTANCE_SOURCE_UNRECOGNIZED"));
        }
        if kinds.iter().any(|kind| *kind != kinds[0]) {
            return (None, false, Some("DISTANCE_SOURCE_CONFLICT"));
        }
        kinds[0]
    } else if positive_parallax {
        Some("parallax")
    } else if positive_distance {
        return (None, false, Some("DISTANCE_SOURCE_REQUIRED"));
    } else {
        return (None, false, Some("DISTANCE_MISSING"));
    };
    let has_model_status = statuses.iter().any(|status| {
        ["VALID_MODEL_DISTANCE", "VALID_MODEL_DISTANCE_NO_INTERVAL"].contains(&status.as_str())
    });
    if path == Some("parallax") {
        if has_model_status {
            return (path, false, Some("DISTANCE_SOURCE_STATUS_MISMATCH"));
        }
        if !positive_parallax && !positive_distance {
            return (path, false, Some("PARALLAX_VALUE_MISSING"));
        }
        return (path, true, None);
    }
    if !positive_distance {
        return (path, false, Some("MODEL_DISTANCE_MISSING"));
    }
    if !has_model_status {
        return (path, false, Some("DISTANCE_SOURCE_STATUS_MISMATCH"));
    }
    if statuses
        .iter()
        .any(|status| status == "VALID_MODEL_DISTANCE_NO_INTERVAL")
    {
        return (path, false, Some("MODEL_DISTANCE_INTERVAL_MISSING"));
    }
    let (_, lower) = present_number(
        Some(row),
        &[
            "distance_lower_pc",
            "distance_gspphot_lower",
            "distance_lower",
        ],
    );
    let (_, lower) = if lower.is_some() {
        (true, lower)
    } else {
        present_number(
            Some(absolute),
            &[
                "distance_lower_pc",
                "distance_gspphot_lower",
                "distance_lower",
            ],
        )
    };
    let (_, upper) = present_number(
        Some(row),
        &[
            "distance_upper_pc",
            "distance_gspphot_upper",
            "distance_upper",
        ],
    );
    let (_, upper) = if upper.is_some() {
        (true, upper)
    } else {
        present_number(
            Some(absolute),
            &[
                "distance_upper_pc",
                "distance_gspphot_upper",
                "distance_upper",
            ],
        )
    };
    let valid = lower
        .zip(upper)
        .zip(distance)
        .is_some_and(|((lower, upper), distance)| {
            lower > 0.0 && upper > 0.0 && lower <= upper && lower <= distance && distance <= upper
        });
    if valid {
        (path, true, None)
    } else {
        (path, false, Some("MODEL_DISTANCE_INTERVAL_MISSING"))
    }
}

fn canonical_band(value: Option<&Value>) -> String {
    let text = value
        .map(python_string)
        .unwrap_or_default()
        .trim()
        .to_owned();
    let key: String = text
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    match key.as_str() {
        "g" | "gaiag" | "gaiadr3g" => "G".into(),
        "v" | "johnsonv" | "johnsoncousinsv" => "V".into(),
        "a0" | "azero" | "a05414nm" | "5414nm" => "A0(541.4 nm)".into(),
        "" | "unknown" | "unk" | "na" | "none" => "unknown".into(),
        _ => text,
    }
}

fn canonical_system(value: Option<&Value>) -> String {
    let text = value
        .map(python_string)
        .unwrap_or_default()
        .trim()
        .to_owned();
    let key: String = text
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    match key.as_str() {
        "gaia" | "gaiavega" | "gaiadr3" | "gaiadr3vega" => "Gaia".into(),
        "johnson" | "johnsonv" | "johnsoncousins" | "johnsoncousinsv" => "Johnson".into(),
        "monochromatic" | "monochromatic5414nm" | "a0" => "monochromatic".into(),
        "" | "unknown" | "unk" | "na" | "none" => "unknown".into(),
        _ => text,
    }
}

fn first_nonempty<'a>(mappings: &[&'a Obj], aliases: &[&str]) -> Option<&'a Value> {
    mappings
        .iter()
        .find_map(|mapping| lookup(Some(mapping), aliases).filter(|value| nonempty(Some(value))))
}

fn extinction_evidence(row: &Obj, absolute: &Obj) -> (bool, Option<&'static str>, Option<f64>) {
    let raw = lookup(Some(row), &["extinction_mag", "extinction", "A_V", "av"]).or_else(|| {
        lookup(
            Some(absolute),
            &["extinction_mag", "extinction", "A_V", "av"],
        )
    });
    let Some(raw) = raw else {
        return (false, Some("EXTINCTION_MISSING"), None);
    };
    let extinction = number(Some(raw));
    if extinction.is_none_or(|value| value < 0.0) {
        return (false, Some("EXTINCTION_INVALID"), extinction);
    }
    let mappings = [row, absolute];
    let band = canonical_band(first_nonempty(
        &mappings,
        &["extinction_band", "ext_band", "extinction_passband"],
    ));
    let system = canonical_system(first_nonempty(
        &mappings,
        &[
            "extinction_system",
            "extinction_photometric_system",
            "ext_system",
        ],
    ));
    let source = first_nonempty(
        &mappings,
        &["extinction_source", "extinction_provenance", "ext_source"],
    );
    if band == "unknown" || system == "unknown" {
        return (false, Some("EXTINCTION_SEMANTICS_REQUIRED"), extinction);
    }
    if source.is_none_or(|value| {
        ["unknown", "unk", "na", "none"].contains(&norm_key(&python_string(value)).as_str())
    }) {
        return (false, Some("EXTINCTION_SOURCE_REQUIRED"), extinction);
    }
    let expected_band = canonical_band(first_nonempty(
        &mappings,
        &["photometric_band", "band", "passband"],
    ));
    let expected_system =
        canonical_system(first_nonempty(&mappings, &["photometric_system", "system"]));
    if (expected_band != "unknown" && expected_band != band)
        || (expected_system != "unknown" && expected_system != system)
    {
        return (false, Some("EXTINCTION_BAND_MISMATCH"), extinction);
    }
    (true, None, extinction)
}

fn error_values(mapping: Option<&Obj>) -> BTreeMap<String, f64> {
    let Some(mapping) = mapping else {
        return BTreeMap::new();
    };
    let mut result = BTreeMap::new();
    for (component, aliases) in ERROR_KEYS {
        if let Some(value) = lookup(Some(mapping), aliases)
            .and_then(|raw| number(Some(raw)))
            .filter(|value| *value >= 0.0)
        {
            result.insert(component.into(), value);
        }
    }
    if let Some(nested) =
        value(Some(mapping), &["error_budget", "errors", "uncertainty"]).and_then(Value::as_object)
    {
        for (key, raw) in nested {
            let raw = if let Some(inner) = raw.as_object() {
                value(Some(inner), &["value", "sigma", "error", "std", "rms"])
            } else {
                Some(raw)
            };
            if let Some(number) = raw
                .and_then(|raw| number(Some(raw)))
                .filter(|number| *number >= 0.0)
            {
                result.entry(norm_key(key)).or_insert(number);
            }
        }
    }
    result
}

fn add_provenance(target: &mut BTreeMap<String, BTreeSet<String>>, mapping: Option<&Obj>) {
    let Some(mapping) = mapping else {
        return;
    };
    let mut mappings = vec![mapping];
    for alias in [
        "provenance",
        "wcs",
        "calibration",
        "calibration_result",
        "absolute",
        "absolute_result",
        "absolute_magnitude",
    ] {
        if let Some(nested) = value(Some(mapping), &[alias]).and_then(Value::as_object) {
            mappings.push(nested);
        }
    }
    for current in mappings {
        for (category, aliases) in PROVENANCE_KEYS {
            let Some(raw) = lookup(Some(current), aliases) else {
                continue;
            };
            if let Some(items) = raw.as_array() {
                for item in items.iter().filter(|item| nonempty(Some(item))) {
                    target
                        .entry(category.into())
                        .or_default()
                        .insert(python_string(item));
                }
            } else if nonempty(Some(raw)) {
                target
                    .entry(category.into())
                    .or_default()
                    .insert(python_string(raw));
            }
        }
    }
}

fn row_key(row: &Obj, index: usize) -> (String, Option<String>, Option<String>) {
    let frame = value(Some(row), &["frame_id", "frame"])
        .filter(|value| nonempty(Some(value)))
        .map(python_string);
    let source = value(Some(row), &["source_id", "detection_id", "id"])
        .filter(|value| nonempty(Some(value)))
        .map(python_string);
    let key = format!(
        "{}:{}",
        frame.as_deref().unwrap_or("frame?"),
        source
            .as_deref()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("row-{index}"))
    );
    (key, frame, source)
}

fn absolute_value_details(
    row: &Obj,
    absolute: &Obj,
) -> (
    bool,
    Option<f64>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    let nested = value(
        Some(row),
        &["absolute_magnitude", "absolute", "absolute_result"],
    )
    .and_then(Value::as_object);
    let mappings: Vec<&Obj> = nested.into_iter().chain([absolute, row]).collect();
    const ALIASES: &[&str] = &[
        "strict_absolute_magnitude",
        "strict_M",
        "M_G",
        "M_V",
        "value",
        "M",
        "absolute_magnitude_value",
        "absolute_mag",
        "m_abs",
    ];
    let mut present = false;
    let mut result = None;
    let mut value_alias = None;
    'outer: for mapping in &mappings {
        for alias in ALIASES {
            let Some(mut raw) = lookup(Some(mapping), &[*alias]) else {
                continue;
            };
            let mut candidate_alias = *alias;
            if let Some(inner) = raw.as_object() {
                let mut found = false;
                for nested_alias in ALIASES {
                    if let Some(item) = lookup(Some(inner), &[*nested_alias]) {
                        raw = item;
                        candidate_alias = nested_alias;
                        found = true;
                        break;
                    }
                }
                if !found {
                    continue;
                }
            }
            if raw.is_null() || raw.as_str().is_some_and(|value| value.trim().is_empty()) {
                continue;
            }
            present = true;
            value_alias = Some(candidate_alias.to_string());
            result = number(Some(raw));
            if result.is_some() {
                break 'outer;
            }
        }
    }
    let band = first_nonempty(
        &mappings,
        &[
            "absolute_magnitude_band",
            "absolute_band",
            "M_band",
            "absolute_passband",
            "photometric_band",
            "band",
        ],
    );
    let system = first_nonempty(
        &mappings,
        &[
            "absolute_magnitude_system",
            "absolute_system",
            "photometric_system",
            "system",
        ],
    );
    let source = first_nonempty(
        &mappings,
        &[
            "absolute_magnitude_source",
            "absolute_source",
            "absolute_method",
            "distance_source",
            "photometric_source",
            "catalog_name",
        ],
    );
    let band = band
        .map(python_string)
        .or_else(|| match value_alias.as_deref() {
            Some("M_G") => Some("G".into()),
            Some("M_V") => Some("V".into()),
            _ => None,
        });
    let canonicalize = |text: Option<String>, system: bool| -> Option<String> {
        let Some(text) = text else {
            return None;
        };
        let key: String = text
            .chars()
            .filter(|ch| ch.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();
        let canonical = if system {
            match key.as_str() {
                "gaia" | "gaiavega" | "gaiadr3" | "gaiadr3vega" => "Gaia",
                "johnson" | "johnsonv" | "johnsoncousins" | "johnsoncousinsv" => "Johnson",
                "monochromatic" | "monochromatic5414nm" | "a0" => "monochromatic",
                "" | "unknown" | "unk" | "na" | "none" => return None,
                _ => return Some(text.trim().to_owned()),
            }
        } else {
            match key.as_str() {
                "g" | "gaiag" | "gaiadr3g" => "G",
                "v" | "johnsonv" | "johnsoncousinsv" => "V",
                "a0" | "azero" | "a05414nm" | "5414nm" => "A0(541.4 nm)",
                "" | "unknown" | "unk" | "na" | "none" => return None,
                _ => return Some(text.trim().to_owned()),
            }
        };
        Some(canonical.into())
    };
    let band = canonicalize(band, false);
    let system = canonicalize(system.map(python_string), true);
    (
        present,
        result,
        band,
        system,
        source
            .map(python_string)
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_owned()),
        value_alias,
    )
}

fn absolute_support_details(row: &Obj, absolute: &Obj) -> Value {
    let nested = value(
        Some(row),
        &["absolute_magnitude", "absolute", "absolute_result"],
    )
    .and_then(Value::as_object);
    let mappings: Vec<&Obj> = nested.into_iter().chain([absolute, row]).collect();
    let first_number = |aliases: &[&str]| -> Value {
        mappings
            .iter()
            .find_map(|mapping| {
                lookup(Some(mapping), aliases)
                    .and_then(|value| number(Some(value)))
                    .map(|number| json!(number))
            })
            .unwrap_or(Value::Null)
    };
    let first_text = |aliases: &[&str]| -> Option<String> {
        mappings.iter().find_map(|mapping| {
            lookup(Some(mapping), aliases)
                .filter(|value| nonempty(Some(value)))
                .map(|value| python_string(value).trim().to_owned())
        })
    };
    let mut interval_status = first_text(&[
        "absolute_magnitude_distance_interval_status",
        "distance_interval_status",
        "distance_interval_state",
    ]);
    if interval_status.is_none() {
        let mut found = None;
        for mapping in &mappings {
            let Some(flags) = value(Some(mapping), &["flags", "flag"]) else {
                continue;
            };
            let items: Vec<String> = if let Some(text) = flags.as_str() {
                vec![norm_status(Some(&Value::String(text.into())))]
            } else {
                flags
                    .as_array()
                    .map(|values| {
                        values
                            .iter()
                            .filter(|v| nonempty(Some(v)))
                            .map(|v| norm_status(Some(v)))
                            .collect()
                    })
                    .unwrap_or_default()
            };
            for flag in items {
                let mapped = match flag.as_str() {
                    "DISTANCE_INTERVAL_INVALID" => Some("INVALID"),
                    "DISTANCE_INTERVAL_DERIVED_FROM_PARALLAX_ERROR" => {
                        Some("DERIVED_FROM_PARALLAX_ERROR")
                    }
                    "DISTANCE_INTERVAL_USED" => Some("PROVIDED"),
                    "DISTANCE_INTERVAL_ONE_SIDED" => Some("ONE_SIDED"),
                    "DISTANCE_INTERVAL_REQUIRED" => Some("REQUIRED"),
                    "DISTANCE_ERROR_NOT_PROVIDED" => Some("NOT_PROVIDED"),
                    _ => None,
                };
                if mapped.is_some() {
                    found = mapped.map(str::to_owned);
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        interval_status = found;
    }
    json!({
        "distance_pc": first_number(&["absolute_magnitude_distance_pc", "distance_pc", "distance"]),
        "distance_lower_pc": first_number(&["absolute_magnitude_distance_lower_pc", "distance_lower_pc", "distance_gspphot_lower", "distance_lower"]),
        "distance_upper_pc": first_number(&["absolute_magnitude_distance_upper_pc", "distance_upper_pc", "distance_gspphot_upper", "distance_upper"]),
        "distance_interval_status": interval_status.map(|value| value.trim().to_uppercase().replace(['-', ' '], "_")).map(Value::String).unwrap_or(Value::Null),
        "extinction_mag": first_number(&["absolute_magnitude_extinction_mag", "extinction_mag", "extinction", "A_V", "av"]),
        "extinction_band": first_text(&["absolute_magnitude_extinction_band", "extinction_band", "ext_band", "extinction_passband"]),
        "extinction_system": first_text(&["absolute_magnitude_extinction_system", "extinction_system", "extinction_photometric_system", "ext_system"]),
        "extinction_source": first_text(&["absolute_magnitude_extinction_source", "extinction_source", "extinction_provenance", "ext_source"]),
    })
}

#[derive(Default)]
struct AuditedRow {
    value: Value,
    violations: Vec<Value>,
    geometry: String,
    rejection_reasons: Vec<String>,
    missing_inputs: Vec<String>,
    error_budget: BTreeMap<String, f64>,
}

fn violation(row_key: &str, field: &str, code: &str, message: String) -> Value {
    json!({"row_key":row_key,"field":field,"code":code,"message":message,"severity":"error"})
}

fn strings_value(values: impl IntoIterator<Item = String>) -> Value {
    Value::Array(values.into_iter().map(Value::String).collect())
}

fn audit_row(
    row: &Obj,
    index: usize,
    calibration: &Obj,
    absolute: &Obj,
    require_wcs: bool,
) -> AuditedRow {
    let (key, frame_id, source_id) = row_key(row, index);
    let explicit_level = level(value(Some(row), &["observability_level", "level"]));
    let row_status = {
        let status = status(Some(row), &["status", "state"]);
        (!status.is_empty()).then_some(status)
    };

    let mut absolute_map = absolute.clone();
    if let Some(nested) = value(
        Some(row),
        &["absolute_magnitude", "absolute", "absolute_result"],
    )
    .and_then(Value::as_object)
    {
        absolute_map.extend(nested.clone());
    }
    let mut calibration_map = calibration.clone();
    if let Some(nested) =
        value(Some(row), &["calibration", "calibration_result"]).and_then(Value::as_object)
    {
        calibration_map.extend(nested.clone());
    }

    let cal_status = status(Some(row), &["calibration_status", "m_cal_status"]);
    let cal_status = if !cal_status.is_empty() {
        Some(cal_status)
    } else {
        let nested = status(
            Some(&calibration_map),
            &["status", "state", "calibration_status"],
        );
        if !nested.is_empty() {
            Some(nested)
        } else {
            row_status.clone()
        }
    };
    let abs_status = status(
        Some(row),
        &["absolute_status", "M_status", "absolute_magnitude_status"],
    );
    let abs_status = if !abs_status.is_empty() {
        Some(abs_status)
    } else {
        let nested = status(Some(&absolute_map), &["status", "state", "absolute_status"]);
        if !nested.is_empty() {
            Some(nested)
        } else {
            row_status.clone()
        }
    };

    let (mut m_inst_present, m_inst) = present_number(
        Some(row),
        &["m_inst", "instrumental_magnitude", "instrumental_mag"],
    );
    let (mut m_cal_present, mut m_cal) = present_number(
        Some(row),
        &[
            "m_cal",
            "calibrated_magnitude",
            "apparent_magnitude",
            "m_std",
            "standard_magnitude",
        ],
    );
    if !m_cal_present {
        (m_cal_present, m_cal) = present_number(
            Some(calibration),
            &["m_cal", "calibrated_magnitude", "apparent_magnitude"],
        );
    }
    let (
        mut absolute_present,
        absolute_magnitude,
        absolute_band,
        absolute_system,
        absolute_source,
        _,
    ) = absolute_value_details(row, &absolute_map);
    let absolute_support = absolute_support_details(row, &absolute_map);
    if !m_inst_present && m_inst.is_none() {
        m_inst_present = explicit_field_present(Some(row), &["m_inst", "instrumental_magnitude"]);
    }
    if !m_cal_present {
        m_cal_present = explicit_field_present(
            Some(row),
            &[
                "m_cal",
                "calibrated_magnitude",
                "apparent_magnitude",
                "m_std",
                "standard_magnitude",
            ],
        );
    }
    if !absolute_present {
        absolute_present = explicit_field_present(Some(row), &["M", "absolute_magnitude_value"]);
    }

    let mut geometry = row
        .get("__report_geometry_state")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    if !["available", "unavailable", "unknown"].contains(&geometry.as_str()) {
        geometry = "unknown".into();
    }
    let explicit_geometry = geometry_state(Some(row), "unknown");
    if explicit_geometry != "unknown" {
        geometry = explicit_geometry;
    }
    let wcs_available = match geometry.as_str() {
        "available" => Value::Bool(true),
        "unavailable" => Value::Bool(false),
        _ => Value::Null,
    };
    let catalog_available =
        catalog_present(row) || catalog_present(&calibration_map) || catalog_present(&absolute_map);

    let calibration_statuses: Vec<String> = [cal_status.clone(), row_status.clone(), {
        let current = status(Some(&calibration_map), &["status"]);
        (!current.is_empty()).then_some(current)
    }]
    .into_iter()
    .flatten()
    .collect();
    let absolute_statuses: Vec<String> = [abs_status.clone(), row_status.clone(), {
        let current = status(Some(&absolute_map), &["status"]);
        (!current.is_empty()).then_some(current)
    }]
    .into_iter()
    .flatten()
    .collect();
    let calibration_valid = m_cal.is_some()
        && calibration_status_valid(
            &calibration_statuses,
            explicit_level.as_deref(),
            &calibration_map,
        );
    let diagnostic_calibration =
        m_cal.is_some() && row_status.as_deref() == Some("CATALOG_INCONSISTENT");
    let calibrated_role = if m_cal.is_some() && calibration_valid {
        "VALID"
    } else if diagnostic_calibration {
        "DIAGNOSTIC_ONLY"
    } else if m_cal.is_some() {
        "INVALID"
    } else {
        "ABSENT"
    };
    let absolute_status_is_valid =
        absolute_magnitude.is_some() && absolute_status_valid(&absolute_statuses);
    let strict_raw = lookup(
        Some(&absolute_map),
        &["absolute_magnitude_is_strict", "is_strict", "strict"],
    );
    let strict_flag_found = strict_raw.is_some();
    let strict_flag = strict_raw.and_then(|value| bool_value(Some(value)));

    let (_, mut parallax) = present_number(
        Some(row),
        &["parallax_mas", "parallax", "corrected_parallax_mas"],
    );
    if parallax.is_none() {
        parallax = present_number(
            Some(&absolute_map),
            &["parallax_mas", "parallax", "corrected_parallax_mas"],
        )
        .1;
    }
    let (_, mut distance) = present_number(Some(row), &["distance_pc", "distance"]);
    if distance.is_none() {
        distance = present_number(Some(&absolute_map), &["distance_pc", "distance"]).1;
    }
    let (distance_path, distance_qualified, distance_issue) =
        distance_evidence(row, &absolute_map, &absolute_statuses, parallax, distance);
    let (extinction_ok, extinction_issue, _) = extinction_evidence(row, &absolute_map);

    let mut violations = Vec::new();
    if m_cal_present && m_cal.is_none() {
        violations.push(violation(
            &key,
            "m_cal",
            "NONFINITE_M_CAL",
            "m_cal is present but is not finite".into(),
        ));
    } else if m_cal.is_some() && !calibration_valid && !diagnostic_calibration {
        let code = if calibration_statuses.is_empty() {
            "M_CAL_STATUS_MISSING"
        } else {
            "M_CAL_STATUS_INVALID"
        };
        let joined = if calibration_statuses.is_empty() {
            "missing".to_owned()
        } else {
            calibration_statuses.join(",")
        };
        violations.push(violation(
            &key,
            "m_cal",
            code,
            format!("m_cal is present but calibration status is not valid: {joined}"),
        ));
    }
    if absolute_present && absolute_magnitude.is_none() {
        violations.push(violation(
            &key,
            "M",
            "NONFINITE_M",
            "M is present but is not finite".into(),
        ));
    } else if absolute_magnitude.is_some() && !absolute_status_is_valid {
        let code = if absolute_statuses.is_empty() {
            "M_STATUS_MISSING"
        } else {
            "M_STATUS_INVALID"
        };
        let joined = if absolute_statuses.is_empty() {
            "missing".to_owned()
        } else {
            absolute_statuses.join(",")
        };
        violations.push(violation(
            &key,
            "M",
            code,
            format!("M is present but absolute status is not valid: {joined}"),
        ));
    }
    if absolute_magnitude.is_some() && absolute_status_is_valid {
        let mut missing_absolute = Vec::new();
        if !calibration_valid {
            missing_absolute.push("calibration");
        }
        if !catalog_available {
            missing_absolute.push("catalog");
        }
        let incomplete_model = distance_path == Some("model")
            && distance_issue == Some("MODEL_DISTANCE_INTERVAL_MISSING")
            && absolute_statuses
                .iter()
                .any(|status| status == "VALID_MODEL_DISTANCE_NO_INTERVAL");
        let distance_provenance_issue =
            distance_issue.is_some_and(|issue| issue != "DISTANCE_MISSING");
        if !distance_qualified && !incomplete_model && !distance_provenance_issue {
            missing_absolute.push("parallax");
        }
        if !extinction_ok {
            missing_absolute.push("extinction");
        }
        if !missing_absolute.is_empty() {
            violations.push(violation(
                &key,
                "M",
                "M_WITH_MISSING_INPUTS",
                format!(
                    "M is present/marked valid but required inputs are missing: {}",
                    missing_absolute.join(",")
                ),
            ));
        }
        if !distance_qualified && distance_provenance_issue && !incomplete_model {
            let issue = distance_issue.unwrap_or("DISTANCE_EVIDENCE_INVALID");
            violations.push(violation(
                &key,
                "distance_source",
                issue,
                format!("M is present/marked valid but distance provenance is not strict: {issue}"),
            ));
        }
    }
    if m_cal.is_none()
        && explicit_level.as_deref().is_some_and(|level| {
            [
                "RELATIVE_CALIBRATED",
                "APPARENT_CALIBRATED",
                "ABSOLUTE_ELIGIBLE",
            ]
            .contains(&level)
        })
    {
        violations.push(violation(
            &key,
            "observability_level",
            "LEVEL_WITHOUT_M_CAL",
            "calibrated level claimed without m_cal".into(),
        ));
    }
    if explicit_level.as_deref() == Some("ABSOLUTE_ELIGIBLE") && absolute_magnitude.is_none() {
        violations.push(violation(
            &key,
            "observability_level",
            "LEVEL_WITHOUT_M",
            "absolute level claimed without M".into(),
        ));
    }

    let mut missing = BTreeSet::new();
    if !m_inst_present || m_inst.is_none() {
        missing.insert("m_inst".to_owned());
    }
    let geometry_failed = require_wcs && geometry != "available";
    if geometry_failed {
        missing.insert("wcs".into());
    }
    if !catalog_available {
        missing.insert("catalog".into());
    }
    if !calibration_valid {
        missing.insert("calibration".into());
    }
    if !distance_qualified {
        missing.insert(
            match distance_issue {
                Some("DISTANCE_SOURCE_REQUIRED") => "distance_source",
                Some("MODEL_DISTANCE_INTERVAL_MISSING") => "distance_interval",
                Some("DISTANCE_MISSING") => "parallax",
                _ => "distance",
            }
            .into(),
        );
    }
    if strict_flag_found && strict_flag != Some(true) {
        missing.insert("strict_absolute_result".into());
    }
    if !extinction_ok {
        missing.insert("extinction".into());
    }
    if absolute_magnitude.is_none() {
        missing.insert("absolute_result".into());
    }
    let row_identity = value(
        Some(row),
        &["photometric_system", "photometric_band", "band"],
    );
    let calibration_identity = value(
        Some(&calibration_map),
        &["photometric_system", "photometric_band", "band"],
    );
    if calibration_valid
        && !row_identity.is_some_and(|value| nonempty(Some(value)))
        && !calibration_identity.is_some_and(|value| nonempty(Some(value)))
    {
        missing.insert("photometric_system".into());
    }

    let mut flags = BTreeSet::new();
    if let Some(raw) = value(Some(row), &["flags", "flag"]) {
        if let Some(text) = raw.as_str() {
            if !text.trim().is_empty() {
                flags.insert(norm_status(Some(raw)));
            }
        } else if let Some(items) = raw.as_array() {
            flags.extend(
                items
                    .iter()
                    .filter(|item| nonempty(Some(item)))
                    .map(|item| norm_status(Some(item))),
            );
        } else if nonempty(Some(raw)) {
            flags.insert(norm_status(Some(raw)));
        }
    }
    let mut reasons = BTreeSet::new();
    if let Some(raw) = value(
        Some(row),
        &["rejection_reasons", "rejection_reason", "reason"],
    ) {
        if let Some(items) = raw.as_array() {
            reasons.extend(items.iter().map(|item| norm_status(Some(item))));
        }
        if raw.as_str().is_some_and(|text| !text.trim().is_empty()) {
            reasons.insert(norm_status(Some(raw)));
        }
    }
    for item in &missing {
        reasons.insert(format!("MISSING_{}", item.to_uppercase()));
    }
    if let Some(issue) = distance_issue.filter(|issue| *issue != "DISTANCE_MISSING") {
        flags.insert(issue.into());
        reasons.insert(issue.into());
    }
    if let Some(issue) = extinction_issue.filter(|issue| *issue != "EXTINCTION_MISSING") {
        flags.insert(issue.into());
        reasons.insert(issue.into());
    }
    if row_status.as_deref().is_some_and(contains_invalid_status) {
        let current = row_status.clone().unwrap();
        flags.insert(current.clone());
        reasons.insert(current);
    }
    if diagnostic_calibration {
        flags.insert("M_CAL_DIAGNOSTIC_ONLY".into());
        reasons.insert("M_CAL_DIAGNOSTIC_ONLY".into());
    }
    if bool_value(value(Some(row), &["quality_passed", "quality_ok"])) == Some(false) {
        flags.insert("QUALITY_REJECTED".into());
        reasons.insert("QUALITY_REJECTED".into());
    }
    for violation in &violations {
        let code = violation["code"].as_str().unwrap_or_default().to_owned();
        flags.insert(code.clone());
        reasons.insert(code);
    }

    let mode = calibration_mode(row, &calibration_map);
    let apparent_valid =
        calibration_valid && m_cal.is_some() && catalog_available && mode == "apparent";
    let relative_valid = calibration_valid && m_cal.is_some();
    let geometry_ok = !require_wcs || geometry == "available";
    let absolute_valid = absolute_status_is_valid
        && absolute_magnitude.is_some()
        && apparent_valid
        && distance_qualified
        && extinction_ok
        && geometry_ok
        && strict_flag != Some(false);
    let mut observability = if geometry_failed {
        "GEOMETRY_UNAVAILABLE"
    } else if absolute_valid {
        "ABSOLUTE_ELIGIBLE"
    } else if apparent_valid {
        "APPARENT_CALIBRATED"
    } else if relative_valid {
        "RELATIVE_CALIBRATED"
    } else if m_inst.is_some() {
        "INSTRUMENTAL_ONLY"
    } else {
        "GEOMETRY_UNAVAILABLE"
    };
    if let Some(explicit) = explicit_level.as_deref() {
        if explicit != "GEOMETRY_UNAVAILABLE" && observability != "GEOMETRY_UNAVAILABLE" {
            let rank = |level: &str| match level {
                "INSTRUMENTAL_ONLY" => 1,
                "RELATIVE_CALIBRATED" => 2,
                "APPARENT_CALIBRATED" => 3,
                "ABSOLUTE_ELIGIBLE" => 4,
                _ => 0,
            };
            if rank(explicit) < rank(observability) {
                observability = explicit;
            }
        } else if explicit == "GEOMETRY_UNAVAILABLE" {
            observability = explicit;
        }
    }
    if observability == "GEOMETRY_UNAVAILABLE" {
        flags.insert("GEOMETRY_UNAVAILABLE".into());
        reasons.insert("GEOMETRY_UNAVAILABLE".into());
    }
    if !violations.is_empty() {
        flags.insert("FALSE_VALID".into());
    }
    if m_cal.is_some() && !calibration_valid && !diagnostic_calibration {
        flags.insert("FALSE_VALID_M_CAL".into());
    }
    if absolute_magnitude.is_some() && !absolute_status_is_valid {
        flags.insert("FALSE_VALID_M".into());
    }
    if strict_flag_found && strict_flag != Some(true) {
        flags.insert("ABSOLUTE_NOT_STRICT".into());
        reasons.insert("ABSOLUTE_NOT_STRICT".into());
    }

    let mut nested_provenance = Obj::new();
    if let Some(provenance) = value(Some(row), &["provenance"]).and_then(Value::as_object) {
        for (name, raw) in provenance {
            if raw.is_null() || raw.is_string() || raw.is_boolean() || raw.is_number() {
                nested_provenance.insert(name.clone(), json_scalar(Some(raw)));
            }
        }
    }
    let error_budget = error_values(Some(row));
    let absolute_strict = absolute_valid && strict_flag != Some(false);
    let row_value = json!({
        "row_key":key,"frame_id":frame_id,"source_id":source_id,
        "observability_level":observability,"observability":observability,"level":observability,
        "status":row_status,"calibration_status":cal_status,"absolute_status":abs_status,
        "m_inst":m_inst,"instrumental_magnitude":m_inst,
        "m_cal":m_cal,"calibrated_magnitude":m_cal,"calibrated_magnitude_value_role":calibrated_role,
        "M":absolute_magnitude,"absolute_magnitude":absolute_magnitude,"strict_M":if absolute_strict { absolute_magnitude } else { None },
        "strict_absolute_magnitude":if absolute_strict { absolute_magnitude } else { None },
        "absolute_magnitude_is_strict":absolute_strict,
        "absolute_magnitude_value_role":if absolute_strict { "STRICT" } else if absolute_magnitude.is_some() { "DIAGNOSTIC_ONLY" } else { "ABSENT" },
        "absolute_magnitude_band":absolute_band,"absolute_magnitude_system":absolute_system,"absolute_magnitude_source":absolute_source,
        "absolute_magnitude_distance_pc":absolute_support["distance_pc"],"absolute_magnitude_distance_lower_pc":absolute_support["distance_lower_pc"],
        "absolute_magnitude_distance_upper_pc":absolute_support["distance_upper_pc"],"absolute_magnitude_distance_interval_status":absolute_support["distance_interval_status"],
        "absolute_magnitude_extinction_mag":absolute_support["extinction_mag"],"absolute_magnitude_extinction_band":absolute_support["extinction_band"],
        "absolute_magnitude_extinction_system":absolute_support["extinction_system"],"absolute_magnitude_extinction_source":absolute_support["extinction_source"],
        "wcs_available":wcs_available,"catalog_available":catalog_available,
        "flags":strings_value(flags.iter().cloned()),"rejection_reasons":strings_value(reasons.iter().cloned()),"missing_inputs":strings_value(missing.iter().cloned()),
        "provenance":nested_provenance,"error_budget":error_budget,
    });
    AuditedRow {
        value: row_value,
        violations,
        geometry,
        rejection_reasons: reasons.into_iter().collect(),
        missing_inputs: missing.into_iter().collect(),
        error_budget,
    }
}

fn summarise_errors(values: &BTreeMap<String, Vec<f64>>) -> Value {
    let mut summary = Obj::new();
    for (component, raw) in values {
        let mut finite: Vec<f64> = raw
            .iter()
            .copied()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .collect();
        finite.sort_by(f64::total_cmp);
        if finite.is_empty() {
            continue;
        }
        let count = finite.len();
        let mean = finite.iter().sum::<f64>() / count as f64;
        let median = if count % 2 == 0 {
            (finite[count / 2 - 1] + finite[count / 2]) / 2.0
        } else {
            finite[count / 2]
        };
        let rms = (finite.iter().map(|value| value * value).sum::<f64>() / count as f64).sqrt();
        summary.insert(component.clone(), json!({"count":count,"min":finite[0],"max":finite[count-1],"mean":mean,"median":median,"rms":rms}));
    }
    Value::Object(summary)
}

fn provenance_summary(
    frame_rows: &[Obj],
    source_rows: &[Obj],
    calibration: &NormalisedResults,
    absolute: &NormalisedResults,
    geometry_counts: &BTreeMap<String, usize>,
    malformed: usize,
) -> Value {
    let mut declared: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for row in frame_rows.iter().chain(source_rows) {
        add_provenance(&mut declared, Some(row));
    }
    for row in calibration
        .globals
        .iter()
        .chain(unique_mappings(calibration.indexed.values()))
    {
        add_provenance(&mut declared, Some(row));
    }
    for row in absolute
        .globals
        .iter()
        .chain(unique_mappings(absolute.indexed.values()))
    {
        add_provenance(&mut declared, Some(row));
    }
    let frame_ids: BTreeSet<String> = frame_rows
        .iter()
        .filter_map(|row| {
            value(Some(row), &["frame_id", "frame"])
                .filter(|item| !item.is_null())
                .map(python_string)
        })
        .collect();
    let source_ids: BTreeSet<String> = source_rows
        .iter()
        .filter_map(|row| {
            value(Some(row), &["source_id", "detection_id", "id"])
                .filter(|item| !item.is_null())
                .map(python_string)
        })
        .collect();
    let mut declared_json = Obj::new();
    for key in PROVENANCE_KEYS
        .iter()
        .map(|(key, _)| *key)
        .collect::<BTreeSet<_>>()
    {
        declared_json.insert(
            key.into(),
            strings_value(
                declared
                    .get(key)
                    .into_iter()
                    .flat_map(|values| values.iter().cloned()),
            ),
        );
    }
    for (key, values) in &declared {
        declared_json.insert(key.clone(), strings_value(values.iter().cloned()));
    }
    json!({
        "input_shapes": {
            "frame_rows":frame_rows.len(),"source_rows":source_rows.len(),
            "calibration_results":calibration.globals.len()+unique_mappings(calibration.indexed.values()).len(),
            "absolute_results":absolute.globals.len()+unique_mappings(absolute.indexed.values()).len(),
            "malformed_rows_ignored":malformed,
        },
        "frame_ids":frame_ids,"source_ids":source_ids,
        "geometry":{"available":geometry_counts.get("available").copied().unwrap_or(0),"unavailable":geometry_counts.get("unavailable").copied().unwrap_or(0),"unknown":geometry_counts.get("unknown").copied().unwrap_or(0)},
        "declared":declared_json,
    })
}

fn first_option<'a>(values: &[Option<&'a Value>]) -> Option<&'a Value> {
    values.iter().copied().flatten().next()
}

/// Build the deterministic JSON observability report for serialized photometry inputs.
///
/// The native boundary intentionally accepts JSON mappings and sequences. It does not
/// deserialize Python objects or import a Python runtime.
pub fn build_photometric_report(input: &Value, options: &Value) -> Result<Value> {
    let mut frame_rows = Vec::new();
    let mut source_rows = Vec::new();
    let mut malformed = 0usize;
    let explicit_frames = options
        .get("frame_rows")
        .is_some_and(|value| !value.is_null())
        || options.get("frames").is_some_and(|value| !value.is_null());
    if !input.is_null() {
        let (frames, sources, bad) = normalise_frame_input(Some(input));
        frame_rows.extend(frames);
        source_rows.extend(sources);
        malformed += bad;
    }
    for key in ["frame_rows", "frames"] {
        if let Some(value) = options.get(key).filter(|value| !value.is_null()) {
            let (frames, sources, bad) = normalise_frame_input(Some(value));
            frame_rows.extend(frames);
            source_rows.extend(sources);
            malformed += bad;
        }
    }
    for key in ["source_rows", "sources", "photometry_rows"] {
        if let Some(value) = options.get(key).filter(|value| !value.is_null()) {
            let (sources, bad) = normalise_source_input(Some(value));
            source_rows.extend(sources);
            malformed += bad;
        }
    }
    let require_wcs = options
        .get("require_wcs")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| {
            explicit_frames
                || source_rows.iter().any(|row| {
                    row.get("__report_frame_context")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                        || geometry_state(Some(row), "unknown") != "unknown"
                })
        });
    let calibration_input = first_option(&[
        options.get("calibration_results"),
        options.get("calibration_rows"),
        options.get("calibration_result"),
        options.get("calibration"),
    ]);
    let absolute_input = first_option(&[
        options.get("absolute_results"),
        options.get("absolute_rows"),
        options.get("absolute_result"),
        options.get("absolute"),
    ]);
    let calibration = normalise_results(calibration_input);
    let absolute = normalise_results(absolute_input);
    malformed += calibration.malformed + absolute.malformed;

    let mut rows = Vec::new();
    let mut violations = Vec::new();
    let mut geometry_counts = BTreeMap::<String, usize>::new();
    let mut rejection_counts = BTreeMap::<String, usize>::new();
    let mut missing_counts = BTreeMap::<String, usize>::new();
    let mut error_summary_values = BTreeMap::<String, Vec<f64>>::new();
    for (index, row) in source_rows.iter().enumerate() {
        let row_calibration = result_for(row, &calibration, &["calibration", "calibration_result"]);
        let row_absolute = result_for(
            row,
            &absolute,
            &["absolute", "absolute_result", "absolute_magnitude"],
        );
        let audited = audit_row(row, index, &row_calibration, &row_absolute, require_wcs);
        *geometry_counts.entry(audited.geometry.clone()).or_default() += 1;
        for reason in &audited.rejection_reasons {
            *rejection_counts.entry(reason.clone()).or_default() += 1;
        }
        for missing in &audited.missing_inputs {
            *missing_counts.entry(missing.clone()).or_default() += 1;
        }
        for (name, value) in &audited.error_budget {
            error_summary_values
                .entry(name.clone())
                .or_default()
                .push(*value);
        }
        for nested_aliases in [
            &["calibration", "calibration_result"][..],
            &["absolute", "absolute_result", "absolute_magnitude"][..],
        ] {
            let nested = value(Some(row), nested_aliases).and_then(Value::as_object);
            for (name, value) in error_values(nested).into_iter() {
                error_summary_values.entry(name).or_default().push(value);
            }
        }
        rows.push(audited.value);
        violations.extend(audited.violations);
    }
    for mapping in calibration
        .globals
        .iter()
        .chain(absolute.globals.iter())
        .chain(unique_mappings(calibration.indexed.values()))
        .chain(unique_mappings(absolute.indexed.values()))
    {
        for (name, value) in error_values(Some(mapping)) {
            error_summary_values.entry(name).or_default().push(value);
        }
    }

    let mut level_counts = Obj::new();
    for level in LEVELS {
        level_counts.insert(level.into(), json!(0));
    }
    for row in &rows {
        if let Some(level) = row["observability_level"].as_str() {
            let count = level_counts.get(level).and_then(Value::as_u64).unwrap_or(0) + 1;
            level_counts.insert(level.into(), json!(count));
        }
    }
    let violation_codes: BTreeSet<String> = violations
        .iter()
        .filter_map(|item| item["code"].as_str().map(str::to_owned))
        .collect();
    let violation_errors: Vec<String> = violations
        .iter()
        .filter_map(|item| item["message"].as_str().map(str::to_owned))
        .collect();
    let gate = json!({
        "checked_rows":rows.len(),"violation_count":violations.len(),"error_count":violations.len(),
        "flag_count":violation_codes.len(),"passed":violations.is_empty(),
        "flags":strings_value(violation_codes.iter().cloned()),"errors":strings_value(violation_errors.iter().cloned()),"violations":violations,
    });
    let mut report_flags = violation_codes;
    if rows.is_empty() {
        report_flags.insert("NO_DATA".into());
    }
    if malformed > 0 {
        report_flags.insert("MALFORMED_ROWS_IGNORED".into());
    }
    if !violation_errors.is_empty() {
        report_flags.insert("FALSE_VALID".into());
    }
    let frame_count = if frame_rows.is_empty() {
        rows.iter()
            .filter_map(|row| row["frame_id"].as_str().map(str::to_owned))
            .collect::<BTreeSet<_>>()
            .len()
    } else {
        frame_rows.len()
    };
    let provenance = provenance_summary(
        &frame_rows,
        &source_rows,
        &calibration,
        &absolute,
        &geometry_counts,
        malformed,
    );
    let rejection_json: Obj = rejection_counts
        .into_iter()
        .map(|(key, count)| (key, json!(count)))
        .collect();
    let missing_json: Obj = missing_counts
        .into_iter()
        .map(|(key, count)| (key, json!(count)))
        .collect();
    let report = json!({
        "frame_count":frame_count,"source_count":rows.len(),
        "counts":{"frames":frame_count,"sources":rows.len(),"observability":level_counts},
        "level_counts":level_counts,"observability_counts":level_counts,"observability_levels":level_counts,
        "reason_counts":rejection_json,"missing_input_counts":missing_json,
        "rejection_reasons":rejection_json,"missing_inputs":missing_json,
        "error_budget":summarise_errors(&error_summary_values),"provenance":provenance,
        "false_valid_gate":gate,"false_valid":!violations.is_empty(),
        "flags":strings_value(report_flags.into_iter()),"errors":strings_value(violation_errors),"ignored_row_count":malformed,"rows":rows,
    });
    let strict = options
        .get("raise_on_false_valid")
        .and_then(Value::as_bool)
        .or_else(|| options.get("strict").and_then(Value::as_bool))
        .unwrap_or(false);
    if strict
        && !report["false_valid_gate"]["passed"]
            .as_bool()
            .unwrap_or(false)
    {
        return Err(format!(
            "false-valid photometry gate failed: {}",
            report["errors"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }
    Ok(report)
}

/// Reproduce the JSON envelope handling used by the standalone photometry command.
pub fn photometry_report_input(payload: &Value) -> (Value, Option<Value>, bool) {
    if !payload.is_object() {
        return (payload.clone(), None, false);
    }
    let mut analysis = payload;
    let mut automatic = false;
    if payload["analysis"].is_object() && payload["analysis"]["detection"].is_object() {
        analysis = &payload["analysis"];
        automatic = true;
    }
    if !analysis.get("detection").is_some_and(Value::is_object) {
        return (payload.clone(), None, false);
    }
    let source_rows = analysis
        .get("source_photometry")
        .filter(|value| value.is_array())
        .cloned()
        .unwrap_or_else(|| json!([]));
    let mut frame = json!({
        "frame_id":analysis.get("path").or_else(|| payload.get("frame_path")).cloned().unwrap_or_else(|| json!("frame")),
        "sources":source_rows,
    });
    if automatic {
        if !payload["affine_wcs"].is_null() {
            frame["wcs"] = payload["affine_wcs"].clone();
        }
    } else if payload.get("wcs").is_some() {
        frame["wcs"] = payload["wcs"].clone();
    }
    let raw_only = !analysis
        .get("source_photometry")
        .is_some_and(|value| value.is_array() && !value.as_array().unwrap().is_empty());
    let calibration = analysis
        .get("photometric_calibration")
        .filter(|value| !value.is_null())
        .cloned();
    (json!([frame]), calibration, raw_only)
}

pub fn report_from_cli_payload(
    payload: &Value,
    require_wcs: Option<bool>,
    strict: bool,
) -> Result<Value> {
    let (input, calibration, _) = photometry_report_input(payload);
    let mut options = json!({"strict":strict});
    if let Some(require_wcs) = require_wcs {
        options["require_wcs"] = json!(require_wcs);
    }
    if let Some(calibration) = calibration {
        options["calibration"] = calibration;
    }
    build_photometric_report(&input, &options)
}

/// Run the file-oriented report contract used by both the standalone CLI and desktop worker.
pub fn run_file_args(args: &[String]) -> Result<(Value, bool)> {
    let mut input = None;
    let mut output = None;
    let mut require_wcs = None;
    let mut strict = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--out" => {
                index += 1;
                output = args.get(index).map(String::as_str);
                if output.is_none() {
                    return Err("--out requires a path".into());
                }
            }
            "--require-wcs" => require_wcs = Some(true),
            "--no-require-wcs" => require_wcs = Some(false),
            "--strict" => strict = true,
            value if value.starts_with('-') => return Err(format!("unknown option: {value}")),
            value => {
                if input.replace(value).is_some() {
                    return Err("unexpected additional input path".into());
                }
            }
        }
        index += 1;
    }
    if input.is_none() {
        return Err("missing photometry JSON input path".into());
    }
    if args.contains(&"--require-wcs".to_owned()) && args.contains(&"--no-require-wcs".to_owned()) {
        return Err("--require-wcs and --no-require-wcs are mutually exclusive".into());
    }
    let input_path = Path::new(input.unwrap());
    let text = std::fs::read_to_string(input_path).map_err(|error| error.to_string())?;
    let payload: Value = serde_json::from_str(&text).map_err(|error| error.to_string())?;
    let (_, _, raw_only) = photometry_report_input(&payload);
    let report = report_from_cli_payload(&payload, require_wcs, strict)?;
    if let Some(output) = output {
        let output = Path::new(output);
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let rendered = serde_json::to_string(&report).map_err(|error| error.to_string())?;
        std::fs::write(output, format!("{rendered}\n")).map_err(|error| error.to_string())?;
    }
    Ok((report, raw_only))
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let (report, raw_only) = run_file_args(args)?;
    if raw_only {
        eprintln!("photometric-report: input has no source_photometry rows; raw detection.sources were intentionally not treated as calibrated photometry");
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_rows() -> Value {
        json!([{
            "frame_id":"f1", "wcs":{"status":"VALID"},
            "sources":[
                {"source_id":"inst", "m_inst":12.0, "status":"INSTRUMENTAL_ONLY"},
                {"source_id":"relative", "m_inst":12.0, "m_cal":11.5, "status":"RELATIVE_CALIBRATED", "calibration_scope":"relative"},
                {"source_id":"apparent", "m_inst":12.0, "m_cal":11.4, "status":"APPARENT_CALIBRATED", "catalog_name":"offline-catalog", "photometric_system":"Gaia Vega", "photometric_band":"G"},
                {"source_id":"absolute", "m_inst":12.0, "m_cal":11.3, "M":3.2, "status":"ABSOLUTE_ELIGIBLE", "absolute_status":"VALID", "catalog_name":"offline-catalog", "photometric_system":"Gaia Vega", "photometric_band":"G", "parallax_mas":10.0, "extinction_mag":0.2, "extinction_band":"G", "extinction_system":"Gaia", "extinction_source":"Gaia DR3 GSP-Phot: ag_gspphot"}
            ]
        }])
    }

    #[test]
    fn frame_rows_preserve_all_observability_levels() {
        let report =
            build_photometric_report(&Value::Null, &json!({"frame_rows":source_rows()})).unwrap();
        assert_eq!(report["frame_count"], 1);
        assert_eq!(report["source_count"], 4);
        assert_eq!(report["level_counts"]["INSTRUMENTAL_ONLY"], 1);
        assert_eq!(report["level_counts"]["RELATIVE_CALIBRATED"], 1);
        assert_eq!(report["level_counts"]["APPARENT_CALIBRATED"], 1);
        assert_eq!(report["level_counts"]["ABSOLUTE_ELIGIBLE"], 1);
        assert_eq!(report["level_counts"]["GEOMETRY_UNAVAILABLE"], 0);
        assert_eq!(report["rows"][3]["strict_absolute_magnitude"], 3.2);
    }

    #[test]
    fn frame_without_wcs_is_explicit_geometry_failure() {
        let report = build_photometric_report(
            &Value::Null,
            &json!({"frame_rows":[{"frame_id":"f", "sources":[{"source_id":"s", "m_inst":12.0}]}]}),
        )
        .unwrap();
        assert_eq!(
            report["rows"][0]["observability_level"],
            "GEOMETRY_UNAVAILABLE"
        );
        assert_eq!(report["missing_inputs"]["wcs"], 1);
    }

    #[test]
    fn numeric_calibrated_value_with_invalid_status_fails_gate() {
        let report = build_photometric_report(&json!([{"source_id":"s", "m_inst":12.0, "m_cal":11.0, "status":"CALIBRATION_INVALID"}]), &Value::Null).unwrap();
        assert_eq!(report["false_valid_gate"]["passed"], false);
        assert_eq!(
            report["false_valid_gate"]["violations"][0]["code"],
            "M_CAL_STATUS_INVALID"
        );
        let strict = build_photometric_report(
            &json!([{"source_id":"s", "m_cal":11.0, "status":"CALIBRATION_INVALID"}]),
            &json!({"strict":true}),
        );
        assert!(strict
            .unwrap_err()
            .contains("false-valid photometry gate failed"));
    }

    #[test]
    fn automatic_photometry_envelope_uses_compact_source_rows() {
        let payload = json!({
            "frame_path":"frame.fits", "affine_wcs":{"rms_residual_px":0.5},
            "analysis":{"path":"frame.fits", "detection":{"sources":[]},
                "photometric_calibration":{"status":"VALID", "photometric_system":"Gaia Vega", "photometric_band":"G"},
                "source_photometry":[{"detection_id":7, "source_id":"gaia-7", "m_inst":-7.5, "calibrated_magnitude":12.4, "catalog_magnitude":12.5, "photometric_system":"Gaia Vega", "photometric_band":"G", "status":"CALIBRATED"}]}
        });
        let report = report_from_cli_payload(&payload, None, false).unwrap();
        assert_eq!(report["source_count"], 1);
        assert_eq!(report["rows"][0]["m_cal"], 12.4);
        assert_eq!(
            report["rows"][0]["observability_level"],
            "APPARENT_CALIBRATED"
        );
    }

    #[test]
    fn diagnostic_and_null_values_are_not_promoted_or_flagged_false_valid() {
        let report = build_photometric_report(
            &Value::Null,
            &json!({
                "source_rows":[
                    {"source_id":"outlier","m_inst":12.0,"m_cal":11.0,"status":"CATALOG_INCONSISTENT","calibration_status":"VALID","catalog_name":"Gaia DR3","photometric_system":"Gaia Vega","photometric_band":"G","wcs_available":true},
                    {"source_id":"missing","m_inst":12.0,"m_cal":null,"M":null,"absolute_magnitude":{"value":null,"status":"NO_PARALLAX"},"status":"INSTRUMENTAL_ONLY","wcs_available":true}
                ],
                "require_wcs":true
            }),
        )
        .unwrap();
        assert_eq!(report["false_valid_gate"]["violation_count"], 0);
        assert_eq!(
            report["rows"][0]["calibrated_magnitude_value_role"],
            "DIAGNOSTIC_ONLY"
        );
        assert_eq!(
            report["rows"][0]["observability_level"],
            "INSTRUMENTAL_ONLY"
        );
        assert!(report["rows"][0]["flags"]
            .as_array()
            .unwrap()
            .contains(&json!("M_CAL_DIAGNOSTIC_ONLY")));
        assert_eq!(
            report["rows"][1]["observability_level"],
            "INSTRUMENTAL_ONLY"
        );
        assert!(!report["rows"][1]["flags"]
            .as_array()
            .unwrap()
            .contains(&json!("NONFINITE_M_CAL")));
    }

    #[test]
    fn model_distance_requires_complete_interval_and_explicit_provenance() {
        let report = build_photometric_report(
            &Value::Null,
            &json!({
                "source_rows":[
                    {"source_id":"model-point","m_inst":12.0,"m_cal":11.0,"status":"CALIBRATED","calibration_status":"VALID","catalog_name":"Gaia DR3","photometric_system":"Gaia Vega","photometric_band":"G","absolute_magnitude":{"value":3.0,"status":"VALID_MODEL_DISTANCE_NO_INTERVAL","distance_source":"Gaia DR3 GSP-Phot","distance_pc":100.0,"extinction_mag":0.2,"extinction_band":"G","extinction_system":"Gaia","extinction_source":"Gaia DR3 GSP-Phot: ag_gspphot"},"wcs_available":true},
                    {"source_id":"unprovenanced","m_inst":12.0,"m_cal":11.0,"M":3.0,"status":"ABSOLUTE_ELIGIBLE","calibration_status":"VALID","catalog_name":"Gaia DR3","photometric_system":"Gaia Vega","photometric_band":"G","distance_pc":100.0,"extinction_mag":0.2,"extinction_band":"G","extinction_system":"Gaia","extinction_source":"Gaia DR3 GSP-Phot: ag_gspphot","wcs_available":true}
                ],
                "require_wcs":true
            }),
        )
        .unwrap();
        assert_eq!(
            report["rows"][0]["observability_level"],
            "APPARENT_CALIBRATED"
        );
        assert_eq!(report["rows"][0]["absolute_magnitude"], 3.0);
        assert_eq!(
            report["rows"][0]["absolute_magnitude_value_role"],
            "DIAGNOSTIC_ONLY"
        );
        assert!(report["rows"][0]["missing_inputs"]
            .as_array()
            .unwrap()
            .contains(&json!("distance_interval")));
        assert_eq!(
            report["rows"][1]["observability_level"],
            "APPARENT_CALIBRATED"
        );
        assert!(report["rows"][1]["flags"]
            .as_array()
            .unwrap()
            .contains(&json!("DISTANCE_SOURCE_REQUIRED")));
        assert_eq!(report["false_valid_gate"]["violation_count"], 1);
    }

    #[test]
    fn keyed_frame_and_absolute_result_inputs_join_by_source_id() {
        let report = build_photometric_report(
            &Value::Null,
            &json!({
                "frames":{"f1":[{"source_id":"s1","m_inst":12.0,"m_cal":11.0,"M":3.0,"status":"ABSOLUTE_ELIGIBLE","catalog":"offline-catalog","photometric_band":"G"}]},
                "absolute_results":{"s1":{"value":3.0,"status":"VALID","parallax_mas":10.0,"extinction_mag":0.0,"extinction_band":"G","extinction_system":"Gaia","extinction_source":"test-catalog"}},
                "require_wcs":false
            }),
        )
        .unwrap();
        assert_eq!(report["frame_count"], 1);
        assert_eq!(report["source_count"], 1);
        assert_eq!(report["rows"][0]["absolute_status"], "VALID");
        assert_eq!(
            report["rows"][0]["observability_level"],
            "ABSOLUTE_ELIGIBLE"
        );
        assert_eq!(report["rows"][0]["strict_absolute_magnitude"], 3.0);
    }

    #[test]
    fn error_budget_and_provenance_are_aggregated_by_declared_field() {
        let report = build_photometric_report(
            &Value::Null,
            &json!({
                "frame_rows":[{"frame_id":"f1","wcs":{"status":"VALID","wcs_source":"offline-solution"},"sources":[{"source_id":"s1","m_inst":12.0,"m_inst_error":0.1,"m_cal":11.0,"m_cal_error":0.2,"status":"APPARENT_CALIBRATED","catalog_name":"offline-catalog","photometric_system":"Gaia Vega","photometric_band":"G","provenance":{"operator":"ignored","source_file":"frame.fits"},"error_budget":{"background":{"sigma":0.3},"not-a-number":"bad"}}]}],
                "calibration":{"status":"VALID","fit_rms_mag":0.04,"validation_rms_mag":0.05}
            }),
        )
        .unwrap();
        assert_eq!(report["error_budget"]["m_inst"]["count"], 1);
        assert_eq!(report["error_budget"]["m_cal"]["max"], 0.2);
        assert_eq!(report["error_budget"]["background"]["mean"], 0.3);
        assert_eq!(report["error_budget"]["calibration_fit"]["mean"], 0.04);
        assert_eq!(
            report["provenance"]["declared"]["catalog"],
            json!(["offline-catalog"])
        );
        assert_eq!(
            report["provenance"]["declared"]["wcs"],
            json!(["offline-solution"])
        );
        assert!(report["provenance"]["declared"].get("operator").is_none());
    }

    #[test]
    fn empty_and_unknown_rows_keep_malformed_counts_and_json_safety() {
        let empty = build_photometric_report(&Value::Null, &Value::Null).unwrap();
        assert_eq!(empty["source_count"], 0);
        assert_eq!(empty["frame_count"], 0);
        assert_eq!(empty["flags"], json!(["NO_DATA"]));
        let unknown = build_photometric_report(
            &Value::Null,
            &json!({"source_rows":[{"future_field":"ignored"}, "not-a-row"]}),
        )
        .unwrap();
        assert_eq!(unknown["source_count"], 1);
        assert_eq!(unknown["ignored_row_count"], 1);
        assert_eq!(
            unknown["rows"][0]["observability_level"],
            "GEOMETRY_UNAVAILABLE"
        );
        assert!(serde_json::to_string(&unknown).is_ok());
    }
}
