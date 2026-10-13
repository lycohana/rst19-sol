//! Builds an auditable, gate-checked innovation evidence package from existing
//! injection summaries. It never reads FITS data or reruns a detector.

use csv::{Terminator, WriterBuilder};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

pub const SCHEMA_VERSION: i64 = 1;
pub const PRIMARY_INNOVATION_ID: &str = "conditional_detectability";
pub const PRIMARY_STATUS: &str = "CONDITIONAL_DEFENSIBLE";
pub const MIN_PRIMARY_INJECTIONS: i64 = 8;
pub const MIN_PRIMARY_LEVELS: i64 = 3;
pub const MIN_PRIMARY_TRIALS: i64 = 3;
const DEFAULT_WILSON_Z: f64 = 1.959_963_984_540_054;
const REQUIRED_FIELDS: [&str; 8] = [
    "stratum",
    "peak_excess_adu",
    "injected_count",
    "candidate_recovered_count",
    "quality_recovered_count",
    "ambiguous_injection_count",
    "unambiguous_injected_count",
    "trial_count",
];
const INT_FIELDS: [&str; 9] = [
    "psf_source_count",
    "injected_count",
    "candidate_recovered_count",
    "quality_recovered_count",
    "ambiguous_injection_count",
    "unambiguous_injected_count",
    "baseline_candidate_count",
    "baseline_quality_count",
    "trial_count",
];
const FLOAT_FIELDS: [&str; 17] = [
    "psf_median_fwhm_px",
    "psf_kernel_sum",
    "peak_excess_adu",
    "candidate_recall",
    "quality_recall",
    "unambiguous_candidate_recall",
    "unambiguous_quality_recall",
    "net_candidate_delta",
    "net_quality_delta",
    "local_background_adu",
    "local_noise_adu",
    "special_pixel_fraction",
    "nearest_baseline_source_px",
    "mean_candidate_count",
    "mean_quality_count",
    "mean_background_candidate_count",
    "mean_background_quality_count",
];
const CSV_FIELDS: [&str; 27] = [
    "stratum",
    "stratum_label",
    "control_class",
    "primary_eligible",
    "peak_excess_adu",
    "injected_count",
    "unambiguous_injected_count",
    "ambiguous_injection_count",
    "denominator_kind",
    "denominator_count",
    "candidate_recovered_count",
    "quality_recovered_count",
    "candidate_recall",
    "candidate_recall_wilson95_low",
    "candidate_recall_wilson95_high",
    "quality_recall",
    "quality_recall_wilson95_low",
    "quality_recall_wilson95_high",
    "quality_filter_gap",
    "rate_semantics",
    "ambiguous_count_is_nonexclusive",
    "minimum_unambiguous_positions_required",
    "local_background_adu",
    "local_noise_adu",
    "nominal_level_over_local_noise",
    "trial_count",
    "source_path",
];

type Row = Map<String, Value>;

fn as_float(value: &Value) -> Option<f64> {
    match value {
        Value::Null | Value::Bool(_) => None,
        Value::Number(value) => value.as_f64().filter(|number| number.is_finite()),
        Value::String(value) => value
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|number| number.is_finite()),
        _ => None,
    }
}

fn as_int(value: &Value) -> Option<i64> {
    let number = as_float(value)?;
    if number.fract() != 0.0 || number < i64::MIN as f64 || number >= i64::MAX as f64 {
        None
    } else {
        Some(number as i64)
    }
}

pub fn wilson_interval(successes: i64, trials: i64, z: f64) -> Result<(f64, f64), String> {
    if trials <= 0 || successes < 0 || successes > trials {
        return Err("Wilson interval requires 0 <= successes <= trials and trials > 0".to_owned());
    }
    if !z.is_finite() || z <= 0.0 {
        return Err("z must be a positive finite number".to_owned());
    }
    let p = successes as f64 / trials as f64;
    let denominator = 1.0 + z * z / trials as f64;
    let centre = (p + z * z / (2.0 * trials as f64)) / denominator;
    let half =
        z * ((p * (1.0 - p) + z * z / (4.0 * trials as f64)) / trials as f64).sqrt() / denominator;
    Ok(((centre - half).max(0.0), (centre + half).min(1.0)))
}

fn read_document(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("evidence artifact not found: {}: {error}", path.display()))?;
    let text = String::from_utf8(bytes)
        .map_err(|error| format!("invalid UTF-8 artifact: {}: {error}", path.display()))?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .eq_ignore_ascii_case("csv")
    {
        let mut reader = csv::ReaderBuilder::new()
            .flexible(true)
            .from_reader(text.as_bytes());
        let headers = reader
            .headers()
            .map_err(|error| format!("{}: {error}", path.display()))?
            .clone();
        let mut rows = Vec::new();
        for record in reader.records() {
            let record = record.map_err(|error| format!("{}: {error}", path.display()))?;
            let mut row = Map::new();
            for (index, raw_header) in headers.iter().enumerate() {
                let header = raw_header.trim();
                if !header.is_empty() {
                    row.insert(
                        header.to_owned(),
                        Value::String(record.get(index).unwrap_or_default().to_owned()),
                    );
                }
            }
            rows.push(Value::Object(row));
        }
        Ok(json!({"rows":rows}))
    } else {
        serde_json::from_str(text)
            .map_err(|error| format!("invalid JSON artifact: {}: {error}", path.display()))
    }
}

fn normalise_row(raw: &Map<String, Value>, index: usize) -> Result<Row, String> {
    let mut row = Map::new();
    for (raw_key, raw_value) in raw {
        let key = raw_key.trim();
        if key.is_empty() {
            continue;
        }
        let value = if raw_value.is_null() || raw_value.as_str().is_some_and(str::is_empty) {
            Value::Null
        } else if INT_FIELDS.contains(&key) {
            let number = as_int(raw_value)
                .ok_or_else(|| format!("rows[{index}].{key} must be an integer"))?;
            json!(number)
        } else if FLOAT_FIELDS.contains(&key) {
            let number = as_float(raw_value)
                .ok_or_else(|| format!("rows[{index}].{key} must be a finite number"))?;
            json!(number)
        } else {
            raw_value.clone()
        };
        row.insert(key.to_owned(), value);
    }
    for field in REQUIRED_FIELDS {
        if row.get(field).is_none_or(Value::is_null) {
            return Err(format!("rows[{index}] is missing required field: {field}"));
        }
    }
    let stratum = row
        .get("stratum")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("rows[{index}].stratum must be a non-empty string"))?
        .trim()
        .to_owned();
    if stratum.is_empty() {
        return Err(format!("rows[{index}].stratum must be a non-empty string"));
    }
    row.insert("stratum".to_owned(), json!(stratum));
    let label = row
        .get("stratum_label")
        .and_then(Value::as_str)
        .unwrap_or(&stratum)
        .trim();
    row.insert(
        "stratum_label".to_owned(),
        json!(if label.is_empty() { &stratum } else { label }),
    );
    Ok(row)
}

fn validate_row(row: &Row, index: usize) -> Result<(), String> {
    for field in INT_FIELDS {
        if let Some(value) = row.get(field).filter(|value| !value.is_null()) {
            let count = as_int(value)
                .ok_or_else(|| format!("rows[{index}].{field} must be a non-negative integer"))?;
            if count < 0 {
                return Err(format!(
                    "rows[{index}].{field} must be a non-negative integer"
                ));
            }
        }
    }
    let int = |field: &str| as_int(&row[field]).expect("required count fields were validated");
    let injected = int("injected_count");
    let candidate = int("candidate_recovered_count");
    let quality = int("quality_recovered_count");
    let ambiguous = int("ambiguous_injection_count");
    let unambiguous = int("unambiguous_injected_count");
    let trials = int("trial_count");
    if trials < 1 {
        return Err(format!("rows[{index}].trial_count must be >= 1"));
    }
    if candidate > injected {
        return Err(format!(
            "rows[{index}] candidate recovery exceeds injected count"
        ));
    }
    if quality > candidate {
        return Err(format!(
            "rows[{index}] quality recovery exceeds candidate recovery"
        ));
    }
    if ambiguous > injected {
        return Err(format!(
            "rows[{index}] ambiguous count exceeds injected count"
        ));
    }
    if unambiguous > injected {
        return Err(format!(
            "rows[{index}] unambiguous count exceeds injected count"
        ));
    }
    for field in [
        "candidate_recall",
        "quality_recall",
        "unambiguous_candidate_recall",
        "unambiguous_quality_recall",
    ] {
        if let Some(value) = row.get(field).filter(|value| !value.is_null()) {
            let number = as_float(value)
                .ok_or_else(|| format!("rows[{index}].{field} must be in [0, 1]"))?;
            if !(0.0..=1.0).contains(&number) {
                return Err(format!("rows[{index}].{field} must be in [0, 1]"));
            }
        }
    }
    if let Some(value) = row.get("local_noise_adu").filter(|value| !value.is_null()) {
        if as_float(value).is_none_or(|number| number < 0.0) {
            return Err(format!(
                "rows[{index}].local_noise_adu must be a finite non-negative number"
            ));
        }
    }
    for field in [
        "mean_candidate_count",
        "mean_quality_count",
        "mean_background_candidate_count",
        "mean_background_quality_count",
    ] {
        if let Some(value) = row.get(field).filter(|value| !value.is_null()) {
            if as_float(value).is_none_or(|number| number < 0.0) {
                return Err(format!(
                    "rows[{index}].{field} must be a finite non-negative number"
                ));
            }
        }
    }
    Ok(())
}

fn load_rows(path: &Path) -> Result<(Vec<Row>, Value), String> {
    let payload = read_document(path)?;
    let raw_rows = payload
        .get("rows")
        .or_else(|| payload.as_array().map(|_| &payload))
        .and_then(Value::as_array)
        .ok_or("evidence artifact must contain a sequence field named 'rows'")?;
    let mut rows = Vec::with_capacity(raw_rows.len());
    let mut seen = HashSet::new();
    for (index, raw) in raw_rows.iter().enumerate() {
        let raw = raw
            .as_object()
            .ok_or_else(|| format!("rows[{index}] must be an object"))?;
        let row = normalise_row(raw, index)?;
        validate_row(&row, index)?;
        let stratum = row["stratum"].as_str().unwrap_or_default().to_owned();
        let level = as_float(&row["peak_excess_adu"]).expect("required level was validated");
        let level_key = if level == 0.0 { 0 } else { level.to_bits() };
        if !seen.insert((stratum.clone(), level_key)) {
            return Err(format!(
                "duplicate stratum/peak grid cell: {stratum} / {level}"
            ));
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("evidence artifact contains no rows".to_owned());
    }
    Ok((rows, payload))
}

fn file_metadata(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let canonical = fs::canonicalize(path).map_err(|error| error.to_string())?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let format = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    Ok(json!({
        "path":canonical.to_string_lossy(),
        "sha256":digest,
        "bytes":bytes.len(),
        "format":if format.is_empty() { "text" } else { &format }
    }))
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    })
}

fn safe_rate(successes: i64, trials: i64) -> Option<f64> {
    (trials > 0).then(|| successes as f64 / trials as f64)
}

fn safe_ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    let (Some(numerator), Some(denominator)) = (numerator, denominator) else {
        return None;
    };
    (denominator != 0.0).then(|| numerator / denominator)
}

fn normalise_optional_float(row: &Row, field: &str) -> Option<f64> {
    row.get(field).and_then(as_float)
}

fn derived_row(row: &Row, index: usize, min_injections: i64) -> Result<Value, String> {
    let ambiguous = as_int(&row["ambiguous_injection_count"]).unwrap();
    let unambiguous = as_int(&row["unambiguous_injected_count"]).unwrap();
    let injected = as_int(&row["injected_count"]).unwrap();
    let candidate_count = as_int(&row["candidate_recovered_count"]).unwrap();
    let quality_count = as_int(&row["quality_recovered_count"]).unwrap();
    let trial_count = as_int(&row["trial_count"]).unwrap();
    let is_core = ambiguous == 0 && unambiguous > 0;
    let minimum_unambiguous_positions = min_injections * trial_count;
    let primary_eligible = is_core && unambiguous >= minimum_unambiguous_positions;
    let (
        denominator,
        denominator_kind,
        rate_semantics,
        candidate_recall,
        quality_recall,
        candidate_ci,
        quality_ci,
    ) = if is_core {
        let candidate_ci = wilson_interval(candidate_count, unambiguous, DEFAULT_WILSON_Z)?;
        let quality_ci = wilson_interval(quality_count, unambiguous, DEFAULT_WILSON_Z)?;
        (
            unambiguous,
            "unambiguous_injected",
            "primary_unambiguous_recall",
            safe_rate(candidate_count, unambiguous),
            safe_rate(quality_count, unambiguous),
            (Some(candidate_ci.0), Some(candidate_ci.1)),
            (Some(quality_ci.0), Some(quality_ci.1)),
        )
    } else {
        let candidate_recall = if unambiguous > 0 {
            normalise_optional_float(row, "candidate_recall")
                .or_else(|| safe_rate(candidate_count, injected))
        } else {
            None
        };
        let quality_recall = if unambiguous > 0 {
            normalise_optional_float(row, "quality_recall")
                .or_else(|| safe_rate(quality_count, injected))
        } else {
            None
        };
        (
            injected,
            "all_injected_diagnostic",
            "diagnostic_all_injected_rate",
            candidate_recall,
            quality_recall,
            (None, None),
            (None, None),
        )
    };
    let level = as_float(&row["peak_excess_adu"]).unwrap();
    let noise = normalise_optional_float(row, "local_noise_adu");
    Ok(json!({
        "row_index":index,
        "source_path":row.get("source_path"),
        "stratum":row["stratum"],
        "stratum_label":row.get("stratum_label").unwrap_or(&row["stratum"]),
        "control_class":if is_core { "core" } else { "hard_control" },
        "primary_eligible":primary_eligible,
        "control_reason":if is_core { Value::Null } else { json!("存在歧义注入位置，或没有可用于主结论的无歧义注入分母；仅作为困难条件对照。") },
        "proposal_mode":row.get("proposal_mode"),
        "psf_model":row.get("psf_model"),
        "peak_excess_adu":level,
        "injected_count":injected,
        "unambiguous_injected_count":unambiguous,
        "ambiguous_injection_count":ambiguous,
        "denominator_kind":denominator_kind,
        "rate_semantics":rate_semantics,
        "ambiguous_count_is_nonexclusive":true,
        "denominator_count":denominator,
        "minimum_unambiguous_positions_required":minimum_unambiguous_positions,
        "candidate_recovered_count":candidate_count,
        "quality_recovered_count":quality_count,
        "candidate_recall":candidate_recall,
        "candidate_recall_wilson95_low":candidate_ci.0,
        "candidate_recall_wilson95_high":candidate_ci.1,
        "quality_recall":quality_recall,
        "quality_recall_wilson95_low":quality_ci.0,
        "quality_recall_wilson95_high":quality_ci.1,
        "quality_filter_gap":candidate_recall.zip(quality_recall).map(|(candidate, quality)| candidate - quality),
        "local_background_adu":normalise_optional_float(row,"local_background_adu"),
        "local_noise_adu":noise,
        "nominal_level_over_local_noise":safe_ratio(Some(level), noise),
        "special_pixel_fraction":normalise_optional_float(row,"special_pixel_fraction"),
        "nearest_baseline_source_px":normalise_optional_float(row,"nearest_baseline_source_px"),
        "trial_count":trial_count,
        "baseline_candidate_count":row.get("baseline_candidate_count"),
        "baseline_quality_count":row.get("baseline_quality_count"),
        "net_candidate_delta":normalise_optional_float(row,"net_candidate_delta"),
        "net_quality_delta":normalise_optional_float(row,"net_quality_delta")
    }))
}

fn profile_summary(stratum: &str, rows: &[Value]) -> Value {
    let mut ordered = rows.to_vec();
    ordered.sort_by(|a, b| {
        as_float(&a["peak_excess_adu"])
            .unwrap()
            .total_cmp(&as_float(&b["peak_excess_adu"]).unwrap())
    });
    let values = |field: &str| {
        ordered
            .iter()
            .filter_map(|row| as_float(&row[field]))
            .collect::<Vec<_>>()
    };
    let candidate = values("candidate_recall");
    let quality = values("quality_recall");
    let noise = values("local_noise_adu");
    let level = |values: &[f64]| {
        if values.is_empty() {
            json!([null, null])
        } else {
            json!([
                values.iter().copied().min_by(f64::total_cmp).unwrap(),
                values.iter().copied().max_by(f64::total_cmp).unwrap()
            ])
        }
    };
    let trial_counts = ordered
        .iter()
        .map(|row| row["trial_count"].as_i64().unwrap())
        .collect::<Vec<_>>();
    json!({
        "stratum":stratum,
        "stratum_label":ordered[0].get("stratum_label").unwrap_or(&json!(stratum)),
        "levels":ordered.iter().map(|row|row["peak_excess_adu"].clone()).collect::<Vec<_>>(),
        "row_count":ordered.len(),
        "denominator_total":ordered.iter().map(|row|row["denominator_count"].as_i64().unwrap()).sum::<i64>(),
        "candidate_recall_range":level(&candidate),
        "quality_recall_range":level(&quality),
        "local_noise_adu_median":median(&noise),
        "local_noise_adu_range":level(&noise),
        "trial_count_min":trial_counts.iter().min(),
        "trial_count_max":trial_counts.iter().max()
    })
}

fn condition_contrasts(grouped: &BTreeMap<String, Vec<Value>>) -> Vec<Value> {
    let names = grouped.keys().cloned().collect::<Vec<_>>();
    let mut output = Vec::new();
    for left_index in 0..names.len() {
        for right_index in (left_index + 1)..names.len() {
            let left_name = &names[left_index];
            let right_name = &names[right_index];
            let level_key = |row: &Value| {
                let level = as_float(&row["peak_excess_adu"]).unwrap();
                if level == 0.0 {
                    0.0_f64.to_bits()
                } else {
                    level.to_bits()
                }
            };
            let left = grouped[left_name]
                .iter()
                .map(|row| (level_key(row), row))
                .collect::<BTreeMap<_, _>>();
            let right = grouped[right_name]
                .iter()
                .map(|row| (level_key(row), row))
                .collect::<BTreeMap<_, _>>();
            let mut shared = left
                .keys()
                .filter(|key| right.contains_key(*key))
                .copied()
                .collect::<Vec<_>>();
            shared.sort_by(|left, right| f64::from_bits(*left).total_cmp(&f64::from_bits(*right)));
            if shared.is_empty() {
                continue;
            }
            let mut observations = Vec::new();
            let mut noise_ratios = Vec::new();
            for key in shared {
                let left_row = left[&key];
                let right_row = right[&key];
                let left_candidate = as_float(&left_row["candidate_recall"]);
                let right_candidate = as_float(&right_row["candidate_recall"]);
                let left_quality = as_float(&left_row["quality_recall"]);
                let right_quality = as_float(&right_row["quality_recall"]);
                let left_noise = as_float(&left_row["local_noise_adu"]);
                let right_noise = as_float(&right_row["local_noise_adu"]);
                let noise_ratio = safe_ratio(right_noise, left_noise);
                if let Some(value) = noise_ratio {
                    noise_ratios.push(value);
                }
                observations.push(json!({
                    "peak_excess_adu":left_row["peak_excess_adu"],
                    "candidate_recall_delta_right_minus_left":right_candidate.zip(left_candidate).map(|(right,left)|right-left),
                    "quality_recall_delta_right_minus_left":right_quality.zip(left_quality).map(|(right,left)|right-left),
                    "left_local_noise_adu":left_noise,
                    "right_local_noise_adu":right_noise,
                    "noise_ratio_right_over_left":noise_ratio
                }));
            }
            output.push(json!({
                "left_stratum":left_name,
                "right_stratum":right_name,
                "shared_levels":observations.iter().map(|row|row["peak_excess_adu"].clone()).collect::<Vec<_>>(),
                "observations":observations,
                "median_noise_ratio_right_over_left":median(&noise_ratios),
                "interpretation":"描述性条件对照；不进行显著性检验，也不据此声称某条件具有普适排序。"
            }));
        }
    }
    output
}

fn source_evidence(
    path: &Path,
    evidence_id: &str,
    role: &str,
    status: &str,
    title: &str,
    limitation: &str,
    observed: Value,
) -> Result<Value, String> {
    Ok(json!({
        "id":evidence_id,
        "role":role,
        "status":status,
        "title":title,
        "source":file_metadata(path)?,
        "observed":observed,
        "limitation":limitation
    }))
}

fn sequence_observed(payload: &Value) -> Value {
    let mut observed = Map::new();
    for (section, keys) in [
        (
            "relation",
            [
                "frame_count",
                "duration_s",
                "interval_median_s",
                "interval_std_s",
            ]
            .as_slice(),
        ),
        ("registration", ["max_cumulative_shift_norm_px"].as_slice()),
        (
            "motion",
            ["track_count", "moving_track_count", "candidate_count"].as_slice(),
        ),
    ] {
        if let Some(values) = payload.get(section).and_then(Value::as_object) {
            for key in keys {
                if let Some(value) = values.get(*key) {
                    observed.insert((*key).to_owned(), value.clone());
                }
            }
        }
    }
    Value::Object(observed)
}

fn feature_matrix_observed(payload: &Value) -> Value {
    let mut observed = Map::new();
    if let Some(rows) = payload.get("rows").and_then(Value::as_array) {
        observed.insert("row_count".to_owned(), json!(rows.len()));
    }
    if let Some(value) = payload.get("correlation_threshold") {
        observed.insert("correlation_threshold".to_owned(), value.clone());
    }
    if let Some(parameters) = payload.get("parameters").and_then(Value::as_object) {
        for key in [
            "pattern_is_not_truth",
            "uses_probability",
            "counting_policy",
        ] {
            if let Some(value) = parameters.get(key) {
                observed.insert(key.to_owned(), value.clone());
            }
        }
    }
    Value::Object(observed)
}

fn optional_evidence(
    path: &Path,
    evidence_id: &str,
    feature_matrix: bool,
) -> Result<Value, String> {
    let payload = read_document(path)?;
    if !payload.is_object() {
        return Err(format!(
            "optional evidence {evidence_id} must be a JSON object"
        ));
    }
    if feature_matrix {
        source_evidence(
            path,
            evidence_id,
            "DIAGNOSTIC",
            "DIAGNOSTIC_ONLY",
            "特征证据矩阵（诊断路由）",
            "没有逐星真值和独立负样本；候选持续性、PSF 相似度和路由比例不能直接解释为恒星概率或误检率。",
            feature_matrix_observed(&payload),
        )
    } else {
        source_evidence(
            path,
            evidence_id,
            "BASIC_SUPPORTING",
            "SUPPORTING_ONLY",
            "15 帧关系与运动证据（基础任务支撑）",
            "图像平面轨迹、配准和差分结果可支撑基础赛项，但没有 WCS 时不能直接解释为角速度、真实速度或轨道参数。",
            sequence_observed(&payload),
        )
    }
}

fn sorted_unique_floats(rows: &[Value]) -> Vec<f64> {
    let mut values = rows
        .iter()
        .filter_map(|row| as_float(&row["peak_excess_adu"]))
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values.dedup_by(|left, right| *left == *right);
    values
}

fn intersection(sets: &[HashSet<u64>]) -> Vec<f64> {
    let Some(first) = sets.first() else {
        return Vec::new();
    };
    let mut values = first
        .iter()
        .filter(|value| sets.iter().skip(1).all(|set| set.contains(value)))
        .map(|value| f64::from_bits(*value))
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values
}

fn level_set(rows: &[Value]) -> HashSet<u64> {
    rows.iter()
        .filter_map(|row| as_float(&row["peak_excess_adu"]))
        .map(|value| {
            if value == 0.0 {
                0.0_f64.to_bits()
            } else {
                value.to_bits()
            }
        })
        .collect()
}

fn optional_min(values: &[i64]) -> Option<i64> {
    values.iter().copied().min()
}

fn optional_max(values: &[i64]) -> Option<i64> {
    values.iter().copied().max()
}

pub fn build_innovation_package(
    injection_artifact: &Path,
    sequence_report: Option<&Path>,
    feature_matrix: Option<&Path>,
    min_injections: i64,
    min_levels: i64,
    min_trials: i64,
) -> Result<Value, String> {
    if min_injections <= 0 {
        return Err("min_injections must be a positive integer".to_owned());
    }
    if min_levels <= 0 {
        return Err("min_levels must be a positive integer".to_owned());
    }
    if min_trials <= 0 {
        return Err("min_trials must be a positive integer".to_owned());
    }
    let primary_min_injections = min_injections.max(MIN_PRIMARY_INJECTIONS);
    let primary_min_levels = min_levels.max(MIN_PRIMARY_LEVELS);
    let primary_min_trials = min_trials.max(MIN_PRIMARY_TRIALS);
    let (raw_rows, _payload) = load_rows(injection_artifact)?;
    let derived_rows = raw_rows
        .iter()
        .enumerate()
        .map(|(index, row)| derived_row(row, index, primary_min_injections))
        .collect::<Result<Vec<_>, _>>()?;

    let all_strata = {
        let values = derived_rows
            .iter()
            .filter_map(|row| row["stratum"].as_str().map(str::to_owned))
            .collect::<HashSet<_>>();
        let mut values = values.into_iter().collect::<Vec<_>>();
        values.sort();
        values
    };
    let all_levels = sorted_unique_floats(&derived_rows);
    let mut core_rows: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut hard_rows: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut incomplete_rows: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for row in &derived_rows {
        let name = row["stratum"].as_str().unwrap_or_default().to_owned();
        let target = if row["primary_eligible"] == true {
            &mut core_rows
        } else if row["control_class"] == "hard_control" {
            &mut hard_rows
        } else {
            &mut incomplete_rows
        };
        target.entry(name).or_default().push(row.clone());
    }

    let core_level_sets = core_rows
        .values()
        .map(|rows| level_set(rows))
        .collect::<Vec<_>>();
    let shared_levels = intersection(&core_level_sets);
    let eligible_for_profile = core_rows
        .iter()
        .filter(|(_, rows)| sorted_unique_floats(rows).len() >= primary_min_levels as usize)
        .map(|(name, rows)| (name.clone(), rows.clone()))
        .collect::<BTreeMap<_, _>>();
    let profile_level_sets = eligible_for_profile
        .values()
        .map(|rows| level_set(rows))
        .collect::<Vec<_>>();
    let profile_shared_levels = intersection(&profile_level_sets);
    let shared_keys = profile_shared_levels
        .iter()
        .map(|value| {
            if *value == 0.0 {
                0.0_f64.to_bits()
            } else {
                value.to_bits()
            }
        })
        .collect::<HashSet<_>>();
    let profile_cells = eligible_for_profile
        .values()
        .flat_map(|rows| rows.iter())
        .filter(|row| {
            let value = as_float(&row["peak_excess_adu"]).unwrap();
            shared_keys.contains(&if value == 0.0 {
                0.0_f64.to_bits()
            } else {
                value.to_bits()
            })
        })
        .collect::<Vec<_>>();
    let profile_trial_counts = profile_cells
        .iter()
        .map(|row| row["trial_count"].as_i64().unwrap())
        .collect::<Vec<_>>();
    let profile_denominators = profile_cells
        .iter()
        .map(|row| row["denominator_count"].as_i64().unwrap())
        .collect::<Vec<_>>();
    let structure_ready = eligible_for_profile.len() >= 2
        && profile_shared_levels.len() >= primary_min_levels as usize;
    let replication_ready = !profile_trial_counts.is_empty()
        && optional_min(&profile_trial_counts).unwrap() >= primary_min_trials;
    let status = if structure_ready && replication_ready {
        PRIMARY_STATUS
    } else if !eligible_for_profile.is_empty() {
        "DIAGNOSTIC_ONLY"
    } else {
        "INCOMPLETE"
    };
    let profile_summaries = eligible_for_profile
        .iter()
        .map(|(name, rows)| profile_summary(name, rows))
        .collect::<Vec<_>>();
    let contrasts = condition_contrasts(&eligible_for_profile);
    let all_trial_counts = derived_rows
        .iter()
        .map(|row| row["trial_count"].as_i64().unwrap())
        .collect::<Vec<_>>();
    let max_profile_trials = optional_max(&profile_trial_counts);
    let replication_status = if !profile_trial_counts.is_empty()
        && optional_min(&profile_trial_counts).unwrap() >= primary_min_trials
    {
        "REPLICATED"
    } else if !profile_trial_counts.is_empty() && max_profile_trials.unwrap() >= 2 {
        "LIMITED_REPLICATION"
    } else {
        "SINGLE_LAYOUT"
    };
    let baseline_available = derived_rows.iter().any(|row| {
        !row["baseline_candidate_count"].is_null() && !row["baseline_quality_count"].is_null()
    });
    let profile_trial_min = optional_min(&profile_trial_counts);
    let profile_trial_max = max_profile_trials;
    let profile_n_min = optional_min(&profile_denominators);
    let profile_n_max = optional_max(&profile_denominators);

    let mut audit_checks = vec![
        json!({"id":"required_fields_and_bounds","status":"PASS","observed":{"row_count":raw_rows.len(),"validated_grid_cells":raw_rows.len()},"meaning":"每个条件/强度单元的计数、召回率范围和分母已通过结构校验。"}),
        json!({"id":"known_injection_denominator","status":if eligible_for_profile.is_empty(){"FAIL"}else{"PASS"},"observed":{"eligible_strata":eligible_for_profile.keys().collect::<Vec<_>>(),"min_injections_per_layout":primary_min_injections,"profile_denominator_min":profile_n_min,"profile_denominator_max":profile_n_max},"meaning":"主剖面只使用无歧义已知注入源作为召回率分母。"}),
        json!({"id":"shared_strength_grid","status":if profile_shared_levels.len()>=primary_min_levels as usize{"PASS"}else{"WARN"},"observed":{"shared_levels":profile_shared_levels,"required_levels":primary_min_levels},"meaning":"不同背景条件在共同注入强度上做描述性对照。"}),
        json!({"id":"paired_baseline","status":if baseline_available{"PASS"}else{"NOT_AVAILABLE"},"observed":{"available":baseline_available},"meaning":"保留注入前后基线字段，但它们不能单独给出 precision/FDR。"}),
        json!({"id":"independent_replication","status":if replication_ready{"PASS"}else{"FAIL"},"observed":{"profile_trial_count_min":profile_trial_min,"profile_trial_count_max":profile_trial_max,"required_min_trials":primary_min_trials,"status":replication_status},"meaning":"主剖面每个条件/强度单元必须有足够的独立位置布局；单布局只能作为诊断结果。"}),
        json!({"id":"false_positive_truth","status":"NOT_AVAILABLE","observed":{"official_per_source_labels":false},"meaning":"没有官方逐星标签或独立负样本，不能从背景检测数计算误检率。"}),
        json!({"id":"absolute_photometric_calibration","status":"NOT_AVAILABLE","observed":{"zero_point_and_color_terms":false},"meaning":"本包不宣称绝对星等、物理灵敏度或跨波段可比性。"}),
    ];
    let _ = &mut audit_checks;
    let mut support_evidence = Vec::new();
    if let Some(path) = sequence_report {
        support_evidence.push(optional_evidence(path, "sequence_relation", false)?);
    }
    if let Some(path) = feature_matrix {
        support_evidence.push(optional_evidence(path, "feature_routing", true)?);
    }
    let hard_controls = hard_rows.iter().map(|(name, rows)| {
        let mut levels = sorted_unique_floats(rows);
        levels.sort_by(f64::total_cmp);
        json!({
            "stratum":name,
            "stratum_label":rows[0].get("stratum_label").unwrap_or(&json!(name)),
            "levels":levels,
            "ambiguous_injection_total":rows.iter().map(|row|row["ambiguous_injection_count"].as_i64().unwrap()).sum::<i64>(),
            "unambiguous_injection_total":rows.iter().map(|row|row["unambiguous_injected_count"].as_i64().unwrap()).sum::<i64>(),
            "reason":rows[0].get("control_reason").unwrap_or(&Value::Null),
            "role":"HARD_CONTROL_ONLY"
        })
    }).collect::<Vec<_>>();
    let selected_claim = if status == PRIMARY_STATUS {
        format!("在当前真实背景、当前 PSF/检测参数、每个主条件/强度单元至少 {}–{} 个无歧义注入位置和 {}–{} 个独立布局的实验口径下，局部背景条件与候选层/质量层回收存在可见差异；严格质量层在困难条件或弱注入下更容易损失源。", profile_n_min.unwrap(), profile_n_max.unwrap(), profile_trial_min.unwrap(), profile_trial_max.unwrap())
    } else {
        format!("主剖面结构已经形成，但当前独立布局数为 {}，未达到交付所需的至少 {} 个布局；本包只能作为实验诊断。", profile_trial_min.map(|value|value.to_string()).unwrap_or_else(||"—".to_owned()), primary_min_trials)
    };
    let profile_denominator_text = |value: Option<i64>| {
        value
            .map(|n| n.to_string())
            .unwrap_or_else(|| "—".to_owned())
    };
    let weakness = format!("主剖面每个条件/强度单元要求至少 {primary_min_injections} 个无歧义位置/布局和 {primary_min_trials} 个独立布局；当前 profile 为 {}–{} 个位置、{}–{} 个布局", profile_denominator_text(profile_n_min), profile_denominator_text(profile_n_max), profile_trial_min.map(|n|n.to_string()).unwrap_or_else(||"—".to_owned()), profile_trial_max.map(|n|n.to_string()).unwrap_or_else(||"—".to_owned()));
    let support = support_evidence;
    let mut evidence = vec![json!({
        "id":PRIMARY_INNOVATION_ID,
        "role":"PRIMARY_INNOVATION",
        "status":status,
        "title":"真实背景分层注入—回收",
        "claim":selected_claim,
        "evidence_strength":["真实图像背景","注入位置和名义强度已知","候选层与质量层分开统计","按背景条件分层"],
        "weaknesses":[weakness,"注入源是模型源，不等同于所有真实恒星","没有独立逐星标签，因此不能计算全图误检率","没有完整相机响应/增益/平场/颜色项校准"]
    })];
    evidence.extend(support);
    let package = json!({
        "schema_version":SCHEMA_VERSION,
        "package_id":PRIMARY_INNOVATION_ID,
        "status":status,
        "title":"创新交付审计：真实背景分层注入—回收的条件化可探测性剖面",
        "selected_innovation":{
            "id":PRIMARY_INNOVATION_ID,
            "role":"PRIMARY_INNOVATION",
            "claim_level":"CONDITIONAL",
            "title":"真实背景分层注入—回收的条件化可探测性剖面",
            "claim":selected_claim,
            "measurement":"在真实 FITS 背景上注入已知位置/名义峰值的 PSF 源，比较候选层和质量层对已知注入源的回收率。",
            "formula":{"recall":"recovered_injected_sources / unambiguous_injected_sources","quality_filter_gap":"candidate_recall - quality_recall","wilson_95":"二项召回率的描述性 Wilson 区间"},
            "not_claimed":["不是整幅图的真实恒星完备率","不是 precision、误检率或 FDR","不是物理仪器灵敏度或绝对星等极限","不是跨仪器/跨波段可迁移的普适曲线"]
        },
        "provenance":{"injection_artifact":file_metadata(injection_artifact)?,"row_count":raw_rows.len(),"strata":all_strata,"peak_levels":all_levels},
        "data_summary":{
            "raw_row_count":raw_rows.len(),
            "primary_row_count":eligible_for_profile.values().map(Vec::len).sum::<usize>(),
            "hard_control_row_count":hard_rows.values().map(Vec::len).sum::<usize>(),
            "incomplete_core_row_count":incomplete_rows.values().map(Vec::len).sum::<usize>(),
            "primary_strata":eligible_for_profile.keys().collect::<Vec<_>>(),
            "hard_control_strata":hard_rows.keys().collect::<Vec<_>>(),
            "incomplete_core_strata":incomplete_rows.keys().collect::<Vec<_>>(),
            "all_levels":all_levels,
            "shared_levels_all_core_candidates":shared_levels,
            "shared_levels_primary_profile":profile_shared_levels,
            "requested_min_injections":min_injections,
            "requested_min_levels":min_levels,
            "requested_min_trials":min_trials,
            "min_injections_gate":primary_min_injections,
            "min_levels_gate":primary_min_levels,
            "min_trials_gate":primary_min_trials,
            "replication_status":replication_status,
            "trial_count_min":optional_min(&all_trial_counts),
            "trial_count_max":optional_max(&all_trial_counts),
            "profile_trial_count_min":profile_trial_min,
            "profile_trial_count_max":profile_trial_max,
            "profile_denominator_min":profile_n_min,
            "profile_denominator_max":profile_n_max
        },
        "profiles":profile_summaries,
        "condition_contrasts":contrasts,
        "hard_controls":hard_controls,
        "rows":derived_rows,
        "evidence":evidence,
        "audit":{
            "checks":audit_checks,
            "claim_boundary":{
                "can_defend":["当前数据条件下的分层经验召回率","候选层到质量层的筛选损失","不同局部噪声/背景条件下的描述性差异"],
                "cannot_defend":["真实恒星总数的普适完备率","误检率、precision 或 FDR","绝对星等和物理灵敏度","未经重复实验支持的显著性结论"]
            },
            "recommended_replication":{"minimum_independent_layouts":3,"recommendation":"每个条件/强度至少 3 个独立位置布局，扩大位置数并报告空间分布。"}
        },
        "delivery_checklist":[
            {"item":"主图：各背景条件的候选/质量召回率剖面","status":if status==PRIMARY_STATUS{"READY"}else{"BLOCKED"}},
            {"item":"表格：每个强度的 n、召回率和 Wilson 区间","status":if status==PRIMARY_STATUS{"READY"}else{"BLOCKED"}},
            {"item":"答辩口径：明确这是条件化经验结果","status":"READY"},
            {"item":"困难条件对照：crowded/special_code/line 不混入主结论","status":"READY"},
            {"item":"独立重复布局 >= 3","status":if replication_status=="REPLICATED"{"READY"}else{"TODO"}},
            {"item":"官方逐星真值/独立负样本","status":"TODO"},
            {"item":"绝对光度定标（零点、波段、颜色项）","status":"TODO"}
        ]
    });
    serde_json::to_string(&package).map_err(|error| error.to_string())?;
    Ok(package)
}

fn fmt_value(value: &Value, digits: usize) -> String {
    if value.is_null() {
        return "—".to_owned();
    }
    if let Some(boolean) = value.as_bool() {
        return if boolean { "是" } else { "否" }.to_owned();
    }
    if let Some(integer) = value.as_i64() {
        return integer.to_string();
    }
    if let Some(integer) = value.as_u64() {
        return integer.to_string();
    }
    if let Some(number) = value.as_f64() {
        return if number.is_finite() {
            format!("{number:.digits$}")
        } else {
            "—".to_owned()
        };
    }
    value.as_str().unwrap_or_default().to_owned()
}

fn pct_value(value: &Value) -> String {
    as_float(value)
        .map(|number| format!("{:.1}%", number * 100.0))
        .unwrap_or_else(|| "—".to_owned())
}

fn value_or_none(value: &Value) -> String {
    if value.is_null() {
        "None".to_owned()
    } else {
        fmt_value(value, 3)
    }
}

fn join_values(values: &[Value], digits: usize) -> String {
    values
        .iter()
        .map(|value| fmt_value(value, digits))
        .collect::<Vec<_>>()
        .join(", ")
}

fn markdown(package: &Value) -> String {
    let selected = &package["selected_innovation"];
    let summary = &package["data_summary"];
    let is_primary = package["status"] == PRIMARY_STATUS;
    let status = package["status"].as_str().unwrap_or_default();
    let profile_min = &summary["profile_denominator_min"];
    let profile_max = &summary["profile_denominator_max"];
    let trial_min = &summary["profile_trial_count_min"];
    let trial_max = &summary["profile_trial_count_max"];
    let profile_trial_range = format!("{}–{}", value_or_none(trial_min), value_or_none(trial_max));
    let profile_denominator_range = format!(
        "{}–{}",
        value_or_none(profile_min),
        value_or_none(profile_max)
    );
    let mut lines = vec![
        "# 创新交付审计".to_owned(),
        String::new(),
        format!("> 状态：**{status}**。本文件冻结当前可答辩边界，不把诊断量包装成物理真值。"),
        String::new(),
        "## 1. 最终选择的创新结果".to_owned(),
        String::new(),
        format!("**{}**", selected["title"].as_str().unwrap_or_default()),
        String::new(),
        selected["claim"].as_str().unwrap_or_default().to_owned(),
        String::new(),
        format!("测量对象：{}", selected["measurement"].as_str().unwrap_or_default()),
        String::new(),
        "计算口径：`recall = recovered_injected_sources / unambiguous_injected_sources`；质量筛选损失为 `candidate_recall - quality_recall`。Wilson 区间只表达当前聚合二项计数的不确定度，不替代独立布局之间的重复性评估。".to_owned(),
        String::new(),
        "## 2. 当前数据证据".to_owned(),
        String::new(),
        format!("- 原始实验单元：{}；主剖面单元：{}。", summary["raw_row_count"], summary["primary_row_count"]),
        format!("- 主剖面条件：{}；共同强度：{} ADU。", if summary["primary_strata"].as_array().is_some_and(Vec::is_empty) { "无".to_owned() } else { summary["primary_strata"].as_array().map(|values|values.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", ")).unwrap_or_default() }, if summary["shared_levels_primary_profile"].as_array().is_some_and(Vec::is_empty) { "无".to_owned() } else { join_values(summary["shared_levels_primary_profile"].as_array().map(Vec::as_slice).unwrap_or_default(), 3) }),
        format!("- 每个条件的独立布局计数范围：{}–{}；当前审计状态：**{}**。", summary["trial_count_min"], summary["trial_count_max"], summary["replication_status"].as_str().unwrap_or_default()),
        "- 主剖面只纳入无歧义注入位置；`crowded`、`special_code`、`line` 等困难条件保留作控制，不混入主结论。".to_owned(),
        String::new(),
        "### 主剖面明细".to_owned(),
        String::new(),
        "| 条件 | 峰值超额 ADU | n | 候选召回 | 候选 Wilson 95% | 质量召回 | 质量 Wilson 95% | 筛选损失 | 局部噪声 ADU |".to_owned(),
        "| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |".to_owned(),
    ];
    for row in package["rows"].as_array().into_iter().flatten() {
        if row["primary_eligible"] != true {
            continue;
        }
        let candidate_ci = format!(
            "{}–{}",
            pct_value(&row["candidate_recall_wilson95_low"]),
            pct_value(&row["candidate_recall_wilson95_high"])
        );
        let quality_ci = format!(
            "{}–{}",
            pct_value(&row["quality_recall_wilson95_low"]),
            pct_value(&row["quality_recall_wilson95_high"])
        );
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            row["stratum"].as_str().unwrap_or_default(),
            fmt_value(&row["peak_excess_adu"], 1),
            row["denominator_count"],
            pct_value(&row["candidate_recall"]),
            candidate_ci,
            pct_value(&row["quality_recall"]),
            quality_ci,
            pct_value(&row["quality_filter_gap"]),
            fmt_value(&row["local_noise_adu"], 2),
        ));
    }
    lines.extend([
        String::new(),
        "### 条件对照".to_owned(),
        String::new(),
        "以下差值均为右侧条件减左侧条件的描述性结果，不代表统计显著性或普适性能排名。".to_owned(),
        String::new(),
        "| 左条件 | 右条件 | 共同强度 ADU | 候选召回差值 | 质量召回差值 | 噪声比（右/左） |"
            .to_owned(),
        "| --- | --- | --- | ---: | ---: | ---: |".to_owned(),
    ]);
    for contrast in package["condition_contrasts"]
        .as_array()
        .into_iter()
        .flatten()
    {
        for observation in contrast["observations"].as_array().into_iter().flatten() {
            lines.push(format!(
                "| {} | {} | {} | {} | {} | {} |",
                contrast["left_stratum"].as_str().unwrap_or_default(),
                contrast["right_stratum"].as_str().unwrap_or_default(),
                fmt_value(&observation["peak_excess_adu"], 1),
                pct_value(&observation["candidate_recall_delta_right_minus_left"]),
                pct_value(&observation["quality_recall_delta_right_minus_left"]),
                fmt_value(&observation["noise_ratio_right_over_left"], 2),
            ));
        }
    }
    lines.extend([
        String::new(),
        "## 3. 哪些可以答辩，哪些只能诊断".to_owned(),
        String::new(),
        if is_primary { "### 可以答辩的主结果" } else { "### 当前只能作为诊断" }.to_owned(),
        String::new(),
        if is_primary {
            format!("1. 在明确的真实背景、PSF 模型、检测参数和注入分母下，给出分层经验召回率；每个主剖面单元的聚合分母为 {profile_denominator_range}。")
        } else {
            "1. 主剖面结构已经形成，但独立布局门槛尚未满足，当前结果不能作为最终答辩主曲线。".to_owned()
        },
        "2. 同一注入强度下比较候选层与质量层，说明“宽筛保召回、严筛保可靠性”的数据依据。".to_owned(),
        if is_primary {
            format!("3. 用 Wilson 区间展示聚合 n={profile_denominator_range} 的不确定性，并同时报告 {profile_trial_range} 个独立布局。")
        } else {
            format!("3. 当前 profile 的独立布局数为 {}–{}，要求至少 {}。", value_or_none(trial_min), value_or_none(trial_max), summary["min_trials_gate"])
        },
        String::new(),
        "### 支撑证据，不单独算创新得分".to_owned(),
        String::new(),
        "- 15 帧运动/配准/差分：用于基础赛项和实验背景说明；没有 WCS 时只能报告像素平面量。".to_owned(),
        "- 注册拼图：用于直观展示固定星场和配准覆盖，不是新的恒星计数或光度定标结果。".to_owned(),
        String::new(),
        "### 诊断工具，不应直接写成创新结论".to_owned(),
        String::new(),
        "- 特征证据矩阵：用于值域、形态、PSF 和路由审计；没有逐星真值，不能称为恒星概率或误检率。".to_owned(),
        "- 困难条件 `crowded`、`special_code`、`line`：可展示算法边界，但歧义/污染使其不适合进入主召回曲线。".to_owned(),
        "- `innovation.py` 的综合 JSON/PNG：是证据导出器，不是自动的比赛结论生成器。".to_owned(),
        String::new(),
        "## 4. 明确不能声称的内容".to_owned(),
        String::new(),
    ]);
    for item in selected["not_claimed"].as_array().into_iter().flatten() {
        lines.push(format!("- {}。", item.as_str().unwrap_or_default()));
    }
    lines.extend([
        String::new(),
        "## 5. 当前未完成项".to_owned(),
        String::new(),
        format!("- 每个主条件/强度单元至少 {} 个独立位置布局，并扩大空间采样；当前 profile `trial_count` 为 {}–{}。", summary["min_trials_gate"], value_or_none(trial_min), value_or_none(trial_max)),
        format!("- 每个独立布局至少 {} 个无歧义位置；当前 profile 聚合分母为 {}–{}。", summary["min_injections_gate"], value_or_none(profile_min), value_or_none(profile_max)),
        "- 获取官方逐星真值或构造独立负样本，才能审计 precision/FDR。".to_owned(),
        "- 以真实相机响应、增益、暗场/平场、波段和标准星零点补足绝对光度链路。".to_owned(),
        "- 用留出星表匹配和空间变 PSF 对照验证跨区域稳定性；当前注入源仍是实验模型源。".to_owned(),
        "- 证明领域差异前要做文献/同类作品对照；本包不自动宣称“独创性”。".to_owned(),
        String::new(),
        "## 6. 推荐答辩表述".to_owned(),
        String::new(),
        format!("> 我们没有把全图检测数直接称为恒星数，而是在真实 FITS 背景中注入已知源，按空白、高背景、边缘等条件分层，分别统计候选层和质量层的回收率。结果表明，检测阈值与质量筛选对弱源的影响依赖局部背景条件；因此我们把它定义为当前数据和当前参数下的条件化经验可探测性剖面。每个独立布局要求至少 {} 个无歧义位置，并要求至少 {} 个独立布局；尚无官方逐星真值时，我们不把它外推为普适完备率、误检率或物理灵敏度。", summary["min_injections_gate"], summary["min_trials_gate"]),
        String::new(),
        "## 7. 生成与复核".to_owned(),
        String::new(),
        "```powershell".to_owned(),
        "cargo run --manifest-path backend/Cargo.toml --bin rst19-innovation-package -- `".to_owned(),
        "  tmp\\stratified-real-background-code-pattern-current-n8\\stratified_real_background_injection.json `".to_owned(),
        "  --out-dir $env:TEMP\\rst19-innovation-package `".to_owned(),
        "  --sequence-report tmp\\innovation-current-psf-support\\innovation_report.json `".to_owned(),
        "  --feature-matrix tmp\\feature-evidence-matrix-code-pattern-current-v3-routing\\feature_evidence_matrix.json `".to_owned(),
        "  --require-defensible".to_owned(),
        "```".to_owned(),
        String::new(),
        "输出 `innovation_package.json`、`innovation_delivery.csv` 和本审计 Markdown。输出目录应放在临时目录或发布附件目录，不把 FITS、tmp、cache 加入 Git。".to_owned(),
        String::new(),
        "## 8. 交付清单".to_owned(),
        String::new(),
    ]);
    for item in package["delivery_checklist"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let ready = item["status"] == "READY";
        lines.push(format!(
            "- [{}] {}（{}）",
            if ready { "x" } else { " " },
            item["item"].as_str().unwrap_or_default(),
            item["status"].as_str().unwrap_or_default()
        ));
    }
    lines.push(String::new());
    lines.join("\n")
}

fn csv_cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => (if *value { "True" } else { "False" }).to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

pub fn write_innovation_package(package: &Value, out_dir: &Path) -> Result<Value, String> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let json_path = out_dir.join("innovation_package.json");
    let csv_path = out_dir.join("innovation_delivery.csv");
    let markdown_path = out_dir.join("innovation_delivery.md");
    let mut json_bytes = serde_json::to_vec_pretty(package).map_err(|error| error.to_string())?;
    json_bytes.push(b'\n');
    fs::write(&json_path, json_bytes).map_err(|error| error.to_string())?;
    let mut file = File::create(&csv_path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    let mut writer = WriterBuilder::new()
        .terminator(Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(CSV_FIELDS)
        .map_err(|error| error.to_string())?;
    for row in package["rows"].as_array().into_iter().flatten() {
        let fields = CSV_FIELDS
            .iter()
            .map(|field| csv_cell(row.get(*field).unwrap_or(&Value::Null)))
            .collect::<Vec<_>>();
        writer
            .write_record(fields)
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    fs::write(&markdown_path, markdown(package)).map_err(|error| error.to_string())?;
    let canonical = |path: &Path| {
        fs::canonicalize(path)
            .map(|path| path.to_string_lossy().into_owned())
            .map_err(|error| error.to_string())
    };
    Ok(json!({
        "json":canonical(&json_path)?,
        "csv":canonical(&csv_path)?,
        "markdown":canonical(&markdown_path)?
    }))
}

fn option_value<'a>(
    args: &'a [String],
    index: &mut usize,
    name: &str,
    inline: Option<&'a str>,
) -> Result<&'a str, String> {
    if let Some(value) = inline {
        return Ok(value);
    }
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| format!("{name} requires a value"))
}

fn positive_arg(value: &str, name: &str) -> Result<i64, String> {
    let parsed = value
        .parse::<i64>()
        .map_err(|error| format!("invalid value for {name}: {error}"))?;
    if parsed <= 0 {
        return Err(format!("{name} must be a positive integer"));
    }
    Ok(parsed)
}

pub fn run_desktop(args: &[String]) -> Result<Value, String> {
    let mut positional = None;
    let mut out_dir = PathBuf::from("tmp/innovation-package");
    let mut sequence_report = None;
    let mut feature_matrix = None;
    let mut min_injections = MIN_PRIMARY_INJECTIONS;
    let mut min_levels = MIN_PRIMARY_LEVELS;
    let mut min_trials = MIN_PRIMARY_TRIALS;
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--require-defensible" {
            index += 1;
            continue;
        }
        let (name, inline) = args[index]
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((args[index].as_str(), None));
        if [
            "--out-dir",
            "--sequence-report",
            "--feature-matrix",
            "--min-injections",
            "--min-levels",
            "--min-trials",
        ]
        .contains(&name)
        {
            let value = option_value(args, &mut index, name, inline)?;
            match name {
                "--out-dir" => out_dir = PathBuf::from(value),
                "--sequence-report" => sequence_report = Some(PathBuf::from(value)),
                "--feature-matrix" => feature_matrix = Some(PathBuf::from(value)),
                "--min-injections" => min_injections = positive_arg(value, name)?,
                "--min-levels" => min_levels = positive_arg(value, name)?,
                _ => min_trials = positive_arg(value, name)?,
            }
        } else if name.starts_with('-') {
            return Err(format!("unknown argument {name}"));
        } else if positional.replace(PathBuf::from(name)).is_some() {
            return Err(format!("unexpected positional argument {name}"));
        }
        index += 1;
    }
    let injection = positional.ok_or("stratified artifact is required")?;
    let package = build_innovation_package(
        &injection,
        sequence_report.as_deref(),
        feature_matrix.as_deref(),
        min_injections,
        min_levels,
        min_trials,
    )?;
    let paths = write_innovation_package(&package, &out_dir)?;
    Ok(json!({
        "tool":"rst19-innovation-package",
        "status":package["status"],
        "package":package,
        "paths":paths,
        "out_dir":fs::canonicalize(out_dir).map_err(|error|error.to_string())?
    }))
}

pub fn run_cli(args: &[String]) -> Result<i32, String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("rst19-innovation-package ARTIFACT.json|csv [--out-dir DIR] [--sequence-report JSON] [--feature-matrix JSON] [--min-injections N] [--min-levels N] [--min-trials N] [--require-defensible]");
        return Ok(0);
    }
    let result = run_desktop(args)?;
    println!("status: {}", result["status"].as_str().unwrap_or_default());
    println!(
        "primary: {}",
        result["package"]["selected_innovation"]["title"]
            .as_str()
            .unwrap_or_default()
    );
    println!(
        "json: {}",
        result["paths"]["json"].as_str().unwrap_or_default()
    );
    println!(
        "csv: {}",
        result["paths"]["csv"].as_str().unwrap_or_default()
    );
    println!(
        "markdown: {}",
        result["paths"]["markdown"].as_str().unwrap_or_default()
    );
    let strict = args.iter().any(|arg| arg == "--require-defensible");
    Ok(if strict && result["status"] != PRIMARY_STATUS {
        2
    } else {
        0
    })
}

pub fn handle_request(request: &Value) -> Result<Value, String> {
    let injection = request["injection_artifact_path"]
        .as_str()
        .ok_or("missing injection_artifact_path")?;
    let path_option = |name: &str| -> Result<Option<PathBuf>, String> {
        request
            .get(name)
            .filter(|value| !value.is_null())
            .map(|value| {
                value
                    .as_str()
                    .map(PathBuf::from)
                    .ok_or_else(|| format!("{name} must be a path string"))
            })
            .transpose()
    };
    let min_injections = request
        .get("min_injections")
        .and_then(Value::as_i64)
        .unwrap_or(MIN_PRIMARY_INJECTIONS);
    let min_levels = request
        .get("min_levels")
        .and_then(Value::as_i64)
        .unwrap_or(MIN_PRIMARY_LEVELS);
    let min_trials = request
        .get("min_trials")
        .and_then(Value::as_i64)
        .unwrap_or(MIN_PRIMARY_TRIALS);
    let package = build_innovation_package(
        Path::new(injection),
        path_option("sequence_report_path")?.as_deref(),
        path_option("feature_matrix_path")?.as_deref(),
        min_injections,
        min_levels,
        min_trials,
    )?;
    let mut response = json!({"status":package["status"],"package":package});
    if let Some(output) = request.get("output_dir").and_then(Value::as_str) {
        let paths = write_innovation_package(&response["package"], Path::new(output))?;
        response["paths"] = paths;
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::condition_contrasts;
    use serde_json::{json, Value};
    use std::collections::BTreeMap;

    fn row(level: f64) -> Value {
        json!({
            "peak_excess_adu":level,
            "candidate_recall":0.5,
            "quality_recall":0.25,
            "local_noise_adu":4.0
        })
    }

    #[test]
    fn condition_contrasts_sort_numeric_levels_and_match_signed_zero() {
        let grouped = BTreeMap::from([
            ("a".to_owned(), vec![row(1.0), row(-0.0), row(-2.0)]),
            ("b".to_owned(), vec![row(-2.0), row(0.0), row(1.0)]),
        ]);
        let contrasts = condition_contrasts(&grouped);
        let levels = contrasts[0]["shared_levels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(levels, vec![-2.0, -0.0, 1.0]);
    }
}
