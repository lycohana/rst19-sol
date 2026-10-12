//! Local JSONL persistence for manual threshold feedback.
//!
//! These records preserve human observations for later review. They are not
//! online training data and do not alter detector measurements or thresholds.

use crate::Result;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

pub const FEEDBACK_SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ManualThresholdFeedback {
    pub timestamp_utc: String,
    pub frame_path: String,
    pub threshold_sigma: f64,
    pub min_flux_snr: f64,
    pub min_distance: i64,
    pub psf_fwhm: f64,
    pub candidate_count: i64,
    pub quality_count: i64,
    pub returned_count: i64,
    pub flag_counts: BTreeMap<String, i64>,
    pub judgement: String,
    pub note: String,
    pub source: String,
}

impl ManualThresholdFeedback {
    pub fn as_dict(&self) -> Value {
        let mut payload = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Value::Object(ref mut object) = payload {
            object.insert("schema_version".into(), json!(FEEDBACK_SCHEMA_VERSION));
        }
        payload
    }
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

fn py_float(value: &Value, field: &str) -> Result<f64> {
    match value {
        Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| format!("{field} must be numeric")),
        Value::String(raw) => raw
            .parse::<f64>()
            .map_err(|error| format!("could not convert string to float: {error}")),
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        _ => Err(format!("{field} must be numeric")),
    }
}

fn py_int(value: &Value, field: &str) -> Result<i64> {
    match value {
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                Ok(value)
            } else if let Some(value) = number.as_u64() {
                i64::try_from(value).map_err(|error| format!("{field}: {error}"))
            } else {
                number
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .and_then(|value| {
                        let value = value.trunc();
                        (value >= i64::MIN as f64 && value <= i64::MAX as f64)
                            .then_some(value as i64)
                    })
                    .ok_or_else(|| format!("{field} must be an integer"))
            }
        }
        Value::String(raw) => raw
            .parse::<i64>()
            .map_err(|error| format!("invalid literal for int(): {error}")),
        Value::Bool(value) => Ok(i64::from(*value)),
        _ => Err(format!("{field} must be an integer")),
    }
}

fn required<'a>(object: &'a Map<String, Value>, field: &str) -> Result<&'a Value> {
    object
        .get(field)
        .ok_or_else(|| format!("missing required field {field}"))
}

pub fn from_value(payload: &Value) -> Result<ManualThresholdFeedback> {
    let object = payload
        .as_object()
        .ok_or("feedback record must be a JSON object")?;
    let flag_counts = match object.get("flag_counts") {
        None => BTreeMap::new(),
        Some(Value::Object(flags)) => flags
            .iter()
            .map(|(key, value)| Ok((key.clone(), py_int(value, "flag_counts value")?)))
            .collect::<Result<BTreeMap<_, _>>>()?,
        Some(_) => return Err("flag_counts must be a JSON object".into()),
    };
    Ok(ManualThresholdFeedback {
        timestamp_utc: python_string(required(object, "timestamp_utc")?),
        frame_path: python_string(required(object, "frame_path")?),
        threshold_sigma: py_float(required(object, "threshold_sigma")?, "threshold_sigma")?,
        min_flux_snr: py_float(required(object, "min_flux_snr")?, "min_flux_snr")?,
        min_distance: py_int(required(object, "min_distance")?, "min_distance")?,
        psf_fwhm: py_float(required(object, "psf_fwhm")?, "psf_fwhm")?,
        candidate_count: py_int(required(object, "candidate_count")?, "candidate_count")?,
        quality_count: py_int(required(object, "quality_count")?, "quality_count")?,
        returned_count: py_int(required(object, "returned_count")?, "returned_count")?,
        flag_counts,
        judgement: object
            .get("judgement")
            .map(python_string)
            .unwrap_or_else(|| "当前平衡".into()),
        note: object.get("note").map(python_string).unwrap_or_default(),
        source: object
            .get("source")
            .map(python_string)
            .unwrap_or_else(|| "tkinter_manual".into()),
    })
}

pub fn append_manual_feedback(path: &Path, feedback: &ManualThresholdFeedback) -> Result<PathBuf> {
    let target = path.to_path_buf();
    if let Some(parent) = target
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&target)
        .map_err(|error| error.to_string())?;
    let encoded = serde_json::to_vec(&feedback.as_dict()).map_err(|error| error.to_string())?;
    file.write_all(&encoded)
        .map_err(|error| error.to_string())?;
    file.write_all(b"\n").map_err(|error| error.to_string())?;
    Ok(target)
}

pub fn load_manual_feedback(path: &Path) -> Result<Vec<ManualThresholdFeedback>> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.to_string()),
    };
    let mut records = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|error| error.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed = serde_json::from_str::<Value>(&line)
            .map_err(|error| format!("人工反馈第 {line_number} 行格式无效：{error}"))?;
        let record = from_value(&parsed)
            .map_err(|error| format!("人工反馈第 {line_number} 行格式无效：{error}"))?;
        records.push(record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture() -> ManualThresholdFeedback {
        ManualThresholdFeedback {
            timestamp_utc: "2026-08-30T00:00:00+00:00".into(),
            frame_path: "F01.fits".into(),
            threshold_sigma: 3.5,
            min_flux_snr: 3.0,
            min_distance: 4,
            psf_fwhm: 3.0,
            candidate_count: 120,
            quality_count: 80,
            returned_count: 120,
            flag_counts: BTreeMap::from([("LOW_FLUX_SNR".into(), 40), ("SPIKE".into(), 2)]),
            judgement: "保留弱星".into(),
            note: String::new(),
            source: "tkinter_manual".into(),
        }
    }

    fn temp_path() -> PathBuf {
        std::env::temp_dir().join(format!(
            "rst19-feedback-{}-{}.jsonl",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn appends_sorted_utf8_jsonl_records_and_round_trips() {
        let path = temp_path();
        let expected = fixture();
        append_manual_feedback(&path, &expected).unwrap();
        append_manual_feedback(&path, &expected).unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let first: Value = serde_json::from_str(contents.lines().next().unwrap()).unwrap();
        assert_eq!(first["schema_version"], FEEDBACK_SCHEMA_VERSION);
        assert_eq!(first["flag_counts"]["LOW_FLUX_SNR"], 40);
        assert!(contents.contains("保留弱星"));
        assert_eq!(
            load_manual_feedback(&path).unwrap(),
            [expected.clone(), expected]
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn missing_path_is_empty_and_corrupt_records_report_their_line() {
        let path = temp_path();
        assert!(load_manual_feedback(&path).unwrap().is_empty());
        fs::write(&path, "{}\nnot-json\n").unwrap();
        let error = load_manual_feedback(&path).unwrap_err();
        assert!(error.starts_with("人工反馈第 1 行格式无效："));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn legacy_optional_fields_and_python_numeric_coercions_are_preserved() {
        let payload = json!({
            "timestamp_utc": "now", "frame_path": "frame.fits", "threshold_sigma": "3.5",
            "min_flux_snr": 3, "min_distance": 4.9, "psf_fwhm": 2,
            "candidate_count": 10, "quality_count": 5, "returned_count": 10,
        });
        let record = from_value(&payload).unwrap();
        assert_eq!(record.min_distance, 4);
        assert_eq!(record.judgement, "当前平衡");
        assert_eq!(record.source, "tkinter_manual");
        assert!(record.flag_counts.is_empty());
    }
}
