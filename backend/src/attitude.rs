//! Auxiliary quaternion/boresight consistency audit.
use crate::{coordinates::auxiliary_boresight_radec, fits, models::AuxiliaryData, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

const NOTE: &str = "四元数-光轴一致性只验证辅助字段之间的约定；它不提供像元角尺度、图像轴方向、畸变或曝光期间姿态变化，因此不能单独构成完整 WCS。";
const AUXILIARY_COUNT: usize = 26;

#[derive(Debug, Clone, Deserialize)]
pub struct AttitudeFrameInput {
    pub path: String,
    #[serde(default)]
    pub header: Map<String, Value>,
    #[serde(default)]
    pub auxiliary_values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditRow {
    frame_index: usize,
    frame_number: usize,
    path: String,
    timestamp: Option<String>,
    auxiliary_ra_deg: Option<f64>,
    auxiliary_dec_deg: Option<f64>,
    derived_ra_deg: Option<f64>,
    derived_dec_deg: Option<f64>,
    delta_ra_arcsec: Option<f64>,
    delta_dec_arcsec: Option<f64>,
    angular_residual_arcsec: Option<f64>,
    quaternion_norm: Option<f64>,
    p_az_deg: Option<f64>,
    p_el_deg: Option<f64>,
    roll_deg: Option<f64>,
    pitch_deg: Option<f64>,
    yaw_deg: Option<f64>,
    header_azimuth_deg: Option<f64>,
    header_elevation_deg: Option<f64>,
    header_pointing_match: Option<bool>,
    status: String,
}

fn optional_float(value: Option<&Value>) -> Option<f64> {
    let value = value?;
    let parsed = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(string) => string.trim().parse::<f64>().ok(),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }?;
    parsed.is_finite().then_some(parsed)
}

fn string_value(value: Option<&Value>) -> Option<String> {
    let value = value?;
    if value.is_null() {
        return None;
    }
    Some(match value {
        Value::String(string) => string.clone(),
        Value::Null => return None,
        Value::Bool(boolean) => if *boolean { "True" } else { "False" }.to_owned(),
        Value::Number(number) => number.to_string(),
        other => other.to_string(),
    })
}

fn row_for_frame(index: usize, input: &AttitudeFrameInput, axis: &[f64]) -> Result<AuditRow> {
    let header_azimuth = optional_float(input.header.get("AZIMUTH"));
    let header_elevation = optional_float(input.header.get("ELEVATIO"));
    let timestamp = string_value(input.header.get("DATE-OBS"));
    let Some(values) = input.auxiliary_values.as_ref() else {
        return Ok(AuditRow {
            frame_index: index,
            frame_number: index + 1,
            path: input.path.clone(),
            timestamp,
            auxiliary_ra_deg: None,
            auxiliary_dec_deg: None,
            derived_ra_deg: None,
            derived_dec_deg: None,
            delta_ra_arcsec: None,
            delta_dec_arcsec: None,
            angular_residual_arcsec: None,
            quaternion_norm: None,
            p_az_deg: None,
            p_el_deg: None,
            roll_deg: None,
            pitch_deg: None,
            yaw_deg: None,
            header_azimuth_deg: header_azimuth,
            header_elevation_deg: header_elevation,
            header_pointing_match: None,
            status: "missing_auxiliary".to_owned(),
        });
    };
    if values.len() != AUXILIARY_COUNT {
        return Err(format!(
            "expected {AUXILIARY_COUNT} auxiliary values, got {}",
            values.len()
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("auxiliary data contains NaN or infinity".into());
    }
    let auxiliary = AuxiliaryData {
        values: values
            .clone()
            .try_into()
            .map_err(|_| "invalid auxiliary value count")?,
    };
    let (derived_ra, derived_dec) = auxiliary_boresight_radec(&auxiliary, axis)?;
    let aux_ra = values[2];
    let aux_dec = values[3];
    let delta_ra = (derived_ra - aux_ra + 180.0).rem_euclid(360.0) - 180.0;
    let delta_ra_arcsec = delta_ra * 3600.0;
    let delta_dec_arcsec = (derived_dec - aux_dec) * 3600.0;
    let residual = (delta_ra_arcsec * aux_dec.to_radians().cos()).hypot(delta_dec_arcsec);
    let p_az = optional_float(Some(&Value::from(values[0])));
    let p_el = optional_float(Some(&Value::from(values[1])));
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9f64.max(1e-9 * a.abs().max(b.abs()));
    let header_match = p_az
        .zip(p_el)
        .zip(header_azimuth.zip(header_elevation))
        .is_some_and(|((p_az, p_el), (header_az, header_el))| {
            close(p_az, header_az) && close(p_el, header_el)
        });
    Ok(AuditRow {
        frame_index: index,
        frame_number: index + 1,
        path: input.path.clone(),
        timestamp,
        auxiliary_ra_deg: Some(aux_ra),
        auxiliary_dec_deg: Some(aux_dec),
        derived_ra_deg: Some(derived_ra),
        derived_dec_deg: Some(derived_dec),
        delta_ra_arcsec: Some(delta_ra_arcsec),
        delta_dec_arcsec: Some(delta_dec_arcsec),
        angular_residual_arcsec: Some(residual),
        quaternion_norm: Some(
            values[10..14]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt(),
        ),
        p_az_deg: p_az,
        p_el_deg: p_el,
        roll_deg: Some(values[14]),
        pitch_deg: Some(values[15]),
        yaw_deg: Some(values[16]),
        header_azimuth_deg: header_azimuth,
        header_elevation_deg: header_elevation,
        header_pointing_match: Some(header_match),
        status: "ok".to_owned(),
    })
}

fn range(values: impl IntoIterator<Item = Option<f64>>) -> Value {
    let values = values.into_iter().flatten().collect::<Vec<_>>();
    let min = values.iter().copied().reduce(f64::min);
    let max = values.iter().copied().reduce(f64::max);
    json!({"min":min,"max":max,"range":min.zip(max).map(|(min,max)| max-min)})
}

fn summarize(rows: &[AuditRow], camera_body_axis: &[f64]) -> Value {
    let residuals = rows
        .iter()
        .filter_map(|row| row.angular_residual_arcsec)
        .collect::<Vec<_>>();
    let norms = rows
        .iter()
        .filter_map(|row| row.quaternion_norm)
        .collect::<Vec<_>>();
    let mean_square = if residuals.is_empty() {
        None
    } else {
        Some(
            (residuals.iter().map(|value| value * value).sum::<f64>() / residuals.len() as f64)
                .sqrt(),
        )
    };
    let ra_range = range(rows.iter().map(|row| row.auxiliary_ra_deg));
    let dec_range = range(rows.iter().map(|row| row.auxiliary_dec_deg));
    let angle_ranges = [
        (
            "roll_deg",
            rows.iter().map(|row| row.roll_deg).collect::<Vec<_>>(),
        ),
        (
            "pitch_deg",
            rows.iter().map(|row| row.pitch_deg).collect::<Vec<_>>(),
        ),
        (
            "yaw_deg",
            rows.iter().map(|row| row.yaw_deg).collect::<Vec<_>>(),
        ),
    ];
    let pointing_rows = rows
        .iter()
        .filter(|row| row.header_pointing_match.is_some())
        .collect::<Vec<_>>();
    json!({
        "frame_count": rows.len(),
        "ok_frame_count": rows.iter().filter(|row| row.status == "ok").count(),
        "missing_auxiliary_count": rows.iter().filter(|row| row.status == "missing_auxiliary").count(),
        "camera_body_axis": camera_body_axis,
        "quaternion_norm": {
            "min": norms.iter().copied().reduce(f64::min),
            "max": norms.iter().copied().reduce(f64::max),
            "max_abs_deviation_from_one": norms.iter().map(|value| (value-1.0).abs()).reduce(f64::max),
        },
        "boresight_residual_arcsec": {
            "max": residuals.iter().copied().reduce(f64::max),
            "rms": mean_square,
        },
        "auxiliary_ra_deg": ra_range.clone(),
        "auxiliary_dec_deg": dec_range.clone(),
        "auxiliary_ra_range_arcsec": ra_range["range"].as_f64().map(|value| value * 3600.0),
        "auxiliary_dec_range_arcsec": dec_range["range"].as_f64().map(|value| value * 3600.0),
        "attitude_ranges": angle_ranges
            .into_iter()
            .map(|(name, values)| (name.to_owned(), range(values)))
            .collect::<Map<_, _>>(),
        "header_pointing_match_count": pointing_rows.iter().filter(|row| row.header_pointing_match == Some(true)).count(),
        "header_pointing_checked_count": pointing_rows.len(),
        "note": NOTE,
    })
}

pub fn audit_frames(
    inputs: &[AttitudeFrameInput],
    camera_body_axis: Option<&[f64]>,
) -> Result<Value> {
    let default_axis = [0.0, -1.0, 0.0];
    let axis = camera_body_axis.unwrap_or(&default_axis);
    let rows = inputs
        .iter()
        .enumerate()
        .map(|(index, frame)| row_for_frame(index, frame, axis))
        .collect::<Result<Vec<_>>>()?;
    let summary = summarize(&rows, axis);
    let frames = rows
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    Ok(json!({"schema_version":1,"summary":summary,"frames":frames}))
}

pub fn audit_paths(paths: &[String], camera_body_axis: Option<&[f64]>) -> Result<Value> {
    let mut inputs = Vec::with_capacity(paths.len());
    for path in paths {
        let frame = fits::read_fits(path, false)?;
        let resolved =
            std::fs::canonicalize(path).unwrap_or_else(|_| Path::new(path).to_path_buf());
        let mut resolved_text = resolved.to_string_lossy().to_string();
        if let Some(stripped) = resolved_text.strip_prefix(r"\\?\") {
            resolved_text = stripped.to_owned();
        }
        inputs.push(AttitudeFrameInput {
            path: resolved_text,
            header: frame.header,
            auxiliary_values: frame.auxiliary.map(|auxiliary| auxiliary.values.to_vec()),
        });
    }
    audit_frames(&inputs, camera_body_axis)
}

const AUDIT_COLUMNS: &[&str] = &[
    "frame_index",
    "frame_number",
    "path",
    "timestamp",
    "auxiliary_ra_deg",
    "auxiliary_dec_deg",
    "derived_ra_deg",
    "derived_dec_deg",
    "delta_ra_arcsec",
    "delta_dec_arcsec",
    "angular_residual_arcsec",
    "quaternion_norm",
    "p_az_deg",
    "p_el_deg",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "header_azimuth_deg",
    "header_elevation_deg",
    "header_pointing_match",
    "status",
];

fn csv_cell(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Bool(value)) => if *value { "True" } else { "False" }.to_owned(),
        Some(Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
    }
}

/// Write the same UTF-8-BOM CSV and JSON artifacts as the Python attitude tool.
pub fn write_artifacts(payload: &Value, output_dir: &Path) -> Result<PathBuf> {
    let frames = payload["frames"]
        .as_array()
        .ok_or("attitude audit payload is missing frames")?;
    let parent = output_dir;
    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;

    let csv_path = parent.join("auxiliary_boresight_audit.csv");
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new());
    writer
        .write_record(AUDIT_COLUMNS)
        .map_err(|error| error.to_string())?;
    for frame in frames {
        let row = AUDIT_COLUMNS
            .iter()
            .map(|column| csv_cell(frame.get(*column)))
            .collect::<Vec<_>>();
        writer
            .write_record(row)
            .map_err(|error| error.to_string())?;
    }
    let csv_bytes = writer.into_inner().map_err(|error| error.to_string())?;
    let mut bom_csv = Vec::with_capacity(csv_bytes.len() + 3);
    bom_csv.extend_from_slice(&[0xef, 0xbb, 0xbf]);
    bom_csv.extend_from_slice(&csv_bytes);
    std::fs::write(csv_path, bom_csv).map_err(|error| error.to_string())?;

    let json_path = parent.join("auxiliary_boresight_audit.json");
    let mut rendered = serde_json::to_string_pretty(payload).map_err(|error| error.to_string())?;
    rendered.push('\n');
    std::fs::write(json_path, rendered).map_err(|error| error.to_string())?;
    Ok(parent.to_path_buf())
}

/// Run the `rst19-aux-audit` command line interface.
pub fn run_cli(args: &[String]) -> Result<()> {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("审计 FITS 辅助四元数与 ra/dec 光轴的一致性\n\nUsage: rst19-aux-audit INPUT_PATH [--out-dir DIR]");
        return Ok(());
    }
    let mut input = None;
    let mut output = PathBuf::from("tmp/auxiliary-attitude-audit");
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--out-dir" {
            index += 1;
            output = PathBuf::from(args.get(index).ok_or("--out-dir requires a value")?);
        } else if let Some(value) = arg.strip_prefix("--out-dir=") {
            output = PathBuf::from(value);
        } else if arg.starts_with('-') {
            return Err(format!("unknown argument {arg}"));
        } else if input.replace(PathBuf::from(arg)).is_some() {
            return Err("only one FITS file or directory is allowed".into());
        }
        index += 1;
    }
    let input = input.ok_or("an input FITS file or directory is required")?;
    let paths = if input.is_file() {
        vec![input.clone()]
    } else if input.is_dir() {
        let mut paths = std::fs::read_dir(&input)
            .map_err(|error| error.to_string())?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "fits")
            })
            .collect::<Vec<_>>();
        paths.sort();
        paths
    } else {
        return Err(format!("input path does not exist: {}", input.display()));
    };
    if paths.is_empty() {
        return Err(format!(
            "directory contains no FITS files: {}",
            input.display()
        ));
    }
    let path_strings = paths
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let payload = audit_paths(&path_strings, None)?;
    let frame_count = payload["frames"].as_array().map_or(0, Vec::len);
    let output = write_artifacts(&payload, &output)?;
    println!("auxiliary audit: {}", output.display());
    println!(
        "frames={frame_count} summary={}",
        output.join("auxiliary_boresight_audit.json").display()
    );
    Ok(())
}
