#[path = "../native_preview.rs"]
mod native_preview;

use serde_json::{json, Value};
use std::{
    fs,
    io::{self, BufRead},
    path::{Path, PathBuf},
};

fn output_path(args: &mut Vec<String>, out_dir: &Path) -> PathBuf {
    if let Some(index) = args.iter().position(|arg| arg == "--json-out") {
        if let Some(path) = args.get(index + 1) {
            return PathBuf::from(path);
        }
    }
    if let Some(path) = args.iter().find_map(|arg| arg.strip_prefix("--json-out=")) {
        return PathBuf::from(path);
    }
    let path = out_dir.join("result.json");
    args.push("--json-out".into());
    args.push(path.to_string_lossy().into_owned());
    path
}

fn string_args(request: &Value) -> Result<Vec<String>, String> {
    request["args"]
        .as_array()
        .ok_or("args must be a string array")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "args must be a string array".to_owned())
        })
        .collect()
}

fn run(request: Value) -> Result<Value, String> {
    if request["operation"] == "feedback.append" {
        let path = PathBuf::from(request["path"].as_str().ok_or("missing path")?);
        let payload = request.get("feedback").ok_or("missing feedback")?;
        let feedback = rst19_core::feedback::from_value(payload)?;
        let written = rst19_core::feedback::append_manual_feedback(&path, &feedback)?;
        return Ok(json!({"path":written, "record":feedback.as_dict()}));
    }
    if request["operation"] == "feedback.load" {
        let path = PathBuf::from(request["path"].as_str().ok_or("missing path")?);
        let records = rst19_core::feedback::load_manual_feedback(&path)?;
        return Ok(
            json!({"records":records.iter().map(|record| record.as_dict()).collect::<Vec<_>>()}),
        );
    }
    if request["operation"] == "preview" {
        let path = PathBuf::from(request["path"].as_str().ok_or("missing FITS path")?);
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        return native_preview::render_preview(
            &path,
            &out_dir,
            request["mode"].as_str().unwrap_or("raw"),
        );
    }
    if request["operation"] == "photometry.calibration.sensitivity" {
        let matches: Vec<rst19_core::matching::CatalogMatch> =
            serde_json::from_value(request["matches"].clone()).map_err(|e| e.to_string())?;
        let detections: Vec<rst19_core::measurement::Detection> =
            serde_json::from_value(request["detections"].clone()).map_err(|e| e.to_string())?;
        let catalog: Vec<rst19_core::catalog::CatalogSource> =
            serde_json::from_value(request["catalog"].clone()).map_err(|e| e.to_string())?;
        let options: rst19_core::calibration_sensitivity::SensitivityOptions =
            serde_json::from_value(request.get("options").cloned().unwrap_or(json!({})))
                .map_err(|e| e.to_string())?;
        return rst19_core::calibration_sensitivity::run(&matches, &detections, &catalog, &options);
    }
    if request["operation"] != "tool" {
        return Err(
            "Rust desktop worker only accepts preview, sensitivity, feedback, or native tools"
                .into(),
        );
    }
    if request["tool"] == "rst19-feature-flag-profile" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::feature_flag_profile::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-feature-flag-interaction" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::feature_flag_interaction::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-feature-gate-margin" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::feature_gate_margin::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-feature-effect-size" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::feature_effect_size::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-feature-class-context" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::feature_class_context::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-pair-mechanism-context" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::pair_mechanism_context::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-pair-mechanism-sensitivity" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::pair_mechanism_sensitivity::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-pair-pixel-topology" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::pair_pixel_topology::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-aperture-peak-ratio" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::aperture_peak_ratio::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-feature-correlation" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::feature_correlation_audit::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-feature-transition-flow" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::feature_transition_flow::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-feature-gate-route" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::feature_gate_route::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-feature-hard-negative" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::feature_hard_negative::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-gaia-merge" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::catalog_merge::run_desktop(&string_args(&request)?, &out_dir);
    }
    if request["tool"] == "rst19-feature-evidence-axes" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::feature_evidence_axes::run_desktop(&string_args(&request)?, &out_dir);
    }
    if request["tool"] == "rst19-direct-neighbor-audit" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::direct_neighbor_audit::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-feature-cross-axis-audit" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::feature_cross_axis_audit::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-source-group-audit" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::source_group_audit::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-source-group-sensitivity" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::source_group_sensitivity::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-source-separation-audit" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::source_separation_audit::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-source-peak-consistency" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::source_peak_consistency::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-source-proposal-peak-audit" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::source_proposal_peak_audit::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-source-proposal-temporal-audit" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::source_proposal_temporal_audit::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-source-geometry-audit" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::source_geometry_audit::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-aux-audit" {
        return rst19_core::attitude::run_desktop(&string_args(&request)?);
    }
    if request["tool"] != "rst19" {
        return Err("Rust desktop worker does not support this tool".into());
    }
    let out_dir = PathBuf::from(
        request["out_dir"]
            .as_str()
            .ok_or("missing output directory")?,
    );
    fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
    fs::write(
        out_dir.join("request.json"),
        serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let mut args = string_args(&request)?;
    let input_path = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .map(PathBuf::from)
        .ok_or("missing FITS path")?;
    let result_path = output_path(&mut args, &out_dir);
    rst19_core::cli::run(&args)?;
    let result: Value =
        serde_json::from_slice(&fs::read(&result_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("Rust single-frame result is invalid JSON: {error}"))?;
    let mode = request["mode"].as_str().unwrap_or("raw");
    let preview = native_preview::render_preview(&input_path, &out_dir, mode)?;
    let mut payload = json!({
        "tool": "rst19",
        "out_dir": out_dir,
        "result_path": result_path,
        "result": result,
    });
    if let (Some(target), Some(source)) = (payload.as_object_mut(), preview.as_object()) {
        target.extend(
            source
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    Ok(payload)
}

fn main() {
    let request = io::stdin()
        .lock()
        .lines()
        .next()
        .ok_or_else(|| "missing JSON request".to_owned())
        .and_then(|line| line.map_err(|error| error.to_string()))
        .and_then(|line| serde_json::from_str(&line).map_err(|error| error.to_string()))
        .and_then(run);
    let (event, exit_code) = match request {
        Ok(payload) => (json!({"event":"done","ok":true,"payload":payload}), 0),
        Err(error) => (json!({"event":"done","ok":false,"error":error}), 1),
    };
    println!("{event}");
    std::process::exit(exit_code);
}

#[cfg(test)]
mod tests {
    use super::output_path;
    use std::path::Path;

    #[test]
    fn json_output_path_is_added_or_respected() {
        let mut args = vec!["frame.fits".into()];
        assert_eq!(
            output_path(&mut args, Path::new("job")),
            Path::new("job/result.json")
        );
        assert_eq!(args[1], "--json-out");

        let mut args = vec![
            "frame.fits".into(),
            "--json-out".into(),
            "custom.json".into(),
        ];
        assert_eq!(
            output_path(&mut args, Path::new("job")),
            Path::new("custom.json")
        );
    }
}
