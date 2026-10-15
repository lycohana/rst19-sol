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
    if request["operation"] == "psf.empirical.estimate" {
        return rst19_core::empirical_psf::handle_request(&request);
    }
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
    if request["operation"] == "sequence.stack_forced_measure_batch" {
        return rst19_core::sequence_photometry::handle_request(&request);
    }
    if request["operation"] == "photometry.relative.fit" {
        return rst19_core::relative_photometry::handle_request(&request);
    }
    if request["operation"] == "photometry.report" {
        return rst19_core::photometric_report::build_photometric_report(
            request.get("input").unwrap_or(&Value::Null),
            request.get("options").unwrap_or(&Value::Null),
        );
    }
    if request["operation"] == "sequence.brief" {
        return rst19_core::sequence_brief::handle_request(&request);
    }
    if request["operation"] == "sequence.analyze_core" {
        return rst19_core::sequence_pipeline::handle_request(&request);
    }
    if request["operation"] == "mosaic.build" {
        let args = serde_json::from_value::<Vec<String>>(
            request.get("args").cloned().unwrap_or(json!([])),
        )
        .map_err(|error| error.to_string())?;
        return rst19_core::mosaic::run_desktop(&args);
    }
    if request["operation"] != "tool" {
        return Err(
            "Rust desktop worker only accepts preview, empirical PSF estimation, sensitivity, feedback, or native tools"
                .into(),
        );
    }
    if matches!(
        request["tool"].as_str(),
        Some(
            "rst19-sequence"
                | "rst19-innovation"
                | "rst19-injection"
                | "rst19-pair-audit"
                | "rst19-motion-proposal-ablation"
        )
    ) {
        let mut args = string_args(&request)?;
        let out = request["out_dir"].as_str().map(PathBuf::from);
        if request["tool"] == "rst19-sequence" {
            return rst19_core::sequence_cli::run_desktop(
                &args,
                out.as_deref().unwrap_or(Path::new("tmp/sequence")),
            );
        }
        if let Some(out) = &out {
            if !args
                .iter()
                .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
            {
                args.extend(["--out-dir".into(), out.to_string_lossy().into_owned()]);
            }
        }
        let result = match request["tool"].as_str().unwrap_or("") {
            "rst19-innovation" => rst19_core::innovation_cli::run_desktop(&args),
            "rst19-injection" => rst19_core::injection::run_desktop(&args),
            "rst19-pair-audit" => rst19_core::source_pair_audit::run_desktop(&args),
            "rst19-motion-proposal-ablation" => {
                rst19_core::motion_proposal_ablation::run_desktop(&args)
            }
            _ => unreachable!(),
        }?;
        if let Some(out) = out {
            fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            fs::write(
                out.join("request.json"),
                serde_json::to_vec_pretty(&request).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        }
        let tool = request["tool"].as_str().unwrap_or("");
        let artifact = match tool {
            "rst19-innovation" => Some("innovation_report.json"),
            "rst19-pair-audit" => Some("source_pair_audit.json"),
            "rst19-motion-proposal-ablation" => Some("manifest.json"),
            _ => None,
        };
        let directory = result["output"]
            .as_str()
            .or_else(|| result["run_dir"].as_str())
            .map(PathBuf::from);
        let artifact_path = directory
            .as_ref()
            .zip(artifact)
            .map(|(dir, file)| dir.join(file));
        let payload = if let Some(path) = &artifact_path {
            serde_json::from_slice::<Value>(&fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?
        } else {
            result.clone()
        };
        return Ok(
            json!({"tool":tool,"out_dir":directory,"result_path":artifact_path,"result":payload,"summary":result}),
        );
    }
    if request["tool"] == "rst19-photometric-report" {
        return rst19_core::photometric_report::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-trails" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::single_frame_trail_audit::run_desktop(&args);
    }
    if request["tool"] == "rst19-sources" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::source_audit::run_desktop(&args);
    }
    if request["tool"] == "rst19-empirical-pair-audit" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::empirical_pair_audit::run_desktop(&args);
    }
    if request["tool"] == "rst19-empirical-pair-fit" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::empirical_pair_fit::run_desktop(&args);
    }
    if request["tool"] == "rst19-empirical-pair-free" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::empirical_pair_free::run_desktop(&args);
    }
    if request["tool"] == "rst19-multipsf-audit" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::multipsf_audit::run_desktop(&args);
    }
    if request["tool"] == "rst19-free-multipsf-audit" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::free_multipsf_audit::run_desktop(&args);
    }
    if request["tool"] == "rst19-forced-stability" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::forced_stability::run_desktop(&args);
    }
    if request["tool"] == "rst19-feature-sequence" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::sequence_feature::run_desktop(&args);
    }
    if request["tool"] == "rst19-feature-audit-replicates" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::feature_audit_replicates::run_desktop(&args);
    }
    if request["tool"] == "rst19-gaia" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out" || arg.starts_with("--out="))
        {
            args.push("--out".into());
            args.push(out_dir.join("gaia.csv").to_string_lossy().into_owned());
        }
        return rst19_core::gaia_remote::run_desktop(&args);
    }
    if request["tool"] == "rst19-gaia-tiled" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out" || arg.starts_with("--out="))
        {
            args.push("--out".into());
            args.push(
                out_dir
                    .join("gaia-tiled.csv")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        return rst19_core::gaia_tiled::run_desktop(&args);
    }
    if request["tool"] == "rst19-gaia-frame" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out" || arg.starts_with("--out="))
        {
            args.push("--out".into());
            args.push(
                out_dir
                    .join("gaia-frame.csv")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        return rst19_core::gaia_frame::run_desktop(&args, &out_dir);
    }
    if request["tool"] == "rst19-plate-solver" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out" || arg.starts_with("--out="))
        {
            args.push("--out".into());
            args.push(
                out_dir
                    .join("plate-solution.json")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        return rst19_core::plate_solver::run_desktop(&args, &out_dir);
    }
    if request["tool"] == "rst19-photometric-auto" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out" || arg.starts_with("--out="))
        {
            args.push("--out".into());
            args.push(
                out_dir
                    .join("photometric-auto.json")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        return rst19_core::photometric_workflow::run_desktop(&args, &out_dir);
    }
    if request["tool"] == "rst19-mosaic" {
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
        return rst19_core::mosaic::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-motion-crossing-ablation" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        let mut args = string_args(&request)?;
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        let result = rst19_core::motion_crossing_ablation::run_desktop(&args)?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return Ok(result);
    }
    if request["tool"] == "rst19-motion-ablation" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        let mut args = string_args(&request)?;
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        let result = rst19_core::motion_association_ablation::run_desktop(&args)?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return Ok(result);
    }
    if request["tool"] == "rst19-crowded-ablation" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        let mut args = string_args(&request)?;
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        let result = rst19_core::crowded_pair_ablation::run_desktop(&args)?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return Ok(result);
    }
    if request["tool"] == "rst19-ablation" {
        let out_dir = PathBuf::from(
            request["out_dir"]
                .as_str()
                .ok_or("missing output directory")?,
        );
        fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        let mut args = string_args(&request)?;
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        let result = rst19_core::point_source_ablation::run_desktop(&args)?;
        fs::write(
            out_dir.join("request.json"),
            serde_json::to_vec_pretty(&request).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        return Ok(result);
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
    if request["tool"] == "rst19-pair-support-audit" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        output_path(&mut args, &out_dir);
        return rst19_core::pair_support_audit::run_desktop(&args);
    }
    if request["tool"] == "rst19-feature-parameter-sensitivity" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::feature_parameter_sensitivity::run_desktop(&args);
    }
    if request["tool"] == "rst19-feature-cross" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::feature_cross_audit::run_desktop(&args);
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
    if request["tool"] == "rst19-aperture-sensitivity" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::aperture_sensitivity::run_desktop(&args);
    }
    if request["tool"] == "rst19-spatial-injection" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::spatial_injection::run_desktop(&args);
    }
    if request["tool"] == "rst19-format-audit" {
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
        return rst19_core::format_audit::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-feature-sequence-compare" {
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
        return rst19_core::sequence_feature_comparison::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-feature-raw-evidence-audit" {
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
        return rst19_core::feature_raw_evidence_audit::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-signed-null-overlap" {
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
        return rst19_core::signed_null_forward_overlap::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-signed-null-audit" {
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
        return rst19_core::signed_null::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-signed-null-sequence" {
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
        return rst19_core::signed_null_sequence::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-wcs-validation" {
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
        return rst19_core::wcs_validation::run_desktop(&string_args(&request)?, &out_dir);
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
    if request["tool"] == "rst19-pair-repair-counterfactual" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::pair_repair_counterfactual::run_desktop(&args);
    }
    if request["tool"] == "rst19-pair-response-attribution" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::pair_response_attribution::run_desktop(&args);
    }
    if request["tool"] == "rst19-pair-covariance-audit" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::pair_flux_covariance::run_desktop(&args);
    }
    if request["tool"] == "rst19-contaminated-pair-injection" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::contaminated_pair_injection::run_desktop(&args);
    }
    if request["tool"] == "rst19-contaminated-pair-class" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::contaminated_pair_class::run_desktop(&args);
    }
    if request["tool"] == "rst19-feature-temporal-consistency-audit" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::feature_temporal_consistency_audit::run_desktop(&args);
    }
    if request["tool"] == "rst19-feature-spatial-context" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::feature_spatial_context::run_desktop(&args);
    }
    if request["tool"] == "rst19-feature-evidence" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::feature_evidence_matrix::run_desktop(&args);
    }
    if request["tool"] == "rst19-feature-gate-temporal-route" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::feature_gate_temporal_route::run_desktop(&args);
    }
    if request["tool"] == "rst19-innovation-package" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::innovation_package::run_desktop(&args);
    }
    if request["tool"] == "rst19-temporal-codes" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::temporal_code::run_desktop(&args);
    }
    if request["tool"] == "rst19-sweep" {
        return rst19_core::detection_sweep::run_desktop(&string_args(&request)?);
    }
    if request["tool"] == "rst19-feature-psf-leaveout" {
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
        if !args
            .iter()
            .any(|arg| arg == "--out-dir" || arg.starts_with("--out-dir="))
        {
            args.push("--out-dir".into());
            args.push(out_dir.to_string_lossy().into_owned());
        }
        return rst19_core::feature_psf_leaveout::run_desktop(&args);
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
    if request["tool"] == "rst19-relative-photometry" {
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
        return rst19_core::relative_photometry::run_desktop(&string_args(&request)?, &out_dir);
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

fn catalog() -> Result<Value, String> {
    let value: Value = serde_json::from_str(include_str!("../tool_catalog.json"))
        .map_err(|error| format!("native tool catalog is invalid: {error}"))?;
    let tools = value
        .as_array()
        .ok_or("native tool catalog must be an array")?;
    let native_count = tools.iter().filter(|tool| tool["native"] == true).count();
    Ok(json!({
        "tools":value,
        "environment":{
            "runtime":"Rust",
            "native_tool_count":native_count,
            "total_tool_count":tools.len()
        }
    }))
}

fn main() {
    let cli = std::env::args().skip(1).collect::<Vec<_>>();
    if cli.first().map(String::as_str) == Some("--tool") {
        let Some(name) = cli.get(1) else {
            eprintln!("--tool requires a native tool name");
            std::process::exit(2);
        };
        if cli[2..].iter().any(|arg| arg == "--help" || arg == "-h") {
            match catalog().and_then(|catalog| {
                catalog["tools"]
                    .as_array()
                    .and_then(|tools| {
                        tools
                            .iter()
                            .find(|tool| tool["name"].as_str() == Some(name.as_str()))
                    })
                    .cloned()
                    .ok_or_else(|| format!("unknown native tool {name}"))
            }) {
                Ok(tool) => {
                    println!("{name}: {}", tool["description"].as_str().unwrap_or(""));
                    for field in tool["fields"].as_array().into_iter().flatten() {
                        println!(
                            "{}  {}",
                            field["flag"].as_str().unwrap_or(""),
                            field["help"].as_str().unwrap_or("")
                        );
                    }
                    return;
                }
                Err(error) => {
                    eprintln!("{error}");
                    std::process::exit(2);
                }
            }
        }
        let args = cli[2..].to_vec();
        let out = args
            .iter()
            .position(|arg| arg == "--out-dir")
            .and_then(|i| args.get(i + 1))
            .cloned()
            .or_else(|| {
                args.iter()
                    .find_map(|arg| arg.strip_prefix("--out-dir=").map(str::to_owned))
            })
            .unwrap_or_else(|| format!("tmp/{name}"));
        match run(json!({"operation":"tool","tool":name,"args":args,"out_dir":out})) {
            Ok(payload) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&payload).expect("native tool result")
                );
                return;
            }
            Err(error) => {
                eprintln!("{name}: {error}");
                std::process::exit(2);
            }
        }
    }
    if std::env::args().any(|arg| arg == "--catalog") {
        let (event, exit_code) = match catalog() {
            Ok(payload) => (json!({"event":"done","ok":true,"payload":payload}), 0),
            Err(error) => (json!({"event":"done","ok":false,"error":error}), 1),
        };
        println!("{event}");
        std::process::exit(exit_code);
    }
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
    use super::{catalog, output_path, run};
    use serde_json::json;
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
    };

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn aperture_sensitivity_tool_runs_through_the_native_worker() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let fits_path = workspace
            .join("backend")
            .join("tests")
            .join("fixtures")
            .join("contaminated_pair_injection_parity.fits");
        let root = std::env::temp_dir().join(format!(
            "rst19-native-aperture-sensitivity-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-aperture-sensitivity",
            "args":[fits_path.to_string_lossy(), "--aperture-radii", "3", "4"],
            "out_dir":out_dir
        }))
        .unwrap();

        assert_eq!(result["tool"], "rst19-aperture-sensitivity");
        assert_eq!(result["result"]["baseline_candidate_count"], 3);
        assert_eq!(result["result"]["psf_source_count"], 3);
        assert_eq!(result["result"]["rows"].as_array().unwrap().len(), 8);
        for artifact in [
            "request.json",
            "aperture_sensitivity_audit.csv",
            "aperture_sensitivity_audit.json",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn spatial_injection_tool_runs_through_the_native_worker() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let fits_path = workspace
            .join("backend")
            .join("tests")
            .join("fixtures")
            .join("contaminated_pair_injection_parity.fits");
        let root = std::env::temp_dir().join(format!(
            "rst19-native-spatial-injection-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-spatial-injection",
            "args":[fits_path.to_string_lossy(), "--grid-size", "1", "--peak-levels", "56", "--aperture-radius", "3"],
            "out_dir":out_dir
        }))
        .unwrap();

        assert_eq!(result["tool"], "rst19-spatial-injection");
        assert_eq!(result["result"]["rows"].as_array().unwrap().len(), 1);
        assert_eq!(result["result"]["rows"][0]["injected_count"], 1);
        for artifact in [
            "request.json",
            "spatial_injection_audit.csv",
            "spatial_injection_audit.json",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rst19_trails_runs_through_the_native_worker_and_writes_artifacts() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let fits_path = workspace
            .join("backend")
            .join("tests")
            .join("fixtures")
            .join("contaminated_pair_injection_parity.fits");
        assert!(
            fits_path.is_file(),
            "missing fixture: {}",
            fits_path.display()
        );
        let root = std::env::temp_dir().join(format!(
            "rst19-native-trails-dispatch-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-trails",
            "args":[fits_path.to_string_lossy()],
            "out_dir":out_dir
        }))
        .unwrap();

        assert_eq!(result["frames"].as_array().unwrap().len(), 1);
        let frame = &result["frames"][0];
        assert_eq!(frame["candidate_count"], 21);
        assert_eq!(frame["quality_count"], 3);
        assert_eq!(frame["trail_count"], 0);
        for artifact in [
            "request.json",
            "single_frame_trails.csv",
            "single_frame_trail_counts.png",
            "single_frame_trail_lengths.png",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rst19_sources_runs_through_the_native_worker_and_writes_artifacts() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let fits_path = workspace
            .join("backend")
            .join("tests")
            .join("fixtures")
            .join("single_frame_trail_audit_streak.fits");
        let root = std::env::temp_dir().join(format!(
            "rst19-native-sources-dispatch-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-sources",
            "args":[fits_path.to_string_lossy()],
            "out_dir":out_dir
        }))
        .unwrap();

        assert!(result["candidate_count"].as_u64().is_some());
        for artifact in [
            "request.json",
            "source_catalog.csv",
            "source_feature_psf_similarity.csv",
            "source_feature_psf_spatial.csv",
            "source_cutout_manifest.csv",
            "source_cutout_contact_sheet.png",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empirical_pair_audit_runs_through_the_native_worker_and_writes_artifacts() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let fixtures = workspace.join("backend/tests/fixtures");
        let frames = fixtures.join("empirical_pair_audit_parity/frames");
        let catalog = fixtures.join("source_audit_parity/source_catalog.csv");
        let root = std::env::temp_dir().join(format!(
            "rst19-native-empirical-pair-dispatch-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-empirical-pair-audit",
            "args":[
                frames.to_string_lossy(),
                "--source-catalog",
                catalog.to_string_lossy(),
                "--primary-id",
                "0",
                "--secondary-id",
                "1"
            ],
            "out_dir":out_dir
        }))
        .unwrap();

        assert!(result["frame_count"].as_u64().is_some());
        for artifact in [
            "request.json",
            "empirical_pair_psf_audit.csv",
            "empirical_pair_psf_audit.json",
            "empirical_pair_psf_similarity.png",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empirical_pair_fit_runs_through_the_native_worker_and_writes_artifacts() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let fixtures = workspace.join("backend/tests/fixtures");
        let frames = fixtures.join("empirical_pair_audit_parity/frames");
        let catalog = fixtures.join("source_audit_parity/source_catalog.csv");
        let root = std::env::temp_dir().join(format!(
            "rst19-native-empirical-pair-fit-dispatch-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-empirical-pair-fit",
            "args":[
                frames.to_string_lossy(),
                "--source-catalog",
                catalog.to_string_lossy(),
                "--primary-id",
                "0",
                "--secondary-id",
                "1"
            ],
            "out_dir":out_dir
        }))
        .unwrap();

        assert!(result["frame_count"].as_u64().is_some());
        for artifact in [
            "request.json",
            "empirical_pair_fit.csv",
            "empirical_pair_fit.json",
            "empirical_pair_fit_bic.png",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empirical_pair_free_runs_through_the_native_worker_and_writes_artifacts() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let fixtures = workspace.join("backend/tests/fixtures");
        let frames = fixtures.join("empirical_pair_audit_parity/frames");
        let catalog = fixtures.join("source_audit_parity/source_catalog.csv");
        let root = std::env::temp_dir().join(format!(
            "rst19-native-empirical-pair-free-dispatch-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool", "tool":"rst19-empirical-pair-free",
            "args":[frames.to_string_lossy(),"--source-catalog",catalog.to_string_lossy(),"--primary-id","0","--secondary-id","1"],
            "out_dir":out_dir
        })).unwrap();
        assert!(result["frame_count"].as_u64().is_some());
        for artifact in [
            "request.json",
            "empirical_pair_free.csv",
            "empirical_pair_free.json",
            "empirical_pair_free_bic.png",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn multipsf_audit_runs_through_the_native_worker_and_writes_artifacts() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let frames = workspace.join("backend/tests/fixtures/empirical_pair_audit_parity/frames");
        let root = std::env::temp_dir().join(format!(
            "rst19-native-multipsf-dispatch-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool", "tool":"rst19-multipsf-audit",
            "args":[frames.to_string_lossy(),"--positions","46,48;48.5,48;50,45","--mask-modes","raw"],
            "out_dir":out_dir
        })).unwrap();
        assert_eq!(result["frame_count"], 1);
        for artifact in [
            "request.json",
            "local_multipsf_audit.csv",
            "local_multipsf_audit.json",
            "local_multipsf_bic.png",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
    }

    #[test]
    fn free_multipsf_audit_runs_through_the_native_worker_and_writes_artifacts() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let frames = workspace.join("backend/tests/fixtures/empirical_pair_audit_parity/frames");
        let root = std::env::temp_dir().join(format!(
            "rst19-native-free-multipsf-dispatch-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool", "tool":"rst19-free-multipsf-audit",
            "args":[frames.to_string_lossy(),"--positions","46,48;48.5,48;50,45","--mask-modes","raw","--position-radius","0.75"],
            "out_dir":out_dir
        })).unwrap();
        assert_eq!(result["frame_count"], 1);
        for artifact in [
            "request.json",
            "local_free_multipsf_audit.csv",
            "local_free_multipsf_audit.json",
            "local_free_multipsf_bic.png",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn forced_stability_runs_through_the_native_worker_and_writes_artifacts() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let frame =
            workspace.join("backend/tests/fixtures/empirical_pair_audit_parity/frames/frame.fits");
        let root = std::env::temp_dir().join(format!(
            "rst19-native-forced-stability-dispatch-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let source_catalog = root.join("source_catalog.csv");
        fs::write(&source_catalog, "detection_id,x,y,peak_x,peak_y,flux_snr,feature_class,feature_class_label\n1,46,48,46,48,10,compact_quality,compact\n").unwrap();
        let persistence_json = root.join("persistence.json");
        fs::write(
            &persistence_json,
            serde_json::to_vec(&json!({
                "frame_count":1,
                "cumulative_shifts":[[0.0,0.0]],
                "frame_rows":[{"frame_index":1,"path":frame,"background_adu":100.0,"noise_adu":3.0}]
            }))
            .unwrap(),
        )
        .unwrap();
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool", "tool":"rst19-forced-stability",
            "args":[source_catalog.to_string_lossy(),persistence_json.to_string_lossy(),"--base-dir",root.to_string_lossy(),"--quiet"],
            "out_dir":out_dir
        })).unwrap();
        assert_eq!(result["frame_count"], 1);
        assert_eq!(result["anchor_count"], 1);
        for artifact in [
            "request.json",
            "forced_stability_frame_metrics.csv",
            "forced_stability_summary.csv",
            "forced_stability.json",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn feature_sequence_runs_through_the_native_worker_and_writes_artifacts() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let data_dir = workspace.join("backend/tests/fixtures/empirical_pair_audit_parity/frames");
        let root = std::env::temp_dir().join(format!(
            "rst19-native-feature-sequence-dispatch-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool", "tool":"rst19-feature-sequence",
            "args":[data_dir.to_string_lossy(), "--out-dir", out_dir.to_string_lossy()],
            "out_dir":out_dir
        }))
        .unwrap();
        assert_eq!(result["frame_count"], 1);
        assert_eq!(result["persistence_rows"].as_array().unwrap().len(), 9);
        for artifact in [
            "request.json",
            "sequence_feature_persistence.csv",
            "sequence_feature_frame_summary.csv",
            "sequence_feature_method_frame_summary.csv",
            "sequence_feature_method_profile.csv",
            "sequence_feature_source_subgroup_persistence.csv",
            "sequence_feature_diagnostic_subgroup_persistence.csv",
            "sequence_feature_diagnostic_sources.csv",
            "sequence_feature_temporal_profile.csv",
            "sequence_feature_class_transition.csv",
            "sequence_feature_persistence.json",
            "sequence_feature_persistence.png",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn feature_audit_replicates_runs_through_the_native_worker_and_writes_artifacts() {
        let root = std::env::temp_dir().join(format!(
            "rst19-native-feature-audit-replicates-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool", "tool":"rst19-feature-audit-replicates",
            "args":["--trials", "1", "--seed", "19019"],
            "out_dir":out_dir
        }))
        .unwrap();
        assert_eq!(result["result"]["trial_count"], 1);
        assert_eq!(result["result"]["rows"].as_array().unwrap().len(), 11);
        for artifact in [
            "request.json",
            "feature_audit_replicates.csv",
            "feature_audit_replicates.json",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    fn fits_card(key: &str, value: &str) -> [u8; 80] {
        let mut card = [b' '; 80];
        let text = format!("{key:<8}= {value:>20}");
        card[..text.len()].copy_from_slice(text.as_bytes());
        card
    }

    fn write_single_frame_test_fits(path: &Path) {
        const WIDTH: usize = 128;
        const HEIGHT: usize = 128;
        let mut bytes = Vec::new();
        for (key, value) in [
            ("SIMPLE", "T".to_owned()),
            ("BITPIX", "16".to_owned()),
            ("NAXIS", "2".to_owned()),
            ("NAXIS1", WIDTH.to_string()),
            ("NAXIS2", HEIGHT.to_string()),
        ] {
            bytes.extend_from_slice(&fits_card(key, &value));
        }
        let mut end = [b' '; 80];
        end[..3].copy_from_slice(b"END");
        bytes.extend_from_slice(&end);
        bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let mut value = 1000.0 + ((x * 17 + y * 31) % 9) as f64 * 2.0 - 8.0;
                for (cx, cy, amplitude) in [
                    (30.0, 40.0, 1000.0),
                    (70.0, 80.0, 900.0),
                    (98.0, 50.0, 1100.0),
                ] {
                    let dx = x as f64 - cx;
                    let dy = y as f64 - cy;
                    value += amplitude * (-0.5 * (dx * dx + dy * dy) / 1.4_f64.powi(2)).exp();
                }
                bytes.extend_from_slice(&(value.round() as i16).to_be_bytes());
            }
        }
        bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
        fs::write(path, bytes).unwrap();
    }

    fn add_test_auxiliary(path: &Path, ra_deg: f64, dec_deg: f64) {
        let mut values = [0.0; 26];
        values[0] = 0.0;
        values[1] = 0.0;
        values[2] = ra_deg;
        values[3] = dec_deg;
        values[10] = 1.0;
        let mut raw = Vec::with_capacity(26 * 8);
        for value in values {
            let mut bytes = value.to_le_bytes();
            for pair in bytes.as_chunks_mut::<2>().0 {
                pair.swap(0, 1);
            }
            raw.extend_from_slice(&bytes);
        }
        let mut fits = fs::read(path).unwrap();
        fits[2880..2880 + raw.len()].copy_from_slice(&raw);
        fs::write(path, fits).unwrap();
    }

    fn write_exposure_line_test_fits(path: &Path) {
        let mut bytes = Vec::new();
        for (key, value) in [
            ("SIMPLE", "T"),
            ("BITPIX", "16"),
            ("NAXIS", "2"),
            ("NAXIS1", "256"),
            ("NAXIS2", "256"),
            ("EXPOSURE", "1500"),
        ] {
            bytes.extend_from_slice(&fits_card(key, value));
        }
        let mut end = [b' '; 80];
        end[..3].copy_from_slice(b"END");
        bytes.extend_from_slice(&end);
        bytes.resize(2880, b' ');
        for y in 0..256 {
            for x in 0..256 {
                let background = 1000.0 + ((x * 17 + y * 31) % 9) as f64 * 2.0 - 8.0;
                let line = if (28..=228).contains(&x) {
                    400.0 * (-0.5 * ((y as f64 - 128.0) / 1.25).powi(2)).exp()
                } else {
                    0.0
                };
                bytes.extend_from_slice(&((background + line).round() as i16).to_be_bytes());
            }
        }
        bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn embedded_catalog_marks_rust_and_pending_tools() {
        let catalog = catalog().unwrap();
        let tools = catalog["tools"].as_array().unwrap();
        assert_eq!(tools.len(), catalog["environment"]["total_tool_count"]);
        assert_eq!(
            tools.iter().filter(|tool| tool["native"] == true).count(),
            catalog["environment"]["native_tool_count"]
                .as_u64()
                .unwrap() as usize
        );
        assert!(tools
            .iter()
            .any(|tool| tool["name"] == "rst19-temporal-codes" && tool["native"] == true));
        assert!(tools.iter().any(|tool| {
            tool["name"] == "rst19-empirical-pair-audit" && tool["native"] == true
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-empirical-pair-fit" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-empirical-pair-free" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-multipsf-audit" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-free-multipsf-audit" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-forced-stability" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-feature-sequence" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| tool["name"] == "rst19-sweep" && tool["native"] == true));
        assert!(tools
            .iter()
            .any(|tool| tool["name"] == "rst19" && tool["native"] == true));
        assert!(tools.iter().any(|tool| {
            tool["name"] == "rst19-feature-psf-leaveout" && tool["native"] == true
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-pair-pixel-topology" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-photometric-report" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-pair-support-audit" && tool["native"] == true }));
        assert!(tools.iter().any(|tool| {
            tool["name"] == "rst19-aperture-sensitivity" && tool["native"] == true
        }));
        assert!(tools.iter().any(|tool| {
            tool["name"] == "rst19-feature-parameter-sensitivity" && tool["native"] == true
        }));
        assert!(tools
            .iter()
            .any(|tool| tool["name"] == "rst19-feature-cross" && tool["native"] == true));
        assert!(tools
            .iter()
            .any(|tool| tool["name"] == "rst19-mosaic" && tool["native"] == true));
        assert!(tools.iter().any(|tool| {
            tool["name"] == "rst19-motion-crossing-ablation" && tool["native"] == true
        }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-motion-ablation" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-crowded-ablation" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-ablation" && tool["native"] == true }));
        assert!(tools.iter().any(|tool| {
            tool["name"] == "rst19-feature-audit-replicates" && tool["native"] == true
        }));
        assert!(tools
            .iter()
            .any(|tool| tool["name"] == "rst19-gaia" && tool["native"] == true));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-gaia-tiled" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-gaia-frame" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-plate-solver" && tool["native"] == true }));
        assert!(tools
            .iter()
            .any(|tool| { tool["name"] == "rst19-photometric-auto" && tool["native"] == true }));
        assert!(tools.iter().any(|tool| {
            tool["name"] == "rst19-contaminated-pair-injection" && tool["native"] == true
        }));
        assert!(tools
            .iter()
            .any(|tool| tool["name"] == "rst19-sequence" && tool["native"] == true));
    }

    #[test]
    fn detection_sweep_is_dispatched_to_rust() {
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-sweep",
            "args":[]
        }))
        .unwrap_err();
        assert_eq!(error, "a FITS input path is required");
    }

    #[test]
    fn empirical_psf_operation_is_dispatched_to_rust() {
        let error = run(json!({"operation":"psf.empirical.estimate"})).unwrap_err();
        assert_eq!(error, "missing FITS path");
    }

    #[test]
    fn single_frame_tool_runs_through_the_native_cli_with_python_absent() {
        let sequence = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rst19-native-single-frame-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let fits_path = root.join("single frame.fits");
        let out_dir = root.join("job output");
        write_single_frame_test_fits(&fits_path);
        let source_bytes = fs::read(&fits_path).unwrap();

        let result = run(json!({
            "operation":"tool",
            "tool":"rst19",
            "args":[fits_path.to_string_lossy()],
            "out_dir":out_dir
        }))
        .unwrap();
        let frame = rst19_core::fits::read_fits(&fits_path, false).unwrap();
        let options: rst19_core::pipeline::FrameOptions =
            serde_json::from_value(json!({"proposal_mode":"hybrid"})).unwrap();
        let expected = rst19_core::pipeline::analyze_frame(frame, None, None, &options, None)
            .unwrap()
            .as_dict();
        let mut star_result = result["result"].clone();
        let trails = star_result
            .as_object_mut()
            .unwrap()
            .remove("single_frame_trails")
            .unwrap();
        assert!(trails.get("error").is_none(), "{trails}");
        assert_eq!(trails["frame_count"], 1);
        assert!(trails["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|track| track["classification"] == "candidate"));
        assert_eq!(star_result, expected);
        assert!(
            result["result"]["detection"]["quality_count"]
                .as_u64()
                .unwrap()
                >= 3
        );
        assert!(Path::new(result["preview"].as_str().unwrap()).is_file());
        assert_eq!(fs::read(&fits_path).unwrap(), source_bytes);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn single_frame_tool_exports_visible_exposure_line_and_preserves_input() {
        let sequence = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rst19-exposure-overlay-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let input = root.join("exposure.fits");
        write_exposure_line_test_fits(&input);
        let original = fs::read(&input).unwrap();
        let result = run(
            json!({"operation":"tool","tool":"rst19","args":[input],"out_dir":root.join("result")}),
        )
        .unwrap();
        let trails = &result["result"]["single_frame_trails"];
        assert!(trails.get("error").is_none(), "{trails}");
        let track = trails["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|track| track["points"][0]["display"]["visible"] == true)
            .expect("exposure line must reach the viewport payload");
        assert_eq!(track["classification"], "candidate");
        assert_eq!(track["evidence_level"], "single_frame_trail");
        let point = &track["points"][0];
        assert!(
            (point["y"].as_f64().unwrap() - 128.0).abs() < 2.0,
            "{point}"
        );
        assert!(point["length_px"].as_f64().unwrap() > 180.0, "{point}");
        assert_eq!(point["display"]["artifact_review"], true);
        assert_eq!(point["display"]["solid"], false);
        let saved: serde_json::Value =
            serde_json::from_slice(&fs::read(result["result_path"].as_str().unwrap()).unwrap())
                .unwrap();
        assert_eq!(&saved["single_frame_trails"], trails);
        assert_eq!(fs::read(&input).unwrap(), original);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn contaminated_pair_injection_runs_through_the_native_worker_without_mutating_fits() {
        let sequence = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rst19-native-contaminated-pair-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let fits_path = root.join("frame.fits");
        let out_dir = root.join("output");
        write_single_frame_test_fits(&fits_path);
        let source_bytes = fs::read(&fits_path).unwrap();
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-contaminated-pair-injection",
            "args":[
                fits_path.to_string_lossy(),
                "--anchor", "synthetic,64,112,1,0",
                "--separations-px", "6",
                "--ratios", "1",
                "--total-peak-adu", "2048",
                "--proposal-mode", "gaussian",
                "--analysis-scope", "local_roi",
                "--roi-half-size-px", "20"
            ],
            "out_dir":out_dir
        }))
        .unwrap();
        assert_eq!(result["tool"], "rst19-contaminated-pair-injection");
        assert_eq!(result["result"]["rows"].as_array().unwrap().len(), 1);
        assert_eq!(
            result["result"]["rows"][0]["roi_bounds"],
            json!([44, 92, 85, 128])
        );
        assert!(out_dir.join("contaminated_pair_injection.csv").is_file());
        assert!(out_dir.join("contaminated_pair_injection.json").is_file());
        assert!(out_dir.join("request.json").is_file());
        assert_eq!(fs::read(&fits_path).unwrap(), source_bytes);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn contaminated_pair_class_runs_through_the_native_worker() {
        let sequence = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rst19-native-contaminated-pair-class-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let fits_path = root.join("frame.fits");
        let catalog_path = root.join("source_catalog.csv");
        let out_dir = root.join("output");
        write_single_frame_test_fits(&fits_path);
        fs::write(
            &catalog_path,
            "detection_id,feature_class,feature_class_label,x,y,filter_snr,flux_snr,peak,quality_passed,flags\n1,linear_artifact,line,64,64,20,10,100,false,LINE_ARTIFACT\n",
        )
        .unwrap();
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-contaminated-pair-class",
            "args":[
                fits_path.to_string_lossy(),
                "--source-catalog", catalog_path.to_string_lossy(),
                "--feature-class", "linear_artifact",
                "--separations-px", "6",
                "--ratios", "1",
                "--total-peak-adu", "2048",
                "--analysis-scope", "local_roi",
                "--roi-half-size-px", "20"
            ],
            "out_dir":out_dir
        }))
        .unwrap();
        assert_eq!(result["tool"], "rst19-contaminated-pair-class");
        assert_eq!(result["result"]["selected_rows"][0]["detection_id"], 1);
        assert_eq!(
            result["result"]["injection"]["rows"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            result["result"]["class_summary"][0]["feature_class"],
            "linear_artifact"
        );
        for artifact in [
            "request.json",
            "contaminated_pair_injection.csv",
            "contaminated_pair_class_selection.csv",
            "contaminated_pair_class_summary.csv",
            "contaminated_pair_quality_reason_summary.csv",
            "contaminated_pair_class_audit.json",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mosaic_tool_is_dispatched_to_rust() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let out_dir = std::env::temp_dir().join(format!(
            "rst19-mosaic-dispatch-{}-{nonce}",
            std::process::id()
        ));
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-mosaic",
            "args":[],
            "out_dir":out_dir
        }))
        .unwrap_err();
        assert_eq!(error, "a FITS directory is required");
        let _ = fs::remove_dir_all(out_dir);
    }

    #[test]
    fn motion_crossing_ablation_is_dispatched_to_rust() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let out_dir = std::env::temp_dir().join(format!(
            "rst19-crossing-dispatch-{}-{nonce}",
            std::process::id()
        ));
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-motion-crossing-ablation",
            "args":[],
            "out_dir":out_dir
        }))
        .unwrap();
        assert_eq!(result["status"], "completed");
        let run_dir = PathBuf::from(result["run_dir"].as_str().unwrap());
        for artifact in ["manifest.json", "summary.csv", "track_points.csv"] {
            assert!(run_dir.join(artifact).is_file(), "missing {artifact}");
        }
        assert!(out_dir.join("request.json").is_file());
        fs::remove_dir_all(out_dir).unwrap();
    }

    #[test]
    fn motion_ablation_is_dispatched_to_the_native_rust_parser() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let out_dir = std::env::temp_dir().join(format!(
            "rst19-motion-ablation-native-dispatch-{}-{nonce}",
            std::process::id()
        ));
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-motion-ablation",
            "args":[],
            "out_dir":out_dir
        }))
        .unwrap_err();
        assert_eq!(error, "--cache is required");
        fs::remove_dir_all(out_dir).unwrap();
    }

    #[test]
    fn crowded_ablation_is_dispatched_to_the_native_rust_parser() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let out_dir = std::env::temp_dir().join(format!(
            "rst19-crowded-ablation-native-dispatch-{}-{nonce}",
            std::process::id()
        ));
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-crowded-ablation",
            "args":[],
            "out_dir":out_dir
        }))
        .unwrap_err();
        assert_eq!(error, "FITS input path is required");
        fs::remove_dir_all(out_dir).unwrap();
    }

    #[test]
    fn point_source_ablation_is_dispatched_to_the_native_rust_parser() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let out_dir = std::env::temp_dir().join(format!(
            "rst19-ablation-native-dispatch-{}-{nonce}",
            std::process::id()
        ));
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-ablation",
            "args":[],
            "out_dir":out_dir
        }))
        .unwrap_err();
        assert_eq!(error, "FITS input path is required");
        fs::remove_dir_all(out_dir).unwrap();
    }

    #[test]
    fn photometric_report_is_dispatched_to_rust() {
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-photometric-report",
            "args":[]
        }))
        .unwrap_err();
        assert_eq!(error, "missing photometry JSON input path");
    }

    #[test]
    fn pair_support_audit_is_dispatched_to_rust() {
        let out_dir =
            std::env::temp_dir().join(format!("rst19-pair-support-empty-{}", std::process::id()));
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-pair-support-audit",
            "args":[],
            "out_dir":out_dir
        }))
        .unwrap_err();
        assert_eq!(error, "data_dir is required");
        let _ = fs::remove_dir_all(out_dir);
    }

    #[test]
    fn feature_parameter_sensitivity_is_dispatched_to_rust() {
        let out_dir = std::env::temp_dir().join(format!(
            "rst19-feature-parameter-sensitivity-empty-{}",
            std::process::id()
        ));
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-feature-parameter-sensitivity",
            "args":[],
            "out_dir":out_dir
        }))
        .unwrap_err();
        assert_eq!(error, "a FITS input path is required");
        let _ = fs::remove_dir_all(out_dir);
    }

    #[test]
    fn feature_cross_is_dispatched_to_rust() {
        let out_dir =
            std::env::temp_dir().join(format!("rst19-feature-cross-empty-{}", std::process::id()));
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-feature-cross",
            "args":[],
            "out_dir":out_dir
        }))
        .unwrap_err();
        assert_eq!(error, "expected source summary and persistence CSV paths");
        let _ = fs::remove_dir_all(out_dir);
    }

    #[test]
    fn feature_psf_leaveout_is_dispatched_to_rust() {
        let out_dir =
            std::env::temp_dir().join(format!("rst19-psf-leaveout-empty-{}", std::process::id()));
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-feature-psf-leaveout",
            "args":[],
            "out_dir":out_dir
        }))
        .unwrap_err();
        let _ = fs::remove_dir_all(out_dir);
        assert_eq!(error, "FITS path is required");
    }

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

    #[test]
    fn pair_repair_tool_is_dispatched_to_the_native_audit() {
        let out_dir = std::env::temp_dir().join(format!(
            "rst19-native-pair-repair-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-pair-repair-counterfactual",
            "out_dir":out_dir,
            "args":[]
        }))
        .unwrap_err();
        assert!(error.contains("an input FITS frame is required"), "{error}");
        fs::remove_dir_all(out_dir).unwrap();
    }

    #[test]
    fn pair_response_tool_is_dispatched_to_the_native_audit() {
        let out_dir = std::env::temp_dir().join(format!(
            "rst19-native-pair-response-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-pair-response-attribution",
            "out_dir":out_dir,
            "args":[]
        }))
        .unwrap_err();
        assert!(error.contains("an input FITS frame is required"), "{error}");
        fs::remove_dir_all(out_dir).unwrap();
    }

    #[test]
    fn pair_flux_covariance_tool_is_dispatched_to_the_native_audit() {
        let out_dir = std::env::temp_dir().join(format!(
            "rst19-native-pair-covariance-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let error = run(json!({
            "operation":"tool",
            "tool":"rst19-pair-covariance-audit",
            "out_dir":out_dir,
            "args":[]
        }))
        .unwrap_err();
        assert!(
            error.contains("a forced stability CSV is required"),
            "{error}"
        );
        fs::remove_dir_all(out_dir).unwrap();
    }

    #[test]
    fn feature_temporal_consistency_tool_is_dispatched_to_the_native_audit() {
        let root = std::env::temp_dir().join(format!(
            "rst19-native-temporal-consistency-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        let metrics = root.join("frame-metrics.csv");
        let out_dir = root.join("output");
        fs::write(
            &catalog,
            "detection_id,feature_class,feature_class_label\n1,compact_quality,紧凑\n",
        )
        .unwrap();
        fs::write(
            &metrics,
            "detection_id,frame_index,flux_snr_raw,negative_extreme_count,repeated_code_count,zero_count\n1,1,10,0,0,0\n",
        )
        .unwrap();
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-feature-temporal-consistency-audit",
            "out_dir":out_dir,
            "args":[
                catalog.to_string_lossy(),
                metrics.to_string_lossy(),
                "--expected-frames=1"
            ]
        }))
        .unwrap();
        assert_eq!(result["tool"], "rst19-feature-temporal-consistency-audit");
        assert_eq!(result["result"]["source_rows"].as_array().unwrap().len(), 1);
        for artifact in [
            "request.json",
            "feature_temporal_consistency_audit.json",
            "feature_temporal_consistency_sources.csv",
            "feature_temporal_consistency_summary.csv",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn feature_spatial_context_tool_is_dispatched_to_the_native_audit() {
        let root = std::env::temp_dir().join(format!(
            "rst19-native-feature-spatial-context-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let catalog = root.join("catalog.csv");
        let out_dir = root.join("output");
        fs::write(
            &catalog,
            "detection_id,x,y,feature_class,feature_class_label,quality_passed,flags,flux_snr,filter_snr,peak\n1,20,30,compact_quality,紧凑质量源,False,LOW_FLUX_SNR,12,15,80\n",
        )
        .unwrap();
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-feature-spatial-context",
            "out_dir":out_dir,
            "args":[
                catalog.to_string_lossy(),
                "--image-width=100",
                "--image-height=100",
                "--top-n=1"
            ]
        }))
        .unwrap();
        assert_eq!(result["tool"], "rst19-feature-spatial-context");
        assert_eq!(result["result"]["summaries"].as_array().unwrap().len(), 1);
        for artifact in [
            "request.json",
            "feature_spatial_context.json",
            "feature_spatial_context_summary.csv",
            "feature_spatial_context_hotspots.csv",
            "feature_spatial_context_targets.csv",
        ] {
            assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn feature_evidence_tool_is_dispatched_to_the_native_matrix() {
        let root = std::env::temp_dir().join(format!(
            "rst19-native-feature-evidence-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let morphology = root.join("morphology.csv");
        let cross = root.join("cross.csv");
        let psf = root.join("psf.csv");
        let out_dir = root.join("output");
        fs::write(&morphology, "feature_class\ncompact_quality\n").unwrap();
        fs::write(&cross, "feature_class\ncompact_quality\n").unwrap();
        fs::write(&psf, "feature_class\ncompact_quality\n").unwrap();
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-feature-evidence",
            "out_dir":out_dir,
            "args":[
                "--source-morphology",
                morphology.to_string_lossy(),
                "--feature-cross",
                cross.to_string_lossy(),
                "--psf-leaveout",
                psf.to_string_lossy()
            ]
        }))
        .unwrap();
        assert_eq!(result["tool"], "rst19-feature-evidence");
        assert_eq!(result["result"]["rows"].as_array().unwrap().len(), 9);
        assert!(out_dir.join("request.json").is_file());
        assert!(out_dir.join("feature_evidence_matrix.csv").is_file());
        assert!(out_dir.join("feature_evidence_matrix.json").is_file());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn feature_gate_temporal_route_tool_is_dispatched_to_native_join() {
        let root = std::env::temp_dir().join(format!(
            "rst19-native-feature-gate-temporal-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let route = root.join("route.csv");
        let diagnostic = root.join("diagnostic.csv");
        let persistence = root.join("persistence.json");
        let out_dir = root.join("output");
        fs::write(
            &route,
            "detection_id,feature_class,feature_class_label,quality_passed,route,numeric_failure_reasons,structural_flags,flags\r\n1,crowded_blend,拥挤,False,rejected_structural_only,,UNRESOLVED_BLEND,UNRESOLVED_BLEND\r\n",
        )
        .unwrap();
        fs::write(
            &diagnostic,
            "detection_id,feature_class,feature_class_label,diagnostic_subgroup,diagnostic_subgroup_label,diagnostic_subgroup_definition,candidate_presence,candidate_same_subgroup_presence,quality_presence,quality_same_subgroup_presence\r\n1,crowded_blend,拥挤,blend_unresolved,拥挤,未解析混合,12,11,0,0\r\n",
        )
        .unwrap();
        fs::write(
            &persistence,
            r#"{"frame_count":15,"required_presence":12,"association_radius_px":1.0,"association_method":"reciprocal_nearest_one_to_one"}"#,
        )
        .unwrap();
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-feature-gate-temporal-route",
            "out_dir":out_dir,
            "args":[
                route.to_string_lossy(),
                diagnostic.to_string_lossy(),
                persistence.to_string_lossy(),
                "--target-id",
                "1"
            ]
        }))
        .unwrap();
        assert_eq!(result["tool"], "rst19-feature-gate-temporal-route");
        assert_eq!(result["result"]["source_row_count"], 1);
        assert_eq!(
            result["result"]["target_rows"][0]["all_three_ge_required"],
            false
        );
        assert!(out_dir.join("request.json").is_file());
        assert!(out_dir
            .join("feature_gate_temporal_route_sources.csv")
            .is_file());
        assert!(out_dir.join("feature_gate_temporal_routes.json").is_file());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn innovation_package_tool_is_dispatched_to_native_builder() {
        let root = std::env::temp_dir().join(format!(
            "rst19-native-innovation-package-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let input = root.join("injection.json");
        let out_dir = root.join("output");
        fs::write(&input, r#"{"rows":[{"stratum":"blank","peak_excess_adu":24,"injected_count":8,"candidate_recovered_count":8,"quality_recovered_count":4,"ambiguous_injection_count":0,"unambiguous_injected_count":8,"trial_count":3}]}"#).unwrap();
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-innovation-package",
            "out_dir":out_dir,
            "args":[input.to_string_lossy()]
        }))
        .unwrap();
        assert_eq!(result["tool"], "rst19-innovation-package");
        assert_eq!(result["package"]["status"], "INCOMPLETE");
        assert!(out_dir.join("innovation_package.json").is_file());
        assert!(out_dir.join("innovation_delivery.csv").is_file());
        assert!(out_dir.join("innovation_delivery.md").is_file());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn gaia_cone_query_tool_uses_native_rust_and_dry_run_stays_offline() {
        let root = std::env::temp_dir().join(format!(
            "rst19-native-gaia-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-gaia",
            "out_dir":out_dir,
            "args":["--ra", "12.5", "--dec", "-1.25", "--radius", "0.4", "--dry-run"]
        }))
        .unwrap();
        assert_eq!(result["network"], false);
        assert!(result["query"]
            .as_str()
            .unwrap()
            .contains("CIRCLE('ICRS', 12.5, -1.25, 0.4)"));
        assert!(out_dir.join("request.json").is_file());
    }

    #[test]
    fn gaia_tiled_tool_uses_native_planner_and_dry_run_stays_offline() {
        let root = std::env::temp_dir().join(format!(
            "rst19-native-gaia-tiled-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let out_dir = root.join("output");
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-gaia-tiled",
            "out_dir":out_dir,
            "args":["--ra", "359.8", "--dec", "70", "--radius", "2", "--tile-radius", "0.5", "--dry-run"]
        }))
        .unwrap();
        assert_eq!(result["network"], false);
        assert!(result["initial_tile_count"].as_u64().unwrap() > 1);
        assert!(out_dir.join("request.json").is_file());
    }

    #[test]
    fn gaia_frame_tool_uses_native_fits_auxiliary_and_stays_offline_in_dry_run() {
        let root = std::env::temp_dir().join(format!(
            "rst19-native-gaia-frame-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let fits = root.join("frame.fits");
        let out_dir = root.join("output");
        write_single_frame_test_fits(&fits);
        let mut bytes = fs::read(&fits).unwrap();
        let mut values = [0.0_f64; 26];
        values[0] = 10.0;
        values[1] = 30.0;
        values[2] = 45.5;
        values[3] = -12.25;
        values[10] = 1.0;
        for (index, value) in values.iter().enumerate() {
            let mut raw = value.to_le_bytes();
            for pair in raw.as_chunks_mut::<2>().0 {
                pair.swap(0, 1);
            }
            bytes[2880 + index * 8..2880 + (index + 1) * 8].copy_from_slice(&raw);
        }
        fs::write(&fits, bytes).unwrap();
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-gaia-frame",
            "args":[fits.to_string_lossy(), "--dry-run"],
            "out_dir":out_dir
        }))
        .unwrap();
        assert_eq!(result["dry_run"], true);
        assert_eq!(result["center_ra_deg"], 45.5);
        assert_eq!(result["center_dec_deg"], -12.25);
        assert_eq!(result["plan"]["network"], false);
        assert!(out_dir.join("request.json").is_file());
    }

    #[test]
    fn plate_solver_tool_uses_native_fits_catalog_and_writes_solution_json() {
        let root = std::env::temp_dir().join(format!(
            "rst19-native-plate-solver-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let fits = root.join("frame.fits");
        let catalog = root.join("catalog.csv");
        let out_dir = root.join("output");
        write_single_frame_test_fits(&fits);
        fs::write(
            &catalog,
            "source_id,ra_deg,dec_deg,magnitude\nstar-a,10.0,20.0,8.5\nstar-b,10.02,20.01,9.0\n",
        )
        .unwrap();
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-plate-solver",
            "args":[
                fits.to_string_lossy(),
                "--catalog", catalog.to_string_lossy(),
                "--center-ra", "10",
                "--center-dec", "20",
                "--pixel-scale-arcsec", "8.5",
                "--out", out_dir.join("solution.json").to_string_lossy()
            ],
            "out_dir":out_dir
        }))
        .unwrap();
        assert!(result["plate_solution"]["valid"].is_boolean());
        assert!(out_dir.join("request.json").is_file());
        assert!(out_dir.join("solution.json").is_file());
    }

    #[test]
    fn automatic_photometry_tool_dispatches_to_rust_and_writes_evidence() {
        let root = std::env::temp_dir().join(format!(
            "rst19-native-photometric-auto-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let fits = root.join("frame.fits");
        let catalog = root.join("catalog.csv");
        let out_dir = root.join("output");
        write_single_frame_test_fits(&fits);
        add_test_auxiliary(&fits, 10.0, 20.0);
        fs::write(
            &catalog,
            "source_id,ra_deg,dec_deg,magnitude\nstar-a,10.0,20.0,8.5\nstar-b,10.02,20.01,9.0\n",
        )
        .unwrap();
        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-photometric-auto",
            "args":[fits.to_string_lossy(), "--catalog", catalog.to_string_lossy()],
            "out_dir":out_dir
        }))
        .unwrap();
        assert!(result["status"].as_str().unwrap().starts_with("WCS_"));
        assert_eq!(result["calibrated"], false);
        assert_eq!(result["plate_solution"]["valid"], false);
        assert_eq!(result["catalog_source_count"], 2);
        assert!(out_dir.join("request.json").is_file());
        assert!(out_dir.join("photometric-auto.json").is_file());
    }

    #[test]
    fn temporal_code_tool_is_dispatched_to_native_audit() {
        let root = std::env::temp_dir().join(format!(
            "rst19-native-temporal-code-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let input_dir = root.join("frames");
        let out_dir = root.join("output");
        fs::create_dir_all(&input_dir).unwrap();
        let mut bytes = Vec::new();
        for (key, value) in [
            ("SIMPLE", "T"),
            ("BITPIX", "16"),
            ("NAXIS", "2"),
            ("NAXIS1", "2"),
            ("NAXIS2", "2"),
        ] {
            let mut card = [b' '; 80];
            let content = format!("{key:<8}= {value:>20}");
            card[..content.len()].copy_from_slice(content.as_bytes());
            bytes.extend_from_slice(&card);
        }
        let mut end = [b' '; 80];
        end[..3].copy_from_slice(b"END");
        bytes.extend_from_slice(&end);
        bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
        for value in [0_i16, 3991, 3992, -1] {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
        fs::write(input_dir.join("frame.fits"), bytes).unwrap();

        let result = run(json!({
            "operation":"tool",
            "tool":"rst19-temporal-codes",
            "out_dir":out_dir,
            "args":[input_dir.to_string_lossy()]
        }))
        .unwrap();
        assert_eq!(result["tool"], "rst19-temporal-codes");
        assert_eq!(result["result"]["frame_count"], 1);
        assert_eq!(result["result"]["exact_stable_pixel_count"], 4);
        assert!(out_dir.join("request.json").is_file());
        assert!(out_dir.join("temporal_code_audit.json").is_file());
        assert!(out_dir.join("temporal_code_focus_series.csv").is_file());
        assert!(out_dir.join("temporal_code_components.csv").is_file());
        fs::remove_dir_all(root).unwrap();
    }
}
