//! Native sequence command, including original-frame audit export.
use crate::{sequence_pipeline, sequence_source_audit, Result};
use serde_json::{json, Map, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const HELP: &str = "检测一组 FITS 并区分稳定星点、运动目标和瞬态候选\n\nUsage: rst19-sequence DATA_DIR [OPTIONS]\n\nOptions: --threshold-sigma --min-distance --aperture-radius --psf-fwhm\n --proposal-mode --temporal-proposal-mode --min-flux-snr --min-psf-support-pixels\n --link-radius-px --min-presence --persistent-min-presence --motion-min-displacement-px\n --motion-line-link-radius-px --max-motion-fit-rms-px --source-limit --background-box-size\n --workers --candidate-consensus-min-snr --temporal-candidate-min-snr --temporal-reference-min-snr\n --temporal-multiscale --temporal-min-psf-correlation --no-stack-faint --stack-reference-mode\n --stack-threshold-sigma --stack-min-flux-snr --stack-frame-min-flux-snr --stack-min-presence\n --float64 --relative-photometry --relative-max-sources --relative-spatial-order\n --relative-validation-fraction --keep-linear-artifacts --json-out --audit-out-dir";

pub struct SequenceCommand {
    pub data_dir: PathBuf,
    pub request: Value,
    pub json_out: Option<PathBuf>,
    pub audit_out_dir: Option<PathBuf>,
}

pub fn parse_args(args: &[String]) -> Result<SequenceCommand> {
    let mut request = Map::new();
    let mut frame = json!({"threshold_sigma":4.0,"min_distance":4,"aperture_radius":4,"psf_fwhm":2.0,"proposal_mode":"hybrid","min_flux_snr":5.0,"min_psf_support_pixels":3,"background_box_size":256});
    let mut tracking =
        json!({"link_radius_px":4.0,"motion_min_displacement_px":2.0,"max_motion_fit_rms_px":0.75});
    let mut data_dir = None;
    let mut json_out = None;
    let mut audit_out = None;
    let mut positional = false;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--" && !positional {
            positional = true;
            index += 1;
            continue;
        }
        if positional || !arg.starts_with('-') {
            if data_dir.replace(PathBuf::from(arg)).is_some() {
                return Err("only one sequence directory is allowed".into());
            }
            index += 1;
            continue;
        }
        let raw = arg
            .strip_prefix("--")
            .ok_or_else(|| format!("unknown sequence argument {arg}"))?;
        let (name, inline) = raw
            .split_once('=')
            .map_or((raw, None), |(name, value)| (name, Some(value)));
        let flag = matches!(
            name,
            "temporal-multiscale"
                | "no-stack-faint"
                | "float64"
                | "relative-photometry"
                | "keep-linear-artifacts"
        );
        if flag {
            if inline.is_some() {
                return Err(format!("--{name} does not take a value"));
            }
            match name {
                "temporal-multiscale" => {
                    request.insert("temporal_multiscale".into(), json!(true));
                }
                "no-stack-faint" => {
                    request.insert("stack_faint_recovery".into(), json!(false));
                }
                "float64" => {
                    frame["use_float32"] = json!(false);
                    frame["fast_sequence"] = json!(false);
                }
                "relative-photometry" => {
                    request.insert("relative_photometry".into(), json!(true));
                }
                _ => {
                    frame["reject_linear_artifacts"] = json!(false);
                }
            }
            index += 1;
            continue;
        }
        let integer = matches!(
            name,
            "min-distance"
                | "aperture-radius"
                | "min-psf-support-pixels"
                | "min-presence"
                | "persistent-min-presence"
                | "source-limit"
                | "background-box-size"
                | "workers"
                | "stack-min-presence"
                | "relative-max-sources"
                | "relative-spatial-order"
        );
        let text = matches!(
            name,
            "proposal-mode"
                | "temporal-proposal-mode"
                | "stack-reference-mode"
                | "json-out"
                | "audit-out-dir"
        );
        let float = matches!(
            name,
            "threshold-sigma"
                | "psf-fwhm"
                | "min-flux-snr"
                | "link-radius-px"
                | "motion-min-displacement-px"
                | "motion-line-link-radius-px"
                | "max-motion-fit-rms-px"
                | "candidate-consensus-min-snr"
                | "temporal-candidate-min-snr"
                | "temporal-reference-min-snr"
                | "temporal-min-psf-correlation"
                | "stack-threshold-sigma"
                | "stack-min-flux-snr"
                | "stack-frame-min-flux-snr"
                | "relative-validation-fraction"
        );
        if !integer && !text && !float {
            return Err(format!("unknown sequence option --{name}"));
        }
        let raw_value = if let Some(value) = inline {
            value
        } else {
            index += 1;
            args.get(index)
                .ok_or_else(|| format!("--{name} requires a value"))?
        };
        let value = if integer {
            let number = raw_value
                .parse::<i64>()
                .map_err(|_| format!("invalid integer for --{name}"))?;
            if number < 0
                || (number == 0 && name != "source-limit" && name != "relative-spatial-order")
            {
                return Err(format!("--{name} must be positive"));
            }
            if name == "relative-spatial-order" && number > 2 {
                return Err("--relative-spatial-order must be 0, 1, or 2".into());
            }
            json!(number)
        } else if float {
            let number = raw_value
                .parse::<f64>()
                .map_err(|_| format!("invalid float for --{name}"))?;
            if !number.is_finite() {
                return Err(format!("--{name} must be finite"));
            }
            json!(number)
        } else {
            json!(raw_value)
        };
        let key = name.replace('-', "_");
        match name {
            "json-out" => {
                json_out = Some(PathBuf::from(raw_value));
            }
            "audit-out-dir" => {
                audit_out = Some(PathBuf::from(raw_value));
            }
            "threshold-sigma"
            | "min-distance"
            | "aperture-radius"
            | "psf-fwhm"
            | "proposal-mode"
            | "min-flux-snr"
            | "min-psf-support-pixels"
            | "background-box-size" => {
                frame[&key] = value;
            }
            "link-radius-px"
            | "min-presence"
            | "persistent-min-presence"
            | "motion-min-displacement-px"
            | "max-motion-fit-rms-px" => {
                tracking[&key] = value;
            }
            "source-limit" => {
                request.insert(
                    "sequence_max_sources".into(),
                    if value == 0 { Value::Null } else { value },
                );
            }
            "relative-max-sources" => {
                request.insert("relative_photometry_max_sources".into(), value);
            }
            "relative-spatial-order" => {
                request.insert("relative_photometry_spatial_order".into(), value);
            }
            "relative-validation-fraction" => {
                request.insert("relative_photometry_validation_fraction".into(), value);
            }
            // Worker count affects scheduling, not the numerical contract.
            "workers" => {
                request.insert("sequence_workers".into(), value);
            }
            _ => {
                request.insert(key, value);
            }
        }
        index += 1;
    }
    request.insert("frame_options".into(), frame);
    request.insert("tracking_options".into(), tracking);
    Ok(SequenceCommand {
        data_dir: data_dir.ok_or("missing sequence data directory")?,
        request: Value::Object(request),
        json_out,
        audit_out_dir: audit_out,
    })
}

pub fn analyze_command(command: &SequenceCommand) -> Result<Value> {
    let mut paths = fs::read_dir(&command.data_dir)
        .map_err(|error| error.to_string())?
        .map(|entry| entry.map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        if cfg!(windows) {
                            extension.eq_ignore_ascii_case("fits")
                        } else {
                            extension == "fits"
                        }
                    })
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "数据目录没有 FITS 文件：{}",
            command.data_dir.display()
        ));
    }
    let mut request = command.request.clone();
    request["paths"] = json!(paths);
    let result = sequence_pipeline::handle_request(&request)?["sequence_result"].clone();
    eprintln!("98.5% · 写入序列分析结果与源级审计文件");
    if let Some(path) = &command.json_out {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let mut bytes = serde_json::to_vec_pretty(&result).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
        fs::write(path, bytes).map_err(|error| error.to_string())?;
    }
    if let Some(path) = &command.audit_out_dir {
        sequence_source_audit::write_sequence_source_audit(&result, path, 6, 10, 2)?;
    }
    eprintln!("99.0% · 序列结果与审计文件已保存");
    Ok(result)
}

pub fn run(args: &[String]) -> Result<()> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{HELP}");
        return Ok(());
    }
    let command = parse_args(args)?;
    let result = analyze_command(&command)?;
    if command.json_out.is_none() {
        println!(
            "{}",
            serde_json::to_string_pretty(&result).map_err(|error| error.to_string())?
        );
    }
    Ok(())
}

pub fn run_desktop(args: &[String], output: &Path) -> Result<Value> {
    let mut command = parse_args(args)?;
    if command.json_out.is_none() {
        command.json_out = Some(output.join("result.json"));
    }
    let result = analyze_command(&command)?;
    Ok(
        json!({"tool":"rst19-sequence","out_dir":output,"output":output,"result_path":command.json_out,"result":result}),
    )
}
