//! Multi-frame aggregation for the source-level sign-flipped negative control.
use crate::{
    fits,
    signed_null::{self, Options},
    statistics, Result,
};
use csv::Terminator;
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const CONCLUSION: &str = "15 帧结果同时保留逐帧和 pooled 计数；pooled 比值只是计数聚合后的 signed-tail leakage，不等于 15 帧真实 precision、FDR、恒星完备率或运动目标概率。值域转移、热像素、宇宙线、亮线和跨帧配准结构必须继续按类别和独立轨迹证据审计。";
const RECURRENCE_DEFINITION: &str = "connected components of measured centroids from different frames within 1.0 px; diagnostic recurrence only, not physical identity";

struct CliArgs {
    inputs: Vec<PathBuf>,
    out_dir: PathBuf,
    options: Options,
}

fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_args(args: &[String]) -> Result<CliArgs> {
    let mut inputs = Vec::new();
    let mut out_dir = PathBuf::from("tmp/signed-null-sequence");
    let mut options = Options::default();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--keep-linear-artifacts" {
            options.reject_linear_artifacts = false;
        } else if arg == "--" {
            inputs.extend(args[index + 1..].iter().map(PathBuf::from));
            break;
        } else if arg == "--out-dir" || arg.starts_with("--out-dir=") {
            out_dir = PathBuf::from(if let Some(value) = arg.strip_prefix("--out-dir=") {
                value.to_owned()
            } else {
                take_value(args, &mut index, "--out-dir")?
            });
        } else if arg == "--snr-thresholds" {
            options.snr_thresholds.clear();
            index += 1;
            while index < args.len() && !args[index].starts_with('-') {
                options.snr_thresholds.push(
                    args[index]
                        .parse()
                        .map_err(|_| "invalid --snr-thresholds value")?,
                );
                index += 1;
            }
            index -= 1;
        } else if let Some(value) = arg.strip_prefix("--snr-thresholds=") {
            options.snr_thresholds = value
                .split(',')
                .map(|part| {
                    part.parse()
                        .map_err(|_| "invalid --snr-thresholds value".to_owned())
                })
                .collect::<Result<Vec<f64>>>()?;
        } else if arg == "--threshold-sigma" || arg.starts_with("--threshold-sigma=") {
            let value = option_value(args, &mut index, arg, "--threshold-sigma")?;
            options.threshold_sigma = value.parse().map_err(|_| "invalid --threshold-sigma")?;
        } else if arg == "--min-distance" || arg.starts_with("--min-distance=") {
            let value = option_value(args, &mut index, arg, "--min-distance")?;
            options.min_distance = value.parse().map_err(|_| "invalid --min-distance")?;
        } else if arg == "--aperture-radius" || arg.starts_with("--aperture-radius=") {
            let value = option_value(args, &mut index, arg, "--aperture-radius")?;
            options.aperture_radius = value.parse().map_err(|_| "invalid --aperture-radius")?;
        } else if arg == "--psf-fwhm" || arg.starts_with("--psf-fwhm=") {
            let value = option_value(args, &mut index, arg, "--psf-fwhm")?;
            options.psf_fwhm = value.parse().map_err(|_| "invalid --psf-fwhm")?;
        } else if arg == "--proposal-mode" || arg.starts_with("--proposal-mode=") {
            options.proposal_mode = option_value(args, &mut index, arg, "--proposal-mode")?;
        } else if arg == "--background-box-size" || arg.starts_with("--background-box-size=") {
            let value = option_value(args, &mut index, arg, "--background-box-size")?;
            options.background_box_size =
                value.parse().map_err(|_| "invalid --background-box-size")?;
        } else if arg == "--min-flux-snr" || arg.starts_with("--min-flux-snr=") {
            let value = option_value(args, &mut index, arg, "--min-flux-snr")?;
            options.min_flux_snr = value.parse().map_err(|_| "invalid --min-flux-snr")?;
        } else if arg == "--min-psf-support-pixels" || arg.starts_with("--min-psf-support-pixels=")
        {
            let value = option_value(args, &mut index, arg, "--min-psf-support-pixels")?;
            options.min_psf_support_pixels = value
                .parse()
                .map_err(|_| "invalid --min-psf-support-pixels")?;
        } else if arg.starts_with('-') {
            return Err(format!("unrecognized argument: {arg}"));
        } else {
            inputs.push(PathBuf::from(arg));
        }
        index += 1;
    }
    Ok(CliArgs {
        inputs,
        out_dir,
        options,
    })
}

fn option_value(args: &[String], index: &mut usize, arg: &str, flag: &str) -> Result<String> {
    if let Some(value) = arg.strip_prefix(&format!("{flag}=")) {
        Ok(value.to_owned())
    } else {
        take_value(args, index, flag)
    }
}

pub fn collect_paths(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut collected = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let mut children = fs::read_dir(input)
                .map_err(|error| {
                    format!("cannot read FITS directory {}: {error}", input.display())
                })?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| {
                    path.is_file()
                        && path
                            .extension()
                            .is_some_and(|extension| extension == "fits" || extension == "FITS")
                })
                .collect::<Vec<_>>();
            children.sort_by_key(|path| file_name(path));
            collected.extend(children);
        } else {
            collected.push(input.clone());
        }
    }
    let mut unique = BTreeMap::<String, PathBuf>::new();
    for path in collected {
        let identity = path
            .canonicalize()
            .map(|canonical| canonical.to_string_lossy().to_lowercase())
            .unwrap_or_else(|_| path.to_string_lossy().to_lowercase());
        unique.insert(identity, path);
    }
    let mut paths = unique.into_values().collect::<Vec<_>>();
    paths.sort_by_key(|path| file_name(path));
    Ok(paths)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

fn add_index_path(row: &Value, frame_index: usize, frame_path: &str) -> Result<Value> {
    let mut object = row
        .as_object()
        .cloned()
        .ok_or("signed-null audit row is not an object")?;
    object.insert("frame_index".into(), json!(frame_index));
    object.insert("frame_path".into(), json!(frame_path));
    Ok(Value::Object(object))
}

fn number(value: &Value, key: &str) -> Result<usize> {
    value[key]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("missing or invalid count {key}"))
}

fn safe_ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn finite_values(rows: &[Value], key: &str) -> Vec<f64> {
    rows.iter()
        .filter_map(|row| row[key].as_f64())
        .filter(|value| value.is_finite())
        .collect()
}

fn optional_summary(values: &[Value], key: &str, median: bool) -> Result<Option<f64>> {
    let samples = finite_values(values, key);
    if samples.is_empty() {
        return Ok(None);
    }
    if median {
        Ok(Some(statistics::median(&samples)?))
    } else {
        Ok(Some(samples.iter().sum::<f64>() / samples.len() as f64))
    }
}

fn find(parent: &mut [usize], index: usize) -> usize {
    if parent[index] != index {
        parent[index] = find(parent, parent[index]);
    }
    parent[index]
}

fn recurrence_rows(rows: &[Value]) -> Result<Vec<Value>> {
    let mut parent = (0..rows.len()).collect::<Vec<_>>();
    for left in 0..rows.len() {
        for right in left + 1..rows.len() {
            if rows[left]["frame_index"] == rows[right]["frame_index"] {
                continue;
            }
            let dx = rows[left]["x"].as_f64().unwrap_or(f64::NAN)
                - rows[right]["x"].as_f64().unwrap_or(f64::NAN);
            let dy = rows[left]["y"].as_f64().unwrap_or(f64::NAN)
                - rows[right]["y"].as_f64().unwrap_or(f64::NAN);
            if dx.hypot(dy) <= 1.0 {
                let left_root = find(&mut parent, left);
                let right_root = find(&mut parent, right);
                if left_root != right_root {
                    parent[right_root] = left_root;
                }
            }
        }
    }
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..rows.len() {
        let root = find(&mut parent, index);
        groups.entry(root).or_default().push(index);
    }
    let mut recurrent = groups
        .into_values()
        .filter(|indices| {
            indices
                .iter()
                .map(|index| rows[*index]["frame_index"].as_u64().unwrap_or_default())
                .collect::<BTreeSet<_>>()
                .len()
                >= 2
        })
        .collect::<Vec<_>>();
    recurrent.sort_by(|left, right| {
        let min_index = |group: &[usize]| {
            group
                .iter()
                .map(|index| rows[*index]["frame_index"].as_u64().unwrap_or_default())
                .min()
                .unwrap_or_default()
        };
        let median_x = |group: &[usize]| {
            let values = group
                .iter()
                .map(|index| rows[*index]["x"].as_f64().unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            statistics::median(&values).unwrap_or(f64::NAN)
        };
        min_index(left)
            .cmp(&min_index(right))
            .then_with(|| median_x(left).total_cmp(&median_x(right)))
    });
    let mut output = Vec::with_capacity(recurrent.len());
    for (zero_id, indices) in recurrent.into_iter().enumerate() {
        let mut frame_indices = indices
            .iter()
            .map(|index| rows[*index]["frame_index"].as_u64().unwrap_or_default() as usize)
            .collect::<Vec<_>>();
        frame_indices.sort_unstable();
        frame_indices.dedup();
        let xs = indices
            .iter()
            .map(|index| rows[*index]["x"].as_f64().unwrap_or(f64::NAN))
            .collect::<Vec<_>>();
        let ys = indices
            .iter()
            .map(|index| rows[*index]["y"].as_f64().unwrap_or(f64::NAN))
            .collect::<Vec<_>>();
        let x_median = statistics::median(&xs)?;
        let y_median = statistics::median(&ys)?;
        let max_radius = indices
            .iter()
            .map(|index| {
                (rows[*index]["x"].as_f64().unwrap_or(f64::NAN) - x_median)
                    .hypot(rows[*index]["y"].as_f64().unwrap_or(f64::NAN) - y_median)
            })
            .fold(0.0f64, f64::max);
        let flux = indices
            .iter()
            .filter_map(|index| rows[*index]["flux_snr"].as_f64())
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let transfer_count = indices
            .iter()
            .filter(|index| rows[**index]["range_transfer_overlap"] == Value::Bool(true))
            .count();
        let frame_label = frame_indices
            .iter()
            .map(|value| format!("{value:02}"))
            .collect::<Vec<_>>()
            .join(",");
        output.push(json!({
            "cluster_id":zero_id + 1,
            "frame_indices":frame_label,
            "frame_count":frame_indices.len(),
            "source_count":indices.len(),
            "x_median":x_median,
            "y_median":y_median,
            "max_radius_px":max_radius,
            "transfer_count":transfer_count,
            "effective_count":indices.len().saturating_sub(transfer_count),
            "median_flux_snr":if flux.is_empty() { None } else { Some(statistics::median(&flux)?) },
        }));
    }
    Ok(output)
}

fn run(paths: &[PathBuf], options: &Options) -> Result<Value> {
    if paths.is_empty() {
        return Err("fits_paths must not be empty".into());
    }
    let total = paths.len();
    let mut frame_rows = Vec::with_capacity(total);
    let mut threshold_rows = Vec::new();
    let mut feature_rows = Vec::new();
    let mut quality_source_rows = Vec::new();
    let mut audits = Vec::with_capacity(total);
    for (zero_index, path) in paths.iter().enumerate() {
        let frame = fits::read_fits(path, false)?;
        let timestamp = frame.header.get("DATE-OBS").cloned().unwrap_or(Value::Null);
        let timestamp = match timestamp {
            Value::Null => Value::Null,
            Value::String(value) => Value::String(value),
            value => Value::String(value.to_string()),
        };
        let audit = signed_null::run_frame_value(frame, options)?;
        let frame_index = zero_index + 1;
        let frame_path = audit["frame_path"]
            .as_str()
            .ok_or("signed-null audit returned no frame path")?;
        let mut row = Map::new();
        for key in [
            "background_adu",
            "positive_noise_adu",
            "negative_noise_adu",
            "positive_candidate_count",
            "positive_quality_count",
            "negative_candidate_count",
            "negative_quality_count",
            "negative_quality_count_after_range_exclusion",
            "candidate_leakage_ratio",
            "quality_leakage_ratio",
            "quality_leakage_ratio_after_range_exclusion",
            "negative_range_transfer_candidate_count",
            "negative_range_transfer_quality_count",
        ] {
            row.insert(key.into(), audit[key].clone());
        }
        row.insert("frame_index".into(), json!(frame_index));
        row.insert("frame_path".into(), json!(frame_path));
        row.insert("timestamp".into(), timestamp);
        frame_rows.push(Value::Object(row));
        for source in audit["thresholds"].as_array().ok_or("no threshold rows")? {
            let mut value = add_index_path(source, frame_index, frame_path)?;
            let object = value
                .as_object_mut()
                .ok_or("threshold row is not an object")?;
            let threshold = object.remove("threshold").ok_or("threshold missing")?;
            object.insert("threshold".into(), threshold);
            threshold_rows.push(value);
        }
        for source in audit["feature_rows"].as_array().ok_or("no feature rows")? {
            feature_rows.push(add_index_path(source, frame_index, frame_path)?);
        }
        for source in audit["negative_quality_sources"]
            .as_array()
            .ok_or("no signed-null source rows")?
        {
            quality_source_rows.push(add_index_path(source, frame_index, frame_path)?);
        }
        audits.push(audit);
    }
    let sum = |key: &str| -> Result<usize> {
        audits.iter().try_fold(0usize, |total, audit| {
            total
                .checked_add(number(audit, key)?)
                .ok_or_else(|| format!("count overflow while summing {key}"))
        })
    };
    let positive_candidate = sum("positive_candidate_count")?;
    let positive_quality = sum("positive_quality_count")?;
    let negative_candidate = sum("negative_candidate_count")?;
    let negative_quality = sum("negative_quality_count")?;
    let negative_effective_quality = sum("negative_quality_count_after_range_exclusion")?;
    let transfer_candidate = sum("negative_range_transfer_candidate_count")?;
    let transfer_quality = sum("negative_range_transfer_quality_count")?;
    let detector = audits[0]["parameters"]["detector"].clone();
    let paths_json = paths
        .iter()
        .map(|path| Value::String(path.to_string_lossy().into_owned()))
        .collect::<Vec<_>>();
    let recurrence_rows = recurrence_rows(&quality_source_rows)?;
    Ok(json!({
        "frame_count":total,
        "frame_row_count":frame_rows.len(),
        "threshold_row_count":threshold_rows.len(),
        "feature_row_count":feature_rows.len(),
        "frame_rows":frame_rows,
        "threshold_rows":threshold_rows,
        "feature_rows":feature_rows,
        "quality_source_row_count":quality_source_rows.len(),
        "quality_source_rows":quality_source_rows,
        "recurrence_row_count":recurrence_rows.len(),
        "recurrence_rows":recurrence_rows,
        "pooled_positive_candidate_count":positive_candidate,
        "pooled_positive_quality_count":positive_quality,
        "pooled_negative_candidate_count":negative_candidate,
        "pooled_negative_quality_count":negative_quality,
        "pooled_negative_quality_count_after_range_exclusion":negative_effective_quality,
        "pooled_candidate_leakage_ratio":safe_ratio(negative_candidate,positive_candidate),
        "pooled_quality_leakage_ratio":safe_ratio(negative_quality,positive_quality),
        "pooled_quality_leakage_ratio_after_range_exclusion":safe_ratio(negative_effective_quality,positive_quality),
        "pooled_negative_range_transfer_candidate_count":transfer_candidate,
        "pooled_negative_range_transfer_quality_count":transfer_quality,
        "frame_candidate_leakage_mean":optional_summary(&frame_rows,"candidate_leakage_ratio",false)?,
        "frame_candidate_leakage_median":optional_summary(&frame_rows,"candidate_leakage_ratio",true)?,
        "frame_quality_leakage_mean":optional_summary(&frame_rows,"quality_leakage_ratio",false)?,
        "frame_quality_leakage_median":optional_summary(&frame_rows,"quality_leakage_ratio",true)?,
        "parameters":{
            "frame_count":total,
            "frame_paths":paths_json,
            "detector":detector,
            "snr_thresholds":options.snr_thresholds,
            "aggregation":{
                "pooled_ratio_definition":"sum(sign_flipped count) / sum(positive count)",
                "frame_mean_median_definition":"mean/median of per-frame ratios over defined frame denominators",
                "feature_rows_definition":"one row per frame, control direction and mutually exclusive primary feature class",
                "quality_source_rows_definition":"all sign_flipped quality sources only; no full candidate catalog is persisted",
                "recurrence_definition":RECURRENCE_DEFINITION
            },
            "not_formal_fdr":true,
            "physical_truth_warning":true
        },
        "conclusion":CONCLUSION
    }))
}

fn csv_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::String(value) => value.clone(),
        value => value.to_string(),
    }
}

fn write_csv(path: &Path, columns: &[&str], rows: &[Value]) -> Result<()> {
    let mut file = File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&[0xef, 0xbb, 0xbf])
        .map_err(|error| error.to_string())?;
    if rows.is_empty() {
        file.write_all(b"\r\n").map_err(|error| error.to_string())?;
        return Ok(());
    }
    let mut writer = csv::WriterBuilder::new()
        .terminator(Terminator::CRLF)
        .from_writer(file);
    writer
        .write_record(columns)
        .map_err(|error| error.to_string())?;
    for row in rows {
        writer
            .write_record(columns.iter().map(|column| csv_value(&row[column])))
            .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    Ok(())
}

fn write_artifacts(result: &Value, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let output = out_dir.canonicalize().map_err(|error| error.to_string())?;
    let summary_path = output.join("signed_null_sequence_summary.json");
    serde_json::to_writer_pretty(
        File::create(summary_path).map_err(|error| error.to_string())?,
        result,
    )
    .map_err(|error| error.to_string())?;
    write_csv(
        &output.join("signed_null_sequence_frames.csv"),
        &[
            "frame_index",
            "frame_path",
            "timestamp",
            "background_adu",
            "positive_noise_adu",
            "negative_noise_adu",
            "positive_candidate_count",
            "positive_quality_count",
            "negative_candidate_count",
            "negative_quality_count",
            "negative_quality_count_after_range_exclusion",
            "candidate_leakage_ratio",
            "quality_leakage_ratio",
            "quality_leakage_ratio_after_range_exclusion",
            "negative_range_transfer_candidate_count",
            "negative_range_transfer_quality_count",
        ],
        result["frame_rows"]
            .as_array()
            .ok_or("frame_rows missing")?,
    )?;
    write_csv(
        &output.join("signed_null_sequence_thresholds.csv"),
        &[
            "frame_index",
            "frame_path",
            "threshold",
            "positive_filter_count",
            "negative_filter_count",
            "positive_flux_count",
            "negative_flux_count",
            "filter_leakage_ratio",
            "flux_leakage_ratio",
        ],
        result["threshold_rows"]
            .as_array()
            .ok_or("threshold_rows missing")?,
    )?;
    write_csv(
        &output.join("signed_null_sequence_features.csv"),
        &[
            "frame_index",
            "frame_path",
            "control",
            "feature_class",
            "feature_class_label",
            "candidate_count",
            "quality_count",
            "quality_fraction",
            "median_flux_snr",
            "max_flux_snr",
            "common_flags",
        ],
        result["feature_rows"]
            .as_array()
            .ok_or("feature_rows missing")?,
    )?;
    write_csv(
        &output.join("signed_null_sequence_quality_sources.csv"),
        &[
            "frame_index",
            "frame_path",
            "detection_id",
            "x",
            "y",
            "peak_x",
            "peak_y",
            "peak",
            "flux_snr",
            "filter_snr",
            "fwhm",
            "psf_support_pixels",
            "raw_peak_adu",
            "raw_negative_pixel_count",
            "raw_negative_anomaly_pixel_count",
            "raw_extreme_negative_pixel_count",
            "raw_minus_one_count",
            "raw_min_adu",
            "raw_max_adu",
            "range_transfer_overlap",
            "raw_evidence_layer",
            "flags",
        ],
        result["quality_source_rows"]
            .as_array()
            .ok_or("quality_source_rows missing")?,
    )?;
    write_csv(
        &output.join("signed_null_sequence_recurrences.csv"),
        &[
            "cluster_id",
            "frame_indices",
            "frame_count",
            "source_count",
            "x_median",
            "y_median",
            "max_radius_px",
            "transfer_count",
            "effective_count",
            "median_flux_snr",
        ],
        result["recurrence_rows"]
            .as_array()
            .ok_or("recurrence_rows missing")?,
    )?;
    Ok(output)
}

fn execute(args: &[String]) -> Result<(Value, PathBuf)> {
    let parsed = parse_args(args)?;
    let paths = collect_paths(&parsed.inputs)?;
    if paths.is_empty() {
        return Err("no FITS files found".into());
    }
    let result = run(&paths, &parsed.options)?;
    let output = write_artifacts(&result, &parsed.out_dir)?;
    Ok((result, output))
}

pub fn run_cli(args: &[String]) -> Result<()> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("逐帧运行真实 FITS 的符号反相 signed-tail leakage 对照，可传目录或多个 FITS 文件\n\nUsage: rst19-signed-null-sequence INPUT... [--threshold-sigma N] [--min-distance N] [--aperture-radius N] [--psf-fwhm N] [--proposal-mode gaussian|hybrid|ensemble] [--background-box-size N] [--min-flux-snr N] [--min-psf-support-pixels N] [--keep-linear-artifacts] [--snr-thresholds N...] [--out-dir DIR]");
        return Ok(());
    }
    let (result, output) = execute(args)?;
    println!("signed-null sequence audit: {}", output.display());
    println!(
        "frames={} pooled positive={}/{} sign_flipped={}/{} effective_quality={}",
        result["frame_count"],
        result["pooled_positive_candidate_count"],
        result["pooled_positive_quality_count"],
        result["pooled_negative_candidate_count"],
        result["pooled_negative_quality_count"],
        result["pooled_negative_quality_count_after_range_exclusion"]
    );
    Ok(())
}

pub fn run_desktop(args: &[String]) -> Result<Value> {
    let (result, output) = execute(args)?;
    Ok(json!({
        "tool":"rst19-signed-null-sequence",
        "out_dir":output,
        "result_path":output.join("signed_null_sequence_summary.json"),
        "result":result
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_inputs_and_preserves_explicit_option_forms() {
        assert!(collect_paths(&[]).unwrap().is_empty());
        let parsed = parse_args(&[
            "frame.fits".into(),
            "--threshold-sigma=5.5".into(),
            "--snr-thresholds".into(),
            "4".into(),
            "7".into(),
            "--keep-linear-artifacts".into(),
        ])
        .unwrap();
        assert_eq!(parsed.inputs, vec![PathBuf::from("frame.fits")]);
        assert_eq!(parsed.options.threshold_sigma, 5.5);
        assert_eq!(parsed.options.snr_thresholds, vec![4.0, 7.0]);
        assert!(!parsed.options.reject_linear_artifacts);
    }

    #[test]
    fn recurrence_joins_cross_frame_neighbors_without_claiming_identity() {
        let rows = vec![
            json!({"frame_index":1,"x":10.0,"y":20.0,"flux_snr":8.0,"range_transfer_overlap":false}),
            json!({"frame_index":2,"x":10.5,"y":20.0,"flux_snr":10.0,"range_transfer_overlap":true}),
            json!({"frame_index":2,"x":30.0,"y":40.0,"flux_snr":null,"range_transfer_overlap":false}),
        ];
        let clusters = recurrence_rows(&rows).unwrap();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0]["frame_indices"], "01,02");
        assert_eq!(clusters[0]["source_count"], 2);
        assert_eq!(clusters[0]["transfer_count"], 1);
        assert_eq!(clusters[0]["effective_count"], 1);
        assert_eq!(clusters[0]["median_flux_snr"], 9.0);
    }
}
