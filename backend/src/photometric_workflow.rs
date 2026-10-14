//! Native automatic plate-solving and apparent-photometry workflow.
//!
//! The workflow uses only a local Gaia-compatible CSV. Network catalog
//! acquisition remains an explicit, separate operation with its own audit.

use crate::{
    calibration_sensitivity::{self, SensitivityOptions},
    catalog_csv,
    coordinates::{AffineProjection, TangentPlaneWcs},
    fits,
    photometric_calibration::PhotometricCalibration,
    pipeline::{self, FrameAnalysis, FrameOptions, RecalibrationOptions},
    plate_solver::{self, PlateSolveOptions, PlateSolveResult},
    wcs_fit::{self, AffineFitOptions, AffineWcsCalibration},
    Result,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

pub const DEFAULT_CAMERA_PIXEL_SCALE_ARCSEC: f64 = 8.5;
pub const DEFAULT_PLATE_SCALE_TOLERANCE: f64 = 0.04;
pub const DEFAULT_PLATE_MIN_MATCHES: usize = 8;
pub const DEFAULT_PLATE_MIN_COVERAGE: f64 = 0.02;
pub const DEFAULT_PLATE_MAX_RMS_PX: f64 = 2.0;
pub const DEFAULT_PLATE_MAX_LOO_RMS_PX: f64 = 3.0;
pub const DEFAULT_PHOTOMETRY_MATCH_RADIUS_PX: f64 = 1.0;

#[derive(Debug, Clone)]
struct Options {
    frame: PathBuf,
    catalog: PathBuf,
    output: PathBuf,
    pixel_scale_arcsec: f64,
    match_radius_px: f64,
    photometry_match_radius_px: Option<f64>,
    min_calibrators: usize,
    include_all_sources: bool,
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.filter(|text| !text.is_empty())
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn provenance_status(path: &Path) -> String {
    let sidecar = PathBuf::from(format!("{}.meta.json", path.to_string_lossy()));
    if !sidecar.is_file() {
        return "PROVENANCE_UNKNOWN".into();
    }
    let payload: Value = match fs::read(&sidecar)
        .map_err(|error| error.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|error| error.to_string()))
    {
        Ok(value) => value,
        Err(_) => return "PROVENANCE_INVALID".into(),
    };
    let Some(object) = payload.as_object() else {
        return "PROVENANCE_INVALID".into();
    };
    if object.get("complete") == Some(&Value::Bool(false)) {
        return "INCOMPLETE".into();
    }
    if object.get("complete") != Some(&Value::Bool(true)) {
        return "PROVENANCE_UNKNOWN".into();
    }
    let Some(expected_hash) = object.get("csv_sha256").and_then(Value::as_str) else {
        return "PROVENANCE_UNBOUND".into();
    };
    let Some(expected_rows) = object.get("csv_row_count").and_then(Value::as_u64) else {
        return "PROVENANCE_UNBOUND".into();
    };
    if expected_hash.len() != 64 {
        return "PROVENANCE_UNBOUND".into();
    }
    let Ok(mut file) = File::open(path) else {
        return "PROVENANCE_INVALID".into();
    };
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => digest.update(&buffer[..count]),
            Err(_) => return "PROVENANCE_INVALID".into(),
        }
    }
    let actual_hash = format!("{:x}", digest.finalize());
    if actual_hash != expected_hash.to_lowercase() {
        return "PROVENANCE_INVALID".into();
    }
    let mut reader = match csv::ReaderBuilder::new().from_path(path) {
        Ok(reader) => reader,
        Err(_) => return "PROVENANCE_INVALID".into(),
    };
    let mut actual_rows = 0_u64;
    for row in reader.records() {
        if row.is_err() {
            return "PROVENANCE_INVALID".into();
        }
        actual_rows += 1;
    }
    if actual_rows != expected_rows {
        return "PROVENANCE_INVALID".into();
    }
    if object.get("coverage_warning").is_some_and(json_truthy) {
        "COMPLETE_WITH_WARNING".into()
    } else {
        "COMPLETE".into()
    }
}

fn reference_wcs_value(wcs: &TangentPlaneWcs) -> Value {
    json!({
        "center_ra_deg":wcs.center_ra_deg,
        "center_dec_deg":wcs.center_dec_deg,
        "pixel_scale_arcsec":wcs.pixel_scale_arcsec,
        "crpix_x":wcs.crpix_x,
        "crpix_y":wcs.crpix_y,
        "rotation_deg":wcs.rotation_deg,
        "parity":wcs.parity
    })
}

fn affine_rejection_reason(
    affine: &AffineWcsCalibration,
    min_matches: usize,
    max_rms_px: f64,
    max_loo_rms_px: f64,
) -> Option<String> {
    let mut failures = Vec::new();
    if affine.inlier_count < min_matches {
        failures.push(format!(
            "仿射内点数 {} < {min_matches}",
            affine.inlier_count
        ));
    }
    if !affine.rms_residual_px.is_finite() {
        failures.push("仿射 RMS 无效".into());
    } else if affine.rms_residual_px > max_rms_px {
        failures.push(format!(
            "仿射 RMS {:.3}px > {max_rms_px:.3}px",
            affine.rms_residual_px
        ));
    }
    if !affine.condition_number.is_finite() || affine.condition_number > 1.0e10 {
        failures.push("仿射条件数无效或过大".into());
    }
    if !affine.anisotropy_ratio.is_finite() || affine.anisotropy_ratio <= 0.0 {
        failures.push("仿射各向异性指标无效".into());
    }
    match affine.leave_one_out_rms_residual_px {
        None if affine.validation_count < 1 => failures.push("仿射留一验证无有效样本".into()),
        None => failures.push("仿射留一残差无效".into()),
        Some(value) if !value.is_finite() => failures.push("仿射留一残差无效".into()),
        Some(value) if value > max_loo_rms_px => {
            failures.push(format!("仿射留一 RMS {value:.3}px > {max_loo_rms_px:.3}px"))
        }
        Some(_) => {}
    }
    (!failures.is_empty()).then(|| failures.join("；"))
}

fn compact_analysis(analysis: &FrameAnalysis, include_all_sources: bool) -> Value {
    let mut payload = analysis.as_dict();
    let Some(object) = payload.as_object_mut() else {
        return payload;
    };
    if include_all_sources {
        object.insert("evidence_scope".into(), json!("full_source_evidence"));
        return payload;
    }
    let mut included_source_count = 0;
    if let Some(detection) = object.get_mut("detection").and_then(Value::as_object_mut) {
        detection.insert("sources".into(), json!([]));
        detection.insert("evidence_scope".into(), json!("summary_only"));
    }
    if let Some(matching) = object.get_mut("matching").and_then(Value::as_object_mut) {
        let unmatched_detection_count = matching
            .remove("unmatched_detection_ids")
            .and_then(|value| value.as_array().map(Vec::len))
            .unwrap_or(0);
        let unmatched_catalog_count = matching
            .remove("unmatched_catalog_ids")
            .and_then(|value| value.as_array().map(Vec::len))
            .unwrap_or(0);
        matching.insert(
            "unmatched_detection_count".into(),
            json!(unmatched_detection_count),
        );
        matching.insert(
            "unmatched_catalog_count".into(),
            json!(unmatched_catalog_count),
        );
        matching.insert("evidence_scope".into(), json!("matches_and_counts"));
    }
    let total_source_count = object
        .get("source_photometry")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    if let Some(rows) = object
        .get_mut("source_photometry")
        .and_then(Value::as_array_mut)
    {
        rows.retain(|row| {
            let keep = row.get("source_id").is_some_and(|id| !id.is_null());
            if keep {
                included_source_count += 1;
            }
            keep
        });
    }
    object.insert("evidence_scope".into(), json!("compact_auto_photometry"));
    object.insert(
        "source_photometry_total_count".into(),
        json!(total_source_count),
    );
    object.insert(
        "source_photometry_included_count".into(),
        json!(included_source_count),
    );
    payload
}

fn build_payload(
    analysis: &FrameAnalysis,
    catalog_path: &Path,
    catalog_count: usize,
    reference_wcs: &TangentPlaneWcs,
    plate_solution: &PlateSolveResult,
    affine_wcs: Option<&AffineWcsCalibration>,
    status: &str,
    reason: &str,
    catalog_provenance_status: &str,
    match_radius_px: f64,
    photometry_match_radius_px: f64,
    sensitivity: Option<&Value>,
    include_all_sources: bool,
) -> Value {
    let photometric_status = analysis
        .photometric_calibration
        .as_ref()
        .map(|calibration| calibration.status.as_str());
    let calibrated = affine_wcs.is_some()
        && plate_solution.valid
        && photometric_status.is_some_and(|value| matches!(value, "VALID" | "VALID_NO_HOLDOUT"))
        && matches!(
            catalog_provenance_status,
            "COMPLETE" | "IN_MEMORY_UNVERIFIED"
        );
    let faintest_scope = analysis
        .faintest
        .as_ref()
        .map(|faintest| faintest.selection_scope.as_str());
    let faintest_global_claim = calibrated && faintest_scope == Some("CALIBRATED_MATCHES");
    json!({
        "schema_version":1,
        "status":status,
        "calibrated":calibrated,
        "reason":reason,
        "frame_path":analysis.frame.path.to_string_lossy(),
        "catalog_path":catalog_path.to_string_lossy(),
        "catalog_provenance_status":catalog_provenance_status,
        "catalog_source_count":catalog_count,
        "matching_policy":{
            "plate_match_radius_px":match_radius_px,
            "selected_plate_match_radius_px":plate_solution.acceptance["match_radius_px"]
                .as_f64().unwrap_or(match_radius_px),
            "photometry_match_radius_px":photometry_match_radius_px,
            "description":"板解使用宽匹配半径提出几何候选；仿射 WCS 通过后，测光重匹配使用更严格半径以降低密集星场近邻误配。"
        },
        "reference_wcs":reference_wcs_value(reference_wcs),
        "plate_solution":plate_solution,
        "affine_wcs":affine_wcs.map(AffineWcsCalibration::as_dict),
        "photometric_status":photometric_status,
        "faintest_global_claim":faintest_global_claim,
        "faintest_selection_scope":faintest_scope,
        "analysis":compact_analysis(analysis,include_all_sources),
        "interpretation":"m_cal is an empirically calibrated apparent magnitude in the declared catalog system; it is not an automatic claim of camera-specific radiometric calibration.",
        "calibration_model_sensitivity":sensitivity
    })
}

fn finish_with_status(
    analysis: &FrameAnalysis,
    catalog_path: &Path,
    catalog_count: usize,
    reference_wcs: &TangentPlaneWcs,
    plate_solution: &PlateSolveResult,
    affine_wcs: Option<&AffineWcsCalibration>,
    status: String,
    reason: String,
    provenance: &str,
    match_radius: f64,
    photometry_radius: f64,
    sensitivity: Option<&Value>,
    include_all_sources: bool,
) -> (Value, bool) {
    let payload = build_payload(
        analysis,
        catalog_path,
        catalog_count,
        reference_wcs,
        plate_solution,
        affine_wcs,
        &status,
        &reason,
        provenance,
        match_radius,
        photometry_radius,
        sensitivity,
        include_all_sources,
    );
    let calibrated = payload["calibrated"] == true;
    (payload, calibrated)
}

fn run_workflow(options: &Options) -> Result<(Value, bool)> {
    let frame_path = options
        .frame
        .canonicalize()
        .map_err(|error| format!("FITS 不存在：{}：{error}", options.frame.display()))?;
    let catalog_path = options
        .catalog
        .canonicalize()
        .map_err(|error| format!("星表文件不存在：{}：{error}", options.catalog.display()))?;
    let frame = fits::read_fits(&frame_path, false)?;
    let Some(auxiliary) = frame.auxiliary.as_ref() else {
        return Err("FITS 没有可用的辅助 RA/DEC，无法自动规划 WCS".into());
    };
    let exposure_s = fits::exposure_seconds(&frame.header, 1.0)?;
    let catalog = catalog_csv::load_catalog_csv(&catalog_path)?;
    if catalog.is_empty() {
        return Err("星表为空，无法进行自动板解".into());
    }
    let provenance = provenance_status(&catalog_path);
    let scale = options.pixel_scale_arcsec;
    let match_radius = options.match_radius_px;
    let photometry_radius = options
        .photometry_match_radius_px
        .unwrap_or(match_radius.min(DEFAULT_PHOTOMETRY_MATCH_RADIUS_PX));
    if !scale.is_finite()
        || scale <= 0.0
        || !match_radius.is_finite()
        || match_radius <= 0.0
        || !photometry_radius.is_finite()
        || photometry_radius <= 0.0
    {
        return Err("像元尺度和两级匹配半径必须为正数".into());
    }
    let reference_wcs = TangentPlaneWcs {
        center_ra_deg: auxiliary.ra_deg(),
        center_dec_deg: auxiliary.dec_deg(),
        pixel_scale_arcsec: scale,
        crpix_x: frame.width as f64 / 2.0,
        crpix_y: frame.height as f64 / 2.0,
        rotation_deg: 0.0,
        parity: 1,
    }
    .validated()?;
    eprintln!("  2.0% · 准备自动 Gaia 测光");
    eprintln!("  5.0% · 正在检测 FITS 星点");
    let analysis = pipeline::analyze_frame(frame, None, None, &FrameOptions::default(), None)?;
    eprintln!(" 67.0% · 检测完成");
    let base_solver = PlateSolveOptions {
        epoch: None,
        image_shape: Some((analysis.frame.height, analysis.frame.width)),
        scale_tolerance: DEFAULT_PLATE_SCALE_TOLERANCE,
        match_radius_px: match_radius,
        min_matches: DEFAULT_PLATE_MIN_MATCHES,
        min_coverage_area: DEFAULT_PLATE_MIN_COVERAGE,
        max_rms_residual_px: DEFAULT_PLATE_MAX_RMS_PX,
        max_leave_one_out_rms_px: DEFAULT_PLATE_MAX_LOO_RMS_PX,
        ..PlateSolveOptions::default()
    };
    eprintln!(" 72.0% · 正在用 Gaia 星对搜索 WCS");
    let screening = plate_solver::solve_plate(
        &analysis.detection.sources,
        &catalog,
        &reference_wcs,
        base_solver.clone(),
    )?;
    let mut plate_solution = screening;
    if photometry_radius < match_radius {
        eprintln!(" 77.0% · 宽门板解完成 · 正在用 {photometry_radius:.2}px 细门复核几何解");
        let fine = plate_solver::solve_plate(
            &analysis.detection.sources,
            &catalog,
            &reference_wcs,
            PlateSolveOptions {
                match_radius_px: photometry_radius,
                ..base_solver
            },
        )?;
        if fine.valid && fine.best.is_some() {
            plate_solution = fine;
        }
    }
    if !plate_solution.valid || plate_solution.best.is_none() {
        eprintln!("100.0% · 自动板解未通过 · {}", plate_solution.reason);
        return Ok(finish_with_status(
            &analysis,
            &catalog_path,
            catalog.len(),
            &reference_wcs,
            &plate_solution,
            None,
            format!("WCS_{}", plate_solution.status),
            format!("自动板解未通过：{}；仅保留 m_inst。", plate_solution.reason),
            &provenance,
            match_radius,
            photometry_radius,
            None,
            options.include_all_sources,
        ));
    }
    eprintln!(
        " 82.0% · 板解通过 · 匹配 {} · 正在细化仿射 WCS",
        plate_solution
            .best
            .as_ref()
            .map_or(0, |best| best.matched_count)
    );
    let affine = match wcs_fit::fit_affine_wcs_from_matches(
        &plate_solution.best.as_ref().unwrap().matches,
        &catalog,
        &reference_wcs,
        None,
        AffineFitOptions {
            min_matches: 6.min(DEFAULT_PLATE_MIN_MATCHES),
            ..AffineFitOptions::default()
        },
    ) {
        Ok(affine) => affine,
        Err(error) => {
            eprintln!("100.0% · WCS 细化失败 · {error}");
            return Ok(finish_with_status(
                &analysis,
                &catalog_path,
                catalog.len(),
                &reference_wcs,
                &plate_solution,
                None,
                "WCS_REFINEMENT_FAILED".into(),
                format!("星对板解通过但仿射细化失败：{error}；仅保留 m_inst。"),
                &provenance,
                match_radius,
                photometry_radius,
                None,
                options.include_all_sources,
            ));
        }
    };
    if let Some(reason) = affine_rejection_reason(
        &affine,
        6.min(DEFAULT_PLATE_MIN_MATCHES),
        DEFAULT_PLATE_MAX_RMS_PX,
        DEFAULT_PLATE_MAX_LOO_RMS_PX,
    ) {
        eprintln!("100.0% · 仿射 WCS 验收失败 · {reason}");
        return Ok(finish_with_status(
            &analysis,
            &catalog_path,
            catalog.len(),
            &reference_wcs,
            &plate_solution,
            None,
            "WCS_REFINEMENT_REJECTED".into(),
            format!("仿射 WCS 验收失败：{reason}；仅保留 m_inst。"),
            &provenance,
            match_radius,
            photometry_radius,
            None,
            options.include_all_sources,
        ));
    }
    let first = catalog.first().ok_or("星表为空，无法进行自动板解")?;
    let system = nonempty(first.photometric_system.as_deref()).unwrap_or("Gaia Vega");
    let band = nonempty(first.photometric_band.as_deref()).unwrap_or("G");
    let color_name = nonempty(first.color_name.as_deref()).unwrap_or("BP-RP");
    eprintln!(" 90.0% · WCS 已细化 · {photometry_radius:.2}px 细匹配拟合多参考星光度零点");
    let mut refined = analysis.clone();
    let refined_projection = crate::matching::Projection::Affine(AffineProjection {
        center_ra_deg: affine.center_ra_deg,
        center_dec_deg: affine.center_dec_deg,
        matrix_px_per_arcsec: affine.matrix_px_per_arcsec,
        offset_px: affine.offset_px,
    });
    if let Err(error) = pipeline::recalibrate_frame_analysis(
        &mut refined,
        &catalog,
        &refined_projection,
        &RecalibrationOptions {
            match_radius_px: photometry_radius,
            epoch: None,
            photometric_system: Some(system.into()),
            photometric_band: Some(band.into()),
            photometric_color_name: Some(color_name.into()),
            photometric_color_order: Some(1),
            photometric_min_calibrators: options.min_calibrators,
            ..RecalibrationOptions::default()
        },
    ) {
        eprintln!("100.0% · 光度层失败 · {error}");
        return Ok(finish_with_status(
            &analysis,
            &catalog_path,
            catalog.len(),
            &reference_wcs,
            &plate_solution,
            Some(&affine),
            "WCS_PHOTOMETRY_ERROR".into(),
            format!("WCS 已通过但光度层发生异常：{error}；仅保留 m_inst。"),
            &provenance,
            match_radius,
            photometry_radius,
            None,
            options.include_all_sources,
        ));
    }
    let sensitivity = if let Some(matching) = refined.matching.as_ref() {
        eprintln!(" 96.0% · 光度层完成 · 正在检查颜色模型对最暗星排序的敏感性");
        let sensitivity_options = SensitivityOptions {
            exposure_s,
            photometric_system: Some(system.into()),
            photometric_band: Some(band.into()),
            color_name: Some(color_name.into()),
            model_orders: vec![0, 1, 2],
            min_snr: refined.detection.parameters["min_flux_snr"]
                .as_f64()
                .unwrap_or(5.0),
            min_calibrators: options.min_calibrators,
        };
        match calibration_sensitivity::run(
            &matching.matches,
            &refined.detection.sources,
            &catalog,
            &sensitivity_options,
        ) {
            Ok(value) => Some(value),
            Err(error) => Some(json!({
                "schema_version":1,
                "status":"ERROR",
                "method":"low_order_color_model_sensitivity",
                "reason":error
            })),
        }
    } else {
        None
    };
    let calibration: Option<&PhotometricCalibration> = refined.photometric_calibration.as_ref();
    let status_and_reason = if calibration
        .is_some_and(|value| matches!(value.status.as_str(), "VALID" | "VALID_NO_HOLDOUT"))
    {
        let calibration = calibration.unwrap();
        let scope = refined
            .faintest
            .as_ref()
            .map(|value| value.selection_scope.as_str())
            .unwrap_or("NO_FAINTEST_RESULT");
        let scope_status = if matches!(scope, "CALIBRATED_MATCHES" | "NO_FAINTEST_RESULT") {
            "CALIBRATED"
        } else {
            "CALIBRATED_PARTIAL"
        };
        let status = if matches!(provenance.as_str(), "COMPLETE" | "IN_MEMORY_UNVERIFIED") {
            scope_status.to_owned()
        } else {
            "CALIBRATED_PROVISIONAL_REFERENCE".into()
        };
        let reason = format!(
            "WCS 与 Gaia 经验光度标定通过：参考星 {}，光度状态 {}；最暗判定集合为 {}。主结果为 {}/{} 表观星等。{}",
            calibration.calibrator_count,
            calibration.status,
            scope,
            system,
            band,
            if status == "CALIBRATED_PROVISIONAL_REFERENCE" {
                format!(" 星表完整性状态为 {provenance}，暂不作为正式校准结果。")
            } else {
                String::new()
            }
        );
        (status, reason)
    } else {
        let photometric_status = calibration.map_or("MISSING", |value| value.status.as_str());
        (
            format!("WCS_VALID_PHOTOMETRY_{photometric_status}"),
            format!("WCS 通过，但光度标定未通过（{photometric_status}）；仅保留 m_inst，不能把未通过的数值作为主星等。"),
        )
    };
    eprintln!("100.0% · {}", status_and_reason.1);
    Ok(finish_with_status(
        &refined,
        &catalog_path,
        catalog.len(),
        &reference_wcs,
        &plate_solution,
        Some(&affine),
        status_and_reason.0,
        status_and_reason.1,
        &provenance,
        match_radius,
        photometry_radius,
        sensitivity.as_ref(),
        options.include_all_sources,
    ))
}

fn parse_number<T: std::str::FromStr>(name: &str, text: String) -> Result<T> {
    text.parse()
        .map_err(|_| format!("{name} has an invalid value: {text}"))
}

impl Options {
    fn parse(args: &[String], default_out_dir: Option<&Path>) -> Result<Self> {
        let mut frame = None;
        let mut catalog = None;
        let mut output = None;
        let mut pixel_scale = DEFAULT_CAMERA_PIXEL_SCALE_ARCSEC;
        let mut match_radius = 3.0;
        let mut photometry_match_radius = None;
        let mut min_calibrators = 6;
        let mut include_all_sources = false;
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            if arg == "-h" || arg == "--help" {
                return Err("__help__".into());
            }
            if arg == "--include-all-sources" {
                include_all_sources = true;
                index += 1;
                continue;
            }
            if !arg.starts_with("--") {
                if frame.replace(PathBuf::from(arg)).is_some() {
                    return Err("only one FITS path is accepted".into());
                }
                index += 1;
                continue;
            }
            let (name, inline) = arg
                .split_once('=')
                .map(|(name, value)| (name, Some(value.to_owned())))
                .unwrap_or((arg.as_str(), None));
            let value = if let Some(value) = inline {
                value
            } else {
                index += 1;
                args.get(index)
                    .cloned()
                    .ok_or_else(|| format!("{name} requires a value"))?
            };
            match name {
                "--catalog" => catalog = Some(PathBuf::from(value)),
                "--out" => output = Some(PathBuf::from(value)),
                "--pixel-scale" => pixel_scale = parse_number(name, value)?,
                "--match-radius" => match_radius = parse_number(name, value)?,
                "--photometry-match-radius" => {
                    photometry_match_radius = Some(parse_number(name, value)?)
                }
                "--min-calibrators" => min_calibrators = parse_number(name, value)?,
                _ => return Err(format!("unknown argument {arg}")),
            }
            index += 1;
        }
        let frame = frame.ok_or("a FITS path is required")?;
        let catalog = catalog.ok_or("--catalog is required")?;
        let output = output
            .or_else(|| default_out_dir.map(|dir| dir.join("photometric-auto.json")))
            .ok_or("--out is required")?;
        if !frame.is_file() {
            return Err(format!("FITS 不存在：{}", frame.display()));
        }
        if !catalog.is_file() {
            return Err(format!("星表不存在：{}", catalog.display()));
        }
        Ok(Self {
            frame,
            catalog,
            output,
            pixel_scale_arcsec: pixel_scale,
            match_radius_px: match_radius,
            photometry_match_radius_px: photometry_match_radius,
            min_calibrators,
            include_all_sources,
        })
    }
}

fn execute(options: &Options) -> Result<(Value, bool)> {
    let (payload, calibrated) = run_workflow(options)?;
    if let Some(parent) = options
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut bytes = serde_json::to_vec_pretty(&payload).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    fs::write(&options.output, bytes)
        .map_err(|error| format!("无法写入输出 JSON {}：{error}", options.output.display()))?;
    Ok((payload, calibrated))
}

pub fn run_desktop(args: &[String], out_dir: &Path) -> Result<Value> {
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let options = Options::parse(args, Some(out_dir))?;
    execute(&options).map(|(payload, _)| payload)
}

const HELP: &str = "使用 FITS 辅助视轴和本地 Gaia CSV 自动板解并拟合经验表观星等\n\nUSAGE: rst19-photometric-auto FITS --catalog CSV --out JSON [OPTIONS]\n\nOptions:\n  --pixel-scale ARCSEC             [default: 8.5]\n  --match-radius PX                [default: 3.0]\n  --photometry-match-radius PX     [default: min(match-radius, 1.0)]\n  --min-calibrators N              [default: 6]\n  --include-all-sources\n  -h, --help";

pub fn run_cli(args: &[String]) -> i32 {
    let options = match Options::parse(args, None) {
        Ok(options) => options,
        Err(error) if error == "__help__" => {
            println!("{HELP}");
            return 0;
        }
        Err(error) => {
            eprintln!("photometric-workflow: {error}\n\n{HELP}");
            return 2;
        }
    };
    match execute(&options) {
        Ok((payload, true)) => {
            eprintln!(
                "{}: {} · evidence -> {}",
                payload["status"].as_str().unwrap_or("CALIBRATED"),
                payload["reason"].as_str().unwrap_or(""),
                options.output.display()
            );
            0
        }
        Ok((payload, false)) => {
            eprintln!(
                "{}: {} · evidence -> {}",
                payload["status"].as_str().unwrap_or("UNCALIBRATED"),
                payload["reason"].as_str().unwrap_or(""),
                options.output.display()
            );
            3
        }
        Err(error) => {
            eprintln!("photometric-workflow: {error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn provenance_requires_bound_hash_and_exact_integer_csv_row_count() {
        let root = std::env::temp_dir().join(format!(
            "rst19-photo-provenance-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let csv = root.join("gaia.csv");
        let audit = root.join("gaia.csv.meta.json");
        fs::write(&csv, "source_id,ra_deg,dec_deg\n1,10,20\n").unwrap();
        assert_eq!(provenance_status(&csv), "PROVENANCE_UNKNOWN");
        fs::write(&audit, r#"{"complete":true}"#).unwrap();
        assert_eq!(provenance_status(&csv), "PROVENANCE_UNBOUND");
        let bytes = fs::read(&csv).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        fs::write(
            &audit,
            json!({"complete":true,"csv_sha256":digest,"csv_row_count":1}).to_string(),
        )
        .unwrap();
        assert_eq!(provenance_status(&csv), "COMPLETE");
        fs::write(&csv, "source_id,ra_deg,dec_deg\n2,10,20\n").unwrap();
        assert_eq!(provenance_status(&csv), "PROVENANCE_INVALID");
        fs::write(
            &audit,
            json!({"complete":true,"csv_sha256":digest,"csv_row_count":1.5}).to_string(),
        )
        .unwrap();
        assert_eq!(provenance_status(&csv), "PROVENANCE_UNBOUND");
    }
}
