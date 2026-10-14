//! Rust command path for the single-frame source artifact audit.

use crate::{
    detection::{self, DetectionOptions},
    fits, source_artifacts, Result,
};
use serde_json::{json, Value};
use std::path::PathBuf;

struct SourceAuditArgs {
    fits_path: PathBuf,
    out_dir: PathBuf,
    detection: DetectionOptions,
    spatial_psf_per_class: usize,
    spatial_psf_grid_size: usize,
}

fn option_value<'a>(
    args: &'a [String],
    index: &mut usize,
    inline: Option<&'a str>,
    name: &str,
) -> Result<&'a str> {
    if let Some(value) = inline {
        return Ok(value);
    }
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| format!("--{name} requires a value"))
}

fn float(value: &str, name: &str) -> Result<f64> {
    let value = value
        .parse::<f64>()
        .map_err(|_| format!("invalid float for --{name}"))?;
    if !value.is_finite() {
        return Err(format!("--{name} must be finite"));
    }
    Ok(value)
}

fn integer(value: &str, name: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid non-negative integer for --{name}"))
}

fn parse_args(args: &[String]) -> Result<SourceAuditArgs> {
    let mut detection = DetectionOptions {
        min_distance: 4,
        proposal_mode: "hybrid".into(),
        ..DetectionOptions::default()
    };
    let mut out_dir = PathBuf::from("tmp/source-audit");
    let mut spatial_psf_per_class = 8;
    let mut spatial_psf_grid_size = 4;
    let mut fits_path = None;
    let mut positional_only = false;
    let mut index = 0;

    while index < args.len() {
        let arg = &args[index];
        if arg == "--" && !positional_only {
            positional_only = true;
            index += 1;
            continue;
        }
        if !positional_only && arg.starts_with("--") {
            let (name, inline) = arg[2..]
                .split_once('=')
                .map(|(name, value)| (name, Some(value)))
                .unwrap_or((&arg[2..], None));
            match name {
                "enable-local-deblend" | "keep-linear-artifacts" => {
                    if inline.is_some() {
                        return Err(format!("--{name} does not take a value"));
                    }
                    if name == "enable-local-deblend" {
                        detection.enable_local_deblend = true;
                    } else {
                        detection.reject_linear_artifacts = false;
                    }
                }
                "threshold-sigma" => {
                    detection.threshold_sigma =
                        float(option_value(args, &mut index, inline, name)?, name)?;
                }
                "min-distance" => {
                    detection.min_distance =
                        integer(option_value(args, &mut index, inline, name)?, name)?;
                }
                "aperture-radius" => {
                    detection.measurement.aperture_radius =
                        integer(option_value(args, &mut index, inline, name)?, name)?;
                }
                "psf-fwhm" => {
                    detection.measurement.psf_fwhm =
                        float(option_value(args, &mut index, inline, name)?, name)?;
                }
                "proposal-mode" => {
                    let value = option_value(args, &mut index, inline, name)?;
                    if !matches!(value, "gaussian" | "hybrid" | "ensemble") {
                        return Err("--proposal-mode must be gaussian, hybrid, or ensemble".into());
                    }
                    detection.proposal_mode = value.to_owned();
                }
                "dog-threshold-sigma" => {
                    detection.dog_threshold_sigma =
                        Some(float(option_value(args, &mut index, inline, name)?, name)?);
                }
                "dog-min-peak-sigma" => {
                    detection.dog_min_peak_sigma =
                        float(option_value(args, &mut index, inline, name)?, name)?;
                }
                "starlet-threshold-sigma" => {
                    detection.starlet_threshold_sigma =
                        Some(float(option_value(args, &mut index, inline, name)?, name)?);
                }
                "starlet-min-peak-sigma" => {
                    detection.starlet_min_peak_sigma =
                        float(option_value(args, &mut index, inline, name)?, name)?;
                }
                "background-box-size" => {
                    detection.background_box_size =
                        integer(option_value(args, &mut index, inline, name)?, name)?;
                }
                "min-flux-snr" => {
                    detection.measurement.min_flux_snr =
                        float(option_value(args, &mut index, inline, name)?, name)?;
                }
                "min-psf-support-pixels" => {
                    detection.measurement.min_psf_support_pixels =
                        integer(option_value(args, &mut index, inline, name)?, name)?;
                }
                "max-sources" => {
                    detection.max_sources = Some(integer(
                        option_value(args, &mut index, inline, name)?,
                        name,
                    )?);
                }
                "spatial-psf-per-class" => {
                    spatial_psf_per_class =
                        integer(option_value(args, &mut index, inline, name)?, name)?;
                }
                "spatial-psf-grid-size" => {
                    spatial_psf_grid_size =
                        integer(option_value(args, &mut index, inline, name)?, name)?;
                }
                "out-dir" => {
                    out_dir = PathBuf::from(option_value(args, &mut index, inline, name)?);
                }
                _ => return Err(format!("unknown argument --{name}")),
            }
        } else if !positional_only && arg.starts_with('-') {
            return Err(format!("unknown argument {arg}"));
        } else if fits_path.replace(PathBuf::from(arg)).is_some() {
            return Err("only one FITS path is allowed".into());
        }
        index += 1;
    }

    let fits_path = fits_path.ok_or("missing FITS path")?;
    if spatial_psf_per_class == 0 || spatial_psf_grid_size == 0 {
        return Err("spatial PSF sample count and grid size must be positive".into());
    }
    Ok(SourceAuditArgs {
        fits_path,
        out_dir,
        detection,
        spatial_psf_per_class,
        spatial_psf_grid_size,
    })
}

/// Run source detection and write every Rust source-audit artifact.
pub fn run_desktop(args: &[String]) -> Result<Value> {
    let args = parse_args(args)?;
    let frame = fits::read_fits(&args.fits_path, false)?;
    let mask = fits::auxiliary_mask(frame.height, frame.width)?;
    let detection = detection::detect_sources(
        &frame.data,
        Some(&mask),
        frame.height,
        frame.width,
        &args.detection,
        None,
    )?;
    source_artifacts::write_detection_source_artifacts(
        &frame,
        &detection,
        &args.out_dir,
        args.spatial_psf_per_class,
        args.spatial_psf_grid_size,
    )?;
    Ok(json!({
        "output":args.out_dir,
        "candidate_count":detection.candidate_count,
        "returned_count":detection.sources.len(),
        "quality_count":detection.quality_count,
        "rejected_count":detection.sources.len().saturating_sub(detection.quality_count),
    }))
}

pub const HELP: &str = "对单张 FITS 输出源表、质量/形态汇总、诊断图和经验 PSF 审计\n\nUsage: rst19-sources FITS_PATH [OPTIONS]\n\nOptions: --threshold-sigma --min-distance --aperture-radius --psf-fwhm --proposal-mode\n         --dog-threshold-sigma --dog-min-peak-sigma --starlet-threshold-sigma\n         --starlet-min-peak-sigma --enable-local-deblend --background-box-size\n         --min-flux-snr --min-psf-support-pixels --max-sources --keep-linear-artifacts\n         --spatial-psf-per-class --spatial-psf-grid-size --out-dir";

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    fn assert_csv_matches_reference(name: &str, actual_path: &std::path::Path, reference: &[u8]) {
        let mut actual_reader = csv::ReaderBuilder::new()
            .from_path(actual_path)
            .unwrap_or_else(|error| panic!("cannot read Rust CSV {name}: {error}"));
        let mut reference_reader = csv::ReaderBuilder::new().from_reader(reference);
        assert_eq!(
            actual_reader.headers().unwrap(),
            reference_reader.headers().unwrap(),
            "CSV fields differ for {name}"
        );
        let actual_rows = actual_reader
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let reference_rows = reference_reader
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            actual_rows.len(),
            reference_rows.len(),
            "row count differs for {name}"
        );
        for (row_index, (actual, expected)) in actual_rows.iter().zip(&reference_rows).enumerate() {
            assert_eq!(
                actual.len(),
                expected.len(),
                "field count differs for {name}"
            );
            for field_index in 0..actual.len() {
                let actual_cell = actual.get(field_index).unwrap_or_default();
                let expected_cell = expected.get(field_index).unwrap_or_default();
                match (actual_cell.parse::<f64>(), expected_cell.parse::<f64>()) {
                    (Ok(actual_number), Ok(expected_number))
                        if actual_number.is_finite() && expected_number.is_finite() =>
                    {
                        let tolerance = 1e-8 + expected_number.abs() * 1e-12;
                        assert!(
                            (actual_number - expected_number).abs() <= tolerance,
                            "{name} row {} field {} numeric mismatch: Rust={actual_number} reference={expected_number} tolerance={tolerance}",
                            row_index + 1,
                            field_index + 1
                        );
                    }
                    _ => assert_eq!(
                        actual_cell,
                        expected_cell,
                        "{name} row {} field {} text mismatch",
                        row_index + 1,
                        field_index + 1
                    ),
                }
            }
        }
    }

    #[test]
    fn source_cli_defaults_and_overrides_match_the_legacy_parser() {
        let parsed = parse_args(&["frame.fits".into()]).unwrap();
        assert_eq!(parsed.detection.threshold_sigma, 4.0);
        assert_eq!(parsed.detection.min_distance, 4);
        assert_eq!(parsed.detection.proposal_mode, "hybrid");
        assert_eq!(parsed.detection.measurement.aperture_radius, 4);
        assert_eq!(parsed.detection.measurement.psf_fwhm, 3.0);
        assert_eq!(parsed.spatial_psf_per_class, 8);
        assert_eq!(parsed.spatial_psf_grid_size, 4);
        assert_eq!(parsed.out_dir, PathBuf::from("tmp/source-audit"));

        let parsed = parse_args(&[
            "frame.fits".into(),
            "--proposal-mode=ensemble".into(),
            "--threshold-sigma".into(),
            "5.25".into(),
            "--enable-local-deblend".into(),
            "--keep-linear-artifacts".into(),
            "--max-sources=17".into(),
            "--out-dir".into(),
            "audit output".into(),
        ])
        .unwrap();
        assert_eq!(parsed.detection.proposal_mode, "ensemble");
        assert_eq!(parsed.detection.threshold_sigma, 5.25);
        assert!(parsed.detection.enable_local_deblend);
        assert!(!parsed.detection.reject_linear_artifacts);
        assert_eq!(parsed.detection.max_sources, Some(17));
        assert_eq!(parsed.out_dir, PathBuf::from("audit output"));
    }

    #[test]
    fn source_cli_rejects_invalid_modes_and_nonfinite_thresholds() {
        assert!(parse_args(&[
            "frame.fits".into(),
            "--proposal-mode".into(),
            "invalid".into(),
        ])
        .is_err());
        assert!(parse_args(&["frame.fits".into(), "--threshold-sigma=NaN".into(),]).is_err());
    }

    #[test]
    fn source_cli_runs_frozen_fits_and_writes_all_rust_artifacts() {
        let root = std::env::temp_dir().join(format!(
            "rst19-sources-cli-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let fits = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/single_frame_trail_audit_streak.fits");
        let summary = run_desktop(&[
            fits.to_string_lossy().into_owned(),
            "--out-dir".into(),
            root.to_string_lossy().into_owned(),
        ])
        .unwrap();
        assert!(summary["candidate_count"].as_u64().is_some());
        for name in [
            "source_catalog.csv",
            "source_feature_summary.csv",
            "source_feature_method_summary.csv",
            "source_feature_morphology.csv",
            "source_feature_subclass_summary.csv",
            "source_feature_spatial.csv",
            "source_spatial_grid.csv",
            "source_quality_summary.csv",
            "source_quality_flags.png",
            "source_feature_classes.png",
            "source_spatial_density.png",
            "source_snr_rank.png",
            "source_flux_snr_distribution.png",
            "source_cutout_manifest.csv",
            "source_cutout_contact_sheet.png",
            "source_feature_psf_similarity.csv",
            "source_feature_psf_spatial.csv",
        ] {
            assert!(root.join(name).is_file(), "missing Rust artifact {name}");
        }
        let csv_references: [(&str, &[u8]); 11] = [
            (
                "source_catalog.csv",
                include_bytes!("../tests/fixtures/source_audit_parity/source_catalog.csv"),
            ),
            (
                "source_cutout_manifest.csv",
                include_bytes!("../tests/fixtures/source_audit_parity/source_cutout_manifest.csv"),
            ),
            (
                "source_feature_method_summary.csv",
                include_bytes!(
                    "../tests/fixtures/source_audit_parity/source_feature_method_summary.csv"
                ),
            ),
            (
                "source_feature_morphology.csv",
                include_bytes!(
                    "../tests/fixtures/source_audit_parity/source_feature_morphology.csv"
                ),
            ),
            (
                "source_feature_psf_similarity.csv",
                include_bytes!(
                    "../tests/fixtures/source_audit_parity/source_feature_psf_similarity.csv"
                ),
            ),
            (
                "source_feature_psf_spatial.csv",
                include_bytes!(
                    "../tests/fixtures/source_audit_parity/source_feature_psf_spatial.csv"
                ),
            ),
            (
                "source_feature_spatial.csv",
                include_bytes!("../tests/fixtures/source_audit_parity/source_feature_spatial.csv"),
            ),
            (
                "source_feature_subclass_summary.csv",
                include_bytes!(
                    "../tests/fixtures/source_audit_parity/source_feature_subclass_summary.csv"
                ),
            ),
            (
                "source_feature_summary.csv",
                include_bytes!("../tests/fixtures/source_audit_parity/source_feature_summary.csv"),
            ),
            (
                "source_quality_summary.csv",
                include_bytes!("../tests/fixtures/source_audit_parity/source_quality_summary.csv"),
            ),
            (
                "source_spatial_grid.csv",
                include_bytes!("../tests/fixtures/source_audit_parity/source_spatial_grid.csv"),
            ),
        ];
        for (name, reference) in csv_references {
            assert_csv_matches_reference(name, &root.join(name), reference);
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
