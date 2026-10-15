use rst19_core::sequence_cli::{analyze_command, parse_args};
use serde_json::{json, Value};
use std::{fs, path::PathBuf};

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}

#[test]
fn legacy_flags_reach_native_frame_track_and_recovery_options() {
    let command = parse_args(&args(&[
        "frames",
        "--source-limit=0",
        "--float64",
        "--relative-photometry",
        "--relative-max-sources",
        "12",
        "--relative-spatial-order=2",
        "--keep-linear-artifacts",
        "--motion-line-link-radius-px",
        "70",
        "--temporal-multiscale",
        "--persistent-min-presence",
        "4",
    ]))
    .unwrap();
    assert_eq!(command.request["sequence_max_sources"], Value::Null);
    assert_eq!(command.request["frame_options"]["use_float32"], false);
    assert_eq!(
        command.request["frame_options"]["reject_linear_artifacts"],
        false
    );
    assert_eq!(
        command.request["tracking_options"]["persistent_min_presence"],
        4
    );
    assert_eq!(command.request["relative_photometry_max_sources"], 12);
    assert_eq!(command.request["relative_photometry_spatial_order"], 2);
    assert_eq!(command.request["motion_line_link_radius_px"], 70.0);
    assert_eq!(command.request["temporal_multiscale"], true);
    for values in [
        vec!["frames", "--source-limit=-1"],
        vec!["frames", "--workers=0"],
        vec!["frames", "--relative-spatial-order=3"],
        vec!["frames", "--temporal-multiscale=true"],
        vec!["frames", "--psf-fwhm=NaN"],
        vec!["frames", "--unknown-option"],
    ] {
        assert!(parse_args(&args(&values)).is_err());
    }
}

#[test]
fn public_sequence_command_writes_json_and_original_frame_audit() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = root
        .join("target/test-artifacts")
        .join(format!("sequence-cli-{}", std::process::id()));
    fs::create_dir_all(dir.join("input")).unwrap();
    let fixture = root.join("tests/fixtures/contaminated_pair_injection_parity.fits");
    for name in ["02.fits", "01.fits"] {
        fs::copy(&fixture, dir.join("input").join(name)).unwrap();
    }
    let output = dir.join("results/sequence.json");
    let audit = dir.join("results/audit");
    let command = parse_args(&vec![
        dir.join("input").to_string_lossy().into_owned(),
        "--json-out".into(),
        output.to_string_lossy().into_owned(),
        "--audit-out-dir".into(),
        audit.to_string_lossy().into_owned(),
        "--no-stack-faint".into(),
        "--source-limit=100".into(),
        "--background-box-size=64".into(),
    ])
    .unwrap();
    let result = analyze_command(&command).unwrap();
    let saved: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(result, saved);
    assert_eq!(result["frame_count"], 2);
    assert!(result["frames"][0]["path"]
        .as_str()
        .unwrap()
        .ends_with("01.fits"));
    assert!(result["motion_features"].is_array());
    assert_eq!(result["stack_faint_candidate_count"], 0);
    assert!(audit.join("source_track_contact_sheet.png").is_file());
    let summary: Value =
        serde_json::from_slice(&fs::read(audit.join("source_track_audit_summary.json")).unwrap())
            .unwrap();
    assert_eq!(summary["frame_count"], json!(2));
    let dir = fs::canonicalize(dir).unwrap();
    assert!(dir.starts_with(fs::canonicalize(root.join("target/test-artifacts")).unwrap()));
    fs::remove_dir_all(dir).unwrap();
}
