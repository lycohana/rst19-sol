use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn temp_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("rst19-trails-{label}-{nonce}"))
}

#[test]
fn standalone_rust_trail_cli_writes_expected_audit_artifacts() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("contaminated_pair_injection_parity.fits");
    let original = fs::read(&fixture).unwrap();
    let root = temp_dir("cli-test");
    let output = Command::new(env!("CARGO_BIN_EXE_rst19-trails"))
        .arg(&fixture)
        .arg("--out-dir")
        .arg(root.join("output"))
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("frame=01 trails=0"), "{stdout}");
    let out_dir = root.join("output");
    let mut reader = csv::Reader::from_path(out_dir.join("single_frame_trails.csv")).unwrap();
    let headers = reader.headers().unwrap().clone();
    let index = |name: &str| headers.iter().position(|value| value == name).unwrap();
    let row = reader.records().next().unwrap().unwrap();
    assert_eq!(row.get(index("timestamp")).unwrap(), "");
    assert_eq!(row.get(index("candidate_count")).unwrap(), "21");
    assert_eq!(row.get(index("quality_count")).unwrap(), "3");
    assert_eq!(row.get(index("trail_count")).unwrap(), "0");
    assert_eq!(row.get(index("max_length_px")).unwrap(), "");
    assert_eq!(row.get(index("max_width_px")).unwrap(), "");
    assert_eq!(row.get(index("max_residual_snr")).unwrap(), "");
    assert_eq!(row.get(index("edge_trail_count")).unwrap(), "0");
    assert_eq!(row.get(index("total_area_pixels")).unwrap(), "0");
    assert_eq!(row.get(index("longest_angle_deg")).unwrap(), "");
    assert_eq!(row.get(index("longest_bbox")).unwrap(), "");
    assert_eq!(row.get(index("accepted_trail_count")).unwrap(), "0");
    assert_eq!(row.get(index("double_star_count")).unwrap(), "0");
    assert_eq!(row.get(index("model_rejected_count")).unwrap(), "0");
    assert_eq!(fs::read(&fixture).unwrap(), original);
    for artifact in [
        "single_frame_trail_counts.png",
        "single_frame_trail_lengths.png",
    ] {
        assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
    }

    let help = Command::new(env!("CARGO_BIN_EXE_rst19-trails"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("Usage: rst19-trails"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn standalone_rust_trail_cli_matches_python_line_audit_without_python() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("single_frame_trail_audit_streak.fits");
    let original = fs::read(&fixture).unwrap();
    let root = temp_dir("line-parity");
    let out_dir = root.join("output");
    let output = Command::new(env!("CARGO_BIN_EXE_rst19-trails"))
        .env("PATH", "")
        .arg(&fixture)
        .args([
            "--residual-sigma",
            "5",
            "--min-residual-adu",
            "10",
            "--min-feature-area",
            "40",
            "--min-axis-ratio",
            "4",
            "--out-dir",
        ])
        .arg(&out_dir)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut reader = csv::Reader::from_path(out_dir.join("single_frame_trails.csv")).unwrap();
    let headers = reader.headers().unwrap().clone();
    let index = |name: &str| headers.iter().position(|value| value == name).unwrap();
    let row = reader.records().next().unwrap().unwrap();
    // Frozen from run_single_frame_trail_audit on the same FITS and options.
    // The geometry proposal is retained, while the single-frame model rejects it.
    assert_eq!(
        row.get(index("timestamp")).unwrap(),
        "2026-10-14T00:00:00.000"
    );
    assert_eq!(row.get(index("candidate_count")).unwrap(), "13");
    assert_eq!(row.get(index("quality_count")).unwrap(), "4");
    assert_eq!(row.get(index("trail_count")).unwrap(), "1");
    assert_eq!(row.get(index("max_length_px")).unwrap(), "");
    assert_eq!(row.get(index("max_width_px")).unwrap(), "");
    assert_eq!(row.get(index("max_residual_snr")).unwrap(), "");
    assert_eq!(row.get(index("edge_trail_count")).unwrap(), "0");
    assert_eq!(row.get(index("total_area_pixels")).unwrap(), "0");
    assert_eq!(row.get(index("longest_angle_deg")).unwrap(), "");
    assert_eq!(row.get(index("longest_bbox")).unwrap(), "");
    assert_eq!(row.get(index("accepted_trail_count")).unwrap(), "0");
    assert_eq!(row.get(index("double_star_count")).unwrap(), "0");
    assert_eq!(row.get(index("model_rejected_count")).unwrap(), "1");
    assert_eq!(fs::read(&fixture).unwrap(), original);
    for artifact in [
        "single_frame_trail_counts.png",
        "single_frame_trail_lengths.png",
    ] {
        assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
    }
    fs::remove_dir_all(root).unwrap();
}
