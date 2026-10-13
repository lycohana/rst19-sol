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
    std::env::temp_dir().join(format!("rst19-aperture-peak-ratio-{label}-{nonce}"))
}

#[test]
fn standalone_rust_cli_runs_without_python_and_writes_parity_artifacts() {
    let root = temp_dir("cli-test");
    fs::create_dir_all(&root).unwrap();
    let catalog = root.join("source catalog.csv");
    fs::write(
        &catalog,
        "detection_id,feature_class,peak,flux,proposal_methods\n1,range,4,24,gaussian|dog_narrow\n2,range,10,20,gaussian\n3,crowded,2,50,gaussian\n4,crowded,5,5,dog_narrow\n",
    )
    .unwrap();
    let out_dir = root.join("output");
    let output = Command::new(env!("CARGO_BIN_EXE_rst19-aperture-peak-ratio"))
        .env("PATH", "")
        .arg(&catalog)
        .arg("--target-ids")
        .arg("1")
        .arg("--out-dir")
        .arg(&out_dir)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("1: range, flux/peak=6.000000, class percentile=1.000000"));
    for artifact in [
        "aperture_peak_ratio_summary.csv",
        "aperture_peak_ratio_method_summary.csv",
        "aperture_peak_ratio_targets.csv",
        "aperture_peak_ratio.json",
    ] {
        assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
    }
    let summary = fs::read(out_dir.join("aperture_peak_ratio_summary.csv")).unwrap();
    assert!(summary.starts_with(&[0xef, 0xbb, 0xbf]));
    let json: serde_json::Value =
        serde_json::from_slice(&fs::read(out_dir.join("aperture_peak_ratio.json")).unwrap())
            .unwrap();
    assert_eq!(json["target_rows"][0]["detection_id"], "1");
    assert_eq!(json["target_rows"][0]["class_percentile"], 1.0);

    let help = Command::new(env!("CARGO_BIN_EXE_rst19-aperture-peak-ratio"))
        .env("PATH", "")
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("Usage: rst19-aperture-peak-ratio"));
    fs::remove_dir_all(root).unwrap();
}
