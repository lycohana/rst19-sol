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
    std::env::temp_dir().join(format!("rst19-pair-class-{label}-{nonce}"))
}

#[test]
fn standalone_class_audit_cli_runs_without_python_and_writes_all_artifacts() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fits = workspace
        .join("tests")
        .join("fixtures")
        .join("contaminated_pair_injection_parity.fits");
    let root = temp_dir("cli-test");
    fs::create_dir_all(&root).unwrap();
    let catalog = root.join("source catalog.csv");
    fs::write(
        &catalog,
        "detection_id,feature_class,feature_class_label,x,y,filter_snr,flux_snr,peak,quality_passed,flags\n1,linear_artifact,line,64,112,20,10,100,false,LINE_ARTIFACT\n",
    )
    .unwrap();
    let out_dir = root.join("output");
    let output = Command::new(env!("CARGO_BIN_EXE_rst19-contaminated-pair-class"))
        .env("PATH", "")
        .arg(&fits)
        .arg("--source-catalog")
        .arg(&catalog)
        .arg("--feature-class")
        .arg("linear_artifact")
        .arg("--separations-px")
        .arg("6")
        .arg("--ratios")
        .arg("1")
        .arg("--total-peak-adu")
        .arg("2048")
        .arg("--analysis-scope")
        .arg("local_roi")
        .arg("--roi-half-size-px")
        .arg("20")
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
    assert!(stdout.contains("linear_artifact: ID 1"), "{stdout}");
    let audit: serde_json::Value = serde_json::from_slice(
        &fs::read(out_dir.join("contaminated_pair_class_audit.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(audit["selected_rows"][0]["detection_id"], 1);
    assert_eq!(audit["injection"]["rows"].as_array().unwrap().len(), 1);
    assert_eq!(
        audit["class_summary"][0]["feature_class"],
        "linear_artifact"
    );
    assert_eq!(audit["quality_reason_summary"][0]["endpoint_count"], 2);
    for artifact in [
        "contaminated_pair_injection.csv",
        "contaminated_pair_injection.json",
        "contaminated_pair_class_selection.csv",
        "contaminated_pair_class_summary.csv",
        "contaminated_pair_quality_reason_summary.csv",
    ] {
        assert!(out_dir.join(artifact).is_file(), "missing {artifact}");
    }

    let help = Command::new(env!("CARGO_BIN_EXE_rst19-contaminated-pair-class"))
        .env("PATH", "")
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("Usage: rst19-contaminated-pair-class"));
    fs::remove_dir_all(root).unwrap();
}
