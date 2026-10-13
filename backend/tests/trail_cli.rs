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
    assert_eq!(row.get(index("candidate_count")).unwrap(), "21");
    assert_eq!(row.get(index("quality_count")).unwrap(), "3");
    assert_eq!(row.get(index("trail_count")).unwrap(), "0");
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
