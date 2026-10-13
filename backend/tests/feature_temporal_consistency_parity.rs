use rst19_core::feature_temporal_consistency_audit::{self, FEATURE_CLASS_SOURCE_CHOICES};
use serde_json::Value;
use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!((a - b).abs() <= 2e-7 + 2e-8 * b.abs(), "{path}: {a} != {b}");
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                compare(a, b, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                b.keys().collect::<Vec<_>>(),
                "{path} keys"
            );
            for (key, value) in b {
                compare(&a[key], value, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

fn write_fixture_inputs(
    directory: &Path,
    fixture: &Value,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let catalog = directory.join("source_catalog.csv");
    let metrics = directory.join("frame_metrics.csv");
    fs::write(
        &catalog,
        format!("\u{feff}{}", fixture["catalog_csv"].as_str().unwrap()),
    )
    .unwrap();
    fs::write(
        &metrics,
        format!("\u{feff}{}", fixture["frame_metrics_csv"].as_str().unwrap()),
    )
    .unwrap();
    (catalog, metrics)
}

#[test]
fn temporal_consistency_matches_python_reference_fields() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/feature_temporal_consistency.json")).unwrap();
    let directory = std::env::temp_dir().join(format!(
        "rst19-temporal-parity-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&directory).unwrap();
    let (catalog, metrics) = write_fixture_inputs(&directory, &fixture);
    let actual =
        feature_temporal_consistency_audit::run(&catalog, &metrics, 3, None, "catalog", None)
            .unwrap();
    let mut actual = actual.as_dict();
    actual["catalog_path"] = "<CATALOG>".into();
    actual["frame_metrics_path"] = "<FRAME_METRICS>".into();
    compare(&actual, &fixture["expected"], "result");
    assert_eq!(FEATURE_CLASS_SOURCE_CHOICES, ["catalog", "frame"]);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn local_classes_alias_cli_engine_and_artifacts_are_native() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/feature_temporal_consistency.json")).unwrap();
    let directory = std::env::temp_dir().join(format!(
        "rst19-temporal-native-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&directory).unwrap();
    let (catalog, metrics) = write_fixture_inputs(&directory, &fixture);
    let local_metrics = directory.join("local_metrics.csv");
    fs::write(
        &local_metrics,
        "detection_id,frame_index,feature_class,flux_snr_raw,negative_extreme_count,repeated_3990_3993_count,zero_count\n1,1,range_anomaly,10,0,0,0\n1,2,range_anomaly,11,0,0,0\n1,3,range_anomaly,9,0,0,0\n2,1,crowded_blend,5,0,0,0\n2,2,crowded_blend,-1,1,0,1\n2,3,crowded_blend,5,0,1,0\n",
    )
    .unwrap();
    let selected = feature_temporal_consistency_audit::run(
        &catalog,
        &local_metrics,
        3,
        None,
        "frame",
        Some(&[2]),
    )
    .unwrap();
    assert_eq!(selected.selected_detection_ids, [2]);
    assert_eq!(selected.feature_classes, ["crowded_blend"]);
    assert!(feature_temporal_consistency_audit::run(
        &catalog,
        &local_metrics,
        3,
        None,
        "catalog",
        None,
    )
    .unwrap_err()
    .contains("feature_class mismatch"));
    let duplicate = fs::read_to_string(&metrics).unwrap() + "1,1,10,0,0,0\n";
    fs::write(&metrics, duplicate).unwrap();
    assert!(
        feature_temporal_consistency_audit::run(&catalog, &metrics, 3, None, "catalog", None)
            .unwrap_err()
            .contains("duplicate detection_id=1, frame_index=1")
    );
    let (catalog, metrics) = write_fixture_inputs(&directory, &fixture);

    let output_dir = directory.join("cli-output");
    let args = vec![
        catalog.to_string_lossy().into_owned(),
        metrics.to_string_lossy().into_owned(),
        "--expected-frames=3".into(),
        "--feature-class-source=catalog".into(),
        "--out-dir".into(),
        output_dir.to_string_lossy().into_owned(),
    ];
    let cli = feature_temporal_consistency_audit::run_desktop(&args).unwrap();
    assert_eq!(cli["result"]["source_rows"].as_array().unwrap().len(), 2);
    for artifact in [
        "feature_temporal_consistency_audit.json",
        "feature_temporal_consistency_sources.csv",
        "feature_temporal_consistency_summary.csv",
    ] {
        assert!(output_dir.join(artifact).is_file(), "missing {artifact}");
    }
    for artifact in [
        "feature_temporal_consistency_sources.csv",
        "feature_temporal_consistency_summary.csv",
    ] {
        assert_eq!(
            fs::read(output_dir.join(artifact)).unwrap(),
            fixture["artifacts"][artifact].as_str().unwrap().as_bytes(),
            "CSV artifact differs from the Python reference: {artifact}"
        );
    }

    let standalone_output = directory.join("standalone-output");
    let status = Command::new(env!(
        "CARGO_BIN_EXE_rst19-feature-temporal-consistency-audit"
    ))
    .args([
        catalog.to_string_lossy().as_ref(),
        metrics.to_string_lossy().as_ref(),
        "--expected-frames=3",
        "--out-dir",
        standalone_output.to_string_lossy().as_ref(),
    ])
    .env("PATH", "")
    .status()
    .unwrap();
    assert!(status.success(), "standalone Rust command failed");
    assert!(standalone_output
        .join("feature_temporal_consistency_audit.json")
        .is_file());

    let engine_output = directory.join("engine-output");
    let mut engine = Command::new(env!("CARGO_BIN_EXE_rst19-engine"))
        .env("PATH", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let request = serde_json::json!({
        "operation":"audit.feature_temporal_consistency",
        "catalog_path":catalog,
        "frame_metrics_path":metrics,
        "expected_frame_count":3,
        "output_dir":engine_output
    });
    writeln!(engine.stdin.take().unwrap(), "{request}").unwrap();
    let output = engine.wait_with_output().unwrap();
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["ok"], true);
    assert_eq!(
        envelope["payload"]["result"]["source_rows"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    fs::remove_dir_all(directory).unwrap();
}
