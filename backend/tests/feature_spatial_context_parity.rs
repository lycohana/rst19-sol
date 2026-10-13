use rst19_core::feature_spatial_context::{
    self, FeatureSpatialContextOptions, SPATIAL_CONTEXT_METRIC_FIELDS,
};
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

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/feature_spatial_context.json")).unwrap()
}

fn write_catalog(directory: &Path, fixture: &Value) -> std::path::PathBuf {
    let catalog = directory.join("source_catalog.csv");
    fs::write(
        &catalog,
        format!("\u{feff}{}", fixture["catalog_csv"].as_str().unwrap()),
    )
    .unwrap();
    catalog
}

#[test]
fn spatial_context_matches_python_result_and_csv_artifacts() {
    let fixture = fixture();
    let directory = std::env::temp_dir().join(format!(
        "rst19-spatial-context-parity-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&directory).unwrap();
    let catalog = write_catalog(&directory, &fixture);
    let excluded = vec!["other_rejected".to_owned()];
    let options = FeatureSpatialContextOptions {
        image_width: 100,
        image_height: 100,
        grid_size: 2,
        edge_margin_px: 5.0,
        high_flux_snr_threshold: 10.0,
        top_n: 1,
        target_ids: vec![2],
        exclude_classes: excluded,
        ..FeatureSpatialContextOptions::default()
    };
    let result = feature_spatial_context::run(&catalog, &options).unwrap();
    let mut actual = result.as_dict();
    actual["catalog_path"] = "<CATALOG>".into();
    compare(&actual, &fixture["expected"], "result");
    assert_eq!(
        SPATIAL_CONTEXT_METRIC_FIELDS,
        ["flux_snr", "filter_snr", "peak"]
    );

    let output = directory.join("artifacts");
    feature_spatial_context::write_artifacts(&result, &output).unwrap();
    for name in [
        "feature_spatial_context_summary.csv",
        "feature_spatial_context_hotspots.csv",
        "feature_spatial_context_targets.csv",
    ] {
        assert_eq!(
            String::from_utf8(fs::read(output.join(name)).unwrap()).unwrap(),
            fixture["artifacts"][name].as_str().unwrap(),
            "CSV artifact differs from the Python reference: {name}"
        );
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn spatial_context_non_square_grid_validation_and_native_routes() {
    let fixture = fixture();
    let directory = std::env::temp_dir().join(format!(
        "rst19-spatial-context-native-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&directory).unwrap();
    let catalog = write_catalog(&directory, &fixture);
    let excluded = vec!["other_rejected".to_owned()];
    let nonsquare_options = FeatureSpatialContextOptions {
        image_width: 200,
        image_height: 100,
        grid_size: 4,
        top_n: 1,
        metric: "filter_snr".into(),
        target_ids: vec![3],
        ..FeatureSpatialContextOptions::default()
    };
    let nonsquare = feature_spatial_context::run(&catalog, &nonsquare_options).unwrap();
    let target = nonsquare
        .targets
        .iter()
        .find(|target| target.detection_id == 3)
        .unwrap();
    assert_eq!(target.grid_row, 3);
    assert_eq!(target.grid_column, 1);
    let missing_target_options = FeatureSpatialContextOptions {
        image_width: 100,
        image_height: 100,
        grid_size: 2,
        edge_margin_px: 5.0,
        top_n: 1,
        target_ids: vec![99],
        exclude_classes: excluded.clone(),
        ..FeatureSpatialContextOptions::default()
    };
    assert!(
        feature_spatial_context::run(&catalog, &missing_target_options)
            .unwrap_err()
            .contains("找不到 detection_id")
    );

    let output_dir = directory.join("cli-output");
    let args = vec![
        catalog.to_string_lossy().into_owned(),
        "--image-width=100".into(),
        "--image-height=100".into(),
        "--grid-size=2".into(),
        "--edge-margin-px=5".into(),
        "--high-flux-snr=10".into(),
        "--top-n=1".into(),
        "--target-id=2".into(),
        "--out-dir".into(),
        output_dir.to_string_lossy().into_owned(),
    ];
    let cli = feature_spatial_context::run_desktop(&args).unwrap();
    assert_eq!(cli["result"]["targets"].as_array().unwrap().len(), 2);
    let standalone_dir = directory.join("standalone-output");
    let status = Command::new(env!("CARGO_BIN_EXE_rst19-feature-spatial-context"))
        .args([
            catalog.to_string_lossy().as_ref(),
            "--image-width=100",
            "--image-height=100",
            "--grid-size=2",
            "--edge-margin-px=5",
            "--high-flux-snr=10",
            "--top-n=1",
            "--target-id=2",
            "--out-dir",
            standalone_dir.to_string_lossy().as_ref(),
        ])
        .env("PATH", "")
        .status()
        .unwrap();
    assert!(status.success());
    assert!(standalone_dir
        .join("feature_spatial_context.json")
        .is_file());

    let engine_output = directory.join("engine-output");
    let mut engine = Command::new(env!("CARGO_BIN_EXE_rst19-engine"))
        .env("PATH", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let request = serde_json::json!({
        "operation":"audit.feature_spatial_context",
        "catalog_path":catalog,
        "image_width":100,
        "image_height":100,
        "grid_size":2,
        "edge_margin_px":5.0,
        "high_flux_snr_threshold":10.0,
        "top_n":1,
        "target_ids":[2],
        "output_dir":engine_output
    });
    writeln!(engine.stdin.take().unwrap(), "{request}").unwrap();
    let response = engine.wait_with_output().unwrap();
    assert!(response.status.success());
    let envelope: Value = serde_json::from_slice(&response.stdout).unwrap();
    assert_eq!(envelope["ok"], true);
    assert_eq!(
        envelope["payload"]["result"]["targets"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert!(engine_output.join("feature_spatial_context.json").is_file());
    fs::remove_dir_all(directory).unwrap();
}
