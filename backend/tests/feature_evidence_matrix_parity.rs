use rst19_core::feature_evidence_matrix::{self, FeatureEvidenceMatrixResult};
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
    serde_json::from_str(include_str!("fixtures/feature_evidence_matrix.json")).unwrap()
}

fn write_inputs(
    directory: &Path,
    fixture: &Value,
) -> (
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    let inputs = &fixture["inputs"];
    let names = ["morphology.csv", "cross.csv", "psf.csv", "spatial.csv"];
    let mut paths = Vec::new();
    for name in names {
        let path = directory.join(name);
        fs::write(&path, format!("\u{feff}{}", inputs[name].as_str().unwrap())).unwrap();
        paths.push(path);
    }
    (
        paths.remove(0),
        paths.remove(0),
        paths.remove(0),
        paths.remove(0),
    )
}

fn normalize_paths(result: &mut Value) {
    result["source_morphology_path"] = "<MORPHOLOGY>".into();
    result["feature_cross_path"] = "<CROSS>".into();
    result["psf_leaveout_path"] = "<PSF>".into();
    result["psf_spatial_path"] = "<SPATIAL>".into();
}

fn result_value(result: &FeatureEvidenceMatrixResult) -> Value {
    let mut value = result.as_dict();
    normalize_paths(&mut value);
    value
}

#[test]
fn evidence_matrix_matches_python_fields_and_csv_bytes() {
    let fixture = fixture();
    let directory = std::env::temp_dir().join(format!(
        "rst19-feature-evidence-parity-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&directory).unwrap();
    let (morphology, cross, psf, spatial) = write_inputs(&directory, &fixture);
    let result =
        feature_evidence_matrix::run(&morphology, &cross, &psf, 0.8, Some(&spatial)).unwrap();
    compare(&result_value(&result), &fixture["expected"], "result");
    assert_eq!(result.rows.len(), 9);
    assert_eq!(result.rows.last().unwrap().feature_class, "other_rejected");

    let output = directory.join("artifacts");
    feature_evidence_matrix::write_artifacts(&result, &output).unwrap();
    let csv = fs::read(output.join("feature_evidence_matrix.csv")).unwrap();
    assert_eq!(
        String::from_utf8(csv).unwrap(),
        fixture["artifact_csv"].as_str().unwrap()
    );
    assert!(output.join("feature_evidence_matrix.json").is_file());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn evidence_matrix_native_cli_and_jsonl_work_without_python_path() {
    let fixture = fixture();
    let directory = std::env::temp_dir().join(format!(
        "rst19-feature-evidence-native-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&directory).unwrap();
    let (morphology, cross, psf, spatial) = write_inputs(&directory, &fixture);
    let output_dir = directory.join("cli-output");
    let args = vec![
        "--source-morphology".into(),
        morphology.to_string_lossy().into_owned(),
        "--feature-cross".into(),
        cross.to_string_lossy().into_owned(),
        "--psf-leaveout".into(),
        psf.to_string_lossy().into_owned(),
        "--psf-spatial".into(),
        spatial.to_string_lossy().into_owned(),
        "--out-dir".into(),
        output_dir.to_string_lossy().into_owned(),
    ];
    let cli = feature_evidence_matrix::run_desktop(&args).unwrap();
    assert_eq!(cli["result"]["rows"].as_array().unwrap().len(), 9);
    let standalone_dir = directory.join("standalone-output");
    let status = Command::new(env!("CARGO_BIN_EXE_rst19-feature-evidence"))
        .args([
            "--source-morphology",
            morphology.to_string_lossy().as_ref(),
            "--feature-cross",
            cross.to_string_lossy().as_ref(),
            "--psf-leaveout",
            psf.to_string_lossy().as_ref(),
            "--psf-spatial",
            spatial.to_string_lossy().as_ref(),
            "--out-dir",
            standalone_dir.to_string_lossy().as_ref(),
        ])
        .env("PATH", "")
        .status()
        .unwrap();
    assert!(status.success());

    let engine_output = directory.join("engine-output");
    let mut engine = Command::new(env!("CARGO_BIN_EXE_rst19-engine"))
        .env("PATH", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let request = serde_json::json!({
        "operation":"audit.feature_evidence_matrix",
        "source_morphology_path":morphology,
        "feature_cross_path":cross,
        "psf_leaveout_path":psf,
        "psf_spatial_path":spatial,
        "output_dir":engine_output
    });
    writeln!(engine.stdin.take().unwrap(), "{request}").unwrap();
    let output = engine.wait_with_output().unwrap();
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["ok"], true);
    assert_eq!(
        envelope["payload"]["result"]["rows"]
            .as_array()
            .unwrap()
            .len(),
        9
    );
    fs::remove_dir_all(directory).unwrap();
}
