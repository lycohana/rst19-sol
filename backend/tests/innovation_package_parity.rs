use rst19_core::innovation_package;
use serde_json::Value;
use std::{
    fs,
    io::Write,
    process::{Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/innovation_package.json")).unwrap()
}

fn setup() -> (
    Value,
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    let fixture = fixture();
    let root = std::env::temp_dir().join(format!(
        "rst19-innovation-package-parity-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let injection = root.join("injection.json");
    let sequence = root.join("sequence.json");
    let feature = root.join("feature.json");
    fs::write(
        &injection,
        fixture["inputs"]["injection.json"].as_str().unwrap(),
    )
    .unwrap();
    fs::write(
        &sequence,
        fixture["inputs"]["sequence.json"].as_str().unwrap(),
    )
    .unwrap();
    fs::write(
        &feature,
        fixture["inputs"]["feature.json"].as_str().unwrap(),
    )
    .unwrap();
    (fixture, root, injection, sequence, feature)
}

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!(
                (a - b).abs() <= 2e-13 + 2e-13 * b.abs(),
                "{path}: {a} != {b}"
            );
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

fn normalize(package: &mut Value) {
    package["provenance"]["injection_artifact"]["path"] = "<INPUT>".into();
    for evidence in package["evidence"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .skip(1)
    {
        match evidence["id"].as_str().unwrap_or_default() {
            "sequence_relation" => evidence["source"]["path"] = "<SEQUENCE>".into(),
            "feature_routing" => evidence["source"]["path"] = "<FEATURE>".into(),
            _ => {}
        }
    }
}

#[test]
fn innovation_package_matches_python_fields_csv_and_markdown() {
    let (fixture, root, injection, sequence, feature) = setup();
    let package = innovation_package::build_innovation_package(
        &injection,
        Some(&sequence),
        Some(&feature),
        8,
        3,
        3,
    )
    .unwrap();
    let mut actual = package.clone();
    normalize(&mut actual);
    compare(&actual, &fixture["expected"], "package");
    let output = root.join("output");
    innovation_package::write_innovation_package(&package, &output).unwrap();
    assert_eq!(
        fs::read(output.join("innovation_delivery.csv")).unwrap(),
        format!(
            "\u{feff}{}",
            fixture["artifacts"]["innovation_delivery.csv"]
                .as_str()
                .unwrap()
        )
        .as_bytes()
    );
    assert_eq!(
        fs::read_to_string(output.join("innovation_delivery.md")).unwrap(),
        fixture["artifacts"]["innovation_delivery.md"]
            .as_str()
            .unwrap()
    );
    let mut artifact: Value =
        serde_json::from_slice(&fs::read(output.join("innovation_package.json")).unwrap()).unwrap();
    normalize(&mut artifact);
    compare(&artifact, &fixture["expected"], "artifact");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn standalone_cli_and_jsonl_builder_run_without_python_path() {
    let (fixture, root, injection, sequence, feature) = setup();
    let output_dir = root.join("standalone");
    let status = Command::new(env!("CARGO_BIN_EXE_rst19-innovation-package"))
        .args([
            injection.to_string_lossy().as_ref(),
            "--sequence-report",
            sequence.to_string_lossy().as_ref(),
            "--feature-matrix",
            feature.to_string_lossy().as_ref(),
            "--out-dir",
            output_dir.to_string_lossy().as_ref(),
            "--require-defensible",
        ])
        .env("PATH", "")
        .status()
        .unwrap();
    assert!(status.success());

    let engine_out = root.join("engine");
    let mut engine = Command::new(env!("CARGO_BIN_EXE_rst19-engine"))
        .env("PATH", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let request = serde_json::json!({
        "operation":"audit.innovation_package",
        "injection_artifact_path":injection,
        "sequence_report_path":sequence,
        "feature_matrix_path":feature,
        "output_dir":engine_out
    });
    writeln!(engine.stdin.take().unwrap(), "{request}").unwrap();
    let output = engine.wait_with_output().unwrap();
    assert!(output.status.success());
    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["ok"], true);
    assert_eq!(response["payload"]["status"], fixture["expected"]["status"]);
    assert_eq!(
        response["payload"]["package"]["rows"]
            .as_array()
            .unwrap()
            .len(),
        fixture["expected"]["rows"].as_array().unwrap().len()
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn primary_evidence_gates_cannot_be_lowered() {
    let (_fixture, root, injection, sequence, feature) = setup();
    let mut input: Value = serde_json::from_slice(&fs::read(&injection).unwrap()).unwrap();
    for row in input["rows"].as_array_mut().unwrap() {
        row["trial_count"] = 1.into();
    }
    let single_layout = root.join("single-layout.json");
    fs::write(&single_layout, serde_json::to_vec(&input).unwrap()).unwrap();
    let package = innovation_package::build_innovation_package(
        &single_layout,
        Some(&sequence),
        Some(&feature),
        1,
        1,
        1,
    )
    .unwrap();
    assert_eq!(package["status"], "DIAGNOSTIC_ONLY");
    assert_eq!(package["data_summary"]["min_injections_gate"], 8);
    assert_eq!(package["data_summary"]["min_levels_gate"], 3);
    assert_eq!(package["data_summary"]["min_trials_gate"], 3);
    assert!(innovation_package::wilson_interval(0, 0, 1.96).is_err());
    fs::remove_dir_all(root).unwrap();
}
