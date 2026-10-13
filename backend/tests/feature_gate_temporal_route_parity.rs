use rst19_core::feature_gate_temporal_route;
use serde_json::Value;
use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/feature_gate_temporal_route.json")).unwrap()
}

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!(
                (a - b).abs() <= 2e-14 + 2e-14 * b.abs(),
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

fn setup() -> (
    Value,
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    let fixture = fixture();
    let root = std::env::temp_dir().join(format!(
        "rst19-feature-gate-temporal-parity-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let route = root.join("route.csv");
    let diagnostic = root.join("diagnostic.csv");
    let persistence = root.join("persistence.json");
    for (path, name) in [
        (&route, "route.csv"),
        (&diagnostic, "diagnostic.csv"),
        (&persistence, "persistence.json"),
    ] {
        fs::write(path, fixture["inputs"][name].as_str().unwrap()).unwrap();
    }
    (fixture, root, route, diagnostic, persistence)
}

fn normalize_paths(value: &mut Value) {
    value["route_csv"] = "<ROUTE>".into();
    value["diagnostic_sources_csv"] = "<DIAGNOSTIC>".into();
    value["persistence_json"] = "<PERSISTENCE>".into();
}

#[test]
fn feature_gate_temporal_route_matches_python_fields_and_csv_artifacts() {
    let (fixture, root, route, diagnostic, persistence) = setup();
    let targets = vec!["2".to_owned(), "2".to_owned(), "3".to_owned()];
    let result =
        feature_gate_temporal_route::run(&route, &diagnostic, &persistence, &targets).unwrap();
    let mut actual = result.as_dict();
    normalize_paths(&mut actual);
    compare(&actual, &fixture["expected"], "result");
    let output = root.join("artifacts");
    feature_gate_temporal_route::write_artifacts(&result, &output).unwrap();
    for name in [
        "feature_gate_temporal_route_sources.csv",
        "feature_gate_temporal_route_summary.csv",
        "feature_gate_temporal_subgroup_summary.csv",
        "feature_gate_temporal_targets.csv",
    ] {
        assert_eq!(
            fs::read(output.join(name)).unwrap(),
            format!("\u{feff}{}", fixture["artifacts"][name].as_str().unwrap()).as_bytes(),
            "{name}"
        );
    }
    let artifact_json: Value = serde_json::from_slice(
        &fs::read(output.join("feature_gate_temporal_routes.json")).unwrap(),
    )
    .unwrap();
    let mut artifact_json = artifact_json;
    normalize_paths(&mut artifact_json);
    let expected_artifact: Value = serde_json::from_str(
        fixture["artifacts"]["feature_gate_temporal_routes.json"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    compare(&artifact_json, &expected_artifact, "artifact json");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn standalone_cli_and_jsonl_join_without_python_path() {
    let (fixture, root, route, diagnostic, persistence) = setup();
    let output_dir = root.join("standalone");
    let status = Command::new(env!("CARGO_BIN_EXE_rst19-feature-gate-temporal-route"))
        .args([
            route.to_string_lossy().as_ref(),
            diagnostic.to_string_lossy().as_ref(),
            persistence.to_string_lossy().as_ref(),
            "--target-id",
            "2",
            "--target-id",
            "3",
            "--out-dir",
            output_dir.to_string_lossy().as_ref(),
        ])
        .env("PATH", "")
        .status()
        .unwrap();
    assert!(status.success());

    let engine_output = root.join("engine");
    let mut engine = Command::new(env!("CARGO_BIN_EXE_rst19-engine"))
        .env("PATH", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let request = serde_json::json!({
        "operation":"audit.feature_gate_temporal_route",
        "route_csv_path":route,
        "diagnostic_sources_csv_path":diagnostic,
        "persistence_json_path":persistence,
        "target_ids":["2","2","3"],
        "output_dir":engine_output
    });
    writeln!(engine.stdin.take().unwrap(), "{request}").unwrap();
    let output = engine.wait_with_output().unwrap();
    assert!(output.status.success());
    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["ok"], true);
    assert_eq!(response["payload"]["result"]["source_row_count"], 3);
    assert_eq!(
        response["payload"]["result"]["target_rows"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        response["payload"]["result"]["frame_count"],
        fixture["expected"]["frame_count"]
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejected_route_without_temporal_coverage_is_rejected() {
    let (_fixture, root, route, diagnostic, persistence) = setup();
    let text = fs::read_to_string(&diagnostic).unwrap();
    let without_weak_row = text
        .lines()
        .filter(|line| !line.starts_with("3,"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(&diagnostic, without_weak_row).unwrap();
    let error =
        feature_gate_temporal_route::run(&route, &diagnostic, &persistence, &[]).unwrap_err();
    assert!(
        error.contains("missing temporal row for rejected route candidate 3"),
        "{error}"
    );
    fs::remove_dir_all(root).unwrap();
}

fn _path_type_anchor(_: &Path) {}
