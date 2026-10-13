use rst19_core::single_frame_trails::single_frame_trail_proposals;
use serde_json::Value;
use std::io::Write;
use std::process::{Command, Stdio};

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!((a - b).abs() <= 2e-8 + 2e-9 * b.abs(), "{path}: {a} != {b}");
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
                "{path} object keys"
            );
            for (key, value) in b {
                compare(&a[key], value, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

#[test]
fn single_frame_trail_proposals_match_python_reference() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/single_frame_trail_proposals.json")).unwrap();
    assert_eq!(
        fixture["source_sha256"],
        "172c8852496fb0ea3cb0176d3726e28921366f912932f9e45f024fe6486841e5"
    );
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let actual = single_frame_trail_proposals(&case["request"]).unwrap();
        compare(&actual, &case["expected"], label);
    }
    for case in fixture["errors"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let actual = single_frame_trail_proposals(&case["request"]).unwrap_err();
        assert_eq!(actual, case["expected_error"].as_str().unwrap(), "{label}");
    }
}

#[test]
fn single_frame_trail_proposals_are_available_through_jsonl_engine() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/single_frame_trail_proposals.json")).unwrap();
    assert_eq!(
        fixture["source_sha256"],
        "172c8852496fb0ea3cb0176d3726e28921366f912932f9e45f024fe6486841e5"
    );
    let case = &fixture["cases"][0];
    let mut request = case["request"].clone();
    request["operation"] = Value::String("trails.single_frame_proposals".into());
    let mut child = Command::new(env!("CARGO_BIN_EXE_rst19-engine"))
        .env("PATH", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.take().unwrap(), "{request}").unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["ok"], true);
    compare(
        &envelope["payload"],
        &case["expected"],
        "jsonl_engine.single_frame_trail_proposals",
    );
}
