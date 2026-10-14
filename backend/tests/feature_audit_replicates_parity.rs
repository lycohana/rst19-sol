use rst19_core::feature_audit_replicates::run_feature_audit_replicates;
use serde_json::Value;
use std::path::Path;

fn assert_json_close(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(
                actual.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>(),
                "object keys differ at {path}"
            );
            for (key, expected_value) in expected {
                assert_json_close(&actual[key], expected_value, &format!("{path}.{key}"));
            }
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(
                actual.len(),
                expected.len(),
                "array length differs at {path}"
            );
            for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
                assert_json_close(actual_value, expected_value, &format!("{path}[{index}]"));
            }
        }
        (Value::Number(actual), Value::Number(expected)) => {
            if let (Some(actual), Some(expected)) = (actual.as_f64(), expected.as_f64()) {
                assert!(
                    (actual - expected).abs() <= 1e-12,
                    "numeric value differs at {path}: {actual} != {expected}"
                );
            } else {
                assert_eq!(actual, expected, "number differs at {path}");
            }
        }
        _ => assert_eq!(actual, expected, "value differs at {path}"),
    }
}

#[test]
fn rust_one_trial_matches_frozen_python_result() {
    let reference = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/feature_audit_replicates_parity/reference.json");
    let expected: Value = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let actual = run_feature_audit_replicates(1, 19_019, |_, _| {}).unwrap();
    assert_json_close(&actual.as_value(), &expected, "$");
}
