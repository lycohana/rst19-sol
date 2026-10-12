use rst19_core::sequence_motion_features::motion_feature_frame_components;
use serde_json::Value;

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            let tolerance = 2e-8 + 2e-10 * b.abs();
            assert!((a - b).abs() <= tolerance, "{path}: {a} != {b}");
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path} length");
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

#[test]
fn motion_feature_components_match_python_reference() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/sequence_motion_components.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let actual = motion_feature_frame_components(&case["request"]).unwrap();
        compare(&actual, &case["expected"], label);
    }
    for case in fixture["errors"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let actual = motion_feature_frame_components(&case["request"]).unwrap_err();
        assert_eq!(actual, case["expected_error"].as_str().unwrap(), "{label}");
    }
}
