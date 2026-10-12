use rst19_core::attitude::{audit_frames, AttitudeFrameInput};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/attitude.json")).unwrap()
}

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!(
                (a - b).abs() <= 1e-7 + 1e-10 * b.abs(),
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
            assert_eq!(a.len(), b.len(), "{path}");
            for (key, value) in b {
                compare(&a[key], value, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

#[test]
fn auxiliary_boresight_audit_and_summary_match_python_reference() {
    for case in fixture()["cases"].as_array().unwrap() {
        let frames: Vec<AttitudeFrameInput> =
            serde_json::from_value(case["frames"].clone()).unwrap();
        let axis: Option<Vec<f64>> = if case["camera_body_axis"].is_null() {
            None
        } else {
            Some(serde_json::from_value(case["camera_body_axis"].clone()).unwrap())
        };
        let actual = audit_frames(&frames, axis.as_deref()).unwrap();
        compare(&actual, &case["expected"], case["label"].as_str().unwrap());
    }
}
