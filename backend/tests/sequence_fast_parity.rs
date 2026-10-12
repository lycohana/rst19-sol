use rst19_core::{
    measurement::Detection,
    sequence_fast::{track_fast_point_movers, FastPointOptions},
};
use serde_json::Value;

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!(
                (a - b).abs() <= 2e-9 + 2e-11 * b.abs(),
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
fn fast_quality_point_motion_tracks_match_python_reference() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/sequence_fast.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let frames: Vec<Vec<Detection>> = serde_json::from_value(case["frames"].clone()).unwrap();
        let shifts: Vec<[f64; 2]> =
            serde_json::from_value(case["cumulative_shifts"].clone()).unwrap();
        let times: Option<Vec<f64>> = serde_json::from_value(case["frame_times"].clone()).unwrap();
        let options: FastPointOptions = serde_json::from_value(case["options"].clone()).unwrap();
        let existing = case["existing_tracks"].as_array().unwrap();
        let actual =
            track_fast_point_movers(&frames, &shifts, times.as_deref(), &options, existing)
                .unwrap();
        compare(&actual, &case["expected"], case["label"].as_str().unwrap());
    }
}
