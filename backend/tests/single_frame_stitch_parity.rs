use rst19_core::sequence_trails::stitch_unassociated_single_frame_points;
use serde_json::Value;

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!(
                (a - b).abs() <= 2e-8 + 2e-10 * b.abs(),
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
fn single_frame_stitch_matches_python() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/single_frame_stitch.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let actual = stitch_unassociated_single_frame_points(&case["request"]).unwrap();
        compare(&actual, &case["expected"], label);
    }
}

#[test]
fn stitching_rejects_nonpositive_controls() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/single_frame_stitch.json")).unwrap();
    let base = &fixture["cases"][0]["request"];
    for (field, value, expected) in [
        (
            "psf_fwhm",
            serde_json::json!(0.0),
            "stitching parameters must be positive",
        ),
        (
            "max_frame_gap",
            serde_json::json!(0),
            "stitching parameters must be positive",
        ),
        (
            "max_prediction_error_px",
            serde_json::json!(-1.0),
            "max_prediction_error_px must be positive",
        ),
    ] {
        let mut request = base.clone();
        request[field] = value;
        assert_eq!(
            stitch_unassociated_single_frame_points(&request).unwrap_err(),
            expected
        );
    }
}
