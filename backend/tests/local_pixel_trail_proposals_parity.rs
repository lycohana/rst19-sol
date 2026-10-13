use rst19_core::single_frame_trails::local_pixel_trail_proposals;
use serde_json::Value;

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
fn local_pixel_trail_proposals_match_python() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/local_pixel_trail_proposals.json")).unwrap();
    assert_eq!(
        fixture["source_hashes"]["single_frame_trails.py"],
        "172c8852496fb0ea3cb0176d3726e28921366f912932f9e45f024fe6486841e5"
    );
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let actual = local_pixel_trail_proposals(&case["request"]).unwrap();
        compare(&actual, &case["expected"], label);
    }
    for case in fixture["errors"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let actual = local_pixel_trail_proposals(&case["request"]).unwrap_err();
        assert_eq!(actual, case["expected_error"].as_str().unwrap(), "{label}");
    }
}
