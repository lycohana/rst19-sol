use rst19_core::sequence_trails::fit_collinear_motion_geometry;
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
fn collinear_motion_geometry_matches_python() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/collinear_motion_geometry.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let actual = fit_collinear_motion_geometry(&case["request"]).unwrap();
        compare(&actual, &case["expected"], label);
    }
}
