use rst19_core::sequence_temporal::{registered_coadd_reference, registered_median_reference};
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
fn registered_median_and_coadd_references_match_python() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/sequence_reference.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let request = &case["request"];
        let operation = case["operation"].as_str().unwrap();
        let actual = match operation {
            "sequence.reference.median" => registered_median_reference(request),
            "sequence.reference.coadd" => registered_coadd_reference(request),
            _ => panic!("unknown operation {operation}"),
        }
        .unwrap();
        compare(&actual, &case["expected"], case["label"].as_str().unwrap());
    }
}
