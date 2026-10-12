use rst19_core::photometry;
use serde_json::Value;
fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!(
                (a - b).abs() <= 2e-11 + 2e-11 * b.abs(),
                "{path}: actual={a}, expected={b}"
            );
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}");
            for (i, (a, b)) in a.iter().zip(b).enumerate() {
                compare(a, b, &format!("{path}[{i}]"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len(), "{path}");
            for (k, b) in b {
                compare(&a[k], b, &format!("{path}.{k}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}
#[test]
fn arithmetic_and_absolute_magnitude_status_error_and_provenance_match_python() {
    let f: Value = serde_json::from_str(include_str!("fixtures/photometry.json")).unwrap();
    for (i, case) in f["cases"].as_array().unwrap().iter().enumerate() {
        let result = photometry::evaluate(
            case["kind"].as_str().unwrap(),
            &case["parameters"],
            case.get("source"),
        );
        if case["error"].as_bool().unwrap() {
            assert!(result.is_err(), "case {i}: accepted {case}");
        } else {
            compare(&result.unwrap(), &case["expected"], &format!("case {i}"));
        }
    }
}
#[test]
fn unknown_photometric_labels_and_unicode_casefold_match_python() {
    let f: Value = serde_json::from_str(include_str!("fixtures/photometry.json")).unwrap();
    for case in f["labels"].as_array().unwrap() {
        assert_eq!(
            serde_json::to_value(photometry::normalise_label(case["input"].as_str())).unwrap(),
            case["expected"]
        );
    }
    assert!(photometry::metadata_matches(
        Some("Straße"),
        Some("g"),
        Some("STRASSE"),
        Some("G")
    ));
    assert!(!photometry::metadata_matches(
        Some("Gaia"),
        Some("G"),
        Some("Gaia"),
        Some("V")
    ));
}
