use rst19_core::sequence_trails::link_motion_evidence_tracks;
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
fn motion_evidence_links_match_python() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/motion_evidence_link.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let actual = link_motion_evidence_tracks(&case["request"]).unwrap();
        compare(&actual, &case["expected"], label);
    }
}

#[test]
fn motion_evidence_links_reject_invalid_thresholds() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/motion_evidence_link.json")).unwrap();
    let base = &fixture["cases"][0]["request"];
    for (field, value, expected) in [
        (
            "min_overlap_fraction",
            serde_json::json!(0.0),
            "min_overlap_fraction must be in (0, 1]",
        ),
        (
            "min_overlap_fraction",
            serde_json::json!(1.01),
            "min_overlap_fraction must be in (0, 1]",
        ),
        (
            "max_perpendicular_distance_px",
            serde_json::json!(0.0),
            "motion evidence link thresholds must be positive",
        ),
        (
            "max_direction_difference_deg",
            serde_json::json!(0.0),
            "motion evidence link thresholds must be positive",
        ),
    ] {
        let mut request = base.clone();
        request[field] = value;
        assert_eq!(link_motion_evidence_tracks(&request).unwrap_err(), expected);
    }
}
