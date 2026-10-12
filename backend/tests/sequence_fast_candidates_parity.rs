use rst19_core::sequence_fast_candidates::{
    associate_fast_candidates, FastCandidateOptions, FastCandidateRecord,
};
use serde_json::Value;

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            assert!(
                (actual - expected).abs() <= 2e-9 + 2e-11 * expected.abs(),
                "{path}: {actual} != {expected}"
            );
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{path}");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                compare(actual, expected, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{path}");
            for (key, expected) in expected {
                compare(&actual[key], expected, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

#[test]
fn raw_fast_candidate_association_matches_python_reference() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/sequence_fast_candidates.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let frames: Vec<Vec<FastCandidateRecord>> =
            serde_json::from_value(case["request"]["frames"].clone()).unwrap();
        let times: Option<Vec<f64>> =
            serde_json::from_value(case["request"]["frame_times"].clone()).unwrap();
        let options: FastCandidateOptions =
            serde_json::from_value(case["request"]["options"].clone()).unwrap();
        let actual = associate_fast_candidates(&frames, times.as_deref(), &options).unwrap();
        compare(&actual, &case["expected"], case["label"].as_str().unwrap());
    }
}
