use rst19_core::{
    photometric_calibration::PhotometricCalibration,
    source_photometry::{build_table, find_faintest, FaintestOptions, TableOptions},
};
use serde_json::Value;
fn compare(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!(
                (a - b).abs() <= 2e-10 + 2e-11 * b.abs(),
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
        _ => assert_eq!(a, b, "{path}"),
    }
}
#[test]
fn source_table_status_flags_metadata_model_fields_and_absolute_evidence_match_python() {
    let f: Value = serde_json::from_str(include_str!("fixtures/source_photometry.json")).unwrap();
    for case in f["tables"].as_array().unwrap() {
        let sources: Vec<rst19_core::measurement::Detection> =
            serde_json::from_value(case["sources"].clone()).unwrap();
        let matches: Vec<rst19_core::matching::CatalogMatch> =
            serde_json::from_value(case["matches"].clone()).unwrap();
        let catalog: Vec<rst19_core::catalog::CatalogSource> =
            serde_json::from_value(case["catalog"].clone()).unwrap();
        let calibration: Option<PhotometricCalibration> =
            serde_json::from_value(case["calibration"].clone()).unwrap();
        let options: TableOptions = serde_json::from_value(case["options"].clone()).unwrap();
        let rows =
            build_table(&sources, &matches, &catalog, calibration.as_ref(), &options).unwrap();
        compare(
            &Value::Array(rows.iter().map(|r| r.as_dict()).collect()),
            &case["expected"],
            case["name"].as_str().unwrap(),
        );
    }
}
#[test]
fn faintest_ranking_partial_scope_consistency_rate_and_ties_match_python() {
    let f: Value = serde_json::from_str(include_str!("fixtures/source_photometry.json")).unwrap();
    for case in f["faintest"].as_array().unwrap() {
        let sources: Vec<rst19_core::measurement::Detection> =
            serde_json::from_value(case["sources"].clone()).unwrap();
        let calibration: Option<PhotometricCalibration> =
            serde_json::from_value(case["calibration"].clone()).unwrap();
        let options: FaintestOptions = serde_json::from_value(case["options"].clone()).unwrap();
        let result = find_faintest(&sources, calibration.as_ref(), &options).unwrap();
        compare(
            &result.map(|r| r.as_dict()).unwrap_or(Value::Null),
            &case["expected"],
            case["name"].as_str().unwrap(),
        );
    }
}
