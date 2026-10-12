use rst19_core::{
    catalog::CatalogSource,
    matching::{match_detections, DetectionPosition, Projection},
};
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/matching.json")).unwrap()
}
#[test]
fn offline_csv_aliases_provenance_quality_and_errors_match_python() {
    let f = fixture();
    for (i, case) in f["csv"].as_array().unwrap().iter().enumerate() {
        let result =
            rst19_core::catalog_csv::load_catalog_bytes(case["csv"].as_str().unwrap().as_bytes());
        if case["error"].as_bool().unwrap() {
            assert!(result.is_err(), "CSV case {i} accepted: {case}");
        } else {
            compare(
                &serde_json::to_value(result.unwrap()).unwrap(),
                &case["expected"],
                &format!("CSV {i}"),
            );
        }
    }
}
fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!(
                (a - b).abs() <= 1e-8 + 1e-11 * b.abs(),
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
fn complete_catalog_model_validation_and_epoch_propagation_match_python() {
    let f = fixture();
    for (i, case) in f["propagation"].as_array().unwrap().iter().enumerate() {
        let source: CatalogSource = serde_json::from_value(case["source"].clone()).unwrap();
        let source = source.validated().unwrap();
        assert_eq!(
            source.extinction_compatibility(),
            case["extinction_compatibility"].as_str().unwrap()
        );
        let propagated = source.at_epoch(case["epoch"].as_f64()).unwrap();
        compare(
            &serde_json::to_value(propagated).unwrap(),
            &case["expected"],
            &format!("propagation {i}"),
        );
    }
    for case in f["invalid_sources"].as_array().unwrap() {
        let result = serde_json::from_value::<CatalogSource>(case.clone())
            .map_err(|e| e.to_string())
            .and_then(CatalogSource::validated);
        assert!(result.is_err(), "invalid source accepted: {case}");
    }
}
#[test]
fn complete_tan_and_affine_catalog_matching_matches_python() {
    let f = fixture();
    for (i, case) in f["matching"].as_array().unwrap().iter().enumerate() {
        let detections: Vec<DetectionPosition> =
            serde_json::from_value(case["detections"].clone()).unwrap();
        let catalog: Vec<CatalogSource> = serde_json::from_value(case["catalog"].clone()).unwrap();
        let wcs: Projection = serde_json::from_value(case["wcs"].clone()).unwrap();
        let actual = match_detections(
            &detections,
            &catalog,
            &wcs,
            case["radius"].as_f64().unwrap(),
            case["epoch"].as_f64(),
            case["mode"].as_str().unwrap(),
        )
        .unwrap();
        compare(
            &serde_json::to_value(actual).unwrap(),
            &case["expected"],
            &format!("matching {i}"),
        );
    }
}
