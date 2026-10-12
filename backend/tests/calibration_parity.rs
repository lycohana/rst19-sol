use rst19_core::{
    catalog::CatalogSource,
    matching::CatalogMatch,
    measurement::Detection,
    photometric_calibration::{fit, CalibrationOptions},
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
fn weighted_calibration_clipping_holdout_quality_roles_and_application_match_python() {
    let f: Value = serde_json::from_str(include_str!("fixtures/calibration.json")).unwrap();
    for case in f["cases"].as_array().unwrap() {
        let matches: Vec<CatalogMatch> = serde_json::from_value(case["matches"].clone()).unwrap();
        let detections: Vec<Detection> =
            serde_json::from_value(case["detections"].clone()).unwrap();
        let catalog: Vec<CatalogSource> = serde_json::from_value(case["catalog"].clone()).unwrap();
        let options: CalibrationOptions = serde_json::from_value(case["options"].clone()).unwrap();
        let actual = fit(&matches, &detections, &catalog, &options).unwrap();
        let name = case["name"].as_str().unwrap();
        compare(&actual.as_dict(), &case["expected"], name);
        for c in case["application"].as_array().unwrap() {
            compare(
                &serde_json::to_value(
                    actual.apply(c["instrumental"].as_f64().unwrap(), c["color"].as_f64()),
                )
                .unwrap(),
                &c["expected"],
                name,
            );
        }
        for c in case["errors"].as_array().unwrap() {
            compare(
                &serde_json::to_value(
                    actual.magnitude_error(
                        c["instrumental_error"].as_f64(),
                        c["color_error"].as_f64(),
                    ),
                )
                .unwrap(),
                &c["expected"],
                name,
            );
        }
    }
}
#[test]
fn invalid_calibration_parameters_match_python_rejections() {
    let f: Value = serde_json::from_str(include_str!("fixtures/calibration.json")).unwrap();
    for case in f["invalid_options"].as_array().unwrap() {
        let result = serde_json::from_value::<CalibrationOptions>(case.clone())
            .map_err(|e| e.to_string())
            .and_then(|o| fit(&[], &[], &[], &o));
        assert!(result.is_err(), "{case}");
    }
}
