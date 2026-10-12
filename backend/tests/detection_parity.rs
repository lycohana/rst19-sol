use rst19_core::{
    detection::{self, DetectionOptions},
    models::PixelData,
};
use serde_json::Value;

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            let tolerance = if path.contains("deblend_") {
                2e-6 + 2e-7 * b.abs()
            } else {
                2e-9 + 2e-11 * b.abs()
            };
            assert!(
                (a - b).abs() <= tolerance,
                "{path}: actual={a:.17} expected={b:.17}"
            );
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}");
            for (i, (a, b)) in a.iter().zip(b).enumerate() {
                compare(a, b, &format!("{path}[{i}]"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                b.keys().collect::<Vec<_>>(),
                "{path}"
            );
            for (k, a) in a {
                compare(a, &b[k], &format!("{path}.{k}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}
#[test]
fn all_default_detection_modes_match_python_including_integer_masks_quality_and_metadata() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/detection.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let values = case["image"].as_array().unwrap();
        let data = if case["dtype"] == "int16" {
            PixelData::I16(values.iter().map(|v| v.as_i64().unwrap() as i16).collect())
        } else if case["dtype"] == "float32" {
            PixelData::F32(values.iter().map(|v| v.as_f64().unwrap() as f32).collect())
        } else {
            PixelData::F64(values.iter().map(|v| v.as_f64().unwrap()).collect())
        };
        let mask: Vec<_> = case["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        let options: DetectionOptions = serde_json::from_value(case["options"].clone()).unwrap();
        let model = case["background_model"].as_object().map(|m| {
            (
                m["background"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<_>>(),
                m["noise"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<_>>(),
                m["height"].as_u64().unwrap() as usize,
                m["width"].as_u64().unwrap() as usize,
            )
        });
        let model = model
            .as_ref()
            .map(|(b, n, h, w)| detection::BackgroundModel {
                background: b,
                noise: n,
                height: *h,
                width: *w,
            });
        let result = detection::detect_sources(
            &data,
            Some(&mask),
            case["height"].as_u64().unwrap() as usize,
            case["width"].as_u64().unwrap() as usize,
            &options,
            model.as_ref(),
        )
        .unwrap();
        compare(
            &result.as_dict(),
            &case["expected"],
            case["name"].as_str().unwrap(),
        );
    }
}
#[test]
fn precision_dispatch_honors_requested_type_and_rejects_invalid_limits() {
    let data = PixelData::I16(vec![100; 256]);
    let options = DetectionOptions {
        use_float32: true,
        ..Default::default()
    };
    let result = detection::detect_sources(&data, None, 16, 16, &options, None).unwrap();
    assert_eq!(result.parameters["use_float32"], 1);
    let options = DetectionOptions {
        max_sources: Some(0),
        ..Default::default()
    };
    assert!(
        detection::detect_sources(&data, None, 16, 16, &options, None)
            .unwrap_err()
            .contains("max_sources")
    );
}
