use rst19_core::{
    catalog::CatalogSource,
    matching::Projection,
    models::{FitsFrame, PixelData},
    pipeline::{analyze_frame, recalibrate_frame_analysis, FrameOptions, RecalibrationOptions},
};
use serde_json::Value;
fn compare(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            let tolerance = if path.contains("deblend_") {
                2e-6 + 2e-7 * b.abs()
            } else {
                2e-9 + 2e-11 * b.abs()
            };
            assert!(
                (a - b).abs() <= tolerance,
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
fn full_single_frame_pipeline_and_recalibration_match_python() {
    let f: Value = serde_json::from_str(include_str!("fixtures/pipeline.json")).unwrap();
    for case in f["cases"].as_array().unwrap() {
        let item = &f["frames"][case["frame"].as_u64().unwrap() as usize];
        let pixels: Vec<f64> = serde_json::from_value(item["image"].clone()).unwrap();
        let frame = FitsFrame {
            path: item["path"].as_str().unwrap().into(),
            header: serde_json::from_value(item["header"].clone()).unwrap(),
            data: if item["dtype"] == "int16" {
                PixelData::I16(pixels.into_iter().map(|v| v as i16).collect())
            } else {
                PixelData::F64(pixels)
            },
            auxiliary: None,
            data_offset: 0,
            width: 97,
            height: 96,
        };
        let catalog: Option<Vec<CatalogSource>> =
            serde_json::from_value(case["catalog"].clone()).unwrap();
        let wcs: Option<Projection> = serde_json::from_value(case["wcs"].clone()).unwrap();
        let options: FrameOptions = serde_json::from_value(case["options"].clone()).unwrap();
        let mut result =
            analyze_frame(frame, catalog.as_deref(), wcs.as_ref(), &options, None).unwrap();
        let name = case["name"].as_str().unwrap();
        compare(&result.as_dict(), &case["expected"], name);
        if !case["recalibration"].is_null() {
            let r = &case["recalibration"];
            let catalog: Vec<CatalogSource> = serde_json::from_value(r["catalog"].clone()).unwrap();
            let wcs: Projection = serde_json::from_value(r["wcs"].clone()).unwrap();
            let options: RecalibrationOptions =
                serde_json::from_value(r["options"].clone()).unwrap();
            let before = result.detection.as_dict();
            recalibrate_frame_analysis(&mut result, &catalog, &wcs, &options).unwrap();
            assert_eq!(before, result.detection.as_dict());
            compare(
                &result.as_dict(),
                &r["expected"],
                &format!("{name} recalibration"),
            );
        }
    }
}
