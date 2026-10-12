use rst19_core::{
    filtering::MergedProposal,
    measurement::{self, Detection, MeasurementContext, MeasurementOptions},
    psf,
};
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/measurement.json")).unwrap()
}
fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn mask(v: &Value) -> Vec<bool> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect()
}
fn compare(actual: &Value, expected: &Value, path: &str, relative: f64, absolute: f64) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!(
                (a - b).abs() <= absolute + relative * b.abs(),
                "{path}: actual {a:.17}, expected {b:.17}"
            );
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}");
            for (i, (a, b)) in a.iter().zip(b).enumerate() {
                compare(a, b, &format!("{path}[{i}]"), relative, absolute);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                b.keys().collect::<Vec<_>>(),
                "{path}"
            );
            for (k, a) in a {
                compare(a, &b[k], &format!("{path}.{k}"), relative, absolute);
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}
#[test]
fn aperture_position_shape_noise_and_all_quality_flags_match_python() {
    for case in fixture()["measurements"].as_array().unwrap() {
        let mut image = nums(&case["image"]);
        for index in case["nonfinite"].as_array().unwrap() {
            image[index.as_u64().unwrap() as usize] = f64::NAN;
        }
        let mask = mask(&case["mask"]);
        let line = case["line_mask"]
            .as_array()
            .map(|_| self::mask(&case["line_mask"]));
        let background = nums(&case["background"]);
        let noise = nums(&case["noise"]);
        let image32: Vec<_> = image.iter().map(|v| *v as f32).collect();
        let ctx = MeasurementContext {
            image: if case["dtype"] == "float32" {
                image32.as_slice().into()
            } else {
                image.as_slice().into()
            },
            mask: &mask,
            line_mask: line.as_deref(),
            background_map: background.as_slice().into(),
            noise_map: noise.as_slice().into(),
            height: case["height"].as_u64().unwrap() as usize,
            width: case["width"].as_u64().unwrap() as usize,
        };
        let options: MeasurementOptions = serde_json::from_value(case["options"].clone()).unwrap();
        let proposal: MergedProposal = serde_json::from_value(case["proposal"].clone()).unwrap();
        let peak = [
            case["peak"][0].as_u64().unwrap() as usize,
            case["peak"][1].as_u64().unwrap() as usize,
        ];
        let actual = measurement::source_from_peak(
            &ctx,
            peak,
            &options,
            &proposal,
            case["filter_snr"].as_f64().unwrap(),
        )
        .unwrap();
        compare(
            &serde_json::to_value(actual).unwrap(),
            &case["expected"],
            case["name"].as_str().unwrap(),
            2e-12,
            1e-10,
        );
    }
}
#[test]
fn fixed_single_pair_psf_evidence_matches_python() {
    for (i, case) in fixture()["pairs"].as_array().unwrap().iter().enumerate() {
        let image = nums(&case["image"]);
        let invalid = mask(&case["mask"]);
        let p = nums(&case["primary"]);
        let s = nums(&case["secondary"]);
        let evidence = psf::pair_psf_evidence(
            &image,
            &invalid,
            case["height"].as_u64().unwrap() as usize,
            case["width"].as_u64().unwrap() as usize,
            [p[0], p[1]],
            [s[0], s[1]],
            case["fwhm"].as_f64().unwrap(),
        )
        .unwrap();
        compare(
            &serde_json::to_value(evidence).unwrap(),
            &case["expected"],
            &format!("PSF {i}"),
            2e-7,
            2e-6,
        );
    }
}
#[test]
fn close_gaussian_pair_guard_preserves_candidates_and_only_rejects_weak_duplicates() {
    for case in fixture()["guards"].as_array().unwrap() {
        let image = nums(&case["image"]);
        let invalid = mask(&case["mask"]);
        let sources: Vec<Detection> = serde_json::from_value(case["sources"].clone()).unwrap();
        let (output, tested, guarded, radius) = psf::guard_unresolved_gaussian_pairs(
            &sources,
            &image,
            &invalid,
            case["height"].as_u64().unwrap() as usize,
            case["width"].as_u64().unwrap() as usize,
            psf::PairGuardOptions {
                psf_fwhm: 3.0,
                delta_bic_min: 10.0,
                component_snr_min: 5.0,
            },
        )
        .unwrap();
        assert_eq!(tested, case["tested"].as_u64().unwrap() as usize);
        assert_eq!(guarded, case["guarded"].as_u64().unwrap() as usize);
        assert_eq!(radius, case["radius"].as_f64().unwrap());
        compare(
            &serde_json::to_value(output).unwrap(),
            &case["expected"],
            "guard",
            2e-7,
            2e-6,
        );
    }
}
