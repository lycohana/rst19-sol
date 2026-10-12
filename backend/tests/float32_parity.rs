use rst19_core::{filtering32, statistics::BackgroundOptions, statistics32};
use serde_json::Value;
fn numbers(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn compare(values: &[f32], expected: &Value, name: &str) {
    let expected = numbers(expected);
    assert_eq!(values.len(), expected.len());
    for (i, (a, b)) in values.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{name}[{i}]: actual={a:.9}, expected={b:.9}"
        );
    }
}
#[test]
fn float32_reduction_and_clipping_match_numpy_rounding() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/float32.json")).unwrap();
    for case in fixture["statistics"].as_array().unwrap() {
        let values = numbers(&case["values"]);
        let mask: Vec<_> = case["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        assert_eq!(
            statistics32::pairwise_sum(&values).to_bits(),
            (case["sum"].as_f64().unwrap() as f32).to_bits(),
            "sum length {}",
            values.len()
        );
        assert_eq!(
            statistics32::median(&values).unwrap().to_bits(),
            (case["median"].as_f64().unwrap() as f32).to_bits(),
            "median length {}",
            values.len()
        );
        assert_eq!(
            statistics32::stddev(&values).to_bits(),
            (case["std"].as_f64().unwrap() as f32).to_bits(),
            "std length {}",
            values.len()
        );
        let (center, noise) =
            statistics32::sigma_clipped_stats(&values, Some(&mask), 3., 4, 1_000_000).unwrap();
        assert_eq!(center, case["center"].as_f64().unwrap());
        assert!((noise - case["noise"].as_f64().unwrap()).abs() < 1e-12);
    }
}
#[test]
fn float32_background_convolution_and_snr_match_stage_by_stage() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/float32.json")).unwrap();
    for case in fixture["stages"].as_array().unwrap() {
        let image = numbers(&case["image"]);
        let valid: Vec<_> = case["valid"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        let mask: Vec<_> = valid.iter().map(|v| !*v).collect();
        let (h, w) = (
            case["shape"][0].as_u64().unwrap() as usize,
            case["shape"][1].as_u64().unwrap() as usize,
        );
        let (b, n) = statistics32::local_background_rms(
            &image,
            &mask,
            h,
            w,
            BackgroundOptions {
                box_size: case["box_size"].as_u64().unwrap() as usize,
                fallback: Some((
                    case["global_stats"][0].as_f64().unwrap(),
                    case["global_stats"][1].as_f64().unwrap(),
                )),
                iterations: 2,
                sample_limit: 100_000,
            },
        )
        .unwrap();
        compare(&b, &case["background"], "background");
        compare(&n, &case["noise"], "noise");
        let residual: Vec<_> = image
            .iter()
            .zip(&b)
            .zip(&valid)
            .map(|((v, b), ok)| if *ok { v - b } else { 0. })
            .collect();
        compare(&residual, &case["residual"], "residual");
        let fast = case["fast"].as_bool().unwrap();
        let (g, mode) = filtering32::masked_gaussian_response(
            &residual,
            &valid,
            h,
            w,
            case["sigma"].as_f64().unwrap(),
            fast,
        )
        .unwrap();
        compare(&g, &case["gaussian"], "Gaussian");
        assert_eq!(mode, case["gaussian_mode"].as_str().unwrap());
        let (s, mode) =
            filtering32::masked_starlet_smooth(&residual, &valid, h, w, 2, fast).unwrap();
        compare(&s, &case["starlet"], "Starlet");
        assert_eq!(mode, case["starlet_mode"].as_str().unwrap());
        let (snr, center, scale) =
            filtering32::standardize_response(&g, &mask, &n, &valid, 1_000_000).unwrap();
        assert_eq!(center, case["response_center"].as_f64().unwrap());
        assert!(
            (scale - case["response_noise"].as_f64().unwrap()).abs() < 1e-12,
            "response noise: actual={scale}, expected={}",
            case["response_noise"]
        );
        compare(&snr, &case["snr"], "SNR");
    }
}
