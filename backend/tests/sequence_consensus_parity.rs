use rst19_core::sequence_consensus::candidate_consensus;
use serde_json::Value;

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!(
                (a - b).abs() <= 2e-9 + 2e-11 * b.abs(),
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
            assert_eq!(a.len(), b.len(), "{path}");
            for (key, value) in b {
                compare(&a[key], value, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

#[test]
fn geometry_only_candidate_consensus_matches_python_reference() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/sequence_consensus.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let candidates = case["candidate_peak_frames"].as_array().unwrap();
        let shifts: Vec<[f64; 2]> =
            serde_json::from_value(case["cumulative_shifts"].clone()).unwrap();
        let quality_tracks = case["quality_tracks"].as_array().unwrap();
        let frame_times: Option<Vec<f64>> =
            serde_json::from_value(case["frame_times"].clone()).unwrap();
        let options = &case["options"];
        let actual = candidate_consensus(
            candidates,
            &shifts,
            quality_tracks,
            frame_times.as_deref(),
            options["link_radius_px"].as_f64().unwrap(),
            options["min_presence"].as_u64().unwrap() as usize,
            options["persistent_min_presence"].as_u64().unwrap() as usize,
            options["motion_min_displacement_px"].as_f64().unwrap(),
            options["max_motion_fit_rms_px"].as_f64().unwrap(),
            options["min_candidate_snr"].as_f64().unwrap(),
            options
                .get("temporal_candidate_min_snr")
                .and_then(Value::as_f64),
            options
                .get("temporal_reference_min_snr")
                .and_then(Value::as_f64),
            options
                .get("temporal_min_psf_correlation")
                .and_then(Value::as_f64)
                .unwrap_or(0.8),
        )
        .unwrap();
        compare(&actual, &case["expected"], case["label"].as_str().unwrap());
    }
}
