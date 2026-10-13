use rst19_core::{
    contaminated_pair_injection::{run, ContaminatedPairAnchor, Options},
    fits,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

const PYTHON_REFERENCE_SHA256: &str =
    "517f7e568a304da390aab34465a78d3bedd71854ae52144f5940547faa4859d9";
const FITS_FIXTURE_SHA256: &str =
    "02a9303a2494cb46dae4d7f5f44d817cbf92f217f405e5da5997aac7008d3c42";

fn compare_json(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            let tolerance = 2e-5 + 2e-6 * actual.abs().max(expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "{path}: Rust {actual:?} differs from Python {expected:?} by more than {tolerance}"
            );
        }
        (Value::Object(actual), Value::Object(expected)) => {
            let actual_keys: Vec<_> = actual.keys().collect();
            let expected_keys: Vec<_> = expected.keys().collect();
            assert_eq!(actual_keys, expected_keys, "field set differs at {path}");
            for key in actual.keys() {
                compare_json(&actual[key], &expected[key], &format!("{path}.{key}"));
            }
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(
                actual.len(),
                expected.len(),
                "array length differs at {path}"
            );
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                compare_json(actual, expected, &format!("{path}[{index}]"));
            }
        }
        _ => assert_eq!(actual, expected, "value differs at {path}"),
    }
}

#[test]
fn contaminated_pair_injection_matches_frozen_python_reference() {
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let frame_path = fixture_dir.join("contaminated_pair_injection_parity.fits");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let reference_path = root.join("src/rst19/contaminated_pair_injection.py");
    let reference_hash = format!("{:x}", Sha256::digest(fs::read(reference_path).unwrap()));
    assert_eq!(
        reference_hash, PYTHON_REFERENCE_SHA256,
        "Python reference changed; refresh the frozen fixture before accepting parity"
    );
    let fits_hash = format!("{:x}", Sha256::digest(fs::read(&frame_path).unwrap()));
    assert_eq!(fits_hash, FITS_FIXTURE_SHA256, "FITS fixture changed");
    let frame = fits::read_fits(&frame_path, false).unwrap();
    let options = Options {
        separations_px: vec![6.0],
        secondary_to_primary_ratios: vec![1.0],
        total_peak_excess_adu: 2048.0,
        proposal_mode: "gaussian".into(),
        analysis_scope: "local_roi".into(),
        roi_half_size_px: 20,
        ..Default::default()
    };
    let anchor = ContaminatedPairAnchor {
        name: "synthetic".into(),
        x: 64.0,
        y: 112.0,
        direction_x: 1.0,
        direction_y: 0.0,
    };
    let actual = serde_json::to_value(run(&frame, &[anchor], &options).unwrap()).unwrap();
    let expected_path = fixture_dir.join("contaminated_pair_injection_parity.json");
    let expected_text = fs::read_to_string(expected_path).unwrap();
    let mut expected: Value = serde_json::from_str(&expected_text).unwrap();
    let frame_label = frame.path.to_string_lossy().into_owned();
    expected["source_path"] = json!(frame_label);
    expected["rows"][0]["source_path"] = json!(frame_label);
    compare_json(&actual, &expected, "result");
}
