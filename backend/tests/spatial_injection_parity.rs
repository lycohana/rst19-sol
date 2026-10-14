use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

const PYTHON_REFERENCE_SHA256: &str =
    "7e34af2134ed908eef53dfb95b23f8c04473c4a88ac3375b8de13e576f69cb77";
const FITS_FIXTURE_SHA256: &str =
    "02a9303a2494cb46dae4d7f5f44d817cbf92f217f405e5da5997aac7008d3c42";
const FLOAT_ABSOLUTE_TOLERANCE: f64 = 2e-5;
const FLOAT_RELATIVE_TOLERANCE: f64 = 2e-6;

fn compare_json(actual: &Value, expected: &Value, path: &str) {
    if path.ends_with("source_path") {
        return;
    }
    if path.ends_with("site_condition_json") && actual.is_string() && expected.is_string() {
        let actual: Value = serde_json::from_str(actual.as_str().unwrap()).unwrap();
        let expected: Value = serde_json::from_str(expected.as_str().unwrap()).unwrap();
        compare_json(&actual, &expected, &format!("{path}<parsed>"));
        return;
    }
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            if let (Some(actual), Some(expected)) = (actual.as_i64(), expected.as_i64()) {
                assert_eq!(actual, expected, "integer differs at {path}");
                return;
            }
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            let allowed = FLOAT_ABSOLUTE_TOLERANCE
                + FLOAT_RELATIVE_TOLERANCE * actual.abs().max(expected.abs());
            assert!(
                (actual - expected).abs() <= allowed,
                "{path}: Rust {actual:?} differs from Python {expected:?} by more than {allowed}"
            );
        }
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(
                actual.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>(),
                "field set differs at {path}"
            );
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

fn compare_csv(actual: &[u8], expected: &[u8]) {
    let actual = actual.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(actual);
    let expected = expected
        .strip_prefix(&[0xef, 0xbb, 0xbf])
        .unwrap_or(expected);
    let mut actual_reader = csv::ReaderBuilder::new().from_reader(actual);
    let mut expected_reader = csv::ReaderBuilder::new().from_reader(expected);
    let actual_headers = actual_reader.headers().unwrap().clone();
    let expected_headers = expected_reader.headers().unwrap().clone();
    assert_eq!(actual_headers, expected_headers, "CSV headers differ");
    let actual_rows = actual_reader
        .records()
        .collect::<std::result::Result<Vec<_>, csv::Error>>()
        .unwrap();
    let expected_rows = expected_reader
        .records()
        .collect::<std::result::Result<Vec<_>, csv::Error>>()
        .unwrap();
    assert_eq!(
        actual_rows.len(),
        expected_rows.len(),
        "CSV row count differs"
    );
    for (row_index, (actual, expected)) in actual_rows.iter().zip(&expected_rows).enumerate() {
        assert_eq!(
            actual.len(),
            expected.len(),
            "CSV width differs at row {row_index}"
        );
        for (column, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let name = &actual_headers[column];
            if name == "source_path" {
                assert!(actual.ends_with("contaminated_pair_injection_parity.fits"));
            } else if name == "site_condition_json" {
                let actual: Value = serde_json::from_str(actual).unwrap();
                let expected: Value = serde_json::from_str(expected).unwrap();
                compare_json(&actual, &expected, "csv.site_condition_json");
            } else if let (Ok(actual), Ok(expected)) =
                (actual.parse::<i64>(), expected.parse::<i64>())
            {
                assert_eq!(actual, expected, "CSV {name} row {row_index}");
            } else if let (Ok(actual), Ok(expected)) =
                (actual.parse::<f64>(), expected.parse::<f64>())
            {
                let allowed = FLOAT_ABSOLUTE_TOLERANCE
                    + FLOAT_RELATIVE_TOLERANCE * actual.abs().max(expected.abs());
                assert!(
                    (actual - expected).abs() <= allowed,
                    "CSV {name} row {row_index}: Rust {actual} differs from Python {expected}"
                );
            } else {
                assert_eq!(actual, expected, "CSV {name} row {row_index}");
            }
        }
    }
}

fn run_and_compare(args: &[&str], expected_json: &[u8], expected_csv: &[u8]) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("tests/fixtures/contaminated_pair_injection_parity.fits");
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "rst19-spatial-injection-parity-{}-{nonce}",
        std::process::id()
    ));
    let status = Command::new(env!("CARGO_BIN_EXE_rst19-spatial-injection"))
        .env("PATH", "")
        .arg(&fixture)
        .args(args)
        .arg("--out-dir")
        .arg(&out_dir)
        .status()
        .expect("Rust CLI must run without Python on PATH");
    assert!(
        status.success(),
        "Rust spatial-injection CLI failed with Python removed from PATH"
    );

    let actual_json_path = out_dir.join("spatial_injection_audit.json");
    let actual: Value = serde_json::from_slice(&fs::read(&actual_json_path).unwrap()).unwrap();
    let mut expected: Value = serde_json::from_slice(expected_json).unwrap();
    expected["source_path"] = actual["source_path"].clone();
    for (actual_row, expected_row) in actual["rows"]
        .as_array()
        .unwrap()
        .iter()
        .zip(expected["rows"].as_array_mut().unwrap())
    {
        expected_row["source_path"] = actual_row["source_path"].clone();
    }
    compare_json(&actual, &expected, "result");

    compare_csv(
        &fs::read(out_dir.join("spatial_injection_audit.csv")).unwrap(),
        expected_csv,
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(fs::read(&fixture).unwrap())),
        FITS_FIXTURE_SHA256,
        "Rust CLI must not modify the FITS fixture"
    );
    let _ = fs::remove_dir_all(Path::new(&out_dir));
}

fn verify_reference_inputs() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(fs::read(root.join("../src/rst19/spatial_injection.py")).unwrap())
        ),
        PYTHON_REFERENCE_SHA256,
        "Python reference changed; refresh frozen outputs before accepting parity"
    );
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(
                fs::read(root.join("tests/fixtures/contaminated_pair_injection_parity.fits"))
                    .unwrap()
            )
        ),
        FITS_FIXTURE_SHA256,
        "FITS fixture changed"
    );
}

#[test]
fn gaussian_multi_cell_and_multi_trial_outputs_match_frozen_python_reference() {
    verify_reference_inputs();
    run_and_compare(
        &[
            "--grid-size",
            "2",
            "--peak-levels",
            "24",
            "56",
            "--trials",
            "2",
            "--sources-per-cell",
            "2",
            "--seed",
            "2026",
        ],
        include_bytes!("fixtures/spatial_injection_gaussian_parity.json"),
        include_bytes!("fixtures/spatial_injection_gaussian_parity.csv"),
    );
}

#[test]
fn empirical_psf_peak_injection_matches_frozen_python_reference() {
    verify_reference_inputs();
    run_and_compare(
        &[
            "--grid-size",
            "1",
            "--peak-levels",
            "56",
            "--trials",
            "1",
            "--sources-per-cell",
            "1",
            "--seed",
            "7",
            "--proposal-mode",
            "gaussian",
            "--psf-model",
            "empirical",
            "--signal-normalization",
            "peak_excess",
            "--aperture-radius",
            "3",
        ],
        include_bytes!("fixtures/spatial_injection_empirical_parity.json"),
        include_bytes!("fixtures/spatial_injection_empirical_parity.csv"),
    );
}

#[test]
fn empirical_psf_integrated_injection_matches_frozen_python_reference() {
    verify_reference_inputs();
    run_and_compare(
        &[
            "--grid-size",
            "1",
            "--peak-levels",
            "56",
            "--trials",
            "1",
            "--sources-per-cell",
            "1",
            "--seed",
            "37",
            "--proposal-mode",
            "hybrid",
            "--psf-model",
            "empirical",
            "--signal-normalization",
            "integrated_excess",
            "--aperture-radius",
            "3",
        ],
        include_bytes!("fixtures/spatial_injection_empirical_integrated_parity.json"),
        include_bytes!("fixtures/spatial_injection_empirical_integrated_parity.csv"),
    );
}
