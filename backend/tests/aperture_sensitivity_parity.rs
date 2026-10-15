use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, process::Command};

const PYTHON_REFERENCE_SHA256: &str =
    "9d9e02c0dad0b4232baf1daa655f0bf2f4f1d3573be0e54f15040df9d8e060b5";
const FITS_FIXTURE_SHA256: &str =
    "02a9303a2494cb46dae4d7f5f44d817cbf92f217f405e5da5997aac7008d3c42";
const FLOAT_TOLERANCE: f64 = 2e-5;
const FLOAT_RELATIVE_TOLERANCE: f64 = 2e-6;
const FLOAT32_SNR_ABSOLUTE_TOLERANCE: f64 = 1.2e-4;

fn tolerance(path: &str, actual: f64, expected: f64) -> f64 {
    let absolute = if path.ends_with(".peak_snr")
        || path.ends_with(".flux_snr")
        || path.ends_with(".median_candidate_flux_snr")
    {
        FLOAT32_SNR_ABSOLUTE_TOLERANCE
    } else {
        FLOAT_TOLERANCE
    };
    absolute + FLOAT_RELATIVE_TOLERANCE * actual.abs().max(expected.abs())
}

fn compare_json(actual: &Value, expected: &Value, path: &str) {
    if path.ends_with(".parameters.summary_json") {
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
            let allowed = tolerance(path, actual, expected);
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
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let expected_rows = expected_reader
        .records()
        .collect::<Result<Vec<_>, _>>()
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
            } else if let (Ok(actual_int), Ok(expected_int)) =
                (actual.parse::<i64>(), expected.parse::<i64>())
            {
                assert_eq!(actual_int, expected_int, "CSV {name} row {row_index}");
            } else if let (Ok(actual_number), Ok(expected_number)) =
                (actual.parse::<f64>(), expected.parse::<f64>())
            {
                let path = format!("csv.{name}");
                let allowed = tolerance(&path, actual_number, expected_number);
                assert!(
                    (actual_number - expected_number).abs() <= allowed,
                    "CSV {name} row {row_index}: Rust {actual_number} differs from Python {expected_number}"
                );
            } else {
                assert_eq!(actual, expected, "CSV {name} row {row_index}");
            }
        }
    }
}

#[test]
fn aperture_sensitivity_cli_matches_frozen_python_outputs_without_python_on_path() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("tests/fixtures/contaminated_pair_injection_parity.fits");
    let python_source = root.join("tests/fixtures/legacy_sources/aperture_sensitivity.txt");
    assert_eq!(
        format!("{:x}", Sha256::digest(fs::read(python_source).unwrap())),
        PYTHON_REFERENCE_SHA256,
        "Python reference changed; refresh the frozen output before accepting parity"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(fs::read(&fixture).unwrap())),
        FITS_FIXTURE_SHA256,
        "FITS fixture changed"
    );

    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "rst19-aperture-sensitivity-parity-{}-{nonce}",
        std::process::id()
    ));
    let status = Command::new(env!("CARGO_BIN_EXE_rst19-aperture-sensitivity"))
        .env("PATH", "")
        .arg(&fixture)
        .args(["--aperture-radii", "3", "4", "--out-dir"])
        .arg(&out_dir)
        .status()
        .expect("launching Rust CLI without PATH must work");
    assert!(
        status.success(),
        "Rust CLI failed with Python removed from PATH"
    );

    let actual: Value =
        serde_json::from_slice(&fs::read(out_dir.join("aperture_sensitivity_audit.json")).unwrap())
            .unwrap();
    let mut expected: Value =
        serde_json::from_slice(include_bytes!("fixtures/aperture_sensitivity_parity.json"))
            .unwrap();
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

    let actual_csv = fs::read(out_dir.join("aperture_sensitivity_audit.csv")).unwrap();
    compare_csv(
        &actual_csv,
        include_bytes!("fixtures/aperture_sensitivity_parity.csv"),
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(fs::read(&fixture).unwrap())),
        FITS_FIXTURE_SHA256,
        "Rust CLI modified the FITS fixture"
    );
}
