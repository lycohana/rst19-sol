use rst19_core::detection_sweep;
use serde_json::Value;
use std::{
    fs,
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let (actual, expected) = (actual.as_f64().unwrap(), expected.as_f64().unwrap());
            assert!(
                (actual - expected).abs() <= 1e-10 + 1e-12 * expected.abs(),
                "{path}: actual={actual:.15}, expected={expected:.15}"
            );
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{path}");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                compare(actual, expected, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(
                actual.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>(),
                "{path} keys"
            );
            for (key, value) in expected {
                compare(&actual[key], value, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

fn fits_card(key: &str, value: &str) -> [u8; 80] {
    let mut card = [b' '; 80];
    let content = format!("{key:<8}= {value:>20}");
    card[..content.len()].copy_from_slice(content.as_bytes());
    card
}

fn write_synthetic_fits(path: &Path) {
    let mut bytes = Vec::new();
    for (key, value) in [
        ("SIMPLE", "T"),
        ("BITPIX", "16"),
        ("NAXIS", "2"),
        ("NAXIS1", "64"),
        ("NAXIS2", "64"),
    ] {
        bytes.extend_from_slice(&fits_card(key, value));
    }
    let mut end = [b' '; 80];
    end[..3].copy_from_slice(b"END");
    bytes.extend_from_slice(&end);
    bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
    for y in 0..64 {
        for x in 0..64 {
            let noise = (((37 * x + 101 * y) % 17) - 8) as f64 * 0.5;
            let radius2 = (x as f64 - 32.0).powi(2) + (y as f64 - 35.0).powi(2);
            let value = (100.0 + noise + 1200.0 * (-0.5 * radius2 / 4.0).exp()).round() as i16;
            bytes.extend_from_slice(&value.to_be_bytes());
        }
    }
    bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
    fs::write(path, bytes).unwrap();
}

#[test]
fn threshold_and_psf_sweep_results_match_python_reference() {
    let root = std::env::temp_dir().join(format!(
        "rst19-detection-sweep-parity-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let input = root.join("source.fits");
    write_synthetic_fits(&input);
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/detection_sweep.json")).unwrap();

    let threshold = detection_sweep::run_desktop(&[
        input.to_string_lossy().into_owned(),
        "--thresholds".into(),
        "4".into(),
        "8".into(),
        "--background-box-size".into(),
        "16".into(),
        "--out-dir".into(),
        root.join("threshold").to_string_lossy().into_owned(),
    ])
    .unwrap();
    compare(&threshold["result"], &fixture["threshold"], "threshold");
    assert!(root
        .join("threshold/detection_threshold_sweep.csv")
        .is_file());
    assert!(root
        .join("threshold/detection_threshold_sweep.png")
        .is_file());

    let psf = detection_sweep::run_desktop(&[
        input.to_string_lossy().into_owned(),
        "--thresholds".into(),
        "4".into(),
        "8".into(),
        "--psf-fwhms".into(),
        "2.5".into(),
        "3.5".into(),
        "--background-box-size".into(),
        "16".into(),
        "--out-dir".into(),
        root.join("psf").to_string_lossy().into_owned(),
    ])
    .unwrap();
    compare(&psf["result"], &fixture["psf_fwhm"], "psf_fwhm");
    assert!(root.join("psf/detection_psf_sweep.csv").is_file());
    assert!(root.join("psf/detection_psf_sweep.png").is_file());
    fs::remove_dir_all(root).unwrap();
}
