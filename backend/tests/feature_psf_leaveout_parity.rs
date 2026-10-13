use serde_json::Value;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn fits_card(key: &str, value: &str) -> [u8; 80] {
    let mut card = [b' '; 80];
    let content = format!("{key:<8}= {value:>20}");
    card[..content.len()].copy_from_slice(content.as_bytes());
    card
}

fn write_fixture(root: &Path) -> (PathBuf, PathBuf) {
    const WIDTH: usize = 96;
    const HEIGHT: usize = 96;
    let fits_path = root.join("frame.fits");
    let catalog_path = root.join("source_catalog.csv");
    let positions = [(16.0f64, 16.0f64), (48.0, 16.0), (16.0, 48.0), (48.0, 48.0)];
    let mut image = vec![20.0f32; WIDTH * HEIGHT];
    let sigma: f64 = 3.0 / 2.354820045;
    let radius = (4.0 * sigma).ceil() as isize;
    for &(cx, cy) in &positions {
        for y in (cy as isize - radius)..=(cy as isize + radius) {
            for x in (cx as isize - radius)..=(cx as isize + radius) {
                let r2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                let kernel = (-r2 / (2.0 * sigma.powi(2))).exp();
                let index = y as usize * WIDTH + x as usize;
                image[index] = (image[index] as f64 + 120.0 * kernel) as f32;
            }
        }
    }
    let mut bytes = Vec::new();
    for (key, value) in [
        ("SIMPLE", "T"),
        ("BITPIX", "-32"),
        ("NAXIS", "2"),
        ("NAXIS1", "96"),
        ("NAXIS2", "96"),
    ] {
        bytes.extend_from_slice(&fits_card(key, value));
    }
    let mut end = [b' '; 80];
    end[..3].copy_from_slice(b"END");
    bytes.extend_from_slice(&end);
    bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
    for value in image {
        bytes.extend_from_slice(&value.to_be_bytes());
    }
    bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
    fs::write(&fits_path, bytes).unwrap();
    let mut catalog = fs::File::create(&catalog_path).unwrap();
    writeln!(catalog, "detection_id,x,y,peak,flux,background,noise,snr,fwhm,flags,flux_snr,ellipticity,quality_passed").unwrap();
    for (index, (x, y)) in positions.into_iter().enumerate() {
        writeln!(catalog, "{index},{x},{y},140,900,20,5,28,3,,45,0.1,true").unwrap();
    }
    (fits_path, catalog_path)
}

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            assert!(
                (actual - expected).abs() <= 5e-6 + 2e-6 * expected.abs(),
                "{path}: {actual} != {expected}"
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
            for (key, expected) in expected {
                compare(&actual[key], expected, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

#[test]
fn standalone_cli_reads_fits_and_catalog_with_python_absent_and_matches_reference() {
    let root = std::env::temp_dir().join(format!(
        "rst19-feature-psf-leaveout-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let (fits, catalog) = write_fixture(&root);
    let output_dir = root.join("result");
    let status = Command::new(env!("CARGO_BIN_EXE_rst19-feature-psf-leaveout"))
        .args([
            fits.to_string_lossy().as_ref(),
            "--source-catalog",
            catalog.to_string_lossy().as_ref(),
            "--per-class",
            "4",
            "--support-radius",
            "3",
            "--max-template-sources",
            "4",
            "--out-dir",
            output_dir.to_string_lossy().as_ref(),
        ])
        .env("PATH", "")
        .status()
        .unwrap();
    assert!(status.success());
    let output: Value =
        serde_json::from_slice(&fs::read(output_dir.join("feature_psf_leaveout.json")).unwrap())
            .unwrap();
    let mut actual = output;
    actual["frame_path"] = "<array>".into();
    actual["catalog_path"] = Value::Null;
    let expected: Value =
        serde_json::from_str(include_str!("fixtures/feature_psf_leaveout.json")).unwrap();
    compare(&actual, &expected, "result");
    assert!(fs::read(output_dir.join("feature_psf_leaveout.csv"))
        .unwrap()
        .starts_with(&[0xef, 0xbb, 0xbf]));
    assert!(fs::read(output_dir.join("feature_psf_leaveout_source.csv"))
        .unwrap()
        .starts_with(&[0xef, 0xbb, 0xbf]));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn jsonl_engine_reads_fits_and_catalog_with_python_absent_and_matches_reference() {
    let root = std::env::temp_dir().join(format!(
        "rst19-feature-psf-leaveout-engine-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let (fits, catalog) = write_fixture(&root);
    let output_dir = root.join("engine-result");
    let request = serde_json::json!({
        "operation":"audit.feature_psf_leaveout",
        "fits_path":fits,
        "source_catalog_path":catalog,
        "per_class":4,
        "support_radius":3,
        "max_template_sources":4,
        "out_dir":output_dir
    });
    let mut engine = Command::new(env!("CARGO_BIN_EXE_rst19-engine"))
        .env("PATH", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(engine.stdin.take().unwrap(), "{request}").unwrap();
    let output = engine.wait_with_output().unwrap();
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["ok"], true);
    let mut actual = envelope["payload"].clone();
    actual["frame_path"] = "<array>".into();
    actual["catalog_path"] = Value::Null;
    actual.as_object_mut().unwrap().remove("out_dir");
    let expected: Value =
        serde_json::from_str(include_str!("fixtures/feature_psf_leaveout.json")).unwrap();
    compare(&actual, &expected, "result");
    fs::remove_dir_all(root).unwrap();
}
