use rst19_core::{
    models::{FitsFrame, PixelData},
    pair_repair_counterfactual::TargetSource,
    pair_response_attribution::{self, Options},
};
use serde_json::Value;
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!((a - b).abs() <= 2e-7 + 2e-8 * b.abs(), "{path}: {a} != {b}");
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                compare(a, b, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                b.keys().collect::<Vec<_>>(),
                "{path} keys"
            );
            for (key, value) in b {
                compare(&a[key], value, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

fn image_values() -> Vec<f64> {
    let mut image = vec![10.0; 60 * 60];
    image[30 * 60 + 30] = 130.0;
    image[30 * 60 + 31] = 99.0;
    image[30 * 60 + 29] = -80.0;
    image[31 * 60 + 30] = 70.0;
    image
}

fn target(detection_id: i64, x: f64, peak_x: f64) -> TargetSource {
    TargetSource {
        detection_id,
        x,
        y: 30.0,
        peak_x: Some(peak_x),
        peak_y: Some(30.0),
        repeated_code_values: vec![99],
        background: Some(10.0),
        noise: Some(2.0),
    }
}

fn options() -> Options {
    Options {
        patch_padding_px: 8,
        psf_fwhm: 2.0,
        negative_anomaly_threshold_adu: -50.0,
        maximum_neighborhood_px: 3,
        maximum_count: 12,
    }
}

fn fits_card(key: &str, value: &str) -> [u8; 80] {
    let mut card = [b' '; 80];
    let content = format!("{key:<8}= {value:>20}");
    card[..content.len()].copy_from_slice(content.as_bytes());
    card
}

fn write_fits(path: &std::path::Path, image: &[f64]) {
    let mut bytes = Vec::new();
    for (key, value) in [
        ("SIMPLE", "T".to_owned()),
        ("BITPIX", "-64".to_owned()),
        ("NAXIS", "2".to_owned()),
        ("NAXIS1", "60".to_owned()),
        ("NAXIS2", "60".to_owned()),
    ] {
        bytes.extend_from_slice(&fits_card(key, &value));
    }
    let mut end = [b' '; 80];
    end[..3].copy_from_slice(b"END");
    bytes.extend_from_slice(&end);
    bytes.resize(bytes.len().div_ceil(2880) * 2880, b' ');
    for value in image {
        bytes.extend_from_slice(&value.to_be_bytes());
    }
    bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
    fs::write(path, bytes).unwrap();
}

#[test]
fn pair_response_result_matches_python_reference_fields() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/pair_response_attribution.json")).unwrap();
    let frame = FitsFrame {
        path: PathBuf::from("<FRAME>"),
        header: Default::default(),
        data: PixelData::F64(image_values()),
        auxiliary: None,
        data_offset: 0,
        width: 60,
        height: 60,
    };
    let actual = pair_response_attribution::run(
        &frame,
        &target(1, 31.0, 31.0),
        &target(2, 28.0, 28.0),
        None,
        &options(),
    )
    .unwrap();
    compare(&actual.as_dict(), &fixture["expected"], "result");
    assert_eq!(actual.target_rows.len(), 8);
    assert_eq!(actual.contribution_rows.len(), 6);
}

#[test]
fn native_cli_engine_and_jsonl_handler_run_without_python_path() {
    let directory = std::env::temp_dir().join(format!(
        "rst19-pair-response-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&directory).unwrap();
    let frame_path = directory.join("pair.fits");
    let catalog_path = directory.join("source_catalog.csv");
    write_fits(&frame_path, &image_values());
    fs::write(
        &catalog_path,
        "\u{feff}detection_id,x,y,peak_x,peak_y,repeated_code_values,background,noise\n1,31,30,31,30,99,10,2\n2,28,30,28,30,99,10,2\n",
    )
    .unwrap();

    let output_dir = directory.join("cli-output");
    let args = vec![
        frame_path.to_string_lossy().into_owned(),
        "--source-catalog".into(),
        catalog_path.to_string_lossy().into_owned(),
        "--primary-id=1".into(),
        "--secondary-id=2".into(),
        "--patch-padding-px=8".into(),
        "--psf-fwhm=2".into(),
        "--negative-anomaly-threshold-adu=-50".into(),
        "--out-dir".into(),
        output_dir.to_string_lossy().into_owned(),
    ];
    let cli = pair_response_attribution::run_desktop(&args).unwrap();
    assert_eq!(cli["result"]["target_rows"].as_array().unwrap().len(), 8);
    for artifact in [
        "pair_response_targets.csv",
        "pair_response_maxima.csv",
        "pair_response_contributions.csv",
        "pair_response_attribution.json",
    ] {
        assert!(output_dir.join(artifact).is_file(), "missing {artifact}");
    }
    assert!(fs::read(output_dir.join("pair_response_targets.csv"))
        .unwrap()
        .starts_with(&[0xef, 0xbb, 0xbf]));

    let standalone_output = directory.join("standalone-output");
    let status = Command::new(env!("CARGO_BIN_EXE_rst19-pair-response-attribution"))
        .args([
            frame_path.to_string_lossy().as_ref(),
            "--source-catalog",
            catalog_path.to_string_lossy().as_ref(),
            "--primary-id=1",
            "--secondary-id=2",
            "--patch-padding-px=8",
            "--psf-fwhm=2",
            "--negative-anomaly-threshold-adu=-50",
            "--out-dir",
            standalone_output.to_string_lossy().as_ref(),
        ])
        .env("PATH", "")
        .status()
        .unwrap();
    assert!(status.success(), "standalone Rust command failed");
    assert!(standalone_output
        .join("pair_response_attribution.json")
        .is_file());

    let engine_output = directory.join("engine-output");
    let mut engine = Command::new(env!("CARGO_BIN_EXE_rst19-engine"))
        .env("PATH", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let request = serde_json::json!({
        "operation":"audit.pair_response_attribution",
        "frame_path":frame_path,
        "source_catalog_path":catalog_path,
        "primary_id":1,
        "secondary_id":2,
        "output_dir":engine_output,
        "options":{
            "patch_padding_px":8,
            "psf_fwhm":2.0,
            "negative_anomaly_threshold_adu":-50.0
        }
    });
    writeln!(engine.stdin.take().unwrap(), "{request}").unwrap();
    let output = engine.wait_with_output().unwrap();
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["ok"], true);
    assert_eq!(
        envelope["payload"]["result"]["target_rows"]
            .as_array()
            .unwrap()
            .len(),
        8
    );
    assert!(engine_output
        .join("pair_response_attribution.json")
        .is_file());

    let request = serde_json::json!({
        "frame_path":frame_path,
        "source_catalog_path":catalog_path,
        "primary_id":1,
        "secondary_id":2,
        "output_dir":directory.join("handler-output"),
        "options":{
            "patch_padding_px":8,
            "psf_fwhm":2.0,
            "negative_anomaly_threshold_adu":-50.0
        }
    });
    let handled = pair_response_attribution::handle_request(&request).unwrap();
    assert_eq!(
        handled["result"]["target_rows"].as_array().unwrap().len(),
        8
    );
    fs::remove_dir_all(directory).unwrap();
}
