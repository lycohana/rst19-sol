use rst19_core::{
    models::{FitsFrame, PixelData},
    pair_repair_counterfactual::{self, Options, TargetSource},
};
use serde_json::Value;
use std::{
    fs,
    path::PathBuf,
    process::Command,
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
    let mut image = Vec::with_capacity(96 * 96);
    let sigma = 2.0 / (2.0 * (2.0_f64.ln()).sqrt() * 2.0_f64.sqrt());
    for y in 0..96 {
        for x in 0..96 {
            let pattern = (((37 * x + 101 * y) % 17) - 8) as f64 * 0.25;
            let mut value = 100.0 + pattern;
            for (cx, cy, amplitude) in [(40.0_f64, 40.0_f64, 800.0_f64), (50.0, 42.0, 720.0)] {
                let distance_squared = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                value += amplitude * (-0.5 * distance_squared / sigma.powi(2)).exp();
            }
            if (x, y) == (44, 41) {
                value = 900.0;
            }
            if (x, y) == (46, 43) {
                value = -250.0;
            }
            image.push(value);
        }
    }
    image
}

fn target(detection_id: i64, x: f64, y: f64, peak_x: f64, peak_y: f64) -> TargetSource {
    TargetSource {
        detection_id,
        x,
        y,
        peak_x: Some(peak_x),
        peak_y: Some(peak_y),
        repeated_code_values: vec![900],
        background: Some(100.0),
        noise: Some(1.0),
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
        ("NAXIS1", "96".to_owned()),
        ("NAXIS2", "96".to_owned()),
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
fn pair_repair_variants_match_python_reference_fields() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/pair_repair_counterfactual.json")).unwrap();
    let frame = FitsFrame {
        path: PathBuf::from("<FRAME>"),
        header: Default::default(),
        data: PixelData::F64(image_values()),
        auxiliary: None,
        data_offset: 0,
        width: 96,
        height: 96,
    };
    let primary = target(1, 40.0, 40.0, 40.0, 40.0);
    let secondary = target(2, 50.0, 42.0, 50.0, 42.0);
    let options = Options {
        patch_padding_px: 18,
        replacement_radius_px: 1,
        target_match_radius_px: 5.0,
        negative_anomaly_threshold_adu: -200.0,
        threshold_sigma: 4.0,
        min_distance: 4,
        aperture_radius: 3,
        psf_fwhm: 2.0,
        background_box_size: 16,
        min_flux_snr: 5.0,
        min_psf_support_pixels: 3,
        proposal_mode: "hybrid".into(),
    };
    let actual =
        pair_repair_counterfactual::run(&frame, &primary, &secondary, None, &options).unwrap();
    compare(&actual.as_dict(), &fixture["expected"], "result");
    assert_eq!(actual.summaries.len(), 4);
    assert_eq!(
        actual
            .summaries
            .iter()
            .map(|row| row.variant.as_str())
            .collect::<Vec<_>>(),
        pair_repair_counterfactual::VARIANTS
    );
}

#[test]
fn native_cli_and_jsonl_route_read_fits_and_write_artifacts() {
    let directory = std::env::temp_dir().join(format!(
        "rst19-pair-repair-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&directory).unwrap();
    let frame_path = directory.join("pair.fits");
    let catalog_path = directory.join("source_catalog.csv");
    write_fits(&frame_path, &image_values());
    fs::write(
        &catalog_path,
        "\u{feff}detection_id,x,y,peak_x,peak_y,repeated_code_values,background,noise\n1,40,40,40,40,900,100,1\n2,50,42,50,42,900,100,1\n",
    )
    .unwrap();
    let output_path = directory.join("cli-output");
    let args = vec![
        frame_path.to_string_lossy().into_owned(),
        "--source-catalog".into(),
        catalog_path.to_string_lossy().into_owned(),
        "--primary-id=1".into(),
        "--secondary-id".into(),
        "2".into(),
        "--patch-padding-px=18".into(),
        "--replacement-radius-px=1".into(),
        "--target-match-radius-px=5".into(),
        "--negative-anomaly-threshold-adu=-200".into(),
        "--aperture-radius=3".into(),
        "--background-box-size=16".into(),
        "--out-dir".into(),
        output_path.to_string_lossy().into_owned(),
    ];
    let cli_result = pair_repair_counterfactual::run_desktop(&args).unwrap();
    assert_eq!(
        cli_result["result"]["summaries"].as_array().unwrap().len(),
        4
    );
    for artifact in [
        "pair_repair_summaries.csv",
        "pair_repair_candidates.csv",
        "pair_repair_replacements.csv",
        "pair_repair_counterfactual.json",
    ] {
        assert!(output_path.join(artifact).is_file(), "missing {artifact}");
    }
    let csv = fs::read(output_path.join("pair_repair_summaries.csv")).unwrap();
    assert!(csv.starts_with(&[0xef, 0xbb, 0xbf]));
    assert!(csv
        .windows(b"variant,repaired_pixel_count,candidate_count".len())
        .any(|window| { window == b"variant,repaired_pixel_count,candidate_count" }));

    let standalone_output = directory.join("standalone-output");
    let status = Command::new(env!("CARGO_BIN_EXE_rst19-pair-repair-counterfactual"))
        .args([
            frame_path.to_string_lossy().as_ref(),
            "--source-catalog",
            catalog_path.to_string_lossy().as_ref(),
            "--primary-id=1",
            "--secondary-id=2",
            "--patch-padding-px=18",
            "--replacement-radius-px=1",
            "--target-match-radius-px=5",
            "--negative-anomaly-threshold-adu=-200",
            "--aperture-radius=3",
            "--background-box-size=16",
            "--out-dir",
            standalone_output.to_string_lossy().as_ref(),
        ])
        .env("PATH", "")
        .status()
        .unwrap();
    assert!(status.success(), "standalone Rust command failed");
    assert!(standalone_output
        .join("pair_repair_counterfactual.json")
        .is_file());

    let worker_output = directory.join("worker-output");
    let request = serde_json::json!({
        "frame_path":frame_path,
        "source_catalog_path":catalog_path,
        "primary_id":1,
        "secondary_id":2,
        "output_dir":worker_output,
        "options":{
            "patch_padding_px":18,
            "replacement_radius_px":1,
            "target_match_radius_px":5.0,
            "negative_anomaly_threshold_adu":-200.0,
            "aperture_radius":3,
            "background_box_size":16
        }
    });
    let worker_result = pair_repair_counterfactual::handle_request(&request).unwrap();
    assert_eq!(
        worker_result["result"]["summaries"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    assert!(worker_output
        .join("pair_repair_counterfactual.json")
        .is_file());
    fs::remove_dir_all(directory).unwrap();
}
