use rst19_core::temporal_code;
use serde_json::Value;
use std::{
    fs,
    io::Write,
    process::{Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

fn fits_card(key: &str, value: &str) -> [u8; 80] {
    let mut card = [b' '; 80];
    let content = format!("{key:<8}= {value:>20}");
    card[..content.len()].copy_from_slice(content.as_bytes());
    card
}

fn write_fits(path: &std::path::Path, image: &[i16]) {
    let mut bytes = Vec::new();
    for (key, value) in [
        ("SIMPLE", "T"),
        ("BITPIX", "16"),
        ("NAXIS", "2"),
        ("NAXIS1", "8"),
        ("NAXIS2", "8"),
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
    fs::write(path, bytes).unwrap();
}

fn frames(root: &std::path::Path) -> std::path::PathBuf {
    let dir = root.join("frames");
    fs::create_dir_all(&dir).unwrap();
    let mut first = vec![0_i16; 64];
    let mut second = first.clone();
    first[2 * 8 + 3] = -1;
    second[2 * 8 + 3] = -1;
    first[3 * 8 + 5] = 3991;
    second[3 * 8 + 5] = 3992;
    first[4 * 8 + 5] = 3991;
    second[4 * 8 + 5] = 3993;
    first[6 * 8 + 6] = 20;
    second[6 * 8 + 6] = 21;
    write_fits(&dir.join("frame-1.fits"), &first);
    write_fits(&dir.join("frame-2.fits"), &second);
    dir
}

#[test]
fn temporal_code_audit_matches_python_counts_components_and_csv_shape() {
    let root = std::env::temp_dir().join(format!(
        "rst19-temporal-code-parity-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    let input = frames(&root);
    let output = root.join("output");
    let response = temporal_code::run_desktop(&[
        input.to_string_lossy().into_owned(),
        "--focus-x".into(),
        "5".into(),
        "--focus-y".into(),
        "3".into(),
        "--sentinel-focus-x".into(),
        "3".into(),
        "--sentinel-focus-y".into(),
        "2".into(),
        "--out-dir".into(),
        output.to_string_lossy().into_owned(),
    ])
    .unwrap();
    let result = &response["result"];
    assert_eq!(result["frame_count"], 2);
    assert_eq!(result["image_shape"], serde_json::json!([8, 8]));
    assert_eq!(result["exact_stable_pixel_count"], 61);
    assert_eq!(result["low_variation_pixel_count"], 64);
    assert_eq!(result["focus_code_pixel_count"], 2);
    assert_eq!(result["focus_code_component_count"], 1);
    assert_eq!(result["focus_code_components_ge_2"], 1);
    assert_eq!(result["focus_code_largest_component"], 2);
    assert_eq!(result["focus_code_component_size"], 2);
    assert_eq!(
        result["focus_code_component_coordinates"],
        serde_json::json!([[5, 3], [5, 4]])
    );
    assert_eq!(result["focus_values_by_frame"][0]["code_value"], 3991);
    assert_eq!(result["focus_values_by_frame"][1]["code_value"], 3992);
    assert_eq!(result["sentinel"]["exact_stable_pixel_count"], 1);
    assert_eq!(result["sentinel"]["focus_component_size"], 1);
    assert_eq!(result["components"].as_array().unwrap().len(), 2);
    assert!(result["interpretation_boundary"]
        .as_str()
        .unwrap()
        .contains("不是已由 FITS 格式说明确认"));

    let focus_csv = fs::read(output.join("temporal_code_focus_series.csv")).unwrap();
    assert!(focus_csv.starts_with(&[0xEF, 0xBB, 0xBF]));
    assert!(String::from_utf8_lossy(&focus_csv).contains("frame_index,path,code_focus_xy"));
    assert!(String::from_utf8_lossy(&focus_csv).contains("3991"));
    let components_csv = fs::read(output.join("temporal_code_components.csv")).unwrap();
    assert!(components_csv.starts_with(&[0xEF, 0xBB, 0xBF]));
    assert!(String::from_utf8_lossy(&components_csv).contains("low_variation_code"));
    assert!(output.join("temporal_code_audit.json").is_file());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn temporal_code_cli_and_engine_work_without_python_path() {
    let root = std::env::temp_dir().join(format!(
        "rst19-temporal-code-routes-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    let input = frames(&root);
    let cli_out = root.join("cli");
    let status = Command::new(env!("CARGO_BIN_EXE_rst19-temporal-codes"))
        .args([
            input.to_string_lossy().as_ref(),
            "--focus-x",
            "5",
            "--focus-y",
            "3",
            "--sentinel-focus-x",
            "3",
            "--sentinel-focus-y",
            "2",
            "--out-dir",
            cli_out.to_string_lossy().as_ref(),
        ])
        .env("PATH", "")
        .status()
        .unwrap();
    assert!(status.success());

    let mut engine = Command::new(env!("CARGO_BIN_EXE_rst19-engine"))
        .env("PATH", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let request = serde_json::json!({
        "operation":"audit.temporal_codes",
        "data_dir":input,
        "focus_x":5,
        "focus_y":3,
        "sentinel_focus_x":3,
        "sentinel_focus_y":2,
        "output_dir":root.join("engine")
    });
    writeln!(engine.stdin.take().unwrap(), "{request}").unwrap();
    let output = engine.wait_with_output().unwrap();
    assert!(output.status.success());
    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["ok"], true);
    assert_eq!(
        response["payload"]["result"]["exact_stable_pixel_count"],
        61
    );
    assert!(root.join("engine/temporal_code_audit.json").is_file());
    fs::remove_dir_all(root).unwrap();
}
