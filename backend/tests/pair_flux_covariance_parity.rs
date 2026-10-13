use rst19_core::pair_flux_covariance::{self, ForcedStabilityFluxRow, PAIR_FLUX_METRICS};
use serde_json::Value;
use std::{
    fs,
    io::Write,
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

fn rows() -> Vec<ForcedStabilityFluxRow> {
    let target_a = [10.0, 20.0, 30.0, 40.0, 50.0];
    let target_b = [5.0, 10.0, 15.0, 20.0, 25.0];
    let controls = [
        (1, [8.0, 7.0, 6.0, 5.0, 4.0]),
        (2, [2.0, 9.0, 3.0, 8.0, 4.0]),
        (3, [4.0, 1.0, 5.0, 2.0, 6.0]),
        (4, [9.0, 3.0, 8.0, 2.0, 7.0]),
    ];
    let mut output = Vec::new();
    for (index, value) in target_a.iter().enumerate() {
        output.push(ForcedStabilityFluxRow {
            detection_id: 82931,
            frame_index: index as i64 + 1,
            flux_snr: Some(*value),
            local_peak_flux_snr: Some(value + 1.0),
        });
    }
    for (index, value) in target_b.iter().enumerate() {
        output.push(ForcedStabilityFluxRow {
            detection_id: 82934,
            frame_index: index as i64 + 1,
            flux_snr: Some(*value),
            local_peak_flux_snr: Some(value + 1.0),
        });
    }
    for (detection_id, values) in controls {
        for (index, value) in values.iter().enumerate() {
            output.push(ForcedStabilityFluxRow {
                detection_id,
                frame_index: index as i64 + 1,
                flux_snr: Some(*value),
                local_peak_flux_snr: Some(value + 0.5),
            });
        }
    }
    output
}

#[test]
fn pair_flux_covariance_matches_python_result_fields() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/pair_flux_covariance.json")).unwrap();
    let metrics = PAIR_FLUX_METRICS.map(str::to_owned);
    let actual = pair_flux_covariance::run(&rows(), 82931, 82934, &metrics).unwrap();
    compare(&actual.as_dict(), &fixture["expected"], "result");
    assert_eq!(actual.frame_count, 5);
    assert_eq!(actual.control_source_count, 4);
    assert_eq!(actual.metrics[0].control_pair_count, 6);
}

#[test]
fn csv_loader_cli_engine_and_artifacts_are_native() {
    let directory = std::env::temp_dir().join(format!(
        "rst19-pair-flux-covariance-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&directory).unwrap();
    let csv_path = directory.join("forced_metrics.csv");
    let mut csv = String::from("\u{feff}detection_id,frame_index,flux_snr,local_peak_flux_snr\n");
    for row in rows() {
        csv.push_str(&format!(
            "{},{},{},{}\n",
            row.detection_id,
            row.frame_index,
            row.flux_snr.unwrap(),
            row.local_peak_flux_snr.unwrap()
        ));
    }
    fs::write(&csv_path, csv).unwrap();
    let loaded = pair_flux_covariance::load_forced_stability_flux_csv(&csv_path).unwrap();
    assert_eq!(loaded.len(), 30);
    let output_dir = directory.join("cli-output");
    let args = vec![
        csv_path.to_string_lossy().into_owned(),
        "--primary-id=82931".into(),
        "--secondary-id".into(),
        "82934".into(),
        "--out-dir".into(),
        output_dir.to_string_lossy().into_owned(),
    ];
    let cli = pair_flux_covariance::run_desktop(&args).unwrap();
    assert_eq!(cli["result"]["metrics"].as_array().unwrap().len(), 2);
    for artifact in ["pair_flux_covariance.csv", "pair_flux_covariance.json"] {
        assert!(output_dir.join(artifact).is_file(), "missing {artifact}");
    }
    let csv_bytes = fs::read(output_dir.join("pair_flux_covariance.csv")).unwrap();
    assert!(csv_bytes.starts_with(&[0xef, 0xbb, 0xbf]));
    let json: Value =
        serde_json::from_slice(&fs::read(output_dir.join("pair_flux_covariance.json")).unwrap())
            .unwrap();
    assert_eq!(json["note"], "这里的量是同一强制测光流程产生的 flux_snr/local_peak_flux_snr；控制分布用于发现异常共变，不是正式显著性检验，也不是双星概率。");

    let standalone_output = directory.join("standalone-output");
    let status = Command::new(env!("CARGO_BIN_EXE_rst19-pair-covariance-audit"))
        .args([
            csv_path.to_string_lossy().as_ref(),
            "--primary-id=82931",
            "--secondary-id=82934",
            "--metric=flux_snr",
            "--out-dir",
            standalone_output.to_string_lossy().as_ref(),
        ])
        .env("PATH", "")
        .status()
        .unwrap();
    assert!(status.success(), "standalone Rust command failed");
    assert!(standalone_output
        .join("pair_flux_covariance.json")
        .is_file());

    let engine_output = directory.join("engine-output");
    let mut engine = Command::new(env!("CARGO_BIN_EXE_rst19-engine"))
        .env("PATH", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let request = serde_json::json!({
        "operation":"audit.pair_flux_covariance",
        "forced_metrics_csv":csv_path,
        "primary_detection_id":82931,
        "secondary_detection_id":82934,
        "output_dir":engine_output
    });
    writeln!(engine.stdin.take().unwrap(), "{request}").unwrap();
    let output = engine.wait_with_output().unwrap();
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["ok"], true);
    assert_eq!(
        envelope["payload"]["result"]["metrics"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert!(engine_output.join("pair_flux_covariance.json").is_file());

    let duplicate = rows()
        .into_iter()
        .chain(std::iter::once(rows()[0].clone()))
        .collect::<Vec<_>>();
    assert!(pair_flux_covariance::run(
        &duplicate,
        82931,
        82934,
        &PAIR_FLUX_METRICS.map(str::to_owned)
    )
    .unwrap_err()
    .contains("duplicate"));
    fs::remove_dir_all(directory).unwrap();
}
