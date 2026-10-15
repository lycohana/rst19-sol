use rst19_core::{
    innovation,
    innovation_artifacts::{difference_rows, write_innovation_artifacts, ImagePlane},
};
use serde_json::{json, Value};
use std::{fs, path::PathBuf};

#[test]
fn registered_difference_statistics_preserve_signed_residuals_and_missing_pixels() {
    let rows = difference_rows(
        &[json!({"frame_index":0,"timestamp":"2026-03-30T00:00:00","background_rms_adu":2.0})],
        &[ImagePlane {
            values: vec![1.0, -2.0, f32::NAN, 0.5],
            width: 2,
            height: 2,
        }],
        &[0.0; 4],
    )
    .unwrap();
    assert_eq!(rows[0]["residual_median_adu"], 0.5);
    assert!((rows[0]["residual_rms_adu"].as_f64().unwrap() - 0.7413).abs() < 1e-12);
    assert_eq!(rows[0]["positive_peak_sigma"], 0.5);
    assert_eq!(rows[0]["negative_peak_sigma"], -1.0);
    assert_eq!(rows[0]["valid_fraction"], 0.75);
}

#[test]
fn full_report_writer_exports_tables_registered_images_motion_cutouts_and_charts() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let frame = root.join("tests/fixtures/contaminated_pair_injection_parity.fits");
    let frame_data = rst19_core::fits::read_fits(&frame, false).unwrap();
    // Synthetic candidate metadata exercises the renderer; the repeated FITS is not motion truth.
    let payload = json!({"frames":[{"path":frame,"timestamp":"2026-03-30T00:00:00","candidate_count":3,"returned_count":3,"quality_count":3},{"path":frame,"timestamp":"2026-03-30T00:00:01.5","candidate_count":3,"returned_count":3,"quality_count":3}],"cumulative_shifts":[[0.0,0.0],[0.0,0.0]],
        "tracks":[],"motion_features":[{"track_id":0,"classification":"candidate","points":[{"frame_index":0,"x":32.0,"y":32.0,"aligned_x":32.0,"aligned_y":32.0,"length_px":12.0,"width_px":3.0,"angle_deg":0.0,"residual_snr":10.0,"bbox":[0,0,frame_data.width-1,frame_data.height-1]}]}]});
    let report = innovation::build_innovation_report_from_payload(
        &payload,
        &root.join("synthetic-report.json"),
    )
    .unwrap();
    assert_eq!(report["relation"]["frame_count"], 2);
    assert_eq!(report["relation"]["duration_s"], 1.5);
    assert_eq!(report["motion"]["candidate_count"], 1);
    assert_eq!(report["photometric_evidence"]["status"], "NOT_PRESENT");
    let output = root
        .join("target/test-artifacts")
        .join(format!("innovation-export-{}", std::process::id()));
    let written = write_innovation_artifacts(&report, &output).unwrap();
    for name in [
        "frame_evidence.csv",
        "motion_evidence.csv",
        "motion_points_evidence.csv",
        "motion_evidence_links.csv",
        "point_motion_evidence.csv",
        "motion_frame_audit.csv",
        "difference_evidence.csv",
        "telemetry_prediction.csv",
        "innovation_report.json",
        "registered_difference.png",
        "registered_mosaic.png",
        "quality_counts.png",
        "registration_shift.png",
        "telemetry_velocity.png",
        "telemetry_position.png",
        "motion_trajectory.png",
        "point_motion_trajectory.png",
        "motion_frame_audit.png",
        "motion_contact_sheet.png",
        "motion_cutouts/track_0000_frame_01.png",
    ] {
        assert!(written.join(name).is_file(), "missing {name}");
    }
    let saved: Value =
        serde_json::from_slice(&fs::read(written.join("innovation_report.json")).unwrap()).unwrap();
    assert_eq!(saved["difference"]["frame_count"], 2);
    assert_eq!(saved["difference"]["residual_rms_median_adu"], 0.0);
    let decoder = png::Decoder::new(std::io::BufReader::new(
        fs::File::open(written.join("telemetry_position.png")).unwrap(),
    ));
    let reader = decoder.read_info().unwrap();
    assert_eq!(reader.info().width, 1200);
    assert_eq!(reader.info().height, 680);
    let canonical = fs::canonicalize(root.join("target/test-artifacts")).unwrap();
    assert!(written.starts_with(canonical));
    fs::remove_dir_all(written).unwrap();
}
