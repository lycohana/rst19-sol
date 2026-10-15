use rst19_core::innovation::fit_constant_velocity;
use serde_json::json;

#[test]
fn small_sample_motion_uses_student_t_and_a_mean_trajectory_confidence_interval() {
    let fit = fit_constant_velocity(
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 1.1, 1.9, 3.2],
        &[0.0, 2.1, 3.9, 6.2],
        Some(2.0),
    )
    .unwrap();
    assert_eq!(fit["degrees_of_freedom"], 2);
    assert_eq!(fit["t95_critical"], 4.303);
    let speed = fit["speed_px_per_s"].as_f64().unwrap();
    assert!(fit["speed_ci95_low_px_per_s"].as_f64().unwrap() < speed);
    assert!(fit["speed_ci95_high_px_per_s"].as_f64().unwrap() > speed);
    assert!(fit["direction_ci95_half_width_deg"].as_f64().unwrap() > 0.0);
    assert!(fit["predicted_x_ci95_half_width_px"].as_f64().unwrap() > 0.0);
    assert_eq!(fit["forecast_elapsed_s"], 5.0);
    assert!((fit["velocity_x_px_per_s"].as_f64().unwrap() - 1.04).abs() < 1e-12);
    assert!((fit["velocity_y_px_per_s"].as_f64().unwrap() - 2.04).abs() < 1e-12);
    // SSE=.042 and Sxx=5: forecast mean variance=.021*(1/4+3.5^2/5).
    let expected = 4.303_f64 * (0.021_f64 * (0.25 + 12.25 / 5.0)).sqrt();
    assert!((fit["predicted_x_ci95_half_width_px"].as_f64().unwrap() - expected).abs() < 1e-12);
}

#[test]
fn two_observations_have_a_forecast_but_no_estimated_uncertainty() {
    let fit = fit_constant_velocity(&[0.0, 2.0], &[1.0, 5.0], &[3.0, 3.0], Some(1.0)).unwrap();
    assert_eq!(fit["degrees_of_freedom"], 0);
    assert_eq!(fit["speed_px_per_s"], 2.0);
    assert_eq!(fit["predicted_x_px"], 7.0);
    for field in [
        "speed_ci95_low_px_per_s",
        "direction_ci95_half_width_deg",
        "predicted_x_ci95_half_width_px",
    ] {
        assert!(fit[field].is_null());
    }
    assert!(fit_constant_velocity(&[1.0, 1.0], &[1.0, 2.0], &[0.0, 0.0], None).is_err());
}

#[test]
fn frame_evidence_normalizes_offsets_and_starts_elapsed_time_at_first_valid_observation() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let frame = root.join("tests/fixtures/contaminated_pair_injection_parity.fits");
    let payload = json!({"frames":[
        {"path":frame,"timestamp":"invalid","auxiliary":{}},
        {"path":frame,"timestamp":"2026-03-30T16:32:05.123456+08:00","auxiliary":{}},
        {"path":frame,"timestamp":"2026-03-30T08:32:06.623456Z","auxiliary":{}}
    ],"cumulative_shifts":[[0.0,0.0],[null,2.0],[3.0,4.0]]});
    let rows = rst19_core::innovation::frame_rows(&payload, &root.join("report.json")).unwrap();
    assert!(rows[0]["elapsed_s"].is_null());
    assert_eq!(rows[1]["timestamp"], "2026-03-30T08:32:05.123456");
    assert_eq!(rows[1]["elapsed_s"], 0.0);
    assert_eq!(rows[2]["elapsed_s"], 1.5);
    assert_eq!(rows[2]["interval_s"], 1.5);
    assert!(rows[1]["cumulative_shift_x_px"].is_null());
    assert!(rows[1]["j2000_position_norm_raw"].is_null());
    assert_eq!(rows[2]["cumulative_shift_norm_px"], 5.0);
}

#[test]
fn vector_evidence_requires_complete_components_and_retains_raw_units() {
    let row = json!({"j2000_x":3.0,"j2000_y":4.0,"j2000_z":12.0,"j2000_xv":0.0,"j2000_yv":3.0,"j2000_zv":4.0});
    let metrics = rst19_core::innovation::vector_metrics(&row);
    assert_eq!(metrics["j2000_position_norm_raw"], 13.0);
    assert_eq!(metrics["j2000_velocity_norm_raw"], 5.0);
    assert!(metrics["wgs84_position_norm_raw"].is_null());
}

#[test]
fn telemetry_uses_central_differences_and_keeps_the_velocity_unit_assumption() {
    let mut rows = (0..3).map(|index|json!({"timestamp":format!("2026-03-30T16:32:{:02}",5+index),"frame_index":index,
        "j2000_x":10.0*index as f64,"j2000_y":0.0,"j2000_z":0.0,"j2000_xv":10.0,"j2000_yv":0.0,"j2000_zv":0.0})).collect::<Vec<_>>();
    let summary = rst19_core::innovation::telemetry_consistency(&mut rows, "j2000").unwrap();
    assert_eq!(summary["valid_sample_count"], 3);
    assert_eq!(summary["interior_sample_count"], 1);
    assert_eq!(summary["position_rate_norm_m_per_s_median"], 10.0);
    assert_eq!(
        rows[1]["j2000_velocity_consistency_relative_error_assuming_m_per_s"],
        0.0
    );
    let prediction = rst19_core::innovation::telemetry_prediction(&rows, "j2000", Some(5.0));
    assert_eq!(
        prediction["predicted_position_m_assuming_m_per_s"],
        json!([70.0, 0.0, 0.0])
    );
    assert_eq!(prediction["unit_status"], "速度单位待主办方确认");
    assert_eq!(
        rst19_core::innovation::telemetry_prediction(&[json!({"j2000_x":1.0})], "j2000", Some(5.0))
            ["status"],
        "unavailable"
    );
}

#[test]
fn motion_evidence_uses_declared_observation_times_and_does_not_invent_single_frame_speed() {
    let frames = vec![
        json!({"timestamp":"2026-03-30T16:32:05.000"}),
        json!({"timestamp":"2026-03-30T16:32:06.500"}),
    ];
    let points = json!([
        {"frame_index":0,"aligned_x":10.0,"aligned_y":20.0,"length_px":40.0,"width_px":3.0,"residual_snr":20.0,"flux_snr":18.0},
        {"frame_index":1,"aligned_x":25.0,"aligned_y":20.0,"length_px":41.0,"width_px":3.0,"residual_snr":22.0,"flux_snr":20.0}]);
    let payload = json!({"motion_features":[{"track_id":3,"classification":"moving","points":points}],"tracks":[{"track_id":12,"classification":"moving","evidence_level":"fast_point_motion","points":points}]});
    let lines = rst19_core::innovation::motion_rows(&payload, &frames).unwrap();
    assert_eq!(lines[0]["duration_s"], 1.5);
    assert_eq!(lines[0]["speed_px_per_s"], 10.0);
    assert_eq!(lines[0]["displacement_px"], 15.0);
    assert_eq!(lines[0]["points"][0]["fit_residual_px"], 0.0);
    let points = rst19_core::innovation::point_motion_rows(&payload, &frames).unwrap();
    assert_eq!(points[0]["median_flux_snr"], 19.0);
    assert_eq!(points[0]["evidence_level"], "fast_point_motion");
    let missing = rst19_core::innovation::point_motion_rows(
        &payload,
        &[frames[0].clone(), json!({"timestamp":null})],
    )
    .unwrap();
    assert!(missing[0]["kinematic_model"].is_null());
    assert!(missing[0]["speed_px_per_s"].is_null());
    let single = rst19_core::innovation::motion_rows(&json!({"motion_features":[{"classification":"candidate","points":[{"frame_index":0,"x":2.0,"y":3.0,"angle_deg":90.0,"length_px":30.0,"width_px":2.0,"residual_snr":10.0}]}]}),&frames[..1]).unwrap();
    assert!(single[0]["speed_px_per_s"].is_null());
    assert_eq!(single[0]["direction_deg_image"], 90.0);
}
