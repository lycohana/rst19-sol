use rst19_core::sequence_motion_tracks::{
    review_temporal_components_against_raw_frames, sequence_line_model_review_supported,
};
use serde_json::{json, Value};

#[test]
fn raw_line_model_review_gate_matches_frozen_python_cases() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/sequence_motion_raw_review.json")).unwrap();
    let psf_fwhm = fixture["psf_fwhm"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let actual = sequence_line_model_review_supported(&case["point"], psf_fwhm).unwrap();
        assert_eq!(actual, case["expected"].as_bool().unwrap(), "{label}");
    }
}

fn temporal_point() -> Value {
    json!({
        "frame_index":0,
        "x":10.5,
        "y":20.0,
        "aligned_x":10.5,
        "aligned_y":20.0,
        "length_px":20.0,
        "width_px":1.0,
        "angle_deg":0.0,
        "residual_snr":12.0,
        "continuity_ratio":0.1,
        "profile_fill_ratio":0.1,
        "line_artifact_overlap_fraction":0.8,
    })
}

fn raw_line(model: &str) -> Value {
    json!({
        "frame_index":0,
        "x":10.0,
        "y":20.0,
        "aligned_x":10.0,
        "aligned_y":20.0,
        "length_px":20.0,
        "width_px":1.0,
        "angle_deg":0.0,
        "line_template_snr":7.0,
        "line_vs_point_delta_bic":3.0,
        "line_vs_double_delta_bic":3.0,
        "line_vs_background_delta_bic":3.0,
        "line_vs_elliptical_delta_bic":3.0,
        "single_frame_model":model,
        "single_frame_model_margin_bic":3.0,
        "continuity_ratio":0.8,
        "profile_fill_ratio":0.9,
        "line_artifact_overlap_fraction":0.0,
    })
}

fn review(temporal: Value, raw: Value) -> Value {
    review_temporal_components_against_raw_frames(
        &json!({
            "status":"analyzed",
            "min_length_px":6.0,
            "thresholds_adu":[100.0],
            "metadata_supported":false,
            "frames":[{"points":[temporal],"audit":{"feature_count":1}}],
        }),
        &[json!({"points":[raw]})],
        &[[0.0, 0.0]],
        2.0,
    )
    .unwrap()
}

#[test]
fn review_copies_original_pixel_evidence_and_rejects_compact_winners() {
    let accepted = review(temporal_point(), raw_line("line"));
    let points = accepted["frames"][0]["points"].as_array().unwrap();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0]["residual_snr"], 12.0);
    assert_eq!(points[0]["line_template_snr"], 7.0);
    assert_eq!(points[0]["continuity_ratio"], 0.8);
    assert_eq!(points[0]["line_artifact_overlap_fraction"], 0.0);
    assert_eq!(accepted["frames"][0]["audit"]["feature_count"], 1);

    let rejected = review(temporal_point(), raw_line("double"));
    assert!(rejected["frames"][0]["points"]
        .as_array()
        .unwrap()
        .is_empty());
    assert_eq!(rejected["review_audit"][0]["model_rejected_count"], 1);
}

#[test]
fn temporal_component_requires_overlapping_raw_frame_geometry() {
    let mut raw = raw_line("line");
    raw["x"] = json!(100.0);
    raw["aligned_x"] = json!(100.0);
    let result = review(temporal_point(), raw);
    assert!(result["frames"][0]["points"].as_array().unwrap().is_empty());
    assert_eq!(result["review_audit"][0]["raw_geometry_match_count"], 0);
}
