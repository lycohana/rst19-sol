use rst19_core::sequence_motion_tracks::{
    classify_motion_component_tracks, public_motion_feature_tracks,
};
use serde_json::{json, Value};

fn point(frame: usize, x: f64) -> Value {
    json!({"frame_index":frame,"x":x,"y":100.0,"aligned_x":x,"aligned_y":100.0,
        "length_px":20.0,"width_px":1.0,"angle_deg":0.0,"residual_snr":20.0,
        "bbox":[x-10.0,99.0,x+10.0,101.0],"touches_edge":false,
        "continuity_ratio":0.9,"profile_fill_ratio":0.9,"line_artifact_overlap_fraction":0.0,
        "single_frame_model":"line","line_template_snr":7.0,
        "line_vs_point_delta_bic":3.0,"line_vs_double_delta_bic":3.0,
        "line_vs_elliptical_delta_bic":3.0,"line_vs_background_delta_bic":3.0})
}

fn classify(points: Vec<Value>, times: &[f64]) -> Value {
    let headers = times
        .iter()
        .map(|_| json!({"DATE-OBS":"2024-01-01T00:00:00Z","EXPTIME":1.0}))
        .collect::<Vec<_>>();
    classify_motion_component_tracks(
        &json!({"status":"analyzed","tracks":[{
        "points":points,"evidence_level":"center_motion"}],"unassociated_single_frame_points":[]}),
        times,
        &headers,
        &vec![[512, 512]; times.len()],
        2.0,
    )
    .unwrap()
}

#[test]
fn all_motion_gates_admit_a_line_and_compact_model_winners_are_rejected() {
    let points = vec![point(0, 100.0), point(1, 120.0), point(2, 150.0)];
    let accepted = classify(points.clone(), &[0.0, 2.0, 5.0]);
    let track = &accepted["tracks"][0];
    assert_eq!(track["classification"], "moving");
    assert!((track["displacement_px"].as_f64().unwrap() - 50.0).abs() < 1e-10);
    assert!((track["speed_px_per_frame"].as_f64().unwrap() - 25.0).abs() < 1e-10);
    assert!(track["rejection_reasons"].as_array().unwrap().is_empty());
    let compact = points
        .into_iter()
        .map(|mut point| {
            point["single_frame_model"] = json!("double");
            point
        })
        .collect();
    let rejected = classify(compact, &[0.0, 2.0, 5.0]);
    assert_eq!(rejected["tracks"][0]["classification"], "candidate");
    assert!(rejected["tracks"][0]["rejection_reasons"]
        .as_array()
        .unwrap()
        .contains(&json!("RAW_LINE_MODEL_MAJORITY_NOT_MET")));
}

#[test]
fn legacy_speed_uses_the_full_sequence_interval_after_a_gap() {
    let result = classify(
        vec![point(0, 100.0), point(2, 120.0), point(3, 200.0)],
        &[0.0, 1.0, 2.0, 10.0],
    );
    assert!((result["tracks"][0]["speed_px_per_frame"].as_f64().unwrap() - 10.0).abs() < 1e-10);
    assert_eq!(result["tracks"][0]["classification"], "candidate");
}

#[test]
fn isolated_raw_review_is_returned_once_as_single_frame_evidence() {
    let raw = point(0, 100.0);
    let result = classify_motion_component_tracks(
        &json!({"status":"analyzed","tracks":[],
        "unassociated_single_frame_points":[{"point":raw,"frame_index":0}]}),
        &[0.0, 1.0],
        &[json!({}), json!({})],
        &[[512, 512]; 2],
        2.0,
    )
    .unwrap();
    assert_eq!(result["tracks"][0]["classification"], "candidate");
    assert_eq!(result["tracks"][0]["evidence_level"], "single_frame_trail");
    assert_eq!(result["tracks"][0]["displacement_px"], 0.0);
}

#[test]
fn public_motion_evidence_preserves_gaps_and_hides_internal_proposal_fields() {
    let mut first = point(0, 100.0);
    first["sequence_candidate_source"] = json!("single_frame_shape");
    first["sequence_stitched"] = json!(true);
    let classified = classify(
        vec![first, point(2, 120.0), point(3, 200.0)],
        &[0.0, 1.0, 2.0, 10.0],
    );
    let tracks = public_motion_feature_tracks(&classified).unwrap();
    assert_eq!(tracks[0]["presence"], 3);
    assert_eq!(tracks[0]["observed_frame_indices"], json!([0, 2, 3]));
    assert_eq!(tracks[0]["observed_frame_ranges"], json!([[0, 0], [2, 3]]));
    assert_eq!(tracks[0]["points"][0]["observation_source"], "detected");
    assert!(tracks[0].get("shape_supported").is_none());
    assert!(tracks[0]["points"][0].get("sequence_stitched").is_none());
    assert!(tracks[0]["evidence_level"]
        .as_str()
        .unwrap()
        .contains("single_frame_stitched"));
    assert_eq!(tracks[0].as_object().unwrap().len(), 15);
    assert!(tracks[0]["points"][0]["single_frame_model_margin_bic"].is_null());
}
