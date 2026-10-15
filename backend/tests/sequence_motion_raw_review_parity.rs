use rst19_core::sequence_motion_tracks::{
    review_temporal_components_against_raw_frames, sequence_line_model_review_supported,
    stitch_and_rescue_motion_component_tracks, supplement_with_coherent_raw_frame_candidates,
};
use serde_json::{json, Value};

#[test]
fn raw_line_model_review_gate_matches_source_derived_boundaries() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/sequence_motion_raw_review.json")).unwrap();
    let psf_fwhm = fixture["psf_fwhm"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let actual = sequence_line_model_review_supported(&case["point"], psf_fwhm).unwrap();
        assert_eq!(actual, case["expected"].as_bool().unwrap(), "{label}");
    }
}

#[test]
fn strong_quantized_ellipse_can_pass_without_beating_the_elliptical_template() {
    let mut point = json!({"single_frame_model":"elliptical_point", "width_px":1.0,"length_px":8.0,
        "line_template_snr":5.0,"line_vs_point_delta_bic":6.0,"line_vs_double_delta_bic":6.0,
        "line_vs_background_delta_bic":6.0,"line_vs_elliptical_delta_bic":-100.0,
        "continuity_ratio":0.80,"profile_fill_ratio":0.80});
    assert!(sequence_line_model_review_supported(&point, 2.0).unwrap());
    point["width_px"] = json!(1.000001);
    assert!(!sequence_line_model_review_supported(&point, 2.0).unwrap());
    point["width_px"] = json!(1.0);
    point["line_vs_double_delta_bic"] = json!(5.999999);
    assert!(!sequence_line_model_review_supported(&point, 2.0).unwrap());
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

#[test]
fn coherent_raw_shape_chain_fills_temporal_candidate_gaps() {
    let times = [0.0, 1.0, 2.0];
    let exposures = [Some(10.0), Some(10.0), Some(10.0)];
    let reviewed = json!({
        "status":"analyzed",
        "frames":[
            {"points":[],"audit":{"feature_count":0}},
            {"points":[],"audit":{"feature_count":0}},
            {"points":[],"audit":{"feature_count":0}},
        ]
    });
    let raw_frames = [0.0, 4.0, 8.0]
        .into_iter()
        .enumerate()
        .map(|(frame_index, x)| {
            let mut point = raw_line("line");
            point["frame_index"] = json!(frame_index);
            point["x"] = json!(x);
            point["aligned_x"] = json!(x);
            json!({"points":[point]})
        })
        .collect::<Vec<_>>();

    let supplemented = supplement_with_coherent_raw_frame_candidates(
        &reviewed,
        &raw_frames,
        &times,
        &exposures,
        2.0,
    )
    .unwrap();

    for frame_index in 0..3 {
        let points = supplemented["frames"][frame_index]["points"]
            .as_array()
            .unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0]["sequence_candidate_source"], "single_frame_shape");
        assert_eq!(
            supplemented["raw_fallback_audit"][frame_index]["coherent_raw_chain_count"],
            1
        );
        assert_eq!(
            supplemented["raw_fallback_audit"][frame_index]["raw_fallback_added_count"],
            1
        );
    }
    assert!(supplemented["unassociated_single_frame_points"]
        .as_array()
        .unwrap()
        .is_empty());

    let associated = rst19_core::sequence_motion_tracks::associate_temporal_component_tracks(
        &supplemented,
        &times,
        &exposures,
        false,
        48.0,
        2.0,
    )
    .unwrap();
    assert_eq!(associated["classification_performed"], false);
    assert_eq!(associated["tracks"].as_array().unwrap().len(), 1);
    assert_eq!(associated["tracks"][0]["point_count"], 3);
}

#[test]
fn nearby_temporal_anchor_admits_raw_shape_candidate_without_a_chain() {
    let times = [0.0, 1.0, 2.0];
    let exposures = [None, None, None];
    let mut anchor = temporal_point();
    anchor["aligned_x"] = json!(0.0);
    anchor["aligned_y"] = json!(0.0);
    let reviewed = json!({
        "status":"analyzed",
        "frames":[
            {"points":[anchor],"audit":{"feature_count":1}},
            {"points":[],"audit":{"feature_count":0}},
            {"points":[],"audit":{"feature_count":0}},
        ]
    });
    let mut anchored_raw = raw_line("line");
    anchored_raw["frame_index"] = json!(2);
    anchored_raw["x"] = json!(10.0);
    anchored_raw["aligned_x"] = json!(10.0);
    let raw_frames = [
        json!({"points": []}),
        json!({"points": []}),
        json!({"points": [anchored_raw]}),
    ];

    let supplemented = supplement_with_coherent_raw_frame_candidates(
        &reviewed,
        &raw_frames,
        &times,
        &exposures,
        2.0,
    )
    .unwrap();
    assert_eq!(
        supplemented["frames"][2]["points"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        supplemented["frames"][2]["points"][0]["sequence_candidate_source"],
        "single_frame_shape"
    );
    assert_eq!(
        supplemented["raw_fallback_audit"][2]["temporal_anchor_count"],
        1
    );
    assert!(supplemented["unassociated_single_frame_points"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[test]
fn isolated_raw_line_stays_in_unassociated_review_audit() {
    let times = [0.0, 1.0, 2.0];
    let exposures = [None, None, None];
    let reviewed = json!({
        "status":"analyzed",
        "frames":[
            {"points":[],"audit":{"feature_count":0}},
            {"points":[],"audit":{"feature_count":0}},
            {"points":[],"audit":{"feature_count":0}},
        ]
    });
    let mut isolated = raw_line("line");
    isolated["frame_index"] = json!(1);
    let raw_frames = [
        json!({"points": []}),
        json!({"points": [isolated]}),
        json!({"points": []}),
    ];

    let supplemented = supplement_with_coherent_raw_frame_candidates(
        &reviewed,
        &raw_frames,
        &times,
        &exposures,
        2.0,
    )
    .unwrap();
    assert!(supplemented["frames"][1]["points"]
        .as_array()
        .unwrap()
        .is_empty());
    let unassociated = supplemented["unassociated_single_frame_points"]
        .as_array()
        .unwrap();
    assert_eq!(unassociated.len(), 1);
    assert_eq!(unassociated[0]["frame_index"], 1);
    assert_eq!(
        unassociated[0]["point"]["sequence_candidate_source"],
        "single_frame_trail_review_only"
    );
}

fn line_at(frame_index: usize, x: f64, length_px: f64) -> Value {
    let mut point = raw_line("line");
    point["frame_index"] = json!(frame_index);
    point["x"] = json!(x);
    point["aligned_x"] = json!(x);
    point["length_px"] = json!(length_px);
    point
}

#[test]
fn short_review_supported_singleton_stitches_into_existing_candidate_track() {
    let times = [0.0, 1.0, 2.0, 3.0];
    let track_points = vec![
        line_at(0, 0.0, 20.0),
        line_at(2, 4.0, 20.0),
        line_at(3, 6.0, 20.0),
    ];
    let mut missing = line_at(1, 2.0, 20.0);
    missing["sequence_candidate_source"] = json!("single_frame_trail_review_only");
    let component_tracks = json!({
        "status":"analyzed",
        "classification_performed":false,
        "tracks":[{"track_id":0,"point_count":3,"points":track_points}],
        "frame_audits":[]
    });
    let unassociated = json!([{"frame_index":1,"point":missing}]);

    let stitched =
        stitch_and_rescue_motion_component_tracks(&component_tracks, &unassociated, &times, 2.0)
            .unwrap();
    assert_eq!(stitched["tracks"].as_array().unwrap().len(), 1);
    assert_eq!(stitched["tracks"][0]["point_count"], 4);
    assert_eq!(stitched["stitching_audit"]["stitched_point_count"], 1);
    assert!(stitched["unassociated_single_frame_points"]
        .as_array()
        .unwrap()
        .is_empty());
    let point = stitched["tracks"][0]["points"]
        .as_array()
        .unwrap()
        .iter()
        .find(|point| point["frame_index"] == 1)
        .unwrap();
    assert_eq!(point["sequence_stitched"], true);
    assert_eq!(point["sequence_candidate_source"], "single_frame_shape");
    assert_eq!(stitched["classification_performed"], false);
}

#[test]
fn collinear_long_singletons_are_rescued_without_classification() {
    let times = [0.0, 1.0, 2.0];
    let component_tracks = json!({
        "status":"analyzed",
        "classification_performed":false,
        "tracks":[
            {"track_id":0,"point_count":1,"points":[line_at(0,0.0,40.0)]},
            {"track_id":1,"point_count":1,"points":[line_at(1,40.0,40.0)]},
            {"track_id":2,"point_count":1,"points":[line_at(2,80.0,40.0)]},
        ],
        "frame_audits":[]
    });
    let rescued =
        stitch_and_rescue_motion_component_tracks(&component_tracks, &json!([]), &times, 2.0)
            .unwrap();
    assert_eq!(rescued["collinear_rescue_audit"]["rescued_group_count"], 1);
    assert_eq!(rescued["collinear_rescue_audit"]["rescued_point_count"], 3);
    assert_eq!(rescued["tracks"].as_array().unwrap().len(), 1);
    assert_eq!(rescued["tracks"][0]["evidence_level"], "collinear_trail");
    assert_eq!(rescued["tracks"][0]["point_count"], 3);
    assert_eq!(rescued["tracks"][0]["candidate_only"], true);
    assert_eq!(rescued["classification_performed"], false);
}
