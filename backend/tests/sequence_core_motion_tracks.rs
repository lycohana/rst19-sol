use rst19_core::sequence_pipeline::handle_request;
use serde_json::json;
use std::path::PathBuf;

#[test]
fn sequence_core_returns_separate_unclassified_motion_component_tracks() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let frame = workspace
        .join("tests")
        .join("fixtures")
        .join("contaminated_pair_injection_parity.fits");
    let result = handle_request(&json!({
        "paths":[frame, frame],
        "fast_point_motion":false,
        "stack_faint_recovery":false,
        "frame_options":{
            "max_sources":100,
            "background_box_size":64,
            "use_float32":true,
            "fast_sequence":true,
        },
    }))
    .unwrap();

    assert_eq!(result["motion_frame_components"]["status"], "analyzed");
    assert_eq!(result["motion_frame_model_review"]["status"], "analyzed");
    assert_eq!(
        result["motion_frame_model_review"]["raw_fallback_audit"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(result["motion_component_tracks"]["status"], "analyzed");
    let public = &result["sequence_result"];
    assert_eq!(public["frame_count"], 2);
    assert_eq!(public["frames"][0]["frame_index"], 0);
    assert_eq!(public["frames"][1]["frame_index"], 1);
    assert_eq!(public["calculation_dtype"], "float32");
    assert_eq!(public["source_working_limit"], 100);
    assert_eq!(public["tracks"], result["tracks"]);
    assert_eq!(public["motion_features"], result["motion_feature_tracks"]);
    assert_eq!(
        public["motion_evidence_links"],
        result["motion_evidence_links"]
    );
    assert_eq!(
        public["motion_evidence_link_count"].as_u64().unwrap() as usize,
        public["motion_evidence_links"].as_array().unwrap().len()
    );
    assert_eq!(
        public["frames"][0]["returned_count"].as_u64().unwrap() as usize,
        result["frame_analyses"][0]["detection"]["sources"]
            .as_array()
            .unwrap()
            .len()
    );
    assert_eq!(
        result["motion_component_tracks"]["classification_performed"],
        false
    );
    assert_eq!(
        result["motion_component_tracks"]["scope"],
        "reviewed_motion_candidates_with_stitching_and_collinear_rescue"
    );
    assert_eq!(
        result["motion_component_tracks"]["collinear_rescue_audit"]["rescued_group_count"],
        0
    );
    assert_eq!(
        result["motion_component_tracks"]["frame_audits"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}
