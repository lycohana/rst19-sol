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
