use rst19_core::{sequence_motion_features, single_frame_overlay};
use serde_json::{json, Value};

/// Frozen real FITS crops, with full-frame Python 50f030b measurements.
/// Recompute the geometry and competing model, then verify it remains visible
/// for review without becoming accepted line/motion evidence.
#[test]
fn first_frame_small_lines_retain_pixel_geometry_and_model_rejection() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/first_frame_short_geometry.json")).unwrap();
    assert_eq!(
        fixture["input_sha256"],
        "6d6926a0546a5e2d28c6e76762bf72127ff3f0e37b589993eb381881bfe0acc6"
    );
    let background = fixture["background"].as_f64().unwrap();
    let noise = fixture["noise"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let residual = case["pixels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|pixel| pixel.as_f64().unwrap() - background)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let height = residual.len();
        let width = residual[0].len();
        let result = sequence_motion_features::motion_feature_frame_components(&json!({
            "residual":residual,"valid_mask":vec![vec![true;width];height],
            "line_artifact_mask":vec![vec![false;width];height],
            "frame_index":0,"shift":[0.0,0.0],"threshold_adu":60.0_f64.max(10.0*noise),
            "residual_noise_adu":noise,"psf_fwhm":3.0,"min_length_px":7.5,
            "min_feature_area":20,"min_axis_ratio":2.5,"bridge_gaps":false,
            "short_min_pixels":18,"fit_line_model":true
        }))
        .unwrap();
        let expected = &case["expected"];
        let x = expected["x"].as_f64().unwrap() - case["offset"][0].as_f64().unwrap();
        let y = expected["y"].as_f64().unwrap() - case["offset"][1].as_f64().unwrap();
        let point = result["points"]
            .as_array()
            .unwrap()
            .iter()
            .find(|point| {
                (point["x"].as_f64().unwrap() - x).abs() < 1e-6
                    && (point["y"].as_f64().unwrap() - y).abs() < 1e-6
            })
            .unwrap_or_else(|| panic!("missing real short line at ({x},{y}): {result}"));
        for field in ["length_px", "width_px"] {
            assert!(
                (point[field].as_f64().unwrap() - expected[field].as_f64().unwrap()).abs() < 1e-6,
                "{field}: {point}"
            );
        }
        let angle_gap = (point["angle_deg"].as_f64().unwrap()
            - expected["angle_deg"].as_f64().unwrap())
        .rem_euclid(180.0);
        assert!(angle_gap.min(180.0 - angle_gap) < 1e-6);
        assert_eq!(point["single_frame_model"], expected["single_frame_model"]);
        let display = single_frame_overlay::display_evidence(point).unwrap();
        assert_eq!(display["geometry_review"], true);
        assert_eq!(display["visible"], false);
        assert_eq!(display["solid"], false);
    }
}
