use rst19_core::{
    detection::DetectionResult,
    models::{FitsFrame, PixelData},
    pipeline::FrameAnalysis,
    sequence_motion_recovery::{augment_multiscale_raw_components, recover_predicted_roi_points},
};
use serde_json::{json, Value};
use std::path::PathBuf;

fn frame(index: usize, inject: bool) -> FrameAnalysis {
    let mut data = vec![100.0_f32; 128 * 128];
    if inject {
        let cx = 30.0 + index as f64 * 4.0;
        let sigma = 2.0 / 2.354820045;
        for y in 0..128 {
            for x in 0..128 {
                let signal = (0..49)
                    .map(|sample| {
                        let t = -6.0 + 12.0 * sample as f64 / 48.0;
                        (-((x as f64 - cx - t).powi(2) + (y as f64 - 64.0).powi(2))
                            / (2.0 * sigma * sigma))
                            .exp()
                    })
                    .sum::<f64>()
                    / 49.0;
                data[y * 128 + x] += (200.0 * signal) as f32;
            }
        }
    }
    FrameAnalysis {
        frame: FitsFrame {
            path: PathBuf::from(format!("synthetic-{index}.fits")),
            header: serde_json::Map::new(),
            data: PixelData::F32(data),
            auxiliary: None,
            data_offset: 0,
            width: 128,
            height: 128,
        },
        detection: DetectionResult {
            image_shape: [128, 128],
            background: 100.0,
            noise: 1.0,
            threshold: 4.0,
            candidate_count: 0,
            sources: vec![],
            quality_count: 0,
            parameters: json!({"mask_zero_pixels":0,"saturation_level":-1.0}),
            candidate_peaks: vec![],
            line_artifact_coordinates: vec![],
        },
        matching: None,
        faintest: None,
        photometric_calibration: None,
        source_photometry: vec![],
    }
}

fn candidates() -> Value {
    let points = [0, 1, 2, 4, 5, 6, 7]
        .into_iter()
        .map(|frame| {
            json!({
                "frame_index":frame,"x":30.0+4.0*frame as f64,"y":64.0,
                "aligned_x":30.0+4.0*frame as f64,"aligned_y":64.0,"length_px":12.0,
            })
        })
        .collect::<Vec<_>>();
    json!({"tracks":[{"points":points,"point_count":7}]})
}

#[test]
fn missing_frame_is_recovered_only_when_original_pixels_support_a_line() {
    let times = (0..9).map(|index| index as f64).collect::<Vec<_>>();
    let shifts = vec![[0.0, 0.0]; 9];
    let analyses = (0..9)
        .map(|index| frame(index, index == 3))
        .collect::<Vec<_>>();
    let mut recovered = candidates();
    recover_predicted_roi_points(&mut recovered, &analyses, &shifts, &times, 2.0).unwrap();
    let point = recovered["tracks"][0]["points"]
        .as_array()
        .unwrap()
        .iter()
        .find(|point| point["frame_index"] == 3)
        .expect("injected line must be remeasured");
    assert_eq!(point["observation_source"], "recovered_by_predicted_roi");
    assert!((point["x"].as_f64().unwrap() - 42.0).abs() <= 2.5);
    assert_eq!(
        recovered["predicted_roi_recovery_audit"]["recovered_point_count"],
        1
    );
    let blank = (0..9).map(|index| frame(index, false)).collect::<Vec<_>>();
    let mut unchanged = candidates();
    recover_predicted_roi_points(&mut unchanged, &blank, &shifts, &times, 2.0).unwrap();
    assert_eq!(unchanged["tracks"][0]["point_count"], 7);
    assert_eq!(
        unchanged["predicted_roi_recovery_audit"]["recovered_point_count"],
        0
    );
}

#[test]
fn multiscale_proposals_measure_original_pixels_and_blank_frames_have_no_additions() {
    let analysis = frame(3, true);
    let mut components = json!({"points":[]});
    augment_multiscale_raw_components(&mut components, &analysis, None, 3, [4.0, 2.0], 2.0, 6.0)
        .unwrap();
    let points = components["points"].as_array().unwrap();
    assert!(
        !points.is_empty(),
        "injected raw line must produce a measured proposal"
    );
    for point in points {
        assert_eq!(point["frame_index"], 3);
        assert!(
            (point["x"].as_f64().unwrap() - point["aligned_x"].as_f64().unwrap() - 4.0).abs()
                < 1e-10
        );
        assert!(point["line_template_snr"].as_f64().unwrap() > 0.0);
    }
    let mut blank = json!({"points":[]});
    augment_multiscale_raw_components(&mut blank, &frame(3, false), None, 3, [4.0, 2.0], 2.0, 6.0)
        .unwrap();
    assert_eq!(blank["multiscale_added_count"], 0);
    assert!(blank["points"].as_array().unwrap().is_empty());
}

#[test]
fn four_frame_streak_matches_the_legacy_source_regression_contract() {
    // Frozen construction and assertions from tests/test_sequence.py:
    // _feature_analysis and test_detect_motion_features_tracks_a_moving_streak_separately.
    // This is a source-derived regression case, not a live Python oracle.
    let analyses = (0..4)
        .map(|index| {
            let mut analysis = frame(index, false);
            let mut pixels = vec![10.0_f32; 96 * 96];
            for offset in 0..42 {
                let x = 6 + 8 * index + offset;
                let y = 15 + 6 * index + offset;
                if x < 96 && y < 96 {
                    pixels[y * 96 + x] = 300.0;
                }
            }
            analysis.frame.width = 96;
            analysis.frame.height = 96;
            analysis.frame.data = PixelData::F32(pixels);
            analysis.detection.image_shape = [96, 96];
            analysis.detection.background = 10.0;
            analysis.detection.noise = 2.0;
            analysis.detection.threshold = 40.0;
            analysis
        })
        .collect::<Vec<_>>();
    let public = native_four_frame_features(&analyses, 3.0);
    assert_eq!(public.len(), 1);
    assert_eq!(public[0]["classification"], "moving");
    assert_eq!(public[0]["presence"], 4);
    assert!((public[0]["displacement_px"].as_f64().unwrap() - 30.0).abs() < 0.5);
}

fn native_four_frame_features(analyses: &[FrameAnalysis], psf: f64) -> Vec<Value> {
    use rst19_core::{
        sequence_motion_features as features, sequence_motion_tracks as tracks, sequence_temporal,
    };
    let shifts = [[0.0, 0.0]; 4];
    let times = [0.0, 1.0, 2.0, 3.0];
    let reference =
        sequence_temporal::registered_reference_from_analyses(&analyses, &shifts, false, 3.0)
            .unwrap();
    let temporal = features::motion_frame_components_from_analyses(
        &analyses,
        reference.as_ref(),
        &shifts,
        psf,
        15.0,
        100.0,
        40,
        4.0,
        None,
        true,
    )
    .unwrap();
    assert_eq!(temporal["thresholds_adu"], json!(vec![100.0; 4]));
    let raw = analyses
        .iter()
        .enumerate()
        .map(|(index, analysis)| {
            let mut components = features::sequence_single_frame_components_from_analysis(
                analysis,
                psf,
                10.0,
                60.0,
                20,
                2.5,
                8.0_f64.max(3.0 * psf),
                20,
            )
            .unwrap();
            for point in components["points"].as_array_mut().unwrap() {
                point["frame_index"] = json!(index);
            }
            components
        })
        .collect::<Vec<_>>();
    let reviewed =
        tracks::review_temporal_components_against_raw_frames(&temporal, &raw, &shifts, psf)
            .unwrap();
    let fallback = tracks::supplement_with_coherent_raw_frame_candidates(
        &reviewed, &raw, &times, &[None; 4], psf,
    )
    .unwrap();
    let associated = tracks::associate_temporal_component_tracks(
        &fallback, &times, &[None; 4], false, 96.0, psf,
    )
    .unwrap();
    let mut assembled = tracks::stitch_and_rescue_motion_component_tracks(
        &associated,
        &fallback["unassociated_single_frame_points"],
        &times,
        psf,
    )
    .unwrap();
    recover_predicted_roi_points(&mut assembled, &analyses, &shifts, &times, psf).unwrap();
    let classified = tracks::classify_motion_component_tracks(
        &assembled,
        &times,
        &[json!({}), json!({}), json!({}), json!({})],
        &[[96, 96]; 4],
        psf,
    )
    .unwrap();
    tracks::public_motion_feature_tracks(&classified).unwrap()
}

#[test]
fn static_quantized_line_is_retained_as_single_frame_evidence() {
    // Source-derived case: test_detect_motion_features_suppresses_a_static_streak_with_temporal_reference.
    let mut analyses = Vec::new();
    for index in 0..4 {
        let mut analysis = frame(index, false);
        let mut pixels = vec![10.0_f32; 96 * 96];
        for offset in 0..42 {
            pixels[(15 + offset) * 96 + 12 + offset] = 300.0;
        }
        analysis.frame.width = 96;
        analysis.frame.height = 96;
        analysis.frame.data = PixelData::F32(pixels);
        analysis.detection.image_shape = [96, 96];
        analysis.detection.background = 10.0;
        analysis.detection.noise = 2.0;
        analyses.push(analysis);
    }
    let public = native_four_frame_features(&analyses, 3.0);
    assert!(!public.is_empty());
    assert!(public
        .iter()
        .all(|track| track["classification"] == "candidate"
            && track["evidence_level"] == "single_frame_trail"
            && track["presence"] == 1));
}
