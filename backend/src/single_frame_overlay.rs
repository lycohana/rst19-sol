//! Current-exposure line evidence for the desktop single-frame view.
//! No temporal reference, cross-frame association, or moving classification.
use crate::{
    pipeline::FrameAnalysis, sequence_motion_features, sequence_motion_tracks, sequence_trails,
    Result,
};
use serde_json::{json, Value};

fn number(point: &Value, key: &str) -> Option<f64> {
    point[key].as_f64().filter(|value| value.is_finite())
}

/// Match the legacy single-frame display gates, including weak line-model
/// winners and the explicitly uncertain long-line/artifact conflict layer.
pub fn display_evidence(point: &Value) -> Result<Value> {
    let shape = sequence_trails::trail_shape_evidence(&json!({"point":point}))?;
    let model_supported = shape["single_frame_model_supported"] == true;
    let review_supported = point["single_frame_model"] == "line"
        && number(point, "line_template_snr").is_some_and(|value| value >= 5.0)
        && [
            "line_vs_point_delta_bic",
            "line_vs_double_delta_bic",
            "line_vs_background_delta_bic",
            "line_vs_elliptical_delta_bic",
            "single_frame_model_margin_bic",
        ]
        .iter()
        .all(|key| number(point, key).is_some_and(|value| value >= 2.0));
    let artifact_overlap =
        number(point, "line_artifact_overlap_fraction").is_some_and(|value| value >= 0.75);
    let artifact_review = artifact_overlap
        && point["single_frame_model"] == "long_line_profile"
        && number(point, "length_px").is_some_and(|value| value > 128.0);
    let visible = (model_supported || review_supported) && (!artifact_overlap || artifact_review);
    let solid =
        visible && model_supported && shape["single_frame_supported"] == true && !artifact_review;
    Ok(
        json!({"visible":visible,"solid":solid,"artifact_review":artifact_review,
        "model_supported":model_supported,"shape_supported":shape["single_frame_supported"],
        "reason":shape["single_frame_reason"]}),
    )
}

fn priority(point: &Value) -> (bool, bool, bool) {
    let display = &point["display"];
    (
        display["shape_supported"] == true,
        display["model_supported"] == true,
        number(point, "line_artifact_overlap_fraction").is_none_or(|value| value < 0.75),
    )
}

pub fn analyze(
    analysis: &FrameAnalysis,
    psf_fwhm: f64,
    progress: &mut dyn FnMut(f32, &str),
) -> Result<Value> {
    progress(0.0, "提取当前曝光的连通线形与局部模型证据");
    let exposure_s = crate::fits::exposure_milliseconds(&analysis.frame.header)?
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| value / 1000.0);
    let thresholds = sequence_trails::motion_proposal_thresholds(&json!({
        "psf_fwhm":psf_fwhm,"exposure_s":[exposure_s],"metadata_supported":exposure_s.is_some(),
        "short_exposure_s":exposure_s
    }))?;
    let min_length = thresholds["min_length_px"]
        .as_f64()
        .ok_or("missing trail length floor")?;
    let short_area = thresholds["short_min_area"]
        .as_u64()
        .ok_or("missing trail area floor")? as usize;
    let mut components = sequence_motion_features::sequence_single_frame_components_from_analysis(
        analysis, psf_fwhm, 10.0, 60.0, 20, 2.5, min_length, short_area,
    )?;
    if analysis.frame.height.min(analysis.frame.width) >= 512 {
        progress(
            0.45,
            "在当前原图上补充多尺度线段提案并复核单星、双星和线段模型",
        );
        crate::sequence_motion_recovery::augment_multiscale_raw_components(
            &mut components,
            analysis,
            None,
            0,
            [0.0, 0.0],
            psf_fwhm,
            min_length,
        )?;
    }
    progress(0.90, "合并共线重复片段并确定实线、虚线显示证据");
    let mut points = components["points"]
        .as_array()
        .cloned()
        .ok_or("missing single-frame trail points")?;
    for (index, point) in points.iter_mut().enumerate() {
        point["display"] = display_evidence(point)?;
        point["overlay_id"] = json!(index);
    }
    points.sort_by(|a, b| {
        priority(b)
            .cmp(&priority(a))
            .then_with(|| {
                number(b, "length_px")
                    .unwrap_or(0.0)
                    .total_cmp(&number(a, "length_px").unwrap_or(0.0))
            })
            .then_with(|| {
                number(b, "residual_snr")
                    .unwrap_or(0.0)
                    .total_cmp(&number(a, "residual_snr").unwrap_or(0.0))
            })
    });
    let mut kept = Vec::new();
    for point in points {
        if !kept
            .iter()
            .any(|existing| sequence_motion_tracks::line_geometries_overlap(&point, existing))
        {
            kept.push(point);
        }
    }
    kept.sort_by_key(|point| point["overlay_id"].as_u64().unwrap_or(0));
    let overlay_count = kept
        .iter()
        .filter(|point| point["display"]["visible"] == true)
        .count();
    let solid_count = kept
        .iter()
        .filter(|point| point["display"]["solid"] == true)
        .count();
    let tracks = kept.into_iter().map(|point| json!({
        "track_id":point["overlay_id"],"classification":"candidate","evidence_level":"single_frame_trail",
        "presence":1,"points":[point]
    })).collect::<Vec<_>>();
    progress(1.0, "单帧曝光线形已完成；保留候选身份与原始像素几何");
    Ok(
        json!({"schema":"rst19.single_frame_trails.v1","path":analysis.frame.path,
        "frame_count":1,"exposure_s":exposure_s,"candidate_count":tracks.len(),
        "overlay_count":overlay_count,"solid_count":solid_count,"review_count":overlay_count-solid_count,
        "tracks":tracks,"audit":components["audit"],"parameters":{
            "psf_fwhm":psf_fwhm,"residual_sigma":10.0,"min_residual_adu":60.0,
            "min_feature_area":20,"min_axis_ratio":2.5,"min_length_px":min_length,"short_min_area":short_area},
        "interpretation":"仅当前曝光的线形候选；实线表示局部线段模型和形态通过，虚线表示待复核；不确认跨帧运动或物理身份。"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line() -> Value {
        json!({"x":40.0,"y":40.0,"length_px":28.0,"width_px":3.0,"angle_deg":0.0,
            "single_frame_model":"line","line_template_snr":30.0,
            "line_vs_point_delta_bic":10.0,"line_vs_double_delta_bic":10.0,
            "line_vs_background_delta_bic":10.0,"line_vs_elliptical_delta_bic":10.0,
            "single_frame_model_margin_bic":10.0,"continuity_ratio":0.95,"profile_fill_ratio":0.95,
            "line_artifact_overlap_fraction":0.0})
    }

    #[test]
    fn current_exposure_display_keeps_strong_weak_and_artifact_evidence_distinct() {
        let mut point = line();
        assert_eq!(display_evidence(&point).unwrap()["solid"], true);
        point["line_vs_double_delta_bic"] = json!(3.0);
        let weak = display_evidence(&point).unwrap();
        assert_eq!(weak["visible"], true);
        assert_eq!(weak["solid"], false);
        point["single_frame_model"] = json!("double");
        assert_eq!(display_evidence(&point).unwrap()["visible"], false);
        point = line();
        point["line_artifact_overlap_fraction"] = json!(0.95);
        assert_eq!(display_evidence(&point).unwrap()["visible"], false);
        point["single_frame_model"] = json!("long_line_profile");
        point["length_px"] = json!(200.0);
        let artifact = display_evidence(&point).unwrap();
        assert_eq!(artifact["visible"], true);
        assert_eq!(artifact["solid"], false);
        assert_eq!(artifact["artifact_review"], true);
    }

    #[test]
    fn raw_exposure_segment_is_exported_without_promoting_a_compact_double() {
        let width = 128;
        let height = 128;
        let data = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let mut value = 1000.0;
                    if (38..=90).contains(&x) {
                        value += 250.0 * (-0.5 * ((y as f64 - 82.0) / 1.25).powi(2)).exp();
                    }
                    for center in [40.0, 46.0] {
                        value += 300.0
                            * (-0.5 * ((x as f64 - center).powi(2) + (y as f64 - 30.0).powi(2))
                                / 1.27_f64.powi(2))
                            .exp();
                    }
                    value as f32
                })
            })
            .collect();
        let frame = crate::models::FitsFrame {
            path: "single-exposure.fits".into(),
            width,
            height,
            auxiliary: None,
            data_offset: 0,
            header: serde_json::from_value(json!({"EXPOSURE":1500})).unwrap(),
            data: crate::models::PixelData::F32(data),
        };
        let analysis = FrameAnalysis {
            frame,
            detection: crate::detection::DetectionResult {
                image_shape: [height, width],
                background: 1000.0,
                noise: 2.0,
                threshold: 1008.0,
                candidate_count: 0,
                quality_count: 0,
                sources: vec![],
                candidate_peaks: vec![],
                line_artifact_coordinates: vec![],
                parameters: json!({"mask_zero_pixels":false,"saturation_level":-1.0}),
            },
            matching: None,
            faintest: None,
            photometric_calibration: None,
            source_photometry: vec![],
        };
        let result = analyze(&analysis, 3.0, &mut |_: f32, _: &str| {}).unwrap();
        let tracks = result["tracks"].as_array().unwrap();
        assert!(!tracks.is_empty());
        assert!(tracks
            .iter()
            .all(|track| track["classification"] == "candidate" && track["presence"] == 1));
        assert!(
            tracks.iter().any(|track| {
                let point = &track["points"][0];
                point["display"]["solid"] == true
                    && (point["y"].as_f64().unwrap() - 82.0).abs() < 2.0
                    && point["length_px"].as_f64().unwrap() > 45.0
            }),
            "{result}"
        );
        assert!(
            !tracks.iter().any(|track| {
                let point = &track["points"][0];
                point["display"]["visible"] == true
                    && (point["y"].as_f64().unwrap() - 30.0).abs() < 5.0
            }),
            "{result}"
        );
        assert_eq!(result["exposure_s"], 1.5);
    }
}
