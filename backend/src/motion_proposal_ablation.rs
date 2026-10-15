//! Paired frozen pixel inputs through the native production motion gates.
use crate::{
    detection::DetectionResult,
    frozen_arrays, innovation_artifacts,
    models::{FitsFrame, PixelData},
    numpy_ziggurat,
    pipeline::FrameAnalysis,
    sequence_motion_features as features, sequence_motion_recovery as recovery,
    sequence_motion_tracks as tracks,
    sequence_temporal::RegisteredReference,
    sequence_trails,
    source_artifacts::Chart,
    source_geometry_audit::NumpyPcg64,
    statistics, Result,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub const HELP:&str="比较基础和多尺度运动线提案；使用同一冻结噪声与注入像素\nUsage: rst19-motion-proposal-ablation --point-run-dir DIR [--association-run-dir DIR] [--out-dir DIR] [--report HTML] [--seed N]";
const N: usize = 4;
const H: usize = 96;
const W: usize = 96;

pub fn frozen_inputs(seed: u128) -> (Vec<f32>, Vec<f32>) {
    let mut rng = NumpyPcg64::new(seed);
    let control = (0..N * H * W)
        .map(|_| (10. + numpy_ziggurat::standard_normal(&mut rng)) as f32)
        .collect::<Vec<_>>();
    let mut injected = control.clone();
    let sigma: f64 = 3. / 2.354820045;
    for frame in 0..N {
        for sample in 0..=24 {
            let sx = 22. + 10. * frame as f64 - 6. + 0.5 * sample as f64;
            for y in 0..H {
                for x in 0..W {
                    let v = (4.
                        * (-0.5 * ((x as f64 - sx).powi(2) + (y as f64 - 40.).powi(2))
                            / sigma.powi(2))
                        .exp()) as f32
                        * 0.5;
                    injected[frame * H * W + y * W + x] += v;
                }
            }
        }
    }
    (control, injected)
}
fn analyses(images: &[f32], prefix: &str) -> Vec<FrameAnalysis> {
    images
        .chunks_exact(H * W)
        .enumerate()
        .map(|(i, pixels)| FrameAnalysis {
            frame: FitsFrame {
                path: PathBuf::from(format!("{prefix}-{i:02}.fits")),
                header: Default::default(),
                data: PixelData::F32(pixels.to_vec()),
                auxiliary: None,
                data_offset: 0,
                width: W,
                height: H,
            },
            detection: DetectionResult {
                image_shape: [H, W],
                background: 10.,
                noise: 1.,
                threshold: 100.,
                candidate_count: 0,
                sources: vec![],
                quality_count: 0,
                parameters: json!({}),
                candidate_peaks: vec![],
                line_artifact_coordinates: vec![],
            },
            matching: None,
            faintest: None,
            photometric_calibration: None,
            source_photometry: vec![],
        })
        .collect()
}
pub fn run_version(images: &[f32], multiscale: bool) -> Result<(Vec<Value>, Vec<Value>)> {
    if images.len() != N * H * W {
        return Err("motion ablation requires four 96x96 frozen frames".into());
    }
    let analyses = analyses(images, "frozen-motion");
    let shifts = vec![[0., 0.]; N];
    let times = vec![0., 1., 2., 3.];
    let exposures = vec![None; N];
    let reference = RegisteredReference {
        values: vec![10.; H * W],
        height: H,
        width: W,
    };
    let temporal = features::motion_frame_components_from_analyses(
        &analyses,
        Some(&reference),
        &shifts,
        3.,
        15.,
        100.,
        40,
        4.,
        Some(8.),
        true,
    )?;
    let thresholds = sequence_trails::motion_proposal_thresholds(
        &json!({"psf_fwhm":3.,"min_length_px":8.,"exposure_s":exposures,"metadata_supported":false,"short_exposure_s":null}),
    )?;
    let short = thresholds["short_min_area"]
        .as_u64()
        .ok_or("missing short proposal area")? as usize;
    let mut raw = Vec::new();
    for (i, analysis) in analyses.iter().enumerate() {
        let mut components = features::sequence_single_frame_components_from_analysis(
            analysis, 3., 10., 60., 20, 2.5, 8., short,
        )?;
        for point in components["points"]
            .as_array_mut()
            .ok_or("missing raw motion points")?
        {
            point["frame_index"] = json!(i);
            point["aligned_x"] = point["x"].clone();
            point["aligned_y"] = point["y"].clone();
        }
        if multiscale {
            recovery::augment_multiscale_raw_components(
                &mut components,
                analysis,
                Some(&reference),
                i,
                [0., 0.],
                3.,
                8.,
            )?;
        }
        raw.push(components);
    }
    let reviewed =
        tracks::review_temporal_components_against_raw_frames(&temporal, &raw, &shifts, 3.)?;
    let reviewed = tracks::supplement_with_coherent_raw_frame_candidates(
        &reviewed, &raw, &times, &exposures, 3.,
    )?;
    let associated =
        tracks::associate_temporal_component_tracks(&reviewed, &times, &exposures, false, 96., 3.)?;
    let mut stitched = tracks::stitch_and_rescue_motion_component_tracks(
        &associated,
        &reviewed["unassociated_single_frame_points"],
        &times,
        3.,
    )?;
    recovery::recover_predicted_roi_points(&mut stitched, &analyses, &shifts, &times, 3.)?;
    let classified = tracks::classify_motion_component_tracks(
        &stitched,
        &times,
        &vec![json!({}); N],
        &vec![[H, W]; N],
        3.,
    )?;
    let public = tracks::public_motion_feature_tracks(&classified)?;
    let audits = reviewed["frames"]
        .as_array()
        .ok_or("missing motion frame audit")?
        .iter()
        .map(|v| v["audit"].clone())
        .collect();
    Ok((public, audits))
}
fn match_errors(tracks: &[Value]) -> Vec<f64> {
    (0..N)
        .filter_map(|i| {
            tracks
                .iter()
                .filter(|t| t["classification"] == "moving")
                .flat_map(|t| t["points"].as_array().into_iter().flatten())
                .filter(|p| p["frame_index"].as_u64() == Some(i as u64))
                .map(|p| {
                    (p["x"].as_f64().unwrap_or(f64::INFINITY) - (22. + 10. * i as f64))
                        .hypot(p["y"].as_f64().unwrap_or(f64::INFINITY) - 40.)
                })
                .filter(|v| *v <= 7.)
                .min_by(f64::total_cmp)
        })
        .collect()
}
fn covers(tracks: &[Value]) -> usize {
    tracks
        .iter()
        .filter(|t| {
            t["classification"] == "moving"
                && (0..N).all(|i| {
                    t["points"].as_array().into_iter().flatten().any(|p| {
                        p["frame_index"].as_u64() == Some(i as u64)
                            && (p["x"].as_f64().unwrap_or(f64::INFINITY) - (22. + 10. * i as f64))
                                .hypot(p["y"].as_f64().unwrap_or(f64::INFINITY) - 40.)
                                <= 7.
                    })
                })
        })
        .count()
}
fn case_sheet(path: &Path, pixels: &[f32], versions: &[Vec<Value>; 2]) -> Result<()> {
    let mut chart = Chart::new(480, 526, [24, 31, 42]);
    for i in 0..N {
        let data = &pixels[i * H * W..(i + 1) * H * W];
        let values = data.iter().map(|v| *v as f64).collect::<Vec<_>>();
        let lo = statistics::percentile(&values, 5.)?;
        let hi = statistics::percentile(&values, 99.8)?;
        let ox = (i % 2) * 240;
        let oy = (i / 2) * 240;
        for y in 0..206 {
            for x in 0..222 {
                let v = data[(y * H / 206) * W + x * W / 222];
                let gray = (((v as f64 - lo) / (hi - lo).max(1e-6)).clamp(0., 1.) * 255.) as u8;
                chart.pixel((ox + 9 + x) as i32, (oy + 22 + y) as i32, [gray; 3]);
            }
        }
        chart.text(
            (ox + 8) as i32,
            (oy + 4) as i32,
            &format!("F{i}"),
            [235, 241, 249],
        );
        let mut marker = |x: f64, y: f64, color: [u8; 3]| {
            let px = (ox as f64 + 9. + x * 222. / W as f64).round() as i32;
            let py = (oy as f64 + 22. + y * 206. / H as f64).round() as i32;
            chart.line((px - 4, py), (px + 4, py), color);
            chart.line((px, py - 4), (px, py + 4), color);
        };
        marker(22. + 10. * i as f64, 40., [255, 222, 80]);
        for (j, tracks) in versions.iter().enumerate() {
            for track in tracks {
                for p in track["points"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|p| p["frame_index"].as_u64() == Some(i as u64))
                {
                    marker(
                        p["x"].as_f64().unwrap_or(0.),
                        p["y"].as_f64().unwrap_or(0.),
                        if j == 0 {
                            [78, 168, 255]
                        } else {
                            [72, 224, 161]
                        },
                    );
                }
            }
        }
    }
    chart.text(
        10,
        492,
        "Yellow=truth Blue=baseline Green=multiscale",
        [235, 241, 249],
    );
    chart.save(path)
}
pub fn run(out: &Path, seed: u128) -> Result<Value> {
    if out.exists()
        && fs::read_dir(out)
            .map_err(|e| e.to_string())?
            .next()
            .is_some()
    {
        return Err(format!(
            "refusing to overwrite non-empty experiment directory: {}",
            out.display()
        ));
    }
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let out = out.canonicalize().map_err(|e| e.to_string())?;
    let (control, injected) = frozen_inputs(seed);
    let shape = [N, H, W];
    let input = out.join("frozen_motion_frames.npz");
    let hash = frozen_arrays::write_npz_f32(
        &input,
        &[
            ("control_frames", &control, &shape),
            ("injected_frames", &injected, &shape),
        ],
    )?;
    let mut summary = Vec::new();
    let mut audits = Vec::new();
    let mut track_rows = Vec::new();
    let mut positive: [Vec<Value>; 2] = [vec![], vec![]];
    let mut runtimes = serde_json::Map::new();
    for (scene, pixels) in [
        ("no_injection_control", &control),
        ("low_peak_moving_trail", &injected),
    ] {
        let array_hash = frozen_arrays::array_sha256(pixels, &shape);
        for multiscale in [false, true] {
            let started = Instant::now();
            let (tracks, frame_audits) = run_version(pixels, multiscale)?;
            let seconds = started.elapsed().as_secs_f64();
            let version = if multiscale {
                "B 多尺度运动线提案"
            } else {
                "A 基础线提案"
            };
            runtimes.insert(
                format!("{scene}:{}", if multiscale { "B" } else { "A" }),
                json!(seconds),
            );
            for audit in &frame_audits {
                let mut row = audit.clone();
                let object = row.as_object_mut().ok_or("invalid motion audit")?;
                object.retain(|key, _| {
                    [
                        "frame_index",
                        "feature_count",
                        "component_count",
                        "support_pixel_count",
                        "threshold_adu",
                        "max_feature_residual_snr",
                    ]
                    .contains(&key.as_str())
                });
                row["scene"] = json!(scene);
                row["version"] = json!(version);
                audits.push(row);
            }
            for (i, track) in tracks.iter().enumerate() {
                let snrs = track["points"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|p| p["residual_snr"].as_f64())
                    .collect::<Vec<_>>();
                track_rows.push(json!({"scene":scene,"version":version,"track_index":i+1,"classification":track["classification"],"presence":track["presence"],"point_count":snrs.len(),"evidence_level":track["evidence_level"],"median_residual_snr":statistics::median(&snrs).ok()}));
            }
            let mut row = json!({"scene":scene,"version":version,"frame_count":N,"input_array_sha256":array_hash,"proposal_feature_observations":frame_audits.iter().filter_map(|a|a["feature_count"].as_u64()).sum::<u64>(),"moving_track_count":tracks.iter().filter(|t|t["classification"]=="moving").count(),"target_recovered_frames":"","target_recovery":"","median_position_error_px":"","p90_position_error_px":"","runtime_seconds":seconds});
            if scene == "low_peak_moving_trail" {
                let errors = match_errors(&tracks);
                row["target_recovered_frames"] = json!(format!("{}/{N}", errors.len()));
                row["target_recovery"] = json!(errors.len() == N && covers(&tracks) == 1);
                row["median_position_error_px"] = json!(statistics::median(&errors).ok());
                row["p90_position_error_px"] = json!(statistics::percentile(&errors, 90.).ok());
                positive[usize::from(multiscale)] = tracks;
            }
            summary.push(row);
        }
    }
    let truth=(0..N).map(|i|json!({"target_id":"LOW-PEAK-01","frame_index":i,"x_px":22.+10.*i as f64,"y_px":40.,"visible":true})).collect::<Vec<_>>();
    for (name, rows) in [
        ("summary.csv", &summary),
        ("frame_audit.csv", &audits),
        ("track_results.csv", &track_rows),
        ("truth_trajectory.csv", &truth),
    ] {
        innovation_artifacts::write_csv(&out.join(name), rows, &[])?;
    }
    case_sheet(&out.join("motion_proposal_cases.png"), &injected, &positive)?;
    let mut manifest = json!({"schema_version":1,"experiment":"synthetic_pixel_level_motion_multiscale_proposal_ablation","status":"completed","seed":seed.to_string(),"input":{"path":input,"sha256":hash,"dtype":"float32","shape_per_scene":shape,"control_array_sha256":frozen_arrays::array_sha256(&control,&shape),"injected_array_sha256":frozen_arrays::array_sha256(&injected,&shape),"paired_noise":true,"frames":N},"truth":{"target_id":"LOW-PEAK-01","type":"synthetic finite-exposure horizontal Gaussian-PSF trail","positions_px":truth.iter().map(|v|json!({"frame_index":v["frame_index"],"x_px":v["x_px"],"y_px":v["y_px"]})).collect::<Vec<_>>(),"line_length_px":12.,"line_kernel_coefficient_adu":4.,"signal_definition":"each 0.5 px line sample adds 0.5 times the kernel coefficient times a Gaussian PSF; the coefficient is not the sampled peak, which is recorded per frame below","sampled_peak_pixel_increment_adu_by_frame":(0..N).map(|i|injected[i*H*W..(i+1)*H*W].iter().zip(&control[i*H*W..(i+1)*H*W]).map(|(a,b)|*a-*b).max_by(f32::total_cmp)).collect::<Vec<_>>(),"frame_to_frame_step_px":10.},"parameters":{"background_adu":10.,"noise_sigma_adu":1.,"psf_fwhm_px":3.,"match_radius_px":7.,"same_detector_options":{"psf_fwhm":3.,"residual_sigma":15.,"min_residual_adu":100.,"min_feature_area":40,"min_axis_ratio":4.,"min_length_px":8.,"bridge_gaps":true,"raw_fallback_workers":1},"ablation_variable":"multiscale_line_proposals=False (A) vs True (B)","temporal_reference":"constant shared background image, same for every frame and version","association_and_shape_gates":"native production motion-feature components, raw model review, association, recovery and classification"},"metrics":{"proposal_feature_observations":"sum of per-frame final MotionFrameAudit.feature_count; not a full-frame false-positive rate","target_recovery":"one moving track has a proposal within the fixed 7 px truth tolerance in all 4 frames","position_error":"distance to the injected trail center among matched moving points","negative_control":"paired noise frames with no injected trail; count observed outputs but do not infer population precision"},"runtime":{"seconds_by_scene_and_version":runtimes},"outputs":{"summary_csv":out.join("summary.csv"),"frame_audit_csv":out.join("frame_audit.csv"),"track_results_csv":out.join("track_results.csv"),"truth_trajectory_csv":out.join("truth_trajectory.csv"),"case_sheet":out.join("motion_proposal_cases.png")},"limitations":["This is a small synthetic pixel-level mechanism test, not a real FITS motion-source recovery test.","No crossing or crowded-trajectory identity test is included.","The negative control is four paired noise frames and is not an estimate of full-frame false-positive rate."]});
    manifest["execution"] = json!({"implementation":"Rust","platform":std::env::consts::OS,"architecture":std::env::consts::ARCH,"core_version":env!("CARGO_PKG_VERSION")});
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(manifest)
}
fn escaped(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut point = None;
    let mut association = None;
    let mut out = None;
    let mut report = PathBuf::from("deliverables/ablation/index.html");
    let mut seed = 19019u128;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let (key, value) = if let Some((k, v)) = arg.split_once('=') {
            (k.to_owned(), v.to_owned())
        } else {
            i += 1;
            (
                arg.clone(),
                args.get(i).ok_or("argument requires a value")?.clone(),
            )
        };
        match key.as_str() {
            "--point-run-dir" => point = Some(PathBuf::from(value)),
            "--association-run-dir" => association = Some(PathBuf::from(value)),
            "--out-dir" => out = Some(PathBuf::from(value)),
            "--report" => report = PathBuf::from(value),
            "--seed" => {
                seed = value
                    .parse()
                    .map_err(|_| "seed must be a non-negative integer")?
            }
            _ => return Err(format!("unknown argument {key}")),
        }
        i += 1;
    }
    let point = point.ok_or("--point-run-dir is required")?;
    if !point.join("manifest.json").is_file() {
        return Err("point run manifest is missing".into());
    }
    if association
        .as_ref()
        .is_some_and(|dir| !dir.join("manifest.json").is_file())
    {
        return Err("association run manifest is missing".into());
    }
    let out = out.unwrap_or_else(|| {
        PathBuf::from(format!(
            "tmp/ablation/motion-proposal-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ))
    });
    let manifest = run(&out, seed)?;
    crate::point_source_ablation::write_report(&point, &report)?;
    let parent = report.parent().unwrap_or(Path::new("."));
    let evidence = parent.join("evidence/motion-proposal");
    fs::create_dir_all(&evidence).map_err(|e| e.to_string())?;
    for file in [
        "manifest.json",
        "summary.csv",
        "frame_audit.csv",
        "track_results.csv",
        "truth_trajectory.csv",
        "motion_proposal_cases.png",
        "frozen_motion_frames.npz",
    ] {
        fs::copy(out.join(file), evidence.join(file)).map_err(|e| e.to_string())?;
    }
    if let Some(dir) = association {
        let dest = parent.join("evidence/motion");
        fs::create_dir_all(&dest).map_err(|e| e.to_string())?;
        for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            if entry.path().is_file() {
                fs::copy(entry.path(), dest.join(entry.file_name())).map_err(|e| e.to_string())?;
            }
        }
    }
    let start = "<!-- RST19-MOTION-PROPOSAL-START -->";
    let end = "<!-- RST19-MOTION-PROPOSAL-END -->";
    let section=format!("{start}<section id=\"motion-proposal\"><h2>冻结像素运动线提案消融</h2><p>基础提案与多尺度提案使用同一四帧冻结噪声和有限曝光注入轨迹。注入回收结果不代表物理恒星身份。</p><p>运行目录：{}</p><ul><li><a href=\"evidence/motion-proposal/summary.csv\">回收与误差汇总</a></li><li><a href=\"evidence/motion-proposal/frame_audit.csv\">逐帧审计</a></li><li><a href=\"evidence/motion-proposal/manifest.json\">输入哈希与参数</a></li><li><a href=\"evidence/motion-proposal/frozen_motion_frames.npz\">冻结像素</a></li></ul><img src=\"evidence/motion-proposal/motion_proposal_cases.png\" alt=\"真值及两个版本的运动提案\"></section>{end}",escaped(&out.to_string_lossy()));
    let mut html = fs::read_to_string(&report).map_err(|e| e.to_string())?;
    if let Some(a) = html.find(start) {
        if let Some(b) = html[a..].find(end) {
            html.replace_range(a..a + b + end.len(), &section);
        } else {
            html.push_str(&section);
        }
    } else {
        let at = html.find("</body>").unwrap_or(html.len());
        html.insert_str(at, &section);
    }
    fs::write(&report, html).map_err(|e| e.to_string())?;
    crate::ablation_report::decorate(&report)?;
    Ok(
        json!({"status":manifest["status"],"run_dir":out.canonicalize().map_err(|e|e.to_string())?,"manifest":out.join("manifest.json"),"report":report,"runtime":manifest["runtime"]["seconds_by_scene_and_version"]}),
    )
}
