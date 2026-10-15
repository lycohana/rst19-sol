//! Native injection/recovery experiments preserve candidate and quality layers.
use crate::measurement::Detection;
use crate::{
    detection::{self, DetectionOptions, DetectionResult},
    empirical_psf::{self, EmpiricalPsf},
    feature_audit_replicates, fits, frozen_arrays,
    injection_sites::{self, Site},
    innovation_artifacts,
    models::{FitsFrame, PixelData},
    numpy_ziggurat,
    source_geometry_audit::NumpyPcg64,
    spatial_injection, statistics, Result,
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub enum Mode {
    Synthetic,
    Feature,
    Real,
    Stratified,
    Comparison,
    PairRatio,
    Crowded,
}
#[derive(Clone, Debug)]
pub struct Options {
    pub mode: Mode,
    pub real_fits: Option<PathBuf>,
    pub out_dir: PathBuf,
    pub trials: usize,
    pub sources: usize,
    pub peaks: Vec<f64>,
    pub detector: DetectionOptions,
    pub injected_psf: Option<f64>,
    pub pair_separation: Option<f64>,
    pub psf_model: String,
    pub normalization: String,
    pub ratios: Vec<f64>,
    pub pairs: usize,
    pub blend_sizes: Vec<usize>,
    pub blend_separations: Vec<f64>,
    pub blend_levels: Vec<f64>,
    pub groups: usize,
    pub strata: Vec<String>,
    pub seed: u128,
}
impl Default for Options {
    fn default() -> Self {
        let mut detector = DetectionOptions::default();
        detector.min_distance = 4;
        detector.proposal_mode = "hybrid".into();
        detector.measurement.psf_fwhm = 3.;
        Self {
            mode: Mode::Synthetic,
            real_fits: None,
            out_dir: PathBuf::from("tmp/injection"),
            trials: 4,
            sources: 24,
            peaks: vec![],
            detector,
            injected_psf: None,
            pair_separation: None,
            psf_model: "gaussian".into(),
            normalization: "peak_excess".into(),
            ratios: vec![1., 0.5, 0.25, 0.125],
            pairs: 4,
            blend_sizes: vec![2, 3],
            blend_separations: vec![3., 5.4, 6.],
            blend_levels: vec![256., 512.],
            groups: 4,
            strata: [
                "blank",
                "high_background",
                "edge",
                "crowded",
                "special_code",
            ]
            .iter()
            .map(|v| v.to_string())
            .collect(),
            seed: 19019,
        }
    }
}
pub const HELP:&str="原生 Rust 注入/回收：合成、特征控制、真实背景、分层、提案比较、双源强弱比和拥挤组\nUsage: rst19-injection [--real-fits FITS] [--feature-audit|--stratified-real|--compare-proposals|--pair-flux-ratio-audit|--crowded-blend-audit] [--out-dir DIR] [--trials N] [--sources N] [--peak-levels ADU ...]";
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}
pub fn match_flags(targets: &[[f64; 2]], sources: &[Detection], radius: f64) -> Vec<bool> {
    let mut remaining = sources.iter().collect::<Vec<_>>();
    targets
        .iter()
        .map(|p| {
            let mut best = None;
            let mut distance = radius;
            for (i, s) in remaining.iter().enumerate() {
                let d = (s.x - p[0]).hypot(s.y - p[1]);
                if d <= distance {
                    best = Some(i);
                    distance = d;
                }
            }
            if let Some(i) = best {
                remaining.remove(i);
                true
            } else {
                false
            }
        })
        .collect()
}
fn hits(targets: &[[f64; 2]], sources: &[Detection], radius: f64) -> usize {
    match_flags(targets, sources, radius)
        .iter()
        .filter(|v| **v)
        .count()
}
fn quality(d: &DetectionResult) -> Vec<Detection> {
    d.sources
        .iter()
        .filter(|s| s.quality_passed)
        .cloned()
        .collect()
}
fn outside(sources: &[Detection], positions: &[[f64; 2]], radius: f64) -> usize {
    sources
        .iter()
        .filter(|s| {
            positions
                .iter()
                .all(|p| (s.x - p[0]).hypot(s.y - p[1]) >= radius)
        })
        .count()
}
fn detect(
    frame: &FitsFrame,
    pixels: &PixelData,
    options: &DetectionOptions,
) -> Result<DetectionResult> {
    let mask = fits::auxiliary_mask(frame.height, frame.width)?;
    detection::detect_sources(
        pixels,
        Some(&mask),
        frame.height,
        frame.width,
        options,
        None,
    )
}
fn injected_options(original: &DetectionOptions, baseline: &DetectionResult) -> DetectionOptions {
    let mut d = original.clone();
    d.measurement.saturation_level = baseline.parameters["saturation_level"]
        .as_f64()
        .filter(|v| *v > 0.);
    d.measurement.negative_overflow_limit = baseline.parameters["negative_overflow_limit"].as_f64();
    d.mask_zero_pixels = Some(
        baseline.parameters["mask_zero_pixels"]
            .as_u64()
            .unwrap_or(0)
            != 0
            || baseline.parameters["mask_zero_pixels"]
                .as_bool()
                .unwrap_or(false),
    );
    d
}
fn psf(frame: &FitsFrame, baseline: &DetectionResult, model: &str) -> Result<Option<EmpiricalPsf>> {
    if model == "gaussian" {
        return Ok(None);
    }
    let p = empirical_psf::estimate(
        frame.data.to_f64().as_slice().into(),
        frame.height,
        frame.width,
        &baseline.sources,
        None,
        7,
        64,
        &[],
    )?;
    if p.is_none() {
        return Err("真实质量源不足，无法建立实测 PSF".into());
    }
    Ok(p)
}
fn psf_fields(row: &mut Value, p: Option<&EmpiricalPsf>, o: &Options) {
    row["psf_model"] = json!(o.psf_model);
    row["psf_source_count"] = json!(p.map_or(0, |p| p.source_count));
    row["psf_median_fwhm_px"] = json!(p.and_then(|p| p.median_fwhm_px));
    row["psf_kernel_sum"] = json!(p.map(|p| p.kernel.iter().map(|v| *v as f64).sum::<f64>()));
}
fn inject(
    image: &mut [f32],
    frame: &FitsFrame,
    position: [f64; 2],
    signal: f64,
    o: &Options,
    p: Option<&EmpiricalPsf>,
) -> Result<f64> {
    let options = spatial_injection::Options {
        psf_fwhm: o.detector.measurement.psf_fwhm,
        injected_psf_fwhm: o.injected_psf,
        psf_model: o.psf_model.clone(),
        signal_normalization: o.normalization.clone(),
        ..Default::default()
    };
    spatial_injection::inject_signal(
        image,
        frame.height,
        frame.width,
        position[0],
        position[1],
        signal,
        &options,
        p,
    )
}
fn freeze(
    o: &Options,
    name: &str,
    image: &[f32],
    frame: &FitsFrame,
    inputs: &mut Vec<Value>,
) -> Result<()> {
    let folder = o.out_dir.join("frozen_inputs");
    fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
    let path = folder.join(format!("{name}.npz"));
    let shape = [frame.height, frame.width];
    let sha = frozen_arrays::write_npz_f32(&path, &[("image", image, &shape)])?;
    inputs.push(json!({"path":path,"sha256":sha,"array_sha256":frozen_arrays::array_sha256(image,&shape),"shape":shape}));
    Ok(())
}
fn random_positions(
    rng: &mut NumpyPcg64,
    count: usize,
    height: usize,
    width: usize,
) -> Result<Vec<[f64; 2]>> {
    if height <= 28 || width <= 28 {
        return Err("synthetic image too small".into());
    }
    let mut result = Vec::<[f64; 2]>::new();
    for _ in 0..1000usize.max(count.saturating_mul(100)) {
        if result.len() == count {
            break;
        }
        let p = [
            rng.uniform(14., (width - 14) as f64),
            rng.uniform(14., (height - 14) as f64),
        ];
        if result
            .iter()
            .all(|v| (p[0] - v[0]).hypot(p[1] - v[1]) >= 12.)
        {
            result.push(p);
        }
    }
    if result.len() != count {
        return Err("无法生成足够非拥挤注入位置".into());
    }
    Ok(result)
}
pub fn synthetic(o: &Options, inputs: &mut Vec<Value>) -> Result<Vec<Value>> {
    let total_trials = o.peaks.len().saturating_mul(o.trials).max(1);
    let report_stride = total_trials.div_ceil(30).max(1);
    let mut completed_trials = 0usize;
    eprintln!("12.0% · 生成固定随机种子的合成背景与注入布局");
    let mut rng = NumpyPcg64::new(o.seed);
    let mut rows = Vec::new();
    let mut detector = o.detector.clone();
    detector.background_box_size = 64;
    let frame = FitsFrame {
        path: PathBuf::from("synthetic-injection.fits"),
        header: Default::default(),
        data: PixelData::F32(vec![]),
        auxiliary: None,
        data_offset: 0,
        width: 256,
        height: 256,
    };
    for (level, peak) in o.peaks.iter().enumerate() {
        let (mut c, mut q) = (0, 0);
        let mut counts = Vec::new();
        let mut quality_counts = Vec::new();
        for trial in 0..o.trials {
            let mut image = (0..256 * 256)
                .map(|_| (21. + 5. * numpy_ziggurat::standard_normal(&mut rng)) as f32)
                .collect::<Vec<_>>();
            let positions = random_positions(&mut rng, o.sources, 256, 256)?;
            let mut signal_options = o.clone();
            signal_options.psf_model = "gaussian".into();
            signal_options.normalization = "peak_excess".into();
            for p in &positions {
                inject(&mut image, &frame, *p, *peak, &signal_options, None)?;
            }
            freeze(
                o,
                &format!("synthetic-level-{level:03}-trial-{trial:03}"),
                &image,
                &frame,
                inputs,
            )?;
            if let Some(input) = inputs.last_mut() {
                input["truth_positions_px"] = json!(positions);
            }
            let d = detect(&frame, &PixelData::F32(image), &detector)?;
            let good = quality(&d);
            let radius = 3f64.max(detector.measurement.psf_fwhm);
            c += hits(&positions, &d.sources, radius);
            q += hits(&positions, &good, radius);
            counts.push(d.sources.len() as f64);
            quality_counts.push(good.len() as f64);
            completed_trials += 1;
            if completed_trials % report_stride == 0 || completed_trials == total_trials {
                let percent = 12.0 + 68.0 * completed_trials as f32 / total_trials as f32;
                eprintln!("{percent:.1}% · 注入回收试验 {completed_trials}/{total_trials}，峰值档位 {}/{}, 候选/质量筛选已完成", level + 1, o.peaks.len());
            }
        }
        let total = o.trials * o.sources;
        rows.push(json!({"proposal_mode":detector.proposal_mode,"detector_psf_fwhm_px":detector.measurement.psf_fwhm,"injected_psf_fwhm_px":o.injected_psf.unwrap_or(detector.measurement.psf_fwhm),"peak_excess_adu":peak,"injected_count":total,"candidate_recovered_count":c,"quality_recovered_count":q,"candidate_recall":c as f64/total as f64,"quality_recall":q as f64/total as f64,"mean_candidate_count":mean(&counts),"mean_quality_count":mean(&quality_counts)}));
    }
    Ok(rows)
}
fn real_context(
    o: &Options,
) -> Result<(
    FitsFrame,
    DetectionResult,
    DetectionOptions,
    Option<EmpiricalPsf>,
)> {
    eprintln!("8.0% · 读取真实背景 FITS 并建立注入前检测基线");
    let frame = fits::read_fits(
        o.real_fits
            .as_ref()
            .ok_or("this mode requires --real-fits")?,
        false,
    )?;
    let baseline = detect(&frame, &frame.data, &o.detector)?;
    eprintln!("12.0% · 基线候选与质量源已计算，估计局部 PSF");
    let options = injected_options(&o.detector, &baseline);
    let empirical = psf(&frame, &baseline, &o.psf_model)?;
    Ok((frame, baseline, options, empirical))
}
pub fn real(o: &Options, stratified: bool, inputs: &mut Vec<Value>) -> Result<Vec<Value>> {
    let (frame, baseline, detector, empirical) = real_context(o)?;
    let mut rng = NumpyPcg64::new(o.seed);
    let special = spatial_injection::special_mask(&frame.data);
    let mut layouts = Vec::<(String, Vec<Vec<Site>>)>::new();
    let psf = o.detector.measurement.psf_fwhm;
    for stratum in if stratified {
        o.strata.clone()
    } else {
        vec!["real".into()]
    } {
        let mut trials = Vec::new();
        for _ in 0..o.trials {
            trials.push(if stratified {
                injection_sites::select_stratum(
                    &frame,
                    &baseline,
                    &special,
                    &mut rng,
                    o.sources,
                    &stratum,
                    psf,
                    o.detector.measurement.aperture_radius,
                )?
            } else {
                injection_sites::select_blank(
                    &frame,
                    &baseline,
                    &mut rng,
                    o.sources,
                    14,
                    12f64.max(2. * psf),
                )?
            });
        }
        layouts.push((stratum, trials));
    }
    let total_trials = layouts
        .iter()
        .map(|(_, trials)| trials.len())
        .sum::<usize>()
        .saturating_mul(o.peaks.len())
        .max(1);
    let report_stride = total_trials.div_ceil(30).max(1);
    let mut completed_trials = 0usize;
    eprintln!("18.0% · 注入位置已按背景条件选定，开始真实图像回收试验");
    let mut rows = Vec::new();
    let radius = 3f64.max(psf);
    let protection = 8f64.max(2. * psf);
    for (stratum, trials) in layouts {
        let conditions = trials
            .iter()
            .enumerate()
            .flat_map(|(t, sites)| {
                sites.iter().enumerate().map(move |(i, s)| {
                    let mut v = serde_json::to_value(s).expect("finite site");
                    v["trial_index"] = json!(t + 1);
                    v["site_index"] = json!(i + 1);
                    v
                })
            })
            .collect::<Vec<_>>();
        for (level, peak) in o.peaks.iter().enumerate() {
            let (mut c, mut q, mut ambiguous, mut unambiguous) = (0, 0, 0, 0);
            let (mut cc, mut qc, mut bc, mut bq, mut bg, mut noise, mut fraction, mut nearest) = (
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            );
            for (t, sites) in trials.iter().enumerate() {
                let mut image = (0..frame.data.len())
                    .map(|i| frame.data.value(i) as f32)
                    .collect::<Vec<_>>();
                let positions = sites.iter().map(|s| [s.x, s.y]).collect::<Vec<_>>();
                let strict = sites
                    .iter()
                    .filter(|s| {
                        !stratified || s.nearest_baseline_source_px.is_none_or(|d| d > radius)
                    })
                    .map(|s| [s.x, s.y])
                    .collect::<Vec<_>>();
                let mut signal_options = o.clone();
                signal_options.injected_psf = None;
                signal_options.normalization = "peak_excess".into();
                for p in &positions {
                    inject(
                        &mut image,
                        &frame,
                        *p,
                        *peak,
                        &signal_options,
                        empirical.as_ref(),
                    )?;
                }
                freeze(
                    o,
                    &format!("{stratum}-level-{level:03}-trial-{t:03}"),
                    &image,
                    &frame,
                    inputs,
                )?;
                let d = detect(&frame, &PixelData::F32(image), &detector)?;
                let good = quality(&d);
                c += hits(&strict, &d.sources, radius);
                q += hits(&strict, &good, radius);
                unambiguous += strict.len();
                ambiguous += sites
                    .iter()
                    .filter(|s| s.baseline_neighbor_count > 0)
                    .count();
                cc.push(d.candidate_count as f64);
                qc.push(good.len() as f64);
                bc.push(outside(&d.sources, &positions, protection) as f64);
                bq.push(outside(&good, &positions, protection) as f64);
                for site in sites {
                    bg.push(site.local_background_adu);
                    noise.push(site.local_noise_adu);
                    fraction.push(site.special_pixel_fraction);
                    if let Some(d) = site.nearest_baseline_source_px {
                        nearest.push(d);
                    }
                }
                completed_trials += 1;
                if completed_trials % report_stride == 0 || completed_trials == total_trials {
                    let percent = 18.0 + 62.0 * completed_trials as f32 / total_trials as f32;
                    eprintln!("{percent:.1}% · 真实背景注入试验 {completed_trials}/{total_trials}，当前分层 {stratum}、峰值档位 {}/{}", level + 1, o.peaks.len());
                }
            }
            let total = o.trials * o.sources;
            let mut row = json!({"source_path":frame.path,"proposal_mode":detector.proposal_mode,"peak_excess_adu":peak,"injected_count":total,"candidate_recovered_count":c,"quality_recovered_count":q,"candidate_recall":c as f64/total as f64,"quality_recall":q as f64/total as f64,"baseline_candidate_count":baseline.candidate_count,"baseline_quality_count":baseline.quality_count,"mean_candidate_count":mean(&cc),"mean_quality_count":mean(&qc),"mean_background_candidate_count":mean(&bc),"mean_background_quality_count":mean(&bq),"net_candidate_delta":mean(&cc)-baseline.candidate_count as f64,"net_quality_delta":mean(&qc)-baseline.quality_count as f64,"local_background_adu":mean(&bg),"local_noise_adu":mean(&noise),"trial_count":o.trials});
            psf_fields(&mut row, empirical.as_ref(), o);
            if stratified {
                let label = match stratum.as_str() {
                    "blank" => "候选稀疏/低纹理背景",
                    "high_background" => "候选稀疏/高局部噪声背景",
                    "edge" => "图像边缘截断",
                    "crowded" => "亮源邻域/拥挤混合",
                    "special_code" => "特殊值域邻域",
                    _ => "线状伪影邻域",
                };
                row["stratum"] = json!(stratum);
                row["stratum_label"] = json!(label);
                row["ambiguous_injection_count"] = json!(ambiguous);
                row["unambiguous_injected_count"] = json!(unambiguous);
                row["unambiguous_candidate_recall"] =
                    json!((unambiguous > 0).then(|| c as f64 / unambiguous as f64));
                row["unambiguous_quality_recall"] =
                    json!((unambiguous > 0).then(|| q as f64 / unambiguous as f64));
                row["special_pixel_fraction"] = json!(mean(&fraction));
                row["nearest_baseline_source_px"] = json!(statistics::median(&nearest).ok());
                row["site_condition_json"] =
                    json!(serde_json::to_string(&conditions).map_err(|e| e.to_string())?);
                row["note"]=json!("该层按注入前图像条件选点；候选/质量回收是已知注入真值的召回率，不是当前真实 FITS 的 precision 或逐星完备率。");
            }
            rows.push(row);
        }
    }
    Ok(rows)
}
pub fn comparison(o: &Options, inputs: &mut Vec<Value>) -> Result<Vec<Value>> {
    if o.pair_separation.is_some() && o.sources % 2 != 0 {
        return Err("sources must be even for paired injections".into());
    }
    let frame = fits::read_fits(
        o.real_fits
            .as_ref()
            .ok_or("comparison requires --real-fits")?,
        false,
    )?;
    let mut options = o.detector.clone();
    options.proposal_mode = "ensemble".into();
    let baseline = detect(&frame, &frame.data, &options)?;
    let mut rng = NumpyPcg64::new(o.seed);
    let psf = options.measurement.psf_fwhm;
    let sep = o.pair_separation.unwrap_or(0.);
    let mut layouts = Vec::new();
    for _ in 0..o.trials {
        let sites = injection_sites::select_blank(
            &frame,
            &baseline,
            &mut rng,
            if o.pair_separation.is_some() {
                o.sources / 2
            } else {
                o.sources
            },
            14usize.max((sep / 2.).ceil() as usize + 10),
            12f64.max(2. * psf + sep),
        )?;
        let mut positions = Vec::new();
        for s in sites {
            if o.pair_separation.is_some() {
                let angle = rng.uniform(0., std::f64::consts::TAU);
                let dx = 0.5 * sep * angle.cos();
                let dy = 0.5 * sep * angle.sin();
                positions.extend([[s.x - dx, s.y - dy], [s.x + dx, s.y + dy]]);
            } else {
                positions.push([s.x, s.y]);
            }
        }
        layouts.push(positions);
    }
    let mut rows = Vec::new();
    let total_trials = layouts.len().saturating_mul(o.peaks.len()).max(1);
    let report_stride = total_trials.div_ceil(30).max(1);
    let mut completed_trials = 0usize;
    eprintln!("18.0% · 注入方案对照开始，逐试验比较 Gaussian、Hybrid 与 Ensemble");
    for (level, peak) in o.peaks.iter().enumerate() {
        let mut row = json!({"source_path":frame.path,"detector_psf_fwhm_px":psf,"injected_psf_fwhm_px":o.injected_psf.unwrap_or(psf),"pair_separation_px":o.pair_separation,"peak_excess_adu":peak,"injected_count":o.trials*o.sources,"trial_count":o.trials});
        let mut sums = [[0f64; 4]; 3];
        for (t, positions) in layouts.iter().enumerate() {
            let mut image = (0..frame.data.len())
                .map(|i| frame.data.value(i) as f32)
                .collect::<Vec<_>>();
            let mut injection_options = o.clone();
            injection_options.psf_model = "gaussian".into();
            injection_options.normalization = "peak_excess".into();
            for p in positions {
                inject(&mut image, &frame, *p, *peak, &injection_options, None)?;
            }
            freeze(
                o,
                &format!("comparison-level-{level:03}-trial-{t:03}"),
                &image,
                &frame,
                inputs,
            )?;
            let pixels = PixelData::F32(image);
            for (i, mode) in ["gaussian", "hybrid", "ensemble"].iter().enumerate() {
                options.proposal_mode = mode.to_string();
                let d = detect(&frame, &pixels, &options)?;
                let good = quality(&d);
                sums[i][0] += hits(positions, &d.sources, 3f64.max(psf)) as f64;
                sums[i][1] += hits(positions, &good, 3f64.max(psf)) as f64;
                sums[i][2] += d.candidate_count as f64;
                sums[i][3] += good.len() as f64;
            }
            completed_trials += 1;
            if completed_trials % report_stride == 0 || completed_trials == total_trials {
                let percent = 18.0 + 62.0 * completed_trials as f32 / total_trials as f32;
                eprintln!("{percent:.1}% · 方案对照试验 {completed_trials}/{total_trials}，峰值档位 {}/{}, 三种检测器已复算", level + 1, o.peaks.len());
            }
        }
        for (i, mode) in ["gaussian", "hybrid", "ensemble"].iter().enumerate() {
            row[format!("{mode}_candidate_recall")] =
                json!(sums[i][0] / (o.trials * o.sources) as f64);
            row[format!("{mode}_quality_recall")] =
                json!(sums[i][1] / (o.trials * o.sources) as f64);
            row[format!("{mode}_mean_candidate_count")] = json!(sums[i][2] / o.trials as f64);
            row[format!("{mode}_mean_quality_count")] = json!(sums[i][3] / o.trials as f64);
        }
        rows.push(row);
    }
    Ok(rows)
}
fn merged(group: &[[f64; 2]], sources: &[Detection], radius: f64) -> bool {
    sources.iter().any(|s| {
        group
            .iter()
            .filter(|p| (s.x - p[0]).hypot(s.y - p[1]) <= radius)
            .count()
            >= 2
    })
}
fn region_count(group: &[[f64; 2]], sources: &[Detection], radius: f64) -> usize {
    let center = [
        group.iter().map(|p| p[0]).sum::<f64>() / group.len() as f64,
        group.iter().map(|p| p[1]).sum::<f64>() / group.len() as f64,
    ];
    let r = group
        .iter()
        .map(|p| (p[0] - center[0]).hypot(p[1] - center[1]) + radius)
        .fold(radius, f64::max);
    sources
        .iter()
        .filter(|s| (s.x - center[0]).hypot(s.y - center[1]) <= r)
        .count()
}
pub fn pairs_or_crowded(o: &Options, crowded: bool, inputs: &mut Vec<Value>) -> Result<Vec<Value>> {
    let (frame, baseline, detector, empirical) = real_context(o)?;
    let mut rng = NumpyPcg64::new(o.seed);
    let psf = detector.measurement.psf_fwhm;
    let radius = 3f64.max(psf);
    let mut layouts = Vec::new();
    for size in if crowded {
        o.blend_sizes.clone()
    } else {
        vec![2]
    } {
        for sep in if crowded {
            o.blend_separations.clone()
        } else {
            vec![o.pair_separation.unwrap_or(5.4)]
        } {
            let extent = (size - 1) as f64 * sep;
            let mut trials = Vec::new();
            for _ in 0..o.trials {
                let sites = injection_sites::select_blank(
                    &frame,
                    &baseline,
                    &mut rng,
                    if crowded { o.groups } else { o.pairs },
                    14usize.max((extent / 2.).ceil() as usize + 10),
                    16f64.max(2. * psf + extent + if crowded { 4. } else { 0. }),
                )?;
                let mut groups = Vec::new();
                for s in &sites {
                    let angle = rng.uniform(0., std::f64::consts::TAU);
                    groups.push(
                        (0..size)
                            .map(|i| {
                                let offset = (i as f64 - 0.5 * (size - 1) as f64) * sep;
                                [s.x + offset * angle.cos(), s.y + offset * angle.sin()]
                            })
                            .collect::<Vec<_>>(),
                    );
                }
                trials.push((groups, sites));
            }
            layouts.push((size, sep, trials));
        }
    }
    let levels = if crowded { &o.blend_levels } else { &o.peaks };
    let ratios = if crowded { vec![1.] } else { o.ratios.clone() };
    let total_trials = layouts
        .iter()
        .map(|(_, _, trials)| trials.len())
        .sum::<usize>()
        .saturating_mul(levels.len())
        .saturating_mul(ratios.len())
        .max(1);
    let report_stride = total_trials.div_ceil(30).max(1);
    let mut completed_trials = 0usize;
    eprintln!("18.0% · 注入布局完成，开始拥挤混合与双星分辨回收");
    let mut rows = Vec::new();
    for (size, sep, trials) in layouts {
        for (level, total_signal) in levels.iter().enumerate() {
            for (ratio_index, ratio) in ratios.iter().enumerate() {
                let mut source_hits = [0usize; 2];
                let mut resolved = [0usize; 2];
                let mut merge = [0usize; 2];
                let mut extra = [0usize; 2];
                let mut primary_hits = [0usize; 2];
                let mut secondary_hits = [0usize; 2];
                let mut region = [vec![], vec![]];
                let mut counts = [vec![], vec![]];
                let mut primary_peaks = Vec::new();
                let mut secondary_peaks = Vec::new();
                let mut background = Vec::new();
                let mut noise = Vec::new();
                let mut total_groups = 0;
                let first_signal = if crowded {
                    *total_signal / size as f64
                } else {
                    *total_signal / (1. + ratio)
                };
                let second_signal = if crowded {
                    first_signal
                } else {
                    *total_signal - first_signal
                };
                for (t, (groups, sites)) in trials.iter().enumerate() {
                    let mut image = (0..frame.data.len())
                        .map(|i| frame.data.value(i) as f32)
                        .collect::<Vec<_>>();
                    for group in groups {
                        for (i, p) in group.iter().enumerate() {
                            let peak = inject(
                                &mut image,
                                &frame,
                                *p,
                                if i == 0 { first_signal } else { second_signal },
                                o,
                                empirical.as_ref(),
                            )?;
                            if i == 0 {
                                primary_peaks.push(peak);
                            } else {
                                secondary_peaks.push(peak);
                            }
                        }
                    }
                    freeze(o,&format!("group-{size}-sep-{:016x}-level-{level:03}-ratio-{ratio_index:03}-trial-{t:03}",sep.to_bits()),&image,&frame,inputs)?;
                    let d = detect(&frame, &PixelData::F32(image), &detector)?;
                    let good = quality(&d);
                    let flat = groups.iter().flatten().copied().collect::<Vec<_>>();
                    for (j, sources) in [&d.sources, &good].iter().enumerate() {
                        let global = if crowded {
                            vec![]
                        } else {
                            match_flags(&flat, sources, radius)
                        };
                        for (g, group) in groups.iter().enumerate() {
                            let flags = if crowded {
                                match_flags(group, sources, radius)
                            } else {
                                global[g * size..(g + 1) * size].to_vec()
                            };
                            source_hits[j] += flags.iter().filter(|v| **v).count();
                            resolved[j] += usize::from(flags.iter().all(|v| *v));
                            primary_hits[j] += usize::from(flags[0]);
                            secondary_hits[j] += usize::from(flags[1]);
                            merge[j] += usize::from(merged(group, sources, radius));
                            let count = region_count(group, sources, radius);
                            region[j].push(count as f64);
                            extra[j] += usize::from(count > size);
                        }
                        counts[j].push(if j == 0 {
                            d.candidate_count as f64
                        } else {
                            good.len() as f64
                        });
                    }
                    total_groups += groups.len();
                    for s in sites {
                        background.push(s.local_background_adu);
                        noise.push(s.local_noise_adu);
                    }
                    completed_trials += 1;
                    if completed_trials % report_stride == 0 || completed_trials == total_trials {
                        let percent = 18.0 + 62.0 * completed_trials as f32 / total_trials as f32;
                        let label = if crowded { "拥挤组" } else { "双星比例" };
                        eprintln!("{percent:.1}% · {label}注入试验 {completed_trials}/{total_trials}，组大小 {size}、间距 {sep:.2}px、信号档位 {}/{}", level + 1, levels.len());
                    }
                }
                let n = total_groups as f64;
                let total = size * total_groups;
                let mut row = json!({"source_path":frame.path,"proposal_mode":detector.proposal_mode,"signal_normalization":o.normalization,"detector_psf_fwhm_px":psf,"injected_psf_fwhm_px":o.injected_psf.unwrap_or(psf),"total_control_signal_adu":total_signal,"injected_count":total,"source_candidate_recall":source_hits[0] as f64/total as f64,"source_quality_recall":source_hits[1] as f64/total as f64,"merged_candidate_fraction":merge[0] as f64/n,"merged_quality_fraction":merge[1] as f64/n,"baseline_candidate_count":baseline.candidate_count,"baseline_quality_count":baseline.quality_count,"mean_candidate_count":mean(&counts[0]),"mean_quality_count":mean(&counts[1]),"local_background_adu":mean(&background),"local_noise_adu":mean(&noise),"trial_count":o.trials});
                psf_fields(&mut row, empirical.as_ref(), o);
                if crowded {
                    row["group_size"] = json!(size);
                    row["nearest_separation_px"] = json!(sep);
                    row["per_source_control_signal_adu"] = json!(first_signal);
                    row["group_count"] = json!(total_groups);
                    row["group_candidate_resolution_recall"] = json!(resolved[0] as f64 / n);
                    row["group_quality_resolution_recall"] = json!(resolved[1] as f64 / n);
                    row["extra_candidate_fraction"] = json!(extra[0] as f64 / n);
                    row["extra_quality_fraction"] = json!(extra[1] as f64 / n);
                    row["mean_candidate_count_in_group"] = json!(mean(&region[0]));
                    row["mean_quality_count_in_group"] = json!(mean(&region[1]));
                    row["note"]=json!(format!("固定同一真实背景和总{}，组内各源等分信号；group resolution 要求每个真值由不同候选一对一命中；merged/extra 仅是注入组邻域诊断，不是 precision。",if o.normalization=="peak_excess"{"峰值超额"}else{"离散积分超额"}));
                } else {
                    row["pair_separation_px"] = json!(sep);
                    row["total_peak_excess_adu"] =
                        json!((o.normalization == "peak_excess").then_some(*total_signal));
                    row["primary_control_signal_adu"] = json!(first_signal);
                    row["secondary_control_signal_adu"] = json!(second_signal);
                    row["secondary_to_primary_ratio"] = json!(ratio);
                    row["primary_peak_excess_adu"] =
                        json!((o.normalization == "peak_excess").then_some(first_signal));
                    row["secondary_peak_excess_adu"] =
                        json!((o.normalization == "peak_excess").then_some(second_signal));
                    row["mean_injected_primary_peak_excess_adu"] = json!(mean(&primary_peaks));
                    row["mean_injected_secondary_peak_excess_adu"] = json!(mean(&secondary_peaks));
                    row["pair_count"] = json!(total_groups);
                    row["primary_candidate_recall"] = json!(primary_hits[0] as f64 / n);
                    row["secondary_candidate_recall"] = json!(secondary_hits[0] as f64 / n);
                    row["primary_quality_recall"] = json!(primary_hits[1] as f64 / n);
                    row["secondary_quality_recall"] = json!(secondary_hits[1] as f64 / n);
                    row["pair_candidate_resolution_recall"] = json!(resolved[0] as f64 / n);
                    row["pair_quality_resolution_recall"] = json!(resolved[1] as f64 / n);
                    row["note"]=json!(format!("固定同一总{}与真实背景，仅改变 secondary/primary；pair resolution 要求一对真值各由不同候选命中，不能把单个中间峰算作两颗。",if o.normalization=="peak_excess"{"峰值超额"}else{"离散积分超额"}));
                }
                rows.push(row);
            }
        }
    }
    Ok(rows)
}
pub fn parse_args(args: &[String]) -> Result<Options> {
    let mut o = Options::default();
    let mut flags = Vec::new();
    let mut feature = false;
    let mut custom_peaks = false;
    let mut total_peaks = None;
    let mut blend_normalization = "integrated_excess".to_owned();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let (key, inline) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(a, b)| (a, Some(b)));
        match key {
            "--feature-audit" => {
                feature = true;
                i += 1;
                continue;
            }
            "--stratified-real" => {
                flags.push(Mode::Stratified);
                i += 1;
                continue;
            }
            "--compare-proposals" => {
                flags.push(Mode::Comparison);
                i += 1;
                continue;
            }
            "--pair-flux-ratio-audit" => {
                flags.push(Mode::PairRatio);
                i += 1;
                continue;
            }
            "--crowded-blend-audit" => {
                flags.push(Mode::Crowded);
                i += 1;
                continue;
            }
            _ => {}
        }
        let mut values = Vec::new();
        if let Some(v) = inline {
            values.push(v.to_owned());
        } else {
            i += 1;
            values.push(
                args.get(i)
                    .ok_or_else(|| format!("{key} requires value"))?
                    .clone(),
            );
        }
        if matches!(
            key,
            "--peak-levels"
                | "--total-peak-levels"
                | "--secondary-ratios"
                | "--blend-sizes"
                | "--blend-separations"
                | "--blend-total-levels"
                | "--strata"
        ) {
            while i + 1 < args.len() && !args[i + 1].starts_with("--") {
                i += 1;
                values.push(args[i].clone());
            }
        }
        let value = &values[0];
        let number = || value.parse::<f64>().map_err(|_| format!("invalid {key}"));
        let integer = || value.parse::<usize>().map_err(|_| format!("invalid {key}"));
        let numbers = || {
            values
                .iter()
                .map(|v| v.parse::<f64>().map_err(|_| format!("invalid {key}")))
                .collect::<Result<Vec<_>>>()
        };
        match key {
            "--out-dir" => o.out_dir = PathBuf::from(value),
            "--real-fits" => o.real_fits = Some(PathBuf::from(value)),
            "--trials" => o.trials = integer()?,
            "--sources" => o.sources = integer()?,
            "--pairs" => o.pairs = integer()?,
            "--blend-groups" => o.groups = integer()?,
            "--threshold-sigma" => o.detector.threshold_sigma = number()?,
            "--min-flux-snr" => o.detector.measurement.min_flux_snr = number()?,
            "--psf-fwhm" => o.detector.measurement.psf_fwhm = number()?,
            "--injected-psf-fwhm" => o.injected_psf = Some(number()?),
            "--pair-separation-px" => o.pair_separation = Some(number()?),
            "--proposal-mode" => o.detector.proposal_mode = value.clone(),
            "--min-distance" => o.detector.min_distance = integer()?,
            "--min-psf-support-pixels" => {
                o.detector.measurement.min_psf_support_pixels = integer()?
            }
            "--psf-model" => o.psf_model = value.clone(),
            "--pair-signal-normalization" => o.normalization = value.clone(),
            "--blend-signal-normalization" => blend_normalization = value.clone(),
            "--peak-levels" => {
                o.peaks = numbers()?;
                custom_peaks = true;
            }
            "--total-peak-levels" => total_peaks = Some(numbers()?),
            "--secondary-ratios" => o.ratios = numbers()?,
            "--blend-separations" => o.blend_separations = numbers()?,
            "--blend-total-levels" => o.blend_levels = numbers()?,
            "--blend-sizes" => {
                o.blend_sizes = values
                    .iter()
                    .map(|v| v.parse().map_err(|_| "invalid blend size".to_owned()))
                    .collect::<Result<_>>()?
            }
            "--strata" => o.strata = values,
            "--seed" => o.seed = value.parse().map_err(|_| "invalid seed")?,
            _ => return Err(format!("unknown injection argument {key}")),
        }
        i += 1;
    }
    if feature {
        o.mode = Mode::Feature;
    } else {
        if flags.len() > 1 {
            return Err("select only one injection mode".into());
        }
        o.mode = flags.pop().unwrap_or(if o.real_fits.is_some() {
            Mode::Real
        } else {
            Mode::Synthetic
        });
    }
    if !matches!(o.mode, Mode::Feature | Mode::Synthetic) && o.real_fits.is_none() {
        return Err("selected mode requires --real-fits".into());
    }
    if o.mode == Mode::Crowded {
        o.normalization = blend_normalization;
    }
    if o.mode == Mode::PairRatio {
        o.peaks = total_peaks.unwrap_or(vec![128., 256., 512.]);
    } else if !custom_peaks {
        o.peaks = match o.mode {
            Mode::Synthetic => vec![8., 12., 16., 24., 36., 56., 84., 128.],
            Mode::Stratified => vec![24., 56., 128.],
            Mode::Comparison => vec![12., 24., 56.],
            _ => vec![12., 16., 24., 36., 56., 84., 128.],
        };
    }
    validate(&o)?;
    Ok(o)
}
fn validate(o: &Options) -> Result<()> {
    let positive =
        |values: &[f64]| !values.is_empty() && values.iter().all(|v| v.is_finite() && *v > 0.);
    if o.trials == 0
        || o.sources == 0
        || o.pairs == 0
        || o.groups == 0
        || !positive(&o.peaks)
        || !positive(&o.ratios)
        || o.ratios.iter().any(|v| *v > 1.)
        || !positive(&o.blend_levels)
        || !positive(&o.blend_separations)
        || o.blend_sizes.is_empty()
        || o.blend_sizes.iter().any(|v| *v < 2)
        || !positive(&[
            o.detector.threshold_sigma,
            o.detector.measurement.psf_fwhm,
            o.detector.measurement.min_flux_snr,
        ])
        || o.detector.min_distance == 0
        || !(1..=9).contains(&o.detector.measurement.min_psf_support_pixels)
        || o.injected_psf.is_some_and(|v| !v.is_finite() || v <= 0.)
        || o.pair_separation.is_some_and(|v| !v.is_finite() || v <= 0.)
    {
        return Err("invalid positive injection, detector, or geometry parameters".into());
    }
    if !["gaussian", "hybrid", "ensemble"].contains(&o.detector.proposal_mode.as_str())
        || !["gaussian", "empirical"].contains(&o.psf_model.as_str())
        || !["peak_excess", "integrated_excess"].contains(&o.normalization.as_str())
    {
        return Err("invalid proposal mode, PSF model, or signal normalization".into());
    }
    let unique = o.strata.iter().collect::<BTreeSet<_>>();
    if unique.len() != o.strata.len()
        || o.strata.is_empty()
        || o.strata.iter().any(|v| {
            !matches!(
                v.as_str(),
                "blank" | "high_background" | "edge" | "crowded" | "special_code" | "line"
            )
        })
    {
        return Err("invalid or duplicate injection strata".into());
    }
    Ok(())
}
pub fn run(o: &Options) -> Result<Value> {
    validate(o)?;
    eprintln!("3.0% · 校验注入回收参数与实验输出目录");
    fs::create_dir_all(&o.out_dir).map_err(|e| e.to_string())?;
    eprintln!("8.0% · 参数通过，准备执行所选注入实验");
    let mut inputs = Vec::new();
    let rows = match o.mode {
        Mode::Synthetic => synthetic(o, &mut inputs)?,
        Mode::Feature => {
            eprintln!("12.0% · 构建特征正控与负控复现实验");
            let mut detector = o.detector.clone();
            detector.background_box_size = 32;
            feature_audit_replicates::run_feature_audit(&detector, o.seed)?
                .iter()
                .map(|v| serde_json::to_value(v).expect("finite feature audit"))
                .collect()
        }
        Mode::Real => real(o, false, &mut inputs)?,
        Mode::Stratified => real(o, true, &mut inputs)?,
        Mode::Comparison => comparison(o, &mut inputs)?,
        Mode::PairRatio => pairs_or_crowded(o, false, &mut inputs)?,
        Mode::Crowded => pairs_or_crowded(o, true, &mut inputs)?,
    };
    eprintln!("82.0% · 注入回收计算完成，汇总召回与质量筛选结果");
    let name = match o.mode {
        Mode::Synthetic => "injection_recovery",
        Mode::Feature => "feature_audit",
        Mode::Real => "real_background_injection",
        Mode::Stratified => "stratified_real_background_injection",
        Mode::Comparison => "proposal_mode_comparison",
        Mode::PairRatio => "pair_flux_ratio_audit",
        Mode::Crowded => "crowded_blend_audit",
    };
    eprintln!("86.0% · 写出逐试验 CSV 与汇总 JSON");
    innovation_artifacts::write_csv(&o.out_dir.join(format!("{name}.csv")), &rows, &[])?;
    let mut payload = json!({"row_count":rows.len(),"rows":rows,"note":"注入真值回收与候选/质量层分开报告；背景检测数、净数量变化和合并响应不能解释成物理身份或真实 precision。"});
    if o.mode == Mode::Feature {
        payload["positive_scenarios"] = json!(rows
            .iter()
            .filter(|r| r["control_type"] == "positive")
            .count());
        payload["negative_scenarios"] = json!(rows
            .iter()
            .filter(|r| r["control_type"] == "negative")
            .count());
    }
    let unique_values = |field: &str| {
        let mut values = Vec::<Value>::new();
        for row in &rows {
            let v = &row[field];
            if !values.contains(v) {
                values.push(v.clone());
            }
        }
        values
    };
    if o.mode == Mode::Stratified {
        payload["strata"] = json!(unique_values("stratum"));
        payload["peak_levels"] = json!(unique_values("peak_excess_adu"));
    }
    if matches!(o.mode, Mode::PairRatio | Mode::Crowded) {
        payload["signal_normalizations"] = json!(unique_values("signal_normalization"));
        payload["total_control_signal_levels"] = json!(unique_values("total_control_signal_adu"));
        if o.mode == Mode::PairRatio {
            payload["secondary_to_primary_ratios"] =
                json!(unique_values("secondary_to_primary_ratio"));
        } else {
            payload["group_sizes"] = json!(unique_values("group_size"));
            payload["nearest_separations_px"] = json!(unique_values("nearest_separation_px"));
        }
    }
    fs::write(
        o.out_dir.join(format!("{name}.json")),
        serde_json::to_vec_pretty(&payload).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let plot_rows = rows
        .iter()
        .filter(|r| o.mode != Mode::Feature || r["control_type"] == "positive")
        .collect::<Vec<_>>();
    let x = plot_rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            if matches!(o.mode, Mode::Synthetic | Mode::Real | Mode::Comparison) {
                r["peak_excess_adu"].as_f64().unwrap_or(i as f64 + 1.)
            } else {
                i as f64 + 1.
            }
        })
        .collect::<Vec<_>>();
    let fields: Vec<&str> = match o.mode {
        Mode::Comparison => vec![
            "gaussian_candidate_recall",
            "hybrid_candidate_recall",
            "ensemble_candidate_recall",
            "gaussian_quality_recall",
            "hybrid_quality_recall",
            "ensemble_quality_recall",
        ],
        Mode::PairRatio => vec![
            "source_candidate_recall",
            "source_quality_recall",
            "pair_candidate_resolution_recall",
            "pair_quality_resolution_recall",
        ],
        Mode::Crowded => vec![
            "group_candidate_resolution_recall",
            "group_quality_resolution_recall",
            "merged_candidate_fraction",
            "merged_quality_fraction",
        ],
        _ => vec!["candidate_recall", "quality_recall"],
    };
    let colors = [
        [215, 148, 50],
        [79, 155, 131],
        [114, 185, 212],
        [165, 108, 193],
        [139, 220, 255],
        [214, 95, 145],
    ];
    let series = fields
        .iter()
        .enumerate()
        .map(|(i, field)| {
            (
                *field,
                plot_rows
                    .iter()
                    .map(|r| r[*field].as_f64().unwrap_or(0.))
                    .collect::<Vec<_>>(),
                colors[i % 6],
            )
        })
        .collect::<Vec<_>>();
    innovation_artifacts::line_chart(
        &o.out_dir.join(if o.mode == Mode::Feature {
            "feature_audit_recall.png".into()
        } else {
            format!("{name}.png")
        }),
        "INJECTION AUDIT - CANDIDATE / QUALITY EVIDENCE",
        &x,
        &series,
        false,
    )?;
    eprintln!("96.0% · 回收率图表生成完成，写出冻结输入清单");
    let manifest = json!({"schema_version":1,"implementation":"Rust","experiment":name,"seed":o.seed.to_string(),"source_fits":o.real_fits,"detector_options":o.detector,"injected_psf_fwhm":o.injected_psf,"frozen_inputs":inputs,"rows":rows,"limitations":["Injected truth is synthetic, not a physical star catalog.","Background counts do not estimate false-positive rate without independent truth."]});
    fs::write(
        o.out_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    eprintln!("99.0% · 注入回收实验结果与清单已保存");
    Ok(
        json!({"output":o.out_dir.canonicalize().map_err(|e|e.to_string())?,"experiment":name,"row_count":rows.len(),"rows":rows}),
    )
}
pub fn run_desktop(args: &[String]) -> Result<Value> {
    run(&parse_args(args)?)
}
