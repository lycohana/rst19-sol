//! Original FITS code statistics and independent-candidate pair PSF audit.
use crate::{
    fits, innovation_artifacts,
    measurement::Detection,
    models::{FitsFrame, PixelData},
    pipeline::{self, FrameOptions},
    psf, source_classification, statistics, Result,
};
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub const HELP: &str = "逐帧复核近邻候选的原始码、孔径泄漏和单/双 PSF 证据\nUsage: rst19-pair-audit DATA_DIR [--primary-x X --primary-y Y --secondary-x X --secondary-y Y] [--shifts-json JSON] [--out-dir DIR]";

#[derive(Clone, Debug)]
pub struct Options {
    pub primary: [f64; 2],
    pub secondary: [f64; 2],
    pub frame: FrameOptions,
    pub match_radius: f64,
    pub search_radius: f64,
    pub shifts: Option<Vec<[f64; 2]>>,
}
impl Default for Options {
    fn default() -> Self {
        let mut frame = FrameOptions::default();
        frame.detection.min_distance = 4;
        frame.detection.proposal_mode = "hybrid".into();
        frame.detection.measurement.refine_local_background = true;
        Self {
            primary: [1269., 3465.],
            secondary: [1274., 3467.],
            frame,
            match_radius: 2.,
            search_radius: 8.,
            shifts: None,
        }
    }
}
pub fn negative_limit(data: &PixelData) -> Option<f64> {
    match data {
        PixelData::I16(_) => Some(-0.9 * i16::MAX as f64),
        PixelData::I32(_) => Some(-0.9 * i32::MAX as f64),
        PixelData::I64(_) => Some(-0.9 * i64::MAX as f64),
        _ => None,
    }
}
pub fn raw_region_statistics(frame: &FitsFrame, center: [f64; 2], radius: usize) -> Value {
    let cx = center[0].round_ties_even() as i64;
    let cy = center[1].round_ties_even() as i64;
    let mut values = Vec::new();
    let mut integers = BTreeMap::<i64, usize>::new();
    let limit = negative_limit(&frame.data);
    let (mut fixed, mut negative, mut positive) = (0, 0, 0);
    for y in (cy - radius as i64).max(0)..(cy + radius as i64 + 1).min(frame.height as i64) {
        for x in (cx - radius as i64).max(0)..(cx + radius as i64 + 1).min(frame.width as i64) {
            if (x as f64 - center[0]).hypot(y as f64 - center[1]) > radius as f64 {
                continue;
            }
            let index = y as usize * frame.width + x as usize;
            let v = frame.data.value(index);
            if !v.is_finite() {
                continue;
            }
            values.push(v);
            if let Some(v) = frame.data.exact_value(index).as_i64() {
                *integers.entry(v).or_default() += 1;
            }
            fixed += usize::from(v == -1.);
            negative += usize::from(limit.is_some_and(|l| v <= l));
            positive += usize::from(limit.is_some_and(|l| v >= -l));
        }
    }
    values.sort_by(f64::total_cmp);
    let mut counts = Vec::<(f64, usize)>::new();
    for &v in &values {
        if let Some(last) = counts.last_mut().filter(|last| last.0 == v) {
            last.1 += 1;
        } else {
            counts.push((v, 1));
        }
    }
    // Exact integer grouping avoids collapsing adjacent int64 storage codes.
    let (unique, mode, mode_count) = if integers.is_empty() {
        let mode = counts
            .iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.total_cmp(&a.0)));
        (counts.len(), mode.map(|v| v.0), mode.map_or(0, |v| v.1))
    } else {
        let mode = integers
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)));
        (
            integers.len(),
            mode.map(|v| *v.0 as f64),
            mode.map_or(0, |v| *v.1),
        )
    };
    json!({"min":values.first(),"max":values.last(),"median":statistics::median(&values).ok(),"unique":unique,"mode":mode,"mode_count":mode_count,"mode_fraction":(!values.is_empty()).then(||mode_count as f64/values.len() as f64),"fixed_minus_one":fixed,"negative_overflow":negative,"positive_extreme":positive})
}
pub fn overlap_statistics(
    frame: &FitsFrame,
    primary: [f64; 2],
    secondary: [f64; 2],
    radius: usize,
    background: Option<f64>,
) -> Result<Value> {
    if radius == 0 || primary.iter().chain(&secondary).any(|v| !v.is_finite()) {
        return Err("invalid aperture overlap parameters".into());
    }
    let mut values = Vec::new();
    let mut shared = Vec::new();
    let x0 = (secondary[0] - radius as f64).floor().max(0.) as usize;
    let y0 = (secondary[1] - radius as f64).floor().max(0.) as usize;
    let x1 = ((secondary[0] + radius as f64).ceil() + 1.)
        .max(0.)
        .min(frame.width as f64) as usize;
    let y1 = ((secondary[1] + radius as f64).ceil() + 1.)
        .max(0.)
        .min(frame.height as f64) as usize;
    for y in y0..y1 {
        for x in x0..x1 {
            let v = frame.data.value(y * frame.width + x);
            if !v.is_finite()
                || (x as f64 - secondary[0]).hypot(y as f64 - secondary[1]) > radius as f64
            {
                continue;
            }
            values.push(v);
            if (x as f64 - primary[0]).hypot(y as f64 - primary[1]) <= radius as f64 {
                shared.push(v);
            }
        }
    }
    let bg = background.filter(|v| v.is_finite());
    let total = (!values.is_empty()).then(|| values.iter().sum::<f64>());
    let overlap = (!shared.is_empty()).then(|| shared.iter().sum::<f64>());
    let net = total.zip(bg).map(|(v, b)| v - b * values.len() as f64);
    let shared_net = overlap.zip(bg).map(|(v, b)| v - b * shared.len() as f64);
    Ok(
        json!({"secondary_pixels":values.len(),"shared_pixels":shared.len(),"background":bg,"secondary_raw_sum":total,"shared_raw_sum":overlap,"secondary_net_sum":net,"shared_net_sum":shared_net,"shared_net_fraction":shared_net.zip(net).and_then(|(s,n)|(n>0.).then_some(s/n)),"shared_max":shared.iter().copied().max_by(f64::total_cmp)}),
    )
}
fn nearest(sources: &[Detection], target: [f64; 2], radius: f64) -> Option<(f64, &Detection)> {
    sources
        .iter()
        .filter_map(|s| {
            let p = s.peak_position();
            let d = (p[0] - target[0]).hypot(p[1] - target[1]);
            (d <= radius).then_some((d, s))
        })
        .min_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.detection_id.cmp(&b.1.detection_id))
        })
}
fn flags(source: Option<&Detection>) -> String {
    source.map_or_else(|| "NOT_FOUND".into(), |s| s.flags.join("|"))
}
fn separation(a: Option<&Detection>, b: Option<&Detection>, peak: bool) -> Option<f64> {
    a.zip(b).map(|(a, b)| {
        let a = if peak { a.peak_position() } else { [a.x, a.y] };
        let b = if peak { b.peak_position() } else { [b.x, b.y] };
        (b[0] - a[0]).hypot(b[1] - a[1])
    })
}
pub fn run(paths: &[PathBuf], o: &Options) -> Result<Value> {
    let d = &o.frame.detection;
    let m = &d.measurement;
    if paths.is_empty()
        || o.primary == o.secondary
        || o.primary.iter().chain(&o.secondary).any(|v| !v.is_finite())
    {
        return Err("pair audit requires frames and distinct finite targets".into());
    }
    if ![
        d.threshold_sigma,
        m.psf_fwhm,
        m.min_flux_snr,
        o.match_radius,
        o.search_radius,
    ]
    .iter()
    .all(|v| v.is_finite() && *v > 0.)
        || d.min_distance < 1
        || m.aperture_radius < 1
        || d.background_box_size < 16
        || d.background_sample_limit < 1
        || !(1..=9).contains(&m.min_psf_support_pixels)
        || d.max_sources == Some(0)
        || o.search_radius < o.match_radius
    {
        return Err("invalid pair audit detector or quality parameters".into());
    }
    let shifts = o
        .shifts
        .clone()
        .unwrap_or_else(|| vec![[0., 0.]; paths.len()]);
    if shifts.len() != paths.len() || shifts.iter().flatten().any(|v| !v.is_finite()) {
        return Err("frame_shifts must match paths and contain finite values".into());
    }
    let distance = (o.secondary[0] - o.primary[0]).hypot(o.secondary[1] - o.primary[1]);
    let pair_radius =
        (m.aperture_radius + 2).max((distance / 2.).ceil() as usize + m.aperture_radius);
    let mut rows = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        let analysis =
            pipeline::analyze_frame(fits::read_fits(path, false)?, None, None, &o.frame, None)?;
        let frame = &analysis.frame;
        let p = [o.primary[0] + shifts[i][0], o.primary[1] + shifts[i][1]];
        let s = [o.secondary[0] + shifts[i][0], o.secondary[1] + shifts[i][1]];
        let primary = nearest(&analysis.detection.sources, p, o.match_radius).map(|x| x.1);
        let secondary = nearest(&analysis.detection.sources, s, o.match_radius).map(|x| x.1);
        let np = nearest(&analysis.detection.sources, p, o.search_radius);
        let ns = nearest(&analysis.detection.sources, s, o.search_radius);
        let raw = raw_region_statistics(frame, s, m.aperture_radius);
        let pair =
            raw_region_statistics(frame, [(p[0] + s[0]) / 2., (p[1] + s[1]) / 2.], pair_radius);
        let overlap = overlap_statistics(
            frame,
            p,
            s,
            m.aperture_radius,
            secondary.map(|x| x.background),
        )?;
        let auxiliary = fits::auxiliary_mask(frame.height, frame.width)?;
        let mut range = auxiliary.clone();
        let mut sentinel = auxiliary.clone();
        let limit = negative_limit(&frame.data);
        for k in 0..frame.data.len() {
            let v = frame.data.value(k);
            range[k] |= limit.is_some_and(|l| v <= l || v >= -l);
            sentinel[k] |= v == -1.;
        }
        let mut evidence = [(None, None); 3];
        if let Some((primary, secondary)) = primary.zip(secondary) {
            let pixels = frame.data.to_f64();
            for (j, mask) in [&auxiliary, &range, &sentinel].iter().enumerate() {
                evidence[j] = psf::pair_psf_evidence(
                    &pixels,
                    mask,
                    frame.height,
                    frame.width,
                    primary.peak_position(),
                    secondary.peak_position(),
                    m.psf_fwhm,
                )?;
            }
        }
        let nps = np.map(|v| v.1);
        let nss = ns.map(|v| v.1);
        let mut row = json!({"frame_index":i+1,"path":frame.path,"frame_shift_x_px":shifts[i][0],"frame_shift_y_px":shifts[i][1],"target_peak_distance_px":distance,"detected_peak_distance_px":separation(primary,secondary,true),"detected_centroid_distance_px":separation(primary,secondary,false),"nearest_primary_distance_px":np.map(|v|v.0),"nearest_secondary_distance_px":ns.map(|v|v.0),"nearest_source_peak_distance_px":separation(nps,nss,true),"nearest_source_same_for_targets":nps.zip(nss).map(|(a,b)|a.detection_id==b.detection_id),"nearest_secondary_detection_id":nss.map(|v|v.detection_id),"nearest_secondary_peak_x":nss.map(|v|v.peak_position()[0]),"nearest_secondary_peak_y":nss.map(|v|v.peak_position()[1]),"nearest_secondary_peak_adu":nss.map(|v|v.peak),"nearest_secondary_quality_passed":nss.map(|v|v.quality_passed),"nearest_secondary_flags":flags(nss),"primary_detection_id":primary.map(|v|v.detection_id),"secondary_detection_id":secondary.map(|v|v.detection_id),"primary_peak_adu":primary.map(|v|v.peak),"secondary_peak_adu":secondary.map(|v|v.peak),"primary_flux_snr":primary.and_then(|v|v.flux_snr),"secondary_flux_snr":secondary.and_then(|v|v.flux_snr),"primary_quality_passed":primary.map(|v|v.quality_passed),"secondary_quality_passed":secondary.map(|v|v.quality_passed),"primary_feature_class":primary.map(source_classification::classify_source_feature),"secondary_feature_class":secondary.map(source_classification::classify_source_feature),"primary_flags":flags(primary),"secondary_flags":flags(secondary),"pair_window_negative_overflow_count":pair["negative_overflow"],"pair_window_positive_extreme_count":pair["positive_extreme"]});
        for (key, input) in [
            ("secondary_aperture_min_adu", "min"),
            ("secondary_aperture_max_adu", "max"),
            ("secondary_aperture_median_adu", "median"),
            ("secondary_aperture_unique_values", "unique"),
            ("secondary_mode_adu", "mode"),
            ("secondary_mode_count", "mode_count"),
            ("secondary_mode_fraction", "mode_fraction"),
            ("secondary_fixed_minus_one_count", "fixed_minus_one"),
            ("secondary_negative_overflow_count", "negative_overflow"),
            ("secondary_positive_extreme_count", "positive_extreme"),
        ] {
            row[key] = raw[input].clone();
        }
        for (j, label) in ["raw", "range_masked", "sentinel_masked"]
            .iter()
            .enumerate()
        {
            row[format!("pair_delta_bic_{label}")] = json!(evidence[j].0);
            row[format!("pair_component_snr_{label}")] = json!(evidence[j].1);
        }
        for (key, input) in [
            ("secondary_aperture_pixel_count", "secondary_pixels"),
            ("secondary_shared_aperture_pixel_count", "shared_pixels"),
            ("secondary_aperture_background_adu", "background"),
            ("secondary_aperture_raw_sum_adu", "secondary_raw_sum"),
            ("secondary_shared_aperture_raw_sum_adu", "shared_raw_sum"),
            ("secondary_aperture_net_sum_adu", "secondary_net_sum"),
            ("secondary_shared_aperture_net_sum_adu", "shared_net_sum"),
            (
                "secondary_shared_aperture_net_fraction",
                "shared_net_fraction",
            ),
            ("secondary_shared_aperture_max_adu", "shared_max"),
        ] {
            row[key] = overlap[input].clone();
        }
        rows.push(row);
    }
    let count = |predicate: fn(&Value) -> bool| rows.iter().filter(|v| predicate(v)).count();
    let found = count(|v| !v["secondary_detection_id"].is_null());
    let quality = count(|v| v["secondary_quality_passed"] == true);
    let extremes = count(|v| {
        v["secondary_negative_overflow_count"].as_u64().unwrap_or(0) > 0
            || v["secondary_positive_extreme_count"].as_u64().unwrap_or(0) > 0
    });
    let supported = count(|v| {
        v["pair_delta_bic_raw"].as_f64().is_some_and(|v| v >= 10.)
            && v["pair_component_snr_raw"]
                .as_f64()
                .is_some_and(|v| v >= 5.)
    });
    let nearby = count(|v| !v["nearest_secondary_detection_id"].is_null());
    let same = count(|v| v["nearest_source_same_for_targets"] == true);
    let (label, note) = if o.shifts.is_some() {
        (
            "按累计平移调整的 detector 目标",
            "仅基于平移注册的 detector 目标诊断",
        )
    } else {
        ("固定 detector 坐标", "固定 detector 坐标的合并/漂移诊断")
    };
    let total = paths.len();
    let conclusion=format!("{label}审计：副目标在 {found}/{total} 帧被候选表找到，其中 {quality}/{total} 帧通过质量层；{extremes}/{total} 帧的副孔径含极端正/负范围码。达到当前双 PSF 工程证据线的帧数为 {supported}/{total}。放宽到目标周围 {} px 后，附近候选在 {nearby}/{total} 帧出现，但两个目标查询落到同一检测源的情况有 {same}/{total} 帧；这只是{note}，不是天空坐标匹配。这些结果支持把两个框解释为‘两个局部提案，副响应需经过异常编码和 PSF 复核’，不能直接报告为两颗已确认恒星。",o.search_radius);
    Ok(
        json!({"frame_count":total,"primary_target_xy":o.primary,"secondary_target_xy":o.secondary,"target_peak_distance_px":distance,"match_radius_px":o.match_radius,"nearest_search_radius_px":o.search_radius,"frame_shifts":shifts,"aperture_radius_px":m.aperture_radius,"association_coordinate_system":if o.shifts.is_some(){"registered detector coordinates from supplied translations; not WCS/sky matching"}else{"fixed detector coordinates; not WCS/sky matching"},"parameters":{"threshold_sigma":d.threshold_sigma,"min_distance":d.min_distance,"aperture_radius":m.aperture_radius,"psf_fwhm":m.psf_fwhm,"background_box_size":d.background_box_size,"background_sample_limit":d.background_sample_limit,"min_flux_snr":m.min_flux_snr,"min_psf_support_pixels":m.min_psf_support_pixels,"proposal_mode":d.proposal_mode,"max_sources":d.max_sources,"match_radius_px":o.match_radius,"nearest_search_radius_px":o.search_radius,"frame_shifts":shifts,"pair_psf_confirmation_line":"delta_bic >= 10 and component_snr >= 5; engineering audit only","range_code_threshold":"signed integer negative <= -0.9 * dtype max; positive counterpart is diagnostic only"},"rows":rows,"conclusion":conclusion}),
    )
}
pub fn write_artifacts(result: &Value, out: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let output = out.canonicalize().map_err(|e| e.to_string())?;
    let rows = result["rows"].as_array().ok_or("missing pair audit rows")?;
    innovation_artifacts::write_csv(&output.join("source_pair_audit.csv"), rows, &[])?;
    let mut payload = result.clone();
    let coordinate = if result["association_coordinate_system"]
        .as_str()
        .unwrap_or("")
        .starts_with("registered detector")
    {
        "按累计平移调整的 registered detector 坐标"
    } else {
        "固定 detector 坐标"
    };
    payload["note"]=json!(format!("ID 只在单次检测运行内有效；secondary 的原始像素统计使用{coordinate}，pair PSF 数值是局部模型比较，不是星表身份或物理真值；secondary_shared_aperture_* 是两个圆孔径的几何通量泄漏诊断，不是去混叠后的独立源通量。"));
    fs::write(
        output.join("source_pair_audit.json"),
        serde_json::to_vec_pretty(&payload).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let x = rows
        .iter()
        .map(|v| v["frame_index"].as_f64().unwrap_or(0.))
        .collect::<Vec<_>>();
    let series = |key: &str| {
        rows.iter()
            .map(|v| v[key].as_f64().unwrap_or(f64::NAN))
            .collect::<Vec<_>>()
    };
    innovation_artifacts::line_chart(
        &output.join("source_pair_raw_codes.png"),
        "SOURCE PAIR AUDIT - RAW RANGE / REPEATED CODE EVIDENCE",
        &x,
        &[
            (
                "secondary negative overflow",
                series("secondary_negative_overflow_count"),
                [199, 53, 158],
            ),
            (
                "secondary positive extreme",
                series("secondary_positive_extreme_count"),
                [215, 148, 50],
            ),
            (
                "secondary exact -1",
                series("secondary_fixed_minus_one_count"),
                [79, 155, 131],
            ),
        ],
        false,
    )?;
    innovation_artifacts::line_chart(
        &output.join("source_pair_psf_evidence.png"),
        "SOURCE PAIR AUDIT - SINGLE / DOUBLE PSF EVIDENCE",
        &x,
        &[
            (
                "raw delta BIC",
                series("pair_delta_bic_raw"),
                [215, 148, 50],
            ),
            (
                "range-masked delta BIC",
                series("pair_delta_bic_range_masked"),
                [79, 155, 131],
            ),
            (
                "confirmation line delta BIC=10",
                vec![10.; rows.len()],
                [199, 53, 158],
            ),
        ],
        false,
    )?;
    Ok(output)
}
pub fn run_desktop(args: &[String]) -> Result<Value> {
    let mut data = None;
    let mut output = PathBuf::from("tmp/source-pair-audit");
    let mut o = Options::default();
    let mut parameters = Map::new();
    let mut shifts_path = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if !arg.starts_with('-') {
            if data.replace(PathBuf::from(arg)).is_some() {
                return Err("only one data directory allowed".into());
            }
            i += 1;
            continue;
        }
        let (key, value) = if let Some((key, value)) = arg.split_once('=') {
            (key.to_owned(), value.to_owned())
        } else {
            i += 1;
            (
                arg.clone(),
                args.get(i)
                    .ok_or_else(|| format!("{arg} requires a value"))?
                    .clone(),
            )
        };
        match key.as_str() {
            "--out-dir" => output = PathBuf::from(value),
            "--shifts-json" => shifts_path = Some(PathBuf::from(value)),
            "--primary-x"
            | "--primary-y"
            | "--secondary-x"
            | "--secondary-y"
            | "--match-radius-px"
            | "--nearest-search-radius-px" => {
                let v = value.parse::<f64>().map_err(|_| format!("invalid {key}"))?;
                match key.as_str() {
                    "--primary-x" => o.primary[0] = v,
                    "--primary-y" => o.primary[1] = v,
                    "--secondary-x" => o.secondary[0] = v,
                    "--secondary-y" => o.secondary[1] = v,
                    "--match-radius-px" => o.match_radius = v,
                    _ => o.search_radius = v,
                }
            }
            "--proposal-mode" => {
                if !["gaussian", "hybrid", "ensemble"].contains(&value.as_str()) {
                    return Err("invalid proposal mode".into());
                }
                parameters.insert("proposal_mode".into(), json!(value));
            }
            "--min-distance"
            | "--aperture-radius"
            | "--background-box-size"
            | "--min-psf-support-pixels"
            | "--max-sources" => {
                let v = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid {key}"))?;
                parameters.insert(
                    key[2..].replace('-', "_"),
                    if key == "--max-sources" && v == 0 {
                        Value::Null
                    } else {
                        json!(v)
                    },
                );
            }
            "--threshold-sigma" | "--psf-fwhm" | "--min-flux-snr" => {
                let v = value.parse::<f64>().map_err(|_| format!("invalid {key}"))?;
                parameters.insert(key[2..].replace('-', "_"), json!(v));
            }
            _ => return Err(format!("unknown pair audit argument {key}")),
        }
        i += 1;
    }
    let mut defaults = serde_json::to_value(&o.frame.detection).map_err(|e| e.to_string())?;
    defaults
        .as_object_mut()
        .ok_or("invalid detector options")?
        .extend(parameters);
    o.frame.detection = serde_json::from_value(defaults).map_err(|e| e.to_string())?;
    if let Some(path) = shifts_path {
        let payload: Value = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        o.shifts = Some(
            serde_json::from_value(payload["cumulative_shifts"].clone())
                .map_err(|e| e.to_string())?,
        );
    }
    let data = data.ok_or("missing FITS directory")?;
    let mut paths = fs::read_dir(data)
        .map_err(|e| e.to_string())?
        .map(|entry| entry.map(|e| e.path()).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>>>()?;
    paths.retain(|p| {
        p.is_file()
            && p.extension()
                .is_some_and(|e| e.to_string_lossy().eq_ignore_ascii_case("fits"))
    });
    paths.sort();
    let result = run(&paths, &o)?;
    let output = write_artifacts(&result, &output)?;
    Ok(
        json!({"output":output,"frame_count":result["frame_count"],"target_distance":result["target_peak_distance_px"],"conclusion":result["conclusion"]}),
    )
}
