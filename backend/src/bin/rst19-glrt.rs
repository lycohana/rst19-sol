//! rst19-glrt: DG-GLRT prototype (M1+M2) command line driver.
//!
//!   rst19-glrt --image <fits> --psf-fwhm 1.93 --lengths 16,32,64,128,256,512 \
//!              --angles 12 --whiten on --threshold-fa 1 --out <dir>
//!
//!   rst19-glrt selftest --out <dir> [--size 512] [--lengths 16,32,64,128]
//!
//! The first form writes <out>/glrt.json (and the optional f32 maps). The
//! selftest form runs the two synthetic acceptance cases (white noise scatter
//! and an injected streak recovery) and writes <out>/selftest.json.

use rst19_core::glrt::{self, Engine, Options};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const USAGE: &str = "\
usage:
  rst19-glrt --image <fits> [--psf-fwhm 1.93] [--lengths 16,32,64,128,256,512,1024]
             [--angles 12] [--whiten on|off] [--threshold-fa 1] [--out <dir>]
             [--engine auto|freq|line] [--write-maps] [--max-detections 200]
             [--calib-size 0] [--floor 2.5] [--background-block 64] [--seed N]
             [--fast] [--phi-step 0.25] [--l-step 1.05] [--phi-range 5]
             [--refine-crop 1536] [--max-refinements 200]
  rst19-glrt selftest [--out <dir>] [--size 512] [--lengths 16,32,64,128]
             [--angles 12] [--psf-fwhm 1.93] [--whiten on|off] [--engine line|freq]";

fn main() {
    if let Err(err) = run() {
        eprintln!("rst19-glrt: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let mut mode = "run".to_string();
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].clone();
        if let Some(rest) = arg.strip_prefix("--") {
            if let Some(eq) = rest.find('=') {
                map.insert(rest[..eq].to_string(), rest[eq + 1..].to_string());
                i += 1;
            } else if let Some(next) = args.get(i + 1) {
                if next.starts_with("--") {
                    map.insert(rest.to_string(), "on".to_string());
                    i += 1;
                } else {
                    map.insert(rest.to_string(), next.clone());
                    i += 2;
                }
            } else {
                map.insert(rest.to_string(), "on".to_string());
                i += 1;
            }
        } else {
            mode = arg;
            i += 1;
        }
    }
    match mode.as_str() {
        "run" => run_image(&map),
        "selftest" => selftest(&map),
        other => Err(format!("unknown mode {other}\n{USAGE}")),
    }
}

fn float(map: &BTreeMap<String, String>, key: &str, default: f64) -> Result<f64, String> {
    match map.get(key) {
        None => Ok(default),
        Some(v) => v
            .parse::<f64>()
            .map_err(|_| format!("--{key} must be a number, got {v}")),
    }
}

fn integer(map: &BTreeMap<String, String>, key: &str, default: usize) -> Result<usize, String> {
    match map.get(key) {
        None => Ok(default),
        Some(v) => v
            .parse::<usize>()
            .map_err(|_| format!("--{key} must be a non-negative integer, got {v}")),
    }
}

fn flag(map: &BTreeMap<String, String>, key: &str) -> bool {
    match map.get(key) {
        None => false,
        Some(v) => !(v == "off" || v == "false" || v == "no" || v == "0"),
    }
}

fn lengths(map: &BTreeMap<String, String>, default: &[f64]) -> Result<Vec<f64>, String> {
    match map.get("lengths") {
        None => Ok(default.to_vec()),
        Some(text) => {
            let mut out = Vec::new();
            for part in text.split(',') {
                let v: f64 = part
                    .trim()
                    .parse()
                    .map_err(|_| format!("--lengths expects comma separated numbers, got {part}"))?;
                if !(v >= 2.0) {
                    return Err(format!("--lengths entries must be >= 2 px, got {v}"));
                }
                out.push(v);
            }
            out.sort_by(|a, b| a.total_cmp(b));
            out.dedup();
            if out.is_empty() {
                return Err("--lengths must not be empty".into());
            }
            Ok(out)
        }
    }
}

fn engine(map: &BTreeMap<String, String>) -> Result<Engine, String> {
    match map.get("engine").map(String::as_str) {
        None | Some("auto") | Some("line") => Ok(Engine::Line),
        Some("freq") => Ok(Engine::Freq),
        Some(other) => Err(format!("--engine must be auto, line or freq, got {other}")),
    }
}

fn options(map: &BTreeMap<String, String>, default_lengths: &[f64]) -> Result<Options, String> {
    Ok(Options {
        psf_fwhm: float(map, "psf-fwhm", 1.93)?,
        lengths: lengths(map, default_lengths)?,
        n_angles: integer(map, "angles", 12)?.max(2),
        whiten: map.get("whiten").map(|_| flag(map, "whiten")).unwrap_or(true),
        threshold_fa: float(map, "threshold-fa", 1.0)?,
        engine: engine(map)?,
        max_detections: integer(map, "max-detections", 200)?,
        max_refinements: integer(map, "max-refinements", 200)?,
        min_separation_px: float(map, "min-separation", 5.0)?,
        calib_size: integer(map, "calib-size", 0)?,
        seed: integer(map, "seed", 20_260_330)? as u64,
        background_block: integer(map, "background-block", 64)?.max(8),
        floor: float(map, "floor", 2.5)?,
        keep_zline: flag(map, "write-maps"),
        refine: !flag(map, "fast"),
        fine: glrt::FineCfg {
            phi_step_deg: float(map, "phi-step", 0.25)?.max(0.005),
            l_step_frac: float(map, "l-step", 1.05)?.max(1.001),
            phi_half_range_deg: float(map, "phi-range", 5.0)?.max(0.25),
            crop_px: integer(map, "refine-crop", 1536)?.clamp(64, 8192),
        },
    })
}

fn out_dir(map: &BTreeMap<String, String>, default: &str) -> PathBuf {
    PathBuf::from(map.get("out").cloned().unwrap_or_else(|| default.to_string()))
}

fn write_json(dir: &Path, name: &str, value: &serde_json::Value) -> Result<(), String> {
    glrt::ensure_dir(dir)?;
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(name), text + "\n")
        .map_err(|e| format!("cannot write {}: {e}", dir.join(name).display()))
}

fn run_image(map: &BTreeMap<String, String>) -> Result<(), String> {
    let image = map
        .get("image")
        .ok_or_else(|| format!("--image is required\n{USAGE}"))?;
    let opts = options(
        map,
        &[16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
    )?;
    let res = glrt::run_fits(Path::new(image), &opts)?;
    let dir = out_dir(map, "tmp/gllrt/run");
    let maps = if flag(map, "write-maps") {
        Some(res.write_maps(&dir).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let json = res.to_json(image, maps);
    write_json(&dir, "glrt.json", &json)?;
    println!(
        "image {image}\n  shape {}x{}  sigma_adu {:.3}  corr_len {:.3} px  n_eff {:.3e}",
        res.shape.0, res.shape.1, res.sigma_adu, res.noise.corr_len_px, res.n_eff
    );
    println!(
        "  library: {} lengths x {} angles = {} hypotheses ({})",
        res.lengths.len(),
        res.n_angles,
        res.n_hypotheses,
        res.engine
    );
    println!(
        "  gumbel mu {:.3} beta {:.3}  thresholds fa1 {:.3} fa10 {:.3} fa100 {:.3}",
        res.gumbel.0, res.gumbel.1, res.thresholds[0], res.thresholds[1], res.thresholds[2]
    );
    println!(
        "  std_z_point {:.4}  std_lambda {:.4}  margin {} px  calib {}",
        res.std_z_point, res.std_lambda, res.margin_px, res.calib_size
    );
    match &res.fine_cfg {
        Some(cfg) => println!(
            "  refinement: phi step {:.3} deg, L step {:.3}x, crop {} px, {} trials max, +{:.3} on the threshold, {} refined detections",
            cfg.phi_step_deg,
            cfg.l_step_frac,
            cfg.crop_px,
            res.refine_trials,
            res.refine_inflation,
            res.refined_detections
        ),
        None => println!("  refinement: off (--fast)"),
    }
    println!("  detections above fa({}): {}", opts.threshold_fa, res.detections.len());
    for (i, d) in res.detections.iter().take(10).enumerate() {
        println!(
            "   {:>3}  x {:8.3} y {:8.3}  lambda {:10.3}  lr {:10.3}  L {:8.2}  pa {:7.2}  sx {:7.4}  coarse {:9.3}",
            i + 1,
            d.x,
            d.y,
            d.lambda,
            d.lr,
            d.length_px,
            d.pa_deg,
            d.sigma_x_px,
            d.coarse_lambda
        );
    }
    for (key, value) in &res.timings_s {
        println!("  timing {key:<18} {value:8.3} s");
    }
    println!("  wrote {}", dir.join("glrt.json").display());
    Ok(())
}

fn nearest<'a>(
    detections: &'a [glrt::Detection],
    x: f64,
    y: f64,
) -> Option<&'a glrt::Detection> {
    detections.iter().min_by(|a, b| {
        let da = (a.x - x).powi(2) + (a.y - y).powi(2);
        let db = (b.x - x).powi(2) + (b.y - y).powi(2);
        da.total_cmp(&db)
    })
}

fn selftest(map: &BTreeMap<String, String>) -> Result<(), String> {
    let size = integer(map, "size", 512)?.max(64);
    let default_lengths: Vec<f64> = if size >= 2048 {
        vec![16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
    } else if size >= 1024 {
        vec![16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
    } else {
        vec![16.0, 32.0, 64.0, 128.0]
    };
    let mut opts = options(map, &default_lengths)?;
    opts.calib_size = opts.calib_size.max(size);
    opts.keep_zline = true;
    if !map.contains_key("floor") {
        opts.floor = 0.0;
    }
    let dir = out_dir(map, "tmp/gllrt/selftest");
    let sigma_psf = opts.psf_fwhm / glrt::FWHM_TO_SIGMA;

    let white = glrt::white_noise(size, size, 1.0, 7);
    let res_white = glrt::run_array(&opts, &white, size, size)?;

    let mut img = glrt::white_noise(size, size, 1.0, 11);
    let cx = (size / 2) as f64;
    let cy = (size / 2) as f64;
    let streak_len = 64.0;
    let streak_pa = 35.0;
    let peak_snr = 15.0;
    let target = glrt::streak_kernel(sigma_psf, streak_len, streak_pa);
    glrt::inject_streak(&mut img, size, size, &target, cx, cy, peak_snr, 1.0);
    let res_streak = glrt::run_array(&opts, &img, size, size)?;

    let hit = nearest(&res_streak.detections, cx, cy).cloned();
    let (found, errors, checks) = match &hit {
        Some(d) => {
            let dx = d.x - cx;
            let dy = d.y - cy;
            let d_len = d.length_px - streak_len;
            let d_pa = d.pa_deg - streak_pa;
            (
                true,
                json!({"dx_px": round(&dx), "dy_px": round(&dy),
                       "dL_px": round(&d_len), "dpa_deg": round(&d_pa)}),
                json!({
                    "lr_gt_6": d.lr > 6.0,
                    "position_within_2px": (dx * dx + dy * dy).sqrt() <= 2.0,
                    "length_within_15px": d_len.abs() <= 15.0,
                    "pa_within_8deg": d_pa.abs() <= 8.0,
                }),
            )
        }
        None => (false, json!(null), json!(null)),
    };

    let out = json!({
        "psf_fwhm": opts.psf_fwhm,
        "engine": if opts.engine == Engine::Freq { "freq" } else { "line" },
        "whiten": opts.whiten,
        "size": size,
        "lengths": opts.lengths,
        "n_angles": opts.n_angles,
        "n_hypotheses": res_white.n_hypotheses,
        "white_noise": {
            "std_z_point": round(&res_white.std_z_point),
            "std_lambda": round(&res_white.std_lambda),
            "std_z_acceptance_band": [0.98, 1.02],
            "std_z_pass": (res_white.std_z_point - 1.0).abs() <= 0.02,
            "gumbel": {"mu": round(&res_white.gumbel.0), "beta": round(&res_white.gumbel.1)},
            "thresholds": {"fa_1": round(&res_white.thresholds[0]),
                           "fa_10": round(&res_white.thresholds[1]),
                           "fa_100": round(&res_white.thresholds[2])},
            "n_eff": round(&res_white.n_eff),
            "margin_px": res_white.margin_px,
            "detections_above_fa1": res_white.detections.len(),
            "timings_s": timings(&res_white),
        },
        "streak": {
            "injected": {"x": cx, "y": cy, "length_px": streak_len, "pa_deg": streak_pa,
                          "peak_snr": peak_snr, "noise_sigma_adu": 1.0},
            "found": found,
            "recovered": hit.as_ref().map(|d| json!({
                "x": round(&d.x), "y": round(&d.y), "lambda": round(&d.lambda),
                "lr": round(&d.lr), "L_px": round(&d.length_px), "pa_deg": round(&d.pa_deg),
                "sigma_x_px": d.sigma_x_px})),
            "errors": errors,
            "acceptance": checks,
            "detections_above_fa1": res_streak.detections.len(),
            "gumbel": {"mu": round(&res_streak.gumbel.0), "beta": round(&res_streak.gumbel.1)},
            "timings_s": timings(&res_streak),
        },
    });
    write_json(&dir, "selftest.json", &out)?;
    println!("{USAGE}\n");
    println!(
        "white noise: std(z) = {:.4} (band 0.98..1.02, pass {}), std(Lambda) = {:.4}, \
         gumbel mu {:.3} beta {:.3}, fa1 threshold {:.3}, {} detections above fa1, margin {} px",
        res_white.std_z_point,
        (res_white.std_z_point - 1.0).abs() <= 0.02,
        res_white.std_lambda,
        res_white.gumbel.0,
        res_white.gumbel.1,
        res_white.thresholds[0],
        res_white.detections.len(),
        res_white.margin_px
    );
    match &hit {
        Some(d) => println!(
            "streak: x {:.3} y {:.3} (truth {cx:.1}, {cy:.1})  LR {:.3}  L {:.2} px (truth 64)  \
             pa {:.2} deg (truth 35)  lambda {:.3}",
            d.x, d.y, d.lr, d.length_px, d.pa_deg, d.lambda
        ),
        None => println!("streak: NOT recovered"),
    }
    println!("wrote {}", dir.join("selftest.json").display());
    Ok(())
}

fn round(v: &f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

fn timings(res: &glrt::GlrtResult) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (key, value) in &res.timings_s {
        map.insert(key.clone(), json!(round(value)));
    }
    serde_json::Value::Object(map)
}
