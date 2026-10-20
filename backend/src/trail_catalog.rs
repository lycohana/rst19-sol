//! Optional DG-GLRT trail catalog (`trail_catalog.json`) written after the source tables.
//!
//! On by default; `--trail-catalog off` removes both the file and its cost. The
//! GLRT statistics live in `glrt` and are only called here, never modified.

use crate::glrt::{self, Engine, Options};
use serde_json::{json, Value};
use std::path::Path;

/// Fixed hypothesis grid of the catalog step.
const LENGTHS: [f64; 7] = [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0];
const ANGLES: usize = 24;
/// Detections shorter than this are point-like and never enter the catalog.
const MIN_TRAIL_PX: f64 = 40.0;
const MAX_TRAILS: usize = 50;
const MAX_DETECTIONS: usize = 3000;

fn round(value: f64, scale: f64) -> f64 {
    let scaled = (value * scale).round() / scale;
    if scaled == 0.0 {
        0.0
    } else {
        scaled
    }
}

/// Strip `--trail-catalog on|off` from `args`; enabled by default.
pub fn take_flag(args: &[String]) -> Result<(bool, Vec<String>), String> {
    let mut enabled = true;
    let mut kept = Vec::with_capacity(args.len());
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let value = if arg == "--trail-catalog" {
            index += 1;
            args.get(index)
                .map(String::as_str)
                .ok_or("--trail-catalog requires a value")?
        } else if let Some(value) = arg.strip_prefix("--trail-catalog=") {
            value
        } else {
            kept.push(arg.clone());
            index += 1;
            continue;
        };
        enabled = match value {
            "on" => true,
            "off" => false,
            other => return Err(format!("--trail-catalog must be on or off, got {other}")),
        };
        index += 1;
    }
    Ok((enabled, kept))
}

/// Core PSF FWHM read from `source_catalog.csv`, when the table exists.
///
/// The *median* FWHM of a detection table is inflated by blends and by saturated
/// cores (3.05 px on the first frame against a 1.93 px PSF), and a kernel that wide
/// biases every trail length downwards. The 15th percentile follows the core instead.
pub fn measured_fwhm(out_dir: &Path) -> Option<f64> {
    let mut reader = csv::Reader::from_path(out_dir.join("source_catalog.csv")).ok()?;
    let column = reader
        .headers()
        .ok()?
        .iter()
        .position(|name| name == "fwhm")?;
    let mut values: Vec<f64> = reader
        .records()
        .filter_map(|record| {
            let record = record.ok()?;
            let value: f64 = record.get(column)?.trim().parse().ok()?;
            (value.is_finite() && value > 0.0).then_some(value)
        })
        .collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let index = (values.len() as f64 * 0.15).round() as usize;
    Some(values[index.min(values.len() - 1)])
}

/// True when two detections describe the same ridge: nearly parallel and within a
/// few pixels of each other's axis.
fn same_ridge(a: &glrt::Detection, b: &glrt::Detection) -> bool {
    let mut delta = (a.pa_deg - b.pa_deg).abs() % 180.0;
    if delta > 90.0 {
        delta = 180.0 - delta;
    }
    if delta > 3.0 {
        return false;
    }
    let theta = a.pa_deg.to_radians();
    let (ct, st) = (theta.cos(), theta.sin());
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    (-st * dx + ct * dy).abs() < 3.0
}

pub struct Catalog {
    /// One log line, e.g. `trail catalog: 9 uniform trails, 1087 long candidates, 23.6 s`.
    pub line: String,
}

/// Run the GLRT once on `fits` and write `out_dir/trail_catalog.json`.
pub fn run(fits: &Path, out_dir: &Path, arg_fwhm: f64, detection_s: f64) -> Result<Catalog, String> {
    let psf_fwhm = measured_fwhm(out_dir).unwrap_or(arg_fwhm);
    if !(psf_fwhm.is_finite() && psf_fwhm > 0.0) {
        return Err("the trail catalog needs a positive PSF FWHM".into());
    }
    let options = Options {
        psf_fwhm,
        lengths: LENGTHS.to_vec(),
        n_angles: ANGLES,
        whiten: true,
        threshold_fa: 1.0,
        engine: Engine::Line,
        max_detections: MAX_DETECTIONS,
        ..Options::default()
    };
    let started = std::time::Instant::now();
    let result = glrt::run_fits(fits, &options)?;
    let seconds = started.elapsed().as_secs_f64();

    let n_long = result
        .detections
        .iter()
        .filter(|detection| detection.length_px >= MIN_TRAIL_PX)
        .count();
    let mut trails: Vec<&glrt::Detection> = result
        .detections
        .iter()
        .filter(|detection| detection.uniform && detection.length_px >= MIN_TRAIL_PX)
        .collect();
    trails.sort_by(|a, b| b.lambda.total_cmp(&a.lambda));
    // The Lambda ridge of a long trail is flat along its axis, so the detector reports
    // the same physical streak at many along-axis positions. Collapse those before
    // publishing, or a single object shows up as dozens of "trails".
    let mut unique: Vec<&glrt::Detection> = Vec::new();
    for candidate in trails {
        if unique
            .iter()
            .any(|kept| same_ridge(kept, candidate))
        {
            continue;
        }
        unique.push(candidate);
    }
    let trails = unique;
    let rows: Vec<Value> = trails
        .iter()
        .take(MAX_TRAILS)
        .map(|detection| {
            json!({
                "x": round(detection.x, 1e4),
                "y": round(detection.y, 1e4),
                "lr": round(detection.lr, 1e4),
                "lambda": round(detection.lambda, 1e4),
                "L_px": round(detection.length_px, 1e4),
                "pa_deg": round(detection.pa_deg, 1e4),
                "duty": round(detection.duty, 1e4),
                "scatter": round(detection.scatter, 1e4),
            })
        })
        .collect();
    let seconds = round(seconds, 10.0);
    let json = json!({
        "psf_fwhm": round(psf_fwhm, 1e4),
        "n_hypotheses": result.n_hypotheses,
        "thresholds": {
            "fa_1": round(result.thresholds[0], 1e4),
            "fa_10": round(result.thresholds[1], 1e4),
            "fa_100": round(result.thresholds[2], 1e4),
        },
        "n_long_candidates": n_long,
        "n_trails": trails.len(),
        "trails": rows,
        "timings_s": {
            "detection": round(detection_s, 10.0),
            "trail_catalog": seconds,
        },
    });
    let path = out_dir.join("trail_catalog.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    let line = format!(
        "trail catalog: {} uniform trails, {} long candidates, {seconds:.1} s",
        trails.len(),
        n_long
    );
    Ok(Catalog { line })
}

/// The summary line of an already written `trail_catalog.json`.
pub fn line_from_file(out_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(out_dir.join("trail_catalog.json")).ok()?;
    let json: Value = serde_json::from_str(&text).ok()?;
    let seconds = json["timings_s"]["trail_catalog"].as_f64()?;
    Some(format!(
        "trail catalog: {} uniform trails, {} long candidates, {seconds:.1} s",
        json["n_trails"], json["n_long_candidates"]
    ))
}

/// Run the optional step and log it; a failure never aborts the source catalog.
pub fn report(fits: &Path, out_dir: &Path, arg_fwhm: f64, detection_s: f64, to_stdout: bool) {
    match run(fits, out_dir, arg_fwhm, detection_s) {
        Ok(catalog) => {
            if to_stdout {
                println!("{}", catalog.line);
            } else {
                eprintln!("{}", catalog.line);
            }
        }
        Err(error) => eprintln!("trail catalog: skipped ({error})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|item| item.to_string()).collect()
    }

    #[test]
    fn flag_defaults_on_and_is_stripped_both_ways() {
        let (enabled, kept) = take_flag(&args(&["frame.fits", "--out-dir", "out"])).unwrap();
        assert!(enabled);
        assert_eq!(kept, args(&["frame.fits", "--out-dir", "out"]));

        let (enabled, kept) =
            take_flag(&args(&["frame.fits", "--trail-catalog", "off", "--out-dir", "out"])).unwrap();
        assert!(!enabled);
        assert_eq!(kept, args(&["frame.fits", "--out-dir", "out"]));

        let (enabled, kept) =
            take_flag(&args(&["--trail-catalog=off", "frame.fits"])).unwrap();
        assert!(!enabled);
        assert_eq!(kept, args(&["frame.fits"]));

        assert!(take_flag(&args(&["--trail-catalog", "maybe"])).is_err());
        assert!(take_flag(&args(&["--trail-catalog"])).is_err());
    }

    #[test]
    fn measured_fwhm_reads_the_catalog_median() {
        let dir = std::env::temp_dir().join(format!("rst19-trail-catalog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("source_catalog.csv"),
            "id,fwhm,flux\n1,1.0,5\n2,2.0,5\n3,9.0,5\n",
        )
        .unwrap();
        assert_eq!(measured_fwhm(&dir), Some(2.0));
        std::fs::remove_dir_all(&dir).ok();
    }
}