//! Reference-star pairing, empirical zero point, k-fold and spatial-block cross validation.
//!
//! The result magnitude field is g_emp (empirical magnitude through an instrument band whose
//! response is unknown); the reference catalog magnitude is never used as a result name.

use rst19_core::{fits, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const BAND_DEFINITION: &str = "450-750 nm nominal, response unknown";
const PHOTOMETRIC_SIGMA_FLOOR_MAG: f64 = 0.05;
const SCHEMA_VERSION: u32 = 1;

struct Config {
    fits: PathBuf,
    sources: PathBuf,
    gaia: PathBuf,
    out: Option<PathBuf>,
    scale: f64,
    rotation: f64,
    parity: i32,
    crpix_x: f64,
    crpix_y: f64,
    center_ra: f64,
    center_dec: f64,
    epoch: f64,
    match_radius: f64,
    bootstrap_radius: f64,
    bootstrap_iters: usize,
    peak_ceiling: f64,
    aperture_radius: f64,
    annulus_inner: f64,
    annulus_outer: f64,
    clip_sigma: f64,
    clip_iterations: usize,
    kfold: usize,
    block_grid: usize,
    max_iter: usize,
    min_pairs: usize,
    gain: Option<f64>,
    include_rejected: bool,
    target_x: f64,
    target_y: f64,
    seed: u64,
}

impl Config {
    fn defaults() -> Self {
        Self {
            fits: PathBuf::from("tmp/frame1_only/20260330163205413_9901.fits"),
            sources: PathBuf::from("tmp/rust-sources-prod/source_catalog.csv"),
            gaia: PathBuf::from("tmp/gaia/frame1_gaia_g13.csv"),
            out: None,
            scale: 8.1606,
            rotation: 129.8507,
            parity: 1,
            crpix_x: 2048.0,
            crpix_y: 2048.0,
            center_ra: 129.533_548_252_294_5,
            center_dec: -1.845_371_993_254_574_2,
            epoch: 2026.2434,
            match_radius: 1.2,
            bootstrap_radius: 4.0,
            bootstrap_iters: 3,
            peak_ceiling: 20000.0,
            aperture_radius: 4.0,
            annulus_inner: 6.0,
            annulus_outer: 10.0,
            clip_sigma: 3.0,
            clip_iterations: 4,
            kfold: 5,
            block_grid: 8,
            max_iter: 12,
            min_pairs: 20,
            gain: None,
            include_rejected: false,
            target_x: 2929.8555,
            target_y: 3915.2171,
            seed: 19,
        }
    }

    fn as_json(&self) -> Value {
        json!({
            "fits": self.fits.to_string_lossy(),
            "sources": self.sources.to_string_lossy(),
            "gaia": self.gaia.to_string_lossy(),
            "scale_arcsec": self.scale,
            "rotation_deg": self.rotation,
            "parity": self.parity,
            "crpix_x": self.crpix_x,
            "crpix_y": self.crpix_y,
            "center_ra_deg": self.center_ra,
            "center_dec_deg": self.center_dec,
            "epoch": self.epoch,
            "match_radius_px": self.match_radius,
            "bootstrap_radius_px": self.bootstrap_radius,
            "bootstrap_iters": self.bootstrap_iters,
            "peak_ceiling": self.peak_ceiling,
            "aperture_radius_px": self.aperture_radius,
            "annulus_inner_px": self.annulus_inner,
            "annulus_outer_px": self.annulus_outer,
            "clip_sigma": self.clip_sigma,
            "clip_iterations": self.clip_iterations,
            "kfold": self.kfold,
            "block_grid": self.block_grid,
            "max_iter": self.max_iter,
            "min_pairs": self.min_pairs,
            "gain": self.gain,
            "include_rejected_sources": self.include_rejected,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "seed": self.seed,
        })
    }
}

fn number(name: &str, text: &str) -> Result<f64> {
    text.parse::<f64>()
        .map_err(|_| format!("{name} has an invalid value: {text}"))
}

fn parse_args(args: &[String]) -> Result<Config> {
    let mut cfg = Config::defaults();
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].clone();
        if arg == "-h" || arg == "--help" {
            return Err("__help__".into());
        }
        if arg == "--include-rejected-sources" {
            cfg.include_rejected = true;
            index += 1;
            continue;
        }
        if !arg.starts_with("--") {
            cfg.fits = PathBuf::from(arg);
            index += 1;
            continue;
        }
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) => (name.to_owned(), Some(value.to_owned())),
            None => (arg.clone(), None),
        };
        let value = match inline {
            Some(value) => value,
            None => {
                index += 1;
                args.get(index)
                    .cloned()
                    .ok_or_else(|| format!("{name} requires a value"))?
            }
        };
        match name.as_str() {
            "--fits" => cfg.fits = PathBuf::from(value),
            "--sources" => cfg.sources = PathBuf::from(value),
            "--gaia" => cfg.gaia = PathBuf::from(value),
            "--out" => cfg.out = Some(PathBuf::from(value)),
            "--scale" => cfg.scale = number(&name, &value)?,
            "--rotation" => cfg.rotation = number(&name, &value)?,
            "--parity" => cfg.parity = number(&name, &value)? as i32,
            "--crpix-x" => cfg.crpix_x = number(&name, &value)?,
            "--crpix-y" => cfg.crpix_y = number(&name, &value)?,
            "--center-ra" => cfg.center_ra = number(&name, &value)?,
            "--center-dec" => cfg.center_dec = number(&name, &value)?,
            "--epoch" => cfg.epoch = number(&name, &value)?,
            "--match-radius" => cfg.match_radius = number(&name, &value)?,
            "--bootstrap-radius" => cfg.bootstrap_radius = number(&name, &value)?,
            "--bootstrap-iters" => cfg.bootstrap_iters = number(&name, &value)? as usize,
            "--peak-ceiling" => cfg.peak_ceiling = number(&name, &value)?,
            "--aperture-radius" => cfg.aperture_radius = number(&name, &value)?,
            "--annulus-inner" => cfg.annulus_inner = number(&name, &value)?,
            "--annulus-outer" => cfg.annulus_outer = number(&name, &value)?,
            "--clip-sigma" => cfg.clip_sigma = number(&name, &value)?,
            "--clip-iterations" => cfg.clip_iterations = number(&name, &value)? as usize,
            "--kfold" => cfg.kfold = number(&name, &value)? as usize,
            "--block-grid" => cfg.block_grid = number(&name, &value)? as usize,
            "--max-iter" => cfg.max_iter = number(&name, &value)? as usize,
            "--min-pairs" => cfg.min_pairs = number(&name, &value)? as usize,
            "--gain" => cfg.gain = Some(number(&name, &value)?),
            "--target-x" => cfg.target_x = number(&name, &value)?,
            "--target-y" => cfg.target_y = number(&name, &value)?,
            "--seed" => cfg.seed = number(&name, &value)? as u64,
            other => return Err(format!("unknown argument {other}")),
        }
        index += 1;
    }
    if cfg.parity != 1 && cfg.parity != -1 {
        return Err("parity must be 1 or -1".into());
    }
    if !cfg.fits.is_file() {
        return Err(format!("FITS not found: {}", cfg.fits.display()));
    }
    if !cfg.sources.is_file() {
        return Err(format!("source catalog not found: {}", cfg.sources.display()));
    }
    if !cfg.gaia.is_file() {
        return Err(format!("reference catalog not found: {}", cfg.gaia.display()));
    }
    Ok(cfg)
}

// ---------------------------------------------------------------- small math

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut copy = values.to_vec();
    copy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(quantile_sorted(&copy, 0.5))
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn mad_sigma(values: &[f64]) -> Option<f64> {
    let center = median(values)?;
    let deviations: Vec<f64> = values.iter().map(|v| (v - center).abs()).collect();
    median(&deviations).map(|m| 1.4826 * m)
}

fn stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

fn solve_dense(matrix: &mut [Vec<f64>], rhs: &mut [f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    for column in 0..n {
        let mut pivot = column;
        for row in column + 1..n {
            if matrix[row][column].abs() > matrix[pivot][column].abs() {
                pivot = row;
            }
        }
        if matrix[pivot][column].abs() < 1e-15 {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let diagonal = matrix[column][column];
        for k in column..n {
            matrix[column][k] /= diagonal;
        }
        rhs[column] /= diagonal;
        for row in 0..n {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor == 0.0 {
                continue;
            }
            for k in column..n {
                matrix[row][k] -= factor * matrix[column][k];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    Some(rhs.to_vec())
}

fn inverted(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut columns = Vec::with_capacity(n);
    for column in 0..n {
        let mut a: Vec<Vec<f64>> = matrix.to_vec();
        let mut rhs: Vec<f64> = (0..n)
            .map(|i| if i == column { 1.0 } else { 0.0 })
            .collect();
        columns.push(solve_dense(&mut a, &mut rhs)?);
    }
    let mut inverse = vec![vec![0.0; n]; n];
    for (column, solution) in columns.into_iter().enumerate() {
        for row in 0..n {
            inverse[row][column] = solution[row];
        }
    }
    Some(inverse)
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn shuffle(&mut self, values: &mut [usize]) {
        for i in (1..values.len()).rev() {
            let j = (self.next() >> 11) as usize % (i + 1);
            values.swap(i, j);
        }
    }
}

fn utc_stamp() -> (String, String) {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = seconds.div_euclid(86_400);
    let rest = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let iso = format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3600,
        (rest % 3600) / 60,
        rest % 60
    );
    let run_id = format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
        rest / 3600,
        (rest % 3600) / 60,
        rest % 60
    );
    (iso, run_id)
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

fn sha256_file(path: &Path) -> String {
    let mut digest = Sha256::new();
    match std::fs::read(path) {
        Ok(bytes) => digest.update(&bytes),
        Err(_) => digest.update(b"<unreadable>"),
    }
    format!("{:x}", digest.finalize())
}

fn sha256_text(text: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(text.as_bytes());
    format!("{:x}", digest.finalize())
}

fn or_null(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

fn or_null_opt(value: Option<f64>) -> Value {
    match value {
        Some(value) => or_null(value),
        None => Value::Null,
    }
}

fn finite(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<f64>().ok().filter(|v| v.is_finite())
}

// ---------------------------------------------------------------- image

struct Image {
    width: usize,
    height: usize,
    dn: Vec<f64>,
    wrapped: Vec<bool>,
    telemetry_pixels: usize,
}

impl Image {
    fn load(path: &Path) -> Result<Self> {
        let frame = fits::read_fits(path, false)?;
        if frame.data.len() != frame.width * frame.height {
            return Err("FITS pixel buffer does not match its dimensions".into());
        }
        let signed16 = frame.data.dtype() == ">i2";
        let mut dn = frame.data.to_f64();
        let mut wrapped = vec![false; dn.len()];
        if signed16 {
            for (index, value) in dn.iter_mut().enumerate() {
                if *value < 0.0 {
                    wrapped[index] = true;
                    *value += 65536.0;
                }
            }
        }
        // y=0, x=0..103 is the permanent telemetry window of this detector.
        let mut telemetry_pixels = 0;
        for x in 0..frame.width.min(104) {
            telemetry_pixels += 1;
            dn[x] = f64::NAN;
        }
        Ok(Self {
            width: frame.width,
            height: frame.height,
            dn,
            wrapped,
            telemetry_pixels,
        })
    }

    fn value(&self, ix: usize, iy: usize) -> Option<f64> {
        let value = self.dn[ix + iy * self.width];
        value.is_finite().then_some(value)
    }

    fn wrapped_at(&self, ix: usize, iy: usize) -> bool {
        self.wrapped[ix + iy * self.width]
    }
}

fn pixel_bounds(center: f64, radius: f64, limit: usize) -> Option<(usize, usize)> {
    let low = (center - radius - 1.0).floor().max(0.0) as usize;
    let high = ((center + radius + 1.0).ceil() as i64).min(limit as i64 - 1);
    if high < 0 || low > high as usize {
        return None;
    }
    Some((low, high as usize))
}

fn aperture_coverage(dx: f64, dy: f64, radius: f64) -> f64 {
    let distance = dx.hypot(dy);
    if distance <= radius - 0.9 {
        return 1.0;
    }
    if distance >= radius + 0.9 {
        return 0.0;
    }
    let samples = 4;
    let mut inside = 0;
    for sx in 0..samples {
        for sy in 0..samples {
            let ox = -0.5 + (sx as f64 + 0.5) / samples as f64;
            let oy = -0.5 + (sy as f64 + 0.5) / samples as f64;
            if (dx + ox).hypot(dy + oy) <= radius {
                inside += 1;
            }
        }
    }
    inside as f64 / (samples * samples) as f64
}

fn clipped_stats(values: &[f64], sigma: f64, iterations: usize) -> Option<(f64, f64, usize)> {
    if values.len() < 8 {
        return None;
    }
    let mut active: Vec<f64> = values.to_vec();
    let mut center = median(&active)?;
    let mut spread = mad_sigma(&active).unwrap_or(0.0);
    for _ in 0..iterations {
        if spread <= 0.0 {
            break;
        }
        let kept: Vec<f64> = active
            .iter()
            .copied()
            .filter(|v| ((v - center) / spread).abs() <= sigma)
            .collect();
        if kept.len() < 8 || kept.len() == active.len() {
            active = kept;
            break;
        }
        active = kept;
        center = median(&active)?;
        spread = mad_sigma(&active).unwrap_or(0.0);
    }
    Some((center, stddev(&active), active.len()))
}

struct Photometry {
    flux: f64,
    flux_error: f64,
    background: f64,
    noise: f64,
    aperture_pixels: f64,
    peak: f64,
    sat_negative: bool,
    sat_peak_ceiling: bool,
    sat_zero_clip: bool,
    annulus_pixels: usize,
}

impl Photometry {
    fn saturated(&self) -> bool {
        self.sat_negative || self.sat_peak_ceiling || self.sat_zero_clip
    }
}

fn measure(image: &Image, x: f64, y: f64, cfg: &Config) -> Option<Photometry> {
    let window = cfg.annulus_outer + 1.0;
    let (x0, x1) = pixel_bounds(x, window, image.width)?;
    let (y0, y1) = pixel_bounds(y, window, image.height)?;
    let mut annulus = Vec::new();
    let mut sat_negative = false;
    let mut peak: f64 = f64::NEG_INFINITY;
    for iy in y0..=y1 {
        for ix in x0..=x1 {
            let Some(value) = image.value(ix, iy) else {
                continue;
            };
            let dx = ix as f64 + 1.0 - x;
            let dy = iy as f64 + 1.0 - y;
            let distance = dx.hypot(dy);
            if distance <= cfg.annulus_outer + 1.0 && image.wrapped_at(ix, iy) {
                sat_negative = true;
            }
            if distance <= cfg.aperture_radius + 1.0 && value > peak {
                peak = value;
            }
            if distance >= cfg.annulus_inner && distance <= cfg.annulus_outer {
                annulus.push(value);
            }
        }
    }
    let (background, noise, annulus_pixels) =
        clipped_stats(&annulus, cfg.clip_sigma, cfg.clip_iterations)?;
    let mut flux = 0.0;
    let mut aperture_pixels = 0.0;
    let mut sat_zero_clip = false;
    for iy in y0..=y1 {
        for ix in x0..=x1 {
            let Some(value) = image.value(ix, iy) else {
                continue;
            };
            let dx = ix as f64 + 1.0 - x;
            let dy = iy as f64 + 1.0 - y;
            let coverage = aperture_coverage(dx, dy, cfg.aperture_radius);
            if coverage <= 0.0 {
                continue;
            }
            if image.wrapped_at(ix, iy) {
                sat_negative = true;
            }
            if coverage > 0.5 && value == 0.0 {
                sat_zero_clip = true;
            }
            flux += coverage * (value - background);
            aperture_pixels += coverage;
            if value > peak {
                peak = value;
            }
        }
    }
    if aperture_pixels < 1.0 || !noise.is_finite() || noise <= 0.0 {
        return None;
    }
    let mut variance = aperture_pixels * noise * noise;
    if let Some(gain) = cfg.gain {
        if gain > 0.0 {
            variance += flux.max(0.0) / gain;
        }
    }
    Some(Photometry {
        flux,
        flux_error: variance.sqrt(),
        background,
        noise,
        aperture_pixels,
        peak,
        sat_negative,
        sat_peak_ceiling: peak.is_finite() && peak >= cfg.peak_ceiling,
        sat_zero_clip,
        annulus_pixels,
    })
}

// ---------------------------------------------------------------- catalogs

struct Detection {
    id: String,
    x: f64,
    y: f64,
    flux: f64,
    flux_error: f64,
    noise: f64,
    snr: f64,
    peak: f64,
    quality_passed: bool,
    quality_class: String,
}

fn column_index(headers: &csv::StringRecord, name: &str) -> Option<usize> {
    headers.iter().position(|h| h.trim() == name)
}

fn load_detections(path: &Path, include_rejected: bool) -> Result<(Vec<Detection>, usize, usize)> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let headers = reader.headers().map_err(|e| e.to_string())?.clone();
    let ix_x = column_index(&headers, "x").ok_or("source catalog lacks column x")?;
    let ix_y = column_index(&headers, "y").ok_or("source catalog lacks column y")?;
    let ix_flux = column_index(&headers, "flux").ok_or("source catalog lacks column flux")?;
    let ix_flux_error = column_index(&headers, "flux_error");
    let ix_noise = column_index(&headers, "noise");
    let ix_snr = column_index(&headers, "snr");
    let ix_peak = column_index(&headers, "peak");
    let ix_quality = column_index(&headers, "quality_passed");
    let ix_class = column_index(&headers, "quality_class");
    let ix_id = column_index(&headers, "detection_id");
    let mut detections = Vec::new();
    let mut rows = 0;
    let mut rejected = 0;
    for record in reader.records() {
        let record = record.map_err(|e| e.to_string())?;
        rows += 1;
        let quality = ix_quality
            .and_then(|i| record.get(i))
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "true" | "t" | "1" | "yes"))
            .unwrap_or(false);
        if !quality {
            rejected += 1;
            if !include_rejected {
                continue;
            }
        }
        let get = |index: Option<usize>| -> Option<f64> {
            index.and_then(|i| record.get(i)).and_then(finite)
        };
        let (Some(x), Some(y), Some(flux)) = (
            record.get(ix_x).and_then(finite),
            record.get(ix_y).and_then(finite),
            record.get(ix_flux).and_then(finite),
        ) else {
            continue;
        };
        detections.push(Detection {
            id: ix_id.and_then(|i| record.get(i)).unwrap_or("unknown").to_owned(),
            x,
            y,
            flux,
            flux_error: get(ix_flux_error).unwrap_or(f64::NAN),
            noise: get(ix_noise).unwrap_or(f64::NAN),
            snr: get(ix_snr).unwrap_or(f64::NAN),
            peak: get(ix_peak).unwrap_or(f64::NAN),
            quality_passed: quality,
            quality_class: ix_class.and_then(|i| record.get(i)).unwrap_or("").to_owned(),
        });
    }
    Ok((detections, rows, rejected))
}

struct Reference {
    source_id: String,
    propagated_ra_deg: f64,
    propagated_dec_deg: f64,
    magnitude: f64,
    magnitude_error: f64,
    color: Option<f64>,
    pm_ra: f64,
    pm_dec: f64,
    ref_epoch: f64,
}

struct ReferenceCatalog {
    stars: Vec<Reference>,
    tangent: Vec<(f64, f64)>,
    rows: usize,
}

fn load_references(path: &Path, cfg: &Config) -> Result<ReferenceCatalog> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let headers = reader.headers().map_err(|e| e.to_string())?.clone();
    let ix_id = column_index(&headers, "source_id").ok_or("reference catalog lacks source_id")?;
    let ix_ra = column_index(&headers, "ra_deg")
        .or_else(|| column_index(&headers, "ra"))
        .ok_or("reference catalog lacks ra_deg")?;
    let ix_dec = column_index(&headers, "dec_deg")
        .or_else(|| column_index(&headers, "dec"))
        .ok_or("reference catalog lacks dec_deg")?;
    let ix_mag = column_index(&headers, "phot_g_mean_mag")
        .or_else(|| column_index(&headers, "magnitude"))
        .ok_or("reference catalog lacks a magnitude column")?;
    let ix_mag_err = column_index(&headers, "magnitude_error");
    let ix_color = column_index(&headers, "color").or_else(|| column_index(&headers, "bp_rp"));
    let ix_pmra = column_index(&headers, "pmra_mas_yr").or_else(|| column_index(&headers, "pmra"));
    let ix_pmdec =
        column_index(&headers, "pmdec_mas_yr").or_else(|| column_index(&headers, "pmdec"));
    let ix_epoch = column_index(&headers, "ref_epoch");
    let mut stars = Vec::new();
    let mut rows = 0;
    for record in reader.records() {
        let record = record.map_err(|e| e.to_string())?;
        rows += 1;
        let get = |index: Option<usize>| -> Option<f64> {
            index.and_then(|i| record.get(i)).and_then(finite)
        };
        let (Some(ra), Some(dec), Some(magnitude)) = (
            record.get(ix_ra).and_then(finite),
            record.get(ix_dec).and_then(finite),
            record.get(ix_mag).and_then(finite),
        ) else {
            continue;
        };
        let pm_ra = get(ix_pmra).unwrap_or(0.0);
        let pm_dec = get(ix_pmdec).unwrap_or(0.0);
        let ref_epoch = get(ix_epoch).unwrap_or(2016.0);
        let years = cfg.epoch - ref_epoch;
        let cos_dec = dec.to_radians().cos().abs().max(1e-8);
        let propagated_ra_deg = (ra + pm_ra * years / cos_dec / 3_600_000.0).rem_euclid(360.0);
        let propagated_dec_deg = dec + pm_dec * years / 3_600_000.0;
        stars.push(Reference {
            source_id: record.get(ix_id).unwrap_or("unknown").to_owned(),
            propagated_ra_deg,
            propagated_dec_deg,
            magnitude,
            magnitude_error: get(ix_mag_err).unwrap_or(0.0).max(0.0),
            color: get(ix_color),
            pm_ra,
            pm_dec,
            ref_epoch,
        });
    }
    let tangent: Vec<(f64, f64)> = stars
        .iter()
        .map(|s| {
            gnomonic(
                cfg.center_ra,
                cfg.center_dec,
                s.propagated_ra_deg,
                s.propagated_dec_deg,
            )
        })
        .collect();
    Ok(ReferenceCatalog {
        stars,
        tangent,
        rows,
    })
}

fn gnomonic(center_ra: f64, center_dec: f64, ra: f64, dec: f64) -> (f64, f64) {
    let ra0 = center_ra.to_radians();
    let dec0 = center_dec.to_radians();
    let dec = dec.to_radians();
    let delta =
        (ra.to_radians() - ra0 + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
    let denominator = dec0.sin() * dec.sin() + dec0.cos() * dec.cos() * delta.cos();
    if denominator <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    (
        (dec.cos() * delta.sin() / denominator).to_degrees() * 3600.0,
        ((dec0.cos() * dec.sin() - dec0.sin() * dec.cos() * delta.cos()) / denominator).to_degrees()
            * 3600.0,
    )
}

// ---------------------------------------------------------------- geometry

#[derive(Clone)]
struct Affine {
    ax: Vec<f64>,
    ay: Vec<f64>,
    terms: usize,
}

impl Affine {
    fn basis(&self, xi: f64, eta: f64) -> Vec<f64> {
        if self.terms == 3 {
            vec![1.0, xi, eta]
        } else {
            vec![1.0, xi, eta, xi * xi, xi * eta, eta * eta]
        }
    }
    fn apply(&self, xi: f64, eta: f64) -> (f64, f64) {
        let basis = self.basis(xi, eta);
        (
            self.ax.iter().zip(basis.iter()).map(|(p, q)| p * q).sum(),
            self.ay.iter().zip(basis.iter()).map(|(p, q)| p * q).sum(),
        )
    }
    fn delta(&self, other: &Affine) -> f64 {
        (0..self.terms.min(other.terms))
            .map(|i| (self.ax[i] - other.ax[i]).abs() + (self.ay[i] - other.ay[i]).abs())
            .fold(0.0, f64::max)
    }
    fn matrix(&self) -> [[f64; 2]; 2] {
        [[self.ax[1], self.ax[2]], [self.ay[1], self.ay[2]]]
    }
}

fn seed_affine(cfg: &Config) -> Affine {
    let theta = cfg.rotation.to_radians();
    let parity = cfg.parity as f64;
    Affine {
        ax: vec![
            cfg.crpix_x,
            parity * theta.cos() / cfg.scale,
            -parity * theta.sin() / cfg.scale,
        ],
        ay: vec![
            cfg.crpix_y,
            theta.sin() / cfg.scale,
            theta.cos() / cfg.scale,
        ],
        terms: 3,
    }
}

struct Grid {
    x0: f64,
    y0: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<u32>>,
}

impl Grid {
    fn new(points: &[(f64, f64)], cell: f64) -> Self {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for (x, y) in points {
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            min_x = min_x.min(*x);
            min_y = min_y.min(*y);
            max_x = max_x.max(*x);
            max_y = max_y.max(*y);
        }
        if !min_x.is_finite() {
            min_x = 0.0;
            min_y = 0.0;
            max_x = 0.0;
            max_y = 0.0;
        }
        let nx = (((max_x - min_x) / cell).floor() as usize) + 1;
        let ny = (((max_y - min_y) / cell).floor() as usize) + 1;
        let mut buckets = vec![Vec::new(); nx * ny];
        for (index, (x, y)) in points.iter().enumerate() {
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            let cx = (((x - min_x) / cell).floor() as usize).min(nx - 1);
            let cy = (((y - min_y) / cell).floor() as usize).min(ny - 1);
            buckets[cx + cy * nx].push(index as u32);
        }
        Self {
            x0: min_x,
            y0: min_y,
            cell,
            nx,
            ny,
            buckets,
        }
    }

    fn nearest(&self, points: &[(f64, f64)], x: f64, y: f64, radius: f64) -> Option<(f64, u32)> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let low_x = (((x - radius - self.x0) / self.cell).floor().max(0.0) as usize).min(self.nx - 1);
        let low_y = (((y - radius - self.y0) / self.cell).floor().max(0.0) as usize).min(self.ny - 1);
        let high_x = (((x + radius - self.x0) / self.cell).floor().max(0.0) as usize).min(self.nx - 1);
        let high_y = (((y + radius - self.y0) / self.cell).floor().max(0.0) as usize).min(self.ny - 1);
        let mut best: Option<(f64, u32)> = None;
        for cy in low_y..=high_y {
            for cx in low_x..=high_x {
                for index in &self.buckets[cx + cy * self.nx] {
                    let (px, py) = points[*index as usize];
                    let distance = (px - x).hypot(py - y);
                    if distance <= radius && best.is_none_or(|(d, _)| distance < d) {
                        best = Some((distance, *index));
                    }
                }
            }
        }
        best
    }
}

struct MatchedPair {
    reference: usize,
    detection: usize,
}

fn mutual_match(
    references: &[(f64, f64)],
    detections: &[(f64, f64)],
    radius: f64,
) -> Vec<MatchedPair> {
    let detection_grid = Grid::new(detections, 8.0);
    let reference_grid = Grid::new(references, 8.0);
    let forward: Vec<Option<(f64, u32)>> = references
        .iter()
        .map(|(x, y)| detection_grid.nearest(detections, *x, *y, radius))
        .collect();
    let mut reverse: Vec<Option<(f64, u32)>> = vec![None; detections.len()];
    for entry in forward.iter().flatten() {
        let detection = entry.1 as usize;
        if reverse[detection].is_none() {
            reverse[detection] = reference_grid.nearest(
                references,
                detections[detection].0,
                detections[detection].1,
                radius,
            );
        }
    }
    let mut matches = Vec::new();
    for (index, entry) in forward.iter().enumerate() {
        let Some((_, detection)) = entry else {
            continue;
        };
        let detection = *detection as usize;
        let Some((_, back)) = reverse[detection] else {
            continue;
        };
        if back as usize != index {
            continue;
        }
        matches.push(MatchedPair {
            reference: index,
            detection,
        });
    }
    matches
}

fn fit_affine(
    tangent: &[(f64, f64)],
    detections: &[(f64, f64)],
    matches: &[MatchedPair],
    terms: usize,
) -> Option<Affine> {
    if matches.len() < terms + 4 {
        return None;
    }
    let mut normal = vec![vec![0.0; terms]; terms];
    let mut rhs_x = vec![0.0; terms];
    let mut rhs_y = vec![0.0; terms];
    for entry in matches {
        let (xi, eta) = tangent[entry.reference];
        if !xi.is_finite() || !eta.is_finite() {
            continue;
        }
        let basis: Vec<f64> = if terms == 3 {
            vec![1.0, xi, eta]
        } else {
            vec![1.0, xi, eta, xi * xi, xi * eta, eta * eta]
        };
        let (x, y) = detections[entry.detection];
        for i in 0..terms {
            for j in 0..terms {
                normal[i][j] += basis[i] * basis[j];
            }
            rhs_x[i] += basis[i] * x;
            rhs_y[i] += basis[i] * y;
        }
    }
    let mut a = normal.clone();
    let mut b = rhs_x;
    let ax = solve_dense(&mut a, &mut b)?;
    let mut a = normal;
    let mut b = rhs_y;
    let ay = solve_dense(&mut a, &mut b)?;
    Some(Affine { ax, ay, terms })
}

fn robust_affine(
    tangent: &[(f64, f64)],
    detections: &[(f64, f64)],
    matches: &[MatchedPair],
    terms: usize,
) -> Option<Affine> {
    let mut active: Vec<usize> = (0..matches.len()).collect();
    let mut affine = fit_affine(tangent, detections, matches, terms)?;
    for _ in 0..6 {
        let mut distances = Vec::with_capacity(active.len());
        for index in &active {
            let entry = &matches[*index];
            let (xi, eta) = tangent[entry.reference];
            let (px, py) = affine.apply(xi, eta);
            let (x, y) = detections[entry.detection];
            distances.push((x - px).hypot(y - py));
        }
        let sigma = mad_sigma(&distances).filter(|v| *v > 0.0).unwrap_or(0.05);
        let window = (3.0 * sigma).clamp(0.08, 3.0);
        let kept: Vec<usize> = active
            .iter()
            .copied()
            .zip(distances.iter())
            .filter(|(_, d)| **d <= window)
            .map(|(index, _)| index)
            .collect();
        if kept.len() < terms + 4 || kept.len() == active.len() {
            break;
        }
        let subset: Vec<MatchedPair> = kept
            .iter()
            .map(|index| MatchedPair {
                reference: matches[*index].reference,
                detection: matches[*index].detection,
            })
            .collect();
        match fit_affine(tangent, detections, &subset, terms) {
            Some(next) => {
                affine = next;
                active = kept;
            }
            None => break,
        }
    }
    Some(affine)
}

// ---------------------------------------------------------------- calibration

struct CalibrationSample {
    magnitude_instrumental: f64,
    sigma_instrumental: f64,
    color: Option<f64>,
    magnitude_reference: f64,
    sigma_reference: f64,
    weight: f64,
    x: f64,
    y: f64,
    radius: f64,
    block: usize,
}

#[derive(Clone)]
struct CalibrationFit {
    zero_point: f64,
    color_term: f64,
    zero_point_error: f64,
    color_term_error: f64,
    covariance: [[f64; 2]; 2],
    n: usize,
    chi_square: f64,
}

fn fit_calibration(samples: &[&CalibrationSample], order: usize) -> Option<CalibrationFit> {
    let parameters = order + 1;
    let mut normal = vec![vec![0.0; parameters]; parameters];
    let mut rhs = vec![0.0; parameters];
    for sample in samples {
        if parameters > 1 && sample.color.is_none() {
            return None;
        }
        let basis = [1.0, sample.color.unwrap_or(0.0)];
        for i in 0..parameters {
            for j in 0..parameters {
                normal[i][j] += sample.weight * basis[i] * basis[j];
            }
            rhs[i] +=
                sample.weight * basis[i] * (sample.magnitude_reference - sample.magnitude_instrumental);
        }
    }
    let mut a = normal.clone();
    let mut b = rhs;
    let solution = solve_dense(&mut a, &mut b)?;
    let mut chi_square = 0.0;
    for sample in samples {
        let residual = sample.magnitude_reference - predict_solution(&solution, sample);
        chi_square += sample.weight * residual * residual;
    }
    let inverse = inverted(&normal)?;
    Some(CalibrationFit {
        zero_point: solution[0],
        color_term: if parameters > 1 { solution[1] } else { 0.0 },
        zero_point_error: inverse[0][0].max(0.0).sqrt(),
        color_term_error: if parameters > 1 {
            inverse[1][1].max(0.0).sqrt()
        } else {
            f64::NAN
        },
        covariance: [
            [inverse[0][0], if parameters > 1 { inverse[0][1] } else { 0.0 }],
            [
                if parameters > 1 { inverse[1][0] } else { 0.0 },
                if parameters > 1 { inverse[1][1] } else { 0.0 },
            ],
        ],
        n: samples.len(),
        chi_square,
    })
}

fn predict_solution(solution: &[f64], sample: &CalibrationSample) -> f64 {
    sample.magnitude_instrumental
        + solution[0]
        + solution.get(1).copied().unwrap_or(0.0) * sample.color.unwrap_or(0.0)
}

fn predict(fit: &CalibrationFit, sample: &CalibrationSample) -> f64 {
    sample.magnitude_instrumental + fit.zero_point + fit.color_term * sample.color.unwrap_or(0.0)
}

struct RobustFit {
    fit: CalibrationFit,
    inliers: Vec<bool>,
    scatter: f64,
    center: f64,
    sigma: f64,
    kept: usize,
    clipped: usize,
}

fn mode_estimate(values: &[f64], bin: f64) -> Option<f64> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    let low = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let high = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut best_center = low;
    let mut best_count = 0_usize;
    let mut edge = low;
    while edge <= high {
        let count = finite
            .iter()
            .filter(|v| **v >= edge && **v < edge + bin)
            .count();
        if count > best_count {
            best_count = count;
            best_center = edge + bin * 0.5;
        }
        edge += bin;
    }
    let near: Vec<f64> = finite
        .iter()
        .copied()
        .filter(|v| (v - best_center).abs() <= bin * 3.0)
        .collect();
    median(&near)
}

fn robust_calibration(
    samples: &[CalibrationSample],
    order: usize,
    clip_sigma: f64,
) -> Option<RobustFit> {
    if order > 0 && samples.iter().any(|s| s.color.is_none()) {
        return None;
    }
    if samples.len() < order + 8 {
        return None;
    }
    let offsets: Vec<f64> = samples
        .iter()
        .map(|s| s.magnitude_reference - s.magnitude_instrumental)
        .collect();
    let center = mode_estimate(&offsets, 0.1)?;
    let seed_window = if order > 0 { 1.2 } else { 0.5 };
    let mut active: Vec<usize> = (0..samples.len())
        .filter(|i| (offsets[*i] - center).abs() <= seed_window)
        .collect();
    if active.len() < order + 8 {
        active = (0..samples.len()).collect();
    }
    for _ in 0..25 {
        let selection: Vec<&CalibrationSample> = active.iter().map(|i| &samples[*i]).collect();
        let Some(fit) = fit_calibration(&selection, order) else {
            break;
        };
        let residuals: Vec<f64> = selection
            .iter()
            .map(|sample| sample.magnitude_reference - predict(&fit, sample))
            .collect();
        let sigma = mad_sigma(&residuals).unwrap_or(0.0);
        let center = median(&residuals).unwrap_or(0.0);
        let window = (clip_sigma * sigma).max(0.03);
        let kept: Vec<usize> = active
            .iter()
            .copied()
            .zip(residuals.iter())
            .filter(|(_, r)| (*r - center).abs() <= window)
            .map(|(index, _)| index)
            .collect();
        if kept.len() < order + 8 {
            break;
        }
        if kept.len() == active.len() {
            active = kept;
            break;
        }
        active = kept;
    }
    let selection: Vec<&CalibrationSample> = active.iter().map(|i| &samples[*i]).collect();
    let fit = fit_calibration(&selection, order)?;
    let residuals: Vec<f64> = selection
        .iter()
        .map(|sample| sample.magnitude_reference - predict(&fit, sample))
        .collect();
    let scatter =
        (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len().max(1) as f64).sqrt();
    let mut inliers = vec![false; samples.len()];
    for index in &active {
        inliers[*index] = true;
    }
    Some(RobustFit {
        fit,
        inliers,
        scatter,
        center: median(&residuals).unwrap_or(0.0),
        sigma: mad_sigma(&residuals).unwrap_or(0.0),
        kept: active.len(),
        clipped: samples.len() - active.len(),
    })
}

fn clone_samples(samples: &[CalibrationSample]) -> Vec<CalibrationSample> {
    samples
        .iter()
        .map(|s| CalibrationSample {
            magnitude_instrumental: s.magnitude_instrumental,
            sigma_instrumental: s.sigma_instrumental,
            color: s.color,
            magnitude_reference: s.magnitude_reference,
            sigma_reference: s.sigma_reference,
            weight: s.weight,
            x: s.x,
            y: s.y,
            radius: s.radius,
            block: s.block,
        })
        .collect()
}

struct CrossValidation {
    sigma_out: f64,
    sigma_out_normalized: f64,
    sigma_out_all: f64,
    fold_sigma: Vec<f64>,
    n_holdout: usize,
    n_scored_all: usize,
}

fn cross_validate(samples: &[CalibrationSample], order: usize, cfg: &Config) -> Option<CrossValidation> {
    let n = samples.len();
    if n < cfg.kfold + order + 7 {
        return None;
    }
    let folds = cfg.kfold.max(2).min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    Lcg(cfg.seed).shuffle(&mut indices);
    let mut assignment = vec![0usize; n];
    for (position, index) in indices.iter().enumerate() {
        assignment[*index] = position % folds;
    }
    let mut fold_sigma = Vec::new();
    let mut total_weight = 0.0;
    let mut total_chi = 0.0;
    let mut total_square = 0.0;
    let mut total_all_square = 0.0;
    let mut count = 0;
    let mut count_all = 0;
    for fold in 0..folds {
        let train: Vec<CalibrationSample> = (0..n)
            .filter(|i| assignment[*i] != fold)
            .map(|i| samples[i].clone())
            .collect();
        let Some(robust) = robust_calibration(&train, order, cfg.clip_sigma) else {
            continue;
        };
        let window = (3.0 * robust.sigma.max(0.03)).max(0.10);
        let mut residuals = Vec::new();
        for index in 0..n {
            if assignment[index] != fold {
                continue;
            }
            let residual = samples[index].magnitude_reference - predict(&robust.fit, &samples[index]);
            total_all_square += residual * residual;
            count_all += 1;
            if (residual - robust.center).abs() > window {
                continue;
            }
            residuals.push(residual);
            total_weight += samples[index].weight;
            total_chi += samples[index].weight * residual * residual;
            total_square += residual * residual;
            count += 1;
        }
        if !residuals.is_empty() {
            fold_sigma.push(stddev(&residuals));
        }
    }
    if count == 0 {
        return None;
    }
    Some(CrossValidation {
        sigma_out: (total_square / count as f64).sqrt(),
        sigma_out_normalized: (total_chi / total_weight.max(1e-12)).sqrt(),
        sigma_out_all: (total_all_square / count_all.max(1) as f64).sqrt(),
        fold_sigma,
        n_holdout: count,
        n_scored_all: count_all,
    })
}

fn leave_one_block_out(
    samples: &[CalibrationSample],
    order: usize,
    cfg: &Config,
) -> Vec<(usize, usize, f64, f64)> {
    let blocks: std::collections::BTreeSet<usize> = samples.iter().map(|s| s.block).collect();
    let mut output = Vec::new();
    for block in blocks {
        let train: Vec<CalibrationSample> = samples
            .iter()
            .filter(|s| s.block != block)
            .cloned()
            .collect();
        let Some(robust) = robust_calibration(&train, order, cfg.clip_sigma) else {
            continue;
        };
        let residuals: Vec<f64> = samples
            .iter()
            .filter(|s| s.block == block)
            .map(|s| s.magnitude_reference - predict(&robust.fit, s))
            .collect();
        if residuals.is_empty() {
            continue;
        }
        let offset = median(&residuals).unwrap_or(f64::NAN);
        output.push((block, residuals.len(), offset, stddev(&residuals)));
    }
    output
}

impl Clone for CalibrationSample {
    fn clone(&self) -> Self {
        CalibrationSample {
            magnitude_instrumental: self.magnitude_instrumental,
            sigma_instrumental: self.sigma_instrumental,
            color: self.color,
            magnitude_reference: self.magnitude_reference,
            sigma_reference: self.sigma_reference,
            weight: self.weight,
            x: self.x,
            y: self.y,
            radius: self.radius,
            block: self.block,
        }
    }
}

fn binned_residuals(
    samples: &[&CalibrationSample],
    residuals: &[f64],
    edges: &[f64],
    selector: impl Fn(&CalibrationSample) -> Option<f64>,
) -> Vec<Value> {
    let mut output = Vec::new();
    for window in edges.windows(2) {
        let (low, high) = (window[0], window[1]);
        let mut values = Vec::new();
        for (sample, residual) in samples.iter().zip(residuals.iter()) {
            let Some(value) = selector(sample) else {
                continue;
            };
            if value >= low && value < high {
                values.push(*residual);
            }
        }
        if values.is_empty() {
            continue;
        }
        output.push(json!({
            "low": low,
            "high": high,
            "n": values.len(),
            "median_residual_mag": or_null(median(&values).unwrap_or(f64::NAN)),
            "mad_sigma_mag": or_null(mad_sigma(&values).unwrap_or(f64::NAN)),
        }));
    }
    output
}

fn binned_offsets(
    samples: &[&CalibrationSample],
    edges: &[f64],
    selector: impl Fn(&CalibrationSample) -> Option<f64>,
) -> Vec<Value> {
    let mut output = Vec::new();
    for window in edges.windows(2) {
        let (low, high) = (window[0], window[1]);
        let mut values = Vec::new();
        for sample in samples {
            let Some(value) = selector(sample) else {
                continue;
            };
            if value >= low && value < high {
                values.push(sample.magnitude_reference - sample.magnitude_instrumental);
            }
        }
        if values.is_empty() {
            continue;
        }
        output.push(json!({
            "low": low,
            "high": high,
            "n": values.len(),
            "median_zero_point": or_null(median(&values).unwrap_or(f64::NAN)),
            "mad_sigma_mag": or_null(mad_sigma(&values).unwrap_or(f64::NAN)),
        }));
    }
    output
}

fn polynomial_solution(samples: &[(f64, f64, f64, f64)], terms: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut normal = vec![vec![0.0; terms]; terms];
    let mut rhs_x = vec![0.0; terms];
    let mut rhs_y = vec![0.0; terms];
    for (xi, eta, x, y) in samples {
        let basis: Vec<f64> = match terms {
            3 => vec![1.0, *xi, *eta],
            _ => vec![1.0, *xi, *eta, xi * xi, xi * eta, eta * eta],
        };
        for i in 0..terms {
            for j in 0..terms {
                normal[i][j] += basis[i] * basis[j];
            }
            rhs_x[i] += basis[i] * x;
            rhs_y[i] += basis[i] * y;
        }
    }
    let mut a = normal.clone();
    let mut b = rhs_x;
    let px = solve_dense(&mut a, &mut b)?;
    let mut a = normal;
    let mut b = rhs_y;
    let py = solve_dense(&mut a, &mut b)?;
    Some((px, py))
}

fn polynomial_rms(samples: &[(f64, f64, f64, f64)], px: &[f64], py: &[f64]) -> f64 {
    let terms = px.len();
    let mut square = 0.0;
    for (xi, eta, x, y) in samples {
        let basis: Vec<f64> = match terms {
            3 => vec![1.0, *xi, *eta],
            _ => vec![1.0, *xi, *eta, xi * xi, xi * eta, eta * eta],
        };
        let dx: f64 = basis.iter().zip(px.iter()).map(|(b, p)| b * p).sum();
        let dy: f64 = basis.iter().zip(py.iter()).map(|(b, p)| b * p).sum();
        square += (x - dx).powi(2) + (y - dy).powi(2);
    }
    (square / samples.len().max(1) as f64).sqrt()
}

fn block_of(width: usize, height: usize, x: f64, y: f64, grid: usize) -> usize {
    let g = grid.max(1);
    let bx = ((((x - 1.0) / width as f64) * g as f64).floor().max(0.0) as usize).min(g - 1);
    let by = ((((y - 1.0) / height as f64) * g as f64).floor().max(0.0) as usize).min(g - 1);
    by * g + bx
}

fn block_label(block: usize, grid: usize) -> (usize, usize) {
    (block % grid.max(1), block / grid.max(1))
}

fn csv_opt(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.6}"),
        _ => String::new(),
    }
}

struct TargetRow {
    detection_id: String,
    x: f64,
    y: f64,
    flux: f64,
    flux_error: f64,
    noise: f64,
    snr: f64,
    peak: f64,
    quality_passed: bool,
    quality_class: String,
}

fn find_target(path: &Path, x: f64, y: f64) -> Option<TargetRow> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_path(path).ok()?;
    let headers = reader.headers().ok()?.clone();
    let ix_x = column_index(&headers, "x")?;
    let ix_y = column_index(&headers, "y")?;
    let ix_flux = column_index(&headers, "flux");
    let ix_flux_error = column_index(&headers, "flux_error");
    let ix_noise = column_index(&headers, "noise");
    let ix_snr = column_index(&headers, "snr");
    let ix_peak = column_index(&headers, "peak");
    let ix_quality = column_index(&headers, "quality_passed");
    let ix_class = column_index(&headers, "quality_class");
    let ix_id = column_index(&headers, "detection_id");
    let mut best: Option<(f64, TargetRow)> = None;
    for record in reader.records().flatten() {
        let (Some(px), Some(py)) = (
            record.get(ix_x).and_then(finite),
            record.get(ix_y).and_then(finite),
        ) else {
            continue;
        };
        let distance = (px - x).hypot(py - y);
        if best.as_ref().is_some_and(|(d, _)| *d <= distance) {
            continue;
        }
        let get = |index: Option<usize>| -> f64 {
            index
                .and_then(|i| record.get(i))
                .and_then(finite)
                .unwrap_or(f64::NAN)
        };
        best = Some((
            distance,
            TargetRow {
                detection_id: ix_id.and_then(|i| record.get(i)).unwrap_or("").to_owned(),
                x: px,
                y: py,
                flux: get(ix_flux),
                flux_error: get(ix_flux_error),
                noise: get(ix_noise),
                snr: get(ix_snr),
                peak: get(ix_peak),
                quality_passed: ix_quality
                    .and_then(|i| record.get(i))
                    .map(|v| matches!(v.trim().to_lowercase().as_str(), "true" | "1" | "t" | "yes"))
                    .unwrap_or(false),
                quality_class: ix_class.and_then(|i| record.get(i)).unwrap_or("").to_owned(),
            },
        ));
    }
    best.map(|(distance, row)| {
        let mut row = row;
        if distance > 2.0 {
            row.quality_class = format!("{} NEAREST_TO_QUERY", row.quality_class);
        }
        row
    })
}

// ---------------------------------------------------------------- run

struct PairRow {
    reference: usize,
    detection: usize,
    resid_x: f64,
    resid_y: f64,
    resid_px: f64,
    photometry: Photometry,
    magnitude_instrumental: f64,
    sigma_instrumental: f64,
    radius: f64,
    block: usize,
    catalog_flux: f64,
    catalog_flux_error: f64,
    catalog_snr: f64,
    catalog_noise: f64,
    quality_passed: bool,
    quality_class: String,
    detection_id: String,
    inlier: bool,
    g_emp: f64,
    g_emp_order0: f64,
    resid_mag: f64,
}

fn run(args: &[String]) -> Result<()> {
    let started = SystemTime::now();
    let cfg = parse_args(args)?;
    let (utc, run_id) = utc_stamp();
    let out_dir = match &cfg.out {
        Some(path) => path.clone(),
        None => PathBuf::from("tmp/rust-pairs").join(&run_id),
    };
    std::fs::create_dir_all(&out_dir)
        .map_err(|error| format!("cannot create {}: {error}", out_dir.display()))?;

    eprintln!("[1/6] frame {}", cfg.fits.display());
    let image = Image::load(&cfg.fits)?;
    let mut negative_pixels = 0_usize;
    let mut zero_pixels = 0_usize;
    for (index, value) in image.dn.iter().enumerate() {
        if image.wrapped[index] {
            negative_pixels += 1;
        } else if *value == 0.0 {
            zero_pixels += 1;
        }
    }

    eprintln!("[2/6] sources {}", cfg.sources.display());
    let (detections, source_rows, rejected_rows) = load_detections(&cfg.sources, cfg.include_rejected)?;
    if detections.len() < cfg.min_pairs {
        return Err(format!(
            "only {} usable detections, need at least {}",
            detections.len(),
            cfg.min_pairs
        ));
    }
    let detection_pixels: Vec<(f64, f64)> = detections.iter().map(|d| (d.x, d.y)).collect();

    eprintln!("[3/6] references {}", cfg.gaia.display());
    let references = load_references(&cfg.gaia, &cfg)?;

    let mut affine = seed_affine(&cfg);
    let mut matches = Vec::new();
    let mut iterations = 0_usize;
    let mut radius_used = cfg.match_radius;
    let mut schedule: Vec<(usize, f64)> = Vec::new();
    for _ in 0..cfg.bootstrap_iters.max(1) {
        schedule.push((3, cfg.bootstrap_radius.max(cfg.match_radius)));
    }
    let intermediate = (cfg.match_radius * 1.7).max(cfg.match_radius);
    schedule.push((6, intermediate));
    schedule.push((6, intermediate));
    for _ in 0..cfg.max_iter.max(1) {
        schedule.push((6, cfg.match_radius));
    }
    for (iteration, (terms, radius)) in schedule.iter().enumerate() {
        iterations = iteration + 1;
        let radius = *radius;
        let terms = *terms;
        let predicted: Vec<(f64, f64)> = references
            .tangent
            .iter()
            .map(|(xi, eta)| affine.apply(*xi, *eta))
            .collect();
        let current = mutual_match(&predicted, &detection_pixels, radius);
        eprintln!(
            "      iteration {iteration}: terms={terms} radius {radius:.2} px -> {} pairs",
            current.len()
        );
        if current.len() < cfg.min_pairs {
            break;
        }
        let Some(next) = robust_affine(&references.tangent, &detection_pixels, &current, terms)
        else {
            break;
        };
        let shift = if next.terms == affine.terms {
            next.delta(&affine)
        } else {
            f64::INFINITY
        };
        matches = current;
        affine = next;
        radius_used = radius;
        if shift < 1e-6 && radius <= cfg.match_radius + 1e-9 && terms == 6 {
            break;
        }
    }
    {
        let predicted: Vec<(f64, f64)> = references
            .tangent
            .iter()
            .map(|(xi, eta)| affine.apply(*xi, *eta))
            .collect();
        let final_matches = mutual_match(&predicted, &detection_pixels, cfg.match_radius);
        if final_matches.len() >= cfg.min_pairs {
            if let Some(next) =
                robust_affine(&references.tangent, &detection_pixels, &final_matches, 6)
            {
                affine = next;
            }
            let predicted: Vec<(f64, f64)> = references
                .tangent
                .iter()
                .map(|(xi, eta)| affine.apply(*xi, *eta))
                .collect();
            matches = mutual_match(&predicted, &detection_pixels, cfg.match_radius);
            radius_used = cfg.match_radius;
        }
    }
    let (affine_only_pairs, affine_only_rms) =
        match fit_affine(&references.tangent, &detection_pixels, &matches, 3) {
            Some(model) => {
                let predicted: Vec<(f64, f64)> = references
                    .tangent
                    .iter()
                    .map(|(xi, eta)| model.apply(*xi, *eta))
                    .collect();
                let candidate = mutual_match(&predicted, &detection_pixels, cfg.match_radius);
                let mut square = 0.0;
                for entry in &candidate {
                    let (xi, eta) = references.tangent[entry.reference];
                    let (px, py) = model.apply(xi, eta);
                    let (x, y) = detection_pixels[entry.detection];
                    square += (x - px).powi(2) + (y - py).powi(2);
                }
                let rms = (square / candidate.len().max(1) as f64).sqrt();
                (candidate.len(), rms)
            }
            None => (0_usize, f64::NAN),
        };
    if matches.len() < cfg.min_pairs {
        return Err(format!(
            "only {} pairs at {:.2} px, need at least {}",
            matches.len(),
            cfg.match_radius,
            cfg.min_pairs
        ));
    }

    eprintln!("[4/6] aperture photometry on {} pairs", matches.len());
    let mut rows: Vec<PairRow> = Vec::new();
    let mut geometry: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut skipped_photometry = 0_usize;
    for entry in &matches {
        let (xi, eta) = references.tangent[entry.reference];
        let (px, py) = affine.apply(xi, eta);
        let detection = &detections[entry.detection];
        let resid_x = detection.x - px;
        let resid_y = detection.y - py;
        geometry.push((xi, eta, detection.x, detection.y));
        let Some(photometry) = measure(&image, detection.x, detection.y, &cfg) else {
            skipped_photometry += 1;
            continue;
        };
        let magnitude_instrumental = if photometry.flux > 0.0 {
            -2.5 * photometry.flux.log10()
        } else {
            f64::NAN
        };
        let sigma_instrumental = if photometry.flux > 0.0 && photometry.flux_error > 0.0 {
            1.0857 * photometry.flux_error / photometry.flux
        } else {
            f64::NAN
        };
        rows.push(PairRow {
            reference: entry.reference,
            detection: entry.detection,
            resid_x,
            resid_y,
            resid_px: resid_x.hypot(resid_y),
            photometry,
            magnitude_instrumental,
            sigma_instrumental,
            radius: (detection.x - cfg.crpix_x).hypot(detection.y - cfg.crpix_y),
            block: block_of(image.width, image.height, detection.x, detection.y, cfg.block_grid),
            catalog_flux: detection.flux,
            catalog_flux_error: detection.flux_error,
            catalog_snr: detection.snr,
            catalog_noise: detection.noise,
            quality_passed: detection.quality_passed,
            quality_class: detection.quality_class.clone(),
            detection_id: detection.id.clone(),
            inlier: false,
            g_emp: f64::NAN,
            g_emp_order0: f64::NAN,
            resid_mag: f64::NAN,
        });
    }

    let mut samples: Vec<CalibrationSample> = Vec::new();
    let mut sample_row: Vec<usize> = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        if row.photometry.saturated() {
            continue;
        }
        if !row.magnitude_instrumental.is_finite()
            || !row.sigma_instrumental.is_finite()
            || row.sigma_instrumental <= 0.0
        {
            continue;
        }
        let reference = &references.stars[row.reference];
        let sigma_reference = reference.magnitude_error.max(1e-3);
        let sigma_total =
            (sigma_reference * sigma_reference + row.sigma_instrumental * row.sigma_instrumental)
                .sqrt()
                .max(PHOTOMETRIC_SIGMA_FLOOR_MAG);
        if !(sigma_total > 0.0) || !sigma_total.is_finite() {
            continue;
        }
        samples.push(CalibrationSample {
            magnitude_instrumental: row.magnitude_instrumental,
            sigma_instrumental: row.sigma_instrumental,
            color: reference.color,
            magnitude_reference: reference.magnitude,
            sigma_reference,
            weight: 1.0 / (sigma_total * sigma_total),
            x: detections[row.detection].x,
            y: detections[row.detection].y,
            radius: row.radius,
            block: row.block,
        });
        sample_row.push(index);
    }
    if samples.len() < cfg.min_pairs {
        return Err(format!(
            "only {} unsaturdated calibrators, need at least {}",
            samples.len(),
            cfg.min_pairs
        ));
    }
    let color_positions: Vec<usize> = (0..samples.len())
        .filter(|i| samples[*i].color.is_some())
        .collect();
    let color_samples: Vec<CalibrationSample> = color_positions.iter().map(|i| samples[*i].clone()).collect();

    eprintln!("[5/6] calibration: {} samples ({} with colour)", samples.len(), color_samples.len());
    let fit_order0_all = robust_calibration(&samples, 0, cfg.clip_sigma)
        .ok_or("order-0 calibration failed")?;
    let fit_order0_color = robust_calibration(&color_samples, 0, cfg.clip_sigma);
    let fit_order1 = robust_calibration(&color_samples, 1, cfg.clip_sigma);
    let cv_order0 = cross_validate(&color_samples, 0, &cfg);
    let cv_order1 = cross_validate(&color_samples, 1, &cfg);
    let mut comparison_indices: Vec<usize> = Vec::new();
    if let (Some(order0), Some(order1)) = (&fit_order0_color, &fit_order1) {
        for index in 0..color_samples.len() {
            if order0.inliers[index] && order1.inliers[index] {
                comparison_indices.push(index);
            }
        }
    }
    let comparison_samples: Vec<CalibrationSample> = comparison_indices
        .iter()
        .map(|index| color_samples[*index].clone())
        .collect();
    let fit_comparison0 = robust_calibration(&comparison_samples, 0, cfg.clip_sigma);
    let fit_comparison1 = robust_calibration(&comparison_samples, 1, cfg.clip_sigma);
    let cv_comparison0 = cross_validate(&comparison_samples, 0, &cfg);
    let cv_comparison1 = cross_validate(&comparison_samples, 1, &cfg);

    if let Some(fit) = &fit_order1 {
        for (position, sample_index) in color_positions.iter().enumerate() {
            if fit.inliers[position] {
                rows[sample_row[*sample_index]].inlier = true;
            }
        }
        for (position, sample_index) in color_positions.iter().enumerate() {
            let row_index = sample_row[*sample_index];
            let sample = &color_samples[position];
            rows[row_index].g_emp = sample.magnitude_instrumental
                + fit.fit.zero_point
                + fit.fit.color_term * sample.color.unwrap_or(0.0);
            rows[row_index].resid_mag = sample.magnitude_reference - rows[row_index].g_emp;
        }
    }
    {
        let fit = &fit_order0_all;
        for (position, row_index) in sample_row.iter().enumerate() {
            let sample = &samples[position];
            rows[*row_index].g_emp_order0 = sample.magnitude_instrumental + fit.fit.zero_point;
        }
    }

    let block_fits = leave_one_block_out(&color_samples, 1, &cfg);
    let mut block_offsets = Vec::new();
    for (block, n, offset, scatter) in &block_fits {
        let (bx, by) = block_label(*block, cfg.block_grid);
        block_offsets.push(json!({
            "block": block,
            "block_x": bx,
            "block_y": by,
            "n": n,
            "median_offset_mag": or_null(*offset),
            "scatter_mag": or_null(*scatter),
        }));
    }
    let offsets: Vec<f64> = block_fits
        .iter()
        .map(|(_, _, offset, _)| *offset)
        .filter(|v| v.is_finite())
        .collect();
    let joined = std::time::Instant::now();
    let _ = joined;

    eprintln!("[6/6] diagnostics");
    let mut position_residuals = Vec::new();
    let mut position_by_radius = Vec::new();
    {
        let mut all: Vec<f64> = rows.iter().map(|r| r.resid_px).collect();
        position_residuals.push(("median_px", median(&all).unwrap_or(f64::NAN)));
        all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        position_residuals.push(("p90_px", quantile_sorted(&all, 0.9)));
        let mut counts = std::collections::BTreeMap::new();
        for row in &rows {
            *counts.entry(row.block).or_insert(0_usize) += 1;
        }
        for window in (0..3200).step_by(400).collect::<Vec<i32>>().windows(2) {
            let (low, high) = (window[0] as f64, window[1] as f64);
            let selected: Vec<&PairRow> = rows
                .iter()
                .filter(|r| r.radius >= low && r.radius < high)
                .collect();
            if selected.is_empty() {
                continue;
            }
            let magnitudes: Vec<f64> = selected.iter().map(|r| r.resid_px).collect();
            let dx: Vec<f64> = selected.iter().map(|r| r.resid_x).collect();
            let dy: Vec<f64> = selected.iter().map(|r| r.resid_y).collect();
            position_by_radius.push(json!({
                "radius_low_px": low,
                "radius_high_px": high,
                "n": selected.len(),
                "median_residual_px": or_null(median(&magnitudes).unwrap_or(f64::NAN)),
                "median_dx_px": or_null(median(&dx).unwrap_or(f64::NAN)),
                "median_dy_px": or_null(median(&dy).unwrap_or(f64::NAN)),
            }));
        }
    }

    let mut distortion = json!({"available": false});
    if geometry.len() >= 20 {
        if let (Some((px3, py3)), Some((px6, py6))) = (
            polynomial_solution(&geometry, 3),
            polynomial_solution(&geometry, 6),
        ) {
            let rms_affine = polynomial_rms(&geometry, &px3, &py3);
            let rms_quadratic = polynomial_rms(&geometry, &px6, &py6);
            distortion = json!({
                "available": true,
                "affine_rms_px": or_null(rms_affine),
                "quadratic_rms_px": or_null(rms_quadratic),
                "rms_reduction_ratio": or_null(if rms_affine > 0.0 { rms_quadratic / rms_affine } else { f64::NAN }),
                "n": geometry.len(),
            });
        }
    }

    let color_edges = [-2.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 4.0];
    let magnitude_edges = [8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 21.0];
    let radius_edges = [0.0, 400.0, 800.0, 1200.0, 1600.0, 2000.0, 2400.0, 2800.0, 3200.0];
    let all_refs: Vec<&CalibrationSample> = samples.iter().collect();
    let zero_point_trends = json!({
        "basis": "all unsaturated matched pairs, offset = g_ref - m_inst",
        "n": all_refs.len(),
        "by_reference_magnitude": binned_offsets(&all_refs, &magnitude_edges, |s| Some(s.magnitude_reference)),
        "by_color": binned_offsets(&all_refs, &color_edges, |s| s.color),
        "by_radius": binned_offsets(&all_refs, &radius_edges, |s| Some(s.radius)),
    });
    let mut trends = json!({});
    if let Some(fit) = &fit_order1 {
        let residuals: Vec<f64> = color_samples
            .iter()
            .map(|s| s.magnitude_reference - predict(&fit.fit, s))
            .collect();
        let inlier_refs: Vec<&CalibrationSample> = color_positions
            .iter()
            .enumerate()
            .filter(|(position, _)| fit.inliers[*position])
            .map(|(_, index)| &samples[*index])
            .collect();
        let inlier_residuals: Vec<f64> = color_positions
            .iter()
            .enumerate()
            .filter(|(position, _)| fit.inliers[*position])
            .map(|(_, index)| {
                let sample = &samples[*index];
                sample.magnitude_reference - predict(&fit.fit, sample)
            })
            .collect();
        trends = json!({
            "n_scored": residuals.len(),
            "by_reference_magnitude": binned_residuals(&inlier_refs, &inlier_residuals, &magnitude_edges, |s| Some(s.magnitude_reference)),
            "by_color": binned_residuals(&inlier_refs, &inlier_residuals, &color_edges, |s| s.color),
            "by_radius": binned_residuals(&inlier_refs, &inlier_residuals, &radius_edges, |s| Some(s.radius)),
        });
    }

    let matched_count = rows.len();
    let negative_wrap = rows.iter().filter(|r| r.photometry.sat_negative).count();
    let peak_ceiling = rows.iter().filter(|r| r.photometry.sat_peak_ceiling).count();
    let zero_clip = rows.iter().filter(|r| r.photometry.sat_zero_clip).count();
    let saturated = rows.iter().filter(|r| r.photometry.saturated()).count();

    let flux_ratios: Vec<f64> = rows
        .iter()
        .filter(|r| r.catalog_flux > 0.0 && r.photometry.flux > 0.0)
        .map(|r| r.photometry.flux / r.catalog_flux)
        .collect();

    let mut reference_magnitudes: Vec<f64> = samples.iter().map(|s| s.magnitude_reference).collect();
    reference_magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut colors: Vec<f64> = samples.iter().filter_map(|s| s.color).collect();
    colors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut instrumental: Vec<f64> = samples.iter().map(|s| s.magnitude_instrumental).collect();
    instrumental.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let target = find_target(&cfg.sources, cfg.target_x, cfg.target_y).map(|row| {
        let photometry = measure(&image, row.x, row.y, &cfg);
        let magnitude_instrumental = photometry
            .as_ref()
            .filter(|p| p.flux > 0.0)
            .map(|p| -2.5 * p.flux.log10())
            .unwrap_or(f64::NAN);
        let sigma_instrumental = photometry
            .as_ref()
            .filter(|p| p.flux > 0.0 && p.flux_error > 0.0)
            .map(|p| 1.0857 * p.flux_error / p.flux)
            .unwrap_or(f64::NAN);
        let g_emp_order1 = if magnitude_instrumental.is_finite() && cfg.gain.is_some() {
            f64::NAN
        } else {
            fit_order1
                .as_ref()
                .map(|f| magnitude_instrumental + f.fit.zero_point)
                .unwrap_or(f64::NAN)
        };
        let faint_limit = reference_magnitudes.last().copied().unwrap_or(f64::NAN);
        let bright_limit = reference_magnitudes.first().copied().unwrap_or(f64::NAN);
        let mut flags: Vec<String> = Vec::new();
        if !g_emp_order1.is_finite() {
            flags.push("NO_EMPIRICAL_MAGNITUDE".into());
        } else {
            if g_emp_order1 > faint_limit {
                flags.push(format!(
                    "FAINTER_THAN_CALIBRATOR_LIMIT_BY_{:.3}_MAG",
                    g_emp_order1 - faint_limit
                ));
            }
            if g_emp_order1 < bright_limit {
                flags.push(format!(
                    "BRIGHTER_THAN_CALIBRATOR_LIMIT_BY_{:.3}_MAG",
                    bright_limit - g_emp_order1
                ));
            }
        }
        flags.push("COLOR_UNKNOWN_NO_COUNTERPART".into());
        json!({
            "detection_id": row.detection_id,
            "image_x": row.x,
            "image_y": row.y,
            "catalog_flux": or_null(row.flux),
            "catalog_flux_error": or_null(row.flux_error),
            "catalog_snr": or_null(row.snr),
            "catalog_noise": or_null(row.noise),
            "catalog_peak": or_null(row.peak),
            "quality_passed": row.quality_passed,
            "quality_class": row.quality_class,
            "flux": or_null(photometry.as_ref().map(|p| p.flux).unwrap_or(f64::NAN)),
            "flux_error": or_null(photometry.as_ref().map(|p| p.flux_error).unwrap_or(f64::NAN)),
            "background": or_null(photometry.as_ref().map(|p| p.background).unwrap_or(f64::NAN)),
            "noise": or_null(photometry.as_ref().map(|p| p.noise).unwrap_or(f64::NAN)),
            "aperture_pixels": or_null(photometry.as_ref().map(|p| p.aperture_pixels).unwrap_or(f64::NAN)),
            "peak": or_null(photometry.as_ref().map(|p| p.peak).unwrap_or(f64::NAN)),
            "saturated": photometry.as_ref().map(|p| p.saturated()),
            "magnitude_instrumental": or_null(magnitude_instrumental),
            "sigma_magnitude_instrumental": or_null(sigma_instrumental),
            "g_emp": or_null(g_emp_order1),
            "g_emp_note": "colour term not applied: no reference counterpart, colour unknown",
            "extrapolation_flags": flags,
        })
    });

    // ---------------------------------------------------------------- outputs
    let mut pairs_csv = String::new();
    let _ = writeln!(
        pairs_csv,
        "pair_id,source_id,image_x,image_y,detection_id,g_ref,g_ref_error,bp_rp,ra_pm_deg,dec_pm_deg,pm_ra_mas_yr,pm_dec_mas_yr,catalog_flux,catalog_flux_error,catalog_snr,catalog_noise,flux,flux_error,background,noise,n_aperture_pixels,peak,snr,m_inst,sigma_m_inst,g_emp,g_emp_order0,resid_mag,weight,sat_negative,sat_peak_ceiling,sat_zero_clip,saturated,photometric_inlier,resid_x_px,resid_y_px,resid_px,radius_px,block_x,block_y,quality_passed,quality_class"
    );
    for (index, row) in rows.iter().enumerate() {
        let reference = &references.stars[row.reference];
        let (bx, by) = block_label(row.block, cfg.block_grid);
        let sigma_reference = reference.magnitude_error.max(1e-3);
        let sigma_total = (sigma_reference * sigma_reference
            + row.sigma_instrumental * row.sigma_instrumental)
            .sqrt()
            .max(PHOTOMETRIC_SIGMA_FLOOR_MAG);
        let weight = if sigma_total > 0.0 && sigma_total.is_finite() {
            1.0 / (sigma_total * sigma_total)
        } else {
            f64::NAN
        };
        let _ = writeln!(
            pairs_csv,
            "{index},{},{:.4},{:.4},{},{:.5},{:.5},{},{:.8},{:.8},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.4},{:.5},{:.5},{:.5},{:.5},{:.5},{:.6},{},{},{},{},{},{:.5},{:.5},{:.5},{:.2},{},{},{},{}",
            reference.source_id,
            detections[row.detection].x,
            detections[row.detection].y,
            row.detection_id,
            reference.magnitude,
            reference.magnitude_error.abs(),
            csv_opt(reference.color),
            reference.propagated_ra_deg,
            reference.propagated_dec_deg,
            reference.pm_ra,
            reference.pm_dec,
            row.catalog_flux,
            row.catalog_flux_error,
            row.catalog_snr,
            row.catalog_noise,
            row.photometry.flux,
            row.photometry.flux_error,
            row.photometry.background,
            row.photometry.noise,
            row.photometry.aperture_pixels,
            row.photometry.peak,
            if row.photometry.flux_error > 0.0 { row.photometry.flux / row.photometry.flux_error } else { f64::NAN },
            row.magnitude_instrumental,
            row.sigma_instrumental,
            row.g_emp,
            row.g_emp_order0,
            row.resid_mag,
            weight,
            row.photometry.sat_negative,
            row.photometry.sat_peak_ceiling,
            row.photometry.sat_zero_clip,
            row.photometry.saturated(),
            row.inlier,
            row.resid_x,
            row.resid_y,
            row.resid_px,
            row.radius,
            bx,
            by,
            row.quality_passed,
            row.quality_class,
        );
    }
    std::fs::write(out_dir.join("pairs.csv"), pairs_csv).map_err(|e| e.to_string())?;

    let mut block_csv = String::from(
        "block,block_x,block_y,n,median_photometric_offset_mag,scatter_mag,median_residual_px,median_dx_px,median_dy_px\n",
    );
    let mut block_ids: std::collections::BTreeSet<usize> =
        rows.iter().map(|r| r.block).collect();
    for block in std::mem::take(&mut block_ids) {
        let selected: Vec<&PairRow> = rows.iter().filter(|r| r.block == block).collect();
        if selected.is_empty() {
            continue;
        }
        let (bx, by) = block_label(block, cfg.block_grid);
        let offset = block_fits
            .iter()
            .find(|(b, _, _, _)| *b == block)
            .map(|(_, _, offset, _)| *offset)
            .unwrap_or(f64::NAN);
        let scatter = block_fits
            .iter()
            .find(|(b, _, _, _)| *b == block)
            .map(|(_, _, _, scatter)| *scatter)
            .unwrap_or(f64::NAN);
        let residuals: Vec<f64> = selected.iter().map(|r| r.resid_px).collect();
        let dx: Vec<f64> = selected.iter().map(|r| r.resid_x).collect();
        let dy: Vec<f64> = selected.iter().map(|r| r.resid_y).collect();
        let _ = writeln!(
            block_csv,
            "{block},{bx},{by},{},{:.5},{:.5},{:.5},{:.5},{:.5}",
            selected.len(),
            offset,
            scatter,
            median(&residuals).unwrap_or(f64::NAN),
            median(&dx).unwrap_or(f64::NAN),
            median(&dy).unwrap_or(f64::NAN),
        );
    }
    std::fs::write(out_dir.join("blocks.csv"), block_csv).map_err(|e| e.to_string())?;

    let order0_summary = |fit: &Option<RobustFit>, cv: &Option<CrossValidation>| -> Value {
        match fit {
            Some(fit) => json!({
                "zero_point": or_null(fit.fit.zero_point),
                "zero_point_error": or_null(fit.fit.zero_point_error),
                "n_used": fit.fit.n,
                "n_inliers": fit.kept,
                "n_clipped": fit.clipped,
                "in_sample_scatter_mag": or_null(fit.scatter),
                "chi_square": or_null(fit.fit.chi_square),
                "holdout_sigma_mag": cv.as_ref().map(|c| or_null(c.sigma_out)).unwrap_or(Value::Null),
                "holdout_sigma_normalized": cv.as_ref().map(|c| or_null(c.sigma_out_normalized)).unwrap_or(Value::Null),
                "holdout_sigma_all_points_mag": cv.as_ref().map(|c| or_null(c.sigma_out_all)).unwrap_or(Value::Null),
                "holdout_n": cv.as_ref().map(|c| c.n_holdout).unwrap_or(0),
                "holdout_n_all": cv.as_ref().map(|c| c.n_scored_all).unwrap_or(0),
                "holdout_fold_sigma": cv.as_ref().map(|c| json!(c.fold_sigma.iter().map(|v| or_null(*v)).collect::<Vec<_>>())).unwrap_or(Value::Null),
            }),
            None => json!({"available": false}),
        }
    };

    let block_spread = if offsets.len() > 1 {
        stddev(&offsets)
    } else {
        f64::NAN
    };
    let block_max_offset = offsets
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));

    let cv0 = cv_order0.as_ref().map(|c| c.sigma_out).unwrap_or(f64::NAN);
    let cv1 = cv_order1.as_ref().map(|c| c.sigma_out).unwrap_or(f64::NAN);
    let cv0c = cv_comparison0.as_ref().map(|c| c.sigma_out).unwrap_or(f64::NAN);
    let cv1c = cv_comparison1.as_ref().map(|c| c.sigma_out).unwrap_or(f64::NAN);

    let calibration = json!({
        "schema_version": SCHEMA_VERSION,
        "producer": "rst19-photometric-pairs",
        "utc": utc,
        "run_id": run_id,
        "band_definition": BAND_DEFINITION,
        "result_field": "g_emp",
        "result_definition": "g_emp = m_inst + ZP + C*(BP-RP); m_inst = -2.5*log10(flux)",
        "models": {
            "order0": {"form": "g_ref = m_inst + ZP", "scope": "all unsaturated pairs"},
            "order1": {"form": "g_ref = m_inst + ZP + C*(BP-RP)", "scope": "unsaturated pairs with a colour"},
        },
        "zero_point": {
            "value": or_null(fit_order0_all.fit.zero_point),
            "sigma": or_null(fit_order0_all.fit.zero_point_error),
            "n": fit_order0_all.fit.n,
            "n_inliers": fit_order0_all.kept,
            "n_clipped": fit_order0_all.clipped,
            "in_sample_scatter_mag": or_null(fit_order0_all.scatter),
        },
        "color_term": {
            "value": fit_order1.as_ref().map(|f| or_null(f.fit.color_term)).unwrap_or(Value::Null),
            "sigma": fit_order1.as_ref().map(|f| or_null(f.fit.color_term_error)).unwrap_or(Value::Null),
            "n": fit_order1.as_ref().map(|f| f.fit.n).unwrap_or(0),
            "n_inliers": fit_order1.as_ref().map(|f| f.kept).unwrap_or(0),
            "n_clipped": fit_order1.as_ref().map(|f| f.clipped).unwrap_or(0),
            "in_sample_scatter_mag": fit_order1.as_ref().map(|f| or_null(f.scatter)).unwrap_or(Value::Null),
        },
        "covariance": {
            "parameters": ["ZP", "C"],
            "matrix": fit_order1.as_ref().map(|f| json!([
                [or_null(f.fit.covariance[0][0]), or_null(f.fit.covariance[0][1])],
                [or_null(f.fit.covariance[1][0]), or_null(f.fit.covariance[1][1])]
            ])).unwrap_or(Value::Null),
            "correlation_zp_c": fit_order1.as_ref().map(|f| {
                let denominator = (f.fit.covariance[0][0] * f.fit.covariance[1][1]).sqrt();
                or_null(if denominator > 0.0 { f.fit.covariance[0][1] / denominator } else { f64::NAN })
            }).unwrap_or(Value::Null),
        },
        "holdout": {
            "kfold": cfg.kfold,
            "order0": order0_summary(&fit_order0_color, &cv_order0),
            "order1": order0_summary(&fit_order1, &cv_order1),
            "sigma_out_order0_mag": or_null(cv0c),
            "sigma_out_order1_mag": or_null(cv1c),
            "sigma_out_improvement_mag": or_null(cv0c - cv1c),
            "sigma_out_ratio": or_null(if cv0c > 0.0 { cv1c / cv0c } else { f64::NAN }),
            "colour_term_significant_over_holdout": if cv0c.is_finite() && cv1c.is_finite() {
                json!(cv1c < cv0c)
            } else {
                Value::Null
            },
            "sigma_out_note": "sigma_out_order* use the common inlier subset so the two models are compared on identical stars",
        },
        "model_comparison": {
            "subset": "intersection of the order-0 and order-1 inlier sets",
            "n": comparison_samples.len(),
            "order0": order0_summary(&fit_comparison0, &cv_comparison0),
            "order1": order0_summary(&fit_comparison1, &cv_comparison1),
            "sigma_out_order0_mag": or_null(cv0c),
            "sigma_out_order1_mag": or_null(cv1c),
            "sigma_out_improvement_mag": or_null(cv0c - cv1c),
            "sigma_out_ratio": or_null(if cv0c > 0.0 { cv1c / cv0c } else { f64::NAN }),
            "verdict": if cv0c.is_finite() && cv1c.is_finite() {
                if cv1c < cv0c * 0.9 {
                    json!("colour term materially reduces holdout scatter")
                } else if cv1c <= cv0c * 1.1 {
                    json!("no material change from the colour term")
                } else {
                    json!("colour term makes holdout scatter worse")
                }
            } else {
                Value::Null
            },
        },
        "spatial_blocks": {
            "grid": cfg.block_grid,
            "blocks_used": block_fits.len(),
            "leave_one_block_out_sigma_out_spread_mag": or_null(block_spread),
            "leave_one_block_out_max_abs_offset_mag": or_null(block_max_offset),
            "blocks": block_offsets,
        },
        "applicable_range": {
            "g_ref_min": or_null(reference_magnitudes.first().copied().unwrap_or(f64::NAN)),
            "g_ref_max": or_null(reference_magnitudes.last().copied().unwrap_or(f64::NAN)),
            "bp_rp_min": or_null(colors.first().copied().unwrap_or(f64::NAN)),
            "bp_rp_max": or_null(colors.last().copied().unwrap_or(f64::NAN)),
            "m_inst_min": or_null(instrumental.first().copied().unwrap_or(f64::NAN)),
            "m_inst_max": or_null(instrumental.last().copied().unwrap_or(f64::NAN)),
            "note": "outside these intervals the empirical zero point is an extrapolation",
        },
        "saturation": {
            "matched_pairs": matched_count,
            "negative_pixel_wraparound": negative_wrap,
            "peak_at_or_above_ceiling": peak_ceiling,
            "aperture_zero_clip": zero_clip,
            "any_saturation_flag": saturated,
            "fraction": or_null(if matched_count > 0 { saturated as f64 / matched_count as f64 } else { f64::NAN }),
            "remaining_after_saturation_cut": matched_count - saturated,
            "peak_ceiling": cfg.peak_ceiling,
        },
        "diagnostics": {
            "pairs": matched_count,
            "matched_before_photometry": matches.len(),
            "photometry_failed": skipped_photometry,
            "calibration_samples": samples.len(),
            "calibration_samples_with_colour": color_samples.len(),
            "iterations": iterations,
            "final_match_radius_px": radius_used,
            "affine": {
                "terms": affine.terms,
                "x": affine.ax.iter().map(|v| or_null(*v)).collect::<Vec<_>>(),
                "y": affine.ay.iter().map(|v| or_null(*v)).collect::<Vec<_>>(),
                "matrix_px_per_arcsec": [[affine.matrix()[0][0], affine.matrix()[0][1]], [affine.matrix()[1][0], affine.matrix()[1][1]]],
            },
            "affine_only_model": {
                "pairs_at_match_radius": affine_only_pairs,
                "rms_px": or_null(affine_only_rms),
                "note": "same match radius with a pure affine (no polynomial distortion terms)",
            },
            "position_residuals": {
                "median_px": or_null(position_residuals.first().map(|(_, v)| *v).unwrap_or(f64::NAN)),
                "p90_px": or_null(position_residuals.get(1).map(|(_, v)| *v).unwrap_or(f64::NAN)),
                "by_radius": position_by_radius,
            },
            "distortion": distortion,
            "photometric_trends": trends,
            "zero_point_trends": zero_point_trends,
            "aperture_reproduction": {
                "median_flux_over_catalog": or_null(median(&flux_ratios).unwrap_or(f64::NAN)),
                "mad_flux_over_catalog": or_null(mad_sigma(&flux_ratios).unwrap_or(f64::NAN)),
                "n": flux_ratios.len(),
            },
            "frame": {
                "negative_pixels": negative_pixels,
                "zero_pixels": zero_pixels,
                "telemetry_pixels": image.telemetry_pixels,
            },
        },
        "target": target,
    });

    let calibration_text = serde_json::to_string_pretty(&calibration).map_err(|e| e.to_string())?;
    std::fs::write(out_dir.join("calibration.json"), format!("{calibration_text}\n"))
        .map_err(|e| e.to_string())?;

    let binary = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("unknown"));
    let config_text = serde_json::to_string(&cfg.as_json()).map_err(|e| e.to_string())?;
    let elapsed = started.elapsed().map(|d| d.as_millis()).unwrap_or(0);
    let manifest = json!({
        "schema_version": SCHEMA_VERSION,
        "producer": "rst19-photometric-pairs",
        "utc": utc,
        "run_id": run_id,
        "output_dir": out_dir.to_string_lossy(),
        "binary": {"path": binary.to_string_lossy(), "sha256": sha256_file(&binary)},
        "inputs": {
            "fits": {"path": cfg.fits.to_string_lossy(), "sha256": sha256_file(&cfg.fits)},
            "sources_csv": {"path": cfg.sources.to_string_lossy(), "sha256": sha256_file(&cfg.sources)},
            "reference_csv": {"path": cfg.gaia.to_string_lossy(), "sha256": sha256_file(&cfg.gaia)},
        },
        "config": cfg.as_json(),
        "config_sha256": sha256_text(&config_text),
        "counts": {
            "source_rows": source_rows,
            "source_rows_rejected_by_quality": rejected_rows,
            "detections_used": detections.len(),
            "reference_rows": references.rows,
            "reference_stars": references.stars.len(),
            "matched_pairs": matched_count,
            "calibration_samples": samples.len(),
        },
        "runtime_ms": elapsed,
    });
    let manifest_text = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    std::fs::write(out_dir.join("run_manifest.json"), format!("{manifest_text}\n"))
        .map_err(|e| e.to_string())?;

    let saturation = calibration["saturation"].clone();
    println!("outputs: {}", out_dir.display());
    println!("matched_pairs={matched_count} iterations={iterations} radius={radius_used:.2}px");
    println!(
        "model terms={} affine_only pairs={} rms={:.3}px",
        affine.terms, affine_only_pairs, affine_only_rms
    );
    println!(
        "position_residual median={:.3}px p90={:.3}px",
        calibration["diagnostics"]["position_residuals"]["median_px"].as_f64().unwrap_or(f64::NAN),
        calibration["diagnostics"]["position_residuals"]["p90_px"].as_f64().unwrap_or(f64::NAN)
    );
    println!(
        "saturation matched={} any={} frac={:.3} remaining={} (neg={} peak={} zero={})",
        saturation["matched_pairs"].as_u64().unwrap_or(0),
        saturation["any_saturation_flag"].as_u64().unwrap_or(0),
        saturation["fraction"].as_f64().unwrap_or(f64::NAN),
        saturation["remaining_after_saturation_cut"].as_u64().unwrap_or(0),
        saturation["negative_pixel_wraparound"].as_u64().unwrap_or(0),
        saturation["peak_at_or_above_ceiling"].as_u64().unwrap_or(0),
        saturation["aperture_zero_clip"].as_u64().unwrap_or(0),
    );
    println!(
        "ZP={:.4}+-{:.4} (n={}) scatter={:.4}",
        fit_order0_all.fit.zero_point,
        fit_order0_all.fit.zero_point_error,
        fit_order0_all.fit.n,
        fit_order0_all.scatter
    );
    if let Some(fit) = &fit_order1 {
        println!(
            "ZP1={:.4}+-{:.4} C={:.4}+-{:.4} (n={}) scatter={:.4}",
            fit.fit.zero_point,
            fit.fit.zero_point_error,
            fit.fit.color_term,
            fit.fit.color_term_error,
            fit.fit.n,
            fit.scatter
        );
    }
    println!(
        "holdout sigma order0={:.4} order1={:.4} (k={})",
        cv0, cv1, cfg.kfold
    );
    println!(
        "block spread={:.4} max|offset|={:.4} over {} blocks",
        block_spread,
        block_max_offset,
        block_fits.len()
    );
    println!(
        "range g_ref=[{:.2},{:.2}] bp_rp=[{:.2},{:.2}]",
        reference_magnitudes.first().copied().unwrap_or(f64::NAN),
        reference_magnitudes.last().copied().unwrap_or(f64::NAN),
        colors.first().copied().unwrap_or(f64::NAN),
        colors.last().copied().unwrap_or(f64::NAN)
    );
    println!(
        "target g_emp={:?}",
        calibration["target"]["g_emp"]
    );
    Ok(())
}

const HELP: &str = "Reference-star pairing and empirical zero point with cross validation.\n\nUSAGE: rst19-photometric-pairs [FITS] [OPTIONS]\n\nOptions:\n  --sources CSV              detection catalog (default tmp/rust-sources-prod/source_catalog.csv)\n  --gaia CSV                 reference catalog (default tmp/gaia/frame1_gaia_g13.csv)\n  --out DIR                  output directory (default tmp/rust-pairs/RUNID)\n  --scale ARCSEC --rotation DEG --parity 1|-1\n  --crpix-x PX --crpix-y PX --center-ra DEG --center-dec DEG\n  --epoch YEAR               propagate reference proper motions to this epoch\n  --match-radius PX          bidirectional nearest-neighbour radius (default 1.2)\n  --bootstrap-radius PX      radius for the first iterations (default 4.0)\n  --peak-ceiling DN          saturation ceiling on the peak pixel (default 20000)\n  --aperture-radius PX       aperture radius (default 4)\n  --annulus-inner PX --annulus-outer PX   local background annulus (default 6, 10)\n  --clip-sigma S --clip-iterations N      robust clipping (default 3.0, 4)\n  --kfold K --block-grid N   cross validation layout (default 5, 8)\n  --gain ADU_PER_ELECTRON    optional camera gain for the Poisson term\n  --include-rejected-sources keep detections that failed quality gating\n  --target-x PX --target-y PX  science target position (default 2929.8555, 3915.2171)\n  -h, --help";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => {}
        Err(error) if error == "__help__" => {
            println!("{HELP}");
        }
        Err(error) => {
            eprintln!("photometric-pairs: {error}");
            std::process::exit(2);
        }
    }
}
