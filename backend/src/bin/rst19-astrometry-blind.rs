//! rst19-astrometry-blind — blind (roll, scale, parity, translation) search between a
//! detection table and a reference catalogue, with built-in synthetic self-calibration and
//! mandatory null controls. No WCS and no magnitude is emitted unless the decision gate passes.
//!
//! Model. Both sides are rasterised onto an N x N grid that lives in *detector* coordinates.
//! A candidate solution maps catalogue arcsec coordinates (xi, eta) onto detector pixels by
//!     px = R(roll) * M(parity) * (xi, eta) / scale + t
//! so the FFT cross-correlation over t returns, for every (parity, roll, scale), the best shift
//! and its significance. The reported translation t is the detector position of the projection
//! centre (ra0, dec0).
//!
//! The search is hierarchical: an exhaustive sweep on a coarse grid (--search-bins), then
//! factorial-style local refinement (4x finer steps, +-1 previous step window) on a
//! successively finer grid up to --bins. Every stage re-solves the translation with a full FFT,
//! so only roll and scale need local scanning.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

const USAGE: &str = r#"rst19-astrometry-blind — blind astrometric pattern search with self-calibration

REQUIRED
  --sources <csv>        detection table (quality_passed, x, y, peak columns)
  --catalog <csv>        reference catalogue (ra_deg/dec_deg/phot_g_mean_mag|magnitude)

DATA SELECTION
  --peak-ceiling <v>     keep detections with peak < v               [20000]
  --mask-telemetry       drop detections inside the telemetry block (x<104 and y<104)
  --catalog-min-g <g>    catalogue G lower bound                     [3]
  --catalog-max-g <g>    catalogue G upper bound                     [13]
  --frame-size <px>      detector frame edge in pixels               [4096]
  --ra0 <deg> --dec0 <deg>  projection centre (default: catalogue centroid)

GRID
  --bins <n>             finest grid edge, power of two              [1024]  -> 4 px/bin
  --search-bins <n>      exhaustive sweep grid edge                  [bins/8]
  --scale-min <v>        arcsec/px lower bound                       [7.6]
  --scale-max <v>        arcsec/px upper bound                       [10.6]
  --roll-zeta <z>        step = z * bin / half_diagonal               [1.0]
  --smooth-sigma <s>     gaussian smoothing of both maps, in bins     [0.0]
  --threads <n>          worker threads                              [cores]
  --keep a,b,c           candidates kept per ladder level            [32,10,3]
  --sample-sigma <k>     coarse trials with full map statistics      [100]

SELF-CALIBRATION
  --selfcal-scale <v>    truth scale for synthetic detections        [8.44]
  --jitter-px <v>        gaussian position jitter                    [1.5]
  --completeness <f>     fraction of catalogue stars injected        [0.35]
  --false-positive-frac <f> uniform false positives per kept star    [0.30]
  --seed <u64>           rng seed                                    [20260330]
  --selfcal-mode <m>     baseline | matrix                           [baseline]
  --scale-grid-shift-frac <f> shift the search scale grid by f*scale [0]

CONTROLS
  --control-trials <k>   null sweeps per control family              [1]

OUTPUT
  --out <dir>            output directory [tmp/rust-astrometry/<runid>]
  --tag <s>              free-form label written into the manifest

MODES
  --selftest             end-to-end geometry check, then exit
  --skip-selfcal --skip-real --skip-controls
"#;

// ----------------------------------------------------------------------------- helpers

fn fin(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else if x > 0.0 {
        json!(1e12)
    } else {
        json!(-1e12)
    }
}

fn utc_stamp(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}{m:02}{d:02}T{h:02}{mi:02}{s:02}Z")
}

fn now_utc() -> (u64, String) {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    (secs, utc_stamp(secs))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = std::io::Read::read(&mut f, &mut buf).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_str(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn median(v: &[f64]) -> f64 {
    let mut c: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if c.is_empty() {
        return f64::NAN;
    }
    c.sort_unstable_by(|a, b| a.total_cmp(b));
    c[c.len() / 2]
}

fn percentile(v: &[f64], p: f64) -> f64 {
    let mut c: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if c.is_empty() {
        return f64::NAN;
    }
    c.sort_unstable_by(|a, b| a.total_cmp(b));
    c[((p * c.len() as f64) as usize).min(c.len() - 1)]
}

// ----------------------------------------------------------------------------- rng

struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
            spare: None,
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state | 1;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let r = u * u + v * v;
            if r > 0.0 && r < 1.0 {
                let f = (-2.0 * r.ln() / r).sqrt();
                self.spare = Some(v * f);
                return u * f;
            }
        }
    }
}

// ----------------------------------------------------------------------------- config

#[derive(Clone, Debug)]
struct Cfg {
    sources: PathBuf,
    catalog: PathBuf,
    out_dir: PathBuf,
    run_id: String,
    mask_telemetry: bool,
    bins: usize,
    search_bins: usize,
    peak_ceiling: f64,
    scale_min: f64,
    scale_max: f64,
    roll_zeta: f64,
    smooth_sigma: f64,
    frame: f64,
    ra0: Option<f64>,
    dec0: Option<f64>,
    catalog_min_g: f64,
    catalog_max_g: f64,
    threads: usize,
    keep: Vec<usize>,
    sample_sigma: usize,
    selfcal_scale: f64,
    jitter_px: f64,
    completeness: f64,
    fp_frac: f64,
    seed: u64,
    selfcal_mode: String,
    scale_grid_shift_frac: f64,
    control_trials: usize,
    skip_selfcal: bool,
    skip_real: bool,
    skip_controls: bool,
    selftest: bool,
    tag: String,
}

struct Cmd {
    m: HashMap<String, String>,
}

impl Cmd {
    fn get(&self, k: &str) -> Option<&str> {
        self.m.get(k).map(|s| s.as_str())
    }
    fn f64_or(&self, k: &str, d: f64) -> Result<f64, String> {
        match self.get(k) {
            None => Ok(d),
            Some(v) => v.parse().map_err(|_| format!("--{k}: not a number: {v}")),
        }
    }
    fn u64_or(&self, k: &str, d: u64) -> Result<u64, String> {
        match self.get(k) {
            None => Ok(d),
            Some(v) => v.parse().map_err(|_| format!("--{k}: not an integer: {v}")),
        }
    }
    fn usize_or(&self, k: &str, d: usize) -> Result<usize, String> {
        match self.get(k) {
            None => Ok(d),
            Some(v) => v.parse().map_err(|_| format!("--{k}: not an integer: {v}")),
        }
    }
    fn has(&self, k: &str) -> bool {
        self.m.contains_key(k)
    }
}

const FLAGS: [&str; 5] = [
    "mask-telemetry",
    "selftest",
    "skip-selfcal",
    "skip-real",
    "skip-controls",
];

fn parse_args(args: &[String]) -> Result<Cmd, String> {
    let mut m: HashMap<String, String> = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let key_body = a
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected positional argument: {a}"))?;
        let (key, inline) = match key_body.split_once('=') {
            Some((k, v)) => (k.to_string(), Some(v.to_string())),
            None => (key_body.to_string(), None),
        };
        if let Some(v) = inline {
            m.insert(key, v);
        } else if FLAGS.contains(&key.as_str()) {
            m.insert(key, "true".into());
        } else {
            let v = args
                .get(i + 1)
                .ok_or_else(|| format!("--{key} needs a value"))?
                .clone();
            m.insert(key, v);
            i += 1;
        }
        i += 1;
    }
    Ok(Cmd { m })
}

fn build_cfg(cmd: &Cmd) -> Result<Cfg, String> {
    let sources = PathBuf::from(cmd.get("sources").ok_or("--sources is required")?);
    let catalog = PathBuf::from(cmd.get("catalog").ok_or("--catalog is required")?);
    let bins = cmd.usize_or("bins", 1024)?;
    let search_bins = if cmd.has("search-bins") {
        cmd.usize_or("search-bins", 128)?
    } else {
        (bins / 8).max(32)
    };
    if !bins.is_power_of_two() || !search_bins.is_power_of_two() {
        return Err("--bins and --search-bins must be powers of two".into());
    }
    if search_bins > bins {
        return Err("--search-bins must be <= --bins".into());
    }
    let keep = match cmd.get("keep") {
        None => vec![32, 10, 3],
        Some(v) => {
            let k: Vec<usize> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            if k.is_empty() {
                return Err("--keep: expected e.g. 32,10,3".into());
            }
            k
        }
    };
    let cores = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(4);
    let threads = cmd.usize_or("threads", cores)?.clamp(1, 64);
    let (_, stamp) = now_utc();
    let run_id = match cmd.get("tag") {
        Some(t) => format!("{t}_{stamp}"),
        None => stamp.clone(),
    };
    let out_dir = cmd
        .get("out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tmp/rust-astrometry").join(&run_id));
    Ok(Cfg {
        sources,
        catalog,
        out_dir,
        run_id,
        mask_telemetry: cmd.has("mask-telemetry"),
        bins,
        search_bins,
        peak_ceiling: cmd.f64_or("peak-ceiling", 20_000.0)?,
        scale_min: cmd.f64_or("scale-min", 7.6)?,
        scale_max: cmd.f64_or("scale-max", 10.6)?,
        roll_zeta: cmd.f64_or("roll-zeta", 1.0)?,
        smooth_sigma: cmd.f64_or("smooth-sigma", 0.0)?,
        frame: cmd.f64_or("frame-size", 4096.0)?,
        ra0: match cmd.get("ra0") {
            Some(v) => Some(v.parse::<f64>().map_err(|_| "--ra0: not a number")?),
            None => None,
        },
        dec0: match cmd.get("dec0") {
            Some(v) => Some(v.parse::<f64>().map_err(|_| "--dec0: not a number")?),
            None => None,
        },
        catalog_min_g: cmd.f64_or("catalog-min-g", 3.0)?,
        catalog_max_g: cmd.f64_or("catalog-max-g", 13.0)?,
        threads,
        keep,
        sample_sigma: cmd.usize_or("sample-sigma", 100)?,
        selfcal_scale: cmd.f64_or("selfcal-scale", 8.44)?,
        jitter_px: cmd.f64_or("jitter-px", 1.5)?,
        completeness: cmd.f64_or("completeness", 0.35)?,
        fp_frac: cmd.f64_or("false-positive-frac", 0.30)?,
        seed: cmd.u64_or("seed", 20_260_330)?,
        selfcal_mode: cmd.get("selfcal-mode").unwrap_or("baseline").to_string(),
        scale_grid_shift_frac: cmd.f64_or("scale-grid-shift-frac", 0.0)?,
        control_trials: cmd.usize_or("control-trials", 1)?,
        skip_selfcal: cmd.has("skip-selfcal"),
        skip_real: cmd.has("skip-real"),
        skip_controls: cmd.has("skip-controls"),
        selftest: cmd.has("selftest"),
        tag: cmd.get("tag").unwrap_or("").to_string(),
    })
}

// ----------------------------------------------------------------------------- inputs

fn read_csv(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("headers {}: {e}", path.display()))?
        .iter()
        .map(|s| s.trim().to_string())
        .collect();
    let mut rows = Vec::new();
    for rec in rdr.records() {
        let rec = rec.map_err(|e| format!("row {}: {e}", path.display()))?;
        rows.push(rec.iter().map(|s| s.to_string()).collect());
    }
    Ok((headers, rows))
}

fn col(headers: &[String], name: &str) -> Result<usize, String> {
    headers
        .iter()
        .position(|h| h == name)
        .ok_or_else(|| format!("missing column '{name}'"))
}

fn truthy(s: &str) -> bool {
    matches!(s.trim().to_ascii_lowercase().as_str(), "true" | "1" | "yes")
}

struct SourceReport {
    points: Vec<(f64, f64)>,
    total_rows: usize,
    rejected_quality: usize,
    rejected_peak: usize,
    telemetry_masked: usize,
    outside_frame: usize,
}

fn load_sources(cfg: &Cfg) -> Result<SourceReport, String> {
    let (headers, rows) = read_csv(&cfg.sources)?;
    let cx = col(&headers, "x")?;
    let cy = col(&headers, "y")?;
    let cp = col(&headers, "peak")?;
    let cq = col(&headers, "quality_passed")?;
    let mut rep = SourceReport {
        points: Vec::new(),
        total_rows: rows.len(),
        rejected_quality: 0,
        rejected_peak: 0,
        telemetry_masked: 0,
        outside_frame: 0,
    };
    for r in &rows {
        if r.len() <= cq.max(cp).max(cy).max(cx) {
            continue;
        }
        if !truthy(&r[cq]) {
            rep.rejected_quality += 1;
            continue;
        }
        let peak: f64 = match r[cp].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !(peak < cfg.peak_ceiling) {
            rep.rejected_peak += 1;
            continue;
        }
        let (Ok(x), Ok(y)) = (r[cx].trim().parse::<f64>(), r[cy].trim().parse::<f64>()) else {
            continue;
        };
        if cfg.mask_telemetry && x < 104.0 && y < 104.0 {
            rep.telemetry_masked += 1;
            continue;
        }
        if !(0.0..cfg.frame).contains(&x) || !(0.0..cfg.frame).contains(&y) {
            rep.outside_frame += 1;
            continue;
        }
        rep.points.push((x, y));
    }
    eprintln!(
        "sources: rows={} kept={} rejected_quality={} rejected_peak={} telemetry_masked={} outside_frame={}",
        rep.total_rows,
        rep.points.len(),
        rep.rejected_quality,
        rep.rejected_peak,
        rep.telemetry_masked,
        rep.outside_frame
    );
    Ok(rep)
}

struct Catalog {
    ra: Vec<f64>,
    dec: Vec<f64>,
    g: Vec<f64>,
}

fn load_catalog(cfg: &Cfg) -> Result<Catalog, String> {
    let (headers, rows) = read_csv(&cfg.catalog)?;
    let cra = col(&headers, "ra_deg")?;
    let cdec = col(&headers, "dec_deg")?;
    let cg = match headers.iter().position(|h| h == "phot_g_mean_mag") {
        Some(i) => i,
        None => col(&headers, "magnitude")?,
    };
    let mut cat = Catalog {
        ra: Vec::new(),
        dec: Vec::new(),
        g: Vec::new(),
    };
    let mut skipped = 0usize;
    let mut out_of_range = 0usize;
    for r in &rows {
        if r.len() <= cra.max(cdec).max(cg) {
            skipped += 1;
            continue;
        }
        let (Ok(ra), Ok(dec)) = (r[cra].trim().parse::<f64>(), r[cdec].trim().parse::<f64>()) else {
            skipped += 1;
            continue;
        };
        let g = r[cg].trim().parse::<f64>().unwrap_or(f64::NAN);
        if g.is_finite() && (g < cfg.catalog_min_g || g > cfg.catalog_max_g) {
            out_of_range += 1;
            continue;
        }
        cat.ra.push(ra);
        cat.dec.push(dec);
        cat.g.push(g);
    }
    eprintln!(
        "catalog: rows={} kept={} skipped={} outside_g_range={}",
        rows.len(),
        cat.ra.len(),
        skipped,
        out_of_range
    );
    if cat.ra.len() < 100 {
        return Err(format!("catalogue too small: {} rows", cat.ra.len()));
    }
    Ok(cat)
}

const ARCSEC_PER_RAD: f64 = 206_264.806_247_096_36;

fn centroid(cat: &Catalog) -> (f64, f64) {
    let d2r = std::f64::consts::PI / 180.0;
    let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
    for i in 0..cat.ra.len() {
        let (ra, dec) = (cat.ra[i] * d2r, cat.dec[i] * d2r);
        sx += dec.cos() * ra.cos();
        sy += dec.cos() * ra.sin();
        sz += dec.sin();
    }
    let r = (sx * sx + sy * sy + sz * sz).sqrt();
    let dec = (sz / r).asin();
    let ra = sy.atan2(sx);
    let ra = if ra < 0.0 {
        ra + 2.0 * std::f64::consts::PI
    } else {
        ra
    };
    (ra / d2r, dec / d2r)
}

fn project(cat: &Catalog, ra0: f64, dec0: f64) -> Vec<(f64, f64)> {
    let d2r = std::f64::consts::PI / 180.0;
    let (r0, d0) = (ra0 * d2r, dec0 * d2r);
    let (s0, c0) = (d0.sin(), d0.cos());
    let mut out = Vec::with_capacity(cat.ra.len());
    for i in 0..cat.ra.len() {
        let dra = cat.ra[i] * d2r - r0;
        let dec = cat.dec[i] * d2r;
        let (sd, cd) = (dec.sin(), dec.cos());
        let k = sd * s0 + cd * c0 * dra.cos();
        if k <= 1e-9 {
            continue;
        }
        let xi = cd * dra.sin() / k;
        let eta = (sd * c0 - cd * s0 * dra.cos()) / k;
        out.push((xi * ARCSEC_PER_RAD, eta * ARCSEC_PER_RAD));
    }
    out
}
// ----------------------------------------------------------------------------- fft

struct Plan1 {
    n: usize,
    rev: Vec<u32>,
    cos: Vec<f64>,
    sin: Vec<f64>,
}

impl Plan1 {
    fn new(n: usize) -> Self {
        let bits = n.trailing_zeros();
        let mut rev = vec![0u32; n];
        for (i, r) in rev.iter_mut().enumerate() {
            *r = (i as u32).reverse_bits() >> (32 - bits);
        }
        let mut cos = vec![0.0; n / 2 + 1];
        let mut sin = vec![0.0; n / 2 + 1];
        for k in 0..=(n / 2) {
            let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            cos[k] = t.cos();
            sin[k] = t.sin();
        }
        Self { n, rev, cos, sin }
    }

    fn fft1(&self, re: &mut [f64], im: &mut [f64], inverse: bool) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i] as usize;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2usize;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            let mut start = 0usize;
            while start < n {
                let mut k = 0usize;
                for j in 0..half {
                    let wr = self.cos[k];
                    let wi = if inverse { self.sin[k] } else { -self.sin[k] };
                    let a = start + j;
                    let b = a + half;
                    let (br, bi) = (re[b], im[b]);
                    let tr = br * wr - bi * wi;
                    let ti = br * wi + bi * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                    k += step;
                }
                start += len;
            }
            len <<= 1;
        }
        if inverse {
            let inv = 1.0 / n as f64;
            for i in 0..n {
                re[i] *= inv;
                im[i] *= inv;
            }
        }
    }
}

fn transpose_blocked(src: &[f64], dst: &mut [f64], n: usize) {
    const B: usize = 32;
    for bi in (0..n).step_by(B) {
        let imax = (bi + B).min(n);
        for bj in (0..n).step_by(B) {
            let jmax = (bj + B).min(n);
            for i in bi..imax {
                let row = i * n;
                for j in bj..jmax {
                    dst[j * n + i] = src[row + j];
                }
            }
        }
    }
}

struct Fft2 {
    plan: Plan1,
    n: usize,
    tr_re: Vec<f64>,
    tr_im: Vec<f64>,
}

impl Fft2 {
    fn new(n: usize) -> Self {
        Self {
            plan: Plan1::new(n),
            n,
            tr_re: vec![0.0; n * n],
            tr_im: vec![0.0; n * n],
        }
    }
    /// unnormalised forward transform along both axes
    fn forward(&mut self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n;
        for r in 0..n {
            let (lo, hi) = (r * n, (r + 1) * n);
            self.plan.fft1(&mut re[lo..hi], &mut im[lo..hi], false);
        }
        transpose_blocked(re, &mut self.tr_re, n);
        transpose_blocked(im, &mut self.tr_im, n);
        for r in 0..n {
            let (lo, hi) = (r * n, (r + 1) * n);
            self.plan
                .fft1(&mut self.tr_re[lo..hi], &mut self.tr_im[lo..hi], false);
        }
        transpose_blocked(&self.tr_re, re, n);
        transpose_blocked(&self.tr_im, im, n);
    }
}

// ----------------------------------------------------------------------------- density grids

fn rasterize(
    re: &mut [f64],
    n: usize,
    inv_bin: f64,
    pts: &[(f64, f64)],
    sign_x: f64,
    cos: f64,
    sin: f64,
) {
    let nn = n as i64;
    for &(x0, y0) in pts {
        let (u0, v0) = (x0 * sign_x, y0);
        let u = (cos * u0 - sin * v0) * inv_bin - 0.5;
        let v = (sin * u0 + cos * v0) * inv_bin - 0.5;
        let f0 = u.floor();
        let g0 = v.floor();
        let fx = u - f0;
        let fy = v - g0;
        let i0 = (f0 as i64).rem_euclid(nn) as usize;
        let i1 = ((f0 as i64) + 1).rem_euclid(nn) as usize;
        let j0 = (g0 as i64).rem_euclid(nn) as usize;
        let j1 = ((g0 as i64) + 1).rem_euclid(nn) as usize;
        let (a, b) = (1.0 - fx, fx);
        let (c, d) = (1.0 - fy, fy);
        re[i0 * n + j0] += a * c;
        re[i1 * n + j0] += b * c;
        re[i0 * n + j1] += a * d;
        re[i1 * n + j1] += b * d;
    }
}

fn gauss_kernel(sigma: f64) -> Vec<f64> {
    let r = (3.0 * sigma).ceil().max(1.0) as usize;
    let mut k: Vec<f64> = (0..=(2 * r))
        .map(|i| {
            let d = i as f64 - r as f64;
            (-0.5 * d * d / (sigma * sigma)).exp()
        })
        .collect();
    let s: f64 = k.iter().sum();
    for w in k.iter_mut() {
        *w /= s;
    }
    k
}

fn smooth(re: &mut [f64], n: usize, kern: &[f64], tmp: &mut [f64]) {
    let kl = kern.len();
    if kl <= 1 {
        return;
    }
    let r = kl / 2;
    let mut idx = vec![0usize; kl];
    for i in 0..n {
        for (t, s) in idx.iter_mut().enumerate() {
            *s = (i + n + t - r) % n;
        }
        for j in 0..n {
            let mut acc = 0.0;
            for (t, w) in kern.iter().enumerate() {
                acc += re[idx[t] * n + j] * w;
            }
            tmp[i * n + j] = acc;
        }
    }
    for j in 0..n {
        for (t, s) in idx.iter_mut().enumerate() {
            *s = (j + n + t - r) % n;
        }
        for i in 0..n {
            let base = i * n;
            let mut acc = 0.0;
            for (t, w) in kern.iter().enumerate() {
                acc += tmp[base + idx[t]] * w;
            }
            re[base + j] = acc;
        }
    }
}

struct MapStats {
    max: f64,
    max_idx: usize,
    median: f64,
    sigma: f64,
}

fn map_stats(map: &[f64], buf: &mut Vec<f64>) -> MapStats {
    let (mut max, mut max_idx) = (f64::NEG_INFINITY, 0usize);
    for (i, &v) in map.iter().enumerate() {
        if v > max {
            max = v;
            max_idx = i;
        }
    }
    buf.clear();
    buf.extend_from_slice(map);
    buf.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = buf.len();
    let med = buf[n / 2];
    let mut dev: Vec<f64> = buf.iter().map(|v| (v - med).abs()).collect();
    dev.sort_unstable_by(|a, b| a.total_cmp(b));
    let mad = dev[dev.len() / 2];
    MapStats {
        max,
        max_idx,
        median: med,
        sigma: 1.4826 * mad,
    }
}

// ----------------------------------------------------------------------------- trials

#[derive(Clone, Copy, Debug)]
struct Trial {
    parity: u8,
    roll: f64, // radians
    scale: f64,
}

#[derive(Clone, Debug)]
struct Hit {
    trial: Trial,
    maxv: f64,
    sigma: f64,
    median: f64,
    peak_sigma: f64,
    t_bin: (usize, usize),
    t_px: (f64, f64),
    n: usize,
}

impl Hit {
    fn summary(&self, frame: f64) -> Value {
        json!({
            "roll_deg": fin(self.trial.roll.to_degrees()),
            "scale_arcsec_per_px": fin(self.trial.scale),
            "parity": self.trial.parity,
            "peak_value": fin(self.maxv),
            "median_value": fin(self.median),
            "map_sigma": fin(self.sigma),
            "peak_sigma": fin(self.peak_sigma),
            "grid": self.n,
            "t_bin": [self.t_bin.0, self.t_bin.1],
            "t_px": [fin(self.t_px.0), fin(self.t_px.1)],
            "t_offset_from_frame_centre_px": [
                fin(self.t_px.0 - frame / 2.0),
                fin(self.t_px.1 - frame / 2.0)
            ],
            "t_signed_px": [
                fin(signed(self.t_bin.0, self.n) as f64 * frame / self.n as f64),
                fin(signed(self.t_bin.1, self.n) as f64 * frame / self.n as f64)
            ]
        })
    }
}

fn signed(k: usize, n: usize) -> i64 {
    if k >= n / 2 {
        k as i64 - n as i64
    } else {
        k as i64
    }
}

struct Worker {
    n: usize,
    fft: Fft2,
    cat_re: Vec<f64>,
    cat_im: Vec<f64>,
    g_re: Vec<f64>,
    g_im: Vec<f64>,
    tmp: Vec<f64>,
    kern: Vec<f64>,
    stat_buf: Vec<f64>,
    inv: f64,
}

impl Worker {
    fn new(n: usize, smooth_sigma: f64) -> Self {
        Self {
            n,
            fft: Fft2::new(n),
            cat_re: vec![0.0; n * n],
            cat_im: vec![0.0; n * n],
            g_re: vec![0.0; n * n],
            g_im: vec![0.0; n * n],
            tmp: vec![0.0; n * n],
            kern: if smooth_sigma > 0.0 {
                gauss_kernel(smooth_sigma)
            } else {
                Vec::new()
            },
            stat_buf: Vec::new(),
            inv: 1.0 / ((n * n) as f64),
        }
    }

    fn transform_catalog(&mut self, cat: &[(f64, f64)], tr: &Trial, inv_bin_scale: f64) {
        let n = self.n;
        self.cat_re.fill(0.0);
        let sign_x = if tr.parity == 1 { -1.0 } else { 1.0 };
        rasterize(
            &mut self.cat_re,
            n,
            inv_bin_scale,
            cat,
            sign_x,
            tr.roll.cos(),
            tr.roll.sin(),
        );
        if self.kern.len() > 1 {
            let (re, tmp, kern) = (&mut self.cat_re, &mut self.tmp, &self.kern);
            smooth(re, n, kern, tmp);
        }
        self.cat_im.fill(0.0);
        self.fft.forward(&mut self.cat_re, &mut self.cat_im);
    }

    fn correlate(&mut self, det_re: &[f64], det_im: &[f64], want_stats: bool) -> (f64, usize, f64, f64) {
        let n2 = self.n * self.n;
        for i in 0..n2 {
            let (ar, ai) = (det_re[i], det_im[i]);
            let (br, bi) = (self.cat_re[i], self.cat_im[i]);
            self.g_re[i] = ar * br + ai * bi;
            self.g_im[i] = ar * bi - ai * br;
        }
        let mut g_re = std::mem::take(&mut self.g_re);
        let mut g_im = std::mem::take(&mut self.g_im);
        self.fft.forward(&mut g_re, &mut g_im);
        std::mem::swap(&mut self.g_re, &mut g_re);
        std::mem::swap(&mut self.g_im, &mut g_im);
        if want_stats {
            let st = map_stats(&self.g_re, &mut self.stat_buf);
            (
                st.max * self.inv,
                st.max_idx,
                st.sigma * self.inv,
                st.median * self.inv,
            )
        } else {
            let inv = self.inv;
            let mut best = f64::NEG_INFINITY;
            let mut bidx = 0usize;
            for (i, &v) in self.g_re.iter().enumerate() {
                let s = v * inv;
                if s > best {
                    best = s;
                    bidx = i;
                }
            }
            (best, bidx, f64::NAN, f64::NAN)
        }
    }
}

struct Prepared {
    det_re: Vec<f64>,
    det_im: Vec<f64>,
    n: usize,
}

fn prepare_det(det: &[(f64, f64)], n: usize, frame: f64, smooth_sigma: f64) -> Prepared {
    let b = frame / n as f64;
    let mut det_re = vec![0.0; n * n];
    let mut det_im = vec![0.0; n * n];
    rasterize(&mut det_re, n, 1.0 / b, det, 1.0, 1.0, 0.0);
    if smooth_sigma > 0.0 {
        let kern = gauss_kernel(smooth_sigma);
        let mut tmp = vec![0.0; n * n];
        smooth(&mut det_re, n, &kern, &mut tmp);
    }
    let mut f = Fft2::new(n);
    f.forward(&mut det_re, &mut det_im);
    Prepared { det_re, det_im, n }
}

#[derive(Clone, Copy, Debug)]
struct TrialEval {
    flat: usize,
    maxv: f64,
    sigma: f64,
    median: f64,
    max_idx: usize,
    sub_dx: f64,
    sub_dy: f64,
}

fn hit_from_eval(cfg: &Cfg, n: usize, tr: Trial, e: &TrialEval) -> Hit {
    let b = cfg.frame / n as f64;
    let (bx, by) = (e.max_idx / n, e.max_idx % n);
    Hit {
        trial: tr,
        maxv: e.maxv,
        sigma: e.sigma,
        median: e.median,
        peak_sigma: if e.sigma > 0.0 {
            (e.maxv - e.median) / e.sigma
        } else {
            f64::INFINITY
        },
        t_bin: (bx, by),
        t_px: ((bx as f64 + e.sub_dx) * b, (by as f64 + e.sub_dy) * b),
        n,
    }
}

struct EvalOut {
    #[allow(dead_code)]
    lo: usize,
    max_per_trial: Vec<f64>,
    top: Vec<(f64, usize)>,
    evals: Vec<TrialEval>,
    sampled: Vec<f64>,
}

fn eval_trials(
    cfg: &Cfg,
    prepared: &Prepared,
    cat: &[(f64, f64)],
    trials: &[Trial],
    threads: usize,
    want_top: usize,
    want_stats: bool,
    sample_stride: usize,
    label: &str,
) -> Vec<EvalOut> {
    let n = prepared.n;
    let b = cfg.frame / n as f64;
    let total = trials.len();
    let counter = AtomicUsize::new(0);
    let counter_ref = &counter;
    let chunk = total.div_ceil(threads.max(1));
    let outs: Vec<EvalOut> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..threads {
            let lo = t * chunk;
            let hi = ((t + 1) * chunk).min(total);
            if lo >= hi {
                continue;
            }
            let det_re = &prepared.det_re;
            let det_im = &prepared.det_im;
            handles.push(s.spawn(move || {
                let mut w = Worker::new(n, cfg.smooth_sigma);
                let mut out = EvalOut {
                    lo,
                    max_per_trial: Vec::new(),
                    top: Vec::new(),
                    evals: Vec::new(),
                    sampled: Vec::new(),
                };
                if !want_stats {
                    out.max_per_trial = vec![0.0; hi - lo];
                }
                for flat in lo..hi {
                    let tr = trials[flat];
                    w.transform_catalog(cat, &tr, 1.0 / b / tr.scale);
                    let (maxv, max_idx, sigma, med) = w.correlate(det_re, det_im, want_stats);
                    if !want_stats {
                        out.max_per_trial[flat - lo] = maxv;
                    }
                    if want_top > 0 && !want_stats {
                        if out.top.len() < want_top {
                            out.top.push((maxv, flat));
                            out.top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                        } else if maxv > out.top[want_top - 1].0 {
                            out.top.pop();
                            out.top.push((maxv, flat));
                            out.top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                        }
                    }
                    if want_stats {
                        let (xi, yi) = ((max_idx / n) as i64, (max_idx % n) as i64);
                        let get = |x: i64, y: i64| -> f64 {
                            let xx = x.rem_euclid(n as i64) as usize;
                            let yy = y.rem_euclid(n as i64) as usize;
                            w.g_re[xx * n + yy] * w.inv
                        };
                        let c = get(xi, yi);
                        let dxn = 2.0 * c - get(xi + 1, yi) - get(xi - 1, yi);
                        let dyn_ = 2.0 * c - get(xi, yi + 1) - get(xi, yi - 1);
                        let mut dx = 0.5 * (get(xi + 1, yi) - get(xi - 1, yi)) / dxn;
                        let mut dy = 0.5 * (get(xi, yi + 1) - get(xi, yi - 1)) / dyn_;
                        if !dx.is_finite() || dx.abs() > 0.5 {
                            dx = 0.0;
                        }
                        if !dy.is_finite() || dy.abs() > 0.5 {
                            dy = 0.0;
                        }
                        out.evals.push(TrialEval {
                            flat,
                            maxv,
                            sigma,
                            median: med,
                            max_idx,
                            sub_dx: dx,
                            sub_dy: dy,
                        });
                    } else if sample_stride > 0 && flat % sample_stride == 0 {
                        let st = map_stats(&w.g_re, &mut w.stat_buf);
                        out.sampled.push(st.sigma * w.inv);
                    }
                    let done = counter_ref.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 50_000 == 0 {
                        eprintln!("  [{label}] {done}/{total}");
                    }
                }
                out
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    outs
}

// ----------------------------------------------------------------------------- hierarchical search

fn roll_step_for(n: usize, frame: f64, zeta: f64) -> f64 {
    let r_max = 0.5 * frame * std::f64::consts::SQRT_2;
    let raw = zeta * (frame / n as f64) / r_max;
    let two_pi = 2.0 * std::f64::consts::PI;
    two_pi / (two_pi / raw).ceil()
}

fn scale_step_for(n: usize, frame: f64, zeta: f64, s_ref: f64) -> f64 {
    let r_max = 0.5 * frame * std::f64::consts::SQRT_2;
    zeta * (frame / n as f64) * s_ref / r_max
}

struct SearchOut {
    refined: Vec<Hit>,
    ladder: Value,
    n_trials_coarse: usize,
    sampled_sigma: Vec<f64>,
    roll_step0: f64,
    scale_step0: f64,
    n_roll: usize,
    n_scale: usize,
    scales: (f64, f64),
}

fn dedup(hits: &mut Vec<Hit>, roll_tol: f64, scale_tol_frac: f64, t_tol_px: f64) -> Vec<Hit> {
    hits.sort_unstable_by(|a, b| b.peak_sigma.total_cmp(&a.peak_sigma));
    let mut out: Vec<Hit> = Vec::new();
    for h in hits.iter() {
        let dup = out.iter().any(|o| {
            o.trial.parity == h.trial.parity
                && (o.trial.roll - h.trial.roll).abs() < roll_tol
                && ((o.trial.scale - h.trial.scale) / h.trial.scale).abs() < scale_tol_frac
                && ((o.t_px.0 - h.t_px.0).powi(2) + (o.t_px.1 - h.t_px.1).powi(2)).sqrt() < t_tol_px
        });
        if !dup {
            out.push(h.clone());
        }
    }
    out
}

fn hierarchical_search(
    cfg: &Cfg,
    det: &[(f64, f64)],
    cat: &[(f64, f64)],
    label: &str,
    keep: &[usize],
    scale_shift_frac: f64,
) -> SearchOut {
    let s_ref = 0.5 * (cfg.scale_min + cfg.scale_max);
    let mut n = cfg.search_bins;
    let roll_step = roll_step_for(n, cfg.frame, cfg.roll_zeta);
    let n_roll = ((2.0 * std::f64::consts::PI) / roll_step).ceil() as usize;
    let roll_step = 2.0 * std::f64::consts::PI / n_roll as f64;
    let scale_step = scale_step_for(n, cfg.frame, cfg.roll_zeta, s_ref);
    let shift = scale_shift_frac * s_ref;
    let mut scales = Vec::new();
    let mut s = cfg.scale_min + shift;
    while s <= cfg.scale_max + 1e-12 {
        scales.push(s);
        s += scale_step;
    }
    if scales.is_empty() {
        scales.push(s_ref + shift);
    }
    let mut trials = Vec::with_capacity(2 * n_roll * scales.len());
    for parity in 0u8..2 {
        for ri in 0..n_roll {
            let roll = ri as f64 * roll_step;
            for &sc in &scales {
                trials.push(Trial {
                    parity,
                    roll,
                    scale: sc,
                });
            }
        }
    }
    let total = trials.len();
    let stride = if cfg.sample_sigma == 0 {
        0
    } else {
        (total / cfg.sample_sigma).max(1)
    };
    let mut ladder: Vec<Value> = Vec::new();
    eprintln!(
        "[{label}] stage0 n={n} n_roll={n_roll}({:.4} deg/step) n_scale={}({:.4} px/step) trials={total}",
        roll_step.to_degrees(),
        scales.len(),
        scale_step
    );
    let t0 = Instant::now();
    let prepared0 = prepare_det(det, n, cfg.frame, cfg.smooth_sigma);
    let outs = eval_trials(
        cfg, &prepared0, cat, &trials, cfg.threads, keep[0], false, stride, label,
    );
    let mut tops: Vec<(f64, usize)> = Vec::new();
    let mut sampled: Vec<f64> = Vec::new();
    for o in outs {
        tops.extend(o.top);
        sampled.extend(o.sampled);
    }
    tops.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    tops.truncate(keep[0]);
    let seed_trials: Vec<Trial> = tops.iter().map(|(_, f)| trials[*f]).collect();
    let coarse_evals = eval_trials(
        cfg,
        &prepared0,
        cat,
        &seed_trials,
        cfg.threads.min(8),
        0,
        true,
        0,
        &format!("{label}/s0stats"),
    );
    let mut coarse_hits: Vec<Hit> = Vec::new();
    for o in coarse_evals {
        for e in o.evals {
            coarse_hits.push(hit_from_eval(cfg, n, seed_trials[e.flat], &e));
        }
    }
    let coarse_hits = dedup(
        &mut coarse_hits,
        roll_step * 0.75,
        0.006,
        2.0 * cfg.frame / n as f64,
    );
    ladder.push(json!({
        "stage": 0,
        "n": n,
        "roll_step_deg": fin(roll_step.to_degrees()),
        "scale_step": fin(scale_step),
        "n_roll": n_roll,
        "n_scale": scales.len(),
        "trials": total,
        "seconds": fin(t0.elapsed().as_secs_f64()),
        "kept": coarse_hits.len(),
        "best_peak_sigma": fin(coarse_hits.first().map(|h| h.peak_sigma).unwrap_or(f64::NAN)),
        "sampled_map_sigma_median": fin(median(&sampled)),
        "raw_top": tops.iter().take(12).map(|(v, f)| json!({
            "raw_max": fin(*v),
            "parity": trials[*f].parity,
            "roll_deg": fin(trials[*f].roll.to_degrees()),
            "scale": fin(trials[*f].scale),
        })).collect::<Vec<_>>(),
    }));
    eprintln!(
        "[{label}] stage0 {:.1}s best_peak_sigma={:.2}",
        t0.elapsed().as_secs_f64(),
        coarse_hits.first().map(|h| h.peak_sigma).unwrap_or(f64::NAN)
    );
    let mut seeds: Vec<Trial> = coarse_hits.iter().map(|h| h.trial).collect();
    let mut current = coarse_hits;
    let mut cur_roll_step = roll_step;
    let mut cur_scale_step = scale_step;
    let mut level = 1usize;
    while n < cfg.bins && !seeds.is_empty() {
        n *= 2;
        let sub = 4.0;
        let roll_step_new = cur_roll_step / sub;
        let scale_step_new = cur_scale_step / sub;
        let mut cand: Vec<Trial> = Vec::new();
        for sd in seeds.iter() {
            for i in -4i32..=4 {
                let roll = sd.roll + i as f64 * roll_step_new;
                for j in -4i32..=4 {
                    let scale = sd.scale + j as f64 * scale_step_new;
                    if scale <= 0.0 {
                        continue;
                    }
                    cand.push(Trial {
                        parity: sd.parity,
                        roll,
                        scale,
                    });
                }
            }
        }
        let t1 = Instant::now();
        let prepared = prepare_det(det, n, cfg.frame, cfg.smooth_sigma);
        let threads = if n * n * 8 * 6 > 300_000_000 {
            cfg.threads.min(6)
        } else {
            cfg.threads
        };
        let outs = eval_trials(
            cfg,
            &prepared,
            cat,
            &cand,
            threads,
            0,
            true,
            0,
            &format!("{label}/stage{level}"),
        );
        let mut hits: Vec<Hit> = Vec::new();
        for o in outs {
            for e in o.evals {
                hits.push(hit_from_eval(cfg, n, cand[e.flat], &e));
            }
        }
        let kept = keep.get(level).copied().unwrap_or(1);
        let hits = dedup(&mut hits, roll_step_new * 0.75, 0.004, 3.0 * cfg.frame / n as f64);
        ladder.push(json!({
            "stage": level,
            "n": n,
            "roll_step_deg": fin(roll_step_new.to_degrees()),
            "scale_step": fin(scale_step_new),
            "seeds": seeds.len(),
            "trials": cand.len(),
            "seconds": fin(t1.elapsed().as_secs_f64()),
            "kept_in_report": hits.len().min(kept),
            "best_peak_sigma": fin(hits.first().map(|h| h.peak_sigma).unwrap_or(f64::NAN)),
        }));
        eprintln!(
            "[{label}] stage{level} n={n} {} trials {:.1}s best_peak_sigma={:.2}",
            cand.len(),
            t1.elapsed().as_secs_f64(),
            hits.first().map(|h| h.peak_sigma).unwrap_or(f64::NAN)
        );
        seeds = hits.iter().take(kept).map(|h| h.trial).collect();
        current = hits.into_iter().take(kept.max(1)).collect();
        cur_roll_step = roll_step_new;
        cur_scale_step = scale_step_new;
        level += 1;
    }
    let _ = cur_scale_step;
    SearchOut {
        refined: current,
        ladder: Value::Array(ladder),
        n_trials_coarse: total,
        sampled_sigma: sampled,
        roll_step0: roll_step,
        scale_step0: scale_step,
        n_roll,
        n_scale: scales.len(),
        scales: (*scales.first().unwrap_or(&0.0), *scales.last().unwrap_or(&0.0)),
    }
}

// ----------------------------------------------------------------------------- synthetic

#[derive(Clone, Copy, Debug)]
struct Truth {
    roll: f64,
    scale: f64,
    parity: u8,
    tx: f64,
    ty: f64,
}

fn synthesize(
    cat: &[(f64, f64)],
    t: &Truth,
    frame: f64,
    mask_telemetry: bool,
    jitter: f64,
    completeness: f64,
    fp_frac: f64,
    rng: &mut Rng,
) -> Vec<(f64, f64)> {
    let (c, s) = (t.roll.cos(), t.roll.sin());
    let sign_x = if t.parity == 1 { -1.0 } else { 1.0 };
    let mut out = Vec::new();
    for &(xi, eta) in cat {
        let (u, v) = (xi * sign_x, eta);
        let x = (c * u - s * v) / t.scale + t.tx;
        let y = (s * u + c * v) / t.scale + t.ty;
        if !(0.0..frame).contains(&x) || !(0.0..frame).contains(&y) {
            continue;
        }
        if rng.uniform() > completeness {
            continue;
        }
        let x = x + if jitter > 0.0 { rng.normal() * jitter } else { 0.0 };
        let y = y + if jitter > 0.0 { rng.normal() * jitter } else { 0.0 };
        if !(0.0..frame).contains(&x) || !(0.0..frame).contains(&y) {
            continue;
        }
        if mask_telemetry && x < 104.0 && y < 104.0 {
            continue;
        }
        out.push((x, y));
    }
    let n_fp = (fp_frac * out.len() as f64).round() as usize;
    for _ in 0..n_fp {
        out.push((rng.uniform() * frame, rng.uniform() * frame));
    }
    out
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut d = (a - b) % two_pi;
    if d > std::f64::consts::PI {
        d -= two_pi;
    }
    if d < -std::f64::consts::PI {
        d += two_pi;
    }
    d
}

// ----------------------------------------------------------------------------- match audit
//
// Independent verification of one candidate solution: map every catalogue star into the detector
// with the candidate parameters and look for a detection inside a small radius. A real solution
// produces an excess of sub-pixel coincidences; a chance alignment does not. The chance level is
// measured by repeating the same audit with the mapping displaced by a fixed offset.

struct DetIndex {
    cell: f64,
    map: HashMap<(i32, i32), Vec<u32>>,
}

impl DetIndex {
    fn build(pts: &[(f64, f64)], cell: f64) -> Self {
        let mut map: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        for (i, &(x, y)) in pts.iter().enumerate() {
            let key = ((x / cell).floor() as i32, (y / cell).floor() as i32);
            map.entry(key).or_default().push(i as u32);
        }
        Self { cell, map }
    }
    fn nearest(&self, pts: &[(f64, f64)], x: f64, y: f64, rmax: f64) -> f64 {
        let cx = (x / self.cell).floor() as i32;
        let cy = (y / self.cell).floor() as i32;
        let span = (rmax / self.cell).ceil() as i32;
        let mut best = f64::INFINITY;
        for i in -span..=span {
            for j in -span..=span {
                if let Some(v) = self.map.get(&(cx + i, cy + j)) {
                    for &idx in v.iter() {
                        let (px, py) = pts[idx as usize];
                        let d = ((px - x).powi(2) + (py - y).powi(2)).sqrt();
                        if d < best {
                            best = d;
                        }
                    }
                }
            }
        }
        best
    }
}

struct AuditOne {
    radii: [f64; 5],
    counts: [usize; 5],
    n_catalog_in_frame: usize,
    median_dist: f64,
    inner_counts: [usize; 5],
    inner_catalog: usize,
    inner_radius: f64,
}

fn audit_positions(
    det: &[(f64, f64)],
    cat: &[(f64, f64)],
    truth: &Truth,
    frame: f64,
    offset: (f64, f64),
) -> AuditOne {
    let radii = [1.0, 2.0, 3.0, 5.0, 8.0];
    let mut counts = [0usize; 5];
    let mut inner_counts = [0usize; 5];
    let mut inner_catalog = 0usize;
    let mut n_in = 0usize;
    let mut dists: Vec<f64> = Vec::new();
    let idx = DetIndex::build(det, 16.0);
    let (c, s) = (truth.roll.cos(), truth.roll.sin());
    let sign_x = if truth.parity == 1 { -1.0 } else { 1.0 };
    let (cx, cy) = (frame / 2.0, frame / 2.0);
    for &(xi, eta) in cat {
        let (u, v) = (xi * sign_x, eta);
        let x = (c * u - s * v) / truth.scale + truth.tx + offset.0;
        let y = (s * u + c * v) / truth.scale + truth.ty + offset.1;
        if !(0.0..frame).contains(&x) || !(0.0..frame).contains(&y) {
            continue;
        }
        n_in += 1;
        let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        let inner = r < frame / 4.0;
        if inner {
            inner_catalog += 1;
        }
        let d = idx.nearest(det, x, y, radii[4]);
        if d <= radii[4] {
            dists.push(d);
        }
        for (k, &rr) in radii.iter().enumerate() {
            if d <= rr {
                counts[k] += 1;
                if inner {
                    inner_counts[k] += 1;
                }
            }
        }
    }
    AuditOne {
        radii,
        counts,
        n_catalog_in_frame: n_in,
        median_dist: median(&dists),
        inner_counts,
        inner_catalog,
        inner_radius: frame / 4.0,
    }
}

fn audit_json(a: &AuditOne) -> Value {
    let mut ser = Map::new();
    for (k, &rr) in a.radii.iter().enumerate() {
        ser.insert(format!("within_{rr:.0}px"), json!(a.counts[k]));
    }
    let mut inner = Map::new();
    for (k, &rr) in a.radii.iter().enumerate() {
        inner.insert(format!("within_{rr:.0}px"), json!(a.inner_counts[k]));
    }
    json!({
        "catalog_stars_in_frame": a.n_catalog_in_frame,
        "matched": Value::Object(ser),
        "median_nearest_detection_px": fin(a.median_dist),
        "inner_quarter": {
            "radius_px": fin(a.inner_radius),
            "catalog_stars": a.inner_catalog,
            "matched": Value::Object(inner),
        }
    })
}
// ----------------------------------------------------------------------------- main

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return;
    }
    let cfg = match parse_args(&args).and_then(|c| build_cfg(&c)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("rst19-astrometry-blind: {e}");
            std::process::exit(2);
        }
    };
    match run(&cfg, &args) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("rst19-astrometry-blind: {e}");
            std::process::exit(2);
        }
    }
}

fn config_json(cfg: &Cfg) -> Value {
    json!({
        "sources": cfg.sources.display().to_string(),
        "catalog": cfg.catalog.display().to_string(),
        "bins": cfg.bins,
        "search_bins": cfg.search_bins,
        "peak_ceiling": fin(cfg.peak_ceiling),
        "mask_telemetry": cfg.mask_telemetry,
        "scale_range": [fin(cfg.scale_min), fin(cfg.scale_max)],
        "roll_zeta": fin(cfg.roll_zeta),
        "smooth_sigma_bins": fin(cfg.smooth_sigma),
        "catalog_g_range": [fin(cfg.catalog_min_g), fin(cfg.catalog_max_g)],
        "frame_size_px": fin(cfg.frame),
        "keep": cfg.keep,
        "sample_sigma": cfg.sample_sigma,
        "control_trials": cfg.control_trials,
        "threads": cfg.threads,
        "seed": cfg.seed,
        "tag": cfg.tag,
    })
}

fn run(cfg: &Cfg, argv: &[String]) -> Result<i32, String> {
    let (epoch, stamp) = now_utc();
    eprintln!("run_id={} utc={}", cfg.run_id, stamp);
    std::fs::create_dir_all(&cfg.out_dir)
        .map_err(|e| format!("create {}: {e}", cfg.out_dir.display()))?;

    let src = load_sources(cfg)?;
    let det = src.points.clone();
    if det.len() < 100 {
        return Err(format!("too few detections: {}", det.len()));
    }
    let catalog = load_catalog(cfg)?;
    let (ra0, dec0) = match (cfg.ra0, cfg.dec0) {
        (Some(a), Some(d)) => (a, d),
        _ => centroid(&catalog),
    };
    let cat = project(&catalog, ra0, dec0);
    let cat_ext = cat
        .iter()
        .fold((0.0f64, 0.0f64), |a, p| (a.0.max(p.0.abs()), a.1.max(p.1.abs())));
    eprintln!(
        "projection centre ra0={ra0:.9} dec0={dec0:.9} catalogue_max_abs_arcsec=({:.1},{:.1})",
        cat_ext.0, cat_ext.1
    );

    if cfg.selftest {
        return selftest(cfg, &cat);
    }

    let manifest_inputs = json!({
        "sources": {
            "path": cfg.sources.display().to_string(),
            "sha256": sha256_file(&cfg.sources)?,
            "csv_rows": src.total_rows,
            "used_rows": det.len(),
            "rejected_quality": src.rejected_quality,
            "rejected_peak": src.rejected_peak,
            "telemetry_masked": src.telemetry_masked,
            "outside_frame": src.outside_frame,
        },
        "catalog": {
            "path": cfg.catalog.display().to_string(),
            "sha256": sha256_file(&cfg.catalog)?,
            "rows_used": cat.len(),
            "g_range": [fin(cfg.catalog_min_g), fin(cfg.catalog_max_g)],
        },
        "projection_centre": {"ra0_deg": fin(ra0), "dec0_deg": fin(dec0)},
        "frame_size_px": fin(cfg.frame),
    });

    let mut exit_code = 0i32;
    let mut baseline_ok = true;
    let mut baseline_value = Value::Null;

    if !cfg.skip_selfcal {
        let (rows, base) = run_selfcal(cfg, &cat)?;
        baseline_ok = base.success;
        baseline_value = json!({
            "success": base.success,
            "attainable_peak_sigma": fin(base.peak_sigma),
            "truth": base.truth,
            "recovered": base.recovered,
            "errors": base.errors,
        });
        let doc = json!({
            "utc": stamp,
            "run_id": cfg.run_id,
            "config": {
                "bins": cfg.bins,
                "search_bins": cfg.search_bins,
                "scale_range": [fin(cfg.scale_min), fin(cfg.scale_max)],
                "roll_zeta": fin(cfg.roll_zeta),
                "smooth_sigma_bins": fin(cfg.smooth_sigma),
                "keep": cfg.keep,
                "frame_size_px": fin(cfg.frame),
                "jitter_px": fin(cfg.jitter_px),
                "completeness": fin(cfg.completeness),
                "false_positive_frac": fin(cfg.fp_frac),
                "seed": cfg.seed,
                "catalog_points": cat.len(),
                "mask_telemetry": cfg.mask_telemetry,
            },
            "baseline": baseline_value,
            "rows": rows,
            "verdict": if base.success { "SELFCAL_OK" } else { "SELFCAL_FAILED" },
        });
        write_json(&cfg.out_dir.join("selfcal.json"), &doc)?;
    }

    if !baseline_ok {
        let doc = json!({
            "status": "METHOD_FAILED",
            "reason": "the built-in synthetic self-calibration did not recover a planted solution with this catalogue and grid; the real-data result must not be interpreted",
            "utc": stamp,
            "run_id": cfg.run_id,
            "config": config_json(cfg),
            "inputs": manifest_inputs,
            "selfcal": if cfg.skip_selfcal { Value::Null } else { json!("selfcal.json") },
            "wcs_emitted": false,
            "magnitudes_emitted": false,
        });
        write_json(&cfg.out_dir.join("blind_search.json"), &doc)?;
        write_manifest(cfg, argv, &stamp, epoch, &manifest_inputs, "METHOD_FAILED")?;
        eprintln!("status=METHOD_FAILED");
        return Ok(1);
    }

    let mut search_json = Value::Null;
    let mut audit_json_final = Value::Null;
    let mut best_real: Option<Hit> = None;
    if !cfg.skip_real {
        let out = hierarchical_search(cfg, &det, &cat, "real", &cfg.keep, cfg.scale_grid_shift_frac);
        best_real = out.refined.first().cloned();
        if let Some(h) = best_real.as_ref() {
            let truth = Truth {
                roll: h.trial.roll,
                scale: h.trial.scale,
                parity: h.trial.parity,
                tx: h.t_px.0,
                ty: h.t_px.1,
            };
            let on = audit_positions(&det, &cat, &truth, cfg.frame, (0.0, 0.0));
            let off = audit_positions(&det, &cat, &truth, cfg.frame, (37.0, 53.0));
            eprintln!(
                "[audit] on-target 2px={} 5px={}; control 2px={} 5px={}",
                on.counts[1], on.counts[3], off.counts[1], off.counts[3]
            );
            audit_json_final = json!({
                "radius_list_px": on.radii.to_vec(),
                "on_target": audit_json(&on),
                "control_offset_37_53_px": audit_json(&off),
            });
        }
        search_json = json!({
            "coarse_grid": {
                "bins": cfg.search_bins,
                "pixel_per_bin": fin(cfg.frame / cfg.search_bins as f64),
                "n_trials": out.n_trials_coarse,
                "n_roll": out.n_roll,
                "n_scale": out.n_scale,
                "roll_step_deg": fin(out.roll_step0.to_degrees()),
                "scale_step_arcsec_per_px": fin(out.scale_step0),
                "scale_grid_range": [fin(out.scales.0), fin(out.scales.1)],
                "sampled_map_sigma": {
                    "n": out.sampled_sigma.len(),
                    "median": fin(median(&out.sampled_sigma)),
                }
            },
            "ladder": out.ladder,
            "top": out.refined.iter().map(|h| h.summary(cfg.frame)).collect::<Vec<_>>(),
            "match_audit": audit_json_final,
        });
    }

    let mut controls: Vec<ControlRow> = Vec::new();
    let mut cstats = ControlStats::default();
    if !cfg.skip_controls {
        let (rows, stats) = run_controls(cfg, &det, &cat)?;
        controls = rows;
        cstats = stats;
    }

    let control_upper = cstats.null_max_peak_sigma;
    let control_p99 = if cstats.null_p99_peak_sigma.is_finite() {
        cstats.null_p99_peak_sigma
    } else {
        control_upper
    };
    let gate = if control_upper.is_finite() {
        (3.0 * control_upper.max(control_p99)).max(8.0)
    } else {
        8.0
    };
    let real_ps = best_real.as_ref().map(|h| h.peak_sigma).unwrap_or(f64::NAN);
    let ratio = if control_upper.is_finite() && control_upper > 0.0 {
        real_ps / control_upper
    } else {
        f64::NAN
    };
    // independent corroboration: an excess of sub-arcsecond coincidences over the displaced control
    let (on2, off2) = match audit_json_final.get("on_target") {
        Some(v) => (
            v["matched"]["within_2px"].as_u64().unwrap_or(0) as f64,
            audit_json_final["control_offset_37_53_px"]["matched"]["within_2px"]
                .as_u64()
                .unwrap_or(0) as f64,
        ),
        None => (0.0, 0.0),
    };
    let audit_ratio = if off2 > 0.0 { on2 / off2 } else { f64::INFINITY };
    let audit_ok = on2 >= 50.0 && audit_ratio >= 3.0;
    let passed_task_gate = !cfg.skip_real && best_real.is_some() && real_ps >= gate;
    let validated = passed_task_gate && audit_ok;
    let status = if validated { "VALIDATED" } else { "UNVALIDATED" };
    if !validated {
        exit_code = 1;
    }
    let verdict = json!({
        "status": status,
        "gate": {
            "rule": "real peak_sigma >= max(8, 3 * control_upper_peak_sigma) AND self-calibration reconstructs AND match audit shows an excess of sub-2px coincidences",
            "real_peak_sigma": fin(real_ps),
            "control_upper_peak_sigma": fin(control_upper),
            "control_invariance_max_peak_sigma": fin(cstats.invariance_max_peak_sigma),
            "control_null_p99_peak_sigma": fin(control_p99),
            "control_null_sweeps_peak_sigma": cstats.null_sweeps.iter().map(|v| fin(*v)).collect::<Vec<_>>(),
            "ratio_real_to_control": fin(ratio),
            "required_peak_sigma": fin(gate),
            "shortfall_peak_sigma": fin(gate - real_ps),
            "selfcal_attainable_peak_sigma": baseline_value
                .get("attainable_peak_sigma")
                .cloned()
                .unwrap_or(Value::Null),
            "task_rule_passed": passed_task_gate,
            "match_audit": {
                "on_target_within_2px": on2,
                "control_within_2px": off2,
                "ratio": fin(audit_ratio),
                "passed": audit_ok,
            },
            "passed": validated,
        },
        "wcs_emitted": false,
        "magnitudes_emitted": false,
        "note": if validated {
            "solution passes the decision gate; a WCS may be fitted in a follow-up step (none is emitted here)"
        } else {
            "no WCS, no catalogue match and no magnitude is reported: the highest peak is not separable from the null controls"
        },
    });
    let doc = json!({
        "status": status,
        "utc": stamp,
        "run_id": cfg.run_id,
        "config": config_json(cfg),
        "inputs": manifest_inputs,
        "search": search_json,
        "controls": {
            "n_rows": controls.len(),
            "null_max_peak_sigma": fin(control_upper),
            "null_p99_peak_sigma": fin(control_p99),
            "null_sweeps_peak_sigma": cstats.null_sweeps.iter().map(|v| fin(*v)).collect::<Vec<_>>(),
            "invariance_max_peak_sigma": fin(cstats.invariance_max_peak_sigma),
            "per_family_max": cstats.families,
            "design_note": "a global translation or a mirror of the catalogue is a searched degree of freedom, not a null: the search must still find the same solution there. Those rows are reported as invariance checks and are excluded from the null upper bound; only per-point perturbation of the catalogue and random/jittered detections measure chance.",
        },
        "verdict": verdict,
    });
    write_json(&cfg.out_dir.join("blind_search.json"), &doc)?;
    write_controls_csv(&cfg.out_dir.join("controls.csv"), &controls)?;
    write_manifest(cfg, argv, &stamp, epoch, &manifest_inputs, status)?;
    eprintln!(
        "status={status} real_peak_sigma={real_ps:.2} control_upper={control_upper:.2} gate={gate:.2} ratio={ratio:.2} audit2px={on2:.0}/{off2:.0}"
    );
    Ok(exit_code)
}

// ----------------------------------------------------------------------------- selfcal

struct BaselineReport {
    success: bool,
    peak_sigma: f64,
    truth: Value,
    recovered: Value,
    errors: Value,
}

struct RowSpec {
    label: &'static str,
    jitter: f64,
    completeness: f64,
    fp: f64,
    scale_shift_frac: f64,
    parity: u8,
    scale: Option<f64>,
    /// place the truth roll exactly half a coarse step away from a grid node (worst case)
    snap_roll_half_step: bool,
}

fn selfcal_specs(cfg: &Cfg) -> Vec<RowSpec> {
    let base = RowSpec {
        label: "baseline",
        jitter: cfg.jitter_px,
        completeness: cfg.completeness,
        fp: cfg.fp_frac,
        scale_shift_frac: cfg.scale_grid_shift_frac,
        parity: 0,
        scale: None,
        snap_roll_half_step: false,
    };
    if cfg.selfcal_mode == "extend" {
        let step_frac = scale_step_for(cfg.search_bins, cfg.frame, cfg.roll_zeta, 9.1)
            / (0.5 * (cfg.scale_min + cfg.scale_max));
        let mut v = Vec::new();
        let mut push = |label: &'static str, j: f64, c: f64, f: f64, ss: f64, snap: bool| {
            v.push(RowSpec {
                label,
                jitter: j,
                completeness: c,
                fp: f,
                scale_shift_frac: ss,
                parity: 0,
                scale: None,
                snap_roll_half_step: snap,
            });
        };
        push("edge_jitter_9.0", 9.0, cfg.completeness, cfg.fp_frac, 0.0, false);
        push("edge_jitter_12.0", 12.0, cfg.completeness, cfg.fp_frac, 0.0, false);
        push("edge_completeness_0.02", cfg.jitter_px, 0.02, cfg.fp_frac, 0.0, false);
        push("edge_completeness_0.05", cfg.jitter_px, 0.05, cfg.fp_frac, 0.0, false);
        push("edge_false_positive_2.00", cfg.jitter_px, cfg.completeness, 2.0, 0.0, false);
        push("edge_false_positive_4.00", cfg.jitter_px, cfg.completeness, 4.0, 0.0, false);
        push("edge_scale_half_step", cfg.jitter_px, cfg.completeness, cfg.fp_frac, 0.5 * step_frac, false);
        push("edge_roll_half_step", cfg.jitter_px, cfg.completeness, cfg.fp_frac, 0.0, true);
        push("edge_roll_and_scale_half_step", cfg.jitter_px, cfg.completeness, cfg.fp_frac, 0.5 * step_frac, true);
        return v;
    }
    if cfg.selfcal_mode != "matrix" {
        return vec![base];
    }
    let mut v = vec![base];
    let mut push = |label: &'static str, j: f64, c: f64, f: f64, ss: f64, p: u8, sc: Option<f64>| {
        v.push(RowSpec {
            label,
            jitter: j,
            completeness: c,
            fp: f,
            scale_shift_frac: ss,
            parity: p,
            scale: sc,
            snap_roll_half_step: false,
        });
    };
    push("jitter_0.0", 0.0, cfg.completeness, cfg.fp_frac, cfg.scale_grid_shift_frac, 0, None);
    push("jitter_3.0", 3.0, cfg.completeness, cfg.fp_frac, cfg.scale_grid_shift_frac, 0, None);
    push("jitter_6.0", 6.0, cfg.completeness, cfg.fp_frac, cfg.scale_grid_shift_frac, 0, None);
    push("completeness_0.10", cfg.jitter_px, 0.10, cfg.fp_frac, cfg.scale_grid_shift_frac, 0, None);
    push("completeness_1.00", cfg.jitter_px, 1.00, cfg.fp_frac, cfg.scale_grid_shift_frac, 0, None);
    push("false_positive_0.00", cfg.jitter_px, cfg.completeness, 0.0, cfg.scale_grid_shift_frac, 0, None);
    push("false_positive_1.00", cfg.jitter_px, cfg.completeness, 1.0, cfg.scale_grid_shift_frac, 0, None);
    push("scale_node_offset_1pct", cfg.jitter_px, cfg.completeness, cfg.fp_frac, 0.01, 0, None);
    push("scale_node_offset_2pct", cfg.jitter_px, cfg.completeness, cfg.fp_frac, 0.02, 0, None);
    push("scale_node_offset_4pct", cfg.jitter_px, cfg.completeness, cfg.fp_frac, 0.04, 0, None);
    push("parity_mirror", cfg.jitter_px, cfg.completeness, cfg.fp_frac, cfg.scale_grid_shift_frac, 1, None);
    push("truth_scale_10.4", cfg.jitter_px, cfg.completeness, cfg.fp_frac, cfg.scale_grid_shift_frac, 0, Some(10.4));
    v
}

fn run_selfcal(cfg: &Cfg, cat: &[(f64, f64)]) -> Result<(Value, BaselineReport), String> {
    let specs = selfcal_specs(cfg);
    let mut rows: Vec<Value> = Vec::new();
    let mut baseline: Option<BaselineReport> = None;
    for (i, spec) in specs.iter().enumerate() {
        let scale = spec.scale.unwrap_or(cfg.selfcal_scale);
        let mut rng = Rng::new(cfg.seed.wrapping_add(i as u64).wrapping_mul(2_654_435_761));
        let mut roll = rng.uniform() * 2.0 * std::f64::consts::PI;
        if spec.snap_roll_half_step {
            let rs = roll_step_for(cfg.search_bins, cfg.frame, cfg.roll_zeta);
            roll = (roll / rs - 0.5).round() * rs + 0.5 * rs;
            roll = roll.rem_euclid(2.0 * std::f64::consts::PI);
        }
        let tx = cfg.frame / 2.0 + (rng.uniform() - 0.5) * 800.0;
        let ty = cfg.frame / 2.0 + (rng.uniform() - 0.5) * 800.0;
        let truth = Truth {
            roll,
            scale,
            parity: spec.parity,
            tx,
            ty,
        };
        let synth = synthesize(
            cat,
            &truth,
            cfg.frame,
            cfg.mask_telemetry,
            spec.jitter,
            spec.completeness,
            spec.fp,
            &mut rng,
        );
        eprintln!(
            "[selfcal {}] synthetic={} truth roll={:.3}deg scale={:.3} parity={} t=({:.1},{:.1})",
            spec.label,
            synth.len(),
            truth.roll.to_degrees(),
            truth.scale,
            truth.parity,
            truth.tx,
            truth.ty
        );
        let out = hierarchical_search(
            cfg,
            &synth,
            cat,
            &format!("selfcal:{}", spec.label),
            &[12, 4, 1],
            spec.scale_shift_frac,
        );
        let best = out.refined.first();
        let roll_tol = out.roll_step0.to_degrees() * 1.5;
        let mut row = Map::new();
        row.insert("label".into(), json!(spec.label));
        row.insert(
            "config".into(),
            json!({
                "jitter_px": fin(spec.jitter),
                "completeness": fin(spec.completeness),
                "false_positive_frac": fin(spec.fp),
                "scale_grid_shift_frac": fin(spec.scale_shift_frac),
                "truth": {
                    "roll_deg": fin(truth.roll.to_degrees()),
                    "scale_arcsec_per_px": fin(truth.scale),
                    "parity": truth.parity,
                    "t_px": [fin(truth.tx), fin(truth.ty)],
                },
                "synthetic_points": synth.len(),
                "coarse_trials": out.n_trials_coarse,
                "coarse_roll_step_deg": fin(out.roll_step0.to_degrees()),
                "coarse_scale_step": fin(out.scale_step0),
            }),
        );
        match best {
            Some(h) => {
                let droll = angle_diff(h.trial.roll, truth.roll).to_degrees();
                let dscale = h.trial.scale / truth.scale - 1.0;
                let dt = ((h.t_px.0 - truth.tx).powi(2) + (h.t_px.1 - truth.ty).powi(2)).sqrt();
                let ok = h.peak_sigma >= 8.0
                    && droll.abs() <= roll_tol
                    && dscale.abs() <= 0.02
                    && dt <= 3.0 * cfg.frame / cfg.bins as f64
                    && h.trial.parity == truth.parity;
                row.insert("peak_sigma".into(), fin(h.peak_sigma));
                row.insert("hit".into(), h.summary(cfg.frame));
                row.insert(
                    "errors".into(),
                    json!({
                        "roll_deg": fin(droll),
                        "scale_frac": fin(dscale),
                        "t_px": fin(dt),
                        "parity_correct": h.trial.parity == truth.parity,
                    }),
                );
                row.insert("success".into(), json!(ok));
                if i == 0 {
                    baseline = Some(BaselineReport {
                        success: ok,
                        peak_sigma: h.peak_sigma,
                        truth: json!({
                            "roll_deg": fin(truth.roll.to_degrees()),
                            "scale_arcsec_per_px": fin(truth.scale),
                            "parity": truth.parity,
                            "t_px": [fin(truth.tx), fin(truth.ty)],
                        }),
                        recovered: h.summary(cfg.frame),
                        errors: json!({
                            "roll_deg": fin(droll),
                            "scale_frac": fin(dscale),
                            "t_px": fin(dt),
                            "parity_correct": h.trial.parity == truth.parity,
                        }),
                    });
                }
            }
            None => {
                row.insert("peak_sigma".into(), fin(f64::NAN));
                row.insert("success".into(), json!(false));
                if i == 0 {
                    baseline = Some(BaselineReport {
                        success: false,
                        peak_sigma: 0.0,
                        truth: Value::Null,
                        recovered: Value::Null,
                        errors: Value::Null,
                    });
                }
            }
        }
        row.insert(
            "sampled_map_sigma_median".into(),
            fin(median(&out.sampled_sigma)),
        );
        rows.push(Value::Object(row));
    }
    Ok((
        Value::Array(rows),
        baseline.unwrap_or(BaselineReport {
            success: false,
            peak_sigma: 0.0,
            truth: Value::Null,
            recovered: Value::Null,
            errors: Value::Null,
        }),
    ))
}

// ----------------------------------------------------------------------------- controls

struct ControlRow {
    family: String,
    kind: &'static str,
    index: usize,
    peak_sigma: f64,
    peak_value: f64,
    sigma: f64,
    roll_deg: f64,
    scale: f64,
    parity: u8,
    t_px: (f64, f64),
}

struct ControlStats {
    /// strongest peak over the genuine null families only
    null_max_peak_sigma: f64,
    /// 99th percentile over the per-sweep null maxima (with few sweeps this equals the maximum)
    null_p99_peak_sigma: f64,
    null_sweeps: Vec<f64>,
    /// strongest peak over the invariance families (translation / mirror are searched degrees of
    /// freedom, so these must reproduce the real peak instead of measuring chance)
    invariance_max_peak_sigma: f64,
    families: Value,
}

impl Default for ControlStats {
    fn default() -> Self {
        Self {
            null_max_peak_sigma: f64::NAN,
            null_p99_peak_sigma: f64::NAN,
            null_sweeps: Vec::new(),
            invariance_max_peak_sigma: f64::NAN,
            families: Value::Null,
        }
    }
}

fn run_controls(
    cfg: &Cfg,
    det: &[(f64, f64)],
    cat: &[(f64, f64)],
) -> Result<(Vec<ControlRow>, ControlStats), String> {
    let mut rows: Vec<ControlRow> = Vec::new();
    let mut sampled: Vec<f64> = Vec::new();
    let s_ref = 0.5 * (cfg.scale_min + cfg.scale_max);
    let mut rng = Rng::new(cfg.seed ^ 0xC0FF_EE12);
    let keep = [10usize, 4, 1];

    for offset_px in [50.0f64, 200.0] {
        for k in 0..cfg.control_trials {
            let ang = rng.uniform() * 2.0 * std::f64::consts::PI;
            let dx = ang.cos() * offset_px * s_ref;
            let dy = ang.sin() * offset_px * s_ref;
            let shifted: Vec<(f64, f64)> = cat.iter().map(|&(x, y)| (x + dx, y + dy)).collect();
            let family = format!("translate_{offset_px:.0}px");
            record_control(
                cfg, det, &shifted, &family, "invariance", k, keep, &mut rows, &mut sampled,
            );
        }
    }
    for k in 0..cfg.control_trials {
        let j = 20.0 * s_ref;
        let jittered: Vec<(f64, f64)> = cat
            .iter()
            .map(|&(x, y)| {
                (
                    x + (rng.uniform() - 0.5) * 2.0 * j,
                    y + (rng.uniform() - 0.5) * 2.0 * j,
                )
            })
            .collect();
        record_control(
            cfg, det, &jittered, "catalog_jitter_20px", "null", k, keep, &mut rows, &mut sampled,
        );
    }
    for k in 0..cfg.control_trials {
        let mirrored: Vec<(f64, f64)> = cat.iter().map(|&(x, y)| (x, -y)).collect();
        record_control(
            cfg, det, &mirrored, "catalog_mirror", "invariance", k, keep, &mut rows, &mut sampled,
        );
    }
    for k in 0..cfg.control_trials {
        let j = 40.0;
        let jittered: Vec<(f64, f64)> = det
            .iter()
            .map(|&(x, y)| (x + (rng.uniform() - 0.5) * 2.0 * j, y + (rng.uniform() - 0.5) * 2.0 * j))
            .collect();
        record_control(
            cfg, det, &jittered, "detections_jitter_40px", "null", k, keep, &mut rows, &mut sampled,
        );
    }
    for k in 0..cfg.control_trials {
        let shuffled: Vec<(f64, f64)> = (0..det.len())
            .map(|_| (rng.uniform() * cfg.frame, rng.uniform() * cfg.frame))
            .collect();
        record_control(
            cfg, &shuffled, cat, "detections_random_uniform", "null", k, keep, &mut rows, &mut sampled,
        );
    }

    let mut families = Map::new();
    for r in rows.iter() {
        let key = format!("{}::{}", r.kind, r.family);
        let e = families.entry(key).or_insert(json!(f64::NEG_INFINITY));
        let cur = e.as_f64().unwrap_or(f64::NEG_INFINITY);
        if r.peak_sigma > cur {
            *e = fin(r.peak_sigma);
        }
    }
    let null_sweeps: Vec<f64> = rows
        .iter()
        .filter(|r| r.kind == "null")
        .map(|r| r.peak_sigma)
        .collect();
    let null_max = null_sweeps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let inv_max = rows
        .iter()
        .filter(|r| r.kind == "invariance")
        .map(|r| r.peak_sigma)
        .fold(f64::NEG_INFINITY, f64::max);
    Ok((
        rows,
        ControlStats {
            null_max_peak_sigma: null_max,
            null_p99_peak_sigma: percentile(&null_sweeps, 0.99),
            null_sweeps,
            invariance_max_peak_sigma: inv_max,
            families: Value::Object(families),
        },
    ))
}

fn record_control(
    cfg: &Cfg,
    det: &[(f64, f64)],
    cat: &[(f64, f64)],
    family: &str,
    kind: &'static str,
    index: usize,
    keep: [usize; 3],
    rows: &mut Vec<ControlRow>,
    sampled: &mut Vec<f64>,
) {
    let out = hierarchical_search(cfg, det, cat, &format!("control:{family}#{index}"), &keep, 0.0);
    if kind == "null" {
        sampled.extend(out.sampled_sigma.iter().copied());
    }
    if let Some(h) = out.refined.first() {
        eprintln!(
            "[control {kind} {family}#{index}] peak_sigma={:.2} roll={:.2}deg scale={:.3} parity={}",
            h.peak_sigma,
            h.trial.roll.to_degrees(),
            h.trial.scale,
            h.trial.parity
        );
        rows.push(ControlRow {
            family: family.to_string(),
            kind,
            index,
            peak_sigma: h.peak_sigma,
            peak_value: h.maxv,
            sigma: h.sigma,
            roll_deg: h.trial.roll.to_degrees(),
            scale: h.trial.scale,
            parity: h.trial.parity,
            t_px: h.t_px,
        });
    }
}

// ----------------------------------------------------------------------------- outputs

fn write_json(path: &Path, v: &Value) -> Result<(), String> {
    let s = serde_json::to_string_pretty(v).map_err(|e| format!("serialise: {e}"))?;
    std::fs::write(path, s + "\n").map_err(|e| format!("write {}: {e}", path.display()))
}

fn write_controls_csv(path: &Path, rows: &[ControlRow]) -> Result<(), String> {
    let mut f = std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    writeln!(
        f,
        "family,kind,index,peak_sigma,peak_value,map_sigma,roll_deg,scale_arcsec_per_px,parity,t_px_x,t_px_y"
    )
    .map_err(|e| e.to_string())?;
    for r in rows {
        writeln!(
            f,
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.3},{:.3}",
            r.family,
            r.kind,
            r.index,
            r.peak_sigma,
            r.peak_value,
            r.sigma,
            r.roll_deg,
            r.scale,
            r.parity,
            r.t_px.0,
            r.t_px.1
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn write_manifest(
    cfg: &Cfg,
    argv: &[String],
    stamp: &str,
    epoch: u64,
    inputs: &Value,
    status: &str,
) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let cfg_json = serde_json::to_string(&config_json(cfg)).unwrap_or_default();
    let manifest = json!({
        "run_id": cfg.run_id,
        "utc": stamp,
        "utc_epoch": epoch,
        "status": status,
        "command": format!("rst19-astrometry-blind {}", argv.join(" ")),
        "binary": {
            "path": exe.display().to_string(),
            "sha256": sha256_file(&exe)?,
            "bytes": std::fs::metadata(&exe).map(|m| m.len()).unwrap_or(0),
        },
        "config": config_json(cfg),
        "config_sha256": sha256_str(&cfg_json),
        "inputs": inputs,
        "outputs": [
            "blind_search.json",
            "selfcal.json",
            "controls.csv",
            "run_manifest.json"
        ],
        "notes": [
            "ra/dec are placed by a gnomonic projection about the catalogue centroid unless --ra0/--dec0 are given",
            "the reported translation is the detector position of that projection centre",
            "no WCS and no magnitude is emitted unless the decision gate passes"
        ],
    });
    write_json(&cfg.out_dir.join("run_manifest.json"), &manifest)
}

// ----------------------------------------------------------------------------- selftest

fn selftest(cfg: &Cfg, cat: &[(f64, f64)]) -> Result<i32, String> {
    let mut sub: Vec<(f64, f64)> = cat.iter().step_by(8).copied().collect();
    sub.truncate(3000);
    let frame = 2048.0;
    let d = Cfg {
        frame,
        ..cfg.clone()
    };
    let truth = Truth {
        roll: 0.7,
        scale: 8.44,
        parity: 0,
        tx: 1000.0,
        ty: 1100.0,
    };
    let mut rng = Rng::new(7);
    let synth = synthesize(&sub, &truth, frame, false, 0.0, 1.0, 0.0, &mut rng);
    eprintln!("selftest: cat={} synth={}", sub.len(), synth.len());
    let n = 256;
    let b = frame / n as f64;
    let mut trials = Vec::new();
    for parity in 0u8..2 {
        for i in -10..=10 {
            let roll = truth.roll + i as f64 * 0.05 * std::f64::consts::PI / 180.0;
            for j in -10..=10 {
                trials.push(Trial {
                    parity,
                    roll,
                    scale: truth.scale * (1.0 + j as f64 * 0.002),
                });
            }
        }
    }
    let prepared = prepare_det(&synth, n, frame, 0.0);
    let outs = eval_trials(&d, &prepared, &sub, &trials, cfg.threads, 0, true, 0, "selftest");
    let mut hits: Vec<Hit> = Vec::new();
    for o in outs {
        for e in o.evals {
            hits.push(hit_from_eval(&d, n, trials[e.flat], &e));
        }
    }
    hits.sort_unstable_by(|a, b| b.peak_sigma.total_cmp(&a.peak_sigma));
    let Some(h) = hits.first().cloned() else {
        println!("selftest ok=false reason=no-hits");
        return Ok(1);
    };
    let droll = angle_diff(h.trial.roll, truth.roll).to_degrees();
    let dscale = h.trial.scale / truth.scale - 1.0;
    let dt = ((h.t_px.0 - truth.tx).powi(2) + (h.t_px.1 - truth.ty).powi(2)).sqrt();
    let ok = h.peak_sigma > 8.0
        && h.trial.parity == truth.parity
        && droll.abs() <= 0.06
        && dscale.abs() <= 0.003
        && dt <= 1.5 * b;
    println!(
        "selftest ok={ok} peak_sigma={:.2} roll={:.4}deg d_roll={:.4} scale={:.4} d_scale={:.4} parity={} t_px=({:.1},{:.1}) d_t={:.1}px truth_t=({:.1},{:.1}) bin={:.1}px",
        h.peak_sigma,
        h.trial.roll.to_degrees(),
        droll,
        h.trial.scale,
        dscale,
        h.trial.parity,
        h.t_px.0,
        h.t_px.1,
        dt,
        truth.tx,
        truth.ty,
        b
    );
    Ok(if ok { 0 } else { 1 })
}
