//! Directional-hypothesis generalised likelihood ratio test (DG-GLRT), prototype M1+M2.
//!
//! Hypothesis family: the point source (a Gaussian PSF) plus line segments of
//! octave length in n_angles orientations. Every hypothesis is a unit-sum
//! kernel k_theta; the statistic is the whitened matched filter
//!
//!     z(x; theta) = <k_theta, W^2 d>(x) / (sigma_w * ||W k_theta||)
//!
//! where d is the background-subtracted frame, W the zero-phase whitening
//! filter (W = I with --whiten off) and sigma_w the noise level of the
//! whitened frame. With W = I this is exactly
//!
//!     z = IFFT(FFT(d) conj(FFT(k))) / (sigma sqrt(sum k^2))
//!
//! which mf_z_freq evaluates directly (zero padded to >= N+M-1). line_z_map
//! evaluates the same operator through the equivalent identity
//!
//!     sum_y k(y) d(x+y) = sum_z seg(z) (g * d)(x+z),      k = g * seg,
//!
//! i.e. a directional line integral of the PSF-smoothed frame, which costs
//! O(1) per (pixel, length) with row prefix sums instead of one inverse FFT per
//! hypothesis. The two paths are cross-checked by
//! tests::freq_and_line_engines_agree.
//!
//! Statistics: Lambda = max_theta z, LR = z(best line) - z(point), the argmax
//! gives (L_hat, phi_hat). Thresholds come from a Gumbel fit to the filtered
//! noise field, so a threshold means "expected false alarms per frame".

use crate::fits;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// FWHM = 2.3548 sigma for a Gaussian.
pub const FWHM_TO_SIGMA: f64 = 2.354_820_045_030_949_3;
/// PSF kernel truncation radius in sigma.
const PSF_TRUNC_SIGMA: f64 = 4.0;
/// Largest rational denominator accepted for a sheared direction.
const MAX_SHEAR_DEN: usize = 8;
/// Half width of the noise-autocorrelation window used by M2.
const NOISE_LAG: usize = 6;
/// Windows up to this many retained samples are summed exactly in the library.
const EXACT_WINDOW: usize = 32;

// ---------------------------------------------------------------------------
// fft
//
// Copied from astrometry_blind.rs (Plan1, Fft2 are private there) and extended
// to rectangular grids plus an explicit inverse transform.
// ---------------------------------------------------------------------------

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

fn transpose_rect(src: &[f64], dst: &mut [f64], rows: usize, cols: usize) {
    const B: usize = 32;
    for bi in (0..rows).step_by(B) {
        let imax = (bi + B).min(rows);
        for bj in (0..cols).step_by(B) {
            let jmax = (bj + B).min(cols);
            for i in bi..imax {
                let row = i * cols;
                for j in bj..jmax {
                    dst[j * rows + i] = src[row + j];
                }
            }
        }
    }
}

/// Separable 2-D FFT on a complex rows x cols grid (both powers of two).
struct Fft2 {
    nx: usize,
    ny: usize,
    px: Plan1,
    py: Plan1,
    ta: Vec<f64>,
    tb: Vec<f64>,
}

impl Fft2 {
    fn new(nx: usize, ny: usize) -> Self {
        assert!(nx.is_power_of_two() && ny.is_power_of_two());
        Self {
            nx,
            ny,
            px: Plan1::new(nx),
            py: Plan1::new(ny),
            ta: vec![0.0; nx * ny],
            tb: vec![0.0; nx * ny],
        }
    }

    fn transform(&mut self, re: &mut [f64], im: &mut [f64], inverse: bool) {
        for r in 0..self.ny {
            let (lo, hi) = (r * self.nx, (r + 1) * self.nx);
            self.px.fft1(&mut re[lo..hi], &mut im[lo..hi], inverse);
        }
        transpose_rect(re, &mut self.ta, self.ny, self.nx);
        transpose_rect(im, &mut self.tb, self.ny, self.nx);
        for r in 0..self.nx {
            let (lo, hi) = (r * self.ny, (r + 1) * self.ny);
            self.py
                .fft1(&mut self.ta[lo..hi], &mut self.tb[lo..hi], inverse);
        }
        transpose_rect(&self.ta, re, self.nx, self.ny);
        transpose_rect(&self.tb, im, self.nx, self.ny);
    }
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

// ---------------------------------------------------------------------------
// small numeric helpers
// ---------------------------------------------------------------------------

fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    // O(n) selection instead of a full sort: this runs on every background block
    values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        values[mid]
    } else {
        let lo = values[..mid]
            .iter()
            .fold(f64::NEG_INFINITY, |acc, v| acc.max(*v));
        0.5 * (lo + values[mid])
    }
}

/// 1.4826 * median(|v|) for the scale of a zero-centred sample.
fn mad_sigma(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    for v in values.iter_mut() {
        *v = v.abs();
    }
    1.482_602_218_505_602 * median(values)
}

fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// xorshift generator; deterministic for a given seed, no dependencies.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn uniform(&mut self) -> f64 {
        let v = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        v.max(1e-12).min(1.0 - 1e-12)
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

/// Unit-sum 1-D Gaussian, truncated at PSF_TRUNC_SIGMA.
fn gauss_1d(sigma: f64) -> Vec<f64> {
    let r = (PSF_TRUNC_SIGMA * sigma).ceil().max(1.0) as usize;
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

/// A sampled kernel on the integer lattice, centred at (rows/2, cols/2).
#[derive(Clone)]
pub struct KernelGrid {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
    pub sum_sq: f64,
}

impl KernelGrid {
    fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        let sum_sq = data.iter().map(|v| v * v).sum();
        Self {
            rows,
            cols,
            data,
            sum_sq,
        }
    }
    fn center(&self) -> (usize, usize) {
        (self.rows / 2, self.cols / 2)
    }
}

/// Separable blur of a rows x cols grid with zero padding at the border.
fn blur_separable_f64(data: &[f64], rows: usize, cols: usize, gx: &[f64], gy: &[f64]) -> Vec<f64> {
    let rx = gx.len() / 2;
    let ry = gy.len() / 2;
    let mut tmp = vec![0.0; rows * cols];
    for y in 0..rows {
        for x in 0..cols {
            let mut acc = 0.0;
            for (k, w) in gx.iter().enumerate() {
                let xx = x as isize + k as isize - rx as isize;
                if xx < 0 || xx >= cols as isize {
                    continue;
                }
                acc += w * data[y * cols + xx as usize];
            }
            tmp[y * cols + x] = acc;
        }
    }
    let mut out = vec![0.0; rows * cols];
    for y in 0..rows {
        for x in 0..cols {
            let mut acc = 0.0;
            for (k, w) in gy.iter().enumerate() {
                let yy = y as isize + k as isize - ry as isize;
                if yy < 0 || yy >= rows as isize {
                    continue;
                }
                acc += w * tmp[yy as usize * cols + x];
            }
            out[y * cols + x] = acc;
        }
    }
    out
}

/// Rasterise lattice samples into the unit-sum kernel g_sigma * segment.
/// samples are (row, col, weight) offsets relative to the segment centre.
fn kernel_from_samples(samples: &[(i64, i64, f64)], sigma: f64) -> KernelGrid {
    if samples.is_empty() {
        return KernelGrid::new(1, 1, vec![1.0]);
    }
    let pad = (PSF_TRUNC_SIGMA * sigma).ceil() as i64 + 1;
    let (mut y0, mut y1, mut x0, mut x1) = (i64::MAX, i64::MIN, i64::MAX, i64::MIN);
    for &(dy, dx, _) in samples {
        y0 = y0.min(dy);
        y1 = y1.max(dy);
        x0 = x0.min(dx);
        x1 = x1.max(dx);
    }
    let hy = y1.max(-y0).max(0) as usize;
    let hx = x1.max(-x0).max(0) as usize;
    let rows = 2 * hy + 1 + 2 * pad as usize;
    let cols = 2 * hx + 1 + 2 * pad as usize;
    let (cy, cx) = (hy + pad as usize, hx + pad as usize);
    let mut seg = vec![0.0; rows * cols];
    for &(dy, dx, wt) in samples {
        let i = (cy as i64 + dy) as usize;
        let j = (cx as i64 + dx) as usize;
        seg[i * cols + j] += wt;
    }
    let g = gauss_1d(sigma);
    let blurred = blur_separable_f64(&seg, rows, cols, &g, &g);
    let s: f64 = blurred.iter().sum();
    let data = if s > 0.0 {
        blurred.iter().map(|v| v / s).collect()
    } else {
        blurred
    };
    KernelGrid::new(rows, cols, data)
}

/// Unit-sum Gaussian point-spread kernel.
pub fn point_kernel(sigma: f64) -> KernelGrid {
    let g = gauss_1d(sigma);
    let n = g.len();
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = g[i] * g[j];
        }
    }
    KernelGrid::new(n, n, data)
}

/// Autocorrelation of a 1-D profile, a[d] = sum_i p[i] p[i+d] for d >= 0.
fn autocorr_1d(profile: &[f64], max_lag: usize) -> Vec<f64> {
    let n = profile.len();
    (0..=max_lag)
        .map(|d| {
            if d >= n {
                return 0.0;
            }
            let mut acc = 0.0;
            for i in 0..n - d {
                acc += profile[i] * profile[i + d];
            }
            acc
        })
        .collect()
}

/// sum k^2 of the kernel built from lattice samples whose separable profile
/// autocorrelations are ay (row axis) and ax (column axis).
fn sum_sq_from_samples(samples: &[(i64, i64, f64)], ay: &[f64], ax: &[f64]) -> f64 {
    let n = samples.len();
    let max_dy = (ay.len() - 1) as i64;
    let max_dx = (ax.len() - 1) as i64;
    let window = 2 * (max_dy.max(max_dx) as usize) + 4;
    let mut acc = 0.0f64;
    for i in 0..n {
        let (yi, xi, wi) = samples[i];
        let lo = i.saturating_sub(window);
        let hi = (i + window + 1).min(n);
        for j in lo..hi {
            let (yj, xj, wj) = samples[j];
            let dy = (yi - yj).abs();
            let dx = (xi - xj).abs();
            if dy > max_dy || dx > max_dx {
                continue;
            }
            acc += wi * wj * ay[dy as usize] * ax[dx as usize];
        }
    }
    acc.max(0.0)
}

/// Lag decomposition of a line kernel's correlation sum.
///
/// For unit-weight samples at arc length d along an axis the pair sum over a
/// window of N consecutive samples is
///   S(N) = sum_{|d| < N} (N - |d|) B(|d|),
/// which only needs the lag-0 term and the prefix sums U(N) = sum_{d<N} B(d),
/// V(N) = sum_{d<N} d B(d):  S(N) = 2 N U(N) - 2 V(N) - N B(0).
/// This is what makes a position-dependent (edge-truncated) normalisation
/// affordable: the retained sample count is data-independent integer bookkeeping
/// and S(N) is one table lookup.
#[derive(Clone)]
struct LagCorr {
    u: Vec<f64>,
    v: Vec<f64>,
    b0: f64,
}

impl LagCorr {
    fn from_lags(lags: &[f64]) -> Self {
        let mut u = Vec::with_capacity(lags.len());
        let mut v = Vec::with_capacity(lags.len());
        let (mut su, mut sv) = (0.0f64, 0.0f64);
        for (d, b) in lags.iter().enumerate() {
            su += *b;
            sv += d as f64 * *b;
            u.push(su);
            v.push(sv);
        }
        Self {
            u,
            v,
            b0: lags.first().copied().unwrap_or(0.0),
        }
    }

    /// Increment of the tail: S(N) grows by this per extra retained sample once
    /// the window is longer than the correlation support.
    fn slope(&self) -> f64 {
        if self.u.is_empty() {
            return 0.0;
        }
        let m = self.u.len();
        2.0 * self.u[m - 1] - self.b0
    }

    /// Scale so that S(n) reproduces an externally computed pair sum exactly.
    fn scaled_to(mut self, n: usize, target: f64) -> Self {
        let cur = self.s_n(n);
        if cur > 1e-30 && target > 1e-30 {
            let f = target / cur;
            for x in self.u.iter_mut().chain(self.v.iter_mut()) {
                *x *= f;
            }
            self.b0 *= f;
        }
        self
    }

    /// Pair sum of the effective kernel autocorrelation over N retained samples.
    fn s_n(&self, n: usize) -> f64 {
        if n == 0 || self.u.is_empty() {
            return 0.0;
        }
        let m = n.min(self.u.len());
        let nf = n as f64;
        2.0 * nf * self.u[m - 1] - 2.0 * self.v[m - 1] - nf * self.b0
    }
}

/// Lags B(d) = a_y(|d| sin) a_x(|d| cos) of a straight line at angle phi, with
/// bilinear interpolation of the two 1-D autocorrelations (continuous angles).
fn line_lags_continuous(phi_deg: f64, ay: &[f64], ax: &[f64], n_max: usize) -> Vec<f64> {
    let rad = phi_deg.to_radians();
    let (c, s) = (rad.cos(), rad.sin());
    let cap = 4 * ay.len().max(ax.len()) + 2;
    let dmax = n_max.clamp(1, cap);
    (0..dmax)
        .map(|d| {
            let b = table_at(ay, d as f64 * s) * table_at(ax, d as f64 * c);
            b
        })
        .collect()
}

// ---------------------------------------------------------------------------
// M2: noise model and zero-phase whitening
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct NoiseModel {
    /// Robust per-pixel noise level of the background-subtracted frame (ADU).
    pub sigma_adu: f64,
    /// sqrt of the product of the two 1-D correlation areas.
    pub corr_len_px: f64,
    /// Autocorrelation along x, rho_x[lag + d] for |d| <= NOISE_LAG, centre = 1.
    pub rho_x: Vec<f64>,
    pub rho_y: Vec<f64>,
    /// Noise pixels per independent cell, 1 / corr_len^2.
    pub n_eff_per_px: f64,
    pub blocks_used: usize,
}

/// Block background (median) with bilinear upsampling.
pub fn background_median(image: &[f64], h: usize, w: usize, block: usize) -> Vec<f64> {
    let block = block.max(4);
    let bh = (h + block - 1) / block;
    let bw = (w + block - 1) / block;
    let mut grid = vec![0.0f64; bh * bw];
    let mut scratch: Vec<f64> = Vec::new();
    for by in 0..bh {
        let y0 = by * block;
        let y1 = ((by + 1) * block).min(h);
        for bx in 0..bw {
            let x0 = bx * block;
            let x1 = ((bx + 1) * block).min(w);
            scratch.clear();
            for y in y0..y1 {
                for x in x0..x1 {
                    scratch.push(image[y * w + x]);
                }
            }
            grid[by * bw + bx] = median(&mut scratch);
        }
    }
    let mut out = vec![0.0f64; h * w];
    for y in 0..h {
        let fy = ((y as f64 + 0.5) / block as f64 - 0.5).clamp(0.0, (bh - 1) as f64);
        let iy = fy.floor() as usize;
        let iy1 = (iy + 1).min(bh - 1);
        let ty = fy - iy as f64;
        for x in 0..w {
            let fx = ((x as f64 + 0.5) / block as f64 - 0.5).clamp(0.0, (bw - 1) as f64);
            let ix = fx.floor() as usize;
            let ix1 = (ix + 1).min(bw - 1);
            let tx = fx - ix as f64;
            let a = grid[iy * bw + ix] * (1.0 - tx) + grid[iy * bw + ix1] * tx;
            let b = grid[iy1 * bw + ix] * (1.0 - tx) + grid[iy1 * bw + ix1] * tx;
            out[y * w + x] = a * (1.0 - ty) + b * ty;
        }
    }
    out
}

/// Estimate the noise level and the separable autocorrelation from source-free
/// blocks of a residual (background already removed).
pub fn estimate_noise(residual: &[f64], h: usize, w: usize) -> NoiseModel {
    let block = 64usize;
    let bh = (h / block).max(1);
    let bw = (w / block).max(1);
    let lag = NOISE_LAG;
    let mut scales = vec![0.0f64; bh * bw];
    let mut scratch: Vec<f64> = Vec::new();
    for by in 0..bh {
        for bx in 0..bw {
            let y0 = (by * block).min(h.saturating_sub(1));
            let x0 = (bx * block).min(w.saturating_sub(1));
            let y1 = ((by + 1) * block).min(h);
            let x1 = ((bx + 1) * block).min(w);
            let mut vals: Vec<f64> = Vec::with_capacity((y1 - y0).max(1) * (x1 - x0).max(1));
            for y in y0..y1 {
                for x in x0..x1 {
                    vals.push(residual[y * w + x]);
                }
            }
            let m = median(&mut vals);
            scratch.clear();
            for v in vals.iter() {
                scratch.push(v - m);
            }
            scales[by * bw + bx] = mad_sigma(&mut scratch);
        }
    }
    let mut ordered = scales.clone();
    let global = median(&mut ordered);
    let mut sorted_shapes = scales.clone();
    sorted_shapes.sort_unstable_by(|a, b| a.total_cmp(b));
    let idx20 = (((sorted_shapes.len() as f64) * 0.2) as usize).min(sorted_shapes.len() - 1);
    let p20 = sorted_shapes[idx20];
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    for by in 0..bh {
        for bx in 0..bw {
            let s = scales[by * bw + bx];
            if s > 0.0 && s <= 1.5 * p20.max(1e-9) {
                blocks.push((by, bx));
            }
        }
    }
    if blocks.is_empty() {
        for by in 0..bh {
            for bx in 0..bw {
                blocks.push((by, bx));
            }
        }
    }
    let stride = (blocks.len() / 96).max(1);
    let blocks: Vec<(usize, usize)> = blocks.into_iter().step_by(stride).collect();
    let mut resid_blocks: Vec<(usize, usize, Vec<f64>)> = Vec::new();
    for &(by, bx) in &blocks {
        let y0 = by * block;
        let x0 = bx * block;
        let y1 = (y0 + block).min(h);
        let x1 = (x0 + block).min(w);
        if y1 <= y0 || x1 <= x0 {
            continue;
        }
        let mut vals: Vec<f64> = Vec::with_capacity((y1 - y0) * (x1 - x0));
        for y in y0..y1 {
            for x in x0..x1 {
                vals.push(residual[y * w + x]);
            }
        }
        // the median helper permutes its argument, so keep the pixel order intact
        let m = median(&mut vals.clone());
        let centred: Vec<f64> = vals.iter().map(|v| v - m).collect();
        resid_blocks.push((y1 - y0, x1 - x0, centred));
    }
    let mut all: Vec<f64> = resid_blocks
        .iter()
        .flat_map(|(_, _, v)| v.iter().copied())
        .collect();
    let mut sigma = mad_sigma(&mut all);
    if !(sigma.is_finite() && sigma > 0.0) {
        sigma = global.max(1e-9);
    }
    let clip = 5.0 * sigma;
    let mut rho_x = vec![0.0f64; 2 * lag + 1];
    let mut rho_y = vec![0.0f64; 2 * lag + 1];
    rho_x[lag] = 1.0;
    rho_y[lag] = 1.0;
    let mut var_acc = 0.0f64;
    let mut n_acc = 0.0f64;
    let mut sum_xx = vec![0.0f64; lag + 1];
    let mut cnt_xx = vec![0.0f64; lag + 1];
    let mut sum_yy = vec![0.0f64; lag + 1];
    let mut cnt_yy = vec![0.0f64; lag + 1];
    for (rows, cols, vals) in &resid_blocks {
        for value in vals.iter() {
            let a = value.clamp(-clip, clip);
            var_acc += a * a;
            n_acc += 1.0;
        }
        for d in 1..=lag {
            let mut sxx = 0.0f64;
            let mut nxx = 0.0f64;
            let mut syy = 0.0f64;
            let mut nyy = 0.0f64;
            for y in 0..*rows {
                for x in 0..*cols {
                    let a = vals[y * cols + x].clamp(-clip, clip);
                    if x + d < *cols {
                        sxx += a * vals[y * cols + x + d].clamp(-clip, clip);
                        nxx += 1.0;
                    }
                    if y + d < *rows {
                        syy += a * vals[(y + d) * cols + x].clamp(-clip, clip);
                        nyy += 1.0;
                    }
                }
            }
            sum_xx[d] += sxx;
            cnt_xx[d] += nxx;
            sum_yy[d] += syy;
            cnt_yy[d] += nyy;
        }
    }
    for d in 1..=lag {
        if cnt_xx[d] > 0.0 {
            rho_x[lag + d] = sum_xx[d] / cnt_xx[d];
        }
        if cnt_yy[d] > 0.0 {
            rho_y[lag + d] = sum_yy[d] / cnt_yy[d];
        }
    }
    let var = (if n_acc > 0.0 { var_acc / n_acc } else { sigma * sigma }).max(1e-30);
    for d in 1..=lag {
        rho_x[lag + d] /= var;
        rho_y[lag + d] /= var;
        rho_x[lag - d] = rho_x[lag + d];
        rho_y[lag - d] = rho_y[lag + d];
    }
    for d in 1..=lag {
        let w = 0.5 * (1.0 + (std::f64::consts::PI * d as f64 / (lag + 1) as f64).cos());
        rho_x[lag + d] *= w;
        rho_x[lag - d] = rho_x[lag + d];
        rho_y[lag + d] *= w;
        rho_y[lag - d] = rho_y[lag + d];
    }
    let area_x: f64 = rho_x.iter().sum::<f64>().max(1.0);
    let area_y: f64 = rho_y.iter().sum::<f64>().max(1.0);
    NoiseModel {
        sigma_adu: sigma,
        corr_len_px: (area_x * area_y).sqrt(),
        rho_x,
        rho_y,
        n_eff_per_px: 1.0 / (area_x * area_y),
        blocks_used: resid_blocks.len(),
    }
}

/// Zero-phase 1-D whitening filter 1/sqrt(S(k)), truncated to taps taps.
fn whitening_fir(rho: &[f64], n_axis: usize, taps: usize) -> Vec<f64> {
    let half = (taps / 2).max(1);
    let lag = (rho.len() - 1) / 2;
    let n = next_pow2(n_axis.max(128));
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    for k in 0..=lag {
        re[k] = rho[lag + k];
        if k > 0 {
            re[n - k] = rho[lag + k];
        }
    }
    let plan = Plan1::new(n);
    plan.fft1(&mut re, &mut im, false);
    let spectrum_max = re[0].max(1e-12);
    for v in re.iter_mut() {
        *v = 1.0 / (*v).max(1e-3 * spectrum_max).sqrt();
    }
    for v in im.iter_mut() {
        *v = 0.0;
    }
    plan.fft1(&mut re, &mut im, true);
    let mut h = vec![0.0f64; 2 * half + 1];
    for d in 0..=half {
        let w = 0.54 + 0.46 * (std::f64::consts::PI * d as f64 / (half + 1) as f64).cos();
        let v = re[d] * w;
        h[half + d] = v;
        h[half - d] = v;
    }
    h
}

#[derive(Clone)]
pub struct Whitener {
    pub hx: Vec<f64>,
    pub hy: Vec<f64>,
}

impl Whitener {
    fn identity() -> Self {
        Self {
            hx: vec![1.0],
            hy: vec![1.0],
        }
    }
    fn is_identity(&self) -> bool {
        self.hx.len() == 1 && self.hy.len() == 1
    }
    /// One whitening pass, zero padded at the border.
    fn apply(&self, data: &[f64], h: usize, w: usize) -> Vec<f64> {
        if self.is_identity() {
            return data.to_vec();
        }
        blur_separable_f64(data, h, w, &self.hx, &self.hy)
    }
    fn support(&self) -> usize {
        (self.hx.len() / 2).max(self.hy.len() / 2)
    }
}

// ---------------------------------------------------------------------------
// hypothesis library
// ---------------------------------------------------------------------------

/// One line-segment hypothesis, expressed in the sheared frame that the line
/// engine evaluates it in (p/q shear, sign of the slope, sample count n).
#[derive(Clone)]
pub struct LineHyp {
    pub length_px: f64,
    pub angle_deg: f64,
    pub steep: bool,
    pub sign: i8,
    pub p: usize,
    pub q: usize,
    pub n: usize,
    /// sigma_w * ||W k|| per phase of the anchor column (q values).
    pub norms: Vec<f64>,
    /// sqrt(sum k^2 / sum (d_x k)^2) for phase 0: position error per unit z.
    pub sigma_x_coeff: f64,
    /// Truncation-aware inverse norm: entry [phase * (n + 1) + N] multiplies the
    /// raw in-frame sample sum for a kernel that keeps N of its n samples.
    pub inv_norm: Vec<f32>,
}

impl LineHyp {
    /// Multiplier that turns a raw in-frame sum over N retained samples into the
    /// unit-variance statistic z. Falls back to the full norm for N = 0.
    #[inline]
    fn inv_norm_at(&self, phase: usize, used: usize) -> f32 {
        let used = used.min(self.n).max(1);
        self.inv_norm[phase * (self.n + 1) + used]
    }
}

impl LineHyp {
    fn n1(&self) -> usize {
        (self.n - 1) / 2
    }
    /// (row, col) sample offsets relative to the response pixel, phase r.
    fn offsets(&self, phase: usize) -> Vec<(i64, i64)> {
        let n1 = self.n1();
        let mid = (phase + n1 * self.p) / self.q;
        (0..self.n)
            .map(|j| {
                let row = ((phase + j * self.p) / self.q) as i64 - mid as i64;
                (row * self.sign as i64, j as i64 - n1 as i64)
            })
            .collect()
    }
    fn weighted_offsets(&self, phase: usize) -> Vec<(i64, i64, f64)> {
        let w = 1.0 / self.n as f64;
        self.offsets(phase)
            .into_iter()
            .map(|(r, c)| (r, c, w))
            .collect()
    }
}

pub struct Library {
    pub lengths: Vec<f64>,
    pub n_angles: usize,
    pub sigma_psf: f64,
    pub point: KernelGrid,
    pub point_norm: f64,
    pub point_sigma_x_coeff: f64,
    pub lines: Vec<LineHyp>,
}

/// Best rational p/q (q <= MAX_SHEAR_DEN) for a slope in [0, 1].
fn rationalise(slope: f64) -> (usize, usize) {
    let target = slope.clamp(0.0, 1.0);
    let mut best = (0usize, 1usize);
    let mut best_err = f64::INFINITY;
    for q in 1..=MAX_SHEAR_DEN {
        let p = (target * q as f64).round() as usize;
        let err = (p as f64 / q as f64 - target).abs();
        if err < best_err - 1e-12 {
            best_err = err;
            best = (p, q);
        }
    }
    let g = gcd(best.0.max(1), best.1);
    (best.0 / g, (best.1 / g).max(1))
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a.max(1)
    } else {
        gcd(b, a % b)
    }
}

fn fold_angle(angle_deg: f64) -> f64 {
    let mut a = angle_deg % 180.0;
    if a < 0.0 {
        a += 180.0;
    }
    a
}

/// Build the library from the optical and noise parameters. profile_plain and
/// profile_grad are the (row, col) autocorrelations of the whitened PSF profile
/// and of its derivative.
pub fn build_library(
    lengths: &[f64],
    n_angles: usize,
    sigma_psf: f64,
    sigma_w: f64,
    profile_plain: (&[f64], &[f64]),
    profile_grad: (&[f64], &[f64]),
) -> Library {
    let (ay, ax) = profile_plain;
    let (by, bx) = profile_grad;
    let point = point_kernel(sigma_psf);
    let one = [(0i64, 0i64, 1.0f64)];
    let point_sum_sq = sum_sq_from_samples(&one, ay, ax);
    let point_norm = sigma_w * point_sum_sq.sqrt();
    let point_sigma_x_coeff =
        (point_sum_sq / sum_sq_from_samples(&one, ay, bx).max(1e-30)).sqrt();
    let mut lines = Vec::new();
    for k in 0..n_angles {
        let nominal = 180.0 * k as f64 / n_angles as f64;
        let rad = nominal.to_radians();
        let (cx, cy) = (rad.cos(), rad.sin());
        let shallow = cx.abs() >= cy.abs();
        let (sign, p, q) = if shallow {
            let slope = cy / cx;
            let s = if slope < 0.0 { -1i8 } else { 1i8 };
            let (p, q) = rationalise(slope.abs());
            (s, p, q)
        } else {
            let slope = cx / cy;
            let s = if slope < 0.0 { -1i8 } else { 1i8 };
            let (p, q) = rationalise(slope.abs());
            (s, p, q)
        };
        let step = (1.0 + (p as f64 / q as f64).powi(2)).sqrt();
        for &length in lengths {
            let n_raw = (length / step).round().max(2.0) as usize;
            let n = (n_raw | 1).min(4095);
            let actual_len = (n - 1) as f64 * step;
            let angle_deg = fold_angle(if shallow {
                (sign as f64 * p as f64 / q as f64).atan().to_degrees()
            } else {
                (1.0f64).atan2(sign as f64 * p as f64 / q as f64).to_degrees()
            });
            let mut hyp = LineHyp {
                length_px: actual_len,
                angle_deg,
                steep: !shallow,
                sign,
                p,
                q,
                n,
                norms: vec![0.0; q],
                sigma_x_coeff: 0.0,
                inv_norm: vec![0.0f32; q * (n + 1)],
            };
            let (plain_y, plain_x) = if shallow { (ay, ax) } else { (ax, ay) };
            let lag_cap = 4 * plain_y.len().max(plain_x.len()) + 2;
            let _ = lag_cap;
            for r in 0..q {
                let samples = hyp.weighted_offsets(r);
                let weighted = sum_sq_from_samples(&samples, plain_y, plain_x);
                hyp.norms[r] = sigma_w * weighted.sqrt();
                // unweighted pair sum over the full kernel, reproduced exactly by
                // the lag decomposition so that N == n returns the interior norm
                let full = weighted * (n * n) as f64;
                // |Delta row| for lag d is floor(d p / q) or one more, in the
                // proportion (d p mod q)/q over the phase cycle, so the tabulated
                // autocorrelation is read at the true fractional lag; lags past
                // the support contribute nothing.
                let lags: Vec<f64> = (0..n.min(lag_cap))
                    .map(|d| {
                        let row = table_at(plain_y, d as f64 * p as f64 / q as f64);
                        let col = table_at(plain_x, d as f64);
                        row * col
                    })
                    .collect();
                // Short windows are summed exactly: the lag decomposition averages
                // the row offset over the phase cycle, which only becomes accurate
                // once a window spans several cycles. Beyond the exact band the
                // decomposition is anchored to the last exact value, so the table
                // is continuous and its tail error is the shape error only.
                let exact_upto = n.min(EXACT_WINDOW);
                let corr = LagCorr::from_lags(&lags);
                for used in 1..=exact_upto {
                    let kept: Vec<(i64, i64, f64)> = samples.iter().take(used).cloned().collect();
                    let s_n = (sum_sq_from_samples(&kept, plain_y, plain_x)
                        * (n * n) as f64)
                        .max(1e-30);
                    hyp.inv_norm[r * (n + 1) + used] = (1.0 / (sigma_w * s_n.sqrt())) as f32;
                }
                if exact_upto < n {
                    let anchor = sum_sq_from_samples(
                        &samples[..exact_upto].to_vec(),
                        plain_y,
                        plain_x,
                    ) * (n * n) as f64;
                    let c = corr.slope().max(1e-30);
                    let k = c * exact_upto as f64 - anchor;
                    for used in exact_upto + 1..=n {
                        let s_n = ((used as f64) * c - k).max(1e-30);
                        hyp.inv_norm[r * (n + 1) + used] = (1.0 / (sigma_w * s_n.sqrt())) as f32;
                    }
                }
            }
            let samples = hyp.weighted_offsets(0);
            let (grad_y, grad_x) = if shallow { (by, bx) } else { (bx, by) };
            let s_sq = sum_sq_from_samples(&samples, plain_y, plain_x);
            let g_sq = sum_sq_from_samples(&samples, grad_y, grad_x);
            hyp.sigma_x_coeff = (s_sq / g_sq.max(1e-30)).sqrt();
            lines.push(hyp);
        }
    }
    Library {
        lengths: lengths.to_vec(),
        n_angles,
        sigma_psf,
        point,
        point_norm,
        point_sigma_x_coeff,
        lines,
    }
}

// ---------------------------------------------------------------------------
// line engine: directional line integrals of the PSF-smoothed frame
// ---------------------------------------------------------------------------

struct DirGroup {
    steep: bool,
    sign: i8,
    p: usize,
    q: usize,
    members: Vec<usize>,
}

fn group_directions(lines: &[LineHyp]) -> Vec<DirGroup> {
    let mut groups: Vec<DirGroup> = Vec::new();
    for (i, h) in lines.iter().enumerate() {
        match groups
            .iter_mut()
            .find(|g| g.steep == h.steep && g.sign == h.sign && g.p == h.p && g.q == h.q)
        {
            Some(g) => g.members.push(i),
            None => groups.push(DirGroup {
                steep: h.steep,
                sign: h.sign,
                p: h.p,
                q: h.q,
                members: vec![i],
            }),
        }
    }
    groups
}

/// Evaluate every member of one sheared direction and fold the result into the
/// Lambda / best-line accumulators. buf is the smoothed frame (rows x cols).
#[allow(clippy::too_many_arguments)]
fn run_direction(
    buf: &[f32],
    rows: usize,
    cols: usize,
    group: &DirGroup,
    lines: &[LineHyp],
    lambda: &mut [f32],
    mut zline: Option<&mut [f32]>,
    floor: f32,
) {
    let p = group.p as i64;
    let q = group.q as i64;
    let sign = group.sign as i64;
    let n1max = group
        .members
        .iter()
        .map(|&i| lines[i].n1())
        .max()
        .unwrap_or(0);
    let mut row_buf = vec![0.0f32; cols];
    let mut prefix = vec![0.0f64; cols + n1max + 2];
    // (index, n1, 1/n, full-length norms, truncation table): the untruncated
    // branch keeps the pre-table arithmetic, which is what almost every pixel
    // outside the edge band takes
    let members: Vec<(usize, usize, f64, Vec<f64>, Vec<f32>)> = group
        .members
        .iter()
        .map(|&i| {
            let h = &lines[i];
            (i, h.n1(), 1.0 / h.n as f64, h.norms.clone(), h.inv_norm.clone())
        })
        .collect();
    for y in 0..rows {
        // The gather row of column x is y + sign * floor(x p / q); floor is
        // non-decreasing in x, so the columns whose gather row stays inside the
        // frame form the prefix [0, c_lim). Everything beyond is outside the
        // frame, hence absent from both numerator and norm.
        let bound = if sign > 0 {
            rows as i64 - 1 - y as i64
        } else {
            y as i64
        };
        let c_lim = if p == 0 {
            cols
        } else if bound < 0 {
            0
        } else {
            ((((bound + 1) * q) + p - 1) / p).clamp(0, cols as i64) as usize
        };
        if c_lim == 0 {
            continue;
        }
        let mut any = false;
        for x in 0..c_lim {
            let sh = (x as i64 * p) / q;
            let src = y as i64 + sign * sh;
            if src < 0 || src >= rows as i64 {
                row_buf[x] = 0.0;
            } else {
                let v = buf[src as usize * cols + x];
                row_buf[x] = v;
                any |= v != 0.0;
            }
        }
        if !any {
            continue;
        }
        for slot in prefix.iter_mut().take(n1max + 1) {
            *slot = 0.0;
        }
        let mut acc = 0.0f64;
        for x in 0..c_lim {
            acc += row_buf[x] as f64;
            prefix[n1max + 1 + x] = acc;
        }
        for (idx, n1, inv_n, norms, inv_norm) in &members {
            let hyp = &lines[*idx];
            let n = hyp.n;
            let qq = hyp.q as i64;
            let mut phase = ((-(*n1 as i64)) % qq + qq) % qq;
            for x in 0..cols {
                let base = x as i64 - *n1 as i64;
                let hi = (base + n as i64).clamp(0, c_lim as i64) as usize;
                let lo = base.clamp(0, c_lim as i64) as usize;
                if hi.saturating_sub(lo) < 2 {
                    phase += 1;
                    if phase == qq {
                        phase = 0;
                    }
                    continue;
                }
                let sum = prefix[n1max + hi] - prefix[n1max + lo];
                let used = hi - lo;
                // the residual norm of a truncated kernel is the pair sum over the
                // retained samples only, so the response of a streak cut by the
                // frame edge is not diluted by the samples that left the frame
                let val = if used == n {
                    (sum * inv_n / norms[phase as usize]) as f32
                } else {
                    (sum * hyp.inv_norm_at(phase as usize, used) as f64) as f32
                };
                if val > floor {
                    // the centre sample sits at column x, so the response belongs
                    // to row y + sign * sh(x) (kernel offsets in LineHyp::offsets
                    // are measured from that sample)
                    let sh = (x as i64 * p) / q;
                    let out = y as i64 + sign * sh;
                    if out >= 0 && out < rows as i64 {
                        let off = out as usize * cols + x;
                        if val > lambda[off] {
                            lambda[off] = val;
                        }
                        if let Some(z) = zline.as_mut() {
                            if val > z[off] {
                                z[off] = val;
                            }
                        }
                    }
                }
                phase += 1;
                if phase == qq {
                    phase = 0;
                }
            }
        }
    }
}

pub struct LineMaps {
    pub lambda: Vec<f32>,
    pub zline: Option<Vec<f32>>,
    pub z_point: Vec<f32>,
}

/// Compute the statistics of one frame. g is the PSF-smoothed double-whitened
/// frame; gt is its transpose (rows = original columns) and is only needed when
/// the library contains steep hypotheses.
#[allow(clippy::too_many_arguments)]
fn line_maps(
    g: &[f32],
    gt: Option<&[f32]>,
    h: usize,
    w: usize,
    library: &Library,
    floor: f32,
    keep_zline: bool,
) -> LineMaps {
    let n = h * w;
    let mut lambda = vec![floor; n];
    let mut zline = if keep_zline {
        Some(vec![floor; n])
    } else {
        None
    };
    // norms already carry sigma_w
    let point_inv = 1.0 / library.point_norm.max(1e-30);
    let z_point: Vec<f32> = g.iter().map(|v| (*v as f64 * point_inv) as f32).collect();
    for i in 0..n {
        if z_point[i] > lambda[i] {
            lambda[i] = z_point[i];
        }
    }
    let groups = group_directions(&library.lines);
    for group in &groups {
        if group.steep {
            if let Some(t) = gt {
                run_direction(
                    t,
                    w,
                    h,
                    group,
                    &library.lines,
                    &mut lambda,
                    zline.as_deref_mut(),
                    floor,
                );
            }
        } else {
            run_direction(
                g,
                h,
                w,
                group,
                &library.lines,
                &mut lambda,
                zline.as_deref_mut(),
                floor,
            );
        }
    }
    LineMaps {
        lambda,
        zline,
        z_point,
    }
}

/// Evaluate every hypothesis at one integer pixel, in library order
/// (the point kernel first, then the lines).
#[allow(clippy::too_many_arguments)]
fn eval_all_at(
    g: &[f32],
    gt: Option<&[f32]>,
    h: usize,
    w: usize,
    library: &Library,
    x: usize,
    y: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(1 + library.lines.len());
    let pk = &library.point;
    let (cy, cx) = pk.center();
    let mut acc = 0.0f64;
    for i in 0..pk.rows {
        let yy = y as isize + i as isize - cy as isize;
        if yy < 0 || yy >= h as isize {
            continue;
        }
        for j in 0..pk.cols {
            let xx = x as isize + j as isize - cx as isize;
            if xx < 0 || xx >= w as isize {
                continue;
            }
            acc += pk.data[i * pk.cols + j] * g[yy as usize * w + xx as usize] as f64;
        }
    }
    out.push(acc / library.point_norm.max(1e-30));
    for hyp in &library.lines {
        let (buf, br, bc, bx, by) = if hyp.steep {
            match gt {
                Some(t) => (t, w, h, y, x),
                None => {
                    out.push(f64::NEG_INFINITY);
                    continue;
                }
            }
        } else {
            (g, h, w, x, y)
        };
        let n1 = hyp.n1();
        let base = bx as i64 - n1 as i64;
        let phase = ((base % hyp.q as i64) + hyp.q as i64) as usize % hyp.q;
        let mid = (phase + n1 * hyp.p) / hyp.q;
        let inv = 1.0 / hyp.n as f64 / hyp.norms[phase];
        let mut acc = 0.0f64;
        for j in 0..hyp.n {
            let col = base + j as i64;
            if col < 0 || col >= bc as i64 {
                continue;
            }
            let row_off = (((phase + j * hyp.p) / hyp.q) as i64 - mid as i64) * hyp.sign as i64;
            let row = by as i64 + row_off;
            if row < 0 || row >= br as i64 {
                continue;
            }
            acc += buf[row as usize * bc + col as usize] as f64;
        }
        out.push(acc * inv);
    }
    out
}

fn blur_1d(data: &[f64], g: &[f64]) -> Vec<f64> {
    let r = g.len() / 2;
    let n = data.len();
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0;
        for (k, w) in g.iter().enumerate() {
            let j = i as isize + k as isize - r as isize;
            if j < 0 || j >= n as isize {
                continue;
            }
            acc += w * data[j as usize];
        }
        out[i] = acc;
    }
    out
}

fn grad_1d(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let a = data[(i + 1).min(n - 1)];
        let b = data[i.saturating_sub(1)];
        out[i] = 0.5 * (a - b);
    }
    out
}

/// The two separable PSF autocorrelations used for the exact kernel norms.
struct ProfileTables {
    plain_x: Vec<f64>,
    plain_y: Vec<f64>,
    grad_x: Vec<f64>,
    grad_y: Vec<f64>,
}

fn profile_tables(sigma_psf: f64, whitener: &Whitener) -> ProfileTables {
    let g = gauss_1d(sigma_psf);
    let wx = blur_1d(&g, &whitener.hx);
    let wy = blur_1d(&g, &whitener.hy);
    let dx = grad_1d(&wx);
    let dy = grad_1d(&wy);
    let lag = g.len() / 2 + whitener.support() + 3;
    ProfileTables {
        plain_x: autocorr_1d(&wx, lag),
        plain_y: autocorr_1d(&wy, lag),
        grad_x: autocorr_1d(&dx, lag),
        grad_y: autocorr_1d(&dy, lag),
    }
}
// ---------------------------------------------------------------------------
// frequency-domain matched filter (zero padded to >= N + M - 1)
// ---------------------------------------------------------------------------

/// z = IFFT(FFT(image) conj(FFT(k))) / (sigma ||k||) on the zero-padded grid.
pub fn mf_z_freq(image: &[f64], h: usize, w: usize, k: &KernelGrid, sigma: f64) -> Vec<f64> {
    let (fy, fx) = (next_pow2(h + k.rows), next_pow2(w + k.cols));
    let mut re = vec![0.0f64; fx * fy];
    let mut im = vec![0.0f64; fx * fy];
    for y in 0..h {
        re[y * fx..y * fx + w].copy_from_slice(&image[y * w..(y + 1) * w]);
    }
    let mut fft = Fft2::new(fx, fy);
    fft.transform(&mut re, &mut im, false);
    let mut kre = vec![0.0f64; fx * fy];
    let mut kim = vec![0.0f64; fx * fy];
    for i in 0..k.rows {
        for j in 0..k.cols {
            kre[i * fx + j] = k.data[i * k.cols + j];
        }
    }
    fft.transform(&mut kre, &mut kim, false);
    for t in 0..re.len() {
        let (ar, ai, br, bi) = (kre[t], kim[t], re[t], im[t]);
        re[t] = ar * br + ai * bi;
        im[t] = ar * bi - ai * br;
    }
    fft.transform(&mut re, &mut im, true);
    let (cy, cx) = k.center();
    let norm = sigma * k.sum_sq.sqrt().max(1e-30);
    let mut out = vec![0.0f64; h * w];
    // IFFT(conj(K) D)[q] = sum_p k[p] d[q + p] with p counted from the kernel's
    // top-left corner, so the centre-referenced value at (y, x) lives at
    // q = (y - cy, x - cx), taken modulo the padded grid.
    for y in 0..h {
        let sy = (y + fy - cy) % fy;
        for x in 0..w {
            let sx = (x + fx - cx) % fx;
            out[y * w + x] = re[sy * fx + sx] / norm;
        }
    }
    out
}

/// Apply the whitening filter to a kernel (symmetric zero padding keeps the
/// kernel centre at its centre).
fn whiten_kernel(k: &KernelGrid, w: &Whitener) -> KernelGrid {
    if w.is_identity() {
        return k.clone();
    }
    let pad = w.support();
    let rows = k.rows + 2 * pad;
    let cols = k.cols + 2 * pad;
    let mut padded = vec![0.0f64; rows * cols];
    for i in 0..k.rows {
        for j in 0..k.cols {
            padded[(i + pad) * cols + j + pad] = k.data[i * k.cols + j];
        }
    }
    let out = blur_separable_f64(&padded, rows, cols, &w.hx, &w.hy);
    KernelGrid::new(rows, cols, out)
}

fn transpose_kernel(k: &KernelGrid) -> KernelGrid {
    let rows = k.cols;
    let cols = k.rows;
    let mut data = vec![0.0f64; rows * cols];
    for i in 0..k.rows {
        for j in 0..k.cols {
            data[j * k.rows + i] = k.data[i * k.cols + j];
        }
    }
    KernelGrid::new(rows, cols, data)
}

/// Reference implementation of the same statistic on the frequency path: one
/// zero-padded inverse FFT per hypothesis (kept for cross checks on small
/// frames, the line engine is the production path).
#[allow(clippy::too_many_arguments)]
fn freq_maps(
    img_w: &[f64],
    h: usize,
    w: usize,
    library: &Library,
    whitener: &Whitener,
    sigma_w: f64,
    floor: f32,
    keep_zline: bool,
) -> LineMaps {
    let n = h * w;
    let mut lambda = vec![floor; n];
    let mut zline = if keep_zline {
        Some(vec![floor; n])
    } else {
        None
    };
    let kp = whiten_kernel(&library.point, whitener);
    let point = mf_z_freq(img_w, h, w, &kp, sigma_w);
    let z_point: Vec<f32> = point.iter().map(|v| *v as f32).collect();
    for i in 0..n {
        if z_point[i] > lambda[i] {
            lambda[i] = z_point[i];
        }
    }
    for hyp in &library.lines {
        let mut grid = kernel_from_samples(&hyp.weighted_offsets(0), library.sigma_psf);
        if hyp.steep {
            grid = transpose_kernel(&grid);
        }
        let kg = whiten_kernel(&grid, whitener);
        let z = mf_z_freq(img_w, h, w, &kg, sigma_w);
        for i in 0..n {
            let v = z[i] as f32;
            if v > lambda[i] {
                lambda[i] = v;
            }
            if let Some(zl) = zline.as_mut() {
                if v > zl[i] {
                    zl[i] = v;
                }
            }
        }
    }
    LineMaps {
        lambda,
        zline,
        z_point,
    }
}

/// The plain Gaussian PSF smoothed frame, as f32.
fn smoothed(img: &[f64], h: usize, w: usize, sigma_psf: f64) -> Vec<f32> {
    let g = gauss_1d(sigma_psf);
    blur_separable_f64(img, h, w, &g, &g)
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

fn transpose_f32(src: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            out[x * h + y] = src[y * w + x];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Gumbel calibration of the filtered noise field
// ---------------------------------------------------------------------------

/// Method-of-quantiles Gumbel fit u(p) = mu + beta * (-ln(-ln p)) on the upper
/// tail of a field.
fn fit_gumbel(values: &mut Vec<f64>) -> (f64, f64) {
    values.sort_unstable_by(|a, b| a.total_cmp(b));
    let levels = [0.99, 0.995, 0.999, 0.9995, 0.9999];
    let mut a: Vec<f64> = Vec::new();
    let mut u: Vec<f64> = Vec::new();
    for &p in levels.iter() {
        let q = quantile_sorted(values, p);
        if !q.is_finite() {
            continue;
        }
        a.push(-(-(p.ln())).ln());
        u.push(q);
    }
    if a.len() < 2 {
        return (0.0, 0.2);
    }
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mu = u.iter().sum::<f64>() / n;
    let mut cov = 0.0f64;
    let mut var = 0.0f64;
    for i in 0..a.len() {
        cov += (a[i] - ma) * (u[i] - mu);
        var += (a[i] - ma) * (a[i] - ma);
    }
    let beta = if var > 1e-12 { cov / var } else { 0.2 };
    let (mu, beta) = (mu - beta * ma, beta.max(1e-3));
    (mu, beta)
}

/// Correlation area of a field: (1 + 2 sum rho_x) * (1 + 2 sum rho_y).
fn correlation_area(field: &[f32], h: usize, w: usize, margin: usize, lags: usize) -> f64 {
    let (y0, y1) = (margin.min(h / 2), h - margin.min(h / 2));
    let (x0, x1) = (margin.min(w / 2), w - margin.min(w / 2));
    if y1 <= y0 + 2 || x1 <= x0 + 2 {
        return 1.0;
    }
    let mut mean = 0.0f64;
    let mut n = 0.0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            mean += field[y * w + x] as f64;
            n += 1.0;
        }
    }
    mean /= n;
    let mut var = 0.0f64;
    let mut rho_x = vec![0.0f64; lags + 1];
    let mut rho_y = vec![0.0f64; lags + 1];
    for y in y0..y1 {
        for x in x0..x1 {
            let v = field[y * w + x] as f64 - mean;
            var += v * v;
        }
    }
    if var <= 0.0 {
        return 1.0;
    }
    for d in 1..=lags {
        let mut sx = 0.0f64;
        let mut nx = 0.0f64;
        let mut sy = 0.0f64;
        let mut ny = 0.0f64;
        for y in y0..y1 {
            for x in x0..x1 {
                let v = field[y * w + x] as f64 - mean;
                if x + d < x1 {
                    sx += v * (field[y * w + x + d] as f64 - mean);
                    nx += 1.0;
                }
                if y + d < y1 {
                    sy += v * (field[(y + d) * w + x] as f64 - mean);
                    ny += 1.0;
                }
            }
        }
        rho_x[d] = if nx > 0.0 { sx / nx / (var / n) } else { 0.0 };
        rho_y[d] = if ny > 0.0 { sy / ny / (var / n) } else { 0.0 };
    }
    let ax = 1.0 + 2.0 * rho_x[1..].iter().sum::<f64>();
    let ay = 1.0 + 2.0 * rho_y[1..].iter().sum::<f64>();
    (ax.max(1.0)) * (ay.max(1.0))
}

/// 1-D spectral density of an autocorrelation function on an n-point grid.
fn spectrum_1d(rho: &[f64], n: usize) -> Vec<f64> {
    let lag = (rho.len() - 1) / 2;
    (0..n)
        .map(|f| {
            let mut s = rho[lag];
            for d in 1..=lag {
                s += 2.0 * rho[lag + d] * (2.0 * std::f64::consts::PI * f as f64 * d as f64 / n as f64).cos();
            }
            s.max(1e-6)
        })
        .collect()
}

/// Synthetic stationary Gaussian field with the estimated autocorrelation and
/// the measured per-pixel scatter (used only for the H0 calibration).
pub fn synthetic_field(size: usize, noise: &NoiseModel, noise_sigma: f64, seed: u64) -> Vec<f64> {
    let s = size.max(8);
    let nn = s * s;
    let mut re = vec![0.0f64; nn];
    let mut im = vec![0.0f64; nn];
    let mut rng = Rng::new(seed | 1);
    for i in 0..nn {
        re[i] = rng.normal();
        im[i] = rng.normal();
    }
    let sx = spectrum_1d(&noise.rho_x, s);
    let sy = spectrum_1d(&noise.rho_y, s);
    for y in 0..s {
        for x in 0..s {
            let f = (sx[x] * sy[y]).sqrt() / (nn as f64).sqrt();
            re[y * s + x] *= f;
            im[y * s + x] *= f;
        }
    }
    let mut fft = Fft2::new(s, s);
    fft.transform(&mut re, &mut im, true);
    let mut scale = mad_sigma(&mut re.clone());
    if !(scale.is_finite() && scale > 0.0) {
        scale = 1.0;
    }
    re.iter().map(|v| v / scale * noise_sigma).collect()
}

// ---------------------------------------------------------------------------
// public options, detections, result
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Auto,
    Freq,
    Line,
}

pub struct Options {
    pub psf_fwhm: f64,
    pub lengths: Vec<f64>,
    pub n_angles: usize,
    pub whiten: bool,
    pub threshold_fa: f64,
    pub engine: Engine,
    pub max_detections: usize,
    pub max_refinements: usize,
    pub min_separation_px: f64,
    pub calib_size: usize,
    pub seed: u64,
    pub background_block: usize,
    pub floor: f64,
    pub keep_zline: bool,
    /// Run the second-stage (fine) search; `--fast` turns it off.
    pub refine: bool,
    pub fine: FineCfg,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            psf_fwhm: 1.93,
            lengths: vec![16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],
            n_angles: 12,
            whiten: true,
            threshold_fa: 1.0,
            engine: Engine::Auto,
            max_detections: 200,
            max_refinements: 200,
            min_separation_px: 5.0,
            calib_size: 0,
            seed: 20_260_330,
            background_block: 64,
            floor: 2.5,
            keep_zline: false,
            refine: true,
            fine: FineCfg::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub x: f64,
    pub y: f64,
    pub lambda: f64,
    pub lr: f64,
    pub length_px: f64,
    pub pa_deg: f64,
    pub sigma_x_px: f64,
    /// Lambda of the coarse (grid) stage at this candidate.
    pub coarse_lambda: f64,
    /// Lambda of the fine stage, when it ran and was accepted.
    pub fine_lambda: Option<f64>,
    pub refined: bool,
}

pub struct GlrtResult {
    pub shape: (usize, usize),
    pub sigma_adu: f64,
    pub noise: NoiseModel,
    pub lengths: Vec<f64>,
    pub n_angles: usize,
    pub n_hypotheses: usize,
    pub gumbel: (f64, f64),
    pub thresholds: [f64; 3],
    /// Thresholds of the coarse stage alone (no multiplicity for the fine search).
    pub thresholds_coarse: [f64; 3],
    /// Largest number of fine trials absorbed by one candidate; 0 without F2.
    pub refine_trials: usize,
    /// ln-based threshold inflation paid for the fine search, in z units.
    pub refine_inflation: f64,
    pub refined_detections: usize,
    pub fine_cfg: Option<FineCfg>,
    pub n_eff: f64,
    pub detections: Vec<Detection>,
    pub std_z_point: f64,
    pub std_lambda: f64,
    pub margin_px: usize,
    pub calib_size: usize,
    pub engine: String,
    pub timings_s: Vec<(String, f64)>,
    pub lambda: Vec<f32>,
    pub lr: Option<Vec<f32>>,
    pub lambda_floor: f64,
}

impl GlrtResult {
    /// JSON matching the agreed output contract (plus a meta block).
    pub fn to_json(&self, image: &str, maps: Option<(String, String)>) -> Value {
        let detections: Vec<Value> = self
            .detections
            .iter()
            .enumerate()
            .map(|(i, d)| {
                json!({
                    "rank": i + 1,
                    "x": round4(d.x),
                    "y": round4(d.y),
                    "lambda": round4(d.lambda),
                    "lr": round4(d.lr),
                    "L_px": round4(d.length_px),
                    "pa_deg": round4(d.pa_deg),
                    "sigma_x_px": round6(d.sigma_x_px),
                    "refined": d.refined,
                    "coarse_lambda": round4(d.coarse_lambda),
                    "fine_lambda": d.fine_lambda.map(round4),
                })
            })
            .collect();
        let refinement = match self.fine_cfg {
            Some(cfg) => json!({
                "enabled": true,
                "trials": self.refine_trials,
                "trials_definition": "fine trials absorbed by the single most refined candidate",
                "phi_step_deg": cfg.phi_step_deg,
                "L_step_frac": cfg.l_step_frac,
                "phi_half_range_deg": cfg.phi_half_range_deg,
                "crop_px": cfg.crop_px,
                "refined_detections": self.refined_detections,
                "threshold_inflation": round4(self.refine_inflation),
                "thresholds_coarse": {
                    "fa_1": round4(self.thresholds_coarse[0]),
                    "fa_10": round4(self.thresholds_coarse[1]),
                    "fa_100": round4(self.thresholds_coarse[2]),
                },
            }),
            None => json!({
                "enabled": false,
                "trials": 0,
                "phi_step_deg": Value::Null,
                "L_step_frac": Value::Null,
            }),
        };
        let mut out = json!({
            "image": image,
            "shape": [self.shape.0, self.shape.1],
            "sigma_adu": round4(self.sigma_adu),
            "noise_correlation_length_px": round4(self.noise.corr_len_px),
            "n_eff": self.n_eff.round(),
            "kernels": {
                "lengths_px": self.lengths,
                "n_angles": self.n_angles,
                "n_hypotheses": self.n_hypotheses,
                "n_hypotheses_total": self.n_hypotheses + self.refine_trials,
            },
            "gumbel": {"mu": round4(self.gumbel.0), "beta": round4(self.gumbel.1)},
            "thresholds": {
                "fa_1": round4(self.thresholds[0]),
                "fa_10": round4(self.thresholds[1]),
                "fa_100": round4(self.thresholds[2]),
            },
            "refinement": refinement,
            "detections": detections,
        });
        if let (Some((l, r)), Some(obj)) = (
            maps.clone(),
            out.as_object_mut(),
        ) {
            obj.insert("maps".into(), json!({"lambda": l, "lr": r}));
            obj.insert("map_shape".into(), json!([self.shape.0, self.shape.1]));
        }
        if let Some(obj) = out.as_object_mut() {
            obj.insert(
                "meta".into(),
                json!({
                    "engine": self.engine,
                    "std_z_point": round4(self.std_z_point),
                    "std_lambda": round4(self.std_lambda),
                    "margin_px": self.margin_px,
                    "calib_size": self.calib_size,
                    "lambda_floor": self.lambda_floor,
                    "noise_blocks_used": self.noise.blocks_used,
                    "refine_trials": self.refine_trials,
                    "refine_inflation": round4(self.refine_inflation),
                    "edge_normalisation": "kernel norm uses only the in-frame samples (position dependent)",
                    "refine_trials": self.refine_trials,
                    "refine_inflation": round4(self.refine_inflation),
                    "edge_normalisation": "kernel norm uses only the in-frame samples (position dependent)",
                    "n_eff_definition": "interior pixels / correlation area of the calibrated Lambda field",
                    "pa_deg_definition": "atan2(dy, dx) of the segment axis in pixel index coordinates, mod 180 deg",
                    "timings_s": self.timings_s.iter().map(|(k, v)| (k.clone(), round3(*v))).collect::<Vec<_>>(),
                }),
            );
        }
        out
    }

    /// Write the optional f32 raw maps (row major, native little endian).
    pub fn write_maps(&self, dir: &Path) -> std::io::Result<(String, String)> {
        use std::io::Write;
        std::fs::create_dir_all(dir)?;
        let mut f = std::fs::File::create(dir.join("lambda.f32"))?;
        let mut bytes = Vec::with_capacity(self.lambda.len() * 4);
        for v in &self.lambda {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&bytes)?;
        let mut f = std::fs::File::create(dir.join("lr.f32"))?;
        let mut bytes = Vec::with_capacity(self.lambda.len() * 4);
        match &self.lr {
            Some(lr) => {
                for v in lr {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            None => {
                for _ in 0..self.lambda.len() {
                    bytes.extend_from_slice(&0.0f32.to_le_bytes());
                }
            }
        }
        f.write_all(&bytes)?;
        Ok(("lambda.f32".into(), "lr.f32".into()))
    }
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}
fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}
fn round6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

// ---------------------------------------------------------------------------
// the pipeline
// ---------------------------------------------------------------------------

struct FrameStats {
    maps: LineMaps,
    g: Vec<f32>,
    gt: Option<Vec<f32>>,
}

#[allow(clippy::too_many_arguments)]
fn frame_stats(
    img_w2: &[f64],
    h: usize,
    w: usize,
    library: &Library,
    sigma_psf: f64,
    floor: f32,
    keep_zline: bool,
) -> FrameStats {
    let g = smoothed(img_w2, h, w, sigma_psf);
    let need_steep = library.lines.iter().any(|l| l.steep);
    let gt = if need_steep {
        Some(transpose_f32(&g, h, w))
    } else {
        None
    };
    let maps = line_maps(&g, gt.as_deref(), h, w, library, floor, keep_zline);
    FrameStats { maps, g, gt }
}

/// Full single-frame run on an in-memory array.
pub fn run_array(opts: &Options, image: &[f64], h: usize, w: usize) -> Result<GlrtResult, String> {
    if h < 3 || w < 3 || image.len() != h * w {
        return Err(format!(
            "image shape {h}x{w} is inconsistent with {} samples",
            image.len()
        ));
    }
    if !(opts.psf_fwhm > 0.0) {
        return Err("--psf-fwhm must be positive".into());
    }
    let mut timings: Vec<(String, f64)> = Vec::new();
    let t_all = Instant::now();
    let mut lap = Instant::now();
    let sigma_psf = opts.psf_fwhm / FWHM_TO_SIGMA;

    let bg = background_median(image, h, w, opts.background_block);
    let residual: Vec<f64> = image
        .iter()
        .zip(bg.iter())
        .map(|(a, b)| a - b)
        .collect();
    let noise = estimate_noise(&residual, h, w);
    timings.push(("background_noise".into(), lap.elapsed().as_secs_f64()));
    lap = Instant::now();

    let whitener = if opts.whiten {
        Whitener {
            hx: whitening_fir(&noise.rho_x, w, 9),
            hy: whitening_fir(&noise.rho_y, h, 9),
        }
    } else {
        Whitener::identity()
    };
    let img_w = whitener.apply(&residual, h, w);
    let mut scratch = img_w.clone();
    let sigma_w = {
        let s = mad_sigma(&mut scratch);
        if s.is_finite() && s > 0.0 {
            s
        } else {
            noise.sigma_adu
        }
    };
    let img_w2 = whitener.apply(&img_w, h, w);
    let profiles = profile_tables(sigma_psf, &whitener);
    let library = build_library(
        &opts.lengths,
        opts.n_angles,
        sigma_psf,
        sigma_w,
        (&profiles.plain_y, &profiles.plain_x),
        (&profiles.grad_y, &profiles.grad_x),
    );
    timings.push(("whiten".into(), lap.elapsed().as_secs_f64()));
    lap = Instant::now();

    let max_length = opts.lengths.iter().cloned().fold(0.0f64, f64::max);
    let psf_pad = (PSF_TRUNC_SIGMA * sigma_psf).ceil() as usize + 1;
    // The kernel norm now depends on position (F1), so a truncated hypothesis at
    // the frame edge is still a unit-variance test. The margin therefore only has
    // to exclude the band where the whitening and the PSF blur themselves lose
    // samples -- half the longest kernel is no longer required, which is what
    // makes an event 240 px from the border detectable at all.
    let guard = 2 * psf_pad + 2 * whitener.support() + 4;
    let margin = guard.min((h.min(w) / 4).max(4));

    let floor = opts.floor as f32;
    let stats = if opts.engine == Engine::Freq {
        if h * w > 4_000_000 {
            return Err("--engine freq is a reference path for small frames only".into());
        }
        let maps = freq_maps(
            &img_w,
            h,
            w,
            &library,
            &whitener,
            sigma_w,
            floor,
            opts.keep_zline,
        );
        let g = smoothed(&img_w2, h, w, sigma_psf);
        let gt = if library.lines.iter().any(|l| l.steep) {
            Some(transpose_f32(&g, h, w))
        } else {
            None
        };
        FrameStats { maps, g, gt }
    } else {
        frame_stats(&img_w2, h, w, &library, sigma_psf, floor, opts.keep_zline)
    };
    timings.push(("matched_filter".into(), lap.elapsed().as_secs_f64()));
    lap = Instant::now();

    // noise-only statistics over the interior margin
    let (y0, y1) = (margin, h - margin);
    let (x0, x1) = (margin, w - margin);
    let mut lambda_in: Vec<f32> = Vec::new();
    let mut zs = 0.0f64;
    let mut zs2 = 0.0f64;
    let mut zn = 0.0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            lambda_in.push(stats.maps.lambda[y * w + x]);
            let v = stats.maps.z_point[y * w + x] as f64;
            zs += v;
            zs2 += v * v;
            zn += 1.0;
        }
    }
    let std_z_point = if zn > 1.0 {
        let mean = zs / zn;
        (zs2 / zn - mean * mean).max(0.0).sqrt()
    } else {
        f64::NAN
    };
    let std_lambda = if lambda_in.len() > 2 {
        let mean = lambda_in.iter().map(|v| *v as f64).sum::<f64>() / lambda_in.len() as f64;
        let var = lambda_in
            .iter()
            .map(|v| (*v as f64 - mean).powi(2))
            .sum::<f64>()
            / lambda_in.len() as f64;
        var.sqrt()
    } else {
        f64::NAN
    };
    timings.push(("interior_stats".into(), lap.elapsed().as_secs_f64()));
    lap = Instant::now();

    // ---- H0 calibration on a synthetic field with the measured noise model
    let min_calib = 2 * margin + 256;
    let calib_size = if opts.calib_size > 0 {
        next_pow2(opts.calib_size.max(128))
    } else {
        next_pow2(min_calib.max(1024).min(4096))
    };
    let synth = synthetic_field(calib_size, &noise, noise.sigma_adu, opts.seed);
    let synth_w = whitener.apply(&synth, calib_size, calib_size);
    let synth_w2 = whitener.apply(&synth_w, calib_size, calib_size);
    let calib_stats = frame_stats(
        &synth_w2,
        calib_size,
        calib_size,
        &library,
        sigma_psf,
        f32::NEG_INFINITY,
        false,
    );
    let calib_margin = margin.min(calib_size / 2 - 1);
    let mut tail: Vec<f64> = Vec::new();
    for y in calib_margin..calib_size - calib_margin {
        for x in calib_margin..calib_size - calib_margin {
            tail.push(calib_stats.maps.lambda[y * calib_size + x] as f64);
        }
    }
    let (mu, beta) = fit_gumbel(&mut tail);
    let area = correlation_area(
        &calib_stats.maps.lambda,
        calib_size,
        calib_size,
        calib_margin,
        4,
    );
    let n_interior = ((y1 - y0) * (x1 - x0)) as f64;
    let n_eff = (n_interior / area).max(1.0);
    // trials = 1 keeps the coarse multiplicity (already inside the fitted Gumbel);
    // trials = m pays the union bound for m independent fine trials at one pixel.
    let thr_for = |nfa: f64, trials: f64| mu + beta * (n_eff * trials / nfa).ln();
    let thresholds_coarse = [thr_for(1.0, 1.0), thr_for(10.0, 1.0), thr_for(100.0, 1.0)];
    let coarse_threshold = if opts.threshold_fa > 0.0 {
        thr_for(opts.threshold_fa, 1.0)
    } else {
        thresholds_coarse[0]
    };
    timings.push(("calibration".into(), lap.elapsed().as_secs_f64()));
    lap = Instant::now();

    // ---- candidate extraction and per-candidate refinement
    let ctx = RefineCtx {
        ax: &profiles.plain_x,
        ay: &profiles.plain_y,
        bx: &profiles.grad_x,
        by: &profiles.grad_y,
        sigma_w,
        max_length: (1.6 * max_length).max(64.0),
    };
    let fine_cfg = if opts.refine { Some(opts.fine) } else { None };
    let (mut detections, refine_trials) = detect(
        &stats,
        h,
        w,
        &library,
        &opts.lengths,
        opts.n_angles,
        coarse_threshold as f32,
        margin,
        opts.max_detections,
        opts.min_separation_px,
        &ctx,
        opts.max_refinements,
        fine_cfg.as_ref(),
    );
    // The fine stage multiplies the number of hypotheses examined per candidate,
    // so the threshold has to move with log(trials); a candidate that was not
    // refined only pays for the coarse multiplicity.
    let refine_mult = refine_trials.max(1) as f64;
    let thresholds = [
        thr_for(1.0, refine_mult),
        thr_for(10.0, refine_mult),
        thr_for(100.0, refine_mult),
    ];
    let threshold = if opts.threshold_fa > 0.0 {
        thr_for(opts.threshold_fa, refine_mult)
    } else {
        thresholds[0]
    };
    let refine_inflation = beta * refine_mult.ln();
    let refined_detections = detections.iter().filter(|d| d.refined).count();
    detections.retain(|d| {
        let thr = if d.refined {
            threshold
        } else {
            coarse_threshold
        };
        d.lambda >= thr
    });

    let mut lr_map: Option<Vec<f32>> = None;
    if let Some(zline) = &stats.maps.zline {
        let mut lr = vec![0.0f32; h * w];
        for i in 0..h * w {
            lr[i] = zline[i] - stats.maps.z_point[i];
        }
        lr_map = Some(lr);
    }
    timings.push(("detections".into(), lap.elapsed().as_secs_f64()));
    timings.push(("total".into(), t_all.elapsed().as_secs_f64()));

    let engine = match opts.engine {
        Engine::Auto => "line (Radon-equivalent line integral of the PSF-smoothed frame)",
        Engine::Line => "line",
        Engine::Freq => "freq",
    };
    Ok(GlrtResult {
        shape: (h, w),
        sigma_adu: noise.sigma_adu,
        noise,
        lengths: opts.lengths.clone(),
        n_angles: opts.n_angles,
        n_hypotheses: 1 + library.lines.len(),
        gumbel: (mu, beta),
        thresholds,
        thresholds_coarse,
        refine_trials,
        refine_inflation,
        refined_detections,
        fine_cfg,
        n_eff,
        detections,
        std_z_point,
        std_lambda,
        margin_px: margin,
        calib_size,
        engine: engine.to_string(),
        timings_s: timings,
        lambda: stats.maps.lambda.clone(),
        lr: lr_map,
        lambda_floor: opts.floor,
    })
}

/// Length and orientation from the (angle, length) grid of z at one pixel.
struct GridInfo {
    best: f64,
    best_i: usize,
    length_px: f64,
    pa_deg: f64,
}

fn grid_info(zs: &[f64], library: &Library, nl: usize, n_angles: usize) -> GridInfo {
    let mut best = f64::NEG_INFINITY;
    let mut best_i = 0usize;
    for (i, z) in zs.iter().enumerate().skip(1) {
        if *z > best {
            best = *z;
            best_i = i - 1;
        }
    }
    let nl = nl.max(1);
    let a_best = best_i / nl;
    let l_best = best_i % nl;
    let grid = |a: usize, l: usize| zs[1 + (a * nl + l).min(zs.len().saturating_sub(2))];
    let last = library.lines.len() - 1;
    let mut length_px = library.lines[best_i.min(last)].length_px;
    if l_best > 0 && l_best + 1 < nl {
        let z0 = grid(a_best, l_best - 1);
        let z1 = grid(a_best, l_best);
        let z2 = grid(a_best, l_best + 1);
        let den = z0 - 2.0 * z1 + z2;
        if den < -1e-9 {
            let u0 = library.lines[(a_best * nl + l_best - 1).min(last)].length_px.ln();
            let u1 = library.lines[(a_best * nl + l_best).min(last)].length_px.ln();
            let u2 = library.lines[(a_best * nl + l_best + 1).min(last)].length_px.ln();
            let du = 0.5 * (u2 - u0);
            let off = 0.5 * du * (z0 - z2) / den;
            length_px = (u1 + off.clamp(-du, du)).exp();
        }
    }
    let mut pa_deg = library.lines[best_i.min(last)].angle_deg;
    if n_angles >= 3 {
        let a_prev = (a_best + n_angles - 1) % n_angles;
        let a_next = (a_best + 1) % n_angles;
        let z0 = grid(a_prev, l_best);
        let z1 = grid(a_best, l_best);
        let z2 = grid(a_next, l_best);
        let den = z0 - 2.0 * z1 + z2;
        if den < -1e-9 {
            let step = 180.0 / n_angles as f64;
            let off = (0.5 * step * (z0 - z2) / den).clamp(-0.5 * step, 0.5 * step);
            pa_deg = fold_angle(pa_deg + off);
        }
    }
    GridInfo {
        best,
        best_i,
        length_px,
        pa_deg,
    }
}

// ---------------------------------------------------------------------------
// continuous refinement of (length, position angle) per candidate
//
// A 15 deg hypothesis grid cannot match a long streak: over L/2 px an angle
// error dtheta displaces the kernel by (L/2) dtheta, so max_theta z is maximised
// by a short kernel unless (L/2) dtheta <~ sigma_psf. The argmax length is
// therefore biased low and has to be recovered by a continuous search. The
// refinement maximises z over (L, phi) with sub-pixel sampling of the smoothed
// frame; it costs O(L) per trial angle and O(1) per additional length thanks to
// a prefix sum along the trial direction.
// ---------------------------------------------------------------------------

struct RefineCtx<'a> {
    ax: &'a [f64],
    ay: &'a [f64],
    bx: &'a [f64],
    by: &'a [f64],
    sigma_w: f64,
    max_length: f64,
}

/// Fine-search configuration of the second stage.
#[derive(Clone, Copy)]
pub struct FineCfg {
    pub phi_step_deg: f64,
    pub l_step_frac: f64,
    pub phi_half_range_deg: f64,
    pub crop_px: usize,
}

impl Default for FineCfg {
    fn default() -> Self {
        Self {
            phi_step_deg: 0.25,
            l_step_frac: 1.05,
            phi_half_range_deg: 5.0,
            crop_px: 1536,
        }
    }
}

struct FineResult {
    z: f64,
    length: f64,
    pa_deg: f64,
    x: f64,
    y: f64,
    sigma_x_coeff: f64,
    trials: usize,
}

fn table_at(table: &[f64], lag: f64) -> f64 {
    let a = lag.abs();
    let i = a as usize;
    if i + 1 >= table.len() {
        return 0.0;
    }
    let f = a - i as f64;
    table[i] * (1.0 - f) + table[i + 1] * f
}

fn bilinear(g: &[f32], h: usize, w: usize, x: f64, y: f64) -> f64 {
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let (x0, y0) = (x0 as isize, y0 as isize);
    let mut acc = 0.0;
    for (dy, wy) in [(0isize, 1.0 - fy), (1, fy)] {
        let yy = y0 + dy;
        if yy < 0 || yy >= h as isize {
            continue;
        }
        for (dx, wx) in [(0isize, 1.0 - fx), (1, fx)] {
            let xx = x0 + dx;
            if xx < 0 || xx >= w as isize {
                continue;
            }
            acc += wy * wx * g[yy as usize * w + xx as usize] as f64;
        }
    }
    acc
}

/// Second-stage search (F2): maximise the same statistic z over a continuous
/// (L, phi) neighbourhood of one coarse candidate, evaluated on a crop centred on
/// that candidate.
///
/// The crop is intersected with the frame, so the profile only reads real
/// pixels, and the running window is clipped to the crop half-length: that
/// clipping *is* the in-frame part of the trial segment, which is why the same
/// position-dependent normalisation as the coarse stage applies. The angle grid
/// is fine enough that the mismatch loss over L/2 px stays below the PSF width,
/// and the window offset is swept to recover the centre of a flat Lambda ridge.
#[allow(clippy::too_many_arguments)]
fn fine_search(
    g: &[f32],
    h: usize,
    w: usize,
    x0: f64,
    y0: f64,
    pa_hint: f64,
    l_hint: f64,
    ctx: &RefineCtx,
    cfg: &FineCfg,
) -> Option<FineResult> {
    if !(ctx.sigma_w > 0.0) {
        return None;
    }
    let cx = (x0.round() as isize).clamp(0, w as isize - 1) as usize;
    let cy = (y0.round() as isize).clamp(0, h as isize - 1) as usize;
    let half = (cfg.crop_px / 2).max(16);
    let hx = half.min(cx).min(w - 1 - cx);
    let hy = half.min(cy).min(h - 1 - cy);
    let reach = hx.min(hy);
    if reach < 24 {
        return None;
    }
    let (ch, cw) = (2 * hy + 1, 2 * hx + 1);
    let mut crop = vec![0.0f32; ch * cw];
    for i in 0..ch {
        let src = (cy - hy + i) * w + (cx - hx);
        crop[i * cw..i * cw + cw].copy_from_slice(&g[src..src + cw]);
    }
    let (ccx, ccy) = (hx as f64, hy as f64);
    let tau = 2 * reach + 1;
    let max_n = (2 * reach - 1) | 1;
    let mut profile = vec![0.0f64; tau];
    let mut prefix = vec![0.0f64; tau + 1];

    let fill = |phi: f64, profile: &mut Vec<f64>, prefix: &mut Vec<f64>| {
        let rad = phi.to_radians();
        let (c, s) = (rad.cos(), rad.sin());
        for (j, slot) in profile.iter_mut().enumerate() {
            let t = j as f64 - reach as f64;
            *slot = bilinear(&crop, ch, cw, ccx + t * c, ccy + t * s);
        }
        prefix[0] = 0.0;
        for j in 0..tau {
            prefix[j + 1] = prefix[j] + profile[j];
        }
    };
    // retained samples and raw sum of an n-sample window shifted by delta
    let window = |n: usize, delta: isize, prefix: &[f64]| -> (usize, f64) {
        let a0 = ((n - 1) / 2) as isize;
        let lo = (delta - a0).max(-(reach as isize));
        let hi = (delta - a0 + (n as isize - 1)).min(reach as isize);
        if hi < lo {
            return (0, 0.0);
        }
        let used = (hi - lo + 1) as usize;
        let a = (lo + reach as isize) as usize;
        let b = (hi + reach as isize + 1) as usize;
        (used, prefix[b] - prefix[a])
    };

    let n_phi = (((2.0 * cfg.phi_half_range_deg / cfg.phi_step_deg).round() as usize) + 1).max(3);
    let l_ceiling = (max_n as f64).min(ctx.max_length.max(8.0));
    let mut phi_c = fold_angle(pa_hint);
    let mut l_lo = (l_hint * 0.5).max(3.0).min(l_ceiling * 0.5);
    let mut l_hi = (l_hint * 2.0).max(l_lo * 2.0).min(l_ceiling);
    let mut trials = 0usize;
    let mut best = f64::NEG_INFINITY;
    let mut best_phi = phi_c;
    let mut best_grid: Vec<f64> = Vec::new();
    let mut best_ns: Vec<usize> = Vec::new();
    let mut best_k = 0usize;
    let mut best_i = 0usize;
    let mut best_phi_c = phi_c;

    for _pass in 0..3 {
        let mut ns: Vec<usize> = Vec::new();
        let mut l = l_lo;
        while l <= l_hi + 1e-9 {
            ns.push((l.round() as usize).max(3));
            l *= cfg.l_step_frac;
        }
        ns.push((l_hi.round() as usize).max(3));
        ns.sort_unstable();
        ns.dedup();
        let n_top = ns[ns.len() - 1];
        let stride = ns.len();
        let mut grid = vec![f64::NEG_INFINITY; n_phi * stride];
        let mut pass_best = f64::NEG_INFINITY;
        let (mut pk, mut pi) = (0usize, 0usize);
        for k in 0..n_phi {
            let phi = phi_c - cfg.phi_half_range_deg + k as f64 * cfg.phi_step_deg;
            fill(phi, &mut profile, &mut prefix);
            let lags = LagCorr::from_lags(&line_lags_continuous(phi, ctx.ay, ctx.ax, n_top + 1));
            for (i, &n) in ns.iter().enumerate() {
                let (used, sum) = window(n, 0, &prefix);
                if used < 2 {
                    continue;
                }
                let s_n = lags.s_n(used);
                if !(s_n > 1e-30) {
                    continue;
                }
                let z = sum / (ctx.sigma_w * s_n.sqrt());
                grid[k * stride + i] = z;
                trials += 1;
                if z > pass_best {
                    pass_best = z;
                    pk = k;
                    pi = i;
                }
            }
        }
        if !pass_best.is_finite() {
            break;
        }
        let mut moved = false;
        if pass_best > best {
            best = pass_best;
            best_grid = grid;
            best_ns = ns.clone();
            best_k = pk;
            best_i = pi;
            best_phi_c = phi_c;
        }
        // a boundary optimum means the search box, not the data, set the answer
        if pi + 1 == stride && l_hi < l_ceiling - 1.0 {
            l_lo = l_hi * 0.5;
            l_hi = (l_hi * 2.0).min(l_ceiling);
            moved = true;
        }
        if pk == 0 {
            phi_c = fold_angle(phi_c - cfg.phi_half_range_deg);
            moved = true;
        } else if pk + 1 == n_phi {
            phi_c = fold_angle(phi_c + cfg.phi_half_range_deg);
            moved = true;
        }
        if !moved {
            break;
        }
    }
    if !best.is_finite() || best_grid.is_empty() {
        return None;
    }
    let stride = best_ns.len();
    // sub-grid interpolation: the 0.25 deg grid still quantises the angle, and the
    // reported angle is what the (L/2) dphi smearing budget is checked with
    let mut phi_off = 0.0f64;
    if best_k > 0 && best_k + 1 < n_phi {
        let z0 = best_grid[(best_k - 1) * stride + best_i];
        let z1 = best_grid[best_k * stride + best_i];
        let z2 = best_grid[(best_k + 1) * stride + best_i];
        let den = z0 - 2.0 * z1 + z2;
        if den < -1e-12 {
            phi_off = (0.5 * (z0 - z2) / den).clamp(-0.5, 0.5);
        }
    }
    best_phi = fold_angle(
        best_phi_c - cfg.phi_half_range_deg
            + best_k as f64 * cfg.phi_step_deg
            + phi_off * cfg.phi_step_deg,
    );
    let mut length = best_ns[best_i] as f64;
    if best_i > 0 && best_i + 1 < stride {
        let z0 = best_grid[best_k * stride + best_i - 1];
        let z1 = best_grid[best_k * stride + best_i];
        let z2 = best_grid[best_k * stride + best_i + 1];
        let den = z0 - 2.0 * z1 + z2;
        if den < -1e-12 {
            let u0 = (best_ns[best_i - 1] as f64).ln();
            let u1 = (best_ns[best_i] as f64).ln();
            let u2 = (best_ns[best_i + 1] as f64).ln();
            let du = 0.5 * (u2 - u0);
            let off = (0.5 * du * (z0 - z2) / den).clamp(-du, du);
            length = (u1 + off).exp();
        }
    }
    // local polish: even the fine grid quantises (L, phi); a dense 21 x 5 patch
    // around the winner removes the residual grid quantisation of both estimates
    let mut pz = best;
    let mut p_phi = best_phi;
    let mut p_len = length;
    let mut p_ns: Vec<usize> = (-2i32..=2)
        .map(|i| ((length * 1.02f64.powi(i)).round() as usize).max(3).min(max_n))
        .collect();
    p_ns.sort_unstable();
    p_ns.dedup();
    let stride_p = p_ns.len();
    let n_top_p = p_ns[stride_p - 1];
    let mut p_grid = vec![f64::NEG_INFINITY; 21 * stride_p];
    let (mut pk, mut pi) = (0usize, 0usize);
    for i in -10i32..=10 {
        let phi = best_phi + 0.05 * i as f64;
        fill(phi, &mut profile, &mut prefix);
        let lags = LagCorr::from_lags(&line_lags_continuous(phi, ctx.ay, ctx.ax, n_top_p + 1));
        for (j, &n) in p_ns.iter().enumerate() {
            let (used, sum) = window(n, 0, &prefix);
            if used < 2 {
                continue;
            }
            let s_n = lags.s_n(used);
            if !(s_n > 1e-30) {
                continue;
            }
            let z = sum / (ctx.sigma_w * s_n.sqrt());
            p_grid[((i + 10) as usize) * stride_p + j] = z;
            trials += 1;
            if z > pz {
                pz = z;
                pk = (i + 10) as usize;
                pi = j;
            }
        }
    }
    if pz > best {
        let mut off_phi = 0.0f64;
        if pk > 0 && pk + 1 < 21 {
            let z0 = p_grid[(pk - 1) * stride_p + pi];
            let z1 = p_grid[pk * stride_p + pi];
            let z2 = p_grid[(pk + 1) * stride_p + pi];
            let den = z0 - 2.0 * z1 + z2;
            if den < -1e-12 {
                off_phi = (0.5 * (z0 - z2) / den).clamp(-0.5, 0.5);
            }
        }
        p_phi = fold_angle(best_phi + 0.05 * (pk as f64 - 10.0 + off_phi));
        p_len = p_ns[pi] as f64;
        if pi > 0 && pi + 1 < stride_p {
            let z0 = p_grid[pk * stride_p + pi - 1];
            let z1 = p_grid[pk * stride_p + pi];
            let z2 = p_grid[pk * stride_p + pi + 1];
            let den = z0 - 2.0 * z1 + z2;
            if den < -1e-12 {
                let u0 = (p_ns[pi - 1] as f64).ln();
                let u1 = (p_ns[pi] as f64).ln();
                let u2 = (p_ns[pi + 1] as f64).ln();
                let du = 0.5 * (u2 - u0);
                let off = (0.5 * du * (z0 - z2) / den).clamp(-du, du);
                p_len = (u1 + off).exp();
            }
        }
    }
    best_phi = p_phi;
    // the estimate stays continuous: only the sampled geometry is quantised
    length = p_len.clamp(2.0, max_n as f64);
    let n_fin = ((length.round() as usize).max(3)).min(max_n);
    let rad = best_phi.to_radians();
    let (c, s) = (rad.cos(), rad.sin());
    // offset sweep: a uniform streak leaves the Lambda ridge flat along its own
    // axis, so the segment centre has to be estimated by maximising the matched
    // response over the position of the segment along that axis
    fill(best_phi, &mut profile, &mut prefix);
    let lags = LagCorr::from_lags(&line_lags_continuous(best_phi, ctx.ay, ctx.ax, n_fin + 1));
    let dmax = (n_fin / 2).min(reach) as isize;
    let mut best_delta = 0isize;
    let mut best_z = f64::NEG_INFINITY;
    let mut best_used = n_fin;
    let mut delta = -dmax;
    while delta <= dmax {
        let (used, sum) = window(n_fin, delta, &prefix);
        trials += 1;
        if used >= 2 {
            let s_n = lags.s_n(used);
            if s_n > 1e-30 {
                let z = sum / (ctx.sigma_w * s_n.sqrt());
                if z > best_z {
                    best_z = z;
                    best_delta = delta;
                    best_used = used;
                }
            }
        }
        delta += 1;
    }
    if !best_z.is_finite() {
        return None;
    }
    // A segment cut by the frame edge only constrains the part that is inside:
    // every nominal length above that part gives the same statistic, so the
    // reported length is the in-frame sample count, not the kernel length.
    if best_used < n_fin {
        length = best_used as f64;
    }
    // position error per unit z from the discrete samples of the winning segment
    let n1 = (n_fin - 1) as f64 / 2.0;
    let weight = 1.0 / n_fin as f64;
    let mut samples = Vec::with_capacity(n_fin);
    for j in 0..n_fin {
        let t = j as f64 - n1;
        samples.push(((t * s).round() as i64, (t * c).round() as i64, weight));
    }
    let plain = sum_sq_from_samples(&samples, ctx.ay, ctx.ax);
    let grad = sum_sq_from_samples(&samples, ctx.ay, ctx.bx);
    Some(FineResult {
        z: best_z,
        length,
        pa_deg: fold_angle(best_phi),
        x: (cx as f64 + best_delta as f64 * c).clamp(0.0, (w - 1) as f64),
        y: (cy as f64 + best_delta as f64 * s).clamp(0.0, (h - 1) as f64),
        sigma_x_coeff: (plain / grad.max(1e-30)).sqrt(),
        trials,
    })
}

/// Extent of the Lambda ridge along the hypothesis axis through (x, y): the
/// larger of the two distances at which the ridge first drops below floor.
///
/// A uniform streak keeps Lambda high along its whole axis, so the radius of the
/// centroid window has to follow the ridge instead of the local peak.
#[allow(clippy::too_many_arguments)]
fn ridge_extent(
    lambda: &[f32],
    h: usize,
    w: usize,
    x: usize,
    y: usize,
    pa_deg: f64,
    max_extent: usize,
    floor: f64,
) -> f64 {
    let rad = pa_deg.to_radians();
    let (c, s) = (rad.cos(), rad.sin());
    let mut best = 0.0f64;
    for dir in [-1.0f64, 1.0] {
        let mut step = 0.0f64;
        let mut reached = 0.0f64;
        while (step as usize) < max_extent {
            step += 1.0;
            let xx = x as f64 + dir * step * c;
            let yy = y as f64 + dir * step * s;
            if xx < 0.0 || yy < 0.0 || xx > (w - 1) as f64 || yy > (h - 1) as f64 {
                break;
            }
            if bilinear(lambda, h, w, xx, yy) < floor {
                break;
            }
            reached = step;
        }
        best = best.max(reached);
    }
    best
}

/// Local maxima of Lambda above the coarse threshold, refined by the two-stage
/// search. Returns the detections and the largest number of fine trials that any
/// single candidate absorbed: the multiplicity the refined threshold pays for.
#[allow(clippy::too_many_arguments)]
fn detect(
    stats: &FrameStats,
    h: usize,
    w: usize,
    library: &Library,
    lengths: &[f64],
    n_angles: usize,
    threshold: f32,
    margin: usize,
    max_detections: usize,
    min_sep: f64,
    ctx: &RefineCtx,
    max_refinements: usize,
    fine: Option<&FineCfg>,
) -> (Vec<Detection>, usize) {
    let lambda = &stats.maps.lambda;
    let (y0, y1) = (margin.max(1), h.saturating_sub(margin.max(1)));
    let (x0, x1) = (margin.max(1), w.saturating_sub(margin.max(1)));
    let mut cands: Vec<(f32, usize, usize)> = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let v = lambda[y * w + x];
            if !(v >= threshold) {
                continue;
            }
            // A long trail leaves a nearly flat ridge in Lambda, so a strict 3x3 maximum
            // test rejects almost every pixel of it (events A and B both vanish or survive
            // only where the ridge happens to wiggle). Compare over a wider plateau window
            // instead; ties keep the first pixel in raster order and the min_sep pass thins.
            let mut is_max = true;
            let radius: i32 = 4;
            'outer: for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let yy = y as i32 + dy;
                    let xx = x as i32 + dx;
                    if yy < 0 || yy >= h as i32 || xx < 0 || xx >= w as i32 {
                        continue;
                    }
                    if lambda[yy as usize * w + xx as usize] > v {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            if is_max {
                cands.push((v, x, y));
            }
        }
    }
    cands.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    let mut accepted: Vec<(usize, usize)> = Vec::new();
    let sep2 = min_sep * min_sep;
    for (_, x, y) in cands {
        if accepted.len() >= max_detections {
            break;
        }
        let clash = accepted.iter().any(|&(ax, ay)| {
            let dx = ax as f64 - x as f64;
            let dy = ay as f64 - y as f64;
            dx * dx + dy * dy < sep2
        });
        if !clash {
            accepted.push((x, y));
        }
    }
    let nl = lengths.len().max(1);
    let last = library.lines.len() - 1;
    let mut out: Vec<Detection> = Vec::new();
    let mut max_trials = 0usize;
    for (rank, (x, y)) in accepted.into_iter().enumerate() {
        let zs = eval_all_at(&stats.g, stats.gt.as_deref(), h, w, library, x, y);
        let info = grid_info(&zs, library, nl, n_angles);
        if !info.best.is_finite() {
            continue;
        }
        // For a uniform streak the Lambda ridge is nearly flat along the axis, so
        // a local parabola wanders while a thresholded centroid is stable (and
        // exact for a point source).
        let cut = 0.55 * info.best.max(lambda[y * w + x] as f64);
        let extent = ridge_extent(
            lambda,
            h,
            w,
            x,
            y,
            info.pa_deg,
            512,
            (0.3 * info.best).max(threshold as f64),
        );
        let radius = ((0.55 * extent).max(0.8 * info.length_px + 4.0) as usize).clamp(6, 320);
        let mut px = x as f64;
        let mut py = y as f64;
        for _ in 0..2 {
            let xa = ((px.round() as isize) - radius as isize).max(0) as usize;
            let xb = (((px.round() as isize) + radius as isize) as usize).min(w - 1);
            let ya = ((py.round() as isize) - radius as isize).max(0) as usize;
            let yb = (((py.round() as isize) + radius as isize) as usize).min(h - 1);
            let mut sw = 0.0f64;
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            for yy in ya..=yb {
                for xx in xa..=xb {
                    let v = lambda[yy * w + xx] as f64;
                    if v > cut {
                        let weight = v - cut;
                        sw += weight;
                        sx += weight * xx as f64;
                        sy += weight * yy as f64;
                    }
                }
            }
            if sw <= 0.0 {
                break;
            }
            let (nx, ny) = (sx / sw, sy / sw);
            if (nx - px).abs() > radius as f64 || (ny - py).abs() > radius as f64 {
                break;
            }
            px = nx;
            py = ny;
        }
        let point_z = zs[0];
        let coarse_z = info.best.max(point_z);
        let mut det = Detection {
            x: px,
            y: py,
            lambda: coarse_z,
            lr: info.best - point_z,
            length_px: info.length_px,
            pa_deg: info.pa_deg,
            sigma_x_px: library.lines[info.best_i.min(last)].sigma_x_coeff
                / info.best.max(1e-9),
            coarse_lambda: coarse_z,
            fine_lambda: None,
            refined: false,
        };
        if let Some(cfg) = fine {
            if rank < max_refinements {
                if let Some(r) = fine_search(
                    &stats.g,
                    h,
                    w,
                    px,
                    py,
                    info.pa_deg,
                    info.length_px,
                    ctx,
                    cfg,
                ) {
                    max_trials = max_trials.max(r.trials);
                    let z = r.z.max(point_z);
                    if z >= det.lambda {
                        det.x = r.x;
                        det.y = r.y;
                        det.length_px = r.length;
                        det.pa_deg = r.pa_deg;
                        det.sigma_x_px = r.sigma_x_coeff / z.max(1e-9);
                        det.lambda = z;
                        det.lr = r.z - point_z;
                        det.fine_lambda = Some(z);
                        det.refined = true;
                    }
                }
            }
        }
        out.push(det);
    }
    out.sort_by(|a, b| b.lambda.total_cmp(&a.lambda));
    // the centroid and the fine search can pull two maxima onto the same object
    let mut unique: Vec<Detection> = Vec::new();
    for d in out {
        let clash = unique
            .iter()
            .any(|u| (u.x - d.x) * (u.x - d.x) + (u.y - d.y) * (u.y - d.y) < sep2);
        if !clash {
            unique.push(d);
        }
    }
    (unique, max_trials)
}

/// Smooth truth kernel for an injected streak (not the DDA kernel used by the
/// detector): a 1-px segment oversampled 4x and blurred by the PSF.
pub fn streak_kernel(sigma_psf: f64, length_px: f64, pa_deg: f64) -> KernelGrid {
    let n = ((length_px * 4.0) as usize).max(8);
    let rad = pa_deg.to_radians();
    let (ux, uy) = (rad.cos(), rad.sin());
    let mut seg = vec![0.0f64; 0];
    let half = length_px / 2.0;
    // bilinear splat onto an integer lattice
    let span_x = (half * ux.abs()).ceil() as i64 + 2;
    let span_y = (half * uy.abs()).ceil() as i64 + 2;
    let cols = (2 * span_x + 1) as usize;
    let rows = (2 * span_y + 1) as usize;
    seg = vec![0.0f64; rows * cols];
    let (cy, cx) = (span_y as f64, span_x as f64);
    for i in 0..=n {
        let t = -half + length_px * i as f64 / n as f64;
        let px = cx + t * ux;
        let py = cy + t * uy;
        let i0 = px.floor();
        let j0 = py.floor();
        let fx = px - i0;
        let fy = py - j0;
        let w = 1.0 / (n + 1) as f64;
        for (ii, wi) in [(0i64, 1.0 - fx), (1, fx)] {
            for (jj, wj) in [(0i64, 1.0 - fy), (1, fy)] {
                let y = (j0 as i64 + jj) as usize;
                let x = (i0 as i64 + ii) as usize;
                if y < rows && x < cols {
                    seg[y * cols + x] += w * wi * wj;
                }
            }
        }
    }
    let g = gauss_1d(sigma_psf);
    let blurred = blur_separable_f64(&seg, rows, cols, &g, &g);
    let s: f64 = blurred.iter().sum();
    let data = if s > 0.0 {
        blurred.iter().map(|v| v / s).collect()
    } else {
        blurred
    };
    KernelGrid::new(rows, cols, data)
}

/// Add an injected streak whose peak equals peak_snr * sigma_noise.
pub fn inject_streak(
    image: &mut [f64],
    h: usize,
    w: usize,
    k: &KernelGrid,
    cx: f64,
    cy: f64,
    peak_snr: f64,
    sigma_noise: f64,
) {
    let peak = k.data.iter().cloned().fold(0.0f64, f64::max).max(1e-30);
    let scale = peak_snr * sigma_noise / peak;
    let (ky, kx) = k.center();
    for i in 0..k.rows {
        for j in 0..k.cols {
            let y = cy.round() as i64 + i as i64 - ky as i64;
            let x = cx.round() as i64 + j as i64 - kx as i64;
            if y < 0 || y >= h as i64 || x < 0 || x >= w as i64 {
                continue;
            }
            image[y as usize * w + x as usize] += scale * k.data[i * k.cols + j];
        }
    }
}

/// Unit-variance white Gaussian noise field.
pub fn white_noise(h: usize, w: usize, sigma: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed | 1);
    (0..h * w).map(|_| rng.normal() * sigma).collect()
}

/// Read the primary image plane of a FITS frame as f64 samples.
pub fn read_image(path: &Path) -> Result<(Vec<f64>, usize, usize), String> {
    let frame = fits::read_fits(path, false)?;
    Ok((frame.data.to_f64(), frame.height, frame.width))
}

/// Convenience wrapper for the CLI: read the frame and run the prototype.
pub fn run_fits(path: &Path, opts: &Options) -> Result<GlrtResult, String> {
    let (image, h, w) = read_image(path)?;
    run_array(opts, &image, h, w)
}

/// Directory helper used by the binaries.
pub fn ensure_dir(dir: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    Ok(dir.to_path_buf())
}
#[cfg(test)]
mod tests {
    use super::*;

    fn lib(lengths: &[f64], n_angles: usize, sigma_psf: f64, sigma_w: f64) -> Library {
        let w = Whitener::identity();
        let p = profile_tables(sigma_psf, &w);
        build_library(
            lengths,
            n_angles,
            sigma_psf,
            sigma_w,
            (&p.plain_y, &p.plain_x),
            (&p.grad_y, &p.grad_x),
        )
    }

    fn std_of(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
    }

    #[test]
    fn kernels_are_unit_sum_centred_and_length_aware() {
        let sigma = 0.82;
        let pk = point_kernel(sigma);
        let total: f64 = pk.data.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "point kernel sum = {total}");
        assert!(pk.rows % 2 == 1 && pk.cols % 2 == 1);
        let (cy, cx) = pk.center();
        let mut peak = (0usize, 0usize, 0.0f64);
        for i in 0..pk.rows {
            for j in 0..pk.cols {
                let v = pk.data[i * pk.cols + j];
                if v > peak.2 {
                    peak = (i, j, v);
                }
            }
        }
        assert_eq!((peak.0, peak.1), (cy, cx), "point kernel is not centred");

        let library = lib(&[16.0, 64.0, 256.0], 12, sigma, 1.0);
        assert_eq!(library.lines.len(), 36);
        let mut norms = Vec::new();
        for (i, hyp) in library.lines.iter().enumerate() {
            let k = kernel_from_samples(&hyp.weighted_offsets(0), sigma);
            let total: f64 = k.data.iter().sum();
            assert!((total - 1.0).abs() < 1e-9, "line kernel sum = {total}");
            assert!(hyp.norms[0] > 0.0);
            assert!(hyp.angle_deg >= 0.0 && hyp.angle_deg < 180.0);
            // every realised axis is within 2 deg of a 15 deg grid step
            let nearest = (hyp.angle_deg / 15.0).round() * 15.0;
            assert!(
                (hyp.angle_deg - nearest).abs() < 2.0,
                "angle {} is not on the 15 deg grid",
                hyp.angle_deg
            );
            // the sampled segment length is within one lattice step of nominal
            let nominal = library.lengths[i % library.lengths.len()];
            assert!(
                (hyp.length_px - nominal).abs() <= 2.5,
                "length {} for nominal {nominal}",
                hyp.length_px
            );
            norms.push(hyp.norms[0]);
        }
        // within one angle a longer segment spreads the unit mass over more pixels
        assert!(norms[0] > norms[2], "line kernel norm did not shrink with L");
    }

    #[test]
    fn point_kernel_z_has_unit_variance_under_white_noise() {
        let (h, w) = (96usize, 96usize);
        let sigma_noise = 1.0;
        let img = white_noise(h, w, sigma_noise, 7);
        let k = point_kernel(0.82);
        let z = mf_z_freq(&img, h, w, &k, sigma_noise);
        let m = k.rows + 1;
        let mut values = Vec::new();
        for y in m..h - m {
            for x in m..w - m {
                values.push(z[y * w + x]);
            }
        }
        let std = std_of(&values);
        assert!(
            (std - 1.0).abs() < 0.03,
            "std(z) of the frequency matched filter = {std}"
        );
    }

    #[test]
    fn freq_and_line_engines_agree() {
        let (h, w) = (40usize, 40usize);
        let sigma_psf = 0.82;
        let sigma_noise = 1.0;
        let img = white_noise(h, w, sigma_noise, 11);
        let library = lib(&[12.0], 4, sigma_psf, sigma_noise);
        let g = smoothed(&img, h, w, sigma_psf);
        let gt = transpose_f32(&g, h, w);
        let m = 11;
        let mut worst = 0.0f64;
        for (idx, hyp) in library.lines.iter().enumerate() {
            assert_eq!(hyp.q, 1, "this test needs a single shear phase");
            let samples = hyp.weighted_offsets(0);
            let a = sum_sq_from_samples(&samples, &library_profile_plain(0), &library_profile_plain(1));
            let err = &mut worst;
            let mut grid = kernel_from_samples(&samples, sigma_psf);
            // match the analytic kernel norm exactly (the grid is renormalised
            // to unit sum, which rescales sum k^2 by ~1e-4)
            let scale = (a / grid.sum_sq).sqrt();
            let data: Vec<f64> = grid.data.iter().map(|v| v * scale).collect();
            grid = KernelGrid::new(grid.rows, grid.cols, data);
            let k_image = if hyp.steep {
                transpose_kernel(&grid)
            } else {
                grid
            };
            let zf = mf_z_freq(&img, h, w, &k_image, sigma_noise);
            for y in m..h - m {
                for x in m..w - m {
                    let zs = eval_all_at(&g, Some(&gt), h, w, &library, x, y);
                    let zl = zs[1 + idx];
                    let rel = (zf[y * w + x] - zl).abs() / (1.0 + zl.abs());
                    if rel > *err {
                        *err = rel;
                    }
                }
            }
        }
        assert!(worst < 1e-7, "frequency and line engines differ by {worst}");
    }

    fn library_profile_plain(which: usize) -> Vec<f64> {
        let p = profile_tables(0.82, &Whitener::identity());
        if which == 0 {
            p.plain_y
        } else {
            p.plain_x
        }
    }

    #[test]
    fn noise_model_of_white_noise_is_white() {
        let (h, w) = (256usize, 256usize);
        let img = white_noise(h, w, 3.0, 5);
        let noise = estimate_noise(&img, h, w);
        assert!(
            (noise.sigma_adu - 3.0).abs() < 0.2,
            "sigma = {}",
            noise.sigma_adu
        );
        assert!(
            (noise.corr_len_px - 1.0).abs() < 0.1,
            "corr_len = {}",
            noise.corr_len_px
        );
        let lag = (noise.rho_x.len() - 1) / 2;
        assert!(noise.rho_x[lag + 1].abs() < 0.05);
        assert!(noise.rho_y[lag + 1].abs() < 0.05);
    }

    #[test]
    fn white_noise_pipeline_has_unit_z_scatter() {
        let (h, w) = (256usize, 256usize);
        let img = white_noise(h, w, 1.0, 17);
        let opts = Options {
            lengths: vec![16.0, 32.0],
            n_angles: 6,
            whiten: false,
            calib_size: 256,
            floor: 0.0,
            ..Default::default()
        };
        let res = run_array(&opts, &img, h, w).unwrap();
        assert!(
            (res.std_z_point - 1.0).abs() < 0.03,
            "std_z_point = {}",
            res.std_z_point
        );
        assert!(res.gumbel.1 > 0.05 && res.gumbel.1 < 1.0, "beta = {}", res.gumbel.1);
        assert!(res.thresholds[0] > res.thresholds[1]);
        assert!(res.thresholds[1] > res.thresholds[2]);
    }

    #[test]
    fn synthetic_streak_is_recovered() {
        let (h, w) = (256usize, 256usize);
        let sigma_psf = 1.93 / FWHM_TO_SIGMA;
        let mut img = white_noise(h, w, 1.0, 3);
        let target = streak_kernel(sigma_psf, 64.0, 35.0);
        inject_streak(&mut img, h, w, &target, 128.0, 128.0, 15.0, 1.0);
        let opts = Options {
            psf_fwhm: 1.93,
            lengths: vec![16.0, 32.0, 64.0, 128.0],
            n_angles: 12,
            whiten: false,
            calib_size: 256,
            floor: 0.0,
            ..Default::default()
        };
        let res = run_array(&opts, &img, h, w).unwrap();
        assert!(!res.detections.is_empty(), "no detection at all");
        let best = res
            .detections
            .iter()
            .min_by(|a, b| {
                let da = (a.x - 128.0).powi(2) + (a.y - 128.0).powi(2);
                let db = (b.x - 128.0).powi(2) + (b.y - 128.0).powi(2);
                da.total_cmp(&db)
            })
            .unwrap();
        let dx = best.x - 128.0;
        let dy = best.y - 128.0;
        assert!(
            dx * dx + dy * dy <= 4.0,
            "position error ({dx}, {dy}) too large"
        );
        assert!(best.lr > 6.0, "LR = {} is not above 6", best.lr);
        assert!(
            (best.length_px - 64.0).abs() <= 16.0,
            "length = {}",
            best.length_px
        );
        assert!(
            (best.pa_deg - 35.0).abs() <= 8.0,
            "pa = {}",
            best.pa_deg
        );
    }


    #[test]
    fn steep_streak_keeps_its_own_position() {
        // a steep streak (PA > 45 deg) is evaluated in the transposed frame; a
        // transposed index anywhere in that path would move the detection to
        // (y, x), which this off-diagonal injection makes impossible to miss
        let (h, w) = (256usize, 256usize);
        let sigma_psf = 1.93 / FWHM_TO_SIGMA;
        let mut img = white_noise(h, w, 1.0, 17);
        let (tx, ty) = (86.0f64, 168.0f64);
        let target = streak_kernel(sigma_psf, 64.0, 122.0);
        inject_streak(&mut img, h, w, &target, tx, ty, 15.0, 1.0);
        let opts = Options {
            psf_fwhm: 1.93,
            lengths: vec![16.0, 32.0, 64.0, 128.0],
            n_angles: 24,
            whiten: false,
            calib_size: 256,
            floor: 0.0,
            ..Default::default()
        };
        let res = run_array(&opts, &img, h, w).unwrap();
        assert!(!res.detections.is_empty(), "no detection at all");
        let best = res
            .detections
            .iter()
            .min_by(|a, b| {
                let da = (a.x - tx).powi(2) + (a.y - ty).powi(2);
                let db = (b.x - tx).powi(2) + (b.y - ty).powi(2);
                da.total_cmp(&db)
            })
            .unwrap();
        let d = ((best.x - tx).powi(2) + (best.y - ty).powi(2)).sqrt();
        assert!(d <= 4.0, "position error {d} px at ({}, {})", best.x, best.y);
        assert!(
            (best.length_px - 64.0).abs() <= 16.0,
            "length = {}",
            best.length_px
        );
        assert!(
            (best.pa_deg - 122.0).abs() <= 6.0,
            "pa = {}",
            best.pa_deg
        );
    }


    #[test]
    fn truncated_norm_matches_the_pair_sum() {
        // F1 rests on norm^2(N) = N*c - k reproducing the exact pair sum
        // sum_{i,j<N} A(x_i - x_j) over the *retained* samples, so compare the
        // table against a direct O(n^2) evaluation for every prefix length.
        // The lag decomposition averages the row offset over the phase cycle, so
        // the短 window limit carries a small residual; the bounds below are what
        // the implementation actually achieves (see the printed maxima).
        let sigma = 1.93 / FWHM_TO_SIGMA;
        let sigma_w = 1.7;
        let library = lib(&[16.0, 64.0, 256.0], 12, sigma, sigma_w);
        let w = Whitener::identity();
        let p = profile_tables(sigma, &w);
        let mut worst_short = 0.0f64;
        let mut worst_long = 0.0f64;
        let mut checked = 0usize;
        for hyp in &library.lines {
            let (py, px) = if hyp.steep {
                (&p.plain_x, &p.plain_y)
            } else {
                (&p.plain_y, &p.plain_x)
            };
            for phase in 0..hyp.q {
                let samples = hyp.weighted_offsets(phase);
                for n_used in 1..=hyp.n {
                    let kept: Vec<(i64, i64, f64)> =
                        samples.iter().take(n_used).cloned().collect();
                    // the reference carries the 1/n weights, the table does not
                    let want = sum_sq_from_samples(&kept, py, px) * (hyp.n * hyp.n) as f64;
                    let table = hyp.inv_norm[phase * (hyp.n + 1) + n_used] as f64;
                    assert!(table > 0.0, "no table entry for N = {n_used}");
                    let got = 1.0 / (sigma_w * table);
                    let rel = ((got * got) - want).abs() / want.max(1.0);
                    if n_used >= 16 {
                        worst_long = worst_long.max(rel);
                    } else {
                        worst_short = worst_short.max(rel);
                    }
                    checked += 1;
                }
            }
        }
        println!("truncated norm: {checked} entries, worst rel err N<16 {worst_short:.4}, N>=16 {worst_long:.4}");
        assert!(checked > 100, "only {checked} entries checked");
        assert!(worst_long <= 0.01, "long-window norm error {worst_long}");
        assert!(worst_short <= 0.001, "short-window norm error {worst_short}");
    }

    #[test]
    fn tiny_frame_with_long_kernel_is_safe() {
        let (h, w) = (16usize, 16usize);
        let img = white_noise(h, w, 2.0, 5);
        let opts = Options {
            lengths: vec![64.0],
            n_angles: 4,
            whiten: false,
            calib_size: 128,
            floor: 0.0,
            ..Default::default()
        };
        let res = run_array(&opts, &img, h, w).unwrap();
        assert_eq!(res.shape, (16, 16));
        assert_eq!(res.lambda.len(), 256);
        assert_eq!(res.n_hypotheses, 1 + 4);
        let json = res.to_json("mem", None);
        assert_eq!(json["shape"][0], 16.0);
        assert!(json["gumbel"]["beta"].as_f64().unwrap() > 0.0);
    }
}
