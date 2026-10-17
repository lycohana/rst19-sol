#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xcheck_phase1b_lib.py -- 阶段一第二批（E3/E4/E10/E7-lite）共用底座。

独立交叉验证工具，非生产实现。只读原始 FITS；不写回原始数据。
所有阈值来自 tools/xcheck/xcheck_phase1b_config.json，脚本内不写死。
"""
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LIB_PATH = Path(__file__).resolve()
SCRIPT_DIR = LIB_PATH.parent
ROOT = LIB_PATH.parents[2]
CONFIG_PATH = SCRIPT_DIR / "xcheck_phase1b_config.json"
RUNID_MARKER = ROOT / "tmp" / "xcheck-frame1" / "phase1b_runid.txt"

LINEAR_MAG = 1.0857362047581294  # 2.5/ln(10)

# ---------------------------------------------------------------- provenance


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest().upper()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest().upper()


def sha256_json(o):
    return sha256_bytes(
        json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("not serializable: %r" % type(o))


def dump_json(path, obj):
    path = Path(path)
    if path.exists():
        raise SystemExit("refuse to overwrite existing file: %s" % path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_np_default), encoding="utf-8")
    print("[json] wrote", path)


def write_json_allow(path, obj):
    """仅用于 run 目录内的中间交换文件（同一次运行内自用）。"""
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_np_default), encoding="utf-8")


def load_config():
    cfg_bytes = CONFIG_PATH.read_bytes()
    return json.loads(cfg_bytes.decode("utf-8")), sha256_bytes(cfg_bytes)


def resolve_runid():
    env = os.environ.get("XCHECK_RUNID")
    if env:
        return env
    if RUNID_MARKER.exists():
        return RUNID_MARKER.read_text(encoding="utf-8").strip()
    return None


def make_runid(cfg_sha):
    rid = resolve_runid()
    if rid is None:
        rid = utc_now().replace("-", "").replace(":", "").replace(".", "")[:18] + "_" + cfg_sha[:8]
        RUNID_MARKER.parent.mkdir(parents=True, exist_ok=True)
        RUNID_MARKER.write_text(rid + "\n", encoding="utf-8")
    return rid


def provenance(script_path, cfg_sha, runid, inputs=None):
    return {
        "tool": "%s (独立交叉验证工具，非生产实现)" % Path(script_path).name,
        "interpreter": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "lib_path": str(LIB_PATH),
        "lib_sha256": sha256_file(LIB_PATH),
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "config_path": str(CONFIG_PATH),
        "config_sha256": cfg_sha,
        "utc_start": utc_now(),
        "runid": runid,
        "inputs": inputs or {},
    }


# ---------------------------------------------------------------- FITS (minimal)


def read_fits_header(path):
    raw = Path(path).read_bytes()[:2880 * 32]
    off = 0
    hdr = {}
    while True:
        block = raw[off : off + 2880]
        if len(block) < 2880:
            raise ValueError("truncated FITS header")
        end = False
        for i in range(36):
            c = block[i * 80 : (i + 1) * 80].decode("latin1")
            if c.startswith("END"):
                end = True
                break
            k = c[:8].strip()
            if k and k not in ("COMMENT", "HISTORY", "END"):
                hdr.setdefault(k, []).append(c[10:].strip())
        off += 2880
        if end:
            break
    nx = int(hdr["NAXIS1"][0].split("/")[0])
    ny = int(hdr["NAXIS2"][0].split("/")[0])
    bitpix = int(hdr["BITPIX"][0].split("/")[0])
    if bitpix != 16 or int(hdr["NAXIS"][0].split("/")[0]) != 2:
        raise ValueError("expected 2-D BITPIX=16")
    bzero = float(hdr.get("BZERO", ["0"])[0].split("/")[0])
    return hdr, ny, nx, off, bzero


def read_fits_data(path, ny, nx, off, bzero=0.0):
    d = np.fromfile(str(path), dtype=">i2", count=nx * ny, offset=off).astype(np.int32)
    if bzero == 32768.0:
        d = d + 32768
    return d.reshape(ny, nx)


def load_frame(path):
    hdr, ny, nx, off, bzero = read_fits_header(path)
    return read_fits_data(path, ny, nx, off, bzero), hdr


def mask_telemetry(dn, cfg):
    """返回探测用副本：遥测区取全帧中位数（永久掩膜语义）。"""
    out = dn.copy()
    r = cfg["mask"]["telemetry_rows"]
    c = cfg["mask"]["telemetry_cols"]
    med = float(np.median(dn[r:, :]))
    out[:r, :c] = med
    return out, med


# ---------------------------------------------------------------- statistics


def sigma_clipped_stats(values, clip=3.0, iters=3):
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    for _ in range(iters):
        if v.size < 4:
            break
        med = float(np.median(v))
        sd = float(np.std(v, ddof=1))
        if not (sd > 0):
            break
        keep = np.abs(v - med) <= clip * sd
        if keep.all():
            break
        v = v[keep]
    med = float(np.median(v))
    sd = float(np.std(v, ddof=1)) if v.size > 1 else float("nan")
    mad = float(1.4826 * np.median(np.abs(v - med))) if v.size else float("nan")
    return med, sd, mad, int(v.size)


def mad_sigma(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    m = float(np.median(v))
    return float(1.4826 * np.median(np.abs(v - m)))


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


# ---------------------------------------------------------------- image ops


def box_stats(dn, box):
    ny, nx = dn.shape
    by, bx = ny // box, nx // box
    sub = dn[: by * box, : bx * box].astype(np.float64).reshape(by, box, bx, box)
    med = np.median(sub, axis=(1, 3))
    sig = np.empty_like(med)
    for i in range(by):
        for j in range(bx):
            _, sd, mad, _ = sigma_clipped_stats(sub[i, :, j, :], 3.0, 3)
            sig[i, j] = mad if mad > 0 else sd
    return med, sig


def upsample_blocky(m, ny, nx):
    by, bx = m.shape
    fy, fx = ny / by, nx / bx
    yy = np.minimum((np.arange(ny) / fy).astype(int), by - 1)
    xx = np.minimum((np.arange(nx) / fx).astype(int), bx - 1)
    return m[np.ix_(yy, xx)]


def dilate_bool(mask, radius):
    out = mask.copy()
    m0 = mask.copy()
    for s in range(1, radius + 1):
        out[:, s:] |= m0[:, :-s]
        out[:, :-s] |= m0[:, s:]
    m1 = out.copy()
    for s in range(1, radius + 1):
        out[s:, :] |= m1[:-s, :]
        out[:-s, :] |= m1[s:, :]
    return out


def gaussian_kernel(fwhm, truncate):
    sigma = fwhm / 2.3548200450309493
    r = max(1, int(math.ceil(truncate * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def convolve1d(a, k, axis):
    r = len(k) // 2
    out = np.zeros_like(a)
    if axis == 0:
        ap = np.pad(a, ((r, r), (0, 0)), mode="edge")
        for i, w in enumerate(k):
            out += w * ap[i : i + a.shape[0], :]
    else:
        ap = np.pad(a, ((0, 0), (r, r)), mode="edge")
        for i, w in enumerate(k):
            out += w * ap[:, i : i + a.shape[1]]
    return out


def local_maxima(s, thresh):
    p = np.pad(s, 1, constant_values=-np.inf)
    m = np.ones(s.shape, dtype=bool)
    h, w = s.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            m &= s > p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return m & (s >= thresh)


def nms(yx, snr, min_dist):
    cell = float(max(1, int(math.floor(min_dist))))
    order = np.argsort(-snr, kind="stable")
    grid, keep = {}, []
    d2 = min_dist * min_dist
    for i in order:
        y, x = int(yx[i, 0]), int(yx[i, 1])
        cx, cy = int(x // cell), int(y // cell)
        ok = True
        for gy in (cy - 1, cy, cy + 1):
            for gx in (cx - 1, cx, cx + 1):
                for (yy, xx) in grid.get((gx, gy), ()):
                    if (yy - y) ** 2 + (xx - x) ** 2 < d2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((cx, cy), []).append((y, x))
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def moments(img, yx, half, absolute=False):
    o = np.arange(-half, half + 1)
    dy, dx = np.meshgrid(o, o, indexing="ij")
    dy, dx = dy.ravel().astype(np.float64), dx.ravel().astype(np.float64)
    ny, nx = img.shape
    n = len(yx)
    fwhm = np.full(n, np.nan)
    ell = np.full(n, np.nan)
    for s0 in range(0, n, 40000):
        sl = slice(s0, min(n, s0 + 40000))
        y, x = yx[sl, 0].astype(int), yx[sl, 1].astype(int)
        edge = (y < half) | (y >= ny - half) | (x < half) | (x >= nx - half)
        y = np.clip(y, half, ny - half - 1)
        x = np.clip(x, half, nx - half - 1)
        win = img[(y[:, None] + dy[None, :]).astype(int), (x[:, None] + dx[None, :]).astype(int)].astype(np.float64)
        if absolute:
            win = np.abs(win)
        win = np.where(win > 0, win, 0.0)
        tot = win.sum(axis=1)
        ok = tot > 0
        safe = np.where(ok, tot, 1.0)
        mx = (win * dx[None, :]).sum(axis=1) / safe
        my = (win * dy[None, :]).sum(axis=1) / safe
        vx = (win * (dx[None, :] - mx[:, None]) ** 2).sum(axis=1) / safe
        vy = (win * (dy[None, :] - my[:, None]) ** 2).sum(axis=1) / safe
        g = np.sqrt(np.sqrt(np.maximum(vx * vy, 0.0)))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.sqrt(np.minimum(vx, vy) / np.maximum(vx, vy))
        good = ok & (~edge)
        fwhm[sl] = np.where(good, 2.3548200450309493 * g, np.nan)
        ell[sl] = np.where(good, 1.0 - r, np.nan)
    return fwhm, ell


# ---------------------------------------------------------------- detection


class Frame:
    """一帧的背景图 + 匹配滤波探测结果。"""

    def __init__(self, dn, cfg):
        self.dn = dn
        self.cfg = cfg
        ny, nx = dn.shape
        self.ny, self.nx = ny, nx
        clean, self.telemetry_fill = mask_telemetry(dn, cfg)
        box = cfg["detect"]["background_box"]
        med_map, sig_map = box_stats(clean, box)
        self.bg_med_map = med_map
        self.bg_sig_map = sig_map
        self.bg_med = upsample_blocky(med_map, ny, nx).astype(np.float32)
        self.bg_sig = upsample_blocky(sig_map, ny, nx).astype(np.float32)
        self.sigma_global = float(np.median(sig_map))
        self.med_global = float(np.median(med_map))
        self.img = clean.astype(np.float32) - self.bg_med
        self.clean_mask = np.ones((ny, nx), dtype=bool)
        r = cfg["mask"]["telemetry_rows"]
        c = cfg["mask"]["telemetry_cols"]
        self.clean_mask[:r, :c] = False

    def detect(self, absolute=False, snr_thresh=None, keep_morph=False):
        c = self.cfg["detect"]
        k = gaussian_kernel(c["fwhm_px"], c["kernel_truncate_sigma"])
        img = self.img if not absolute else (-self.img).astype(np.float32)
        s1 = convolve1d(img, k, 0)
        s2 = convolve1d(s1, k, 1)
        sumk2 = float((np.outer(k, k) ** 2).sum())
        snr = s2 / (self.bg_sig * math.sqrt(sumk2))
        thresh = c["mf_threshold_snr"] if snr_thresh is None else snr_thresh
        peaks = local_maxima(snr, thresh) & self.clean_mask
        yx = np.argwhere(peaks)
        sv = snr[peaks]
        keep = nms(yx, sv, c["nms_min_distance_px"])
        yx, sv = yx[keep], sv[keep]
        fwhm = ell = None
        if keep_morph:
            fwhm, ell = moments(self.img, yx, c["morphology_window"] // 2, absolute=absolute)
            lo, hi = c["fwhm_accept_px"]
            ok = np.isfinite(fwhm) & (fwhm >= lo) & (fwhm <= hi) & np.isfinite(ell) & (ell <= c["ellip_max"])
            yx, sv, fwhm, ell = yx[ok], sv[ok], fwhm[ok], ell[ok]
        return {"yx": yx, "snr": sv, "fwhm": fwhm, "ell": ell, "snr_map": snr}

    def peak_dn(self, yx):
        return self.dn[yx[:, 0].astype(int), yx[:, 1].astype(int)].astype(np.float64)


# ---------------------------------------------------------------- photometry


def _win_index(ys, xs, half, ny, nx):
    o = np.arange(-half, half + 1)
    yy, xx = np.meshgrid(o, o, indexing="ij")
    iy = np.floor(ys)[:, None].astype(np.int64) + yy.ravel()[None, :]
    ix = np.floor(xs)[:, None].astype(np.int64) + xx.ravel()[None, :]
    valid = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    iy = np.clip(iy, 0, ny - 1)
    ix = np.clip(ix, 0, nx - 1)
    off_y = iy - ys[:, None]
    off_x = ix - xs[:, None]
    return iy, ix, valid, np.hypot(off_x, off_y)


def aperture_measure(dn, ys, xs, r_ap, r_in, r_out, frac=False, chunk=4000, npix=None):
    """几何孔径（d<=r_ap）或分数像元权重孔径（w=clip(r_ap+0.5-d,0,1)），环带 r_in..r_out 取中位数为天光。"""
    ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
    xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
    n = ys.size
    out = {
        "ys": ys, "xs": xs,
        "flux": np.full(n, np.nan), "sky": np.full(n, np.nan),
        "n_ap": np.zeros(n, dtype=np.int64), "n_ap_float": np.full(n, np.nan),
        "sky_sigma": np.full(n, np.nan),
        "ap_snr": np.full(n, np.nan),
    }
    half = int(math.ceil(r_out)) + 1
    ny, nx = dn.shape
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(n, s0 + chunk))
        iy, ix, valid, d = _win_index(ys[sl], xs[sl], half, ny, nx)
        ann = valid & (d >= r_in) & (d <= r_out)
        vals = dn[iy, ix].astype(np.float64)
        ann_vals = np.where(ann, vals, np.nan)
        with np.errstate(all="ignore"):
            sky = np.nanmedian(ann_vals, axis=1)
        sky = np.where(np.isfinite(sky), sky, 0.0)
        if frac:
            wgt = np.clip(r_ap + 0.5 - d, 0.0, 1.0) * valid
        else:
            wgt = (d <= r_ap) & valid
        tot = ((vals - sky[:, None]) * wgt).sum(axis=1)
        n_ap = wgt.sum(axis=1)
        out["flux"][sl] = tot
        out["sky"][sl] = sky
        out["n_ap"][sl] = np.round(n_ap).astype(np.int64)
        out["n_ap_float"][sl] = n_ap
        resid = np.where(ann, vals - sky[:, None], np.nan)
        out["sky_sigma"][sl] = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=1)[:, None]), axis=1)
        var = np.maximum(n_ap, 1) * out["sky_sigma"][sl] ** 2 * 1.0
        with np.errstate(all="ignore"):
            out["ap_snr"][sl] = tot / np.sqrt(var)
    return out


def centroid(dn, ys, xs, r=3.0, iters=4, clip_negative=True, sky_in=6.0, sky_out=8.0):
    """迭代质心。天光从 r_in..r_out 环带中位数取（不用窗口内中位数：后者会被源自身的翼污染，
    产生随子像素相位变化的约 3% 系统偏差）。窗口 d<=r 内以 (I-sky) 正部为权。"""
    ys = np.atleast_1d(np.asarray(ys, dtype=np.float64)).copy()
    xs = np.atleast_1d(np.asarray(xs, dtype=np.float64)).copy()
    ny, nx = dn.shape
    half = int(math.ceil(max(r, sky_out))) + 1
    for _ in range(iters):
        iy, ix, valid, d = _win_index(ys, xs, half, ny, nx)
        vals = dn[iy, ix].astype(np.float64)
        ann = valid & (d >= sky_in) & (d <= sky_out)
        with np.errstate(all="ignore"):
            sky = np.nanmedian(np.where(ann, vals, np.nan), axis=1)
        sky = np.where(np.isfinite(sky), sky, 0.0)
        w = (d <= r) & valid
        v = np.where(w, vals - sky[:, None], 0.0)
        if clip_negative:
            v = np.where(v > 0, v, 0.0)
        tot = v.sum(axis=1)
        ok = tot > 0
        safe = np.where(ok, tot, np.nan)
        off_y = iy - np.floor(ys)[:, None]
        off_x = ix - np.floor(xs)[:, None]
        dy = (v * off_y).sum(axis=1) / safe
        dx = (v * off_x).sum(axis=1) / safe
        ys = np.where(ok & np.isfinite(dy), np.floor(ys) + dy, ys)
        xs = np.where(ok & np.isfinite(dx), np.floor(xs) + dx, xs)
    return ys, xs


def nearest_neighbor_px(yx, radius):
    """到最近其他源的距离（px），用 cell 哈希，O(n)。"""
    n = len(yx)
    if n == 0:
        return np.zeros(0)
    cell = max(1.0, float(radius) / 2.0)
    grid = {}
    for i in range(n):
        y, x = float(yx[i, 0]), float(yx[i, 1])
        grid.setdefault((int(x // cell), int(y // cell)), []).append(i)
    out = np.full(n, np.inf)
    for i in range(n):
        y, x = float(yx[i, 0]), float(yx[i, 1])
        cx, cy = int(x // cell), int(y // cell)
        best = np.inf
        for gy in range(cy - 1, cy + 2):
            for gx in range(cx - 1, cx + 2):
                for j in grid.get((gx, gy), ()):
                    if j == i:
                        continue
                    dd = (yx[j, 0] - y) ** 2 + (yx[j, 1] - x) ** 2
                    if dd < best:
                        best = dd
        out[i] = math.sqrt(best)
    return out


# ---------------------------------------------------------------- growth curve


def growth_curve(dn, ys, xs, r_max, sky_in, sky_out, chunk=800):
    """A(r), r=1..r_max；天光由 sky_in..sky_out 环带中位数给出。"""
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    n = ys.size
    A = np.full((n, r_max), np.nan)
    skys = np.full(n, np.nan)
    ny, nx = dn.shape
    half = int(math.ceil(sky_out)) + 1
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(n, s0 + chunk))
        iy, ix, valid, d = _win_index(ys[sl], xs[sl], half, ny, nx)
        vals = dn[iy, ix].astype(np.float64)
        ann = valid & (d >= sky_in) & (d <= sky_out)
        with np.errstate(all="ignore"):
            sky = np.nanmedian(np.where(ann, vals, np.nan), axis=1)
        sky = np.where(np.isfinite(sky), sky, 0.0)
        skys[sl] = sky
        resid = vals - sky[:, None]
        for r in range(1, r_max + 1):
            m = valid & (d <= r)
            A[sl, r - 1] = (resid * m).sum(axis=1)
    return A, skys


def growth_total(A, plateau_r, extrap_r):
    """两种总通量：r>=12 平台均值 与 A(r)=Finf-b/r 的 1/r 外推。"""
    i0 = plateau_r[0] - 1
    i1 = plateau_r[1]
    plat = np.nanmean(A[:, i0:i1], axis=1)
    rs = np.arange(extrap_r[0], extrap_r[1] + 1, dtype=np.float64)
    x = 1.0 / rs
    y = A[:, extrap_r[0] - 1 : extrap_r[1]]
    ok = np.isfinite(y).all(axis=1)
    finf = np.full(y.shape[0], np.nan)
    bcoef = np.full(y.shape[0], np.nan)
    if ok.any():
        yy = y[ok]
        xm, ym = x.mean(), yy.mean(axis=1, keepdims=True)
        sxx = ((x - xm) ** 2).sum()
        b = ((yy - ym) * (x - xm)[None, :]).sum(axis=1) / sxx
        a = ym[:, 0] - b * xm
        finf[ok] = a
        bcoef[ok] = b
    return plat, finf, bcoef


# ---------------------------------------------------------------- PSF


def psf_align(wins, frac_y, frac_x, pad=10):
    """用 Fourier 相位斜坡把每颗星的亚像素中心精确对齐到窗口中心像元，再求和成 1px PSF。

    wins: (n, S, S)，S 为奇数，中心像元 = floor(天体位置)。
    frac_y/frac_x: 天体相对该像元的偏移（0..1）。
    pad: 零填充边距（px），压制循环卷积回绕。
    返回 (aligned_1px, n_used)。aligned_1px 的 sum ≈ 每颗星归一化后的总通量均值。
    """
    wins = np.asarray(wins, dtype=np.float64)
    if wins.ndim == 2:
        S = int(round(math.sqrt(wins.shape[1])))
        wins = wins.reshape(wins.shape[0], S, S)
    n, S, S2 = wins.shape
    assert S == S2 and S % 2 == 1, "need odd square windows"
    P = S + 2 * pad
    buf = np.zeros((n, P, P), dtype=np.float64)
    c0 = (P - S) // 2
    buf[:, c0:c0 + S, c0:c0 + S] = wins
    k = np.fft.fftfreq(P)
    # 要把位于 +frac 的源移到 0：乘以 exp(+2j*pi*k*frac)
    ramp = np.exp(2j * np.pi * (k[None, :, None] * frac_y[:, None, None] + k[None, None, :] * frac_x[:, None, None]))
    F = np.fft.fft2(buf, axes=(1, 2)) * ramp
    aligned = np.real(np.fft.ifft2(F, axes=(1, 2)))
    return aligned.mean(axis=0)[c0:c0 + S, c0:c0 + S], int(n)


def fourier_upsample(P1, os_):
    """带限零填充上采样：格距 1/os_ px，中心对齐在 (N//2, N//2)，通量按 os^2 正确缩放。"""
    S = P1.shape[0]
    assert S % 2 == 1
    N = S * int(os_)
    f = np.fft.fft2(P1)
    h = S // 2
    new = np.zeros((N, N), dtype=complex)
    new[: h + 1, : h + 1] = f[: h + 1, : h + 1]
    new[: h + 1, N - h:] = f[: h + 1, S - h:]
    new[N - h:, : h + 1] = f[S - h:, : h + 1]
    new[N - h:, N - h:] = f[S - h:, S - h:]
    out = np.real(np.fft.ifft2(new))
    return out


def recenter_psf(Pos, os_, r_max=6.0):
    """把过采样 PSF 用双线性重采样精确居中到格点中心（不用 Fourier 平移，避免边缘振铃）。

    返回 (out, cy, cx)，cy/cx 为测到的残差中心（px）。
    """
    G = Pos.shape[0]
    c = (G - 1) / 2.0
    g = np.arange(G)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    gy = (yy - c) / os_
    gx = (xx - c) / os_
    m = np.hypot(gy, gx) <= r_max
    w = np.where(m, Pos, 0.0)
    tot = float(w.sum())
    cy = float((w * gy).sum() / tot)
    cx = float((w * gx).sum() / tot)
    if abs(cy) < 1e-3 and abs(cx) < 1e-3:
        return Pos, cy, cx
    sy = g + cy * os_
    sx = g + cx * os_
    y0 = np.floor(sy).astype(np.int64)
    x0 = np.floor(sx).astype(np.int64)
    fy = sy - y0
    fx = sx - x0
    y0 = np.clip(y0, 0, G - 2)
    x0 = np.clip(x0, 0, G - 2)
    out = (Pos[np.ix_(y0, x0)] * (1 - fy)[:, None] * (1 - fx)[None, :]
           + Pos[np.ix_(y0, x0 + 1)] * (1 - fy)[:, None] * fx[None, :]
           + Pos[np.ix_(y0 + 1, x0)] * fy[:, None] * (1 - fx)[None, :]
           + Pos[np.ix_(y0 + 1, x0 + 1)] * fy[:, None] * fx[None, :])
    return out, cy, cx


def psf_upsample_center(N):
    return (N - 1) // 2


def sample_psf(Pos, os_, dy, dx):
    """在过采样 PSF 上按双线性取样。dy/dx 为相对中心的像素偏移（可为分数）。"""
    G = Pos.shape[0]
    c = (G - 1) / 2.0
    gy = np.atleast_1d(dy) * os_ + c
    gx = np.atleast_1d(dx) * os_ + c
    y0 = np.floor(gy).astype(np.int64)
    x0 = np.floor(gx).astype(np.int64)
    fy = gy - y0
    fx = gx - x0
    ok = (y0 >= 0) & (y0 < G - 1) & (x0 >= 0) & (x0 < G - 1)
    y0 = np.clip(y0, 0, G - 2)
    x0 = np.clip(x0, 0, G - 2)
    v = (Pos[y0, x0] * (1 - fy) * (1 - fx) + Pos[y0, x0 + 1] * (1 - fy) * fx
         + Pos[y0 + 1, x0] * fy * (1 - fx) + Pos[y0 + 1, x0 + 1] * fy * fx)
    return np.where(ok, v, np.nan)

def psf_kernel_at_phase(P1, fy, fx, half=8, pad=10):
    """给出亚像素相位 (fy,fx) 下每个整数像素该读到的值（带限精确，不用插值）。

    P1: 1px 间距的叠合剖面（奇数尺寸，天体在中心像元）。
    返回 (2*half+1)^2 数组：像素偏移 di 对应 P1_cont(di - fy)。
    """
    S = P1.shape[0]
    assert S % 2 == 1
    P = S + 2 * pad
    buf = np.zeros((P, P), dtype=np.float64)
    c0 = (P - S) // 2
    buf[c0:c0 + S, c0:c0 + S] = P1
    k = np.fft.fftfreq(P)
    # psf_align 的约定：ramp 用 +frac 时星从 +frac 移到 0；要把它放到 +f 需用 -f
    ramp = np.exp(-2j * np.pi * (k[:, None] * fy + k[None, :] * fx))
    out = np.real(np.fft.ifft2(np.fft.fft2(buf) * ramp))
    cc = (P - 1) // 2
    return out[cc - half:cc + half + 1, cc - half:cc + half + 1]


def psf_radial(P, os_, r_max, n_bins=None, step=0.2):
    """从过采样 PSF 取方位平均径向 profile（每像素通量/px^2 归一到总和 1）。"""
    G = P.shape[0]
    c = (G - 1) / 2.0
    g = np.arange(G)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    rr = np.hypot(yy - c, xx - c) / os_
    edges = np.arange(0.0, r_max + step / 2, step)
    rows = []
    tot = float(P.sum())
    for i in range(len(edges) - 1):
        m = (rr >= edges[i]) & (rr < edges[i + 1])
        if not m.any():
            continue
        a = float(P[m].mean()) * os_ * os_  # 每 px^2 的通量
        rmid = 0.5 * (edges[i] + edges[i + 1])
        rows.append({"r_px": rmid, "p_r_per_px2": a, "n_cells": int(m.sum()),
                     "frac_of_total": float(P[m].sum() / tot)})
    return rows, tot


def psf_moments(P, os_, r_max=None):
    """二阶矩 FWHM/椭率。r_max 限制积分半径：二阶矩对边缘噪声/台基极其敏感，必须截断。"""
    G = P.shape[0]
    c = (G - 1) / 2.0
    g = np.arange(G)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    gy = (yy - c) / os_
    gx = (xx - c) / os_
    if r_max is not None:
        m = np.hypot(gy, gx) <= r_max
        P = np.where(m, P, 0.0)
        P = P / max(P.sum(), 1e-30)
    tot = P.sum()
    my = (P * gy).sum() / tot
    mx = (P * gx).sum() / tot
    vy = (P * (gy - my) ** 2).sum() / tot
    vx = (P * (gx - mx) ** 2).sum() / tot
    vxy = (P * (gx - mx) * (gy - my)).sum() / tot
    tr = vx + vy
    det = vx * vy - vxy ** 2
    disc = max(tr * tr / 4 - det, 0.0)
    l1 = tr / 2 + math.sqrt(disc)
    l2 = tr / 2 - math.sqrt(disc)
    fwhm = 2.3548200450309493 * math.sqrt(math.sqrt(max(l1 * l2, 0.0)))
    return {
        "fwhm_px": fwhm,
        "fwhm_x_px": 2.3548200450309493 * math.sqrt(vx),
        "fwhm_y_px": 2.3548200450309493 * math.sqrt(vy),
        "ellipticity": 1.0 - math.sqrt(max(l2, 0.0) / max(l1, 1e-30)),
        "centroid_offset_y_px": my, "centroid_offset_x_px": mx,
        "vxy": vxy,
    }


def encircled(P, os_, radii):
    G = P.shape[0]
    c = (G - 1) / 2.0
    g = np.arange(G)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    rr = np.hypot(yy - c, xx - c) / os_
    tot = float(P.sum())
    return [{"r_px": r, "ee": float(P[rr <= r].sum() / tot)} for r in radii]


# ---------------------------------------------------------------- misc


def logistic_50(x, y):
    """最小二乘 logistic 拟合 y=1..0 与 x，返回 50% 点（单调性由数据保证）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.min() > 0.5 or y.max() < 0.5:
        return None
    # 单调插值（x 已按序），线性内插 50% 交叉点
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    for i in range(1, xs.size):
        if ys[i - 1] >= 0.5 > ys[i]:
            t = (ys[i - 1] - 0.5) / (ys[i - 1] - ys[i])
            return float(xs[i - 1] + t * (xs[i] - xs[i - 1]))
    return None
