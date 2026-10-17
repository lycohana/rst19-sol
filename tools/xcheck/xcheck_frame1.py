#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xcheck_frame1.py -- 独立交叉验证工具，非生产实现。

目的：在不依赖生产 Rust 代码、不依赖 astropy 的前提下，对第 1 帧做自洽的
仪器层面分析（E0 像元编码 / E1 噪声结构 / E6 假阳性基线 / E2 误差模型算术）。

硬性边界：
  * 只读输入 FITS，不写回、不改名。
  * 不下载星表、不做板解、不做光度标定、不声明精度。
  * 所有可调参数来自 tools/xcheck/xcheck_config.json；脚本内不写死阈值。
  * 缺 astropy 时使用自带的最小 FITS 读取（2880 字节卡片块 + 大端 int16）。

用法：
  python tools/xcheck/xcheck_frame1.py --section all
  python tools/xcheck/xcheck_frame1.py --section e0 --runid <existing-runid>
"""
import argparse
import hashlib
import json
import math
import platform
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
CONFIG_PATH = ROOT / "tools" / "xcheck" / "xcheck_config.json"

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
    raise TypeError("not JSON serializable: %r" % type(o))


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_np_default), encoding="utf-8")


# ---------------------------------------------------------------- minimal FITS


def read_fits(path):
    raw = Path(path).read_bytes()
    cards, off = [], 0
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
            cards.append(c.rstrip())
        off += 2880
        if end:
            break
    hdr = {}
    for c in cards:
        k = c[:8].strip()
        if k and k not in ("COMMENT", "HISTORY"):
            hdr.setdefault(k, []).append(c[10:].strip())
    naxis = int(hdr["NAXIS"][0].split("/")[0])
    if naxis != 2:
        raise ValueError("only 2-D images supported")
    nx = int(hdr["NAXIS1"][0].split("/")[0])
    ny = int(hdr["NAXIS2"][0].split("/")[0])
    bitpix = int(hdr["BITPIX"][0].split("/")[0])
    bzero = float(hdr.get("BZERO", ["0"])[0].split("/")[0])
    if bitpix != 16:
        raise ValueError("expected BITPIX=16, got %d" % bitpix)
    data = np.frombuffer(raw, dtype=">i2", count=nx * ny, offset=off).astype(np.int32)
    if bzero == 32768.0:
        data = data + 32768
    return hdr, cards, data.reshape(ny, nx), off, len(raw)


def decode_aux(raw_bytes, data_offset, n=26):
    """26 个 double：相邻字节对交换后按小端解包（格式说明约定）。"""
    b = bytearray(raw_bytes[data_offset : data_offset + n * 8])
    if len(b) != n * 8:
        raise ValueError("aux region truncated")
    for i in range(0, len(b) - 1, 2):
        b[i], b[i + 1] = b[i + 1], b[i]
    return list(struct.unpack("<%dd" % n, bytes(b)))


# ---------------------------------------------------------------- helpers


def sigma_clipped_stats(values, clip=3.0, iters=3):
    v = np.asarray(values, dtype=np.float64).ravel()
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
    mad = float(1.4826 * np.median(np.abs(v - med)))
    return med, sd, mad, int(v.size)


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
    """可分布尔膨胀（方形结构元，半径 radius）。"""
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
    n = len(yx)
    fwhm = np.full(n, np.nan)
    ell = np.full(n, np.nan)
    step = 40000
    ny, nx = img.shape
    for s0 in range(0, n, step):
        sl = slice(s0, min(n, s0 + step))
        y, x = yx[sl, 0], yx[sl, 1]
        edge = (y < half) | (y >= ny - half) | (x < half) | (x >= nx - half)
        y = np.clip(y, half, ny - half - 1)
        x = np.clip(x, half, nx - half - 1)
        win = img[y[:, None] + dy[None, :].astype(int), x[:, None] + dx[None, :].astype(int)].astype(np.float64)
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


def inverse_normal_upper(p):
    """z 使 P(Z>z)=p，双侧数值二分（确定性）。"""
    lo, hi = 0.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(mid / math.sqrt(2.0)) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def aperture_replica(dn, x, y, r_ap, r_in, r_out, clip=3.0, iters=4):
    """复刻生产语义：几何判据 distance<=r 纳入孔径，环带 r_in..r_out。"""
    h, w = dn.shape
    y0, y1 = max(0, y - r_out), min(h, y + r_out + 1)
    x0, x1 = max(0, x - r_out), min(w, x + r_out + 1)
    sub = dn[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    ap = d <= r_ap
    an = (d >= r_in) & (d <= r_out)
    n_ap = int(ap.sum())
    n_an = int(an.sum())
    med, sd, mad, n_eff = sigma_clipped_stats(sub[an], clip, iters)
    flux = float((sub[ap] - med).sum())
    var = n_ap * sd * sd + (n_ap * n_ap) / max(n_an, 1) * sd * sd
    var_mad = n_ap * mad * mad + (n_ap * n_ap) / max(n_an, 1) * mad * mad
    return {
        "x": x, "y": y,
        "n_aperture_pixels": n_ap, "n_annulus_pixels": n_an, "n_annulus_after_clip": n_eff,
        "background_median_adu": med, "annulus_std_adu": sd, "annulus_mad_sigma_adu": mad,
        "flux_adu": flux,
        "flux_error_adu_std": math.sqrt(var) if var > 0 else float("nan"),
        "snr_std": flux / math.sqrt(var) if var > 0 else float("nan"),
        "flux_error_adu_mad": math.sqrt(var_mad) if var_mad > 0 else float("nan"),
        "snr_mad": flux / math.sqrt(var_mad) if var_mad > 0 else float("nan"),
        "x_float_reported": None,
    }


def geometry_counts(r_ap, r_in, r_out):
    R = int(math.ceil(r_out))
    o = np.arange(-R, R + 1)
    yy, xx = np.meshgrid(o, o, indexing="ij")
    d = np.hypot(xx, yy)
    return int((d <= r_ap).sum()), int(((d >= r_in) & (d <= r_out)).sum()), d


# ---------------------------------------------------------------- E0


def run_e0(dn, raw, data_offset, hdr, cards, cfg, out):
    c = cfg["e0"]
    ny, nx = dn.shape
    flat = dn.ravel()
    neg_mask = flat < 0
    neg_idx = np.flatnonzero(neg_mask)
    ys, xs = np.divmod(neg_idx, nx)

    stats = {
        "min_dn": int(flat.min()), "max_dn": int(flat.max()),
        "n_total_pixels": int(flat.size),
        "n_negative": int(neg_mask.sum()),
        "n_minus_one": int((flat == -1).sum()),
        "n_below_minus_1000": int((flat < -1000).sum()),
        "n_band_32760_32767": int(((flat >= c["plateau_band_low"]) & (flat <= c["plateau_band_high"])).sum()),
        "n_exactly_32767": int((flat == 32767).sum()),
        "n_exactly_32750": int((flat == 32750).sum()),
        "n_zero": int((flat == 0).sum()),
        "n_ge_20000": int((flat >= 20000).sum()),
        "n_ge_30000": int((flat >= 30000).sum()),
        "n_le_minus_30000": int((flat <= -30000).sum()),
    }

    # 全域直方图（粗）+ 负值区细分
    coarse_edges = list(range(-32768, 32769, 1024))
    coarse = np.histogram(flat, bins=coarse_edges)[0].tolist()
    neg_vals = flat[neg_mask]
    neg_edges = list(range(-32768, 1, 512))
    neg_hist = np.histogram(neg_vals, bins=neg_edges)[0].tolist()

    # 最负 N 个像元
    order = np.argsort(neg_vals, kind="stable")[: c["worst_n"]]
    worst = []
    nb = c["neighbour_radius"]
    for k in order:
        v = int(neg_vals[k])
        y, x = int(ys[k]), int(xs[k])
        y0, y1 = max(0, y - nb), min(ny, y + nb + 1)
        x0, x1 = max(0, x - nb), min(nx, x + nb + 1)
        win = dn[y0:y1, x0:x1].astype(np.float64).ravel()
        win = np.delete(win, (y - y0) * (x1 - x0) + (x - x0))
        pos = win[win >= 0]
        worst.append({
            "dn": v, "unsigned_if_wrapped": v + 65536, "x": x, "y": y,
            "neighbour_mean_all": float(win.mean()),
            "neighbour_mean_nonneg": float(pos.mean()) if pos.size else None,
            "neighbour_max": float(win.max()),
        })

    # 每颗负值像元的“是否落在亮核”判据
    nb_med = np.empty(neg_idx.size)
    nb_max = np.empty(neg_idx.size)
    for k in range(neg_idx.size):
        y, x = int(ys[k]), int(xs[k])
        win = dn[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2].astype(np.float64).ravel()
        win = np.delete(win, min(4, win.size - 1))
        pos = win[win >= 0]
        nb_med[k] = np.median(pos) if pos.size else np.nan
        nb_max[k] = win.max()
    sky_med, sky_sig = 21.0, None
    med_global = float(np.median(flat))
    _, sd_global, mad_global, _ = sigma_clipped_stats(flat[::37], 3.0, 3)
    sky_med = med_global
    sky_sig = mad_global if mad_global > 0 else sd_global
    core_thr = c["bright_core_neighbour_median_adu"]
    med_thr = c["medium_core_neighbour_median_adu"]
    in_bright = float(np.mean(nb_med > core_thr)) if neg_idx.size else float("nan")
    in_medium = float(np.mean(nb_med > med_thr)) if neg_idx.size else float("nan")
    in_sky = int(np.sum(~(nb_med > sky_med + 5 * sky_sig))) if neg_idx.size else 0

    in_tel = (ys == 0) & (xs < 104)
    nbq = {str(q): float(np.nanpercentile(nb_med, q)) for q in [0, 5, 25, 50, 75, 95, 100]}
    code_groups = []
    for label in ("minus_1", "not_minus_1"):
        sel = (neg_vals == -1) if label == "minus_1" else (neg_vals != -1)
        if not sel.any():
            continue
        code_groups.append({
            "group": label, "n": int(sel.sum()),
            "nb_med_p50_adu": float(np.nanmedian(nb_med[sel])),
            "frac_nb_med_gt_200_adu": float(np.mean(nb_med[sel] > core_thr)),
            "frac_nb_med_le_58_adu": float(np.mean(nb_med[sel] <= 58.0)),
        })
    partition = {
        "n_negative_total": int(neg_idx.size),
        "n_negative_in_telemetry_region": int(in_tel.sum()),
        "n_negative_outside_telemetry": int((~in_tel).sum()),
        "n_negative_sky_region_outside_telemetry": int(np.sum((~in_tel) & ~(nb_med > sky_med + 5 * sky_sig))),
        "frac_outside_telemetry_in_bright_core": float(np.mean(nb_med[~in_tel] > core_thr)) if (~in_tel).any() else None,
        "neighbour_median_percentiles_adu": nbq,
        "code_groups": code_groups,
    }

    # 编码码值结构：是否有单一码值尖峰（显式标记码的特征）
    uniq, cnt = np.unique(neg_vals, return_counts=True)
    order2 = np.argsort(-cnt, kind="stable")[: c["code_spike_top_n"]]
    top_codes = [{"dn": int(uniq[i]), "count": int(cnt[i])} for i in order2]
    top_share = float(cnt[order2[0]] / neg_vals.size) if neg_vals.size else float("nan")

    # 平台（>=3 连续）检测：对 32760-32767 带 与 全饱和码 -1 各做一次
    def runs(mask_2d, min_run):
        n_runs, max_len = 0, 0
        for axis in (0, 1):
            m = mask_2d if axis == 0 else mask_2d.T
            for row in m:
                if not row.any():
                    continue
                idx = np.flatnonzero(row)
                if idx.size == 0:
                    continue
                breaks = np.flatnonzero(np.diff(idx) > 1)
                starts = np.concatenate(([0], breaks + 1))
                ends = np.concatenate((breaks, [idx.size - 1]))
                for s, e in zip(starts, ends):
                    L = e - s + 1
                    if L >= min_run:
                        n_runs += 1
                        max_len = max(max_len, L)
        return n_runs, max_len

    band = (dn >= c["plateau_band_low"]) & (dn <= c["plateau_band_high"])
    plateau = {
        "band_runs_ge_min": runs(band, c["plateau_min_run"])[0],
        "band_longest_run": runs(band, c["plateau_min_run"])[1],
        "band_pixels": int(band.sum()),
        "minus1_runs_ge_min": runs(dn == -1, c["plateau_min_run"])[0],
        "minus1_longest_run": runs(dn == -1, c["plateau_min_run"])[1],
        "minus1_pixels": int((dn == -1).sum()),
    }

    # 遥测
    aux = decode_aux(raw, data_offset)
    qnorm = float(math.sqrt(sum(v * v for v in aux[10:14])))
    az_card = float(hdr["AZIMUTH"][0].split("/")[0].split()[0])
    aux_block = {
        "decoded_26_doubles": aux,
        "field0_vs_AZIMUTH_card": {"aux": aux[0], "card": az_card, "abs_diff": abs(aux[0] - az_card)},
        "ra_deg": aux[2], "dec_deg": aux[3],
        "quaternion_q1q2q3q4": aux[10:14], "quaternion_norm": qnorm,
        "quaternion_norm_dev_from_1": abs(qnorm - 1.0),
        "aux_bytes_in_pixel_array": 26 * 8,
        "aux_pixels_in_row0": 26 * 8 // 2,
        "permanent_mask": {
            "y": 0, "x_first": 0, "x_last": 26 * 8 // 2 - 1,
            "reason": "26 个小端 double 遥测覆盖像元阵列前 208 字节",
        },
    }

    # 判定取舍
    hypotheses = {
        "A_unsigned_wrap": {
            "prediction": "负数 = 真实 ADC 值 32768..65535 的补码回绕；应 100% 落在亮核内、码值连续分布、满量程平台以 -1 出现",
            "evidence": {
                "fraction_negatives_in_bright_core": in_bright,
                "fraction_negatives_in_medium_core": in_medium,
                "n_negatives_in_sky_region": in_sky,
                "top_code_share": top_share,
                "n_distinct_negative_codes": int(uniq.size),
                "minus1_plateau_runs": plateau["minus1_runs_ge_min"],
                "minus1_longest_run": plateau["minus1_longest_run"],
            },
        },
        "B_explicit_marker": {
            "prediction": "饱和/坏点用固定码值标记；应出现单一码值尖峰与平台，且码值不随亮度连续变化",
            "evidence": {"top_codes": top_codes, "top_code_share": top_share,
                          "n_distinct_negative_codes": int(uniq.size)},
        },
        "C_true_negative": {
            "prediction": "真实负读数；应出现在天空区而非亮核内",
            "evidence": {"n_negatives_in_sky_region": in_sky,
                          "fraction_in_sky": (in_sky / max(1, int(neg_idx.size)))},
        },
    }

    # 诊断图 1：最亮饱和核 64x64 切块
    cand = int(np.nanargmax(nb_med)) if neg_idx.size else None
    half = c["cutout_half"]
    if cand is not None:
        y, x = int(ys[cand]), int(xs[cand])
        y0, y1 = max(0, y - half), min(ny, y + half)
        x0, x1 = max(0, x - half), min(nx, x + half)
        crop = dn[y0:y1, x0:x1].astype(np.float64)
        pos = crop[crop >= 0]
        if pos.size:
            vmin, vmax = np.percentile(pos, [1, 99.5])
            vmax = max(vmax, vmin + 1)
            g = np.clip((np.clip(crop, vmin, vmax) - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
            rgb = np.stack([g, g, g], axis=-1)
            rgb[crop < 0] = [255, 0, 0]
            up = c["cutout_upscale"]
            rgb = np.repeat(np.repeat(rgb, up, axis=0), up, axis=1)
            Image.fromarray(rgb).save(out / "e0_cutout_bright_core.png")
            cutout_info = {"center_x": x, "center_y": y, "dn_at_center": int(dn[y, x]),
                            "n_negatives_in_crop": int((crop < 0).sum()),
                            "display_vmin": float(vmin), "display_vmax": float(vmax)}
        else:
            cutout_info = {"error": "no positive pixels in crop"}
    else:
        cutout_info = {"error": "no negative pixels"}

    # 诊断图 2：全帧负值分布缩略图
    nm = np.zeros((ny, nx), dtype=np.uint8)
    nm[ys, xs] = 255
    size = c["neg_map_size"]
    f = ny // size
    nm_small = nm[: size * f, : size * f].reshape(size, f, size, f).max(axis=(1, 3))
    Image.fromarray(nm_small).save(out / "e0_negative_map.png")

    res = {
        "section": "E0",
        "core_statistics": stats,
        "histogram_coarse_edges": coarse_edges,
        "histogram_coarse_counts": coarse,
        "histogram_negative_edges": neg_edges,
        "histogram_negative_counts": neg_hist,
        "worst_n_negative_pixels": worst,
        "negative_neighbour_stats": {
            "median_of_neighbour_medians_adu": float(np.nanmedian(nb_med)) if nb_med.size else None,
            "min_of_neighbour_medians_adu": float(np.nanmin(nb_med)) if nb_med.size else None,
            "sky_median_adu": sky_med, "sky_sigma_adu": sky_sig,
            "core_threshold_adu": core_thr, "medium_threshold_adu": med_thr,
        },
        "negative_pixel_partition": partition,
        "plateau": plateau,
        "aux_telemetry": aux_block,
        "hypotheses": hypotheses,
        "cutout": cutout_info,
        "artifacts": ["e0_cutout_bright_core.png", "e0_negative_map.png"],
    }
    return res


# ---------------------------------------------------------------- E1


def e1_box_sigma_local(patch):
    """生产同款噪声估计器：max(1.4826*MAD, std)。"""
    v = np.asarray(patch, dtype=np.float64).ravel()
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    sd = float(np.std(v, ddof=1))
    return max(mad, sd, 1e-6)


def run_e1(dn, cfg, out):
    c = cfg["e1"]
    ny, nx = dn.shape
    box = c["box"]
    med_map, sig_map = box_stats(dn, box)
    sig_flat = sig_map.ravel()
    qs = [0, 1, 5, 16, 50, 84, 95, 99, 100]
    pct = {str(q): float(np.percentile(sig_flat, q)) for q in qs}

    mg = c["macro_grid"]
    by, bx = sig_map.shape
    macro = np.zeros((mg, mg))
    for i in range(mg):
        for j in range(mg):
            macro[i, j] = float(sig_map[i * by // mg : (i + 1) * by // mg, j * bx // mg : (j + 1) * bx // mg].mean())
    cy, cx = np.mgrid[0:by, 0:bx]
    A = np.column_stack([np.ones(by * bx), cx.ravel(), cy.ravel()])
    coef, *_ = np.linalg.lstsq(A, sig_flat, rcond=None)
    trend = {
        "macro_cell_sigma_adu": macro.tolist(),
        "macro_cell_max_over_min": float(macro.max() / max(macro.min(), 1e-9)),
        "linear_fit_sigma_vs_box_grid": {"c0": float(coef[0]), "d_sigma_per_box_x": float(coef[1]),
                                         "d_sigma_per_box_y": float(coef[2])},
        "sigma_dynamic_range_p999_over_p5": float(pct["99"] / max(pct["5"], 1e-9)),
        "sigma_dynamic_range_max_over_min": float(sig_flat.max() / max(sig_flat.min(), 1e-9)),
    }

    # source-free 采样
    bg_map = upsample_blocky(med_map, ny, nx)
    sg_map = upsample_blocky(sig_map, ny, nx)
    resid = dn.astype(np.float32) - bg_map.astype(np.float32)
    star = np.abs(resid) > c["clean_sigma_mult"] * sg_map
    star[0, : 104] = True  # 遥测区永久排除
    dilated = dilate_bool(star, c["dilate_px"])
    allowed = ~dilated
    m = c["edge_margin"]
    allowed[:m, :] = False; allowed[-m:, :] = False; allowed[:, :m] = False; allowed[:, -m:] = False

    rng = np.random.default_rng(cfg["e1"].get("rng_seed", 20260330))
    ay, ax = np.nonzero(allowed)
    n_req = c["n_sample_positions"]
    if ay.size < n_req:
        raise RuntimeError("not enough clean positions: %d" % ay.size)
    sel = rng.choice(ay.size, size=n_req, replace=False)
    py, px = ay[sel], ax[sel]

    ph = c["patch_half"]
    off = np.arange(-ph, ph + 1)
    dy, dx = np.meshgrid(off, off, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()
    patches = dn[py[:, None] + dy[None, :], px[:, None] + dx[None, :]].astype(np.float64)

    radii = c["aperture_radii"]
    sig_ap, n_geo = {}, {}
    for r in radii:
        sel_r = np.flatnonzero(dy**2 + dx**2 <= r * r)
        v = patches[:, sel_r].var(axis=1, ddof=1)
        sig_ap[r] = float(np.median(v)) ** 0.5
        R = int(math.ceil(r))
        g = np.mgrid[-R : R + 1, -R : R + 1]
        n_geo[r] = int(((g[0] ** 2 + g[1] ** 2) <= r * r).sum())

    r1 = radii[0]
    var1 = sig_ap[r1] ** 2 / n_geo[r1]
    rows = []
    for r in radii:
        v = sig_ap[r] ** 2
        R = v / (n_geo[r] * var1)
        rows.append({"r_px": r, "n_geo": n_geo[r], "sigma_ap_adu": sig_ap[r],
                     "var_ap_adu2": v, "N_eff": v / var1, "kappa_r": math.sqrt(R) if R > 0 else float("nan"),
                     "R_over_white_noise": R})
    ng = np.array([n_geo[r] for r in radii], dtype=float)
    vv = np.array([sig_ap[r] ** 2 for r in radii], dtype=float)
    A2 = np.column_stack([np.ones_like(ng), ng])
    coef2, *_ = np.linalg.lstsq(A2, vv, rcond=None)
    fit = {"intercept_adu2": float(coef2[0]), "slope_adu2_per_pixel": float(coef2[1]),
           "sigma_pix_from_anchor_adu": math.sqrt(var1),
           "kappa_sq_from_slope_over_anchor": float(coef2[1] / var1)}
    fit["kappa"] = math.sqrt(fit["kappa_sq_from_slope_over_anchor"]) if fit["kappa_sq_from_slope_over_anchor"] > 0 else float("nan")
    fit["effective_zero_point_pixels"] = float(-coef2[0] / coef2[1]) if coef2[1] != 0 else float("nan")

    # 生产估计量本身的方差（环带背景已扣）：误差模型应当比较的量
    est_rows = []
    for r in [2, 3, 4, 6]:
        ap_idx = np.flatnonzero(dy**2 + dx**2 <= r * r)
        ann_idx = np.flatnonzero((dy**2 + dx**2 >= 36) & (dy**2 + dx**2 <= 64))
        med = np.median(patches[:, ann_idx], axis=1)
        flux_i = (patches[:, ap_idx] - med[:, None]).sum(axis=1)
        n_ap_r = int(ap_idx.size)
        n_ann_r = int(ann_idx.size)
        centre = float(np.median(flux_i))
        s_rob = float(1.4826 * np.median(np.abs(flux_i - centre)))
        s_std = float(np.std(flux_i, ddof=1))
        sb = float(np.median([e1_box_sigma_local(patches[i]) for i in range(0, n_req, 25)]))
        pred = math.sqrt(n_ap_r * sb * sb + (n_ap_r * n_ap_r) / n_ann_r * sb * sb)
        est_rows.append({
            "aperture_r_px": r, "n_aperture_pixels": n_ap_r, "n_annulus_pixels": n_ann_r,
            "empirical_flux_rms_robust_adu": s_rob, "empirical_flux_rms_std_adu": s_std,
            "local_sigma_pix_median_adu": sb,
            "model_flux_error_adu_using_local_sigma": pred,
            "ratio_empirical_over_model": (s_rob / pred) if pred > 0 else None,
            "N_eff_equiv_empirical": (s_rob * s_rob / (sb * sb) - (n_ap_r * n_ap_r) / n_ann_r) if sb > 0 else None,
        })
    estimator = {
        "description": "在无源位置直接测量生产估计量 flux=sum_aperture(I-median(annulus 6-8px)) 的 RMS，与生产误差模型 sqrt(N_ap*s^2+N_ap^2/N_ann*s^2) 比较。",
        "rows": est_rows,
        "local_sigma_note": "local_sigma_pix 用生产同款估计器 max(1.4826*MAD,std) 在每位置 17x17 片内估计（每 25 个位置取一个，再取中位数）。",
    }

    # 背景梯度
    tile = c["tile"]
    slopes, quads, trec = [], [], []
    tny, tnx = ny // tile, nx // tile
    for i in range(tny):
        for j in range(tnx):
            sub = dn[i * tile : (i + 1) * tile, j * tile : (j + 1) * tile].astype(np.float64)
            v = sub.ravel()
            for _ in range(c["clip_iters"]):
                med = np.median(v)
                sd = np.std(v, ddof=1)
                if not (sd > 0):
                    break
                keep = np.abs(v - med) <= c["clip_sigma"] * sd
                if keep.all():
                    break
                v = v[keep]
            med = np.median(v)
            sd = np.std(v, ddof=1)
            keep = (np.abs(sub - med) <= c["clip_sigma"] * sd)
            yy, xx = np.mgrid[0:tile, 0:tile]
            yy = yy - tile / 2.0
            xx = xx - tile / 2.0
            z = sub[keep]
            X = np.column_stack([np.ones(keep.sum()), xx[keep], yy[keep]])
            b1, *_ = np.linalg.lstsq(X, z, rcond=None)
            g = math.hypot(b1[1], b1[2])
            X2 = np.column_stack([np.ones(keep.sum()), xx[keep], yy[keep], xx[keep] ** 2, yy[keep] ** 2, (xx * yy)[keep]])
            b2, *_ = np.linalg.lstsq(X2, z, rcond=None)
            # 对称环带 rho≈7 px: 一阶项精确抵消；二阶项残差 = (c3+c4)*rho^2/2
            bias = (b2[3] + b2[4]) * (7.0**2 / 2.0)
            slopes.append(g)
            quads.append(bias)
            trec.append({"tile_iy": i, "tile_ix": j, "slope_adu_per_px": g,
                         "quadratic_annulus_bias_adu": bias, "resid_std_adu": sd})
    slopes = np.array(slopes)
    quads = np.array(quads)
    n_ap4 = n_geo[4]
    sigma_ap4 = sig_ap[4]
    grad = {
        "n_tiles": int(slopes.size),
        "slope_percentiles_adu_per_px": {str(q): float(np.percentile(slopes, q)) for q in [50, 84, 95, 99, 100]},
        "quadratic_bias_percentiles_adu": {str(q): float(np.percentile(np.abs(quads), q)) for q in [50, 84, 95, 100]},
        "aperture_r4_noise_adu": sigma_ap4,
        "quadratic_bias_over_r4_noise_p50": float(np.percentile(np.abs(quads), 50) / sigma_ap4),
        "quadratic_bias_over_r4_noise_p95": float(np.percentile(np.abs(quads), 95) / sigma_ap4),
        "linear_term_note": "对称环带的环带均值等于中心处的一阶背景值，线性项按构造精确抵消；风险来自二阶/高阶结构与环带非对称（邻星掩膜）。",
        "tiles": trec,
    }

    answer = {
        "sigma_dynamic_range_max_over_min": trend["sigma_dynamic_range_max_over_min"],
        "sigma_p999_over_p5": trend["sigma_dynamic_range_p999_over_p5"],
        "is_single_global_noise_usable": bool(trend["sigma_dynamic_range_p999_over_p5"] < 1.25),
        "note": "若 p99/p5 超过 ~1.25，则单一全局噪声会在亮/暗背景区之间造成 >10% 的误差条偏差。",
    }
    return {
        "section": "E1",
        "box_sigma": {"box_px": box, "n_boxes": int(sig_flat.size), "percentiles_adu": pct,
                       "median_adu": float(np.median(sig_flat))},
        "spatial_trend": trend,
        "aperture_sigma_curve": {"n_positions": int(n_req), "radii": radii, "rows": rows, "linear_fit": fit},
        "estimator_variance": estimator,
        "background_gradient": grad,
        "answer_global_noise": answer,
        "sampling": {"clean_sigma_mult": c["clean_sigma_mult"], "dilate_px": c["dilate_px"],
                      "n_allowed_positions": int(ay.size), "rng_seed": int(cfg["e1"].get("rng_seed", 20260330))},
    }


# ---------------------------------------------------------------- E6


def detect(img, cfg, bg_sig_map, absolute=False):
    c = cfg["e6"]
    k = gaussian_kernel(c["fwhm_px"], c["kernel_truncate_sigma"])
    s1 = convolve1d(img, k, 0)
    s2 = convolve1d(s1, k, 1)
    sumk2 = float((np.outer(k, k) ** 2).sum())
    snr = s2 / (bg_sig_map * math.sqrt(sumk2))
    peaks = local_maxima(snr, c["proposal_sigma"])
    yx = np.argwhere(peaks)
    sv = snr[peaks]
    keep = nms(yx, sv, c["min_distance_px"])
    yx, sv = yx[keep], sv[keep]
    fwhm, ell = moments(img, yx, c["morphology_window"] // 2, absolute=absolute)
    lo, hi = c["fwhm_accept_px"]
    morph_ok = np.isfinite(fwhm) & (fwhm >= lo) & (fwhm <= hi) & np.isfinite(ell) & (ell <= c["ellip_max"])
    snr_ok = sv >= c["snr_cut"]
    res = {
        "n_peaks_above_proposal": int(yx.shape[0]),
        "n_snr_ge_cut": int(np.sum(snr_ok)),
        "n_snr_ge_cut_and_morphology": int(np.sum(snr_ok & morph_ok)),
        "max_snr": float(sv.max()) if sv.size else float("nan"),
        "snr_percentiles": {str(q): float(np.percentile(sv, q)) for q in [50, 90, 99, 100]} if sv.size else {},
    }
    return res, yx, sv, fwhm, ell, morph_ok, snr


def run_e6(dn, cfg, out):
    c = cfg["e6"]
    ny, nx = dn.shape
    box = c["background_box"]
    med_map, sig_map = box_stats(dn, box)
    bg_med = upsample_blocky(med_map, ny, nx).astype(np.float32)
    bg_sig = upsample_blocky(sig_map, ny, nx).astype(np.float32)
    img = dn.astype(np.float32) - bg_med
    global_med = float(np.median(dn))
    _, gsd, gmad, _ = sigma_clipped_stats(dn[::37, ::37].ravel(), 3.0, 3)
    sigma_global = float(np.median(sig_map))

    real, ryx, rsv, rfwhm, rell, rmorph, rsnr = detect(img, cfg, bg_sig)

    # 反相：对已扣背景的图取负，形态统计用 |win| 保证形状不变
    inv_img = (-img).astype(np.float32)
    inv_res, iyx, isv, ifwhm, iell, imorph, isnr = detect(inv_img, cfg, bg_sig, absolute=True)
    negdil = dilate_bool(dn < 0, 3)
    isel = (isv >= c["snr_cut"]) & imorph
    inv_res["n_quality_on_or_near_negative_pixel"] = int(negdil[iyx[isel, 0], iyx[isel, 1]].sum())
    inv_res["frac_quality_on_or_near_negative_pixel"] = (float(negdil[iyx[isel, 0], iyx[isel, 1]].mean()) if isel.any() else None)
    rsel = (rsv >= c["snr_cut"]) & rmorph
    real["n_quality_on_or_near_negative_pixel"] = int(negdil[ryx[rsel, 0], ryx[rsel, 1]].sum())
    cz = dn[iyx[isel, 0], iyx[isel, 1]].astype(np.float64)
    inv_res["inverted_center_pixel_stats"] = {
        "median_dn": float(np.median(cz)) if cz.size else None,
        "frac_center_dn_le_1": float(np.mean(cz <= 1)) if cz.size else None,
        "frac_center_dn_eq_0": float(np.mean(cz == 0)) if cz.size else None,
    }
    czr = dn[ryx[rsel, 0], ryx[rsel, 1]].astype(np.float64)
    real["real_center_pixel_stats"] = {"median_dn": float(np.median(czr)) if czr.size else None}

    # 纯高斯噪声
    rng = np.random.default_rng(c["synthetic_seed"])
    n_indep = float(nx * ny) / (math.pi * (c["fwhm_px"] / 2.0) ** 2)
    z_exp = inverse_normal_upper(1.0 / n_indep)
    synth = []
    for i in range(c["n_synthetic"]):
        f = rng.normal(global_med, sigma_global, size=(ny, nx)).astype(np.float32)
        smed = float(np.median(f))
        ssig = np.full((ny, nx), float(np.std(f)), dtype=np.float32)
        r, syx, ssv, *_ = detect(f - smed, cfg, ssig)
        r["realization"] = i
        r["empirical_max_snr"] = r["max_snr"]
        synth.append(r)

    # 生产候选位置探针
    px, py = c["production_probe_xy"]
    d2 = (ryx[:, 1] - px) ** 2 + (ryx[:, 0] - py) ** 2
    j = int(np.argmin(d2)) if ryx.size else -1
    probe = {"reported_x": px, "reported_y": py, "match_radius_px": c["probe_match_radius_px"]}
    if j >= 0 and math.sqrt(d2[j]) <= c["probe_match_radius_px"]:
        probe.update({"found": True, "x": int(ryx[j, 1]), "y": int(ryx[j, 0]),
                       "offset_px": float(math.sqrt(d2[j])), "matched_filter_snr": float(rsv[j]),
                       "fwhm_px": float(rfwhm[j]), "ellipticity": float(rell[j]),
                       "passes_morphology": bool(rmorph[j]), "passes_snr5": bool(rsv[j] >= c["snr_cut"])})
    else:
        probe.update({"found": False, "nearest_offset_px": float(math.sqrt(d2[j])) if j >= 0 else None,
                       "nearest_snr": float(rsv[j]) if j >= 0 else None})

    res = {
        "section": "E6",
        "declaration": "这不是生产检测器：FWHM=3 px 高斯匹配滤波 + 4 sigma 局部极大 + 3 px 抑制 + 形态筛；SNR 定义为匹配滤波 SNR，与生产的 r=4 孔径 SNR 不是同一个统计量。",
        "background": {"box_px": box, "global_median_adu": global_med,
                        "sigma_map_median_adu": sigma_global, "sigma_clip_global_adu": gsd,
                        "sigma_mad_global_adu": gmad},
        "real_image": real,
        "inverted_image": inv_res,
        "synthetic_gaussian": {
            "n_realizations": c["n_synthetic"], "sigma_used_adu": sigma_global,
            "n_indep_beams": n_indep, "expected_max_snr_evt": z_exp,
            "realizations": synth,
        },
        "production_probe": probe,
        "answer_indistinguishable": None,
    }
    n_syn_avg = float(np.mean([s["n_snr_ge_cut_and_morphology"] for s in synth]))
    n_inv = inv_res["n_snr_ge_cut_and_morphology"]
    probe_snr = probe.get("matched_filter_snr")
    noise_max = float(max(s["max_snr"] for s in synth))
    quiet_max = max(noise_max, z_exp)
    res["answer_indistinguishable"] = {
        "question": "记录到的 SNR=5.11（r=4 孔径统计）最暗质量源，是否与纯噪声极值不可区分？",
        "statistic_note": "孔径 SNR 与匹配滤波 SNR 是两个不同统计量；极值论证只在同一统计量内成立。",
        "probe_matched_filter_snr": probe_snr,
        "probe_fwhm_px": probe.get("fwhm_px"),
        "probe_ellipticity": probe.get("ellipticity"),
        "noise_only_max_snr_evt_expected": z_exp,
        "noise_only_max_snr_empirical_per_realization": [s["max_snr"] for s in synth],
        "noise_only_quality_count_per_frame": [s["n_snr_ge_cut_and_morphology"] for s in synth],
        "noise_only_quality_count_mean": n_syn_avg,
        "artifact_count_from_inverted_image": n_inv,
        "artifact_fraction_on_or_near_negative_pixel": inv_res["frac_quality_on_or_near_negative_pixel"],
        "matched_filter_verdict_probe_above_noise_ceiling": bool(probe_snr is not None and probe_snr > quiet_max),
        "aperture_snr_rule_R1_would_flag_noise": bool(5.110702614293284 <= z_exp),
        "indistinguishable_from_pure_noise_extreme_matched_filter": bool(not (probe_snr is not None and probe_snr > quiet_max)),
    }
    return res

# ---------------------------------------------------------------- E2


def run_e2(dn, cfg, e1):
    c = cfg["e2"]
    n_ap, n_an, _ = geometry_counts(c["aperture_radius"], c["annulus_inner_radius"], c["annulus_outer_radius"])
    flux = c["flux_adu"]
    snr_rep = c["flux_snr_reported"]
    err_rep = flux / snr_rep
    sigma_implied = err_rep / math.sqrt(n_ap + (n_ap * n_ap) / n_an)

    rows = []
    for g in c["gain_grid"]:
        for rn in c["read_noise_adu_grid"]:
            src = (flux / g) if g else 0.0
            bgv = n_ap * sigma_implied**2 + (n_ap * n_ap) / n_an * sigma_implied**2
            var = src + n_ap * rn * rn + bgv
            var_nodbl = src + bgv
            rows.append({
                "gain_e_per_adu": g if g else "UNKNOWN",
                "read_noise_adu": rn,
                "source_poisson_var_adu2": src,
                "read_noise_term_adu2": n_ap * rn * rn,
                "background_var_adu2": bgv,
                "total_var_adu2": var,
                "snr_as_coded": flux / math.sqrt(var),
                "sigma_mag_as_coded": 1.0857362047581294 / (flux / math.sqrt(var)),
                "snr_no_double_count": flux / math.sqrt(var_nodbl),
                "sigma_mag_no_double_count": 1.0857362047581294 / (flux / math.sqrt(var_nodbl)),
                "delta_sigma_mag_vs_reported": 1.0857362047581294 / (flux / math.sqrt(var)) - c["instrumental_magnitude_error_reported"],
            })

    # 独立复测该位置
    px, py = cfg["e6"]["production_probe_xy"]
    probes = []
    for iy in range(int(math.floor(py)) - 1, int(math.floor(py)) + 3):
        for ix in range(int(math.floor(px)) - 1, int(math.floor(px)) + 3):
            a = aperture_replica(dn, ix, iy, c["aperture_radius"], c["annulus_inner_radius"], c["annulus_outer_radius"])
            a["x_float_reported"] = px
            a["y_float_reported"] = py
            a["abs_flux_diff_vs_reported"] = abs(a["flux_adu"] - flux)
            probes.append(a)
    probes.sort(key=lambda d: d["abs_flux_diff_vs_reported"])

    # --- 噪声估计器辨识：为什么独立复算的 SNR 与生产不一致 -----------------
    best = probes[0]
    ix, iy = int(best["x"]), int(best["y"])
    R = int(math.ceil(c["annulus_outer_radius"]))
    o = np.arange(-R, R + 1)
    yy, xx = np.meshgrid(o, o, indexing="ij")
    dd = np.hypot(xx, yy)
    win = dn[iy - R : iy + R + 1, ix - R : ix + R + 1].astype(np.float64)
    av = win[(dd >= c["annulus_inner_radius"]) & (dd <= c["annulus_outer_radius"])]

    def est_all(v):
        v = np.asarray(v, dtype=np.float64)
        med, sd, mad, n = sigma_clipped_stats(v, 3.0, 4)
        q16, q50, q84 = np.percentile(v, [15.865, 50, 84.135])
        q25, q75 = np.percentile(v, [25, 75])
        med0 = np.median(v)
        clip_mad = np.abs(v - med0) <= 3.0 * 1.4826 * np.median(np.abs(v - med0))
        return {
            "n": int(v.size),
            "std_raw": float(np.std(v, ddof=1)),
            "mad_sigma_raw": float(1.4826 * np.median(np.abs(v - med0))),
            "iqr_over_1_349_raw": float((q75 - q25) / 1.349),
            "p16_84_halfwidth_raw": float((q84 - q16) / 2.0),
            "median_raw": float(med0),
            "clipped_std": float(sd), "n_after_clip": int(n), "clipped_median": float(med),
            "clipped_mad_sigma": float(mad),
            "mad3sigma_clip_std": float(np.std(v[clip_mad], ddof=1)) if clip_mad.sum() > 2 else None,
        }

    estimators_annulus = est_all(av)
    bx = iy // cfg["e6"]["background_box"]
    bxs = ix // cfg["e6"]["background_box"]
    bb = cfg["e6"]["background_box"]
    boxvals = dn[bx * bb : (bx + 1) * bb, bxs * bb : (bxs + 1) * bb].astype(np.float64)
    estimators_box128 = est_all(boxvals)
    target = sigma_implied
    cand = {"annulus_std_raw": estimators_annulus["std_raw"],
            "annulus_mad_sigma_raw": estimators_annulus["mad_sigma_raw"],
            "annulus_iqr": estimators_annulus["iqr_over_1_349_raw"],
            "annulus_p16_84": estimators_annulus["p16_84_halfwidth_raw"],
            "annulus_clipped_std": estimators_annulus["clipped_std"],
            "annulus_clipped_mad_sigma": estimators_annulus["clipped_mad_sigma"],
            "annulus_mad3sigma_clip_std": estimators_annulus["mad3sigma_clip_std"],
            "box128_std_raw": estimators_box128["std_raw"],
            "box128_mad_sigma_raw": estimators_box128["mad_sigma_raw"],
            "box128_clipped_std": estimators_box128["clipped_std"],
            "box128_clipped_mad_sigma": estimators_box128["clipped_mad_sigma"]}
    ident = {"implied_sigma_from_production_snr": target,
             "candidates": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cand.items()},
             "closest_candidate": min([k for k, v in cand.items() if v], key=lambda k: abs(cand[k] - target)),
             "closest_abs_diff": min([abs(v - target) for v in cand.values() if v])}
    ident["ratio_implied_over_annulus_std"] = target / estimators_annulus["std_raw"]
    ident["annulus_values_adu"] = [round(float(v), 3) for v in av]
    ident["probe_position"] = {"x": ix, "y": iy}

    neff = None
    if e1:
        for row in e1["aperture_sigma_curve"]["rows"]:
            if row["r_px"] == c["aperture_radius"]:
                neff = row["N_eff"]
    var_neff = None
    if neff:
        var_neff = n_ap * sigma_implied**2 * (neff / n_ap) + (n_ap * n_ap) / n_an * sigma_implied**2

    return {
        "section": "E2",
        "inputs": {
            "flux_adu": flux, "flux_snr_reported": snr_rep,
            "exposure_s": c["exposure_s"], "exposure_source": "FITS EXPOSURE 卡片（非标准格式，注释单位 ms，本项目解释为 1500 ms）",
            "instrumental_magnitude_reported": c["instrumental_magnitude_reported"],
            "instrumental_magnitude_error_reported": c["instrumental_magnitude_error_reported"],
        },
        "geometry": {"aperture_radius_px": c["aperture_radius"], "n_aperture_pixels_geometric": n_ap,
                      "annulus_inner_px": c["annulus_inner_radius"], "annulus_outer_px": c["annulus_outer_radius"],
                      "n_annulus_pixels_geometric": n_an},
        "implied": {
            "flux_error_adu": err_rep,
            "sigma_per_pixel_adu_implied": sigma_implied,
            "formula": "var = N_ap*sigma^2 + N_ap^2/N_ann*sigma^2 (gain UNKNOWN, read_noise 0, 生产当前默认)",
            "magnitude_error_check": 1.0857362047581294 / snr_rep,
        },
        "grid": rows,
        "neff_sensitivity": {"N_eff_at_r4": neff,
                              "snr_if_N_eff_used": (flux / math.sqrt(var_neff)) if var_neff else None,
                              "sigma_mag_if_N_eff_used": (1.0857362047581294 / (flux / math.sqrt(var_neff))) if var_neff else None},
        "independent_replica": probes[:4],
        "noise_estimator_identification": ident,
        "assumptions": ["环带内无第三方掩膜（生产端 mask 未知）", "图像未做任何像素域修正（bias/dark/flat 状态 UNKNOWN）",
                         "sigma 由环带 sigma-clip 统计给出，其估计器在 std 与 1.4826*MAD 之间待定，两者的数值都列出"],
    }


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all", choices=["all", "e0", "e1", "e6", "e2"])
    ap.add_argument("--runid", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg_bytes = CONFIG_PATH.read_bytes()
    cfg = json.loads(cfg_bytes.decode("utf-8"))
    cfg_sha = sha256_bytes(cfg_bytes)
    script_sha = sha256_file(SCRIPT_PATH)
    fits_path = ROOT / cfg["input"]["fits_relative"]
    t0 = utc_now()
    runid = args.runid or (t0.replace("-", "").replace(":", "").replace(".", "")[:18] + "_" + cfg_sha[:8])
    out = ROOT / cfg["output_root_relative"] / runid
    out.mkdir(parents=True, exist_ok=True)

    prov = {
        "tool": "xcheck_frame1.py (独立交叉验证工具，非生产实现)",
        "interpreter": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pillow_version": Image.__version__,
        "script_path": str(SCRIPT_PATH),
        "script_sha256": script_sha,
        "config_path": str(CONFIG_PATH),
        "config_sha256": cfg_sha,
        "input_fits_path": str(fits_path),
        "input_fits_size_bytes": fits_path.stat().st_size,
        "input_fits_sha256": None,
        "utc_start": t0,
        "runid": runid,
    }

    print("[prov] interpreter:", prov["interpreter"].split()[0], "numpy", np.__version__, "PIL", Image.__version__)
    print("[prov] script sha256:", script_sha)
    print("[prov] config sha256:", cfg_sha)
    print("[prov] runid:", runid)
    print("[prov] out:", out)

    print("[fits] hashing input ...")
    in_sha = sha256_file(fits_path)
    prov["input_fits_sha256"] = in_sha
    print("[fits] input sha256:", in_sha)
    match = in_sha == cfg["input"]["expected_sha256"].upper()
    prov["input_sha256_matches_expected"] = bool(match)
    print("[fits] matches expected:", match)
    if not match and not args.force:
        raise SystemExit("input sha256 mismatch; abort (use --force to override)")

    hdr, cards, dn, data_off, nbytes = read_fits(fits_path)
    prov["header_cards"] = len(cards)
    prov["data_offset_bytes"] = int(data_off)
    prov["shape"] = list(dn.shape)
    prov["bitpix"] = int(hdr["BITPIX"][0].split("/")[0])
    prov["file_size_bytes"] = int(nbytes)
    expected_size = data_off + dn.shape[0] * dn.shape[1] * 2
    prov["file_size_matches_pixels"] = bool(nbytes == expected_size)
    absent = [k for k in ["GAIN", "SATURATE", "FILTER", "TIMESYS", "TELESCOP", "INSTRUME", "BUNIT", "RDNOISE",
                          "CTYPE1", "CRVAL1", "CRPIX1", "CD1_1"] if k not in hdr]
    prov["absent_header_keys"] = absent
    prov["exposure_raw_card"] = [c for c in cards if c.startswith("EXPOSURE")][0]
    print("[fits] cards:", len(cards), "offset:", data_off, "shape:", dn.shape, "absent:", ",".join(absent))

    raw = fits_path.read_bytes()
    written = []
    if args.section in ("all", "e0"):
        print("[e0] running ...")
        r = run_e0(dn, raw, data_off, hdr, cards, cfg, out)
        r["provenance"] = prov
        dump_json(out / "e0_encoding.json", r)
        written.append("e0_encoding.json")
        cs = r["core_statistics"]
        print("[e0] min/max:", cs["min_dn"], cs["max_dn"], "neg:", cs["n_negative"], "-1:", cs["n_minus_one"],
              "band:", cs["n_band_32760_32767"], "in_bright_core:", round(r["hypotheses"]["A_unsigned_wrap"]["evidence"]["fraction_negatives_in_bright_core"], 4))
    e1res = None
    e1path = out / "e1_noise.json"
    if e1path.exists():
        e1res = json.loads(e1path.read_text(encoding="utf-8"))
    if args.section in ("all", "e1"):
        print("[e1] running ...")
        e1res = run_e1(dn, cfg, out)
        e1res["provenance"] = prov
        dump_json(e1path, e1res)
        written.append("e1_noise.json")
        print("[e1] box sigma p50/p5/p99:", round(e1res["box_sigma"]["percentiles_adu"]["50"], 3),
              round(e1res["box_sigma"]["percentiles_adu"]["5"], 3), round(e1res["box_sigma"]["percentiles_adu"]["99"], 3))
    if args.section in ("all", "e6"):
        print("[e6] running ...")
        r = run_e6(dn, cfg, out)
        r["provenance"] = prov
        dump_json(out / "e6_falsepositives.json", r)
        written.append("e6_falsepositives.json")
        a = r["answer_indistinguishable"]
        print("[e6] real:", r["real_image"]["n_snr_ge_cut_and_morphology"],
              "inverted-artifacts:", a["artifact_count_from_inverted_image"],
              "(on neg px %.3f)" % (a["artifact_fraction_on_or_near_negative_pixel"] or 0.0),
              "noise-only mean:", round(a["noise_only_quality_count_mean"], 2),
              "probe MF snr:", round(a["probe_matched_filter_snr"], 3),
              "noise ceiling:", round(max(a["noise_only_max_snr_empirical_per_realization"]), 3))
    if args.section in ("all", "e2"):
        print("[e2] running ...")
        r = run_e2(dn, cfg, e1res)
        r["provenance"] = prov
        (out / "e2_snr_table.json").write_text(json.dumps(r, indent=2, ensure_ascii=False, default=_np_default), encoding="utf-8")
        written.append("e2_snr_table.json")
        print("[e2] N_ap:", r["geometry"]["n_aperture_pixels_geometric"],
              "N_ann:", r["geometry"]["n_annulus_pixels_geometric"],
              "sigma_implied:", round(r["implied"]["sigma_per_pixel_adu_implied"], 4))

    prov["utc_end"] = utc_now()
    prov["sections_written"] = written
    dump_json(out / "run_manifest.json", prov)
    print("[done] wrote:", ", ".join(written + ["run_manifest.json"]))


if __name__ == "__main__":
    main()
