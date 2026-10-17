#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xcheck_phase1b_e4.py -- E4 增长曲线 / 孔径修正 / 经验 PSF（第 1 帧）。

输入：原始 FITS 第 1 帧（只读）+ 冻结配置。
输出：e4_growth_curve.json, epf_profile.json, epf_profile_oversampled.npy
不声明任何星等/精度/标定。
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import xcheck_phase1b_lib as L


def mag(x):
    return 2.5 * math.log10(x) if (x and np.isfinite(x) and x > 0) else None


def build_psf(fr, ys, xs, sky, norm, half, os_, r_max_moment):
    """从一组星叠出经验 PSF。返回 (P1, Pos, bsum, n_used)。"""
    ny, nx = fr.dn.shape
    iy, ix, valid, d = L._win_index(ys, xs, half, ny, nx)
    vals = fr.dn[iy, ix].astype(np.float64)
    res = np.where(valid, vals - sky[:, None], 0.0)
    bsum = res.sum(axis=1)
    wins = res / bsum[:, None]
    P1, n = L.psf_align(wins, ys - np.floor(ys), xs - np.floor(xs), pad=10)
    Pos = L.fourier_upsample(P1, os_)
    Pos, oy, ox = L.recenter_psf(Pos, os_)
    return P1, Pos, bsum, n, oy, ox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stars", type=int, default=None)
    args = ap.parse_args()
    cfg, cfg_sha = L.load_config()
    runid = L.make_runid(cfg_sha)
    outdir = L.ROOT / cfg["output_root_relative"] / runid
    outdir.mkdir(parents=True, exist_ok=True)
    c = cfg["e4"]
    apc = cfg["aperture"]
    p0 = cfg["prior_results"]
    sigma_ap = p0["sigma_ap_r4_adu_from_e1"]

    fits = L.ROOT / cfg["input"]["fits_relative"] / cfg["input"]["reference_fits"]
    prov = L.provenance(Path(__file__).resolve(), cfg_sha, runid)
    prov["input_fits_path"] = str(fits)
    prov["input_fits_size_bytes"] = fits.stat().st_size
    prov["input_fits_sha256"] = L.sha256_file(fits)
    prov["input_sha256_matches_expected"] = prov["input_fits_sha256"] == cfg["input"]["expected_sha256_reference"]
    print("[prov]", prov["script_sha256"], "cfg", cfg_sha, "runid", runid)
    if not prov["input_sha256_matches_expected"]:
        raise SystemExit("reference FITS sha256 mismatch")

    dn, hdr = L.load_frame(fits)
    print("[fits] shape", dn.shape, "median", float(np.median(dn)))
    fr = L.Frame(dn, cfg)
    print("[bg] sigma map median", round(fr.sigma_global, 3))

    det = fr.detect(keep_morph=False)
    yx, mf_snr = det["yx"], det["snr"]
    print("[detect] peaks >= %.1f sigma: %d" % (cfg["detect"]["mf_threshold_snr"], len(yx)))

    peak_dn = fr.peak_dn(yx)
    apm = L.aperture_measure(fr.dn, yx[:, 0].astype(float), yx[:, 1].astype(float),
                             apc["r_ap"], apc["r_in"], apc["r_out"])
    flux_ap = apm["flux"]
    snr_bg = flux_ap / sigma_ap
    iso = L.nearest_neighbor_px(yx.astype(float), c["isolation_px_growth"])

    sel = (snr_bg >= c["ap_snr_min"]) & (peak_dn < c["peak_max_dn"]) & (iso >= c["isolation_px_growth"])
    idx = np.flatnonzero(sel)
    print("[select] snr_bg>=%.0f:%d peak<%.0f:%d iso>=%.0f:%d all:%d" % (
        c["ap_snr_min"], int((snr_bg >= c["ap_snr_min"]).sum()),
        c["peak_max_dn"], int((peak_dn < c["peak_max_dn"]).sum()),
        c["isolation_px_growth"], int((iso >= c["isolation_px_growth"]).sum()), idx.size))

    fwhm0, ell0 = L.moments(fr.img, yx[idx], cfg["detect"]["morphology_window"] // 2)
    lo, hi = cfg["detect"]["fwhm_accept_px"]
    m_ok = np.isfinite(fwhm0) & (fwhm0 >= lo) & (fwhm0 <= hi) & np.isfinite(ell0) & (ell0 <= cfg["detect"]["ellip_max"])
    idx = idx[m_ok]
    print("[select] morphology ok: %d" % idx.size)

    ys0 = yx[idx, 0].astype(float)
    xs0 = yx[idx, 1].astype(float)
    cy, cx = L.centroid(fr.dn, ys0, xs0, r=3.0)
    cy4, cx4 = L.centroid(fr.dn, ys0, xs0, r=4.0)
    cspread = np.hypot(cy4 - cy, cx4 - cx)

    apm_c = L.aperture_measure(fr.dn, cy, cx, apc["r_ap"], apc["r_in"], apc["r_out"])
    flux_c = apm_c["flux"]
    keep = np.isfinite(flux_c) & np.isfinite(cy) & np.isfinite(cx)
    srt = np.argsort(-flux_c)[: (args.max_stars or flux_c.size)]
    srt = srt[keep[srt]]
    print("[sample] growth-curve stars:", srt.size)

    ysg, xsg = cy[srt], cx[srt]
    A, sky = L.growth_curve(fr.dn, ysg, xsg, c["r_max"], c["sky_in"], c["sky_out"])
    finf_plat, finf_fit, bcoef = L.growth_total(A, c["plateau_r"], c["extrap_r"])
    ok = np.isfinite(finf_fit) & (finf_fit > 0) & np.isfinite(finf_plat) & np.isfinite(A).all(axis=1)
    print("[growth] usable:", int(ok.sum()), "/", srt.size)
    A = A[ok]
    finf_fit = finf_fit[ok]
    finf_plat = finf_plat[ok]
    bcoef = bcoef[ok]
    ysg, xsg, sky = ysg[ok], xsg[ok], sky[ok]
    mf_snr_g = mf_snr[idx][srt][ok]
    cspread_g = cspread[srt][ok]
    peak_g = peak_dn[idx][srt][ok]
    fwhm_g = fwhm0[srt][ok]
    flux_r4 = apm_c["flux"][srt][ok]

    # --- 污染/缺陷剔除：r<=14 内有负像元 或 盒子通量与 A(10) 不符 ---
    H = c["psf_half_px"]
    iy, ix, valid, d = L._win_index(ysg, xsg, H, dn.shape[0], dn.shape[1])
    raw = dn[iy, ix].astype(np.float64)
    negmask = (raw < 0) & valid
    box = np.where(valid, raw - sky[:, None], 0.0)
    bsum = box.sum(axis=1)
    neg_in_box = int(negmask.any(axis=1).sum())
    a10 = A[:, 9]
    # 盒子通量与 A(10) 的允许差：纯噪声项 + 3% 系统项（盒子角落 vs 圆形孔径、天光估计）
    tol = (c["psf_box_sigma_mult"] * math.sqrt(441 + 78) * p0["sigma_pix_adu"] + c["psf_box_systematic_frac"] * a10) / np.maximum(a10, 1e-9)
    ratio_box = bsum / np.maximum(a10, 1e-9)
    contam = np.abs(ratio_box - 1.0)
    clean = (contam < tol) & (~negmask.any(axis=1))
    print("[clean] neg:%d  box-ratio pct:%s  contamin:%d  remain:%d" % (
        neg_in_box, {q: round(float(np.percentile(ratio_box, q)), 3) for q in [5, 50, 95]},
        int((contam >= tol).sum()), int(clean.sum())))

    tot_ratio = finf_fit / finf_plat
    rs = np.arange(c["extrap_r"][0], c["extrap_r"][1] + 1, dtype=float)
    fit_resid = A[:, c["extrap_r"][0] - 1: c["extrap_r"][1]] - (finf_fit[:, None] - bcoef[:, None] / rs[None, :])

    # --- 总通量不确定度：孔径噪声 + 天光估计误差随孔径面积的放大 ---
    def n_pix_ann(d_in, d_out):
        n = 0
        for iy_ in range(-int(math.ceil(d_out)), int(math.ceil(d_out)) + 1):
            for ix_ in range(-int(math.ceil(d_out)), int(math.ceil(d_out)) + 1):
                dd = math.hypot(iy_, ix_)
                if d_in <= dd <= d_out:
                    n += 1
        return n

    n_ann_sky = n_pix_ann(c["sky_in"], c["sky_out"])
    n14 = n_pix_ann(0.0, float(c["r_max"]))
    sig_sky = 1.2533141 * p0["sigma_pix_adu"] / math.sqrt(max(n_ann_sky, 1))
    sig_ap14 = c["sigma_ap_r_r_gt_8_adu"]
    sig_F = math.sqrt(sig_ap14 ** 2 + (n14 * sig_sky) ** 2)
    frac_err_F = sig_F / np.maximum(finf_plat, 1e-9)
    measurable = np.isfinite(frac_err_F) & (frac_err_F < c["ac_measurable_frac_err_max"]) & (finf_plat > 0)

    AC_plat = finf_plat[:, None] / np.where(A == 0, np.nan, A)
    AC_fit = finf_fit[:, None] / np.where(A == 0, np.nan, A)
    ac4 = AC_plat[:, 3]
    ac4_fit = AC_fit[:, 3]
    print("[ac] n_ann_sky=%d n14=%d sig_sky=%.3f sig_F=%.1f ADU | measurable:%d" % (
        n_ann_sky, n14, sig_sky, sig_F, int(measurable.sum())))

    # AC(4) 随亮度的诊断表（说明为什么必须限制样本）
    lf = np.log10(np.maximum(finf_plat, 1.0))
    qs = np.quantile(lf, np.linspace(0, 1, 7))
    trend_rows = []
    for i in range(6):
        m = (lf >= qs[i]) & (lf <= qs[i + 1] if i == 5 else lf < qs[i + 1])
        if m.sum() < 5:
            continue
        trend_rows.append({"log10_Ftotal_range": [float(qs[i]), float(qs[i + 1])],
                           "F_total_adu_range": [10 ** float(qs[i]), 10 ** float(qs[i + 1])],
                           "n": int(m.sum()), "n_measurable": int((m & measurable).sum()),
                           "ac4_median": float(np.nanmedian(ac4[m])),
                           "ac4_mad_sigma": L.mad_sigma(ac4[m]),
                           "frac_err_F_median": float(np.median(frac_err_F[m]))})

    # ---------------- 经验 PSF ----------------
    os_ = c["psf_oversample"]
    pidx = np.flatnonzero(clean & measurable)
    pidx = pidx[np.argsort(-finf_plat[pidx])][: c["max_stars_psf"]]
    print("[psf] candidates clean&measurable:", int((clean & measurable).sum()), "->", pidx.size)
    P1, Pos, bsum_p, npsf, off_y, off_x = build_psf(fr, ysg[pidx], xsg[pidx], sky[pidx], finf_fit[pidx], H, os_, c["psf_moment_rmax"])
    G = Pos.shape[0]
    cen = (G - 1) // 2
    mom = L.psf_moments(Pos, os_, c["psf_moment_rmax"])
    prof, ptot = L.psf_radial(Pos, os_, r_max=9.0, step=0.2)
    ee = L.encircled(Pos, os_, [1, 2, 3, 4, 5, 6, 8, 10])
    ee4 = [e["ee"] for e in ee if e["r_px"] == 4][0]
    ee_vals = [e["ee"] for e in ee]
    ee_monotone = all(ee_vals[i + 1] >= ee_vals[i] - 1e-4 for i in range(len(ee_vals) - 1))
    kern_min_pre = None
    print("[psf] stars=%d  FWHM=%.3f  ell=%.3f  EE(4)=%.4f  sum(Pos)=%.4f" % (
        npsf, mom["fwhm_px"], mom["ellipticity"], ee4, Pos.sum()))

    # 半高法 FWHM（径向 profile）
    pk = max(r["p_r_per_px2"] for r in prof)
    fwhm_hm = None
    for i in range(1, len(prof)):
        if prof[i]["p_r_per_px2"] <= 0.5 * pk < prof[i - 1]["p_r_per_px2"]:
            r1, p1 = prof[i - 1]["r_px"], prof[i - 1]["p_r_per_px2"]
            r2, p2 = prof[i]["r_px"], prof[i]["p_r_per_px2"]
            fwhm_hm = 2.0 * (r1 + (p1 - 0.5 * pk) / (p1 - p2) * (r2 - r1))
            break

    # 1px 间距数值表（注入用）+ 象素化 kernel
    ktab = 8
    tab = Pos[cen - ktab * os_: cen + ktab * os_ + 1: os_, cen - ktab * os_: cen + ktab * os_ + 1: os_]
    tab = tab / tab.sum()
    kern = np.zeros((2 * ktab + 1, 2 * ktab + 1))
    for i, dy in enumerate(range(-ktab, ktab + 1)):
        for j, dx in enumerate(range(-ktab, ktab + 1)):
            kern[i, j] = L.sample_psf(Pos, os_, [float(dy)], [float(dx)])[0]
    kern_sum = float(kern.sum())
    kern = kern / kern_sum

    # 4x4 网格 FWHM 与 AC(4)
    gny, gnx = c["fwhm_grid"]
    grid_rows = []
    ny, nx = dn.shape
    for gy in range(gny):
        for gx in range(gnx):
            y0, y1 = gy * ny / gny, (gy + 1) * ny / gny
            x0, x1 = gx * nx / gnx, (gx + 1) * nx / gnx
            m = (ysg[pidx] >= y0) & (ysg[pidx] < y1) & (xsg[pidx] >= x0) & (xsg[pidx] < x1)
            mm = (ysg >= y0) & (ysg < y1) & (xsg >= x0) & (xsg < x1) & clean & measurable
            row = {"cell": [gy, gx], "n_psf": int(m.sum()), "n_ac4": int(mm.sum()),
                   "y_range_px": [y0, y1], "x_range_px": [x0, x1],
                   "ac4_median": float(np.median(ac4[mm])) if mm.sum() >= 10 else None}
            if m.sum() >= 30:
                _, Pc, _, _, _, _ = build_psf(fr, ysg[pidx][m], xsg[pidx][m], sky[pidx][m], finf_fit[pidx][m], H, os_, c["psf_moment_rmax"])
                mc = L.psf_moments(Pc, os_, c["psf_moment_rmax"])
                eec = L.encircled(Pc, os_, [4])[0]["ee"]
                row.update({"fwhm_px": mc["fwhm_px"], "ellipticity": mc["ellipticity"], "ee4": eec})
            else:
                row.update({"fwhm_px": None, "ellipticity": None, "ee4": None,
                            "note": "n<30，不足以叠 PSF"})
            grid_rows.append(row)
    fwhms = [r["fwhm_px"] for r in grid_rows if r["fwhm_px"]]
    acs = [r["ac4_median"] for r in grid_rows if r["ac4_median"]]
    fwhm_spread = (max(fwhms) - min(fwhms)) if len(fwhms) >= 2 else None
    ac_spread_mag = (mag(max(acs)) - mag(min(acs))) if len(acs) >= 2 else None

    # --- AC(4) 分箱 ---
    def bin_summary(vals, key_vals, n_bins=None, labels=None, edges=None):
        if edges is not None:
            k = np.clip(np.searchsorted(edges, key_vals, side="right") - 1, 0, len(edges) - 2)
        else:
            q = np.unique(np.quantile(key_vals, np.linspace(0, 1, n_bins + 1)))
            edges = q
            k = np.clip(np.searchsorted(q, key_vals, side="right") - 1, 0, len(q) - 2)
        rows = []
        for i in range(len(edges) - 1):
            m = (k == i) & np.isfinite(vals)
            if m.sum() < 5:
                continue
            v = vals[m]
            med = float(np.median(v))
            sig = L.mad_sigma(v)
            rows.append({"bin": (labels[i] if labels else i),
                         "key_range": [float(edges[i]), float(edges[i + 1])],
                         "n": int(m.sum()), "ac4_median": med, "ac4_mad_sigma": sig,
                         "ac4_se": sig / math.sqrt(m.sum()), "corr_mag_r4": mag(med)})
        return rows

    med_ac4 = float(np.nanmedian(ac4))
    sig_ac4 = L.mad_sigma(ac4)
    sel_ac = clean & measurable & np.isfinite(ac4)
    n_ac = int(sel_ac.sum())
    med_ac4_clean = float(np.median(ac4[sel_ac]))
    sig_ac4_clean = L.mad_sigma(ac4[sel_ac])
    se_ac4 = sig_ac4_clean / math.sqrt(max(n_ac, 1))
    bright_rows = bin_summary(ac4[sel_ac], finf_plat[sel_ac], n_bins=c["n_bright_bins"],
                              labels=["faint_third", "mid_third", "bright_third"])
    quad = (ysg >= ny / 2).astype(int) * 2 + (xsg >= nx / 2).astype(int)
    quad_rows = []
    for q in range(4):
        m = (quad == q) & sel_ac & np.isfinite(ac4)
        if m.sum() < 5:
            continue
        v = ac4[m]
        quad_rows.append({"quadrant": ["SW", "SE", "NW", "NE"][q], "n": int(m.sum()),
                          "ac4_median": float(np.median(v)), "ac4_mad_sigma": L.mad_sigma(v),
                          "ac4_se": L.mad_sigma(v) / math.sqrt(m.sum()), "corr_mag_r4": mag(float(np.median(v)))})

    def spread(rows, key):
        m = [r[key] for r in rows if r.get(key) is not None]
        return (max(m) - min(m)) if len(m) >= 2 else None

    crit = 0.05
    bright_spread = spread(bright_rows, "corr_mag_r4")
    quad_spread = spread(quad_rows, "corr_mag_r4")

    ac_rows = []
    for r in range(1, c["r_max"] + 1):
        ac = AC_plat[sel_ac, r - 1]
        ac_all = AC_plat[:, r - 1]
        o = np.arange(-r, r + 1)
        nap = int((o[:, None] ** 2 + o[None, :] ** 2 <= r * r).sum())
        a_med = float(np.nanmedian(A[sel_ac, r - 1]))
        f_med = float(np.nanmedian(finf_plat[sel_ac]))
        ee_r = ([e["ee"] for e in ee if e["r_px"] == r][0] if any(e["r_px"] == r for e in ee) else None)
        ac_rows.append({"r_px": r, "n_ap_geo": nap, "n_stars": int(sel_ac.sum()),
                        "A_median_adu": a_med, "F_total_median_adu": f_med,
                        "A_over_F_median": a_med / f_med,
                        "ac_median": float(np.nanmedian(ac)), "ac_mad_sigma": L.mad_sigma(ac),
                        "corr_mag": mag(float(np.nanmedian(ac))),
                        "ac_median_full_sample_all": float(np.nanmedian(ac_all)),
                        "psf_ee_in_r": ee_r,
                        "delta_vs_psf_ee_mag": (mag(float(np.nanmedian(ac))) - mag(1.0 / ee_r)) if ee_r else None})

    prov["utc_end"] = L.utc_now()
    res = {
        "section": "E4",
        "declaration": "本文件只给出第 1 帧的仪器级通量增长曲线与孔径修正算术；不声明任何星等、零点、Gaia 标定、极限星等或完备深度。曝光 1.5 s 的解读来自 FITS EXPOSURE 卡片注释（前批已记录为待确认）。",
        "sample": {
            "n_selected": int(srt.size), "n_used_growth": int(A.shape[0]),
            "n_clean_for_psf_and_bins": int(clean.sum()),
            "criteria": {
                "bg_limited_snr_min": c["ap_snr_min"],
                "snr_definition": "flux(r=4 几何孔径, 环带 6-8px 中位天光) / sigma_ap_r4_frozen = %.6f ADU" % sigma_ap,
                "snr_caveat": "这是背景极限 SNR，未含源泊松项（GAIN 卡片缺失，UNKNOWN）。真实总 SNR 只会更低。",
                "peak_max_dn": c["peak_max_dn"],
                "peak_caveat": "SATURATE 卡片缺失，20000 DN 是保守经验上限，真实满量程 UNKNOWN",
                "isolation_px": c["isolation_px_growth"],
                "isolation_caveat": "比题面 3xFWHM(=10.2px) 更严，因为增长曲线要测到 r=14",
                "fwhm_accept_px": cfg["detect"]["fwhm_accept_px"], "ellip_max": cfg["detect"]["ellip_max"],
                "contamination_cut": "r<=14 窗口内无负像元，且 |21x21 盒子通量/A(10) - 1| < (%.1f*sqrt(519)*sigma_pix + %.2f)/A(10)" % (c["psf_box_sigma_mult"], c["psf_box_systematic_frac"]),
            },
            "n_peaks_above_8sigma": int(len(yx)),
            "n_stars_with_negative_pixel_in_box": neg_in_box,
            "n_contaminated_cut": int((contam >= tol).sum()),
            "box_over_a10_percentiles": {str(q): float(np.percentile(ratio_box, q)) for q in [1, 5, 50, 95, 99]},
            "flux_r4_adu_percentiles": {str(q): float(np.percentile(flux_r4, q)) for q in [0, 16, 50, 84, 100]},
            "peak_dn_max": float(peak_g.max()),
            "centroid_window_spread_px_median": float(np.median(cspread_g)),
            "centroid_window_spread_note": "r=3 与 r=4 两种质心窗口的差值中位数（质心精度上界，不是误差条）",
            "per_star_fwhm_7x7_median_px": float(np.nanmedian(fwhm_g)),
            "per_star_fwhm_7x7_note": "7x7 窗口的二阶矩估计，截断导致系统性偏小，只用于形态筛",
        },
        "growth_curve": {"r_max_px": c["r_max"], "sky_annulus_px": [c["sky_in"], c["sky_out"]], "rows": ac_rows},
        "total_flux": {
            "plateau_r": c["plateau_r"], "extrap_r": c["extrap_r"],
            "extrap_model": "A(r) = F_total - b/r 对 1/r 做最小二乘",
            "finf_fit_median_adu": float(np.nanmedian(finf_fit)),
            "finf_plateau_median_adu": float(np.nanmedian(finf_plat)),
            "ratio_fit_over_plateau_median": float(np.nanmedian(tot_ratio)),
            "ratio_fit_over_plateau_mad_sigma": L.mad_sigma(tot_ratio),
            "b_median_adu_px": float(np.nanmedian(bcoef)),
            "fit_resid_rms_adu": float(np.sqrt(np.nanmean(fit_resid ** 2))),
            "fit_resid_rms_frac": float(np.sqrt(np.nanmean(fit_resid ** 2)) / np.nanmedian(finf_fit)),
        },
        "aperture_correction": {
            "primary_total_flux": "A(r) 在 r=12..14 的平台均值（直接测量，不外推）",
            "n_stars_used": n_ac,
            "ac4_median": med_ac4_clean, "ac4_mad_sigma": sig_ac4_clean, "ac4_se": se_ac4,
            "corr_mag_r4_median": mag(med_ac4_clean),
            "corr_mag_r4_se": (L.LINEAR_MAG * se_ac4 / med_ac4_clean),
            "ac4_median_full_sample_all_6074": med_ac4,
            "ac4_median_full_sample_note": "全样本中位数：最暗端的星被天光误差与比值噪声向上偏置，不可用作修正量",
            "corr_mag_r4_from_1overR_extrap": mag(float(np.median(ac4_fit[sel_ac]))),
            "ac4_from_1overR_extrap": float(np.median(ac4_fit[sel_ac])),
            "definition": "AC(r) = F_total / A(r)，修正量 = 2.5*log10(AC(r))",
            "error_model": {
                "sigma_sky_annulus_adu": sig_sky, "n_annulus_pix": n_ann_sky,
                "sigma_ap_r14_adu": sig_ap14, "n_ap_r14_pix": n14,
                "sigma_Ftotal_adu": sig_F,
                "note": "天光中位数的误差乘上 r=14 孔径面积 (n14) 会成为总通量的主导系统项；这是限制样本的根本原因",
                "select_frac_err_max": c["ac_measurable_frac_err_max"],
            },
        },
        "ac4_dependence": {
            "brightness_bins": bright_rows, "position_quadrants": quad_rows,
            "ac4_vs_flux_trend_diagnostic": trend_rows,
            "position_4x4_grid": grid_rows,
            "verdict": {
                "threshold_mag": crit, "brightness_spread_mag": bright_spread,
                "position_quadrant_spread_mag": quad_spread, "position_4x4_spread_mag": ac_spread_mag,
                "scalar_correction_sufficient": bool(
                    (bright_spread is not None and bright_spread <= crit)
                    and (quad_spread is not None and quad_spread <= crit)),
                "rule": "若 AC(4) 的系统差异（空间或亮度）> 0.05 mag，则该修正必须做成位置/亮度相关项，不能只用一个标量。",
                "caveat": "差异里含样本噪声（各箱 n 只有几百），判据只比中位修正量的峰峰值。",
                "why_faint_stars_excluded": "天光中位数误差 x 613 像元 ≈ %.0f ADU 的总通量误差；暗星的增长曲线在大 r 处被它主导" % (n14 * sig_sky),
            },
        },
        "sky_sensitivity": {},
        "provenance": prov,
    }

    # 天光环带敏感性
    A2, sky2 = L.growth_curve(fr.dn, ysg[pidx], xsg[pidx], c["r_max"], c["analytic_sky_in"], c["analytic_sky_out"])
    f2, _, _ = L.growth_total(A2, c["plateau_r"], c["extrap_r"])
    ac4_alt = f2 / A2[:, 3]
    res["sky_sensitivity"] = {
        "note": "天光环带 17-21px 换成 14.5-21px 后 AC(4) 的中位变化（同一批 PSF 星）",
        "ac4_median_primary": float(np.nanmedian(ac4[pidx])),
        "ac4_median_alt_sky": float(np.nanmedian(ac4_alt)),
        "delta_mag": (mag(float(np.nanmedian(ac4_alt))) - mag(float(np.nanmedian(ac4[pidx])))),
        "sky_offset_median_adu": float(np.median(sky2 - sky[pidx])),
        "n_stars": int(pidx.size),
        "interpretation": "这一项给出'天光口径选择'带来的系统不确定度，量级应远小于位置/亮度的判据阈值才有意义",
    }

    prof_res = {
        "section": "EPF",
        "declaration": "经验 PSF（第 1 帧孤立星叠加，Fourier 相位斜坡对齐 + 带限零填充上采样）。是图像域经验剖面，不是光学设计值；不声明 PSF 在全帧恒定。",
        "construction": {
            "n_stars": int(npsf), "half_window_px": H, "oversample": os_,
            "grid_cells": int(G), "cell_px": 1.0 / os_,
            "alignment": "每颗星用 Fourier 相位斜坡按其质心小数部分精确平移对齐（带限，无插值平滑）",
            "normalization": "每颗星扣 17-21px 环带天光，再除以该星 21x21 盒子通量；叠合后再归一到 sum=1",
            "upsample": "带限零填充（频域补零），格距 0.2 px",
            "recenter": "叠合后用一阶矩（r<=6px）测残差中心，再 Fourier 平移到格点中心",
            "recenter_offset_px": [off_y, off_x],
            "recenter_note": "非零偏移反映单星质心估计的系统偏置；它只平移叠合剖面，不展宽",
            "contamination_cut": "见 e4_growth_curve.json sample.criteria.contamination_cut",
            "n_stars_in_field_total_used": int(clean.sum()),
        },
        "moments": mom, "moment_rmax_px": c["psf_moment_rmax"],
        "moment_rmax_note": "二阶矩对边缘台基极敏感，必须截断积分半径；截断本身又使 FWHM 略偏小",
        "fwhm_half_max_px": fwhm_hm,
        "ellipticity": mom["ellipticity"],
        "encircled_energy": ee,
        "ee_monotone_ok": bool(ee_monotone),
        "ee_monotone_note": "包围能量必须随半径单调不减；若不是，说明叠合剖面有负值/振铃，不可用于注入",
        "kernel_min_value": float(kern.min()),
        "kernel_negative_frac": float((kern < 0).sum() / kern.size),
        "radial_profile": prof,
        "radial_profile_units": "p_r_per_px2 = 每 px^2 的归一化通量；frac_of_total = 该环通量占比",
        "psf_1px_grid_17x17": tab.tolist(),
        "psf_1px_grid_note": "1px 间距抽样并归一到 sum=1，仅示意",
        "psf_1px_stack_21x21": P1.tolist(),
        "psf_1px_stack_note": "1px 间距、天体在中心像元的叠合剖面（未重采样）。E3/E7 用带限相位移在此表上取任意亚像素相位",
        "psf_pixel_kernel_17x17": kern.tolist(),
        "psf_pixel_kernel_note": "在各整数像素偏移处采样连续剖面得到的像素化 kernel，已归一到 sum=1；E3/E7 注入用它",
        "psf_pixel_kernel_raw_sum": kern_sum,
        "fwhm_vs_position_4x4": grid_rows,
        "fwhm_position_spread_px": fwhm_spread,
        "ac4_position_4x4_spread_mag": ac_spread_mag,
        "consistency": {
            "psf_ee_r4": ee4, "psf_implied_ac4": 1.0 / ee4,
            "growth_implied_ac4_clean": med_ac4_clean,
            "delta_mag": (mag(1.0 / ee4) - mag(med_ac4_clean)),
            "psf_centroid_offset_after_recenter_px": [mom["centroid_offset_y_px"], mom["centroid_offset_x_px"]],
            "growth_delta_vs_psf_by_radius_mag": [(row["r_px"], row["delta_vs_psf_ee_mag"]) for row in ac_rows],
            "note": "两条独立路径（PSF 包围能量 vs 逐星增长曲线）的 AC(4) 应只差几个百分点",
        },
        "unknowns": [
            "PSF 是否随视场系统变化：4x4 的 FWHM 差异只能给量级，不能区分光学像差与导星抖动",
            "SATURATE/GAIN 缺失：peak<20000 DN 是保守经验上限",
            "像素积分效应：叠合出的剖面本身已含矩形像元积分，注入时只在像素中心点采样，两者相差 ≲2%",
        ],
        "provenance": prov,
    }

    L.dump_json(outdir / "e4_growth_curve.json", res)
    L.dump_json(outdir / "epf_profile.json", prof_res)
    npy = outdir / "epf_profile_oversampled.npy"
    if npy.exists():
        raise SystemExit("refuse to overwrite %s" % npy)
    np.save(npy, Pos)
    np.save(outdir / "epf_profile_pixel_kernel.npy", kern)
    np.save(outdir / "epf_profile_p1_21x21.npy", P1)

    print("\n[E4] n=%d clean=%d measurable=%d ac4sample=%d | F_total(plateau)=%.1f ADU | AC(4)=%.4f (%.4f mag)" % (
        A.shape[0], int(clean.sum()), int(measurable.sum()), n_ac,
        np.nanmedian(finf_plat), med_ac4_clean, mag(med_ac4_clean) or 0))
    print("[E4] AC(4) from 1/r extrap=%.4f (%.4f mag) | sig_F=%.1f ADU" % (
        float(np.median(ac4_fit[sel_ac])), mag(float(np.median(ac4_fit[sel_ac]))) or 0, sig_F))
    print("[E4] brightness spread %s mag | quadrant spread %s mag | 4x4 spread %s mag | scalar ok=%s" % (
        bright_spread, quad_spread, ac_spread_mag, res["ac4_dependence"]["verdict"]["scalar_correction_sufficient"]))
    print("[PSF] FWHM=%.3f (half-max %s) ell=%.3f EE4=%.4f -> AC4=%.4f" % (
        mom["fwhm_px"], (round(fwhm_hm, 3) if fwhm_hm else None), mom["ellipticity"], ee4, 1.0 / ee4))
    print("[PSF] 4x4 FWHM spread: %s px | EE monotone=%s | kernel min=%.5f neg=%.3f" % (
        fwhm_spread, ee_monotone, float(kern.min()), float((kern < 0).mean())))
    print("[done] outdir", outdir)


if __name__ == "__main__":
    main()
