#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xcheck_phase1b_e3.py -- E3 孔径几何与子像素相位（两条独立证据）。

证据 1（真实星）：按质心小数部分分 10 个相位桶，比较同一 r=4 几何孔径的三种圆心/权重：
  a = 整数峰为心（几何）
  b = 质心为心（几何）
  c = 质心为心（分数像元权重 w=clip(4.5-d,0,1)）
  d = 质心为心（分数权重），但天光固定取整数峰环带（解耦天空-质心耦合）
证据 2（注入）：把 E4 的经验 PSF 以 5x5=25 个亚像素相位注入无源天空区，走同一测量。

判据：相位间通量散布 <= 2x 噪声预期 -> 孔径几何可接受；否则必须改"质心 + 分数像元权重"。
"""
import json
import math
from pathlib import Path

import numpy as np

import xcheck_phase1b_lib as L


def inject_at(img, y0i, x0i, kern):
    h = kern.shape[0] // 2
    img[y0i - h:y0i + h + 1, x0i - h:x0i + h + 1] += kern
    return img


def bucket_table(vals, frac, nbins):
    edges = np.linspace(0.0, 1.0, nbins + 1)
    k = np.clip(np.searchsorted(edges, frac, side="right") - 1, 0, nbins - 1)
    rows = []
    for i in range(nbins):
        m = k == i
        if m.sum() < 3:
            rows.append({"bucket": i, "phase_range": [float(edges[i]), float(edges[i + 1])],
                         "n": int(m.sum()), "median_pct": None})
            continue
        v = vals[m]
        med = float(np.median(v))
        mad = L.mad_sigma(v)
        rows.append({"bucket": i, "phase_range": [float(edges[i]), float(edges[i + 1])],
                     "phase_center": float(0.5 * (edges[i] + edges[i + 1])),
                     "n": int(m.sum()), "median_pct": 100.0 * med,
                     "mad_sigma_pct": 100.0 * mad, "se_pct": 100.0 * mad / math.sqrt(m.sum())})
    return rows


def spread_of(rows):
    m = [r["median_pct"] for r in rows if r.get("median_pct") is not None]
    return (max(m) - min(m)) if len(m) >= 2 else None


def main():
    cfg, cfg_sha = L.load_config()
    runid = L.make_runid(cfg_sha)
    outdir = L.ROOT / cfg["output_root_relative"] / runid
    outdir.mkdir(parents=True, exist_ok=True)
    c = cfg["e3"]
    apc = cfg["aperture"]
    p0 = cfg["prior_results"]
    sigma_ap = p0["sigma_ap_r4_adu_from_e1"]
    sigma_pix = p0["sigma_pix_adu"]

    epf = json.loads((outdir / "epf_profile.json").read_text(encoding="utf-8"))
    P1 = np.asarray(epf["psf_1px_stack_21x21"], dtype=np.float64)
    P1 = P1 / P1.sum()

    fits = L.ROOT / cfg["input"]["fits_relative"] / cfg["input"]["reference_fits"]
    prov = L.provenance(Path(__file__).resolve(), cfg_sha, runid)
    prov["input_fits_path"] = str(fits)
    prov["input_fits_sha256"] = L.sha256_file(fits)
    prov["psf_sha256"] = L.sha256_json(P1.tolist())
    prov["epf_profile_sha256"] = L.sha256_file(outdir / "epf_profile.json")
    print("[prov] script", prov["script_sha256"], "runid", runid)

    dn, hdr = L.load_frame(fits)
    fr = L.Frame(dn, cfg)
    det = fr.detect(keep_morph=False)
    yx = det["yx"]
    peak_dn = fr.peak_dn(yx)
    apm = L.aperture_measure(fr.dn, yx[:, 0].astype(float), yx[:, 1].astype(float),
                             apc["r_ap"], apc["r_in"], apc["r_out"])
    snr_bg = apm["flux"] / sigma_ap
    iso = L.nearest_neighbor_px(yx.astype(float), c["isolation_px"])
    sel = (snr_bg >= c["ap_snr_min"]) & (peak_dn < c["peak_max_dn"]) & (iso >= c["isolation_px"])
    idx = np.flatnonzero(sel)
    fwhm0, ell0 = L.moments(fr.img, yx[idx], cfg["detect"]["morphology_window"] // 2)
    lo, hi = cfg["detect"]["fwhm_accept_px"]
    m_ok = np.isfinite(fwhm0) & (fwhm0 >= lo) & (fwhm0 <= hi) & np.isfinite(ell0) & (ell0 <= cfg["detect"]["ellip_max"])
    idx = idx[m_ok]
    print("[sample] snr_bg>%.0f iso>=%.1fpx morph-ok: %d" % (c["ap_snr_min"], c["isolation_px"], idx.size))

    ys0 = yx[idx, 0].astype(float)
    xs0 = yx[idx, 1].astype(float)
    cy, cx = L.centroid(fr.dn, ys0, xs0, r=3.0)
    iy, ix, valid, dd = L._win_index(cy, cx, int(apc["r_out"]) + 2, dn.shape[0], dn.shape[1])
    bad = ((dn[iy, ix] < 0) & valid).any(axis=1)
    keep = ~bad & np.isfinite(cy) & np.isfinite(cx)
    cy, cx, ys0, xs0 = cy[keep], cx[keep], ys0[keep], xs0[keep]
    print("[sample] after negative-pixel cut: %d (cut %d)" % (cy.size, int(bad.sum())))

    fa = L.aperture_measure(fr.dn, ys0, xs0, apc["r_ap"], apc["r_in"], apc["r_out"])
    fb = L.aperture_measure(fr.dn, cy, cx, apc["r_ap"], apc["r_in"], apc["r_out"])
    fc = L.aperture_measure(fr.dn, cy, cx, apc["r_ap"], apc["r_in"], apc["r_out"], frac=True)
    okf = (np.isfinite(fa["flux"]) & np.isfinite(fb["flux"]) & np.isfinite(fc["flux"])
           & (fa["flux"] > 0) & (fb["flux"] > 0) & (fc["flux"] > 0))
    fref = fb["flux"][okf]
    d_ab = (fb["flux"][okf] - fa["flux"][okf]) / fref
    d_bc = (fc["flux"][okf] - fb["flux"][okf]) / fref
    d_ac = (fc["flux"][okf] - fa["flux"][okf]) / fref
    n_ap_a = fa["n_ap"][okf].astype(int)
    n_ap_b = fb["n_ap"][okf].astype(int)
    fracy = (cy - np.floor(cy))[okf]
    fracx = (cx - np.floor(cx))[okf]
    noise_frac = sigma_ap / float(np.median(fref))
    print("[real] n=%d median flux %.1f ADU | noise/star %.3f%% | n_ap a%s b%s" % (
        okf.sum(), float(np.median(fref)), 100 * noise_frac,
        sorted(set(n_ap_a.tolist())), sorted(set(n_ap_b.tolist()))[:8]))

    nbins = c["n_phase_buckets"]
    b_ab = bucket_table(d_ab, fracx, nbins)
    b_ab_y = bucket_table(d_ab, fracy, nbins)
    b_bc = bucket_table(d_bc, fracx, nbins)
    b_ac = bucket_table(d_ac, fracx, nbins)
    spreads = {"delta_ab_fracx": spread_of(b_ab), "delta_ab_fracy": spread_of(b_ab_y),
               "delta_bc_fracx": spread_of(b_bc), "delta_ac_fracx": spread_of(b_ac)}
    se_level = float(np.median([r["se_pct"] for r in b_bc if r.get("se_pct") is not None]))
    print("[real] spreads %% %s | bucket SE %.3f%%" % (
        {k: (round(v, 3) if v is not None else None) for k, v in spreads.items()}, se_level))

    # ---------------- 证据 2：注入 ----------------
    ny, nx = dn.shape
    srcmask = np.zeros((ny, nx), dtype=bool)
    srcmask[yx[:, 0].astype(int), yx[:, 1].astype(int)] = True
    srcmask = L.dilate_bool(srcmask, int(c["inject_clean_dilate_px"]))
    rng = np.random.default_rng(c["inject_seed"])
    cand_y = rng.integers(200, ny - 200, size=40000)
    cand_x = rng.integers(200, nx - 200, size=40000)
    cand = ~srcmask[cand_y, cand_x]
    pos = []
    for y, x in zip(cand_y[cand], cand_x[cand]):
        if dn[y - 20:y + 21, x - 20:x + 21].min() < 0:
            continue
        pos.append((int(y), int(x)))
        if len(pos) >= c["n_inject_positions"]:
            break
    pos = np.asarray(pos, dtype=int)
    print("[inject] clean positions: %d" % len(pos))

    phases = np.array(c["inject_phase_grid"], dtype=float)
    half = 8
    kern_at = {}
    kern_sums = {}
    for py in phases:
        for px in phases:
            k = L.psf_kernel_at_phase(P1, float(py), float(px), half=half)
            ksum = float(k.sum())
            # 归一化：P1 非严格带限，格点和随相位有 ~0.8% 抖动，会对注入注入伪相位依赖
            kern_at[(float(py), float(px))] = k / ksum
            kern_sums["%.1f_%.1f" % (py, px)] = ksum
    n_ann = 79
    sig_sky_ann = 1.2533141 * sigma_pix / math.sqrt(n_ann)
    sig_meas = math.sqrt(sigma_ap ** 2 + (49 * sig_sky_ann) ** 2)
    print("[inject] kernel sums: min %.5f max %.5f | sigma_meas=%.1f ADU (sky-term dominated)" % (
        min(kern_sums.values()), max(kern_sums.values()), sig_meas))

    inj = {}
    for flux_inj in c["inject_flux_grid_adu"]:
        rows = []
        for (py, px) in kern_at:
            k = kern_at[(py, px)] * flux_inj
            ratios = []
            cbis = []
            for (y0i, x0i) in pos:
                patch = fr.dn[y0i - 30:y0i + 31, x0i - 30:x0i + 31].astype(np.float64).copy()
                inject_at(patch, 30, 30, k)
                sub = patch[28:33, 28:33]
                dmax = np.unravel_index(int(np.argmax(sub)), sub.shape)
                ipy, ipx = 28 + dmax[0], 28 + dmax[1]
                ca = L.aperture_measure(patch, np.array([float(ipy)]), np.array([float(ipx)]),
                                        apc["r_ap"], apc["r_in"], apc["r_out"])
                ccy, ccx = L.centroid(patch, np.array([float(ipy)]), np.array([float(ipx)]), r=3.0)
                cb = L.aperture_measure(patch, ccy, ccx, apc["r_ap"], apc["r_in"], apc["r_out"])
                cc = L.aperture_measure(patch, ccy, ccx, apc["r_ap"], apc["r_in"], apc["r_out"], frac=True)
                # d: 质心 + 分数像元权重，但天光固定取整数峰环带（解耦"天空跟着质心走"的耦合）
                fd = cc["flux"][0] + cc["n_ap_float"][0] * (cc["sky"][0] - ca["sky"][0])
                cbis.append((ccy[0] - (30.0 + py), ccx[0] - (30.0 + px)))
                ratios.append((ca["flux"][0] / flux_inj, cb["flux"][0] / flux_inj,
                               cc["flux"][0] / flux_inj, fd / flux_inj))
            r = np.asarray(ratios)
            cb_ = np.asarray(cbis)
            rows.append({"phase": [py, px], "n": int(r.shape[0]),
                         "ratio_a": float(np.median(r[:, 0])), "ratio_b": float(np.median(r[:, 1])),
                         "ratio_c": float(np.median(r[:, 2])), "ratio_d": float(np.median(r[:, 3])),
                         "centroid_bias_y_px": float(np.mean(cb_[:, 0])),
                         "centroid_bias_x_px": float(np.mean(cb_[:, 1])),
                         "se_a_pct": 100.0 * L.mad_sigma(r[:, 0]) / math.sqrt(r.shape[0])})
        out = {"phases": rows}
        for key in ("a", "b", "c", "d"):
            v = np.array([x["ratio_" + key] for x in rows])
            out["p2p_" + key + "_pct"] = float(100.0 * (v.max() - v.min()))
            out["mean_" + key] = float(v.mean())
            out["rms_" + key + "_pct"] = float(100.0 * np.std(v, ddof=1))
        inj["%g" % flux_inj] = out
        cbias = np.array([[x["centroid_bias_y_px"], x["centroid_bias_x_px"]] for x in rows])
        out["centroid_bias_p2p_px"] = [float(cbias[:, 0].ptp()), float(cbias[:, 1].ptp())]
        print("[inject] F=%6g: p2p a=%.2f%% b=%.2f%% c=%.2f%% d=%.2f%% | mean a=%.4f | se_a~%.2f%% | ctr bias p2p %.3f/%.3f px" % (
            flux_inj, out["p2p_a_pct"], out["p2p_b_pct"], out["p2p_c_pct"], out["p2p_d_pct"],
            out["mean_a"], float(np.median([x["se_a_pct"] for x in rows])),
            cbias[:, 0].ptp(), cbias[:, 1].ptp()))

    verdict = {
        "criterion": "相位间通量散布 (peak-to-peak, %) <= 2 x 噪声预期 (%) -> 孔径几何可接受",
        "noise_definition": {
            "sigma_ap_r4_adu": sigma_ap,
            "note": "sigma_ap_r4 是前批 E1 在 2000 个无源位置实测的 r=4 孔径和 RMS（已含 kappa 膨胀）。真实的光子噪声泊松项 UNKNOWN：GAIN 缺失。",
            "real_star_per_measurement_pct": 100.0 * noise_frac,
            "real_star_criterion_pct": 200.0 * noise_frac,
            "injection_sigma_meas_adu": sig_meas,
            "injection_sigma_meas_note": "sky-term dominated: 49 * (1.25*sigma_pix/sqrt(79)) ；这正是暗注入无法测几何效应的原因",
        },
        "real_star": {
            "n": int(okf.sum()),
            "spreads_pct": spreads,
            "bucket_se_pct": se_level,
            "criterion_pct": 200.0 * noise_frac,
            "criterion_se_level_pct": 2.0 * se_level,
            "pass_delta_ab": bool(spreads["delta_ab_fracx"] is not None and spreads["delta_ab_fracx"] <= 200.0 * noise_frac),
            "pass_delta_bc": bool(spreads["delta_bc_fracx"] is not None and spreads["delta_bc_fracx"] <= 200.0 * noise_frac),
            "pass_delta_bc_se_level": bool(spreads["delta_bc_fracx"] is not None and spreads["delta_bc_fracx"] <= 2.0 * se_level),
        },
        "injection": {},
    }
    for f, d in inj.items():
        exp_pct = 100.0 * sig_meas / float(f)
        verdict["injection"][f] = {
            "per_position_noise_pct": exp_pct,
            "criterion_pct": 2 * exp_pct,
            "p2p_a_pct": d["p2p_a_pct"], "p2p_b_pct": d["p2p_b_pct"], "p2p_c_pct": d["p2p_c_pct"],
            "p2p_d_pct": d["p2p_d_pct"],
            "centroid_bias_p2p_px": d["centroid_bias_p2p_px"],
            "variant_d_note": "质心+分数权重，但天光固定取整数峰环带：用来区分'孔径几何'与'天空跟着质心走'的耦合",
            "informative": bool(d["p2p_a_pct"] > 2 * exp_pct or exp_pct < 1.0),
            "integer_aperture_pass": bool(d["p2p_a_pct"] <= 2 * exp_pct),
            "frac_weight_pass": bool(d["p2p_c_pct"] <= 2 * exp_pct),
        }
    informative = [f for f in verdict["injection"] if verdict["injection"][f]["informative"]]
    verdict["overall"] = {
        "informative_injection_levels": informative,
        "real_star_geometric_phase_effect_pct": spreads["delta_bc_fracx"],
        "injection_phase_effect_pct": {f: verdict["injection"][f]["p2p_a_pct"] for f in informative},
        "pass": bool(verdict["real_star"]["pass_delta_ab"] and verdict["real_star"]["pass_delta_bc"]
                     and all(verdict["injection"][f]["integer_aperture_pass"] for f in informative)),
        "recommendation": None,
    }
    if verdict["overall"]["pass"]:
        verdict["overall"]["recommendation"] = "孔径几何可接受：两条证据的相位散布都在 2x 噪声预期内，r=4 几何孔径可保留"
    else:
        verdict["overall"]["recommendation"] = (
            "必须改'质心 + 分数像元权重'：整数峰为心的几何孔径存在超出噪声预期的相位依赖；"
            "分数像元权重方案的相位散布见 injection.p2p_c_pct 与 real_star.spreads_pct.delta_ac_fracx")

    res = {
        "section": "E3",
        "declaration": "只测孔径几何的子像素相位敏感性；不声明任何星等、精度或标定。",
        "inputs": {
            "psf_source": "epf_profile.json (E4): psf_1px_stack_21x21，归一到 sum=1",
            "psf_sha256": prov["psf_sha256"],
            "psf_fwhm_moment_px": epf["moments"]["fwhm_px"],
            "psf_ee4": [e["ee"] for e in epf["encircled_energy"] if e["r_px"] == 4][0],
            "note": "注入用带限相位移（psf_kernel_at_phase）在 1px 叠合剖面取任意亚像素相位，不做双线性插值（双线性会随相位引入 3% 级别的伪平滑）",
        },
        "evidence1_real_stars": {
            "n_stars": int(okf.sum()),
            "median_flux_r4_adu": float(np.median(fref)),
            "per_star_noise_pct": 100.0 * noise_frac,
            "n_aperture_pixels_integer": {str(k): int((n_ap_a == k).sum()) for k in sorted(set(n_ap_a.tolist()))},
            "n_aperture_pixels_centroid": {str(k): int((n_ap_b == k).sum()) for k in sorted(set(n_ap_b.tolist()))},
            "delta_ab_median_pct": 100.0 * float(np.median(d_ab)),
            "delta_bc_median_pct": 100.0 * float(np.median(d_bc)),
            "delta_ac_median_pct": 100.0 * float(np.median(d_ac)),
            "buckets_fracx_delta_ab": b_ab, "buckets_fracy_delta_ab": b_ab_y,
            "buckets_fracx_delta_bc": b_bc, "buckets_fracx_delta_ac": b_ac,
            "caveat": "真实星的相位分布不均匀且相位与位置相关，因此本项会被'样本选择偏差'污染。delta_bc 不需要总通量，是最干净的'几何离散化偏差 vs 相位'测量。",
        },
        "evidence2_injection": {
            "n_positions": int(pos.size),
            "positions_head": [[int(y), int(x)] for y, x in pos[:20]],
            "positions_sha256": L.sha256_json([[int(y), int(x)] for y, x in pos]),
            "phase_grid": phases.tolist(), "n_phases": int(phases.size ** 2),
            "flux_grid_adu": c["inject_flux_grid_adu"],
            "kernel_sums": kern_sums,
            "kernel_sum_note": "带限相位移的格点和应恒为 1；实测 0.974-0.982 且随相位抖 ~0.8%，说明 P1 非严格带限（含负环与噪声）。已按每个相位归一化后再注入，原始格点和记在 kernel_sums 里。这是注入法固有的模型限制。",
            "seed": c["inject_seed"],
            "per_flux": inj,
            "caveat": "注入用 E4 的单一叠合剖面，因此本项会被'PSF 模型'污染；它只回答'纯子像素相位引起的孔径几何效应有多大'。",
        },
        "verdict": verdict,
        "provenance": prov,
    }
    prov["utc_end"] = L.utc_now()
    L.dump_json(outdir / "e3_aperture_geometry.json", res)
    print("\n[E3] real delta_bc spread=%.3f%% (bucket SE %.3f%%) | delta_ab spread=%.3f%%" % (
        spreads["delta_bc_fracx"], se_level, spreads["delta_ab_fracx"]))
    for f in verdict["injection"]:
        v = verdict["injection"][f]
        print("[E3] inj %6s: p2p a=%.2f%% c=%.2f%% | noise/pos %.2f%% crit %.2f%% | a_pass=%s c_pass=%s" % (
            f, v["p2p_a_pct"], v["p2p_c_pct"], v["per_position_noise_pct"], v["criterion_pct"],
            v["integer_aperture_pass"], v["frac_weight_pass"]))
    print("[E3] overall pass=%s : %s" % (verdict["overall"]["pass"], verdict["overall"]["recommendation"]))
    print("[done] outdir", outdir)


if __name__ == "__main__":
    main()
