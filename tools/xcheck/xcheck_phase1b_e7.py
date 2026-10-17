#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xcheck_phase1b_e7.py -- E7-lite 注入回收最小版本。

用 E4 的经验 PSF 注入到第 1 帧的无源天空区，注入亮度只用浮点相对量（相对该帧 flux=219 ADU
的倍数，或 m_inst = -2.5 log10(F/t)）。不声称任何星等系统。
分 3 背景箱 x 3 拥挤度箱，每箱 >=200 次注入、>=5 次实现；注入表冻结并记录哈希与种子。
"""
import json
import math
from pathlib import Path

import numpy as np

import xcheck_phase1b_lib as L


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def crossing(levels, fracs):
    """在 levels（递增）上线性内插 50% 回收点（回收率随通量单调递增）。"""
    for i in range(1, len(levels)):
        if fracs[i - 1] < 0.5 <= fracs[i]:
            t = (0.5 - fracs[i - 1]) / (fracs[i] - fracs[i - 1])
            return levels[i - 1] + t * (levels[i] - levels[i - 1])
    if fracs and fracs[0] >= 0.5:
        return levels[0]
    return None


def main():
    cfg, cfg_sha = L.load_config()
    runid = L.make_runid(cfg_sha)
    outdir = L.ROOT / cfg["output_root_relative"] / runid
    outdir.mkdir(parents=True, exist_ok=True)
    c = cfg["e7"]
    apc = cfg["aperture"]
    p0 = cfg["prior_results"]
    sigma_ap = p0["sigma_ap_r4_adu_from_e1"]
    sigma_pix = p0["sigma_pix_adu"]
    t_exp = c["exposure_s"]

    epf = json.loads((outdir / "epf_profile.json").read_text(encoding="utf-8"))
    P1 = np.asarray(epf["psf_1px_stack_21x21"], dtype=np.float64)
    P1 = P1 / P1.sum()

    fits = L.ROOT / cfg["input"]["fits_relative"] / cfg["input"]["reference_fits"]
    prov = L.provenance(Path(__file__).resolve(), cfg_sha, runid)
    prov["input_fits_sha256"] = L.sha256_file(fits)
    prov["psf_sha256"] = L.sha256_json(P1.tolist())
    print("[prov] script", prov["script_sha256"], "runid", runid)

    dn, hdr = L.load_frame(fits)
    fr = L.Frame(dn, cfg)
    det = fr.detect(keep_morph=False)
    yx = det["yx"]
    ny, nx = dn.shape
    print("[det] peaks", yx.shape[0])

    # ---- 位置池：避开已知源 3xFWHM 以上、无负像元、远离边缘 ----
    srcmask = np.zeros((ny, nx), dtype=bool)
    srcmask[yx[:, 0].astype(int), yx[:, 1].astype(int)] = True
    srcmask = L.dilate_bool(srcmask, int(c["clean_dilate_px"]))
    rng = np.random.default_rng(c["seed"])
    ntry = 60000
    cy = rng.integers(120, ny - 120, size=ntry)
    cx = rng.integers(120, nx - 120, size=ntry)
    ok = ~srcmask[cy, cx]
    cy, cx = cy[ok], cx[ok]
    keep = []
    for y, x in zip(cy, cx):
        if dn[y - 20:y + 21, x - 20:x + 21].min() < 0:
            continue
        keep.append((int(y), int(x)))
        if len(keep) >= 6000:
            break
    pool = np.asarray(keep, dtype=int)
    print("[pool] clean positions:", len(pool))

    bgv = fr.bg_med[pool[:, 0], pool[:, 1]].astype(np.float64)
    # 拥挤度：20px 内的已知峰数
    cnt = np.zeros(len(pool), dtype=int)
    cell = 20.0
    grid = {}
    srcs = yx.astype(np.float64)
    for j in range(srcs.shape[0]):
        grid.setdefault((int(srcs[j, 1] // cell), int(srcs[j, 0] // cell)), []).append(j)
    for i, (y, x) in enumerate(pool):
        cxx, cyy = int(x // cell), int(y // cell)
        n = 0
        for gy in (cyy - 1, cyy, cyy + 1):
            for gx in (cxx - 1, cxx, cxx + 1):
                for j in grid.get((gx, gy), ()):
                    if (srcs[j, 0] - y) ** 2 + (srcs[j, 1] - x) ** 2 <= 400.0:
                        n += 1
        cnt[i] = n
    bq = np.quantile(bgv, [1 / 3, 2 / 3])
    cq = np.quantile(cnt, [1 / 3, 2 / 3])
    bbi = np.clip(np.searchsorted(bq, bgv, side="right"), 0, 2)
    cci = np.clip(np.searchsorted(cq, cnt, side="right"), 0, 2)
    print("[bins] bg edges", np.round(bq, 3).tolist(), "crowd edges", cq.tolist())

    levels = np.array(c["inject_flux_rel_to_219"], dtype=float) * p0["probe_flux_adu"]
    nlev = levels.size
    nreal = c["n_realizations"]
    per_bin = c["n_injections_per_bin"]
    per_level_real = max(1, per_bin // (nlev * nreal))
    print("[plan] levels", np.round(levels, 1).tolist(), "per level/real", per_level_real)

    phases = []
    table = []
    rng2 = np.random.default_rng(c["seed"] + 1)
    for bi in range(3):
        for ci in range(3):
            sel = np.flatnonzero((bbi == bi) & (cci == ci))
            if sel.size < 20:
                continue
            for ri in range(nreal):
                for fi in range(nlev):
                    for _ in range(per_level_real):
                        pi = int(rng2.choice(sel))
                        py = float(rng2.uniform(0, 1))
                        px = float(rng2.uniform(0, 1))
                        table.append([bi, ci, ri, fi, int(pool[pi, 0]), int(pool[pi, 1]),
                                      round(py, 6), round(px, 6)])

    rows = [{"bin": [bi, ci], "realization": ri, "flux_index": fi, "y": y, "x": x,
             "phase": [py, px], "flux_injected_adu": None} for (bi, ci, ri, fi, y, x, py, px) in table]
    table_sha = L.sha256_json(table)
    print("[freeze] injection table entries:", len(table), "sha256:", table_sha)

    n_ann = 88
    sig_sky_ann = 1.2533141 * sigma_pix / math.sqrt(n_ann)
    sig_meas = math.sqrt(sigma_ap ** 2 + (49 * sig_sky_ann) ** 2)

    half = 8
    cache = {}
    rec = []
    for (bi, ci, ri, fi, y, x, py, px) in table:
        key = (py, px)
        if key not in cache:
            k = L.psf_kernel_at_phase(P1, py, px, half=half)
            cache[key] = k / float(k.sum())
        flux = float(levels[fi])
        patch = dn[y - 30:y + 31, x - 30:x + 31].astype(np.float64).copy()
        patch[30 - half:30 + half + 1, 30 - half:30 + half + 1] += cache[key] * flux
        # 在真值位置做局部重定心再测（与生产同语义：质心 + 固定几何 r=4 孔径）
        vy, vx = L.centroid(patch, np.array([30.0 + py]), np.array([30.0 + px]), r=3.0)
        meas = L.aperture_measure(patch, vy, vx, apc["r_ap"], apc["r_in"], apc["r_out"])
        f = float(meas["flux"][0])
        snr = f / sigma_ap
        dpos = math.hypot(vy[0] - (30.0 + py), vx[0] - (30.0 + px))
        rel_ok = abs(f / flux - 1.0) <= c["rel_flux_tolerance"]
        ok_rec = bool(snr >= c["detect_snr_recover_min"] and rel_ok
                      and abs(f - flux) <= c["flux_recover_sigma_mult"] * sig_meas
                      and dpos <= c["pos_recover_px"])
        rec.append((bi, ci, ri, fi, ok_rec, snr, f / flux))
    rec = np.array([(r[0], r[1], r[2], r[3], int(r[4]), r[5], r[6]) for r in rec], dtype=float)
    print("[inject] done:", rec.shape[0])

    # ---- 回收率曲线 ----
    out_bins = []
    for bi in range(3):
        for ci in range(3):
            m = (rec[:, 0] == bi) & (rec[:, 1] == ci)
            if m.sum() < 50:
                continue
            rows_out = []
            fr_list, lev_list = [], []
            for fi in range(nlev):
                mm = m & (rec[:, 3] == fi)
                n = int(mm.sum())
                k = int(rec[mm, 4].sum())
                p = k / n if n else float("nan")
                ci_lo, ci_hi = wilson(k, n)
                fr_list.append(p)
                lev_list.append(float(levels[fi]))
                rows_out.append({"flux_adu": float(levels[fi]),
                                 "flux_rel_to_219": float(levels[fi] / p0["probe_flux_adu"]),
                                 "m_inst": -2.5 * math.log10(float(levels[fi]) / t_exp),
                                 "n": n, "n_recovered": k, "rate": p,
                                 "ci95": [ci_lo, ci_hi],
                                 "median_snr": float(np.median(rec[mm, 5])) if n else None,
                                 "median_flux_ratio": float(np.median(rec[mm, 6])) if n else None})
            x50 = crossing(lev_list, fr_list)
            boot = []
            rng3 = np.random.default_rng(c["seed"] + 7)
            for _ in range(400):
                fs = []
                for fi in range(nlev):
                    mm = m & (rec[:, 3] == fi)
                    n = int(mm.sum())
                    k = int(rec[mm, 4].sum())
                    if n == 0:
                        fs.append(float("nan"))
                        continue
                    fs.append(rng3.binomial(n, k / n) / n)
                x = crossing(lev_list, fs)
                if x is not None:
                    boot.append(x)
            out_bins.append({
                "background_bin": bi, "crowding_bin": ci,
                "bg_adu_range": [float(bq[bi - 1]) if bi > 0 else float(bgv.min()),
                                 float(bq[bi]) if bi < 2 else float(bgv.max())],
                "crowding_range": [float(cq[ci - 1]) if ci > 0 else float(cnt.min()),
                                   float(cq[ci]) if ci < 2 else float(cnt.max())],
                "n_injections": int(m.sum()), "n_realizations": int(np.unique(rec[m, 2]).size),
                "curve": rows_out,
                "flux_50pct_adu": x50,
                "flux_50pct_ci95_adu": [float(np.percentile(boot, 16)), float(np.percentile(boot, 84))] if len(boot) > 20 else None,
                "m_inst_50pct": (-2.5 * math.log10(x50 / t_exp)) if x50 else None,
                "m_inst_50pct_ci95": ([-2.5 * math.log10(np.percentile(boot, 84) / t_exp),
                                        -2.5 * math.log10(np.percentile(boot, 16) / t_exp)] if len(boot) > 20 else None),
            })
    allf = [b["m_inst_50pct"] for b in out_bins if b["m_inst_50pct"] is not None]
    print("[E7] bins:", len(out_bins), "m_inst(50%) range:", (round(min(allf), 3), round(max(allf), 3)) if allf else None)

    res = {
        "section": "E7",
        "declaration": "这是流程完备度（注入-回收率），不是天体身份真值，也不是任何星等系统的标定。注入亮度只用浮点相对量：flux_adu，以及 m_inst = -2.5*log10(F/t_exp) 这一纯仪器相对标度（t_exp=1.5s 来自 FITS EXPOSURE 注释，前批已标为待确认）。",
        "inputs": {"psf_source": "epf_profile.json psf_1px_stack_21x21", "psf_sha256": prov["psf_sha256"],
                   "sigma_ap_r4_adu": sigma_ap, "sigma_pix_adu": sigma_pix,
                   "sigma_meas_per_injection_adu": sig_meas,
                   "sigma_meas_note": "sqrt(sigma_ap^2+(49*sigma_sky_ann)^2): 天光项主导；GAIN 缺失，源泊松项 UNKNOWN"},
        "frozen_injection_table": {
            "n_entries": int(len(table)),
            "sha256": table_sha,
            "seed": c["seed"], "seed_phase": c["seed"] + 1,
            "columns": ["background_bin", "crowding_bin", "realization", "flux_index", "y", "x", "phase_y", "phase_x"],
            "head": table[:10],
            "note": "表在注入前一次性生成并哈希；重跑同 seed 应得到同表同哈希。表本身较长，这里只存头部与哈希。",
        },
        "binning": {"background_bins_adu_edges": [float(x) for x in bq],
                    "crowding_bins_n_sources_within_20px_edges": [float(x) for x in cq],
                    "background_definition": "128px 盒中位数背景图在该位置的取值",
                    "crowding_definition": "该位置 20px 内已探测峰数"},
        "recovery_criterion": {
            "detect_snr_min": c["detect_snr_recover_min"],
            "snr_definition": "r=4 孔径通量 / sigma_ap_r4 = %.4f ADU（背景极限；未含源泊松项）" % sigma_ap,
            "flux_tolerance_sigma": c["flux_recover_sigma_mult"],
            "flux_tolerance_adu": c["flux_recover_sigma_mult"] * sig_meas,
            "relative_flux_tolerance": c["rel_flux_tolerance"],
            "relative_flux_note": "回收要求 |F_meas/F_inj - 1| <= %.2f。这一条在暗端是主导：通量噪声 sigma_meas=%.1f ADU 远大于生产误差模型的 %.3f ADU。" % (
                c["rel_flux_tolerance"], sig_meas, sigma_ap),
            "pos_tolerance_px": c["pos_recover_px"],
        },
        "bins": out_bins,
        "m_inst_50pct_range": [float(min(allf)), float(max(allf))] if allf else None,
        "caveats": [
            "回收率定义为'孔径 SNR>=5 且通量在 3x sigma_meas 内且位置在 2px 内'，是流程层面的可测性，不是天体是否存在的判定。",
            "注入用单一经验 PSF，未模拟 PSF 的空间变化（E4 实测 4x4 网格 FWHM 散布 ~0.5px）；因此不同背景/拥挤度箱的差异里含 PSF 模型误差。",
            "m_inst = -2.5log10(F/1.5s) 是纯相对仪器标度：没有零点、没有滤光片、没有大气/探测器标定，不可与任何星等系统比较。",
            "15 帧不是独立样本；本文件只用第 1 帧，不做 sqrt(15) 的统计增益。",
        ],
        "provenance": prov,
    }
    prov["utc_end"] = L.utc_now()
    L.dump_json(outdir / "e7_injection_lite.json", res)
    for b in out_bins:
        print("[E7] bg%d crowd%d n=%d  m_inst(50%%)=%s" % (
            b["background_bin"], b["crowding_bin"], b["n_injections"],
            round(b["m_inst_50pct"], 3) if b["m_inst_50pct"] is not None else None))
    print("[done] outdir", outdir)


if __name__ == "__main__":
    main()
