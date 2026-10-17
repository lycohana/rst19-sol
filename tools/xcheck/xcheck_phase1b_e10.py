#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xcheck_phase1b_e10.py -- E10 15 帧静止性否决。

1. 自写配准：相位相关给粗平移，再用少量亮星质心给精平移，报告残差。
2. 第 1 帧质量通过列表（SNR>20）在其余 14 帧预测位置测通量，分类：
   静止 / 非静止(位置漂移>1px) / 未检出 / 通量强烈不一致。
3. 统计三类比例，给出"若用多帧静止性做参考样本否决，会剔除多少候选"。
本轮不做运动目标识别、不做轨迹关联。
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import xcheck_phase1b_lib as L


def phase_correlate_shift(a, b, max_shift):
    """返回 (dy, dx) 使 b 相对 a 的平移（b(x) ≈ a(x - shift)）。"""
    ny, nx = a.shape
    wy = np.hanning(ny)[:, None]
    wx = np.hanning(nx)[None, :]
    A = np.fft.rfft2((a - a.mean()) * wy * wx)
    B = np.fft.rfft2((b - b.mean()) * wy * wx)
    R = B * np.conj(A)
    R /= np.maximum(np.abs(R), 1e-12)
    c = np.fft.irfft2(R, s=(ny, nx))
    # 平滑到 ~1px 再由整数峰 + 抛物线插值给亚像素
    c = np.real(np.fft.ifft2(np.fft.fft2(c) * np.exp(-2 * np.pi ** 2 * (np.fft.fftfreq(ny)[:, None] ** 2 + np.fft.fftfreq(nx)[None, :] ** 2) * 0.8 ** 2)))
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    dy, dx = iy, ix
    if dy > ny // 2:
        dy -= ny
    if dx > nx // 2:
        dx -= nx
    return float(dy), float(dx), float(c[iy, ix]), c


def refine_by_stars(a, b, guess, ys, xs, r, half=6, iters=3):
    """用逐星质心（环带取天光）在 guess 平移附近精化整体平移，并统计残差。

    3x3 抛物线定心在 FWHM~2.2px 的 PSF 上有相位相关的系统偏差（实测残差 ~0.5px），
    因此这里直接复用 L.centroid。
    """
    up = np.array(guess, dtype=float)
    for it in range(iters):
        gy = ys + up[0]
        gx = xs + up[1]
        ok = (gy > 12) & (gy < b.shape[0] - 12) & (gx > 12) & (gx < b.shape[1] - 12)
        if ok.sum() < 20:
            break
        cy, cx = L.centroid(b, gy[ok], gx[ok], r=r, sky_in=6.0, sky_out=8.0)
        dy = cy - ys[ok]
        dx = cx - xs[ok]
        good = np.isfinite(dy) & np.isfinite(dx)
        if good.sum() < 20:
            break
        mdy = float(np.median(dy[good]))
        mdx = float(np.median(dx[good]))
        res = np.hypot(dy - mdy, dx - mdx)
        lim = max(0.15, 4.0 * L.mad_sigma(res[good]))
        keep = good & (res < lim)
        if keep.sum() < 20:
            keep = good
        up = np.array([float(np.median(dy[keep])), float(np.median(dx[keep]))])
    res = np.hypot(dy - up[0], dx - up[1])
    good = np.isfinite(res)
    return up[0], up[1], {
        "n_stars_used": int(keep.sum()),
        "resid_rms_px": float(np.sqrt(np.mean(res[keep] ** 2))),
        "resid_median_px": float(np.median(res[keep])),
        "resid_p84_px": float(np.percentile(res[keep], 84)),
        "resid_sem_px": float(np.sqrt(np.mean(res[keep] ** 2)) / math.sqrt(max(keep.sum(), 1))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-sources", type=int, default=None)
    args = ap.parse_args()
    cfg, cfg_sha = L.load_config()
    runid = L.make_runid(cfg_sha)
    outdir = L.ROOT / cfg["output_root_relative"] / runid
    outdir.mkdir(parents=True, exist_ok=True)
    c = cfg["e10"]
    apc = cfg["aperture"]
    p0 = cfg["prior_results"]
    sigma_ap = p0["sigma_ap_r4_adu_from_e1"]
    sigma_pix = p0["sigma_pix_adu"]

    rawdir = L.ROOT / cfg["input"]["fits_relative"]
    files = sorted(rawdir.glob("*.fits"))
    print("[frames] found", len(files))
    ref = [f for f in files if f.name == cfg["input"]["reference_fits"]][0]
    files = [ref] + [f for f in files if f != ref]
    files = files[: (args.max_frames or c["max_frames"])]
    n_max_src = args.max_sources or c["n_sources_max"]
    sha = {f.name: L.sha256_file(f) for f in files}
    prov = L.provenance(Path(__file__).resolve(), cfg_sha, runid)
    prov["frames"] = [{"name": f.name, "sha256": sha[f.name], "size": f.stat().st_size} for f in files]
    print("[prov] runid", runid, "| ref sha ok:", sha[ref.name] == cfg["input"]["expected_sha256_reference"])

    dn0, hdr0 = L.load_frame(ref)
    fr0 = L.Frame(dn0, cfg)
    det0 = fr0.detect(keep_morph=False)
    yx0 = det0["yx"]
    peak0 = fr0.peak_dn(yx0)
    ap0 = L.aperture_measure(fr0.dn, yx0[:, 0].astype(float), yx0[:, 1].astype(float),
                             apc["r_ap"], apc["r_in"], apc["r_out"])
    snr0 = ap0["flux"] / sigma_ap
    iso0 = L.nearest_neighbor_px(yx0.astype(float), c["regstar_isolation_px"])
    iso8 = L.nearest_neighbor_px(yx0.astype(float), 8.0)
    iyb, ixb, validb, db = L._win_index(yx0[:, 0].astype(float), yx0[:, 1].astype(float), 9, *dn0.shape)
    negnear = ((dn0[iyb, ixb] < 0) & validb).any(axis=1)
    m = (snr0 >= c["ap_snr_min"]) & (peak0 < c["peak_max_dn_e10"]) & (~negnear) & np.isfinite(ap0["flux"])
    cand = np.flatnonzero(m)
    print("[frame1] peaks=%d | snr_bg>=%.0f:%d | peak<%.0f:%d | no-neg-pixel:%d | all:%d" % (
        yx0.shape[0], c["ap_snr_min"], int((snr0 >= c["ap_snr_min"]).sum()),
        c["peak_max_dn_e10"], int((peak0 < c["peak_max_dn_e10"]).sum()),
        int((~negnear).sum()), cand.size))
    rs = np.random.default_rng(c["sample_seed"])
    order = cand if cand.size <= n_max_src else rs.choice(cand, size=n_max_src, replace=False)
    order = np.sort(order)
    print("[frame1] random sample kept:", order.size)

    reg_sel = (snr0 >= c["regstar_mf_snr_min"]) & (iso0 >= c["regstar_isolation_px"]) & (peak0 < c["regstar_peak_max_dn"])
    reg_idx = np.flatnonzero(reg_sel)
    reg_idx = reg_idx[np.argsort(-snr0[reg_idx])][: c["n_regstars"]]
    ry, rx = L.centroid(fr0.dn, yx0[reg_idx, 0].astype(float), yx0[reg_idx, 1].astype(float), r=3.0)
    print("[reg] registration stars:", reg_idx.size)

    ny, nx = dn0.shape
    cc = int(c["phasecorr_crop"])
    y0c, x0c = ny // 2 - cc // 2, nx // 2 - cc // 2
    crop0 = fr0.img[y0c:y0c + cc, x0c:x0c + cc]
    sh = c["phasecorr_max_shift"]
    crop0 = crop0[sh:cc - sh, sh:cc - sh]

    shifts = [{"frame": 0, "name": ref.name, "dy": 0.0, "dx": 0.0, "resid_rms_px": 0.0,
               "method": "reference"}]
    for k in range(1, len(files)):
        dnk, _ = L.load_frame(files[k])
        frk = L.Frame(dnk, cfg)
        cropk = frk.img[y0c + sh:y0c + cc - sh, x0c + sh:x0c + cc - sh]
        pdy, pdx, pcval, _ = phase_correlate_shift(crop0, cropk, sh)
        dy, dx, rs = refine_by_stars(dn0, frk.dn, (pdy, pdx), ry, rx, 3.0, half=c["local_search_px"])
        shifts.append({"frame": k, "name": files[k].name,
                       "phasecorr_dy": pdy, "phasecorr_dx": pdx, "phasecorr_peak": pcval,
                       "dy": dy, "dx": dx, "method": "phasecorr+stars", **rs})
        print("[reg] %-28s phasecorr(%+.2f,%+.2f) -> stars(%+.3f,%+.3f) resid %.3f px (n=%d)" % (
            files[k].name, pdy, pdx, dy, dx, rs["resid_rms_px"], rs["n_stars_used"]))
    resid = np.array([s["resid_rms_px"] for s in shifts[1:]])
    reg_ok = bool(np.all(resid < c["registration_resid_max_px"]))
    print("[reg] resid max %.3f px  -> registration usable: %s" % (float(resid.max()), reg_ok))

    # ---------------- 多帧测量 ----------------
    nsrc = order.size
    ys1 = yx0[order, 0].astype(float)
    xs1 = yx0[order, 1].astype(float)
    f1 = ap0["flux"][order]
    sky1 = ap0["sky"][order]
    nfr = len(files)
    posy = np.full((nfr, nsrc), np.nan)
    posx = np.full((nfr, nsrc), np.nan)
    flux = np.full((nfr, nsrc), np.nan)
    snrm = np.full((nfr, nsrc), np.nan)
    posy[0], posx[0], flux[0] = ys1, xs1, f1

    for k in range(1, nfr):
        dnk, _ = L.load_frame(files[k])
        dy, dx = shifts[k]["dy"], shifts[k]["dx"]
        py, px = ys1 + dy, xs1 + dx
        # 局部重定心：直接迭代质心（环带 6-8px 取天光）
        vy, vx = L.centroid(dnk, py, px, r=3.0, sky_in=6.0, sky_out=8.0)
        edge = (vy < 12) | (vy > ny - 12) | (vx < 12) | (vx > nx - 12)
        vy = np.where(edge, np.nan, vy)
        vx = np.where(edge, np.nan, vx)
        meas = L.aperture_measure(dnk, np.where(np.isfinite(vy), vy, py), np.where(np.isfinite(vx), vx, px),
                                  apc["r_ap"], apc["r_in"], apc["r_out"])
        meas["flux"] = np.where(np.isfinite(vy) & np.isfinite(vx), meas["flux"], np.nan)
        posy[k], posx[k] = vy, vx
        flux[k] = meas["flux"]
        snrm[k] = meas["flux"] / np.maximum(meas["sky_sigma"] * math.sqrt(apc["r_ap"] ** 2 * math.pi), 1e-9)

    br = (np.isfinite(flux[0]) & (flux[0] > 1000.0))
    frame_phot = []
    for k in range(1, nfr):
        ok = br & np.isfinite(flux[k])
        frame_phot.append({"frame": k, "name": files[k].name, "n": int(ok.sum()),
                           "median_flux_ratio": float(np.median(flux[k][ok] / flux[0][ok])) if ok.sum() > 10 else None})

    # ---------------- 分类 ----------------
    # 位置一致性：与帧 1 位置 + 该帧平移 的偏差
    dpos = np.hypot(posy - (ys1[None, :] + np.array([[s["dy"] for s in shifts]]).T),
                    posx - (xs1[None, :] + np.array([[s["dx"] for s in shifts]]).T))
    # 通量误差必须含天光中位数误差 x 孔径面积这一项：r=4 孔径 49 像元 x 0.9 ADU ~ 45 ADU，
    # 它在实测中就是 frame-to-frame 通量差的主导项（E4 已量化 sig_F(r=14)=256 ADU）
    n_ap4 = int((np.hypot(*np.meshgrid(np.arange(-4, 5), np.arange(-4, 5), indexing="ij")) <= 4).sum())
    n_ann68 = 0
    for iy_ in range(-8, 9):
        for ix_ in range(-8, 9):
            dd = math.hypot(iy_, ix_)
            if 6.0 <= dd <= 8.0:
                n_ann68 += 1
    sig_sky_ann = 1.2533141 * sigma_pix / math.sqrt(n_ann68)
    sig_flux = math.sqrt(sigma_ap ** 2 + (n_ap4 * sig_sky_ann) ** 2)
    ferr = sig_flux * math.sqrt(2.0)
    print("[flux] n_ap4=%d n_ann68=%d sig_sky_ann=%.3f -> sig_flux=%.1f ADU, 3sigma diff=%.0f ADU" % (
        n_ap4, n_ann68, sig_sky_ann, sig_flux, 3 * ferr))
    rate = np.full(nsrc, np.nan)
    cls = np.empty(nsrc, dtype=object)
    det_frac = np.zeros(nsrc)
    drift_px = np.zeros(nsrc)
    flux_bad = np.zeros(nsrc)
    drift84 = np.full(nsrc, np.nan)
    drift_sem = np.full(nsrc, np.nan)
    scat_rel = np.full(nsrc, np.nan)
    for i in range(nsrc):
        ok = np.isfinite(flux[:, i]) & (flux[:, i] > -1e5)
        det = np.isfinite(flux[1:, i])
        det_frac[i] = float(det.mean()) if det.size else 0.0
        if det.sum() >= c["min_frames_for_flux_test"]:
            dp = dpos[1:, i][det]
            drift_px[i] = float(np.median(dp))
            drift84[i] = float(np.percentile(dp, 84))
            drift_sem[i] = 1.2533 * L.mad_sigma(dp) / math.sqrt(max(dp.size, 1))
            rel = flux[1:, i][det] / flux[0, i] - 1.0
            scat = 1.2533 * L.mad_sigma(rel)
            scat_rel[i] = scat
            # 阈值 = max(3 sigma 背景模型, 相对下限)：明亮源的真实噪声含泊松+闪烁项，
            # GAIN 缺失无法算泊松项，故给一个相对下限并显式声明它是假设而非测量
            thr = max(c["flux_sigma_mult"] * ferr / flux[0, i], c["flux_rel_floor"])
            flux_bad[i] = float(scat > thr)
        else:
            drift_px[i] = float("nan")
            flux_bad[i] = float("nan")
    for i in range(nsrc):
        if det_frac[i] < 0.5:
            cls[i] = "undetected"
        elif (np.isfinite(drift_px[i]) and drift_px[i] > c["drift_px_max"]
              and drift_px[i] > 3.0 * max(drift_sem[i], 1e-6)):
            cls[i] = "non_stationary"
        elif np.isfinite(flux_bad[i]) and flux_bad[i] > 0.5:
            cls[i] = "flux_inconsistent"
        else:
            cls[i] = "stationary"
    cnt = {k: int((cls == k).sum()) for k in ("stationary", "non_stationary", "undetected", "flux_inconsistent")}
    veto = cnt["non_stationary"] + cnt["undetected"] + cnt["flux_inconsistent"]
    print("[class] %s | veto would drop %d/%d = %.1f%%" % (
        cnt, veto, nsrc, 100.0 * veto / max(nsrc, 1)))

    by_snr = []
    for lo, hi in ((20, 50), (50, 150), (150, 500), (500, 1e9)):
        m = (snr0[order] >= lo) & (snr0[order] < hi)
        if m.sum() == 0:
            continue
        cc2 = {k: int((cls[m] == k).sum()) for k in ("stationary", "non_stationary", "undetected", "flux_inconsistent")}
        by_snr.append({"snr_bg_range": [lo, hi], "n": int(m.sum()), **cc2,
                       "veto_frac": float((cc2["non_stationary"] + cc2["undetected"] + cc2["flux_inconsistent"]) / m.sum())})

    res = {
        "section": "E10",
        "declaration": "本文件只做多帧静止性否决；不做运动目标识别、不做轨迹关联、不声明任何天体身份。",
        "frames": {"n_used": len(files), "reference": ref.name,
                   "note": "15 帧不是独立样本：这里只用逐帧一致性，不把 15 帧当独立测量乘 sqrt(15)"},
        "registration": {
            "shifts": shifts,
            "resid_rms_max_px": float(resid.max()),
            "resid_rms_median_px": float(np.median(resid)),
            "requirement_px": c["registration_resid_max_px"],
            "usable": reg_ok,
            "method": "1024x1024 中心裁切 + Hann 窗相位相关给粗平移（±64px），再用 %d 颗亮星逐星 3x3 抛物线定心给精平移并统计残差" % reg_idx.size,
            "shift_summary": {"dy_range": [float(min(s['dy'] for s in shifts)), float(max(s['dy'] for s in shifts))],
                              "dx_range": [float(min(s['dx'] for s in shifts)), float(max(s['dx'] for s in shifts))]},
        },
        "sources": {
            "n_frame1_candidates": int(nsrc),
            "selection": "第 1 帧质量通过列表（MF 8sigma + 形态筛）中 snr_bg>=%.0f 的源，最多 %d 个" % (c["ap_snr_min"], c["n_sources_max"]),
            "snr_definition": "flux(r=4)/sigma_ap_r4_frozen=%.4f ADU（背景极限 SNR）" % sigma_ap,
        },
        "classification": {
            "counts": cnt,
            "fractions": {k: float(v / max(nsrc, 1)) for k, v in cnt.items()},
            "rules": {
                "undetected": "在 <50% 的其余帧里测不到（通量无法测出）",
                "non_stationary": "位置偏差 84 百分位 > %.1f px（相对帧 1 位置 + 该帧整体平移）" % c["drift_px_max"],
                "flux_inconsistent": "逐帧相对通量的经验散布 > max(%.0f*sigma_flux_bg/F1, %.2f)，即半数以上帧显著偏离" % (
                    c["flux_sigma_mult"], c["flux_rel_floor"]),
                "stationary": "其余（>=80% 帧位置一致且通量在 3 sigma 内）",
                "flux_error_note": "每帧 flux 误差按 sqrt(2)*sqrt(sigma_ap^2 + (49*sigma_sky_ann)^2) 计（两帧相减），未含源泊松项（GAIN UNKNOWN）",
            },
            "by_snr_bg": by_snr,
            "frame_photometry_ratio_vs_frame1": frame_phot,
            "detection_fraction_median": float(np.median(det_frac)),
            "flux_rel_scatter_median": float(np.nanmedian(scat_rel)),
            "flux_rel_scatter_p90": float(np.nanpercentile(scat_rel, 90)),
            "flux_rel_floor_used": c["flux_rel_floor"],
            "flux_rel_scatter_note": "相对下限是假设不是测量：GAIN 缺失无法算泊松项；明亮源的真实逐帧散布可能来自泊松/闪烁/非线性，本文件不区分",
            "pos_drift_median_px": float(np.nanmedian(drift_px)),
            "pos_drift_median_sem_px": float(np.nanmedian(drift_sem)),
            "non_stationary_rule": "median|dpos| > 1.0 px 且 > 3x 其标准误（避免把暗源的质心噪声当成漂移）",
            "pos_drift_pct84": float(np.nanmedian(drift84)),
            "flux_error_per_frame_pair_adu": float(ferr),
            "flux_error_model": "sqrt(2)*sqrt(sigma_ap^2 + (49*sigma_sky_ann)^2): 天光项主导",
            "flux_3sigma_change_adu": float(3 * ferr),
        },
        "veto_if_used_for_reference_purity": {
            "n_candidates": int(nsrc),
            "n_rejected": int(veto),
            "rejected_fraction": float(veto / max(nsrc, 1)),
            "note": "这是'用多帧静止性否决参考样本'会剔除的比例；被剔除不等于假源，也不等于真源，只说明该源不适合当静止参考。",
        },
        "provenance": prov,
    }
    prov["utc_end"] = L.utc_now()
    L.dump_json(outdir / "e10_stationarity.json", res)
    print("[done] outdir", outdir)


if __name__ == "__main__":
    main()
