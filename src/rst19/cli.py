"""rst19 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import load_catalog_csv
from .pipeline import analyze_frame
from .wcs import TangentPlaneWCS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="读取 rst19 FITS，检测星点并按先验 WCS 匹配离线星表")
    parser.add_argument("fits_path", type=Path, help="输入 FITS 文件")
    parser.add_argument("--catalog", type=Path, help="离线 CSV 星表，必需列 source_id, ra_deg, dec_deg")
    parser.add_argument("--center-ra", type=float, help="WCS 光轴赤经（deg）")
    parser.add_argument("--center-dec", type=float, help="WCS 光轴赤纬（deg）")
    parser.add_argument("--pixel-scale-arcsec", type=float, help="像元角尺度（arcsec/pixel）")
    parser.add_argument("--rotation-deg", type=float, default=0.0, help="天球东/北到图像坐标的旋转角")
    parser.add_argument("--parity", type=int, choices=(-1, 1), default=1, help="RA 轴方向，取 1 或 -1")
    parser.add_argument("--crpix-x", type=float, help="WCS 参考像素 x；默认图像中心")
    parser.add_argument("--crpix-y", type=float, help="WCS 参考像素 y；默认图像中心")
    parser.add_argument("--epoch", type=float, help="将星表自行传播到指定 Julian 年")
    parser.add_argument("--match-radius-px", type=float, default=3.0, help="匹配半径（pixel）")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="匹配滤波检测阈值；默认 4σ，优先保留候选")
    parser.add_argument("--min-distance", type=int, default=3, help="候选峰最小间距（pixel）")
    parser.add_argument("--aperture-radius", type=int, default=4, help="通量估计半径（pixel）")
    parser.add_argument("--max-sources", type=int, help="最多保留的检测源数")
    parser.add_argument("--psf-fwhm", type=float, default=3.0, help="Gaussian 点扩散函数 FWHM（pixel）")
    parser.add_argument("--background-box-size", type=int, default=128, help="局部背景/RMS 统计块大小（pixel）")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="可信星点的孔径通量 SNR 下限")
    parser.add_argument("--min-fwhm", type=float, default=0.8, help="可信点源的 FWHM 下限（pixel）")
    parser.add_argument("--max-fwhm", type=float, default=12.0, help="可信点源的 FWHM 上限（pixel）")
    parser.add_argument("--max-ellipticity", type=float, default=0.65, help="可信点源椭圆率上限")
    parser.add_argument("--min-sharpness", type=float, default=0.005, help="点源 sharpness 下限")
    parser.add_argument("--max-sharpness", type=float, default=0.85, help="尖峰伪迹 sharpness 上限")
    parser.add_argument("--min-footprint-pixels", type=int, default=2, help="点源孔径内超过 1 RMS 的最少像素数")
    parser.add_argument("--gain-e-per-adu", type=float, help="可选 CCD 增益（electron/ADU）")
    parser.add_argument("--read-noise-adu", type=float, default=0.0, help="读出噪声（ADU）")
    parser.add_argument("--keep-zero-pixels", action="store_true", help="不把整数图像中的精确 0 自动标记为无效像素")
    parser.add_argument("--zero-point", type=float, help="可选仪器星等零点；不提供时只输出仪器星等")
    parser.add_argument("--json-out", type=Path, help="可选 JSON 输出文件；不提供时输出到 stdout")
    return parser


def _build_wcs(args: argparse.Namespace, width: int, height: int) -> TangentPlaneWCS | None:
    wcs_fields = (args.center_ra, args.center_dec, args.pixel_scale_arcsec)
    if args.catalog is None and all(value is None for value in wcs_fields):
        return None
    if args.catalog is None:
        raise ValueError("提供 WCS 参数时必须同时提供 --catalog")
    if any(value is None for value in wcs_fields):
        raise ValueError("--catalog 需要同时提供 --center-ra、--center-dec 和 --pixel-scale-arcsec")
    return TangentPlaneWCS(
        center_ra_deg=args.center_ra,
        center_dec_deg=args.center_dec,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
        crpix_x=width / 2.0 if args.crpix_x is None else args.crpix_x,
        crpix_y=height / 2.0 if args.crpix_y is None else args.crpix_y,
        rotation_deg=args.rotation_deg,
        parity=args.parity,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from .fits import read_fits

        frame = read_fits(args.fits_path)
        catalog = load_catalog_csv(args.catalog) if args.catalog else None
        wcs = _build_wcs(args, frame.width, frame.height)
        result = analyze_frame(
            str(args.fits_path),
            catalog=catalog,
            wcs=wcs,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            max_sources=args.max_sources,
            psf_fwhm=args.psf_fwhm,
            background_box_size=args.background_box_size,
            min_flux_snr=args.min_flux_snr,
            min_fwhm=args.min_fwhm,
            max_fwhm=args.max_fwhm,
            max_ellipticity=args.max_ellipticity,
            min_sharpness=args.min_sharpness,
            max_sharpness=args.max_sharpness,
            min_footprint_pixels=args.min_footprint_pixels,
            gain_e_per_adu=args.gain_e_per_adu,
            read_noise_adu=args.read_noise_adu,
            mask_zero_pixels=False if args.keep_zero_pixels else None,
            match_radius_px=args.match_radius_px,
            epoch=args.epoch,
            zero_point=args.zero_point,
        )
        payload = result.as_dict()
        if wcs is not None:
            payload["wcs"] = wcs.as_dict()
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0
    except (OSError, ValueError) as exc:
        print(f"rst19: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
