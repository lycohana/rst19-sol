"""真实 FITS 星点检测参数扫描命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .experiments import (
    run_detection_psf_sweep,
    run_detection_threshold_sweep,
    write_detection_psf_sweep_artifacts,
    write_detection_sweep_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在同一真实 FITS 上扫描星点候选阈值并写出表格/曲线")
    parser.add_argument("fits_path", type=Path, help="输入 FITS 文件")
    parser.add_argument("--thresholds", type=float, nargs="+", default=(4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0), help="候选阈值 sigma 列表")
    parser.add_argument("--psf-fwhms", type=float, nargs="+", help="改为扫描匹配滤波 PSF FWHM 列表（pixel）")
    parser.add_argument("--min-distance", type=int, default=4, help="候选峰最小间距（pixel）")
    parser.add_argument("--aperture-radius", type=int, default=4, help="孔径测光半径（pixel）")
    parser.add_argument("--psf-fwhm", type=float, default=3.0, help="Gaussian PSF FWHM（pixel）")
    parser.add_argument("--background-box-size", type=int, default=128, help="局部背景/RMS 网格大小（pixel）")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量层孔径通量 SNR 下限")
    parser.add_argument("--min-psf-support-pixels", type=int, default=3, help="候选峰中心 3×3 内超过 PSF 支持阈值的最少像素数")
    parser.add_argument("--max-sources", type=int, help="可选返回源上限；留空表示全量")
    parser.add_argument("--keep-linear-artifacts", action="store_true", help="不标记线状结构为 LINE_ARTIFACT")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/detection-sweep"), help="CSV/PNG 输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.psf_fwhms:
            rows = run_detection_psf_sweep(
                args.fits_path,
                psf_levels=args.psf_fwhms,
                threshold_sigma=args.thresholds[0],
                min_distance=args.min_distance,
                aperture_radius=args.aperture_radius,
                background_box_size=args.background_box_size,
                min_flux_snr=args.min_flux_snr,
                min_psf_support_pixels=args.min_psf_support_pixels,
                max_sources=args.max_sources,
                reject_linear_artifacts=not args.keep_linear_artifacts,
            )
            output = write_detection_psf_sweep_artifacts(rows, args.out_dir)
            print(f"detection PSF sweep: {output}")
        else:
            rows = run_detection_threshold_sweep(
                args.fits_path,
                threshold_levels=args.thresholds,
                min_distance=args.min_distance,
                aperture_radius=args.aperture_radius,
                psf_fwhm=args.psf_fwhm,
                background_box_size=args.background_box_size,
                min_flux_snr=args.min_flux_snr,
                min_psf_support_pixels=args.min_psf_support_pixels,
                max_sources=args.max_sources,
                reject_linear_artifacts=not args.keep_linear_artifacts,
            )
            output = write_detection_sweep_artifacts(rows, args.out_dir)
            print(f"detection sweep: {output}")
        print(f"levels={len(rows)} first={rows[0].as_dict()}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"rst19-sweep: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
