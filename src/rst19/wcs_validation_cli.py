"""15 帧逐帧局部 WCS 验证命令。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .catalog import load_catalog_csv
from .wcs import TangentPlaneWCS
from .wcs_validation import run_sequence_wcs_validation, write_wcs_validation_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用用户提供的离线星表，对一组 FITS 做逐帧局部 WCS 验证")
    parser.add_argument("data_dir", type=Path, help="包含 FITS 序列的目录")
    parser.add_argument("catalog_csv", type=Path, help="用户提供的离线 CSV 星表")
    parser.add_argument("--center-ra", type=float, required=True, help="先验光轴 RA / deg")
    parser.add_argument("--center-dec", type=float, required=True, help="先验光轴 DEC / deg")
    parser.add_argument("--pixel-scale", type=float, required=True, help="先验像元角尺度 / arcsec px^-1")
    parser.add_argument("--crpix-x", type=float, default=2048.0, help="先验 CRPIX X，默认按 4096 图像中心")
    parser.add_argument("--crpix-y", type=float, default=2048.0, help="先验 CRPIX Y，默认按 4096 图像中心")
    parser.add_argument("--rotation", type=float, default=0.0, help="先验旋转 / deg")
    parser.add_argument("--parity", type=int, choices=(-1, 1), default=1, help="RA 轴奇偶性")
    parser.add_argument("--match-radius", type=float, default=3.0, help="先验 WCS 匹配半径 / px")
    parser.add_argument("--min-matches", type=int, default=6, help="每帧最少唯一匹配点数")
    parser.add_argument("--epoch", type=float, help="星表传播目标历元，Julian year")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="匹配滤波候选阈值 sigma")
    parser.add_argument("--min-distance", type=int, default=4, help="候选峰最小间距 pixel")
    parser.add_argument("--aperture-radius", type=int, default=4, help="孔径测光半径 pixel")
    parser.add_argument("--psf-fwhm", type=float, default=3.0, help="Gaussian 匹配滤波 FWHM pixel")
    parser.add_argument("--background-box-size", type=int, default=128, help="局部背景/RMS 网格大小 pixel")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量层通量 SNR 下限")
    parser.add_argument("--max-sources", type=int, help="可选返回源上限；留空表示全量")
    parser.add_argument("--keep-linear-artifacts", action="store_true", help="不标记线状结构为 LINE_ARTIFACT")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp") / "wcs-validation", help="CSV/JSON/PNG 输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(sorted(args.data_dir.glob("*.fits")))
    if not paths:
        print(f"rst19-wcs-validation: no FITS files in {args.data_dir}", file=sys.stderr)
        return 2
    try:
        catalog = load_catalog_csv(args.catalog_csv)
        wcs = TangentPlaneWCS(
            center_ra_deg=args.center_ra,
            center_dec_deg=args.center_dec,
            pixel_scale_arcsec=args.pixel_scale,
            crpix_x=args.crpix_x,
            crpix_y=args.crpix_y,
            rotation_deg=args.rotation,
            parity=args.parity,
        )

        def progress(index: int, total: int) -> None:
            print(f"matching frame {index}/{total}", file=sys.stderr)

        report = run_sequence_wcs_validation(
            paths,
            catalog,
            wcs,
            detector_kwargs={
                "threshold_sigma": args.threshold_sigma,
                "min_distance": args.min_distance,
                "aperture_radius": args.aperture_radius,
                "max_sources": args.max_sources,
                "psf_fwhm": args.psf_fwhm,
                "background_box_size": args.background_box_size,
                "min_flux_snr": args.min_flux_snr,
                "reject_linear_artifacts": not args.keep_linear_artifacts,
            },
            match_radius_px=args.match_radius,
            epoch=args.epoch,
            min_matches=args.min_matches,
            progress=progress,
        )
        output = write_wcs_validation_artifacts(report, args.out_dir)
    except (OSError, ValueError) as exc:
        print(f"rst19-wcs-validation: {exc}", file=sys.stderr)
        return 2
    print(f"wcs validation: {output}")
    print(
        f"frames={report.frame_count} validated={report.validated_count} "
        f"validation_ratio={report.validation_ratio:.2%} catalog={report.catalog_source_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

