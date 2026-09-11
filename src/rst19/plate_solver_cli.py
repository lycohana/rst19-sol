"""用本地 FITS 检测点和本地星表执行显式星对几何解算。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import load_catalog_csv
from .detection import detect_sources
from .fits import auxiliary_mask, read_fits
from .plate_solver import solve_plate
from .wcs import TangentPlaneWCS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用检测点与本地 Gaia/星表 CSV 的星对几何候选估计图像 WCS"
    )
    parser.add_argument("frame", type=Path, help="输入 FITS 文件")
    parser.add_argument("--catalog", type=Path, required=True, help="本地 CatalogSource 兼容 CSV")
    parser.add_argument("--center-ra", type=float, required=True, help="光轴先验赤经（度）")
    parser.add_argument("--center-dec", type=float, required=True, help="光轴先验赤纬（度）")
    parser.add_argument("--pixel-scale-arcsec", type=float, required=True, help="像元角尺度先验（角秒/像素）")
    parser.add_argument("--epoch", type=float, help="星表传播历元；省略则不传播")
    parser.add_argument("--threshold-sigma", type=float, default=8.0, help="检测候选阈值，默认 8σ")
    parser.add_argument("--min-distance", type=int, default=4, help="候选峰最小距离，默认 4 px")
    parser.add_argument("--aperture-radius", type=int, default=4, help="孔径半径，默认 4 px")
    parser.add_argument("--psf-fwhm", type=float, default=2.0, help="PSF FWHM，默认 2 px")
    parser.add_argument("--max-detections", type=int, default=4000, help="检测阶段最多保留候选，默认 4000")
    parser.add_argument("--max-image-points", type=int, default=80, help="几何解算最多使用图像点数")
    parser.add_argument("--max-catalog-points", type=int, default=160, help="几何解算最多使用星表点数")
    parser.add_argument("--scale-tolerance", type=float, default=0.04, help="尺度先验相对容差，默认 4%%")
    parser.add_argument("--match-radius-px", type=float, default=3.0, help="验收匹配半径，默认 3 px")
    parser.add_argument("--min-matches", type=int, default=8, help="最少唯一匹配数，默认 8")
    parser.add_argument("--min-coverage-area", type=float, default=0.02, help="最小视场覆盖面积比例")
    parser.add_argument("--max-rms-px", type=float, default=2.0, help="最大匹配 RMS，默认 2 px")
    parser.add_argument("--out", type=Path, required=True, help="输出 JSON 路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frame = read_fits(args.frame)
        catalog = load_catalog_csv(args.catalog)
        epoch = args.epoch
        reference_wcs = TangentPlaneWCS(
            center_ra_deg=args.center_ra,
            center_dec_deg=args.center_dec,
            pixel_scale_arcsec=args.pixel_scale_arcsec,
            crpix_x=(frame.width - 1) / 2.0,
            crpix_y=(frame.height - 1) / 2.0,
        )
        detection = detect_sources(
            frame.data,
            mask=auxiliary_mask(frame.data.shape),
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            max_sources=args.max_detections,
            psf_fwhm=args.psf_fwhm,
            min_flux_snr=5.0,
            proposal_mode="hybrid",
        )
        result = solve_plate(
            detection.quality_sources,
            catalog,
            reference_wcs,
            epoch=epoch,
            image_shape=frame.data.shape,
            scale_tolerance=args.scale_tolerance,
            match_radius_px=args.match_radius_px,
            min_matches=args.min_matches,
            min_coverage_area=args.min_coverage_area,
            max_rms_residual_px=args.max_rms_px,
            max_image_points=args.max_image_points,
            max_catalog_points=args.max_catalog_points,
        )
        payload = {
            "frame": str(frame.path),
            "catalog": str(args.catalog),
            "detection": {
                "candidate_count": detection.candidate_count,
                "returned_count": detection.returned_count,
                "quality_count": detection.star_count,
            },
            "reference_wcs": reference_wcs.as_dict(),
            "plate_solution": result.as_dict(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(
            f"plate-solver: {result.status}; {result.reason}; output={args.out}",
            file=sys.stderr,
        )
        return 0 if result.valid else 3
    except (OSError, ValueError) as exc:
        print(f"plate-solver: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

