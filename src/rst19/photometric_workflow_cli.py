"""自动板解和 Gaia 经验测光命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .photometric_workflow import (
    DEFAULT_CAMERA_PIXEL_SCALE_ARCSEC,
    DEFAULT_PHOTOMETRY_MATCH_RADIUS_PX,
    run_auto_photometric_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 FITS 辅助视轴和本地 Gaia CSV 自动板解并拟合经验表观星等"
    )
    parser.add_argument("fits", type=Path, help="输入 FITS 文件")
    parser.add_argument("--catalog", type=Path, required=True, help="Gaia 兼容 CSV 星表")
    parser.add_argument("--out", type=Path, required=True, help="输出 JSON 证据")
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=DEFAULT_CAMERA_PIXEL_SCALE_ARCSEC,
        help="像元角尺度先验（arcsec/px），默认 8.5",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=3.0,
        help="板解阶段的宽匹配半径（px），默认 3.0",
    )
    parser.add_argument(
        "--photometry-match-radius",
        type=float,
        default=None,
        help=(
            "仿射 WCS 通过后的测光细匹配半径（px）；默认取 min(--match-radius, "
            f"{DEFAULT_PHOTOMETRY_MATCH_RADIUS_PX:.1f})"
        ),
    )
    parser.add_argument("--min-calibrators", type=int, default=6, help="最少光度参考星数量")
    parser.add_argument(
        "--include-all-sources",
        action="store_true",
        help="将全量检测源写入 JSON；默认只写检测统计、匹配星和逐源标定结果",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.fits.is_file():
        print(f"photometric-workflow: FITS 不存在：{args.fits}", file=sys.stderr)
        return 2
    if not args.catalog.is_file():
        print(f"photometric-workflow: 星表不存在：{args.catalog}", file=sys.stderr)
        return 2
    try:
        result = run_auto_photometric_workflow(
            args.fits,
            args.catalog,
            pixel_scale_arcsec=args.pixel_scale,
            match_radius_px=args.match_radius,
            photometry_match_radius_px=args.photometry_match_radius,
            photometric_min_calibrators=args.min_calibrators,
            progress=lambda value, label: print(f"{value:5.1f}% · {label}", file=sys.stderr),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                result.as_dict(include_all_sources=args.include_all_sources),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"photometric-workflow: {exc}", file=sys.stderr)
        return 2
    print(
        f"{result.status}: {result.reason} · evidence -> {args.out.resolve()}",
        file=sys.stderr,
    )
    return 0 if result.calibrated else 3


if __name__ == "__main__":
    raise SystemExit(main())
