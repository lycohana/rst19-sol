"""源目录最近邻间距群体控制的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .source_separation_audit import (
    run_source_separation_audit,
    write_source_separation_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统计源目录的质心/峰坐标最近邻间距，并定位指定源在总体分布中的位置"
    )
    parser.add_argument("catalog_csv", type=Path, help="包含 x/y、peak_x/peak_y 和 feature_class 的源目录")
    parser.add_argument(
        "--target-id",
        type=int,
        action="append",
        default=[],
        help="指定要回查的 detection_id；可重复传入",
    )
    parser.add_argument(
        "--radii-px",
        type=float,
        nargs="+",
        default=[3.0, 4.0, 4.5, 5.0, 6.0],
        help="最近邻半径列表，单位为像素",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/source-separation-audit"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_source_separation_audit(
            args.catalog_csv,
            target_ids=args.target_id,
            radii_px=args.radii_px,
        )
        output = write_source_separation_artifacts(result, args.out_dir)
        print(f"source separation audit: {output}")
        print(
            f"sources={result.source_count} quality={result.quality_count} "
            f"targets={len(result.targets)}"
        )
        print(f"centroid_quantiles={result.centroid_quantiles_px}")
        print(f"peak_quantiles={result.peak_quantiles_px}")
        return 0
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-source-separation-audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
