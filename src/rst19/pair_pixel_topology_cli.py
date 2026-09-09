"""近邻双框原始像素拓扑审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .empirical_pair import load_detection_catalog_csv
from .pair_pixel_topology import (
    run_pair_pixel_topology_audit,
    write_pair_pixel_topology_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查近邻候选是否属于同一原始正响应连通块，以及候选峰是否为 raw 局部极大值"
    )
    parser.add_argument("frame", type=Path, help="原始 FITS 文件")
    parser.add_argument("--source-catalog", type=Path, required=True, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument("--primary-id", type=int, required=True, help="主候选 detection_id")
    parser.add_argument("--secondary-id", type=int, required=True, help="副候选 detection_id")
    parser.add_argument("--patch-padding-px", type=int, default=10, help="候选峰周围局部窗口外扩像素")
    parser.add_argument("--sigma-levels", type=float, nargs="+", default=(3.0, 5.0, 8.0, 10.0))
    parser.add_argument("--raw-maximum-neighborhoods-px", type=int, nargs="+", default=(3, 5, 7))
    parser.add_argument("--raw-maximum-sigma-level", type=float, default=3.0)
    parser.add_argument("--special-negative-threshold-adu", type=float, default=-1000.0)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/pair-pixel-topology"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sources = load_detection_catalog_csv(args.source_catalog)
        by_id = {source.detection_id: source for source in sources}
        missing = [source_id for source_id in (args.primary_id, args.secondary_id) if source_id not in by_id]
        if missing:
            raise ValueError(f"source catalog does not contain detection_id={missing[0]}")
        result = run_pair_pixel_topology_audit(
            args.frame,
            by_id[args.primary_id],
            by_id[args.secondary_id],
            patch_padding_px=args.patch_padding_px,
            sigma_levels=args.sigma_levels,
            raw_maximum_neighborhoods_px=args.raw_maximum_neighborhoods_px,
            raw_maximum_sigma_level=args.raw_maximum_sigma_level,
            special_negative_threshold_adu=args.special_negative_threshold_adu,
        )
        output = write_pair_pixel_topology_artifacts(result, args.out_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-pair-pixel-topology: {exc}", file=sys.stderr)
        return 2

    print(f"pair pixel topology audit: {output}")
    print(result.conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
