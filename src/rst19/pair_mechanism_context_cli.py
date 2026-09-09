"""几何条件化近邻机制审计命令行入口。"""

from __future__ import annotations

import argparse

from .pair_mechanism_context import (
    run_pair_mechanism_context,
    write_pair_mechanism_context_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把指定候选 pair 放回相同峰/质心几何条件的 detector-level 机制背景中。"
    )
    parser.add_argument("catalog", help="rst19-sources 导出的 source_catalog.csv")
    parser.add_argument(
        "--target-id",
        action="append",
        type=int,
        required=True,
        help="目标 detection_id；必须重复传入两次",
    )
    parser.add_argument("--max-centroid-distance", type=float, default=3.5)
    parser.add_argument("--peak-distance-min", type=float, default=3.5)
    parser.add_argument("--peak-distance-max", type=float, default=4.8)
    parser.add_argument("--out-dir", default="tmp/pair-mechanism-context")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_pair_mechanism_context(
        args.catalog,
        args.target_id,
        max_centroid_distance_px=args.max_centroid_distance,
        peak_distance_min_px=args.peak_distance_min,
        peak_distance_max_px=args.peak_distance_max,
    )
    output = write_pair_mechanism_context_artifacts(result, args.out_dir)
    print(result.conclusion)
    print(f"输出目录: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
