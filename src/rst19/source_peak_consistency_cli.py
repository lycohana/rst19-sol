"""候选峰与原始像素局部峰一致性审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .source_peak_consistency import run_source_peak_consistency, write_source_peak_consistency_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把现有 source_catalog 候选放回原始 FITS，检查 3x3/5x5/7x7 raw 局部峰一致性"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument("fits_path", type=Path, help="与源表对应的原始 FITS 文件")
    parser.add_argument(
        "--target-id",
        type=int,
        action="append",
        default=[],
        help="指定需要额外导出局部最大位置的 detection_id；可重复传入",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/source-peak-consistency"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_source_peak_consistency(
            args.catalog,
            args.fits_path,
            target_ids=args.target_id,
        )
        output = write_source_peak_consistency_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-source-peak-consistency: {exc}", file=sys.stderr)
        return 2

    print(f"rst19-source-peak-consistency: 结果已写入 {output}")
    for row in result.class_summaries:
        print(
            f"{row.feature_class}: n={row.candidate_count} "
            f"raw_local_max(r1/r2/r3)="
            f"{row.raw_local_maximum_count_r1}/{row.raw_local_maximum_count_r2}/{row.raw_local_maximum_count_r3}"
        )
    print(result.conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
