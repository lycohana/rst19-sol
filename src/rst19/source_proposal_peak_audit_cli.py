"""提议器来源与 raw 峰交叉审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .source_proposal_peak_audit import (
    run_source_proposal_peak_audit,
    write_source_proposal_peak_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="连接 source_catalog 与 raw 峰一致性表，按特征类别和提议器来源汇总"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument(
        "peak_consistency",
        type=Path,
        help="rst19-source-peak-consistency 生成的 source_peak_consistency.csv",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        action="append",
        default=[],
        help="指定需要导出来源子组相对位置的 detection_id；可重复传入",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/source-proposal-peak-audit"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_source_proposal_peak_audit(
            args.catalog,
            args.peak_consistency,
            target_ids=args.target_id,
        )
        output = write_source_proposal_peak_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-source-proposal-peak-audit: {exc}", file=sys.stderr)
        return 2

    print(f"rst19-source-proposal-peak-audit: 结果已写入 {output}")
    for row in result.summaries:
        print(
            f"{row.feature_class}/{row.method_group}: n={row.candidate_count} "
            f"quality={row.quality_count} raw_local_max_r1="
            f"{row.raw_local_maximum_count_r1}/{row.candidate_count}"
        )
    print(result.conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
