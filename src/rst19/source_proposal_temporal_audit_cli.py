"""提议器来源与 15 帧时序持久性交叉审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .source_proposal_temporal_audit import (
    run_source_proposal_temporal_audit,
    write_source_proposal_temporal_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="连接 source_catalog、非紧凑逐源时序和紧凑来源子组时序汇总"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument(
        "diagnostic_sources",
        type=Path,
        help="rst19-feature-sequence 生成的 sequence_feature_diagnostic_sources.csv",
    )
    parser.add_argument(
        "subgroup_persistence",
        type=Path,
        help="rst19-feature-sequence 生成的 sequence_feature_source_subgroup_persistence.csv",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        action="append",
        default=[],
        help="指定需要导出来源子组和逐源时序的 detection_id；可重复传入",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/source-proposal-temporal-audit"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_source_proposal_temporal_audit(
            args.catalog,
            args.diagnostic_sources,
            args.subgroup_persistence,
            target_ids=args.target_id,
        )
        output = write_source_proposal_temporal_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-source-proposal-temporal-audit: {exc}", file=sys.stderr)
        return 2

    print(f"rst19-source-proposal-temporal-audit: 结果已写入 {output}")
    for row in result.summaries:
        print(
            f"{row.feature_class}/{row.method_group}: n={row.candidate_count} "
            f"candidate_ge={row.candidate_presence_ge_required_count}/{row.candidate_count} "
            f"quality_ge={row.quality_presence_ge_required_count}/{row.candidate_count} "
            f"detail={row.detail_level}"
        )
    print(result.conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
