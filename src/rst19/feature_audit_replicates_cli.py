"""多次重复特征类别注入控制的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_audit_replicates import (
    run_feature_audit_replicates,
    write_feature_audit_replicate_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="重复运行正/负特征类别注入控制并分开聚合回收与泄漏")
    parser.add_argument("--trials", type=int, default=16, help="随机背景重复次数")
    parser.add_argument("--seed", type=int, default=19019, help="首个随机种子")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-audit-replicates"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def report_progress(completed: int, total: int) -> None:
        print(f"feature-audit-replicates: trial {completed}/{total}", file=sys.stderr)

    try:
        result = run_feature_audit_replicates(trials=args.trials, seed=args.seed, progress=report_progress)
        output = write_feature_audit_replicate_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-audit-replicates: 运行失败：{exc}", file=sys.stderr)
        return 2

    print(f"rst19-feature-audit-replicates: 结果已写入 {output}")
    for row in result.rows:
        if row.truth_count:
            summary = (
                f"recall={row.candidate_recall:.4f}/{row.quality_recall:.4f} "
                f"candidate_ci=[{row.candidate_recall_wilson95_low:.4f},{row.candidate_recall_wilson95_high:.4f}] "
                f"quality_ci=[{row.quality_recall_wilson95_low:.4f},{row.quality_recall_wilson95_high:.4f}]"
            )
        else:
            summary = (
                f"leak_mean={row.mean_nearby_candidate_count:.2f}/{row.mean_nearby_quality_count:.2f} "
                f"leak_max={row.max_nearby_candidate_count}/{row.max_nearby_quality_count}"
            )
        print(f"{row.feature_class}/{row.scenario}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
