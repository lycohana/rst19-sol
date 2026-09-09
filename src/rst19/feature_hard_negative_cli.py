"""类别内高显著性落选候选审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_hard_negative import (
    HARD_NEGATIVE_METRIC_FIELDS,
    run_feature_hard_negative_audit,
    write_feature_hard_negative_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="列出各特征类别内 SNR 较高但未通过质量层的 hard-negative 候选"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 导出的 source_catalog.csv")
    parser.add_argument("--top-n", type=int, default=5, help="每个类别保留的落选候选数，默认 5")
    parser.add_argument(
        "--metric",
        choices=HARD_NEGATIVE_METRIC_FIELDS,
        default="flux_snr",
        help="类别内排序指标，默认 flux_snr",
    )
    parser.add_argument(
        "--include-other-rejected",
        action="store_true",
        help="把 other_rejected 也作为一个类别输出；默认排除汇总桶",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-hard-negative"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    excluded = () if args.include_other_rejected else ("other_rejected",)
    try:
        result = run_feature_hard_negative_audit(
            args.catalog,
            top_n=args.top_n,
            metric=args.metric,
            exclude_classes=excluded,
        )
        output = write_feature_hard_negative_artifacts(result, args.out_dir)
        print(
            f"rst19-feature-hard-negative: {len(result.summaries)} 个类别、"
            f"{len(result.rows)} 个 hard-negative 已写入 {output}"
        )
        for summary in result.summaries:
            print(
                f"{summary.feature_class}: rejected={summary.class_rejected_count} "
                f"top_id={summary.top_detection_id} top_{summary.metric}={summary.top_metric_value}"
            )
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"rst19-feature-hard-negative: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
