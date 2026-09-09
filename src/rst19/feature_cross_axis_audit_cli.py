"""首要特征类别跨证据轴审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_cross_axis_audit import (
    run_feature_cross_axis_audit,
    write_feature_cross_axis_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对齐源目录、raw 局部证据、raw 峰审计和逐帧响应，检查证据轴冲突"
    )
    parser.add_argument("catalog", type=Path, help="source_catalog.csv")
    parser.add_argument("raw_summary", type=Path, help="stratified_raw_source_summary.csv")
    parser.add_argument("peak_consistency", type=Path, help="source_peak_consistency.csv")
    parser.add_argument(
        "temporal_sources",
        type=Path,
        help="feature_temporal_consistency_sources.csv",
    )
    parser.add_argument(
        "--feature-class",
        action="append",
        default=[],
        help="只汇总指定 feature_class；可重复传入，默认使用抽样中的全部类别",
    )
    parser.add_argument(
        "--detection-id",
        action="append",
        type=int,
        default=[],
        help="只审计指定 detection_id；可重复传入，默认使用全部抽样源",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-cross-axis-audit"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_cross_axis_audit(
            args.catalog,
            args.raw_summary,
            args.peak_consistency,
            args.temporal_sources,
            feature_classes=args.feature_class or None,
            detection_ids=args.detection_id or None,
        )
        output = write_feature_cross_axis_artifacts(result, args.out_dir)
    except (OSError, ValueError, TypeError) as exc:
        print(f"rst19-feature-cross-axis-audit: {exc}", file=sys.stderr)
        return 2

    print(f"rst19-feature-cross-axis-audit: 结果已写入 {output}")
    print(
        f"sources={len(result.source_rows)} classes={len(result.class_summaries)} "
        f"patterns={len(result.pattern_summaries)}"
    )
    for row in result.class_summaries:
        print(
            f"{row.feature_class}: raw_peak={row.raw_local_maximum_fraction:.1%} "
            f"value_anomaly={row.value_domain_anomaly_fraction:.1%} "
            f"rMAD={row.temporal_flux_snr_relative_mad_median:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
