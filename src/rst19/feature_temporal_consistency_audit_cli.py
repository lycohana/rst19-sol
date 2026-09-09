"""各特征类别原始逐帧响应一致性审计的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_temporal_consistency_audit import (
    FEATURE_CLASS_SOURCE_CHOICES,
    run_feature_temporal_consistency_audit,
    write_feature_temporal_consistency_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="连接 source_catalog 与逐帧原始孔径摘要，审计各类别的时间一致性"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 导出的 source_catalog.csv")
    parser.add_argument(
        "frame_metrics",
        type=Path,
        help="stratified_raw_source_frame_metrics.csv 或同字段逐帧摘要",
    )
    parser.add_argument(
        "--feature-class",
        action="append",
        default=[],
        help="需要汇总的 feature_class；可重复传入，默认采用逐帧表中的全部类别",
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=15,
        help="预期帧数，默认 15；仅用于记录覆盖比例",
    )
    parser.add_argument(
        "--feature-class-source",
        choices=FEATURE_CLASS_SOURCE_CHOICES,
        default="catalog",
        help="类别来源：catalog 默认要求同产物一致；frame 用于显式采用逐帧表的本地产物类别",
    )
    parser.add_argument(
        "--detection-id",
        action="append",
        type=int,
        default=[],
        help="只审计指定 detection_id；可重复传入，默认审计逐帧表中的全部源",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-temporal-consistency-audit"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_temporal_consistency_audit(
            args.catalog,
            args.frame_metrics,
            expected_frame_count=args.expected_frames,
            feature_classes=args.feature_class or None,
            feature_class_source=args.feature_class_source,
            detection_ids=args.detection_id or None,
        )
        output = write_feature_temporal_consistency_artifacts(result, args.out_dir)
    except (OSError, ValueError, TypeError) as exc:
        print(f"rst19-feature-temporal-consistency-audit: {exc}", file=sys.stderr)
        return 2

    print(f"rst19-feature-temporal-consistency-audit: 结果已写入 {output}")
    print(
        "类别摘要："
        + ", ".join(
            f"{row.feature_class}={row.source_count}"
            for row in result.class_summaries
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
