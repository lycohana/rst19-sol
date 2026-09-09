"""各特征类别原始局部证据审计的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_raw_evidence_audit import (
    RAW_EVIDENCE_METRICS,
    run_feature_raw_evidence_audit,
    write_feature_raw_evidence_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="连接 source_catalog 与原始 FITS 抽样摘要，比较各特征类别的局部证据"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 导出的 source_catalog.csv")
    parser.add_argument(
        "raw_summary",
        type=Path,
        help="stratified_raw_source_summary.csv；由原始 FITS 抽样审计生成",
    )
    parser.add_argument(
        "--feature-class",
        action="append",
        default=[],
        help="需要汇总的 feature_class；可重复传入，默认采用抽样表中的全部类别",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-raw-evidence-audit"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_raw_evidence_audit(
            args.catalog,
            args.raw_summary,
            feature_classes=args.feature_class or None,
        )
        output = write_feature_raw_evidence_artifacts(result, args.out_dir)
    except (OSError, ValueError, TypeError) as exc:
        print(f"rst19-feature-raw-evidence-audit: {exc}", file=sys.stderr)
        return 2

    print(f"rst19-feature-raw-evidence-audit: 结果已写入 {output}")
    print(
        "类别样本数："
        + ", ".join(f"{name}={result.sample_counts[name]}" for name in result.feature_classes)
    )
    print(f"证据轴记录：{len(result.rows)} 条；可用字段：{', '.join(RAW_EVIDENCE_METRICS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
