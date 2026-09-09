"""源级特征相关性审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_correlation_audit import (
    DEFAULT_CORRELATION_FEATURES,
    run_feature_correlation_audit,
    write_feature_correlation_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="计算源级特征相关性和各类别中位数，不生成物理恒星概率"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument("--features", nargs="*", default=list(DEFAULT_CORRELATION_FEATURES))
    parser.add_argument("--strong-threshold", type=float, default=0.45)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/feature-correlation-audit"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_correlation_audit(
            args.catalog,
            features=args.features,
            strong_threshold=args.strong_threshold,
        )
        output = write_feature_correlation_artifacts(result, args.out_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-correlation: {exc}", file=sys.stderr)
        return 2
    print(f"feature correlation audit: {output}")
    for row in result.correlation_rows:
        if row.strong_absolute_correlation:
            print(f"{row.feature_a}~{row.feature_b}: rho={row.spearman_rho:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
