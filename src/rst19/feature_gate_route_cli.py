"""质量门规则路径审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_gate_route import run_feature_gate_route, write_feature_gate_route_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按候选拆分数值质量门和结构旗标路径，不重新读取 FITS"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/feature-gate-route"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_gate_route(args.catalog)
        output = write_feature_gate_route_artifacts(result, args.out_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-gate-route: {exc}", file=sys.stderr)
        return 2
    print(f"feature gate route: {output}")
    print(f"sources={len(result.source_rows)} classes={len(result.summary_rows)}")
    for row in result.summary_rows:
        print(
            f"{row.feature_class}: quality={row.quality_count}/{row.candidate_count} "
            f"numeric={row.numeric_failure_count} structural={row.structural_flag_count} "
            f"rejected(numeric={row.rejected_numeric_only_count}, "
            f"structural={row.rejected_structural_only_count}, "
            f"mixed={row.rejected_numeric_and_structural_count}, "
            f"unexplained={row.rejected_unexplained_count})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
