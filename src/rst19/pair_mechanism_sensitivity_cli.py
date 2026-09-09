"""几何条件化近邻机制窗口敏感性审计的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pair_mechanism_sensitivity import (
    DEFAULT_PAIR_MECHANISM_SENSITIVITY_CONFIGS,
    PairMechanismSensitivityConfiguration,
    run_pair_mechanism_sensitivity,
    write_pair_mechanism_sensitivity_artifacts,
)


def _parse_configuration(raw: str) -> PairMechanismSensitivityConfiguration:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--configuration format is name,centroid_max,peak_min,peak_max")
    try:
        return (parts[0], float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError as exc:
        raise ValueError("--configuration distances must be numeric") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="重复多组质心/整数峰几何窗口，检查近邻机制背景的敏感性"
    )
    parser.add_argument("catalog_csv", type=Path, help="包含当前候选源的 source_catalog.csv")
    parser.add_argument(
        "--target-id",
        type=int,
        action="append",
        default=[],
        help="目标 detection_id；必须重复传入两次",
    )
    parser.add_argument(
        "--configuration",
        action="append",
        default=[],
        help="自定义窗口 name,centroid_max,peak_min,peak_max；可重复传入，省略则使用默认五组",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/pair-mechanism-sensitivity"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configurations = (
            tuple(_parse_configuration(raw) for raw in args.configuration)
            if args.configuration
            else DEFAULT_PAIR_MECHANISM_SENSITIVITY_CONFIGS
        )
        result = run_pair_mechanism_sensitivity(
            args.catalog_csv,
            args.target_id,
            configurations=configurations,
        )
        output = write_pair_mechanism_sensitivity_artifacts(result, args.out_dir)
        print(result.conclusion)
        print(f"输出目录: {output}")
        for row in result.rows:
            print(
                f"{row.configuration}: pairs={row.pair_count} "
                f"same={row.same_feature_class_count} cross={row.cross_feature_class_count} "
                f"both_quality={row.both_quality_count}"
            )
        return 0
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-pair-mechanism-sensitivity: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
