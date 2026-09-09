"""源级坐标几何审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .source_geometry_audit import (
    DEFAULT_CONTROL_CLASS,
    run_source_geometry_audit,
    write_source_geometry_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读取 source_catalog，审计各特征类别的 detector 坐标细长性"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 导出的 source_catalog.csv")
    parser.add_argument(
        "--target-class",
        action="append",
        default=[],
        help="需要审计的 feature_class；可重复传入，默认审计全部类别",
    )
    parser.add_argument(
        "--control-class",
        default=DEFAULT_CONTROL_CLASS,
        help="等样本量抽样对照类别，默认 compact_quality",
    )
    parser.add_argument("--trials", type=int, default=5000, help="每个类别的对照抽样次数")
    parser.add_argument("--seed", type=int, default=1909, help="对照抽样随机种子")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/source-geometry-audit"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_source_geometry_audit(
            args.catalog,
            target_classes=args.target_class or None,
            control_class=args.control_class,
            control_trial_count=args.trials,
            random_seed=args.seed,
        )
        output = write_source_geometry_artifacts(result, args.out_dir)
    except (OSError, ValueError, TypeError) as exc:
        print(f"rst19-source-geometry-audit: {exc}", file=sys.stderr)
        return 2

    print(f"rst19-source-geometry-audit: 结果已写入 {output}")
    for row in result.summaries:
        tail = (
            "-"
            if row.control_axis_ratio_upper_tail_fraction is None
            else f"{row.control_axis_ratio_upper_tail_fraction:.4f}"
        )
        print(
            f"{row.feature_class}: n={row.candidate_count} "
            f"axis_ratio={row.pca_axis_ratio:.3f} "
            f"residual_p50/p90={row.perpendicular_residual_median_px:.3f}/"
            f"{row.perpendicular_residual_p90_px:.3f} control_tail={tail}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
