"""类别证据轴同步关系的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_evidence_axes import run_feature_evidence_axes, write_feature_evidence_axes_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从已有 feature_evidence_matrix.csv 汇总候选位置与质量响应的证据轴关系"
    )
    parser.add_argument("matrix", type=Path, help="rst19-feature-evidence 生成的 feature_evidence_matrix.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-evidence-axes"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_evidence_axes(args.matrix)
        output = write_feature_evidence_axes_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-evidence-axes: {exc}", file=sys.stderr)
        return 2

    print(f"rst19-feature-evidence-axes: 结果已写入 {output}")
    print(f"rst19-feature-evidence-axes: 敏感性配置 {len({row.configuration for row in result.sensitivity_rows})} 组")
    print(result.conclusion)
    for row in result.rows:
        conditional = (
            "NA"
            if row.quality_response_given_candidate_persistent_fraction is None
            else f"{row.quality_response_given_candidate_persistent_fraction:.4f}"
        )
        print(
            f"{row.feature_class}: pattern={row.evidence_axis_pattern} "
            f"candidate_persistent={row.candidate_persistent_count} "
            f"quality_response={row.quality_response_count} Q|P={conditional}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
