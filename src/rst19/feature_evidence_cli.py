"""类别形态、15 帧持久性和留一 PSF 证据矩阵命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_evidence_matrix import (
    build_feature_evidence_matrix,
    write_feature_evidence_matrix_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="合并各特征类别的形态、15 帧持久性和留一 PSF 证据")
    parser.add_argument("--source-morphology", type=Path, required=True, help="source_feature_morphology.csv")
    parser.add_argument("--feature-cross", type=Path, required=True, help="feature_cross_audit.csv")
    parser.add_argument("--psf-leaveout", type=Path, required=True, help="feature_psf_leaveout.csv")
    parser.add_argument(
        "--psf-spatial",
        type=Path,
        default=None,
        help="可选 source_feature_psf_spatial.csv，用于合并局部 PSF 控制字段",
    )
    parser.add_argument("--correlation-threshold", type=float, default=0.8, help="PSF 形状审计线，不是恒星概率")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-evidence-matrix"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_feature_evidence_matrix(
            args.source_morphology,
            args.feature_cross,
            args.psf_leaveout,
            correlation_threshold=args.correlation_threshold,
            psf_spatial_path=args.psf_spatial,
        )
        output = write_feature_evidence_matrix_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-evidence: 矩阵生成失败：{exc}", file=sys.stderr)
        return 2

    print(f"rst19-feature-evidence: 矩阵已写入 {output}")
    print(result.conclusion)
    for row in result.rows:
        print(
            f"{row.feature_class}: pattern={row.evidence_pattern} "
            f"candidate_persistence={row.candidate_persistence_fraction} "
            f"quality_response={row.quality_response_fraction} "
            f"candidate_same_class={row.candidate_same_class_response_fraction} "
            f"quality_same_class={row.quality_same_class_response_fraction} "
            f"leaveout_corr={row.leaveout_correlation_median} "
            f"spatial_local_corr={row.spatial_local_correlation_median} "
            f"audit_stage={row.recommended_audit_stage} "
            f"counting_policy={row.counting_policy}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
