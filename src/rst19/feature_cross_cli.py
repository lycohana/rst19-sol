"""单帧特征与 15 帧持久性联合反例审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .experiments import run_feature_cross_audit, write_feature_cross_audit_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="连接 source_feature_summary.csv 与 sequence_feature_persistence.csv，输出跨类别反例统计"
    )
    parser.add_argument("source_summary_csv", type=Path, help="rst19-sources 生成的 source_feature_summary.csv")
    parser.add_argument(
        "persistence_csv",
        type=Path,
        help="rst19-feature-sequence 生成的 sequence_feature_persistence.csv",
    )
    parser.add_argument(
        "--class-transition-csv",
        type=Path,
        help="可选的 sequence_feature_class_transition.csv；提供后追加任意/同类响应率",
    )
    parser.add_argument(
        "--high-snr-threshold",
        type=float,
        default=10.0,
        help="标记‘高 SNR 但质量层拒绝’的最大 flux_SNR 阈值，默认 10",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-cross-audit"),
        help="CSV/JSON/PNG 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_cross_audit(
            args.source_summary_csv,
            args.persistence_csv,
            high_snr_threshold=args.high_snr_threshold,
            class_transition_path=args.class_transition_csv,
        )
        output = write_feature_cross_audit_artifacts(result, args.out_dir)
        print(f"rst19-feature-cross: 审计已写入 {output}")
        print(
            f"classes={len(result.rows)} frames={result.frame_count} "
            f"required_presence={result.required_presence} "
            f"high_snr_threshold={result.high_snr_threshold:g}"
        )
        for row in result.rows:
            candidate_fraction = "-" if row.candidate_presence_fraction is None else f"{row.candidate_presence_fraction:.1%}"
            quality_fraction = "-" if row.quality_presence_fraction is None else f"{row.quality_presence_fraction:.1%}"
            print(
                f"{row.feature_class}: candidate={candidate_fraction} quality={quality_fraction} "
                f"high_snr_rejected={row.high_snr_rejected}"
            )
        print(result.conclusion)
        return 0
    except (OSError, ValueError) as exc:
        print(f"rst19-feature-cross: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
