"""旗标单项与组合交互审计的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_flag_interaction import run_feature_flag_interaction, write_feature_flag_interaction_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统计 source_catalog.csv 中质量旗标的单项/两项交互，不重新读取 FITS"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument(
        "--high-snr-threshold",
        type=float,
        default=10.0,
        help="hard-negative 统计的 flux_snr 阈值，默认 10",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-flag-interaction"),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="终端显示候选数最多的前 N 个交互，产物仍保存全部交互",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_n < 1:
        print("rst19-feature-flag-interaction: --top-n must be positive", file=sys.stderr)
        return 2
    try:
        result = run_feature_flag_interaction(
            args.catalog,
            high_snr_threshold=args.high_snr_threshold,
        )
        output = write_feature_flag_interaction_artifacts(result, args.out_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-flag-interaction: {exc}", file=sys.stderr)
        return 2
    print(f"feature flag interaction: {output}")
    print(f"records={result.record_count} interactions={len(result.rows)}")
    for row in result.rows[: args.top_n]:
        print(
            f"{row.interaction}: candidates={row.candidate_count} "
            f"quality={row.quality_count} ({row.quality_fraction:.1%}) "
            f"exact={row.exact_set_count} high_snr_rejected={row.high_snr_rejected_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
