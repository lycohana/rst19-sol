"""近邻候选跨帧强制测光共变审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pair_flux_covariance import (
    PAIR_FLUX_METRICS,
    load_forced_stability_flux_csv,
    run_pair_flux_covariance_audit,
    write_pair_flux_covariance_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较一对候选在 15 帧强制测光中的共变，并与其它源对照"
    )
    parser.add_argument("forced_metrics_csv", type=Path, help="forced_stability_frame_metrics.csv")
    parser.add_argument("--primary-id", type=int, required=True, help="主候选 detection_id")
    parser.add_argument("--secondary-id", type=int, required=True, help="副候选 detection_id")
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        choices=PAIR_FLUX_METRICS,
        help="指定测量列；可重复传入，默认同时审计固定和局部 flux SNR",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/pair-flux-covariance-audit"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = load_forced_stability_flux_csv(args.forced_metrics_csv)
        result = run_pair_flux_covariance_audit(
            rows,
            args.primary_id,
            args.secondary_id,
            metrics=tuple(args.metrics or PAIR_FLUX_METRICS),
        )
        output = write_pair_flux_covariance_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-pair-covariance-audit: {exc}", file=sys.stderr)
        return 2

    print(f"rst19-pair-covariance-audit: 审计已写入 {output}")
    print(result.conclusion)
    for metric in result.metrics:
        print(
            f"{metric.metric_name}: pearson={metric.primary_pearson} "
            f"normalized={metric.frame_median_normalized_pearson} "
            f"control_tail={metric.normalized_upper_tail_fraction}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
