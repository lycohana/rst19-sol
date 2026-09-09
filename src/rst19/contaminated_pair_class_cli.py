"""按特征类别选择污染锚点并运行双源注入审计。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contaminated_pair_class import (
    CLASS_SELECTION_METRICS,
    CLASS_SELECTION_MODES,
    run_feature_class_contaminated_pair_audit,
    write_feature_class_contaminated_pair_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 source_catalog.csv 按类别选择代表锚点，复用统一的真实污染背景双源注入审计"
    )
    parser.add_argument("frame", type=Path, help="原始 FITS 文件")
    parser.add_argument("--source-catalog", type=Path, required=True, help="rst19-sources 输出的 source_catalog.csv")
    parser.add_argument(
        "--feature-class",
        action="append",
        dest="feature_classes",
        help="只审计指定类别；可重复传入，默认审计全部有效类别",
    )
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--selection-metric", choices=CLASS_SELECTION_METRICS, default="filter_snr")
    parser.add_argument("--selection-mode", choices=CLASS_SELECTION_MODES, default="median")
    parser.add_argument("--separations-px", type=float, nargs="+", default=(2.738, 4.123))
    parser.add_argument("--ratios", type=float, nargs="+", default=(0.143, 1.0), help="secondary/primary")
    parser.add_argument("--total-peak-adu", type=float, default=4096.0)
    parser.add_argument("--psf-fwhm", type=float, default=2.0)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument(
        "--analysis-scope",
        choices=("full_frame", "local_roi"),
        default="full_frame",
        help="注入后检测范围；local_roi 只用于研究加速，计数明确标为 ROI 局部",
    )
    parser.add_argument("--roi-half-size-px", type=int, default=192)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/contaminated-pair-class-audit"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def report(index: int, total: int) -> None:
        print(f"contaminated pair class audit: {index}/{total}", file=sys.stderr, flush=True)

    try:
        result = run_feature_class_contaminated_pair_audit(
            args.frame,
            args.source_catalog,
            feature_classes=args.feature_classes,
            per_class=args.per_class,
            selection_metric=args.selection_metric,
            selection_mode=args.selection_mode,
            separations_px=args.separations_px,
            secondary_to_primary_ratios=args.ratios,
            total_peak_excess_adu=args.total_peak_adu,
            psf_fwhm=args.psf_fwhm,
            max_sources=args.max_sources,
            analysis_scope=args.analysis_scope,
            roi_half_size_px=args.roi_half_size_px,
            progress=report,
        )
        output = write_feature_class_contaminated_pair_artifacts(result, args.out_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-contaminated-pair-class: {exc}", file=sys.stderr)
        return 2

    print(f"contaminated pair class audit: {output}")
    print(result.injection.conclusion)
    for row in result.selected_rows:
        print(
            f"{row.feature_class}: ID {row.detection_id}, "
            f"{row.selection_metric}={row.metric_value:.4g}, "
            f"quality={row.quality_passed}, flags={row.flags or 'none'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
