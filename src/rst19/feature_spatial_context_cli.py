"""特征类别空间条件化审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_spatial_context import (
    SPATIAL_CONTEXT_METRIC_FIELDS,
    run_feature_spatial_context,
    write_feature_spatial_context_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 detector 空间条件审计特征类别与 hard-negative，不重新读取 FITS"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 导出的 source_catalog.csv")
    parser.add_argument("--target-id", type=int, action="append", default=[], help="显式追加候选 ID，可重复传入")
    parser.add_argument("--top-n", type=int, default=5, help="每个类别按指标保留的落选代表数")
    parser.add_argument(
        "--metric",
        choices=SPATIAL_CONTEXT_METRIC_FIELDS,
        default="flux_snr",
        help="hard-negative 排名指标；只影响代表抽样，不改变质量层",
    )
    parser.add_argument("--grid-size", type=int, default=4, help="空间网格边数")
    parser.add_argument("--image-width", type=int, default=4096, help="detector 宽度（像素）")
    parser.add_argument("--image-height", type=int, default=4096, help="detector 高度（像素）")
    parser.add_argument("--edge-margin-px", type=float, default=16.0, help="边缘诊断距离（像素）")
    parser.add_argument("--high-flux-snr", type=float, default=10.0, help="高 SNR 空间切片阈值")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-spatial-context"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_spatial_context(
            args.catalog,
            image_width=args.image_width,
            image_height=args.image_height,
            grid_size=args.grid_size,
            edge_margin_px=args.edge_margin_px,
            high_flux_snr_threshold=args.high_flux_snr,
            top_n=args.top_n,
            metric=args.metric,
            target_ids=args.target_id,
        )
        output = write_feature_spatial_context_artifacts(result, args.out_dir)
        print(f"rst19-feature-spatial-context: 空间条件化审计已写入 {output}")
        for summary in result.summaries:
            hot = (
                f"r{summary.high_snr_hot_grid_row}c{summary.high_snr_hot_grid_column}"
                if summary.high_snr_hot_grid_row is not None
                else "-"
            )
            print(
                f"{summary.feature_class}: n={summary.class_count} "
                f"edge={summary.edge_fraction:.3f} high_snr={summary.high_snr_rejected_count} "
                f"hot={hot}"
            )
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"rst19-feature-spatial-context: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
