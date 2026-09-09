"""孔径通量/峰值比类别审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .aperture_peak_ratio import (
    run_aperture_peak_ratio_audit,
    write_aperture_peak_ratio_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按特征类别审计 source_catalog 的 flux/peak，不生成物理恒星概率"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument(
        "--target-ids",
        nargs="*",
        default=["82931", "82934"],
        help="需要输出类别分位的候选 ID",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/aperture-peak-ratio-audit"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_aperture_peak_ratio_audit(args.catalog, target_ids=args.target_ids)
        output = write_aperture_peak_ratio_artifacts(result, args.out_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-aperture-peak-ratio: {exc}", file=sys.stderr)
        return 2
    print(f"aperture/peak ratio audit: {output}")
    for row in result.target_rows:
        print(
            f"{row.detection_id}: {row.feature_class}, "
            f"flux/peak={row.flux_to_peak:.6f}, "
            f"class percentile={row.class_percentile:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
