"""直接近邻边审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .direct_neighbor_audit import run_direct_neighbor_audit, write_direct_neighbor_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在源目录上保留直接近邻边，区分局部 pair 与传递父组"
    )
    parser.add_argument(
        "catalog_csv",
        type=Path,
        help="rst19-sources 导出的 source_catalog.csv",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        action="append",
        default=[],
        help="指定目标 pair 的两个 detection_id；可重复传入",
    )
    parser.add_argument("--psf-fwhm", type=float, default=2.0)
    parser.add_argument(
        "--pair-radius",
        type=float,
        default=None,
        help="直接近邻质心半径；默认 max(2.5, 1.5×PSF FWHM)",
    )
    parser.add_argument("--delta-bic-min", type=float, default=10.0)
    parser.add_argument("--component-snr-min", type=float, default=5.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/direct-neighbor-audit"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_direct_neighbor_audit(
            args.catalog_csv,
            target_ids=args.target_id,
            psf_fwhm=args.psf_fwhm,
            pair_radius_px=args.pair_radius,
            delta_bic_min=args.delta_bic_min,
            component_snr_min=args.component_snr_min,
        )
        output = write_direct_neighbor_artifacts(result, args.out_dir)
        print(f"direct neighbor audit: {output}")
        print(
            f"sources={result.source_count:,} quality={result.quality_count:,} "
            f"pairs={result.direct_pair_count:,} unresolved={result.unresolved_pair_count:,} "
            f"independent_candidates={result.independent_pair_candidate_count:,}"
        )
        if result.target is not None:
            target = result.target
            print(
                f"target={target.detection_id_a}|{target.detection_id_b} "
                f"distance={target.centroid_distance_px:.3f}px "
                f"in_radius={target.in_pair_radius} classification={target.classification}"
            )
        return 0
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-direct-neighbor-audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
