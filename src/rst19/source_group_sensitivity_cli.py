"""父源组半径敏感性审计的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .source_group_sensitivity import (
    run_source_group_sensitivity,
    write_source_group_sensitivity_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较不同父源组半径下的候选分组和目标分流，不读取 FITS"
    )
    parser.add_argument(
        "catalog_csv",
        type=Path,
        help="rst19-sources 导出的 source_catalog.csv",
    )
    parser.add_argument(
        "--group-radius",
        dest="group_radii",
        type=float,
        action="append",
        default=None,
        help="待比较的组半径（可重复传入；默认 2.5/3/3.5/4 px）",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        action="append",
        default=[],
        help="指定要回查的 detection_id；可重复传入",
    )
    parser.add_argument("--psf-fwhm", type=float, default=2.0)
    parser.add_argument("--delta-bic-min", type=float, default=10.0)
    parser.add_argument("--component-snr-min", type=float, default=5.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="半径配置并行数；默认 1，源表-only 扫描可显式提高",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/source-group-radius-sensitivity"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_source_group_sensitivity(
            args.catalog_csv,
            group_radii_px=tuple(args.group_radii or (2.5, 3.0, 3.5, 4.0)),
            target_ids=args.target_id,
            psf_fwhm=args.psf_fwhm,
            delta_bic_min=args.delta_bic_min,
            component_snr_min=args.component_snr_min,
            workers=args.workers,
        )
        output = write_source_group_sensitivity_artifacts(result, args.out_dir)
        print(f"source group radius sensitivity: {output}")
        for row in result.rows:
            print(
                f"radius={row.group_radius_px:g} sources={row.source_count:,} "
                f"groups={row.group_count:,} multi={row.multi_member_group_count:,} "
                f"unresolved={row.unresolved_group_count:,} "
                f"independent_candidates={row.independent_group_candidate_count:,}"
            )
        for target in result.targets:
            print(
                f"radius={target.group_radius_px:g} target={target.target_detection_ids} "
                f"group={target.group_id} members={target.group_detection_ids} "
                f"classification={target.classification}"
            )
        return 0
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-source-group-radius-sensitivity: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
