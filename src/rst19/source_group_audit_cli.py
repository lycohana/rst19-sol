"""候选父源组审计的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .source_group_audit import run_candidate_group_audit, write_candidate_group_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从源目录建立父源组—子候选关系，区分近邻混合组与独立候选组"
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
        help="指定要回查的 detection_id；可重复传入",
    )
    parser.add_argument("--psf-fwhm", type=float, default=2.0, help="组半径推导使用的 PSF FWHM，单位为像素")
    parser.add_argument(
        "--group-radius",
        type=float,
        default=None,
        help="显式覆盖父源组半径；默认 max(2.5, 1.5×PSF FWHM)",
    )
    parser.add_argument("--delta-bic-min", type=float, default=10.0)
    parser.add_argument("--component-snr-min", type=float, default=5.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/source-group-audit"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_candidate_group_audit(
            args.catalog_csv,
            target_ids=args.target_id,
            psf_fwhm=args.psf_fwhm,
            group_radius_px=args.group_radius,
            delta_bic_min=args.delta_bic_min,
            component_snr_min=args.component_snr_min,
        )
        output = write_candidate_group_artifacts(result, args.out_dir)
        print(f"source group audit: {output}")
        print(
            f"sources={result.source_count:,} quality={result.quality_count:,} "
            f"groups={result.group_count:,} multi={result.multi_member_group_count:,} "
            f"unresolved={result.unresolved_group_count:,} "
            f"independent_candidates={result.independent_group_candidate_count:,}"
        )
        for target in result.targets:
            print(
                f"target={target.target_detection_ids} group={target.group_id} "
                f"members={target.group_detection_ids} classification={target.classification}"
            )
        return 0
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-source-group-audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
