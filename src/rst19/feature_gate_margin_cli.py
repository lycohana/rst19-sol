"""质量门余量审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_gate_margin import run_feature_gate_margin, write_feature_gate_margin_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 feature_class 统计数值质量门余量，不重新读取 FITS"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument("--min-flux-snr", type=float, default=5.0)
    parser.add_argument("--min-psf-support-pixels", type=int, default=3)
    parser.add_argument("--min-fwhm", type=float, default=0.8)
    parser.add_argument("--max-fwhm", type=float, default=12.0)
    parser.add_argument("--max-ellipticity", type=float, default=0.65)
    parser.add_argument("--min-sharpness", type=float, default=0.005)
    parser.add_argument("--max-sharpness", type=float, default=0.85)
    parser.add_argument("--min-footprint-pixels", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/feature-gate-margin"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_gate_margin(
            args.catalog,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            min_fwhm=args.min_fwhm,
            max_fwhm=args.max_fwhm,
            max_ellipticity=args.max_ellipticity,
            min_sharpness=args.min_sharpness,
            max_sharpness=args.max_sharpness,
            min_footprint_pixels=args.min_footprint_pixels,
        )
        output = write_feature_gate_margin_artifacts(result, args.out_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-gate-margin: {exc}", file=sys.stderr)
        return 2
    print(f"feature gate margin: {output}")
    print(f"rows={len(result.rows)}")
    for row in result.rows:
        print(
            f"{row.feature_class}/{row.metric}: median={row.median_margin:.4g} "
            f"p10={row.p10_margin:.4g} p90={row.p90_margin:.4g} "
            f"negative={row.negative_margin_count}/{row.value_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
