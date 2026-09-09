"""各类特征参数敏感性审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_parameter_sensitivity import (
    run_feature_parameter_sensitivity,
    write_feature_parameter_sensitivity_artifacts,
)


def _float_levels(value: str) -> tuple[float, ...]:
    try:
        levels = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("参数列表必须是逗号分隔的数字") from exc
    if not levels:
        raise argparse.ArgumentTypeError("参数列表不能为空")
    return levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="固定同一真实 FITS，扫描候选阈值/flux SNR 并追踪各类候选稳定性"
    )
    parser.add_argument("path", type=Path, help="一张 FITS 首帧")
    parser.add_argument("--threshold-levels", type=_float_levels, default=(4.0, 5.0, 6.0, 8.0))
    parser.add_argument("--flux-snr-levels", type=_float_levels, default=(3.0, 5.0, 7.0, 9.0))
    parser.add_argument("--baseline-threshold-sigma", type=float, default=4.0)
    parser.add_argument("--baseline-min-flux-snr", type=float, default=5.0)
    parser.add_argument("--min-distance", type=int, default=4)
    parser.add_argument("--aperture-radius", type=int, default=4)
    parser.add_argument("--psf-fwhm", type=float, default=2.0)
    parser.add_argument("--background-box-size", type=int, default=128)
    parser.add_argument("--background-sample-limit", type=int, default=100_000)
    parser.add_argument("--min-psf-support-pixels", type=int, default=3)
    parser.add_argument("--proposal-mode", choices=("gaussian", "hybrid", "ensemble"), default="hybrid")
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--association-radius-px", type=float, default=1.5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-parameter-sensitivity-current"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_sources == 0:
        args.max_sources = None

    def report(index: int, total: int, label: str) -> None:
        print(f"feature parameter sensitivity: {index}/{total} · {label}", file=sys.stderr, flush=True)

    try:
        result = run_feature_parameter_sensitivity(
            args.path,
            threshold_levels=args.threshold_levels,
            flux_snr_levels=args.flux_snr_levels,
            baseline_threshold_sigma=args.baseline_threshold_sigma,
            baseline_min_flux_snr=args.baseline_min_flux_snr,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            psf_fwhm=args.psf_fwhm,
            background_box_size=args.background_box_size,
            background_sample_limit=args.background_sample_limit,
            min_psf_support_pixels=args.min_psf_support_pixels,
            proposal_mode=args.proposal_mode,
            max_sources=args.max_sources,
            association_radius_px=args.association_radius_px,
            progress=report,
        )
        output = write_feature_parameter_sensitivity_artifacts(result, args.out_dir)
        print(f"rst19-feature-parameter-sensitivity: 审计已写入 {output}")
        print(
            f"baseline candidates={result.observations['baseline_candidate_count']} "
            f"quality={result.observations['baseline_quality_count']}"
        )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-parameter-sensitivity: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
