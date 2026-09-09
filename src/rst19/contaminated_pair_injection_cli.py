"""真实污染结构双源注入审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contaminated_pair_injection import (
    ContaminatedPairAnchor,
    run_contaminated_pair_injection_audit,
    write_contaminated_pair_injection_artifacts,
)


def _parse_anchor(value: str) -> ContaminatedPairAnchor:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 5:
        raise ValueError("--anchor must be NAME,X,Y,DIRECTION_X,DIRECTION_Y")
    try:
        return ContaminatedPairAnchor(
            name=parts[0],
            x=float(parts[1]),
            y=float(parts[2]),
            direction_x=float(parts[3]),
            direction_y=float(parts[4]),
        )
    except ValueError as exc:
        raise ValueError(f"invalid --anchor {value!r}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在真实局部背景/紧凑源/线状结构上注入已知双源，区分 baseline、injected 和 new 命中"
    )
    parser.add_argument("frame", type=Path, help="原始 FITS 文件")
    parser.add_argument(
        "--anchor",
        action="append",
        required=True,
        help="污染位置：NAME,X,Y,DIRECTION_X,DIRECTION_Y；可重复传入",
    )
    parser.add_argument("--separations-px", type=float, nargs="+", default=(2.738, 4.123))
    parser.add_argument("--ratios", type=float, nargs="+", default=(0.143, 1.0), help="secondary/primary")
    parser.add_argument("--total-peak-adu", type=float, default=4096.0)
    parser.add_argument("--psf-fwhm", type=float, default=2.0)
    parser.add_argument("--injected-psf-fwhm", type=float, default=None)
    parser.add_argument("--proposal-mode", default="hybrid", choices=("gaussian", "hybrid", "ensemble"))
    parser.add_argument("--threshold-sigma", type=float, default=4.0)
    parser.add_argument("--min-distance", type=int, default=4)
    parser.add_argument("--aperture-radius", type=int, default=4)
    parser.add_argument("--background-box-size", type=int, default=128)
    parser.add_argument("--min-flux-snr", type=float, default=5.0)
    parser.add_argument("--min-psf-support-pixels", type=int, default=3)
    parser.add_argument("--match-radius-px", type=float, default=2.0)
    parser.add_argument(
        "--analysis-scope",
        choices=("full_frame", "local_roi"),
        default="full_frame",
        help="注入后检测范围；local_roi 只用于研究加速，结果中的计数是 ROI 局部计数",
    )
    parser.add_argument(
        "--roi-half-size-px",
        type=int,
        default=192,
        help="local_roi 的半窗口边长（像素）",
    )
    parser.add_argument("--allow-linear-artifacts", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/contaminated-pair-injection"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        anchors = tuple(_parse_anchor(value) for value in args.anchor)
        def report(index: int, total: int) -> None:
            print(f"contaminated pair injection: {index}/{total}", file=sys.stderr, flush=True)

        result = run_contaminated_pair_injection_audit(
            args.frame,
            anchors,
            separations_px=args.separations_px,
            secondary_to_primary_ratios=args.ratios,
            total_peak_excess_adu=args.total_peak_adu,
            psf_fwhm=args.psf_fwhm,
            injected_psf_fwhm=args.injected_psf_fwhm,
            proposal_mode=args.proposal_mode,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            background_box_size=args.background_box_size,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            match_radius_px=args.match_radius_px,
            reject_linear_artifacts=not args.allow_linear_artifacts,
            analysis_scope=args.analysis_scope,
            roi_half_size_px=args.roi_half_size_px,
            progress=report,
        )
        output = write_contaminated_pair_injection_artifacts(result, args.out_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-contaminated-pair-injection: {exc}", file=sys.stderr)
        return 2
    print(f"contaminated pair injection audit: {output}")
    print(result.conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
