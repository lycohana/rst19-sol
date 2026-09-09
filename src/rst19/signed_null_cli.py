"""符号反相源级阴性对照命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .signed_null import SIGNED_NULL_SNR_THRESHOLDS, run_signed_null_audit, write_signed_null_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用真实 FITS 的背景符号反相图做源级 signed-tail leakage 对照，不输出源目录"
    )
    parser.add_argument("fits_path", type=Path, help="输入 FITS 文件")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="匹配滤波候选阈值 sigma")
    parser.add_argument("--min-distance", type=int, default=4, help="候选峰最小间距 pixel")
    parser.add_argument("--aperture-radius", type=int, default=4, help="孔径测光半径 pixel")
    parser.add_argument("--psf-fwhm", type=float, default=2.0, help="Gaussian 匹配滤波 FWHM pixel")
    parser.add_argument(
        "--proposal-mode",
        choices=("gaussian", "hybrid", "ensemble"),
        default="hybrid",
        help="宽筛选提案：Gaussian、Gaussian+DoG，或再加入 starlet 小波",
    )
    parser.add_argument("--background-box-size", type=int, default=128, help="局部背景/RMS 网格大小 pixel")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量层孔径通量 SNR 下限")
    parser.add_argument("--min-psf-support-pixels", type=int, default=3, help="候选峰中心 3x3 内的最少 PSF 支持像素")
    parser.add_argument("--keep-linear-artifacts", action="store_true", help="不标记线状结构为 LINE_ARTIFACT")
    parser.add_argument(
        "--snr-thresholds",
        type=float,
        nargs="+",
        default=list(SIGNED_NULL_SNR_THRESHOLDS),
        help="输出 SNR 分档下限，必须严格递增",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/signed-null-audit"), help="JSON/CSV 输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(value: float, label: str) -> None:
        print(f"[{value:5.1f}%] {label}", file=sys.stderr, flush=True)

    try:
        result = run_signed_null_audit(
            args.fits_path,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            psf_fwhm=args.psf_fwhm,
            background_box_size=args.background_box_size,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            proposal_mode=args.proposal_mode,
            reject_linear_artifacts=not args.keep_linear_artifacts,
            snr_thresholds=args.snr_thresholds,
            progress=progress,
        )
        output = write_signed_null_artifacts(result, args.out_dir)
        print(f"signed-null audit: {output}")
        print(
            f"positive candidate={result.positive_candidate_count} quality={result.positive_quality_count}; "
            f"sign_flipped candidate={result.negative_candidate_count} quality={result.negative_quality_count}"
        )
        print(
            f"candidate_leakage={result.candidate_leakage_ratio!r} "
            f"quality_leakage={result.quality_leakage_ratio!r}"
        )
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"rst19-signed-null-audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
