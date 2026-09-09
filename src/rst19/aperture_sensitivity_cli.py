"""经验 PSF 孔径敏感性审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .aperture_sensitivity import (
    run_aperture_sensitivity_audit,
    write_aperture_sensitivity_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在同一真实背景和同一注入位置上扫描经验 PSF 的测光孔径"
    )
    parser.add_argument("fits_path", type=Path, help="单张 FITS 图像")
    parser.add_argument(
        "--aperture-radii",
        nargs="+",
        type=int,
        default=(3, 4, 5, 6),
        help="要对照的测光孔径半径；默认 3 4 5 6 pixel",
    )
    parser.add_argument("--reference-aperture-radius", type=int, default=4, help="模板/位置选择的参考孔径")
    parser.add_argument("--control-signal-adu", type=float, default=512.0, help="固定离散积分注入信号 ADU")
    parser.add_argument("--grid-size", type=int, default=2, help="位置选择网格每轴单元数；默认 2，即 2×2")
    parser.add_argument("--psf-fwhm", type=float, default=2.0, help="检测器 Gaussian 匹配尺度 FWHM")
    parser.add_argument("--background-box-size", type=int, default=128, help="局部背景网格尺寸 pixel")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量层通量 SNR 下限")
    parser.add_argument("--min-psf-support-pixels", type=int, default=3, help="3×3 PSF 支持像素下限")
    parser.add_argument("--min-distance", type=int, default=4, help="候选峰最小间距 pixel")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="候选阈值 sigma")
    parser.add_argument("--max-sources", type=int, help="可选检测返回上限；默认不截断")
    parser.add_argument("--proposal-mode", choices=("gaussian", "hybrid", "ensemble"), default="hybrid")
    parser.add_argument("--empirical-psf-radius", type=int, default=7, help="经验 PSF 支持半径 pixel")
    parser.add_argument("--empirical-psf-sources", type=int, default=64, help="最多模板源数")
    parser.add_argument("--seed", type=int, default=19019, help="自动选择空间位置时的随机种子")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/aperture-sensitivity-audit"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.fits_path.is_file():
        print(f"rst19-aperture-sensitivity: FITS 不存在：{args.fits_path}", file=sys.stderr)
        return 2
    progress_state = {"last": (-1, -1)}

    def report(index: int, total: int) -> None:
        state = (int(index), int(total))
        if state != progress_state["last"]:
            progress_state["last"] = state
            print(f"aperture sensitivity progress: {index}/{total}", file=sys.stderr, flush=True)

    try:
        result = run_aperture_sensitivity_audit(
            args.fits_path,
            aperture_radii=tuple(args.aperture_radii),
            reference_aperture_radius=args.reference_aperture_radius,
            control_signal_adu=args.control_signal_adu,
            grid_size=args.grid_size,
            psf_fwhm=args.psf_fwhm,
            proposal_mode=args.proposal_mode,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            background_box_size=args.background_box_size,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            max_sources=args.max_sources,
            empirical_psf_radius=args.empirical_psf_radius,
            empirical_psf_sources=args.empirical_psf_sources,
            seed=args.seed,
            progress=report,
        )
        output = write_aperture_sensitivity_artifacts(result, args.out_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"rst19-aperture-sensitivity: 审计失败：{exc}", file=sys.stderr)
        return 2

    print(f"rst19-aperture-sensitivity: artifacts written to {output}")
    print(result.conclusion)
    for summary in result.summary_by_aperture:
        print(
            f"r={int(summary['aperture_radius'])} "
            f"candidate={summary['candidate_recall']:.3f} "
            f"quality={summary['quality_recall']:.3f} "
            f"median_flux_snr={summary['median_candidate_flux_snr']} "
            f"control={summary['paired_control_candidate_count']}/"
            f"{summary['paired_control_quality_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
