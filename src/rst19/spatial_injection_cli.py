"""空间分区真实背景留出注入命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .spatial_injection import run_spatial_injection_audit, write_spatial_injection_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在真实 FITS 背景的 detector 空间单元中做留出注入-回收"
    )
    parser.add_argument("fits_path", type=Path, help="单张 FITS 图像")
    parser.add_argument("--grid-size", type=int, default=2, help="每轴空间单元数；默认 2，即 2×2")
    parser.add_argument(
        "--peak-levels",
        nargs="+",
        type=float,
        default=(56.0, 128.0),
        help="控制信号 ADU 档位；integrated_excess 时表示离散积分超额",
    )
    parser.add_argument("--trials", type=int, default=1, help="每个信号档位的重复次数")
    parser.add_argument("--sources-per-cell", type=int, default=1, help="每个空间单元每次注入源数")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="候选阈值 sigma")
    parser.add_argument("--min-distance", type=int, default=4, help="候选峰最小间距 pixel")
    parser.add_argument("--aperture-radius", type=int, default=4, help="孔径测光半径 pixel")
    parser.add_argument("--psf-fwhm", type=float, default=2.0, help="检测 PSF FWHM pixel")
    parser.add_argument("--injected-psf-fwhm", type=float, help="Gaussian 注入 PSF FWHM；默认跟随 --psf-fwhm")
    parser.add_argument("--background-box-size", type=int, default=128, help="局部背景网格尺寸 pixel")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量层通量 SNR 下限")
    parser.add_argument("--min-psf-support-pixels", type=int, default=3, help="3×3 PSF 支持像素下限")
    parser.add_argument("--max-sources", type=int, help="可选检测返回上限；默认不截断")
    parser.add_argument("--match-radius-px", type=float, help="注入回收匹配半径；默认 max(3,FWHM)")
    parser.add_argument("--seed", type=int, default=19019, help="空间位置选择随机种子")
    parser.add_argument("--proposal-mode", choices=("gaussian", "hybrid", "ensemble"), default="hybrid")
    parser.add_argument("--psf-model", choices=("gaussian", "empirical"), default="gaussian")
    parser.add_argument(
        "--signal-normalization",
        choices=("peak_excess", "integrated_excess"),
        default="integrated_excess",
        help="固定峰值超额或固定离散积分超额；默认后者便于 PSF 公平比较",
    )
    parser.add_argument("--empirical-psf-radius", type=int, default=7, help="实测 PSF 支持半径 pixel")
    parser.add_argument("--empirical-psf-sources", type=int, default=64, help="实测 PSF 最多模板源数")
    parser.add_argument("--keep-linear-artifacts", action="store_true", help="不在质量层排除线状结构")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/spatial-injection-audit"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.fits_path.is_file():
        print(f"rst19-spatial-injection: FITS 不存在：{args.fits_path}", file=sys.stderr)
        return 2
    progress_state = {"last": (-1, -1)}

    def report(index: int, total: int) -> None:
        state = (int(index), int(total))
        if state != progress_state["last"]:
            progress_state["last"] = state
            print(f"spatial injection progress: {index}/{total}", file=sys.stderr, flush=True)

    try:
        result = run_spatial_injection_audit(
            args.fits_path,
            grid_size=args.grid_size,
            peak_levels=tuple(args.peak_levels),
            trials_per_level=args.trials,
            sources_per_cell=args.sources_per_cell,
            psf_fwhm=args.psf_fwhm,
            injected_psf_fwhm=args.injected_psf_fwhm,
            proposal_mode=args.proposal_mode,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            background_box_size=args.background_box_size,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            max_sources=args.max_sources,
            match_radius_px=args.match_radius_px,
            seed=args.seed,
            reject_linear_artifacts=not args.keep_linear_artifacts,
            psf_model=args.psf_model,
            signal_normalization=args.signal_normalization,
            empirical_psf_radius=args.empirical_psf_radius,
            empirical_psf_sources=args.empirical_psf_sources,
            progress=report,
        )
        output = write_spatial_injection_artifacts(result, args.out_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"rst19-spatial-injection: 审计失败：{exc}", file=sys.stderr)
        return 2

    print(f"rst19-spatial-injection: artifacts written to {output}")
    print(result.conclusion)
    for row in result.rows:
        print(
            f"{row.cell_id} signal={row.control_signal_adu:g} "
            f"candidate={row.candidate_recall:.3f} quality={row.quality_recall:.3f} "
            f"noise={row.local_noise_adu:.3f} "
            f"candidate_delta={row.candidate_delta_vs_paired_control:g} "
            f"quality_delta={row.quality_delta_vs_paired_control:g} "
            f"filter_noise_delta={row.filter_noise_delta_vs_paired_control:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
