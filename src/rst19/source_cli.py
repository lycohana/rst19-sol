"""单帧源级星点研究表和 SNR 图命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .experiments import write_detection_source_artifacts
from .pipeline import analyze_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对单张 FITS 输出全量源表、质量标记、形态分层和 SNR 研究图")
    parser.add_argument("fits_path", type=Path, help="输入 FITS 文件")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="匹配滤波候选阈值 sigma")
    parser.add_argument("--min-distance", type=int, default=4, help="候选峰最小间距 pixel")
    parser.add_argument("--aperture-radius", type=int, default=4, help="孔径测光半径 pixel")
    parser.add_argument("--psf-fwhm", type=float, default=3.0, help="Gaussian 匹配滤波 FWHM pixel")
    parser.add_argument("--proposal-mode", choices=("gaussian", "hybrid", "ensemble"), default="hybrid", help="宽筛选提案：Gaussian、Gaussian+DoG，或再加入 starlet 小波")
    parser.add_argument("--dog-threshold-sigma", type=float, help="DoG 提案阈值；默认 max(6σ, Gaussian+2σ)")
    parser.add_argument("--dog-min-peak-sigma", type=float, default=2.0, help="DoG 候选在原始残差图上的最低正峰显著性")
    parser.add_argument("--starlet-threshold-sigma", type=float, help="starlet 响应阈值；默认 max(7σ, Gaussian+3σ)")
    parser.add_argument("--starlet-min-peak-sigma", type=float, default=2.5, help="starlet 候选在原始残差图上的最低正峰显著性")
    parser.add_argument("--enable-local-deblend", action="store_true", help="启用强主峰残差双源去混叠；较慢，默认关闭")
    parser.add_argument("--background-box-size", type=int, default=128, help="局部背景/RMS 网格大小 pixel")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量层孔径通量 SNR 下限")
    parser.add_argument("--min-psf-support-pixels", type=int, default=3, help="候选峰中心 3×3 内超过 PSF 支持阈值的最少像素数")
    parser.add_argument("--max-sources", type=int, help="可选返回源上限；留空表示全量")
    parser.add_argument("--keep-linear-artifacts", action="store_true", help="不标记线状结构为 LINE_ARTIFACT")
    parser.add_argument("--spatial-psf-per-class", type=int, default=8, help="空间 PSF 诊断每类抽样数；只影响研究表")
    parser.add_argument("--spatial-psf-grid-size", type=int, default=4, help="空间 PSF 诊断每轴网格数；只影响研究表")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/source-audit"), help="CSV/PNG 输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        analysis = analyze_frame(
            args.fits_path,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            psf_fwhm=args.psf_fwhm,
            proposal_mode=args.proposal_mode,
            dog_threshold_sigma=args.dog_threshold_sigma,
            dog_min_peak_sigma=args.dog_min_peak_sigma,
            starlet_threshold_sigma=args.starlet_threshold_sigma,
            starlet_min_peak_sigma=args.starlet_min_peak_sigma,
            enable_local_deblend=args.enable_local_deblend,
            background_box_size=args.background_box_size,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            max_sources=args.max_sources,
            reject_linear_artifacts=not args.keep_linear_artifacts,
        )
        output = write_detection_source_artifacts(
            analysis.detection,
            args.out_dir,
            frame=analysis.frame,
            spatial_psf_per_class=args.spatial_psf_per_class,
            spatial_psf_grid_size=args.spatial_psf_grid_size,
        )
        detection = analysis.detection
        print(f"source audit: {output}")
        print(
            f"candidate={detection.candidate_count} returned={detection.returned_count} "
            f"quality={detection.star_count} rejected={detection.rejected_count}"
        )
        return 0
    except (OSError, ValueError) as exc:
        print(f"rst19-sources: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
