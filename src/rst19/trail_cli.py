"""单图长线候选审计命令。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .experiments import run_single_frame_trail_audit, write_single_frame_trail_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="逐张 FITS 审计单图长线候选并写出表格/曲线")
    parser.add_argument("input_path", type=Path, help="FITS 文件或包含 FITS 文件的目录")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="点源候选阈值 sigma")
    parser.add_argument("--min-distance", type=int, default=4, help="点源候选峰最小间距（pixel）")
    parser.add_argument("--aperture-radius", type=int, default=4, help="孔径测光半径（pixel）")
    parser.add_argument("--psf-fwhm", type=float, default=3.0, help="匹配滤波 Gaussian PSF FWHM（pixel）")
    parser.add_argument("--background-box-size", type=int, default=128, help="局部背景/RMS 网格大小（pixel）")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量层孔径通量 SNR 下限")
    parser.add_argument("--max-sources", type=int, help="可选返回源上限；留空表示全量")
    parser.add_argument("--residual-sigma", type=float, default=15.0, help="长线残差显著性阈值 sigma")
    parser.add_argument("--min-residual-adu", type=float, default=100.0, help="长线残差最低 ADU")
    parser.add_argument("--min-feature-area", type=int, default=40, help="长线最小连通域面积（pixel）")
    parser.add_argument("--min-axis-ratio", type=float, default=4.0, help="长线 PCA 最小轴比")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp") / "single-frame-trails", help="CSV/PNG 输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input_path
    paths = sorted(input_path.glob("*.fits")) if input_path.is_dir() else [input_path]

    def progress(index: int, total: int) -> None:
        print(f"trail audit: {index}/{total}", file=sys.stderr, flush=True)

    try:
        rows = run_single_frame_trail_audit(
            paths,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            psf_fwhm=args.psf_fwhm,
            background_box_size=args.background_box_size,
            min_flux_snr=args.min_flux_snr,
            max_sources=args.max_sources,
            residual_sigma=args.residual_sigma,
            min_residual_adu=args.min_residual_adu,
            min_feature_area=args.min_feature_area,
            min_axis_ratio=args.min_axis_ratio,
            progress=progress,
        )
        output = write_single_frame_trail_artifacts(rows, args.out_dir)
    except (OSError, ValueError) as exc:
        print(f"rst19-trails: {exc}", file=sys.stderr)
        return 2

    print(f"single-frame trail audit: {output}")
    for row in rows:
        length = f"{row.max_length_px:.1f}px" if row.max_length_px is not None else "—"
        print(f"frame={row.frame_number:02d} trails={row.trail_count} max_length={length} edge={row.edge_trail_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
