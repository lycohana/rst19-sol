"""15 帧特征类别持久性审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .experiments import run_sequence_feature_persistence, write_sequence_feature_persistence_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在真实 FITS 序列中比较各类首帧候选的跨帧响应持久性"
    )
    parser.add_argument("data_dir", type=Path, help="包含 FITS 序列的目录")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="匹配滤波候选阈值 sigma")
    parser.add_argument("--min-distance", type=int, default=4, help="候选峰最小间距 pixel")
    parser.add_argument("--aperture-radius", type=int, default=4, help="孔径测光半径 pixel")
    parser.add_argument("--psf-fwhm", type=float, default=2.0, help="Gaussian 匹配滤波 FWHM pixel")
    parser.add_argument(
        "--proposal-mode",
        choices=("gaussian", "hybrid", "ensemble"),
        default="hybrid",
        help="宽筛选提案模式",
    )
    parser.add_argument("--background-box-size", type=int, default=128, help="局部背景网格尺寸 pixel")
    parser.add_argument("--background-sample-limit", type=int, default=100_000, help="每个背景块的统计抽样上限")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量层孔径通量 SNR 下限")
    parser.add_argument(
        "--min-psf-support-pixels",
        type=int,
        default=3,
        help="候选峰中心 3×3 内超过支持阈值的最少像素数",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        help="每帧源级测量返回上限；留空为全量，传 0 也表示全量",
    )
    parser.add_argument(
        "--association-radius-px",
        type=float,
        default=1.0,
        help="首帧坐标的跨帧邻域半径；默认 1 px，不是一对一星表匹配半径",
    )
    parser.add_argument("--registration-radius-px", type=float, default=8.0, help="配准源关联半径 pixel")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/sequence-feature-audit"), help="CSV/JSON/PNG 输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(sorted(args.data_dir.glob("*.fits")))
    if not paths:
        print(f"rst19-feature-sequence: 数据目录没有 FITS 文件：{args.data_dir}", file=sys.stderr)
        return 2
    if args.max_sources == 0:
        args.max_sources = None

    def report(index: int, total: int) -> None:
        print(f"feature persistence: frame {index}/{total}", file=sys.stderr, flush=True)

    try:
        result = run_sequence_feature_persistence(
            paths,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            psf_fwhm=args.psf_fwhm,
            background_box_size=args.background_box_size,
            background_sample_limit=args.background_sample_limit,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            proposal_mode=args.proposal_mode,
            max_sources=args.max_sources,
            registration_radius_px=args.registration_radius_px,
            association_radius_px=args.association_radius_px,
            progress=report,
        )
        output = write_sequence_feature_persistence_artifacts(result, args.out_dir)
        print(f"rst19-feature-sequence: 审计已写入 {output}")
        print(
            f"frames={result.frame_count} radius={result.association_radius_px:g} "
            f"required_presence={result.required_presence}"
        )
        for row in result.persistence_rows:
            print(
                f"{row.feature_class}: anchor={row.anchor_candidate_count} "
                f"candidate>={row.required_presence}={row.candidate_presence_ge_required_count} "
                f"quality>={row.required_presence}={row.quality_presence_ge_required_count}"
            )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-sequence: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
