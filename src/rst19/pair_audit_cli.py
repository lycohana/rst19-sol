"""固定坐标近邻双源原始证据审计命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .experiments import run_source_pair_audit, write_source_pair_audit_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="复核两个近邻检测框是否是两颗可分辨点源，并输出逐帧原始证据"
    )
    parser.add_argument("data_dir", type=Path, help="包含 FITS 序列的目录")
    parser.add_argument("--primary-x", type=float, default=1269.0, help="主响应 detector X；默认使用截图审计坐标")
    parser.add_argument("--primary-y", type=float, default=3465.0, help="主响应 detector Y；默认使用截图审计坐标")
    parser.add_argument("--secondary-x", type=float, default=1274.0, help="副响应 detector X；默认使用截图审计坐标")
    parser.add_argument("--secondary-y", type=float, default=3467.0, help="副响应 detector Y；默认使用截图审计坐标")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="匹配滤波候选阈值 sigma")
    parser.add_argument("--min-distance", type=int, default=4, help="候选峰最小间距 pixel")
    parser.add_argument("--aperture-radius", type=int, default=4, help="原始像素审计孔径半径 pixel")
    parser.add_argument("--psf-fwhm", type=float, default=2.0, help="单/双 PSF 比较的 FWHM pixel")
    parser.add_argument("--proposal-mode", choices=("gaussian", "hybrid", "ensemble"), default="hybrid", help="宽筛选提案模式")
    parser.add_argument("--background-box-size", type=int, default=128, help="局部背景/RMS 网格尺寸 pixel")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量层孔径通量 SNR 下限")
    parser.add_argument("--min-psf-support-pixels", type=int, default=3, help="3×3 PSF 支持像素最少数量")
    parser.add_argument("--max-sources", type=int, help="可选返回源上限；留空表示全量")
    parser.add_argument("--match-radius-px", type=float, default=2.0, help="按 detector 峰坐标寻找对应候选的半径")
    parser.add_argument("--nearest-search-radius-px", type=float, default=8.0, help="判断合并/漂移响应时检索附近候选的半径")
    parser.add_argument(
        "--shifts-json",
        type=Path,
        help="可选序列 JSON；读取 cumulative_shifts 后按注册 detector 位置审计",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/source-pair-audit"), help="CSV/JSON/PNG 输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(sorted(args.data_dir.glob("*.fits")))
    if not paths:
        print(f"rst19-pair-audit: 数据目录没有 FITS 文件：{args.data_dir}", file=sys.stderr)
        return 2
    if args.max_sources == 0:
        args.max_sources = None

    frame_shifts = None
    if args.shifts_json is not None:
        try:
            payload = json.loads(args.shifts_json.read_text(encoding="utf-8"))
            raw_shifts = payload["cumulative_shifts"]
            frame_shifts = tuple((float(item[0]), float(item[1])) for item in raw_shifts)
        except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
            print(f"rst19-pair-audit: 无法读取 --shifts-json：{exc}", file=sys.stderr)
            return 2

    def report(index: int, total: int) -> None:
        print(f"pair audit: frame {index}/{total}", file=sys.stderr, flush=True)

    try:
        result = run_source_pair_audit(
            paths,
            (args.primary_x, args.primary_y),
            (args.secondary_x, args.secondary_y),
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            psf_fwhm=args.psf_fwhm,
            background_box_size=args.background_box_size,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            proposal_mode=args.proposal_mode,
            max_sources=args.max_sources,
            match_radius_px=args.match_radius_px,
            nearest_search_radius_px=args.nearest_search_radius_px,
            frame_shifts=frame_shifts,
            progress=report,
        )
        output = write_source_pair_audit_artifacts(result, args.out_dir)
        print(f"rst19-pair-audit: 审计已写入 {output}")
        print(f"target_distance={result.target_peak_distance_px:.3f}px")
        print(result.conclusion)
        for row in result.rows:
            shared_fraction = (
                f"{row.secondary_shared_aperture_net_fraction:.1%}"
                if row.secondary_shared_aperture_net_fraction is not None
                else "—"
            )
            print(
                f"F{row.frame_index:02d}: primary={row.primary_detection_id} "
                f"secondary={row.secondary_detection_id} "
                f"secondary_quality={row.secondary_quality_passed} "
                f"nearest_secondary={row.nearest_secondary_detection_id} "
                f"same_nearest={row.nearest_source_same_for_targets} "
                f"shared_aperture={row.secondary_shared_aperture_pixel_count}/"
                f"{row.secondary_aperture_pixel_count} "
                f"shared_net={shared_fraction} "
                f"neg={row.secondary_negative_overflow_count} "
                f"-1={row.secondary_fixed_minus_one_count} "
                f"raw_delta_bic={row.pair_delta_bic_raw}"
            )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-pair-audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
