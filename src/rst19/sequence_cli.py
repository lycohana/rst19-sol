"""15 帧序列分析命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .sequence import DEFAULT_SEQUENCE_SOURCE_WORKING_LIMIT, DEFAULT_SEQUENCE_WORKERS, analyze_sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检测一组 FITS 并区分稳定星点、运动目标和瞬态候选")
    parser.add_argument("data_dir", type=Path, help="包含 FITS 序列的目录")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="匹配滤波候选阈值")
    parser.add_argument("--min-distance", type=int, default=3, help="候选峰最小间距（pixel）")
    parser.add_argument("--aperture-radius", type=int, default=4, help="孔径半径（pixel）")
    parser.add_argument("--psf-fwhm", type=float, default=3.0, help="Gaussian PSF FWHM（pixel）")
    parser.add_argument(
        "--proposal-mode",
        choices=("gaussian", "hybrid", "ensemble"),
        default="gaussian",
        help="宽筛选提案：Gaussian、Gaussian+DoG，或再加入 Starlet 小波",
    )
    parser.add_argument(
        "--temporal-proposal-mode",
        choices=("median", "coadd", "both"),
        default="median",
        help="15 帧共同证据提案：时间中值、稳健叠加，或两者并集；默认中值",
    )
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量源通量 SNR 下限")
    parser.add_argument(
        "--min-psf-support-pixels",
        type=int,
        default=3,
        help="候选峰中心 3×3 内超过 PSF 支持阈值的最少像素数；默认 3",
    )
    parser.add_argument("--link-radius-px", type=float, default=4.0, help="轨迹关联半径（pixel）")
    parser.add_argument("--min-presence", type=int, help="轨迹至少出现的帧数；默认要求约 80%% 帧")
    parser.add_argument(
        "--persistent-min-presence",
        type=int,
        help="较低置信静态候选至少出现的帧数；默认要求约 50%% 帧，且不超过 --min-presence",
    )
    parser.add_argument("--motion-min-displacement-px", type=float, default=2.0, help="运动轨迹最小总位移（pixel）")
    parser.add_argument("--max-motion-fit-rms-px", type=float, default=0.75, help="运动直线拟合最大 RMS（pixel）")
    parser.add_argument(
        "--source-limit",
        type=int,
        default=DEFAULT_SEQUENCE_SOURCE_WORKING_LIMIT,
        help=f"每帧进入序列配准/点轨迹关联的最高 SNR 源数；默认 {DEFAULT_SEQUENCE_SOURCE_WORKING_LIMIT}，传 0 使用全量（更慢）",
    )
    parser.add_argument(
        "--background-box-size",
        type=int,
        default=256,
        help="序列局部背景网格尺寸（pixel）；默认 256",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_SEQUENCE_WORKERS,
        help=f"并行检测的帧数；默认 {DEFAULT_SEQUENCE_WORKERS}，内存紧张时传 1",
    )
    parser.add_argument(
        "--candidate-consensus-min-snr",
        type=float,
        default=15.0,
        help="全量滤波候选进入跨帧稳定共识的最低候选 SNR；默认 15.0",
    )
    parser.add_argument(
        "--temporal-candidate-min-snr",
        type=float,
        help="时间参考坐标回到逐帧原图时的强制响应 SNR；默认 max(5, 0.5×普通候选门槛)，只影响时序补提案",
    )
    parser.add_argument(
        "--temporal-reference-min-snr",
        type=float,
        help="时间中值/稳健叠加补提案的参考图最低 SNR；默认跟随 --candidate-consensus-min-snr，可单独降到 12 做召回抽测",
    )
    parser.add_argument(
        "--temporal-multiscale",
        action="store_true",
        help="时间参考宽筛使用窄/基准/宽三组 PSF；研究漏检用，默认关闭，候选数不能直接视为星表真值",
    )
    parser.add_argument(
        "--temporal-min-psf-correlation",
        type=float,
        default=0.8,
        help="时序补提案逐帧 Gaussian 形状相关系数下限；默认 0.8，作为细筛证据",
    )
    parser.add_argument(
        "--float64",
        action="store_true",
        help="禁用整数 FITS 序列的 float32 中间阵列，使用 float64 做对照",
    )
    parser.add_argument("--keep-linear-artifacts", action="store_true", help="保留线状结构候选；默认将长线标记为 LINE_ARTIFACT")
    parser.add_argument("--json-out", type=Path, help="可选 JSON 输出路径")
    parser.add_argument(
        "--audit-out-dir",
        type=Path,
        help="可选星点时序审计目录；导出严格静态/持续候选/瞬态轨迹的代表性原图小窗",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(sorted(args.data_dir.glob("*.fits")))
    if not paths:
        print(f"rst19-sequence: 数据目录没有 FITS 文件：{args.data_dir}", file=sys.stderr)
        return 2
    try:
        if args.source_limit < 0:
            raise ValueError("--source-limit must be zero or a positive integer")
        if args.workers < 1:
            raise ValueError("--workers must be a positive integer")
        result = analyze_sequence(
            paths,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            psf_fwhm=args.psf_fwhm,
            proposal_mode=args.proposal_mode,
            temporal_proposal_mode=args.temporal_proposal_mode,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            link_radius_px=args.link_radius_px,
            min_presence=args.min_presence,
            persistent_min_presence=args.persistent_min_presence,
            motion_min_displacement_px=args.motion_min_displacement_px,
            max_motion_fit_rms_px=args.max_motion_fit_rms_px,
            sequence_max_sources=None if args.source_limit == 0 else args.source_limit,
            background_box_size=args.background_box_size,
            sequence_workers=args.workers,
            candidate_consensus_min_snr=args.candidate_consensus_min_snr,
            temporal_candidate_min_snr=args.temporal_candidate_min_snr,
            temporal_reference_min_snr=args.temporal_reference_min_snr,
            temporal_multiscale=args.temporal_multiscale,
            temporal_min_psf_correlation=args.temporal_min_psf_correlation,
            use_float32=not args.float64,
            reject_linear_artifacts=not args.keep_linear_artifacts,
        )
        rendered = json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        if args.audit_out_dir is not None:
            from .experiments import write_sequence_source_audit

            audit_output = write_sequence_source_audit(result, args.audit_out_dir)
            print(f"rst19-sequence: 星点时序审计已写入 {audit_output}", file=sys.stderr)
        return 0
    except (OSError, ValueError) as exc:
        print(f"rst19-sequence: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
