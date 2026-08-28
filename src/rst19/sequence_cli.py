"""15 帧序列分析命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .sequence import analyze_sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检测一组 FITS 并区分稳定星点、运动目标和瞬态候选")
    parser.add_argument("data_dir", type=Path, help="包含 FITS 序列的目录")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="匹配滤波候选阈值")
    parser.add_argument("--min-distance", type=int, default=3, help="候选峰最小间距（pixel）")
    parser.add_argument("--aperture-radius", type=int, default=4, help="孔径半径（pixel）")
    parser.add_argument("--psf-fwhm", type=float, default=3.0, help="Gaussian PSF FWHM（pixel）")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量源通量 SNR 下限")
    parser.add_argument("--link-radius-px", type=float, default=4.0, help="轨迹关联半径（pixel）")
    parser.add_argument("--min-presence", type=int, help="轨迹至少出现的帧数；默认要求约 80%% 帧")
    parser.add_argument("--motion-min-displacement-px", type=float, default=2.0, help="运动轨迹最小总位移（pixel）")
    parser.add_argument("--max-motion-fit-rms-px", type=float, default=0.75, help="运动直线拟合最大 RMS（pixel）")
    parser.add_argument("--json-out", type=Path, help="可选 JSON 输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(sorted(args.data_dir.glob("*.fits")))
    if not paths:
        print(f"rst19-sequence: 数据目录没有 FITS 文件：{args.data_dir}", file=sys.stderr)
        return 2
    try:
        result = analyze_sequence(
            paths,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            psf_fwhm=args.psf_fwhm,
            min_flux_snr=args.min_flux_snr,
            link_radius_px=args.link_radius_px,
            min_presence=args.min_presence,
            motion_min_displacement_px=args.motion_min_displacement_px,
            max_motion_fit_rms_px=args.max_motion_fit_rms_px,
        )
        rendered = json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0
    except (OSError, ValueError) as exc:
        print(f"rst19-sequence: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
