"""近邻双框局部异常值修复反事实命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .empirical_pair import load_detection_catalog_csv
from .pair_repair_counterfactual import (
    run_pair_repair_counterfactual,
    write_pair_repair_counterfactual_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用局部稳健邻域值替换重复码/极端负值后重跑近邻双候选 detector"
    )
    parser.add_argument("frame", type=Path, help="原始 FITS 文件")
    parser.add_argument("--source-catalog", type=Path, required=True, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument("--primary-id", type=int, required=True, help="主候选 detection_id")
    parser.add_argument("--secondary-id", type=int, required=True, help="副候选 detection_id")
    parser.add_argument("--patch-padding-px", type=int, default=128, help="目标峰周围局部裁剪外扩像素")
    parser.add_argument("--replacement-radius-px", type=int, default=1, help="邻域中位数补值半径")
    parser.add_argument("--target-match-radius-px", type=float, default=4.0, help="将重跑候选归入目标窗口的质心半径")
    parser.add_argument("--repeated-code-values", default=None, help="可选逗号分隔重复工程码；省略时从源表推断")
    parser.add_argument("--negative-anomaly-threshold-adu", type=float, default=-1000.0, help="极端负值阈值")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="候选阈值 sigma")
    parser.add_argument("--min-distance", type=int, default=4, help="候选峰最小间距 pixel")
    parser.add_argument("--aperture-radius", type=int, default=4, help="孔径半径 pixel")
    parser.add_argument("--psf-fwhm", type=float, default=2.0, help="Gaussian PSF FWHM pixel")
    parser.add_argument("--background-box-size", type=int, default=128, help="背景/RMS 网格尺寸 pixel")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量层孔径通量 SNR 下限")
    parser.add_argument("--min-psf-support-pixels", type=int, default=3, help="PSF 支持像素下限")
    parser.add_argument("--proposal-mode", choices=("gaussian", "hybrid", "ensemble"), default="hybrid")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/pair-repair-counterfactual"), help="CSV/JSON 输出目录")
    return parser


def _load_codes(value: str | None):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sources = load_detection_catalog_csv(args.source_catalog)
        source_by_id = {source.detection_id: source for source in sources}
        primary = source_by_id[args.primary_id]
        secondary = source_by_id[args.secondary_id]
        result = run_pair_repair_counterfactual(
            args.frame,
            primary,
            secondary,
            patch_padding_px=args.patch_padding_px,
            replacement_radius_px=args.replacement_radius_px,
            target_match_radius_px=args.target_match_radius_px,
            repeated_code_values=_load_codes(args.repeated_code_values),
            negative_anomaly_threshold_adu=args.negative_anomaly_threshold_adu,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            psf_fwhm=args.psf_fwhm,
            background_box_size=args.background_box_size,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            proposal_mode=args.proposal_mode,
        )
        output = write_pair_repair_counterfactual_artifacts(result, args.out_dir)
    except KeyError as exc:
        print(f"rst19-pair-repair-counterfactual: source catalog 中不存在 detection_id={exc.args[0]}", file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-pair-repair-counterfactual: 审计失败：{exc}", file=sys.stderr)
        return 2

    print(f"pair repair counterfactual: {output}")
    print(result.conclusion)
    for row in result.summaries:
        print(
            f"{row.variant}: target_candidates={row.target_window_candidate_count} "
            f"target_quality={row.target_window_quality_count} repaired={row.repaired_pixel_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
