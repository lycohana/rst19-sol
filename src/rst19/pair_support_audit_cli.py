"""近邻双候选独立支持留出审计命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .empirical_pair import load_detection_catalog_csv
from .pair_support_audit import (
    PAIR_SUPPORT_MASK_MODES,
    run_pair_support_audit,
    write_pair_support_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在局部裁剪和多种留出掩膜下检查一对近邻候选的独立像素支持"
    )
    parser.add_argument("data_dir", type=Path, help="包含 FITS 序列的目录")
    parser.add_argument(
        "--source-catalog",
        type=Path,
        required=True,
        help="rst19-sources 产生的首帧 source_catalog.csv",
    )
    parser.add_argument("--primary-id", type=int, required=True, help="主候选 detection_id")
    parser.add_argument("--secondary-id", type=int, required=True, help="副候选 detection_id")
    parser.add_argument(
        "--mask-modes",
        default=",".join(PAIR_SUPPORT_MASK_MODES),
        help="逗号分隔的留出模式；默认执行全部模式",
    )
    parser.add_argument(
        "--repeated-code-values",
        default=None,
        help="可选逗号分隔的重复工程码；省略时从 source_catalog 推断",
    )
    parser.add_argument(
        "--shifts-json",
        type=Path,
        help="可选序列 JSON；优先读取 cumulative_shifts，兼容 frame_shifts 后平移首帧候选坐标",
    )
    parser.add_argument("--crop-padding-px", type=float, default=128.0, help="局部裁剪 padding pixel")
    parser.add_argument("--aperture-radius", type=int, default=4, help="孔径半径 pixel")
    parser.add_argument("--match-radius-px", type=float, default=2.0, help="局部候选一对一匹配半径 pixel")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="Gaussian 候选阈值 sigma")
    parser.add_argument("--min-distance", type=int, default=4, help="候选峰最小间距 pixel")
    parser.add_argument("--psf-fwhm", type=float, default=2.0, help="匹配滤波 PSF FWHM pixel")
    parser.add_argument("--background-box-size", type=int, default=128, help="局部背景/RMS 网格尺寸 pixel")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量层孔径通量 SNR 下限")
    parser.add_argument("--min-psf-support-pixels", type=int, default=3, help="3×3 PSF 支持像素下限")
    parser.add_argument(
        "--proposal-mode",
        choices=("gaussian", "hybrid", "ensemble"),
        default="hybrid",
        help="局部重检宽筛模式",
    )
    parser.add_argument("--keep-linear-artifacts", action="store_true", help="不标记线状结构")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/pair-support-audit"),
        help="CSV/JSON 输出目录",
    )
    return parser


def _load_shifts(path: Path | None):
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shifts JSON must contain an object")
    raw_shifts = payload.get("cumulative_shifts")
    field_name = "cumulative_shifts"
    if raw_shifts is None:
        raw_shifts = payload.get("frame_shifts")
        field_name = "frame_shifts"
    if raw_shifts is None:
        raise ValueError("shifts JSON must contain cumulative_shifts or frame_shifts")
    try:
        return tuple((float(item[0]), float(item[1])) for item in raw_shifts)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain x/y pairs") from exc


def _load_codes(value: str | None):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(sorted(args.data_dir.glob("*.fits")))
    if not paths:
        print(f"rst19-pair-support-audit: 数据目录没有 FITS 文件：{args.data_dir}", file=sys.stderr)
        return 2
    try:
        sources = load_detection_catalog_csv(args.source_catalog)
        source_by_id = {source.detection_id: source for source in sources}
        primary = source_by_id[args.primary_id]
        secondary = source_by_id[args.secondary_id]
        modes = tuple(item.strip() for item in args.mask_modes.split(",") if item.strip())
        shifts = _load_shifts(args.shifts_json)
        codes = _load_codes(args.repeated_code_values)
    except KeyError as exc:
        print(
            f"rst19-pair-support-audit: source catalog 中不存在 detection_id={exc.args[0]}",
            file=sys.stderr,
        )
        return 2
    except (IndexError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"rst19-pair-support-audit: 输入读取失败：{exc}", file=sys.stderr)
        return 2

    def report(index: int, total: int) -> None:
        print(f"pair support audit: frame {index}/{total}", file=sys.stderr, flush=True)

    try:
        result = run_pair_support_audit(
            paths,
            (primary.x, primary.y),
            (secondary.x, secondary.y),
            template_sources=sources,
            primary_detection_id=primary.detection_id,
            secondary_detection_id=secondary.detection_id,
            frame_shifts=shifts,
            mask_modes=modes,
            repeated_code_values=codes,
            crop_padding_px=args.crop_padding_px,
            aperture_radius=args.aperture_radius,
            match_radius_px=args.match_radius_px,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            psf_fwhm=args.psf_fwhm,
            background_box_size=args.background_box_size,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            proposal_mode=args.proposal_mode,
            reject_linear_artifacts=not args.keep_linear_artifacts,
            progress=report,
        )
        output = write_pair_support_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-pair-support-audit: 审计失败：{exc}", file=sys.stderr)
        return 2

    print(f"rst19-pair-support-audit: 审计已写入 {output}")
    print(result.conclusion)
    for row in result.rows:
        if row.frame_index == 1:
            print(
                f"F01 {row.mask_mode}: assigned={row.assigned_candidate_count} "
                f"primary={row.primary_re_detected_id} secondary={row.secondary_re_detected_id} "
                f"shared_fraction=({row.primary_shared_positive_fraction},"
                f"{row.secondary_shared_positive_fraction}) "
                f"exclusive_snr=({row.primary_exclusive_snr},{row.secondary_exclusive_snr})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
