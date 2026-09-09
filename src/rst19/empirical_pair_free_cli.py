"""自由位置经验 PSF 近邻双源审计命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .empirical_pair import load_detection_catalog_csv
from .empirical_pair_free import (
    run_empirical_free_pair_audit,
    write_empirical_free_pair_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在同一经验 PSF 和掩膜下联合优化 K=1/K=2 的源位置、幅度和背景"
    )
    parser.add_argument("data_dir", type=Path, help="包含 FITS 序列的目录")
    parser.add_argument("--source-catalog", type=Path, required=True, help="首帧 source_catalog.csv")
    parser.add_argument("--primary-id", type=int, required=True, help="主候选 detection_id")
    parser.add_argument("--secondary-id", type=int, required=True, help="副候选 detection_id")
    parser.add_argument("--support-radius", type=int, default=7, help="经验 PSF 支持半径 pixel")
    parser.add_argument("--max-template-sources", type=int, default=64, help="经验 PSF 最多模板源数")
    parser.add_argument(
        "--patch-padding-px",
        type=float,
        default=None,
        help="局部窗口相对 pair 包围盒的 padding；默认 max(radius+3,10)",
    )
    parser.add_argument(
        "--single-position-radius-px",
        type=float,
        default=2.0,
        help="K=1 相对 pair 中点的位置边界半径；默认 2 pixel",
    )
    parser.add_argument(
        "--double-position-radius-px",
        type=float,
        default=1.25,
        help="K=2 每个分量相对候选锚点的位置边界半径；默认 1.25 pixel",
    )
    parser.add_argument(
        "--minimum-fitted-separation-px",
        type=float,
        default=1.0,
        help="确认线要求的拟合后最小双源间距；默认 1 pixel",
    )
    parser.add_argument(
        "--mask-modes",
        default="raw,range_masked,sentinel_masked",
        help="逗号分隔的掩膜模式；也支持 repeated_code_masked 组合",
    )
    parser.add_argument(
        "--repeated-code-values",
        default=None,
        help="可选逗号分隔的工程重复码；省略时从源表推断",
    )
    parser.add_argument("--shifts-json", type=Path, help="可选序列 JSON；读取 cumulative_shifts")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/empirical-pair-free-audit"),
        help="CSV/JSON/PNG 输出目录",
    )
    return parser


def _target_position(source) -> tuple[float, float]:
    return float(source.x), float(source.y)


def _load_frame_shifts(path: Path | None):
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple((float(item[0]), float(item[1])) for item in payload["cumulative_shifts"])


def _load_repeated_code_values(value: str | None):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(sorted(args.data_dir.glob("*.fits")))
    if not paths:
        print(f"rst19-empirical-pair-free: 数据目录没有 FITS 文件：{args.data_dir}", file=sys.stderr)
        return 2
    try:
        sources = load_detection_catalog_csv(args.source_catalog)
        source_by_id = {source.detection_id: source for source in sources}
        primary = source_by_id[args.primary_id]
        secondary = source_by_id[args.secondary_id]
        mask_modes = tuple(item.strip() for item in args.mask_modes.split(",") if item.strip())
        frame_shifts = _load_frame_shifts(args.shifts_json)
        repeated_code_values = _load_repeated_code_values(args.repeated_code_values)
    except KeyError as exc:
        print(
            f"rst19-empirical-pair-free: source catalog 中不存在 detection_id={exc.args[0]}",
            file=sys.stderr,
        )
        return 2
    except (IndexError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"rst19-empirical-pair-free: 输入读取失败：{exc}", file=sys.stderr)
        return 2

    def report(index: int, total: int) -> None:
        print(f"empirical free pair audit: frame {index}/{total}", file=sys.stderr, flush=True)

    try:
        result = run_empirical_free_pair_audit(
            paths,
            _target_position(primary),
            _target_position(secondary),
            template_sources=sources,
            primary_detection_id=primary.detection_id,
            secondary_detection_id=secondary.detection_id,
            template_frame=paths[0],
            template_catalog_path=args.source_catalog,
            frame_shifts=frame_shifts,
            mask_modes=mask_modes,
            repeated_code_values=repeated_code_values,
            support_radius=args.support_radius,
            max_template_sources=args.max_template_sources,
            patch_padding_px=args.patch_padding_px,
            single_position_radius_px=args.single_position_radius_px,
            double_position_radius_px=args.double_position_radius_px,
            minimum_fitted_separation_px=args.minimum_fitted_separation_px,
            progress=report,
        )
        output = write_empirical_free_pair_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-empirical-pair-free: 审计失败：{exc}", file=sys.stderr)
        return 2

    print(f"rst19-empirical-pair-free: 审计已写入 {output}")
    print(result.conclusion)
    for row in result.rows:
        if row.mask_mode == "raw":
            print(
                f"F{row.frame_index:02d}: delta_bic={row.delta_bic_single_minus_double} "
                f"second_snr={row.double_second_component_snr} "
                f"separation={row.fitted_separation_px} "
                f"at_bound={row.double_positions_at_bound}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
