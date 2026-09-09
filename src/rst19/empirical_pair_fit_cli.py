"""经验 PSF 单源/双源锚点敏感性审计命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .empirical_pair import load_detection_catalog_csv
from .empirical_pair_fit import (
    run_empirical_pair_fit_audit,
    write_empirical_pair_fit_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用经验 PSF 比较近邻候选的固定双源模型与最佳单源位置控制"
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
        "--coordinate-mode",
        choices=("centroid", "peak"),
        default="centroid",
        help="候选位置使用质心 x/y 或 peak_x/peak_y；默认质心",
    )
    parser.add_argument("--support-radius", type=int, default=7, help="经验 PSF 半径 pixel")
    parser.add_argument("--max-template-sources", type=int, default=64, help="经验 PSF 模板最多使用的源数")
    parser.add_argument(
        "--patch-padding-px",
        type=float,
        default=None,
        help="双源局部窗口相对候选包围盒的 padding；默认 max(radius+3, 10)",
    )
    parser.add_argument(
        "--best-single-search-radius-px",
        type=float,
        default=2.0,
        help="最佳单源位置控制相对双源中点的搜索半径；默认 2 pixel",
    )
    parser.add_argument(
        "--best-single-grid-step-px",
        type=float,
        default=0.5,
        help="最佳单源位置网格步长；默认 0.5 pixel",
    )
    parser.add_argument(
        "--mask-modes",
        default="raw,range_masked,sentinel_masked",
        help=(
            "逗号分隔：raw、range_masked、sentinel_masked、range_sentinel_masked；"
            "也可用 repeated_code_masked 及其组合"
        ),
    )
    parser.add_argument(
        "--repeated-code-values",
        default=None,
        help="可选逗号分隔的工程重复码；省略时从 source_catalog 的 repeated_code_values 推断",
    )
    parser.add_argument(
        "--shifts-json",
        type=Path,
        help="可选序列 JSON；读取 cumulative_shifts 后平移候选 detector 坐标",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/empirical-pair-fit-audit"),
        help="CSV/JSON/PNG 输出目录",
    )
    return parser


def _target_position(source, mode: str) -> tuple[float, float]:
    if mode == "peak" and source.peak_x is not None and source.peak_y is not None:
        return float(source.peak_x), float(source.peak_y)
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
        print(f"rst19-empirical-pair-fit: 数据目录没有 FITS 文件：{args.data_dir}", file=sys.stderr)
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
            f"rst19-empirical-pair-fit: source catalog 中不存在 detection_id={exc.args[0]}",
            file=sys.stderr,
        )
        return 2
    except (IndexError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"rst19-empirical-pair-fit: 输入读取失败：{exc}", file=sys.stderr)
        return 2

    def report(index: int, total: int) -> None:
        print(f"empirical pair fit audit: frame {index}/{total}", file=sys.stderr, flush=True)

    try:
        result = run_empirical_pair_fit_audit(
            paths,
            _target_position(primary, args.coordinate_mode),
            _target_position(secondary, args.coordinate_mode),
            template_sources=sources,
            primary_detection_id=primary.detection_id,
            secondary_detection_id=secondary.detection_id,
            template_frame=paths[0],
            template_catalog_path=args.source_catalog,
            frame_shifts=frame_shifts,
            coordinate_mode=args.coordinate_mode,
            mask_modes=mask_modes,
            repeated_code_values=repeated_code_values,
            support_radius=args.support_radius,
            max_template_sources=args.max_template_sources,
            patch_padding_px=args.patch_padding_px,
            best_single_search_radius_px=args.best_single_search_radius_px,
            best_single_grid_step_px=args.best_single_grid_step_px,
            progress=report,
        )
        output = write_empirical_pair_fit_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-empirical-pair-fit: 审计失败：{exc}", file=sys.stderr)
        return 2

    print(f"rst19-empirical-pair-fit: 审计已写入 {output}")
    print(result.conclusion)
    raw_rows = [row for row in result.rows if row.mask_mode == "raw"]
    for row in raw_rows:
        print(
            f"F{row.frame_index:02d}: fixed_delta={row.fixed_delta_bic} "
            f"control_delta={row.best_single_grid_delta_bic} "
            f"second_snr={row.double_second_component_snr}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
