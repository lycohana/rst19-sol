"""经验 PSF 近邻候选形状审计命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .empirical_pair import (
    load_detection_catalog_csv,
    run_empirical_psf_pair_audit,
    write_empirical_psf_pair_audit_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用源表构造全局/局部经验 PSF，逐帧复核两个近邻候选的形状"
    )
    parser.add_argument("data_dir", type=Path, help="包含 FITS 序列的目录")
    parser.add_argument("--source-catalog", type=Path, required=True, help="rst19-sources 产生的首帧 source_catalog.csv")
    parser.add_argument("--primary-id", type=int, required=True, help="模板源表中的主候选 detection_id")
    parser.add_argument("--secondary-id", type=int, required=True, help="模板源表中的副候选 detection_id")
    parser.add_argument(
        "--coordinate-mode",
        choices=("centroid", "peak"),
        default="centroid",
        help="候选裁剪中心使用测量质心 x/y 或整数匹配峰 peak_x/peak_y；默认质心",
    )
    parser.add_argument("--support-radius", type=int, default=7, help="经验 PSF 半径 pixel")
    parser.add_argument("--max-template-sources", type=int, default=64, help="全局/局部模板最多使用的源数")
    parser.add_argument("--grid-size", type=int, default=4, help="局部模板空间网格每轴格数")
    parser.add_argument(
        "--shifts-json",
        type=Path,
        help="可选序列 JSON；读取 cumulative_shifts 后按注册 detector 位置移动候选",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/empirical-pair-psf-audit"), help="CSV/JSON/PNG 输出目录")
    return parser


def _target_position(source, mode: str) -> tuple[float, float]:
    if mode == "peak" and source.peak_x is not None and source.peak_y is not None:
        return float(source.peak_x), float(source.peak_y)
    return float(source.x), float(source.y)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(sorted(args.data_dir.glob("*.fits")))
    if not paths:
        print(f"rst19-empirical-pair-audit: 数据目录没有 FITS 文件：{args.data_dir}", file=sys.stderr)
        return 2
    try:
        sources = load_detection_catalog_csv(args.source_catalog)
        source_by_id = {source.detection_id: source for source in sources}
        primary = source_by_id[args.primary_id]
        secondary = source_by_id[args.secondary_id]
    except KeyError as exc:
        print(f"rst19-empirical-pair-audit: source catalog 中不存在 detection_id={exc.args[0]}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"rst19-empirical-pair-audit: 无法读取源表：{exc}", file=sys.stderr)
        return 2

    frame_shifts = None
    if args.shifts_json is not None:
        try:
            payload = json.loads(args.shifts_json.read_text(encoding="utf-8"))
            frame_shifts = tuple((float(item[0]), float(item[1])) for item in payload["cumulative_shifts"])
        except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
            print(f"rst19-empirical-pair-audit: 无法读取 --shifts-json：{exc}", file=sys.stderr)
            return 2

    def report(index: int, total: int) -> None:
        print(f"empirical pair PSF audit: frame {index}/{total}", file=sys.stderr, flush=True)

    try:
        result = run_empirical_psf_pair_audit(
            paths,
            _target_position(primary, args.coordinate_mode),
            _target_position(secondary, args.coordinate_mode),
            template_sources=sources,
            primary_detection_id=primary.detection_id,
            secondary_detection_id=secondary.detection_id,
            template_frame=paths[0],
            template_catalog_path=args.source_catalog,
            frame_shifts=frame_shifts,
            support_radius=args.support_radius,
            max_template_sources=args.max_template_sources,
            grid_size=args.grid_size,
            progress=report,
        )
        output = write_empirical_psf_pair_audit_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-empirical-pair-audit: {exc}", file=sys.stderr)
        return 2

    print(f"rst19-empirical-pair-audit: 审计已写入 {output}")
    print(result.conclusion)
    for row in result.rows:
        print(
            f"F{row.frame_index:02d}: global_corr="
            f"({row.primary_global_correlation}, {row.secondary_global_correlation}) "
            f"global_resid=({row.primary_global_relative_residual}, {row.secondary_global_relative_residual}) "
            f"local_template={row.local_template_available}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
