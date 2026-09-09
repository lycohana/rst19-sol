"""近邻双框匹配响应反事实归因命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .empirical_pair import load_detection_catalog_csv
from .pair_response_attribution import (
    run_pair_response_attribution,
    write_pair_response_attribution_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较近邻双候选在重复码/极端负值留出下的局部 Gaussian 响应和像素归因"
    )
    parser.add_argument("frame", type=Path, help="原始 FITS 文件")
    parser.add_argument("--source-catalog", type=Path, required=True, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument("--primary-id", type=int, required=True, help="主候选 detection_id")
    parser.add_argument("--secondary-id", type=int, required=True, help="副候选 detection_id")
    parser.add_argument("--patch-padding-px", type=int, default=10, help="候选峰周围局部窗口外扩像素")
    parser.add_argument("--psf-fwhm", type=float, default=2.0, help="Gaussian 匹配滤波 FWHM pixel")
    parser.add_argument("--repeated-code-values", default=None, help="可选逗号分隔的重复工程码；省略时从源表推断")
    parser.add_argument("--negative-anomaly-threshold-adu", type=float, default=-1000.0, help="极端负值留出阈值")
    parser.add_argument("--maximum-neighborhood-px", type=int, default=3, help="局部 Gaussian 峰判定窗口；须为奇数")
    parser.add_argument("--maximum-count", type=int, default=12, help="每个掩膜模式最多导出多少个局部响应峰")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/pair-response-attribution"), help="CSV/JSON 输出目录")
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
        result = run_pair_response_attribution(
            args.frame,
            primary,
            secondary,
            patch_padding_px=args.patch_padding_px,
            psf_fwhm=args.psf_fwhm,
            repeated_code_values=_load_codes(args.repeated_code_values),
            negative_anomaly_threshold_adu=args.negative_anomaly_threshold_adu,
            maximum_neighborhood_px=args.maximum_neighborhood_px,
            maximum_count=args.maximum_count,
        )
        output = write_pair_response_attribution_artifacts(result, args.out_dir)
    except KeyError as exc:
        print(f"rst19-pair-response-attribution: source catalog 中不存在 detection_id={exc.args[0]}", file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-pair-response-attribution: 审计失败：{exc}", file=sys.stderr)
        return 2

    print(f"pair response attribution: {output}")
    print(result.conclusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
