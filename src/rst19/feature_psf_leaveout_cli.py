"""按特征类别做逐源留一法经验 PSF 交叉核验的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .empirical_pair import load_detection_catalog_csv
from .feature_psf_leaveout import (
    run_feature_psf_leaveout_audit,
    write_feature_psf_leaveout_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按各首要特征类别比较全局经验 PSF 与逐源留一经验 PSF"
    )
    parser.add_argument("fits_path", type=Path, help="生成 source_catalog.csv 的首帧 FITS")
    parser.add_argument("--source-catalog", type=Path, required=True, help="rst19-sources 产生的全量源表")
    parser.add_argument("--per-class", type=int, default=16, help="每个特征类别的确定性抽样上限")
    parser.add_argument("--support-radius", type=int, default=7, help="经验 PSF 支持半径 pixel")
    parser.add_argument("--max-template-sources", type=int, default=20, help="每次经验 PSF 最多使用的模板源数")
    parser.add_argument("--correlation-threshold", type=float, default=0.8, help="形状相关度审计线，不是恒星概率")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-psf-leaveout-audit"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sources = load_detection_catalog_csv(args.source_catalog)
        result = run_feature_psf_leaveout_audit(
            args.fits_path,
            sources,
            catalog_path=args.source_catalog,
            per_class=args.per_class,
            support_radius=args.support_radius,
            max_template_sources=args.max_template_sources,
            correlation_threshold=args.correlation_threshold,
            progress=lambda index, total: print(
                f"feature PSF leave-one-out: sample {index}/{total}", file=sys.stderr, flush=True
            ),
        )
        output = write_feature_psf_leaveout_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-psf-leaveout: 审计失败：{exc}", file=sys.stderr)
        return 2

    print(f"rst19-feature-psf-leaveout: 审计已写入 {output}")
    print(result.conclusion)
    for row in result.class_rows:
        print(
            f"{row.feature_class}: sample={row.sample_count} leaveout_valid={row.leaveout_valid_count} "
            f"corr_median={row.leaveout_correlation_median} "
            f"residual_median={row.leaveout_residual_median}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
