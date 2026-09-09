"""特征类别非参数效应量审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_effect_size import (
    DEFAULT_EFFECT_FEATURES,
    run_feature_effect_size_audit,
    write_feature_effect_size_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较 compact_quality 与各特征类别的源级字段分布，不生成物理恒星概率"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument("--reference-class", default="compact_quality")
    parser.add_argument("--comparison-classes", nargs="*", default=None)
    parser.add_argument("--features", nargs="*", default=list(DEFAULT_EFFECT_FEATURES))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/feature-effect-size"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_effect_size_audit(
            args.catalog,
            reference_class=args.reference_class,
            comparison_classes=args.comparison_classes,
            features=args.features,
            top_n=args.top_n,
        )
        output = write_feature_effect_size_artifacts(result, args.out_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-effect-size: {exc}", file=sys.stderr)
        return 2
    print(f"feature effect-size audit: {output}")
    for class_name, features in result.top_features_by_class.items():
        print(f"{class_name}: {', '.join(features)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
