"""指定候选的类别内经验分位审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_class_context import run_feature_class_context, write_feature_class_context_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把指定 detection ID 放回其首要特征类别的经验分布中，不重新读取 FITS"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 导出的 source_catalog.csv")
    parser.add_argument(
        "--target-id",
        type=int,
        action="append",
        required=True,
        help="需要审计的 detection ID；可重复传入多个",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/feature-class-context"),
        help="CSV/JSON 输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_class_context(args.catalog, args.target_id)
        output = write_feature_class_context_artifacts(result, args.out_dir)
        print(f"rst19-feature-class-context: 类别上下文已写入 {output}")
        for target in result.targets:
            print(
                f"ID {target.detection_id}: class={target.feature_class} "
                f"quality={target.quality_passed} class_count={target.class_count}"
            )
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"rst19-feature-class-context: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
