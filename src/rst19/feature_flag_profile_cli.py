"""按类别拆解质量/审计旗标的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_flag_profile import run_feature_flag_profile, write_feature_flag_profile_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按首要特征类别统计重叠质量旗标，不重新读取 FITS"
    )
    parser.add_argument("catalog", type=Path, help="rst19-sources 生成的 source_catalog.csv")
    parser.add_argument(
        "--target-ids",
        nargs="*",
        default=["82931", "82934", "44132"],
        help="输出原始旗标的候选 ID",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/feature-flag-profile"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_flag_profile(args.catalog, target_ids=args.target_ids)
        output = write_feature_flag_profile_artifacts(result, args.out_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-flag-profile: {exc}", file=sys.stderr)
        return 2
    print(f"feature flag profile: {output}")
    for row in result.target_rows:
        print(f"{row.detection_id}: {row.feature_class}, quality={row.quality_passed}, flags={row.flags or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
