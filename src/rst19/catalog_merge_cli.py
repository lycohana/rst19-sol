"""合并亮星定标层与深星层 Gaia CSV。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .catalog_merge import CatalogMergeError, merge_catalog_csvs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="合并分层 Gaia DR3 CSV，并继承输入完整性审计"
    )
    parser.add_argument(
        "catalogs",
        type=Path,
        nargs="+",
        help="至少两份 CSV；通常一份亮星定标层、一份深星层",
    )
    parser.add_argument("--out", type=Path, required=True, help="合并后的 CatalogSource 兼容 CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = merge_catalog_csvs(args.catalogs, args.out)
    except (CatalogMergeError, OSError, ValueError) as exc:
        print(f"gaia-merge: {exc}", file=sys.stderr)
        return 2
    state = "complete" if result.complete else "incomplete"
    print(
        f"Gaia merged {state}: {result.row_count:,} unique rows -> {result.output_path} "
        f"(duplicates={result.duplicate_count:,}, conflicts={result.conflict_count:,}; audit: {result.audit_path})",
        file=sys.stderr,
    )
    return 0 if result.complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
