"""比赛 FITS 实际格式和值域审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .format_audit import run_format_audit, write_format_audit_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="审计 FITS 实际尺寸、字节序、特殊值和辅助位置-速度自洽性，不修改检测器"
    )
    parser.add_argument("input_path", type=Path, help="单个 FITS 或包含 FITS 的目录")
    parser.add_argument(
        "--extreme-fraction",
        type=float,
        default=0.9,
        help="统计有符号 16 bit 正负极值的相对阈值，默认 0.9",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/format-audit"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_format_audit(args.input_path, extreme_fraction=args.extreme_fraction)
        output = write_format_audit_artifacts(result, args.out_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-format-audit: {exc}", file=sys.stderr)
        return 2
    summary = result.summary
    velocity = summary["velocity_consistency"]
    print(f"format audit: {output}")
    print(
        f"frames={summary['frame_count']} shape={summary['shape_counts']} "
        f"file_size_ok={summary['all_file_sizes_match_header']} "
        f"auxiliary={summary['complete_auxiliary_frames']} "
        f"velocity_j2000_median={velocity['j2000_relative_error_median']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
