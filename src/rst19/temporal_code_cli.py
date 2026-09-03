"""15 帧 detector 固定码/低变化像素审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .experiments import run_temporal_code_audit, write_temporal_code_audit_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较 15 帧 FITS 的 detector 固定值和低变化特殊码，不改变星点检测结果"
    )
    parser.add_argument("data_dir", type=Path, help="包含 FITS 序列的目录")
    parser.add_argument(
        "--low-variation-span-adu",
        type=int,
        default=2,
        help="低变化像素允许的跨帧最大范围（ADU）",
    )
    parser.add_argument("--code-min-adu", type=int, default=3990, help="低变化正码候选的首帧下限（ADU）")
    parser.add_argument("--code-max-adu", type=int, default=3993, help="低变化正码候选的首帧上限（ADU）")
    parser.add_argument("--sentinel-value", type=int, default=-1, help="固定 sentinel 值")
    parser.add_argument("--focus-x", type=int, default=1274, help="正码候选关注坐标 X")
    parser.add_argument("--focus-y", type=int, default=3466, help="正码候选关注坐标 Y")
    parser.add_argument("--sentinel-focus-x", type=int, default=1271, help="sentinel 关注坐标 X")
    parser.add_argument("--sentinel-focus-y", type=int, default=3465, help="sentinel 关注坐标 Y")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/temporal-code-audit"), help="JSON/CSV 输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(sorted(args.data_dir.glob("*.fits")))
    if not paths:
        print(f"rst19-temporal-codes: 数据目录没有 FITS 文件：{args.data_dir}", file=sys.stderr)
        return 2

    def report(index: int, total: int) -> None:
        print(f"temporal code audit: frame {index}/{total}", file=sys.stderr, flush=True)

    try:
        result = run_temporal_code_audit(
            paths,
            low_variation_span_adu=args.low_variation_span_adu,
            code_min_adu=args.code_min_adu,
            code_max_adu=args.code_max_adu,
            sentinel_value=args.sentinel_value,
            code_focus_xy=(args.focus_x, args.focus_y),
            sentinel_focus_xy=(args.sentinel_focus_x, args.sentinel_focus_y),
            progress=report,
        )
        output = write_temporal_code_audit_artifacts(result, args.out_dir)
        print(f"rst19-temporal-codes: 审计已写入 {output}")
        print(
            f"frames={result.frame_count} exact_stable={result.exact_stable_pixel_count} "
            f"low_variation={result.low_variation_pixel_count} "
            f"code_pixels={result.code_pixel_count} "
            f"code_components={result.code_component_count} "
            f"sentinel_pixels={result.sentinel_exact_stable_pixel_count}"
        )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-temporal-codes: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
