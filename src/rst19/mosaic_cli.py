"""注册 15 帧联合视场大图命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .mosaic import (
    build_registered_mosaic,
    render_mosaic_preview,
    write_mosaic_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 15 帧序列的累计平移生成可审计的注册联合视场大图"
    )
    parser.add_argument("data_dir", type=Path, help="包含 FITS 序列的目录")
    parser.add_argument(
        "--sequence-json",
        type=Path,
        help="rst19-sequence 的 JSON 结果；不提供时只读取同目录中的序列结果不会自动猜测平移",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp") / "registered-mosaic",
        help="大图产物目录，默认 tmp/registered-mosaic",
    )
    parser.add_argument(
        "--mode",
        choices=("robust_mean", "median"),
        default="robust_mean",
        help="重叠区融合：稳健均值（默认）或时间中值",
    )
    parser.add_argument("--clip-sigma", type=float, default=3.0, help="稳健均值的 MAD 裁剪倍数")
    parser.add_argument("--tile-rows", type=int, default=256, help="分块高度，控制内存峰值")
    parser.add_argument("--raw-preview", action="store_true", help="同时按原始显示口径输出预览")
    return parser


def _load_shifts(path: Path) -> tuple[tuple[float, float], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("sequence"), dict):
        payload = payload["sequence"]
    shifts = payload.get("cumulative_shifts")
    if not isinstance(shifts, list):
        raise ValueError(f"JSON 缺少 cumulative_shifts：{path}")
    return tuple((float(row[0]), float(row[1])) for row in shifts)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(sorted(args.data_dir.glob("*.fits")))
    if len(paths) < 2:
        print(f"rst19-mosaic: 至少需要 2 个 FITS 文件：{args.data_dir}", file=sys.stderr)
        return 2
    if args.sequence_json is None:
        print("rst19-mosaic: 必须提供 15 帧序列 JSON，以避免在未配准时伪造大图", file=sys.stderr)
        return 2
    try:
        shifts = _load_shifts(args.sequence_json)
        if len(shifts) != len(paths):
            raise ValueError(f"序列 JSON 有 {len(shifts)} 个平移，但数据目录有 {len(paths)} 帧")

        def progress(value: float, label: str) -> None:
            print(f"{value:6.1f}%  {label}", file=sys.stderr, flush=True)

        result = build_registered_mosaic(
            paths,
            shifts,
            combine_mode=args.mode,
            clip_sigma=args.clip_sigma,
            tile_rows=args.tile_rows,
            progress=progress,
        )
        preview = render_mosaic_preview(result.image, mode="enhanced")
        write_mosaic_artifacts(args.out_dir, result, preview=preview)
        if args.raw_preview:
            render_mosaic_preview(result.image, mode="raw").save(args.out_dir / "registered_mosaic_raw_preview.png")
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"rst19-mosaic: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
