"""固定位置 K 源局部 PSF 模型敏感性审计命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .experiments import run_local_multipsf_audit, write_local_multipsf_audit_artifacts


def _parse_positions(value: str) -> tuple[tuple[float, float], ...]:
    positions: list[tuple[float, float]] = []
    for item in value.split(";"):
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"位置必须是 x,y;x,y 格式：{item}")
        positions.append((float(parts[0]), float(parts[1])))
    if len(positions) < 2:
        raise ValueError("至少需要两个位置")
    return tuple(positions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较同一局部的固定位置 K=1..N PSF 模型，审计多个框是否需要多个点源"
    )
    parser.add_argument("data_dir", type=Path, help="包含 FITS 序列的目录")
    parser.add_argument(
        "--positions",
        default="1269,3465;1274,3467;1272,3453",
        help="按 detector 坐标给出位置，格式为 x,y;x,y;...；默认使用截图局部三个候选",
    )
    parser.add_argument("--psf-fwhm", type=float, default=2.0, help="固定 Gaussian PSF 的 FWHM（pixel）")
    parser.add_argument(
        "--mask-modes",
        nargs="+",
        choices=("raw", "range_masked", "sentinel_masked", "range_sentinel_masked"),
        default=("raw", "range_masked", "sentinel_masked"),
        help="并行比较的原始/异常码掩膜口径",
    )
    parser.add_argument("--shifts-json", type=Path, help="可选 JSON；读取 cumulative_shifts 后按帧移动 detector 位置")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/local-multipsf-audit"), help="CSV/JSON/PNG 输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = tuple(sorted(args.data_dir.glob("*.fits")))
    if not paths:
        print(f"rst19-multipsf-audit: 数据目录没有 FITS 文件：{args.data_dir}", file=sys.stderr)
        return 2
    try:
        positions = _parse_positions(args.positions)
    except (TypeError, ValueError) as exc:
        print(f"rst19-multipsf-audit: {exc}", file=sys.stderr)
        return 2

    frame_shifts = None
    if args.shifts_json is not None:
        try:
            payload = json.loads(args.shifts_json.read_text(encoding="utf-8"))
            frame_shifts = tuple(
                (float(item[0]), float(item[1])) for item in payload["cumulative_shifts"]
            )
        except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
            print(f"rst19-multipsf-audit: 无法读取 --shifts-json：{exc}", file=sys.stderr)
            return 2

    def report(index: int, total: int) -> None:
        print(f"multipsf audit: frame {index}/{total}", file=sys.stderr, flush=True)

    try:
        result = run_local_multipsf_audit(
            paths,
            positions,
            psf_fwhm=args.psf_fwhm,
            mask_modes=tuple(args.mask_modes),
            frame_shifts=frame_shifts,
            progress=report,
        )
        output = write_local_multipsf_audit_artifacts(result, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rst19-multipsf-audit: {exc}", file=sys.stderr)
        return 2

    print(f"rst19-multipsf-audit: 审计已写入 {output}")
    print(result.conclusion)
    for row in result.rows:
        if row.component_count >= 2 and row.mask_mode == "raw":
            print(
                f"F{row.frame_index:02d} {row.mask_mode} K={row.component_count}: "
                f"delta_bic={row.delta_bic_from_single} "
                f"component_snr={row.component_snr} "
                f"active={row.active_component_count} "
                f"confirmed={row.confirmation_line}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
