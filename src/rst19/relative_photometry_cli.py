"""Fit a robust relative magnitude scale from a JSON observation table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .relative_photometry import fit_relative_photometry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对 15 帧观测行拟合相对星等标尺（不联网、不产生绝对星等）"
    )
    parser.add_argument("input", type=Path, help="JSON 数组，或包含 observations/rows/items 的 JSON")
    parser.add_argument("--out", type=Path, help="可选结果 JSON；不提供时输出到 stdout")
    parser.add_argument("--spatial-order", type=int, choices=(0, 1, 2), default=0, help="空间响应项阶数")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="源分块留出比例，默认 0.2")
    parser.add_argument("--random-state", type=int, default=0, help="留出分块随机种子")
    parser.add_argument("--mad-threshold", type=float, default=5.0, help="迭代 MAD 剔除阈值")
    return parser


def _observations(payload: object) -> object:
    if isinstance(payload, dict):
        for key in ("observations", "relative_observations", "rows", "items"):
            if key in payload:
                return payload[key]
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        result = fit_relative_photometry(
            _observations(payload),
            spatial_order=args.spatial_order,
            validation_fraction=args.validation_fraction,
            random_state=args.random_state,
            mad_threshold=args.mad_threshold,
        )
        rendered = json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"relative-photometry: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
