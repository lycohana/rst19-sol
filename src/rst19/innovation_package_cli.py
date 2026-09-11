"""命令行生成创新交付包。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .innovation_package import PRIMARY_STATUS, build_innovation_package, write_innovation_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从真实背景分层注入实验表生成可审计的创新交付包。不会读取 FITS，也不会重跑检测。"
    )
    parser.add_argument("stratified_artifact", type=Path, help="分层注入 JSON 或 CSV")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp") / "innovation-package",
        help="输出目录（默认：tmp/innovation-package；建议发布前改到临时目录）",
    )
    parser.add_argument("--sequence-report", type=Path, help="可选：15 帧 innovation_report.json，仅作为基础任务支撑")
    parser.add_argument("--feature-matrix", type=Path, help="可选：特征证据矩阵 JSON，仅作为诊断证据")
    parser.add_argument("--min-injections", type=int, default=8, help="进入主剖面的最小无歧义注入数，默认 8")
    parser.add_argument("--min-levels", type=int, default=3, help="进入主剖面的最小共同强度层数，默认 3")
    parser.add_argument(
        "--require-defensible",
        action="store_true",
        help="若未达到条件可答辩门槛则返回 2",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        package = build_innovation_package(
            args.stratified_artifact,
            sequence_report=args.sequence_report,
            feature_matrix=args.feature_matrix,
            min_injections=args.min_injections,
            min_levels=args.min_levels,
        )
        paths = write_innovation_package(package, args.out_dir)
    except (OSError, ValueError) as exc:
        print(f"innovation-package error: {exc}", file=sys.stderr)
        return 2

    print(f"status: {package['status']}")
    print(f"primary: {package['selected_innovation']['title']}")
    print(f"json: {paths['json']}")
    print(f"csv: {paths['csv']}")
    print(f"markdown: {paths['markdown']}")
    if args.require_defensible and package["status"] != PRIMARY_STATUS:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
