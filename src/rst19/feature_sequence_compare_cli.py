"""已完成序列特征审计结果的比较命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .sequence_feature_comparison import compare_sequence_feature_runs, write_sequence_feature_comparison_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较多组已生成的 15 帧特征持久性 JSON，不重新运行 FITS 检测"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=JSON",
        help="一组运行标签和 JSON 路径，例如 SNR5=tmp/sequence_feature_persistence.json；至少两组",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/sequence-feature-parameter-comparison"),
        help="比较表输出目录",
    )
    return parser


def _parse_runs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"--run 格式应为 NAME=JSON，收到：{value!r}")
        name = name.strip()
        if name in result:
            raise ValueError(f"重复的运行标签：{name}")
        result[name] = Path(raw_path.strip())
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runs = _parse_runs(args.run)
        result = compare_sequence_feature_runs(runs)
        output = write_sequence_feature_comparison_artifacts(result, args.out_dir)
        print(f"rst19-feature-sequence-compare: 比较表已写入 {output}")
        for summary in result.runs:
            print(
                f"{summary.configuration}: frames={summary.unique_frame_count} "
                f"candidate={summary.first_frame_candidate_count} "
                f"quality={summary.first_frame_quality_count}"
            )
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"rst19-feature-sequence-compare: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
