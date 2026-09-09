"""类别转移流向的快速后处理命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_transition_flow import (
    summarize_sequence_feature_transition_flow,
    write_sequence_feature_transition_flow_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从已生成的 sequence_feature_class_transition.csv 提取类别流向，不重新运行 FITS 检测"
    )
    parser.add_argument("csv_path", type=Path, help="sequence_feature_class_transition.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/sequence-feature-transition-flow"),
        help="输出目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = summarize_sequence_feature_transition_flow(args.csv_path)
        output = write_sequence_feature_transition_flow_artifacts(result, args.out_dir)
        print(
            "rst19-feature-transition-flow: "
            f"groups={result.included_group_count} skipped_zero_anchor={result.skipped_zero_anchor_group_count} "
            f"output={output}"
        )
        for row in result.rows:
            top = (
                f"{row.top_non_same_feature_class}={row.top_non_same_matched_count}"
                if row.top_non_same_feature_class is not None
                else "-"
            )
            print(
                f"{row.layer}/{row.anchor_feature_class}: "
                f"q_any={row.q_any!r} q_same={row.q_same!r} top_non_same={top}"
            )
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"rst19-feature-transition-flow: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
