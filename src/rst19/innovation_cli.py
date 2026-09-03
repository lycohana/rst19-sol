"""创新分析证据导出命令。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .innovation import build_innovation_report, write_innovation_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 rst19 序列 JSON 生成 15 帧创新分析证据")
    parser.add_argument("sequence_json", type=Path, help="SequenceResult.as_dict 导出的 JSON")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp") / "innovation", help="报告输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_innovation_report(args.sequence_json)
    output = write_innovation_artifacts(report, args.out_dir)
    relation = report["relation"]
    motion = report["motion"]
    print(f"innovation report: {output}")
    print(f"frames={relation['frame_count']} duration_s={relation['duration_s']} motion={motion['moving_count']} candidates={motion['candidate_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
