"""辅助姿态/光轴一致性审计命令。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .attitude import audit_auxiliary_boresight, write_auxiliary_boresight_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="审计比赛 FITS 辅助四元数与 ra/dec 光轴的一致性")
    parser.add_argument("input_path", type=Path, help="单个 FITS 或包含 FITS 的目录")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/auxiliary-attitude-audit"), help="CSV/JSON 输出目录")
    return parser


def _paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.fits"))
    raise FileNotFoundError(f"输入路径不存在：{input_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = _paths(args.input_path)
        if not paths:
            raise ValueError(f"目录中没有 FITS 文件：{args.input_path}")
        rows = audit_auxiliary_boresight(paths)
        output = write_auxiliary_boresight_artifacts(rows, args.out_dir)
        print(f"auxiliary audit: {output}")
        print(f"frames={len(rows)} summary={output / 'auxiliary_boresight_audit.json'}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"rst19-aux-audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
