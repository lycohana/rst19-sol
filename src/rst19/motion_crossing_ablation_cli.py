"""Command-line entry point for deterministic crossing-track ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

from .motion_crossing_ablation import run_motion_crossing_ablation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="冻结两种合成交叉轨迹 Detection 输入，比较基础关联与速度预测补充关联"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp") / "ablation" / f"motion-crossing-{datetime.now():%Y%m%d-%H%M%S}",
        help="产物目录；非空目录不会被覆盖",
    )
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_motion_crossing_ablation(args.out_dir)
        repo_root = Path(__file__).resolve().parents[2]
        source_paths = (
            "src/rst19/motion_crossing_ablation.py",
            "src/rst19/motion_crossing_ablation_cli.py",
            "src/rst19/sequence.py",
            "src/rst19/detection.py",
            "pyproject.toml",
        )
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=False, capture_output=True, text=True
        ).stdout.strip() or None
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", *source_paths],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        manifest["execution"] = {
            "argv": sys.argv if argv is None else ["rst19-motion-crossing-ablation", *argv],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "source_revision": revision,
            "source_worktree_dirty": dirty,
            "source_file_sha256": {
                relative: _sha256(repo_root / relative)
                for relative in source_paths
            },
        }
        manifest_path = Path(args.out_dir).resolve() / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"rst19-motion-crossing-ablation: failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_dir": str(Path(args.out_dir).resolve()),
                "manifest": str(manifest_path),
                "summary": str(Path(args.out_dir).resolve() / "summary.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
