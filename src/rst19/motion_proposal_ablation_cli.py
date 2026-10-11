"""CLI for the paired pixel-level sequence motion proposal ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import PIL
import scipy

from .ablation import write_ablation_html
from .motion_proposal_ablation import run_motion_proposal_ablation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在同一冻结像素序列和真值轨迹下比较基础运动提案与多尺度线段提案"
    )
    parser.add_argument(
        "--point-run-dir",
        type=Path,
        required=True,
        help="配套点源 A/B/C 实验目录",
    )
    parser.add_argument(
        "--association-run-dir",
        type=Path,
        help="已完成的冻结检测列表轨迹关联消融目录；提供后合并到同一 HTML 报告",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp") / "ablation" / f"motion-proposal-{time.strftime('%Y%m%d-%H%M%S')}",
        help="运动像素提案实验产物目录；非空目录不会被覆盖",
    )
    parser.add_argument("--report", type=Path, default=Path("deliverables/ablation/index.html"))
    parser.add_argument("--seed", type=int, default=19019)
    return parser


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_execution_metadata() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip() or None
    source_paths = (
        "src/rst19/motion_proposal_ablation.py",
        "src/rst19/sequence.py",
        "src/rst19/single_frame_trails.py",
    )
    source_hashes = {
        relative: _sha256_bytes((repo_root / relative).read_bytes())
        for relative in source_paths
    }
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--", *source_paths],
        cwd=repo_root,
        check=False,
        capture_output=True,
    ).stdout
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "src/rst19", "pyproject.toml", "tests"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "source_revision": revision,
        "source_worktree_dirty": dirty,
        "source_file_sha256": source_hashes,
        "relevant_tracked_diff_sha256": _sha256_bytes(diff),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_motion_proposal_ablation(args.out_dir, seed=args.seed)
        manifest["execution"] = {
            "argv": sys.argv if argv is None else ["rst19-motion-proposal-ablation", *argv],
            "python": sys.version,
            "platform": platform.platform(),
            "scipy": scipy.__version__,
            "pillow": PIL.__version__,
            **_source_execution_metadata(),
        }
        manifest_path = Path(args.out_dir).resolve() / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        report = write_ablation_html(
            args.point_run_dir,
            args.report,
            motion_run_dir=args.association_run_dir,
            motion_proposal_run_dir=args.out_dir,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"rst19-motion-proposal-ablation: failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_dir": str(Path(args.out_dir).resolve()),
                "manifest": str(manifest_path),
                "report": str(report),
                "runtime": manifest["runtime"]["seconds_by_scene_and_version"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
