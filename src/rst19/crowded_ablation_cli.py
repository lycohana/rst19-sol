"""CLI for paired crowded-companion ablation on a frozen FITS background."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from .ablation import write_ablation_html
from .crowded_ablation import run_crowded_pair_ablation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在同一真实 FITS 背景块与冻结真值上比较拥挤双源、质量门和局部去混叠"
    )
    parser.add_argument("fits_path", type=Path, help="真实背景 FITS；会校验输入 SHA-256")
    parser.add_argument("--reuse-manifest", type=Path, required=True, help="点源实验的固定目标 manifest，用作候选背景锚点")
    parser.add_argument("--point-run-dir", type=Path, required=True, help="统一 A/B/C 点源实验目录，用于合并 HTML")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp") / "ablation" / f"crowded-{time.strftime('%Y%m%d-%H%M%S')}",
        help="实验产物目录；非空目录不会被覆盖",
    )
    parser.add_argument("--report", type=Path, default=Path("deliverables/ablation/index.html"), help="合并报告路径")
    parser.add_argument("--anchors", type=int, default=8, help="通过同图无注入筛选的固定 FITS 背景块数量")
    parser.add_argument("--match-radius-px", type=float, default=3.0, help="统一真值与候选匹配半径，pixel")
    parser.add_argument("--motion-run-dir", type=Path, help="合并已有运动轨迹关联实验")
    parser.add_argument("--motion-proposal-run-dir", type=Path, help="合并已有像素级运动提案实验")
    parser.add_argument("--motion-crossing-run-dir", type=Path, help="合并已有交叉轨迹压力测试")
    parser.add_argument("--demo-summary", type=Path, help="单帧 CLI 演示摘要 JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(message: str) -> None:
        print(f"crowded-ablation: {message}", file=sys.stderr, flush=True)

    try:
        manifest = run_crowded_pair_ablation(
            args.fits_path,
            args.reuse_manifest,
            args.out_dir,
            anchor_count=args.anchors,
            match_radius_px=args.match_radius_px,
            progress=progress,
        )
        repo_root = Path(__file__).resolve().parents[2]
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=False, capture_output=True, text=True
        ).stdout.strip() or None
        source_paths = (
            "src/rst19/crowded_ablation.py",
            "src/rst19/crowded_ablation_cli.py",
            "src/rst19/ablation.py",
            "src/rst19/detection.py",
            "src/rst19/experiments.py",
            "src/rst19/fits.py",
        )
        manifest["execution"] = {
            "argv": sys.argv if argv is None else ["rst19-crowded-ablation", *argv],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "source_revision": revision,
            "source_file_sha256": {
                relative: hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
                for relative in source_paths
            },
            "source_worktree_dirty": bool(subprocess.run(
                ["git", "status", "--porcelain", "--", "src/rst19", "pyproject.toml", "tests"],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()),
        }
        manifest_path = Path(args.out_dir).resolve() / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report = write_ablation_html(
            args.point_run_dir,
            args.report,
            motion_run_dir=args.motion_run_dir,
            motion_proposal_run_dir=args.motion_proposal_run_dir,
            motion_crossing_run_dir=args.motion_crossing_run_dir,
            demo_summary_path=args.demo_summary,
            crowded_run_dir=args.out_dir,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"rst19-crowded-ablation: failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "status": manifest["status"],
        "run_dir": str(Path(args.out_dir).resolve()),
        "manifest": str(manifest_path),
        "report": str(report),
        "anchors": manifest["patch_selection"]["accepted_anchor_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
