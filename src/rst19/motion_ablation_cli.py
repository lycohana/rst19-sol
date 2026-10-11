"""Command-line entry point for the frozen-source motion-association ablation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import PIL
import scipy

from .ablation import write_ablation_html
from .motion_ablation import run_motion_association_ablation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在同一冻结逐帧质量源列表和轨迹真值下比较普通关联与速度预测补充关联"
    )
    parser.add_argument("--cache", type=Path, required=True, help="已有序列结果缓存；原始 FITS SHA-256 会逐帧复核")
    parser.add_argument("--point-run-dir", type=Path, required=True, help="配套点源提案/质量门实验目录")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp") / "ablation" / f"motion-{time.strftime('%Y%m%d-%H%M%S')}",
        help="运动关联产物目录；非空目录不会被覆盖",
    )
    parser.add_argument("--report", type=Path, default=Path("deliverables/ablation/index.html"), help="合并报告路径")
    parser.add_argument("--seed", type=int, default=19019, help="只用于一次性轨迹选址；坐标保存在输出 JSON")
    parser.add_argument("--min-presence", type=int, help="最少观测帧数；默认复用缓存值")
    parser.add_argument("--link-radius-px", type=float, help="普通最近邻半径；默认复用缓存值")
    parser.add_argument("--registration-radius-px", type=float, default=8.0)
    parser.add_argument("--min-flux-snr", type=float, default=7.0)
    parser.add_argument("--max-step-px", type=float, default=20.0)
    parser.add_argument("--prediction-gate-px", type=float, default=4.5)
    parser.add_argument("--max-fit-rms-px", type=float, default=1.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(message: str) -> None:
        print(f"motion-ablation: {message}", file=sys.stderr, flush=True)

    try:
        manifest = run_motion_association_ablation(
            args.cache,
            args.out_dir,
            seed=args.seed,
            min_presence=args.min_presence,
            link_radius_px=args.link_radius_px,
            registration_radius_px=args.registration_radius_px,
            min_flux_snr=args.min_flux_snr,
            max_step_px=args.max_step_px,
            prediction_gate_px=args.prediction_gate_px,
            max_fit_rms_px=args.max_fit_rms_px,
            progress=progress,
        )
        repo_root = Path(__file__).resolve().parents[2]
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=False, capture_output=True, text=True
        ).stdout.strip() or None
        source_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--", "src/rst19", "pyproject.toml", "tests/test_ablation.py"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip())
        manifest["execution"] = {
            "argv": sys.argv if argv is None else ["rst19-motion-ablation", *argv],
            "python": sys.version,
            "platform": platform.platform(),
            "scipy": scipy.__version__,
            "pillow": PIL.__version__,
            "source_revision": revision,
            "source_worktree_dirty": source_dirty,
        }
        manifest_path = Path(args.out_dir).resolve() / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report = write_ablation_html(args.point_run_dir, args.report, motion_run_dir=args.out_dir)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"rst19-motion-ablation: failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "status": manifest["status"],
        "run_dir": str(Path(args.out_dir).resolve()),
        "manifest": str(manifest_path),
        "report": str(report),
        "baseline_seconds": manifest["runtime"]["baseline_seconds"],
        "supplement_seconds": manifest["runtime"]["supplement_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
