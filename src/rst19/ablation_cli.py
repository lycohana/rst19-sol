"""Command-line entry point for the shared-input ablation experiment."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import scipy

from .ablation import run_point_source_ablation, write_ablation_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在同一真实 FITS 背景、冻结注入真值和参数下运行点源提案/质量门消融"
    )
    parser.add_argument("fits_path", type=Path, help="真实背景 FITS；会记录文件 SHA-256")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp") / "ablation" / f"run-{time.strftime('%Y%m%d-%H%M%S')}",
        help="实验产物目录；非空目录不会被覆盖",
    )
    parser.add_argument("--report", type=Path, default=Path("deliverables/ablation/index.html"), help="中文 HTML 报告路径")
    parser.add_argument("--peak-levels", nargs="+", type=float, default=(12.0, 24.0, 56.0), help="每个目标的 Gaussian 峰值超额 ADU")
    parser.add_argument("--targets", type=int, default=8, help="一组固定注入位置数")
    parser.add_argument("--seed", type=int, default=19019, help="只用于一次性选址；实际坐标保存在 manifest")
    parser.add_argument("--psf-fwhm", type=float, default=2.0, help="检测 PSF FWHM，pixel")
    parser.add_argument("--injected-psf-fwhm", type=float, help="注入 PSF FWHM；默认等于 --psf-fwhm")
    parser.add_argument("--threshold-sigma", type=float, default=4.0, help="候选阈值 sigma")
    parser.add_argument("--min-distance", type=int, default=4, help="非极大值抑制最小间距，pixel")
    parser.add_argument("--aperture-radius", type=int, default=4, help="测光孔径半径，pixel")
    parser.add_argument("--background-box-size", type=int, default=128, help="局部背景网格间距，pixel")
    parser.add_argument("--min-flux-snr", type=float, default=5.0, help="质量门孔径通量 SNR 下限")
    parser.add_argument("--min-psf-support-pixels", type=int, default=3, help="3×3 核心最低支持像素数")
    parser.add_argument("--match-radius-px", type=float, help="请求匹配半径；代码实际下限为 3 px")
    parser.add_argument("--reuse-manifest", type=Path, help="复用既有 manifest 中冻结的注入坐标")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(message: str) -> None:
        print(f"ablation: {message}", file=sys.stderr, flush=True)

    try:
        manifest = run_point_source_ablation(
            args.fits_path,
            args.out_dir,
            peak_levels_adu=tuple(args.peak_levels),
            target_count=args.targets,
            seed=args.seed,
            psf_fwhm_px=args.psf_fwhm,
            injected_psf_fwhm_px=args.injected_psf_fwhm,
            threshold_sigma=args.threshold_sigma,
            min_distance_px=args.min_distance,
            aperture_radius_px=args.aperture_radius,
            background_box_size_px=args.background_box_size,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            match_radius_px=args.match_radius_px,
            fixed_targets_manifest_path=args.reuse_manifest,
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
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pillow": PIL.__version__,
            "source_revision": revision,
            "source_worktree_dirty": source_dirty,
        }
        manifest_path = Path(args.out_dir).resolve() / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report = write_ablation_html(args.out_dir, args.report)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"rst19-ablation: failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "status": manifest["status"],
        "run_dir": str(Path(args.out_dir).resolve()),
        "manifest": str(Path(args.out_dir).resolve() / "manifest.json"),
        "report": str(report),
        "total_experiment_seconds": manifest["runtime"]["total_experiment_seconds"],
        "target_count": args.targets,
        "signal_levels_adu": list(args.peak_levels),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
