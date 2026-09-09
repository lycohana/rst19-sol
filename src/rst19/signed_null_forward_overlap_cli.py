"""符号反相质量源与正向检测近邻交叉审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .signed_null_forward_overlap import (
    run_signed_null_forward_overlap,
    write_signed_null_forward_overlap_artifacts,
)


def _collect_paths(inputs: list[Path]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.fits"), key=lambda path: path.name))
            paths.extend(sorted(item.glob("*.FITS"), key=lambda path: path.name))
        else:
            paths.append(item)
    deduplicated: dict[str, Path] = {}
    for path in paths:
        deduplicated[str(path.resolve()).casefold()] = path
    return tuple(sorted(deduplicated.values(), key=lambda path: path.name))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="逐帧查找符号反相质量源在原图正向检测中的局部近邻"
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="FITS 文件或包含 FITS 的目录")
    parser.add_argument(
        "--reverse-source-csv",
        type=Path,
        required=True,
        help="signed_null_sequence_quality_sources.csv",
    )
    parser.add_argument("--threshold-sigma", type=float, default=4.0)
    parser.add_argument("--min-distance", type=int, default=4)
    parser.add_argument("--aperture-radius", type=int, default=4)
    parser.add_argument("--psf-fwhm", type=float, default=2.0)
    parser.add_argument("--background-box-size", type=int, default=128)
    parser.add_argument("--background-sample-limit", type=int, default=100_000)
    parser.add_argument("--min-flux-snr", type=float, default=5.0)
    parser.add_argument("--min-psf-support-pixels", type=int, default=3)
    parser.add_argument(
        "--proposal-mode",
        choices=("gaussian", "hybrid", "ensemble"),
        default="hybrid",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/signed-null-positive-overlap"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = _collect_paths(args.inputs)
    if not paths:
        print("rst19-signed-null-overlap: no FITS files found", file=sys.stderr)
        return 2

    def progress(index: int, total: int) -> None:
        print(f"forward overlap: frame {index}/{total}", file=sys.stderr, flush=True)

    try:
        result = run_signed_null_forward_overlap(
            paths,
            args.reverse_source_csv,
            threshold_sigma=args.threshold_sigma,
            min_distance=args.min_distance,
            aperture_radius=args.aperture_radius,
            psf_fwhm=args.psf_fwhm,
            background_box_size=args.background_box_size,
            background_sample_limit=args.background_sample_limit,
            min_flux_snr=args.min_flux_snr,
            min_psf_support_pixels=args.min_psf_support_pixels,
            proposal_mode=args.proposal_mode,
            progress=progress,
        )
        output = write_signed_null_forward_overlap_artifacts(result, args.out_dir)
        print(f"signed-null forward overlap: {output}")
        print(result.conclusion)
        print(f"overlap_classes={result.overlap_class_counts}")
        return 0
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-signed-null-overlap: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
