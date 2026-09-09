"""质量门路径 × 跨帧响应审计命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_gate_temporal_route import (
    run_feature_gate_temporal_route,
    write_feature_gate_temporal_route_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="连接质量门路径与已生成的逐源跨帧响应，不重新读取 FITS"
    )
    parser.add_argument("route_csv", type=Path, help="rst19-feature-gate-route 的逐候选 CSV")
    parser.add_argument("diagnostic_sources_csv", type=Path, help="rst19-feature-sequence 的诊断源 CSV")
    parser.add_argument("persistence_json", type=Path, help="同一次序列实验的 persistence JSON")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/feature-gate-temporal-route"))
    parser.add_argument(
        "--target-id",
        action="append",
        default=[],
        help="可重复指定候选 ID，输出其相对所属诊断子组的经验分布位置",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_feature_gate_temporal_route(
            args.route_csv,
            args.diagnostic_sources_csv,
            args.persistence_json,
            target_ids=tuple(args.target_id),
        )
        output = write_feature_gate_temporal_route_artifacts(result, args.out_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"rst19-feature-gate-temporal-route: {exc}", file=sys.stderr)
        return 2
    print(
        f"feature gate temporal route: {output} "
        f"frames={result.frame_count} required={result.required_presence} "
        f"radius={result.association_radius_px:g} subgroups={len(result.subgroup_summary_rows)}"
    )
    for row in result.summary_rows:
        candidate = "-" if row.candidate_presence_ge_required_count is None else str(
            row.candidate_presence_ge_required_count
        )
        quality = "-" if row.quality_presence_ge_required_count is None else str(
            row.quality_presence_ge_required_count
        )
        print(
            f"{row.feature_class}/{row.route}: n={row.route_count} "
            f"temporal={row.temporal_source_count} candidate>={result.required_presence}:{candidate} "
            f"quality>={result.required_presence}:{quality}"
        )
    for row in result.target_rows:
        candidate = "-" if row.candidate_presence is None else str(row.candidate_presence)
        same = "-" if row.candidate_same_subgroup_presence is None else str(
            row.candidate_same_subgroup_presence
        )
        quality = "-" if row.quality_presence is None else str(row.quality_presence)
        print(
            f"target {row.detection_id}/{row.diagnostic_subgroup or 'unavailable'}: "
            f"candidate={candidate} same={same} quality={quality} "
            f"all_three={row.all_three_ge_required}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
