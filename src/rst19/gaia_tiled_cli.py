"""Gaia DR3 大视场分块查询命令行入口。

这个入口只在用户明确运行命令时访问公共 Gaia TAP 服务。查询完成后同时
写出可供现有星表读取器使用的 CSV 和分块完整性审计 JSON；若远端结果
可能被截断、某块失败或达到查询上限，进程返回非零状态，避免把部分
星表误当成可用于光度定标的完整参考样本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .gaia_remote import DEFAULT_GAIA_TAP_SYNC_URL, GaiaError, write_catalog_csv
from .gaia_tiled import (
    GaiaTiledQueryError,
    GaiaTilingError,
    plan_gaia_tiles,
    query_gaia_tiled,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按天空坐标分块查询公共 Gaia DR3 星表并导出完整性审计结果"
    )
    parser.add_argument("--ra", type=float, required=True, help="查询中心赤经（度，0 <= ra <= 360）")
    parser.add_argument("--dec", type=float, required=True, help="查询中心赤纬（度，-90 <= dec <= 90）")
    parser.add_argument("--radius", type=float, required=True, help="目标查询半径（度，0 < radius <= 180）")
    parser.add_argument("--out", type=Path, help="输出 CatalogSource 兼容 CSV；非 dry-run 时必填")
    parser.add_argument("--tile-radius", type=float, default=1.0, help="每块圆锥半径，默认 1 度")
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="每块 Gaia TOP/MAXREC 上限，默认 5000；达到上限会触发细分并保持不完整状态",
    )
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="不主动设置每块行数上限；由于服务端隐含上限不可见，结果仍标记为不完整",
    )
    parser.add_argument(
        "--strategy",
        choices=("auto", "small_circle", "grid"),
        default="auto",
        help="查询规划策略，默认按目标半径自动选择",
    )
    parser.add_argument("--min-g-mag", type=float, help="排除过亮参考星的 Gaia G 星等下限")
    parser.add_argument("--max-g-mag", type=float, help="按图像极限限制 Gaia G 星等上限")
    parser.add_argument("--max-depth", type=int, default=2, help="饱和块的最大自适应细分深度，默认 2")
    parser.add_argument("--max-queries", type=int, default=10000, help="最大块查询数，默认 10000")
    parser.add_argument(
        "--saturated-policy",
        choices=("subdivide", "record", "raise"),
        default="subdivide",
        help="块达到上限时的策略，默认 subdivide",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="每块 HTTP 超时秒数，默认 30")
    parser.add_argument("--endpoint", default=DEFAULT_GAIA_TAP_SYNC_URL, help="Gaia TAP 同步服务地址")
    parser.add_argument("--dry-run", action="store_true", help="只规划分块并输出摘要，不联网、不写文件")
    return parser


def _write_result_artifacts(result: object, output_path: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    # Keep this helper duck-typed so partial results from GaiaTiledQueryError
    # can be written with exactly the same audit format as normal results.
    csv_path = write_catalog_csv(result.rows, output_path)  # type: ignore[attr-defined]
    metadata_path = csv_path.with_suffix(csv_path.suffix + ".tiled.meta.json")
    metadata = {
        "catalog": "Gaia DR3",
        "table": "gaiadr3.gaia_source",
        "query_center_ra_deg": args.ra,
        "query_center_dec_deg": args.dec,
        "query_radius_deg": args.radius,
        "tile_radius_deg": args.tile_radius,
        "tile_limit": None if args.no_limit else args.limit,
        "min_g_mag": args.min_g_mag,
        "max_g_mag": args.max_g_mag,
        "max_subdivide_depth": args.max_depth,
        "max_queries": args.max_queries,
        "saturated_policy": args.saturated_policy,
        "endpoint": args.endpoint,
        "row_count": len(result.rows),  # type: ignore[attr-defined]
        "complete": result.complete,  # type: ignore[attr-defined]
        "initial_tile_count": result.initial_tile_count,  # type: ignore[attr-defined]
        "queried_tile_count": result.queried_tile_count,  # type: ignore[attr-defined]
        "duplicate_count": result.duplicate_count,  # type: ignore[attr-defined]
        "outside_scope_count": result.outside_scope_count,  # type: ignore[attr-defined]
        "unresolved_tile_ids": list(result.unresolved_tile_ids),  # type: ignore[attr-defined]
        "truncated_tile_ids": list(result.truncated_tile_ids),  # type: ignore[attr-defined]
        "errors": list(result.errors),  # type: ignore[attr-defined]
        "tile_records": [record.as_dict() for record in result.tile_records],  # type: ignore[attr-defined]
        "magnitude_error_source": (
            "local first-order approximation from phot_g_mean_flux_error; "
            "Gaia does not publish a symmetric phot_g_mean_mag_error column"
        ),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, metadata_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.dry_run:
            tiles = plan_gaia_tiles(
                args.ra,
                args.dec,
                args.radius,
                tile_radius_deg=args.tile_radius,
                strategy=args.strategy,
            )
            print(
                json.dumps(
                    {
                        "strategy": args.strategy,
                        "tile_radius_deg": args.tile_radius,
                        "initial_tile_count": len(tiles),
                        "first_tile": tiles[0].as_dict() if tiles else None,
                        "last_tile": tiles[-1].as_dict() if tiles else None,
                        "network": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.out is None:
            raise GaiaError("--out is required unless --dry-run is used")
        result = query_gaia_tiled(
            args.ra,
            args.dec,
            args.radius,
            tile_radius_deg=args.tile_radius,
            tile_limit=None if args.no_limit else args.limit,
            strategy=args.strategy,
            min_g_mag=args.min_g_mag,
            max_g_mag=args.max_g_mag,
            timeout=args.timeout,
            endpoint=args.endpoint,
            saturated_policy=args.saturated_policy,
            max_subdivide_depth=args.max_depth,
            max_queries=args.max_queries,
            raise_on_error=False,
        )
        csv_path, metadata_path = _write_result_artifacts(result, args.out, args)
        if result.complete:
            print(
                f"Gaia DR3 tiled: wrote {len(result.rows)} rows to {csv_path} "
                f"({result.queried_tile_count} tiles; audit: {metadata_path})",
                file=sys.stderr,
            )
            return 0
        print(
            f"gaia-tiled: incomplete result wrote {len(result.rows)} rows to {csv_path}; "
            f"see audit {metadata_path} for unresolved/truncated blocks",
            file=sys.stderr,
        )
        return 3
    except GaiaTiledQueryError as exc:
        if args.out is not None:
            csv_path, metadata_path = _write_result_artifacts(exc.partial_result, args.out, args)
            print(
                f"gaia-tiled: query failed; partial result wrote {len(exc.partial_result.rows)} rows to "
                f"{csv_path}; audit: {metadata_path}; {exc}",
                file=sys.stderr,
            )
        else:
            print(f"gaia-tiled: {exc}", file=sys.stderr)
        return 2
    except (GaiaError, GaiaTilingError, OSError) as exc:
        print(f"gaia-tiled: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

