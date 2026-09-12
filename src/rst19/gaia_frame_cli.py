"""从单张 FITS 的辅助指向显式获取 Gaia DR3 参考子表。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .gaia_remote import DEFAULT_GAIA_TAP_SYNC_URL, GaiaError
from .gaia_tiled import GaiaTilingError
from .public_catalog import (
    DEFAULT_GAIA_MAX_G_MAG,
    DEFAULT_GAIA_MAX_QUERIES,
    DEFAULT_GAIA_MAX_SUBDIVIDE_DEPTH,
    DEFAULT_GAIA_MIN_G_MAG,
    DEFAULT_GAIA_TILE_LIMIT,
    DEFAULT_GAIA_TILE_RADIUS_DEG,
    download_public_gaia_catalog_for_frame,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 FITS 辅助 RA/DEC 显式获取 Gaia DR3 参考星表并保存完整性审计"
    )
    parser.add_argument("fits", type=Path, help="输入 FITS 文件")
    parser.add_argument("--out", type=Path, required=True, help="输出 Gaia 兼容 CSV 文件")
    parser.add_argument("--radius", type=float, help="查询半径（度）；默认覆盖 9.78° 正方形视场外接圆")
    parser.add_argument("--min-g", type=float, default=DEFAULT_GAIA_MIN_G_MAG, help="Gaia G 下限")
    parser.add_argument(
        "--max-g",
        type=float,
        default=DEFAULT_GAIA_MAX_G_MAG,
        help="Gaia G 上限；默认 13.5，是标定参考子表工作点，不代表全图完整星数",
    )
    parser.add_argument("--tile-radius", type=float, default=DEFAULT_GAIA_TILE_RADIUS_DEG, help="分块圆半径（度）")
    parser.add_argument("--tile-limit", type=int, default=DEFAULT_GAIA_TILE_LIMIT, help="每块 TOP/MAXREC 上限")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_GAIA_MAX_SUBDIVIDE_DEPTH, help="饱和块最大细分深度")
    parser.add_argument("--max-queries", type=int, default=DEFAULT_GAIA_MAX_QUERIES, help="最大块查询数")
    parser.add_argument(
        "--include-gspphot-model",
        action="store_true",
        help="额外连接 Gaia astrophysical_parameters，保留 GSP-Phot mg_gspphot 及 16/84%% 分位界",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="单块 HTTP 超时秒数")
    parser.add_argument("--endpoint", default=DEFAULT_GAIA_TAP_SYNC_URL, help="Gaia TAP sync 地址；可选 https://gaia.aip.de/tap/sync")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.fits.is_file():
        print(f"gaia-frame: FITS 不存在：{args.fits}", file=sys.stderr)
        return 2
    try:
        result = download_public_gaia_catalog_for_frame(
            args.fits,
            args.out,
            search_radius_deg=args.radius,
            min_g_mag=args.min_g,
            max_g_mag=args.max_g,
            tile_radius_deg=args.tile_radius,
            tile_limit=args.tile_limit,
            max_subdivide_depth=args.max_depth,
            max_queries=args.max_queries,
            include_gspphot_model=args.include_gspphot_model,
            timeout=args.timeout,
            endpoint=args.endpoint,
            progress=lambda message: print(message, file=sys.stderr),
        )
    except (GaiaError, GaiaTilingError, OSError, ValueError) as exc:
        print(f"gaia-frame: {exc}", file=sys.stderr)
        return 2
    state = "complete" if result.complete else "incomplete"
    print(
        f"Gaia DR3 {state}: {result.row_count:,} rows -> {result.output_path} "
        f"({result.queried_tile_count} tiles; audit: {result.audit_path})",
        file=sys.stderr,
    )
    return 0 if result.complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
