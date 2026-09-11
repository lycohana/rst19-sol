"""Gaia DR3 公共星表查询命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .gaia_remote import (
    DEFAULT_GAIA_TAP_SYNC_URL,
    GaiaError,
    build_gaia_adql,
    query_gaia,
    write_catalog_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按天空坐标查询公共 Gaia DR3 星表并导出 CSV")
    parser.add_argument("--ra", type=float, required=True, help="查询中心赤经（度，0 <= ra < 360）")
    parser.add_argument("--dec", type=float, required=True, help="查询中心赤纬（度，-90 <= dec <= 90）")
    parser.add_argument("--radius", type=float, required=True, help="查询半径（度，0 < radius <= 180）")
    parser.add_argument("--out", type=Path, help="输出 CatalogSource 兼容 CSV 路径；非 dry-run 时必填")
    parser.add_argument("--limit", type=int, help="可选的 Gaia TOP 行数限制（正整数）")
    parser.add_argument("--min-g-mag", type=float, help="可选 Gaia G 星等下限；用于排除过亮饱和参考星")
    parser.add_argument("--max-g-mag", type=float, help="可选 Gaia G 星等上限；建议按图像检测极限控制返回量")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP 超时秒数，默认 30")
    parser.add_argument("--dry-run", action="store_true", help="只输出 ADQL，不联网、不写文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        adql = build_gaia_adql(
            args.ra,
            args.dec,
            args.radius,
            limit=args.limit,
            min_g_mag=args.min_g_mag,
            max_g_mag=args.max_g_mag,
        )
        if args.dry_run:
            print(adql)
            return 0
        if args.out is None:
            raise GaiaError("--out is required unless --dry-run is used")
        if args.radius > 1.0:
            print(
                "gaia-remote: warning: broad cone queries may be truncated by the TAP service; "
                "verify row count and magnitude distribution or query smaller tiles",
                file=sys.stderr,
            )
        rows = query_gaia(
            args.ra,
            args.dec,
            args.radius,
            limit=args.limit,
            timeout=args.timeout,
            min_g_mag=args.min_g_mag,
            max_g_mag=args.max_g_mag,
        )
        output_path = write_catalog_csv(rows, args.out)
        metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        metadata_path.write_text(
            json.dumps(
                {
                    "catalog": "Gaia DR3",
                    "table": "gaiadr3.gaia_source",
                    "query_center_ra_deg": args.ra,
                    "query_center_dec_deg": args.dec,
                    "query_radius_deg": args.radius,
                    "limit": args.limit,
                    "min_g_mag": args.min_g_mag,
                    "max_g_mag": args.max_g_mag,
                    "row_count": len(rows),
                    "tap_maxrec": args.limit,
                    "endpoint": DEFAULT_GAIA_TAP_SYNC_URL,
                    "magnitude_error_source": (
                        "local first-order approximation from phot_g_mean_flux_error; "
                        "Gaia does not publish a symmetric phot_g_mean_mag_error column"
                    ),
                    "gspphot_fields": (
                        "distance_gspphot/ag_gspphot and their 16th/84th percentile bounds "
                        "are preserved when available; they are model-based single-star estimates"
                    ),
                    "coverage_warning": (
                        "broad cone result requires completeness audit"
                        if args.radius > 1.0
                        else None
                    ),
                    "query": adql,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"Gaia DR3: wrote {len(rows)} rows to {output_path} "
            f"(query metadata: {metadata_path})",
            file=sys.stderr,
        )
        return 0
    except (GaiaError, OSError) as exc:
        print(f"gaia-remote: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
