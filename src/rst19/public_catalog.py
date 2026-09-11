"""把公开星表查询接成可审计的显式工作流。

这个模块是 GUI/CLI 的薄封装：只有调用
``download_public_gaia_catalog`` 时才会访问远程服务。它不在导入时联网，
也不把任何离线星表写进仓库。查询结果旁边始终保存一个 JSON 审计文件，
记录视场、星等窗口、分块数量和是否可能截断；``complete=False`` 的结果
可以用于人工检查，但不能被描述成完整参考星表。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fits import read_fits
from .gaia_tiled import GaiaTiledResult, query_gaia_tiled
from .gaia_remote import DEFAULT_GAIA_TAP_SYNC_URL, write_catalog_csv
from .models import FitsFrame


# These are the externally supplied nominal camera values.  They are used only
# to plan a public-catalog search footprint; they are not silently promoted to
# a validated WCS or a measured photometric passband.
DEFAULT_CAMERA_FOV_WIDTH_DEG = 9.78
DEFAULT_CAMERA_FOV_HEIGHT_DEG = 9.78
DEFAULT_GAIA_MIN_G_MAG = 5.0
DEFAULT_GAIA_MAX_G_MAG = 13.5
DEFAULT_GAIA_TILE_RADIUS_DEG = 1.5
DEFAULT_GAIA_TILE_LIMIT = 5_000
DEFAULT_GAIA_MAX_SUBDIVIDE_DEPTH = 2
DEFAULT_GAIA_MAX_QUERIES = 2_000


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_footprint_radius_deg(
    width_deg: object = DEFAULT_CAMERA_FOV_WIDTH_DEG,
    height_deg: object = DEFAULT_CAMERA_FOV_HEIGHT_DEG,
) -> float:
    """Return the spherical-query radius covering a rectangular camera field.

    The half diagonal is deliberately a conservative *catalog-query* radius.
    It is not used as an image-plane WCS scale and does not claim that the
    corners have identical spherical projection.
    """

    width = _finite_positive(width_deg, name="width_deg")
    height = _finite_positive(height_deg, name="height_deg")
    return 0.5 * math.hypot(width, height)


@dataclass(frozen=True, slots=True)
class PublicCatalogDownload:
    """公开星表下载结果以及可复核的查询边界。"""

    output_path: Path
    audit_path: Path
    frame_path: Path | None
    center_ra_deg: float
    center_dec_deg: float
    search_radius_deg: float
    min_g_mag: float | None
    max_g_mag: float | None
    row_count: int
    complete: bool
    initial_tile_count: int
    queried_tile_count: int
    duplicate_count: int
    outside_scope_count: int
    tiled_result: GaiaTiledResult
    include_gspphot_model: bool = False

    def as_dict(self) -> dict[str, object]:
        # The CSV is the row-level artifact.  Keep the sidecar audit compact;
        # embedding every returned row here would duplicate a large catalog
        # and make a wide-field query unnecessarily expensive to inspect.
        tiled_audit = self.tiled_result.as_dict()
        tiled_audit.pop("rows", None)
        return {
            "catalog": "Gaia DR3",
            "table": (
                "gaiadr3.gaia_source LEFT OUTER JOIN gaiadr3.astrophysical_parameters"
                if self.include_gspphot_model
                else "gaiadr3.gaia_source"
            ),
            "frame_path": str(self.frame_path) if self.frame_path is not None else None,
            "output_path": str(self.output_path),
            "audit_path": str(self.audit_path),
            "center_ra_deg": self.center_ra_deg,
            "center_dec_deg": self.center_dec_deg,
            "search_radius_deg": self.search_radius_deg,
            "min_g_mag": self.min_g_mag,
            "max_g_mag": self.max_g_mag,
            "row_count": self.row_count,
            "csv_row_count": self.row_count,
            "csv_sha256": _sha256(self.output_path),
            "csv_byte_count": self.output_path.stat().st_size,
            "complete": self.complete,
            "initial_tile_count": self.initial_tile_count,
            "queried_tile_count": self.queried_tile_count,
            "duplicate_count": self.duplicate_count,
            "outside_scope_count": self.outside_scope_count,
            "tiled": tiled_audit,
            "provenance": {
                "source": "Gaia DR3 gaiadr3.gaia_source",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
                "magnitude_error": (
                    "local first-order approximation from phot_g_mean_flux_error; "
                    "Gaia does not publish a symmetric phot_g_mean_mag_error column"
                ),
                "gspphot_fields": (
                    "distance_gspphot and ag_gspphot, with 16th/84th percentile bounds; "
                    "these are Gaia DR3 GSP-Phot model estimates under a single-star assumption; "
                    + (
                        "mg_gspphot and its 16th/84th percentile bounds are included"
                        if self.include_gspphot_model
                        else "mg_gspphot is not requested"
                    )
                ),
            },
            "interpretation": (
                "This is an external reference catalog for identity and apparent-magnitude "
                "calibration. It is not the camera-specific 450-750 nm response or an "
                "absolute photometric zero point."
            ),
        }


def _write_audit(result: PublicCatalogDownload) -> None:
    result.audit_path.parent.mkdir(parents=True, exist_ok=True)
    result.audit_path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def download_public_gaia_catalog(
    center_ra_deg: object,
    center_dec_deg: object,
    output_path: str | Path,
    *,
    search_radius_deg: object | None = None,
    fov_width_deg: object = DEFAULT_CAMERA_FOV_WIDTH_DEG,
    fov_height_deg: object = DEFAULT_CAMERA_FOV_HEIGHT_DEG,
    min_g_mag: object | None = DEFAULT_GAIA_MIN_G_MAG,
    max_g_mag: object | None = DEFAULT_GAIA_MAX_G_MAG,
    tile_radius_deg: object = DEFAULT_GAIA_TILE_RADIUS_DEG,
    tile_limit: object | None = DEFAULT_GAIA_TILE_LIMIT,
    max_subdivide_depth: object = DEFAULT_GAIA_MAX_SUBDIVIDE_DEPTH,
    max_queries: object = DEFAULT_GAIA_MAX_QUERIES,
    include_gspphot_model: bool = False,
    timeout: object = 30.0,
    endpoint: str = DEFAULT_GAIA_TAP_SYNC_URL,
    frame_path: str | Path | None = None,
    progress: Callable[[str], object] | None = None,
    query_fn: Callable[..., Any] | None = None,
) -> PublicCatalogDownload:
    """显式查询一个相机视场的 Gaia DR3 参考子表并写出审计文件。

    ``max_g_mag`` 默认设为 13.5，是面向当前数据的标定参考子表工作点，
    不是“全图所有星”的隐含上限。需要更深目录时可显式提高它，但应同
    时关注分块完整性和服务端负载。
    """

    ra = float(center_ra_deg)
    dec = float(center_dec_deg)
    if not math.isfinite(ra) or not 0.0 <= ra < 360.0:
        raise ValueError("center_ra_deg must be finite and in [0, 360)")
    if not math.isfinite(dec) or not -90.0 <= dec <= 90.0:
        raise ValueError("center_dec_deg must be finite and in [-90, 90]")
    radius = (
        camera_footprint_radius_deg(fov_width_deg, fov_height_deg)
        if search_radius_deg is None
        else _finite_positive(search_radius_deg, name="search_radius_deg")
    )
    if radius > 180.0:
        raise ValueError("search_radius_deg must be at most 180 degrees")
    output = Path(output_path)
    if output.suffix.lower() != ".csv":
        raise ValueError("output_path must use the .csv suffix")
    frame = None if frame_path is None else Path(frame_path).resolve()

    def on_tile(record: object) -> None:
        if progress is None:
            return
        sequence = getattr(record, "sequence", 0) + 1
        returned = getattr(record, "returned_rows", 0)
        new_rows = getattr(record, "new_rows", 0)
        status = getattr(record, "status", "unknown")
        progress(f"Gaia DR3 · 已查询第 {sequence} 块 · 返回 {returned:,} 行 · 新增 {new_rows:,} · {status}")

    result = query_gaia_tiled(
        ra,
        dec,
        radius,
        tile_radius_deg=tile_radius_deg,
        tile_limit=tile_limit,
        min_g_mag=min_g_mag,
        max_g_mag=max_g_mag,
        max_subdivide_depth=max_subdivide_depth,
        max_queries=max_queries,
        include_gspphot_model=include_gspphot_model,
        timeout=timeout,
        endpoint=endpoint,
        raise_on_error=False,
        on_tile=on_tile,
        query_fn=query_fn,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_catalog_csv(result.rows, output)
    download = PublicCatalogDownload(
        output_path=output.resolve(),
        audit_path=output.with_suffix(output.suffix + ".meta.json").resolve(),
        frame_path=frame,
        center_ra_deg=ra,
        center_dec_deg=dec,
        search_radius_deg=radius,
        min_g_mag=None if min_g_mag is None else float(min_g_mag),
        max_g_mag=None if max_g_mag is None else float(max_g_mag),
        row_count=len(result.rows),
        complete=result.complete,
        initial_tile_count=result.initial_tile_count,
        queried_tile_count=result.queried_tile_count,
        duplicate_count=result.duplicate_count,
        outside_scope_count=result.outside_scope_count,
        tiled_result=result,
        include_gspphot_model=include_gspphot_model,
    )
    _write_audit(download)
    return download


def download_public_gaia_catalog_for_frame(
    frame: str | Path | FitsFrame,
    output_path: str | Path,
    **kwargs: object,
) -> PublicCatalogDownload:
    """从 FITS 辅助区的 RA/DEC 取中心后执行显式 Gaia 查询。"""

    loaded = frame if isinstance(frame, FitsFrame) else read_fits(frame)
    if loaded.auxiliary is None:
        raise ValueError("FITS 没有可用的辅助 RA/DEC，无法规划公共星表查询")
    return download_public_gaia_catalog(
        loaded.auxiliary.ra_deg,
        loaded.auxiliary.dec_deg,
        output_path,
        frame_path=loaded.path,
        **kwargs,
    )


__all__ = [
    "DEFAULT_CAMERA_FOV_HEIGHT_DEG",
    "DEFAULT_CAMERA_FOV_WIDTH_DEG",
    "DEFAULT_GAIA_MAX_G_MAG",
    "DEFAULT_GAIA_MAX_QUERIES",
    "DEFAULT_GAIA_MAX_SUBDIVIDE_DEPTH",
    "DEFAULT_GAIA_MIN_G_MAG",
    "DEFAULT_GAIA_TILE_LIMIT",
    "DEFAULT_GAIA_TILE_RADIUS_DEG",
    "PublicCatalogDownload",
    "camera_footprint_radius_deg",
    "download_public_gaia_catalog",
    "download_public_gaia_catalog_for_frame",
]
