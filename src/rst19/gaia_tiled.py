"""可靠的 Gaia 大视场分块查询。

单个数度级 Gaia cone query 同时有两个容易被忽略的风险：服务端可能按
``TOP``/``MAXREC`` 截断返回，而且大圆锥的返回顺序并不是一个可以用来
代表完整样本的科学抽样。这个模块把大视场拆成有重叠的小圆查询，并在
本地用球面距离做最终范围裁剪；跨块重复的 Gaia ``source_id`` 只保留一
行。

本模块刻意只依赖现有的 :func:`rst19.gaia_remote.query_gaia` 和
:func:`rst19.gaia_remote.build_gaia_adql`。导入、规划和 ADQL 构造都不会
访问网络；只有显式调用 :func:`query_gaia_tiled` 才会执行同步查询。查询
函数可以通过 ``query_fn`` 注入，因此测试和离线演示无需触碰 Gaia 服务。

“返回行数等于 tile limit”只能证明结果可能被截断，不能证明它一定被截断。
默认策略是继续细分该块；达到细分深度仍然饱和时，结果会明确标记为
``complete=False``，不会把偏亮或不完整样本包装成完整星表。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any, Literal

from .gaia_remote import (
    DEFAULT_GAIA_TAP_SYNC_URL,
    build_gaia_adql,
    query_gaia,
)


_ANGLE_EPSILON_DEG = 1.0e-9
# Adjacent grid centers are no farther apart than 0.8 tile radii in either
# local spherical coordinate.  This leaves useful overlap while avoiding the
# roughly 4x query count of an unnecessarily dense half-radius grid.
_GRID_STEP_FACTOR = 0.8

TileStrategy = Literal["auto", "small_circle", "grid"]
SaturatedPolicy = Literal["subdivide", "record", "raise"]


class GaiaTilingError(ValueError):
    """分块规划参数或返回数据不满足契约。"""


class GaiaTiledQueryError(RuntimeError):
    """一个分块查询失败，或无法证明分块结果没有截断。"""

    def __init__(self, message: str, *, partial_result: "GaiaTiledResult") -> None:
        self.partial_result = partial_result
        super().__init__(message)


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise GaiaTilingError(f"{name} must be a finite number, not a boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GaiaTilingError(f"{name} must be a finite number; got {value!r}") from exc
    if not math.isfinite(parsed):
        raise GaiaTilingError(f"{name} must be a finite number; got {value!r}")
    return parsed


def _coordinate_ra(value: object, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if not 0.0 <= parsed <= 360.0:
        raise GaiaTilingError(f"{name} must be in [0, 360] degrees; got {parsed}")
    # 360 degrees is the same meridian as zero.  Accepting it here is useful
    # for hand-written catalog fixtures, while generated Gaia queries use [0,360).
    return parsed % 360.0


def _coordinate_dec(value: object, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if not -90.0 <= parsed <= 90.0:
        raise GaiaTilingError(f"{name} must be in [-90, 90] degrees; got {parsed}")
    return parsed


def _radius(value: object, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if not 0.0 < parsed <= 180.0:
        raise GaiaTilingError(f"{name} must be greater than 0 and at most 180 degrees; got {parsed}")
    return parsed


def _positive_int(
    value: object | None,
    *,
    name: str,
    allow_none: bool = False,
    allow_zero: bool = False,
) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise GaiaTilingError(f"{name} must be a positive integer, not a boolean")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise GaiaTilingError(f"{name} must be a positive integer; got {value!r}")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise GaiaTilingError(f"{name} must be a positive integer; got {value!r}") from exc
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        description = "non-negative integer" if allow_zero else "positive integer"
        raise GaiaTilingError(f"{name} must be a {description}; got {value!r}")
    return parsed


def _normalise_strategy(value: object) -> TileStrategy:
    if not isinstance(value, str):
        raise GaiaTilingError(f"strategy must be auto, small_circle, or grid; got {value!r}")
    normalised = value.strip().lower().replace("-", "_")
    if normalised not in {"auto", "small_circle", "grid"}:
        raise GaiaTilingError(f"strategy must be auto, small_circle, or grid; got {value!r}")
    return normalised  # type: ignore[return-value]


def _normalise_saturated_policy(value: object) -> SaturatedPolicy:
    if not isinstance(value, str):
        raise GaiaTilingError(f"saturated_policy must be subdivide, record, or raise; got {value!r}")
    normalised = value.strip().lower().replace("-", "_")
    if normalised not in {"subdivide", "record", "raise"}:
        raise GaiaTilingError(f"saturated_policy must be subdivide, record, or raise; got {value!r}")
    return normalised  # type: ignore[return-value]


def _resolve_tile_radius(
    primary: object | None,
    alias: object | None,
    *,
    default: object | None = None,
) -> float:
    """Resolve the explicit ``*_deg`` name and its short public alias."""

    if primary is not None and alias is not None:
        raise GaiaTilingError("provide only one of tile_radius_deg and tile_radius")
    value = alias if alias is not None else primary
    if value is None:
        if default is None:
            raise GaiaTilingError("tile_radius_deg is required")
        value = default
    return _radius(value, name="tile_radius_deg")


@dataclass(frozen=True, slots=True)
class GaiaTile:
    """一个 Gaia cone query 的球面中心和半径。"""

    tile_id: str
    ra_deg: float
    dec_deg: float
    radius_deg: float
    strategy: str = "grid"
    depth: int = 0
    parent_id: str | None = None

    def __post_init__(self) -> None:
        if not self.tile_id:
            raise GaiaTilingError("tile_id cannot be empty")
        # Keep manually-created tiles valid for the existing ADQL builder as
        # well.  In particular, RA=360 is a useful input convention but must
        # be normalized to zero before it reaches build_gaia_adql.
        object.__setattr__(self, "ra_deg", _coordinate_ra(self.ra_deg, name="tile.ra_deg"))
        object.__setattr__(self, "dec_deg", _coordinate_dec(self.dec_deg, name="tile.dec_deg"))
        object.__setattr__(self, "radius_deg", _radius(self.radius_deg, name="tile.radius_deg"))
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise GaiaTilingError("tile.depth must be a non-negative integer")

    def as_dict(self) -> dict[str, object]:
        return {
            "tile_id": self.tile_id,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "radius_deg": self.radius_deg,
            "strategy": self.strategy,
            "depth": self.depth,
            "parent_id": self.parent_id,
        }


@dataclass(frozen=True, slots=True)
class TileQueryRecord:
    """一块查询的可审计记录。"""

    sequence: int
    tile: GaiaTile
    requested_limit: int | None
    returned_rows: int
    in_scope_rows: int
    new_rows: int
    duplicate_rows: int
    outside_scope_rows: int
    truncation_status: str
    status: str
    adql: str
    error: str | None = None

    @property
    def possibly_truncated(self) -> bool:
        return self.truncation_status in {"saturated", "unknown_limit"}

    @property
    def tile_id(self) -> str:
        return self.tile.tile_id

    def as_dict(self) -> dict[str, object]:
        payload = self.tile.as_dict()
        payload.update(
            {
                "sequence": self.sequence,
                "requested_limit": self.requested_limit,
                "returned_rows": self.returned_rows,
                "in_scope_rows": self.in_scope_rows,
                "new_rows": self.new_rows,
                "duplicate_rows": self.duplicate_rows,
                "outside_scope_rows": self.outside_scope_rows,
                "truncation_status": self.truncation_status,
                "possibly_truncated": self.possibly_truncated,
                "status": self.status,
                "adql": self.adql,
                "error": self.error,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class GaiaTiledResult:
    """分块查询的合并结果和完整性证据。"""

    rows: tuple[dict[str, object], ...]
    tile_records: tuple[TileQueryRecord, ...]
    center_ra_deg: float
    center_dec_deg: float
    search_radius_deg: float
    strategy: str
    tile_radius_deg: float
    tile_limit: int | None
    complete: bool
    duplicate_count: int
    outside_scope_count: int
    unresolved_tile_ids: tuple[str, ...]
    errors: tuple[str, ...]
    initial_tile_count: int

    @property
    def truncated_tile_ids(self) -> tuple[str, ...]:
        return tuple(
            record.tile_id
            for record in self.tile_records
            if record.possibly_truncated
        )

    @property
    def queried_tile_count(self) -> int:
        return len(self.tile_records)

    @property
    def records(self) -> tuple[TileQueryRecord, ...]:
        """``tile_records`` 的简短别名。"""

        return self.tile_records

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": [dict(row) for row in self.rows],
            "tile_records": [record.as_dict() for record in self.tile_records],
            "center_ra_deg": self.center_ra_deg,
            "center_dec_deg": self.center_dec_deg,
            "search_radius_deg": self.search_radius_deg,
            "strategy": self.strategy,
            "tile_radius_deg": self.tile_radius_deg,
            "tile_limit": self.tile_limit,
            "complete": self.complete,
            "duplicate_count": self.duplicate_count,
            "outside_scope_count": self.outside_scope_count,
            "unresolved_tile_ids": list(self.unresolved_tile_ids),
            "truncated_tile_ids": list(self.truncated_tile_ids),
            "errors": list(self.errors),
            "initial_tile_count": self.initial_tile_count,
            "queried_tile_count": self.queried_tile_count,
        }


def angular_distance_deg(
    ra1_deg: object,
    dec1_deg: object,
    ra2_deg: object,
    dec2_deg: object,
) -> float:
    """返回两点在天球上的大圆距离，单位为度。

    实现使用 haversine 形式，能正确处理 RA 的 0/360° 接缝，也比直接
    用平面 x/y 距离更适合大视场边界裁剪。
    """

    ra1 = math.radians(_coordinate_ra(ra1_deg, name="ra1_deg"))
    ra2 = math.radians(_coordinate_ra(ra2_deg, name="ra2_deg"))
    dec1 = math.radians(_coordinate_dec(dec1_deg, name="dec1_deg"))
    dec2 = math.radians(_coordinate_dec(dec2_deg, name="dec2_deg"))
    delta_dec = dec2 - dec1
    delta_ra = (ra2 - ra1 + math.pi) % (2.0 * math.pi) - math.pi
    haversine = math.sin(delta_dec / 2.0) ** 2 + math.cos(dec1) * math.cos(dec2) * math.sin(delta_ra / 2.0) ** 2
    haversine = min(1.0, max(0.0, haversine))
    return math.degrees(2.0 * math.atan2(math.sqrt(haversine), math.sqrt(1.0 - haversine)))


def _row_value(row: Mapping[str, object], names: tuple[str, ...]) -> object | None:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        if name in row:
            return row[name]
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return None


def _row_coordinates(row: Mapping[str, object], *, row_number: int) -> tuple[float, float]:
    raw_ra = _row_value(row, ("ra_deg", "ra"))
    raw_dec = _row_value(row, ("dec_deg", "dec"))
    if raw_ra is None or str(raw_ra).strip() == "":
        raise GaiaTilingError(f"row {row_number}: missing ra_deg/ra")
    if raw_dec is None or str(raw_dec).strip() == "":
        raise GaiaTilingError(f"row {row_number}: missing dec_deg/dec")
    return (
        _coordinate_ra(raw_ra, name=f"row {row_number} ra"),
        _coordinate_dec(raw_dec, name=f"row {row_number} dec"),
    )


def _row_source_id(row: Mapping[str, object], *, row_number: int) -> str:
    value = _row_value(row, ("source_id",))
    source_id = "" if value is None else str(value).strip()
    if not source_id:
        raise GaiaTilingError(f"row {row_number}: missing source_id")
    return source_id


def filter_rows_to_spherical_cap(
    rows: Iterable[Mapping[str, object]],
    center_ra_deg: object,
    center_dec_deg: object,
    radius_deg: object,
) -> tuple[dict[str, object], ...]:
    """只保留目标球面小圆内的行，不使用经度矩形近似。"""

    center_ra = _coordinate_ra(center_ra_deg, name="center_ra_deg")
    center_dec = _coordinate_dec(center_dec_deg, name="center_dec_deg")
    radius = _radius(radius_deg, name="radius_deg")
    selected: list[dict[str, object]] = []
    for row_number, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise GaiaTilingError(f"row {row_number}: expected a mapping")
        row = dict(raw_row)
        row_ra, row_dec = _row_coordinates(row, row_number=row_number)
        if angular_distance_deg(center_ra, center_dec, row_ra, row_dec) <= radius + _ANGLE_EPSILON_DEG:
            selected.append(row)
    return tuple(selected)


def deduplicate_source_rows(
    rows: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """按 ``source_id`` 去重并保留首次出现的整行。"""

    unique: dict[str, dict[str, object]] = {}
    for row_number, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise GaiaTilingError(f"row {row_number}: expected a mapping")
        row = dict(raw_row)
        source_id = _row_source_id(row, row_number=row_number)
        unique.setdefault(source_id, row)
    return tuple(unique.values())


def _grid_values(low: float, high: float, step: float) -> list[float]:
    span = max(0.0, high - low)
    count = max(1, int(math.ceil(span / step)))
    actual_step = span / count
    return [low + actual_step * index for index in range(count + 1)]


def _longitude_half_width(
    center_dec_rad: float,
    row_dec_rad: float,
    cap_radius_rad: float,
) -> float | None:
    """给定纬度上与球面小圆相交的 RA 半宽；None 表示没有交集。"""

    if cap_radius_rad >= math.pi - 1.0e-12:
        return math.pi
    denominator = math.cos(center_dec_rad) * math.cos(row_dec_rad)
    cosine_limit = math.cos(cap_radius_rad)
    if abs(denominator) <= 1.0e-14:
        base = math.sin(center_dec_rad) * math.sin(row_dec_rad)
        if base > cosine_limit + 1.0e-12:
            return None
        return math.pi
    needed_cosine = (cosine_limit - math.sin(center_dec_rad) * math.sin(row_dec_rad)) / denominator
    if needed_cosine > 1.0 + 1.0e-12:
        return None
    if needed_cosine < -1.0 - 1.0e-12:
        return math.pi
    return math.acos(min(1.0, max(-1.0, needed_cosine)))


def _longitude_step_rad(row_dec_rad: float, tile_radius_rad: float) -> float:
    # Keeping the coordinate step below one tile radius gives a diagonal gap
    # below one tile radius in the tangent plane, leaving overlap even after
    # the final spherical filter.
    target = max(tile_radius_rad * _GRID_STEP_FACTOR, math.radians(1.0e-6))
    cos_lat = abs(math.cos(row_dec_rad))
    if cos_lat <= 1.0e-14:
        return 2.0 * math.pi
    return min(2.0 * math.pi, target / cos_lat)


def plan_small_circle_tiles(
    center_ra_deg: object,
    center_dec_deg: object,
    search_radius_deg: object,
    *,
    tile_radius_deg: object | None = None,
    tile_radius: object | None = None,
) -> tuple[GaiaTile, ...]:
    """为一个不超过 tile 半径的视场生成单个精确小圆查询。"""

    center_ra = _coordinate_ra(center_ra_deg, name="center_ra_deg")
    center_dec = _coordinate_dec(center_dec_deg, name="center_dec_deg")
    search_radius = _radius(search_radius_deg, name="search_radius_deg")
    control_radius = (
        search_radius
        if tile_radius_deg is None and tile_radius is None
        else _resolve_tile_radius(tile_radius_deg, tile_radius)
    )
    if search_radius > control_radius + _ANGLE_EPSILON_DEG:
        raise GaiaTilingError(
            "small_circle strategy requires search_radius_deg <= tile_radius_deg; use grid for a larger field"
        )
    return (
        GaiaTile(
            tile_id="tile-00000",
            ra_deg=center_ra,
            dec_deg=center_dec,
            radius_deg=search_radius,
            strategy="small_circle",
        ),
    )


def plan_grid_tiles(
    center_ra_deg: object,
    center_dec_deg: object,
    search_radius_deg: object,
    tile_radius_deg: object | None = None,
    *,
    tile_radius: object | None = None,
) -> tuple[GaiaTile, ...]:
    """生成覆盖目标球面小圆的重叠 RA/DEC 网格。

    网格不是在图像平面上画矩形，而是在纬度方向用固定角距、在每一条
    纬线上按 ``cos(dec)`` 自适应 RA 间距。候选中心扩展到
    ``search_radius + tile_radius``，因此目标小圆边缘也被 tile 圆覆盖。
    最终是否落在目标小圆内仍由 :func:`filter_rows_to_spherical_cap` 决定。
    """

    center_ra = _coordinate_ra(center_ra_deg, name="center_ra_deg")
    center_dec = _coordinate_dec(center_dec_deg, name="center_dec_deg")
    search_radius = _radius(search_radius_deg, name="search_radius_deg")
    tile_radius_value = _resolve_tile_radius(tile_radius_deg, tile_radius)
    if search_radius <= tile_radius_value + _ANGLE_EPSILON_DEG:
        return (
            GaiaTile(
                tile_id="tile-00000",
                ra_deg=center_ra,
                dec_deg=center_dec,
                radius_deg=search_radius,
                strategy="grid",
            ),
        )

    extended_radius = min(180.0, search_radius + tile_radius_value)
    extended_radius_rad = math.radians(extended_radius)
    center_dec_rad = math.radians(center_dec)
    lat_low = max(-90.0, center_dec - extended_radius)
    lat_high = min(90.0, center_dec + extended_radius)
    lat_step = max(math.radians(tile_radius_value * _GRID_STEP_FACTOR), math.radians(1.0e-6))
    candidate_centers: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()

    for row_dec in _grid_values(lat_low, lat_high, math.degrees(lat_step)):
        row_dec_rad = math.radians(row_dec)
        half_width = _longitude_half_width(center_dec_rad, row_dec_rad, extended_radius_rad)
        if half_width is None:
            continue
        longitude_step = _longitude_step_rad(row_dec_rad, math.radians(tile_radius_value))
        if half_width >= math.pi - 1.0e-12:
            count = max(1, int(math.ceil((2.0 * math.pi) / longitude_step)))
            ras = [center_ra + math.degrees(2.0 * math.pi * index / count) for index in range(count)]
        else:
            width = 2.0 * half_width
            count = max(1, int(math.ceil(width / longitude_step)))
            low_ra = math.radians(center_ra) - half_width
            ras = [math.degrees(low_ra + width * index / count) for index in range(count + 1)]

        for raw_ra in ras:
            ra = raw_ra % 360.0
            # At an exact pole all RA values denote the same point.  Keeping
            # one representative avoids a large set of geometrically equal
            # tiles without changing the spherical coverage.
            if abs(abs(row_dec) - 90.0) <= 1.0e-10:
                ra = center_ra
            key = (round(ra * 1.0e9), round(row_dec * 1.0e9))
            if key in seen:
                continue
            if angular_distance_deg(center_ra, center_dec, ra, row_dec) <= extended_radius + _ANGLE_EPSILON_DEG:
                seen.add(key)
                candidate_centers.append((ra, row_dec))

    if not candidate_centers:
        # This should only be reachable for a numerically degenerate cap, but
        # failing loudly is safer than silently returning an uncovered field.
        raise GaiaTilingError("grid planner generated no tile centers")

    candidate_centers.sort(
        key=lambda item: (
            angular_distance_deg(center_ra, center_dec, item[0], item[1]),
            item[1],
            item[0],
        )
    )
    return tuple(
        GaiaTile(
            tile_id=f"tile-{index:05d}",
            ra_deg=ra,
            dec_deg=dec,
            radius_deg=tile_radius_value,
            strategy="grid",
        )
        for index, (ra, dec) in enumerate(candidate_centers)
    )


def plan_gaia_tiles(
    center_ra_deg: object,
    center_dec_deg: object,
    search_radius_deg: object,
    *,
    tile_radius_deg: object | None = None,
    strategy: object = "auto",
    tile_radius: object | None = None,
) -> tuple[GaiaTile, ...]:
    """统一入口：小视场用单小圆，大视场用球面网格。"""

    normalised = _normalise_strategy(strategy)
    search_radius = _radius(search_radius_deg, name="search_radius_deg")
    tile_radius_value = _resolve_tile_radius(tile_radius_deg, tile_radius, default=1.0)
    if normalised == "small_circle" or (normalised == "auto" and search_radius <= tile_radius_value):
        return plan_small_circle_tiles(
            center_ra_deg,
            center_dec_deg,
            search_radius,
            tile_radius_deg=tile_radius_value,
        )
    return plan_grid_tiles(center_ra_deg, center_dec_deg, search_radius, tile_radius_value)


# Short aliases keep the pure planning API convenient without duplicating logic.
plan_small_circle = plan_small_circle_tiles
plan_grid = plan_grid_tiles


def build_tile_adql(
    tile: GaiaTile,
    *,
    tile_limit: int | None,
    min_g_mag: object | None = None,
    max_g_mag: object | None = None,
) -> str:
    """用现有 ``build_gaia_adql`` 构造一块的可审计 ADQL。"""

    return build_gaia_adql(
        tile.ra_deg,
        tile.dec_deg,
        tile.radius_deg,
        limit=tile_limit,
        min_g_mag=min_g_mag,
        max_g_mag=max_g_mag,
    )


def _materialise_query_rows(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, (str, bytes, bytearray)):
        raise GaiaTilingError("query_fn must return an iterable of row mappings")
    try:
        rows = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise GaiaTilingError("query_fn must return an iterable of row mappings") from exc
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise GaiaTilingError(f"query result row {row_number}: expected a mapping")
    return rows  # type: ignore[return-value]


def _subdivide_tile(tile: GaiaTile) -> tuple[GaiaTile, ...]:
    # Re-use the same spherical coverage construction for adaptive retries.
    # A child grid with half the parent radius covers the complete parent cap;
    # its extra margin is harmless because the final result is filtered to the
    # original requested cap.
    child_radius = tile.radius_deg * 0.5
    if child_radius <= 1.0e-6:
        return ()
    planned = plan_grid_tiles(tile.ra_deg, tile.dec_deg, tile.radius_deg, child_radius)
    return tuple(
        GaiaTile(
            tile_id=f"{tile.tile_id}.{index:04d}",
            ra_deg=child.ra_deg,
            dec_deg=child.dec_deg,
            radius_deg=child.radius_deg,
            strategy="adaptive_grid",
            depth=tile.depth + 1,
            parent_id=tile.tile_id,
        )
        for index, child in enumerate(planned)
    )


def _source_sort_key(source_id: str) -> tuple[int, object, str]:
    try:
        return (0, int(source_id), source_id)
    except ValueError:
        return (1, source_id, source_id)


def _assemble_result(
    *,
    rows_by_source_id: Mapping[str, dict[str, object]],
    records: Iterable[TileQueryRecord],
    center_ra: float,
    center_dec: float,
    search_radius: float,
    strategy: str,
    tile_radius: float,
    tile_limit: int | None,
    unresolved: Iterable[str],
    errors: Iterable[str],
    duplicate_count: int,
    outside_scope_count: int,
    initial_tile_count: int,
) -> GaiaTiledResult:
    record_tuple = tuple(records)
    unresolved_tuple = tuple(dict.fromkeys(unresolved))
    error_tuple = tuple(errors)
    rows = tuple(rows_by_source_id[source_id] for source_id in sorted(rows_by_source_id, key=_source_sort_key))
    return GaiaTiledResult(
        rows=rows,
        tile_records=record_tuple,
        center_ra_deg=center_ra,
        center_dec_deg=center_dec,
        search_radius_deg=search_radius,
        strategy=strategy,
        tile_radius_deg=tile_radius,
        tile_limit=tile_limit,
        complete=not unresolved_tuple and not error_tuple,
        duplicate_count=duplicate_count,
        outside_scope_count=outside_scope_count,
        unresolved_tile_ids=unresolved_tuple,
        errors=error_tuple,
        initial_tile_count=initial_tile_count,
    )


def query_gaia_tiled(
    center_ra_deg: object,
    center_dec_deg: object,
    search_radius_deg: object,
    *,
    tile_radius_deg: object | None = None,
    tile_limit: object | None = 5000,
    strategy: object = "auto",
    tile_radius: object | None = None,
    min_g_mag: object | None = None,
    max_g_mag: object | None = None,
    timeout: object = 30.0,
    endpoint: str = DEFAULT_GAIA_TAP_SYNC_URL,
    response_format: str = "csv",
    opener: Callable[..., Any] | None = None,
    query_fn: Callable[..., Iterable[Mapping[str, object]]] | None = None,
    saturated_policy: object = "subdivide",
    max_subdivide_depth: object = 2,
    max_queries: object = 10000,
    raise_on_error: bool = True,
    on_tile: Callable[[TileQueryRecord], object] | None = None,
) -> GaiaTiledResult:
    """同步执行可靠分块查询。

    ``tile_limit`` 是每一块独立传给现有 ``query_gaia`` 的 ``limit``，而
    不是整个大视场的总上限。若返回行数达到该值，该块会被标记为可能
    截断；默认继续用半径更小的球面网格覆盖该块。显式传
    ``tile_limit=None`` 可以请求不带 ``TOP`` 的块，但由于远端服务仍可
    有隐含上限，结果会始终标记为 ``complete=False``，以免把不可观测的
    不确定性误报成完整星表。

    ``query_fn`` 的调用约定与现有 ``query_gaia`` 相同。默认值为现有
    ``query_gaia``，但它只在这个函数被显式调用时才产生网络请求。
    """

    center_ra = _coordinate_ra(center_ra_deg, name="center_ra_deg")
    center_dec = _coordinate_dec(center_dec_deg, name="center_dec_deg")
    search_radius = _radius(search_radius_deg, name="search_radius_deg")
    tile_radius_value = _resolve_tile_radius(tile_radius_deg, tile_radius, default=1.0)
    tile_limit_value = _positive_int(tile_limit, name="tile_limit", allow_none=True)
    max_depth = _positive_int(
        max_subdivide_depth,
        name="max_subdivide_depth",
        allow_none=False,
        allow_zero=True,
    )
    max_queries_value = _positive_int(max_queries, name="max_queries", allow_none=False)
    policy = _normalise_saturated_policy(saturated_policy)
    if not isinstance(raise_on_error, bool):
        raise GaiaTilingError("raise_on_error must be a boolean")
    if on_tile is not None and not callable(on_tile):
        raise GaiaTilingError("on_tile must be callable or None")

    normalised_strategy = _normalise_strategy(strategy)
    initial_tiles = plan_gaia_tiles(
        center_ra,
        center_dec,
        search_radius,
        tile_radius_deg=tile_radius_value,
        strategy=normalised_strategy,
    )
    query_callable = query_gaia if query_fn is None else query_fn
    if not callable(query_callable):
        raise GaiaTilingError("query_fn must be callable or None")

    pending: deque[GaiaTile] = deque(initial_tiles)
    records: list[TileQueryRecord] = []
    rows_by_source_id: dict[str, dict[str, object]] = {}
    unresolved: list[str] = []
    errors: list[str] = []
    duplicate_count = 0
    outside_scope_count = 0

    while pending:
        tile = pending.popleft()
        if len(records) >= max_queries_value:
            message = f"maximum tile query count {max_queries_value} reached before {tile.tile_id}"
            unresolved.append(tile.tile_id)
            errors.append(message)
            partial = _assemble_result(
                rows_by_source_id=rows_by_source_id,
                records=records,
                center_ra=center_ra,
                center_dec=center_dec,
                search_radius=search_radius,
                strategy=normalised_strategy,
                tile_radius=tile_radius_value,
                tile_limit=tile_limit_value,
                unresolved=unresolved,
                errors=errors,
                duplicate_count=duplicate_count,
                outside_scope_count=outside_scope_count,
                initial_tile_count=len(initial_tiles),
            )
            if raise_on_error:
                raise GaiaTiledQueryError(message, partial_result=partial)
            break

        adql = build_tile_adql(
            tile,
            tile_limit=tile_limit_value,
            min_g_mag=min_g_mag,
            max_g_mag=max_g_mag,
        )
        try:
            raw_rows = _materialise_query_rows(
                query_callable(
                    tile.ra_deg,
                    tile.dec_deg,
                    tile.radius_deg,
                    limit=tile_limit_value,
                    timeout=timeout,
                    endpoint=endpoint,
                    response_format=response_format,
                    min_g_mag=min_g_mag,
                    max_g_mag=max_g_mag,
                    opener=opener,
                )
            )
            returned_rows = len(raw_rows)
            local_rows: dict[str, dict[str, object]] = {}
            local_duplicate_count = 0
            outside_rows = 0
            for row_number, raw_row in enumerate(raw_rows, start=1):
                row = dict(raw_row)
                source_id = _row_source_id(row, row_number=row_number)
                row_ra, row_dec = _row_coordinates(row, row_number=row_number)
                if angular_distance_deg(center_ra, center_dec, row_ra, row_dec) > search_radius + _ANGLE_EPSILON_DEG:
                    outside_rows += 1
                    continue
                if source_id in local_rows:
                    local_duplicate_count += 1
                    continue
                local_rows[source_id] = row

            new_rows = 0
            cross_tile_duplicates = 0
            for source_id, row in local_rows.items():
                if source_id in rows_by_source_id:
                    cross_tile_duplicates += 1
                else:
                    rows_by_source_id[source_id] = row
                    new_rows += 1
            duplicate_rows = local_duplicate_count + cross_tile_duplicates
            duplicate_count += duplicate_rows
            outside_scope_count += outside_rows

            if tile_limit_value is None:
                truncation_status = "unknown_limit"
                saturated = True
            elif returned_rows >= tile_limit_value:
                truncation_status = "saturated"
                saturated = True
            else:
                truncation_status = "not_saturated"
                saturated = False

            # With no caller-provided limit there is no observable saturation
            # signal to guide an adaptive retry.  Subdividing would merely
            # multiply queries while leaving the same remote hard-cap
            # uncertainty, so report it once as unknown instead.
            if (
                saturated
                and tile_limit_value is not None
                and policy == "subdivide"
                and tile.depth < max_depth
            ):
                children = _subdivide_tile(tile)
                if children:
                    pending.extendleft(reversed(children))
                    status = "saturated_subdivided"
                else:
                    unresolved.append(tile.tile_id)
                    status = "saturated_unresolved"
            elif saturated:
                unresolved.append(tile.tile_id)
                status = "saturated_unresolved"
            else:
                status = "ok"

            record = TileQueryRecord(
                sequence=len(records),
                tile=tile,
                requested_limit=tile_limit_value,
                returned_rows=returned_rows,
                in_scope_rows=len(local_rows),
                new_rows=new_rows,
                duplicate_rows=duplicate_rows,
                outside_scope_rows=outside_rows,
                truncation_status=truncation_status,
                status=status,
                adql=adql,
            )
            records.append(record)
            if on_tile is not None:
                on_tile(record)
            if saturated and policy == "raise":
                partial = _assemble_result(
                    rows_by_source_id=rows_by_source_id,
                    records=records,
                    center_ra=center_ra,
                    center_dec=center_dec,
                    search_radius=search_radius,
                    strategy=normalised_strategy,
                    tile_radius=tile_radius_value,
                    tile_limit=tile_limit_value,
                    unresolved=unresolved,
                    errors=errors,
                    duplicate_count=duplicate_count,
                    outside_scope_count=outside_scope_count,
                    initial_tile_count=len(initial_tiles),
                )
                raise GaiaTiledQueryError(
                    f"tile {tile.tile_id} returned {returned_rows} rows at limit {tile_limit_value!r}",
                    partial_result=partial,
                )
        except GaiaTiledQueryError:
            raise
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            errors.append(f"{tile.tile_id}: {error_text}")
            unresolved.append(tile.tile_id)
            record = TileQueryRecord(
                sequence=len(records),
                tile=tile,
                requested_limit=tile_limit_value,
                returned_rows=0,
                in_scope_rows=0,
                new_rows=0,
                duplicate_rows=0,
                outside_scope_rows=0,
                truncation_status="query_error",
                status="error",
                adql=adql,
                error=error_text,
            )
            records.append(record)
            if on_tile is not None:
                on_tile(record)
            partial = _assemble_result(
                rows_by_source_id=rows_by_source_id,
                records=records,
                center_ra=center_ra,
                center_dec=center_dec,
                search_radius=search_radius,
                strategy=normalised_strategy,
                tile_radius=tile_radius_value,
                tile_limit=tile_limit_value,
                unresolved=unresolved,
                errors=errors,
                duplicate_count=duplicate_count,
                outside_scope_count=outside_scope_count,
                initial_tile_count=len(initial_tiles),
            )
            if raise_on_error:
                raise GaiaTiledQueryError(
                    f"tile {tile.tile_id} query failed: {error_text}",
                    partial_result=partial,
                ) from exc

    return _assemble_result(
        rows_by_source_id=rows_by_source_id,
        records=records,
        center_ra=center_ra,
        center_dec=center_dec,
        search_radius=search_radius,
        strategy=normalised_strategy,
        tile_radius=tile_radius_value,
        tile_limit=tile_limit_value,
        unresolved=unresolved,
        errors=errors,
        duplicate_count=duplicate_count,
        outside_scope_count=outside_scope_count,
        initial_tile_count=len(initial_tiles),
    )


query_gaia_grid = query_gaia_tiled


__all__ = [
    "GaiaTile",
    "GaiaTiledQueryError",
    "GaiaTiledResult",
    "GaiaTilingError",
    "TileQueryRecord",
    "angular_distance_deg",
    "build_tile_adql",
    "deduplicate_source_rows",
    "filter_rows_to_spherical_cap",
    "plan_gaia_tiles",
    "plan_grid",
    "plan_grid_tiles",
    "plan_small_circle",
    "plan_small_circle_tiles",
    "query_gaia_grid",
    "query_gaia_tiled",
]
