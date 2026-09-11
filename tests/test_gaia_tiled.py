from __future__ import annotations

import json
import math

import pytest

from rst19 import gaia_tiled
from rst19.gaia_tiled import (
    GaiaTiledQueryError,
    angular_distance_deg,
    deduplicate_source_rows,
    filter_rows_to_spherical_cap,
    plan_gaia_tiles,
    plan_grid_tiles,
    plan_small_circle_tiles,
    query_gaia_tiled,
)


def _destination(ra_deg: float, dec_deg: float, distance_deg: float, bearing_deg: float) -> tuple[float, float]:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    distance = math.radians(distance_deg)
    bearing = math.radians(bearing_deg)
    sin_dec = math.sin(dec) * math.cos(distance) + math.cos(dec) * math.sin(distance) * math.cos(bearing)
    out_dec = math.asin(max(-1.0, min(1.0, sin_dec)))
    out_ra = ra + math.atan2(
        math.sin(bearing) * math.sin(distance) * math.cos(dec),
        math.cos(distance) - math.sin(dec) * math.sin(out_dec),
    )
    return math.degrees(out_ra) % 360.0, math.degrees(out_dec)


def test_planning_is_pure_and_does_not_call_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("tile planning must not query Gaia")

    monkeypatch.setattr(gaia_tiled, "query_gaia", fail)

    tiles = plan_gaia_tiles(359.8, 0.2, 2.2, tile_radius_deg=0.8, strategy="grid")

    assert len(tiles) > 1
    assert all(0.0 <= tile.ra_deg < 360.0 for tile in tiles)
    assert all(angular_distance_deg(359.8, 0.2, tile.ra_deg, tile.dec_deg) <= 3.0 + 1.0e-7 for tile in tiles)


def test_small_circle_is_one_exact_query_and_accepts_ra_wrap() -> None:
    tiles = plan_small_circle_tiles(360.0, -1.0, 0.25, tile_radius_deg=0.5)

    assert len(tiles) == 1
    assert tiles[0].ra_deg == 0.0
    assert tiles[0].dec_deg == -1.0
    assert tiles[0].radius_deg == 0.25
    assert tiles[0].strategy == "small_circle"


def test_short_tile_radius_alias_is_supported() -> None:
    tiles = plan_grid_tiles(10.0, 20.0, 1.5, tile_radius=0.5)

    assert len(tiles) > 1
    assert all(tile.radius_deg == 0.5 for tile in tiles)


def test_small_circle_rejects_a_field_larger_than_control_radius() -> None:
    with pytest.raises(gaia_tiled.GaiaTilingError, match="use grid"):
        plan_small_circle_tiles(1.0, 2.0, 1.1, tile_radius_deg=1.0)


def test_manually_created_tile_normalises_ra_360_for_adql() -> None:
    tile = gaia_tiled.GaiaTile("manual", 360.0, 0.0, 0.2)

    assert tile.ra_deg == 0.0
    assert "CIRCLE('ICRS', 0, 0, 0.2)" in gaia_tiled.build_tile_adql(tile, tile_limit=10)


def test_grid_covers_spherical_cap_across_ra_seam() -> None:
    center_ra, center_dec, search_radius, tile_radius = 359.8, 0.5, 2.2, 0.8
    tiles = plan_grid_tiles(center_ra, center_dec, search_radius, tile_radius)

    for distance in (0.0, 0.5, 1.1, search_radius - 1.0e-7, search_radius):
        for bearing in range(0, 360, 10):
            ra, dec = _destination(center_ra, center_dec, distance, bearing)
            assert min(
                angular_distance_deg(ra, dec, tile.ra_deg, tile.dec_deg) - tile.radius_deg for tile in tiles
            ) <= 1.0e-7


def test_large_692_degree_field_is_not_sent_as_one_query() -> None:
    tiles = plan_gaia_tiles(
        129.533548,
        -1.845372,
        6.92,
        tile_radius_deg=1.0,
        strategy="grid",
    )

    assert len(tiles) > 1
    assert all(tile.radius_deg <= 1.0 for tile in tiles)
    assert max(
        angular_distance_deg(129.533548, -1.845372, tile.ra_deg, tile.dec_deg)
        for tile in tiles
    ) <= 7.92 + 1.0e-7


def test_grid_covers_a_cap_near_the_north_pole() -> None:
    center_ra, center_dec, search_radius, tile_radius = 37.0, 88.0, 1.5, 0.6
    tiles = plan_grid_tiles(center_ra, center_dec, search_radius, tile_radius)

    for distance in (0.0, 0.75, search_radius):
        for bearing in range(0, 360, 15):
            ra, dec = _destination(center_ra, center_dec, distance, bearing)
            assert min(
                angular_distance_deg(ra, dec, tile.ra_deg, tile.dec_deg) - tile.radius_deg for tile in tiles
            ) <= 1.0e-7


def test_filter_uses_great_circle_distance_and_dedupes_source_rows() -> None:
    rows = [
        {"source_id": "1", "ra": "359.9", "dec": "0"},
        {"source_id": "1", "ra": "359.9", "dec": "0"},
        {"source_id": "2", "ra_deg": "0.2", "dec_deg": "0"},
        {"source_id": "3", "ra_deg": "2.0", "dec_deg": "0"},
    ]

    filtered = filter_rows_to_spherical_cap(rows, 0.0, 0.0, 0.5)
    unique = deduplicate_source_rows(filtered)

    assert [row["source_id"] for row in filtered] == ["1", "1", "2"]
    assert [row["source_id"] for row in unique] == ["1", "2"]


def test_tiled_query_filters_scope_deduplicates_and_records_each_tile() -> None:
    calls: list[dict[str, object]] = []

    def fake_query(ra: float, dec: float, radius: float, **kwargs: object):
        calls.append({"ra": ra, "dec": dec, "radius": radius, **kwargs})
        return (
            {"source_id": "10", "ra_deg": "359.9", "dec_deg": "0.0", "phot_g_mean_mag": "12"},
            {"source_id": "11", "ra_deg": "0.2", "dec_deg": "0.1", "phot_g_mean_mag": "13"},
            {"source_id": "outside", "ra_deg": "5.0", "dec_deg": "0.0"},
        )

    result = query_gaia_tiled(
        0.0,
        0.0,
        1.0,
        tile_radius_deg=0.45,
        tile_limit=100,
        strategy="grid",
        query_fn=fake_query,
    )

    assert result.complete
    assert len(calls) == result.queried_tile_count == result.initial_tile_count
    assert [row["source_id"] for row in result.rows] == ["10", "11"]
    assert result.duplicate_count > 0
    assert result.outside_scope_count == len(calls)
    assert all(call["limit"] == 100 for call in calls)
    assert all(record.status == "ok" for record in result.tile_records)
    assert all("CIRCLE('ICRS'" in record.adql for record in result.tile_records)

    encoded = json.dumps(result.as_dict(), ensure_ascii=False)
    assert '"complete": true' in encoded


def test_saturated_tile_is_subdivided_and_children_can_complete() -> None:
    calls: list[tuple[float, float, float, int | None]] = []

    def fake_query(ra: float, dec: float, radius: float, **kwargs: object):
        limit = kwargs["limit"]
        calls.append((ra, dec, radius, limit))
        if radius >= 0.999:
            return (
                {"source_id": "1", "ra_deg": "0.0", "dec_deg": "0.0"},
                {"source_id": "2", "ra_deg": "0.1", "dec_deg": "0.0"},
            )
        return ({"source_id": "1", "ra_deg": "0.0", "dec_deg": "0.0"},)

    result = query_gaia_tiled(
        0.0,
        0.0,
        1.0,
        tile_radius_deg=1.0,
        tile_limit=2,
        strategy="small_circle",
        query_fn=fake_query,
        max_subdivide_depth=1,
    )

    assert result.complete
    assert len(calls) > 1
    assert any(record.status == "saturated_subdivided" for record in result.tile_records)
    assert result.unresolved_tile_ids == ()
    assert set(result.truncated_tile_ids) == {"tile-00000"}
    assert {row["source_id"] for row in result.rows} == {"1", "2"}


def test_unresolved_saturation_is_not_reported_as_complete() -> None:
    def fake_query(*args: object, **kwargs: object):
        return (
            {"source_id": "1", "ra_deg": "0.0", "dec_deg": "0.0"},
            {"source_id": "2", "ra_deg": "0.1", "dec_deg": "0.0"},
        )

    result = query_gaia_tiled(
        0.0,
        0.0,
        1.0,
        tile_radius_deg=1.0,
        tile_limit=2,
        strategy="small_circle",
        query_fn=fake_query,
        max_subdivide_depth=0,
        saturated_policy="record",
    )

    assert not result.complete
    assert result.unresolved_tile_ids == ("tile-00000",)
    assert result.tile_records[0].truncation_status == "saturated"


def test_tile_limit_none_is_explicitly_unknown_and_not_complete() -> None:
    def fake_query(*args: object, **kwargs: object):
        assert kwargs["limit"] is None
        return ({"source_id": "1", "ra_deg": "0", "dec_deg": "0"},)

    result = query_gaia_tiled(
        0.0,
        0.0,
        0.1,
        tile_radius_deg=0.2,
        tile_limit=None,
        query_fn=fake_query,
        max_subdivide_depth=0,
        saturated_policy="record",
    )

    assert not result.complete
    assert result.tile_records[0].truncation_status == "unknown_limit"
    assert result.tile_records[0].possibly_truncated


def test_query_error_keeps_a_record_and_can_return_partial_result() -> None:
    def failing_query(*args: object, **kwargs: object):
        raise RuntimeError("synthetic connector failure")

    result = query_gaia_tiled(
        0.0,
        0.0,
        0.1,
        tile_radius_deg=0.2,
        query_fn=failing_query,
        raise_on_error=False,
    )

    assert not result.complete
    assert len(result.tile_records) == 1
    assert result.tile_records[0].status == "error"
    assert "synthetic connector failure" in result.tile_records[0].error


def test_query_error_default_raises_with_partial_result() -> None:
    def failing_query(*args: object, **kwargs: object):
        raise RuntimeError("synthetic connector failure")

    with pytest.raises(GaiaTiledQueryError) as caught:
        query_gaia_tiled(0.0, 0.0, 0.1, tile_radius_deg=0.2, query_fn=failing_query)

    assert not caught.value.partial_result.complete
    assert caught.value.partial_result.tile_records[0].status == "error"


@pytest.mark.parametrize(
    ("ra", "dec", "radius"),
    [
        (0.0, 91.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (361.0, 0.0, 1.0),
    ],
)
def test_planner_rejects_invalid_spherical_inputs(ra: object, dec: object, radius: object) -> None:
    with pytest.raises(gaia_tiled.GaiaTilingError):
        plan_gaia_tiles(ra, dec, radius)
