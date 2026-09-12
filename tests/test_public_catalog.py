from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from rst19.public_catalog import camera_footprint_radius_deg, download_public_gaia_catalog


def _query_rows(*_args: object, **_kwargs: object) -> tuple[dict[str, object], ...]:
    return (
        {
            "source_id": "42",
            "ra": 129.5,
            "dec": -1.8,
            "phot_g_mean_mag": 11.2,
            "phot_bp_mean_mag": 11.7,
            "phot_rp_mean_mag": 10.9,
        },
    )


def test_camera_query_radius_covers_square_field_diagonal() -> None:
    assert math.isclose(camera_footprint_radius_deg(9.78, 9.78), 6.915504, rel_tol=0.0, abs_tol=1.0e-6)


def test_public_catalog_download_is_explicit_and_writes_audit(tmp_path: Path) -> None:
    output = tmp_path / "gaia.csv"
    result = download_public_gaia_catalog(
        129.5,
        -1.8,
        output,
        search_radius_deg=0.1,
        tile_radius_deg=1.0,
        min_g_mag=5.0,
        max_g_mag=13.5,
        query_fn=_query_rows,
    )

    assert result.complete is True
    assert result.row_count == 1
    assert result.output_path == output.resolve()
    assert result.audit_path.is_file()
    with output.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["source_id"] == "42"
    metadata = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert metadata["catalog"] == "Gaia DR3"
    assert metadata["complete"] is True
    assert metadata["csv_row_count"] == 1
    assert metadata["csv_sha256"]
    assert metadata["csv_byte_count"] == output.stat().st_size
    assert metadata["provenance"]["photometric_band"] == "G"
    assert "rows" not in metadata["tiled"]


def test_public_catalog_records_selected_endpoint(tmp_path: Path) -> None:
    endpoint = "https://gaia.aip.de/tap/sync"

    def query(*args: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        assert kwargs["endpoint"] == endpoint
        return _query_rows()

    result = download_public_gaia_catalog(
        129.5, -1.8, tmp_path / "gaia.csv", search_radius_deg=0.1,
        endpoint=endpoint, query_fn=query,
    )
    assert json.loads(result.audit_path.read_text(encoding="utf-8"))["endpoint"] == endpoint


def test_public_catalog_can_preserve_gspphot_model_absolute_magnitude(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []

    def query_rows(*_args: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        seen.append(kwargs)
        return (
            {
                "source_id": "43",
                "ra": 129.5,
                "dec": -1.8,
                "phot_g_mean_mag": 11.2,
                "phot_bp_mean_mag": 11.7,
                "phot_rp_mean_mag": 10.9,
                "mg_gspphot": 4.2,
                "mg_gspphot_lower": 3.9,
                "mg_gspphot_upper": 4.5,
            },
        )

    result = download_public_gaia_catalog(
        129.5,
        -1.8,
        tmp_path / "gaia-mg.csv",
        search_radius_deg=0.1,
        tile_radius_deg=1.0,
        include_gspphot_model=True,
        query_fn=query_rows,
    )

    assert seen[0]["include_gspphot_model"] is True
    assert result.as_dict()["table"] == (
        "gaiadr3.gaia_source LEFT OUTER JOIN gaiadr3.astrophysical_parameters"
    )
    assert result.as_dict()["provenance"]["gspphot_fields"].startswith("distance_gspphot")
    assert "mg_gspphot" in result.as_dict()["provenance"]["gspphot_fields"]
    csv_text = result.output_path.read_text(encoding="utf-8")
    assert "mg_gspphot" in csv_text
    assert ",4.2," in csv_text
