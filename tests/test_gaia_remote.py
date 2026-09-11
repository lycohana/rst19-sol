from __future__ import annotations

import io
import math
import urllib.error
import urllib.parse
import urllib.request

import pytest

from rst19 import gaia_remote
from rst19 import gaia_remote_cli
from rst19.catalog import load_catalog_csv
from rst19.gaia_remote import (
    GaiaHTTPError,
    GaiaInputError,
    GaiaMissingColumnsError,
    GaiaTimeoutError,
    build_gaia_adql,
    parse_gaia_csv,
    parse_gaia_json,
    query_gaia,
    write_catalog_csv,
)


GAIA_HEADER = ",".join(gaia_remote.GAIA_REQUIRED_COLUMNS)
GAIA_ROW = "123,129.5,-1.25,2016.0,1.2,-2.3,15.4,1000,10,100,15.8,14.9,1.2,2.1,0.1,1.05,false,12,NOT_AVAILABLE,100.0,90.0,110.0,0.12,0.10,0.15,0.08"


def test_build_gaia_adql_selects_required_columns_and_encodes_numeric_values() -> None:
    query = build_gaia_adql(129.5, -1.25, 0.2, limit=25)

    assert "SELECT TOP 25" in query
    assert "FROM gaiadr3.gaia_source" in query
    assert "POINT('ICRS', ra, dec)" in query
    assert "CIRCLE('ICRS', 129.5, -1.25, 0.2)" in query
    for column in gaia_remote.GAIA_REQUIRED_COLUMNS:
        assert column in query


def test_build_gaia_adql_supports_explicit_g_magnitude_window() -> None:
    query = build_gaia_adql(129.5, -1.25, 0.2, min_g_mag=8, max_g_mag=18.5)

    assert "phot_g_mean_mag >= 8" in query
    assert "phot_g_mean_mag <= 18.5" in query


def test_build_gaia_adql_rejects_an_inverted_or_unbounded_magnitude_window() -> None:
    with pytest.raises(GaiaInputError, match="min_g_mag"):
        build_gaia_adql(129.5, -1.25, 0.2, min_g_mag=19, max_g_mag=18)
    with pytest.raises(GaiaInputError):
        build_gaia_adql(129.5, -1.25, 0.2, max_g_mag=51)


@pytest.mark.parametrize(
    ("ra", "dec", "radius"),
    [
        ("129.5); DROP TABLE gaiadr3.gaia_source; --", -1.25, 0.2),
        (129.5, "-1.25 OR 1=1", 0.2),
        (129.5, -1.25, "0.2); SELECT * FROM secret; --"),
        (360.0, 0.0, 1.0),
        (0.0, 91.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    ],
)
def test_build_gaia_adql_rejects_injection_and_invalid_coordinates(ra: object, dec: object, radius: object) -> None:
    with pytest.raises(GaiaInputError):
        build_gaia_adql(ra, dec, radius)


def test_parse_gaia_csv_returns_catalog_compatible_aliases() -> None:
    rows = parse_gaia_csv(f"{GAIA_HEADER}\n{GAIA_ROW}\n")

    assert len(rows) == 1
    row = rows[0]
    assert row["source_id"] == "123"
    assert row["ra_deg"] == "129.5"
    assert row["dec_deg"] == "-1.25"
    assert row["magnitude"] == "15.4"
    assert float(row["magnitude_error"]) == pytest.approx(
        (2.5 / math.log(10.0)) * 10.0 / 1000.0
    )
    assert row["phot_g_mean_mag_error"] == row["magnitude_error"]
    assert row["phot_g_mean_flux_over_error"] == "100"
    assert row["phot_bp_rp_excess_factor"] == "1.2"
    assert row["phot_variable_flag"] == "NOT_AVAILABLE"
    assert row["pmra_mas_yr"] == "1.2"
    assert row["pmdec_mas_yr"] == "-2.3"
    assert row["parallax_mas"] == "2.1"
    assert row["distance_pc"] == "100.0"
    assert row["distance_lower_pc"] == "90.0"
    assert row["distance_upper_pc"] == "110.0"
    assert row["distance_source"] == "Gaia DR3 GSP-Phot"
    assert row["extinction_mag"] == "0.12"
    assert float(row["extinction_error_mag"]) == pytest.approx(0.025)
    assert row["extinction_band"] == "G"
    assert row["extinction_system"] == "Gaia"
    assert row["extinction_source"] == "Gaia DR3 GSP-Phot: ag_gspphot"
    assert row["color"] == "0.9"
    assert row["color_name"] == "BP-RP"
    assert row["catalog_name"] == "Gaia DR3"


def test_parse_gaia_json_supports_metadata_and_array_rows() -> None:
    payload = {
        "metadata": [{"name": column} for column in gaia_remote.GAIA_REQUIRED_COLUMNS],
        "data": [[value for value in GAIA_ROW.split(",")]],
    }

    rows = parse_gaia_json(payload)

    assert rows[0]["source_id"] == "123"
    assert rows[0]["magnitude"] == "15.4"
    assert rows[0]["ruwe"] == "1.05"


def test_parse_gaia_json_preserves_a0_semantics_without_treating_it_as_gaia_g() -> None:
    values = GAIA_ROW.split(",")
    # Remove the standard Gaia GSP-Phot A_G triplet so this fixture represents
    # an A0-only response carrying an optional monochromatic extinction value.
    values[22:25] = ["", "", ""]
    columns = [*gaia_remote.GAIA_REQUIRED_COLUMNS, "a0"]
    payload = {
        "metadata": [{"name": column} for column in columns],
        "data": [values + ["0.3"]],
    }

    rows = parse_gaia_json(payload)

    assert rows[0]["extinction_mag"] == "0.3"
    assert rows[0]["extinction_band"] == "A0(541.4 nm)"
    assert rows[0]["extinction_system"] == "monochromatic"
    assert rows[0]["extinction_source"] == "response column: a0"


def test_parse_gaia_response_reports_missing_columns_clearly() -> None:
    with pytest.raises(GaiaMissingColumnsError, match="phot_rp_mean_mag"):
        parse_gaia_csv("source_id,ra,dec\n1,1,2\n")


def test_query_gaia_uses_encoded_tap_request_and_parses_success() -> None:
    class Response:
        status = 200
        reason = "OK"

        def read(self) -> bytes:
            return f"{GAIA_HEADER}\n{GAIA_ROW}\n".encode("utf-8")

        def close(self) -> None:
            return None

    calls: list[tuple[object, float]] = []

    def opener(request: object, *, timeout: float) -> Response:
        calls.append((request, timeout))
        return Response()

    rows = query_gaia(
        1.0,
        2.0,
        0.1,
        limit=1,
        timeout=12.5,
        endpoint="https://example.invalid/tap/sync",
        opener=opener,
    )

    assert rows[0]["source_id"] == "123"
    assert len(calls) == 1
    request, timeout = calls[0]
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)  # type: ignore[attr-defined]
    assert params["FORMAT"] == ["csv"]
    assert params["MAXREC"] == ["1"]
    assert "TOP 1" in params["QUERY"][0]
    assert timeout == 12.5


def test_download_gaia_reports_http_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise urllib.error.HTTPError(
            "https://example.invalid/tap/sync",
            503,
            "Service Unavailable",
            {},
            io.BytesIO(b"upstream unavailable"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)

    with pytest.raises(GaiaHTTPError, match="503.*upstream unavailable"):
        gaia_remote.download_gaia(1.0, 2.0, 0.1, endpoint="https://example.invalid/tap/sync")


def test_download_gaia_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fail)

    with pytest.raises(GaiaTimeoutError, match="timed out"):
        gaia_remote.download_gaia(1.0, 2.0, 0.1, endpoint="https://example.invalid/tap/sync")


def test_cli_dry_run_prints_adql_without_network_or_output_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("dry-run must not open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    output = tmp_path / "should-not-exist.csv"

    assert (
        gaia_remote_cli.main(
            [
                "--ra",
                "1",
                "--dec",
                "2",
                "--radius",
                "0.1",
                "--out",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "SELECT" in captured.out
    assert "CIRCLE('ICRS', 1, 2, 0.1)" in captured.out
    assert captured.err == ""
    assert not output.exists()


def test_write_catalog_csv_keeps_downstream_columns(tmp_path) -> None:
    rows = parse_gaia_csv(f"{GAIA_HEADER}\n{GAIA_ROW}\n")
    path = write_catalog_csv(rows, tmp_path / "gaia.csv")

    text = path.read_text(encoding="utf-8")
    assert "source_id,ra_deg,dec_deg,magnitude" in text.splitlines()[0]
    assert "123,129.5,-1.25,15.4" in text


def test_write_catalog_csv_round_trips_gspphot_distance_and_extinction(tmp_path) -> None:
    rows = parse_gaia_csv(f"{GAIA_HEADER}\n{GAIA_ROW}\n")
    path = write_catalog_csv(rows, tmp_path / "gaia-gspphot.csv")

    source = load_catalog_csv(path)[0]

    assert source.distance_pc == pytest.approx(100.0)
    assert source.distance_lower_pc == pytest.approx(90.0)
    assert source.distance_upper_pc == pytest.approx(110.0)
    assert source.distance_source == "Gaia DR3 GSP-Phot"
    assert source.extinction_mag == pytest.approx(0.12)
    assert source.extinction_error_mag == pytest.approx(0.025)
    assert source.extinction_band == "G"
    assert source.extinction_system == "Gaia"
    assert source.extinction_source == "Gaia DR3 GSP-Phot: ag_gspphot"


def test_load_catalog_csv_keeps_legacy_generic_extinction_unknown(tmp_path) -> None:
    path = tmp_path / "legacy-extinction.csv"
    path.write_text(
        "source_id,ra,dec,phot_g_mean_mag,extinction_mag\n"
        "legacy,10.0,20.0,12.5,0.2\n",
        encoding="utf-8",
    )

    source = load_catalog_csv(path)[0]

    assert source.extinction_mag == pytest.approx(0.2)
    assert source.extinction_band == "unknown"
    assert source.extinction_system == "unknown"
    assert source.extinction_source == "CSV column: extinction_mag"
    assert source.extinction_compatibility() == "unknown"


def test_load_catalog_csv_accepts_gaia_g_extinction_only_when_semantics_match(tmp_path) -> None:
    path = tmp_path / "gaia-g-extinction.csv"
    path.write_text(
        "source_id,ra,dec,phot_g_mean_mag,extinction_mag,extinction_band,"
        "extinction_system,extinction_source\n"
        "g,10.0,20.0,12.5,0.2,G,Gaia,calibration\n",
        encoding="utf-8",
    )

    source = load_catalog_csv(path)[0]

    assert source.extinction_band == "G"
    assert source.extinction_system == "Gaia"
    assert source.extinction_source == "calibration"
    assert source.extinction_compatibility() == "compatible"


@pytest.mark.parametrize(
    ("column", "value"),
    [("a_v", "0.2"), ("azero_gspphot", "0.3")],
)
def test_load_catalog_csv_rejects_known_extinction_band_mismatch_with_gaia_g(
    tmp_path, column: str, value: str
) -> None:
    path = tmp_path / f"mismatch-{column}.csv"
    path.write_text(
        f"source_id,ra,dec,phot_g_mean_mag,{column}\n"
        f"bad,10.0,20.0,12.5,{value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incompatible"):
        load_catalog_csv(path)
