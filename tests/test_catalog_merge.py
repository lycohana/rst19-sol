from __future__ import annotations

import csv
import json
from pathlib import Path

from rst19.catalog import load_catalog_csv
from rst19.catalog_merge import merge_catalog_csvs


def _write_catalog(path: Path, rows: list[dict[str, str]]) -> None:
    fields = tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_audit(path: Path, *, min_g: float, max_g: float, complete: bool = True) -> None:
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "catalog": "Gaia DR3",
                "complete": complete,
                "center_ra_deg": 129.5,
                "center_dec_deg": -1.8,
                "search_radius_deg": 6.92,
                "min_g_mag": min_g,
                "max_g_mag": max_g,
            }
        ),
        encoding="utf-8",
    )


def test_merge_layers_preserves_richer_duplicate_and_audit(tmp_path: Path) -> None:
    bright = tmp_path / "bright.csv"
    deep = tmp_path / "deep.csv"
    _write_catalog(
        bright,
        [
            {
                "source_id": "1",
                "ra_deg": "129.5",
                "dec_deg": "-1.8",
                "magnitude": "6.2",
                "color": "0.4",
                "color_name": "BP-RP",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
            },
            {
                "source_id": "2",
                "ra_deg": "129.6",
                "dec_deg": "-1.7",
                "magnitude": "7.1",
                "color": "0.8",
                "color_name": "BP-RP",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
            },
        ],
    )
    _write_catalog(
        deep,
        [
            {
                "source_id": "2",
                "ra_deg": "129.6",
                "dec_deg": "-1.7",
                "magnitude": "7.1",
                "color": "0.8",
                "color_name": "BP-RP",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
                "parallax_mas": "2.3",
            },
            {
                "source_id": "3",
                "ra_deg": "129.7",
                "dec_deg": "-1.6",
                "magnitude": "15.4",
                "color": "1.1",
                "color_name": "BP-RP",
                "photometric_system": "Gaia Vega",
                "photometric_band": "G",
            },
        ],
    )
    _write_audit(bright, min_g=5.0, max_g=13.5)
    _write_audit(deep, min_g=8.0, max_g=18.0)

    result = merge_catalog_csvs((bright, deep), tmp_path / "merged.csv")

    assert result.complete is True
    assert result.row_count == 3
    assert result.duplicate_count == 1
    assert result.conflict_count == 0
    merged = load_catalog_csv(result.output_path)
    source_two = next(source for source in merged if source.source_id == "2")
    assert source_two.parallax_mas == 2.3
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["complete"] is True
    assert len(audit["inputs"]) == 2


def test_merge_without_complete_input_stays_incomplete(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    row = {
        "source_id": "1",
        "ra_deg": "129.5",
        "dec_deg": "-1.8",
        "magnitude": "6.2",
        "color": "0.4",
        "color_name": "BP-RP",
        "photometric_system": "Gaia Vega",
        "photometric_band": "G",
    }
    _write_catalog(first, [row])
    _write_catalog(second, [dict(row, source_id="2", magnitude="15.0")])
    _write_audit(first, min_g=5.0, max_g=13.5, complete=True)
    _write_audit(second, min_g=8.0, max_g=18.0, complete=False)

    result = merge_catalog_csvs((first, second), tmp_path / "merged.csv")

    assert result.complete is False
    assert result.input_audits_complete is False
    payload = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert payload["incomplete_reasons"] == ["至少一份输入缺少 complete=true 审计"]


def test_merge_rejects_critical_conflict(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    row = {
        "source_id": "1",
        "ra_deg": "129.5",
        "dec_deg": "-1.8",
        "magnitude": "6.2",
        "color": "0.4",
        "color_name": "BP-RP",
        "photometric_system": "Gaia Vega",
        "photometric_band": "G",
    }
    _write_catalog(first, [row])
    _write_catalog(second, [dict(row, magnitude="6.9")])
    _write_audit(first, min_g=5.0, max_g=13.5)
    _write_audit(second, min_g=8.0, max_g=18.0)

    result = merge_catalog_csvs((first, second), tmp_path / "merged.csv")

    assert result.complete is False
    assert result.conflict_count == 1
    assert result.conflicting_source_ids == ("1",)
