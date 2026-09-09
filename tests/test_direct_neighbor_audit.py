from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rst19.direct_neighbor_audit import run_direct_neighbor_audit, write_direct_neighbor_artifacts


def _write_catalog(path: Path) -> None:
    fields = (
        "detection_id",
        "x",
        "y",
        "peak_x",
        "peak_y",
        "peak",
        "flux_snr",
        "filter_snr",
        "fwhm",
        "deblend_delta_bic",
        "deblend_component_snr",
        "quality_passed",
        "feature_class",
        "flags",
    )
    rows = (
        (1, 0.0, 0.0, 0, 0, 3992, 204.6, 1195, 2.59, -4.09, 0.87, "False", "range_anomaly", "CODE_PATTERN"),
        (2, 2.738, 0.0, 4, 0, 569, 115.5, 300, 2.44, -4.09, 0.87, "False", "crowded_blend", "UNRESOLVED_BLEND"),
        (3, 10.0, 0.0, 10, 0, 250, 12, 20, 2.0, "", "", "True", "compact_quality", ""),
        (4, 20.0, 0.0, 20, 0, 900, 20, 100, 2.0, 12, 6, "True", "compact_quality", ""),
        (5, 23.0, 0.0, 23, 0, 400, 10, 60, 2.0, "", "", "True", "compact_quality", ""),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def test_direct_neighbors_keep_edges_without_transitive_grouping(tmp_path: Path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    result = run_direct_neighbor_audit(catalog, target_ids=(1, 2), pair_radius_px=4.0)

    assert result.source_count == 5
    assert result.direct_pair_count == 2
    assert result.unresolved_pair_count == 1
    assert result.independent_pair_candidate_count == 1
    assert [(row.detection_id_a, row.detection_id_b) for row in result.pairs] == [(1, 2), (4, 5)]
    assert result.target is not None
    assert result.target.in_pair_radius is True
    assert result.target.classification == "unresolved_pair"
    assert result.target.centroid_distance_px == pytest.approx(2.738)


def test_direct_neighbor_target_can_be_outside_radius(tmp_path: Path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    result = run_direct_neighbor_audit(catalog, target_ids=(1, 2), pair_radius_px=2.0)

    assert result.direct_pair_count == 0
    assert result.target is not None
    assert result.target.in_pair_radius is False
    assert result.target.classification == "outside_pair_radius"


def test_direct_neighbor_writes_machine_artifacts(tmp_path: Path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    result = run_direct_neighbor_audit(catalog, target_ids=(1, 2))

    output = write_direct_neighbor_artifacts(result, tmp_path / "out")

    assert (output / "direct_neighbor_pairs.csv").is_file()
    assert (output / "direct_neighbor_summaries.csv").is_file()
    assert (output / "direct_neighbor_targets.csv").is_file()
    payload = json.loads((output / "direct_neighbor_audit.json").read_text(encoding="utf-8"))
    assert payload["target"]["classification"] == "unresolved_pair"
    assert "传递连通" in payload["interpretation_guardrails"][0]
