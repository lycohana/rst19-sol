from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rst19.source_group_audit import run_candidate_group_audit, write_candidate_group_artifacts


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
        (3, 20.0, 0.0, 20, 0, 250, 12, 20, 2.0, "", "", "True", "compact_quality", ""),
        (4, 40.0, 0.0, 40, 0, 900, 20, 100, 2.0, 12, 6, "True", "compact_quality", ""),
        (5, 42.0, 0.0, 42, 0, 400, 10, 60, 2.0, "", "", "True", "compact_quality", ""),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def test_group_audit_keeps_target_as_one_unresolved_parent_group(tmp_path: Path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    result = run_candidate_group_audit(catalog, target_ids=(1, 2), psf_fwhm=2.0)

    assert result.source_count == 5
    assert result.group_count == 3
    assert result.multi_member_group_count == 2
    assert result.unresolved_group_count == 1
    assert result.independent_group_candidate_count == 1
    assert len(result.targets) == 1
    target = result.targets[0]
    assert target.target_detection_ids == "1|2"
    assert target.group_detection_ids == "1|2"
    assert target.classification == "unresolved_group"
    assert target.centroid_span_px == pytest.approx(2.738)
    assert target.peak_span_px == pytest.approx(4.0)
    assert target.independent_psf_evidence is False
    assert "CODE_PATTERN" in target.flags
    assert "UNRESOLVED_BLEND" in target.flags

    compact = next(row for row in result.classes if row.feature_class == "compact_quality")
    assert compact.source_count == 3
    assert compact.multi_group_source_count == 2
    assert compact.unresolved_group_source_count == 0
    assert compact.independent_group_source_count == 2


def test_group_audit_writes_json_and_rejects_missing_target(tmp_path: Path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    result = run_candidate_group_audit(catalog, target_ids=(1,))
    output = write_candidate_group_artifacts(result, tmp_path / "out")

    assert (output / "source_groups.csv").is_file()
    assert (output / "source_group_class_summary.csv").is_file()
    assert (output / "source_group_targets.csv").is_file()
    payload = json.loads((output / "source_group_summary.json").read_text(encoding="utf-8"))
    assert payload["targets"][0]["target_detection_ids"] == "1"
    assert "物理恒星分组" in payload["interpretation_guardrails"][0]

    with pytest.raises(ValueError, match="not in source catalog"):
        run_candidate_group_audit(catalog, target_ids=(99,))


def test_group_radius_can_be_overridden(tmp_path: Path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)

    result = run_candidate_group_audit(catalog, target_ids=(1, 2), group_radius_px=2.0)

    assert result.multi_member_group_count == 1
    assert result.targets[0].classification == "isolated"
