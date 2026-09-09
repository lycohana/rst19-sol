from __future__ import annotations

import csv
import json
from pathlib import Path

from rst19.source_proposal_peak_audit import (
    proposal_method_group,
    run_source_proposal_peak_audit,
    write_source_proposal_peak_artifacts,
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: tuple[tuple[object, ...], ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    catalog = tmp_path / "source_catalog.csv"
    _write_csv(
        catalog,
        (
            "detection_id",
            "feature_class",
            "feature_class_label",
            "quality_passed",
            "flags",
            "proposal_methods",
            "flux_snr",
            "filter_snr",
        ),
        (
            (1, "compact_quality", "compact", "True", "", "gaussian|dog_narrow|dog_broad", 10, 20),
            (2, "compact_quality", "compact", "True", "", "gaussian", 8, 11),
            (3, "crowded_blend", "blend", "False", "UNRESOLVED_BLEND", "dog_narrow|dog_broad", 9, 30),
            (4, "crowded_blend", "blend", "False", "UNRESOLVED_BLEND", "gaussian", 7, 12),
        ),
    )
    peak = tmp_path / "source_peak_consistency.csv"
    _write_csv(
        peak,
        (
            "detection_id",
            "raw_valid",
            "raw_local_maximum_r1",
            "raw_unique_local_maximum_r1",
            "raw_local_max_gap_adu_r1",
        ),
        (
            (1, "True", "True", "True", 0),
            (2, "True", "False", "False", 4),
            (3, "True", "True", "True", 0),
            (4, "True", "False", "False", 9),
        ),
    )
    return catalog, peak


def test_method_group_is_deterministic() -> None:
    assert proposal_method_group("gaussian|dog_narrow|dog_broad") == "all_three"
    assert proposal_method_group("gaussian") == "gaussian_only"
    assert proposal_method_group("dog_narrow|dog_broad") == "dog_only"
    assert proposal_method_group("gaussian|dog_narrow") == "partial"
    assert proposal_method_group("") == "no_method"


def test_source_proposal_peak_audit_joins_categories_and_targets(tmp_path: Path) -> None:
    catalog, peak = _inputs(tmp_path)
    result = run_source_proposal_peak_audit(catalog, peak, target_ids=(2, 4))

    summaries = {(row.feature_class, row.method_group): row for row in result.summaries}
    assert summaries[("compact_quality", "all_three")].raw_local_maximum_fraction_r1 == 1.0
    assert summaries[("compact_quality", "gaussian_only")].nonlocal_peak_gap_median_adu_r1 == 4.0
    assert summaries[("crowded_blend", "gaussian_only")].quality_count == 0

    targets = {row.detection_id: row for row in result.targets}
    assert targets[2].method_group == "gaussian_only"
    assert targets[2].method_group_count == 1
    assert targets[2].method_group_share_in_class == 0.5
    assert targets[4].raw_local_maximum_r1 is False
    assert targets[4].raw_local_max_gap_adu_r1 == 9.0


def test_source_proposal_peak_audit_writes_artifacts(tmp_path: Path) -> None:
    catalog, peak = _inputs(tmp_path)
    result = run_source_proposal_peak_audit(catalog, peak, target_ids=(4,))
    output = write_source_proposal_peak_artifacts(result, tmp_path / "out")

    assert (output / "source_proposal_peak_rows.csv").is_file()
    assert (output / "source_proposal_peak_summary.csv").is_file()
    assert (output / "source_proposal_peak_targets.csv").is_file()
    payload = json.loads((output / "source_proposal_peak_audit.json").read_text(encoding="utf-8"))
    assert payload["targets"][0]["detection_id"] == 4
    assert "不是独立投票" in payload["interpretation_boundary"]
