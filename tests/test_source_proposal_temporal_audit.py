from __future__ import annotations

import csv
import json
from pathlib import Path

from rst19.source_proposal_temporal_audit import (
    run_source_proposal_temporal_audit,
    write_source_proposal_temporal_artifacts,
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: tuple[tuple[object, ...], ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
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
        ),
        (
            (1, "compact_quality", "紧凑", "True", "", "gaussian|dog_narrow|dog_broad"),
            (2, "compact_quality", "紧凑", "True", "", "dog_narrow|dog_broad"),
            (3, "crowded_blend", "拥挤", "False", "UNRESOLVED_BLEND", "gaussian"),
            (4, "crowded_blend", "拥挤", "False", "UNRESOLVED_BLEND", "dog_narrow|dog_broad"),
        ),
    )
    diagnostic = tmp_path / "sequence_feature_diagnostic_sources.csv"
    _write_csv(
        diagnostic,
        (
            "detection_id",
            "feature_class",
            "feature_class_label",
            "diagnostic_subgroup",
            "diagnostic_subgroup_label",
            "quality_passed",
            "proposal_methods",
            "candidate_presence",
            "candidate_same_subgroup_presence",
            "quality_presence",
            "quality_same_subgroup_presence",
            "frame_count",
            "required_presence",
            "association_radius_px",
        ),
        (
            (3, "crowded_blend", "拥挤", "blend_unresolved", "未分辨", "False", "gaussian", 4, 3, 0, 0, 5, 4, 1.0),
            (
                4,
                "crowded_blend",
                "拥挤",
                "blend_unresolved",
                "未分辨",
                "False",
                "dog_narrow|dog_broad",
                5,
                5,
                2,
                1,
                5,
                4,
                1.0,
            ),
        ),
    )
    subgroup = tmp_path / "sequence_feature_source_subgroup_persistence.csv"
    _write_csv(
        subgroup,
        (
            "feature_class",
            "feature_class_label",
            "proposal_subgroup",
            "proposal_subgroup_label",
            "anchor_count",
            "anchor_quality_count",
            "frame_count",
            "required_presence",
            "association_radius_px",
            "candidate_mean_presence",
            "candidate_presence_ge_required_count",
            "candidate_presence_all_frames_count",
            "quality_mean_presence",
            "quality_presence_ge_required_count",
            "quality_presence_all_frames_count",
        ),
        (
            ("compact_quality", "紧凑", "all_three", "三路", 1, 1, 5, 4, 1.0, 5, 1, 1, 5, 1, 1),
            ("compact_quality", "紧凑", "dog_only_no_deblend", "DoG", 1, 1, 5, 4, 1.0, 2, 0, 0, 1, 0, 0),
        ),
    )
    return catalog, diagnostic, subgroup


def test_source_proposal_temporal_audit_joins_exact_and_aggregate_rows(tmp_path: Path) -> None:
    catalog, diagnostic, subgroup = _inputs(tmp_path)
    result = run_source_proposal_temporal_audit(catalog, diagnostic, subgroup, target_ids=(1, 3))

    summaries = {(row.feature_class, row.method_group): row for row in result.summaries}
    assert summaries[("compact_quality", "all_three")].detail_level == "subgroup_aggregate"
    assert summaries[("compact_quality", "all_three")].candidate_presence_ge_required_count == 1
    assert summaries[("compact_quality", "dog_only")].quality_presence_ge_required_count == 0
    assert summaries[("crowded_blend", "gaussian_only")].detail_level == "source_exact"
    assert summaries[("crowded_blend", "gaussian_only")].candidate_presence_ge_required_count == 1

    targets = {row.detection_id: row for row in result.targets}
    assert targets[1].temporal_detail_available is False
    assert targets[1].candidate_presence is None
    assert "不是零响应" in targets[1].note
    assert targets[3].temporal_detail_available is True
    assert targets[3].candidate_presence == 4
    assert targets[3].quality_presence == 0


def test_source_proposal_temporal_audit_writes_artifacts(tmp_path: Path) -> None:
    catalog, diagnostic, subgroup = _inputs(tmp_path)
    result = run_source_proposal_temporal_audit(catalog, diagnostic, subgroup, target_ids=(3,))
    output = write_source_proposal_temporal_artifacts(result, tmp_path / "out")

    assert (output / "source_proposal_temporal_rows.csv").is_file()
    assert (output / "source_proposal_temporal_summary.csv").is_file()
    assert (output / "source_proposal_temporal_targets.csv").is_file()
    payload = json.loads((output / "source_proposal_temporal_audit.json").read_text(encoding="utf-8"))
    assert payload["target_rows"][0]["detection_id"] == 3
    assert "不是星表身份" in payload["interpretation_guardrails"][1]
