from __future__ import annotations

import json

from rst19.experiments import FeatureAuditRow
from rst19.feature_audit_replicates import (
    run_feature_audit_replicates,
    write_feature_audit_replicate_artifacts,
)


def _row(
    scenario: str,
    control_type: str,
    *,
    candidate_hits: int,
    quality_hits: int,
    truth_count: int,
    nearby_candidates: int,
    nearby_quality: int,
) -> FeatureAuditRow:
    return FeatureAuditRow(
        feature_class="compact_quality" if control_type == "positive" else "linear_artifact",
        scenario=scenario,
        control_type=control_type,
        truth_count=truth_count,
        candidate_true_hits=candidate_hits,
        quality_true_hits=quality_hits,
        candidate_recall=candidate_hits / truth_count if truth_count else None,
        quality_recall=quality_hits / truth_count if truth_count else None,
        nearby_candidate_count=nearby_candidates,
        nearby_quality_count=nearby_quality,
        candidate_count=nearby_candidates + 1,
        quality_count=nearby_quality,
        nearby_flags="LINE_ARTIFACT:1" if control_type == "negative" else "",
        note="test",
    )


def test_feature_audit_replicates_separates_recall_and_negative_leakage(monkeypatch, tmp_path) -> None:
    calls: list[int] = []

    def fake_run_feature_audit(*, seed: int, **_kwargs):
        calls.append(seed)
        return (
            _row(
                "known",
                "positive",
                candidate_hits=1,
                quality_hits=0,
                truth_count=1,
                nearby_candidates=1,
                nearby_quality=0,
            ),
            _row(
                "negative",
                "negative",
                candidate_hits=0,
                quality_hits=0,
                truth_count=0,
                nearby_candidates=2,
                nearby_quality=0,
            ),
        )

    monkeypatch.setattr("rst19.feature_audit_replicates.run_feature_audit", fake_run_feature_audit)
    result = run_feature_audit_replicates(trials=2, seed=11, detector_mode="test")
    rows = {row.scenario: row for row in result.rows}

    assert calls == [11, 1_000_014]
    assert rows["known"].truth_count == 2
    assert rows["known"].candidate_recall == 1.0
    assert rows["known"].quality_recall == 0.0
    assert rows["known"].quality_recall_wilson95_low == 0.0
    assert rows["negative"].truth_count == 0
    assert rows["negative"].candidate_recall is None
    assert rows["negative"].mean_nearby_candidate_count == 2.0
    assert rows["negative"].max_nearby_candidate_count == 2
    assert rows["negative"].nearby_flags == "LINE_ARTIFACT:2"

    output = write_feature_audit_replicate_artifacts(result, tmp_path / "replicates")
    assert (output / "feature_audit_replicates.csv").is_file()
    payload = json.loads((output / "feature_audit_replicates.json").read_text(encoding="utf-8"))
    assert payload["trial_count"] == 2
    assert payload["negative_scenarios"] == 1
