from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rst19.models import FitsFrame
from rst19.signed_null import SignedNullAuditResult, SignedNullFeatureRow, SignedNullThresholdRow
from rst19.signed_null_sequence import run_signed_null_sequence, write_signed_null_sequence_artifacts


def _audit(path: Path, *, positive_quality: int, negative_quality: int) -> SignedNullAuditResult:
    return SignedNullAuditResult(
        frame_path=str(path),
        image_shape=(12, 12),
        background_adu=10.0,
        positive_noise_adu=2.0,
        negative_noise_adu=2.1,
        positive_candidate_count=10,
        positive_quality_count=positive_quality,
        negative_candidate_count=2,
        negative_quality_count=negative_quality,
        negative_quality_count_after_range_exclusion=0,
        candidate_leakage_ratio=0.2,
        quality_leakage_ratio=negative_quality / positive_quality if positive_quality else None,
        quality_leakage_ratio_after_range_exclusion=0.0 if positive_quality else None,
        negative_range_transfer_candidate_count=1,
        negative_range_transfer_quality_count=negative_quality,
        thresholds=(
            SignedNullThresholdRow(
                threshold=5.0,
                positive_filter_count=8,
                negative_filter_count=1,
                positive_flux_count=4,
                negative_flux_count=0,
                filter_leakage_ratio=0.125,
                flux_leakage_ratio=0.0,
            ),
        ),
        feature_rows=(
            SignedNullFeatureRow(
                control="positive",
                feature_class="compact_quality",
                feature_class_label="紧凑质量候选",
                candidate_count=positive_quality,
                quality_count=positive_quality,
                quality_fraction=1.0 if positive_quality else None,
                median_flux_snr=10.0,
                max_flux_snr=20.0,
                common_flags="",
            ),
        ),
        parameters={"detector": {"threshold_sigma": 4.0}},
        conclusion="test",
    )


def test_signed_null_sequence_preserves_frames_and_pooled_denominators(monkeypatch, tmp_path) -> None:
    def fake_read_fits(path: Path) -> FitsFrame:
        return FitsFrame(
            path=Path(path),
            header={"DATE-OBS": "2026-03-30T16:32:00"},
            data=np.zeros((12, 12), dtype=np.int16),
            auxiliary=None,
            data_offset=0,
        )

    def fake_audit(frame: FitsFrame, **_kwargs) -> SignedNullAuditResult:
        return _audit(frame.path, positive_quality=4, negative_quality=1 if frame.path.name == "b.fits" else 0)

    monkeypatch.setattr("rst19.signed_null_sequence.read_fits", fake_read_fits)
    monkeypatch.setattr("rst19.signed_null_sequence.run_signed_null_audit", fake_audit)
    result = run_signed_null_sequence([Path("b.fits"), Path("a.fits")])

    assert result.frame_count == 2
    assert [row.frame_path for row in result.frame_rows] == ["b.fits", "a.fits"]
    assert result.pooled_positive_candidate_count == 20
    assert result.pooled_positive_quality_count == 8
    assert result.pooled_negative_candidate_count == 4
    assert result.pooled_negative_quality_count == 1
    assert result.pooled_negative_quality_count_after_range_exclusion == 0
    assert result.pooled_candidate_leakage_ratio == 0.2
    assert result.pooled_quality_leakage_ratio == 0.125
    assert result.pooled_quality_leakage_ratio_after_range_exclusion == 0.0
    assert result.frame_candidate_leakage_mean == 0.2
    assert result.frame_quality_leakage_median == 0.125

    output = write_signed_null_sequence_artifacts(result, tmp_path / "sequence")
    assert (output / "signed_null_sequence_summary.json").is_file()
    assert (output / "signed_null_sequence_frames.csv").is_file()
    assert (output / "signed_null_sequence_thresholds.csv").is_file()
    assert (output / "signed_null_sequence_features.csv").is_file()
    assert (output / "signed_null_sequence_quality_sources.csv").is_file()
    assert (output / "signed_null_sequence_recurrences.csv").is_file()
    payload = json.loads((output / "signed_null_sequence_summary.json").read_text(encoding="utf-8"))
    assert payload["frame_count"] == 2
    assert payload["pooled_negative_quality_count_after_range_exclusion"] == 0


def test_signed_null_sequence_requires_at_least_one_path() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        run_signed_null_sequence(())
