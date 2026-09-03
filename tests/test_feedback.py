from __future__ import annotations

import json

from rst19.feedback import ManualThresholdFeedback, append_manual_feedback, load_manual_feedback


def test_manual_threshold_feedback_round_trips_as_jsonl(tmp_path) -> None:
    path = tmp_path / "manual-feedback.jsonl"
    feedback = ManualThresholdFeedback(
        timestamp_utc="2026-08-30T00:00:00+00:00",
        frame_path="F01.fits",
        threshold_sigma=3.5,
        min_flux_snr=3.0,
        min_distance=4,
        psf_fwhm=3.0,
        candidate_count=120,
        quality_count=80,
        returned_count=120,
        flag_counts={"LOW_FLUX_SNR": 40, "SPIKE": 2},
        judgement="保留弱星",
    )

    append_manual_feedback(path, feedback)
    records = load_manual_feedback(path)

    assert records == (feedback,)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["flag_counts"] == {"LOW_FLUX_SNR": 40, "SPIKE": 2}


def test_load_manual_feedback_missing_file_is_empty(tmp_path) -> None:
    assert load_manual_feedback(tmp_path / "does-not-exist.jsonl") == ()
