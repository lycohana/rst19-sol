from __future__ import annotations

import json

import numpy as np
import pytest

from rst19.detection import Detection, DetectionResult
from rst19.signed_null import _raw_evidence_layer, run_signed_null_audit, write_signed_null_artifacts


def _source(
    detection_id: int,
    *,
    filter_snr: float,
    flux_snr: float,
    quality_passed: bool,
    flags: tuple[str, ...] = (),
    peak_x: float | None = None,
    peak_y: float | None = None,
) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=float(detection_id),
        y=2.0,
        peak=20.0,
        flux=50.0,
        background=10.0,
        noise=2.0,
        snr=filter_snr,
        fwhm=2.0,
        flags=flags,
        flux_snr=flux_snr,
        filter_snr=filter_snr,
        quality_passed=quality_passed,
        peak_x=peak_x,
        peak_y=peak_y,
    )


def _result(sources: tuple[Detection, ...]) -> DetectionResult:
    return DetectionResult(
        image_shape=(12, 12),
        background=10.0,
        noise=2.0,
        threshold=18.0,
        candidate_count=len(sources),
        sources=sources,
        quality_count=sum(source.quality_passed for source in sources),
        parameters={"test": True},
    )


def test_signed_null_audit_keeps_positive_and_sign_flipped_layers_separate(monkeypatch, tmp_path) -> None:
    calls: list[np.ndarray] = []
    positive = _result(
        (
            _source(1, filter_snr=20.0, flux_snr=15.0, quality_passed=True),
            _source(2, filter_snr=6.0, flux_snr=4.0, quality_passed=False, flags=("SPIKE",)),
        )
    )
    negative = _result(
        (_source(3, filter_snr=7.0, flux_snr=5.0, quality_passed=False, flags=("SPIKE",)),)
    )

    def fake_stats(*_args, **_kwargs):
        return 10.0, 2.0

    def fake_detect(image, **_kwargs):
        calls.append(np.asarray(image))
        return positive if len(calls) == 1 else negative

    monkeypatch.setattr("rst19.signed_null.sigma_clipped_stats", fake_stats)
    monkeypatch.setattr("rst19.signed_null.detect_sources", fake_detect)
    result = run_signed_null_audit(np.full((12, 12), 10, dtype=np.int16))

    assert result.background_adu == 10.0
    assert result.positive_candidate_count == 2
    assert result.positive_quality_count == 1
    assert result.negative_candidate_count == 1
    assert result.negative_quality_count == 0
    assert result.candidate_leakage_ratio == 0.5
    assert result.quality_leakage_ratio == 0.0
    thresholds = {row.threshold: row for row in result.thresholds}
    assert thresholds[5.0].positive_filter_count == 2
    assert thresholds[5.0].negative_filter_count == 1
    assert thresholds[5.0].positive_flux_count == 1
    assert thresholds[5.0].negative_flux_count == 1
    assert thresholds[5.0].filter_leakage_ratio == 0.5
    assert thresholds[10.0].negative_filter_count == 0
    controls = {row.control for row in result.feature_rows}
    assert controls == {"positive", "sign_flipped"}
    assert np.allclose(calls[0], 10.0)
    assert np.allclose(calls[1], 10.0)

    output = write_signed_null_artifacts(result, tmp_path / "signed-null")
    assert (output / "signed_null_summary.json").is_file()
    assert (output / "signed_null_thresholds.csv").is_file()
    assert (output / "signed_null_feature_counts.csv").is_file()
    payload = json.loads((output / "signed_null_summary.json").read_text(encoding="utf-8"))
    assert payload["positive_candidate_count"] == 2
    assert payload["parameters"]["not_formal_fdr"] is True


def test_signed_null_audit_zero_denominator_is_undefined(monkeypatch) -> None:
    empty = _result(())
    calls = 0

    def fake_detect(_image, **_kwargs):
        nonlocal calls
        calls += 1
        return empty

    monkeypatch.setattr("rst19.signed_null.sigma_clipped_stats", lambda *_args, **_kwargs: (10.0, 2.0))
    monkeypatch.setattr("rst19.signed_null.detect_sources", fake_detect)
    result = run_signed_null_audit(np.zeros((12, 12), dtype=np.float64))

    assert calls == 2
    assert result.candidate_leakage_ratio is None
    assert result.quality_leakage_ratio is None
    assert all(row.filter_leakage_ratio is None for row in result.thresholds)
    assert all(row.flux_leakage_ratio is None for row in result.thresholds)


def test_signed_null_audit_exposes_integer_range_transfer(monkeypatch) -> None:
    image = np.full((12, 12), 10, dtype=np.int16)
    image[5, 5] = -32000
    positive = _result(())
    negative = _result(
        (_source(1, filter_snr=30.0, flux_snr=20.0, quality_passed=True, peak_x=5.0, peak_y=5.0),)
    )
    calls = 0

    def fake_detect(_image, **_kwargs):
        nonlocal calls
        calls += 1
        return positive if calls == 1 else negative

    monkeypatch.setattr("rst19.signed_null.sigma_clipped_stats", lambda *_args, **_kwargs: (10.0, 2.0))
    monkeypatch.setattr("rst19.signed_null.detect_sources", fake_detect)
    result = run_signed_null_audit(image)

    assert result.negative_quality_count == 1
    assert result.negative_quality_count_after_range_exclusion == 0
    assert result.negative_range_transfer_candidate_count == 1
    assert result.negative_range_transfer_quality_count == 1
    assert len(result.negative_quality_sources) == 1
    source = result.negative_quality_sources[0]
    assert source.range_transfer_overlap is True
    assert source.raw_evidence_layer == "extreme_range_transfer"
    assert source.raw_peak_adu == -32000.0
    assert source.raw_extreme_negative_pixel_count >= 1
    assert result.quality_leakage_ratio is None
    assert result.quality_leakage_ratio_after_range_exclusion is None


def test_signed_null_thresholds_must_be_strictly_increasing(monkeypatch) -> None:
    monkeypatch.setattr("rst19.signed_null.sigma_clipped_stats", lambda *_args, **_kwargs: (10.0, 2.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        run_signed_null_audit(np.zeros((12, 12)), snr_thresholds=(4.0, 4.0))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    (
        (
            {
                "range_transfer_overlap": True,
                "raw_negative_pixel_count": 4,
                "raw_negative_anomaly_pixel_count": 4,
                "raw_minus_one_count": 4,
            },
            "extreme_range_transfer",
        ),
        (
            {
                "range_transfer_overlap": False,
                "raw_negative_pixel_count": 4,
                "raw_negative_anomaly_pixel_count": 2,
                "raw_minus_one_count": 4,
            },
            "raw_negative_anomaly",
        ),
        (
            {
                "range_transfer_overlap": False,
                "raw_negative_pixel_count": 1,
                "raw_negative_anomaly_pixel_count": 0,
                "raw_minus_one_count": 1,
            },
            "fixed_sentinel_candidate",
        ),
        (
            {
                "range_transfer_overlap": False,
                "raw_negative_pixel_count": 1,
                "raw_negative_anomaly_pixel_count": 0,
                "raw_minus_one_count": 0,
            },
            "raw_negative_other",
        ),
        (
            {
                "range_transfer_overlap": False,
                "raw_negative_pixel_count": 0,
                "raw_negative_anomaly_pixel_count": 0,
                "raw_minus_one_count": 0,
            },
            "no_negative_evidence",
        ),
    ),
)
def test_raw_evidence_layer_priority(kwargs: dict[str, object], expected: str) -> None:
    assert _raw_evidence_layer(**kwargs) == expected
