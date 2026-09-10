from __future__ import annotations

import pytest

from rst19.detection import Detection
from rst19.photometry import (
    absolute_magnitude_from_apparent,
    absolute_magnitude_from_parallax,
    calibrated_apparent_magnitude,
    find_faintest_source,
    inferred_zero_point_for_comparison,
    instrumental_magnitude,
)


def _source(
    detection_id: int,
    flux: float,
    *,
    snr: float = 8.0,
    flags: tuple[str, ...] = (),
) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=20.0 + detection_id,
        y=30.0,
        peak=100.0,
        flux=flux,
        background=10.0,
        noise=2.0,
        snr=snr,
        fwhm=2.0,
        flags=flags,
    )


def test_instrumental_magnitude_uses_positive_flux() -> None:
    assert instrumental_magnitude(10.0) == pytest.approx(-2.5)
    assert instrumental_magnitude(10.0, exposure_s=2.0) == pytest.approx(-1.747425)
    assert instrumental_magnitude(0.0) is None
    assert instrumental_magnitude(-1.0) is None


def test_find_faintest_source_uses_quality_filtered_low_flux() -> None:
    result = find_faintest_source(
        (
            _source(0, 100.0),
            _source(1, 10.0),
            _source(2, 1.0, snr=3.0),
        ),
        min_snr=5.0,
    )

    assert result is not None
    assert result.detection_id == 1
    assert result.instrumental_magnitude == pytest.approx(-2.5)
    assert result.calibrated_magnitude is None


def test_find_faintest_source_excludes_bad_quality_flags_and_applies_zero_point() -> None:
    result = find_faintest_source(
        (
            _source(0, 8.0, flags=("EDGE",)),
            _source(1, 12.0, flags=("MASKED",)),
            _source(2, 20.0),
        ),
        min_snr=5.0,
        zero_point=25.0,
    )

    assert result is not None
    assert result.detection_id == 2
    assert result.instrumental_magnitude == pytest.approx(-3.252574989)
    assert result.calibrated_magnitude == pytest.approx(21.747425, abs=1e-5)


def test_find_faintest_source_keeps_flux_rate_belonging_to_selected_source() -> None:
    # The last iterated source is deliberately brighter than the selected
    # faint source.  This catches loop-variable leakage in the result field.
    result = find_faintest_source(
        (
            _source(1, 10.0),
            _source(99, 100.0),
        ),
        min_snr=5.0,
        exposure_s=2.0,
    )

    assert result is not None
    assert result.detection_id == 1
    assert result.flux_rate == pytest.approx(5.0)


def test_standard_and_absolute_magnitude_are_separate_quantities() -> None:
    instrumental = instrumental_magnitude(219.0, exposure_s=1.5)
    assert instrumental == pytest.approx(-5.410882139461092)
    apparent = calibrated_apparent_magnitude(instrumental, zero_point=18.970882139461092)
    assert apparent == pytest.approx(13.56)
    # The same apparent value becomes a different absolute value at every distance.
    assert absolute_magnitude_from_apparent(apparent, distance_pc=100.0) == pytest.approx(8.56)
    assert absolute_magnitude_from_parallax(apparent, parallax_mas=10.0) == pytest.approx(8.56)
    assert inferred_zero_point_for_comparison(instrumental, 13.56) == pytest.approx(18.970882139461092)


def test_absolute_magnitude_rejects_missing_or_non_positive_distance() -> None:
    with pytest.raises(ValueError):
        absolute_magnitude_from_apparent(13.56, distance_pc=0.0)
    with pytest.raises(ValueError):
        absolute_magnitude_from_parallax(13.56, parallax_mas=-1.0)
    with pytest.raises(ValueError):
        calibrated_apparent_magnitude(1.0, zero_point=18.0, color_coefficient=0.2)
