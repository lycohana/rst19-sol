from __future__ import annotations

import pytest

from rst19.calibration_sensitivity import build_calibration_model_sensitivity
from rst19.catalog import CatalogSource
from rst19.detection import Detection
from rst19.matching import CatalogMatch
from rst19.photometry import instrumental_magnitude


def _fixture(count: int = 12):
    detections = tuple(
        Detection(
            detection_id=index,
            x=100.0 + index,
            y=200.0 + index,
            peak=200.0 + index,
            flux=100.0 + index * 8.0,
            background=10.0,
            noise=2.0,
            snr=30.0,
            flux_snr=30.0,
            fwhm=2.0,
            flags=(),
        )
        for index in range(count)
    )
    catalog = tuple(
        CatalogSource(
            str(index),
            10.0 + index * 0.01,
            20.0 + index * 0.01,
            magnitude=(
                instrumental_magnitude(detection.flux)
                + 20.0
                + 0.25 * (index / 5.0)
                + 0.04 * (index / 5.0) ** 2
            ),
            color=index / 5.0,
            color_name="BP-RP",
            photometric_system="Gaia Vega",
            photometric_band="G",
        )
        for index, detection in enumerate(detections)
    )
    matches = tuple(
        CatalogMatch(
            index,
            str(index),
            detection.x,
            detection.y,
            detection.x,
            detection.y,
            0.1,
            catalog[index].magnitude,
            catalog_color=catalog[index].color,
            catalog_color_name="BP-RP",
            photometric_system="Gaia Vega",
            photometric_band="G",
        )
        for index, detection in enumerate(detections)
    )
    return matches, detections, catalog


def test_model_sensitivity_reports_low_order_models_and_faintest_agreement() -> None:
    matches, detections, catalog = _fixture()

    result = build_calibration_model_sensitivity(
        matches,
        detections,
        catalog,
        exposure_s=1.0,
        photometric_system="Gaia Vega",
        photometric_band="G",
        color_name="BP-RP",
        min_calibrators=6,
    )

    assert result["status"] in {"MODEL_STABLE", "MODEL_SENSITIVE"}
    assert result["model_count_requested"] == 3
    # The zero-order model intentionally fails the residual gate for this
    # colour-dependent fixture; the two colour-aware variants remain usable.
    assert result["model_count_usable"] == 2
    assert result["matched_measurement_count"] == 12
    assert result["faintest_ranking"]["valid_model_count"] == 2
    assert 0.0 <= result["faintest_ranking"]["winner_agreement_fraction"] <= 1.0
    assert all(row["valid_model_count"] >= 1 for row in result["sources"])
    assert all(row["model_spread_mag"] is not None for row in result["sources"])


def test_model_sensitivity_does_not_call_unlabelled_offset_a_standard_result() -> None:
    matches, detections, catalog = _fixture()

    result = build_calibration_model_sensitivity(
        matches,
        detections,
        catalog,
        photometric_system="unknown",
        photometric_band="unknown",
        min_calibrators=6,
    )

    assert result["status"] == "DIAGNOSTIC_ONLY"
    assert result["model_count_usable"] == 0
    assert all(model["usable"] is False for model in result["models"])
    assert "Gaia G 等于开运一号真实标准波段星等" in result["claim_boundary"]["cannot_claim"]


def test_model_order_validation_is_explicit() -> None:
    matches, detections, catalog = _fixture()
    with pytest.raises(ValueError, match="model_orders"):
        build_calibration_model_sensitivity(matches, detections, catalog, model_orders=(3,))
