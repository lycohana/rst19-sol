from __future__ import annotations

import math

import pytest

from rst19.catalog import CatalogSource
from rst19.detection import Detection
from rst19.matching import CatalogMatch
from rst19.photometry import (
    absolute_magnitude_estimate_from_distance,
    absolute_magnitude_estimate_from_parallax,
    absolute_magnitude_from_catalog,
    absolute_magnitude_from_apparent,
    absolute_magnitude_from_parallax,
    build_source_photometry,
    calibrated_apparent_magnitude,
    find_faintest_source,
    fit_photometric_calibration,
    inferred_zero_point_for_comparison,
    instrumental_magnitude,
    PhotometricCalibration,
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


def _catalog_source_with_forced_extinction_semantics(
    *,
    extinction_band: str,
    extinction_system: str,
    extinction_source: str,
) -> CatalogSource:
    """Build a legacy-shaped source, then exercise the photometry gate.

    The current CatalogSource constructor rejects an explicitly contradictory
    row early.  Bypassing that constructor check here lets this test cover the
    defensive contract of absolute_magnitude_from_catalog as well (for a
    deserialized/legacy object that can still reach the calculation layer).
    """

    source = CatalogSource(
        "extinction-gate",
        10.0,
        20.0,
        magnitude=13.56,
        photometric_system="Gaia Vega",
        photometric_band="G",
        parallax_mas=10.0,
        parallax_error_mas=0.1,
        extinction_mag=None,
    )
    object.__setattr__(source, "extinction_mag", 0.2)
    object.__setattr__(source, "extinction_band", extinction_band)
    object.__setattr__(source, "extinction_system", extinction_system)
    object.__setattr__(source, "extinction_source", extinction_source)
    return source


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
    assert result.instrumental_magnitude_error == pytest.approx(1.0857362047581296 / 8.0)


def test_find_faintest_source_ranks_by_calibrated_magnitude_when_available() -> None:
    calibration = PhotometricCalibration(
        photometric_system="test",
        photometric_band="G",
        color_name="color",
        color_order=1,
        coefficients=(0.0, 1.0),
        calibrator_count=8,
        inlier_count=8,
        validation_count=2,
        fit_rms_mag=0.01,
        validation_rms_mag=0.02,
        residual_mad_mag=0.01,
        color_min=0.0,
        color_max=5.0,
        status="VALID",
    )
    result = find_faintest_source(
        (_source(0, 100.0), _source(1, 200.0)),
        min_snr=5.0,
        photometric_calibration=calibration,
        source_colors={0: 0.0, 1: 5.0},
    )

    assert result is not None
    assert result.detection_id == 1
    assert result.calibrated_magnitude == pytest.approx(result.instrumental_magnitude + 5.0)
    assert result.selection_scope == "CALIBRATED_MATCHES"
    assert result.eligible_candidate_count == 2
    assert result.calibrated_candidate_count == 2


def test_find_faintest_source_marks_partial_calibration_domain() -> None:
    calibration = PhotometricCalibration(
        photometric_system="test",
        photometric_band="G",
        color_name="color",
        color_order=1,
        coefficients=(20.0, 0.1),
        calibrator_count=8,
        inlier_count=8,
        validation_count=2,
        fit_rms_mag=0.01,
        validation_rms_mag=0.02,
        residual_mad_mag=0.01,
        color_min=0.0,
        color_max=1.0,
        status="VALID",
    )
    result = find_faintest_source(
        (_source(0, 100.0), _source(1, 10.0)),
        photometric_calibration=calibration,
        source_colors={0: 0.5},
    )

    assert result is not None
    assert result.detection_id == 0
    assert result.calibrated_magnitude is not None
    assert result.selection_scope == "CALIBRATED_MATCHES_PARTIAL"
    assert result.eligible_candidate_count == 2
    assert result.calibrated_candidate_count == 1


def test_find_faintest_source_exposes_no_usable_calibration_instead_of_silent_claim() -> None:
    calibration = PhotometricCalibration(
        photometric_system="test",
        photometric_band="G",
        color_name="color",
        color_order=1,
        coefficients=(20.0, 0.1),
        calibrator_count=8,
        inlier_count=8,
        validation_count=2,
        fit_rms_mag=0.01,
        validation_rms_mag=0.02,
        residual_mad_mag=0.01,
        color_min=0.0,
        color_max=1.0,
        status="VALID",
    )
    result = find_faintest_source(
        (_source(0, 100.0),),
        photometric_calibration=calibration,
        source_colors={},
    )

    assert result is not None
    assert result.calibrated_magnitude is None
    assert result.selection_scope == "QUALITY_DETECTIONS_NO_USABLE_CALIBRATION"
    assert result.eligible_candidate_count == 1
    assert result.calibrated_candidate_count == 0


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


def test_fit_photometric_calibration_recovers_zero_point_and_color_term() -> None:
    detections = tuple(_source(index, 10.0 + index * 3.0, snr=20.0) for index in range(8))
    catalog: list[CatalogSource] = []
    matches: list[CatalogMatch] = []
    for index, detection in enumerate(detections):
        color = 0.2 + 0.25 * index
        instrumental = instrumental_magnitude(detection.flux)
        assert instrumental is not None
        catalog.append(
            CatalogSource(
                source_id=f"g{index}",
                ra_deg=10.0 + index,
                dec_deg=20.0,
                magnitude=instrumental + 20.0 + 0.3 * color,
                color=color,
                color_name="BP-RP",
                catalog_name="Gaia DR3",
                photometric_system="Gaia Vega",
                photometric_band="G",
            )
        )
        matches.append(
            CatalogMatch(
                detection_id=detection.detection_id,
                source_id=f"g{index}",
                detection_x=detection.x,
                detection_y=detection.y,
                predicted_x=detection.x,
                predicted_y=detection.y,
                residual_px=0.1,
                catalog_magnitude=catalog[-1].magnitude,
            )
        )

    calibration = fit_photometric_calibration(
        matches,
        detections,
        catalog,
        exposure_s=1.0,
        photometric_system="Gaia Vega",
        photometric_band="G",
        color_name="BP-RP",
        min_calibrators=3,
    )

    assert calibration.status == "VALID"
    assert calibration.zero_point == pytest.approx(20.0, abs=1e-8)
    assert calibration.color_coefficient == pytest.approx(0.3, abs=1e-8)
    assert calibration.calibrator_count == 8
    assert calibration.inlier_count == 6
    assert calibration.validation_count >= 1
    assert calibration.validation_rms_mag == pytest.approx(0.0, abs=1e-8)
    assert calibration.apply(-5.0, color=1.0) == pytest.approx(15.3)
    assert dict(calibration.calibration_sample_roles)
    assert set(dict(calibration.calibration_sample_roles).values()) <= {
        "validation",
        "training_inlier",
        "training_outlier",
    }


def test_fit_photometric_calibration_rejects_bad_holdout_quality() -> None:
    detections = tuple(_source(index, 20.0 + index * 4.0, snr=30.0) for index in range(10))
    catalog: list[CatalogSource] = []
    matches: list[CatalogMatch] = []
    for index, detection in enumerate(detections):
        color = 0.2 + 0.15 * index
        instrumental = instrumental_magnitude(detection.flux)
        assert instrumental is not None
        offset = 1.0 if index == 9 else 0.0
        catalog_magnitude = instrumental + 20.0 + 0.2 * color + offset
        catalog.append(
            CatalogSource(
                source_id=f"holdout-{index}",
                ra_deg=10.0 + index,
                dec_deg=20.0,
                magnitude=catalog_magnitude,
                color=color,
                color_name="BP-RP",
                catalog_name="Gaia DR3",
                photometric_system="Gaia Vega",
                photometric_band="G",
            )
        )
        matches.append(
            CatalogMatch(
                detection_id=detection.detection_id,
                source_id=catalog[-1].source_id,
                detection_x=detection.x,
                detection_y=detection.y,
                predicted_x=detection.x,
                predicted_y=detection.y,
                residual_px=0.1,
                catalog_magnitude=catalog_magnitude,
            )
        )

    calibration = fit_photometric_calibration(
        matches,
        detections,
        catalog,
        photometric_system="Gaia Vega",
        photometric_band="G",
        color_name="BP-RP",
        min_calibrators=3,
    )

    assert calibration.status == "PHOTOMETRY_REJECTED_QUALITY_GATE"
    assert calibration.validation_rms_mag is not None
    assert calibration.validation_rms_mag > 0.35
    assert "VALIDATION_RMS_EXCEEDS_LIMIT" in calibration.flags
    assert calibration.apply(-5.0, color=1.0) is None


def test_photometric_outlier_is_retained_but_not_used_for_faintest_selection() -> None:
    good = _source(0, 100.0, snr=30.0)
    outlier = _source(1, 10.0, snr=30.0)
    good_instrumental = instrumental_magnitude(good.flux)
    outlier_instrumental = instrumental_magnitude(outlier.flux)
    assert good_instrumental is not None and outlier_instrumental is not None
    catalog = (
        CatalogSource(
            source_id="good",
            ra_deg=10.0,
            dec_deg=20.0,
            magnitude=good_instrumental + 20.0,
            color=0.5,
            color_name="BP-RP",
            catalog_name="Gaia DR3",
            photometric_system="Gaia Vega",
            photometric_band="G",
        ),
        CatalogSource(
            source_id="outlier",
            ra_deg=10.1,
            dec_deg=20.0,
            magnitude=outlier_instrumental + 25.0,
            color=0.5,
            color_name="BP-RP",
            catalog_name="Gaia DR3",
            photometric_system="Gaia Vega",
            photometric_band="G",
        ),
    )
    matches = (
        CatalogMatch(0, "good", good.x, good.y, good.x, good.y, 0.1, catalog[0].magnitude),
        CatalogMatch(1, "outlier", outlier.x, outlier.y, outlier.x, outlier.y, 0.1, catalog[1].magnitude),
    )
    calibration = PhotometricCalibration(
        photometric_system="Gaia Vega",
        photometric_band="G",
        color_name="BP-RP",
        color_order=1,
        coefficients=(20.0, 0.0),
        calibrator_count=8,
        inlier_count=8,
        validation_count=2,
        fit_rms_mag=0.01,
        validation_rms_mag=0.02,
        residual_mad_mag=0.01,
        color_min=0.0,
        color_max=2.0,
        status="VALID",
        max_source_residual_mag=0.5,
    )
    rows = build_source_photometry(
        (good, outlier),
        matches=matches,
        catalog=catalog,
        photometric_calibration=calibration,
    )

    assert rows[0].photometric_consistent is True
    assert rows[1].photometric_consistent is False
    assert rows[1].status == "CATALOG_INCONSISTENT"
    assert rows[1].photometric_outlier_reason == "RESIDUAL_EXCEEDS_LIMIT"
    assert rows[1].photometric_residual_mag == pytest.approx(5.0)
    assert rows[1].absolute_magnitude is None
    faintest = find_faintest_source(
        (good, outlier),
        photometric_calibration=calibration,
        source_colors={0: 0.5, 1: 0.5},
        source_photometry={row.detection_id: row for row in rows},
    )
    assert faintest is not None
    assert faintest.detection_id == 0
    assert faintest.selection_scope == "CALIBRATED_MATCHES_PARTIAL"


def test_fit_photometric_calibration_filters_explicit_gaia_quality_flags() -> None:
    detections = tuple(_source(index, 10.0 + index * 3.0, snr=30.0) for index in range(12))
    catalog: list[CatalogSource] = []
    matches: list[CatalogMatch] = []
    for index, detection in enumerate(detections):
        color = 0.2 + 0.12 * index
        instrumental = instrumental_magnitude(detection.flux)
        assert instrumental is not None
        catalog.append(
            CatalogSource(
                source_id=f"quality-{index}",
                ra_deg=10.0 + index,
                dec_deg=20.0,
                magnitude=instrumental + 20.0 + 0.25 * color,
                color=color,
                color_name="BP-RP",
                catalog_name="Gaia DR3",
                photometric_system="Gaia Vega",
                photometric_band="G",
                phot_g_mean_flux_over_error=5.0 if index == 8 else 100.0,
                ruwe=2.0 if index == 9 else 1.1,
                duplicated_source=index == 7,
                visibility_periods_used=3 if index == 10 else 12,
                phot_variable_flag="VARIABLE" if index == 11 else "NOT_AVAILABLE",
            )
        )
        matches.append(
            CatalogMatch(
                detection_id=detection.detection_id,
                source_id=catalog[-1].source_id,
                detection_x=detection.x,
                detection_y=detection.y,
                predicted_x=detection.x,
                predicted_y=detection.y,
                residual_px=0.1,
                catalog_magnitude=catalog[-1].magnitude,
            )
        )

    calibration = fit_photometric_calibration(
        matches,
        detections,
        catalog,
        photometric_system="Gaia Vega",
        photometric_band="G",
        color_name="BP-RP",
        min_calibrators=3,
    )

    assert calibration.status == "VALID"
    assert calibration.calibrator_count == 7
    assert dict(calibration.catalog_filter_counts) == {
        "DUPLICATED_SOURCE": 1,
        "HIGH_RUWE": 1,
        "LOW_CATALOG_FLUX_SNR": 1,
        "LOW_VISIBILITY_PERIODS": 1,
        "VARIABLE_SOURCE": 1,
    }
    assert calibration.zero_point == pytest.approx(20.0, abs=1e-8)
    assert calibration.color_coefficient == pytest.approx(0.25, abs=1e-8)


def test_fit_photometric_calibration_reports_insufficient_references() -> None:
    detection = _source(0, 10.0, snr=20.0)
    catalog = (CatalogSource("g0", 10.0, 20.0, magnitude=12.0, color=0.5),)
    matches = (CatalogMatch(0, "g0", detection.x, detection.y, detection.x, detection.y, 0.1, 12.0),)

    calibration = fit_photometric_calibration(matches, (detection,), catalog, min_calibrators=3)

    assert calibration.status == "INSUFFICIENT_CALIBRATORS"
    assert calibration.zero_point is None
    assert "INSUFFICIENT_CALIBRATORS" in calibration.flags


def test_absolute_magnitude_estimate_has_quality_gate_and_error() -> None:
    estimate = absolute_magnitude_estimate_from_parallax(
        13.56,
        parallax_mas=10.0,
        parallax_error_mas=0.1,
        extinction_mag=0.2,
        apparent_magnitude_error=0.05,
        extinction_error_mag=0.03,
        extinction_band="G",
        extinction_system="Gaia",
    )

    assert estimate.status == "VALID"
    assert estimate.value == pytest.approx(8.36)
    assert estimate.error is not None and estimate.error > 0
    assert estimate.distance_interval_status == "DERIVED_FROM_PARALLAX_ERROR"
    assert estimate.distance_lower_pc == pytest.approx(1000.0 / 10.1)
    assert estimate.distance_upper_pc == pytest.approx(1000.0 / 9.9)
    assert estimate.distance_error_pc == pytest.approx(
        (1000.0 / 9.9 - 1000.0 / 10.1) / 2.0
    )
    assert estimate.error_status == "AVAILABLE"
    assert estimate.is_strict is True
    assert estimate.is_strict_with_uncertainty is True

    rejected = absolute_magnitude_estimate_from_parallax(
        13.56,
        parallax_mas=10.0,
        parallax_error_mas=3.0,
        extinction_mag=0.2,
    )
    assert rejected.value is None
    assert rejected.status == "LOW_PARALLAX_SNR"
    assert "USE_DISTANCE_POSTERIOR" in rejected.flags

    missing_parallax_error = absolute_magnitude_estimate_from_parallax(
        13.56,
        parallax_mas=10.0,
        parallax_error_mas=None,
        extinction_mag=0.2,
        extinction_band="G",
        extinction_system="Gaia",
    )
    assert missing_parallax_error.value is None
    assert missing_parallax_error.status == "NO_PARALLAX_ERROR"
    assert missing_parallax_error.distance_interval_status == "NOT_PROVIDED"
    assert "DISTANCE_ERROR_NOT_PROVIDED" in missing_parallax_error.flags


def test_generic_absolute_magnitude_helpers_carry_optional_extinction_provenance() -> None:
    parallax_estimate = absolute_magnitude_estimate_from_parallax(
        13.56,
        parallax_mas=10.0,
        parallax_error_mas=0.1,
        extinction_mag=0.2,
        extinction_band="G",
        extinction_system="Gaia",
        extinction_source="test-catalog",
    )
    assert parallax_estimate.status == "VALID"
    assert parallax_estimate.extinction_band == "G"
    assert parallax_estimate.extinction_system == "Gaia"
    assert parallax_estimate.extinction_source == "test-catalog"
    assert parallax_estimate.as_dict()["extinction_band"] == "G"
    assert parallax_estimate.as_dict()["extinction_system"] == "Gaia"
    assert parallax_estimate.as_dict()["extinction_source"] == "test-catalog"

    distance_estimate = absolute_magnitude_estimate_from_distance(
        13.56,
        distance_pc=100.0,
        distance_lower_pc=95.0,
        distance_upper_pc=106.0,
        distance_source="test-distance",
        extinction_mag=0.2,
        extinction_band="G",
        extinction_system="Gaia",
        extinction_source="test-catalog",
    )
    assert distance_estimate.status == "VALID_MODEL_DISTANCE"
    assert distance_estimate.extinction_band == "G"
    assert distance_estimate.extinction_system == "Gaia"
    assert distance_estimate.extinction_source == "test-catalog"
    assert distance_estimate.error is None
    assert distance_estimate.error_status == "INCOMPLETE"
    assert "APPARENT_MAGNITUDE_ERROR_NOT_PROVIDED" in distance_estimate.flags
    assert "EXTINCTION_ERROR_NOT_PROVIDED" in distance_estimate.flags

    # A bare numeric extinction remains a supported generic API input, but no
    # photometric semantics are invented for it.
    legacy_estimate = absolute_magnitude_estimate_from_parallax(
        13.56,
        parallax_mas=10.0,
        parallax_error_mas=0.1,
        extinction_mag=0.2,
    )
    assert legacy_estimate.status == "EXTINCTION_SEMANTICS_REQUIRED"
    assert legacy_estimate.value is None
    assert legacy_estimate.extinction_band is None
    assert legacy_estimate.extinction_system is None
    assert legacy_estimate.extinction_source is None
    assert legacy_estimate.distance_interval_status == "DERIVED_FROM_PARALLAX_ERROR"


def test_catalog_absolute_magnitude_requires_distance_metadata() -> None:
    source = CatalogSource(
        "g1",
        10.0,
        20.0,
        magnitude=13.56,
        magnitude_error=0.04,
        photometric_system="Gaia Vega",
        photometric_band="G",
        parallax_mas=10.0,
        parallax_error_mas=0.1,
        extinction_mag=0.2,
        extinction_error_mag=0.03,
        extinction_band="G",
        extinction_system="Gaia",
        extinction_source="Gaia DR3 GSP-Phot: ag_gspphot",
    )
    estimate = absolute_magnitude_from_catalog(
        source,
        required_photometric_system="Gaia Vega",
        required_photometric_band="G",
    )
    assert estimate.status == "VALID"
    assert estimate.value == pytest.approx(8.36)
    assert estimate.extinction_band == "G"
    assert estimate.extinction_system == "Gaia"
    assert estimate.extinction_source == "Gaia DR3 GSP-Phot: ag_gspphot"

    missing_distance = absolute_magnitude_from_catalog(
        CatalogSource("g2", 10.0, 20.0, magnitude=13.56)
    )
    assert missing_distance.value is None
    assert missing_distance.status == "NO_PARALLAX"


@pytest.mark.parametrize(
    ("extinction_band", "extinction_system", "expected_status"),
    (
        ("V", "Johnson", "EXTINCTION_BAND_MISMATCH"),
        ("A0(541.4 nm)", "monochromatic", "EXTINCTION_BAND_MISMATCH"),
        ("unknown", "unknown", "EXTINCTION_SEMANTICS_REQUIRED"),
    ),
)
def test_catalog_absolute_magnitude_rejects_mismatched_or_unknown_extinction_semantics(
    extinction_band: str,
    extinction_system: str,
    expected_status: str,
) -> None:
    source = _catalog_source_with_forced_extinction_semantics(
        extinction_band=extinction_band,
        extinction_system=extinction_system,
        extinction_source="test-extinction",
    )

    estimate = absolute_magnitude_from_catalog(
        source,
        required_photometric_system="Gaia Vega",
        required_photometric_band="G",
    )

    assert estimate.value is None
    assert estimate.status == expected_status
    assert expected_status in estimate.flags
    assert estimate.extinction_band == extinction_band
    assert estimate.extinction_system == extinction_system
    assert estimate.extinction_source == "test-extinction"


def test_catalog_extinction_gate_also_applies_to_legacy_no_argument_path() -> None:
    source = _catalog_source_with_forced_extinction_semantics(
        extinction_band="unknown",
        extinction_system="unknown",
        extinction_source="legacy-unlabelled-column",
    )

    # A structured catalog result must not become strict merely because the
    # caller used the historical no-argument form.
    legacy = absolute_magnitude_from_catalog(source)

    assert legacy.status == "EXTINCTION_SEMANTICS_REQUIRED"
    assert legacy.value is None
    assert legacy.extinction_band == "unknown"
    assert legacy.extinction_system == "unknown"
    assert legacy.extinction_source == "legacy-unlabelled-column"


def test_catalog_absolute_magnitude_uses_declared_model_distance_when_parallax_is_missing() -> None:
    source = CatalogSource(
        "gsp-1",
        10.0,
        20.0,
        magnitude=13.56,
        magnitude_error=0.04,
        photometric_system="Gaia Vega",
        photometric_band="G",
        distance_pc=100.0,
        distance_lower_pc=95.0,
        distance_upper_pc=106.0,
        distance_source="Gaia DR3 GSP-Phot",
        extinction_mag=0.2,
        extinction_error_mag=0.03,
        extinction_band="G",
        extinction_system="Gaia",
        extinction_source="Gaia DR3 GSP-Phot: ag_gspphot",
    )

    estimate = absolute_magnitude_from_catalog(source)

    assert estimate.status == "VALID_MODEL_DISTANCE"
    assert estimate.value == pytest.approx(8.36)
    assert estimate.distance_source == "Gaia DR3 GSP-Phot"
    assert estimate.distance_lower_pc == pytest.approx(95.0)
    assert estimate.distance_upper_pc == pytest.approx(106.0)
    assert "DISTANCE_MODEL_USED" in estimate.flags
    assert "DISTANCE_INTERVAL_USED" in estimate.flags
    assert estimate.error is not None and estimate.error > 0


def test_low_snr_parallax_falls_back_to_declared_model_distance() -> None:
    source = CatalogSource(
        "gsp-2",
        10.0,
        20.0,
        magnitude=13.56,
        photometric_system="Gaia Vega",
        photometric_band="G",
        parallax_mas=10.0,
        parallax_error_mas=3.0,
        distance_pc=100.0,
        distance_lower_pc=90.0,
        distance_upper_pc=115.0,
        distance_source="Gaia DR3 GSP-Phot",
        extinction_mag=0.2,
        extinction_band="G",
        extinction_system="Gaia",
        extinction_source="Gaia DR3 GSP-Phot: ag_gspphot",
    )

    estimate = absolute_magnitude_from_catalog(source)

    assert estimate.status == "VALID_MODEL_DISTANCE"
    assert estimate.value == pytest.approx(8.36)
    assert "PARALLAX_QUALITY_FALLBACK_TO_MODEL_DISTANCE" in estimate.flags
    assert estimate.is_strict is True
    assert estimate.is_strict_with_uncertainty is False


def test_model_distance_requires_explicit_provenance() -> None:
    source = CatalogSource(
        "gsp-3",
        10.0,
        20.0,
        magnitude=13.56,
        photometric_system="Gaia Vega",
        photometric_band="G",
        distance_pc=100.0,
        extinction_band="G",
        extinction_system="Gaia",
        extinction_source="Gaia DR3 GSP-Phot: ag_gspphot",
        extinction_mag=0.2,
    )

    estimate = absolute_magnitude_from_catalog(source)

    assert estimate.value is None
    assert estimate.status == "MODEL_DISTANCE_SOURCE_REQUIRED"
    assert "DISTANCE_SOURCE_REQUIRED" in estimate.flags


def test_model_distance_without_extinction_keeps_distance_but_not_numeric_absolute_magnitude() -> None:
    estimate = absolute_magnitude_estimate_from_distance(
        13.56,
        distance_pc=100.0,
        distance_lower_pc=95.0,
        distance_upper_pc=106.0,
        distance_source="Gaia DR3 GSP-Phot",
    )

    assert estimate.value is None
    assert estimate.status == "MODEL_DISTANCE_NO_EXTINCTION"
    assert estimate.distance_pc == pytest.approx(100.0)
    assert "EXTINCTION_NOT_PROVIDED" in estimate.flags


def test_model_distance_without_complete_interval_is_not_strict_absolute_magnitude() -> None:
    estimate = absolute_magnitude_estimate_from_distance(
        13.56,
        distance_pc=100.0,
        distance_source="Gaia DR3 GSP-Phot",
        extinction_mag=0.2,
        extinction_error_mag=0.03,
        extinction_band="G",
        extinction_system="Gaia",
    )

    assert estimate.value is None
    assert estimate.status == "MODEL_DISTANCE_INTERVAL_REQUIRED"
    assert estimate.distance_pc == pytest.approx(100.0)
    assert estimate.distance_interval_status == "REQUIRED"
    assert "DISTANCE_ERROR_NOT_PROVIDED" in estimate.flags
    assert "DISTANCE_INTERVAL_REQUIRED" in estimate.flags
    assert estimate.is_strict is False


def test_model_distance_one_sided_interval_is_diagnostic_only() -> None:
    estimate = absolute_magnitude_estimate_from_distance(
        13.56,
        distance_pc=100.0,
        distance_lower_pc=95.0,
        distance_source="Gaia DR3 GSP-Phot",
        extinction_mag=0.2,
        extinction_band="G",
        extinction_system="Gaia",
    )

    assert estimate.value is None
    assert estimate.status == "MODEL_DISTANCE_INTERVAL_REQUIRED"
    assert estimate.distance_lower_pc == pytest.approx(95.0)
    assert estimate.distance_upper_pc is None
    assert estimate.distance_interval_status == "ONE_SIDED"
    assert "DISTANCE_INTERVAL_ONE_SIDED" in estimate.flags
    assert "DISTANCE_INTERVAL_REQUIRED" in estimate.flags


def test_model_distance_requires_source_provenance_for_strict_result() -> None:
    estimate = absolute_magnitude_estimate_from_distance(
        13.56,
        distance_pc=100.0,
        distance_lower_pc=95.0,
        distance_upper_pc=106.0,
        extinction_mag=0.2,
        extinction_band="G",
        extinction_system="Gaia",
    )

    assert estimate.value is None
    assert estimate.status == "MODEL_DISTANCE_SOURCE_REQUIRED"
    assert estimate.distance_interval_status == "PROVIDED"
    assert "DISTANCE_SOURCE_REQUIRED" in estimate.flags


def test_catalog_absolute_magnitude_propagates_catalog_and_distance_uncertainty() -> None:
    source = CatalogSource(
        "g-error",
        10.0,
        20.0,
        magnitude=13.56,
        magnitude_error=0.04,
        photometric_system="Gaia Vega",
        photometric_band="G",
        parallax_mas=10.0,
        parallax_error_mas=0.1,
        extinction_mag=0.2,
        extinction_error_mag=0.03,
        extinction_band="G",
        extinction_system="Gaia",
        extinction_source="Gaia DR3 GSP-Phot: ag_gspphot",
    )

    estimate = absolute_magnitude_from_catalog(
        source,
        required_photometric_system="Gaia Vega",
        required_photometric_band="G",
    )

    expected_error = math.sqrt(
        0.04**2
        + (5.0 / math.log(10.0) * 0.1 / 10.0) ** 2
        + 0.03**2
    )
    assert estimate.status == "VALID"
    assert estimate.error == pytest.approx(expected_error)
    assert estimate.error_status == "AVAILABLE"
    assert estimate.distance_interval_status == "DERIVED_FROM_PARALLAX_ERROR"
    assert estimate.as_dict()["distance_error_lower_pc"] == pytest.approx(
        100.0 - 1000.0 / 10.1
    )
    assert estimate.as_dict()["distance_error_upper_pc"] == pytest.approx(
        1000.0 / 9.9 - 100.0
    )


def test_build_source_photometry_keeps_instrumental_calibrated_and_absolute_layers() -> None:
    detection = _source(0, 10.0, snr=20.0)
    instrumental = instrumental_magnitude(detection.flux)
    assert instrumental is not None
    catalog = (
        CatalogSource(
            "g0",
            10.0,
            20.0,
            magnitude=instrumental + 20.0 + 0.3 * 0.5,
            magnitude_error=0.02,
            color=0.5,
            color_name="BP-RP",
            catalog_name="Gaia DR3",
            photometric_system="Gaia Vega",
            photometric_band="G",
            magnitude_source="Gaia DR3 phot_g_mean_mag",
            parallax_mas=10.0,
            parallax_error_mas=0.1,
            extinction_mag=0.2,
            mg_gspphot=4.2,
            mg_gspphot_lower=4.0,
            mg_gspphot_upper=4.4,
            mg_gspphot_source="Gaia DR3 GSP-Phot: mg_gspphot",
            extinction_band="G",
            extinction_system="Gaia",
            extinction_source="Gaia DR3 GSP-Phot: ag_gspphot",
        ),
    )
    match = CatalogMatch(0, "g0", detection.x, detection.y, detection.x, detection.y, 0.1, catalog[0].magnitude)
    # A single reference is deliberately insufficient for a real fit;
    # construct a known-valid calibration to test the result-layer wiring.
    calibration = PhotometricCalibration(
        photometric_system="Gaia Vega",
        photometric_band="G",
        color_name="BP-RP",
        color_order=1,
        coefficients=(20.0, 0.3),
        calibrator_count=8,
        inlier_count=8,
        validation_count=2,
        fit_rms_mag=0.01,
        validation_rms_mag=0.02,
        residual_mad_mag=0.01,
        color_min=0.2,
        color_max=1.5,
        status="VALID",
    )
    rows = build_source_photometry(
        (detection,),
        matches=(match,),
        catalog=catalog,
        photometric_calibration=calibration,
    )
    assert len(rows) == 1
    assert rows[0].status == "CALIBRATED"
    assert rows[0].instrumental_magnitude == pytest.approx(instrumental)
    assert rows[0].calibrated_magnitude == pytest.approx(instrumental + 20.15)
    assert rows[0].catalog_magnitude_error == pytest.approx(0.02)
    assert rows[0].magnitude_source == "Gaia DR3 phot_g_mean_mag"
    assert rows[0].catalog_mg_gspphot == pytest.approx(4.2)
    assert rows[0].catalog_mg_gspphot_lower == pytest.approx(4.0)
    assert rows[0].catalog_mg_gspphot_upper == pytest.approx(4.4)
    assert rows[0].catalog_mg_gspphot_source == "Gaia DR3 GSP-Phot: mg_gspphot"
    assert rows[0].absolute_magnitude is not None
    assert rows[0].absolute_magnitude.status == "VALID"
