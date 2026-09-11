from __future__ import annotations

import numpy as np
import pytest

from rst19.relative_photometry import Observation, fit_relative_photometry


def _synthetic_rows(
    *,
    frame_count: int = 15,
    source_count: int = 6,
    magnitude_error: float = 0.01,
    noise_seed: int | None = None,
) -> tuple[list[Observation], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(noise_seed)
    frame_zero_points = np.linspace(-0.18, 0.21, frame_count)
    frame_zero_points -= np.median(frame_zero_points)
    source_magnitudes = np.linspace(10.1, 14.6, source_count)
    rows: list[Observation] = []
    for frame_id in range(frame_count):
        for source_id in range(source_count):
            noise = float(rng.normal(0.0, magnitude_error * 0.35)) if noise_seed is not None else 0.0
            rows.append(
                Observation(
                    frame_id=frame_id,
                    source_id=f"s{source_id}",
                    instrumental_magnitude=(
                        source_magnitudes[source_id] + frame_zero_points[frame_id] + noise
                    ),
                    magnitude_error=magnitude_error,
                    x=100.0 + 13.0 * source_id + 0.15 * frame_id,
                    y=80.0 + 7.0 * source_id - 0.12 * frame_id,
                    quality_passed=True,
                    is_moving=False,
                )
            )
    return rows, frame_zero_points, source_magnitudes


def test_recovers_known_frame_offsets_and_relative_source_magnitudes() -> None:
    rows, true_zero_points, true_source_magnitudes = _synthetic_rows(noise_seed=4)

    result = fit_relative_photometry(
        rows,
        validation_fraction=0.2,
        random_state=12,
    )

    assert result.status == "VALID"
    assert np.median(list(result.frame_zero_points.values())) == pytest.approx(0.0, abs=1e-12)
    assert result.reference_sample_count == len(rows)
    assert result.training_residual_rms is not None and result.training_residual_rms < 0.01
    assert result.training_residual_mad is not None and result.training_residual_mad < 0.01

    # The synthetic z values already use the requested gauge, so the recovered
    # source values can be compared directly to the generating values.
    for frame_id, expected in enumerate(true_zero_points):
        assert result.frame_zero_points[frame_id] == pytest.approx(expected, abs=0.015)
    for source_id, expected in enumerate(true_source_magnitudes):
        assert result.relative_magnitudes[f"s{source_id}"] == pytest.approx(expected, abs=0.015)


def test_moving_and_quality_failed_rows_never_enter_reference_fit() -> None:
    rows, _, _ = _synthetic_rows(frame_count=6, source_count=4)
    rows.extend(
        [
            Observation(
                frame_id=frame_id,
                source_id="moving",
                instrumental_magnitude=3.0 + frame_id,
                magnitude_error=0.01,
                x=20.0 + frame_id,
                y=30.0,
                quality_passed=True,
                is_moving=True,
            )
            for frame_id in range(6)
        ]
    )
    rows.append(
        Observation(
            frame_id=0,
            source_id="bad-quality",
            instrumental_magnitude=1.0,
            magnitude_error=0.01,
            x=20.0,
            y=30.0,
            quality_passed=False,
            is_moving=False,
        )
    )
    rows.append(
        Observation(
            frame_id=0,
            source_id="variable",
            instrumental_magnitude=2.0,
            magnitude_error=0.01,
            x=20.0,
            y=30.0,
            quality_passed=True,
            is_moving=False,
            is_variable_optional=True,
        )
    )

    result = fit_relative_photometry(rows, validation_fraction=0.0)

    assert result.status == "VALID"
    assert "MOVING_EXCLUDED" in result.flags
    assert "QUALITY_FAILED_EXCLUDED" in result.flags
    assert "VARIABLE_EXCLUDED" in result.flags
    assert "moving" not in result.relative_magnitudes
    assert "bad-quality" not in result.relative_magnitudes
    assert "variable" not in result.relative_magnitudes
    assert result.reference_sample_count == 6 * 4
    assert result.excluded_observation_count == 8


def test_source_block_holdout_reports_independent_validation_residual() -> None:
    rows, _, _ = _synthetic_rows(noise_seed=9)

    result = fit_relative_photometry(
        rows,
        validation_fraction=0.25,
        random_state=0,
    )

    assert result.status == "VALID"
    assert "HOLDOUT_SOURCE_BLOCK" in result.flags
    assert result.validation_source_ids
    assert result.validation_sample_count == len(result.validation_source_ids) * 15
    assert result.validation_residual_rms is not None
    assert result.validation_residual_rms < 0.01
    assert result.validation_residual_mad is not None
    assert result.validation_residual_mad < 0.01


def test_iterative_mad_rejects_a_bad_reference_row() -> None:
    rows, true_zero_points, _ = _synthetic_rows(frame_count=8, source_count=5)
    bad_index = 3 * 5 + 2
    bad_row = rows[bad_index]
    rows[bad_index] = Observation(
        frame_id=bad_row.frame_id,
        source_id=bad_row.source_id,
        instrumental_magnitude=bad_row.instrumental_magnitude + 1.5,
        magnitude_error=bad_row.magnitude_error,
        x=bad_row.x,
        y=bad_row.y,
        quality_passed=True,
        is_moving=False,
    )

    result = fit_relative_photometry(rows, validation_fraction=0.0, mad_threshold=4.0)

    assert result.status == "VALID"
    assert bad_index in result.rejected_observation_indices
    assert "MAD_REJECTION_APPLIED" in result.flags
    assert result.reference_sample_count == len(rows) - 1
    for frame_id, expected in enumerate(true_zero_points):
        assert result.frame_zero_points[frame_id] == pytest.approx(expected, abs=0.02)


def test_degenerate_spatial_design_returns_status_instead_of_raising() -> None:
    rows, _, _ = _synthetic_rows(frame_count=4, source_count=4)
    constant_position_rows = [
        Observation(
            frame_id=row.frame_id,
            source_id=row.source_id,
            instrumental_magnitude=row.instrumental_magnitude,
            magnitude_error=row.magnitude_error,
            x=100.0,
            y=200.0,
            quality_passed=True,
            is_moving=False,
        )
        for row in rows
    ]

    result = fit_relative_photometry(
        constant_position_rows,
        spatial_order=1,
        validation_fraction=0.0,
    )

    assert result.status == "DEGENERATE_DESIGN"
    assert "DEGENERATE_DESIGN" in result.flags
    assert result.frame_zero_points == {}
    assert result.relative_magnitudes == {}


def test_optional_spatial_terms_are_fitted_as_relative_nuisance_terms() -> None:
    rng = np.random.default_rng(33)
    frame_zero_points = np.linspace(-0.12, 0.16, 6)
    frame_zero_points -= np.median(frame_zero_points)
    source_magnitudes = np.linspace(10.0, 14.0, 5)
    coordinates = rng.normal(size=(30, 2))
    x_center, y_center = np.mean(coordinates, axis=0)
    x_scale, y_scale = np.std(coordinates, axis=0)
    rows: list[Observation] = []
    coordinate_index = 0
    for frame_id in range(6):
        for source_id in range(5):
            x, y = coordinates[coordinate_index]
            x_normalized = (x - x_center) / x_scale
            y_normalized = (y - y_center) / y_scale
            rows.append(
                Observation(
                    frame_id=frame_id,
                    source_id=f"s{source_id}",
                    instrumental_magnitude=(
                        source_magnitudes[source_id]
                        + frame_zero_points[frame_id]
                        + 0.08 * x_normalized
                        - 0.05 * y_normalized
                    ),
                    magnitude_error=0.01,
                    x=float(x),
                    y=float(y),
                    quality_passed=True,
                    is_moving=False,
                )
            )
            coordinate_index += 1

    result = fit_relative_photometry(rows, spatial_terms=True, validation_fraction=0.0)

    assert result.status == "VALID"
    assert result.spatial_order == 1
    assert len(result.spatial_coefficients) == 2
    assert "SPATIAL_TERMS_ENABLED" in result.flags
    assert result.training_residual_rms is not None and result.training_residual_rms < 1e-10


def test_uncertainties_are_reported_when_residual_degrees_of_freedom_exist() -> None:
    rows, _, _ = _synthetic_rows(noise_seed=21)

    result = fit_relative_photometry(rows, validation_fraction=0.0)

    assert result.status == "VALID"
    assert "UNCERTAINTY_ESTIMATED" in result.flags
    assert set(result.frame_zero_point_uncertainties) == set(result.frame_zero_points)
    assert set(result.relative_magnitude_uncertainties) == set(result.relative_magnitudes)
    assert all(
        uncertainty is not None and np.isfinite(uncertainty) and uncertainty >= 0
        for uncertainty in result.frame_zero_point_uncertainties.values()
    )
    assert all(
        uncertainty is not None and np.isfinite(uncertainty) and uncertainty >= 0
        for uncertainty in result.relative_magnitude_uncertainties.values()
    )


def test_result_as_dict_is_json_friendly_and_keeps_relative_scope_explicit() -> None:
    rows, _, _ = _synthetic_rows(frame_count=3, source_count=3)

    result = fit_relative_photometry(rows, validation_fraction=0.0)
    payload = result.as_dict()

    assert "relative_magnitudes" in payload
    assert "calibrated_magnitudes" not in payload
    assert all(isinstance(key, str) for key in payload["relative_magnitudes"])
