from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from rst19.innovation import _difference_rows, _fit_constant_velocity, _motion_audit_rows, _motion_point_rows, _motion_rows, _telemetry_consistency, _telemetry_position_chart, _telemetry_prediction, _vector_metrics


def test_constant_velocity_fit_reports_small_sample_uncertainty() -> None:
    fit = _fit_constant_velocity(
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.1, 1.9, 3.2],
        [0.0, 2.1, 3.9, 6.2],
        2.0,
    )

    assert fit["degrees_of_freedom"] == 2
    assert fit["t95_critical"] == pytest.approx(4.303)
    assert fit["speed_ci95_low_px_per_s"] < fit["speed_px_per_s"] < fit["speed_ci95_high_px_per_s"]
    assert fit["direction_ci95_half_width_deg"] > 0
    assert fit["predicted_x_ci95_half_width_px"] > 0
    assert fit["predicted_y_ci95_half_width_px"] > 0
    assert fit["forecast_elapsed_s"] == pytest.approx(5.0)


def test_constant_velocity_fit_does_not_invent_uncertainty_from_two_points() -> None:
    fit = _fit_constant_velocity([0.0, 2.0], [1.0, 5.0], [3.0, 3.0], 1.0)

    assert fit["degrees_of_freedom"] == 0
    assert fit["speed_px_per_s"] == pytest.approx(2.0)
    assert fit["predicted_x_px"] == pytest.approx(7.0)
    assert fit["speed_ci95_low_px_per_s"] is None
    assert fit["direction_ci95_half_width_deg"] is None
    assert fit["predicted_x_ci95_half_width_px"] is None


def test_motion_rows_uses_observation_time_for_pixel_speed() -> None:
    frame_rows = [
        {"timestamp": "2026-03-30T16:32:05.000", "frame_index": 0},
        {"timestamp": "2026-03-30T16:32:06.500", "frame_index": 1},
    ]
    payload = {
        "motion_features": [
            {
                "track_id": 3,
                "classification": "moving",
                "points": [
                    {"frame_index": 0, "aligned_x": 10.0, "aligned_y": 20.0, "length_px": 40.0, "width_px": 3.0, "residual_snr": 20.0},
                    {"frame_index": 1, "aligned_x": 25.0, "aligned_y": 20.0, "length_px": 41.0, "width_px": 3.0, "residual_snr": 22.0},
                ],
            }
        ]
    }

    rows = _motion_rows(payload, frame_rows)

    assert rows[0]["duration_s"] == pytest.approx(1.5)
    assert rows[0]["speed_px_per_s"] == pytest.approx(10.0)
    assert rows[0]["displacement_px"] == pytest.approx(15.0)
    assert rows[0]["direction_deg_image"] == pytest.approx(0.0)
    assert rows[0]["points"][0]["fit_residual_px"] == pytest.approx(0.0)
    assert rows[0]["points"][1]["fit_residual_px"] == pytest.approx(0.0)


def test_motion_rows_keeps_single_frame_candidate_without_fake_speed() -> None:
    rows = _motion_rows(
        {"motion_features": [{"track_id": 1, "classification": "candidate", "points": [{"frame_index": 0, "x": 2.0, "y": 3.0, "angle_deg": 90.0, "length_px": 30.0, "width_px": 2.0, "residual_snr": 10.0}]}]},
        [{"timestamp": "2026-03-30T16:32:05.000", "frame_index": 0}],
    )

    assert rows[0]["duration_s"] == pytest.approx(0.0)
    assert rows[0]["speed_px_per_s"] is None
    assert rows[0]["direction_deg_image"] == pytest.approx(90.0)


def test_motion_point_rows_exposes_one_based_frame_number() -> None:
    rows = _motion_point_rows(
        [
            {
                "track_id": 7,
                "classification": "candidate",
                "points": [{"frame_index": 0, "x": 12.0, "y": 8.0}],
            }
        ]
    )

    assert rows[0]["frame_index"] == 0
    assert rows[0]["frame_number"] == 1


def test_motion_audit_rows_exposes_one_based_frame_number_and_filter_counts() -> None:
    rows = _motion_audit_rows(
        {
            "motion_frame_audits": [
                {
                    "frame_index": 8,
                    "component_count": 12,
                    "area_pass_count": 4,
                    "geometry_pass_count": 2,
                    "edge_rejected_count": 1,
                    "feature_count": 1,
                }
            ]
        }
    )

    assert rows[0]["frame_index"] == 8
    assert rows[0]["frame_number"] == 9
    assert rows[0]["geometry_pass_count"] == 2
    assert rows[0]["feature_count"] == 1


def test_difference_rows_preserves_raw_adu_units() -> None:
    rows = _difference_rows(
        [{"frame_index": 0, "timestamp": "2026-03-30T16:32:05.000", "background_rms_adu": 2.0}],
        [np.array([[10.0, 14.0], [8.0, np.nan]])],
        np.full((2, 2), 10.0),
    )

    assert rows[0]["residual_median_adu"] == pytest.approx(0.0)
    assert rows[0]["residual_rms_adu"] == pytest.approx(2.9652)
    assert rows[0]["residual_p99_abs_adu"] == pytest.approx(3.96, abs=0.01)
    assert rows[0]["positive_peak_sigma"] == pytest.approx(2.0)
    assert rows[0]["negative_peak_sigma"] == pytest.approx(-1.0)
    assert rows[0]["pixels_abs_gt_5sigma"] == 0
    assert rows[0]["valid_fraction"] == pytest.approx(0.75)


def test_vector_metrics_reports_position_and_velocity_directions() -> None:
    metrics = _vector_metrics(
        {
            "j2000_x": 3.0,
            "j2000_y": 4.0,
            "j2000_z": 0.0,
            "j2000_xv": 0.0,
            "j2000_yv": 1.0,
            "j2000_zv": 0.0,
        }
    )

    assert metrics["j2000_position_norm_raw"] == pytest.approx(5.0)
    assert metrics["j2000_position_azimuth_deg"] == pytest.approx(53.1301024)
    assert metrics["j2000_position_elevation_deg"] == pytest.approx(0.0)
    assert metrics["j2000_velocity_norm_raw"] == pytest.approx(1.0)
    assert metrics["j2000_velocity_azimuth_deg"] == pytest.approx(90.0)
    assert metrics["j2000_velocity_elevation_deg"] == pytest.approx(0.0)


def test_telemetry_consistency_uses_central_difference_for_interior_frames() -> None:
    rows = [
        {"timestamp": "2026-03-30T16:32:05.000", "j2000_x": 0.0, "j2000_y": 0.0, "j2000_z": 0.0, "j2000_xv": 10.0, "j2000_yv": 0.0, "j2000_zv": 0.0},
        {"timestamp": "2026-03-30T16:32:06.000", "j2000_x": 10.0, "j2000_y": 0.0, "j2000_z": 0.0, "j2000_xv": 10.0, "j2000_yv": 0.0, "j2000_zv": 0.0},
        {"timestamp": "2026-03-30T16:32:07.000", "j2000_x": 20.0, "j2000_y": 0.0, "j2000_z": 0.0, "j2000_xv": 10.0, "j2000_yv": 0.0, "j2000_zv": 0.0},
    ]

    summary = _telemetry_consistency(rows, "j2000")

    assert summary["valid_sample_count"] == 3
    assert summary["interior_sample_count"] == 1
    assert summary["position_rate_norm_m_per_s_median"] == pytest.approx(10.0)
    assert summary["interior_relative_error_assuming_m_per_s_median"] == pytest.approx(0.0)
    assert rows[1]["j2000_position_rate_norm_m_per_s"] == pytest.approx(10.0)
    assert rows[1]["j2000_velocity_consistency_relative_error_assuming_m_per_s"] == pytest.approx(0.0)


def test_telemetry_prediction_uses_last_state_and_declares_unit_assumption() -> None:
    rows = [
        {
            "frame_index": 0,
            "j2000_x": 100.0,
            "j2000_y": 200.0,
            "j2000_z": 300.0,
            "j2000_xv": 2.0,
            "j2000_yv": -1.0,
            "j2000_zv": 0.5,
        },
        {
            "frame_index": 1,
            "j2000_x": 110.0,
            "j2000_y": 195.0,
            "j2000_z": 302.5,
            "j2000_xv": 2.0,
            "j2000_yv": -1.0,
            "j2000_zv": 0.5,
        },
    ]

    prediction = _telemetry_prediction(rows, "j2000", 5.0)

    assert prediction["status"] == "available"
    assert prediction["frame_index"] == 1
    assert prediction["predicted_position_m_assuming_m_per_s"] == pytest.approx([120.0, 190.0, 305.0])
    assert prediction["unit_status"] == "速度单位待主办方确认"


def test_telemetry_prediction_is_unavailable_without_valid_last_state() -> None:
    prediction = _telemetry_prediction([{"j2000_x": 1.0}], "j2000", 5.0)

    assert prediction["status"] == "unavailable"
    assert prediction["predicted_position_m_assuming_m_per_s"] is None


def test_telemetry_position_chart_writes_observation_and_forecast_panels(tmp_path) -> None:
    rows = [
        {
            "j2000_x": 100.0,
            "j2000_y": 200.0,
            "wgs84_x": 500.0,
            "wgs84_y": 800.0,
        },
        {
            "j2000_x": 110.0,
            "j2000_y": 195.0,
            "wgs84_x": 510.0,
            "wgs84_y": 790.0,
        },
    ]
    predictions = {
        "j2000": {"predicted_position_m_assuming_m_per_s": [120.0, 190.0, 305.0]},
        "wgs84": {"predicted_position_m_assuming_m_per_s": [520.0, 780.0, 905.0]},
    }

    output = tmp_path / "telemetry_position.png"
    _telemetry_position_chart(output, rows, predictions)

    assert output.is_file()
    with Image.open(output) as image:
        assert image.size == (1200, 680)
