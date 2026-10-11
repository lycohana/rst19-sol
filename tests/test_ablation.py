import json
from types import SimpleNamespace

import numpy as np
import pytest

from rst19.ablation import _match_one_to_one, _save_case_sheet, _wilson, write_ablation_html
from rst19.motion_ablation import _make_truth_paths


def test_ablation_matching_assigns_each_detection_once() -> None:
    truths = [
        {"target_id": "T1", "x_px": 10.0, "y_px": 10.0},
        {"target_id": "T2", "x_px": 11.0, "y_px": 10.0},
    ]
    detections = [SimpleNamespace(detection_id=7, x=10.4, y=10.0)]

    matches, nearby = _match_one_to_one(truths, detections, radius_px=3.0)

    assert len(matches) == 1
    assert matches["T1"].detection_id == 7
    assert "T2" not in matches
    assert nearby == {"T1": 1, "T2": 1}


def test_wilson_interval_handles_empty_and_all_hits() -> None:
    assert _wilson(0, 0) == (None, None)
    low, high = _wilson(8, 8)
    assert low is not None and 0.0 < low < 1.0
    assert high == 1.0


def test_case_sheet_accepts_csv_rows_with_blank_detection_coordinates(tmp_path) -> None:
    row = {
        "gaussian_center_amplitude_adu": 12.0,
        "target_id": "PSF-001",
        "x_px": 32.0,
        "y_px": 32.0,
        "gaussian_candidate_hit": False,
        "hybrid_candidate_hit": False,
        "quality_gate_hit": False,
        "gaussian_x_px": "",
        "gaussian_y_px": "",
        "hybrid_x_px": None,
        "hybrid_y_px": None,
    }
    output = tmp_path / "cases.png"

    _save_case_sheet({12.0: np.zeros((64, 64), dtype=np.float32)}, [row], output, 12.0)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_report_rejects_stale_point_ablation_schema(tmp_path) -> None:
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version 2"):
        write_ablation_html(run_dir, tmp_path / "report.html")


def test_motion_truth_paths_are_constant_velocity_in_observation_time() -> None:
    frame_times = np.asarray([0.0, 1.0, 2.5, 3.5, 5.0, 6.0, 7.5, 8.5, 10.0, 11.0, 12.5, 13.5, 15.0, 16.0, 17.5])
    targets = _make_truth_paths(
        4096,
        4096,
        len(frame_times),
        frame_times,
        [(0.0, 0.0)] * len(frame_times),
        [[] for _ in frame_times],
        seed=19019,
    )
    target = targets[0]
    positions = np.asarray(target["registered_positions"], dtype=np.float64)
    reference_steps = (frame_times - frame_times[0]) / np.median(np.diff(frame_times))
    velocity = np.asarray([
        target["vx_px_per_reference_interval"],
        target["vy_px_per_reference_interval"],
    ])

    assert np.allclose(positions, positions[0] + reference_steps[:, None] * velocity)
