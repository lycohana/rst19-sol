from types import SimpleNamespace

import numpy as np

from rst19.ablation import _match_one_to_one, _save_case_sheet, _wilson


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
        "signal_peak_excess_adu": 12.0,
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
