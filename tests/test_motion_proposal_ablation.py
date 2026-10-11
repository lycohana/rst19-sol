import csv
from types import SimpleNamespace

import pytest

from rst19.motion_proposal_ablation import (
    _moving_track_count_covering_truth,
    run_motion_proposal_ablation,
)


def test_motion_proposal_ablation_pairs_the_same_noise_and_recovers_low_peak_trail(tmp_path) -> None:
    manifest = run_motion_proposal_ablation(tmp_path / "motion-proposal", seed=19019)
    with (tmp_path / "motion-proposal" / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_scene_version = {(row["scene"], row["version"]): row for row in rows}

    control_a = by_scene_version[("no_injection_control", "A 基础线提案")]
    control_b = by_scene_version[("no_injection_control", "B 多尺度运动线提案")]
    target_a = by_scene_version[("low_peak_moving_trail", "A 基础线提案")]
    target_b = by_scene_version[("low_peak_moving_trail", "B 多尺度运动线提案")]

    assert control_a["input_array_sha256"] == control_b["input_array_sha256"]
    assert control_a["moving_track_count"] == control_b["moving_track_count"] == "0"
    assert target_a["target_recovered_frames"] == "0/4"
    assert target_b["target_recovered_frames"] == "4/4"
    assert target_b["target_recovery"] == "True"
    assert float(target_b["median_position_error_px"]) <= 1.0
    assert manifest["input"]["sha256"]
    assert (tmp_path / "motion-proposal" / "frame_audit.csv").is_file()
    assert manifest["truth"]["line_kernel_coefficient_adu"] == 4.0
    assert manifest["truth"]["sampled_peak_pixel_increment_adu_by_frame"] == pytest.approx(
        [12.77] * 4, abs=0.02
    )


def test_complete_truth_track_count_ignores_non_moving_tracks() -> None:
    truth = [(10.0 + frame, 20.0) for frame in range(4)]
    points = [
        SimpleNamespace(frame_index=frame, x=x, y=y)
        for frame, (x, y) in enumerate(truth)
    ]
    moving = SimpleNamespace(classification="moving", points=points)
    stationary = SimpleNamespace(classification="stationary", points=points)

    assert _moving_track_count_covering_truth([moving, stationary], truth, 1.0) == 1
