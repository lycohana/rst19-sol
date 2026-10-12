"""Freeze coordinate-only fast-candidate association reference cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import _FastCandidateRecord, _associate_fast_candidate_records


def record(row_index: int, x: float, y: float, snr: float) -> _FastCandidateRecord:
    return _FastCandidateRecord(row_index, x, y, x, y, snr)


def case(label, frames, *, frame_times=None, options=None):
    options = options or {
        "link_radius_px": 4.0,
        "min_presence": 3,
        "motion_min_displacement_px": 2.0,
        "max_step_px": 20.0,
        "gate_px": 2.5,
        "max_fit_rms_px": 1.0,
    }
    shifts = [(0.0, 0.0)] * len(frames)
    expected = _associate_fast_candidate_records(
        frames,
        shifts,
        frame_times=frame_times,
        link_radius_px=options["link_radius_px"],
        min_presence=options["min_presence"],
        motion_min_displacement_px=options["motion_min_displacement_px"],
        max_step_px=options["max_step_px"],
        gate_px=options["gate_px"],
        max_fit_rms_px=options["max_fit_rms_px"],
    )
    return {
        "label": label,
        "request": {
            "frames": [
                [
                    {
                        "row_index": item.row_index,
                        "aligned_x": item.aligned_x,
                        "aligned_y": item.aligned_y,
                        "filter_snr": item.filter_snr,
                    }
                    for item in frame
                ]
                for frame in frames
            ],
            "frame_times": frame_times,
            "options": options,
        },
        "expected": {
            "tracks": [
                {
                    "observations": [
                        {"frame_index": frame_index, "row_index": item.row_index}
                        for frame_index, item in observations
                    ]
                }
                for observations in expected
            ]
        },
    }


def main() -> None:
    cases = []
    cases.append(
        case(
            "irregular_time_axis_linear_candidate",
            [
                [record(17, 100.0, 30.0, 18.0)],
                [record(25, 110.0, 30.0, 19.0)],
                [record(31, 130.0, 30.0, 20.0)],
                [record(42, 140.0, 30.0, 21.0)],
            ],
            frame_times=[0.0, 1.0, 3.0, 4.0],
        )
    )
    cases.append(
        case(
            "single_interior_frame_gap_is_bridged",
            [
                [record(1, 50.0, 40.0, 20.0)],
                [record(4, 60.0, 40.0, 22.0)],
                [record(8, 70.0, 40.0, 24.0)],
                [],
                [record(16, 90.0, 40.0, 26.0)],
            ],
            options={
                "link_radius_px": 4.0,
                "min_presence": 4,
                "motion_min_displacement_px": 2.0,
                "max_step_px": 20.0,
                "gate_px": 2.5,
                "max_fit_rms_px": 1.0,
            },
        )
    )
    cases.append(
        case(
            "two_consecutive_missing_frames_stop_following",
            [
                [record(2, 0.0, 10.0, 20.0)],
                [record(3, 10.0, 10.0, 20.0)],
                [record(5, 20.0, 10.0, 20.0)],
                [],
                [],
                [record(9, 50.0, 10.0, 20.0)],
            ],
            options={
                "link_radius_px": 4.0,
                "min_presence": 3,
                "motion_min_displacement_px": 2.0,
                "max_step_px": 20.0,
                "gate_px": 2.5,
                "max_fit_rms_px": 1.0,
            },
        )
    )
    cases.append(
        case(
            "separate_tracks_keep_deterministic_fit_priority",
            [
                [record(0, 20.0, 20.0, 25.0), record(1, 20.5, 20.0, 18.0)],
                [record(5, 30.0, 20.0, 25.0), record(6, 30.8, 20.0, 18.0)],
                [record(9, 40.0, 20.0, 25.0), record(10, 40.1, 20.0, 18.0)],
            ],
        )
    )
    cases.append(
        case(
            "invalid_time_metadata_falls_back_to_indices",
            [
                [record(4, 10.0, 5.0, 12.0)],
                [record(6, 20.0, 5.0, 12.0)],
                [record(8, 30.0, 5.0, 12.0)],
                [record(12, 40.0, 5.0, 12.0)],
            ],
            frame_times=[0.0, 1.0, 1.0, 3.0],
            options={
                "link_radius_px": 4.0,
                "min_presence": 3,
                "motion_min_displacement_px": 2.0,
                "max_step_px": 20.0,
                "gate_px": 2.5,
                "max_fit_rms_px": 1.0,
            },
        )
    )
    source = ROOT / "src/rst19/sequence.py"
    fixture = {
        "source_hashes": {"sequence.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": cases,
    }
    target = Path(__file__).parent / "tests/fixtures/sequence_fast_candidates.json"
    target.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
