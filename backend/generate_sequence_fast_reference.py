"""Freeze Python reference cases for supplemental quality-point motion tracks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.detection import Detection
from rst19.sequence import SourceTrack, TrackPoint, track_fast_point_movers


def source(detection_id: int, x: float, y: float, *, flux_snr: float = 25.0) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=x,
        y=y,
        peak=100.0,
        flux=500.0,
        background=10.0,
        noise=2.0,
        snr=20.0,
        fwhm=3.0,
        flags=(),
        flux_error=20.0,
        flux_snr=flux_snr,
        quality_passed=True,
    )


def static_track(detection_id: int, x: float, y: float, count: int) -> SourceTrack:
    points = tuple(
        TrackPoint(index, detection_id + index, x, y, x, y, 25.0)
        for index in range(count)
    )
    return SourceTrack(detection_id, "static", points, 0.0, 0.0, 0.0)


def capture(label: str, frames, shifts, options, existing_tracks=(), frame_times=None):
    result = track_fast_point_movers(
        frames,
        shifts,
        frame_times=frame_times,
        existing_tracks=existing_tracks,
        **options,
    )
    return {
        "label": label,
        "frames": [[item.as_dict() for item in frame] for frame in frames],
        "cumulative_shifts": [list(shift) for shift in shifts],
        "frame_times": frame_times,
        "options": options,
        "existing_tracks": [track.as_dict() for track in existing_tracks],
        "expected": {"tracks": [track.as_dict() for track in result]},
    }


def main() -> None:
    cases = []
    frames = []
    for index in range(15):
        frames.append(
            [
                source(1000 + index, 100.0, 100.0),
                source(2000 + index, 300.0, 300.0),
                source(3000 + index, 1795.0 + 8.0 * index, 919.0 - 6.0 * index),
            ]
        )
    options = {
        "link_radius_px": 4.0,
        "min_presence": 12,
        "motion_min_displacement_px": 2.0,
        "min_flux_snr": 7.0,
        "max_step_px": 20.0,
        "gate_px": 4.5,
        "max_fit_rms_px": 1.5,
    }
    cases.append(
        capture(
            "moving_quality_source_avoids_existing_static_tracks",
            frames,
            [(0.0, 0.0)] * len(frames),
            options,
            [static_track(1000, 100.0, 100.0, 15), static_track(2000, 300.0, 300.0, 15)],
        )
    )

    stationary = [
        [source(100 + frame, 50.0, 50.0)]
        for frame in range(5)
    ]
    cases.append(
        capture(
            "stationary_quality_sources_are_not_relabelled",
            stationary,
            [(0.0, 0.0)] * 5,
            {**options, "min_presence": 5},
        )
    )

    times = [0.0, 1.2, 2.7, 4.0, 5.4, 6.8, 8.2, 9.5]
    shifts = [(0.3 * index, -0.2 * index) for index in range(len(times))]
    irregular = []
    static_points = []
    for index, time_s in enumerate(times):
        rows = [source(5000 + index, 180.0 + shifts[index][0], 40.0 + shifts[index][1])]
        static_points.append(
            TrackPoint(
                index,
                5000 + index,
                180.0 + shifts[index][0],
                40.0 + shifts[index][1],
                180.0,
                40.0,
                25.0,
            )
        )
        if index != 3:
            rows.append(
                source(
                    6000 + index,
                    30.0 + 7.0 * time_s / 1.4 + shifts[index][0],
                    80.0 - 5.0 * time_s / 1.4 + shifts[index][1],
                )
            )
        irregular.append(rows)
    irregular_track = SourceTrack(5000, "static", tuple(static_points), 0.0, 0.0, 0.0)
    cases.append(
        capture(
            "irregular_elapsed_time_with_one_missing_frame",
            irregular,
            shifts,
            {
                **options,
                "min_presence": 6,
                "motion_min_displacement_px": 12.0,
                "max_step_px": 14.0,
                "gate_px": 4.5,
            },
            [irregular_track],
            times,
        )
    )

    payload = {
        "source_hashes": {
            name: hashlib.sha256((ROOT / "src/rst19" / name).read_bytes()).hexdigest()
            for name in ("sequence.py", "detection.py")
        },
        "cases": cases,
    }
    destination = ROOT / "backend/tests/fixtures/sequence_fast.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination} with {len(cases)} cases")


if __name__ == "__main__":
    main()
