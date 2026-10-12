"""Freeze Python geometry-only candidate-consensus reference scenarios."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import SourceTrack, TrackPoint, _candidate_consensus_tracks


def static_track(x: float, y: float, frame_indices: tuple[int, ...], *, track_id: int = 0) -> SourceTrack:
    points = tuple(
        TrackPoint(index, track_id + 100 + index, x, y, x, y, 20.0)
        for index in frame_indices
    )
    return SourceTrack(track_id, "static", points, 0.0, 0.0, 0.0)


def capture(label: str, candidates, shifts, options, quality_tracks=(), frame_times=None):
    audit = {}
    tracks = _candidate_consensus_tracks(
        candidates,
        shifts,
        quality_tracks,
        frame_times=frame_times,
        audit_sink=audit,
        **options,
    )
    return {
        "label": label,
        "candidate_peak_frames": candidates,
        "cumulative_shifts": [list(shift) for shift in shifts],
        "frame_times": frame_times,
        "options": options,
        "quality_tracks": [track.as_dict() for track in quality_tracks],
        "expected": {"tracks": [track.as_dict() for track in tracks], "audit": audit},
    }


def main() -> None:
    base = {
        "link_radius_px": 2.0,
        "min_presence": 4,
        "persistent_min_presence": 2,
        "motion_min_displacement_px": 2.0,
        "max_motion_fit_rms_px": 0.75,
        "min_candidate_snr": 15.0,
    }
    cases = []
    candidates = [
        [[20, 20, 18.0], [60, 60, 22.0]],
        [[20, 20, 17.0], [60, 60, 21.0]],
        [[20, 20, 16.0], [60, 60, 20.0]],
        [[20, 20, 18.0], [60, 60, 23.0]],
    ]
    cases.append(
        capture(
            "repeated_candidates_deduplicate_persistent_quality_track",
            candidates,
            [(0.0, 0.0)] * 4,
            base,
            [static_track(20.0, 20.0, (0, 1, 2, 3))],
        )
    )

    later = [[], [[30.0, 30.0, 8.0, 1.0, 20.0]], [[30.0, 30.0, 8.0, 1.0, 20.0]], [[30.0, 30.0, 8.0, 1.0, 20.0]]]
    cases.append(
        capture(
            "temporal_candidate_can_seed_after_missing_first_frame",
            later,
            [(0.0, 0.0)] * 4,
            {**base, "temporal_candidate_min_snr": 7.5},
        )
    )

    temporal_low = [[[30.0, 30.0, 8.0, 1.0, 12.5]] for _ in range(4)]
    cases.append(
        capture(
            "temporal_reference_gate_is_independent",
            temporal_low,
            [(0.0, 0.0)] * 4,
            {**base, "temporal_candidate_min_snr": 7.5, "temporal_reference_min_snr": 12.0},
        )
    )

    normal_low = [[[30.0, 30.0, 8.0]] for _ in range(4)]
    cases.append(
        capture(
            "short_quality_track_does_not_suppress_repeated_candidate",
            normal_low,
            [(0.0, 0.0)] * 4,
            {**base, "temporal_candidate_min_snr": 7.5},
            [static_track(30.0, 30.0, (0,), track_id=7)],
        )
    )

    below_temporal_gate = [[[30.0, 30.0, 6.0, 1.0, 20.0]] for _ in range(4)]
    cases.append(
        capture(
            "temporal_response_gate_rejects_low_filter_snr",
            below_temporal_gate,
            [(0.0, 0.0)] * 4,
            {**base, "temporal_candidate_min_snr": 7.5},
        )
    )

    moving = [
        [[10.0 + 10.0 * index, 10.0, 20.0]]
        for index in range(4)
    ]
    cases.append(
        capture(
            "moving_candidate_is_rejected_by_motion_fit",
            moving,
            [(0.0, 0.0)] * 4,
            {**base, "link_radius_px": 100.0, "motion_min_displacement_px": 2.0},
        )
    )

    payload = {
        "source_hashes": {
            "sequence.py": hashlib.sha256((ROOT / "src/rst19/sequence.py").read_bytes()).hexdigest()
        },
        "cases": cases,
    }
    destination = ROOT / "backend/tests/fixtures/sequence_consensus.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination} with {len(cases)} cases")


if __name__ == "__main__":
    main()
