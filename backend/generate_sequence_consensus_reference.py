"""Freeze Python candidate-consensus geometry and image-gate scenarios."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.sequence import SourceTrack, TrackPoint, _candidate_consensus_tracks


def static_track(x: float, y: float, frame_indices: tuple[int, ...], *, track_id: int = 0) -> SourceTrack:
    points = tuple(
        TrackPoint(index, track_id + 100 + index, x, y, x, y, 20.0)
        for index in frame_indices
    )
    return SourceTrack(track_id, "static", points, 0.0, 0.0, 0.0)


def capture(
    label: str,
    candidates,
    shifts,
    options,
    quality_tracks=(),
    frame_times=None,
    *,
    images=None,
    backgrounds=None,
    noises=None,
    parameters=None,
    line_artifact_frames=None,
):
    audit = {}
    frame_analyses = None
    image_support = None
    if images is not None:
        if backgrounds is None:
            backgrounds = [100.0] * len(images)
        if noises is None:
            noises = [4.0] * len(images)
        if parameters is None:
            parameters = [
                {
                    "aperture_radius": 4,
                    "min_psf_support_pixels": 3,
                    "psf_fwhm": 3.0,
                    "mask_zero_pixels": 0,
                    "saturation_level": -1.0,
                }
                for _ in images
            ]
        if line_artifact_frames is None:
            line_artifact_frames = [[] for _ in images]
        frame_analyses = [
            SimpleNamespace(
                frame=SimpleNamespace(data=np.asarray(image, dtype=np.float32)),
                detection=SimpleNamespace(
                    background=background,
                    noise=noise,
                    parameters=parameter,
                ),
            )
            for image, background, noise, parameter in zip(
                images, backgrounds, noises, parameters, strict=True
            )
        ]
        image_support = {
            "frames": [np.asarray(image, dtype=np.float32).tolist() for image in images],
            "backgrounds": backgrounds,
            "noises": noises,
            "parameters": parameters,
            "line_artifact_frames": line_artifact_frames,
        }
    tracks = _candidate_consensus_tracks(
        candidates,
        shifts,
        quality_tracks,
        frame_analyses=frame_analyses,
        frame_times=frame_times,
        line_artifact_frames=line_artifact_frames if frame_analyses is not None else None,
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
        "image_support": image_support,
        "expected": {"tracks": [track.as_dict() for track in tracks], "audit": audit},
    }


def point_frames(count: int = 4, *, center: tuple[int, int] = (12, 12), sigma: float = 1.25):
    yy, xx = np.mgrid[:24, :24]
    x, y = center
    return [
        (100.0 + 180.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))).astype(np.float32)
        for _ in range(count)
    ]


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

    gaussian_frames = point_frames()
    repeated_point = [[[12.0, 12.0, 20.0]] for _ in gaussian_frames]
    cases.append(
        capture(
            "pixel_support_accepts_repeated_gaussian_candidate",
            repeated_point,
            [(0.0, 0.0)] * len(gaussian_frames),
            base,
            frame_times=[0.0, 1.0, 2.0, 3.0],
            images=gaussian_frames,
        )
    )

    cases.append(
        capture(
            "line_artifact_overlap_rejects_repeated_candidate",
            repeated_point,
            [(0.0, 0.0)] * len(gaussian_frames),
            base,
            frame_times=[0.0, 1.0, 2.0, 3.0],
            images=gaussian_frames,
            line_artifact_frames=[[[12.0, 12.0]], [], [], []],
        )
    )

    spike_frames = []
    for _ in range(4):
        image = np.full((24, 24), 100.0, dtype=np.float32)
        image[12, 12] = 300.0
        spike_frames.append(image)
    cases.append(
        capture(
            "single_pixel_spike_fails_temporal_support_gate",
            repeated_point,
            [(0.0, 0.0)] * 4,
            base,
            frame_times=[0.0, 1.0, 2.0, 3.0],
            images=spike_frames,
        )
    )

    masked_frames = point_frames()
    masked_frames[0][12, 12] = 0.0
    masked_parameters = [
        {
            "aperture_radius": 4,
            "min_psf_support_pixels": 3,
            "psf_fwhm": 3.0,
            "mask_zero_pixels": 1,
            "saturation_level": -1.0,
        }
        for _ in masked_frames
    ]
    cases.append(
        capture(
            "invalid_center_pixel_fails_temporal_validity_gate",
            repeated_point,
            [(0.0, 0.0)] * 4,
            base,
            frame_times=[0.0, 1.0, 2.0, 3.0],
            images=masked_frames,
            parameters=masked_parameters,
        )
    )

    temporal_candidates = [[[12.0, 12.0, 8.0, 1.0, 20.0] for _ in range(1)] for _ in range(4)]
    cases.append(
        capture(
            "temporal_reference_candidate_passes_pixel_and_psf_gates",
            temporal_candidates,
            [(0.0, 0.0)] * 4,
            {**base, "temporal_candidate_min_snr": 7.5},
            frame_times=[0.0, 1.0, 2.0, 3.0],
            images=point_frames(),
        )
    )

    cases.append(
        capture(
            "temporal_reference_psf_correlation_gate_rejects_mismatch",
            temporal_candidates,
            [(0.0, 0.0)] * 4,
            {
                **base,
                "temporal_candidate_min_snr": 7.5,
                "temporal_min_psf_correlation": 1.0,
            },
            frame_times=[0.0, 1.0, 2.0, 3.0],
            images=point_frames(),
        )
    )

    yy, xx = np.mgrid[:24, :24]
    elongated_frames = [
        (
            100.0
            + 180.0
            * np.exp(-((xx - 12) ** 2 / (2.0 * 2.5**2) + (yy - 12) ** 2 / (2.0 * 0.8**2)))
        ).astype(np.float32)
        for _ in range(4)
    ]
    cases.append(
        capture(
            "extended_shape_fails_temporal_source_gate",
            repeated_point,
            [(0.0, 0.0)] * 4,
            base,
            frame_times=[0.0, 1.0, 2.0, 3.0],
            images=elongated_frames,
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
