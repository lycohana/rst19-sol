"""Compare Rust FITS-to-track orchestration with its Python constituent stages."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.detection import _working_mask, build_background_model  # noqa: E402
from rst19.fits import auxiliary_mask, read_fits  # noqa: E402
from rst19.pipeline import analyze_frame  # noqa: E402
from rst19.sequence import (  # noqa: E402
    _registered_median_reference,
    _registered_coadd_reference,
    _temporal_reference_candidate_frames,
    _fit_sequence_relative_photometry,
    _observation_times_seconds,
    _stack_forced_frame_measure_batch,
    _candidate_consensus_tracks,
    _track_fast_candidate_movers,
    audit_fixed_sentinel,
    audit_fixed_sentinel_impacts,
    track_detections,
    track_fast_point_movers,
)

FRAME_OPTIONS: dict[str, Any] = {
    "threshold_sigma": 4.0,
    "min_distance": 3,
    "max_sources": 6_000,
    "background_box_size": 256,
    "background_sample_limit": 4_096,
    "use_float32": True,
    "fast_sequence": True,
    "refine_local_background": False,
}


def card(name: str, value: str) -> bytes:
    return f"{name:<8}= {value:>20}".ljust(80).encode("ascii")


def make_fits(path: Path, image: np.ndarray, timestamp: str) -> None:
    height, width = image.shape
    header = b"".join(
        [
            card("SIMPLE", "T"),
            card("BITPIX", "16"),
            card("NAXIS", "2"),
            card("NAXIS1", str(width)),
            card("NAXIS2", str(height)),
            card("DATE-OBS", f"'{timestamp}'"),
            card("EXPTIME", "1.5"),
            b"END".ljust(80),
        ]
    ).ljust(2880, b" ")
    payload = header + np.asarray(image, dtype=">i2").tobytes()
    path.write_bytes(payload + bytes((-len(payload)) % 2880))


def synthetic_frames(directory: Path) -> list[Path]:
    rng = np.random.default_rng(1913)
    yy, xx = np.indices((128, 128), dtype=np.float64)
    paths = []
    static_stars = [(20, 20, 360), (43, 29, 270), (81, 22, 330), (30, 77, 310), (91, 86, 420)]
    for index in range(3):
        image = rng.normal(100.0, 2.0, size=(128, 128))
        for x, y, amplitude in static_stars:
            image += amplitude * np.exp(-0.5 * (((xx - x) / 1.25) ** 2 + ((yy - y) / 1.25) ** 2))
        image += 440 * np.exp(
            -0.5 * (((xx - (55 + index * 4)) / 1.25) ** 2 + ((yy - (57 + index * 2)) / 1.25) ** 2)
        )
        image[20, 22] = -1
        path = directory / f"frame-{index:02d}.fits"
        make_fits(path, np.rint(image).astype(np.int16), f"2026-07-03T12:00:{index * 2:02d}.000Z")
        paths.append(path)
    return paths


def synthetic_candidate_recovery_frames(directory: Path) -> tuple[list[Path], list[tuple[float, float]]]:
    rng = np.random.default_rng(77531)
    yy, xx = np.indices((128, 128), dtype=np.float64)
    static_stars = [(16, 19, 900), (108, 23, 850), (22, 105, 820), (105, 104, 780)]
    paths = []
    truth = []
    for index in range(3):
        image = rng.normal(100.0, 2.0, size=(128, 128))
        for x, y, amplitude in static_stars:
            image += amplitude * np.exp(
                -0.5 * (((xx - x) / 1.25) ** 2 + ((yy - y) / 1.25) ** 2)
            )
        x = 54 + index * 4
        y = 57 + index * 2
        truth.append((float(x), float(y)))
        image += 220 * np.exp(
            -0.5 * (((xx - x) / 1.25) ** 2 + ((yy - y) / 1.25) ** 2)
        )
        path = directory / f"candidate-frame-{index:02d}.fits"
        make_fits(path, np.rint(image).astype(np.int16), f"2026-07-03T12:01:{index * 2:02d}.000Z")
        paths.append(path)
    return paths, truth


def compare(actual: Any, expected: Any, path: str = "result") -> None:
    if is_dataclass(expected):
        expected = asdict(expected)
    if isinstance(expected, bool) or expected is None or isinstance(expected, (str, int)):
        if actual != expected:
            raise AssertionError(f"{path}: Rust returned {actual!r}; Python returned {expected!r}")
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isclose(
            actual, expected, rel_tol=5e-7, abs_tol=5e-7
        ):
            raise AssertionError(f"{path}: Rust returned {actual!r}; Python returned {expected!r}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list length or type differs")
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            compare(got, wanted, f"{path}[{index}]")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            if isinstance(actual, dict):
                detail = f"actual-only={sorted(actual.keys() - expected.keys())}; expected-only={sorted(expected.keys() - actual.keys())}"
            else:
                detail = f"actual type={type(actual).__name__}"
            raise AssertionError(f"{path}: object keys differ ({detail})")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    else:
        raise TypeError(f"unsupported baseline value at {path}: {type(expected)}")


def invoke(engine: Path, request: dict[str, Any]) -> dict[str, Any]:
    response = subprocess.run(
        [str(engine)],
        input=json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n",
        text=True,
        capture_output=True,
        encoding="utf-8",
        timeout=300,
    )
    events = [json.loads(line) for line in response.stdout.splitlines()]
    if response.returncode != 0 or len(events) != 1 or events[0].get("ok") is not True:
        raise RuntimeError(f"Rust engine failed: exit={response.returncode}; {response.stdout}; {response.stderr}")
    return events[0]["payload"]


def verify(engine: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    paths = synthetic_frames(out)
    frames = [read_fits(path) for path in paths]
    shared_background = build_background_model(
        frames[0].data,
        mask=auxiliary_mask(frames[0].data.shape),
        background_box_size=FRAME_OPTIONS["background_box_size"],
        background_sample_limit=FRAME_OPTIONS["background_sample_limit"],
        refine_local_background=FRAME_OPTIONS["refine_local_background"],
        use_float32=FRAME_OPTIONS["use_float32"],
    )
    python_analyses = [
        analyze_frame(path, background_model=shared_background, **FRAME_OPTIONS)
        for path in paths
    ]
    python_fixed_sentinel_audit = audit_fixed_sentinel(
        [analysis.frame.data for analysis in python_analyses]
    )
    python_fixed_sentinel_impact_audit = audit_fixed_sentinel_impacts(
        python_analyses,
        python_fixed_sentinel_audit,
    )
    frame_times = _observation_times_seconds(python_analyses).tolist()
    python_tracking_result = track_detections(
        [analysis.detection.quality_sources for analysis in python_analyses],
        frame_names=[str(path) for path in paths],
        frame_times=frame_times,
    )
    python_tracking = python_tracking_result.as_dict()
    effective_motion_gate = python_tracking_result.effective_motion_min_displacement_px or 2.0
    python_fast_point_tracks = track_fast_point_movers(
        [analysis.detection.quality_sources for analysis in python_analyses],
        python_tracking_result.cumulative_shifts,
        frame_times=frame_times,
        link_radius_px=python_tracking_result.link_radius_px,
        min_presence=python_tracking_result.min_presence,
        motion_min_displacement_px=effective_motion_gate,
        existing_tracks=python_tracking_result.tracks,
    )
    python_fast_point_rows = []
    for index, track in enumerate(python_fast_point_tracks):
        row = track.as_dict()
        row["track_id"] = len(python_tracking_result.tracks) + index
        python_fast_point_rows.append(row)
    request = {
        "operation": "sequence.analyze_core",
        "paths": [str(path) for path in paths],
        "frame_options": FRAME_OPTIONS,
    }
    actual = invoke(engine.resolve(), request)
    if actual["schema"] != "rst19.sequence.core.v1" or not actual["shared_background_model"]:
        raise AssertionError("Rust did not execute the sequence shared-background core")
    if len(actual["frame_analyses"]) != len(python_analyses):
        raise AssertionError("frame count differs")
    for index, (native, python) in enumerate(zip(actual["frame_analyses"], python_analyses)):
        compare(native, python.as_dict(), f"frame_analyses[{index}]")
    compare(
        actual["fixed_sentinel_audit"],
        json.loads(json.dumps(asdict(python_fixed_sentinel_audit), allow_nan=False)),
        "fixed_sentinel_audit",
    )
    compare(
        actual["fixed_sentinel_impact_audit"],
        json.loads(json.dumps(asdict(python_fixed_sentinel_impact_audit), allow_nan=False)),
        "fixed_sentinel_impact_audit",
    )
    for key in (
        "frame_count",
        "stable_source_count",
        "persistent_source_count",
        "moving_track_count",
        "transient_track_count",
        "min_presence",
        "registration_model",
        "registration_affine_frame_count",
        "effective_motion_min_displacement_px",
        "cumulative_shifts",
        "tracks",
    ):
        compare(actual["tracking"][key], python_tracking[key], f"tracking.{key}")
    compare(actual["fast_point_tracks"], python_fast_point_rows, "fast_point_tracks")
    compare(actual["tracking"]["fast_point_motion_count"], len(python_fast_point_rows), "tracking.fast_point_motion_count")
    compare(actual["observation_time_axis"]["frame_times"], frame_times, "observation_time_axis.frame_times")
    disabled = invoke(engine.resolve(), {**request, "fast_point_motion": False})
    compare(disabled["fast_point_tracks"], [], "disabled.fast_point_tracks")
    compare(disabled["tracking"]["fast_point_motion_count"], 0, "disabled.tracking.fast_point_motion_count")
    compare(disabled["fast_candidate_proposals"]["status"], "disabled", "disabled.fast_candidate_proposals.status")

    candidate_dir = out / "fast-candidate-recovery"
    candidate_dir.mkdir()
    candidate_paths, candidate_truth = synthetic_candidate_recovery_frames(candidate_dir)
    candidate_request = {
        "operation": "sequence.analyze_core",
        "paths": [str(path) for path in candidate_paths],
        "frame_options": {**FRAME_OPTIONS, "max_sources": 2},
        "relative_photometry": True,
        "relative_photometry_max_sources": 2,
        "relative_photometry_validation_fraction": 0.0,
    }
    candidate_result = invoke(engine.resolve(), candidate_request)
    candidate_stage = candidate_result["fast_candidate_proposals"]
    if candidate_stage["status"] != "proposals_available":
        raise AssertionError("Rust did not propose the truncated moving-source peak")
    if candidate_stage["evidence_level"] != "unverified_candidate_proposal":
        raise AssertionError("candidate peaks were promoted beyond proposal evidence")
    if candidate_stage["original_pixel_measurement_applied"] is not True:
        raise AssertionError("Rust did not remeasure candidate coordinates in original pixels")
    if candidate_stage["accepted_track_count"] < 1:
        raise AssertionError("Rust original-pixel gates did not admit the synthetic moving source")
    if any(frame["detection"]["returned_count"] >= frame["detection"]["candidate_count"]
           for frame in candidate_result["frame_analyses"]):
        raise AssertionError("candidate fixture did not exercise source-list truncation")
    matched_truth = False
    for proposal in candidate_stage["proposals"]:
        observations = proposal["observations"]
        if len(observations) != len(candidate_truth):
            continue
        if all(
            abs(observation["x"] - x) <= 1.5 and abs(observation["y"] - y) <= 1.5
            for observation, (x, y) in zip(observations, candidate_truth)
        ):
            matched_truth = True
            if proposal["verification_status"] != "original_pixel_quality_and_motion_gates_passed":
                raise AssertionError("synthetic moving source failed its original-pixel gates")
            break
    if not matched_truth:
        raise AssertionError("Rust candidate proposals did not recover the known moving peak")

    candidate_frame_options = {**FRAME_OPTIONS, "max_sources": 2}
    candidate_frames = [read_fits(path) for path in candidate_paths]
    candidate_background = build_background_model(
        candidate_frames[0].data,
        mask=auxiliary_mask(candidate_frames[0].data.shape),
        background_box_size=FRAME_OPTIONS["background_box_size"],
        background_sample_limit=FRAME_OPTIONS["background_sample_limit"],
        refine_local_background=FRAME_OPTIONS["refine_local_background"],
        use_float32=FRAME_OPTIONS["use_float32"],
    )
    candidate_analyses = [
        analyze_frame(path, background_model=candidate_background, **candidate_frame_options)
        for path in candidate_paths
    ]
    candidate_times = _observation_times_seconds(candidate_analyses).tolist()
    candidate_tracking_result = track_detections(
        [analysis.detection.quality_sources for analysis in candidate_analyses],
        frame_names=[str(path) for path in candidate_paths],
        frame_times=candidate_times,
    )
    python_relative_photometry = _fit_sequence_relative_photometry(
        candidate_analyses,
        candidate_tracking_result.tracks,
        min_presence=candidate_tracking_result.min_presence,
        max_sources=2,
        spatial_order=0,
        validation_fraction=0.0,
    )
    compare(
        candidate_result["relative_photometry"],
        json.loads(json.dumps(asdict(python_relative_photometry), allow_nan=False)),
        "candidate.relative_photometry",
    )
    candidate_quality_fast_tracks = track_fast_point_movers(
        [analysis.detection.quality_sources for analysis in candidate_analyses],
        candidate_tracking_result.cumulative_shifts,
        frame_times=candidate_times,
        link_radius_px=candidate_tracking_result.link_radius_px,
        min_presence=candidate_tracking_result.min_presence,
        motion_min_displacement_px=(
            candidate_tracking_result.effective_motion_min_displacement_px or 2.0
        ),
        existing_tracks=candidate_tracking_result.tracks,
    )
    candidate_python_tracks = _track_fast_candidate_movers(
        [analysis.detection.candidate_peaks for analysis in candidate_analyses],
        candidate_analyses,
        candidate_tracking_result.cumulative_shifts,
        frame_times=candidate_times,
        link_radius_px=candidate_tracking_result.link_radius_px,
        min_presence=candidate_tracking_result.min_presence,
        motion_min_displacement_px=(
            candidate_tracking_result.effective_motion_min_displacement_px or 2.0
        ),
        min_filter_snr=7.0,
        max_step_px=20.0,
        gate_px=4.5,
        max_fit_rms_px=1.5,
        candidate_limit=20_000,
        existing_tracks=candidate_tracking_result.tracks + candidate_quality_fast_tracks,
        aperture_radius=4,
        min_psf_support_pixels=3,
        min_fwhm=0.8,
        max_fwhm=12.0,
        max_ellipticity=0.65,
        line_artifact_frames=[
            analysis.detection.line_artifact_coordinates for analysis in candidate_analyses
        ],
    )
    candidate_python_track_rows = []
    for index, track in enumerate(candidate_quality_fast_tracks + candidate_python_tracks):
        row = track.as_dict()
        row["track_id"] = len(candidate_tracking_result.tracks) + index
        candidate_python_track_rows.append(row)
    compare(
        candidate_result["fast_point_tracks"],
        candidate_python_track_rows,
        "candidate.fast_point_tracks",
    )
    candidate_temporal_reference = _registered_median_reference(
        candidate_analyses,
        candidate_tracking_result.cumulative_shifts,
    )
    candidate_temporal_frames, candidate_temporal_count = _temporal_reference_candidate_frames(
        candidate_analyses,
        candidate_tracking_result.cumulative_shifts,
        candidate_temporal_reference,
        psf_fwhm=3.0,
        min_candidate_snr=15.0,
    )
    candidate_coadd_reference = _registered_coadd_reference(
        candidate_analyses,
        candidate_tracking_result.cumulative_shifts,
    )
    candidate_coadd_frames, candidate_coadd_count = _temporal_reference_candidate_frames(
        candidate_analyses,
        candidate_tracking_result.cumulative_shifts,
        candidate_coadd_reference,
        psf_fwhm=3.0,
        min_candidate_snr=15.0,
    )

    def merge_candidate_rows(temporal_frames: Any) -> list[np.ndarray]:
        merged = []
        for analysis, temporal_rows in zip(candidate_analyses, temporal_frames, strict=True):
            raw = np.asarray(analysis.detection.candidate_peaks, dtype=np.float32)
            if raw.size == 0:
                base_rows = np.empty((0, 3), dtype=np.float32)
            elif raw.ndim == 2 and raw.shape[1] >= 3:
                base_rows = raw[:, :3]
            else:
                base_rows = np.empty((0, 3), dtype=np.float32)
            if base_rows.size:
                base_rows = np.column_stack(
                    (
                        base_rows[:, :3],
                        np.zeros(base_rows.shape[0], dtype=np.float32),
                        np.full(base_rows.shape[0], np.nan, dtype=np.float32),
                    )
                ).astype(np.float32, copy=False)
            if temporal_rows.size == 0:
                merged.append(base_rows)
            elif base_rows.size == 0:
                merged.append(temporal_rows)
            else:
                merged.append(
                    np.vstack((base_rows, temporal_rows)).astype(np.float32, copy=False)
                )
        return merged

    candidate_consensus_frames = merge_candidate_rows(candidate_temporal_frames)
    compare(
        candidate_result["temporal_reference_candidate_count"],
        candidate_temporal_count,
        "candidate.temporal_reference_candidate_count",
    )
    compare(
        candidate_result["temporal_coadd_candidate_count"],
        0,
        "candidate.temporal_coadd_candidate_count",
    )
    candidate_consensus_audit: dict[str, int] = {}
    candidate_consensus_tracks = _candidate_consensus_tracks(
        candidate_consensus_frames,
        candidate_tracking_result.cumulative_shifts,
        candidate_tracking_result.tracks,
        frame_analyses=candidate_analyses,
        frame_times=candidate_times,
        link_radius_px=candidate_tracking_result.link_radius_px,
        min_presence=candidate_tracking_result.min_presence,
        persistent_min_presence=candidate_tracking_result.persistent_min_presence,
        motion_min_displacement_px=(
            candidate_tracking_result.effective_motion_min_displacement_px or 2.0
        ),
        max_motion_fit_rms_px=candidate_tracking_result.max_motion_fit_rms_px,
        min_candidate_snr=15.0,
        temporal_candidate_min_snr=7.5,
        temporal_reference_min_snr=15.0,
        temporal_min_psf_correlation=0.8,
        line_artifact_frames=[
            analysis.detection.line_artifact_coordinates
            for analysis in candidate_analyses
        ],
        audit_sink=candidate_consensus_audit,
    )
    candidate_consensus_rows = []
    for index, track in enumerate(candidate_consensus_tracks):
        row = track.as_dict()
        row["track_id"] = (
            len(candidate_tracking_result.tracks)
            + len(candidate_quality_fast_tracks)
            + len(candidate_python_tracks)
            + index
        )
        candidate_consensus_rows.append(row)
    compare(
        candidate_result["candidate_consensus"]["tracks"],
        candidate_consensus_rows,
        "candidate.candidate_consensus.tracks",
    )
    compare(
        candidate_result["candidate_consensus"]["audit"],
        candidate_consensus_audit,
        "candidate.candidate_consensus.audit",
    )
    compare(
        candidate_result["tracking"]["candidate_consensus_count"],
        len(candidate_consensus_rows),
        "candidate.tracking.candidate_consensus_count",
    )
    for mode, temporal_rows in (
        ("coadd", candidate_coadd_frames),
        (
            "both",
            tuple(
                np.vstack((median_rows, coadd_rows)).astype(np.float32, copy=False)
                if median_rows.size and coadd_rows.size
                else coadd_rows
                if coadd_rows.size
                else median_rows
                for median_rows, coadd_rows in zip(
                    candidate_temporal_frames,
                    candidate_coadd_frames,
                    strict=True,
                )
            ),
        ),
    ):
        mode_result = invoke(engine.resolve(), {**candidate_request, "temporal_proposal_mode": mode})
        compare(
            mode_result["temporal_reference_candidate_count"],
            candidate_temporal_count,
            f"candidate.{mode}.temporal_reference_candidate_count",
        )
        compare(
            mode_result["temporal_coadd_candidate_count"],
            candidate_coadd_count,
            f"candidate.{mode}.temporal_coadd_candidate_count",
        )
        expected_mode_audit: dict[str, int] = {}
        expected_mode_tracks = _candidate_consensus_tracks(
            merge_candidate_rows(temporal_rows),
            candidate_tracking_result.cumulative_shifts,
            candidate_tracking_result.tracks,
            frame_analyses=candidate_analyses,
            frame_times=candidate_times,
            link_radius_px=candidate_tracking_result.link_radius_px,
            min_presence=candidate_tracking_result.min_presence,
            persistent_min_presence=candidate_tracking_result.persistent_min_presence,
            motion_min_displacement_px=(
                candidate_tracking_result.effective_motion_min_displacement_px or 2.0
            ),
            max_motion_fit_rms_px=candidate_tracking_result.max_motion_fit_rms_px,
            min_candidate_snr=15.0,
            temporal_candidate_min_snr=7.5,
            temporal_reference_min_snr=15.0,
            temporal_min_psf_correlation=0.8,
            line_artifact_frames=[
                analysis.detection.line_artifact_coordinates
                for analysis in candidate_analyses
            ],
            audit_sink=expected_mode_audit,
        )
        expected_mode_rows = []
        for index, track in enumerate(expected_mode_tracks):
            row = track.as_dict()
            row["track_id"] = (
                len(candidate_tracking_result.tracks)
                + len(candidate_quality_fast_tracks)
                + len(candidate_python_tracks)
                + index
            )
            expected_mode_rows.append(row)
        actual_mode_rows = mode_result["candidate_consensus"]["tracks"]
        if mode == "both":
            actual_without_tie_scores = json.loads(json.dumps(actual_mode_rows))
            expected_without_tie_scores = json.loads(json.dumps(expected_mode_rows))
            for track_index, (actual_track, expected_track) in enumerate(
                zip(actual_without_tie_scores, expected_without_tie_scores, strict=True)
            ):
                actual_track.pop("evidence_level", None)
                expected_track.pop("evidence_level", None)
                for point_index, (actual_point, expected_point) in enumerate(
                    zip(actual_track["points"], expected_track["points"], strict=True)
                ):
                    actual_point.pop("detection_id", None)
                    expected_point.pop("detection_id", None)
                    frame_index = int(actual_point["frame_index"])
                    x = float(actual_point["x"])
                    y = float(actual_point["y"])
                    available_scores = []
                    raw = np.asarray(
                        candidate_analyses[frame_index].detection.candidate_peaks,
                        dtype=np.float64,
                    )
                    for rows in (raw, candidate_temporal_frames[frame_index], candidate_coadd_frames[frame_index]):
                        for row in rows:
                            if abs(float(row[0]) - x) <= 1e-6 and abs(float(row[1]) - y) <= 1e-6:
                                available_scores.append(float(row[2]))
                    actual_score = actual_point.get("candidate_snr")
                    if actual_score is not None and not any(
                        math.isclose(float(actual_score), score, rel_tol=5e-7, abs_tol=5e-7)
                        for score in available_scores
                    ):
                        raise AssertionError(
                            f"candidate.both track {track_index} point {point_index} score does not come from an input candidate"
                        )
                    actual_point["candidate_snr"] = None
                    expected_point["candidate_snr"] = None
            compare(
                actual_without_tie_scores,
                expected_without_tie_scores,
                "candidate.both.candidate_consensus.tracks_without_duplicate_row_tie_fields",
            )
        else:
            compare(
                actual_mode_rows,
                expected_mode_rows,
                f"candidate.{mode}.candidate_consensus.tracks",
            )
        actual_mode_audit = mode_result["candidate_consensus"]["audit"]
        if mode == "both":
            if actual_mode_audit.get("accepted") != expected_mode_audit.get("accepted"):
                raise AssertionError("candidate.both accepted track count differs")
            actual_mode_audit = dict(actual_mode_audit)
            expected_mode_audit = dict(expected_mode_audit)
            for key in ("accepted_candidate_consensus", "accepted_temporal_reference"):
                actual_mode_audit.pop(key, None)
                expected_mode_audit.pop(key, None)
        compare(
            actual_mode_audit,
            expected_mode_audit,
            f"candidate.{mode}.candidate_consensus.audit",
        )
    for proposal_index, proposal in enumerate(candidate_stage["proposals"]):
        for observation_index, observation in enumerate(proposal["observations"]):
            frame_index = observation["frame_index"]
            analysis = candidate_analyses[frame_index]
            image = np.asarray(analysis.frame.data)
            saturation_level = analysis.detection.parameters.get("saturation_level")
            if saturation_level is not None:
                saturation_level = float(saturation_level)
                if saturation_level <= 0:
                    saturation_level = None
            effective_mask, _inferred_saturation, _used_zero_mask = _working_mask(
                image,
                auxiliary_mask(image.shape),
                background=float(analysis.detection.background),
                mask_zero_pixels=bool(analysis.detection.parameters.get("mask_zero_pixels")),
                saturation_level=saturation_level,
                numeric=np.asarray(image, dtype=np.float32),
            )
            expected_measurement = _stack_forced_frame_measure_batch(
                image,
                effective_mask,
                np.asarray([observation["x"]], dtype=np.float64),
                np.asarray([observation["y"]], dtype=np.float64),
                aperture_radius=4,
                min_psf_support_pixels=3,
                global_background=float(analysis.detection.background),
                global_noise=float(analysis.detection.noise),
            )
            flux_snr, support, center_valid, fwhm, ellipticity = expected_measurement
            expected = {
                "flux_snr": float(flux_snr[0]),
                "psf_support_pixels": int(support[0]),
                "center_valid": bool(center_valid[0]),
                "fwhm": float(fwhm[0]),
                "ellipticity": float(ellipticity[0]),
            }
            compare(
                observation["original_pixel_measurement"],
                expected,
                f"candidate.proposals[{proposal_index}].observations[{observation_index}].original_pixel_measurement",
            )
    report = {
        "engine": str(engine.resolve()),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "frame_count": len(paths),
        "frame_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "python_baseline_sha256": {
            module: hashlib.sha256((ROOT / "src" / "rst19" / module).read_bytes()).hexdigest()
            for module in ("detection.py", "fits.py", "pipeline.py", "sequence.py")
        },
        "per_frame_analysis_fields_compared": True,
        "quality_track_fields_compared": [
            "frame_count",
            "stable_source_count",
            "persistent_source_count",
            "moving_track_count",
            "transient_track_count",
            "min_presence",
            "registration_model",
            "registration_affine_frame_count",
            "effective_motion_min_displacement_px",
            "cumulative_shifts",
            "tracks",
        ],
        "fast_point_motion_track_count": len(python_fast_point_rows),
        "fast_point_motion_disable_verified": True,
        "fast_candidate_proposal_recovery_verified": True,
        "fast_candidate_original_pixel_measurement_and_quality_gates_verified": True,
        "fast_candidate_tracks_match_python_fields": True,
        "candidate_consensus_tracks_and_audit_match_python": True,
        "temporal_reference_candidate_recovery_matches_python": True,
        "both_mode_duplicate_row_tie_note": "SciPy and Rust may select different source rows for exact-coordinate raw/temporal duplicates; track positions and classifications match, while evidence labels, synthetic detection IDs, per-point scores, and acceptance-channel counts can differ; every Rust score is checked against an input candidate row",
        "candidate_consensus_count": len(candidate_consensus_rows),
        "fast_candidate_proposal_count": candidate_stage["proposal_count"],
        "fast_candidate_accepted_track_count": candidate_stage["accepted_track_count"],
        "relative_photometry_orchestration_matches_python": True,
        "relative_photometry_status": candidate_result["relative_photometry"]["status"],
        "fixed_sentinel_and_impact_audits_match_python": True,
        "observation_times_seconds": frame_times,
        "scope": "Rust FITS detection, shared first-frame background, time parsing, ordinary quality tracking, quality-source fast-point tracking, registered temporal-reference candidate recovery, matched-filter candidate association with original-pixel forced measurement and quality gates, raw-image-gated static candidate consensus, optional relative-photometry orchestration, and fixed-sentinel impact audits; trail auditing and stack recovery remain outside this core operation",
    }
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.engine, arguments.out), ensure_ascii=False, indent=2))
