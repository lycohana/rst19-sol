import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rst19.ablation import _match_one_to_one, _save_case_sheet, _wilson, run_point_source_ablation, write_ablation_html
from rst19.detection import Detection, DetectionResult
from rst19.fits import FitsFrame
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


def test_case_sheet_marks_hybrid_candidate_rejected_by_quality_gate(tmp_path) -> None:
    row = {
        "gaussian_center_amplitude_adu": 24.0,
        "target_id": "PSF-001",
        "x_px": 32.5,
        "y_px": 32.5,
        "gaussian_candidate_hit": False,
        "hybrid_candidate_hit": True,
        "quality_gate_hit": False,
        "gaussian_x_px": "",
        "gaussian_y_px": "",
        "hybrid_x_px": 32.5,
        "hybrid_y_px": 32.5,
        "quality_x_px": None,
        "quality_y_px": None,
        "quality_rejection_flags": "LOW_FLUX_SNR",
    }
    output = tmp_path / "cases.png"

    _save_case_sheet({24.0: np.zeros((64, 64), dtype=np.float32)}, [row], output, 24.0)

    pixels = np.asarray(Image.open(output).convert("RGB"))
    assert np.any(np.all(pixels == (255, 91, 91), axis=2))


def test_report_rejects_stale_point_ablation_schema(tmp_path) -> None:
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version 3"):
        write_ablation_html(run_dir, tmp_path / "report.html")


def test_point_ablation_runs_paired_no_injection_a_b_and_reuses_b_for_c(monkeypatch, tmp_path) -> None:
    source = tmp_path / "input.fits"
    source.write_bytes(b"test fits identity")
    image = np.zeros((64, 64), dtype=np.int16)
    frame = FitsFrame(source, {}, image, None, 0)
    monkeypatch.setattr("rst19.ablation.read_fits", lambda _path: frame)
    monkeypatch.setattr(
        "rst19.ablation._select_real_background_positions",
        lambda *_args, **_kwargs: ([(32.5, 32.5)], [(0.0, 1.0)]),
    )
    calls: list[tuple[str, np.ndarray, dict[str, object]]] = []

    def fake_detect(input_image, *, proposal_mode, **options):
        calls.append((proposal_mode, np.asarray(input_image).copy(), options))
        injected = bool(np.any(input_image))
        if injected:
            x, y, passed, flags = 32.5, 32.5, proposal_mode == "gaussian", (() if proposal_mode == "gaussian" else ("LOW_FLUX_SNR",))
            sources = (
                Detection(
                    detection_id=1,
                    x=x,
                    y=y,
                    peak=20.0,
                    flux=30.0,
                    background=0.0,
                    noise=1.0,
                    snr=20.0,
                    fwhm=2.0,
                    flags=flags,
                    quality_passed=passed,
                ),
            )
        elif proposal_mode == "gaussian":
            sources = (
                Detection(1, 5.0, 5.0, 10, 10, 0, 1, 10, 2, (), quality_passed=True),
            )
        elif proposal_mode == "hybrid":
            sources = (
                Detection(2, 7.0, 7.0, 10, 10, 0, 1, 10, 2, ("LOW_FLUX_SNR",), quality_passed=False),
            )
        else:
            sources = ()
        return DetectionResult(
            image_shape=tuple(input_image.shape),
            background=0.0,
            noise=1.0,
            threshold=4.0,
            candidate_count=len(sources),
            sources=sources,
            parameters={"proposal_mode": proposal_mode, **{key: value for key, value in options.items() if key != "mask"}},
            quality_count=sum(source.quality_passed for source in sources),
        )

    monkeypatch.setattr("rst19.ablation.detect_sources", fake_detect)
    output = tmp_path / "run"
    manifest = run_point_source_ablation(
        source,
        output,
        peak_levels_adu=(24.0,),
        target_count=1,
        psf_fwhm_px=2.0,
        injected_psf_fwhm_px=2.0,
        min_distance_px=4,
        background_box_size_px=8,
    )

    assert [call[0] for call in calls] == ["ensemble", "gaussian", "hybrid", "gaussian", "hybrid"]
    assert calls[1][1].dtype == np.float32 and calls[2][1].dtype == np.float32
    assert np.array_equal(calls[1][1], calls[2][1])
    assert {key: value for key, value in calls[1][2].items() if key != "mask"} == {
        key: value for key, value in calls[2][2].items() if key != "mask"
    }
    assert manifest["shared_control"]["versions"][2]["candidate_count"] == 1
    assert manifest["shared_control"]["versions"][2]["output_count"] == 0
    assert (output / "control_summary.csv").is_file()
    assert set(manifest["outputs"]["case_sheets"]) == {"24"}
    report = write_ablation_html(output, tmp_path / "report.html")
    report_text = report.read_text(encoding="utf-8")
    for replay_arg in (
        "--seed 19019",
        "--psf-fwhm 2",
        "--injected-psf-fwhm 2",
        "--threshold-sigma 4",
        "--min-distance 4",
        "--aperture-radius 4",
        "--background-box-size 8",
        "--min-flux-snr 5",
        "--min-psf-support-pixels 3",
    ):
        assert replay_arg in report_text


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
