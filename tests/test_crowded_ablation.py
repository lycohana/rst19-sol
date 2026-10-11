import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rst19.crowded_ablation import run_crowded_pair_ablation
from rst19.detection import Detection, DetectionResult


def test_crowded_ablation_freezes_paired_inputs_and_reports_companion_gate_tradeoff(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source.fits"
    source.write_bytes(b"real-fits-identity-for-test")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "point-manifest.json"
    manifest_path.write_text(json.dumps({
        "experiment": "point_source_detection_ablation",
        "source_fits": {"sha256": source_sha},
        "targets": [{"target_id": "ANCHOR-1", "x_px": 96.0, "y_px": 96.0}],
    }), encoding="utf-8")
    raw = np.full((192, 192), 100, dtype=np.int16)
    monkeypatch.setattr("rst19.crowded_ablation.read_fits", lambda _path: SimpleNamespace(data=raw))
    calls: list[tuple[np.ndarray, str, bool]] = []

    def fake_detect(image, *, proposal_mode, enable_local_deblend=False, **options):
        values = np.asarray(image)
        is_injected = float(values[48, 48]) > 180.0
        is_pair = is_injected and float(values[48, 54]) > 115.0
        calls.append((values.copy(), proposal_mode, enable_local_deblend))
        sources = []
        if is_injected:
            sources.append(Detection(1, 48.0, 48.0, 260, 500, 100, 2, 80, 1.4, (), quality_passed=True))
            if is_pair and proposal_mode == "hybrid":
                sources.append(Detection(2, 54.0, 48.0, 140, 40, 100, 2, 5, 0.8, ("LOW_FLUX_SNR",), quality_passed=False))
            if enable_local_deblend:
                sources.append(Detection(3, 54.0, 48.0, 140, 40, 100, 2, 5, 0.8, ("UNRESOLVED_BLEND",), quality_passed=False))
        options.pop("mask", None)
        detections = tuple(sources)
        return DetectionResult(
            image_shape=tuple(values.shape),
            background=100.0,
            noise=2.0,
            threshold=12.0,
            candidate_count=len(detections),
            sources=detections,
            parameters={"proposal_mode": proposal_mode, "enable_local_deblend": enable_local_deblend, **options},
            quality_count=sum(source.quality_passed for source in detections),
        )

    monkeypatch.setattr("rst19.crowded_ablation.detect_sources", fake_detect)
    output = tmp_path / "crowded-run"
    manifest = run_crowded_pair_ablation(source, manifest_path, output, anchor_count=1)

    pair_rows = [row for row in _read_csv(output / "summary.csv") if row["scene"] == "pair"]
    single_rows = [row for row in _read_csv(output / "summary.csv") if row["scene"] == "single"]
    b_pair = next(row for row in pair_rows if row["version"] == "B")
    c_pair = next(row for row in pair_rows if row["version"] == "C")
    d_single = next(row for row in single_rows if row["version"] == "D")
    assert (int(b_pair["candidate_matched_targets"]), int(b_pair["quality_matched_targets"])) == (2, 1)
    assert (int(c_pair["candidate_matched_targets"]), int(c_pair["quality_matched_targets"])) == (1, 1)
    assert int(d_single["negative_companion_site_candidates"]) == 1
    assert manifest["patch_selection"]["accepted_anchor_count"] == 1

    # Three versions receive the exact same frozen array for each scene; C is
    # a post-hoc filter and therefore does not trigger a fourth detector call.
    injected_calls = [item for item in calls if float(item[0][48, 48]) > 180.0]
    assert len(injected_calls) == 6
    for scene_group in (injected_calls[:3], injected_calls[3:]):
        assert all(np.array_equal(scene_group[0][0], item[0]) for item in scene_group[1:])
        assert {item[1] for item in scene_group} == {"gaussian", "hybrid"}
        assert sum(item[2] for item in scene_group) == 1
    assert len(list((output / "frozen_inputs").glob("*.npy"))) == 3
    assert (output / "crowded_pair_cases.png").is_file()
    assert all(Path(path).is_file() for path in _manifest_scene_files(manifest))


def test_crowded_ablation_rejects_changed_fits_identity_before_running(tmp_path) -> None:
    source = tmp_path / "source.fits"
    source.write_bytes(b"source")
    manifest_path = tmp_path / "point-manifest.json"
    manifest_path.write_text(json.dumps({
        "source_fits": {"sha256": "not-the-current-input"},
        "targets": [{"target_id": "ANCHOR-1", "x_px": 96, "y_px": 96}],
    }), encoding="utf-8")

    try:
        run_crowded_pair_ablation(source, manifest_path, tmp_path / "out", anchor_count=1)
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("changed FITS input must not reuse frozen anchor coordinates")


def _read_csv(path):
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _manifest_scene_files(manifest):
    anchor = manifest["patch_selection"]["accepted_anchors"][0]
    return [anchor["scenes"][scene]["file"] for scene in ("pair", "single")]
