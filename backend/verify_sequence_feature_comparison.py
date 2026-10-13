"""Compare the Rust sequence feature comparison routes with the Python baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_detection import compare
from rst19.sequence_feature_comparison import (
    compare_sequence_feature_runs,
    write_sequence_feature_comparison_artifacts,
)


ARTIFACT_NAMES = (
    "sequence_feature_parameter_runs.csv",
    "sequence_feature_parameter_profiles.csv",
    "sequence_feature_parameter_transitions.csv",
)


def fixture(*, quality_presence: int = 7) -> dict:
    return {
        "frame_count": 2,
        "association_radius_px": 1.0,
        "required_presence": 2,
        "frame_rows": [
            {"frame_index": 1, "path": "frame-1.fits", "candidate_count": 100, "quality_count": 40},
            {"frame_index": 1, "path": "frame-1.fits", "candidate_count": 100, "quality_count": 40},
            {"frame_index": 2, "path": "frame-2.fits", "candidate_count": 101, "quality_count": 41},
        ],
        "persistence_rows": [
            {
                "feature_class": "compact_quality",
                "feature_class_label": "紧凑质量源",
                "anchor_candidate_count": 10,
                "candidate_presence_ge_required_count": 9,
                "candidate_presence_all_frames_count": 8,
                "anchor_quality_count": 8,
                "quality_presence_ge_required_count": quality_presence,
                "quality_presence_all_frames_count": 6,
            },
            {
                "feature_class": "range_anomaly",
                "feature_class_label": "异常值域",
                "anchor_candidate_count": 0,
                "candidate_presence_ge_required_count": 0,
                "candidate_presence_all_frames_count": 0,
                "anchor_quality_count": 0,
                "quality_presence_ge_required_count": 0,
                "quality_presence_all_frames_count": 0,
            },
            {
                "feature_class": "unlisted_class",
                "feature_class_label": "未知类别",
                "anchor_candidate_count": 3,
                "candidate_presence_ge_required_count": 1,
                "candidate_presence_all_frames_count": 0,
                "anchor_quality_count": 2,
                "quality_presence_ge_required_count": 1,
                "quality_presence_all_frames_count": 0,
            },
        ],
        "class_transition_rows": [
            {
                "layer": "quality",
                "anchor_feature_class": "compact_quality",
                "response_feature_class": "compact_quality",
                "anchor_count": 8,
                "frame_count": 2,
                "possible_match_count": 16,
                "matched_count": 14,
                "match_rate": 0.875,
                "mean_matches_per_anchor": 1.75,
            },
            {
                "layer": "candidate",
                "anchor_feature_class": "compact_quality",
                "response_feature_class": "range_anomaly",
                "anchor_count": 10,
                "frame_count": 2,
                "possible_match_count": 20,
                "matched_count": 2,
                "match_rate": 0.1,
                "mean_matches_per_anchor": 0.2,
            },
        ],
    }


def run_worker(worker: Path, args: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=False)
    request = {
        "operation": "tool",
        "tool": "rst19-feature-sequence-compare",
        "args": args,
        "out_dir": str(out_dir.resolve()),
    }
    process = subprocess.run(
        [str(worker.resolve())], input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode == 0 and len(events) == 1 and events[0]["event"] == "done" \
        and events[0]["ok"], (process.returncode, process.stdout, process.stderr)
    return events[0]["payload"]


def verify(engine: Path, cli: Path, worker: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    low_path = inputs / "low.json"
    high_path = inputs / "高质量.json"
    low_payload = fixture()
    high_payload = fixture(quality_presence=8)
    low_path.write_text(json.dumps(low_payload, ensure_ascii=False), encoding="utf-8")
    high_path.write_text(json.dumps(high_payload, ensure_ascii=False), encoding="utf-8")
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (low_path, high_path)
    }
    runs = {"SNR3": low_path, "SNR9": high_path}
    expected = compare_sequence_feature_runs(runs).as_dict()
    python_out = out / "python-artifacts"
    write_sequence_feature_comparison_artifacts(
        compare_sequence_feature_runs(runs), python_out
    )

    request = {
        "operation": "sequence.feature_parameter_compare",
        "runs": [
            {"configuration": name, "path": str(path)}
            for name, path in runs.items()
        ],
    }
    engine_process = subprocess.run(
        [str(engine.resolve())], input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    engine_events = [json.loads(line) for line in engine_process.stdout.splitlines()]
    assert engine_process.returncode == 0 and len(engine_events) == 1 \
        and engine_events[0]["ok"], (
            engine_process.returncode, engine_process.stdout, engine_process.stderr
        )
    compare(engine_events[0]["payload"], expected)

    cli_out = out / "cli-artifacts"
    cli_runs = [f"--run={name}={path}" for name, path in runs.items()]
    cli_process = subprocess.run(
        [str(cli.resolve()), *cli_runs, "--out-dir", str(cli_out.resolve())],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert cli_process.returncode == 0, (cli_process.stdout, cli_process.stderr)
    cli_json = json.loads(
        (cli_out / "sequence_feature_parameter_comparison.json").read_text(encoding="utf-8")
    )
    compare(cli_json, expected)

    worker_out = out / "worker-artifacts"
    worker_args: list[str] = []
    for name, path in runs.items():
        worker_args.extend(["--run", f"{name}={path}"])
    worker_args.extend(["--out-dir", str(worker_out / "artifacts")])
    worker_result = run_worker(worker, worker_args, worker_out)
    compare(worker_result["result"], expected)

    for name in ARTIFACT_NAMES:
        expected_bytes = (python_out / name).read_bytes()
        assert (cli_out / name).read_bytes() == expected_bytes, f"CLI artifact differs: {name}"
        assert (worker_out / "artifacts" / name).read_bytes() == expected_bytes, (
            f"desktop worker artifact differs: {name}"
        )
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (low_path, high_path)
    }
    assert before == after, "input JSON changed"
    return {
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "python_source_sha256": hashlib.sha256(
            (ROOT / "src/rst19/sequence_feature_comparison.py").read_bytes()
        ).hexdigest(),
        "run_count": len(expected["runs"]),
        "profile_row_count": len(expected["rows"]),
        "transition_row_count": len(expected["transitions"]),
        "engine_fields_match": True,
        "cli_and_worker_fields_match": True,
        "comparison_json_fields_match": True,
        "all_csv_artifacts_byte_identical": True,
        "inputs_sha256_before_after": before,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("cli", type=Path)
    parser.add_argument("worker", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.engine, args.cli, args.worker, args.out), ensure_ascii=False, indent=2))
