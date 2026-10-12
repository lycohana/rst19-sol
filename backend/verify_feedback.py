"""Compare Rust feedback JSONL persistence with the Python reference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rst19.feedback import (  # noqa: E402
    ManualThresholdFeedback,
    append_manual_feedback,
    load_manual_feedback,
)


def invoke(executable: Path, request: dict[str, object]) -> dict:
    process = subprocess.run(
        [str(executable.resolve())], input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=True,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(events) == 1 and events[0]["ok"], events
    return events[0]["payload"]


def invoke_error(executable: Path, request: dict[str, object]) -> str:
    process = subprocess.run(
        [str(executable.resolve())], input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode != 0 and len(events) == 1 and not events[0]["ok"], events
    return events[0]["error"]


def feedback(timestamp: str, *, note: str = "") -> ManualThresholdFeedback:
    return ManualThresholdFeedback(
        timestamp_utc=timestamp,
        frame_path="星场/F01.fits",
        threshold_sigma=3.5,
        min_flux_snr=3.0,
        min_distance=4,
        psf_fwhm=3.0,
        candidate_count=120,
        quality_count=80,
        returned_count=120,
        flag_counts={"LOW_FLUX_SNR": 40, "SPIKE": 2},
        judgement="保留弱星",
        note=note,
        source="native_manual",
    )


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="rst19-feedback-") as temporary:
        root = Path(temporary)
        python_path = root / "python" / "feedback.jsonl"
        native_path = root / "native" / "feedback.jsonl"
        first = feedback("2026-10-13T00:00:00+00:00")
        second = feedback("2026-10-13T00:01:00+00:00", note="第二次手工阈值试验")
        append_manual_feedback(python_path, first)
        append_manual_feedback(python_path, second)

        engine_append = invoke(engine, {
            "operation": "feedback.append", "path": str(native_path), "feedback": first.as_dict(),
        })
        worker_append = invoke(worker, {
            "operation": "feedback.append", "path": str(native_path), "feedback": second.as_dict(),
        })
        assert engine_append["record"] == first.as_dict()
        assert worker_append["record"] == second.as_dict()
        native_records = [json.loads(line) for line in native_path.read_text(encoding="utf-8").splitlines()]
        python_records = [json.loads(line) for line in python_path.read_text(encoding="utf-8").splitlines()]
        assert native_records == python_records

        expected = [record.as_dict() for record in load_manual_feedback(python_path)]
        engine_load = invoke(engine, {"operation": "feedback.load", "path": str(native_path)})
        worker_load = invoke(worker, {"operation": "feedback.load", "path": str(native_path)})
        assert engine_load == {"records": expected}
        assert worker_load == {"records": expected}

        legacy_path = root / "legacy.jsonl"
        legacy_record = {
            "timestamp_utc": "legacy", "frame_path": "old.fits", "threshold_sigma": "3.5",
            "min_flux_snr": 3, "min_distance": 4.9, "psf_fwhm": 2,
            "candidate_count": 10, "quality_count": 5, "returned_count": 10,
        }
        legacy_path.write_text(json.dumps(legacy_record, ensure_ascii=False) + "\n", encoding="utf-8")
        legacy_expected = [record.as_dict() for record in load_manual_feedback(legacy_path)]
        assert invoke(engine, {"operation": "feedback.load", "path": str(legacy_path)}) == {
            "records": legacy_expected,
        }
        assert invoke(worker, {"operation": "feedback.load", "path": str(legacy_path)}) == {
            "records": legacy_expected,
        }

        missing_path = root / "missing.jsonl"
        assert invoke(engine, {"operation": "feedback.load", "path": str(missing_path)}) == {"records": []}
        assert invoke(worker, {"operation": "feedback.load", "path": str(missing_path)}) == {"records": []}

        corrupt_path = root / "corrupt.jsonl"
        corrupt_path.write_text("\nnot-json\n", encoding="utf-8")
        python_error = None
        try:
            load_manual_feedback(corrupt_path)
        except ValueError as error:
            python_error = str(error)
        assert python_error is not None and python_error.startswith("人工反馈第 2 行格式无效：")
        engine_error = invoke_error(engine, {"operation": "feedback.load", "path": str(corrupt_path)})
        worker_error = invoke_error(worker, {"operation": "feedback.load", "path": str(corrupt_path)})
        assert engine_error.startswith("人工反馈第 2 行格式无效：")
        assert worker_error.startswith("人工反馈第 2 行格式无效：")

        report = {
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "appended_jsonl_records_match": True,
            "loaded_records_match": True,
            "legacy_defaults_and_numeric_coercions_match": True,
            "missing_file_returns_empty": True,
            "corrupt_record_line_number_preserved": True,
        }
    (out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.worker, arguments.engine, arguments.out), ensure_ascii=False, indent=2))
