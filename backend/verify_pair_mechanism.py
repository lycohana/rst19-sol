"""Compare native pair-mechanism audits and artifacts with Python baselines."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rst19.pair_mechanism_context import (  # noqa: E402
    run_pair_mechanism_context,
    write_pair_mechanism_context_artifacts,
)
from rst19.pair_mechanism_sensitivity import (  # noqa: E402
    run_pair_mechanism_sensitivity,
    write_pair_mechanism_sensitivity_artifacts,
)

CATALOG = (
    "detection_id,x,y,peak_x,peak_y,quality_passed,feature_class,flags\n"
    '2,2.738,0,4,0,false,crowded_blend,"UNRESOLVED_BLEND,MANUAL"\n'
    "4,25,0,25,0,true,compact_quality,\n"
    '1,0,0,0,0,true,range_anomaly,"CODE_PATTERN,MANUAL"\n'
    "3,-0.5,0.5,3.5,0,true,crowded_blend,\n"
)
TARGET_IDS = (2, 1, 2)
CONFIGURATIONS = (
    ("narrow", 1.0, 3.5, 4.8),
    ("target", 3.5, 3.5, 4.8),
)


def invoke(executable: Path, request: dict[str, object]) -> dict:
    process = subprocess.run(
        [str(executable.resolve())],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    )
    events = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(events) == 1 and events[0]["ok"], events
    return events[0]["payload"]


def assert_files_equal(first: Path, second: Path) -> None:
    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()
    if first_bytes != second_bytes:
        left = first_bytes.decode("utf-8-sig").splitlines()
        right = second_bytes.decode("utf-8-sig").splitlines()
        difference = "\n".join(difflib.unified_diff(left, right, fromfile=str(first), tofile=str(second)))
        raise AssertionError(f"artifact bytes differ:\n{difference}")


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="rst19-pair-mechanism-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        catalog.write_bytes(("\ufeff" + CATALOG).encode("utf-8"))
        before = hashlib.sha256(catalog.read_bytes()).hexdigest()

        context = run_pair_mechanism_context(catalog, TARGET_IDS)
        expected_context = context.as_dict()
        context_python = root / "context-python"
        context_native = root / "context-native"
        write_pair_mechanism_context_artifacts(context, context_python)
        worker_context = invoke(
            worker,
            {
                "operation": "tool",
                "tool": "rst19-pair-mechanism-context",
                "args": [
                    str(catalog), "--target-id", "2", "--target-id", "1", "--target-id", "2",
                    "--out-dir", str(context_native),
                ],
                "out_dir": str((root / "context-job").resolve()),
            },
        )
        engine_context = invoke(
            engine,
            {
                "operation": "audit.pair_mechanism_context",
                "catalog_path": str(catalog),
                "target_ids": list(TARGET_IDS),
            },
        )
        assert worker_context["result"] == expected_context
        assert engine_context == expected_context
        context_csvs = (
            "pair_mechanism_context_pairs.csv",
            "pair_mechanism_context_summary.csv",
            "pair_mechanism_context_target.csv",
        )
        for name in context_csvs:
            assert_files_equal(context_native / name, context_python / name)
        assert json.loads((context_native / "pair_mechanism_context.json").read_text(encoding="utf-8")) == expected_context

        sensitivity = run_pair_mechanism_sensitivity(
            catalog, TARGET_IDS, configurations=CONFIGURATIONS
        )
        expected_sensitivity = sensitivity.as_dict()
        sensitivity_python = root / "sensitivity-python"
        sensitivity_native = root / "sensitivity-native"
        write_pair_mechanism_sensitivity_artifacts(sensitivity, sensitivity_python)
        worker_sensitivity = invoke(
            worker,
            {
                "operation": "tool",
                "tool": "rst19-pair-mechanism-sensitivity",
                "args": [
                    str(catalog), "--target-id", "2", "--target-id", "1", "--target-id", "2",
                    "--configuration", "narrow,1,3.5,4.8",
                    "--configuration", "target,3.5,3.5,4.8",
                    "--out-dir", str(sensitivity_native),
                ],
                "out_dir": str((root / "sensitivity-job").resolve()),
            },
        )
        engine_sensitivity = invoke(
            engine,
            {
                "operation": "audit.pair_mechanism_sensitivity",
                "catalog_path": str(catalog),
                "target_ids": list(TARGET_IDS),
                "configurations": [list(config) for config in CONFIGURATIONS],
            },
        )
        assert worker_sensitivity["result"] == expected_sensitivity
        assert engine_sensitivity == expected_sensitivity
        assert_files_equal(
            sensitivity_native / "pair_mechanism_sensitivity.csv",
            sensitivity_python / "pair_mechanism_sensitivity.csv",
        )
        assert json.loads(
            (sensitivity_native / "pair_mechanism_sensitivity.json").read_text(encoding="utf-8")
        ) == expected_sensitivity
        assert hashlib.sha256(catalog.read_bytes()).hexdigest() == before
        result = {
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "catalog_sha256": before,
            "context_result_fields_match": True,
            "context_csvs_byte_identical": True,
            "context_json_fields_match": True,
            "sensitivity_result_fields_match": True,
            "sensitivity_csv_byte_identical": True,
            "sensitivity_json_fields_match": True,
            "source_catalog_unchanged": True,
            "source_count": context.source_count,
            "context_pair_count": len(context.pairs),
            "sensitivity_configuration_count": len(sensitivity.rows),
            "target_ids_deduplicated": list(context.target_ids),
        }
    (out / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.worker, args.engine, args.out), ensure_ascii=False, indent=2))
