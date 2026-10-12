"""Compare the native feature-flag interaction audit with its Python baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rst19.feature_flag_interaction import (
    run_feature_flag_interaction,
    write_feature_flag_interaction_artifacts,
)


CATALOG = (
    "detection_id,quality_passed,flags,flux_snr\n"
    "1,true,ALPHA|BETA|ALPHA,12\n"
    "2,false,BETA|ALPHA,10\n"
    "3,false,ALPHA|BETA|GAMMA,20\n"
    "4,true,GAMMA,3\n"
    "5,true,,15\n"
)
THRESHOLD = 10.0


def run_process(executable: Path, request: dict[str, object]) -> dict:
    response = subprocess.run(
        [str(executable.resolve())],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    )
    events = [json.loads(line) for line in response.stdout.splitlines()]
    assert len(events) == 1 and events[0]["ok"], events
    return events[0]["payload"]


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="rst19-feature-flag-interaction-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        catalog.write_bytes(("\ufeff" + CATALOG).encode("utf-8"))
        before = hashlib.sha256(catalog.read_bytes()).hexdigest()
        baseline = run_feature_flag_interaction(catalog, high_snr_threshold=THRESHOLD)
        expected = baseline.as_dict()
        python_out = root / "python"
        native_out = root / "native"
        write_feature_flag_interaction_artifacts(baseline, python_out)
        worker_payload = run_process(
            worker,
            {
                "operation": "tool",
                "tool": "rst19-feature-flag-interaction",
                "args": [
                    str(catalog),
                    "--high-snr-threshold",
                    str(THRESHOLD),
                    "--top-n",
                    "2",
                    "--out-dir",
                    str(native_out),
                ],
                "out_dir": str((root / "job").resolve()),
            },
        )
        engine_payload = run_process(
            engine,
            {
                "operation": "audit.feature_flag_interaction",
                "catalog_path": str(catalog),
                "high_snr_threshold": THRESHOLD,
            },
        )
        assert worker_payload["result"] == expected
        assert engine_payload == expected
        name = "feature_flag_interactions.csv"
        assert (native_out / name).read_bytes() == (python_out / name).read_bytes()
        native_json = json.loads((native_out / "feature_flag_interactions.json").read_text(encoding="utf-8"))
        assert native_json == expected
        assert hashlib.sha256(catalog.read_bytes()).hexdigest() == before
        result = {
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "catalog_sha256": before,
            "all_result_fields_match": True,
            "csv_artifact_byte_identical": True,
            "json_artifact_fields_match": True,
            "source_catalog_unchanged": True,
            "record_count": expected["record_count"],
            "interaction_count": len(expected["rows"]),
        }
    (out / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.worker, args.engine, args.out), ensure_ascii=False, indent=2))
