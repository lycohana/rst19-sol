"""Compare the Rust aperture/peak audit with the Python baseline."""
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
from rst19.aperture_peak_ratio import (  # noqa: E402
    run_aperture_peak_ratio_audit,
    write_aperture_peak_ratio_artifacts,
)

CATALOG = (
    "detection_id,feature_class,peak,flux,proposal_methods\n"
    '82931,range_anomaly,4,24,"gaussian|dog_narrow"\n'
    "82932,range_anomaly,10,20,gaussian\n"
    "82934,crowded_blend,2,50,gaussian\n"
    "82935,crowded_blend,5,5,dog_narrow\n"
    '82936,complete_set,4,48,"dog_broad|gaussian|dog_narrow|gaussian"\n'
    '82937,dog_only,3,45,"dog_narrow|dog_broad"\n'
    "82938,no_method,2,18,\n"
    "82939,range_anomaly,0,99,gaussian\n"
    "82940,range_anomaly,NaN,12,gaussian\n"
    ",missing_id,2,12,gaussian\n"
    "82941,,2,12,gaussian\n"
)
TARGET_IDS = ("82934", "82931", "82934", " missing ")


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


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="rst19-aperture-peak-ratio-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        catalog.write_bytes(("\ufeff" + CATALOG).encode("utf-8"))
        before = hashlib.sha256(catalog.read_bytes()).hexdigest()
        baseline = run_aperture_peak_ratio_audit(catalog, target_ids=TARGET_IDS)
        expected = baseline.as_dict()
        python_out = root / "python"
        native_out = root / "native"
        write_aperture_peak_ratio_artifacts(baseline, python_out)
        worker_payload = invoke(
            worker,
            {
                "operation": "tool",
                "tool": "rst19-aperture-peak-ratio",
                "args": [
                    str(catalog), "--target-ids", *TARGET_IDS, "--out-dir", str(native_out),
                ],
                "out_dir": str((root / "worker-job").resolve()),
            },
        )
        engine_payload = invoke(
            engine,
            {
                "operation": "audit.aperture_peak_ratio",
                "catalog_path": str(catalog),
                "target_ids": list(TARGET_IDS),
            },
        )
        assert worker_payload["result"] == expected
        assert engine_payload == expected
        artifacts = (
            "aperture_peak_ratio_summary.csv",
            "aperture_peak_ratio_method_summary.csv",
            "aperture_peak_ratio_targets.csv",
        )
        for name in artifacts:
            assert (native_out / name).read_bytes() == (python_out / name).read_bytes(), name
        assert json.loads((native_out / "aperture_peak_ratio.json").read_text(encoding="utf-8")) == expected
        assert hashlib.sha256(catalog.read_bytes()).hexdigest() == before
        result = {
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "catalog_sha256": before,
            "all_result_fields_match": True,
            "summary_csv_byte_identical": True,
            "method_summary_csv_byte_identical": True,
            "target_csv_byte_identical": True,
            "json_artifact_fields_match": True,
            "source_catalog_unchanged": True,
            "record_count": len(CATALOG.splitlines()) - 1,
            "feature_class_count": len(expected["summaries"]),
            "method_group_count": len(expected["method_summaries"]),
            "target_row_count": len(expected["target_rows"]),
            "method_groups": sorted({row["method_group"] for row in expected["method_summaries"]}),
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
