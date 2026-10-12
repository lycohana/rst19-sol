"""Compare the native feature-class context audit with its Python baseline."""
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
from rst19.feature_class_context import (  # noqa: E402
    run_feature_class_context,
    write_feature_class_context_artifacts,
)


CATALOG = (
    "detection_id,feature_class,feature_class_label,quality_passed,flags,flux_snr,filter_snr,"
    "fwhm,ellipticity,sharpness,psf_support_pixels,footprint_pixels,centroid_shift_px,peak\n"
    "1,compact_quality,紧凑质量源,True,,5,10,2,0.1,0.3,5,20,0.1,50\n"
    "2,compact_quality,紧凑质量源,False,LOW_FLUX_SNR,10,20,2.2,0.2,0.4,4,18,0.2,100\n"
    "3,compact_quality,紧凑质量源,True,,15,30,2.1,0.15,0.35,6,22,NaN,150\n"
)
TARGET_IDS = (2, 1, 2)


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
    with tempfile.TemporaryDirectory(prefix="rst19-feature-class-context-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        catalog.write_bytes(("\ufeff" + CATALOG).encode("utf-8"))
        before = hashlib.sha256(catalog.read_bytes()).hexdigest()
        baseline = run_feature_class_context(catalog, TARGET_IDS)
        expected = baseline.as_dict()
        python_out = root / "python"
        native_out = root / "native"
        write_feature_class_context_artifacts(baseline, python_out)
        worker_payload = run_process(
            worker,
            {
                "operation": "tool",
                "tool": "rst19-feature-class-context",
                "args": [
                    str(catalog),
                    *[item for target_id in TARGET_IDS for item in ("--target-id", str(target_id))],
                    "--out-dir",
                    str(native_out),
                ],
                "out_dir": str((root / "job").resolve()),
            },
        )
        engine_payload = run_process(
            engine,
            {
                "operation": "audit.feature_class_context",
                "catalog_path": str(catalog),
                "target_ids": list(TARGET_IDS),
            },
        )
        assert worker_payload["result"] == expected
        assert engine_payload == expected
        for name in (
            "feature_class_context_targets.csv",
            "feature_class_context_metrics.csv",
        ):
            assert (native_out / name).read_bytes() == (python_out / name).read_bytes()
        native_json = json.loads(
            (native_out / "feature_class_context.json").read_text(encoding="utf-8")
        )
        assert native_json == expected
        assert hashlib.sha256(catalog.read_bytes()).hexdigest() == before
        result = {
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "catalog_sha256": before,
            "all_result_fields_match": True,
            "target_csv_byte_identical": True,
            "metric_csv_byte_identical": True,
            "json_artifact_fields_match": True,
            "source_catalog_unchanged": True,
            "record_count": len(CATALOG.splitlines()) - 1,
            "target_count_after_deduplication": len(expected["targets"]),
            "metric_count": len(expected["metrics"]),
            "within_class_percentiles_match": True,
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
