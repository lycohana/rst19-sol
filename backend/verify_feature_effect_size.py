"""Compare the native feature effect-size audit with its Python baseline."""
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
from rst19.feature_effect_size import (  # noqa: E402
    run_feature_effect_size_audit,
    write_feature_effect_size_artifacts,
)


CATALOG = (
    "feature_class,flux_snr,fwhm\n"
    "compact_quality,10,2\n"
    "compact_quality,20,2.2\n"
    "compact_quality,30,2.1\n"
    "weak_or_background,1,3\n"
    "weak_or_background,20,2.2\n"
    "spike_or_support,5,4\n"
    "spike_or_support,NaN,3.8\n"
    "other_rejected,200,0.5\n"
)
FEATURES = ("flux_snr", "fwhm", "flux_snr")
COMPARISON_CLASSES = ("spike_or_support", "weak_or_background", "spike_or_support")
TOP_N = 2


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
    with tempfile.TemporaryDirectory(prefix="rst19-feature-effect-size-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        catalog.write_bytes(("\ufeff" + CATALOG).encode("utf-8"))
        before = hashlib.sha256(catalog.read_bytes()).hexdigest()
        baseline = run_feature_effect_size_audit(
            catalog,
            comparison_classes=COMPARISON_CLASSES,
            features=FEATURES,
            top_n=TOP_N,
        )
        expected = baseline.as_dict()
        python_out = root / "python"
        native_out = root / "native"
        write_feature_effect_size_artifacts(baseline, python_out)
        worker_payload = run_process(
            worker,
            {
                "operation": "tool",
                "tool": "rst19-feature-effect-size",
                "args": [
                    str(catalog),
                    "--comparison-classes",
                    *COMPARISON_CLASSES,
                    "--features",
                    *FEATURES,
                    "--top-n",
                    str(TOP_N),
                    "--out-dir",
                    str(native_out),
                ],
                "out_dir": str((root / "job").resolve()),
            },
        )
        engine_payload = run_process(
            engine,
            {
                "operation": "audit.feature_effect_size",
                "catalog_path": str(catalog),
                "comparison_classes": list(COMPARISON_CLASSES),
                "features": list(FEATURES),
                "top_n": TOP_N,
            },
        )
        assert worker_payload["result"] == expected
        assert engine_payload == expected
        assert (native_out / "feature_effect_sizes.csv").read_bytes() == (
            python_out / "feature_effect_sizes.csv"
        ).read_bytes()
        native_json = json.loads(
            (native_out / "feature_effect_sizes.json").read_text(encoding="utf-8")
        )
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
            "record_count": len(CATALOG.splitlines()) - 1,
            "comparison_class_count": len(expected["comparison_classes"]),
            "feature_count": len(expected["features"]),
            "effect_row_count": len(expected["rows"]),
            "tie_aware_auc_and_top_order_match": True,
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
