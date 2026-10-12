"""Compare the Rust feature-correlation audit with its Python baseline."""
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
from rst19.feature_correlation_audit import (  # noqa: E402
    run_feature_correlation_audit,
    write_feature_correlation_artifacts,
)

CATALOG = (
    "feature_class,feature_class_label,a,b,c\n"
    'weak,"弱,背景",4,4,1\n'
    "weak,ignored,5,5,2\n"
    "weak,ignored,6,6,3\n"
    'weak,"ignored,again",,8,4\n'
    'compact_quality,"紧凑,清晰",1,3,10\n'
    "compact_quality,ignored,2,2,20\n"
    "compact_quality,ignored,3,1,30\n"
    ",,7,7,NaN\n"
)
FEATURES = (" a ", "b", "c", "a")
THRESHOLD = 0.75


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
    with tempfile.TemporaryDirectory(prefix="rst19-feature-correlation-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        catalog.write_bytes(("\ufeff" + CATALOG).encode("utf-8"))
        before = hashlib.sha256(catalog.read_bytes()).hexdigest()
        baseline = run_feature_correlation_audit(
            catalog,
            features=FEATURES,
            strong_threshold=THRESHOLD,
        )
        expected = baseline.as_dict()
        python_out = root / "python"
        native_out = root / "native"
        write_feature_correlation_artifacts(baseline, python_out)
        worker_payload = invoke(
            worker,
            {
                "operation": "tool",
                "tool": "rst19-feature-correlation",
                "args": [
                    str(catalog), "--features", *FEATURES,
                    "--strong-threshold", str(THRESHOLD), "--out-dir", str(native_out),
                ],
                "out_dir": str((root / "worker-job").resolve()),
            },
        )
        engine_payload = invoke(
            engine,
            {
                "operation": "audit.feature_correlation",
                "catalog_path": str(catalog),
                "features": list(FEATURES),
                "strong_threshold": THRESHOLD,
            },
        )
        assert worker_payload["result"] == expected
        assert engine_payload == expected
        artifacts = (
            "feature_correlations.csv",
            "feature_class_medians.csv",
            "feature_class_correlations.csv",
        )
        for name in artifacts:
            assert (native_out / name).read_bytes() == (python_out / name).read_bytes(), name
        assert json.loads((native_out / "feature_correlation_audit.json").read_text(encoding="utf-8")) == expected
        assert hashlib.sha256(catalog.read_bytes()).hexdigest() == before
        result = {
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "catalog_sha256": before,
            "all_result_fields_match": True,
            "correlation_csv_byte_identical": True,
            "class_median_csv_byte_identical": True,
            "class_correlation_csv_byte_identical": True,
            "json_artifact_fields_match": True,
            "source_catalog_unchanged": True,
            "record_count": len(CATALOG.splitlines()) - 1,
            "features_after_normalization": expected["features"],
            "global_pair_count": len(expected["correlations"]),
            "class_count": len(expected["class_medians"]),
            "class_pair_count": len(expected["class_correlations"]),
            "pairwise_sample_counts": [row["sample_count"] for row in expected["correlations"]],
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
