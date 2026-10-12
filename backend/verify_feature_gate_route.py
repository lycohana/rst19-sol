"""Compare the Rust quality-gate route audit with its Python baseline."""
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
from rst19.feature_gate_route import (  # noqa: E402
    run_feature_gate_route,
    write_feature_gate_route_artifacts,
)

CATALOG = (
    "detection_id,feature_class,feature_class_label,quality_passed,flags,flux_snr,"
    "psf_support_pixels,fwhm,ellipticity,sharpness,footprint_pixels\r\n"
    '1,compact_quality,"紧凑,清晰",True,,12,5,2.5,0.1,0.3,20\r\n'
    "2,weak_or_background,弱,False,LOW_FLUX_SNR,3,5,2.5,0.1,0.3,20\r\n"
    "3,linear_artifact,线状,False,LINE_ARTIFACT,20,5,2.5,0.1,0.3,20\r\n"
    "4,crowded_blend,拥挤,False,LOW_FLUX_SNR|UNRESOLVED_BLEND,3,2,2.5,0.1,0.3,20\r\n"
    '5,crowded_blend,"拥挤;重复",yes,"SMALL_FOOTPRINT;SMALL_FOOTPRINT|LOW_FLUX_SNR",9,5,13,0.1,0.3,1\r\n'
    "6,other_rejected,边缘,False,INSUFFICIENT_PSF_SUPPORT,NaN,,nan,0.1,0.3,20\r\n"
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


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="rst19-feature-gate-route-") as temporary:
        root = Path(temporary)
        source = root / "source catalog.csv"
        source.write_bytes(("\ufeff" + CATALOG).encode("utf-8"))
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        baseline = run_feature_gate_route(source)
        expected = baseline.as_dict()
        python_out = root / "python"
        native_out = root / "native"
        write_feature_gate_route_artifacts(baseline, python_out)
        worker_payload = invoke(
            worker,
            {
                "operation": "tool",
                "tool": "rst19-feature-gate-route",
                "args": [str(source), "--out-dir", str(native_out)],
                "out_dir": str(root / "worker-job"),
            },
        )
        engine_payload = invoke(
            engine,
            {
                "operation": "audit.feature_gate_route",
                "catalog_path": str(source),
            },
        )
        assert worker_payload["result"] == expected
        assert engine_payload == expected
        artifact_names = (
            "feature_gate_route_sources.csv",
            "feature_gate_route_summary.csv",
        )
        for name in artifact_names:
            assert (native_out / name).read_bytes() == (python_out / name).read_bytes(), name
        native_json = json.loads(
            (native_out / "feature_gate_routes.json").read_text(encoding="utf-8")
        )
        assert native_json == expected
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        sources = {row["detection_id"]: row for row in expected["source_rows"]}
        summary = {
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "source_catalog_sha256": source_hash,
            "all_result_fields_match": True,
            "source_csv_byte_identical": True,
            "summary_csv_byte_identical": True,
            "json_fields_match": True,
            "source_catalog_unchanged": True,
            "source_count": len(expected["source_rows"]),
            "summary_class_count": len(expected["summary_rows"]),
            "numeric_only": sources["2"]["route"],
            "structural_only": sources["3"]["route"],
            "mixed_flags": sources["4"]["structural_flags"],
            "satisfied_numeric_aliases_retained": sources["5"]["structural_flags"],
            "quality_passed_label_preserved": sources["1"]["quality_passed"],
        }
    (out / "verification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.worker, args.engine, args.out), ensure_ascii=False, indent=2))
