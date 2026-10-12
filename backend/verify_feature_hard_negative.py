"""Compare the Rust hard-negative audit with its Python baseline."""
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
from rst19.feature_hard_negative import (  # noqa: E402
    run_feature_hard_negative_audit,
    write_feature_hard_negative_artifacts,
)

CATALOG = (
    "detection_id,feature_class,feature_class_label,quality_passed,flags,flux_snr,filter_snr,"
    "peak,fwhm,ellipticity,sharpness,psf_support_pixels,footprint_pixels\r\n"
    '1,compact_quality,"紧凑,清晰",True,,2,4,20,2,0.1,0.3,5,20\r\n'
    "2,compact_quality,紧凑,False,LOW_FLUX_SNR,12,24,120,2,0.1,0.3,5,20\r\n"
    "3,compact_quality,紧凑,False,UNRESOLVED_BLEND,20,40,200,2,0.1,0.3,5,20\r\n"
    "6,compact_quality,紧凑,False,UNRESOLVED_BLEND,20,500,300,2,0.1,0.3,5,20\r\n"
    "4,range_anomaly,范围,False,NEGATIVE_OVERFLOW,50,10,500,NaN,0.8,,2,3\r\n"
    "5,other_rejected,其他,False,OTHER,100,200,1000,2,0.1,0.3,5,20\r\n"
)
CASES = (
    {"metric": "flux_snr", "top_n": 2, "exclude_classes": ["other_rejected"]},
    {"metric": "filter_snr", "top_n": 1, "exclude_classes": ["other_rejected"]},
    {"metric": "peak", "top_n": 1, "exclude_classes": []},
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
    with tempfile.TemporaryDirectory(prefix="rst19-feature-hard-negative-") as temporary:
        root = Path(temporary)
        source = root / "source catalog.csv"
        source.write_bytes(("\ufeff" + CATALOG).encode("utf-8"))
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        case_summaries = []
        for case in CASES:
            metric = case["metric"]
            top_n = case["top_n"]
            excluded = case["exclude_classes"]
            baseline = run_feature_hard_negative_audit(
                source,
                metric=metric,
                top_n=top_n,
                exclude_classes=excluded,
            )
            expected = baseline.as_dict()
            case_root = root / metric
            python_out = case_root / "python"
            native_out = case_root / "native"
            write_feature_hard_negative_artifacts(baseline, python_out)
            args = [str(source), "--metric", metric, "--top-n", str(top_n)]
            if not excluded:
                args.append("--include-other-rejected")
            args += ["--out-dir", str(native_out)]
            worker_payload = invoke(
                worker,
                {
                    "operation": "tool",
                    "tool": "rst19-feature-hard-negative",
                    "args": args,
                    "out_dir": str(case_root / "worker-job"),
                },
            )
            engine_payload = invoke(
                engine,
                {
                    "operation": "audit.feature_hard_negative",
                    "catalog_path": str(source),
                    "metric": metric,
                    "top_n": top_n,
                    "exclude_classes": excluded,
                },
            )
            assert worker_payload["result"] == expected
            assert engine_payload == expected
            artifacts = (
                "feature_hard_negative_summary.csv",
                "feature_hard_negative_rows.csv",
            )
            for name in artifacts:
                assert (native_out / name).read_bytes() == (python_out / name).read_bytes(), name
            native_json = json.loads(
                (native_out / "feature_hard_negative.json").read_text(encoding="utf-8")
            )
            assert native_json == expected
            case_summaries.append(
                {
                    "metric": metric,
                    "top_n": top_n,
                    "excluded_classes": excluded,
                    "summary_count": len(expected["summaries"]),
                    "detail_count": len(expected["rows"]),
                    "all_result_fields_match": True,
                    "csv_artifacts_byte_identical": True,
                    "json_fields_match": True,
                }
            )
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        summary = {
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "source_catalog_sha256": source_hash,
            "source_catalog_unchanged": True,
            "cases": case_summaries,
            "flux_snr_tie_order": [row["detection_id"] for row in expected_tie_rows(source)],
        }
    (out / "verification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def expected_tie_rows(source: Path) -> list[dict[str, object]]:
    result = run_feature_hard_negative_audit(source, top_n=2)
    return [row.as_dict() for row in result.rows if row.feature_class == "compact_quality"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.worker, args.engine, args.out), ensure_ascii=False, indent=2))
