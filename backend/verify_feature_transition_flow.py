"""Compare the Rust transition-flow audit with its Python baseline."""
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
from rst19.feature_transition_flow import (  # noqa: E402
    summarize_sequence_feature_transition_flow,
    write_sequence_feature_transition_flow_artifacts,
)

CATALOG = (
    "layer,anchor_feature_class,anchor_feature_class_label,response_feature_class,"
    "response_feature_class_label,anchor_count,frame_count,possible_match_count,matched_count\r\n"
    'candidate,compact_quality,"紧凑,清晰",compact_quality,紧凑,4,3,12,3\r\n'
    "candidate,compact_quality,ignored,weak_or_background,背景,4,3,12,2\r\n"
    "candidate,compact_quality,ignored,linear_artifact,线伪迹,4,3,12,2\r\n"
    "quality,masked_or_edge,边缘,masked_or_edge,边缘,5,3,15,0\r\n"
    "quality,other_rejected,拒绝,other_rejected,拒绝,0,0,0,0\r\n"
    ",,,,,,,,\r\n"
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
    with tempfile.TemporaryDirectory(prefix="rst19-feature-transition-flow-") as temporary:
        root = Path(temporary)
        source = root / "sequence transition.csv"
        source.write_bytes(("\ufeff" + CATALOG).encode("utf-8"))
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        baseline = summarize_sequence_feature_transition_flow(source)
        expected = baseline.as_dict()
        python_out = root / "python"
        native_out = root / "native"
        write_sequence_feature_transition_flow_artifacts(baseline, python_out)
        worker_payload = invoke(
            worker,
            {
                "operation": "tool",
                "tool": "rst19-feature-transition-flow",
                "args": [str(source), "--out-dir", str(native_out)],
                "out_dir": str(root / "worker-job"),
            },
        )
        engine_payload = invoke(
            engine,
            {
                "operation": "audit.feature_transition_flow",
                "csv_path": str(source),
            },
        )
        assert worker_payload["result"] == expected
        assert engine_payload == expected
        assert (
            native_out / "sequence_feature_transition_flow.csv"
        ).read_bytes() == (
            python_out / "sequence_feature_transition_flow.csv"
        ).read_bytes()
        expected_json = json.loads(
            (python_out / "sequence_feature_transition_flow.json").read_text(encoding="utf-8")
        )
        native_json = json.loads(
            (native_out / "sequence_feature_transition_flow.json").read_text(encoding="utf-8")
        )
        assert native_json == expected_json == expected
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        summary = {
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "source_csv_sha256": source_hash,
            "all_result_fields_match": True,
            "csv_byte_identical": True,
            "json_fields_match": True,
            "source_csv_unchanged": True,
            "input_row_count": expected["input_row_count"],
            "included_group_count": expected["included_group_count"],
            "skipped_zero_anchor_group_count": expected["skipped_zero_anchor_group_count"],
            "ordered_groups": [
                f"{row['layer']}/{row['anchor_feature_class']}"
                for row in expected["rows"]
            ],
            "zero_response_ratios_are_null": (
                expected["rows"][1]["q_any"] == 0.0
                and expected["rows"][1]["q_same"] is None
            ),
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
