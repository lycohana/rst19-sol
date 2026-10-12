"""Run every current sequence-native process verifier against one release engine."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

VERIFIERS = (
    "verify_wcs_sequence.py",
    "verify_sequence_fast.py",
    "verify_sequence_consensus.py",
    "verify_sequence_reference.py",
    "verify_sequence_temporal_reference.py",
    "verify_sequence_trail_metrics.py",
    "verify_sequence_trail_shape.py",
    "verify_sequence_motion_feature_gates.py",
    "verify_sequence_motion_feature_continuity.py",
    "verify_sequence_motion_feature_exposure.py",
    "verify_sequence_motion_feature_confirmation.py",
    "verify_sequence_motion_proposal_thresholds.py",
    "verify_sequence_motion_time_axis.py",
    "verify_sequence_observation_time_axis.py",
    "verify_sequence_motion_feature_pair_geometry.py",
    "verify_collinear_motion_geometry.py",
    "verify_collinear_motion_group.py",
    "verify_single_frame_stitch.py",
    "verify_motion_evidence_link.py",
    "verify_collinear_motion_rescue.py",
    "verify_coherent_raw_motion_points.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    engine = args.engine.resolve()
    root = Path(__file__).resolve().parent
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    engine_hash = hashlib.sha256(engine.read_bytes()).hexdigest()
    reports = []
    for verifier in VERIFIERS:
        name = Path(verifier).stem
        destination = output / name
        result = subprocess.run(
            [sys.executable, str(root / verifier), str(engine), "--out", str(destination)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=1800,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"{verifier} failed with exit {result.returncode}\n{result.stdout}\n{result.stderr}"
            )
        report_path = destination / "verification.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("engine_sha256") != engine_hash:
            raise SystemExit(f"{verifier} used an unexpected engine build")
        if report.get("all_fields_compared") is not True:
            raise SystemExit(f"{verifier} did not compare all fields")
        count = int(report["native_requests_verified"])
        reports.append(
            {
                "directory": str(destination),
                "requests": count,
                "engine_sha256": engine_hash,
                "all_fields_compared": True,
            }
        )
        print(f"{verifier}: {count} requests passed")
    summary = {
        "report_count": len(reports),
        "native_requests_verified": sum(report["requests"] for report in reports),
        "engine_sha256": engine_hash,
        "all_fields_compared": True,
        "reports": reports,
    }
    (output / "verification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
