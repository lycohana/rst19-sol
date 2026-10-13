from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.feature_gate_temporal_route import (  # noqa: E402
    run_feature_gate_temporal_route,
    write_feature_gate_temporal_route_artifacts,
)


def csv_text(rows: list[dict[str, str]]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + stream.getvalue()


def main() -> None:
    route_rows = [
        {
            "detection_id": "1",
            "feature_class": "compact_quality",
            "feature_class_label": "紧凑",
            "quality_passed": "True",
            "route": "quality_passed",
            "numeric_failure_reasons": "",
            "structural_flags": "",
            "flags": "",
        },
        {
            "detection_id": "2",
            "feature_class": "crowded_blend",
            "feature_class_label": "拥挤",
            "quality_passed": "False",
            "route": "rejected_structural_only",
            "numeric_failure_reasons": "",
            "structural_flags": "UNRESOLVED_BLEND",
            "flags": "UNRESOLVED_BLEND",
        },
        {
            "detection_id": "3",
            "feature_class": "weak_or_background",
            "feature_class_label": "弱",
            "quality_passed": "False",
            "route": "rejected_numeric_only",
            "numeric_failure_reasons": "flux_snr",
            "structural_flags": "",
            "flags": "LOW_FLUX_SNR",
        },
    ]
    diagnostic_rows = [
        {
            "detection_id": "2",
            "feature_class": "crowded_blend",
            "feature_class_label": "拥挤",
            "diagnostic_subgroup": "blend_unresolved",
            "diagnostic_subgroup_label": "拥挤",
            "diagnostic_subgroup_definition": "未解析混合",
            "candidate_presence": "12",
            "candidate_same_subgroup_presence": "11",
            "quality_presence": "0",
            "quality_same_subgroup_presence": "0",
        },
        {
            "detection_id": "3",
            "feature_class": "weak_or_background",
            "feature_class_label": "弱",
            "diagnostic_subgroup": "weak_low_flux_snr",
            "diagnostic_subgroup_label": "弱",
            "diagnostic_subgroup_definition": "低通量信噪比",
            "candidate_presence": "15",
            "candidate_same_subgroup_presence": "15",
            "quality_presence": "2",
            "quality_same_subgroup_presence": "1",
        },
    ]
    persistence = {
        "frame_count": 15,
        "required_presence": 12,
        "association_radius_px": 1.0,
        "association_method": "reciprocal_nearest_one_to_one",
    }
    inputs = {
        "route.csv": csv_text(route_rows),
        "diagnostic.csv": csv_text(diagnostic_rows),
        "persistence.json": json.dumps(persistence, ensure_ascii=False, indent=2),
    }
    import tempfile

    with tempfile.TemporaryDirectory(prefix="rst19-feature-gate-temporal-reference-") as temp:
        root = Path(temp)
        paths = {name: root / name for name in inputs}
        for name, value in inputs.items():
            paths[name].write_text(value, encoding="utf-8")
        result = run_feature_gate_temporal_route(
            paths["route.csv"],
            paths["diagnostic.csv"],
            paths["persistence.json"],
            target_ids=("2", "3"),
        )
        expected = result.as_dict()
        expected["route_csv"] = "<ROUTE>"
        expected["diagnostic_sources_csv"] = "<DIAGNOSTIC>"
        expected["persistence_json"] = "<PERSISTENCE>"
        output = write_feature_gate_temporal_route_artifacts(result, root / "output")
        artifacts = {
            path.name: (
                path.read_bytes().decode("utf-8-sig")
                if path.suffix == ".csv"
                else path.read_text(encoding="utf-8")
            )
            for path in output.iterdir()
        }
        artifact_json = json.loads(artifacts["feature_gate_temporal_routes.json"])
        artifact_json["route_csv"] = "<ROUTE>"
        artifact_json["diagnostic_sources_csv"] = "<DIAGNOSTIC>"
        artifact_json["persistence_json"] = "<PERSISTENCE>"
        artifacts["feature_gate_temporal_routes.json"] = json.dumps(
            artifact_json, ensure_ascii=False, indent=2
        )
    payload = {"inputs": inputs, "expected": expected, "artifacts": artifacts}
    destination = Path(__file__).parent / "tests" / "fixtures" / "feature_gate_temporal_route.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
