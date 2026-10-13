from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.innovation_package import build_innovation_package, write_innovation_package  # noqa: E402


def row(stratum: str, level: float, candidate: int, quality: int, *, ambiguous: int = 0, unambiguous: int = 24, noise: float = 5.0) -> dict[str, object]:
    return {
        "source_path": "fixture.fits",
        "stratum": stratum,
        "stratum_label": stratum,
        "proposal_mode": "hybrid",
        "psf_model": "empirical",
        "peak_excess_adu": level,
        "injected_count": 24,
        "candidate_recovered_count": candidate,
        "quality_recovered_count": quality,
        "candidate_recall": candidate / 24,
        "quality_recall": quality / 24,
        "ambiguous_injection_count": ambiguous,
        "unambiguous_injected_count": unambiguous,
        "local_noise_adu": noise,
        "trial_count": 3,
        "baseline_candidate_count": 100,
        "baseline_quality_count": 50,
        "mean_candidate_count": 84573.0,
        "mean_quality_count": 29268.666666666668,
    }


def normalize(package: dict[str, object]) -> None:
    package["provenance"]["injection_artifact"]["path"] = "<INPUT>"
    for evidence in package["evidence"][1:]:
        if evidence["id"] == "sequence_relation":
            evidence["source"]["path"] = "<SEQUENCE>"
        elif evidence["id"] == "feature_routing":
            evidence["source"]["path"] = "<FEATURE>"


def main() -> None:
    rows: list[dict[str, object]] = []
    values = {
        "blank": (0, 8, 0, 5.0),
        "high_background": (2, 7, 1, 10.0),
        "edge": (0, 7, 0, 7.5),
    }
    for stratum, (low, middle, high, noise) in values.items():
        for level, candidate, quality in ((24.0, low, 0), (56.0, middle, 0), (128.0, 24, high)):
            rows.append(row(stratum, level, candidate, quality, noise=noise))
    rows.extend([
        row("crowded", 24.0, 3, 0, ambiguous=24, unambiguous=24, noise=20.0),
        row("special_code", 24.0, 0, 0, ambiguous=24, unambiguous=0, noise=67.0),
        row("line", 24.0, 0, 0, ambiguous=24, unambiguous=24, noise=600.0),
    ])
    injection = {"row_count": len(rows), "rows": rows}
    sequence = {
        "relation": {"frame_count": 15, "duration_s": 21.014, "interval_median_s": 1.5},
        "registration": {"max_cumulative_shift_norm_px": 0.254},
        "motion": {"track_count": 4, "moving_track_count": 1, "candidate_count": 9},
    }
    feature = {
        "correlation_threshold": 0.8,
        "parameters": {"pattern_is_not_truth": True, "counting_policy": "candidate_only"},
        "rows": [{"feature_class": "compact_quality"}],
    }
    with tempfile.TemporaryDirectory(prefix="rst19-innovation-package-reference-") as temporary:
        root = Path(temporary)
        inputs = {
            "injection.json": json.dumps(injection, ensure_ascii=False),
            "sequence.json": json.dumps(sequence, ensure_ascii=False),
            "feature.json": json.dumps(feature, ensure_ascii=False),
        }
        paths = {name: root / name for name in inputs}
        for name, text in inputs.items():
            paths[name].write_text(text, encoding="utf-8")
        package = build_innovation_package(
            paths["injection.json"],
            sequence_report=paths["sequence.json"],
            feature_matrix=paths["feature.json"],
        )
        expected = json.loads(json.dumps(package, ensure_ascii=False))
        normalize(expected)
        output = write_innovation_package(package, root / "output")
        artifacts = {
            "innovation_delivery.csv": Path(output["csv"]).read_bytes().decode("utf-8-sig"),
            "innovation_delivery.md": Path(output["markdown"]).read_text(encoding="utf-8"),
        }
        artifact_json = json.loads(Path(output["json"]).read_text(encoding="utf-8"))
        normalize(artifact_json)
        artifacts["innovation_package.json"] = json.dumps(artifact_json, ensure_ascii=False, indent=2) + "\n"
        artifacts["innovation_delivery.md"] = artifacts["innovation_delivery.md"].replace(
            "python -m rst19.innovation_package_cli `",
            "cargo run --manifest-path backend/Cargo.toml --bin rst19-innovation-package -- `",
        )
    payload = {"inputs": inputs, "expected": expected, "artifacts": artifacts}
    destination = Path(__file__).parent / "tests" / "fixtures" / "innovation_package.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
