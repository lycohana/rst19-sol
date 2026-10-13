"""Freeze evidence-matrix results and CSV output from the Python baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.feature_evidence_matrix import (  # noqa: E402
    build_feature_evidence_matrix,
    write_feature_evidence_matrix_artifacts,
)
from rst19.feature_evidence_cli import main as cli_main  # noqa: E402,F401


INPUTS = {
    "morphology.csv": (
        "feature_class,feature_class_label,candidate_count,quality_count,quality_fraction,"
        "high_flux_snr_rejection_fraction,spike_or_support_fraction,weak_or_background_fraction\n"
        "range_anomaly,范围异常,20,0,0.0,1.0,0.0,0.0\n"
        "crowded_blend,拥挤,100,0,0.0,1.0,0.5,0.2\n"
        "compact_quality,紧凑,100,90,0.9,0.0,0.0,0.0\n"
    ),
    "cross.csv": (
        "feature_class,feature_class_label,anchor_candidate_count,candidate_presence_ge_required_count,"
        "candidate_presence_fraction,quality_presence_ge_required_count,quality_presence_fraction,"
        "candidate_any_response_fraction,candidate_same_class_response_fraction,"
        "quality_any_response_fraction,quality_same_class_response_fraction\n"
        "range_anomaly,范围异常,20,2,0.1,0,,0.2,0.1,,\n"
        "crowded_blend,拥挤,100,80,0.8,0,,0.8,0.5,,\n"
        "compact_quality,紧凑,100,80,0.8,70,0.7,0.9,0.8,0.95,0.85\n"
    ),
    "psf.csv": (
        "feature_class,feature_class_label,sample_count,leaveout_valid_count,"
        "leaveout_correlation_median,leaveout_correlation_ge_threshold_fraction,leaveout_residual_median\n"
        "range_anomaly,范围异常,2,2,0.75,0.2,0.7\n"
        "crowded_blend,拥挤,2,2,0.2,0.0,0.98\n"
        "compact_quality,紧凑,2,2,0.9,1.0,0.4\n"
    ),
    "spatial.csv": (
        "feature_class,feature_class_label,local_template_available_cell_count,"
        "local_template_source_count_median,local_valid_count,local_fallback_count,"
        "paired_correlation_count,local_correlation_median,local_residual_median,"
        "local_correlation_improved_fraction,local_residual_reduced_fraction,"
        "median_correlation_delta_local_minus_global,median_residual_delta_local_minus_global\n"
        "crowded_blend,拥挤,3,6.0,11,5,11,0.17,0.985,0.64,0.64,0.002,-0.0002\n"
        "compact_quality,紧凑,3,6.0,11,5,11,0.84,0.55,0.45,0.45,-0.002,0.0002\n"
    ),
}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rst19-feature-evidence-reference-") as temporary:
        root = Path(temporary)
        paths = {}
        for name, content in INPUTS.items():
            path = root / name
            path.write_text(content, encoding="utf-8-sig")
            paths[name] = path
        result = build_feature_evidence_matrix(
            paths["morphology.csv"],
            paths["cross.csv"],
            paths["psf.csv"],
            psf_spatial_path=paths["spatial.csv"],
        )
        expected = result.as_dict()
        expected["source_morphology_path"] = "<MORPHOLOGY>"
        expected["feature_cross_path"] = "<CROSS>"
        expected["psf_leaveout_path"] = "<PSF>"
        expected["psf_spatial_path"] = "<SPATIAL>"
        output = root / "artifacts"
        write_feature_evidence_matrix_artifacts(result, output)
        artifact = (output / "feature_evidence_matrix.csv").read_bytes().decode("utf-8")
    payload = {
        "source_hashes": {
            name: hashlib.sha256((ROOT / "src/rst19" / name).read_bytes()).hexdigest()
            for name in ("feature_evidence_matrix.py", "feature_evidence_cli.py")
        },
        "inputs": INPUTS,
        "artifact_csv": artifact,
        "expected": expected,
    }
    destination = ROOT / "backend/tests/fixtures/feature_evidence_matrix.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
