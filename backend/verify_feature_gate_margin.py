"""Compare the native feature-gate margin audit with its Python baseline."""
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
from rst19.feature_gate_margin import (  # noqa: E402
    run_feature_gate_margin,
    write_feature_gate_margin_artifacts,
)


FIELDS = (
    "detection_id,feature_class,feature_class_label,quality_passed,flux_snr,"
    "psf_support_pixels,fwhm,ellipticity,sharpness,footprint_pixels\n"
)
CATALOG = FIELDS + (
    "1,compact_quality,紧凑,true,12,5,2.5,0.1,0.3,20\n"
    "2,compact_quality,忽略,false,3,4,2.5,0.1,0.3,20\n"
    "3,shape_outlier,形状,false,25,6,3.5,0.7,0.3,20\n"
)
THRESHOLDS = {
    "min_flux_snr": 6.0,
    "min_psf_support_pixels": 4,
    "min_fwhm": 0.7,
    "max_fwhm": 10.0,
    "max_ellipticity": 0.6,
    "min_sharpness": 0.01,
    "max_sharpness": 0.8,
    "min_footprint_pixels": 3,
}
CLI_THRESHOLD_ARGS = (
    "--min-flux-snr", str(THRESHOLDS["min_flux_snr"]),
    "--min-psf-support-pixels", str(THRESHOLDS["min_psf_support_pixels"]),
    "--min-fwhm", str(THRESHOLDS["min_fwhm"]),
    "--max-fwhm", str(THRESHOLDS["max_fwhm"]),
    "--max-ellipticity", str(THRESHOLDS["max_ellipticity"]),
    "--min-sharpness", str(THRESHOLDS["min_sharpness"]),
    "--max-sharpness", str(THRESHOLDS["max_sharpness"]),
    "--min-footprint-pixels", str(THRESHOLDS["min_footprint_pixels"]),
)


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
    with tempfile.TemporaryDirectory(prefix="rst19-feature-gate-margin-") as temporary:
        root = Path(temporary)
        catalog = root / "source catalog.csv"
        catalog.write_bytes(("\ufeff" + CATALOG).encode("utf-8"))
        before = hashlib.sha256(catalog.read_bytes()).hexdigest()
        baseline = run_feature_gate_margin(catalog, **THRESHOLDS)
        expected = baseline.as_dict()
        python_out = root / "python"
        native_out = root / "native"
        write_feature_gate_margin_artifacts(baseline, python_out)
        worker_payload = run_process(
            worker,
            {
                "operation": "tool",
                "tool": "rst19-feature-gate-margin",
                "args": [
                    str(catalog),
                    *CLI_THRESHOLD_ARGS,
                    "--out-dir",
                    str(native_out),
                ],
                "out_dir": str((root / "job").resolve()),
            },
        )
        engine_payload = run_process(
            engine,
            {
                "operation": "audit.feature_gate_margin",
                "catalog_path": str(catalog),
                "thresholds": THRESHOLDS,
            },
        )
        assert worker_payload["result"] == expected
        assert engine_payload == expected
        assert (native_out / "feature_gate_margins.csv").read_bytes() == (
            python_out / "feature_gate_margins.csv"
        ).read_bytes()
        native_json = json.loads(
            (native_out / "feature_gate_margins.json").read_text(encoding="utf-8")
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
            "feature_class_count": len({"compact_quality", "shape_outlier"}),
            "margin_row_count": len(expected["rows"]),
            "thresholds_match": True,
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
