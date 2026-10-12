"""Compare the Rust layered-catalog merge against its Python reference."""
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
from rst19.catalog_merge import merge_catalog_csvs  # noqa: E402


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


def write_catalog(path: Path, rows: list[dict[str, str]]) -> None:
    import csv

    fields = tuple(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_audit(path: Path, *, complete: bool, center_ra: float = 129.5) -> None:
    import csv

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        count = sum(1 for _ in csv.DictReader(stream))
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "catalog": "Gaia DR3",
                "complete": complete,
                "center_ra_deg": center_ra,
                "center_dec_deg": -1.8,
                "search_radius_deg": 6.92,
                "min_g_mag": 5.0,
                "max_g_mag": 18.0,
                "csv_row_count": count,
                "csv_sha256": digest,
            }
        ),
        encoding="utf-8",
    )


def prepare_case(root: Path, name: str) -> tuple[list[Path], Path]:
    first = root / f"{name}-bright.csv"
    second = root / f"{name}-deep.csv"
    base = {
        "source_id": "2",
        "ra_deg": "129.6",
        "dec_deg": "-1.7",
        "magnitude": "7.1",
        "color": "0.8",
        "color_name": "BP-RP",
        "photometric_system": "Gaia Vega",
        "photometric_band": "G",
        "note": "亮星,定标层",
    }
    deep = dict(base, magnitude="7.1000000001", parallax_mas="2.3")
    if name == "conflict":
        deep["magnitude"] = "7.9"
    first_rows = [
        dict(base, source_id="1", magnitude="6.2"),
        base,
    ]
    second_rows = [deep, dict(base, source_id="3", magnitude="15.4")]
    write_catalog(first, first_rows)
    write_catalog(second, second_rows)

    if name != "missing-audit":
        write_audit(first, complete=True, center_ra=129.5)
        write_audit(second, complete=name != "incomplete", center_ra=129.5)
    if name == "footprint-mismatch":
        write_audit(second, complete=True, center_ra=130.0)
    if name == "stale-binding":
        first.write_text(
            first.read_text(encoding="utf-8-sig").replace("129.5", "129.5001"),
            encoding="utf-8",
        )
    if name == "alternate-sidecar":
        for path in (first, second):
            sidecar = path.with_suffix(path.suffix + ".meta.json")
            sidecar.rename(path.with_suffix(path.suffix + ".tiled.meta.json"))
    if name == "invalid-sidecar":
        first.with_suffix(first.suffix + ".meta.json").write_text("{broken", encoding="utf-8")
    if name == "non-object-sidecar":
        first.with_suffix(first.suffix + ".meta.json").write_text("[]", encoding="utf-8")
    if name == "ra-wrap":
        write_audit(second, complete=True, center_ra=-230.5)
    return [first, second], root / f"{name}-merged.csv"


def normalize_paths(value: object) -> object:
    if isinstance(value, list):
        return [normalize_paths(item) for item in value]
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if key in {"output_path", "audit_path"}:
                normalized[key] = f"<{key}>"
            else:
                normalized[key] = normalize_paths(item)
        return normalized
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(worker: Path, engine: Path, out: Path) -> dict[str, object]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    case_reports = []
    with tempfile.TemporaryDirectory(prefix="rst19-catalog-merge-") as temporary:
        root = Path(temporary)
        for name in (
            "complete",
            "conflict",
            "stale-binding",
            "incomplete",
            "missing-audit",
            "footprint-mismatch",
            "alternate-sidecar",
            "invalid-sidecar",
            "non-object-sidecar",
            "ra-wrap",
        ):
            case_root = root / name
            case_root.mkdir()
            paths, baseline_output = prepare_case(case_root, name)
            before = {str(path): sha256(path) for path in paths}
            sidecars = [
                Path(f"{path}.meta.json")
                for path in paths
            ] + [Path(f"{path}.tiled.meta.json") for path in paths]
            before.update({str(sidecar): sha256(sidecar) for sidecar in sidecars if sidecar.is_file()})

            baseline_result = merge_catalog_csvs(paths, baseline_output)
            baseline_path = baseline_result.audit_path
            expected = json.loads(baseline_path.read_text(encoding="utf-8"))

            engine_output = case_root / "native" / "merged.csv"
            engine_payload = invoke(
                engine,
                {
                    "operation": "catalog.merge_csvs",
                    "input_paths": [str(path.resolve()) for path in paths],
                    "output_path": str(engine_output.resolve()),
                },
            )
            worker_output = case_root / "worker" / "merged.csv"
            worker_payload = invoke(
                worker,
                {
                    "operation": "tool",
                    "tool": "rst19-gaia-merge",
                    "args": [
                        *(str(path.resolve()) for path in paths),
                        "--out",
                        str(worker_output.resolve()),
                    ],
                    "out_dir": str((case_root / "worker-job").resolve()),
                },
            )
            assert normalize_paths(engine_payload) == normalize_paths(expected), name
            assert normalize_paths(worker_payload["result"]) == normalize_paths(expected), name
            assert engine_output.read_bytes() == baseline_output.read_bytes(), name
            assert worker_output.read_bytes() == baseline_output.read_bytes(), name
            for native_output in (engine_output, worker_output):
                native_json = json.loads(
                    native_output.with_suffix(native_output.suffix + ".meta.json").read_text(
                        encoding="utf-8"
                    )
                )
                assert normalize_paths(native_json) == normalize_paths(expected), name
            after = {str(path): sha256(path) for path in paths}
            after.update({str(sidecar): sha256(sidecar) for sidecar in sidecars if sidecar.is_file()})
            assert after == before, name
            case_reports.append(
                {
                    "case": name,
                    "complete": expected["complete"],
                    "rows": expected["row_count"],
                    "duplicates": expected["duplicate_count"],
                    "conflicts": expected["conflict_count"],
                    "engine_and_worker_fields_match": True,
                    "csv_byte_identical": True,
                    "sidecar_fields_match": True,
                    "inputs_unchanged": True,
                }
            )

    report = {
        "worker_sha256": sha256(worker),
        "engine_sha256": sha256(engine),
        "cases": case_reports,
        "all_result_fields_match": True,
        "all_csv_artifacts_byte_identical": True,
        "all_inputs_unchanged": True,
    }
    (out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.worker, arguments.engine, arguments.out), ensure_ascii=False, indent=2))
