"""Compare native pair-pixel topology results and artifacts with Python."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rst19.empirical_pair import load_detection_catalog_csv  # noqa: E402
from rst19.fits import read_fits  # noqa: E402
from rst19.pair_pixel_topology import (  # noqa: E402
    run_pair_pixel_topology_audit,
    write_pair_pixel_topology_artifacts,
)


def card(key: str, value: str) -> bytes:
    return f"{key:<8}= {value:>20}".ljust(80).encode("ascii")


def make_frame(path: Path) -> None:
    image = np.full((40, 40), 10, dtype=np.int16)
    image[20, 20] = 130
    image[20, 21] = 99
    image[21, 19] = 80
    image[21, 18] = 70
    image[21, 17] = 50
    image[21, 16] = 40
    header = b"".join(
        (
            card("SIMPLE", "T"),
            card("BITPIX", "16"),
            card("NAXIS", "2"),
            card("NAXIS1", "40"),
            card("NAXIS2", "40"),
            b"END".ljust(80),
        )
    ).ljust(2880, b" ")
    payload = struct.pack(f">{image.size}h", *image.reshape(-1).tolist())
    payload = payload.ljust(((len(payload) + 2879) // 2880) * 2880, b"\0")
    path.write_bytes(header + payload)


CATALOG = (
    "detection_id,x,y,peak,flux,background,noise,snr,quality_passed,peak_x,peak_y,repeated_code_values\n"
    "1,21,20,99,100,10,2,20,false,21,20,99\n"
    "2,16,21,40,100,10,2,20,false,16,21,99\n"
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
    with tempfile.TemporaryDirectory(prefix="rst19-pair-pixel-topology-") as temporary:
        root = Path(temporary)
        frame_path = root / "近邻原始像素.fits"
        catalog_path = root / "source catalog.csv"
        make_frame(frame_path)
        catalog_path.write_text(CATALOG, encoding="utf-8")
        frame_before = hashlib.sha256(frame_path.read_bytes()).hexdigest()
        catalog_before = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
        sources = load_detection_catalog_csv(catalog_path)
        by_id = {source.detection_id: source for source in sources}
        frame = read_fits(frame_path)
        baseline = run_pair_pixel_topology_audit(
            frame,
            by_id[1],
            by_id[2],
            patch_padding_px=4,
            sigma_levels=(3.0, 5.0),
            raw_maximum_neighborhoods_px=(3,),
        )
        expected = baseline.as_dict()
        python_out = root / "python"
        native_out = root / "native"
        write_pair_pixel_topology_artifacts(baseline, python_out)
        args = [
            str(frame_path),
            "--source-catalog",
            str(catalog_path),
            "--primary-id",
            "1",
            "--secondary-id",
            "2",
            "--patch-padding-px",
            "4",
            "--sigma-levels",
            "3",
            "5",
            "--raw-maximum-neighborhoods-px",
            "3",
            "--out-dir",
            str(native_out),
        ]
        worker_payload = invoke(
            worker,
            {
                "operation": "tool",
                "tool": "rst19-pair-pixel-topology",
                "args": args,
                "out_dir": str((root / "worker-job").resolve()),
            },
        )
        engine_payload = invoke(
            engine,
            {
                "operation": "audit.pair_pixel_topology",
                "frame_path": str(frame_path),
                "primary_detection": by_id[1].as_dict(),
                "secondary_detection": by_id[2].as_dict(),
                "options": {
                    "patch_padding_px": 4,
                    "sigma_levels": [3.0, 5.0],
                    "raw_maximum_neighborhoods_px": [3],
                },
            },
        )
        assert worker_payload["result"] == expected
        assert engine_payload == expected
        artifact_names = (
            "pair_pixel_topology_components.csv",
            "pair_pixel_topology_maxima.csv",
        )
        for name in artifact_names:
            assert (native_out / name).read_bytes() == (python_out / name).read_bytes(), name
        assert json.loads((native_out / "pair_pixel_topology.json").read_text(encoding="utf-8")) == expected
        native_svg = (native_out / "pair_pixel_topology.svg").read_bytes()
        python_svg = (python_out / "pair_pixel_topology.svg").read_bytes()
        if native_svg != python_svg:
            native_lines = native_svg.decode("utf-8").splitlines()
            python_lines = python_svg.decode("utf-8").splitlines()
            difference = "\n".join(difflib.unified_diff(python_lines, native_lines, fromfile="python.svg", tofile="native.svg"))
            raise AssertionError(f"SVG differs:\n{difference}")
        assert hashlib.sha256(frame_path.read_bytes()).hexdigest() == frame_before
        assert hashlib.sha256(catalog_path.read_bytes()).hexdigest() == catalog_before
        result = {
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "frame_sha256": frame_before,
            "catalog_sha256": catalog_before,
            "result_fields_match": True,
            "component_csv_byte_identical": True,
            "maximum_csv_byte_identical": True,
            "json_fields_match": True,
            "svg_byte_identical": True,
            "frame_and_catalog_unchanged": True,
            "patch_bounds": expected["patch_bounds"],
            "component_row_count": len(expected["component_rows"]),
            "maximum_row_count": len(expected["maximum_rows"]),
            "repeated_codes": expected["repeated_code_values"],
        }
    (out / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.worker, args.engine, args.out), ensure_ascii=False, indent=2))
