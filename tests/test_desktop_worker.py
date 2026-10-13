from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rst19.desktop_worker import catalog, preview, registry, run


def make_frame(path: Path) -> None:
    def card(key: str, value: str) -> bytes:
        return f"{key:<8}= {value:>20}".ljust(80).encode("ascii")
    header = b"".join([card("SIMPLE", "T"), card("BITPIX", "16"), card("NAXIS", "2"),
                       card("NAXIS1", "128"), card("NAXIS2", "128"), b"END".ljust(80)]).ljust(2880, b" ")
    rng = np.random.default_rng(19)
    data = rng.normal(1000, 8, (128, 128))
    y, x = np.indices(data.shape)
    for px, py in [(30, 40), (70, 80), (98, 50)]:
        data += 1000 * np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * 1.4 ** 2))
    path.write_bytes(header + np.rint(data).astype(">i2").tobytes())


def test_catalog_covers_all_scientific_entry_points():
    entries = registry()
    tools = catalog()
    assert {tool["name"] for tool in tools} == set(entries)
    assert all("gui" not in target and "launcher" not in target for target in entries.values())
    assert all(tool["fields"] for tool in tools)
    format_audit = next(tool for tool in tools if tool["name"] == "rst19-format-audit")
    assert entries["rst19-format-audit"] == "native:rst19-format-audit"
    assert [field["key"] for field in format_audit["fields"]] == [
        "input_path", "extreme_fraction", "out_dir"
    ]
    sequence_compare = next(tool for tool in tools if tool["name"] == "rst19-feature-sequence-compare")
    assert entries["rst19-feature-sequence-compare"] == "native:rst19-feature-sequence-compare"
    assert sequence_compare["fields"][0]["repeat"] is True
    raw_evidence = next(tool for tool in tools if tool["name"] == "rst19-feature-raw-evidence-audit")
    assert entries["rst19-feature-raw-evidence-audit"] == "native:rst19-feature-raw-evidence-audit"
    assert [field["key"] for field in raw_evidence["fields"]] == [
        "catalog", "raw_summary", "feature_class", "out_dir"
    ]
    assert raw_evidence["fields"][2]["repeat"] is True
    signed_null = next(tool for tool in tools if tool["name"] == "rst19-signed-null-overlap")
    assert entries["rst19-signed-null-overlap"] == "native:rst19-signed-null-overlap"
    assert [field["key"] for field in signed_null["fields"]] == [
        "inputs", "reverse_source_csv", "threshold_sigma", "min_distance",
        "aperture_radius", "psf_fwhm", "background_box_size",
        "background_sample_limit", "min_flux_snr", "min_psf_support_pixels",
        "proposal_mode", "out_dir",
    ]
    signed_null_audit = next(tool for tool in tools if tool["name"] == "rst19-signed-null-audit")
    assert entries["rst19-signed-null-audit"] == "native:rst19-signed-null-audit"
    assert [field["key"] for field in signed_null_audit["fields"]] == [
        "fits_path", "threshold_sigma", "min_distance", "aperture_radius", "psf_fwhm",
        "proposal_mode", "background_box_size", "min_flux_snr", "min_psf_support_pixels",
        "keep_linear_artifacts", "snr_thresholds", "out_dir",
    ]
    signed_null_sequence = next(tool for tool in tools if tool["name"] == "rst19-signed-null-sequence")
    assert entries["rst19-signed-null-sequence"] == "native:rst19-signed-null-sequence"
    assert [field["key"] for field in signed_null_sequence["fields"]] == [
        "inputs", "threshold_sigma", "min_distance", "aperture_radius", "psf_fwhm",
        "proposal_mode", "background_box_size", "min_flux_snr", "min_psf_support_pixels",
        "keep_linear_artifacts", "snr_thresholds", "out_dir",
    ]
    assert signed_null_sequence["fields"][0]["nargs"] == "+"
    migrated_tools = {
        "rst19-gaia-merge": ["catalogs", "out"],
        "rst19-relative-photometry": ["input", "out", "spatial_order", "validation_fraction", "random_state", "mad_threshold"],
        "rst19-wcs-validation": ["data_dir", "catalog_csv", "center_ra", "center_dec", "pixel_scale",
                                  "crpix_x", "crpix_y", "rotation", "parity", "match_radius",
                                  "min_matches", "epoch", "threshold_sigma", "min_distance",
                                  "aperture_radius", "psf_fwhm", "background_box_size", "min_flux_snr",
                                  "max_sources", "keep_linear_artifacts", "out_dir"],
        "rst19-aux-audit": ["input_path", "out_dir"],
        "rst19-source-geometry-audit": ["catalog", "target_class", "control_class", "trials", "seed", "out_dir"],
        "rst19-photometric-report": ["input", "out", "require_wcs", "no_require_wcs", "strict"],
        "rst19-pair-support-audit": ["data_dir", "source_catalog", "primary_id", "secondary_id", "mask_modes",
                                     "repeated_code_values", "shifts_json", "crop_padding_px", "aperture_radius",
                                     "match_radius_px", "threshold_sigma", "min_distance", "psf_fwhm",
                                     "background_box_size", "min_flux_snr", "min_psf_support_pixels",
                                     "proposal_mode", "keep_linear_artifacts", "out_dir"],
    }
    for tool_name, expected_fields in migrated_tools.items():
        tool = next(item for item in tools if item["name"] == tool_name)
        assert entries[tool_name] == f"native:{tool_name}"
        assert [field["key"] for field in tool["fields"]] == expected_fields


def test_format_audit_cannot_fall_back_to_python(tmp_path):
    request = {"operation": "tool", "tool": "rst19-format-audit", "args": [],
               "out_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="必须由 native worker 执行"):
        run(request)


def test_sequence_feature_comparison_cannot_fall_back_to_python(tmp_path):
    request = {"operation": "tool", "tool": "rst19-feature-sequence-compare", "args": [],
               "out_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="必须由 native worker 执行"):
        run(request)


def test_photometric_report_cannot_fall_back_to_python(tmp_path):
    request = {"operation": "tool", "tool": "rst19-photometric-report", "args": [],
               "out_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="必须由 native worker 执行"):
        run(request)


def test_pair_support_audit_cannot_fall_back_to_python(tmp_path):
    request = {"operation": "tool", "tool": "rst19-pair-support-audit", "args": [],
               "out_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="必须由 native worker 执行"):
        run(request)


def test_feature_raw_evidence_audit_cannot_fall_back_to_python(tmp_path):
    request = {"operation": "tool", "tool": "rst19-feature-raw-evidence-audit", "args": [],
               "out_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="必须由 native worker 执行"):
        run(request)


def test_signed_null_overlap_cannot_fall_back_to_python(tmp_path):
    request = {"operation": "tool", "tool": "rst19-signed-null-overlap", "args": [],
               "out_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="必须由 native worker 执行"):
        run(request)


def test_signed_null_audit_cannot_fall_back_to_python(tmp_path):
    request = {"operation": "tool", "tool": "rst19-signed-null-audit", "args": [],
               "out_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="必须由 native worker 执行"):
        run(request)


def test_signed_null_sequence_cannot_fall_back_to_python(tmp_path):
    request = {"operation": "tool", "tool": "rst19-signed-null-sequence", "args": [],
               "out_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="必须由 native worker 执行"):
        run(request)


@pytest.mark.parametrize("tool_name", [
    "rst19-gaia-merge", "rst19-relative-photometry", "rst19-wcs-validation",
    "rst19-aux-audit", "rst19-source-geometry-audit",
])
def test_rust_migrated_entrypoints_cannot_fall_back_to_python(tool_name, tmp_path):
    request = {"operation": "tool", "tool": tool_name, "args": [], "out_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="必须由 native worker 执行"):
        run(request)


def test_frame_job_preserves_science_and_raw_data(tmp_path):
    path = tmp_path / "星图 with spaces.fits"
    make_frame(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = run({"operation": "tool", "tool": "rst19", "args": [str(path)], "out_dir": str(tmp_path / "job")})
    from rst19.pipeline import analyze_frame
    expected = analyze_frame(path, proposal_mode="hybrid").as_dict()
    assert result["result"] == expected
    assert result["result"]["detection"]["quality_count"] >= 3
    assert Path(result["preview"]).is_file()
    assert json.loads(Path(result["result_path"]).read_text(encoding="utf-8")) == expected
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    for mode in ["raw", "noise", "enhanced"]:
        output = tmp_path / mode
        output.mkdir()
        assert preview(str(path), output, mode)["width"] == 128
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_worker_has_exactly_one_terminal_event_for_errors(tmp_path):
    request = {"operation": "tool", "tool": "rst19", "args": [str(tmp_path / "missing.fits")], "out_dir": str(tmp_path / "failed")}
    result = subprocess.run([sys.executable, "-m", "rst19.desktop_worker"], input=json.dumps(request),
                            capture_output=True, text=True, encoding="utf-8")
    events = [json.loads(line) for line in result.stdout.splitlines()]
    terminal = [event for event in events if event["event"] == "done"]
    assert result.returncode != 0
    assert len(terminal) == 1
    assert terminal[0]["ok"] is False
    assert "2" in terminal[0]["error"]


def test_worker_does_not_allow_arbitrary_imports(tmp_path):
    with pytest.raises(ValueError, match="未知研究工具"):
        run({"operation": "tool", "tool": "os:system", "args": [], "out_dir": str(tmp_path)})
