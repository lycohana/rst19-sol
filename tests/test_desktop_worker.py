from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from rst19.desktop_worker import NATIVE_TOOL_NAMES, _schema_registry, catalog, registry, run


def test_catalog_covers_all_scientific_entry_points():
    entries = registry()
    schema_entries = _schema_registry()
    tools = catalog()
    # The Rust worker owns the complete UI catalog. The Python compatibility
    # catalog exposes only commands with a Python parser/schema.
    assert {tool["name"] for tool in tools} == set(entries) & set(schema_entries)
    assert all("gui" not in target and "launcher" not in target for target in entries.values())
    assert all(tool["fields"] for tool in tools)
    root = Path(__file__).resolve().parents[1]
    static_catalog = json.loads((root / "desktop/src/tool_catalog.json").read_text(encoding="utf-8"))
    static_native = {tool["name"] for tool in static_catalog if tool["native"]}
    assert static_native == NATIVE_TOOL_NAMES
    backend_source = (root / "desktop/src/backend.rs").read_text(encoding="utf-8").split("\nimpl Backend")[0]
    routed_native = set(re.findall(r'"(rst19(?:-[a-z0-9-]+)?)"', backend_source))
    assert routed_native == static_native
    single_frame = next(tool for tool in tools if tool["name"] == "rst19")
    assert entries["rst19"] == "native:rst19"
    assert single_frame["native"] is True
    assert [field["key"] for field in single_frame["fields"]][0] == "fits_path"
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
        "rst19-feature-parameter-sensitivity": ["path", "threshold_levels", "flux_snr_levels",
                                                "baseline_threshold_sigma", "baseline_min_flux_snr",
                                                "min_distance", "aperture_radius", "psf_fwhm",
                                                "background_box_size", "background_sample_limit",
                                                "min_psf_support_pixels", "proposal_mode", "max_sources",
                                                "association_radius_px", "out_dir"],
        "rst19-feature-cross": ["source_summary_csv", "persistence_csv", "class_transition_csv",
                                "high_snr_threshold", "out_dir"],
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


def test_feature_parameter_sensitivity_cannot_fall_back_to_python(tmp_path):
    request = {"operation": "tool", "tool": "rst19-feature-parameter-sensitivity", "args": [],
               "out_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="必须由 native worker 执行"):
        run(request)


def test_feature_cross_cannot_fall_back_to_python(tmp_path):
    request = {"operation": "tool", "tool": "rst19-feature-cross", "args": [],
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


def test_single_frame_tool_cannot_fall_back_to_python(tmp_path):
    request = {
        "operation": "tool",
        "tool": "rst19",
        "args": [str(tmp_path / "not-opened.fits")],
        "out_dir": str(tmp_path / "job"),
    }
    with pytest.raises(ValueError, match="必须由 native worker 执行"):
        run(request)


@pytest.mark.parametrize("tool_name", sorted(NATIVE_TOOL_NAMES))
def test_every_native_catalog_tool_refuses_python_fallback(tool_name, tmp_path):
    request = {"operation": "tool", "tool": tool_name, "args": [], "out_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="必须由 native worker 执行"):
        run(request)


def test_worker_has_exactly_one_terminal_event_for_errors(tmp_path):
    request = {"operation": "tool", "tool": "rst19", "args": [], "out_dir": str(tmp_path / "failed")}
    result = subprocess.run([sys.executable, "-m", "rst19.desktop_worker"], input=json.dumps(request),
                            capture_output=True, text=True, encoding="utf-8")
    events = [json.loads(line) for line in result.stdout.splitlines()]
    terminal = [event for event in events if event["event"] == "done"]
    assert result.returncode != 0
    assert len(terminal) == 1
    assert terminal[0]["ok"] is False
    assert "native worker" in terminal[0]["error"]


def test_worker_does_not_allow_arbitrary_imports(tmp_path):
    with pytest.raises(ValueError, match="未知研究工具"):
        run({"operation": "tool", "tool": "os:system", "args": [], "out_dir": str(tmp_path)})
