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
    assert len(tools) == 78
    assert all("gui" not in target and "launcher" not in target for target in entries.values())
    assert all(tool["fields"] for tool in tools)
    format_audit = next(tool for tool in tools if tool["name"] == "rst19-format-audit")
    assert entries["rst19-format-audit"] == "native:rst19-format-audit"
    assert [field["key"] for field in format_audit["fields"]] == [
        "input_path", "extreme_fraction", "out_dir"
    ]


def test_format_audit_cannot_fall_back_to_python(tmp_path):
    request = {"operation": "tool", "tool": "rst19-format-audit", "args": [],
               "out_dir": str(tmp_path)}
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
