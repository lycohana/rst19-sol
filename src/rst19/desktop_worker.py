"""One isolated native-desktop job per process; stdout is strictly JSONL.

The worker invokes the existing public command parsers and scientific routines.
It deliberately does not import Tkinter or change the original FITS data.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

PROTOCOL = sys.stdout
if hasattr(PROTOCOL, "reconfigure"):
    PROTOCOL.reconfigure(encoding="utf-8", errors="strict")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="strict")


def emit(kind: str, **fields: object) -> None:
    PROTOCOL.write(json.dumps({"event": kind, **fields}, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    PROTOCOL.flush()


def registry() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file():
        import tomllib
        entries = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    else:
        entries = {ep.name: ep.value for ep in importlib.metadata.distribution("rst19-sol").entry_points
                   if ep.group == "console_scripts"}
    return {name: target for name, target in entries.items()
            if target not in ("rst19.gui:main", "rst19.desktop_launcher:main")}


def module_for(name: str):
    target = registry().get(name)
    if target is None:
        raise ValueError(f"未知研究工具：{name}")
    return importlib.import_module(target.split(":")[0])


def catalog() -> list[dict]:
    tools = []
    for name in registry():
        module = module_for(name)
        parser = module.build_parser()
        fields = []
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            flags = action.option_strings
            fields.append({
                "key": action.dest, "flag": next((f for f in flags if f.startswith("--")), ""),
                "help": action.help or "", "required": action.required,
                "default": None if action.default is argparse.SUPPRESS else action.default,
                "choices": list(action.choices) if action.choices is not None else [],
                "kind": "bool" if action.nargs == 0 else "value",
                "nargs": action.nargs, "path": action.type is Path,
                "toggle_value": getattr(action, "const", True) if action.nargs == 0 else None,
                "repeat": isinstance(action, (argparse._AppendAction, argparse._AppendConstAction)),
            })
        tools.append({"name": name, "description": parser.description or name, "fields": fields})
    return tools


class JobLog:
    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")
        self.pending = ""

    def write(self, value: str) -> int:
        self.file.write(value)
        self.file.flush()
        self.pending += value
        while "\n" in self.pending or len(self.pending) > 4096:
            if "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
            else:
                line, self.pending = self.pending[:4096], self.pending[4096:]
            emit("log", text=line)
        return len(value)

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        if self.pending:
            emit("log", text=self.pending)
        self.file.close()


def preview(path: str, out: Path, mode: str) -> dict:
    import numpy as np
    from PIL import Image, ImageFilter
    from .fits import read_fits, auxiliary_mask
    frame = read_fits(path)
    values = np.asarray(frame.data, dtype=np.float32).copy()
    values[auxiliary_mask(values.shape)] = np.nan
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        raise ValueError("图像没有可预览的有限像素")
    lo, q25, mid, q75, hi = np.percentile(valid, [0.5, 25, 50, 75, 99.8])
    noise = max(float(q75 - q25) / 1.3489795, 1e-6)
    gamma = 1.0
    if mode == "noise":
        lo, hi, gamma = max(lo, mid - 2 * noise), mid + 7 * noise, 0.82
    elif mode == "enhanced":
        lo, hi, gamma = max(lo, mid - noise), max(hi, mid + 12 * noise), 0.7
    pixels = np.nan_to_num(np.power(np.clip((values - lo) / max(float(hi - lo), 1), 0, 1), gamma))
    image = Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).convert("RGB")
    if mode == "enhanced":
        image = image.filter(ImageFilter.GaussianBlur(0.35))
    # Keep full resolution so zooming preserves original pixel locations.
    target = out / "preview.png"
    image.save(target)
    return {"preview": str(target), "width": frame.width, "height": frame.height,
            "path": str(frame.path), "header": dict(frame.header),
            "auxiliary": frame.auxiliary.as_dict() if frame.auxiliary else None}


def run(request: dict) -> dict:
    out = Path(request["out_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if request["operation"] == "preview":
        return preview(request["path"], out, request.get("mode", "raw"))
    name = request["tool"]
    module = module_for(name)
    argv = request.get("args", [])
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("args 必须是字符串数组")
    # argparse validates inputs before any scientific work.
    parser = module.build_parser()
    parsed = parser.parse_args(argv)
    result_path = out / "result.json"
    if any("--json-out" in a.option_strings for a in parser._actions) and "--json-out" not in argv:
        argv = [*argv, "--json-out", str(result_path)]
        parsed = parser.parse_args(argv)
    stdout_path = out / "stdout.txt"
    stderr_path = out / "stderr.txt"
    stdout, stderr = JobLog(stdout_path), JobLog(stderr_path)
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = module.main(argv)
    finally:
        stdout.close()
        stderr.close()
    if code not in (None, 0):
        raise RuntimeError(f"{name} 返回错误码 {code}；详见 {stderr_path}")
    json_out = getattr(parsed, "json_out", None)
    named_out = getattr(parsed, "out", None)
    if named_out is not None and Path(named_out).suffix.lower() != ".json":
        named_out = None
    actual_result = json_out or named_out or result_path
    payload = None
    if Path(actual_result).is_file():
        payload = json.loads(Path(actual_result).read_text(encoding="utf-8"))
    else:
        try:
            payload = json.loads(stdout_path.read_text(encoding="utf-8"))
            result_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        except json.JSONDecodeError:
            pass
    response = {"tool": name, "out_dir": str(out), "result_path": str(actual_result), "result": payload}
    # Only single-frame detection carries the exact image/result identity.
    if name == "rst19" and payload is not None:
        response.update(preview(str(parsed.fits_path), out, request.get("mode", "raw")))
    (out / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    return response


def main() -> int:
    try:
        if "--catalog" in sys.argv:
            import numpy, scipy, PIL
            emit("done", ok=True, payload={"tools": catalog(), "environment": {
                "python": sys.version.split()[0], "numpy": numpy.__version__,
                "scipy": scipy.__version__, "pillow": PIL.__version__,
                "frozen": bool(getattr(sys, "frozen", False))}})
        else:
            request = json.loads(sys.stdin.readline())
            emit("done", ok=True, payload=run(request))
        return 0
    except (Exception, SystemExit) as exc:
        emit("done", ok=False, error=str(exc), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
