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
    # `rst19` is a Rust executable. Keep its parser only as the desktop form
    # schema; the desktop backend dispatches the calculation to Rust.
    entries.setdefault("rst19", "rst19.cli:main")
    # The format audit is Rust-only. Keep its form contract here until the
    # remaining Python catalog bootstrap is replaced by native Rust metadata.
    entries.setdefault("rst19-format-audit", "native:rst19-format-audit")
    entries.setdefault("rst19-feature-sequence-compare", "native:rst19-feature-sequence-compare")
    entries.setdefault("rst19-feature-raw-evidence-audit", "native:rst19-feature-raw-evidence-audit")
    entries.setdefault("rst19-signed-null-overlap", "native:rst19-signed-null-overlap")
    entries.setdefault("rst19-signed-null-audit", "native:rst19-signed-null-audit")
    return {name: target for name, target in entries.items()
            if target not in ("rst19.gui:main", "rst19.desktop_launcher:main")}


def module_for(name: str):
    target = registry().get(name)
    if target is None:
        raise ValueError(f"未知研究工具：{name}")
    if target.startswith("native:"):
        raise ValueError(f"{name} 已迁移到 Rust，必须由 native worker 执行")
    return importlib.import_module(target.split(":")[0])


def catalog() -> list[dict]:
    tools = []
    for name, target in registry().items():
        if target == "native:rst19-format-audit":
            tools.append({"name": name,
                          "description": "审计 FITS 尺寸、字节序、特殊值和辅助位置—速度自洽性，不修改检测器",
                          "fields": [
                              {"key": "input_path", "flag": "", "help": "单个 FITS 或包含 FITS 的目录",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "extreme_fraction", "flag": "--extreme-fraction",
                               "help": "统计有符号 16 bit 正负极值的相对阈值，默认 0.9",
                               "required": False, "default": 0.9, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "out_dir", "flag": "--out-dir", "help": "产物输出目录",
                               "required": False, "default": "tmp/format-audit", "choices": [],
                               "kind": "value", "nargs": None, "path": True, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-feature-sequence-compare":
            tools.append({"name": name,
                          "description": "比较多组已生成的序列特征持久性 JSON，不重新运行 FITS 检测",
                          "fields": [
                              {"key": "run", "flag": "--run", "help": "运行标签和 JSON 路径，NAME=JSON；至少两组",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": True},
                              {"key": "out_dir", "flag": "--out-dir", "help": "比较表输出目录",
                               "required": False, "default": "tmp/sequence-feature-parameter-comparison",
                               "choices": [], "kind": "value", "nargs": None, "path": True,
                               "repeat": False},
                          ]})
            continue
        if target == "native:rst19-feature-raw-evidence-audit":
            tools.append({"name": name,
                          "description": "连接源目录与原始 FITS 抽样 CSV，比较各特征类别的局部证据",
                          "fields": [
                              {"key": "catalog", "flag": "", "help": "rst19-sources 导出的 source_catalog.csv",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "raw_summary", "flag": "", "help": "stratified_raw_source_summary.csv",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "feature_class", "flag": "--feature-class",
                               "help": "需要汇总的 feature_class；可重复传入",
                               "required": False, "default": [], "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": True},
                              {"key": "out_dir", "flag": "--out-dir", "help": "CSV/JSON 输出目录",
                               "required": False, "default": "tmp/feature-raw-evidence-audit",
                               "choices": [], "kind": "value", "nargs": None, "path": True,
                               "repeat": False},
                          ]})
            continue
        if target == "native:rst19-signed-null-overlap":
            tools.append({"name": name,
                          "description": "逐帧查找符号反相质量源在原图正向检测中的局部近邻",
                          "fields": [
                              {"key": "inputs", "flag": "", "help": "FITS 文件或包含 FITS 的目录",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": "+", "path": True, "repeat": False},
                              {"key": "reverse_source_csv", "flag": "--reverse-source-csv",
                               "help": "signed_null_sequence_quality_sources.csv",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "threshold_sigma", "flag": "--threshold-sigma", "help": "检测阈值 sigma",
                               "required": False, "default": 4.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_distance", "flag": "--min-distance", "help": "候选峰最小间距",
                               "required": False, "default": 4, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "aperture_radius", "flag": "--aperture-radius", "help": "测光孔径半径",
                               "required": False, "default": 4, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "psf_fwhm", "flag": "--psf-fwhm", "help": "检测器 PSF FWHM",
                               "required": False, "default": 2.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "background_box_size", "flag": "--background-box-size", "help": "背景块尺寸",
                               "required": False, "default": 128, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "background_sample_limit", "flag": "--background-sample-limit", "help": "背景采样上限",
                               "required": False, "default": 100000, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_flux_snr", "flag": "--min-flux-snr", "help": "质量层最低孔径通量 SNR",
                               "required": False, "default": 5.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_psf_support_pixels", "flag": "--min-psf-support-pixels", "help": "最低 PSF 支持像素数",
                               "required": False, "default": 3, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "proposal_mode", "flag": "--proposal-mode", "help": "候选提议方式",
                               "required": False, "default": "hybrid", "choices": ["gaussian", "hybrid", "ensemble"],
                               "kind": "value", "nargs": None, "path": False, "repeat": False},
                              {"key": "out_dir", "flag": "--out-dir", "help": "CSV/JSON 输出目录",
                               "required": False, "default": "tmp/signed-null-positive-overlap", "choices": [],
                               "kind": "value", "nargs": None, "path": True, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-signed-null-audit":
            tools.append({"name": name,
                          "description": "用真实 FITS 的背景符号反相图做源级 signed-tail leakage 对照，不输出源目录",
                          "fields": [
                              {"key": "fits_path", "flag": "", "help": "输入 FITS 文件",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "threshold_sigma", "flag": "--threshold-sigma", "help": "匹配滤波候选阈值 sigma",
                               "required": False, "default": 4.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_distance", "flag": "--min-distance", "help": "候选峰最小间距 pixel",
                               "required": False, "default": 4, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "aperture_radius", "flag": "--aperture-radius", "help": "孔径测光半径 pixel",
                               "required": False, "default": 4, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "psf_fwhm", "flag": "--psf-fwhm", "help": "Gaussian 匹配滤波 FWHM pixel",
                               "required": False, "default": 2.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "proposal_mode", "flag": "--proposal-mode", "help": "宽筛选提案方式",
                               "required": False, "default": "hybrid", "choices": ["gaussian", "hybrid", "ensemble"],
                               "kind": "value", "nargs": None, "path": False, "repeat": False},
                              {"key": "background_box_size", "flag": "--background-box-size", "help": "局部背景/RMS 网格大小 pixel",
                               "required": False, "default": 128, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_flux_snr", "flag": "--min-flux-snr", "help": "质量层孔径通量 SNR 下限",
                               "required": False, "default": 5.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_psf_support_pixels", "flag": "--min-psf-support-pixels",
                               "help": "候选峰中心 3x3 内的最少 PSF 支持像素", "required": False,
                               "default": 3, "choices": [], "kind": "value", "nargs": None,
                               "path": False, "repeat": False},
                              {"key": "keep_linear_artifacts", "flag": "--keep-linear-artifacts",
                               "help": "不标记线状结构为 LINE_ARTIFACT", "required": False,
                               "default": False, "choices": [], "kind": "bool", "nargs": 0,
                               "path": False, "toggle_value": True, "repeat": False},
                              {"key": "snr_thresholds", "flag": "--snr-thresholds", "help": "输出 SNR 分档下限，必须严格递增",
                               "required": False, "default": [4.0, 5.0, 7.0, 10.0, 15.0, 20.0],
                               "choices": [], "kind": "value", "nargs": "+", "path": False,
                               "repeat": False},
                              {"key": "out_dir", "flag": "--out-dir", "help": "JSON/CSV 输出目录",
                               "required": False, "default": "tmp/signed-null-audit", "choices": [],
                               "kind": "value", "nargs": None, "path": True, "repeat": False},
                          ]})
            continue
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
