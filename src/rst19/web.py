"""本地星图分析工作台的轻量 HTTP 服务。"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

from .fits import auxiliary_mask, read_fits
from .pipeline import analyze_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "doc" / "00-项目资料" / "原始数据"
UI_DIR = Path(__file__).resolve().parent / "ui"


def _bounded_float(query: dict[str, list[str]], name: str, default: float, low: float, high: float) -> float:
    raw = query.get(name, [str(default)])[0]
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"参数 {name} 不是有效数字") from exc
    if not low <= value <= high:
        raise ValueError(f"参数 {name} 应在 {low} 至 {high} 之间")
    return value


def _bounded_int(query: dict[str, list[str]], name: str, default: int, low: int, high: int) -> int:
    raw = query.get(name, [str(default)])[0]
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"参数 {name} 不是有效整数") from exc
    if not low <= value <= high:
        raise ValueError(f"参数 {name} 应在 {low} 至 {high} 之间")
    return value


def _safe_frame_path(data_dir: Path, filename: str | None) -> Path:
    if not filename:
        raise ValueError("缺少 filename 参数")
    root = data_dir.resolve()
    candidate = (root / unquote(filename)).resolve()
    if candidate.parent != root or candidate.suffix.lower() != ".fits" or not candidate.is_file():
        raise ValueError("只能读取数据目录内的 FITS 文件")
    return candidate


def _preview_png(path: Path, max_side: int = 960) -> bytes:
    frame = read_fits(path)
    values = np.asarray(frame.data, dtype=np.float64).copy()
    values[auxiliary_mask(values.shape)] = np.nan
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        raise ValueError("图像没有可预览的有限像素")
    low, high = np.percentile(valid, [1.0, 99.8])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    image = Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), mode="L")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _handler_for(data_dir: Path):
    class Rst19Handler(BaseHTTPRequestHandler):
        server_version = "rst19-ui/0.1"

        def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8", status)

        def _error(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
            self._send_json({"ok": False, "error": message}, status)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            request = urlparse(self.path)
            query = parse_qs(request.query)
            try:
                if request.path == "/api/health":
                    self._send_json({"ok": True, "mode": "offline", "data_dir": str(data_dir)})
                    return
                if request.path == "/api/frames":
                    frames = [
                        {"filename": path.name, "size": path.stat().st_size}
                        for path in sorted(data_dir.glob("*.fits"))
                    ]
                    self._send_json({"ok": True, "frames": frames})
                    return
                if request.path == "/api/preview":
                    path = _safe_frame_path(data_dir, query.get("filename", [None])[0])
                    max_side = _bounded_int(query, "max_side", 960, 320, 1600)
                    encoded = base64.b64encode(_preview_png(path, max_side)).decode("ascii")
                    self._send_json({"ok": True, "filename": path.name, "data_url": f"data:image/png;base64,{encoded}"})
                    return
                if request.path == "/api/analyze":
                    path = _safe_frame_path(data_dir, query.get("filename", [None])[0])
                    threshold_sigma = _bounded_float(query, "threshold_sigma", 5.0, 2.0, 12.0)
                    min_distance = _bounded_int(query, "min_distance", 4, 1, 20)
                    max_sources = _bounded_int(query, "max_sources", 800, 50, 5000)
                    result = analyze_frame(
                        path,
                        threshold_sigma=threshold_sigma,
                        min_distance=min_distance,
                        max_sources=max_sources,
                    )
                    self._send_json({"ok": True, **result.as_dict()})
                    return
                self._serve_static(request.path)
            except FileNotFoundError as exc:
                self._error(str(exc), HTTPStatus.NOT_FOUND)
            except (OSError, ValueError) as exc:
                self._error(str(exc))

        def _serve_static(self, request_path: str) -> None:
            relative = unquote(request_path.lstrip("/")) or "index.html"
            candidate = (UI_DIR / relative).resolve()
            if UI_DIR.resolve() not in candidate.parents and candidate != UI_DIR.resolve():
                self._error("静态资源路径无效", HTTPStatus.NOT_FOUND)
                return
            if not candidate.is_file():
                candidate = UI_DIR / "index.html"
            body = candidate.read_bytes()
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            if content_type == "text/html":
                content_type += "; charset=utf-8"
            self._send_bytes(body, content_type)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[rst19-ui] {self.address_string()} - {format % args}")

    return Rst19Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 rst19 本地星图分析工作台")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="FITS 数据目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"rst19-ui: 数据目录不存在：{data_dir}")
        return 2
    server = ThreadingHTTPServer((args.host, args.port), _handler_for(data_dir))
    print(f"rst19-ui: http://{args.host}:{args.port}")
    print(f"rst19-ui: data_dir={data_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nrst19-ui: stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
