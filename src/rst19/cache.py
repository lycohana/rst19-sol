"""单帧检测结果的本地压缩缓存。"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from .detection import Detection, DetectionResult
from .fits import FitsFrame
from .photometry import FaintestSource

# 几何孔径的 MASKED/SATURATED 判定和线状伪迹质量语义发生过变化；
# 旧缓存不能继续冒充当前检测结果，必须自动失效并重新计算。
CACHE_VERSION = 3
CACHE_SUFFIXES = {".gz", ".json", ".tmp"}


def cache_key(
    frame: FitsFrame,
    *,
    threshold_sigma: float,
    min_distance: int,
    aperture_radius: int,
    max_sources: int | None,
    zero_point: float | None,
    psf_fwhm: float = 3.0,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    min_fwhm: float = 0.8,
    max_fwhm: float = 12.0,
    max_ellipticity: float = 0.65,
    min_sharpness: float = 0.005,
    max_sharpness: float = 0.85,
    min_footprint_pixels: int = 2,
    gain_e_per_adu: float | None = None,
    read_noise_adu: float = 0.0,
    mask_zero_pixels: bool | None = None,
    reject_linear_artifacts: bool = True,
) -> str:
    """根据输入文件状态和检测参数生成稳定缓存键。"""

    stat = frame.path.stat()
    descriptor = {
        "cache_version": CACHE_VERSION,
        "path": str(frame.path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "threshold_sigma": threshold_sigma,
        "min_distance": min_distance,
        "aperture_radius": aperture_radius,
        "max_sources": max_sources,
        "zero_point": zero_point,
        "psf_fwhm": psf_fwhm,
        "background_box_size": background_box_size,
        "min_flux_snr": min_flux_snr,
        "min_fwhm": min_fwhm,
        "max_fwhm": max_fwhm,
        "max_ellipticity": max_ellipticity,
        "min_sharpness": min_sharpness,
        "max_sharpness": max_sharpness,
        "min_footprint_pixels": min_footprint_pixels,
        "gain_e_per_adu": gain_e_per_adu,
        "read_noise_adu": read_noise_adu,
        "mask_zero_pixels": mask_zero_pixels,
        "reject_linear_artifacts": reject_linear_artifacts,
    }
    encoded = json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json.gz"


def load_analysis(cache_dir: Path, key: str, frame: FitsFrame) -> Any | None:
    """读取缓存并恢复为绑定当前 FITS frame 的 FrameAnalysis。"""

    path = cache_path(cache_dir, key)
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("cache_version") != CACHE_VERSION:
            return None
        detection_payload = payload["detection"]
        sources = tuple(
            Detection(
                detection_id=int(source["detection_id"]),
                x=float(source["x"]),
                y=float(source["y"]),
                peak=float(source["peak"]),
                flux=float(source["flux"]),
                background=float(source["background"]),
                noise=float(source["noise"]),
                snr=float(source["snr"]),
                fwhm=float(source["fwhm"]) if source["fwhm"] is not None else None,
                flags=tuple(str(flag) for flag in source.get("flags", [])),
                flux_error=float(source["flux_error"]) if source.get("flux_error") is not None else None,
                flux_snr=float(source["flux_snr"]) if source.get("flux_snr") is not None else None,
                filter_snr=float(source["filter_snr"]) if source.get("filter_snr") is not None else None,
                fwhm_x=float(source["fwhm_x"]) if source.get("fwhm_x") is not None else None,
                fwhm_y=float(source["fwhm_y"]) if source.get("fwhm_y") is not None else None,
                ellipticity=float(source["ellipticity"]) if source.get("ellipticity") is not None else None,
                sharpness=float(source["sharpness"]) if source.get("sharpness") is not None else None,
                footprint_pixels=int(source["footprint_pixels"]) if source.get("footprint_pixels") is not None else None,
                quality_passed=bool(source.get("quality_passed", True)),
            )
            for source in detection_payload["sources"]
        )
        detection = DetectionResult(
            image_shape=tuple(int(value) for value in detection_payload["image_shape"]),
            background=float(detection_payload["background"]),
            noise=float(detection_payload["noise"]),
            threshold=float(detection_payload["threshold"]),
            candidate_count=int(detection_payload["candidate_count"]),
            sources=sources,
            parameters={key: value for key, value in detection_payload.get("parameters", {}).items()},
            quality_count=int(detection_payload["quality_count"]) if detection_payload.get("quality_count") is not None else None,
        )
        faintest_payload = payload.get("faintest_detected")
        faintest = (
            FaintestSource(
                detection_id=int(faintest_payload["detection_id"]),
                x=float(faintest_payload["x"]),
                y=float(faintest_payload["y"]),
                flux=float(faintest_payload["flux"]),
                snr=float(faintest_payload["snr"]),
                instrumental_magnitude=float(faintest_payload["instrumental_magnitude"]),
                calibrated_magnitude=(
                    float(faintest_payload["calibrated_magnitude"])
                    if faintest_payload["calibrated_magnitude"] is not None
                    else None
                ),
                flags=tuple(str(flag) for flag in faintest_payload.get("flags", [])),
                flux_snr=float(faintest_payload["flux_snr"]) if faintest_payload.get("flux_snr") is not None else None,
                flux_rate=float(faintest_payload["flux_rate"]) if faintest_payload.get("flux_rate") is not None else None,
            )
            if faintest_payload is not None
            else None
        )
        from .pipeline import FrameAnalysis

        return FrameAnalysis(frame=frame, detection=detection, matching=None, faintest=faintest)
    except (OSError, EOFError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_analysis(cache_dir: Path, key: str, analysis: Any) -> Path:
    """以 gzip JSON 保存完整检测结果，方便之后重画叠加和导出。"""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, key)
    temporary_path = path.with_suffix(".tmp")
    payload = {"cache_version": CACHE_VERSION, **analysis.as_dict()}
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def clear_cache(cache_dir: Path) -> int:
    """删除本项目缓存目录中的当前及旧版缓存文件，不触碰原始数据。"""

    if not cache_dir.is_dir():
        return 0
    removed = 0
    for path in cache_dir.iterdir():
        if path.is_file() and path.suffix in CACHE_SUFFIXES:
            path.unlink()
            removed += 1
    try:
        cache_dir.rmdir()
    except OSError:
        pass
    return removed
