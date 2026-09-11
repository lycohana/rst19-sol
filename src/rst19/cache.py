"""单帧检测结果的本地压缩缓存。"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections.abc import Mapping as ABCMapping
from pathlib import Path
from typing import Any, Mapping, Sequence

from .detection import Detection, DetectionResult
from .fits import FitsFrame
from .matching import CatalogMatch, MatchResult
from .photometry import AbsoluteMagnitudeEstimate, FaintestSource, PhotometricCalibration, SourcePhotometry

# 几何孔径的 MASKED/SATURATED 判定、局部零值坏像素容错、线状伪迹质量语义、
# PSF 支持审计和背景网格抽样
# 口径发生过变化；
# 旧缓存不能继续冒充当前检测结果，必须自动失效并重新计算。
# 本轮（16）修正 CODE_PATTERN 触发条件：只有候选峰本身的像素值落在
# 全幅异常重复高位码上、且孔径含异常负值时才拒绝；此前“孔径内含重复码
# 即拒”会误伤真实亮星的饱和/溢出出血列。
# 本轮（17）近邻 Gaussian 联合门控把带值域审计旗标的强邻峰纳入配对，
# 并用测量质心而不是整数峰坐标寻找重叠邻域；旧缓存不能继续冒充当前结果。
# 本轮（18）加入光度标定证据、逐源 m_inst/m_cal/M 层和视差质量状态；
# 旧缓存没有这些字段，必须重新计算，避免 UI 显示混合口径。
# 本轮（19）把原始输入内容摘要、缓存载荷身份和星表匹配结果纳入边界；
# 仅凭路径/大小/mtime 或丢失 matching 的旧载荷都不能继续复用。
# 本轮（20）把 Gaia 目录质量字段贯通到光度定标，并在载荷中记录被
# 排除的参考星原因；旧的 calibration payload 没有这组证据，必须失效。
# 本轮（21）把绝对星等的距离来源和 GSP-Phot 距离区间纳入逐源载荷；
# 旧缓存即使有数值，也不能继续冒充带来源的模型/视差结果。
CACHE_VERSION = 21
# 序列结果新增逐帧线状筛选审计和配准工作集口径；此前切换到只在
# 16 位 FITS 序列路径使用 float32 中间阵列并调整快速背景统计迭代数，
# 本轮又增加确定性网格抽样、注册时间中值补提案、时序候选响应分层，
# 以及时序补提案在预测坐标附近的 ±1 px 局部 PSF 峰定位；此前再加入
# 序列级固定异常码审计结果及其对候选的孔径影响关联；本轮加入高位
# 重复码与局部异常负值联合门控，以及极近 Gaussian 双 PSF 质量门。
# 本轮（35）新增叠加参考图暗星恢复层（evidence_level="stack_faint"）：
# 序列结果现在含 stack_faint 轨迹及其参数，旧缓存不具备该口径。
# 本轮（36）新增高速点状目标补充关联（evidence_level="fast_point_motion"），
# 旧序列缓存没有该轨迹层，必须重新计算，不能把旧的 transient 结果冒充
# 当前的点源运动判定。
# 本轮（37）把高速点源的宽筛从截断后的 quality_sources 扩展到完整候选峰
# 工作集，并回到原始 ADU 做选择性 PSF/flux SNR 复核；旧的 36 缓存可能
# 漏掉被 6000 条源级工作集截断的高速点，不能继续复用。
# 旧缓存不具备相同计算口径，必须重新计算，否则 UI 可能把不同精度的
# 结果混在一起。
# 本轮（38）加入显式可选的 15 帧相对光度标尺；旧序列缓存不含这项
# 创新证据，必须失效后重新计算，避免 GUI 显示不完整的相对标尺。
# 本轮（39）把整组 FITS 内容/目录清单和序列科学上下文纳入身份，并要求
# 保存的输入清单与当前文件一致，避免旧目录内容在同一 key 下静默复用。
SEQUENCE_CACHE_VERSION = 39
# 其中 ``.npz`` 是注册 15 帧大图的数组缓存；它和检测/序列 JSON 一样
# 属于可随时删除的派生数据，绝不能与原始 FITS 混为一谈。
CACHE_SUFFIXES = {".gz", ".json", ".tmp", ".npz"}

_IDENTITY_VERSION = 1
_HASH_CHUNK_BYTES = 1024 * 1024
_PATH_PARAMETER_KEYS = {
    "path",
    "file",
    "filename",
    "directory",
    "dir",
    "data_dir",
    "catalog_path",
    "catalog_file",
    "catalog_csv",
    "wcs_path",
    "extinction_map_path",
    "input_directory",
}


def _file_identity(path: str | Path) -> dict[str, object]:
    """返回文件的可复现身份，而不是只依赖易碰撞的 stat 三元组。

    内容摘要是故意放在 key 中的：FITS/CSV 被原地替换且保留大小和 mtime
    时，缓存也必须失效。若文件在摘要期间发生变化则直接报错，让调用方
    重新尝试，而不是把一个不完整的文件写成可复用结果。
    """

    resolved = Path(path).resolve()
    before = resolved.stat()
    if not resolved.is_file():
        raise OSError(f"cache input is not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while True:
            chunk = stream.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError(f"cache input changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _directory_identity(path: str | Path) -> dict[str, object]:
    """返回目录及其递归文件内容的确定性身份。

    该函数只用于显式传入的目录上下文（例如星表目录或消光图目录）。
    序列 FITS 的默认目录清单由 ``_sequence_input_identity`` 限定为直接
    ``*.fits`` 文件，避免同目录的缓存/说明文件无关地触发重算。
    """

    resolved = Path(path).resolve()
    directory_stat = resolved.stat()
    if not resolved.is_dir():
        raise OSError(f"cache input is not a directory: {resolved}")
    entries: list[dict[str, object]] = []
    children = sorted(
        (child for child in resolved.rglob("*") if child.is_file()),
        key=lambda child: child.relative_to(resolved).as_posix(),
    )
    for child in children:
        entries.append(
            {
                "relative_path": child.relative_to(resolved).as_posix(),
                "file": _file_identity(child),
            }
        )
    return {
        "path": str(resolved),
        "mtime_ns": int(directory_stat.st_mtime_ns),
        "files": entries,
    }


def _path_identity(value: str | Path) -> dict[str, object]:
    """将 PathLike 上下文转为存在性、路径和内容均明确的身份。"""

    resolved = Path(value).resolve()
    if resolved.is_file():
        return {"kind": "file", "file": _file_identity(resolved)}
    if resolved.is_dir():
        return {"kind": "directory", "directory": _directory_identity(resolved)}
    return {"kind": "missing", "path": str(resolved)}


def _is_path_parameter_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.strip().lower()
    return normalized in _PATH_PARAMETER_KEYS or normalized.endswith(("_path", "_file", "_dir", "_directory"))


def _canonical_identity(value: object, *, depth: int = 0) -> object:
    """递归规范化 key 上下文，覆盖 Path、目录、对象和非字符串映射键。

    星表/WCS 对象使用其 ``as_dict`` 作为身份；不可序列化的对象会让 key
    生成失败，而不会退化成不稳定的 ``repr``，从而避免漏掉科学参数。
    """

    if depth > 32:
        raise ValueError("cache identity nesting is too deep")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cache identity contains a non-finite float")
        return value
    if isinstance(value, os.PathLike):
        return _path_identity(os.fspath(value))
    if isinstance(value, bytes):
        return {"kind": "bytes", "sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, ABCMapping):
        items: list[tuple[object, object]] = []
        for key, item in value.items():
            canonical_key = _canonical_identity(key, depth=depth + 1)
            if _is_path_parameter_key(key) and isinstance(item, (str, os.PathLike)):
                canonical_value = _path_identity(os.fspath(item))
            else:
                canonical_value = _canonical_identity(item, depth=depth + 1)
            items.append((canonical_key, canonical_value))
        items.sort(
            key=lambda item: json.dumps(
                item[0], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        )
        return {"kind": "mapping", "items": [[key, item] for key, item in items]}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_canonical_identity(item, depth=depth + 1) for item in value],
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonical_identity(item, depth=depth + 1) for item in value]
        items.sort(key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return {"kind": "set", "items": items}
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return {
            "kind": "object",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _canonical_identity(as_dict(), depth=depth + 1),
        }
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _canonical_identity(item_method(), depth=depth + 1)
        except (TypeError, ValueError):
            pass
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"cache identity contains unsupported value {type(value).__name__}") from exc
    return value


def _fits_directory_manifest(
    directory: Path,
    known_files: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    """列出一个输入父目录中的直接 FITS 文件及其内容身份。"""

    resolved = directory.resolve()
    directory_stat = resolved.stat()
    entries: list[dict[str, object]] = []
    for child in sorted(resolved.iterdir(), key=lambda item: item.name.casefold()):
        if not child.is_file() or child.suffix.lower() not in {".fits", ".fit", ".fts"}:
            continue
        child_resolved = str(child.resolve())
        file_identity = known_files.get(child_resolved) or _file_identity(child)
        entries.append({"name": child.name, "file": file_identity})
    return {
        "path": str(resolved),
        "mtime_ns": int(directory_stat.st_mtime_ns),
        "files": entries,
    }


def _sequence_input_identity(paths: Sequence[str | Path]) -> dict[str, object]:
    """返回有序 FITS 输入和其所在目录内容的身份清单。"""

    file_identities = [_file_identity(path) for path in paths]
    known_files = {str(item["path"]): item for item in file_identities}
    directories = sorted({str(Path(path).resolve().parent) for path in paths})
    return {
        "identity_version": _IDENTITY_VERSION,
        "kind": "sequence",
        "frames": file_identities,
        "directories": [_fits_directory_manifest(Path(directory), known_files) for directory in directories],
    }


def _frame_input_identity(frame: FitsFrame) -> dict[str, object]:
    return {
        "identity_version": _IDENTITY_VERSION,
        "kind": "frame",
        "file": _file_identity(frame.path),
    }


def _scientific_context(
    *,
    catalog: object | None,
    catalog_path: str | Path | None,
    catalog_identity: object | None,
    wcs: object | None,
    epoch: float | None,
    match_radius_px: float | None,
    photometric_system: str | None,
    photometric_band: str | None,
    photometric_color_name: str | None,
    photometric_color_order: int | None,
    photometric_min_calibrators: int | None,
    parallax: object | None,
    parallax_zero_point_mas: float | None,
    max_fractional_parallax_error: float | None,
    extinction: object | None,
    extinction_model: object | None,
    extinction_mag: float | None,
    extinction_error_mag: float | None,
    relative_photometry: object | None,
    relative_photometry_parameters: Mapping[str, Any] | None,
    input_directory: str | Path | None,
    cache_context: Mapping[str, Any] | None,
) -> object:
    """统一收集星表、WCS、测光、距离/消光和相对光度上下文。"""

    return _canonical_identity(
        {
            "catalog": catalog,
            "catalog_path": catalog_path,
            "catalog_identity": catalog_identity,
            "wcs": wcs,
            "epoch": epoch,
            "match_radius_px": match_radius_px,
            "photometric_system": photometric_system,
            "photometric_band": photometric_band,
            "photometric_color_name": photometric_color_name,
            "photometric_color_order": photometric_color_order,
            "photometric_min_calibrators": photometric_min_calibrators,
            "parallax": parallax,
            "parallax_zero_point_mas": parallax_zero_point_mas,
            "max_fractional_parallax_error": max_fractional_parallax_error,
            "extinction": extinction,
            "extinction_model": extinction_model,
            "extinction_mag": extinction_mag,
            "extinction_error_mag": extinction_error_mag,
            "relative_photometry": relative_photometry,
            "relative_photometry_parameters": relative_photometry_parameters,
            "input_directory": input_directory,
            "cache_context": cache_context,
        }
    )


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
    min_psf_support_pixels: int = 3,
    gain_e_per_adu: float | None = None,
    read_noise_adu: float = 0.0,
    mask_zero_pixels: bool | None = None,
    allow_partial_zero_mask: bool | None = None,
    reject_linear_artifacts: bool = True,
    proposal_mode: str = "gaussian",
    dog_threshold_sigma: float | None = None,
    dog_min_peak_sigma: float = 2.0,
    dog_blend_radius_factor: float = 2.5,
    starlet_threshold_sigma: float | None = None,
    starlet_min_peak_sigma: float = 2.5,
    deblend_delta_bic_min: float = 10.0,
    deblend_component_snr_min: float = 5.0,
    deblend_primary_snr_min: float = 12.0,
    deblend_min_residual_sigma: float = 4.0,
    deblend_search_radius_factor: float = 2.0,
    enable_local_deblend: bool = False,
    refine_local_background: bool = True,
    use_float32: bool = False,
    fit_photometry: bool = False,
    photometric_system: str | None = None,
    photometric_band: str | None = None,
    photometric_color_name: str | None = None,
    photometric_color_order: int = 1,
    photometric_min_calibrators: int = 6,
    parallax_zero_point_mas: float = 0.0,
    max_fractional_parallax_error: float = 0.2,
    catalog: object | None = None,
    catalog_path: str | Path | None = None,
    catalog_identity: object | None = None,
    wcs: object | None = None,
    epoch: float | None = None,
    match_radius_px: float | None = 3.0,
    parallax: object | None = None,
    extinction: object | None = None,
    extinction_model: object | None = None,
    extinction_mag: float | None = None,
    extinction_error_mag: float | None = None,
    relative_photometry: object | None = None,
    relative_photometry_parameters: Mapping[str, Any] | None = None,
    input_directory: str | Path | None = None,
    cache_context: Mapping[str, Any] | None = None,
) -> str:
    """根据输入内容、检测参数和星等链上下文生成稳定缓存键。

    ``catalog``/``catalog_path``、``wcs``、``epoch``、光度系统/波段、
    视差/消光策略以及相对光度参数都属于科学身份的一部分。调用方可用
    ``cache_context`` 扩展尚未有专用形参的参数；PathLike 文件和目录会
    记录内容摘要，而不是只记录路径。
    """

    frame_identity = _frame_input_identity(frame)
    descriptor = {
        "cache_version": CACHE_VERSION,
        "input": frame_identity,
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
        "min_psf_support_pixels": min_psf_support_pixels,
        "gain_e_per_adu": gain_e_per_adu,
        "read_noise_adu": read_noise_adu,
        "mask_zero_pixels": mask_zero_pixels,
        "allow_partial_zero_mask": allow_partial_zero_mask,
        "reject_linear_artifacts": reject_linear_artifacts,
        "proposal_mode": proposal_mode,
        "dog_threshold_sigma": dog_threshold_sigma,
        "dog_min_peak_sigma": dog_min_peak_sigma,
        "dog_blend_radius_factor": dog_blend_radius_factor,
        "starlet_threshold_sigma": starlet_threshold_sigma,
        "starlet_min_peak_sigma": starlet_min_peak_sigma,
        "deblend_delta_bic_min": deblend_delta_bic_min,
        "deblend_component_snr_min": deblend_component_snr_min,
        "deblend_primary_snr_min": deblend_primary_snr_min,
        "deblend_min_residual_sigma": deblend_min_residual_sigma,
        "deblend_search_radius_factor": deblend_search_radius_factor,
        "enable_local_deblend": bool(enable_local_deblend),
        "refine_local_background": refine_local_background,
        "use_float32": use_float32,
        "fit_photometry": bool(fit_photometry),
        "photometric_system": photometric_system,
        "photometric_band": photometric_band,
        "photometric_color_name": photometric_color_name,
        "photometric_color_order": photometric_color_order,
        "photometric_min_calibrators": photometric_min_calibrators,
        "parallax_zero_point_mas": parallax_zero_point_mas,
        "max_fractional_parallax_error": max_fractional_parallax_error,
        "scientific_context": _scientific_context(
            catalog=catalog,
            catalog_path=catalog_path,
            catalog_identity=catalog_identity,
            wcs=wcs,
            epoch=epoch,
            match_radius_px=match_radius_px,
            photometric_system=photometric_system,
            photometric_band=photometric_band,
            photometric_color_name=photometric_color_name,
            photometric_color_order=photometric_color_order,
            photometric_min_calibrators=photometric_min_calibrators,
            parallax=parallax,
            parallax_zero_point_mas=parallax_zero_point_mas,
            max_fractional_parallax_error=max_fractional_parallax_error,
            extinction=extinction,
            extinction_model=extinction_model,
            extinction_mag=extinction_mag,
            extinction_error_mag=extinction_error_mag,
            relative_photometry=relative_photometry,
            relative_photometry_parameters=relative_photometry_parameters,
            input_directory=input_directory,
            cache_context=cache_context,
        ),
    }
    encoded = json.dumps(
        descriptor,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
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
        required_keys = {
            "cache_version",
            "cache_kind",
            "cache_key",
            "input_identity",
            "detection",
            "matching",
            "faintest_detected",
            "photometric_calibration",
            "source_photometry",
        }
        if (
            not isinstance(payload, ABCMapping)
            or not required_keys.issubset(payload)
            or payload.get("cache_version") != CACHE_VERSION
            or payload.get("cache_kind") != "frame-analysis"
            or payload.get("cache_key") != key
            or payload.get("input_identity") != _frame_input_identity(frame)
        ):
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
                psf_support_pixels=(
                    int(source["psf_support_pixels"])
                    if source.get("psf_support_pixels") is not None
                    else None
                ),
                quality_passed=bool(source.get("quality_passed", True)),
                peak_x=float(source["peak_x"]) if source.get("peak_x") is not None else None,
                peak_y=float(source["peak_y"]) if source.get("peak_y") is not None else None,
                centroid_shift_px=(
                    float(source["centroid_shift_px"])
                    if source.get("centroid_shift_px") is not None
                    else None
                ),
                proposal_methods=tuple(str(method) for method in source.get("proposal_methods", ())),
                proposal_scales=tuple(float(scale) for scale in source.get("proposal_scales", ())),
                proposal_snr=float(source["proposal_snr"]) if source.get("proposal_snr") is not None else None,
                nearest_gaussian_px=(
                    float(source["nearest_gaussian_px"])
                    if source.get("nearest_gaussian_px") is not None
                    else None
                ),
                deblend_delta_bic=(
                    float(source["deblend_delta_bic"])
                    if source.get("deblend_delta_bic") is not None
                    else None
                ),
                deblend_component_snr=(
                    float(source["deblend_component_snr"])
                    if source.get("deblend_component_snr") is not None
                    else None
                ),
                repeated_code_count=(
                    int(source["repeated_code_count"])
                    if source.get("repeated_code_count") is not None
                    else None
                ),
                range_anomaly_pixel_count=(
                    int(source["range_anomaly_pixel_count"])
                    if source.get("range_anomaly_pixel_count") is not None
                    else None
                ),
                repeated_code_values=tuple(
                    int(value) for value in source.get("repeated_code_values", ())
                ),
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
        matching_payload = payload["matching"]
        matching = None
        if matching_payload is not None:
            if not isinstance(matching_payload, ABCMapping):
                return None
            matching = MatchResult(
                matches=tuple(
                    CatalogMatch(
                        detection_id=int(match["detection_id"]),
                        source_id=str(match["source_id"]),
                        detection_x=float(match["detection_x"]),
                        detection_y=float(match["detection_y"]),
                        predicted_x=float(match["predicted_x"]),
                        predicted_y=float(match["predicted_y"]),
                        residual_px=float(match["residual_px"]),
                        catalog_magnitude=(
                            float(match["catalog_magnitude"])
                            if match.get("catalog_magnitude") is not None
                            else None
                        ),
                        catalog_magnitude_error=(
                            float(match["catalog_magnitude_error"])
                            if match.get("catalog_magnitude_error") is not None
                            else None
                        ),
                        catalog_color=(
                            float(match["catalog_color"])
                            if match.get("catalog_color") is not None
                            else None
                        ),
                        catalog_color_name=match.get("catalog_color_name"),
                        photometric_system=match.get("photometric_system"),
                        photometric_band=match.get("photometric_band"),
                    )
                    for match in matching_payload.get("matches", [])
                ),
                unmatched_detection_ids=tuple(
                    int(value) for value in matching_payload.get("unmatched_detection_ids", [])
                ),
                unmatched_catalog_ids=tuple(
                    str(value) for value in matching_payload.get("unmatched_catalog_ids", [])
                ),
                max_residual_px=_float_or_none(matching_payload.get("max_residual_px")),
                rms_residual_px=_float_or_none(matching_payload.get("rms_residual_px")),
                inlier_ratio=float(matching_payload["inlier_ratio"]),
                radius_px=float(matching_payload["radius_px"]),
                assignment_mode=str(matching_payload.get("assignment_mode", "global")),
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
                instrumental_magnitude_error=(
                    float(faintest_payload["instrumental_magnitude_error"])
                    if faintest_payload.get("instrumental_magnitude_error") is not None
                    else None
                ),
                photometric_system=faintest_payload.get("photometric_system"),
                photometric_band=faintest_payload.get("photometric_band"),
                calibration_status=str(faintest_payload.get("calibration_status", "INSTRUMENTAL")),
                calibrated_magnitude_error=(
                    float(faintest_payload["calibrated_magnitude_error"])
                    if faintest_payload.get("calibrated_magnitude_error") is not None
                    else None
                ),
                selection_scope=str(faintest_payload.get("selection_scope", "QUALITY_DETECTIONS")),
                eligible_candidate_count=int(faintest_payload.get("eligible_candidate_count", 0)),
                calibrated_candidate_count=int(faintest_payload.get("calibrated_candidate_count", 0)),
                absolute_magnitude=(
                    AbsoluteMagnitudeEstimate(
                        value=(
                            float(faintest_payload["absolute_magnitude"]["value"])
                            if faintest_payload["absolute_magnitude"].get("value") is not None
                            else None
                        ),
                        error=(
                            float(faintest_payload["absolute_magnitude"]["error"])
                            if faintest_payload["absolute_magnitude"].get("error") is not None
                            else None
                        ),
                        distance_pc=(
                            float(faintest_payload["absolute_magnitude"]["distance_pc"])
                            if faintest_payload["absolute_magnitude"].get("distance_pc") is not None
                            else None
                        ),
                        corrected_parallax_mas=(
                            float(faintest_payload["absolute_magnitude"]["corrected_parallax_mas"])
                            if faintest_payload["absolute_magnitude"].get("corrected_parallax_mas") is not None
                            else None
                        ),
                        extinction_mag=faintest_payload["absolute_magnitude"].get("extinction_mag"),
                        status=str(faintest_payload["absolute_magnitude"]["status"]),
                        flags=tuple(str(flag) for flag in faintest_payload["absolute_magnitude"].get("flags", [])),
                        distance_source=faintest_payload["absolute_magnitude"].get("distance_source"),
                        distance_lower_pc=(
                            float(faintest_payload["absolute_magnitude"]["distance_lower_pc"])
                            if faintest_payload["absolute_magnitude"].get("distance_lower_pc") is not None
                            else None
                        ),
                        distance_upper_pc=(
                            float(faintest_payload["absolute_magnitude"]["distance_upper_pc"])
                            if faintest_payload["absolute_magnitude"].get("distance_upper_pc") is not None
                            else None
                        ),
                    )
                    if faintest_payload.get("absolute_magnitude") is not None
                    else None
                ),
            )
            if faintest_payload is not None
            else None
        )
        calibration_payload = payload.get("photometric_calibration")
        calibration = (
            PhotometricCalibration(
                photometric_system=str(calibration_payload["photometric_system"]),
                photometric_band=str(calibration_payload["photometric_band"]),
                color_name=calibration_payload.get("color_name"),
                color_order=int(calibration_payload["color_order"]),
                coefficients=tuple(float(value) for value in calibration_payload.get("coefficients", ())),
                calibrator_count=int(calibration_payload["calibrator_count"]),
                inlier_count=int(calibration_payload["inlier_count"]),
                validation_count=int(calibration_payload["validation_count"]),
                fit_rms_mag=(
                    float(calibration_payload["fit_rms_mag"])
                    if calibration_payload.get("fit_rms_mag") is not None
                    else None
                ),
                validation_rms_mag=(
                    float(calibration_payload["validation_rms_mag"])
                    if calibration_payload.get("validation_rms_mag") is not None
                    else None
                ),
                residual_mad_mag=(
                    float(calibration_payload["residual_mad_mag"])
                    if calibration_payload.get("residual_mad_mag") is not None
                    else None
                ),
                color_min=(float(calibration_payload["color_min"]) if calibration_payload.get("color_min") is not None else None),
                color_max=(float(calibration_payload["color_max"]) if calibration_payload.get("color_max") is not None else None),
                status=str(calibration_payload["status"]),
                flags=tuple(str(flag) for flag in calibration_payload.get("flags", [])),
                catalog_filter_counts=tuple(
                    (str(item.get("reason")), int(item.get("count", 0)))
                    for item in calibration_payload.get("catalog_filter_counts", [])
                    if isinstance(item, ABCMapping) and item.get("reason") is not None
                ),
                validation_bias_mag=_float_or_none(calibration_payload.get("validation_bias_mag")),
                max_fit_rms_mag=_float_or_none(calibration_payload.get("max_fit_rms_mag")),
                max_validation_rms_mag=_float_or_none(calibration_payload.get("max_validation_rms_mag")),
                max_residual_mad_mag=_float_or_none(calibration_payload.get("max_residual_mad_mag")),
                max_validation_bias_mag=_float_or_none(calibration_payload.get("max_validation_bias_mag")),
                min_validation_count=int(calibration_payload.get("min_validation_count", 0)),
                max_source_residual_mag=_float_or_none(calibration_payload.get("max_source_residual_mag")),
                calibration_sample_roles=tuple(
                    (str(item.get("source_id")), str(item.get("role")))
                    for item in calibration_payload.get("calibration_sample_roles", [])
                    if isinstance(item, ABCMapping)
                    and item.get("source_id") is not None
                    and item.get("role") is not None
                ),
            )
            if calibration_payload is not None
            else None
        )
        source_photometry = tuple(
            SourcePhotometry(
                detection_id=int(row["detection_id"]),
                source_id=row.get("source_id"),
                instrumental_magnitude=(
                    float(row["instrumental_magnitude"])
                    if row.get("instrumental_magnitude") is not None
                    else None
                ),
                instrumental_magnitude_error=(
                    float(row["instrumental_magnitude_error"])
                    if row.get("instrumental_magnitude_error") is not None
                    else None
                ),
                calibrated_magnitude=(
                    float(row["calibrated_magnitude"])
                    if row.get("calibrated_magnitude") is not None
                    else None
                ),
                calibrated_magnitude_error=(
                    float(row["calibrated_magnitude_error"])
                    if row.get("calibrated_magnitude_error") is not None
                    else None
                ),
                catalog_magnitude=(
                    float(row["catalog_magnitude"])
                    if row.get("catalog_magnitude") is not None
                    else None
                ),
                catalog_magnitude_error=(
                    float(row["catalog_magnitude_error"])
                    if row.get("catalog_magnitude_error") is not None
                    else None
                ),
                color=float(row["color"]) if row.get("color") is not None else None,
                color_name=row.get("color_name"),
                photometric_system=row.get("photometric_system"),
                photometric_band=row.get("photometric_band"),
                absolute_magnitude=(
                    AbsoluteMagnitudeEstimate(
                        value=(float(row["absolute_magnitude"]["value"]) if row["absolute_magnitude"].get("value") is not None else None),
                        error=(float(row["absolute_magnitude"]["error"]) if row["absolute_magnitude"].get("error") is not None else None),
                        distance_pc=(float(row["absolute_magnitude"]["distance_pc"]) if row["absolute_magnitude"].get("distance_pc") is not None else None),
                        corrected_parallax_mas=(float(row["absolute_magnitude"]["corrected_parallax_mas"]) if row["absolute_magnitude"].get("corrected_parallax_mas") is not None else None),
                        extinction_mag=row["absolute_magnitude"].get("extinction_mag"),
                        status=str(row["absolute_magnitude"]["status"]),
                        flags=tuple(str(flag) for flag in row["absolute_magnitude"].get("flags", [])),
                        distance_source=row["absolute_magnitude"].get("distance_source"),
                        distance_lower_pc=(
                            float(row["absolute_magnitude"]["distance_lower_pc"])
                            if row["absolute_magnitude"].get("distance_lower_pc") is not None
                            else None
                        ),
                        distance_upper_pc=(
                            float(row["absolute_magnitude"]["distance_upper_pc"])
                            if row["absolute_magnitude"].get("distance_upper_pc") is not None
                            else None
                        ),
                    )
                    if row.get("absolute_magnitude") is not None
                    else None
                ),
                status=str(row["status"]),
                flags=tuple(str(flag) for flag in row.get("flags", [])),
                photometric_residual_mag=_float_or_none(row.get("photometric_residual_mag")),
                photometric_residual_limit_mag=_float_or_none(row.get("photometric_residual_limit_mag")),
                photometric_consistent=(
                    bool(row["photometric_consistent"])
                    if row.get("photometric_consistent") is not None
                    else None
                ),
                photometric_outlier_reason=row.get("photometric_outlier_reason"),
                calibration_sample_role=row.get("calibration_sample_role"),
            )
            for row in payload.get("source_photometry", [])
        )
        from .pipeline import FrameAnalysis

        return FrameAnalysis(
            frame=frame,
            detection=detection,
            matching=matching,
            faintest=faintest,
            photometric_calibration=calibration,
            source_photometry=source_photometry,
        )
    except (OSError, EOFError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_analysis(cache_dir: Path, key: str, analysis: Any) -> Path:
    """以 gzip JSON 保存完整检测结果，方便之后重画叠加和导出。"""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, key)
    temporary_path = path.with_suffix(".tmp")
    frame = getattr(analysis, "frame", None)
    if not isinstance(frame, FitsFrame):
        raise TypeError("analysis cache requires a FrameAnalysis with a FitsFrame")
    payload = dict(analysis.as_dict())
    payload.update(
        {
            "cache_version": CACHE_VERSION,
            "cache_kind": "frame-analysis",
            "cache_key": key,
            "input_identity": _frame_input_identity(frame),
        }
    )
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def sequence_cache_key(
    paths: Sequence[str | Path],
    *,
    parameters: Mapping[str, Any],
    catalog: object | None = None,
    catalog_path: str | Path | None = None,
    catalog_identity: object | None = None,
    wcs: object | None = None,
    epoch: float | None = None,
    match_radius_px: float | None = None,
    photometric_system: str | None = None,
    photometric_band: str | None = None,
    photometric_color_name: str | None = None,
    photometric_color_order: int | None = None,
    photometric_min_calibrators: int | None = None,
    parallax: object | None = None,
    parallax_zero_point_mas: float | None = None,
    max_fractional_parallax_error: float | None = None,
    extinction: object | None = None,
    extinction_model: object | None = None,
    extinction_mag: float | None = None,
    extinction_error_mag: float | None = None,
    relative_photometry: object | None = None,
    relative_photometry_parameters: Mapping[str, Any] | None = None,
    input_directory: str | Path | None = None,
    cache_context: Mapping[str, Any] | None = None,
) -> str:
    """按整组 FITS 内容、目录清单和序列/星等参数生成缓存键。

    ``parameters`` 保留现有调用方的完整序列配置；专用上下文形参用于让
    星表/WCS/历元/波段/距离/消光/相对光度配置不依赖调用方是否记得把它们
    拼进普通字典。显式 ``input_directory`` 还会纳入目录内所有文件的内容
    清单；默认则纳入各输入父目录的直接 FITS 清单。
    """

    frame_paths = tuple(paths)
    descriptor = {
        "cache_version": CACHE_VERSION,
        "sequence_cache_version": SEQUENCE_CACHE_VERSION,
        "input": _sequence_input_identity(frame_paths),
        "parameters": _canonical_identity(parameters),
        "scientific_context": _scientific_context(
            catalog=catalog,
            catalog_path=catalog_path,
            catalog_identity=catalog_identity,
            wcs=wcs,
            epoch=epoch,
            match_radius_px=match_radius_px,
            photometric_system=photometric_system,
            photometric_band=photometric_band,
            photometric_color_name=photometric_color_name,
            photometric_color_order=photometric_color_order,
            photometric_min_calibrators=photometric_min_calibrators,
            parallax=parallax,
            parallax_zero_point_mas=parallax_zero_point_mas,
            max_fractional_parallax_error=max_fractional_parallax_error,
            extinction=extinction,
            extinction_model=extinction_model,
            extinction_mag=extinction_mag,
            extinction_error_mag=extinction_error_mag,
            relative_photometry=relative_photometry,
            relative_photometry_parameters=relative_photometry_parameters,
            input_directory=input_directory,
            cache_context=cache_context,
        ),
    }
    encoded = json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sequence_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.sequence.json.gz"


def _float_or_none(value: object) -> float | None:
    return float(value) if value is not None else None


def _sequence_result_from_dict(payload: Mapping[str, Any]) -> Any:
    """从缓存 JSON 恢复 SequenceResult；导入放在函数内避免包初始化环。"""

    from .sequence import (
        FixedSentinelImpactAudit,
        FixedSentinelSourceImpact,
        FrameSequenceSummary,
        MotionFeaturePoint,
        MotionFeatureTrack,
        MotionFrameAudit,
        SequenceResult,
        SourceTrack,
        TrackPoint,
        FixedSentinelAudit,
    )
    from .relative_photometry import RelativePhotometryResult

    frames = tuple(
        FrameSequenceSummary(
            frame_index=int(frame["frame_index"]),
            path=str(frame["path"]),
            candidate_count=int(frame["candidate_count"]),
            returned_count=int(frame["returned_count"]),
            quality_count=int(frame["quality_count"]),
            timestamp=str(frame["timestamp"]) if frame.get("timestamp") is not None else None,
            exposure_ms=_float_or_none(frame.get("exposure_ms")),
            auxiliary=tuple((str(key), float(value)) for key, value in dict(frame.get("auxiliary", {})).items()),
            width_px=int(frame["width_px"]) if frame.get("width_px") is not None else None,
            height_px=int(frame["height_px"]) if frame.get("height_px") is not None else None,
        )
        for frame in payload.get("frames", [])
    )
    tracks = []
    for track in payload.get("tracks", []):
        points = tuple(
            TrackPoint(
                frame_index=int(point["frame_index"]),
                detection_id=int(point["detection_id"]),
                x=float(point["x"]),
                y=float(point["y"]),
                aligned_x=float(point["aligned_x"]),
                aligned_y=float(point["aligned_y"]),
                flux_snr=_float_or_none(point.get("flux_snr")),
                quality_passed=bool(point.get("quality_passed", True)),
                candidate_snr=_float_or_none(point.get("candidate_snr")),
            )
            for point in track.get("points", [])
        )
        tracks.append(
            SourceTrack(
                track_id=int(track["track_id"]),
                classification=str(track["classification"]),
                points=points,
                displacement_px=float(track["displacement_px"]),
                speed_px_per_frame=float(track["speed_px_per_frame"]),
                fit_rms_px=_float_or_none(track.get("fit_rms_px")),
                evidence_level=str(track.get("evidence_level", "quality")),
            )
        )
    motion_features = []
    for track in payload.get("motion_features", []):
        points = tuple(
            MotionFeaturePoint(
                frame_index=int(point["frame_index"]),
                x=float(point["x"]),
                y=float(point["y"]),
                aligned_x=float(point["aligned_x"]),
                aligned_y=float(point["aligned_y"]),
                residual_snr=float(point["residual_snr"]),
                area_pixels=int(point["area_pixels"]),
                length_px=float(point["length_px"]),
                width_px=float(point["width_px"]),
                angle_deg=float(point["angle_deg"]),
                bbox=tuple(int(value) for value in point["bbox"]),
                touches_edge=bool(point["touches_edge"]),
            )
            for point in track.get("points", [])
        )
        motion_features.append(
            MotionFeatureTrack(
                track_id=int(track["track_id"]),
                classification=str(track["classification"]),
                points=points,
                displacement_px=float(track["displacement_px"]),
                speed_px_per_frame=float(track["speed_px_per_frame"]),
                fit_rms_px=_float_or_none(track.get("fit_rms_px")),
            )
        )
    motion_frame_audits = tuple(
        MotionFrameAudit(
            frame_index=int(audit["frame_index"]),
            threshold_adu=float(audit["threshold_adu"]),
            residual_noise_adu=float(audit["residual_noise_adu"]),
            valid_pixel_count=int(audit["valid_pixel_count"]),
            support_pixel_count=int(audit["support_pixel_count"]),
            component_count=int(audit["component_count"]),
            area_pass_count=int(audit["area_pass_count"]),
            geometry_pass_count=int(audit["geometry_pass_count"]),
            edge_rejected_count=int(audit["edge_rejected_count"]),
            feature_count=int(audit["feature_count"]),
            max_feature_residual_snr=_float_or_none(audit.get("max_feature_residual_snr")),
        )
        for audit in payload.get("motion_frame_audits", [])
    )
    fixed_sentinel_payload = payload.get("fixed_sentinel_audit")
    fixed_sentinel_audit = None
    if isinstance(fixed_sentinel_payload, Mapping):
        image_shape_payload = fixed_sentinel_payload.get("image_shape")
        image_shape = (
            (int(image_shape_payload[0]), int(image_shape_payload[1]))
            if isinstance(image_shape_payload, Sequence)
            and not isinstance(image_shape_payload, (str, bytes))
            and len(image_shape_payload) == 2
            else None
        )
        fixed_coordinates: list[tuple[int, int]] = []
        coordinates_payload = fixed_sentinel_payload.get("fixed_coordinates", [])
        if isinstance(coordinates_payload, Sequence) and not isinstance(coordinates_payload, (str, bytes)):
            for coordinate in coordinates_payload:
                if (
                    isinstance(coordinate, Sequence)
                    and not isinstance(coordinate, (str, bytes))
                    and len(coordinate) == 2
                ):
                    fixed_coordinates.append((int(coordinate[0]), int(coordinate[1])))
        per_frame_payload = fixed_sentinel_payload.get("per_frame_occurrences", [])
        per_frame_occurrences = (
            tuple(int(value) for value in per_frame_payload)
            if isinstance(per_frame_payload, Sequence)
            and not isinstance(per_frame_payload, (str, bytes))
            else ()
        )
        fixed_sentinel_audit = FixedSentinelAudit(
            sentinel_value=int(fixed_sentinel_payload.get("sentinel_value", -1)),
            frame_count=int(fixed_sentinel_payload.get("frame_count", len(per_frame_occurrences))),
            image_shape=image_shape,
            same_shape=bool(fixed_sentinel_payload.get("same_shape", True)),
            total_occurrences=int(fixed_sentinel_payload.get("total_occurrences", 0)),
            per_frame_occurrences=per_frame_occurrences,
            fixed_coordinate_count=int(fixed_sentinel_payload.get("fixed_coordinate_count", len(fixed_coordinates))),
            fixed_coordinates=tuple(fixed_coordinates),
            coordinates_truncated=bool(fixed_sentinel_payload.get("coordinates_truncated", False)),
        )
    fixed_sentinel_impact_payload = payload.get("fixed_sentinel_impact_audit")
    fixed_sentinel_impact_audit = None
    if isinstance(fixed_sentinel_impact_payload, Mapping):
        impact_records: list[FixedSentinelSourceImpact] = []
        records_payload = fixed_sentinel_impact_payload.get("records", [])
        if isinstance(records_payload, Sequence) and not isinstance(records_payload, (str, bytes)):
            for record in records_payload:
                if not isinstance(record, Mapping):
                    continue
                impact_records.append(
                    FixedSentinelSourceImpact(
                        frame_index=int(record.get("frame_index", 0)),
                        detection_id=int(record.get("detection_id", -1)),
                        x=float(record.get("x", 0.0)),
                        y=float(record.get("y", 0.0)),
                        peak_x=float(record.get("peak_x", record.get("x", 0.0))),
                        peak_y=float(record.get("peak_y", record.get("y", 0.0))),
                        nearest_fixed_distance_px=float(record.get("nearest_fixed_distance_px", 0.0)),
                        fixed_coordinate_count=int(record.get("fixed_coordinate_count", 0)),
                        peak=float(record.get("peak", 0.0)),
                        flux_snr=_float_or_none(record.get("flux_snr")),
                        quality_passed=bool(record.get("quality_passed", False)),
                        flags=tuple(str(flag) for flag in record.get("flags", [])),
                    )
                )
        fixed_sentinel_impact_audit = FixedSentinelImpactAudit(
            frame_count=int(fixed_sentinel_impact_payload.get("frame_count", 0)),
            affected_frame_count=int(fixed_sentinel_impact_payload.get("affected_frame_count", 0)),
            candidate_peak_data_available=bool(fixed_sentinel_impact_payload.get("candidate_peak_data_available", False)),
            candidate_peak_frame_count=int(fixed_sentinel_impact_payload.get("candidate_peak_frame_count", 0)),
            candidate_peak_count=int(fixed_sentinel_impact_payload.get("candidate_peak_count", 0)),
            candidate_peak_affected_count=int(fixed_sentinel_impact_payload.get("candidate_peak_affected_count", 0)),
            affected_returned_source_count=int(fixed_sentinel_impact_payload.get("affected_returned_source_count", 0)),
            affected_quality_source_count=int(fixed_sentinel_impact_payload.get("affected_quality_source_count", 0)),
            records=tuple(impact_records),
            record_count=int(fixed_sentinel_impact_payload.get("record_count", len(impact_records))),
            records_truncated=bool(fixed_sentinel_impact_payload.get("records_truncated", False)),
        )
    relative_payload = payload.get("relative_photometry")
    relative_result = None
    if isinstance(relative_payload, Mapping):
        def float_mapping(name: str) -> dict[str, float]:
            raw = relative_payload.get(name, {})
            if not isinstance(raw, Mapping):
                return {}
            result: dict[str, float] = {}
            for key, value in raw.items():
                if value is None:
                    continue
                result[str(key)] = float(value)
            return result

        def optional_float_mapping(name: str) -> dict[str, float | None]:
            raw = relative_payload.get(name, {})
            if not isinstance(raw, Mapping):
                return {}
            result: dict[str, float | None] = {}
            for key, value in raw.items():
                result[str(key)] = float(value) if value is not None else None
            return result

        validation_source_ids = relative_payload.get("validation_source_ids", [])
        rejected_indices = relative_payload.get("rejected_observation_indices", [])
        relative_result = RelativePhotometryResult(
            status=str(relative_payload.get("status", "UNKNOWN")),
            flags=tuple(str(flag) for flag in relative_payload.get("flags", [])),
            frame_zero_points=float_mapping("frame_zero_points"),
            relative_magnitudes=float_mapping("relative_magnitudes"),
            frame_zero_point_uncertainties=optional_float_mapping("frame_zero_point_uncertainties"),
            relative_magnitude_uncertainties=optional_float_mapping("relative_magnitude_uncertainties"),
            reference_sample_count=int(relative_payload.get("reference_sample_count", 0)),
            reference_sample_count_by_frame={
                str(key): int(value) for key, value in dict(relative_payload.get("reference_sample_count_by_frame", {})).items()
            },
            reference_sample_count_by_source={
                str(key): int(value) for key, value in dict(relative_payload.get("reference_sample_count_by_source", {})).items()
            },
            training_residual_rms=_float_or_none(relative_payload.get("training_residual_rms")),
            training_residual_mad=_float_or_none(relative_payload.get("training_residual_mad")),
            validation_residual_rms=_float_or_none(relative_payload.get("validation_residual_rms")),
            validation_residual_mad=_float_or_none(relative_payload.get("validation_residual_mad")),
            validation_sample_count=int(relative_payload.get("validation_sample_count", 0)),
            validation_source_ids=tuple(str(value) for value in validation_source_ids),
            rejected_observation_indices=tuple(int(value) for value in rejected_indices),
            excluded_observation_count=int(relative_payload.get("excluded_observation_count", 0)),
            spatial_coefficients=tuple(float(value) for value in relative_payload.get("spatial_coefficients", [])),
            spatial_order=int(relative_payload.get("spatial_order", 0)),
        )
    return SequenceResult(
        frames=frames,
        cumulative_shifts=tuple((float(shift[0]), float(shift[1])) for shift in payload.get("cumulative_shifts", [])),
        tracks=tuple(tracks),
        link_radius_px=float(payload["link_radius_px"]),
        min_presence=int(payload["min_presence"]),
        motion_min_displacement_px=float(payload["motion_min_displacement_px"]),
        max_motion_fit_rms_px=float(payload["max_motion_fit_rms_px"]),
        motion_features=tuple(motion_features),
        motion_residual_threshold_adu=float(payload.get("motion_residual_threshold_adu", 100.0)),
        motion_reference_mode=str(payload.get("motion_reference_mode", "per_frame_background")),
        motion_frame_audits=motion_frame_audits,
        source_working_limit=(
            int(payload["source_working_limit"])
            if payload.get("source_working_limit") is not None
            else None
        ),
        calculation_dtype=str(payload.get("calculation_dtype", "float64")),
        background_sample_limit=(
            int(payload["background_sample_limit"])
            if payload.get("background_sample_limit") is not None
            else None
        ),
        fast_sequence=bool(payload.get("fast_sequence", False)),
        background_model_mode=str(payload.get("background_model_mode", "per_frame_local")),
        persistent_min_presence=int(payload.get("persistent_min_presence", 0)),
        temporal_reference_candidate_count=int(payload.get("temporal_reference_candidate_count", 0)),
        temporal_coadd_candidate_count=int(payload.get("temporal_coadd_candidate_count", 0)),
        temporal_proposal_mode=str(payload.get("temporal_proposal_mode", "median")),
        temporal_candidate_min_snr=float(payload.get("temporal_candidate_min_snr", 7.5)),
        temporal_reference_min_snr=float(payload.get("temporal_reference_min_snr", 15.0)),
        temporal_multiscale=bool(payload.get("temporal_multiscale", False)),
        temporal_min_psf_correlation=float(payload.get("temporal_min_psf_correlation", 0.8)),
        candidate_consensus_audit=tuple(
            (str(key), int(value))
            for key, value in dict(payload.get("candidate_consensus_audit", {})).items()
        ),
        stack_faint_candidate_count=int(payload.get("stack_faint_candidate_count", 0)),
        stack_reference_mode=str(payload.get("stack_reference_mode", "median")),
        stack_threshold_sigma=float(payload.get("stack_threshold_sigma", 4.0)),
        stack_min_flux_snr=float(payload.get("stack_min_flux_snr", 5.0)),
        stack_frame_min_flux_snr=float(payload.get("stack_frame_min_flux_snr", 3.0)),
        stack_min_presence=int(payload.get("stack_min_presence", 0)),
        fixed_sentinel_audit=fixed_sentinel_audit,
        fixed_sentinel_impact_audit=fixed_sentinel_impact_audit,
        relative_photometry=relative_result,
    )


def load_sequence_result(
    cache_dir: Path,
    key: str,
    paths: Sequence[str | Path] | None = None,
) -> Any | None:
    """读取整组序列缓存；损坏或版本不符时按缓存未命中处理。"""

    path = sequence_cache_path(cache_dir, key)
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if (
            not isinstance(payload, ABCMapping)
            or payload.get("cache_version") != CACHE_VERSION
            or payload.get("sequence_cache_version") != SEQUENCE_CACHE_VERSION
            or payload.get("cache_kind") != "sequence"
            or payload.get("cache_key") != key
            or not isinstance(payload.get("sequence"), ABCMapping)
            or "relative_photometry" not in payload["sequence"]
        ):
            return None
        sequence_payload = payload["sequence"]
        frame_payloads = sequence_payload.get("frames", [])
        if not isinstance(frame_payloads, Sequence) or isinstance(frame_payloads, (str, bytes)):
            return None
        payload_paths = tuple(
            str(frame_payload["path"])
            for frame_payload in frame_payloads
            if isinstance(frame_payload, ABCMapping) and "path" in frame_payload
        )
        if len(payload_paths) != len(frame_payloads):
            return None
        expected_paths = tuple(paths) if paths is not None else payload_paths
        if _sequence_input_identity(expected_paths) != payload.get("input_identity"):
            return None
        if paths is not None and tuple(str(Path(value).resolve()) for value in paths) != tuple(
            str(Path(value).resolve()) for value in payload_paths
        ):
            return None
        return _sequence_result_from_dict(sequence_payload)
    except (OSError, EOFError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_sequence_result(cache_dir: Path, key: str, result: Any) -> Path:
    """原子写入整组序列结果 gzip 缓存。"""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = sequence_cache_path(cache_dir, key)
    temporary_path = path.with_suffix(".tmp")
    sequence_payload = dict(result.as_dict())
    frame_payloads = sequence_payload.get("frames", [])
    if not isinstance(frame_payloads, Sequence) or isinstance(frame_payloads, (str, bytes)):
        raise TypeError("sequence cache requires a SequenceResult with frame summaries")
    frame_paths = tuple(
        str(frame_payload["path"])
        for frame_payload in frame_payloads
        if isinstance(frame_payload, ABCMapping) and "path" in frame_payload
    )
    if len(frame_paths) != len(frame_payloads):
        raise ValueError("sequence cache frame summaries must contain paths")
    payload = {
        "cache_version": CACHE_VERSION,
        "sequence_cache_version": SEQUENCE_CACHE_VERSION,
        "cache_kind": "sequence",
        "cache_key": key,
        "input_identity": _sequence_input_identity(frame_paths),
        "sequence": sequence_payload,
    }
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
