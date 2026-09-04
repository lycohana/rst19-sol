"""单帧检测结果的本地压缩缓存。"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .detection import Detection, DetectionResult
from .fits import FitsFrame
from .photometry import FaintestSource

# 几何孔径的 MASKED/SATURATED 判定、局部零值坏像素容错、线状伪迹质量语义、
# PSF 支持审计和背景网格抽样
# 口径发生过变化；
# 旧缓存不能继续冒充当前检测结果，必须自动失效并重新计算。
# 本轮（16）修正 CODE_PATTERN 触发条件：只有候选峰本身的像素值落在
# 全幅异常重复高位码上、且孔径含异常负值时才拒绝；此前“孔径内含重复码
# 即拒”会误伤真实亮星的饱和/溢出出血列。
CACHE_VERSION = 16
# 序列结果新增逐帧线状筛选审计和配准工作集口径；此前切换到只在
# 16 位 FITS 序列路径使用 float32 中间阵列并调整快速背景统计迭代数，
# 本轮又增加确定性网格抽样、注册时间中值补提案、时序候选响应分层，
# 以及时序补提案在预测坐标附近的 ±1 px 局部 PSF 峰定位；此前再加入
# 序列级固定异常码审计结果及其对候选的孔径影响关联；本轮加入高位
# 重复码与局部异常负值联合门控，以及极近 Gaussian 双 PSF 质量门。
# 本轮（35）新增叠加参考图暗星恢复层（evidence_level="stack_faint"）：
# 序列结果现在含 stack_faint 轨迹及其参数，旧缓存不具备该口径。
# 旧缓存不具备相同计算口径，必须重新计算，否则 UI 可能把不同精度的
# 结果混在一起。
SEQUENCE_CACHE_VERSION = 35
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


def sequence_cache_key(paths: Sequence[str | Path], *, parameters: Mapping[str, Any]) -> str:
    """按整组 FITS 文件状态和序列/检测参数生成缓存键。"""

    frames = []
    for path in paths:
        frame_path = Path(path)
        stat = frame_path.stat()
        frames.append({"path": str(frame_path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    descriptor = {
        "cache_version": CACHE_VERSION,
        "sequence_cache_version": SEQUENCE_CACHE_VERSION,
        "frames": frames,
        "parameters": dict(parameters),
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
    )


def load_sequence_result(cache_dir: Path, key: str) -> Any | None:
    """读取整组序列缓存；损坏或版本不符时按缓存未命中处理。"""

    path = sequence_cache_path(cache_dir, key)
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("cache_version") != CACHE_VERSION or payload.get("sequence_cache_version") != SEQUENCE_CACHE_VERSION:
            return None
        return _sequence_result_from_dict(payload["sequence"])
    except (OSError, EOFError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_sequence_result(cache_dir: Path, key: str, result: Any) -> Path:
    """原子写入整组序列结果 gzip 缓存。"""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = sequence_cache_path(cache_dir, key)
    temporary_path = path.with_suffix(".tmp")
    payload = {
        "cache_version": CACHE_VERSION,
        "sequence_cache_version": SEQUENCE_CACHE_VERSION,
        "sequence": result.as_dict(),
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
