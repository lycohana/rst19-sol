"""各类候选对检测参数的敏感性审计。

本模块不把参数扫描结果当作真值标签。它固定同一张真实 FITS，在候选
阈值和质量层 ``flux SNR`` 两个正交方向上重复检测，然后把每个运行与
默认配置的候选位置做互相最近一对一配对。输出用于回答三个问题：

* 候选数量变化来自宽筛阈值，还是来自质量层；
* 某个首要特征类别在参数变化后是否仍保留同一批位置；
* 通过质量层的类别是否只是因为某个门槛被放宽。

``candidate``、``quality`` 和 ``baseline match`` 仍然是三个不同层次：
它们都不能单独替代星表、注入真值或跨帧身份确认。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .detection import Detection
from .experiments import classify_source_feature
from .fits import FitsFrame, read_fits
from .pipeline import analyze_frame


FEATURE_CLASS_ORDER: tuple[str, ...] = (
    "range_anomaly",
    "linear_artifact",
    "masked_or_edge",
    "crowded_blend",
    "spike_or_support",
    "weak_or_background",
    "shape_outlier",
    "compact_quality",
    "other_rejected",
)


@dataclass(frozen=True, slots=True)
class FeatureParameterRunRow:
    """一个参数配置下的整图数量统计。"""

    source_path: str
    parameter_family: str
    parameter_value: float
    threshold_sigma: float
    min_flux_snr: float
    candidate_count: int
    returned_count: int
    quality_count: int
    rejected_count: int
    background_adu: float
    noise_adu: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureParameterProfileRow:
    """默认候选按首要类别在一个参数配置下的响应。"""

    source_path: str
    parameter_family: str
    parameter_value: float
    threshold_sigma: float
    min_flux_snr: float
    feature_class: str
    baseline_candidate_anchor_count: int
    current_candidate_count: int
    current_quality_count: int
    current_quality_fraction: float | None
    baseline_candidate_matched_count: int
    baseline_same_class_matched_count: int
    baseline_candidate_match_fraction: float | None
    baseline_same_class_match_fraction: float | None
    baseline_quality_anchor_count: int
    baseline_quality_matched_count: int
    baseline_quality_still_passed_count: int
    baseline_quality_match_fraction: float | None
    baseline_quality_still_passed_fraction: float | None
    note: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureParameterTransitionRow:
    """默认类别与当前类别之间的同坐标转移计数。"""

    source_path: str
    parameter_family: str
    parameter_value: float
    threshold_sigma: float
    min_flux_snr: float
    baseline_feature_class: str
    current_feature_class: str
    matched_count: int
    current_quality_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureParameterSensitivityResult:
    """参数敏感性审计的机器可读结果。"""

    source_path: str
    image_shape: tuple[int, int]
    baseline_threshold_sigma: float
    baseline_min_flux_snr: float
    association_radius_px: float
    detector_parameters: dict[str, object]
    requested_threshold_levels: tuple[float, ...]
    requested_flux_snr_levels: tuple[float, ...]
    runs: tuple[FeatureParameterRunRow, ...]
    profiles: tuple[FeatureParameterProfileRow, ...]
    transitions: tuple[FeatureParameterTransitionRow, ...]
    observations: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "image_shape": list(self.image_shape),
            "baseline_threshold_sigma": self.baseline_threshold_sigma,
            "baseline_min_flux_snr": self.baseline_min_flux_snr,
            "association_radius_px": self.association_radius_px,
            "detector_parameters": dict(self.detector_parameters),
            "requested_threshold_levels": list(self.requested_threshold_levels),
            "requested_flux_snr_levels": list(self.requested_flux_snr_levels),
            "runs": [row.as_dict() for row in self.runs],
            "profiles": [row.as_dict() for row in self.profiles],
            "transitions": [row.as_dict() for row in self.transitions],
            "observations": dict(self.observations),
        }


def _finite_point(source: Detection) -> tuple[float, float] | None:
    x = source.peak_x if source.peak_x is not None else source.x
    y = source.peak_y if source.peak_y is not None else source.y
    if not np.isfinite(float(x)) or not np.isfinite(float(y)):
        return None
    return float(x), float(y)


def _reciprocal_match_indices(
    anchor_sources: Sequence[Detection],
    current_sources: Sequence[Detection],
    *,
    radius_px: float,
) -> dict[int, int]:
    """返回默认源到当前源的互相最近一对一匹配。"""

    if radius_px <= 0 or not anchor_sources or not current_sources:
        return {}
    anchor_rows = [(index, _finite_point(source)) for index, source in enumerate(anchor_sources)]
    current_rows = [(index, _finite_point(source)) for index, source in enumerate(current_sources)]
    anchor_valid = [(index, point) for index, point in anchor_rows if point is not None]
    current_valid = [(index, point) for index, point in current_rows if point is not None]
    if not anchor_valid or not current_valid:
        return {}

    anchor_points = np.asarray([point for _index, point in anchor_valid], dtype=np.float64)
    current_points = np.asarray([point for _index, point in current_valid], dtype=np.float64)
    anchor_tree = cKDTree(anchor_points)
    current_tree = cKDTree(current_points)
    anchor_distances, current_positions = current_tree.query(
        anchor_points,
        distance_upper_bound=float(radius_px),
    )
    current_distances, anchor_positions = anchor_tree.query(
        current_points,
        distance_upper_bound=float(radius_px),
    )
    result: dict[int, int] = {}
    for anchor_position, current_position in enumerate(np.asarray(current_positions).reshape(-1)):
        if not np.isfinite(anchor_distances[anchor_position]):
            continue
        current_position = int(current_position)
        if current_position >= len(current_valid):
            continue
        if not np.isfinite(current_distances[current_position]):
            continue
        if int(anchor_positions[current_position]) != anchor_position:
            continue
        result[anchor_valid[anchor_position][0]] = current_valid[current_position][0]
    return result


def _fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _profiles_for_run(
    *,
    source_path: str,
    parameter_family: str,
    parameter_value: float,
    threshold_sigma: float,
    min_flux_snr: float,
    baseline_sources: Sequence[Detection],
    current_sources: Sequence[Detection],
    association_radius_px: float,
) -> tuple[FeatureParameterProfileRow, ...]:
    baseline_classes = [classify_source_feature(source) for source in baseline_sources]
    current_classes = [classify_source_feature(source) for source in current_sources]
    matches = _reciprocal_match_indices(
        baseline_sources,
        current_sources,
        radius_px=association_radius_px,
    )
    current_class_counts = {
        feature_class: sum(current_class == feature_class for current_class in current_classes)
        for feature_class in FEATURE_CLASS_ORDER
    }
    current_quality_counts = {
        feature_class: sum(
            current_class == feature_class and bool(source.quality_passed)
            for current_class, source in zip(current_classes, current_sources, strict=True)
        )
        for feature_class in FEATURE_CLASS_ORDER
    }
    rows: list[FeatureParameterProfileRow] = []
    for feature_class in FEATURE_CLASS_ORDER:
        anchor_indices = [
            index for index, current_class in enumerate(baseline_classes) if current_class == feature_class
        ]
        matched_indices = [index for index in anchor_indices if index in matches]
        same_class_count = sum(
            current_classes[matches[index]] == feature_class
            for index in matched_indices
        )
        quality_anchor_indices = [
            index
            for index in anchor_indices
            if bool(baseline_sources[index].quality_passed)
        ]
        quality_matched_indices = [index for index in quality_anchor_indices if index in matches]
        quality_still_passed_count = sum(
            bool(current_sources[matches[index]].quality_passed)
            for index in quality_matched_indices
        )
        current_count = current_class_counts[feature_class]
        current_quality_count = current_quality_counts[feature_class]
        rows.append(
            FeatureParameterProfileRow(
                source_path=source_path,
                parameter_family=parameter_family,
                parameter_value=float(parameter_value),
                threshold_sigma=float(threshold_sigma),
                min_flux_snr=float(min_flux_snr),
                feature_class=feature_class,
                baseline_candidate_anchor_count=len(anchor_indices),
                current_candidate_count=current_count,
                current_quality_count=current_quality_count,
                current_quality_fraction=_fraction(current_quality_count, current_count),
                baseline_candidate_matched_count=len(matched_indices),
                baseline_same_class_matched_count=same_class_count,
                baseline_candidate_match_fraction=_fraction(len(matched_indices), len(anchor_indices)),
                baseline_same_class_match_fraction=_fraction(same_class_count, len(anchor_indices)),
                baseline_quality_anchor_count=len(quality_anchor_indices),
                baseline_quality_matched_count=len(quality_matched_indices),
                baseline_quality_still_passed_count=quality_still_passed_count,
                baseline_quality_match_fraction=_fraction(
                    len(quality_matched_indices), len(quality_anchor_indices)
                ),
                baseline_quality_still_passed_fraction=_fraction(
                    quality_still_passed_count, len(quality_anchor_indices)
                ),
                note=(
                    "baseline_match 是同一首帧坐标在参数变化后的互相最近一对一响应；"
                    "feature_class 是算法首要类别，不是物理恒星类别。"
                ),
            )
        )
    return tuple(rows)


def _transitions_for_run(
    *,
    source_path: str,
    parameter_family: str,
    parameter_value: float,
    threshold_sigma: float,
    min_flux_snr: float,
    baseline_sources: Sequence[Detection],
    current_sources: Sequence[Detection],
    association_radius_px: float,
) -> tuple[FeatureParameterTransitionRow, ...]:
    """按同坐标匹配统计首要类别转移，不把未匹配源当作转移。"""

    baseline_classes = [classify_source_feature(source) for source in baseline_sources]
    current_classes = [classify_source_feature(source) for source in current_sources]
    matches = _reciprocal_match_indices(
        baseline_sources,
        current_sources,
        radius_px=association_radius_px,
    )
    counts: dict[tuple[str, str], list[int]] = {}
    for baseline_index, current_index in matches.items():
        key = (baseline_classes[baseline_index], current_classes[current_index])
        values = counts.setdefault(key, [0, 0])
        values[0] += 1
        values[1] += int(bool(current_sources[current_index].quality_passed))
    rows: list[FeatureParameterTransitionRow] = []
    for (baseline_class, current_class), (matched_count, current_quality_count) in sorted(
        counts.items(),
        key=lambda item: (FEATURE_CLASS_ORDER.index(item[0][0]), FEATURE_CLASS_ORDER.index(item[0][1])),
    ):
        rows.append(
            FeatureParameterTransitionRow(
                source_path=source_path,
                parameter_family=parameter_family,
                parameter_value=float(parameter_value),
                threshold_sigma=float(threshold_sigma),
                min_flux_snr=float(min_flux_snr),
                baseline_feature_class=baseline_class,
                current_feature_class=current_class,
                matched_count=int(matched_count),
                current_quality_count=int(current_quality_count),
            )
        )
    return tuple(rows)


def _normalise_levels(levels: Iterable[float], *, name: str) -> tuple[float, ...]:
    values = tuple(float(level) for level in levels)
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{name} must contain finite positive values")
    return tuple(dict.fromkeys(values))


def run_feature_parameter_sensitivity(
    path: str | Path | FitsFrame,
    *,
    threshold_levels: Sequence[float] = (4.0, 5.0, 6.0, 8.0),
    flux_snr_levels: Sequence[float] = (3.0, 5.0, 7.0, 9.0),
    baseline_threshold_sigma: float = 4.0,
    baseline_min_flux_snr: float = 5.0,
    min_distance: int = 4,
    aperture_radius: int = 4,
    psf_fwhm: float = 2.0,
    background_box_size: int = 128,
    background_sample_limit: int = 100_000,
    min_psf_support_pixels: int = 3,
    proposal_mode: str = "hybrid",
    max_sources: int | None = None,
    association_radius_px: float = 1.5,
    progress: Callable[[int, int, str], None] | None = None,
) -> FeatureParameterSensitivityResult:
    """在同一真实首帧上扫描阈值和通量 SNR，并按类别回溯默认候选。

    运行之间共享同一个已读入的 ``FitsFrame``，但每个参数配置仍独立
    建立背景模型和检测结果。默认配置会被强制加入内部运行集，哪怕
    调用方自定义扫描 levels 时没有把 ``4σ/5`` 写进去。
    """

    threshold_values = _normalise_levels(threshold_levels, name="threshold_levels")
    flux_values = _normalise_levels(flux_snr_levels, name="flux_snr_levels")
    if baseline_threshold_sigma <= 0 or baseline_min_flux_snr <= 0:
        raise ValueError("baseline thresholds must be positive")
    if min_distance < 1 or aperture_radius < 1 or psf_fwhm <= 0:
        raise ValueError("detector geometry parameters must be positive")
    if background_box_size < 16 or background_sample_limit < 1:
        raise ValueError("background parameters are invalid")
    if not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("min_psf_support_pixels must be within 1..9")
    if association_radius_px <= 0:
        raise ValueError("association_radius_px must be positive")

    frame = path if isinstance(path, FitsFrame) else read_fits(path)
    source_path = str(frame.path)
    # family, value, threshold_sigma, min_flux_snr
    requested_specs = [
        ("threshold_sigma", value, value, float(baseline_min_flux_snr))
        for value in threshold_values
    ] + [
        ("min_flux_snr", value, float(baseline_threshold_sigma), value)
        for value in flux_values
    ]
    unique_configs: list[tuple[float, float]] = []
    for _family, _value, threshold_sigma, min_flux_snr in requested_specs + [
        ("baseline", 0.0, float(baseline_threshold_sigma), float(baseline_min_flux_snr))
    ]:
        config = (float(threshold_sigma), float(min_flux_snr))
        if config not in unique_configs:
            unique_configs.append(config)

    detector_parameters = {
        "min_distance": int(min_distance),
        "aperture_radius": int(aperture_radius),
        "psf_fwhm": float(psf_fwhm),
        "background_box_size": int(background_box_size),
        "background_sample_limit": int(background_sample_limit),
        "min_psf_support_pixels": int(min_psf_support_pixels),
        "proposal_mode": str(proposal_mode),
        "max_sources": max_sources,
        "refine_local_background": True,
        "reject_linear_artifacts": True,
    }
    analyses: dict[tuple[float, float], tuple[Detection, ...]] = {}
    detection_summary: dict[tuple[float, float], tuple[int, int, int, float, float]] = {}
    total = len(unique_configs)
    for index, (threshold_sigma, min_flux_snr) in enumerate(unique_configs, start=1):
        analysis = analyze_frame(
            frame,
            threshold_sigma=threshold_sigma,
            min_distance=min_distance,
            aperture_radius=aperture_radius,
            max_sources=max_sources,
            psf_fwhm=psf_fwhm,
            background_box_size=background_box_size,
            background_sample_limit=background_sample_limit,
            min_flux_snr=min_flux_snr,
            min_psf_support_pixels=min_psf_support_pixels,
            proposal_mode=proposal_mode,
            reject_linear_artifacts=True,
            refine_local_background=True,
        )
        detection = analysis.detection
        analyses[(threshold_sigma, min_flux_snr)] = tuple(detection.sources)
        detection_summary[(threshold_sigma, min_flux_snr)] = (
            int(detection.candidate_count),
            int(detection.returned_count),
            int(detection.star_count),
            float(detection.background),
            float(detection.noise),
        )
        if progress is not None:
            progress(index, total, f"threshold={threshold_sigma:g}, flux_snr={min_flux_snr:g}")

    baseline_config = (float(baseline_threshold_sigma), float(baseline_min_flux_snr))
    baseline_sources = analyses[baseline_config]
    run_rows: list[FeatureParameterRunRow] = []
    profile_rows: list[FeatureParameterProfileRow] = []
    transition_rows: list[FeatureParameterTransitionRow] = []
    for family, value, threshold_sigma, min_flux_snr in requested_specs:
        config = (float(threshold_sigma), float(min_flux_snr))
        candidate_count, returned_count, quality_count, background, noise = detection_summary[config]
        run_rows.append(
            FeatureParameterRunRow(
                source_path=source_path,
                parameter_family=family,
                parameter_value=float(value),
                threshold_sigma=float(threshold_sigma),
                min_flux_snr=float(min_flux_snr),
                candidate_count=candidate_count,
                returned_count=returned_count,
                quality_count=quality_count,
                rejected_count=max(0, returned_count - quality_count),
                background_adu=background,
                noise_adu=noise,
            )
        )
        profile_rows.extend(
            _profiles_for_run(
                source_path=source_path,
                parameter_family=family,
                parameter_value=float(value),
                threshold_sigma=float(threshold_sigma),
                min_flux_snr=float(min_flux_snr),
                baseline_sources=baseline_sources,
                current_sources=analyses[config],
                association_radius_px=float(association_radius_px),
            )
        )
        transition_rows.extend(
            _transitions_for_run(
                source_path=source_path,
                parameter_family=family,
                parameter_value=float(value),
                threshold_sigma=float(threshold_sigma),
                min_flux_snr=float(min_flux_snr),
                baseline_sources=baseline_sources,
                current_sources=analyses[config],
                association_radius_px=float(association_radius_px),
            )
        )

    threshold_runs = [row for row in run_rows if row.parameter_family == "threshold_sigma"]
    flux_runs = [row for row in run_rows if row.parameter_family == "min_flux_snr"]
    threshold_candidate_counts = [row.candidate_count for row in threshold_runs]
    flux_candidate_counts = [row.candidate_count for row in flux_runs]
    flux_quality_by_level = [row.quality_count for row in sorted(flux_runs, key=lambda row: row.parameter_value)]
    observations: dict[str, object] = {
        "baseline_candidate_count": int(detection_summary[baseline_config][0]),
        "baseline_returned_count": int(detection_summary[baseline_config][1]),
        "baseline_quality_count": int(detection_summary[baseline_config][2]),
        "threshold_candidate_count_range": [
            int(min(threshold_candidate_counts)) if threshold_candidate_counts else None,
            int(max(threshold_candidate_counts)) if threshold_candidate_counts else None,
        ],
        "flux_candidate_pool_stable": bool(
            flux_candidate_counts and len(set(flux_candidate_counts)) == 1
        ),
        "flux_quality_monotone_nonincreasing_with_threshold": bool(
            all(left >= right for left, right in zip(flux_quality_by_level, flux_quality_by_level[1:], strict=False))
        ),
        "interpretation": (
            "参数扫描只衡量同一坐标的算法响应稳定性；候选持续不等于物理恒星，"
            "质量通过也不替代星表/WCS、注入回收和跨帧身份确认。"
        ),
    }
    return FeatureParameterSensitivityResult(
        source_path=source_path,
        image_shape=tuple(int(value) for value in frame.data.shape),
        baseline_threshold_sigma=float(baseline_threshold_sigma),
        baseline_min_flux_snr=float(baseline_min_flux_snr),
        association_radius_px=float(association_radius_px),
        detector_parameters=detector_parameters,
        requested_threshold_levels=threshold_values,
        requested_flux_snr_levels=flux_values,
        runs=tuple(run_rows),
        profiles=tuple(profile_rows),
        transitions=tuple(transition_rows),
        observations=observations,
    )


def write_feature_parameter_sensitivity_artifacts(
    result: FeatureParameterSensitivityResult,
    out_dir: str | Path,
) -> Path:
    """写入参数运行表、类别追踪表和 JSON。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_rows = [row.as_dict() for row in result.runs]
    profile_rows = [row.as_dict() for row in result.profiles]
    with (output / "feature_parameter_runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(run_rows[0]) if run_rows else ["parameter_family"])
        writer.writeheader()
        writer.writerows(run_rows)
    with (output / "feature_parameter_profiles.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profile_rows[0]) if profile_rows else ["feature_class"])
        writer.writeheader()
        writer.writerows(profile_rows)
    transition_rows = [row.as_dict() for row in result.transitions]
    with (output / "feature_parameter_transitions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(transition_rows[0]) if transition_rows else ["baseline_feature_class"],
        )
        writer.writeheader()
        writer.writerows(transition_rows)
    (output / "feature_parameter_sensitivity.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output
