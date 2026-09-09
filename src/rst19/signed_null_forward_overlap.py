"""符号反相质量源与正向检测结果的局部交叉审计。

符号反相对照只能说明负向尾部会不会被检测器提出。为了检查一个反相质量源
是否同时对应原图中的正向响应，本模块在同一组 FITS 帧上重新运行当前正向
检测器，并只在反相源坐标周围做邻域查询。它不保存正向全量源表，不修改检测
缓存，也不把“正向近邻”解释成星表身份或物理真值。
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from scipy.spatial import cKDTree

from .experiments import analyze_frame, classify_source_feature


_INNER_RADIUS_PX = 1.0
_MATCH_RADIUS_PX = 2.0
_REPORT_RADIUS_PX = 4.0


@dataclass(frozen=True, slots=True)
class SignedNullForwardOverlapSourceRow:
    """一个反相质量源对应的正向候选/质量源邻域证据。"""

    frame_index: int
    frame_path: str
    reverse_detection_id: int
    reverse_x: float
    reverse_y: float
    raw_evidence_layer: str
    reverse_flux_snr: float | None
    candidate_count_r1: int
    candidate_count_r2: int
    candidate_count_r4: int
    quality_count_r1: int
    quality_count_r2: int
    quality_count_r4: int
    overlap_class: str
    nearest_candidate_detection_id: int | None
    nearest_candidate_x: float | None
    nearest_candidate_y: float | None
    nearest_candidate_peak_x: float | None
    nearest_candidate_peak_y: float | None
    nearest_candidate_peak: float | None
    nearest_candidate_flux_snr: float | None
    nearest_candidate_filter_snr: float | None
    nearest_candidate_fwhm: float | None
    nearest_candidate_feature_class: str | None
    nearest_candidate_quality_passed: bool | None
    nearest_candidate_flags: str
    nearest_candidate_distance_px: float | None
    nearest_quality_detection_id: int | None
    nearest_quality_x: float | None
    nearest_quality_y: float | None
    nearest_quality_peak_x: float | None
    nearest_quality_peak_y: float | None
    nearest_quality_peak: float | None
    nearest_quality_flux_snr: float | None
    nearest_quality_filter_snr: float | None
    nearest_quality_fwhm: float | None
    nearest_quality_feature_class: str | None
    nearest_quality_quality_passed: bool | None
    nearest_quality_flags: str
    nearest_quality_distance_px: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SignedNullForwardOverlapFrameRow:
    """逐帧正向检测计数及反相质量源数量。"""

    frame_index: int
    frame_path: str
    candidate_count: int
    returned_count: int
    quality_count: int
    reverse_quality_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SignedNullForwardOverlapResult:
    """符号反相源和正向检测结果的局部交叉审计结果。"""

    frame_count: int
    reverse_quality_source_count: int
    source_rows: tuple[SignedNullForwardOverlapSourceRow, ...]
    frame_rows: tuple[SignedNullForwardOverlapFrameRow, ...]
    raw_evidence_layer_counts: dict[str, int]
    overlap_class_counts: dict[str, int]
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "reverse_quality_source_count": self.reverse_quality_source_count,
            "source_row_count": len(self.source_rows),
            "frame_rows": [row.as_dict() for row in self.frame_rows],
            "frame_summaries": [row.as_dict() for row in self.frame_rows],
            "raw_evidence_layer_counts": dict(self.raw_evidence_layer_counts),
            "overlap_class_counts": dict(self.overlap_class_counts),
            "parameters": self.parameters,
            "detector_parameters": self.parameters,
            "conclusion": self.conclusion,
            "interpretation": self.conclusion,
        }


def _load_reverse_source_rows(path: str | Path) -> tuple[dict[str, str], ...]:
    """读取 ``signed_null_sequence_quality_sources.csv`` 的必要字段。"""

    source_path = Path(path)
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {"frame_index", "detection_id", "x", "y", "raw_evidence_layer", "flux_snr"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"reverse source CSV is missing fields: {', '.join(missing)}")
        return tuple(dict(row) for row in reader)


def _source_summary(source: object) -> dict[str, object]:
    """把检测对象压缩成可写入交叉表的字段。"""

    return {
        "detection_id": int(source.detection_id),
        "x": float(source.x),
        "y": float(source.y),
        "peak_x": float(source.peak_x if source.peak_x is not None else source.x),
        "peak_y": float(source.peak_y if source.peak_y is not None else source.y),
        "peak": float(source.peak),
        "flux_snr": float(source.flux_snr) if source.flux_snr is not None else None,
        "filter_snr": float(source.filter_snr) if source.filter_snr is not None else None,
        "fwhm": float(source.fwhm) if source.fwhm is not None else None,
        "feature_class": str(classify_source_feature(source)),
        "quality_passed": bool(source.quality_passed),
        "flags": "|".join(str(flag) for flag in source.flags),
    }


def _make_tree(sources: tuple[object, ...]) -> cKDTree | None:
    if not sources:
        return None
    return cKDTree([(float(source.x), float(source.y)) for source in sources])


def _nearest(
    point: tuple[float, float],
    sources: tuple[object, ...],
    tree: cKDTree | None,
) -> dict[str, object] | None:
    if tree is None:
        return None
    distance, index = tree.query(point, k=1)
    result = _source_summary(sources[int(index)])
    result["distance_px"] = float(distance)
    return result


def _count_within(point: tuple[float, float], tree: cKDTree | None, radius: float) -> int:
    if tree is None:
        return 0
    return int(len(tree.query_ball_point(point, radius)))


def _row_from_values(
    *,
    frame_index: int,
    frame_path: Path,
    reverse: dict[str, str],
    candidate_sources: tuple[object, ...],
    candidate_tree: cKDTree | None,
    quality_sources: tuple[object, ...],
    quality_tree: cKDTree | None,
) -> SignedNullForwardOverlapSourceRow:
    point = (float(reverse["x"]), float(reverse["y"]))
    candidate_counts = tuple(
        _count_within(point, candidate_tree, radius)
        for radius in (_INNER_RADIUS_PX, _MATCH_RADIUS_PX, _REPORT_RADIUS_PX)
    )
    quality_counts = tuple(
        _count_within(point, quality_tree, radius)
        for radius in (_INNER_RADIUS_PX, _MATCH_RADIUS_PX, _REPORT_RADIUS_PX)
    )
    nearest_candidate = _nearest(point, candidate_sources, candidate_tree)
    nearest_quality = _nearest(point, quality_sources, quality_tree)
    if quality_counts[1] > 0:
        overlap_class = "quality_counterpart_within_2px"
    elif candidate_counts[1] > 0:
        overlap_class = "candidate_only_within_2px"
    elif candidate_counts[2] > 0:
        overlap_class = "candidate_only_within_4px"
    else:
        overlap_class = "no_forward_candidate_within_4px"

    def value(source: dict[str, object] | None, key: str) -> object:
        return source.get(key) if source is not None else None

    return SignedNullForwardOverlapSourceRow(
        frame_index=frame_index,
        frame_path=str(frame_path),
        reverse_detection_id=int(reverse["detection_id"]),
        reverse_x=point[0],
        reverse_y=point[1],
        raw_evidence_layer=str(reverse["raw_evidence_layer"]),
        reverse_flux_snr=float(reverse["flux_snr"]) if reverse["flux_snr"] else None,
        candidate_count_r1=candidate_counts[0],
        candidate_count_r2=candidate_counts[1],
        candidate_count_r4=candidate_counts[2],
        quality_count_r1=quality_counts[0],
        quality_count_r2=quality_counts[1],
        quality_count_r4=quality_counts[2],
        overlap_class=overlap_class,
        nearest_candidate_detection_id=value(nearest_candidate, "detection_id"),
        nearest_candidate_x=value(nearest_candidate, "x"),
        nearest_candidate_y=value(nearest_candidate, "y"),
        nearest_candidate_peak_x=value(nearest_candidate, "peak_x"),
        nearest_candidate_peak_y=value(nearest_candidate, "peak_y"),
        nearest_candidate_peak=value(nearest_candidate, "peak"),
        nearest_candidate_flux_snr=value(nearest_candidate, "flux_snr"),
        nearest_candidate_filter_snr=value(nearest_candidate, "filter_snr"),
        nearest_candidate_fwhm=value(nearest_candidate, "fwhm"),
        nearest_candidate_feature_class=value(nearest_candidate, "feature_class"),
        nearest_candidate_quality_passed=value(nearest_candidate, "quality_passed"),
        nearest_candidate_flags=str(value(nearest_candidate, "flags") or ""),
        nearest_candidate_distance_px=value(nearest_candidate, "distance_px"),
        nearest_quality_detection_id=value(nearest_quality, "detection_id"),
        nearest_quality_x=value(nearest_quality, "x"),
        nearest_quality_y=value(nearest_quality, "y"),
        nearest_quality_peak_x=value(nearest_quality, "peak_x"),
        nearest_quality_peak_y=value(nearest_quality, "peak_y"),
        nearest_quality_peak=value(nearest_quality, "peak"),
        nearest_quality_flux_snr=value(nearest_quality, "flux_snr"),
        nearest_quality_filter_snr=value(nearest_quality, "filter_snr"),
        nearest_quality_fwhm=value(nearest_quality, "fwhm"),
        nearest_quality_feature_class=value(nearest_quality, "feature_class"),
        nearest_quality_quality_passed=value(nearest_quality, "quality_passed"),
        nearest_quality_flags=str(value(nearest_quality, "flags") or ""),
        nearest_quality_distance_px=value(nearest_quality, "distance_px"),
    )


def run_signed_null_forward_overlap(
    frame_paths: Iterable[str | Path],
    reverse_source_csv: str | Path,
    *,
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    aperture_radius: int = 4,
    psf_fwhm: float = 2.0,
    background_box_size: int = 128,
    background_sample_limit: int = 100_000,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    proposal_mode: str = "hybrid",
    progress: Callable[[int, int], None] | None = None,
) -> SignedNullForwardOverlapResult:
    """在当前正向检测器中查找反相质量源的局部对应物。

    ``frame_index`` 使用 CSV 中的 1-based 帧号。匹配只在原始 detector 坐标
    中进行，且不使用跨帧注册；这正是为了回答“同一帧同一坐标附近是否还有
    正向检测”的局部问题。``quality_counterpart`` 只表示当前质量规则下的
    邻近测量，不表示独立恒星、双星或星表身份。
    """

    paths = tuple(Path(path) for path in frame_paths)
    if not paths:
        raise ValueError("frame_paths must not be empty")
    if threshold_sigma <= 0 or min_distance < 1 or aperture_radius < 1 or psf_fwhm <= 0:
        raise ValueError("detector parameters must be positive")
    if background_box_size < 16 or background_sample_limit < 1 or min_flux_snr <= 0:
        raise ValueError("background and quality parameters are invalid")
    if not 1 <= min_psf_support_pixels <= 9:
        raise ValueError("min_psf_support_pixels must be within 1..9")
    reverse_rows = _load_reverse_source_rows(reverse_source_csv)
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in reverse_rows:
        try:
            frame_index = int(row["frame_index"])
            int(row["detection_id"])
            float(row["x"])
            float(row["y"])
            if row["flux_snr"]:
                float(row["flux_snr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("reverse source CSV contains an invalid numeric row") from exc
        if not 1 <= frame_index <= len(paths):
            raise ValueError(f"reverse source frame_index is outside 1..{len(paths)}")
        by_frame[frame_index].append(row)

    detector_parameters = {
        "threshold_sigma": threshold_sigma,
        "min_distance": min_distance,
        "aperture_radius": aperture_radius,
        "psf_fwhm": psf_fwhm,
        "background_box_size": background_box_size,
        "background_sample_limit": background_sample_limit,
        "min_flux_snr": min_flux_snr,
        "min_psf_support_pixels": min_psf_support_pixels,
        "proposal_mode": proposal_mode,
        "reject_linear_artifacts": True,
        "refine_local_background": True,
        "max_sources": None,
        "inner_radius_px": _INNER_RADIUS_PX,
        "match_radius_px": _MATCH_RADIUS_PX,
        "report_radius_px": _REPORT_RADIUS_PX,
        "reverse_source_csv": str(reverse_source_csv),
    }
    source_rows: list[SignedNullForwardOverlapSourceRow] = []
    frame_rows: list[SignedNullForwardOverlapFrameRow] = []
    layers = Counter(row["raw_evidence_layer"] for row in reverse_rows)
    overlap_classes: Counter[str] = Counter()
    for frame_index, path in enumerate(paths, start=1):
        analysis = analyze_frame(
            path,
            threshold_sigma=threshold_sigma,
            min_distance=min_distance,
            aperture_radius=aperture_radius,
            psf_fwhm=psf_fwhm,
            background_box_size=background_box_size,
            background_sample_limit=background_sample_limit,
            min_flux_snr=min_flux_snr,
            min_psf_support_pixels=min_psf_support_pixels,
            max_sources=None,
            reject_linear_artifacts=True,
            proposal_mode=proposal_mode,
            refine_local_background=True,
        )
        candidate_sources = tuple(analysis.detection.sources)
        quality_sources = tuple(analysis.detection.quality_sources)
        frame_rows.append(
            SignedNullForwardOverlapFrameRow(
                frame_index=frame_index,
                frame_path=str(path),
                candidate_count=int(analysis.detection.candidate_count),
                returned_count=int(analysis.detection.returned_count),
                quality_count=int(analysis.detection.star_count),
                reverse_quality_count=len(by_frame.get(frame_index, ())),
            )
        )
        candidate_tree = _make_tree(candidate_sources)
        quality_tree = _make_tree(quality_sources)
        for reverse in by_frame.get(frame_index, ()):
            row = _row_from_values(
                frame_index=frame_index,
                frame_path=path,
                reverse=reverse,
                candidate_sources=candidate_sources,
                candidate_tree=candidate_tree,
                quality_sources=quality_sources,
                quality_tree=quality_tree,
            )
            source_rows.append(row)
            overlap_classes[row.overlap_class] += 1
        if progress is not None:
            progress(frame_index, len(paths))

    no_quality = sum(row.quality_count_r4 == 0 for row in source_rows)
    no_candidate = sum(
        row.overlap_class == "no_forward_candidate_within_4px" for row in source_rows
    )
    conclusion = (
        f"{no_quality}/{len(source_rows)} reverse quality sources have no forward quality "
        f"neighbor within 4 px; {no_candidate}/{len(source_rows)} have no forward candidate "
        "within 4 px. This is detector-level local overlap, not a source-identity match."
    )
    return SignedNullForwardOverlapResult(
        frame_count=len(paths),
        reverse_quality_source_count=len(reverse_rows),
        source_rows=tuple(source_rows),
        frame_rows=tuple(frame_rows),
        raw_evidence_layer_counts=dict(layers),
        overlap_class_counts=dict(overlap_classes),
        parameters=detector_parameters,
        conclusion=conclusion,
    )


def write_signed_null_forward_overlap_artifacts(
    result: SignedNullForwardOverlapResult,
    out_dir: str | Path,
) -> Path:
    """写入局部交叉 CSV 和摘要 JSON，不写检测缓存。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_rows = [row.as_dict() for row in result.source_rows]
    if source_rows:
        with (output / "signed_null_forward_overlap.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
            writer.writeheader()
            writer.writerows(source_rows)
    else:
        (output / "signed_null_forward_overlap.csv").write_text("\n", encoding="utf-8")
    payload = result.as_dict()
    payload["outputs"] = [
        "signed_null_forward_overlap.csv",
        "signed_null_forward_overlap_summary.json",
    ]
    with (output / "signed_null_forward_overlap_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output
