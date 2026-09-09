"""比较已生成的 15 帧特征持久性审计结果。

这个模块只读取 ``feature_sequence_cli`` 已经写出的 JSON，不重新读取 FITS，
也不重新运行检测器。它用于把不同 ``min_flux_snr`` 或其它配置的结果放到
同一个口径下比较，避免把候选池、质量层和类别转移混成一个数字。

输入 JSON 的 ``frame_rows`` 可能一帧对应多行（每个特征类别一行），因此
比较器会显式检查唯一帧数，而不是把行数误当成帧数。
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
class SequenceFeatureRunSummary:
    """单个 JSON 审计产物的口径摘要。"""

    configuration: str
    json_path: str
    source_path: str
    frame_count: int
    unique_frame_count: int
    required_presence: int
    association_radius_px: float
    first_frame_candidate_count: int
    first_frame_quality_count: int
    first_frame_quality_fraction: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SequenceFeatureParameterRow:
    """一个配置、一个特征类别的候选/质量持久性。"""

    configuration: str
    feature_class: str
    feature_class_label: str
    anchor_candidate_count: int
    candidate_presence_ge_required_count: int
    candidate_presence_all_frames_count: int
    candidate_persistence_fraction: float | None
    anchor_quality_count: int
    quality_presence_ge_required_count: int
    quality_presence_all_frames_count: int
    quality_persistence_fraction: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SequenceFeatureTransitionComparisonRow:
    """一个配置下首帧类别到后续响应类别的转移。"""

    configuration: str
    layer: str
    anchor_feature_class: str
    response_feature_class: str
    anchor_count: int
    frame_count: int
    possible_match_count: int
    matched_count: int
    match_rate: float | None
    mean_matches_per_anchor: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SequenceFeatureParameterComparisonResult:
    """多组已完成序列审计的可复核比较结果。"""

    runs: tuple[SequenceFeatureRunSummary, ...]
    rows: tuple[SequenceFeatureParameterRow, ...]
    transitions: tuple[SequenceFeatureTransitionComparisonRow, ...]
    observations: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "runs": [row.as_dict() for row in self.runs],
            "rows": [row.as_dict() for row in self.rows],
            "transitions": [row.as_dict() for row in self.transitions],
            "observations": dict(self.observations),
        }


def _number(value: Any, *, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: 字段 {field!r} 不是数字") from exc
    return result


def _integer(value: Any, *, field: str, path: Path) -> int:
    result = _number(value, field=field, path=path)
    if not result.is_integer():
        raise ValueError(f"{path}: 字段 {field!r} 不是整数")
    return int(result)


def _fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _class_sort_key(feature_class: str) -> tuple[int, str]:
    try:
        return FEATURE_CLASS_ORDER.index(feature_class), feature_class
    except ValueError:
        return len(FEATURE_CLASS_ORDER), feature_class


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"无法读取序列审计 JSON：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"序列审计 JSON 无法解析：{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"序列审计 JSON 顶层必须是对象：{path}")
    return payload


def _load_run(configuration: str, path: Path) -> tuple[
    SequenceFeatureRunSummary,
    tuple[SequenceFeatureParameterRow, ...],
    tuple[SequenceFeatureTransitionComparisonRow, ...],
    dict[str, Any],
]:
    payload = _read_json(path)
    frame_count = _integer(payload.get("frame_count"), field="frame_count", path=path)
    required_presence = _integer(payload.get("required_presence"), field="required_presence", path=path)
    association_radius_px = _number(
        payload.get("association_radius_px"),
        field="association_radius_px",
        path=path,
    )
    if frame_count < 1 or required_presence < 1 or association_radius_px <= 0:
        raise ValueError(f"{path}: frame_count/required_presence/association_radius_px 不合法")

    frame_rows = payload.get("frame_rows")
    if not isinstance(frame_rows, list) or not frame_rows:
        raise ValueError(f"{path}: 缺少非空 frame_rows")
    frame_indices = {
        _integer(row.get("frame_index"), field="frame_index", path=path)
        for row in frame_rows
        if isinstance(row, dict) and "frame_index" in row
    }
    if len(frame_indices) != frame_count:
        raise ValueError(
            f"{path}: metadata frame_count={frame_count}，但 frame_rows 的唯一帧数为 {len(frame_indices)}"
        )
    first_frame_rows = [
        row for row in frame_rows if isinstance(row, dict) and _integer(row.get("frame_index"), field="frame_index", path=path) == 1
    ]
    if not first_frame_rows:
        first_frame_rows = [row for row in frame_rows if isinstance(row, dict)]
    if not first_frame_rows:
        raise ValueError(f"{path}: frame_rows 没有可用记录")
    first = first_frame_rows[0]
    first_candidate_count = _integer(first.get("candidate_count"), field="candidate_count", path=path)
    first_quality_count = _integer(first.get("quality_count"), field="quality_count", path=path)
    source_path = str(first.get("path", ""))

    persistence_rows = payload.get("persistence_rows")
    if not isinstance(persistence_rows, list):
        raise ValueError(f"{path}: 缺少 persistence_rows")
    rows: list[SequenceFeatureParameterRow] = []
    for item in persistence_rows:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: persistence_rows 中存在非对象记录")
        feature_class = str(item.get("feature_class", ""))
        if not feature_class:
            raise ValueError(f"{path}: persistence_rows 存在空 feature_class")
        anchor_candidate_count = _integer(
            item.get("anchor_candidate_count"), field="anchor_candidate_count", path=path
        )
        candidate_ge_required = _integer(
            item.get("candidate_presence_ge_required_count"),
            field="candidate_presence_ge_required_count",
            path=path,
        )
        candidate_all_frames = _integer(
            item.get("candidate_presence_all_frames_count"),
            field="candidate_presence_all_frames_count",
            path=path,
        )
        anchor_quality_count = _integer(item.get("anchor_quality_count"), field="anchor_quality_count", path=path)
        quality_ge_required = _integer(
            item.get("quality_presence_ge_required_count"),
            field="quality_presence_ge_required_count",
            path=path,
        )
        quality_all_frames = _integer(
            item.get("quality_presence_all_frames_count"),
            field="quality_presence_all_frames_count",
            path=path,
        )
        rows.append(
            SequenceFeatureParameterRow(
                configuration=configuration,
                feature_class=feature_class,
                feature_class_label=str(item.get("feature_class_label", feature_class)),
                anchor_candidate_count=anchor_candidate_count,
                candidate_presence_ge_required_count=candidate_ge_required,
                candidate_presence_all_frames_count=candidate_all_frames,
                candidate_persistence_fraction=_fraction(candidate_ge_required, anchor_candidate_count),
                anchor_quality_count=anchor_quality_count,
                quality_presence_ge_required_count=quality_ge_required,
                quality_presence_all_frames_count=quality_all_frames,
                quality_persistence_fraction=_fraction(quality_ge_required, anchor_quality_count),
            )
        )

    transition_payload = payload.get("class_transition_rows", [])
    if not isinstance(transition_payload, list):
        raise ValueError(f"{path}: class_transition_rows 必须是列表")
    transitions: list[SequenceFeatureTransitionComparisonRow] = []
    for item in transition_payload:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: class_transition_rows 中存在非对象记录")
        transitions.append(
            SequenceFeatureTransitionComparisonRow(
                configuration=configuration,
                layer=str(item.get("layer", "")),
                anchor_feature_class=str(item.get("anchor_feature_class", "")),
                response_feature_class=str(item.get("response_feature_class", "")),
                anchor_count=_integer(item.get("anchor_count"), field="anchor_count", path=path),
                frame_count=_integer(item.get("frame_count"), field="frame_count", path=path),
                possible_match_count=_integer(
                    item.get("possible_match_count"), field="possible_match_count", path=path
                ),
                matched_count=_integer(item.get("matched_count"), field="matched_count", path=path),
                match_rate=None
                if item.get("match_rate") is None
                else _number(item.get("match_rate"), field="match_rate", path=path),
                mean_matches_per_anchor=None
                if item.get("mean_matches_per_anchor") is None
                else _number(item.get("mean_matches_per_anchor"), field="mean_matches_per_anchor", path=path),
            )
        )

    summary = SequenceFeatureRunSummary(
        configuration=configuration,
        json_path=str(path),
        source_path=source_path,
        frame_count=frame_count,
        unique_frame_count=len(frame_indices),
        required_presence=required_presence,
        association_radius_px=association_radius_px,
        first_frame_candidate_count=first_candidate_count,
        first_frame_quality_count=first_quality_count,
        first_frame_quality_fraction=_fraction(first_quality_count, first_candidate_count),
    )
    metadata = {
        "payload": payload,
        "summary": summary,
    }
    return summary, tuple(rows), tuple(transitions), metadata


def compare_sequence_feature_runs(
    runs: Mapping[str, str | Path],
) -> SequenceFeatureParameterComparisonResult:
    """比较两组或更多已完成的序列审计 JSON。

    所有输入必须使用相同的帧数、持久性门槛和关联半径；否则比较会被拒绝，
    以免把不同实验设计误当成参数敏感性。
    """

    if len(runs) < 2:
        raise ValueError("至少需要两组序列审计 JSON 才能进行比较")
    loaded = [_load_run(str(configuration), Path(path)) for configuration, path in runs.items()]
    summaries = tuple(item[0] for item in loaded)
    reference = summaries[0]
    for summary in summaries[1:]:
        if summary.frame_count != reference.frame_count:
            raise ValueError("各运行的 frame_count 不一致，不能比较")
        if summary.required_presence != reference.required_presence:
            raise ValueError("各运行的 required_presence 不一致，不能比较")
        if abs(summary.association_radius_px - reference.association_radius_px) > 1e-9:
            raise ValueError("各运行的 association_radius_px 不一致，不能比较")

    rows = tuple(row for _summary, run_rows, _transitions, _metadata in loaded for row in run_rows)
    transitions = tuple(
        transition
        for _summary, _run_rows, run_transitions, _metadata in loaded
        for transition in run_transitions
    )
    rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.configuration,
                _class_sort_key(row.feature_class),
            ),
        )
    )
    transitions = tuple(
        sorted(
            transitions,
            key=lambda row: (
                row.configuration,
                row.layer,
                _class_sort_key(row.anchor_feature_class),
                _class_sort_key(row.response_feature_class),
            ),
        )
    )

    quality_persistence: dict[str, dict[str, float | None]] = {}
    for summary in summaries:
        quality_persistence[summary.configuration] = {
            row.feature_class: row.quality_persistence_fraction
            for row in rows
            if row.configuration == summary.configuration
        }
    observations: dict[str, object] = {
        "same_sequence_geometry": True,
        "same_frame_count": reference.frame_count,
        "same_required_presence": reference.required_presence,
        "same_association_radius_px": reference.association_radius_px,
        "first_frame_counts": {
            summary.configuration: {
                "candidate_count": summary.first_frame_candidate_count,
                "quality_count": summary.first_frame_quality_count,
                "quality_fraction": summary.first_frame_quality_fraction,
            }
            for summary in summaries
        },
        "quality_persistence_by_configuration": quality_persistence,
        "interpretation_guardrails": [
            "configuration 是调用者提供的标签；JSON 本身未声明 min_flux_snr 等命令行参数。",
            "candidate persistence 不是一对一身份确认，仍可能被邻近源偶然命中。",
            "quality persistence 只描述通过当前质量层的响应，不能单独替代星表真值。",
        ],
    }
    return SequenceFeatureParameterComparisonResult(
        runs=summaries,
        rows=rows,
        transitions=transitions,
        observations=observations,
    )


def _write_csv(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8-sig")
        return
    first = rows[0]
    if not hasattr(first, "as_dict"):
        raise TypeError("CSV rows must expose as_dict()")
    fieldnames = list(first.as_dict().keys())  # type: ignore[union-attr]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())  # type: ignore[union-attr]


def write_sequence_feature_comparison_artifacts(
    result: SequenceFeatureParameterComparisonResult,
    out_dir: str | Path,
) -> Path:
    """写出 JSON 与两张 CSV 比较表。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "sequence_feature_parameter_comparison.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(output / "sequence_feature_parameter_runs.csv", result.runs)
    _write_csv(output / "sequence_feature_parameter_profiles.csv", result.rows)
    _write_csv(output / "sequence_feature_parameter_transitions.csv", result.transitions)
    return output


__all__ = [
    "SequenceFeatureRunSummary",
    "SequenceFeatureParameterRow",
    "SequenceFeatureTransitionComparisonRow",
    "SequenceFeatureParameterComparisonResult",
    "compare_sequence_feature_runs",
    "write_sequence_feature_comparison_artifacts",
]
