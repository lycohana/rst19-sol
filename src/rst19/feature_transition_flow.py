"""从已完成的序列类别转移 CSV 提取机制流向摘要。

这个模块是一个快速后处理器：它只读取
``rst19-feature-sequence`` 已经写出的
``sequence_feature_class_transition.csv``，不重新读取 FITS，也不重新运行
检测器。它把“任意类别响应”“同类响应”和“最大跨类响应”分开，避免把
类别标签误当作逐星物理身份。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .sequence_feature_comparison import FEATURE_CLASS_ORDER


@dataclass(frozen=True, slots=True)
class SequenceFeatureTransitionFlowRow:
    """一个层级和首帧特征类别的转移摘要。"""

    layer: str
    anchor_feature_class: str
    anchor_feature_class_label: str
    anchor_count: int
    frame_count: int
    possible_match_count: int
    matched_count: int
    same_class_match_count: int
    q_any: float | None
    q_same: float | None
    mean_matches_per_anchor: float | None
    top_non_same_feature_class: str | None
    top_non_same_feature_class_label: str | None
    top_non_same_matched_count: int
    top_non_same_fraction: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SequenceFeatureTransitionFlowResult:
    """可复核的类别流向后处理结果。"""

    source_csv: str
    input_row_count: int
    included_group_count: int
    skipped_zero_anchor_group_count: int
    rows: tuple[SequenceFeatureTransitionFlowRow, ...]
    interpretation_guardrails: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_csv": self.source_csv,
            "input_row_count": self.input_row_count,
            "included_group_count": self.included_group_count,
            "skipped_zero_anchor_group_count": self.skipped_zero_anchor_group_count,
            "rows": [row.as_dict() for row in self.rows],
            "interpretation_guardrails": list(self.interpretation_guardrails),
        }


_REQUIRED_FIELDS = {
    "layer",
    "anchor_feature_class",
    "anchor_feature_class_label",
    "response_feature_class",
    "response_feature_class_label",
    "anchor_count",
    "frame_count",
    "possible_match_count",
    "matched_count",
}


def _number(value: Any, *, field: str, path: Path) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: 字段 {field!r} 不是数字") from exc


def _integer(value: Any, *, field: str, path: Path) -> int:
    result = _number(value, field=field, path=path)
    if not result.is_integer():
        raise ValueError(f"{path}: 字段 {field!r} 不是整数")
    return int(result)


def _class_sort_key(feature_class: str) -> tuple[int, str]:
    try:
        return FEATURE_CLASS_ORDER.index(feature_class), feature_class
    except ValueError:
        return len(FEATURE_CLASS_ORDER), feature_class


def _layer_sort_key(layer: str) -> tuple[int, str]:
    return ({"candidate": 0, "quality": 1}.get(layer, 2), layer)


def _read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing = sorted(_REQUIRED_FIELDS - fields)
            if missing:
                raise ValueError(f"{path}: 缺少字段 {', '.join(missing)}")
            rows = [dict(row) for row in reader if any(str(value or "").strip() for value in row.values())]
    except OSError as exc:
        raise OSError(f"无法读取类别转移 CSV：{path}") from exc
    if not rows:
        raise ValueError(f"{path}: 类别转移 CSV 为空")
    return rows


def summarize_sequence_feature_transition_flow(
    csv_path: str | Path,
) -> SequenceFeatureTransitionFlowResult:
    """汇总已生成的类别转移 CSV，不触碰 FITS。"""

    path = Path(csv_path)
    source_rows = _read_rows(path)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        layer = str(row.get("layer", "")).strip()
        anchor_class = str(row.get("anchor_feature_class", "")).strip()
        if not layer or not anchor_class:
            raise ValueError(f"{path}: layer/anchor_feature_class 不能为空")
        groups[(layer, anchor_class)].append(row)

    result_rows: list[SequenceFeatureTransitionFlowRow] = []
    skipped_zero_anchor = 0
    for (layer, anchor_class), group in groups.items():
        first = group[0]
        anchor_count = _integer(first["anchor_count"], field="anchor_count", path=path)
        frame_count = _integer(first["frame_count"], field="frame_count", path=path)
        possible_match_count = _integer(
            first["possible_match_count"], field="possible_match_count", path=path
        )
        if anchor_count <= 0:
            skipped_zero_anchor += 1
            continue
        if frame_count <= 0 or possible_match_count < 0:
            raise ValueError(f"{path}: {layer}/{anchor_class} 的元数据不合法")

        response_counts: dict[str, int] = {}
        response_labels: dict[str, str] = {}
        matched_count = 0
        for row in group:
            row_anchor_count = _integer(row["anchor_count"], field="anchor_count", path=path)
            row_frame_count = _integer(row["frame_count"], field="frame_count", path=path)
            row_possible = _integer(
                row["possible_match_count"], field="possible_match_count", path=path
            )
            if (row_anchor_count, row_frame_count, row_possible) != (
                anchor_count,
                frame_count,
                possible_match_count,
            ):
                raise ValueError(f"{path}: {layer}/{anchor_class} 的元数据在响应类别之间不一致")
            response_class = str(row.get("response_feature_class", "")).strip()
            if not response_class:
                raise ValueError(f"{path}: response_feature_class 不能为空")
            matched = _integer(row["matched_count"], field="matched_count", path=path)
            if matched < 0:
                raise ValueError(f"{path}: matched_count 不能为负数")
            response_counts[response_class] = response_counts.get(response_class, 0) + matched
            response_labels.setdefault(
                response_class,
                str(row.get("response_feature_class_label", response_class)).strip() or response_class,
            )
            matched_count += matched

        same_class_match_count = response_counts.get(anchor_class, 0)
        non_same = [
            (feature_class, count)
            for feature_class, count in response_counts.items()
            if feature_class != anchor_class and count > 0
        ]
        non_same.sort(key=lambda item: (-item[1], _class_sort_key(item[0])))
        if non_same:
            top_class, top_count = non_same[0]
            top_label = response_labels[top_class]
            top_fraction = top_count / matched_count if matched_count else None
        else:
            top_class = None
            top_count = 0
            top_label = None
            top_fraction = None

        result_rows.append(
            SequenceFeatureTransitionFlowRow(
                layer=layer,
                anchor_feature_class=anchor_class,
                anchor_feature_class_label=(
                    str(first.get("anchor_feature_class_label", anchor_class)).strip() or anchor_class
                ),
                anchor_count=anchor_count,
                frame_count=frame_count,
                possible_match_count=possible_match_count,
                matched_count=matched_count,
                same_class_match_count=same_class_match_count,
                q_any=(matched_count / possible_match_count if possible_match_count else None),
                q_same=(same_class_match_count / matched_count if matched_count else None),
                mean_matches_per_anchor=(matched_count / anchor_count if anchor_count else None),
                top_non_same_feature_class=top_class,
                top_non_same_feature_class_label=top_label,
                top_non_same_matched_count=top_count,
                top_non_same_fraction=top_fraction,
            )
        )

    result_rows.sort(key=lambda row: (_layer_sort_key(row.layer), _class_sort_key(row.anchor_feature_class)))
    return SequenceFeatureTransitionFlowResult(
        source_csv=str(path),
        input_row_count=len(source_rows),
        included_group_count=len(result_rows),
        skipped_zero_anchor_group_count=skipped_zero_anchor,
        rows=tuple(result_rows),
        interpretation_guardrails=(
            "q_any 是首帧类别在后续帧得到任意邻域响应的比例，不是逐星身份概率。",
            "q_same 是发生响应后的条件同类率，不是类别分类准确率。",
            "最大跨类流向描述算法类别重写，不证明物理结构发生了变化。",
            "输入是已生成的类别转移 CSV；本后处理不重新读取 FITS 或运行检测器。",
        ),
    )


def _write_csv(path: Path, rows: Iterable[SequenceFeatureTransitionFlowRow]) -> None:
    materialized = list(rows)
    fieldnames = [field.name for field in fields(SequenceFeatureTransitionFlowRow)]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row.as_dict())


def write_sequence_feature_transition_flow_artifacts(
    result: SequenceFeatureTransitionFlowResult,
    out_dir: str | Path,
) -> Path:
    """写出类别流向 CSV 和 JSON。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "sequence_feature_transition_flow.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(output / "sequence_feature_transition_flow.csv", result.rows)
    return output


__all__ = [
    "SequenceFeatureTransitionFlowRow",
    "SequenceFeatureTransitionFlowResult",
    "summarize_sequence_feature_transition_flow",
    "write_sequence_feature_transition_flow_artifacts",
]
