"""把创新候选整理成可交付、可复核的证据包。

本模块只读取已经生成的实验结果（JSON 或 CSV），不会读取 FITS、重跑检测，
也不会把缓存中的背景检测数解释为真实恒星真值。当前默认的主创新是：

    真实背景分层注入—回收的条件化可探测性剖面

它回答的是“在当前真实背景、当前注入模型和当前检测参数下，已知注入源在
不同环境中的候选层/质量层召回率如何变化”。这是一个有边界的经验测量，
不是整幅图的物理完备率、precision、FDR 或绝对星等灵敏度。

``innovation.py`` 继续负责 15 帧关系、运动和配准证据；本模块负责把创新
交付边界单独冻结出来，避免把诊断图表直接当成比赛结论。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
PRIMARY_INNOVATION_ID = "conditional_detectability"
PRIMARY_STATUS = "CONDITIONAL_DEFENSIBLE"
MIN_PRIMARY_INJECTIONS = 8
MIN_PRIMARY_LEVELS = 3
MIN_PRIMARY_TRIALS = 3

_REQUIRED_FIELDS = (
    "stratum",
    "peak_excess_adu",
    "injected_count",
    "candidate_recovered_count",
    "quality_recovered_count",
    "ambiguous_injection_count",
    "unambiguous_injected_count",
    "trial_count",
)

_INT_FIELDS = {
    "psf_source_count",
    "injected_count",
    "candidate_recovered_count",
    "quality_recovered_count",
    "ambiguous_injection_count",
    "unambiguous_injected_count",
    "baseline_candidate_count",
    "baseline_quality_count",
    "trial_count",
}

_FLOAT_FIELDS = {
    "psf_median_fwhm_px",
    "psf_kernel_sum",
    "peak_excess_adu",
    "candidate_recall",
    "quality_recall",
    "unambiguous_candidate_recall",
    "unambiguous_quality_recall",
    "net_candidate_delta",
    "net_quality_delta",
    "local_background_adu",
    "local_noise_adu",
    "special_pixel_fraction",
    "nearest_baseline_source_px",
    # These values are means over independent injection layouts.  They are
    # expected to be fractional when the per-layout counts differ, so they
    # must not be parsed as integer event counts.
    "mean_candidate_count",
    "mean_quality_count",
    "mean_background_candidate_count",
    "mean_background_quality_count",
}

_COUNT_FIELDS = _INT_FIELDS - {"trial_count"}


def _as_float(value: object) -> float | None:
    """把输入转成有限浮点数；空值和非有限值统一为 ``None``。"""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: object) -> int | None:
    """把 JSON/CSV 中的整数转成 int，不悄悄截断小数。"""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """返回二项召回率的 Wilson 95% 区间。

    区间仅用于表达小样本不确定度，不用于显著性检验，也不把当前 8 个位置
    包装成大量独立观测。``successes`` 和 ``trials`` 必须是整数计数。
    """

    if isinstance(successes, bool) or isinstance(trials, bool):
        raise ValueError("successes/trials must be integers")
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise ValueError("successes/trials must be integers")
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials > 0")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be a positive finite number")

    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _read_text_document(source: str | Path) -> tuple[Any, Path | None]:
    """读取 JSON/CSV 输入，返回载荷和真实路径。"""

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"evidence artifact not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv":
        return {"rows": list(csv.DictReader(io.StringIO(text)))}, path.resolve()
    try:
        return json.loads(text), path.resolve()
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path}: {exc}") from exc


def _payload(source: str | Path | Mapping[str, Any]) -> tuple[Any, Path | None]:
    if isinstance(source, Mapping):
        return source, None
    return _read_text_document(source)


def _normalise_row(raw: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for raw_key, raw_value in raw.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip()
        if not key:
            continue
        if raw_value == "" or raw_value is None:
            value: Any = None
        elif key in _INT_FIELDS:
            value = _as_int(raw_value)
            if value is None:
                raise ValueError(f"rows[{row_index}].{key} must be an integer")
        elif key in _FLOAT_FIELDS:
            value = _as_float(raw_value)
            if value is None:
                raise ValueError(f"rows[{row_index}].{key} must be a finite number")
        else:
            value = raw_value
        row[key] = value

    for field in _REQUIRED_FIELDS:
        if field not in row or row[field] is None:
            raise ValueError(f"rows[{row_index}] is missing required field: {field}")
    if not isinstance(row["stratum"], str) or not row["stratum"].strip():
        raise ValueError(f"rows[{row_index}].stratum must be a non-empty string")
    row["stratum"] = row["stratum"].strip()
    row.setdefault("stratum_label", row["stratum"])
    if not isinstance(row["stratum_label"], str) or not row["stratum_label"].strip():
        row["stratum_label"] = row["stratum"]
    row["stratum_label"] = row["stratum_label"].strip()
    return row


def _validate_row(row: Mapping[str, Any], row_index: int) -> None:
    for field in _COUNT_FIELDS | {"trial_count"}:
        value = row.get(field)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"rows[{row_index}].{field} must be a non-negative integer")

    injected = row["injected_count"]
    candidate = row["candidate_recovered_count"]
    quality = row["quality_recovered_count"]
    ambiguous = row["ambiguous_injection_count"]
    unambiguous = row["unambiguous_injected_count"]
    trials = row["trial_count"]
    if trials < 1:
        raise ValueError(f"rows[{row_index}].trial_count must be >= 1")
    if candidate > injected:
        raise ValueError(f"rows[{row_index}] candidate recovery exceeds injected count")
    if quality > candidate:
        raise ValueError(f"rows[{row_index}] quality recovery exceeds candidate recovery")
    if ambiguous > injected:
        raise ValueError(f"rows[{row_index}] ambiguous count exceeds injected count")
    if unambiguous > injected:
        raise ValueError(f"rows[{row_index}] unambiguous count exceeds injected count")
    for field in ("candidate_recall", "quality_recall", "unambiguous_candidate_recall", "unambiguous_quality_recall"):
        value = row.get(field)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
            raise ValueError(f"rows[{row_index}].{field} must be in [0, 1]")
    noise = row.get("local_noise_adu")
    if noise is not None and (not isinstance(noise, (int, float)) or not math.isfinite(float(noise)) or float(noise) < 0):
        raise ValueError(f"rows[{row_index}].local_noise_adu must be a finite non-negative number")
    for field in (
        "mean_candidate_count",
        "mean_quality_count",
        "mean_background_candidate_count",
        "mean_background_quality_count",
    ):
        value = row.get(field)
        if value is not None and (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"rows[{row_index}].{field} must be a finite non-negative number")


def load_stratified_rows(source: str | Path | Mapping[str, Any]) -> list[dict[str, Any]]:
    """加载并校验“一行=一个条件/注入强度”实验表。

    支持当前实验 JSON（顶层含 ``rows``）和同字段 CSV。为避免把重复实验
    静默合并，要求 ``(stratum, peak_excess_adu)`` 唯一；重复实验应先在实验
    端聚合，并在 ``trial_count`` 中保留独立布局数。
    """

    payload, _ = _payload(source)
    if isinstance(payload, Mapping):
        raw_rows = payload.get("rows")
    else:
        raw_rows = payload
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("evidence artifact must contain a sequence field named 'rows'")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"rows[{index}] must be an object")
        row = _normalise_row(raw, index)
        _validate_row(row, index)
        key = (row["stratum"], float(row["peak_excess_adu"]))
        if key in seen:
            raise ValueError(f"duplicate stratum/peak grid cell: {key[0]} / {key[1]}")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("evidence artifact contains no rows")
    return rows


def _file_metadata(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return {"path": None, "sha256": None, "bytes": None, "format": "mapping"}
    path = Path(source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size, "format": path.suffix.lower().lstrip(".") or "text"}


def _safe_rate(successes: int, trials: int) -> float | None:
    return successes / trials if trials > 0 else None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _normalise_float(value: object) -> float | None:
    return _as_float(value)


def _derived_row(row: Mapping[str, Any], min_injections: int) -> dict[str, Any]:
    ambiguous = int(row["ambiguous_injection_count"])
    unambiguous = int(row["unambiguous_injected_count"])
    injected = int(row["injected_count"])
    candidate_count = int(row["candidate_recovered_count"])
    quality_count = int(row["quality_recovered_count"])
    trial_count = int(row["trial_count"])
    is_core = ambiguous == 0 and unambiguous > 0
    minimum_unambiguous_positions = min_injections * trial_count
    primary_eligible = is_core and unambiguous >= minimum_unambiguous_positions
    if is_core:
        denominator = unambiguous
        denominator_kind = "unambiguous_injected"
        rate_semantics = "primary_unambiguous_recall"
        candidate_recall = _safe_rate(candidate_count, denominator)
        quality_recall = _safe_rate(quality_count, denominator)
        candidate_ci = wilson_interval(candidate_count, denominator) if denominator > 0 else (None, None)
        quality_ci = wilson_interval(quality_count, denominator) if denominator > 0 else (None, None)
    else:
        denominator = injected
        denominator_kind = "all_injected_diagnostic"
        rate_semantics = "diagnostic_all_injected_rate"
        # 困难条件的注入位置不能独立归因时，不输出看起来像普通召回率的
        # 0/8；保留原始计数和 denominator_kind 供机制审计即可。
        if unambiguous > 0:
            candidate_recall = _normalise_float(row.get("candidate_recall"))
            quality_recall = _normalise_float(row.get("quality_recall"))
            if candidate_recall is None:
                candidate_recall = _safe_rate(candidate_count, denominator)
            if quality_recall is None:
                quality_recall = _safe_rate(quality_count, denominator)
        else:
            candidate_recall = None
            quality_recall = None
        candidate_ci = (None, None)
        quality_ci = (None, None)
    noise = _normalise_float(row.get("local_noise_adu"))
    peak = float(row["peak_excess_adu"])
    return {
        "row_index": row.get("row_index"),
        "source_path": row.get("source_path"),
        "stratum": row["stratum"],
        "stratum_label": row.get("stratum_label", row["stratum"]),
        "control_class": "core" if is_core else "hard_control",
        "primary_eligible": primary_eligible,
        "control_reason": None if is_core else (
            "存在歧义注入位置，或没有可用于主结论的无歧义注入分母；仅作为困难条件对照。"
        ),
        "proposal_mode": row.get("proposal_mode"),
        "psf_model": row.get("psf_model"),
        "peak_excess_adu": peak,
        "injected_count": injected,
        "unambiguous_injected_count": unambiguous,
        "ambiguous_injection_count": ambiguous,
        "denominator_kind": denominator_kind,
        "rate_semantics": rate_semantics,
        "ambiguous_count_is_nonexclusive": True,
        "denominator_count": denominator,
        "minimum_unambiguous_positions_required": minimum_unambiguous_positions,
        "candidate_recovered_count": candidate_count,
        "quality_recovered_count": quality_count,
        "candidate_recall": candidate_recall,
        "candidate_recall_wilson95_low": candidate_ci[0],
        "candidate_recall_wilson95_high": candidate_ci[1],
        "quality_recall": quality_recall,
        "quality_recall_wilson95_low": quality_ci[0],
        "quality_recall_wilson95_high": quality_ci[1],
        "quality_filter_gap": (
            candidate_recall - quality_recall
            if candidate_recall is not None and quality_recall is not None
            else None
        ),
        "local_background_adu": _normalise_float(row.get("local_background_adu")),
        "local_noise_adu": noise,
        "nominal_level_over_local_noise": _safe_ratio(peak, noise),
        "special_pixel_fraction": _normalise_float(row.get("special_pixel_fraction")),
        "nearest_baseline_source_px": _normalise_float(row.get("nearest_baseline_source_px")),
        "trial_count": trial_count,
        "baseline_candidate_count": row.get("baseline_candidate_count"),
        "baseline_quality_count": row.get("baseline_quality_count"),
        "net_candidate_delta": _normalise_float(row.get("net_candidate_delta")),
        "net_quality_delta": _normalise_float(row.get("net_quality_delta")),
    }


def _profile_summary(stratum: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda item: float(item["peak_excess_adu"]))
    candidate = [float(item["candidate_recall"]) for item in ordered if item["candidate_recall"] is not None]
    quality = [float(item["quality_recall"]) for item in ordered if item["quality_recall"] is not None]
    noise = [float(item["local_noise_adu"]) for item in ordered if item["local_noise_adu"] is not None]
    return {
        "stratum": stratum,
        "stratum_label": ordered[0].get("stratum_label", stratum),
        "levels": [float(item["peak_excess_adu"]) for item in ordered],
        "row_count": len(ordered),
        "denominator_total": sum(int(item["denominator_count"]) for item in ordered),
        "candidate_recall_range": [min(candidate), max(candidate)] if candidate else [None, None],
        "quality_recall_range": [min(quality), max(quality)] if quality else [None, None],
        "local_noise_adu_median": median(noise) if noise else None,
        "local_noise_adu_range": [min(noise), max(noise)] if noise else [None, None],
        "trial_count_min": min(int(item["trial_count"]) for item in ordered),
        "trial_count_max": max(int(item["trial_count"]) for item in ordered),
    }


def _condition_contrasts(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    names = sorted(grouped)
    for left_name, right_name in combinations(names, 2):
        left = {float(row["peak_excess_adu"]): row for row in grouped[left_name]}
        right = {float(row["peak_excess_adu"]): row for row in grouped[right_name]}
        levels = sorted(set(left) & set(right))
        observations: list[dict[str, Any]] = []
        noise_ratios: list[float] = []
        for level in levels:
            left_row = left[level]
            right_row = right[level]
            left_candidate = left_row.get("candidate_recall")
            right_candidate = right_row.get("candidate_recall")
            left_quality = left_row.get("quality_recall")
            right_quality = right_row.get("quality_recall")
            left_noise = left_row.get("local_noise_adu")
            right_noise = right_row.get("local_noise_adu")
            noise_ratio = _safe_ratio(right_noise, left_noise)
            if noise_ratio is not None:
                noise_ratios.append(noise_ratio)
            observations.append({
                "peak_excess_adu": level,
                "candidate_recall_delta_right_minus_left": (
                    right_candidate - left_candidate
                    if right_candidate is not None and left_candidate is not None
                    else None
                ),
                "quality_recall_delta_right_minus_left": (
                    right_quality - left_quality
                    if right_quality is not None and left_quality is not None
                    else None
                ),
                "left_local_noise_adu": left_noise,
                "right_local_noise_adu": right_noise,
                "noise_ratio_right_over_left": noise_ratio,
            })
        if observations:
            result.append({
                "left_stratum": left_name,
                "right_stratum": right_name,
                "shared_levels": levels,
                "observations": observations,
                "median_noise_ratio_right_over_left": median(noise_ratios) if noise_ratios else None,
                "interpretation": "描述性条件对照；不进行显著性检验，也不据此声称某条件具有普适排序。",
            })
    return result


def _source_evidence(source: str | Path | Mapping[str, Any], evidence_id: str, role: str, status: str, title: str, limitation: str, observed: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metadata = _file_metadata(source)
    return {
        "id": evidence_id,
        "role": role,
        "status": status,
        "title": title,
        "source": metadata,
        "observed": dict(observed or {}),
        "limitation": limitation,
    }


def _sequence_observed(payload: Mapping[str, Any]) -> dict[str, Any]:
    relation = payload.get("relation")
    registration = payload.get("registration")
    motion = payload.get("motion")
    observed: dict[str, Any] = {}
    if isinstance(relation, Mapping):
        for key in ("frame_count", "duration_s", "interval_median_s", "interval_std_s"):
            if key in relation:
                observed[key] = relation[key]
    if isinstance(registration, Mapping) and "max_cumulative_shift_norm_px" in registration:
        observed["max_cumulative_shift_norm_px"] = registration["max_cumulative_shift_norm_px"]
    if isinstance(motion, Mapping):
        for key in ("track_count", "moving_track_count", "candidate_count"):
            if key in motion:
                observed[key] = motion[key]
    return observed


def _feature_matrix_observed(payload: Mapping[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    rows = payload.get("rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        observed["row_count"] = len(rows)
    for key in ("correlation_threshold",):
        if key in payload:
            observed[key] = payload[key]
    parameters = payload.get("parameters")
    if isinstance(parameters, Mapping):
        for key in ("pattern_is_not_truth", "uses_probability", "counting_policy"):
            if key in parameters:
                observed[key] = parameters[key]
    return observed


def _optional_evidence(source: str | Path | Mapping[str, Any], evidence_id: str, *, feature_matrix: bool) -> dict[str, Any]:
    payload, _ = _payload(source)
    if not isinstance(payload, Mapping):
        raise ValueError(f"optional evidence {evidence_id} must be a JSON object")
    if feature_matrix:
        return _source_evidence(
            source,
            evidence_id,
            "DIAGNOSTIC",
            "DIAGNOSTIC_ONLY",
            "特征证据矩阵（诊断路由）",
            "没有逐星真值和独立负样本；候选持续性、PSF 相似度和路由比例不能直接解释为恒星概率或误检率。",
            _feature_matrix_observed(payload),
        )
    return _source_evidence(
        source,
        evidence_id,
        "BASIC_SUPPORTING",
        "SUPPORTING_ONLY",
        "15 帧关系与运动证据（基础任务支撑）",
        "图像平面轨迹、配准和差分结果可支撑基础赛项，但没有 WCS 时不能直接解释为角速度、真实速度或轨道参数。",
        _sequence_observed(payload),
    )


def build_innovation_package(
    injection_artifact: str | Path | Mapping[str, Any],
    *,
    sequence_report: str | Path | Mapping[str, Any] | None = None,
    feature_matrix: str | Path | Mapping[str, Any] | None = None,
    min_injections: int = 8,
    min_levels: int = 3,
    min_trials: int = 3,
) -> dict[str, Any]:
    """构建创新交付包。

    ``min_injections`` 和 ``min_levels`` 是“进入主结论”的审计门槛，不是
    检测算法的阈值。``min_trials`` 是每个主剖面条件/强度单元要求的独立
    位置布局数。为防止调用参数把交付门槛悄悄调低，主结论始终不会低于
    8 个位置/布局、3 个共同强度层和 3 个独立布局；更低的参数只会用于
    生成诊断数据，不能让 ``CONDITIONAL_DEFENSIBLE`` 通过。
    """

    if not isinstance(min_injections, int) or isinstance(min_injections, bool) or min_injections <= 0:
        raise ValueError("min_injections must be a positive integer")
    if not isinstance(min_levels, int) or isinstance(min_levels, bool) or min_levels <= 0:
        raise ValueError("min_levels must be a positive integer")
    if not isinstance(min_trials, int) or isinstance(min_trials, bool) or min_trials <= 0:
        raise ValueError("min_trials must be a positive integer")

    primary_min_injections = max(min_injections, MIN_PRIMARY_INJECTIONS)
    primary_min_levels = max(min_levels, MIN_PRIMARY_LEVELS)
    primary_min_trials = max(min_trials, MIN_PRIMARY_TRIALS)

    raw_rows = load_stratified_rows(injection_artifact)
    derived_rows = [
        _derived_row({**row, "row_index": index}, primary_min_injections)
        for index, row in enumerate(raw_rows)
    ]
    all_strata = sorted({str(row["stratum"]) for row in derived_rows})
    all_levels = sorted({float(row["peak_excess_adu"]) for row in derived_rows})
    core_rows_by_stratum: dict[str, list[dict[str, Any]]] = {}
    hard_rows_by_stratum: dict[str, list[dict[str, Any]]] = {}
    incomplete_core_rows_by_stratum: dict[str, list[dict[str, Any]]] = {}
    for row in derived_rows:
        if row["primary_eligible"]:
            target = core_rows_by_stratum
        elif row["control_class"] == "hard_control":
            target = hard_rows_by_stratum
        else:
            target = incomplete_core_rows_by_stratum
        target.setdefault(str(row["stratum"]), []).append(row)

    eligible_strata = sorted(core_rows_by_stratum)
    level_sets = [set(float(row["peak_excess_adu"]) for row in rows) for rows in core_rows_by_stratum.values()]
    shared_levels = sorted(set.intersection(*level_sets)) if level_sets else []
    eligible_for_profile = {
        name: rows
        for name, rows in core_rows_by_stratum.items()
        if len({float(row["peak_excess_adu"]) for row in rows}) >= primary_min_levels
    }
    profile_level_sets = [set(float(row["peak_excess_adu"]) for row in rows) for rows in eligible_for_profile.values()]
    profile_shared_levels = sorted(set.intersection(*profile_level_sets)) if profile_level_sets else []
    profile_cells = [
        row
        for rows in eligible_for_profile.values()
        for row in rows
        if float(row["peak_excess_adu"]) in profile_shared_levels
    ]
    profile_trial_counts = [int(row["trial_count"]) for row in profile_cells]
    profile_denominators = [int(row["denominator_count"]) for row in profile_cells]
    structure_ready = len(eligible_for_profile) >= 2 and len(profile_shared_levels) >= primary_min_levels
    replication_ready = bool(profile_trial_counts) and min(profile_trial_counts) >= primary_min_trials
    if structure_ready and replication_ready:
        status = PRIMARY_STATUS
    elif eligible_for_profile:
        status = "DIAGNOSTIC_ONLY"
    else:
        status = "INCOMPLETE"

    profile_summaries = [
        _profile_summary(name, sorted(rows, key=lambda item: float(item["peak_excess_adu"])))
        for name, rows in sorted(eligible_for_profile.items())
    ]
    contrasts = _condition_contrasts(eligible_for_profile)
    all_trial_counts = [int(row["trial_count"]) for row in derived_rows]
    if profile_trial_counts and min(profile_trial_counts) >= primary_min_trials:
        replication_status = "REPLICATED"
    elif profile_trial_counts and max(profile_trial_counts) >= 2:
        replication_status = "LIMITED_REPLICATION"
    else:
        replication_status = "SINGLE_LAYOUT"

    baseline_available = any(
        row.get("baseline_candidate_count") is not None and row.get("baseline_quality_count") is not None
        for row in derived_rows
    )
    audit_checks = [
        {
            "id": "required_fields_and_bounds",
            "status": "PASS",
            "observed": {"row_count": len(raw_rows), "validated_grid_cells": len(raw_rows)},
            "meaning": "每个条件/强度单元的计数、召回率范围和分母已通过结构校验。",
        },
        {
            "id": "known_injection_denominator",
            "status": "PASS" if eligible_for_profile else "FAIL",
            "observed": {
                "eligible_strata": sorted(eligible_for_profile),
                "min_injections_per_layout": primary_min_injections,
                "profile_denominator_min": min(profile_denominators) if profile_denominators else None,
                "profile_denominator_max": max(profile_denominators) if profile_denominators else None,
            },
            "meaning": "主剖面只使用无歧义已知注入源作为召回率分母。",
        },
        {
            "id": "shared_strength_grid",
            "status": "PASS" if len(profile_shared_levels) >= primary_min_levels else "WARN",
            "observed": {"shared_levels": profile_shared_levels, "required_levels": primary_min_levels},
            "meaning": "不同背景条件在共同注入强度上做描述性对照。",
        },
        {
            "id": "paired_baseline",
            "status": "PASS" if baseline_available else "NOT_AVAILABLE",
            "observed": {"available": baseline_available},
            "meaning": "保留注入前后基线字段，但它们不能单独给出 precision/FDR。",
        },
        {
            "id": "independent_replication",
            "status": "PASS" if replication_ready else "FAIL",
            "observed": {
                "profile_trial_count_min": min(profile_trial_counts) if profile_trial_counts else None,
                "profile_trial_count_max": max(profile_trial_counts) if profile_trial_counts else None,
                "required_min_trials": primary_min_trials,
                "status": replication_status,
            },
            "meaning": "主剖面每个条件/强度单元必须有足够的独立位置布局；单布局只能作为诊断结果。",
        },
        {
            "id": "false_positive_truth",
            "status": "NOT_AVAILABLE",
            "observed": {"official_per_source_labels": False},
            "meaning": "没有官方逐星标签或独立负样本，不能从背景检测数计算误检率。",
        },
        {
            "id": "absolute_photometric_calibration",
            "status": "NOT_AVAILABLE",
            "observed": {"zero_point_and_color_terms": False},
            "meaning": "本包不宣称绝对星等、物理灵敏度或跨波段可比性。",
        },
    ]

    support_evidence: list[dict[str, Any]] = []
    if sequence_report is not None:
        support_evidence.append(_optional_evidence(sequence_report, "sequence_relation", feature_matrix=False))
    if feature_matrix is not None:
        support_evidence.append(_optional_evidence(feature_matrix, "feature_routing", feature_matrix=True))

    hard_controls = []
    for name, rows in sorted(hard_rows_by_stratum.items()):
        hard_controls.append({
            "stratum": name,
            "stratum_label": rows[0].get("stratum_label", name),
            "levels": sorted(float(row["peak_excess_adu"]) for row in rows),
            "ambiguous_injection_total": sum(int(row["ambiguous_injection_count"]) for row in rows),
            "unambiguous_injection_total": sum(int(row["unambiguous_injected_count"]) for row in rows),
            "reason": rows[0].get("control_reason"),
            "role": "HARD_CONTROL_ONLY",
        })

    profile_trial_min = min(profile_trial_counts) if profile_trial_counts else None
    profile_trial_max = max(profile_trial_counts) if profile_trial_counts else None
    profile_n_min = min(profile_denominators) if profile_denominators else None
    profile_n_max = max(profile_denominators) if profile_denominators else None
    selected_claim = (
        "在当前真实背景、当前 PSF/检测参数、每个主条件/强度单元至少 "
        f"{profile_n_min}–{profile_n_max} 个无歧义注入位置和 "
        f"{profile_trial_min}–{profile_trial_max} 个独立布局的实验口径下，"
        "局部背景条件与候选层/质量层回收存在可见差异；严格质量层在困难条件或弱注入下更容易损失源。"
        if status == PRIMARY_STATUS
        else (
            "主剖面结构已经形成，但当前独立布局数为 "
            f"{profile_trial_min if profile_trial_min is not None else '—'}，"
            f"未达到交付所需的至少 {primary_min_trials} 个布局；本包只能作为实验诊断。"
        )
    )
    package: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "package_id": PRIMARY_INNOVATION_ID,
        "status": status,
        "title": "创新交付审计：真实背景分层注入—回收的条件化可探测性剖面",
        "selected_innovation": {
            "id": PRIMARY_INNOVATION_ID,
            "role": "PRIMARY_INNOVATION",
            "claim_level": "CONDITIONAL",
            "title": "真实背景分层注入—回收的条件化可探测性剖面",
            "claim": selected_claim,
            "measurement": "在真实 FITS 背景上注入已知位置/名义峰值的 PSF 源，比较候选层和质量层对已知注入源的回收率。",
            "formula": {
                "recall": "recovered_injected_sources / unambiguous_injected_sources",
                "quality_filter_gap": "candidate_recall - quality_recall",
                "wilson_95": "二项召回率的描述性 Wilson 区间",
            },
            "not_claimed": [
                "不是整幅图的真实恒星完备率",
                "不是 precision、误检率或 FDR",
                "不是物理仪器灵敏度或绝对星等极限",
                "不是跨仪器/跨波段可迁移的普适曲线",
            ],
        },
        "provenance": {
            "injection_artifact": _file_metadata(injection_artifact),
            "row_count": len(raw_rows),
            "strata": all_strata,
            "peak_levels": all_levels,
        },
        "data_summary": {
            "raw_row_count": len(raw_rows),
            "primary_row_count": sum(len(rows) for rows in eligible_for_profile.values()),
            "hard_control_row_count": sum(len(rows) for rows in hard_rows_by_stratum.values()),
            "incomplete_core_row_count": sum(len(rows) for rows in incomplete_core_rows_by_stratum.values()),
            "primary_strata": sorted(eligible_for_profile),
            "hard_control_strata": sorted(hard_rows_by_stratum),
            "incomplete_core_strata": sorted(incomplete_core_rows_by_stratum),
            "all_levels": all_levels,
            "shared_levels_all_core_candidates": shared_levels,
            "shared_levels_primary_profile": profile_shared_levels,
            "requested_min_injections": min_injections,
            "requested_min_levels": min_levels,
            "requested_min_trials": min_trials,
            "min_injections_gate": primary_min_injections,
            "min_levels_gate": primary_min_levels,
            "min_trials_gate": primary_min_trials,
            "replication_status": replication_status,
            "trial_count_min": min(all_trial_counts) if all_trial_counts else None,
            "trial_count_max": max(all_trial_counts) if all_trial_counts else None,
            "profile_trial_count_min": profile_trial_min,
            "profile_trial_count_max": profile_trial_max,
            "profile_denominator_min": profile_n_min,
            "profile_denominator_max": profile_n_max,
        },
        "profiles": profile_summaries,
        "condition_contrasts": contrasts,
        "hard_controls": hard_controls,
        "rows": derived_rows,
        "evidence": [
            {
                "id": PRIMARY_INNOVATION_ID,
                "role": "PRIMARY_INNOVATION",
                "status": status,
                "title": "真实背景分层注入—回收",
                "claim": selected_claim,
                "evidence_strength": [
                    "真实图像背景",
                    "注入位置和名义强度已知",
                    "候选层与质量层分开统计",
                    "按背景条件分层",
                ],
                "weaknesses": [
                    f"主剖面每个条件/强度单元要求至少 {primary_min_injections} 个无歧义位置/布局和 {primary_min_trials} 个独立布局；当前 profile 为 {profile_n_min if profile_n_min is not None else '—'}–{profile_n_max if profile_n_max is not None else '—'} 个位置、{profile_trial_min if profile_trial_min is not None else '—'}–{profile_trial_max if profile_trial_max is not None else '—'} 个布局",
                    "注入源是模型源，不等同于所有真实恒星",
                    "没有独立逐星标签，因此不能计算全图误检率",
                    "没有完整相机响应/增益/平场/颜色项校准",
                ],
            },
            *support_evidence,
        ],
        "audit": {
            "checks": audit_checks,
            "claim_boundary": {
                "can_defend": [
                    "当前数据条件下的分层经验召回率",
                    "候选层到质量层的筛选损失",
                    "不同局部噪声/背景条件下的描述性差异",
                ],
                "cannot_defend": [
                    "真实恒星总数的普适完备率",
                    "误检率、precision 或 FDR",
                    "绝对星等和物理灵敏度",
                    "未经重复实验支持的显著性结论",
                ],
            },
            "recommended_replication": {
                "minimum_independent_layouts": 3,
                "recommendation": "每个条件/强度至少 3 个独立位置布局，扩大位置数并报告空间分布。",
            },
        },
        "delivery_checklist": [
            {"item": "主图：各背景条件的候选/质量召回率剖面", "status": "READY" if status == PRIMARY_STATUS else "BLOCKED"},
            {"item": "表格：每个强度的 n、召回率和 Wilson 区间", "status": "READY" if status == PRIMARY_STATUS else "BLOCKED"},
            {"item": "答辩口径：明确这是条件化经验结果", "status": "READY"},
            {"item": "困难条件对照：crowded/special_code/line 不混入主结论", "status": "READY"},
            {"item": "独立重复布局 >= 3", "status": "TODO" if replication_status != "REPLICATED" else "READY"},
            {"item": "官方逐星真值/独立负样本", "status": "TODO"},
            {"item": "绝对光度定标（零点、波段、颜色项）", "status": "TODO"},
        ],
    }
    # 这是防止后续新增字段意外把 NaN/Infinity 写进比赛材料的最后一道关。
    json.dumps(package, ensure_ascii=False, allow_nan=False)
    return package


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: object) -> str:
    number = _as_float(value)
    return "—" if number is None else f"{number * 100:.1f}%"


def _markdown(package: Mapping[str, Any]) -> str:
    selected = package["selected_innovation"]
    summary = package["data_summary"]
    lines = [
        "# 创新交付审计",
        "",
        f"> 状态：**{package['status']}**。本文件冻结当前可答辩边界，不把诊断量包装成物理真值。",
        "",
        "## 1. 最终选择的创新结果",
        "",
        f"**{selected['title']}**",
        "",
        selected["claim"],
        "",
        f"测量对象：{selected['measurement']}",
        "",
        "计算口径：`recall = recovered_injected_sources / unambiguous_injected_sources`；"
        "质量筛选损失为 `candidate_recall - quality_recall`。Wilson 区间只表达当前聚合二项计数的不确定度，"
        "不替代独立布局之间的重复性评估。",
        "",
        "## 2. 当前数据证据",
        "",
        f"- 原始实验单元：{summary['raw_row_count']}；主剖面单元：{summary['primary_row_count']}。",
        f"- 主剖面条件：{', '.join(summary['primary_strata']) or '无'}；共同强度：{', '.join(_fmt(x) for x in summary['shared_levels_primary_profile']) or '无'} ADU。",
        f"- 每个条件的独立布局计数范围：{summary['trial_count_min']}–{summary['trial_count_max']}；当前审计状态：**{summary['replication_status']}**。",
        "- 主剖面只纳入无歧义注入位置；`crowded`、`special_code`、`line` 等困难条件保留作控制，不混入主结论。",
        "",
        "### 主剖面明细",
        "",
        "| 条件 | 峰值超额 ADU | n | 候选召回 | 候选 Wilson 95% | 质量召回 | 质量 Wilson 95% | 筛选损失 | 局部噪声 ADU |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for row in package["rows"]:
        if not row["primary_eligible"]:
            continue
        candidate_ci = f"{_pct(row['candidate_recall_wilson95_low'])}–{_pct(row['candidate_recall_wilson95_high'])}"
        quality_ci = f"{_pct(row['quality_recall_wilson95_low'])}–{_pct(row['quality_recall_wilson95_high'])}"
        lines.append(
            f"| {row['stratum']} | {_fmt(row['peak_excess_adu'], 1)} | {row['denominator_count']} | "
            f"{_pct(row['candidate_recall'])} | {candidate_ci} | {_pct(row['quality_recall'])} | {quality_ci} | "
            f"{_pct(row['quality_filter_gap'])} | {_fmt(row['local_noise_adu'], 2)} |"
        )

    lines.extend([
        "",
        "### 条件对照",
        "",
        "以下差值均为右侧条件减左侧条件的描述性结果，不代表统计显著性或普适性能排名。",
        "",
        "| 左条件 | 右条件 | 共同强度 ADU | 候选召回差值 | 质量召回差值 | 噪声比（右/左） |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ])
    for contrast in package["condition_contrasts"]:
        for observation in contrast["observations"]:
            lines.append(
                f"| {contrast['left_stratum']} | {contrast['right_stratum']} | {_fmt(observation['peak_excess_adu'], 1)} | "
                f"{_pct(observation['candidate_recall_delta_right_minus_left'])} | "
                f"{_pct(observation['quality_recall_delta_right_minus_left'])} | "
                f"{_fmt(observation['noise_ratio_right_over_left'], 2)} |"
            )

    lines.extend([
        "",
        "## 3. 哪些可以答辩，哪些只能诊断",
        "",
        "### 可以答辩的主结果" if package["status"] == PRIMARY_STATUS else "### 当前只能作为诊断",
        "",
        (
            f"1. 在明确的真实背景、PSF 模型、检测参数和注入分母下，给出分层经验召回率；每个主剖面单元的聚合分母为 {summary['profile_denominator_min']}–{summary['profile_denominator_max']}。"
            if package["status"] == PRIMARY_STATUS
            else "1. 主剖面结构已经形成，但独立布局门槛尚未满足，当前结果不能作为最终答辩主曲线。"
        ),
        "2. 同一注入强度下比较候选层与质量层，说明“宽筛保召回、严筛保可靠性”的数据依据。",
        (
            f"3. 用 Wilson 区间展示聚合 n={summary['profile_denominator_min']}–{summary['profile_denominator_max']} 的不确定性，并同时报告 {summary['profile_trial_count_min']}–{summary['profile_trial_count_max']} 个独立布局。"
            if package["status"] == PRIMARY_STATUS
            else f"3. 当前 profile 的独立布局数为 {summary['profile_trial_count_min'] if summary['profile_trial_count_min'] is not None else '—'}–{summary['profile_trial_count_max'] if summary['profile_trial_count_max'] is not None else '—'}，要求至少 {summary['min_trials_gate']}。"
        ),
        "",
        "### 支撑证据，不单独算创新得分",
        "",
        "- 15 帧运动/配准/差分：用于基础赛项和实验背景说明；没有 WCS 时只能报告像素平面量。",
        "- 注册拼图：用于直观展示固定星场和配准覆盖，不是新的恒星计数或光度定标结果。",
        "",
        "### 诊断工具，不应直接写成创新结论",
        "",
        "- 特征证据矩阵：用于值域、形态、PSF 和路由审计；没有逐星真值，不能称为恒星概率或误检率。",
        "- 困难条件 `crowded`、`special_code`、`line`：可展示算法边界，但歧义/污染使其不适合进入主召回曲线。",
        "- `innovation.py` 的综合 JSON/PNG：是证据导出器，不是自动的比赛结论生成器。",
        "",
        "## 4. 明确不能声称的内容",
        "",
    ])
    for item in selected["not_claimed"]:
        lines.append(f"- {item}。")
    lines.extend([
        "",
        "## 5. 当前未完成项",
        "",
        f"- 每个主条件/强度单元至少 {summary['min_trials_gate']} 个独立位置布局，并扩大空间采样；当前 profile `trial_count` 为 {summary['profile_trial_count_min'] if summary['profile_trial_count_min'] is not None else '—'}–{summary['profile_trial_count_max'] if summary['profile_trial_count_max'] is not None else '—'}。",
        f"- 每个独立布局至少 {summary['min_injections_gate']} 个无歧义位置；当前 profile 聚合分母为 {summary['profile_denominator_min'] if summary['profile_denominator_min'] is not None else '—'}–{summary['profile_denominator_max'] if summary['profile_denominator_max'] is not None else '—'}。",
        "- 获取官方逐星真值或构造独立负样本，才能审计 precision/FDR。",
        "- 以真实相机响应、增益、暗场/平场、波段和标准星零点补足绝对光度链路。",
        "- 用留出星表匹配和空间变 PSF 对照验证跨区域稳定性；当前注入源仍是实验模型源。",
        "- 证明领域差异前要做文献/同类作品对照；本包不自动宣称“独创性”。",
        "",
        "## 6. 推荐答辩表述",
        "",
        f"> 我们没有把全图检测数直接称为恒星数，而是在真实 FITS 背景中注入已知源，按空白、高背景、边缘等条件分层，分别统计候选层和质量层的回收率。结果表明，检测阈值与质量筛选对弱源的影响依赖局部背景条件；因此我们把它定义为当前数据和当前参数下的条件化经验可探测性剖面。每个独立布局要求至少 {summary['min_injections_gate']} 个无歧义位置，并要求至少 {summary['min_trials_gate']} 个独立布局；尚无官方逐星真值时，我们不把它外推为普适完备率、误检率或物理灵敏度。",
        "",
        "## 7. 生成与复核",
        "",
        "```powershell",
        "python -m rst19.innovation_package_cli `",
        "  tmp\\stratified-real-background-code-pattern-current-n8\\stratified_real_background_injection.json `",
        "  --out-dir $env:TEMP\\rst19-innovation-package `",
        "  --sequence-report tmp\\innovation-current-psf-support\\innovation_report.json `",
        "  --feature-matrix tmp\\feature-evidence-matrix-code-pattern-current-v3-routing\\feature_evidence_matrix.json `",
        "  --require-defensible",
        "```",
        "",
        "输出 `innovation_package.json`、`innovation_delivery.csv` 和本审计 Markdown。输出目录应放在临时目录或发布附件目录，不把 FITS、tmp、cache 加入 Git。",
        "",
        "## 8. 交付清单",
        "",
    ])
    for item in package["delivery_checklist"]:
        lines.append(f"- [{ 'x' if item['status'] == 'READY' else ' ' }] {item['item']}（{item['status']}）")
    lines.append("")
    return "\n".join(lines)


_CSV_FIELDS = (
    "stratum",
    "stratum_label",
    "control_class",
    "primary_eligible",
    "peak_excess_adu",
    "injected_count",
    "unambiguous_injected_count",
    "ambiguous_injection_count",
    "denominator_kind",
    "denominator_count",
    "candidate_recovered_count",
    "quality_recovered_count",
    "candidate_recall",
    "candidate_recall_wilson95_low",
    "candidate_recall_wilson95_high",
    "quality_recall",
    "quality_recall_wilson95_low",
    "quality_recall_wilson95_high",
    "quality_filter_gap",
    "rate_semantics",
    "ambiguous_count_is_nonexclusive",
    "minimum_unambiguous_positions_required",
    "local_background_adu",
    "local_noise_adu",
    "nominal_level_over_local_noise",
    "trial_count",
    "source_path",
)


def write_innovation_package(package: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    """写出 JSON、CSV 和答辩友好的 Markdown，不写入仓库默认路径。"""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "innovation_package.json"
    csv_path = target / "innovation_delivery.csv"
    markdown_path = target / "innovation_delivery.md"

    serialised = json.dumps(package, ensure_ascii=False, indent=2, allow_nan=False)
    json_path.write_text(serialised + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in package.get("rows", []):
            writer.writerow({field: row.get(field) for field in _CSV_FIELDS})
    markdown_path.write_text(_markdown(package), encoding="utf-8")
    return {"json": str(json_path.resolve()), "csv": str(csv_path.resolve()), "markdown": str(markdown_path.resolve())}


__all__ = [
    "MIN_PRIMARY_INJECTIONS",
    "MIN_PRIMARY_LEVELS",
    "MIN_PRIMARY_TRIALS",
    "PRIMARY_INNOVATION_ID",
    "PRIMARY_STATUS",
    "SCHEMA_VERSION",
    "build_innovation_package",
    "load_stratified_rows",
    "wilson_interval",
    "write_innovation_package",
]
