"""多次重复的特征类别注入控制。

``run_feature_audit`` 负责一个随机种子的机制场景；本模块只重复调用它并
按场景聚合，严格把正样本的已知真值回收和负样本的结构泄漏分开。输出的
区间是观测回收计数的描述性区间，不是当前真实 FITS 的 precision、完备率
或物理伪影概率。
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Iterable

import numpy as np

from .experiments import FeatureAuditRow, run_feature_audit


@dataclass(frozen=True, slots=True)
class FeatureAuditAggregateRow:
    """一个场景在多个随机背景重复试验中的聚合结果。"""

    feature_class: str
    scenario: str
    control_type: str
    trial_count: int
    truth_count: int
    candidate_true_hits: int
    quality_true_hits: int
    candidate_recall: float | None
    quality_recall: float | None
    candidate_recall_wilson95_low: float | None
    candidate_recall_wilson95_high: float | None
    quality_recall_wilson95_low: float | None
    quality_recall_wilson95_high: float | None
    mean_nearby_candidate_count: float
    max_nearby_candidate_count: int
    mean_nearby_quality_count: float
    max_nearby_quality_count: int
    mean_candidate_count: float
    mean_quality_count: float
    nearby_flags: str
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_class": self.feature_class,
            "scenario": self.scenario,
            "control_type": self.control_type,
            "trial_count": self.trial_count,
            "truth_count": self.truth_count,
            "candidate_true_hits": self.candidate_true_hits,
            "quality_true_hits": self.quality_true_hits,
            "candidate_recall": self.candidate_recall,
            "quality_recall": self.quality_recall,
            "candidate_recall_wilson95_low": self.candidate_recall_wilson95_low,
            "candidate_recall_wilson95_high": self.candidate_recall_wilson95_high,
            "quality_recall_wilson95_low": self.quality_recall_wilson95_low,
            "quality_recall_wilson95_high": self.quality_recall_wilson95_high,
            "mean_nearby_candidate_count": self.mean_nearby_candidate_count,
            "max_nearby_candidate_count": self.max_nearby_candidate_count,
            "mean_nearby_quality_count": self.mean_nearby_quality_count,
            "max_nearby_quality_count": self.max_nearby_quality_count,
            "mean_candidate_count": self.mean_candidate_count,
            "mean_quality_count": self.mean_quality_count,
            "nearby_flags": self.nearby_flags,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class FeatureAuditReplicateResult:
    """多次特征机制控制的结果和固定参数。"""

    trial_count: int
    base_seed: int
    rows: tuple[FeatureAuditAggregateRow, ...]
    parameters: dict[str, object]
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "trial_count": self.trial_count,
            "base_seed": self.base_seed,
            "row_count": len(self.rows),
            "positive_scenarios": sum(row.control_type == "positive" for row in self.rows),
            "negative_scenarios": sum(row.control_type == "negative" for row in self.rows),
            "parameters": self.parameters,
            "note": self.note,
            "rows": [row.as_dict() for row in self.rows],
        }


def _wilson95_interval(successes: int, trials: int) -> tuple[float, float]:
    """为已知真值回收计数计算描述性 Wilson 95% 区间。"""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError(f"invalid recall counts: successes={successes}, trials={trials}")
    z = NormalDist().inv_cdf(0.975)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == trials else min(1.0, center + half_width)
    return float(lower), float(upper)


def _merge_flags(rows: Iterable[FeatureAuditRow]) -> str:
    counts: Counter[str] = Counter()
    for row in rows:
        trial_flags: set[str] = set()
        for token in row.nearby_flags.split("|"):
            token = token.strip()
            if token:
                # 每次 trial 的行可能已经带有局部数量（例如
                # ``LINE_ARTIFACT:3``）。聚合时保留机制名并按 trial
                # 计数，避免把不同 trial 的数量误当成不同旗标。
                trial_flags.add(token.split(":", 1)[0])
        counts.update(trial_flags)
    return "|".join(f"{name}:{counts[name]}" for name in sorted(counts))


def _aggregate_rows(
    grouped: dict[str, list[FeatureAuditRow]],
    trial_count: int,
) -> tuple[FeatureAuditAggregateRow, ...]:
    output: list[FeatureAuditAggregateRow] = []
    for scenario in sorted(grouped):
        rows = grouped[scenario]
        first = rows[0]
        if any(
            row.feature_class != first.feature_class or row.control_type != first.control_type
            for row in rows
        ):
            raise ValueError(f"scenario {scenario!r} changed feature class or control type across trials")
        truth_count = sum(row.truth_count for row in rows)
        candidate_hits = sum(row.candidate_true_hits for row in rows)
        quality_hits = sum(row.quality_true_hits for row in rows)
        candidate_recall = candidate_hits / truth_count if truth_count else None
        quality_recall = quality_hits / truth_count if truth_count else None
        candidate_interval = _wilson95_interval(candidate_hits, truth_count) if truth_count else (None, None)
        quality_interval = _wilson95_interval(quality_hits, truth_count) if truth_count else (None, None)
        nearby_candidate_counts = [row.nearby_candidate_count for row in rows]
        nearby_quality_counts = [row.nearby_quality_count for row in rows]
        output.append(
            FeatureAuditAggregateRow(
                feature_class=first.feature_class,
                scenario=first.scenario,
                control_type=first.control_type,
                trial_count=trial_count,
                truth_count=truth_count,
                candidate_true_hits=candidate_hits,
                quality_true_hits=quality_hits,
                candidate_recall=candidate_recall,
                quality_recall=quality_recall,
                candidate_recall_wilson95_low=candidate_interval[0],
                candidate_recall_wilson95_high=candidate_interval[1],
                quality_recall_wilson95_low=quality_interval[0],
                quality_recall_wilson95_high=quality_interval[1],
                mean_nearby_candidate_count=float(np.mean(nearby_candidate_counts)),
                max_nearby_candidate_count=max(nearby_candidate_counts),
                mean_nearby_quality_count=float(np.mean(nearby_quality_counts)),
                max_nearby_quality_count=max(nearby_quality_counts),
                mean_candidate_count=float(np.mean([row.candidate_count for row in rows])),
                mean_quality_count=float(np.mean([row.quality_count for row in rows])),
                nearby_flags=_merge_flags(rows),
                note=first.note,
            )
        )
    return tuple(output)


def run_feature_audit_replicates(
    *,
    trials: int = 16,
    seed: int = 19019,
    progress: Callable[[int, int], None] | None = None,
    **feature_audit_kwargs: object,
) -> FeatureAuditReplicateResult:
    """用多个随机背景重复特征审计并聚合正/负样本。

    ``feature_audit_kwargs`` 透传给 :func:`run_feature_audit`，因此研究命令
    可以固定与单次审计相同的检测参数。每个重复只改变随机背景种子。
    """

    if trials < 1:
        raise ValueError("trials must be positive")
    grouped: dict[str, list[FeatureAuditRow]] = defaultdict(list)
    expected_scenarios: tuple[str, ...] | None = None
    seed_stride = 1_000_003
    for trial_index in range(trials):
        rows = run_feature_audit(seed=int(seed) + trial_index * seed_stride, **feature_audit_kwargs)
        scenario_names = tuple(row.scenario for row in rows)
        if len(set(scenario_names)) != len(scenario_names):
            raise ValueError("run_feature_audit returned duplicate scenarios")
        if expected_scenarios is None:
            expected_scenarios = scenario_names
        elif scenario_names != expected_scenarios:
            raise ValueError("run_feature_audit scenario order changed across trials")
        for row in rows:
            grouped[row.scenario].append(row)
        if progress is not None:
            progress(trial_index + 1, trials)

    parameters = {
        "trial_count": int(trials),
        "base_seed": int(seed),
        "seed_stride": seed_stride,
        "feature_audit_kwargs": dict(feature_audit_kwargs),
        "positive_recall_denominator": "sum(truth_count) across repeated known-source scenes",
        "negative_leakage_denominator": "trial count; no injected truth source exists",
        "interval_definition": "descriptive Wilson 95% interval for positive known-source hits only",
        "physical_truth_warning": True,
    }
    return FeatureAuditReplicateResult(
        trial_count=int(trials),
        base_seed=int(seed),
        rows=_aggregate_rows(grouped, int(trials)),
        parameters=parameters,
        note=(
            "正样本的候选/质量回收率来自已知注入坐标；负样本只统计结构邻域泄漏。"
            "二者不合并为 accuracy、precision、真实伪影率或物理恒星概率。"
        ),
    )


def write_feature_audit_replicate_artifacts(
    result: FeatureAuditReplicateResult,
    out_dir: str | Path,
) -> Path:
    """写入多次特征审计的 CSV/JSON。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [row.as_dict() for row in result.rows]
    fields = tuple(rows[0].keys()) if rows else ()
    with (output / "feature_audit_replicates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "feature_audit_replicates.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output
