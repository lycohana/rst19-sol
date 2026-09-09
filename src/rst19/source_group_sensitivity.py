"""父源组半径敏感性审计。

该模块只在已经导出的 ``source_catalog.csv`` 上重复运行父源组后处理，
比较不同近邻半径是否改变目标 pair 的分流和全图多成员组数量。它不读取
FITS、不重跑检测、不修改默认质量层或缓存；半径扫描结果也不是光学分辨率
或物理双星概率。
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .source_group_audit import CandidateGroupAuditResult, run_candidate_group_audit


@dataclass(frozen=True, slots=True)
class SourceGroupSensitivityRow:
    """一个组半径配置的全量汇总。"""

    group_radius_px: float
    source_count: int
    quality_count: int
    group_count: int
    multi_member_group_count: int
    unresolved_group_count: int
    independent_group_candidate_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceGroupSensitivityTargetRow:
    """一个组半径配置下的目标定位。"""

    group_radius_px: float
    target_detection_ids: str
    group_id: int
    group_detection_ids: str
    classification: str
    member_count: int
    centroid_span_px: float
    peak_span_px: float
    feature_classes: str
    flags: str
    independent_psf_evidence: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceGroupSensitivityResult:
    """父源组半径扫描结果。"""

    catalog_path: str
    target_detection_ids: str
    rows: tuple[SourceGroupSensitivityRow, ...]
    targets: tuple[SourceGroupSensitivityTargetRow, ...]
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "target_detection_ids": self.target_detection_ids,
            "rows": [row.as_dict() for row in self.rows],
            "targets": [row.as_dict() for row in self.targets],
            "parameters": dict(self.parameters),
            "conclusion": self.conclusion,
            "interpretation_guardrails": [
                "组半径是 detector-level 敏感性参数，不是光学分辨率或物理星间距阈值。",
                "半径改变会改变比较总体；不同半径的组数不能拼接或相加。",
                "目标在某个半径下 isolated 只表示没有进入该半径近邻组，不表示已确认独立恒星。",
            ],
        }


def _validate_radii(group_radii_px: Iterable[float]) -> tuple[float, ...]:
    radii = tuple(float(value) for value in group_radii_px)
    if not radii:
        raise ValueError("group_radii_px must contain at least one radius")
    if any(not math.isfinite(value) or value <= 0 for value in radii):
        raise ValueError("group radii must be finite and positive")
    if len(set(radii)) != len(radii):
        raise ValueError("group radii must be unique")
    return radii


def _run_one_radius(
    catalog_csv: str | Path,
    *,
    group_radius_px: float,
    target_ids: tuple[int, ...],
    psf_fwhm: float,
    delta_bic_min: float,
    component_snr_min: float,
) -> CandidateGroupAuditResult:
    return run_candidate_group_audit(
        catalog_csv,
        target_ids=target_ids,
        psf_fwhm=psf_fwhm,
        group_radius_px=group_radius_px,
        delta_bic_min=delta_bic_min,
        component_snr_min=component_snr_min,
    )


def run_source_group_sensitivity(
    catalog_csv: str | Path,
    *,
    group_radii_px: Sequence[float] = (2.5, 3.0, 3.5, 4.0),
    target_ids: Iterable[int] = (),
    psf_fwhm: float = 2.0,
    delta_bic_min: float = 10.0,
    component_snr_min: float = 5.0,
    workers: int = 1,
) -> SourceGroupSensitivityResult:
    """比较多个父源组半径；所有配置都保持同一源表和证据线。"""

    radii = _validate_radii(group_radii_px)
    target_id_tuple = tuple(dict.fromkeys(int(value) for value in target_ids))
    if workers <= 0:
        raise ValueError("workers must be positive")
    jobs = [
        dict(
            catalog_csv=catalog_csv,
            group_radius_px=radius,
            target_ids=target_id_tuple,
            psf_fwhm=psf_fwhm,
            delta_bic_min=delta_bic_min,
            component_snr_min=component_snr_min,
        )
        for radius in radii
    ]
    if workers == 1 or len(jobs) == 1:
        audit_results = tuple(_run_one_radius(**job) for job in jobs)
    else:
        max_workers = min(int(workers), len(jobs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_one_radius, **job) for job in jobs]
            audit_results = tuple(future.result() for future in futures)

    rows = tuple(
        SourceGroupSensitivityRow(
            group_radius_px=radius,
            source_count=result.source_count,
            quality_count=result.quality_count,
            group_count=result.group_count,
            multi_member_group_count=result.multi_member_group_count,
            unresolved_group_count=result.unresolved_group_count,
            independent_group_candidate_count=result.independent_group_candidate_count,
        )
        for radius, result in zip(radii, audit_results, strict=True)
    )
    target_rows: list[SourceGroupSensitivityTargetRow] = []
    for radius, result in zip(radii, audit_results, strict=True):
        target_rows.extend(
            SourceGroupSensitivityTargetRow(
                group_radius_px=radius,
                target_detection_ids=target.target_detection_ids,
                group_id=target.group_id,
                group_detection_ids=target.group_detection_ids,
                classification=target.classification,
                member_count=target.member_count,
                centroid_span_px=target.centroid_span_px,
                peak_span_px=target.peak_span_px,
                feature_classes=target.feature_classes,
                flags=target.flags,
                independent_psf_evidence=target.independent_psf_evidence,
                reason=target.reason,
            )
            for target in result.targets
        )
    descriptions = ", ".join(
        f"{row.group_radius_px:g}px→{row.multi_member_group_count}个多成员组"
        for row in rows
    )
    target_description = "未指定目标" if not target_id_tuple else "|".join(map(str, target_id_tuple))
    conclusion = (
        f"在同一源表和证据线下扫描组半径 {descriptions}；"
        f"目标 {target_description} 的变化只反映 detector-level 近邻分流，"
        "不能解释为物理分辨率、星数或双星概率。"
    )
    return SourceGroupSensitivityResult(
        catalog_path=str(catalog_csv),
        target_detection_ids="|".join(map(str, target_id_tuple)),
        rows=rows,
        targets=tuple(target_rows),
        parameters={
            "group_radii_px": list(radii),
            "psf_fwhm": float(psf_fwhm),
            "delta_bic_min": float(delta_bic_min),
            "component_snr_min": float(component_snr_min),
            "workers": int(workers),
            "comparison_boundary": "same source catalog, same quality and PSF evidence thresholds",
        },
        conclusion=conclusion,
    )


def _write_csv(path: Path, rows: Sequence[object]) -> None:
    import csv
    from dataclasses import fields

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    field_names = [field.name for field in fields(type(rows[0]))]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_source_group_sensitivity_artifacts(
    result: SourceGroupSensitivityResult,
    out_dir: str | Path,
) -> Path:
    """写出半径汇总、目标表和 JSON。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "source_group_radius_sensitivity.csv", result.rows)
    _write_csv(output / "source_group_radius_targets.csv", result.targets)
    (output / "source_group_radius_sensitivity.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


__all__ = [
    "SourceGroupSensitivityResult",
    "SourceGroupSensitivityRow",
    "SourceGroupSensitivityTargetRow",
    "run_source_group_sensitivity",
    "write_source_group_sensitivity_artifacts",
]
