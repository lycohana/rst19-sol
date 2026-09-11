"""合并分层 Gaia 子表，并把完整性证据传递到结果层。

星等工作流通常需要两种公共目录：一份包含足够亮参考星的定标层，
另一份覆盖检测极限的深星层。它们可以分别查询，再通过本模块合并成一
个供 ``rst19-photometric-auto`` 使用的 CSV。合并本身不创造任何星，且
只有当所有输入都有完整性审计、天空覆盖一致、行级关键字段没有冲突时，
输出才会标记为 ``complete=True``。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .catalog import load_catalog_csv
from .gaia_remote import CATALOG_COMPAT_COLUMNS


class CatalogMergeError(ValueError):
    """输入星表不满足可审计合并契约。"""


_CRITICAL_FIELDS = frozenset(
    {
        "ra",
        "dec",
        "ra_deg",
        "dec_deg",
        "magnitude",
        "magnitude_source",
        "phot_g_mean_mag",
        "parallax",
        "parallax_mas",
        "distance_pc",
        "distance_gspphot",
        "extinction_mag",
        "ag_gspphot",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _numeric_equal(left: str, right: str) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return left == right
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return left == right
    return math.isclose(left_value, right_value, rel_tol=1.0e-9, abs_tol=1.0e-12)


def _values_equal(field: str, left: str, right: str) -> bool:
    if left == right:
        return True
    if field.strip().lower() in _CRITICAL_FIELDS:
        return _numeric_equal(left, right)
    return False


def _read_csv(path: Path) -> tuple[tuple[dict[str, str], ...], tuple[str, ...]]:
    if not path.is_file():
        raise CatalogMergeError(f"星表文件不存在：{path}")
    if path.suffix.lower() != ".csv":
        raise CatalogMergeError(f"星表必须是 CSV：{path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            raw_fieldnames = tuple(reader.fieldnames or ())
            fieldnames = tuple(str(name).strip() for name in raw_fieldnames if str(name).strip())
            source_field = next(
                (name for name in fieldnames if name.casefold() == "source_id"),
                None,
            )
            if source_field is None:
                raise CatalogMergeError(f"星表缺少 source_id 列：{path}")
            rows: list[dict[str, str]] = []
            for row_number, raw_row in enumerate(reader, start=2):
                row = {str(key).strip(): _normalise_text(value) for key, value in raw_row.items() if key is not None}
                source_id = row.get(source_field, "").strip()
                if not source_id:
                    raise CatalogMergeError(f"{path} 第 {row_number} 行缺少 source_id")
                row["source_id"] = source_id
                rows.append(row)
    except UnicodeDecodeError as exc:
        raise CatalogMergeError(f"星表不是有效 UTF-8 CSV：{path}") from exc
    return tuple(rows), tuple(dict.fromkeys(("source_id", *fieldnames)))


def _audit_path(path: Path) -> Path | None:
    candidates = (
        path.with_suffix(path.suffix + ".meta.json"),
        path.with_suffix(path.suffix + ".tiled.meta.json"),
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _read_audit(path: Path) -> tuple[dict[str, object] | None, Path | None, str | None]:
    audit_path = _audit_path(path)
    if audit_path is None:
        return None, None, "缺少 .meta.json 或 .tiled.meta.json"
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, audit_path, f"审计文件不可读取：{type(exc).__name__}"
    if not isinstance(payload, Mapping):
        return None, audit_path, "审计文件不是 JSON 对象"
    return dict(payload), audit_path, None


def _audit_value(payload: Mapping[str, object], *names: str) -> float | None:
    for name in names:
        value = payload.get(name)
        if value is None or str(value).strip() == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _footprint(payload: Mapping[str, object]) -> tuple[float, float, float] | None:
    ra = _audit_value(payload, "center_ra_deg", "query_center_ra_deg")
    dec = _audit_value(payload, "center_dec_deg", "query_center_dec_deg")
    radius = _audit_value(payload, "search_radius_deg", "query_radius_deg")
    if ra is None or dec is None or radius is None:
        return None
    return ra % 360.0, dec, radius


def _audit_binding_error(
    rows: Sequence[Mapping[str, str]],
    payload: Mapping[str, object] | None,
    file_sha256: str,
) -> str | None:
    """验证 complete sidecar 是否仍然绑定当前 CSV。"""

    if payload is None or payload.get("complete") is not True:
        return None
    declared_sha = payload.get("csv_sha256", payload.get("sha256"))
    if not isinstance(declared_sha, str) or not declared_sha.strip():
        return "complete=true 审计缺少 csv_sha256，无法绑定当前 CSV"
    if declared_sha.strip().lower() != file_sha256.lower():
        return "complete=true 审计的 csv_sha256 与当前 CSV 不一致"
    declared_count = payload.get("csv_row_count", payload.get("row_count"))
    try:
        count = int(declared_count) if declared_count is not None else None
    except (TypeError, ValueError):
        count = None
    if count is None:
        return "complete=true 审计缺少 csv_row_count，无法核对当前 CSV 行数"
    if count != len(rows):
        return f"complete=true 审计的 csv_row_count={count} 与当前 CSV 行数={len(rows)} 不一致"
    return None


def _same_footprint(left: tuple[float, float, float], right: tuple[float, float, float]) -> bool:
    return all(math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-6) for a, b in zip(left, right, strict=True))


def _row_richness(row: Mapping[str, str]) -> int:
    return sum(bool(_normalise_text(value)) for value in row.values())


def _merge_rows(
    rows_by_source_id: dict[str, dict[str, str]],
    incoming: Mapping[str, str],
    *,
    conflict_ids: set[str],
) -> bool:
    source_id = _normalise_text(incoming.get("source_id"))
    existing = rows_by_source_id.get(source_id)
    if existing is None:
        rows_by_source_id[source_id] = dict(incoming)
        return False

    # Start with the richer representation, then fill missing values from the
    # other query.  This is important when the calibration query and the deep
    # query requested different column sets.
    if _row_richness(incoming) > _row_richness(existing):
        base = dict(incoming)
        supplement = existing
    else:
        base = dict(existing)
        supplement = incoming
    conflict = False
    for field, value in supplement.items():
        value_text = _normalise_text(value)
        current_text = _normalise_text(base.get(field))
        if not current_text and value_text:
            base[field] = value_text
        elif current_text and value_text and not _values_equal(field, current_text, value_text):
            conflict = True
    rows_by_source_id[source_id] = base
    if conflict:
        conflict_ids.add(source_id)
    return True


@dataclass(frozen=True, slots=True)
class CatalogMergeResult:
    """分层星表合并结果及其完整性摘要。"""

    output_path: Path
    audit_path: Path
    input_paths: tuple[Path, ...]
    row_count: int
    duplicate_count: int
    conflict_count: int
    complete: bool
    footprint_consistent: bool
    input_audits_complete: bool
    input_audit_bindings_valid: bool
    conflicting_source_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "catalog": "Gaia DR3 merged layers",
            "output_path": str(self.output_path),
            "audit_path": str(self.audit_path),
            "input_paths": [str(path) for path in self.input_paths],
            "row_count": self.row_count,
            "duplicate_count": self.duplicate_count,
            "conflict_count": self.conflict_count,
            "conflicting_source_ids": list(self.conflicting_source_ids),
            "complete": self.complete,
            "footprint_consistent": self.footprint_consistent,
            "input_audits_complete": self.input_audits_complete,
            "input_audit_bindings_valid": self.input_audit_bindings_valid,
            "interpretation": (
                "A merged catalog is eligible for formal downstream provenance only when complete is true. "
                "Duplicate source_id rows are collapsed; field conflicts and missing input audits keep it incomplete."
            ),
        }


def merge_catalog_csvs(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
) -> CatalogMergeResult:
    """合并至少两份 CSV，并继承输入审计的完整性边界。"""

    paths = tuple(Path(path).expanduser().resolve() for path in input_paths)
    if len(paths) < 2:
        raise CatalogMergeError("至少需要两份星表才能执行分层合并")
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".csv":
        raise CatalogMergeError("output_path 必须使用 .csv 后缀")
    if output in paths:
        raise CatalogMergeError("输出文件不能覆盖输入星表")

    rows_by_source_id: dict[str, dict[str, str]] = {}
    all_fields: list[str] = list(CATALOG_COMPAT_COLUMNS)
    conflict_ids: set[str] = set()
    duplicate_count = 0
    input_records: list[dict[str, object]] = []
    footprints: list[tuple[float, float, float]] = []
    input_audits_complete = True
    input_audit_bindings_valid = True

    for path in paths:
        rows, fieldnames = _read_csv(path)
        # Validate every input using the same parser used by the photometry
        # workflow before any merged artifact is written.
        try:
            load_catalog_csv(path)
        except (OSError, ValueError) as exc:
            raise CatalogMergeError(f"星表无法按 CatalogSource 读取：{path}：{exc}") from exc
        for field in (*fieldnames,):
            if field and field not in all_fields:
                all_fields.append(field)

        audit, audit_path, audit_error = _read_audit(path)
        file_sha256 = _sha256(path)
        binding_error = _audit_binding_error(rows, audit, file_sha256)
        if binding_error is not None:
            audit_error = binding_error if audit_error is None else f"{audit_error}; {binding_error}"
            input_audit_bindings_valid = False
        audit_complete = bool(audit is not None and audit.get("complete") is True and not audit_error)
        input_audits_complete = input_audits_complete and audit_complete
        footprint = _footprint(audit) if audit is not None else None
        if footprint is not None:
            footprints.append(footprint)
        input_records.append(
            {
                "path": str(path),
                "sha256": file_sha256,
                "row_count": len(rows),
                "audit_path": str(audit_path) if audit_path is not None else None,
                "audit_complete": audit_complete,
                "audit_error": audit_error,
                "audit_binding_valid": binding_error is None,
                "footprint": (
                    {
                        "center_ra_deg": footprint[0],
                        "center_dec_deg": footprint[1],
                        "radius_deg": footprint[2],
                    }
                    if footprint is not None
                    else None
                ),
                "min_g_mag": audit.get("min_g_mag") if audit is not None else None,
                "max_g_mag": audit.get("max_g_mag") if audit is not None else None,
            }
        )
        for row in rows:
            duplicate_count += int(
                _merge_rows(rows_by_source_id, row, conflict_ids=conflict_ids)
            )

    footprint_consistent = bool(footprints) and len(footprints) == len(paths) and all(
        _same_footprint(footprints[0], footprint) for footprint in footprints[1:]
    )
    complete = (
        input_audits_complete
        and input_audit_bindings_valid
        and footprint_consistent
        and not conflict_ids
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(dict.fromkeys(all_fields)),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for source_id in sorted(rows_by_source_id):
            row = rows_by_source_id[source_id]
            writer.writerow({field: _normalise_text(row.get(field, "")) for field in writer.fieldnames})

    # Validate the artifact after writing as well.  This catches malformed
    # header combinations introduced by a custom CSV before provenance is
    # advertised to the downstream workflow.
    try:
        merged_rows = load_catalog_csv(output)
    except (OSError, ValueError) as exc:
        raise CatalogMergeError(f"合并结果无法按 CatalogSource 读取：{output}：{exc}") from exc
    if len(merged_rows) != len(rows_by_source_id):
        raise CatalogMergeError("合并结果行数与唯一 source_id 数量不一致")

    result = CatalogMergeResult(
        output_path=output,
        audit_path=output.with_suffix(output.suffix + ".meta.json"),
        input_paths=paths,
        row_count=len(rows_by_source_id),
        duplicate_count=duplicate_count,
        conflict_count=len(conflict_ids),
        complete=complete,
        footprint_consistent=footprint_consistent,
        input_audits_complete=input_audits_complete,
        input_audit_bindings_valid=input_audit_bindings_valid,
        conflicting_source_ids=tuple(sorted(conflict_ids)),
    )
    payload = result.as_dict()
    payload["csv_row_count"] = result.row_count
    payload["csv_sha256"] = _sha256(output)
    payload["csv_byte_count"] = output.stat().st_size
    payload["inputs"] = input_records
    if not complete:
        reasons: list[str] = []
        if not input_audits_complete:
            reasons.append("至少一份输入缺少 complete=true 审计")
        if not input_audit_bindings_valid:
            reasons.append("至少一份 complete=true 审计未与当前 CSV 的 SHA-256/行数绑定")
        if not footprint_consistent:
            reasons.append("输入天空覆盖不一致或缺少覆盖字段")
        if conflict_ids:
            reasons.append(f"{len(conflict_ids)} 个 source_id 存在关键字段冲突")
        payload["incomplete_reasons"] = reasons
    result.audit_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = ["CatalogMergeError", "CatalogMergeResult", "merge_catalog_csvs"]
