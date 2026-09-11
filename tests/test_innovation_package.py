from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rst19.innovation_package import (
    PRIMARY_STATUS,
    build_innovation_package,
    load_stratified_rows,
    wilson_interval,
    write_innovation_package,
)
from rst19.innovation_package_cli import main


def _row(
    stratum: str,
    level: float,
    candidate: int,
    quality: int,
    *,
    ambiguous: int = 0,
    unambiguous: int | None = None,
    noise: float = 5.0,
    trial_count: int = 1,
) -> dict[str, object]:
    injected_total = 8 * trial_count
    candidate_total = candidate * trial_count
    quality_total = quality * trial_count
    unambiguous_total = injected_total if unambiguous is None else unambiguous
    return {
        "source_path": "fixture.fits",
        "stratum": stratum,
        "stratum_label": stratum,
        "proposal_mode": "hybrid",
        "psf_model": "empirical",
        "peak_excess_adu": level,
        "injected_count": injected_total,
        "candidate_recovered_count": candidate_total,
        "quality_recovered_count": quality_total,
        "candidate_recall": candidate_total / injected_total,
        "quality_recall": quality_total / injected_total,
        "ambiguous_injection_count": ambiguous,
        "unambiguous_injected_count": unambiguous_total,
        "local_noise_adu": noise,
        "trial_count": trial_count,
        "baseline_candidate_count": 100,
        "baseline_quality_count": 50,
    }


def _write_artifact(tmp_path: Path, rows: list[dict[str, object]], name: str = "injection.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"row_count": len(rows), "rows": rows}, ensure_ascii=False), encoding="utf-8")
    return path


def _core_rows(*, trial_count: int = 1) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    values = {
        "blank": (0, 8, 0, 5.0),
        "high_background": (2, 7, 1, 10.0),
        "edge": (0, 7, 0, 7.5),
    }
    for stratum, (low, middle, high, noise) in values.items():
        for level, candidate, quality in ((24.0, low, 0), (56.0, middle, 0), (128.0, 8, high)):
            rows.append(_row(stratum, level, candidate, quality, noise=noise, trial_count=trial_count))
    return rows


def test_build_selects_conditional_primary_and_keeps_wide_quality_layers_separate(tmp_path: Path) -> None:
    rows = _core_rows(trial_count=3)
    rows.extend(
        [
            _row("crowded", 24.0, 1, 0, ambiguous=8, unambiguous=8, noise=20.0),
            _row("special_code", 24.0, 0, 0, ambiguous=8, unambiguous=0, noise=67.0),
            _row("line", 24.0, 0, 0, ambiguous=8, unambiguous=8, noise=600.0),
        ]
    )
    package = build_innovation_package(_write_artifact(tmp_path, rows))

    assert package["status"] == PRIMARY_STATUS
    assert package["data_summary"]["primary_strata"] == ["blank", "edge", "high_background"]
    assert package["data_summary"]["hard_control_strata"] == ["crowded", "line", "special_code"]
    primary_rows = [row for row in package["rows"] if row["primary_eligible"]]
    assert len(primary_rows) == 9
    blank_56 = next(row for row in primary_rows if row["stratum"] == "blank" and row["peak_excess_adu"] == 56.0)
    assert blank_56["denominator_kind"] == "unambiguous_injected"
    assert blank_56["denominator_count"] == 24
    assert blank_56["minimum_unambiguous_positions_required"] == 24
    assert blank_56["candidate_recall"] == pytest.approx(1.0)
    assert blank_56["quality_recall"] == pytest.approx(0.0)
    assert blank_56["quality_filter_gap"] == pytest.approx(1.0)
    low, high = wilson_interval(24, 24)
    assert blank_56["candidate_recall_wilson95_low"] == pytest.approx(low)
    assert blank_56["candidate_recall_wilson95_high"] == pytest.approx(high)
    assert package["audit"]["checks"][-1]["status"] == "NOT_AVAILABLE"
    special = next(row for row in package["rows"] if row["stratum"] == "special_code")
    assert special["candidate_recall"] is None
    assert special["quality_recall"] is None
    assert special["candidate_recall_wilson95_low"] is None


def test_controls_are_retained_as_diagnostic_not_primary(tmp_path: Path) -> None:
    rows = _core_rows()
    rows.append(_row("line", 24.0, 8, 0, ambiguous=8, unambiguous=8, noise=600.0))
    package = build_innovation_package(_write_artifact(tmp_path, rows))

    line = next(row for row in package["rows"] if row["stratum"] == "line")
    assert line["control_class"] == "hard_control"
    assert line["primary_eligible"] is False
    assert line["denominator_kind"] == "all_injected_diagnostic"
    assert package["hard_controls"][0]["role"] == "HARD_CONTROL_ONLY"
    assert package["evidence"][0]["role"] == "PRIMARY_INNOVATION"
    assert any("误检率" in item for item in package["selected_innovation"]["not_claimed"])


def test_loader_rejects_bad_counts_and_duplicate_grid_cells(tmp_path: Path) -> None:
    invalid = _row("blank", 24.0, 3, 4)
    with pytest.raises(ValueError, match="quality recovery exceeds candidate"):
        load_stratified_rows(_write_artifact(tmp_path, [invalid], "bad.json"))

    duplicate_a = _row("blank", 24.0, 1, 0)
    duplicate_b = _row("blank", 24.0, 2, 0)
    with pytest.raises(ValueError, match="duplicate stratum/peak grid cell"):
        load_stratified_rows(_write_artifact(tmp_path, [duplicate_a, duplicate_b], "duplicate.json"))


def test_optional_evidence_is_explicitly_classified(tmp_path: Path) -> None:
    injection_path = _write_artifact(tmp_path, _core_rows())
    sequence_path = tmp_path / "sequence.json"
    sequence_path.write_text(
        json.dumps(
            {
                "relation": {"frame_count": 15, "duration_s": 21.014},
                "registration": {"max_cumulative_shift_norm_px": 0.254},
            }
        ),
        encoding="utf-8",
    )
    feature_path = tmp_path / "feature.json"
    feature_path.write_text(
        json.dumps(
            {
                "correlation_threshold": 0.8,
                "parameters": {"pattern_is_not_truth": True},
                "rows": [{"feature_class": "compact_quality"}],
            }
        ),
        encoding="utf-8",
    )

    package = build_innovation_package(injection_path, sequence_report=sequence_path, feature_matrix=feature_path)
    by_id = {item["id"]: item for item in package["evidence"]}
    assert by_id["sequence_relation"]["role"] == "BASIC_SUPPORTING"
    assert by_id["sequence_relation"]["status"] == "SUPPORTING_ONLY"
    assert by_id["sequence_relation"]["observed"]["frame_count"] == 15
    assert by_id["feature_routing"]["role"] == "DIAGNOSTIC"
    assert by_id["feature_routing"]["status"] == "DIAGNOSTIC_ONLY"
    assert by_id["feature_routing"]["observed"]["pattern_is_not_truth"] is True


def test_writer_outputs_stable_json_csv_and_markdown(tmp_path: Path) -> None:
    package = build_innovation_package(_write_artifact(tmp_path, _core_rows(trial_count=3)))
    output = tmp_path / "out"
    paths = write_innovation_package(package, output)

    json_path = Path(paths["json"])
    csv_path = Path(paths["csv"])
    markdown_path = Path(paths["markdown"])
    assert json_path.is_file()
    assert csv_path.is_file()
    assert markdown_path.is_file()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["status"] == PRIMARY_STATUS
    assert "不能声称" in markdown_path.read_text(encoding="utf-8")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(package["rows"])
    assert csv_rows[0]["denominator_kind"] == "unambiguous_injected"
    assert csv_rows[0]["rate_semantics"] == "primary_unambiguous_recall"
    assert csv_rows[0]["minimum_unambiguous_positions_required"] == "24"
    assert "trial_count=1" not in markdown_path.read_text(encoding="utf-8")


def test_cli_main_writes_package_and_require_defensible(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    injection_path = _write_artifact(tmp_path, _core_rows(trial_count=3))
    output = tmp_path / "cli-out"
    assert main([str(injection_path), "--out-dir", str(output), "--require-defensible"]) == 0
    stdout = capsys.readouterr().out
    assert "status: CONDITIONAL_DEFENSIBLE" in stdout
    assert (output / "innovation_package.json").is_file()


def test_single_layout_is_diagnostic_and_strict_cli_rejects(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    injection_path = _write_artifact(tmp_path, _core_rows())
    package = build_innovation_package(injection_path)

    assert package["status"] == "DIAGNOSTIC_ONLY"
    assert package["data_summary"]["replication_status"] == "SINGLE_LAYOUT"
    assert package["audit"]["checks"][4]["status"] == "FAIL"

    output = tmp_path / "diagnostic-cli-out"
    assert main([str(injection_path), "--out-dir", str(output), "--require-defensible"]) == 2
    stdout = capsys.readouterr().out
    assert "status: DIAGNOSTIC_ONLY" in stdout


def test_primary_gates_cannot_be_lowered_by_callers(tmp_path: Path) -> None:
    package = build_innovation_package(
        _write_artifact(tmp_path, _core_rows()),
        min_injections=1,
        min_levels=1,
        min_trials=1,
    )

    assert package["status"] == "DIAGNOSTIC_ONLY"
    assert package["data_summary"]["min_injections_gate"] == 8
    assert package["data_summary"]["min_levels_gate"] == 3
    assert package["data_summary"]["min_trials_gate"] == 3


def test_wilson_interval_validates_counts() -> None:
    assert wilson_interval(0, 8)[0] == pytest.approx(0.0)
    assert wilson_interval(0, 8)[1] > 0.0
    with pytest.raises(ValueError):
        wilson_interval(9, 8)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
