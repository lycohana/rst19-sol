import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from rst19.motion_crossing_ablation import run_motion_crossing_ablation
from rst19.motion_crossing_ablation_cli import build_parser


def _read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _independent_track_metrics(point_rows):
    by_track = defaultdict(list)
    for row in point_rows:
        by_track[(row["scenario"], row["version"], row["output_track_id"])].append(row)
    metrics = {}
    for key, rows in by_track.items():
        rows.sort(key=lambda row: int(row["frame_index"]))
        labels = [row["truth_target_id"] for row in rows if row["truth_target_id"]]
        metrics[key] = {
            "switches": sum(left != right for left, right in zip(labels, labels[1:])),
            "mixed": len(set(labels)) > 1,
            "targets": set(labels),
        }
    return metrics


def test_crossing_ablation_freezes_inputs_and_recomputes_expected_metrics(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = run_motion_crossing_ablation(first_dir)
    second = run_motion_crossing_ablation(second_dir)

    for artifact in (
        "truth.csv",
        "track_level.csv",
        "target_summary.csv",
        "track_points.csv",
        "frozen_inputs/slow_linked_identity_swap.csv.gz",
        "frozen_inputs/fast_orthogonal_supplement.csv.gz",
    ):
        assert (first_dir / artifact).read_bytes() == (second_dir / artifact).read_bytes()

    assert first["status"] == "completed"
    assert first["schema_version"] == 1
    for scenario, run in first["scenarios"].items():
        frozen = run["input"]
        assert frozen["same_input_sha256_for_A_and_B"] is True
        assert frozen["A_and_B_same_Detection_list"] is True
        assert frozen["A_input_fingerprint"] == frozen["B_input_fingerprint"]
        assert frozen["B_input_fingerprint"] == frozen["post_B_input_fingerprint"]
        assert run["registration"]["zero_verified"] is True
        assert run["registration"]["max_shift_px"] == 0.0
        assert len(run["registration"]["cumulative_shifts_px"]) == 15

    summary_rows = _read_csv(first_dir / "summary.csv")
    second_summary_rows = _read_csv(second_dir / "summary.csv")
    for row in summary_rows + second_summary_rows:
        row.pop("runtime_seconds", None)
    assert summary_rows == second_summary_rows
    summary = {(row["scenario"], row["version"]): row for row in summary_rows}
    targets = _read_csv(first_dir / "target_summary.csv")
    target_index = {
        (row["scenario"], row["version"], row["target_id"]): row
        for row in targets
    }

    slow_a = summary[("slow_linked_identity_swap", "A")]
    slow_b = summary[("slow_linked_identity_swap", "B_union")]
    assert slow_a["targets_with_full_clean_track_present"] == "0"
    assert slow_a["identity_switch_count"] == "2"
    assert slow_a["mixed_truth_track_count"] == "2"
    assert slow_a["target_track_assignments_fragment_count_sum"] == "4"
    assert slow_b["identity_switch_count"] == slow_a["identity_switch_count"]
    assert slow_b["mixed_truth_track_count"] == slow_a["mixed_truth_track_count"]
    assert slow_b["predicted_track_count"] == slow_a["predicted_track_count"]
    assert summary[("slow_linked_identity_swap", "B_supplement")]["predicted_track_count"] == "0"
    assert {
        target_index[("slow_linked_identity_swap", "A", target_id)]["predicted_tracks_containing_target"]
        for target_id in ("SLOW-CROSS-A", "SLOW-CROSS-B")
    } == {"2"}

    fast_a = summary[("fast_orthogonal_supplement", "A")]
    fast_supplement = summary[("fast_orthogonal_supplement", "B_supplement")]
    fast_union = summary[("fast_orthogonal_supplement", "B_union")]
    assert fast_a["targets_with_full_clean_track_present"] == "0"
    assert fast_a["target_track_assignments_fragment_count_sum"] == "30"
    assert fast_supplement["targets_with_full_clean_track_present"] == "2"
    assert fast_supplement["targets_with_exclusive_full_track"] == "2"
    assert fast_supplement["identity_switch_count"] == "0"
    assert fast_union["targets_with_full_clean_track_present"] == "2"
    assert fast_union["targets_with_exclusive_full_track"] == "0"
    assert fast_union["target_track_assignments_fragment_count_sum"] == "32"
    assert fast_union["duplicated_truth_observation_assignments"] == "30"
    assert {
        target_index[("fast_orthogonal_supplement", "B_supplement", target_id)]["predicted_tracks_containing_target"]
        for target_id in ("FAST-CROSS-A", "FAST-CROSS-B")
    } == {"1"}

    point_rows = _read_csv(first_dir / "track_points.csv")
    independent = _independent_track_metrics(point_rows)
    for key, row in summary.items():
        selected = {
            track_key: values
            for track_key, values in independent.items()
            if track_key[:2] == key
        }
        assert sum(values["switches"] for values in selected.values()) == int(row["identity_switch_count"])
        assert sum(values["mixed"] for values in selected.values()) == int(row["mixed_truth_track_count"])
        target_ids = (
            ("SLOW-CROSS-A", "SLOW-CROSS-B")
            if key[0] == "slow_linked_identity_swap"
            else ("FAST-CROSS-A", "FAST-CROSS-B")
        )
        for target_id in target_ids:
            fragment_count = sum(target_id in values["targets"] for values in selected.values())
            assert fragment_count == int(
                target_index[(key[0], key[1], target_id)]["predicted_tracks_containing_target"]
            )

    truth_rows = _read_csv(first_dir / "truth.csv")
    assert len(truth_rows) == 60
    manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "FITS pixels" in manifest["limitations"][0]
    for artifact in manifest["outputs"].values():
        artifact_path = Path(artifact["path"])
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]


def test_crossing_ablation_cli_has_safe_output_option() -> None:
    args = build_parser().parse_args(["--out-dir", "tmp/crossing-run"])
    assert str(args.out_dir) == "tmp\\crossing-run"
