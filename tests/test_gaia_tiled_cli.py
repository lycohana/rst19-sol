from __future__ import annotations

import json

from rst19 import gaia_tiled_cli


def test_dry_run_plans_tiles_without_network(capsys) -> None:
    code = gaia_tiled_cli.main(
        [
            "--ra",
            "129.5",
            "--dec",
            "-1.8",
            "--radius",
            "2.0",
            "--tile-radius",
            "1.0",
            "--dry-run",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["network"] is False
    assert payload["initial_tile_count"] > 1


def test_incomplete_result_has_nonzero_exit_and_audit(monkeypatch, tmp_path, capsys) -> None:
    class FakeRecord:
        def as_dict(self):
            return {"tile_id": "tile-00000", "status": "saturated_unresolved"}

    class FakeResult:
        rows = ({"source_id": "1", "ra_deg": "129.5", "dec_deg": "-1.8", "magnitude": "12"},)
        complete = False
        initial_tile_count = 1
        queried_tile_count = 1
        duplicate_count = 0
        outside_scope_count = 0
        unresolved_tile_ids = ("tile-00000",)
        truncated_tile_ids = ("tile-00000",)
        errors = ()
        tile_records = (FakeRecord(),)

    monkeypatch.setattr(gaia_tiled_cli, "query_gaia_tiled", lambda *args, **kwargs: FakeResult())
    output = tmp_path / "gaia.csv"
    code = gaia_tiled_cli.main(
        [
            "--ra",
            "129.5",
            "--dec",
            "-1.8",
            "--radius",
            "0.2",
            "--out",
            str(output),
        ]
    )

    assert code == 3
    assert output.exists()
    metadata = json.loads(output.with_suffix(".csv.tiled.meta.json").read_text(encoding="utf-8"))
    assert metadata["complete"] is False
    assert metadata["truncated_tile_ids"] == ["tile-00000"]
    assert "incomplete result" in capsys.readouterr().err

