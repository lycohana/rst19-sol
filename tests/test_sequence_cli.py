from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import rst19.sequence_cli as sequence_cli


def test_sequence_cli_defaults_match_competition_gui_profile() -> None:
    parser = sequence_cli.build_parser()
    args = parser.parse_args(["data"])

    assert args.min_distance == 4
    assert args.psf_fwhm == 2.0
    assert args.proposal_mode == "hybrid"


def test_sequence_cli_reports_progress_on_stderr(tmp_path: Path, monkeypatch, capsys) -> None:
    (tmp_path / "frame-01.fits").write_bytes(b"placeholder")

    def fake_analyze_sequence(*paths, **kwargs):
        assert kwargs["progress"] is not None
        assert kwargs["detail_progress"] is not None
        kwargs["progress"]("prepare", 0, 1)
        kwargs["detail_progress"](1, 1, 100.0, "检测完成")
        kwargs["progress"]("frame", 1, 1)
        kwargs["progress"]("complete", 1, 1)
        return SimpleNamespace(as_dict=lambda: {"status": "TEST"})

    monkeypatch.setattr(sequence_cli, "analyze_sequence", fake_analyze_sequence)

    assert sequence_cli.main([str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "逐帧检测" in captured.err
    assert '"status": "TEST"' in captured.out
