"""Compare native raw-peak fast-candidate association with fixed Python cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from verify_detection import compare


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "tests/fixtures/sequence_fast_candidates.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    engine = args.engine.resolve()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for name, expected in fixture["source_hashes"].items():
        actual = hashlib.sha256((ROOT / "src/rst19" / name).read_bytes()).hexdigest()
        assert actual == expected, f"Python baseline changed: {name}"

    records = []
    for case in fixture["cases"]:
        response = subprocess.run(
            [str(engine)],
            input=json.dumps(
                {"operation": "sequence.fast_candidate_associate", **case["request"]},
                ensure_ascii=False,
                allow_nan=False,
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        events = [json.loads(line) for line in response.stdout.splitlines()]
        assert response.returncode == 0 and len(events) == 1 and events[0]["ok"], (
            case["label"], response.returncode, response.stdout, response.stderr
        )
        compare(events[0]["payload"], case["expected"])
        records.append({"label": case["label"], "all_fields_compared": True})

    report = {
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "python_source_hashes": fixture["source_hashes"],
        "native_requests_verified": len(records),
        "records": records,
        "all_fields_compared": True,
        "scope": "constant-velocity association of prefiltered raw candidate peaks before original-image forced photometry; proposals remain unconfirmed until downstream measurement gates",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
