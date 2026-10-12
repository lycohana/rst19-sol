"""Launch the Rust desktop from an installed Python entry point."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    filename = "rst19-desktop.exe" if os.name == "nt" else "rst19-desktop"
    explicit = os.environ.get("RST19_DESKTOP")
    candidates = ([Path(explicit)] if explicit else [])
    candidates += sorted((root / "dist").glob(f"RST19-*/{filename}"), reverse=True)
    candidates += [root / "desktop/target/release" / filename]
    for path in candidates:
        if path.is_file():
            environment = os.environ.copy()
            environment.setdefault("RST19_PYTHON", sys.executable)
            return subprocess.call([str(path), *(sys.argv[1:] if argv is None else argv)], env=environment)
    print("Rust 界面尚未构建。运行 python desktop/build.py 生成包含全部运行环境的发布包，"
          "或 cargo build --release --manifest-path desktop/Cargo.toml。", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
