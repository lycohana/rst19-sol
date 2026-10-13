"""Build a complete native desktop distribution on the target OS.

python desktop/build.py
Never delete an existing package: each build uses a fresh timestamped directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import venv

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "desktop"


def command(args: list[str], **kwargs) -> None:
    subprocess.run(args, check=True, cwd=ROOT, **kwargs)


def finalize(package: Path) -> Path:
    hashes = {str(p.relative_to(package)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.rglob("*")) if p.is_file() and p.name != "SHA256.json"}
    (package / "SHA256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return Path(shutil.make_archive(str(package), "zip", package.parent, package.name))


def bundle_font(cached: Path, assets: Path) -> None:
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    font = TTFont(cached / "NotoSansSC-Regular.ttf")
    # egui reads default outlines. Make the intended Regular weight explicit,
    # rather than rendering the variable font's very thin default instance.
    static = instantiateVariableFont(font, {"wght": 400}, inplace=True)
    static.save(assets / "NotoSansSC-Regular.ttf")


def build() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-build-environment", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.in_build_environment:
        environment = DESKTOP / ".build-venv"
        if not environment.is_dir():
            venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command([str(python), "-m", "pip", "install", "-r", str(DESKTOP / "requirements-build.txt")])
        command([str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation", str(ROOT)])
        command([str(python), str(Path(__file__).resolve()), "--in-build-environment"])
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    label = f"RST19-{platform.system()}-{platform.machine()}-{stamp}"
    staging = DESKTOP / "build" / stamp
    package = ROOT / "dist" / label
    package.mkdir(parents=True, exist_ok=False)
    command(["cargo", "build", "--release", "--locked", "--manifest-path", str(DESKTOP / "Cargo.toml"), "--bins"])
    command(["cargo", "build", "--release", "--locked", "--manifest-path", str(ROOT / "backend/Cargo.toml"),
             "--bin", "rst19", "--bin", "rst19-format-audit", "--bin", "rst19-feature-sequence-compare", "--bin", "rst19-feature-raw-evidence-audit", "--bin", "rst19-signed-null-overlap", "--bin", "rst19-gaia-merge", "--bin", "rst19-aux-audit", "--bin", "rst19-source-geometry-audit",
              "--bin", "rst19-relative-photometry", "--bin", "rst19-wcs-validation",
              "--bin", "rst19-innovation-package",
              "--bin", "rst19-temporal-codes",
              "--bin", "rst19-sweep",
              "--bin", "rst19-feature-psf-leaveout",
              "--bin", "rst19-pair-pixel-topology",
              "--bin", "rst19-signed-null-audit", "--bin", "rst19-signed-null-sequence"])
    entries = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    modules = sorted({target.split(":")[0] for target in entries.values()
                      if target not in ("rst19.gui:main", "rst19.desktop_launcher:main")}
                     | {"rst19.cli"})
    freeze = [sys.executable, "-m", "PyInstaller", "--onedir", "--noupx", "--name", "rst19-worker",
              "--distpath", str(staging / "frozen"), "--workpath", str(staging / "work"),
              "--specpath", str(staging), "--paths", str(ROOT / "src"),
              "--copy-metadata", "rst19-sol", "--exclude-module", "tkinter",
              "--exclude-module", "rst19.gui"]
    for module in modules:
        freeze += ["--hidden-import", module]
    freeze.append(str(DESKTOP / "worker_entry.py"))
    command(freeze)
    shutil.copytree(staging / "frozen/rst19-worker", package / "runtime")
    native_worker = "rst19-native-worker.exe" if os.name == "nt" else "rst19-native-worker"
    shutil.copy2(DESKTOP / "target/release" / native_worker, package / "runtime" / native_worker)
    single_frame_cli = "rst19.exe" if os.name == "nt" else "rst19"
    shutil.copy2(ROOT / "backend/target/release" / single_frame_cli, package / "runtime" / single_frame_cli)
    format_audit_cli = "rst19-format-audit.exe" if os.name == "nt" else "rst19-format-audit"
    shutil.copy2(ROOT / "backend/target/release" / format_audit_cli, package / "runtime" / format_audit_cli)
    sequence_feature_compare = "rst19-feature-sequence-compare.exe" if os.name == "nt" else "rst19-feature-sequence-compare"
    shutil.copy2(ROOT / "backend/target/release" / sequence_feature_compare, package / "runtime" / sequence_feature_compare)
    raw_evidence_audit = "rst19-feature-raw-evidence-audit.exe" if os.name == "nt" else "rst19-feature-raw-evidence-audit"
    shutil.copy2(ROOT / "backend/target/release" / raw_evidence_audit, package / "runtime" / raw_evidence_audit)
    signed_null_overlap = "rst19-signed-null-overlap.exe" if os.name == "nt" else "rst19-signed-null-overlap"
    shutil.copy2(ROOT / "backend/target/release" / signed_null_overlap, package / "runtime" / signed_null_overlap)
    signed_null_audit = "rst19-signed-null-audit.exe" if os.name == "nt" else "rst19-signed-null-audit"
    shutil.copy2(ROOT / "backend/target/release" / signed_null_audit, package / "runtime" / signed_null_audit)
    signed_null_sequence = "rst19-signed-null-sequence.exe" if os.name == "nt" else "rst19-signed-null-sequence"
    shutil.copy2(ROOT / "backend/target/release" / signed_null_sequence, package / "runtime" / signed_null_sequence)
    gaia_merge = "rst19-gaia-merge.exe" if os.name == "nt" else "rst19-gaia-merge"
    shutil.copy2(ROOT / "backend/target/release" / gaia_merge, package / "runtime" / gaia_merge)
    auxiliary_audit = "rst19-aux-audit.exe" if os.name == "nt" else "rst19-aux-audit"
    shutil.copy2(ROOT / "backend/target/release" / auxiliary_audit, package / "runtime" / auxiliary_audit)
    geometry_audit = "rst19-source-geometry-audit.exe" if os.name == "nt" else "rst19-source-geometry-audit"
    shutil.copy2(ROOT / "backend/target/release" / geometry_audit, package / "runtime" / geometry_audit)
    relative_photometry = "rst19-relative-photometry.exe" if os.name == "nt" else "rst19-relative-photometry"
    shutil.copy2(ROOT / "backend/target/release" / relative_photometry, package / "runtime" / relative_photometry)
    wcs_validation = "rst19-wcs-validation.exe" if os.name == "nt" else "rst19-wcs-validation"
    shutil.copy2(ROOT / "backend/target/release" / wcs_validation, package / "runtime" / wcs_validation)
    innovation_package = "rst19-innovation-package.exe" if os.name == "nt" else "rst19-innovation-package"
    shutil.copy2(ROOT / "backend/target/release" / innovation_package, package / "runtime" / innovation_package)
    temporal_codes = "rst19-temporal-codes.exe" if os.name == "nt" else "rst19-temporal-codes"
    shutil.copy2(ROOT / "backend/target/release" / temporal_codes, package / "runtime" / temporal_codes)
    detection_sweep = "rst19-sweep.exe" if os.name == "nt" else "rst19-sweep"
    shutil.copy2(ROOT / "backend/target/release" / detection_sweep, package / "runtime" / detection_sweep)
    feature_psf_leaveout = "rst19-feature-psf-leaveout.exe" if os.name == "nt" else "rst19-feature-psf-leaveout"
    shutil.copy2(ROOT / "backend/target/release" / feature_psf_leaveout, package / "runtime" / feature_psf_leaveout)
    pair_pixel_topology = "rst19-pair-pixel-topology.exe" if os.name == "nt" else "rst19-pair-pixel-topology"
    shutil.copy2(ROOT / "backend/target/release" / pair_pixel_topology, package / "runtime" / pair_pixel_topology)
    executable = "rst19-desktop.exe" if os.name == "nt" else "rst19-desktop"
    shutil.copy2(DESKTOP / "target/release" / executable, package / executable)
    assets = package / "assets"
    assets.mkdir()
    # A redistributable CJK font: do not copy proprietary system fonts.
    cached = DESKTOP / ".assets"
    cached.mkdir(exist_ok=True)
    urls = {
        "NotoSansSC-Regular.ttf": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/Variable/TTF/Subset/NotoSansSC-VF.ttf",
        "NotoSansSC-LICENSE.txt": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/LICENSE",
    }
    for name, url in urls.items():
        target = cached / name
        if not target.is_file():
            temporary = target.with_suffix(".download")
            with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(target)
        if not name.endswith(".ttf"):
            shutil.copy2(target, assets / name)
    bundle_font(cached, assets)
    shutil.copy2(DESKTOP / "README.md", package / "使用说明.md")
    license_dir = package / "licenses"
    license_dir.mkdir()
    for distribution in ("numpy", "scipy", "Pillow", "pyinstaller"):
        import importlib.metadata
        metadata = importlib.metadata.distribution(distribution)
        for file in metadata.files or ():
            if "license" in str(file).lower() or "copying" in str(file).lower():
                source = Path(metadata.locate_file(file))
                if source.is_file():
                    destination = license_dir / distribution / str(file)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        shutil.copy2(python_license, license_dir / "Python-LICENSE.txt")
    # Frozen catalog imports every advertised algorithm module, including native extensions.
    worker = package / "runtime" / ("rst19-worker.exe" if os.name == "nt" else "rst19-worker")
    verification_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([str(worker), "--catalog"], capture_output=True, text=True, encoding="utf-8",
                            check=True, cwd=package, env=verification_env)
    catalog = json.loads(result.stdout)
    expected_tools = {name for name, target in entries.items()
                      if target not in ("rst19.gui:main", "rst19.desktop_launcher:main")}
    expected_tools.add("rst19")
    catalog_tools = {tool["name"] for tool in catalog.get("payload", {}).get("tools", [])}
    missing_tools = expected_tools - catalog_tools
    if not catalog.get("ok") or missing_tools:
        raise RuntimeError(
            f"Frozen environment verification failed; missing tools {sorted(missing_tools)}: {result.stdout}"
        )
    (package / "environment.json").write_text(json.dumps(catalog["payload"]["environment"], indent=2), encoding="utf-8")
    archive = finalize(package)
    print(f"\n完整运行包：{package}\n分发 ZIP：{archive}")


if __name__ == "__main__":
    build()
