"""Generate the complete module/function and entry-point migration inventory."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import tomllib

ROOT=Path(__file__).resolve().parents[1]


def symbols(tree:ast.AST,prefix:str="")->list[str]:
    result=[]
    for node in getattr(tree,"body",[]):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            name=f"{prefix}{node.name}"
            result.append(name)
            if isinstance(node,ast.ClassDef):result.extend(symbols(node,f"{name}."))
    return result


if __name__=="__main__":
    scripts=tomllib.loads((ROOT/"pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    tools={name:target for name,target in scripts.items() if target not in ("rst19.gui:main","rst19.desktop_launcher:main")}
    administrative={"gui.py","desktop_worker.py","desktop_launcher.py","__init__.py","__main__.py"}
    modules=[]
    for path in sorted((ROOT/"src/rst19").glob("*.py")):
        if path.name in administrative:continue
        text=path.read_text(encoding="utf-8")
        modules.append({"python_module":f"rst19.{path.stem}","sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                        "line_count":len(text.splitlines()),"symbols":symbols(ast.parse(text)),
                        "migration_status":"in_progress" if path.stem in {"fits","models","detection","wcs","attitude"} else "pending"})
    payload={"objective":"all backend calculations implemented in Rust; production runtime has no Python dependency",
             "complete":False,"python_runtime_removed":False,"modules":modules,"tools":tools,
             "native_implemented":["fits raw typed primary-HDU reader","auxiliary decode and plausibility",
                                   "exposure units","f64 sigma-clipped statistics","f64 local background maps",
                                   "TAN projection","quaternion boresight","affine projection and velocity transform"],
             "native_detection_helpers":["f64 masked Gaussian and starlet filters","sliding maximum filter",
                                          "peak-window extraction and NMS","Gaussian-anchor multichannel merging",
                                          "connected-component PCA line audit","exact 2-D KD-tree queries",
                                          "source aperture photometry shape and noise","fixed single/pair PSF BIC and covariance",
                                          "Gaussian close-pair guard","local residual deblend proposals","legacy compact KD-tree tie handling"],
             "native_detection_execution":"float64 Gaussian/hybrid/ensemble; optional local deblend; shared-background shape validation",
             "verification":["tests/parity.rs","tests/measurement_parity.rs","tests/detection_parity.rs","tests/spatial_parity.rs","verify_fits.py","verify_detection.py","verify_spatial_ties.py","verify_full_snapshot.py"],
             "remaining":["float32 sequence calculation path and full detection acceptance coverage","catalog and photometric calibration","sequence and motion",
                          "all scientific audits and experiments","all CLI input/output contracts","native desktop routing",
                          "remove Python from runtime packaging","full scientific and cross-platform acceptance"]}
    target=ROOT/"backend/migration.json"
    target.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"{len(modules)} Python backend modules, {len(tools)} tools, {sum(len(m['symbols']) for m in modules)} symbols inventoried")
