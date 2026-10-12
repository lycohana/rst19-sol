"""Freeze sparse matching and rectangular solver contracts from Python/SciPy."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from scipy.optimize import linear_sum_assignment
from rst19.matching import _global_assignment,_greedy_assignment

if __name__=="__main__":
    rng=np.random.default_rng(195);matrices=[];graphs=[]
    for rows,columns in [(0,4),(4,0),(1,1),(3,7),(7,3),(16,16),(9,11),(11,9)]:
        for kind in ["constant","integer","random"]:
            a=np.zeros((rows,columns)) if kind=="constant" else (rng.integers(-4,6,(rows,columns)).astype(float) if kind=="integer" else rng.normal(0,3,(rows,columns)))
            r,c=linear_sum_assignment(a)
            matrices.append({"rows":rows,"columns":columns,"cost":a.ravel().tolist(),"expected":list(zip(r.tolist(),c.tolist()))})
    # Crowded conflicts, duplicated edges, disconnected components, and ties.
    for d,c in [(0,0),(0,5),(5,0),(1,1),(3,5),(5,3),(12,13),(30,29)]:
        for iteration in range(30):
            radius=3.;edges=[]
            for i in range(d):
                for j in range(c):
                    if rng.random()<.17:
                        residual=float(rng.integers(0,4)) if iteration%2 else float(rng.uniform(0,radius))
                        edges.append((residual,i,j))
                        if rng.random()<.1:edges.append((min(radius,residual+.1),i,j))
            for mode in ["global","greedy"]:
                expected=_global_assignment(edges,detection_count=d,catalog_count=c,radius_px=radius) if mode=="global" else _greedy_assignment(edges)
                graphs.append({"detection_count":d,"catalog_count":c,"radius":radius,"mode":mode,"edges":edges,"expected":expected})
    target=ROOT/"backend/tests/fixtures/assignment.json"
    target.write_text(json.dumps({"source_sha256":hashlib.sha256((ROOT/"src/rst19/matching.py").read_bytes()).hexdigest(),"matrices":matrices,"graphs":graphs},indent=2),encoding="utf-8")
    print(target)
