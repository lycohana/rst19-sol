"""Freeze nearest-neighbor tie results from the supported SciPy baseline."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

if __name__=="__main__":
    rng=np.random.default_rng(177);cases=[]
    for count in [3,16,17,101,1001]:
        points=rng.integers(0,100,size=(count,2)).astype(float);queries=rng.integers(0,100,size=(300,2)).astype(float)
        distances,indices=cKDTree(points).query(queries)
        cases.append({"points":points.tolist(),"queries":queries.tolist(),"expected":[[int(i),float(d)] for i,d in zip(indices,distances)]})
    target=Path(__file__).resolve().parent/"tests/fixtures/spatial.json";target.write_text(json.dumps(cases,indent=2),encoding="utf-8");print(target)
