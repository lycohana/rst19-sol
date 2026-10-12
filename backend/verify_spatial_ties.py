"""Verify legacy neighbor selection against SciPy, including equidistant candidates."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial import cKDTree


def check(engine,points,queries,label):
    tree=cKDTree(points);distances,indices=tree.query(queries)
    request={"operation":"spatial.compatible_query","points":points.tolist(),"queries":queries.tolist()}
    response=subprocess.run([str(engine)],input=json.dumps(request),capture_output=True,text=True,encoding="utf-8",timeout=60)
    event=json.loads(response.stdout);assert response.returncode==0 and event["ok"],event
    actual=event["payload"]["nearest"]
    differences=[{"query":queries[i].tolist(),"native":a,"scipy":[int(b),float(d)]} for i,(a,b,d) in enumerate(zip(actual,indices,distances)) if a[0]!=int(b) or abs(a[1]-d)>1e-10]
    return {"label":label,"points":len(points),"queries":len(queries),"differences":differences}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("engine",type=Path);parser.add_argument("--full-result",type=Path);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    rng=np.random.default_rng(177);records=[]
    for count in [3,16,17,101,1001]:
        points=rng.integers(0,100,size=(count,2)).astype(float);queries=rng.integers(0,100,size=(300,2)).astype(float)
        records.append(check(args.engine.resolve(),points,queries,f"seeded-{count}"))
    if args.full_result:
        data=json.load(gzip.open(args.full_result,"rt",encoding="utf-8"));gaussian=[s for s in data["sources"] if "gaussian" in s["proposal_methods"]]
        gaussian.sort(key=lambda s:(-s["proposal_snr"],-s["filter_snr"]))
        points=np.array([[s["peak_x"],s["peak_y"]] for s in gaussian]);queries=np.array([[s["peak_x"],s["peak_y"]] for s in data["sources"] if "gaussian" not in s["proposal_methods"] and s["nearest_gaussian_px"] is not None and s["nearest_gaussian_px"]<7.5])
        records.append(check(args.engine.resolve(),points,queries,"full-frame-Gaussian-neighbor-selection"))
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps([{k:v if k!="differences" else len(v) for k,v in r.items()} for r in records],indent=2))
    if any(r["differences"] for r in records):raise SystemExit(1)
