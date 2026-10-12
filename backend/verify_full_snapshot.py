"""Re-run a native full frame against a source/data-bound Python snapshot captured in this migration."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_detection import compare,ROOT


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("engine",type=Path);parser.add_argument("frame",type=Path);parser.add_argument("baseline",type=Path);parser.add_argument("--provenance",type=Path,required=True);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    provenance=json.loads(args.provenance.read_text(encoding="utf-8"))
    data_hash=hashlib.sha256(args.frame.read_bytes()).hexdigest();source_hash=hashlib.sha256((ROOT/"src/rst19/detection.py").read_bytes()).hexdigest()
    assert data_hash==provenance["real_fits_sha256_unchanged"] and source_hash==provenance["python_detection_sha256"],"snapshot source/data provenance changed"
    expected=json.load(gzip.open(args.baseline,"rt",encoding="utf-8"));assert expected["parameters"]["proposal_mode"]=="hybrid" and expected["parameters"]["max_sources"]==-1
    request={"operation":"detection.fits","path":str(args.frame.resolve()),"options":{"proposal_mode":"hybrid"}}
    started=time.perf_counter();response=subprocess.run([str(args.engine.resolve())],input=json.dumps(request),capture_output=True,text=True,encoding="utf-8",timeout=420);elapsed=time.perf_counter()-started
    event=json.loads(response.stdout);assert response.returncode==0 and event["ok"],(response.returncode,response.stdout[:1000],response.stderr)
    actual=event["payload"]
    args.out.mkdir(parents=True,exist_ok=False)
    with gzip.open(args.out/"native.json.gz","wt",encoding="utf-8") as stream:json.dump(actual,stream,ensure_ascii=False,allow_nan=False)
    compare(actual,expected)
    assert hashlib.sha256(args.frame.read_bytes()).hexdigest()==data_hash
    report={"engine_sha256":hashlib.sha256(args.engine.read_bytes()).hexdigest(),"frame_sha256":data_hash,"python_source_sha256":source_hash,"baseline_sha256":hashlib.sha256(args.baseline.read_bytes()).hexdigest(),"image_shape":actual["image_shape"],"candidate_count":actual["candidate_count"],"returned_count":actual["returned_count"],"quality_count":actual["quality_count"],"all_fields_compared":True,"unlimited_sources":True,"native_seconds_including_startup_and_json":elapsed,"mode":"hybrid","scope":"one full frame float64 hybrid; no sequence or all-tools claim"}
    (args.out/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
