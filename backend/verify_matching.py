"""Exercise catalog/CSV/matching operations through the native process."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import csv
import sys
sys.path.insert(0,str(Path(__file__).parent))
from verify_detection import compare,ROOT

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("engine",type=Path);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    path=Path(__file__).parent/"tests/fixtures/matching.json";fixture=json.loads(path.read_text(encoding="utf-8"))
    for name,digest in fixture["source_hashes"].items():assert hashlib.sha256((ROOT/"src/rst19"/name).read_bytes()).hexdigest()==digest,"Python baseline changed"
    count=0
    def check(request,expected=None,error=False):
        global count
        response=subprocess.run([str(args.engine.resolve())],input=json.dumps(request,ensure_ascii=False),capture_output=True,text=True,encoding="utf-8",timeout=30)
        event=json.loads(response.stdout)
        if error:assert response.returncode==1 and not event["ok"],event
        else:
            assert response.returncode==0 and event["ok"],event
            compare(event["payload"],expected)
        count+=1
    for case in fixture["propagation"]:
        check({"operation":"catalog.propagate","source":case["source"],"epoch":case["epoch"]},{"source":case["expected"],"extinction_compatibility":case["extinction_compatibility"]})
    for source in fixture["invalid_sources"]:check({"operation":"catalog.propagate","source":source},error=True)
    for source in fixture["invalid_sources"]:
        check({"operation":"matching.catalog","detections":[],"catalog":[source],"wcs":fixture["matching"][0]["wcs"]},error=True)
    with tempfile.TemporaryDirectory() as directory:
        for i,case in enumerate(fixture["csv"]):
            csv_path=Path(directory)/f"{i}.csv";csv_path.write_text(case["csv"],encoding="utf-8",newline="")
            check({"operation":"catalog.load","path":str(csv_path)},None if case["error"] else {"sources":case["expected"]},case["error"])
            assert csv_path.read_bytes()==case["csv"].encode("utf-8"),"CSV input changed"
    for case in fixture["matching"]:
        request={"operation":"matching.catalog",**{k:v for k,v in case.items() if k!="expected"}}
        check(request,case["expected"])
    with tempfile.TemporaryDirectory() as directory:
        for i,case in enumerate(fixture["matching"]):
            if not case["catalog"]:continue
            csv_path=Path(directory)/f"match-{i}.csv"
            with csv_path.open("w",encoding="utf-8-sig",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(case["catalog"][0]));writer.writeheader();writer.writerows(case["catalog"])
            digest=hashlib.sha256(csv_path.read_bytes()).hexdigest()
            request={"operation":"matching.catalog",**{k:v for k,v in case.items() if k not in {"expected","catalog"}},"catalog_path":str(csv_path)}
            check(request,case["expected"])
            assert hashlib.sha256(csv_path.read_bytes()).hexdigest()==digest,"matching changed CSV input"
    report={"engine_sha256":hashlib.sha256(args.engine.read_bytes()).hexdigest(),"fixture_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"python_source_hashes":fixture["source_hashes"],"propagation_cases":len(fixture["propagation"]),"invalid_source_cases":len(fixture["invalid_sources"]),"csv_cases":len(fixture["csv"]),"matching_cases":len(fixture["matching"]),"native_requests_verified":count,"all_fields_compared":True,"scope":"catalog model, offline CSV, epoch propagation, TAN/affine matching; no CLI/desktop or full workflow acceptance"}
    args.out.mkdir(parents=True,exist_ok=False);(args.out/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
