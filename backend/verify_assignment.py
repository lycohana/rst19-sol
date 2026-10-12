"""Check all frozen assignment contracts through the native JSONL process."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("engine",type=Path);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    fixture_path=Path(__file__).parent/"tests/fixtures/assignment.json";fixture=json.loads(fixture_path.read_text(encoding="utf-8"));requests=[];expected=[]
    for case in fixture["matrices"]:
        requests.append({"operation":"assignment.solve","rows":case["rows"],"columns":case["columns"],"cost":case["cost"]});expected.append({"assignment":case["expected"]})
    for case in fixture["graphs"]:
        requests.append({"operation":"matching.assign",**{k:v for k,v in case.items() if k!="expected"}});expected.append({"matches":case["expected"]})
    for i,(request,target) in enumerate(zip(requests,expected)):
        response=subprocess.run([str(args.engine.resolve())],input=json.dumps(request),capture_output=True,text=True,encoding="utf-8",timeout=60)
        assert response.returncode==0,response.stderr
        event=json.loads(response.stdout)
        assert event["ok"] and event["payload"]==target,(i,event,target)
    report={"engine_sha256":hashlib.sha256(args.engine.read_bytes()).hexdigest(),"fixture_sha256":hashlib.sha256(fixture_path.read_bytes()).hexdigest(),"python_source_sha256":fixture["source_sha256"],"matrix_cases":len(fixture["matrices"]),"sparse_graph_cases":len(fixture["graphs"]),"all_fields_compared":True,"scope":"assignment kernel and native JSONL; catalog propagation/projection and full matching entrypoint remain pending"}
    args.out.mkdir(parents=True,exist_ok=False);(args.out/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
