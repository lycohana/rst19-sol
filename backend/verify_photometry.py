"""Exercise photometric arithmetic, absolute gates and fits through native JSONL."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).parent))
from verify_detection import compare,ROOT

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("engine",type=Path);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    source_hash=hashlib.sha256((ROOT/"src/rst19/photometry.py").read_bytes()).hexdigest();count=0
    def check(request,expected=None,error=False):
        global count
        response=subprocess.run([str(args.engine.resolve())],input=json.dumps(request,ensure_ascii=False),capture_output=True,text=True,encoding="utf-8",timeout=30);event=json.loads(response.stdout)
        if error:assert response.returncode==1 and not event["ok"],event
        else:
            assert response.returncode==0 and event["ok"],event
            compare(event["payload"],expected)
        count+=1
    hashes={};case_counts={}
    for name in ["photometry","calibration"]:
        path=Path(__file__).parent/"tests/fixtures"/f"{name}.json";fixture=json.loads(path.read_text(encoding="utf-8"));assert fixture["source_sha256"]==source_hash,"Python baseline changed";hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest();case_counts[name]=len(fixture["cases"])
        for case in fixture["cases"]:
            if name=="photometry":check({"operation":"photometry.evaluate","kind":case["kind"],"parameters":case["parameters"],"source":case["source"]},case["expected"],case["error"])
            else:
                check({"operation":"photometry.calibration.fit",**{k:case[k] for k in ["matches","detections","catalog","options"]}},case["expected"])
                for application in case["application"]:
                    check({"operation":"photometry.calibration.apply","calibration":case["expected"],"instrumental":application["instrumental"],"color":application["color"]},{"value":application["expected"]})
                for error in case["errors"]:
                    check({"operation":"photometry.calibration.error","calibration":case["expected"],"instrumental_error":error["instrumental_error"],"color_error":error["color_error"]},{"value":error["expected"]})
        if name=="calibration":
            for options in fixture["invalid_options"]:check({"operation":"photometry.calibration.fit","matches":[],"detections":[],"catalog":[],"options":options},error=True)
    report={"engine_sha256":hashlib.sha256(args.engine.read_bytes()).hexdigest(),"python_photometry_sha256":source_hash,"fixture_hashes":hashes,"case_counts":case_counts,"native_requests_verified":count,"all_fields_compared":True,"scope":"photometric scalar/absolute gates, weighted calibration, application and errors; faintest/source table and complete frame workflow remain pending"}
    args.out.mkdir(parents=True,exist_ok=False);(args.out/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
