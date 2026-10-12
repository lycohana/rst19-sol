"""Run native detection through its actual JSONL protocol and compare all source fields."""
from __future__ import annotations
import argparse
import hashlib
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from rst19.fits import read_fits,auxiliary_mask
from rst19.detection import detect_sources


def compare(actual,expected,path="result"):
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and actual.keys()==expected.keys(),(path,"keys")
        for key,value in expected.items():compare(actual[key],value,f"{path}.{key}")
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected),(path,"length",len(actual),len(expected))
        for index,(a,b) in enumerate(zip(actual,expected)):compare(a,b,f"{path}[{index}]")
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        assert isinstance(actual,(int,float)) and not isinstance(actual,bool),(path,"type")
        tolerance=(2e-6+abs(expected)*2e-7) if "deblend_" in path else (2e-9+abs(expected)*2e-11)
        assert abs(actual-expected)<=tolerance,(path,actual,expected,tolerance)
    else:assert actual==expected,(path,actual,expected)


def check(engine,path,image,mask,options):
    started=time.perf_counter();expected=detect_sources(image,mask=mask,**options).as_dict();python_seconds=time.perf_counter()-started
    request={"operation":"detection.array","height":image.shape[0],"width":image.shape[1],"dtype":str(image.dtype),"image":image.ravel().tolist(),"mask":mask.ravel().tolist(),"options":options}
    started=time.perf_counter();response=subprocess.run([str(engine)],input=json.dumps(request,allow_nan=False),capture_output=True,text=True,encoding="utf-8",timeout=180)
    native_seconds=time.perf_counter()-started
    events=[json.loads(line) for line in response.stdout.splitlines()]
    assert response.returncode==0 and len(events)==1 and events[0]["ok"],(response.returncode,response.stdout,response.stderr)
    actual=events[0]["payload"];compare(actual,expected)
    record={"image":path,"shape":list(image.shape),"options":options,"candidate_count":actual["candidate_count"],"returned_count":actual["returned_count"],"quality_count":actual["quality_count"],"all_fields_compared":True,"python_seconds":python_seconds,"native_seconds_including_startup_and_json":native_seconds}
    return record


def check_full_fits(engine,path,frame,options,out):
    print(f"Python full-frame baseline: {path.name}",flush=True)
    started=time.perf_counter();expected=detect_sources(frame.data,mask=auxiliary_mask(frame.data.shape),**options).as_dict();python_seconds=time.perf_counter()-started
    print(f"Native full-frame: {path.name}; Python took {python_seconds:.2f}s",flush=True)
    request={"operation":"detection.fits","path":str(path.resolve()),"options":options}
    started=time.perf_counter();response=subprocess.run([str(engine)],input=json.dumps(request),capture_output=True,text=True,encoding="utf-8",timeout=420)
    native_seconds=time.perf_counter()-started
    events=[json.loads(line) for line in response.stdout.splitlines()]
    assert response.returncode==0 and len(events)==1 and events[0]["ok"],(response.returncode,response.stderr,response.stdout[:1000])
    actual=events[0]["payload"]
    for label,payload in [("python",expected),("native",actual)]:
        with gzip.open(out/f"full-{label}.json.gz","wt",encoding="utf-8") as stream:json.dump(payload,stream,ensure_ascii=False,allow_nan=False)
    differences=[]
    for a,b in zip(actual["sources"],expected["sources"]):
        if a["flags"]!=b["flags"] or a["quality_passed"]!=b["quality_passed"]:
            differences.append({"native":a,"python":b})
    diagnostic={"native_counts":{k:actual[k] for k in ["candidate_count","returned_count","quality_count"]},"python_counts":{k:expected[k] for k in ["candidate_count","returned_count","quality_count"]},"quality_differences":differences,"python_seconds":python_seconds,"native_seconds":native_seconds}
    (out/"full-differences.json").write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding="utf-8")
    compare(actual,expected)
    return {"image":path.name,"shape":list(frame.data.shape),"options":options,"candidate_count":actual["candidate_count"],"returned_count":actual["returned_count"],"quality_count":actual["quality_count"],"all_fields_compared":True,"python_seconds":python_seconds,"native_seconds_including_startup_and_json":native_seconds,"native_input":"original FITS path; no pixel transfer through Python"}


def verify(engine,out,real_frame,full_frame=False):
    out.mkdir(parents=True,exist_ok=False);records=[]
    yy,xx=np.indices((96,97));rng=np.random.default_rng(314);synthetic=rng.normal(100.,2.,yy.shape)
    for x,y,amplitude in [(22,30,120),(26,30,40),(70,60,8)]:synthetic+=amplitude*np.exp(-.5*(((xx-x)/1.3)**2+((yy-y)/1.3)**2))
    invalid=np.zeros(synthetic.shape,dtype=bool);invalid[0,:16]=True
    for mode in ["gaussian","hybrid","ensemble"]:
        records.append(check(engine,"synthetic",synthetic,invalid,{"proposal_mode":mode,"enable_local_deblend":True,"background_box_size":32}))
    real_hash=None
    if real_frame:
        real_hash=hashlib.sha256(real_frame.read_bytes()).hexdigest();frame=read_fits(real_frame)
        # Read-only ROI includes the known long structure; this is not full-frame acceptance.
        x,y,w,h=1250,160,512,512
        image=frame.data[y:y+h,x:x+w].copy().astype(np.int16)
        invalid=auxiliary_mask(frame.data.shape)[y:y+h,x:x+w].copy()
        for mode in ["gaussian","hybrid","ensemble"]:
            records.append(check(engine,f"{real_frame.name} ROI({x},{y},{w},{h})",image,invalid,{"proposal_mode":mode,"max_sources":300,"background_box_size":128}))
        if full_frame:
            records.append(check_full_fits(engine,real_frame,frame,{"proposal_mode":"hybrid"},out))
        assert hashlib.sha256(real_frame.read_bytes()).hexdigest()==real_hash
    report={"engine":str(engine.resolve()),"engine_sha256":hashlib.sha256(engine.read_bytes()).hexdigest(),"python_detection_sha256":hashlib.sha256((ROOT/"src/rst19/detection.py").read_bytes()).hexdigest(),"records":records,"real_fits_sha256_unchanged":real_hash,"scope":"synthetic, explicit bounded ROI, plus one full-frame hybrid run if requested; no sequence claim"}
    (out/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("engine",type=Path);parser.add_argument("--out",type=Path,required=True);parser.add_argument("--real-frame",type=Path)
    parser.add_argument("--full-frame",action="store_true")
    args=parser.parse_args();print(json.dumps(verify(args.engine.resolve(),args.out,args.real_frame,args.full_frame),ensure_ascii=False,indent=2))
