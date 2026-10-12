"""Exercise actual native float32 detection, compare all fields and preserve source/data fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_detection import check,ROOT
import numpy as np
from rst19.fits import read_fits,auxiliary_mask

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("engine",type=Path);parser.add_argument("--out",type=Path,required=True);parser.add_argument("--real-frame",type=Path);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(132);yy,xx=np.indices((96,97));image=rng.normal(100,2,yy.shape)
    for x,y,amp in [(22.2,30.1,100),(26.3,30.2,35),(70.2,60.5,8)]:image+=amp*np.exp(-.5*(((xx-x)/1.3)**2+((yy-y)/1.3)**2))
    mask=np.zeros(image.shape,dtype=bool);mask[0,:16]=True;records=[]
    for dtype in ["float32","int16"]:
        for mode in ["gaussian","hybrid","ensemble"]:
            for fast in [False,True]:
                options={"use_float32":True,"fast_sequence":fast,"refine_local_background":False,"proposal_mode":mode,"background_box_size":32,"enable_local_deblend":True}
                records.append(check(args.engine.resolve(),f"synthetic-{dtype}",image.astype(dtype),mask,options))
    original=None
    if args.real_frame:
        original=hashlib.sha256(args.real_frame.read_bytes()).hexdigest();frame=read_fits(args.real_frame);x,y,w,h=1250,160,512,512;roi=frame.data[y:y+h,x:x+w].copy().astype(np.int16);mask=auxiliary_mask(frame.data.shape)[y:y+h,x:x+w]
        for mode in ["gaussian","hybrid","ensemble"]:
            records.append(check(args.engine.resolve(),f"{args.real_frame.name}-ROI(1250,160,512,512)",roi,mask,{"use_float32":True,"fast_sequence":True,"refine_local_background":False,"proposal_mode":mode,"max_sources":300}))
        assert hashlib.sha256(args.real_frame.read_bytes()).hexdigest()==original
    report={"engine_sha256":hashlib.sha256(args.engine.read_bytes()).hexdigest(),"python_source_sha256":hashlib.sha256((ROOT/"src/rst19/detection.py").read_bytes()).hexdigest(),"records":records,"real_fits_sha256_unchanged":original,"scope":"float32 detector and bounded ROI, not full sequence"}
    (args.out/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
