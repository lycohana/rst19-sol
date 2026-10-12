"""Freeze representative end-to-end detection results from the unchanged Python baseline."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from rst19.detection import detect_sources


def generate():
    rng=np.random.default_rng(1919);cases=[]
    for dtype,precision32 in [("float64",False),("int16",False),("int16",True),("float32",True),("float32",False)]:
        for mode in ["gaussian","hybrid","ensemble"]:
            for variant in ["stars","artifacts","truncated","fast","deblend","deblend_masked","deblend_edge","shared_background","shape_fallback","float64_coercion"]:
                if variant=="float64_coercion" and dtype!="float64":continue
                yy,xx=np.indices((64,65));image=rng.normal(100.,2.,yy.shape)
                for x,y,amplitude,sigma in [(20.2,25.1,100,1.3),(24.5,25.2,25,1.3),(47.4,43.2,6,.8)]:
                    image+=amplitude*np.exp(-.5*(((xx-x)/sigma)**2+((yy-y)/sigma)**2))
                mask=np.zeros(image.shape,dtype=bool);mask[0,:12]=True
                if variant=="artifacts":
                    image[40:42,4:35]+=80;image[16,37]=2000;image[25,21]=0;image[30,30]=-32000
                image=image.astype(dtype)
                options={"proposal_mode":mode,"threshold_sigma":4.0,"min_distance":3,"background_box_size":16,
                         "dog_threshold_sigma":5.0,"starlet_threshold_sigma":6.0}
                if precision32:options["use_float32"]=True
                if variant=="truncated":options["max_sources"]=1
                if variant=="fast":options.update(fast_sequence=True,refine_local_background=False)
                if variant=="float64_coercion":options["use_float32"]=True
                if variant.startswith("deblend"):
                    options["enable_local_deblend"]=True
                    if variant=="deblend_masked":mask[25,19]=True
                    if variant=="deblend_edge":
                        image=image.astype(np.float64)
                        image+=150*np.exp(-.5*(((xx-4)/1.3)**2+((yy-5)/1.3)**2))
                        image+=45*np.exp(-.5*(((xx-7)/1.3)**2+((yy-5)/1.3)**2))
                        image=image.astype(dtype)
                model=None
                if variant in ["shared_background","shape_fallback"]:
                    shape=image.shape if variant=="shared_background" else (1,image.size)
                    model=(np.full(shape,100.),np.full(shape,2.))
                output=detect_sources(image,mask=mask,background_model=model,**options)
                cases.append({"name":f"{dtype}-precision32-{precision32}-{mode}-{variant}","dtype":dtype,"height":64,"width":65,"image":image.ravel().tolist(),"mask":mask.ravel().tolist(),"options":options,"expected":output.as_dict(),"background_model":None if model is None else {"height":model[0].shape[0],"width":model[0].shape[1],"background":model[0].ravel().tolist(),"noise":model[1].ravel().tolist()}})
    return {"source_sha256":hashlib.sha256((ROOT/"src/rst19/detection.py").read_bytes()).hexdigest(),"cases":cases}


if __name__=="__main__":
    target=ROOT/"backend/tests/fixtures/detection.json";target.write_text(json.dumps(generate(),ensure_ascii=False,allow_nan=False,indent=2),encoding="utf-8");print(target)
