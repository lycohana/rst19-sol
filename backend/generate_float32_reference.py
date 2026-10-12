"""Freeze dtype-specific reduction, convolution and background stages."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from rst19.detection import sigma_clipped_stats,local_background_rms,_masked_gaussian_response,_masked_starlet_smooth,_standardize_response

if __name__=="__main__":
    rng=np.random.default_rng(32);stats=[]
    for n in [1,2,7,8,9,127,128,129,4096,8191,8192,8193,10001,16384,16385]:
        values=rng.normal(1234,15,n).astype(np.float32);mask=rng.random(n)<.03;mask[0]=False
        center,noise=sigma_clipped_stats(values,mask=mask)
        stats.append({"values":values.tolist(),"mask":mask.tolist(),"sum":float(values.sum()),"median":float(np.median(values)),"std":float(np.std(values)),"center":center,"noise":noise})
    stages=[]
    for density in [.003,.15,0.0]:
        image=rng.normal(100,2,(33,35)).astype(np.float32);yy,xx=np.indices(image.shape);image+=(100*np.exp(-.5*(((xx-15.3)/1.3)**2+((yy-17.1)/1.3)**2))).astype(np.float32);valid=rng.random(image.shape)>=density;mask=~valid
        sigma=1.3;box_size=16
        if density==0.0:
            local_rng=np.random.default_rng(132);yy,xx=np.indices((96,97));image=local_rng.normal(100,2,yy.shape)
            for x,y,amp in [(22.2,30.1,100),(26.3,30.2,35),(70.2,60.5,8)]:image+=amp*np.exp(-.5*(((xx-x)/1.3)**2+((yy-y)/1.3)**2))
            image=image.astype(np.float32);mask=np.zeros(image.shape,dtype=bool);mask[0,:16]=True;valid=~mask;sigma=3/2.35482;box_size=32
        global_stats=sigma_clipped_stats(image,mask=mask)
        background,noise=local_background_rms(image,mask,box_size=box_size,iterations=2,fallback=global_stats)
        residual=np.where(valid,image-background,0.)
        for fast in [False,True]:
            gaussian,mode=_masked_gaussian_response(residual,valid,sigma=sigma,calculation_dtype=np.float32,fast_sequence=fast)
            starlet,s_mode=_masked_starlet_smooth(residual,valid,step=2,calculation_dtype=np.float32,fast_sequence=fast)
            snr,center,scale=_standardize_response(gaussian,effective_mask=mask,noise_map=noise,valid=valid,sample_limit=1_000_000)
            selected=gaussian.ravel()[valid.ravel()]
            for iteration in range(5):
                mean=np.mean(selected);squares=(selected-mean)**2
                diagnostic_center=float(np.median(selected));diagnostic_mad=float(np.median(np.abs(selected-diagnostic_center)))
                stats.append({"values":selected.tolist(),"mask":np.zeros(selected.size,dtype=bool).tolist(),"sum":float(selected.sum()),"median":diagnostic_center,"std":float(np.std(selected)),"center":sigma_clipped_stats(selected)[0],"noise":sigma_clipped_stats(selected)[1],"mean":float(mean),"square_sum":float(squares.sum()),"iteration":iteration})
                if iteration<4:selected=selected[np.abs(selected-diagnostic_center)<=3*(1.4826*diagnostic_mad)]
            stages.append({"shape":list(image.shape),"sigma":sigma,"box_size":box_size,"global_stats":global_stats,"image":image.ravel().tolist(),"valid":valid.ravel().tolist(),"background":background.ravel().tolist(),"noise":noise.ravel().tolist(),"residual":residual.ravel().tolist(),"fast":fast,"gaussian":gaussian.ravel().tolist(),"gaussian_mode":mode,"starlet":starlet.ravel().tolist(),"starlet_mode":s_mode,"snr":snr.ravel().tolist(),"response_center":center,"response_noise":scale})
    target=ROOT/"backend/tests/fixtures/float32.json";target.write_text(json.dumps({"statistics":stats,"stages":stages},indent=2,allow_nan=False),encoding="utf-8");print(target)
