"""Source and PSF parity cases; Python is used only as a development oracle."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from rst19.detection import _source_from_peak,_pair_psf_evidence,_guard_unresolved_gaussian_pairs


def generate():
    rng=np.random.default_rng(210)
    cases=[]
    for dtype,mode in [(d,m) for d in ["float64","float32"] for m in ["point","weak","edge","partial_zero","strict_zero","masked_center","nan_neighbor","saturated","negative_overflow","code_pattern","unresolved","line","empty","gain_noise","no_refinement","tiny"]]:
        h,w=(12,12) if mode=="tiny" else (48,49)
        xp,yp=(2,2) if mode=="edge" else (w//2,h//2)
        yy,xx=np.indices((h,w));image=rng.normal(100,2,(h,w));mask=np.zeros((h,w),dtype=bool);line=None
        amplitude=4 if mode=="weak" else 100
        image+=amplitude*np.exp(-((xx-xp-.15)**2+(yy-yp+.2)**2)/(2*1.3**2))
        image=image.astype(dtype)
        options={"aperture_radius":4,"saturation_level":None,"min_flux_snr":5.0,"min_fwhm":0.8,"max_fwhm":12.0,"max_ellipticity":0.65,"min_sharpness":0.005,"max_sharpness":0.85,"min_footprint_pixels":2,"min_psf_support_pixels":3,"gain_e_per_adu":None,"read_noise_adu":0.0,"refine_local_background":True,"psf_fwhm":3.0,"dog_blend_radius_px":7.5,"allow_partial_zero_mask":False,"negative_overflow_limit":None,"repeated_high_code_values":[]}
        methods=("gaussian",);scales=(3.0,);nearest=None
        if mode in ["partial_zero","strict_zero"]:
            image[yp,xp+2]=0;mask[yp,xp+2]=True;options["allow_partial_zero_mask"]=mode=="partial_zero"
        if mode=="masked_center":mask[yp,xp]=True
        if mode=="nan_neighbor":image[yp,xp+2]=np.nan
        if mode=="saturated":image[yp,xp+2]=4000;options["saturation_level"]=3000.0;mask[yp,xp+2]=True
        if mode=="negative_overflow":image[yp,xp+2]=-32000;options["negative_overflow_limit"]=-29490.3
        if mode=="code_pattern":image[yp,xp]=3991;image[yp,xp+2]=-2000;options["repeated_high_code_values"]=[3991,3992]
        if mode=="unresolved":methods=("dog_narrow",);scales=(2.5,);nearest=5.0
        if mode=="line":line=np.zeros((h,w),dtype=bool);line[yp,xp+2]=True
        if mode=="empty":image[:]=100.0
        if mode=="gain_noise":options["gain_e_per_adu"]=1.7;options["read_noise_adu"]=3.2
        if mode=="no_refinement":options["refine_local_background"]=False
        background=np.full((h,w),100.0);noise=np.full((h,w),2.0)
        expected=_source_from_peak(image,mask,line,xp,yp,background,noise,4,options["saturation_level"],12.0,5.0,.8,12.0,.65,.005,.85,2,3,options["gain_e_per_adu"],options["read_noise_adu"],options["refine_local_background"],3.0,methods,scales,12.0,nearest,7.5,allow_partial_zero_mask=options["allow_partial_zero_mask"],negative_overflow_limit=options["negative_overflow_limit"],repeated_high_code_values=tuple(options["repeated_high_code_values"]))
        sanitized=np.nan_to_num(image,nan=0).ravel().tolist()
        nonfinite=np.nonzero(~np.isfinite(image.ravel()))[0].tolist()
        cases.append({"name":f"{dtype}-{mode}","dtype":dtype,"height":h,"width":w,"image":sanitized,"nonfinite":nonfinite,"mask":mask.ravel().tolist(),"line_mask":line.ravel().tolist() if line is not None else None,"background":background.ravel().tolist(),"noise":noise.ravel().tolist(),"peak":[xp,yp],"options":options,"proposal":{"x":xp,"y":yp,"score":12.0,"methods":methods,"scales":scales,"nearest_gaussian_px":nearest},"filter_snr":12.0,"expected":expected.as_dict()})
    pairs=[]
    for separation in [0.0,.001,2.0,4.0,7.0]:
        for secondary_amplitude in [0.0,45.0]:
            yy,xx=np.indices((48,49));p=(22.3,24.1);s=(p[0]+separation,p[1]+.15 if separation else p[1]);sigma=3/2.35482
            image=rng.normal(100,2,(48,49))+.04*xx-.03*yy
            image+=150*np.exp(-.5*(((xx-p[0])/sigma)**2+((yy-p[1])/sigma)**2))
            image+=secondary_amplitude*np.exp(-.5*(((xx-s[0])/sigma)**2+((yy-s[1])/sigma)**2))
            mask=rng.random(image.shape)<.01
            delta,snr=_pair_psf_evidence(image,mask,p,s,psf_fwhm=3.0)
            pairs.append({"height":48,"width":49,"image":image.ravel().tolist(),"mask":mask.ravel().tolist(),"primary":p,"secondary":s,"fwhm":3.0,"expected":[delta,snr]})
    # Both a real pair and a repeated shoulder exercise the quality guard.
    guards=[]
    for separation in [2,4]:
        yy,xx=np.indices((48,49));image=rng.normal(100,2,(48,49));sigma=3/2.35482
        image+=130*np.exp(-.5*(((xx-22)/sigma)**2+((yy-24)/sigma)**2))
        image+=30*np.exp(-.5*(((xx-22-separation)/sigma)**2+((yy-24)/sigma)**2))
        mask=np.zeros(image.shape,dtype=bool);sources=[]
        for x in [22,22+separation]:
            sources.append(_source_from_peak(image,mask,None,x,24,np.full(image.shape,100.),np.full(image.shape,2.),4,None,12.,1.,.1,20.,1.,0.,1.,1,1,None,0.,False,3.,("gaussian",),(3.,),12.,None,7.5))
        result,tested,guarded,radius=_guard_unresolved_gaussian_pairs(sources,image,mask,psf_fwhm=3.,delta_bic_min=10.,component_snr_min=5.)
        guards.append({"height":48,"width":49,"image":image.ravel().tolist(),"mask":mask.ravel().tolist(),"sources":[s.as_dict() for s in sources],"expected":[s.as_dict() for s in result],"tested":tested,"guarded":guarded,"radius":radius})
    return {"source_sha256":hashlib.sha256((ROOT/"src/rst19/detection.py").read_bytes()).hexdigest(),"measurements":cases,"pairs":pairs,"guards":guards}


if __name__=="__main__":
    target=ROOT/"backend/tests/fixtures/measurement.json"
    target.write_text(json.dumps(generate(),ensure_ascii=False,allow_nan=False,indent=2),encoding="utf-8")
    print(target)
