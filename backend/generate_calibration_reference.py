"""Freeze weighted photometric fits, sample roles and application gates."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from dataclasses import replace
import numpy as np
from rst19.catalog import CatalogSource
from rst19.detection import Detection
from rst19.matching import CatalogMatch
from rst19.photometry import fit_photometric_calibration,calibrated_magnitude_error

if __name__=="__main__":
    rng=np.random.default_rng(198);cases=[]
    for count in [0,3,6,12,30]:
        for order in [0,1,2]:
            for variant in ["normal","outliers","holdout_bias","missing_color","metadata_mismatch","color_mismatch","duplicate","catalog_quality","edge","rank","scatter","no_holdout"]:
                catalog=[];detections=[];matches=[]
                for i in range(count):
                    color=float(rng.uniform(-.2,2.));instrumental=float(rng.uniform(-11.,-8.));flux=2.*10**(-.4*instrumental)
                    magnitude=instrumental+24.+(.2*color if order else 0)+(.04*color*color if order==2 else 0)+float(rng.normal(0,.01 if variant!="scatter" else .7))
                    if variant=="rank":color=.4
                    if variant=="outliers" and i%7==3:magnitude+=2.
                    if variant=="holdout_bias" and i in [0,count-1]:magnitude+=1.
                    source=CatalogSource(source_id=f"星{i:03}",ra_deg=10.+i/100,dec_deg=20.,magnitude=magnitude,magnitude_error=.02 if i%2 else 0.,color=color,color_name="BP-RP",photometric_system="Gaia Vega",photometric_band="G")
                    detection=Detection(detection_id=i+1,x=float(i),y=0.,peak=150.,flux=flux,background=100.,noise=2.,snr=100.,fwhm=3.,flags=(),flux_snr=80.+i)
                    if i==0:
                        if variant=="missing_color":source=replace(source,color=None)
                        if variant=="metadata_mismatch":source=replace(source,photometric_band="V")
                        if variant=="color_mismatch":source=replace(source,color_name="V-I")
                        if variant=="catalog_quality":source=replace(source,duplicated_source=True,phot_variable_flag="VARIABLE",ruwe=2.,visibility_periods_used=3,phot_g_mean_flux_over_error=5.)
                        if variant=="edge":detection=replace(detection,flags=("EDGE",),quality_passed=False)
                    catalog.append(source);detections.append(detection)
                    matches.append(CatalogMatch(detection_id=i+1,source_id=source.source_id,detection_x=float(i),detection_y=0.,predicted_x=float(i)+.1,predicted_y=0.,residual_px=.1,catalog_magnitude=magnitude,catalog_magnitude_error=.02))
                if variant=="duplicate" and matches:matches.append(matches[0])
                options={"exposure_s":2.,"photometric_system":"Gaia Vega","photometric_band":"G","color_name":"BP-RP","color_order":order,"min_calibrators":6,"validation_fraction":0. if variant=="no_holdout" else .2}
                result=fit_photometric_calibration(matches,detections,catalog,**options)
                apply=[{"instrumental":-10.,"color":c,"expected":result.apply(-10.,color=c)} for c in [None,-1.,0.,1.,3.]]
                errors=[{"instrumental_error":e,"color_error":c,"expected":calibrated_magnitude_error(e,calibration=result,color_error=c)} for e,c in [(None,None),(.02,None),(.02,.1),(-.02,.1)]]
                cases.append({"matches":[m.as_dict() for m in matches],"detections":[d.as_dict() for d in detections],"catalog":[s.as_dict() for s in catalog],"options":options,"expected":result.as_dict(),"application":apply,"errors":errors,"name":f"n{count}-order{order}-{variant}"})
    invalid=[]
    for field,value in [("exposure_s",0.),("min_snr",0.),("color_order",3),("min_calibrators",1),("sigma_clip",0.),("max_iterations",0),("validation_fraction",.5),("max_fit_rms_mag",0.),("min_validation_count",True),("min_catalog_flux_snr",0.),("max_catalog_ruwe",-1.),("min_catalog_visibility_periods",-1),("reject_duplicated_sources",1)]:
        options={"photometric_system":"Gaia Vega","photometric_band":"G",field:value}
        try:fit_photometric_calibration([],[],[],**options)
        except ValueError:invalid.append(options)
        else:raise AssertionError((field,value))
    for options in [{},{"photometric_system":"Gaia","photometric_band":"unknown"},{"photometric_system":None,"photometric_band":"G"},{"photometric_system":"Gaia Vega","photometric_band":None}]:
        result=fit_photometric_calibration([],[],[],**options);cases.append({"matches":[],"detections":[],"catalog":[],"options":options,"expected":result.as_dict(),"application":[],"errors":[],"name":"unknown metadata"})
    target=ROOT/"backend/tests/fixtures/calibration.json";target.write_text(json.dumps({"source_sha256":hashlib.sha256((ROOT/"src/rst19/photometry.py").read_bytes()).hexdigest(),"cases":cases,"invalid_options":invalid},ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");print(target,len(cases))
