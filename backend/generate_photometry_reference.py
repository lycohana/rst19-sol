"""Freeze arithmetic, absolute-magnitude gates and their derived evidence."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from rst19 import photometry as p
from rst19.catalog import CatalogSource

if __name__=="__main__":
    cases=[]
    def capture(kind,kwargs,source=None):
        try:
            if kind=="catalog":result=p.absolute_magnitude_from_catalog(CatalogSource(**source),**kwargs)
            else:result=getattr(p,kind)(**kwargs)
            expected=result.as_dict() if hasattr(result,"as_dict") else result;error=False
        except ValueError:expected=None;error=True
        cases.append({"kind":kind,"parameters":kwargs,"source":source,"expected":expected,"error":error})
    for flux,exposure in itertools.product([-5.,0.,1.,100.,1e-300],[0.,-1.,1.,.02,1e100]):capture("instrumental_magnitude",{"flux":flux,"exposure_s":exposure})
    for color,coefficient in itertools.product([None,-1.,0.,.6],[0.,.2]):capture("calibrated_apparent_magnitude",{"instrumental":-12.,"zero_point":25.,"color":color,"color_coefficient":coefficient,"extinction_mag":.1})
    for distance in [-1.,0.,.1,10.,100.,1500.]:capture("absolute_magnitude_from_apparent",{"apparent_magnitude":15.,"distance_pc":distance,"extinction_mag":.2})
    for parallax in [-5.,0.,.1,1.,100.]:capture("absolute_magnitude_from_parallax",{"apparent_magnitude":15.,"parallax_mas":parallax,"extinction_mag":.2})
    for parallax,error,extinction in itertools.product([-1.,0.,1.,5.],[None,0.,.1,.2,1.,5.,-.1],[None,-.1,0.,.3]):
        capture("absolute_magnitude_estimate_from_parallax",{"apparent_magnitude":15.,"parallax_mas":parallax,"parallax_error_mas":error,"extinction_mag":extinction,"apparent_magnitude_error":.02,"extinction_error_mag":.01,"extinction_system":"Gaia Vega","extinction_band":"G","extinction_source":"survey"})
    base={"apparent_magnitude":15.,"parallax_mas":5.,"parallax_error_mas":.1,"extinction_mag":.2,"apparent_magnitude_error":.02,"extinction_error_mag":.01,"extinction_system":"Gaia Vega","extinction_band":"G"}
    for field,values in [("extinction_system",[None,"unknown"," N/A ","?","Gaia"]),("extinction_band",[None,"unknown","—","G"]),("apparent_magnitude_error",[None,0.,-.1]),("extinction_error_mag",[None,0.,-.1]),("parallax_zero_point_mas",[0.,-.02,10.]),("max_fractional_parallax_error",[0.,-1.,.01,1.])]:
        for value in values:capture("absolute_magnitude_estimate_from_parallax",{**base,field:value})
    for lower,upper in [(None,None),(None,120.),(80.,None),(80.,120.),(100.,100.),(120.,80.),(110.,120.),(80.,90.),(0.,120.),(-1.,120.)]:
        for source,extinction in itertools.product([None,"unknown","GSP-Phot"],[None,-.1,0.,.3]):
            capture("absolute_magnitude_estimate_from_distance",{"apparent_magnitude":15.,"distance_pc":100.,"distance_lower_pc":lower,"distance_upper_pc":upper,"distance_source":source,"extinction_mag":extinction,"apparent_magnitude_error":.02,"extinction_error_mag":.01,"extinction_system":"Gaia","extinction_band":"G"})
    base_distance={"apparent_magnitude":15.,"distance_pc":100.,"distance_lower_pc":80.,"distance_upper_pc":120.,"distance_source":"GSP-Phot","extinction_mag":.2,"apparent_magnitude_error":.02,"extinction_error_mag":.01,"extinction_system":"Gaia","extinction_band":"G"}
    for field,values in [("distance_pc",[-1.,0.,150.]),("extinction_system",[None,"unknown","Gaia"]),("extinction_band",[None,"unknown","G"]),("apparent_magnitude_error",[None,0.,-.1]),("extinction_error_mag",[None,0.,-.1])]:
        for value in values:capture("absolute_magnitude_estimate_from_distance",{**base_distance,field:value})
    base_source={"source_id":"α","ra_deg":10.,"dec_deg":20.,"magnitude":15.,"magnitude_error":.02,"photometric_system":"Gaia Vega","photometric_band":"G","extinction_mag":.2,"extinction_error_mag":.01,"extinction_band":"G","extinction_system":"Gaia","extinction_source":"survey"}
    for parallax,error,distance,source in itertools.product([None,-1.,5.],[None,.1,5.],[None,100.],[None,"GSP-Phot"]):
        row={**base_source,"parallax_mas":parallax,"parallax_error_mas":error,"distance_pc":distance,"distance_lower_pc":80. if distance else None,"distance_upper_pc":120. if distance else None,"distance_source":source}
        capture("catalog",{},row)
        capture("catalog",{"apparent_magnitude":16.,"apparent_magnitude_error":None,"required_photometric_system":"Gaia Vega","required_photometric_band":"G"},row)
    for field,values in [("magnitude",[None,15.]),("extinction_mag",[None,-.1]),("extinction_band",["unknown","G"]),("photometric_system",[None,"unknown","Gaia Vega"])]:
        for value in values:capture("catalog",{},{**base_source,"parallax_mas":5.,"parallax_error_mas":.1,field:value})
    for system,band in [(None,"G"),("Gaia Vega",None),("Gaia Vega","V"),(" unknown ","G"),(" gaia\tvega ","g")]:capture("catalog",{"required_photometric_system":system,"required_photometric_band":band},{**base_source,"parallax_mas":5.,"parallax_error_mas":.1})
    capture("absolute_magnitude_estimate_from_parallax",{**base,"parallax_error_mas":0.,"apparent_magnitude_error":0.,"extinction_error_mag":0.})
    capture("absolute_magnitude_estimate_from_distance",{**base_distance,"distance_lower_pc":100.,"distance_upper_pc":100.,"apparent_magnitude_error":0.,"extinction_error_mag":0.})
    for band,system in [("V","Johnson"),("unknown","unknown"),("A0(541.4 nm)","monochromatic")]:
        source=CatalogSource(**{**base_source,"parallax_mas":5.,"parallax_error_mas":.1})
        # Simulate old cached objects that predate constructor conflict gates.
        object.__setattr__(source,"extinction_band",band);object.__setattr__(source,"extinction_system",system)
        expected=p.absolute_magnitude_from_catalog(source).as_dict()
        cases.append({"kind":"catalog","parameters":{},"source":source.as_dict(),"expected":expected,"error":False,"scope":"legacy cached extinction semantics"})
    labels=[None,"", " unknown ","N/A","—","Gaia\tVega","Straße","STRASSE","Σ","ς","\u001c unknown \u001f"]
    normalisations=[{"input":v,"expected":p._normalise_photometric_label(v)} for v in labels]
    path=ROOT/"backend/tests/fixtures/photometry.json";path.write_text(json.dumps({"source_sha256":hashlib.sha256((ROOT/"src/rst19/photometry.py").read_bytes()).hexdigest(),"cases":cases,"labels":normalisations},ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");print(path,len(cases))
