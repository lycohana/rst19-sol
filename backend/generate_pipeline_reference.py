"""Freeze complete synthetic frame analyses and detection-preserving recalibration."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from rst19.models import FitsFrame
from rst19.catalog import CatalogSource
from rst19.pipeline import analyze_frame,recalibrate_frame_analysis
from rst19.photometry import instrumental_magnitude
from rst19.wcs import TangentPlaneWCS

if __name__=="__main__":
    frames=[];cases=[];rng=np.random.default_rng(199);yy,xx=np.indices((96,97));data=rng.normal(100,2,yy.shape)
    for i,(x,y) in enumerate([(x,y) for y in [20.2,45.1,70.3] for x in [15.2,36.1,57.3,78.2]]):data+=(50.+i*15.)*np.exp(-.5*(((xx-x)/1.3)**2+((yy-y)/1.3)**2))
    wcs_spec={"center_ra_deg":359.99,"center_dec_deg":23.,"pixel_scale_arcsec":2.3,"crpix_x":48.,"crpix_y":48.,"rotation_deg":31.,"parity":-1};wcs=TangentPlaneWCS(**wcs_spec)
    for dtype in ["float64","int16"]:
        image=data.astype(dtype);header={"EXPTIME":2.,"BITPIX":-64 if dtype=="float64" else 16,"NAXIS1":97,"NAXIS2":96,"DATE-OBS":"2026-03-30T16:32:05","IGNORED_HEADER":42}
        frame=FitsFrame(Path(f"synthetic-{dtype}.fits"),header,image,None,0);index=len(frames);frames.append({"path":str(frame.path),"header":header,"height":96,"width":97,"dtype":dtype,"image":image.ravel().tolist()})
        baseline=analyze_frame(frame,background_box_size=32)
        catalog=[]
        for i,d in enumerate(baseline.detection.quality_sources):
            ra,dec=wcs.pixel_to_world(d.x,d.y);color=i/max(1,len(baseline.detection.quality_sources)-1)
            catalog.append(CatalogSource(source_id=f"s{i:03}",ra_deg=ra,dec_deg=dec,magnitude=instrumental_magnitude(d.flux,exposure_s=2.)+24.+.2*color,magnitude_error=.02,color=color,color_name="BP-RP",photometric_system="Gaia Vega",photometric_band="G",parallax_mas=5.,parallax_error_mas=.1,extinction_mag=.2,extinction_error_mag=.01,extinction_band="G",extinction_system="Gaia",extinction_source="synthetic fixture"))
        for mode in ["gaussian","hybrid","ensemble"]:
            for variant in ["no_catalog","catalog_no_fit","fit","zero_point","empty_catalog","fit_no_catalog","fit_band_mismatch","float32_fast"]:
                options={"proposal_mode":mode,"background_box_size":32};selected=None if variant in {"no_catalog","zero_point","fit_no_catalog"} else catalog;selected_wcs=wcs if selected is not None else None
                if variant in {"fit","empty_catalog","fit_no_catalog","fit_band_mismatch","float32_fast"}:options["fit_photometry"]=True
                if variant=="empty_catalog":selected=[]
                if variant=="zero_point":options["zero_point"]=24.
                if variant=="fit_band_mismatch":options["photometric_band"]="V"
                if variant=="float32_fast":options.update(use_float32=True,fast_sequence=True,refine_local_background=False)
                result=analyze_frame(frame,catalog=selected,wcs=selected_wcs,**options)
                recalibrated=None
                if variant in {"fit","catalog_no_fit","fit_band_mismatch"}:
                    recalibrated=recalibrate_frame_analysis(result,catalog,wcs).as_dict()
                cases.append({"frame":index,"name":f"{dtype}-{mode}-{variant}","catalog":None if selected is None else [s.as_dict() for s in selected],"wcs":None if selected_wcs is None else wcs_spec,"options":options,"expected":result.as_dict(),"recalibration":None if recalibrated is None else {"catalog":[s.as_dict() for s in catalog],"wcs":wcs_spec,"options":{},"expected":recalibrated}})
    hashes={p:hashlib.sha256((ROOT/"src/rst19"/p).read_bytes()).hexdigest() for p in ["pipeline.py","photometry.py","detection.py","matching.py","fits.py"]}
    target=ROOT/"backend/tests/fixtures/pipeline.json";target.write_text(json.dumps({"source_hashes":hashes,"frames":frames,"cases":cases,"scope":"synthetic images and catalogs derived from baseline measurements; no physical identity claim"},ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");print(target,len(cases))
