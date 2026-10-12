"""Freeze catalog validation/propagation and complete matching contracts."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from dataclasses import asdict
import numpy as np
from rst19.catalog import CatalogSource,load_catalog_csv
import csv
import io
import tempfile
from rst19.detection import Detection
from rst19.matching import match_detections
from rst19.wcs import TangentPlaneWCS,AffineWCSCalibration

if __name__=="__main__":
    rng=np.random.default_rng(197);propagation=[];matches=[];invalid=[]
    base={"source_id":"高自行 α","ra_deg":359.99,"dec_deg":-35.,"pmra_mas_yr":100000.,"pmdec_mas_yr":50000.,"magnitude":16.,"magnitude_error":.02,"photometric_system":"Gaia","photometric_band":"G","magnitude_source":"  measured G  ","extinction_mag":.1,"extinction_band":"gaia dr3 g","extinction_system":"Gaia Vega","extinction_source":"  survey  ","distance_pc":150.,"distance_lower_pc":140.,"distance_upper_pc":160.,"distance_source":"catalog model","mg_gspphot":4.,"mg_gspphot_lower":3.,"mg_gspphot_upper":5.,"mg_gspphot_source":"Gaia model","duplicated_source":False,"visibility_periods_used":12,"ruwe":1.1,"phot_variable_flag":"NOT_AVAILABLE"}
    for dec in [-90.,-89.9999,-35.,0.,35.,89.9999,90.]:
        source={**base,"dec_deg":dec,"pmdec_mas_yr":0.}
        for epoch in [None,2016.,2010.,2030.]:
            s=CatalogSource(**source);propagation.append({"source":source,"epoch":epoch,"expected":s.at_epoch(epoch).as_dict(),"extinction_compatibility":s.extinction_compatibility()})
    for field,value in [("source_id",""),("ra_deg",361.),("dec_deg",91.),("magnitude_error",-.1),("distance_pc",0.),("distance_lower_pc",200.),("mg_gspphot",9.),("phot_variable_flag"," "),("distance_source",""),("mg_gspphot_source"," "),("extinction_band","V"),("visibility_periods_used",True),("visibility_periods_used",-1),("duplicated_source",1)]:
        request={**base,field:value}
        try:CatalogSource(**request)
        except ValueError:invalid.append(request)
        else:raise AssertionError((field,value))
    for field in ["catalog_name","photometric_system","photometric_band","magnitude_source","extinction_band","extinction_system","extinction_source"]:
        source={**base,field:None};s=CatalogSource(**source)
        propagation.append({"source":source,"epoch":2026.,"expected":s.at_epoch(2026.).as_dict(),"extinction_compatibility":s.extinction_compatibility()})
    for kind in ["tan","affine"]:
        tan_spec={"center_ra_deg":359.999,"center_dec_deg":23.,"pixel_scale_arcsec":2.3,"crpix_x":128.,"crpix_y":120.,"rotation_deg":31.,"parity":-1}
        tan=TangentPlaneWCS(**tan_spec)
        affine_spec={"center_ra_deg":359.999,"center_dec_deg":23.,"matrix_px_per_arcsec":[[-.4,.035],[.02,.5]],"offset_px":[128.,120.]}
        if kind=="tan":wcs=tan;spec=tan_spec
        else:
            # Metadata does not participate in the affine forward projection.
            wcs=AffineWCSCalibration(**affine_spec,plate_scale_arcsec_per_pixel=2.3,rotation_deg=0.,parity=-1,anisotropy_ratio=1.1,matched_count=0,inlier_count=0,rms_residual_px=0.,max_residual_px=0.,all_rms_residual_px=0.,all_max_residual_px=0.,condition_number=1.,inlier_source_ids=())
            spec=affine_spec
        for count in [0,1,3,20,70]:
            catalog=[];detections=[]
            for i in range(count):
                px,py=rng.uniform(0,256,2);ra,dec=tan.pixel_to_world(px,py)
                source=CatalogSource(source_id=f"星-{i}",ra_deg=ra,dec_deg=dec,pmra_mas_yr=float(rng.normal(0,200)),pmdec_mas_yr=float(rng.normal(0,200)),magnitude=15.+i/100,magnitude_error=.02,color=.5,color_name="BP-RP",photometric_system="Gaia",photometric_band="G")
                catalog.append(source)
                x,y=wcs.world_to_pixel(ra,dec);dx,dy=rng.normal(0,.2,2)
                detections.append(Detection(detection_id=(-i if i%2 else count-i),x=float(x+dx),y=float(y+dy),peak=150.,flux=500.,background=100.,noise=2.,snr=25.,fwhm=3.,flags=()))
            # Unprojectable source, two detections competing for one source,
            # duplicates, and unmatched points preserve ordering contracts.
            catalog.append(CatalogSource(source_id="behind",ra_deg=180.,dec_deg=-23.))
            if detections:
                d=detections[0];detections.append(Detection(detection_id=999,x=d.x+.1,y=d.y+.1,peak=150.,flux=500.,background=100.,noise=2.,snr=25.,fwhm=3.,flags=()))
            detections.append(Detection(detection_id=777,x=2000.,y=-1000.,peak=150.,flux=500.,background=100.,noise=2.,snr=25.,fwhm=3.,flags=()))
            for epoch in [None,2027.]:
                for mode in ["global","greedy"]:
                    expected=match_detections(detections,catalog,wcs,radius_px=3.,epoch=epoch,assignment_mode=mode).as_dict()
                    matches.append({"detections":[d.as_dict() for d in detections],"catalog":[s.as_dict() for s in catalog],"wcs":spec,"radius":3.,"epoch":epoch,"mode":mode,"expected":expected})
    csv_cases=[]
    def capture_csv(text):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"catalog.csv";path.write_text(text,encoding="utf-8",newline="")
            try:expected=[s.as_dict() for s in load_catalog_csv(path)];error=False
            except ValueError:expected=None;error=True
        csv_cases.append({"csv":text,"expected":expected,"error":error})
    capture_csv("source_id,ra_deg,dec_deg\n")
    capture_csv("\ufeffSOURCE_ID, RA ,DEC,phot_g_mean_mag,bp_rp,pmra,pmdec,ref_epoch\r\nα,359.9,20,15.2,.8,120,-30,2016\r\n")
    capture_csv('source_id,ra_deg,dec_deg,magnitude,magnitude_source\n"id, quoted",10,20,15,"provenance\nwith newline"\n\n')
    capture_csv('source_id,ra_deg,dec_deg,optional\nid,10,20\nid2,20,30,a,extra\n')
    capture_csv('source_id,RA,ra,dec_deg\nid,10,20,30\n')
    capture_csv('source_id,ra_deg,ra_deg,dec_deg\nid,10,20,30\n')
    row={"source_id":"full","ra_deg":"10","dec_deg":"20","phot_g_mean_mag":"16","magnitude":"16","magnitude_error":".02","bp_rp":".7","pmra":"120","pmdec":"-50","ref_epoch":"2015","parallax":"5","parallax_error":".1","ag_gspphot":".2","extinction_mag":".2","ag_gspphot_error":".01","distance_gspphot":"200","distance_gspphot_lower":"180","distance_gspphot_upper":"230","mg_gspphot":"4.2","mg_gspphot_lower":"4","mg_gspphot_upper":"4.5","phot_g_mean_flux_over_error":"120","phot_bp_rp_excess_factor":"1.1","ruwe":"1.05","duplicated_source":"false","visibility_periods_used":"12.0","phot_variable_flag":"NOT_AVAILABLE"}
    def row_csv(row):
        out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(row));writer.writeheader();writer.writerow(row);return out.getvalue()
    capture_csv(row_csv(row))
    for field in ["extinction_mag","a_band","a_g","ag","ag_gspphot","azero_gspphot","a0_gspphot","a0","azero","a_v","av"]:
        capture_csv(row_csv({"source_id":field,"ra":10,"dec":20,"magnitude":16,field:.2}))
    for field in ["distance_pc","distance_gspphot","distance","r_med_geo","r_med_photogeo","r_med_photogeometric"]:
        capture_csv(row_csv({"source_id":field,"ra":10,"dec":20,field:200}))
    for field in ["mg_gspphot","mg_g","gsp_phot_mg"]:
        capture_csv(row_csv({"source_id":field,"ra":10,"dec":20,field:4.2}))
    for value in ["true","t","1","YES","y","false","f","0","No","n",""]:
        capture_csv(row_csv({"source_id":value or "blank","ra":10,"dec":20,"duplicate_source":value}))
    for field,value in [("magnitude","17"),("extinction_mag",".3"),("a_v",".2"),("extinction_band","V"),("distance_gspphot","0"),("distance_gspphot_lower","500"),("mg_gspphot","10"),("duplicated_source","maybe"),("visibility_periods_used","3.5"),("ra_deg","nan"),("pmra","inf"),("magnitude_error","-.1")]:
        capture_csv(row_csv({**row,field:value}))
    for text in ["","\n","\r\n\r\n","\nsource_id,ra,dec\na,10,20\n","source_id,ra,dec\na,10,20\na,20,30\n","source_id,ra,dec\n,10,20\n","source_id,ra,dec\na,10,\n"]:capture_csv(text)
    source_hashes={p:hashlib.sha256((ROOT/"src/rst19"/p).read_bytes()).hexdigest() for p in ["catalog.py","matching.py","wcs.py"]}
    target=ROOT/"backend/tests/fixtures/matching.json";target.write_text(json.dumps({"source_hashes":source_hashes,"propagation":propagation,"invalid_sources":invalid,"matching":matches,"csv":csv_cases},ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");print(target)
