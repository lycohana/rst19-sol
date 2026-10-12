"""Freeze source rows and faintest ranking with provenance/consistency gates."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from dataclasses import replace
from rst19.catalog import CatalogSource
from rst19.detection import Detection
from rst19.matching import CatalogMatch
from rst19.photometry import PhotometricCalibration,build_source_photometry,find_faintest_source

if __name__=="__main__":
    tables=[];faintest=[]
    calibration=PhotometricCalibration(photometric_system="Gaia Vega",photometric_band="G",color_name="BP-RP",color_order=1,coefficients=(24.,.2),calibrator_count=10,inlier_count=8,validation_count=2,fit_rms_mag=.03,validation_rms_mag=.04,residual_mad_mag=.02,color_min=0.,color_max=2.,status="VALID",max_source_residual_mag=.5,calibration_sample_roles=(("s0","validation"),("s1","training_inlier")))
    for variant in ["normal","empty","no_matches","no_catalog","no_calibration","invalid_calibration","zero_color_order","missing_color","outside_color","metadata_mismatch","color_metadata_mismatch","no_catalog_magnitude","missing_catalog_error","catalog_inconsistent","quality_rejected","zero_flux","snr_fallback","partial_match","model_distance","no_distance","unknown_extinction","negative_extinction","duplicate_flags","match_fallback_metadata"]:
        sources=[];catalog=[];matches=[];c=calibration
        for i,flux in enumerate([1000.,100.,300.]):
            source=Detection(detection_id=i+1,x=float(i*10),y=20.,peak=150.,flux=flux,background=100.,noise=2.,snr=10.+i,fwhm=3.,flags=(),flux_snr=20.+i)
            # m_inst(rate) + zero point + colour term.
            import math
            magnitude=-2.5*math.log10(flux/2.)+24.+.2*(i/2.)
            star=CatalogSource(source_id=f"s{i}",ra_deg=10.+i/100.,dec_deg=20.,magnitude=magnitude,magnitude_error=.02,color=i/2.,color_name="BP-RP",photometric_system="Gaia Vega",photometric_band="G",parallax_mas=5.,parallax_error_mas=.1,extinction_mag=.2,extinction_error_mag=.01,extinction_band="G",extinction_system="Gaia",extinction_source="survey",mg_gspphot=4.,mg_gspphot_lower=3.8,mg_gspphot_upper=4.2,mg_gspphot_source="model G")
            match=CatalogMatch(detection_id=i+1,source_id=star.source_id,detection_x=source.x,detection_y=20.,predicted_x=source.x+.1,predicted_y=20.,residual_px=.1,catalog_magnitude=magnitude,catalog_magnitude_error=.02,catalog_color=star.color,catalog_color_name=star.color_name,photometric_system=star.photometric_system,photometric_band=star.photometric_band)
            if i==1:
                if variant=="missing_color":star=replace(star,color=None);match=replace(match,catalog_color=None)
                if variant=="outside_color":star=replace(star,color=3.)
                if variant=="metadata_mismatch":star=replace(star,photometric_band="V",extinction_mag=None)
                if variant=="color_metadata_mismatch":star=replace(star,color_name="V-I")
                if variant=="no_catalog_magnitude":star=replace(star,magnitude=None);match=replace(match,catalog_magnitude=None)
                if variant=="missing_catalog_error":star=replace(star,magnitude_error=None);match=replace(match,catalog_magnitude_error=None)
                if variant=="catalog_inconsistent":star=replace(star,magnitude=magnitude+2.)
                if variant=="quality_rejected":source=replace(source,quality_passed=False,flags=("EDGE",))
                if variant=="zero_flux":source=replace(source,flux=0.)
                if variant=="snr_fallback":source=replace(source,flux_snr=None)
                if variant=="model_distance":star=replace(star,parallax_mas=None,distance_pc=200.,distance_lower_pc=180.,distance_upper_pc=230.,distance_source="model posterior")
                if variant=="no_distance":star=replace(star,parallax_mas=None)
                if variant=="unknown_extinction":star=replace(star,extinction_band="unknown",extinction_system="unknown")
                if variant=="negative_extinction":star=replace(star,extinction_mag=-.2)
                if variant=="duplicate_flags":source=replace(source,flags=("CUSTOM","CUSTOM"))
                if variant=="match_fallback_metadata":star=replace(star,magnitude=None,magnitude_error=None,color=None,color_name=None,photometric_system=None,photometric_band=None)
            sources.append(source);catalog.append(star);matches.append(match)
        if variant=="empty":sources=[]
        if variant=="no_matches":matches=[]
        if variant=="no_catalog":catalog=[]
        if variant=="partial_match":matches=matches[:1]
        if variant=="no_calibration":c=None
        if variant=="invalid_calibration":c=replace(c,status="PHOTOMETRY_REJECTED_QUALITY_GATE")
        if variant=="zero_color_order":c=replace(c,color_order=0,coefficients=(24.,),color_min=None,color_max=None)
        rows=build_source_photometry(sources,matches=matches,catalog=catalog,exposure_s=2.,photometric_calibration=c)
        tables.append({"name":variant,"sources":[s.as_dict() for s in sources],"matches":[m.as_dict() for m in matches],"catalog":[s.as_dict() for s in catalog],"calibration":c.as_dict() if c else None,"options":{"exposure_s":2.},"expected":[r.as_dict() for r in rows]})
        for scope in ["checked_rows","unchecked_rows","empty_rows"]:
            row_map={r.detection_id:r for r in rows} if scope=="checked_rows" else ({} if scope=="empty_rows" else None)
            colors={s.detection_id:r.color for s,r in zip(sources,rows) if r.color is not None};absolute={r.detection_id:r.absolute_magnitude for r in rows if r.absolute_magnitude is not None}
            options={"exposure_s":2.,"source_colors":colors,"source_absolute_magnitudes":absolute,"source_photometry":row_map}
            result=find_faintest_source(sources,photometric_calibration=c,**options)
            serial={**options,"source_absolute_magnitudes":{k:v.as_dict() for k,v in absolute.items()},"source_photometry":None if row_map is None else {k:v.as_dict() for k,v in row_map.items()}}
            faintest.append({"name":f"{variant}-{scope}","sources":[s.as_dict() for s in sources],"calibration":c.as_dict() if c else None,"options":serial,"expected":result.as_dict() if result else None})
    for zero_point in [None,24.]:
        sources=[Detection(detection_id=i,x=float(i),y=0.,peak=150.,flux=100.,background=100.,noise=2.,snr=20.,fwhm=3.,flags=()) for i in [3,9,1]]
        result=find_faintest_source(sources,zero_point=zero_point)
        faintest.append({"name":"equal flux tie and zero point","sources":[s.as_dict() for s in sources],"calibration":None,"options":{"zero_point":zero_point},"expected":result.as_dict()})
    target=ROOT/"backend/tests/fixtures/source_photometry.json";target.write_text(json.dumps({"source_sha256":hashlib.sha256((ROOT/"src/rst19/photometry.py").read_bytes()).hexdigest(),"tables":tables,"faintest":faintest},ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");print(target,len(tables),len(faintest))
