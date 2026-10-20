"""Reaggregate retained paired evidence; export the same raw crop for both views."""
from pathlib import Path
import csv, hashlib, importlib.util, json, math, statistics
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
records=[]
def bind(p):
    records.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
def load(rel):
    p=ROOT/rel;bind(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(rel):
    p=ROOT/rel;bind(p);return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
geo=load('tmp/paper/motion/trail_geometry.json')
cat=load('tmp/flag_on/02-rst19-sources/trail_catalog.json')
request=load('tmp/flag_on/02-rst19-sources/request.json')
on=rows('tmp/flag_on/02-rst19-sources/source_catalog.csv')
off=rows('tmp/flag_off/02-rst19-sources/source_catalog.csv')
assert (ROOT/'tmp/flag_on/02-rst19-sources/source_catalog.csv').read_bytes()==(ROOT/'tmp/flag_off/02-rst19-sources/source_catalog.csv').read_bytes()
a,b=geo['ridge_fit']['x_equals_a_plus_b_y']
selected=[r for r in off if 0<=float(r['y'])<=600 and abs(float(r['x'])-a-b*float(r['y']))/math.sqrt(1+b*b)<3]
assert len(selected)==69 and all(r['quality_passed'].lower()=='false' and 'LINE_ARTIFACT' in r['flags'] for r in selected)
baseline=[{'id':r['detection_id'],'x':float(r['x']),'y':float(r['y']),'quality_passed':False,'flags':r['flags']} for r in selected]
white=load('tmp/glrt_val/h0_white.json')
fa=white['summary']['fa_table'][0]
def count(m,u):
    data=sorted((float(k),v) for k,v in m['peak_counts'].items())
    for (x,y),(xx,yy) in zip(data,data[1:]):
        if x<=u<=xx:
            return float(yy) if min(y,yy)<=0 or y==yy else math.exp(math.log(y)+(u-x)/(xx-x)*(math.log(yy)-math.log(y)))
    raise ValueError('threshold outside saved grid')
policies=[]
for name,key in [('单点口径','u_standard_pc'),('按核族标定','u_empirical')]:
    vals=[count(m,fa[key]) for m in white['metrics']]
    mean=statistics.mean(vals);sd=statistics.stdev(vals)
    assert abs(mean-fa['count_'+key]['mean'])<1e-12
    assert abs(sd-fa['count_'+key]['std'])<1e-12
    policies.append({'label':name,'threshold':fa[key],'counts':vals,'mean':mean,'std':sd})
fits=ROOT/'doc/00-项目资料/原始数据/20260330163205413_9901.fits';bind(fits)
reader=ROOT/'tools/glrt_validate/ref_glrt.py';bind(reader)
spec=importlib.util.spec_from_file_location('fits_reader',reader);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
image,_=module.read_fits(fits)
# Horizontal display transposes the real crop: display x = original y,
# display y = original x - 1440. Both comparisons share this exact file.
crop=image[:620,1440:1580].T.astype(float)
lo,hi=20.,float(np.percentile(crop,99.7));scaled=np.arcsinh(10*np.clip((crop-lo)/(hi-lo),0,1))/np.arcsinh(10)
im=Image.fromarray(np.uint8(np.stack([scaled*.85,scaled*.94,scaled],axis=-1)*255))
im.save(HERE/'figs/jury-event-A.png')
bind(HERE/'figs/jury-event-A.png')
cal=load('doc/05-论文答辩/latex-论文/evidence/computed_results.json')
scope={'case':'real first frame A, geometric band selected using historical independent pixel profile',
       'rule':{'y_min':0,'y_max':600,'perpendicular_distance_lt_px':3,'x_equals_a_plus_b_y':[a,b]},
       'display':{'original_x':[1440,1580],'original_y':[0,620],'transpose':True,'stretch':'asinh, identical before/after','lo_DN':lo,'hi_DN':hi},
       'baseline':baseline,'baseline_n':len(baseline),'quality_n':0,'production_trail':cat['trails'][0],
       'reference_geometry':{'visible_length_px':geo['length_px_in_frame'],'image_axis_angle_deg':90+geo['position_angle']['image_frame_angle_of_axis_from_plus_y_deg']},
       'direction_absolute_difference_deg':abs(cat['trails'][0]['pa_deg']-(90+geo['position_angle']['image_frame_angle_of_axis_from_plus_y_deg'])),
       'scope_limit':'Single observed case. Geometric reference is an independent pixel estimate, not external physical truth. No whole-frame recall or identity claim.'}
data={'date':'2026-10-20','inputs':records,'real_case':scope,
 'h0':{'n_frames':len(white['metrics']),'size':[4096,4096],'family_size':white['summary']['family_size'],
       'policies':policies,'target_false_alarms_per_frame':1,
       'count_method':'Per-frame log-linear interpolation on saved 0.1 response threshold grid, exactly matching h0_calibration.py::count_above',
       'scope':'Calibration self-check on the same 20 synthetic white-noise frames used for the tail fit; no held-out or production FDR claim'},
 'on_off':{'source_catalog_byte_identical':True,'source_n':len(on),'quality_n':sum(r['quality_passed'].lower()=='true' for r in on),
           'on_files':len(list((ROOT/'tmp/flag_on/02-rst19-sources').iterdir())),'off_files':len(list((ROOT/'tmp/flag_off/02-rst19-sources').iterdir())),
           'timings_on_run':cat['timings_s'],'trail_records':cat['n_trails'],'production_psf_fwhm':cat['psf_fwhm']},
 'photometry':cal['photometry'],'photometry_source_n':cal['source_count'],
 'editorial':{'main_slides':12,'backup_slides':8,'target_seconds':415,'timing_is':'planned allocation, not measured rehearsal'}}
(HERE/'evidence/jury-evidence.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'baseline_case_candidates':len(selected),'case_quality':0,'h0_means':[p['mean'] for p in policies],
                  'production_source_n':len(on),'reference_angle_difference_deg':scope['direction_absolute_difference_deg']},ensure_ascii=False))
