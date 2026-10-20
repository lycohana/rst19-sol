"""Export real, numerical algorithm traces by compiling current Rust source copies.

Production files are read only. Capture wrappers expose private intermediates in
copies under output/playwright/rst19-deck/science-capture. No algorithm is changed.
"""
from pathlib import Path
import hashlib, json, subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'output/playwright/rst19-deck/science-capture'
OUT.mkdir(parents=True,exist_ok=True)

GLRT_WRAPPER=r'''
pub fn capture(image:&[f64], h:usize,w:usize)->Value {
 let bg=background_median(image,h,w,32);
 let residual:Vec<f64>=image.iter().zip(&bg).map(|(v,b)|v-b).collect();
 let noise=estimate_noise(&residual,h,w);
 let wh=Whitener{hx:whitening_fir(&noise.rho_x,w,9),hy:whitening_fir(&noise.rho_y,h,9)};
 let dw=wh.apply(&residual,h,w);let mut scratch=dw.clone();let sigma=mad_sigma(&mut scratch);
 let spsf=1.93/FWHM_TO_SIGMA;let profiles=profile_tables(spsf,&wh);
 let lib=build_library(&[8.0,16.0,32.0],24,spsf,sigma,(&profiles.plain_y,&profiles.plain_x),(&profiles.grad_y,&profiles.grad_x));
 let cy=32;let cx=32;let idx=cy*w+cx;
 let mut frames=Vec::new();
 let mut push=|k:KernelGrid,len:f64,angle:f64|{
  let kw=whiten_kernel(&k,&wh);let map=mf_z_freq(&dw,h,w,&kw,sigma);
  frames.push(json!({"length":len,"angle":angle,"kernel_rows":kw.rows,"kernel_cols":kw.cols,"kernel":kw.data,"sum_sq":kw.sum_sq,"norm":sigma*kw.sum_sq.sqrt(),"z":map[idx],"map":map}));
 };
 push(lib.point.clone(),0.0,0.0);
 for hyp in &lib.lines{
  let mut k=kernel_from_samples(&hyp.weighted_offsets(0),spsf);
  if hyp.steep{k=transpose_kernel(&k)};
  push(k,hyp.length_px,hyp.angle_deg);
 }
 json!({"width":w,"height":h,"raw":image,"background":bg,"residual":residual,"whitened":dw,"sigma_w":sigma,"target":[cx,cy],"psf_fwhm":1.93,"frames":frames,"scope":"Real crop; exact current glrt.rs frequency reference engine, phase 0; whitened crop noise model; not a rerun of the full-frame Line catalog"})
}
'''
PHOTO_WRAPPER=r'''
pub fn capture(path:&Path,x:f64,y:f64)->Value{
 let image=Image::load(path).unwrap();let cfg=Config::defaults();
 let m=measure(&image,x,y,&cfg).unwrap();let mut pixels=Vec::new();let mut ring=Vec::new();
 let x0=x.floor() as usize-12;let y0=y.floor() as usize-12;
 for j in 0..25{for i in 0..25{
  let xx=x0+i;let yy=y0+j;let v=image.value(xx,yy).unwrap();let dx=xx as f64-x;let dy=yy as f64-y;
  let weight=aperture_coverage(dx,dy,cfg.aperture_radius);
  let distance=dx.hypot(dy);let in_ring=distance>=cfg.annulus_inner&&distance<=cfg.annulus_outer;
  if in_ring{ring.push(v)};
  pixels.push(json!({"x":i,"y":j,"value":v,"weight":weight,"term":weight*(v-m.background),"ring":in_ring}));
 }}
 let mut stages=Vec::new();let mut active=ring.clone();
 for _ in 0..cfg.clip_iterations{
  let med=median(&active).unwrap();let spread=mad_sigma(&active).unwrap_or(0.0);
  let next:Vec<f64>=active.iter().copied().filter(|v|(*v-med).abs()<=cfg.clip_sigma*spread).collect();
  stages.push(json!({"input_n":active.len(),"median":med,"mad_sigma":spread,"kept_n":next.len()}));
  if next.len()==active.len()||spread<=0.0{break};active=next;
 }
 json!({"width":25,"height":25,"origin":[x0,y0],"center":[x-x0 as f64,y-y0 as f64],"radius":cfg.aperture_radius,"annulus_inner":cfg.annulus_inner,"annulus_outer":cfg.annulus_outer,"pixels":pixels,"background":m.background,"noise":m.noise,"flux":m.flux,"flux_error":m.flux_error,"m_inst":-2.5*m.flux.log10(),"clip_stages":stages,"scope":"Exact current empirical_photometry.rs measure + aperture_coverage on the original FITS"})
}
'''

for src,name,append in [('glrt.rs','captured_glrt.rs',GLRT_WRAPPER),('empirical_photometry.rs','captured_photo.rs',PHOTO_WRAPPER)]:
    (OUT/name).write_text((ROOT/'backend/src'/src).read_text(encoding='utf-8')+'\n'+append,encoding='utf-8')

driver=r'''
pub use rst19_core::{fits,spatial};
pub type Result<T>=rst19_core::Result<T>;
#[path="__STATS__"] pub mod statistics;
#[path="__FILTER__"] pub mod filtering;
#[path="captured_glrt.rs"] mod cg;
#[path="captured_photo.rs"] mod cp;
use serde_json::json;
use std::path::Path;
fn crop(img:&[f64],w:usize,x0:usize,y0:usize,n:usize)->Vec<f64>{let mut a=Vec::new();for y in y0..y0+n{a.extend_from_slice(&img[y*w+x0..y*w+x0+n])}a}
fn main(){
 let args:Vec<String>=std::env::args().collect();let path=Path::new(&args[1]);
 let (img,_,w)=cg::read_image(path).unwrap();let n=64;let origin=[2898,3884];
 let raw=crop(&img,w,origin[0],origin[1],n);let mask:Vec<bool>=raw.iter().map(|v|!v.is_finite()||*v<0.0).collect();let valid:Vec<bool>=mask.iter().map(|m|!*m).collect();
 let (bg,noise)=statistics::local_background_rms(&raw,&mask,n,n,statistics::BackgroundOptions{box_size:16,fallback:None,iterations:4,sample_limit:100000}).unwrap();
 let residual:Vec<f64>=raw.iter().zip(&bg).map(|(v,b)|v-b).collect();let sigma=1.93/2.35482;
 let (filtered,_)=filtering::masked_gaussian_response(&residual,&valid,n,n,sigma,false).unwrap();
 let (response,rb,rs)=filtering::standardize_response(&filtered,&mask,&noise,&valid,1000000).unwrap();
 let rad=(4.0*sigma+0.5).floor() as i32;let mut kernel:Vec<f64>=(-rad..=rad).map(|x|(-0.5/(sigma*sigma)*(x*x) as f64).exp()).collect();let sum:f64=kernel.iter().sum();for v in &mut kernel{*v/=sum};
 let mut thresholds=Vec::new();for threshold in 2..=8{
  let proposals=filtering::response_proposals(&response,&valid,n,n,&filtering::ProposalOptions{threshold_sigma:threshold as f64,min_distance:3,scale_fwhm:1.93,method:"gaussian".into()},None).unwrap();
  thresholds.push(json!({"threshold":threshold,"peaks":proposals.iter().map(|p|json!([p.x,p.y,p.score])).collect::<Vec<_>>()}));
 }
 let point=json!({"width":n,"height":n,"origin":origin,"raw":raw,"background":bg,"noise":noise,"residual":residual,"filtered":filtered,"response":response,"response_background":rb,"response_scale":rs,"sigma":sigma,"kernel_1d":kernel,"thresholds":thresholds,"scope":"Exact current statistics.rs and filtering.rs functions on a frozen 64x64 real crop; local box 16, not full-frame parameters"});
 let a=crop(&img,w,1486,276,64);let mut trail=cg::capture(&a,64,64);trail["origin"]=json!([1486,276]);
 let photo=cp::capture(path,2929.8555143839876,3915.2170822163916);
 std::fs::write(&args[2],serde_json::to_vec(&json!({"point":point,"trail":trail,"photometry":photo})).unwrap()).unwrap();
}
'''
driver=driver.replace('__STATS__',(ROOT/'backend/src/statistics.rs').as_posix()).replace('__FILTER__',(ROOT/'backend/src/filtering.rs').as_posix())
(OUT/'capture.rs').write_text(driver,encoding='utf-8')
deps=ROOT/'backend/target/release/deps'
cmd=['rustc',str(OUT/'capture.rs'),'--edition=2021','-O','-C','lto=thin','-A','warnings','-L',f'dependency={deps}','-o',str(OUT/'capture.exe')]
bound=[]
for name in ['rst19_core','serde','serde_json','sha2','csv']:
    p=next(deps.glob(f'lib{name}-*.rlib'));cmd.extend(['--extern',f'{name}={p}']);bound.append(p)
subprocess.run(cmd,cwd=ROOT,check=True)
fits=ROOT/'doc/00-项目资料/原始数据/20260330163205413_9901.fits'
dest=HERE/'evidence/science-traces.json'
subprocess.run([str(OUT/'capture.exe'),str(fits),str(dest)],cwd=ROOT,check=True)
trace=json.loads(dest.read_text(encoding='utf-8'))
prototype=ROOT/'tmp/uni_A3/glrt.json'
trace['dedup']={'candidates':[d for d in json.loads(prototype.read_text(encoding='utf-8'))['detections'] if d['uniform'] and d['L_px']>=40],
               'scope':'Retained fixed-PSF 1.93 prototype candidates; not the PSF 2.1513 production catalog'}
trace['provenance']={'scope':'New numerical exports from source copies; original Rust files unchanged',
 'frame':fits.relative_to(ROOT).as_posix(),'frame_sha256':hashlib.sha256(fits.read_bytes()).hexdigest(),
 'source_bindings':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [ROOT/'backend/src'/x for x in ['glrt.rs','filtering.rs','statistics.rs','empirical_photometry.rs']]],
 'linked_libraries':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in bound],
 'compiler':subprocess.check_output(['rustc','--version'],text=True).strip(),
 'exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'compiled_capture_sha256':hashlib.sha256((OUT/'capture.exe').read_bytes()).hexdigest(),
 'capture_source_sha256':hashlib.sha256((OUT/'capture.rs').read_bytes()).hexdigest(),
 'dedup_input_sha256':hashlib.sha256(prototype.read_bytes()).hexdigest()}
dest.write_text(json.dumps(trace,separators=(',',':'),ensure_ascii=False),encoding='utf-8')
print(json.dumps({'point_threshold_counts':[(v['threshold'],len(v['peaks'])) for v in trace['point']['thresholds']],
 'photo':{k:trace['photometry'][k] for k in ['background','flux','m_inst']},'trail_hypotheses':len(trace['trail']['frames']),
 'bytes':dest.stat().st_size},ensure_ascii=False))
