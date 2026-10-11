"""Rebuild paper figures from existing artifacts; never run the detector or write its cache."""
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import csv, gzip, hashlib, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from datetime import datetime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Microsoft YaHei', 'font.size':9, 'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
                     'savefig.dpi':200, 'axes.labelcolor':'#253348','text.color':'#253348'})
COLORS = ['#174C78','#C85A2D','#2C8B83','#9872A3']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    plt.close(fig)
def writecsv(name, rows):
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

sp=ROOT/'.rst19-cache/3cbeaf34b7381421ab13d688048b6eaa1a084da0c91cf0862800759e9e52a955.sequence.json.gz'
fp=ROOT/'.rst19-cache/fb1332214b8c0c8664c02c55e6993a5bfeac8c83b5a7e378c0bf9bd8f53e62ab.json.gz'
mp=ROOT/'.rst19-cache/2c5cab1b6609450e5a2d4eaf9d61cad7148500fb7b333fa4d6ef177d1c472455.mosaic.npz'
sd=json.load(gzip.open(sp,'rt',encoding='utf-8'));s=sd['sequence']
fd=json.load(gzip.open(fp,'rt',encoding='utf-8'));d=fd['detection'];sources=d['sources']
assert len(sources)==d['candidate_count']==84594
assert sum(x['quality_passed'] for x in sources)==d['quality_count']==29260
groups=[('range_anomaly',{'SATURATED','NEGATIVE_OVERFLOW','CODE_PATTERN'}),
        ('linear_artifact',{'LINE_ARTIFACT'}),('masked_or_edge',{'MASKED','PARTIAL_MASKED','EDGE'}),
        ('crowded_blend',{'UNRESOLVED_BLEND'}),('spike_or_support',{'SPIKE','NARROW','SMALL_FOOTPRINT','INSUFFICIENT_PSF_SUPPORT'}),
        ('weak_or_background',{'LOW_FLUX_SNR','NON_POSITIVE_FLUX','BACKGROUND_UNCERTAIN'}),
        ('shape_outlier',{'NO_SHAPE','BROAD','ELONGATED','DIFFUSE'})]
def classify(x):
    for k,flags in groups:
        if set(x['flags'])&flags:return k
    return 'compact_quality' if x['quality_passed'] else 'other_rejected'
labels={'compact_quality':'紧凑质量','spike_or_support':'尖峰／支持不足','weak_or_background':'弱通量／背景',
        'masked_or_edge':'边缘／掩膜','crowded_blend':'拥挤／未分辨','linear_artifact':'线状／拖影','range_anomaly':'范围异常','shape_outlier':'形状异常'}
order=list(labels)
counts=Counter(classify(x) for x in sources);quality=Counter(classify(x) for x in sources if x['quality_passed'])
rows=[dict(feature_class=k,label=labels[k],candidate_count=counts[k],quality_count=quality[k]) for k in order]
assert sum(r['candidate_count'] for r in rows)==84594
writecsv('source_classes.csv',rows)
fig,axs=plt.subplots(1,2,figsize=(10,4),gridspec_kw={'width_ratios':[1.15,1]})
y=np.arange(len(order));axs[0].barh(y, [counts[k] for k in order],color='#B8CDDB',label='候选层')
axs[0].barh(y, [quality[k] for k in order],color=COLORS[0],label='质量层')
axs[0].set_yticks(y,[labels[k] for k in order]);axs[0].invert_yaxis();axs[0].set_xlabel('数量');axs[0].legend(frameon=False,fontsize=8)
for i,k in enumerate(order): axs[0].text(counts[k]+300,i,f'{counts[k]:,} / {quality[k]:,}',va='center',fontsize=7)
axs[0].set_xlim(0,36000);axs[0].set_title('(a) 首要类别统计')
for passed,c,l in [(False,'#ABB8C6','未通过质量层'),(True,COLORS[0],'质量通过')]:
    a=[x['flux_snr'] for x in sources if x['quality_passed']==passed and x['flux_snr'] and x['flux_snr']>0]
    axs[1].hist(a,bins=np.geomspace(.05,4000,65),histtype='step',lw=1.5,color=c,label=l)
axs[1].set_xscale('log');axs[1].set_yscale('log');axs[1].axvline(5,color=COLORS[1],ls='--',lw=1)
axs[1].set_xlabel('孔径通量 SNR');axs[1].set_ylabel('候选数');axs[1].set_title('(b) 高 SNR 仍可能落选');axs[1].legend(frameon=False,fontsize=8)
fig.tight_layout(w_pad=3);save(fig,'source_quality')

# Verify raw identities and FITS headers without any processing or cache writes.
rawdir=ROOT/'doc/00-项目资料/原始数据'
rawrows=[];arrays=[]
for ident in sd['input_identity']['frames']:
    p=rawdir/Path(ident['path']).name
    actual=sha(p); assert actual==ident['sha256'],f'Input hash mismatch: {p.name}'
    with p.open('rb') as f: header=f.read(2880).decode('ascii',errors='replace')
    cards={header[i:i+8].strip():header[i+10:i+80].split('/')[0].strip().strip("'") for i in range(0,2880,80) if header[i+8:i+10]=='= '}
    assert int(cards['NAXIS1'])==4096 and int(cards['NAXIS2'])==4096 and int(cards['BITPIX'])==16
    a=np.memmap(p,dtype='>i2',mode='r',offset=2880,shape=(4096,4096))
    arrays.append(a)
    rawrows.append(dict(file=p.name,sha256=actual,size=p.stat().st_size,date_obs=cards['DATE-OBS'],
                        width=4096,height=4096,exposure_ms=float(cards['EXPOSURE'].split()[-1]),
                        wcs_core=all(k in cards for k in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2','CTYPE1','CTYPE2'])))
writecsv('input_manifest.csv',rawrows)
times=np.array([(datetime.fromisoformat(f['timestamp'])-datetime.fromisoformat(s['frames'][0]['timestamp'])).total_seconds() for f in s['frames']])
shifts=np.array(s['cumulative_shifts'])
fig,axs=plt.subplots(1,2,figsize=(9,3.3))
for k,l,c in [(0,'Δx',COLORS[0]),(1,'Δy',COLORS[1])]:axs[0].plot(times,shifts[:,k],'o-',ms=3,lw=1.3,label=l,color=c)
axs[0].set_xlabel('相对首帧时间 / s');axs[0].set_ylabel('累计平移 / px');axs[0].legend(frameon=False);axs[0].set_title('(a) 平移坐标记录')
axs[1].plot(np.arange(1,16),[f['quality_count'] for f in s['frames']],'o-',ms=3,color=COLORS[0],label='工作集质量源')
axs[1].axhline(3000,color=COLORS[1],ls='--',label='工作集上限 3000');axs[1].set_ylim(2600,3060)
axs[1].set_xlabel('帧号');axs[1].set_ylabel('数量');axs[1].set_xticks([1,5,10,15]);axs[1].legend(frameon=False,fontsize=8);axs[1].set_title('(b) 序列工作集口径')
fig.tight_layout(w_pad=3);save(fig,'registration')

# First-frame crop, measured centroid anchors, and raw value topology.
pair=[next(x for x in sources if x['detection_id']==i) for i in [82931,82934]]
x0,y0=2424,4007;crop=np.asarray(arrays[0][y0:y0+32,x0:x0+32],dtype=float)
fig,axs=plt.subplots(1,3,figsize=(10,3.3))
im=axs[0].imshow(crop,origin='upper',extent=(x0-.5,x0+31.5,y0+31.5,y0-.5),cmap='gray',norm=SymLogNorm(linthresh=50,vmin=-30000,vmax=14000))
for p,c in zip(pair,COLORS):
    axs[0].scatter(p['x'],p['y'],s=65,marker='x',color=c,lw=1.5)
    axs[0].add_patch(plt.Circle((p['x'],p['y']),4,fill=False,color=c,lw=1))
axs[0].set_title('(a) 原始 ADU 与质心孔径');axs[0].set_xlabel('x / px');axs[0].set_ylabel('y / px')
mask=np.zeros(crop.shape);mask[np.isin(crop,[3990,3991,3992,3993])]=1;mask[crop<-1000]=2
axs[1].imshow(mask,origin='upper',cmap=matplotlib.colors.ListedColormap(['#F1F3F5',COLORS[1],COLORS[0]]),vmin=0,vmax=2)
axs[1].set_title('(b) 重复高码与极端负值');axs[1].set_xlabel('窗口列');axs[1].set_ylabel('窗口行')
from matplotlib.patches import Patch
axs[1].legend(handles=[Patch(color=COLORS[1],label='重复高位码'),Patch(color=COLORS[0],label='< −1000 ADU')],loc='lower right',fontsize=6,framealpha=.85)
cx=int(round(np.mean([p['peak_x'] for p in pair])));row=int(round(np.mean([p['peak_y'] for p in pair])))
axs[2].plot(np.arange(x0,x0+32),np.asarray(arrays[0][row,x0:x0+32]),'o-',ms=2,lw=1,color=COLORS[0]);axs[2].axhline(0,color='#AAA',lw=.6)
axs[2].set_title(f'(c) 原始横截面 y={row}');axs[2].set_xlabel('x / px');axs[2].set_ylabel('ADU')
fig.tight_layout(w_pad=2);save(fig,'pair_raw')

# Historical, documented experiment; no claim of a rerun.
hits={'低纹理':[2,22,24],'高背景':[3,21,24],'边缘':[1,19,24]}
fine={'低纹理':[0,1,16],'高背景':[0,2,16],'边缘':[0,0,0]}
def wilson(k,n=24):
    z=1.95996398454;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;w=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-w,c+w
ir=[];fig,axs=plt.subplots(1,3,figsize=(9.5,3.2),sharey=True)
for ax,(label,vals) in zip(axs,hits.items()):
    for name,series,c,marker in [('候选层',vals,COLORS[0],'o'),('质量层',fine[label],COLORS[1],'s')]:
        rate=np.array(series)/24;ci=np.array([wilson(k) for k in series]);xx=np.array([24,56,128])
        ax.errorbar(xx,rate,yerr=np.array([rate-ci[:,0],ci[:,1]-rate]),marker=marker,color=c,capsize=3,lw=1.3,ms=4,label=name)
        for x,k,p,l,u in zip(xx,series,rate,ci[:,0],ci[:,1]):ir.append(dict(stratum=label,layer=name,peak_adu=int(x),hits=k,n=24,recall=float(p),wilson_low=float(l),wilson_high=float(u),provenance='doc/04-创新分析/创新交付审计.md; documented historical experiment'))
    ax.set_xticks([24,56,128]);ax.set_ylim(-.06,1.08);ax.set_xlabel('名义注入峰值 / ADU');ax.set_title(label);ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('条件化回收率');axs[-1].legend(frameon=False,fontsize=8,loc='center right')
fig.tight_layout();save(fig,'injection');writecsv('injection_documented.csv',ir)

motions=[t for t in s['motion_features'] if t['classification']=='moving']
mr=[];fig,axs=plt.subplots(2,2,figsize=(9,6))
for ax,t,c in zip(axs.flat,motions,COLORS):
    pts=t['points']; xy=np.array([[p['aligned_x'],p['aligned_y']] for p in pts]);tt=np.array([times[p['frame_index']] for p in pts])
    X=np.column_stack([np.ones(len(tt)),tt]);coef=np.linalg.lstsq(X,xy,rcond=None)[0];speed=float(np.linalg.norm(coef[1]));rms=float(np.sqrt(np.mean(np.sum((xy-X@coef)**2,axis=1))))
    ax.plot(xy[:,0],xy[:,1],'o-',color=c,lw=1,ms=3)
    for p in [pts[0],pts[-1]]:ax.annotate('F%02d'%(p['frame_index']+1),(p['aligned_x'],p['aligned_y']),fontsize=7,xytext=(4,4),textcoords='offset points')
    ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('注册 x / px');ax.set_ylabel('注册 y / px');ax.set_title(f"L{t['track_id']}：{len(pts)} 帧，{speed:.2f} px/s",fontsize=9)
    mr.append(dict(track='L'+str(t['track_id']),presence=t['presence'],first_frame=min(p['frame_index'] for p in pts)+1,last_frame=max(p['frame_index'] for p in pts)+1,
                   displacement_px=t['displacement_px'],cache_rms_px=t['fit_rms_px'],time_ols_rms_px=rms,vx_px_s=float(coef[1,0]),vy_px_s=float(coef[1,1]),speed_px_s=speed,
                   exposure_velocity_ratio=t['exposure_velocity_ratio'],line_overlap_max=max(p['line_artifact_overlap_fraction'] for p in pts),
                   detected_points=sum(p.get('observation_source')=='detected' for p in pts),predicted_roi_points=sum(p.get('observation_source')!='detected' for p in pts),evidence_level=t['evidence_level']))
fig.tight_layout(h_pad=2,w_pad=3);save(fig,'motion_tracks');writecsv('motion_cache.csv',mr)
# Raw images for L1/L4, inspect actual morphology instead of reusing old contact sheets.
fig,axs=plt.subplots(2,4,figsize=(10,4.8))
for row,t in enumerate([motions[0],motions[-1]]):
    indices=np.linspace(0,len(t['points'])-1,4).astype(int)
    for ax,j in zip(axs[row],indices):
        p=t['points'][j];x,y=int(round(p['x'])),int(round(p['y']));radius=48; xx0=max(0,x-radius);yy0=max(0,y-radius); xx1=min(4096,x+radius+1);yy1=min(4096,y+radius+1)
        a=np.asarray(arrays[p['frame_index']][yy0:yy1,xx0:xx1],dtype=float)
        ax.imshow(a,cmap='gray',origin='upper',norm=Normalize(vmin=np.percentile(a,20),vmax=np.percentile(a,99.5)),extent=(xx0-.5,xx1-.5,yy1-.5,yy0-.5))
        ax.scatter(p['x'],p['y'],marker='+',s=55,color=COLORS[1],lw=.8);ax.set_title(f"L{t['track_id']} / F{p['frame_index']+1:02d}",fontsize=8);ax.tick_params(labelsize=6)
fig.tight_layout();save(fig,'motion_raw')

z=np.load(mp,allow_pickle=False);meta=json.loads(str(z['metadata'])); image=z['image'];coverage=z['coverage'];scatter=z['scatter']
assert meta['cumulative_shifts']==s['cumulative_shifts']
assert meta['max_coverage']==int(coverage.max())
fig,axs=plt.subplots(1,3,figsize=(10,3.5))
v=image[::8,::8]; finite=v[np.isfinite(v)];axs[0].imshow(v,cmap='gray',origin='upper',vmin=np.percentile(finite,10),vmax=np.percentile(finite,99));axs[0].set_title('(a) 稳健 ADU 合成显示')
im=axs[1].imshow(coverage[::8,::8],origin='upper',cmap='viridis',vmin=0,vmax=15);fig.colorbar(im,ax=axs[1],fraction=.046,pad=.04);axs[1].set_title('(b) 覆盖帧数')
sv=scatter[::8,::8];im=axs[2].imshow(sv,cmap='magma',origin='upper',vmin=0,vmax=np.nanpercentile(sv,99));fig.colorbar(im,ax=axs[2],fraction=.046,pad=.04);axs[2].set_title('(c) 帧间离散度 / ADU')
for ax in axs:ax.set_xticks([0,256,512],['0','2048','4096']);ax.set_yticks([0,256,512],['0','2048','4096']);ax.set_xlabel('输出 x / px')
fig.tight_layout(w_pad=1);save(fig,'mosaic')

summary={'verified_on':'2026-10-10','frame_cache_sha256':sha(fp),'sequence_cache_sha256':sha(sp),'mosaic_cache_sha256':sha(mp),
 'frame_cache_version':fd['cache_version'],'sequence_cache_version':sd['sequence_cache_version'],
 'single_frame':{k:v for k,v in d.items() if k!='sources'},'faintest':fd['faintest_detected'],'classes':rows,
 'sequence':{k:s[k] for k in ['stable_source_count','persistent_source_count','moving_track_count','motion_evidence_group_count','source_working_limit','registration_model','registration_affine_frame_count','registration_residual_median_px','link_radius_px','min_presence','persistent_min_presence','temporal_reference_count','stack_faint_count','detector_parameters']},
 'sequence_layer_counts':{'/'.join(k):v for k,v in Counter((t['classification'],t['evidence_level']) for t in s['tracks']).items()},
 'sequence_frame_counts':[{k:f[k] for k in ['frame_index','timestamp','candidate_count','returned_count','quality_count']} for f in s['frames']],
 'time_span_s':float(times[-1]),'median_interval_s':float(np.median(np.diff(times))),'max_translation_px':float(np.max(np.linalg.norm(shifts,axis=1))),
 'motion':mr,'mosaic':meta,'raw_identity_verified_count':len(rawrows),'injection_provenance':'Historical document only; original machine outputs absent in tmp; not rerun.',
 'detector_rerun':False,'git_head':'4a2dd99307065c9454f31182e75944f0b510c606','git_worktree':'dirty; source differs from HEAD'}
(HERE/'evidence.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'classes':rows,'motion':mr,'max_translation':summary['max_translation_px'],'span':summary['time_span_s'],'figures':len(list(FIG.glob('*.pdf')))},ensure_ascii=False,indent=2))
