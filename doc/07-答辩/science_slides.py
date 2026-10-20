"""Numerical replay pages; no decorative motion. Used by build_deck.py."""
from pathlib import Path
import json, re
HERE=Path(__file__).resolve().parent

def revise(slides, glrt):
    traces=json.loads((HERE/'evidence/science-traces.json').read_text(encoding='utf-8'))
    traces['dedup']={'candidates':[d for d in glrt['detections'] if d['uniform'] and d['L_px']>=40],
                     'scope':'Retained fixed-PSF 1.93 prototype candidates; not the PSF 2.1513 production catalog'}
    specs={
      3:('background','真实像素如何转化为背景扣除图','同一 64 × 64 FITS 裁剪，逐步查看原始 DN、背景估计与逐像素相减结果。',
         '数据路径：statistics::local_background_rms → residual = raw − background。局部教学实验使用 16 px 分块。',
         '当前像素的 I、B、I−B 均可核对；本页是局部计算，不替代全帧背景模型。'),
      4:('convolution','卷积响应由像素与核权重逐项相乘得到','移动到任一像素，显示 Gaussian 权重和加权和；响应图随实际计算次序展开。',
         '当前源码：masked_gaussian_response → standardize_response → response_proposals。阈值改变时重新选择已导出的真实候选。',
         '播放推进的是像素计算次序。阈值改变会影响候选数量；提案响应不等于全帧虚警概率。'),
      5:('photometry','净通量由 65 个孔径像素的贡献累加得到','在原始 FITS 的同一目标位置，重放背景环估计、面积权重和净通量累加。',
         '当前源码：empirical_photometry::measure 与 aperture_coverage。孔径半径 4 px，背景环 6–10 px。',
         '结束时 F = 268.75，m_inst = −6.073371；未完成的累积量仅用于解释计算过程。'),
      10:('kernels','方向与长度变化会改变同一拖影的响应','73 个核来自当前 GLRT 频域参照路径；显示真实裁剪、白化核及对应响应图。',
          '当前源码：kernel_from_samples → whiten_kernel → mf_z_freq。白化模型由该裁剪估计；3 长度 × 24 方向 + 点核。',
          'Λ 随已检验假设逐步取最大值；本页为局部频域实验，不能与全帧 Line 目录的数值混用。'),
      17:('dedup','共线去重逐条判断真实候选是否属于同一脊线','对固定 PSF 原型中 9 条均匀长候选按 Λ 降序重放，显式计算方向差和垂直距离。',
          '实际判据：轴方向差 ≤ 3° 且至保留脊线垂距 < 3 px；满足时抑制当前候选。',
          '此处重放原型候选的去重过程；产品目录 3 条记录来自另一 PSF 配置，分别保留。')}
    for idx,(kind,title,sub,method,claim) in specs.items():
        body=f'''<div class="science-layout" data-lab="{kind}"><div class="science-main"><canvas id="lab-{kind}" width="880" height="370" aria-label="{title}"></canvas>
        <div class="science-controls"><button data-action="reset">复位</button><button data-action="play">播放计算</button><button data-action="step">下一步</button><input data-action="seek" aria-label="计算进度" type="range" min="0" value="0"><output data-output="progress"></output></div>
        <div class="science-caption" data-output="caption"></div></div><aside class="science-readout"><div class="science-tag">真实像素 / 数值计算</div><div data-output="values"></div>'''
        if kind=='convolution': body+='<label class="threshold-control">提案阈值 <input data-action="threshold" type="range" min="2" max="8" value="4" step="1"><output data-output="threshold">4</output></label>'
        if kind=='kernels': body+='<label class="threshold-control">直接选择核 <select data-action="hypothesis" aria-label="核假设"></select></label>'
        body+=f'<p class="science-method">{method}</p></aside></div>'
        s=slides[idx]
        s=re.sub(r'<h1>.*?</h1><p class="subtitle">.*?</p>',f'<h1>{title}</h1><p class="subtitle">{sub}</p>',s,flags=re.S)
        s=re.sub(r'<div class="content">.*?</div>\s*<div class="claim">.*?</div>',f'<div class="content">{body}</div><div class="claim">{claim}</div>',s,flags=re.S)
        s=re.sub(r'data-title="[^"]*"',f'data-title="{title}"',s)
        s=re.sub(r'<span class="pill [^"]*">.*?</span>','<span class="pill measured">数值实验</span>',s,count=1)
        s=re.sub(r'<footer><span>.*?</span>',f'<footer><span>evidence/science-traces.json · 原始 FITS SHA-256 与本次编译源码绑定 · {kind}</span>',s)
        s=re.sub(r'<aside class="speaker">.*?</aside>',f'<aside class="speaker"><h2>计算来源与范围</h2><p>{method}</p><p>{claim}</p><p>详见 science-traces.json 及 prepare_science.py。当前源码复制到独立采集模块，追加只读导出入口后编译；原有程序文件未改动。</p></aside>',s,flags=re.S)
        slides[idx]=s
    return slides,traces

CSS=r'''
/* Only numerical state changes move; all ornamental animation is removed. */
*{animation:none!important;transition:none!important;stroke-dashoffset:0!important}.reveal{opacity:1!important;transform:none!important}.bar{transform:none!important}.scanner{display:none!important}
.science-layout{display:grid;grid-template-columns:880px 1fr;gap:30px;align-items:start}.science-main{min-width:0}.science-main canvas{background:#eef3f8;border:1px solid #d8e2ec;width:880px;height:370px;border-radius:4px}.science-controls{display:flex;gap:9px;align-items:center;margin-top:13px}.science-controls button{border:1px solid #b7c9d9;background:white;color:#1c4966;padding:6px 12px;border-radius:4px;font-size:16px;white-space:nowrap}.science-controls button:hover{background:#dcecef}.science-controls input{flex:1;min-width:120px;accent-color:#087f83}.science-controls output{font-size:16px;min-width:79px;text-align:right;color:#265367}.science-caption{font-size:15px;color:#536f83;margin-top:10px;line-height:1.6;min-height:24px}.science-tag{font-size:14px;letter-spacing:1px;color:#087f83;border-bottom:2px solid #67aaa9;padding:3px 0 10px;margin-bottom:13px}.science-readout dl{display:grid;grid-template-columns:1fr auto;gap:9px 8px;margin:0;font-size:18px;line-height:1.6}.science-readout dt{color:#596f81}.science-readout dd{margin:0;text-align:right;color:#124c6a;font-variant-numeric:tabular-nums;font-family:Consolas,"Microsoft YaHei",sans-serif}.science-method{font-size:16px;line-height:1.8;color:#556d80;margin-top:16px!important}.threshold-control{font-size:16px;display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:17px;color:#23586d}.threshold-control input{width:130px;accent-color:#087f83}.threshold-control select{width:100%;padding:6px;font-size:15px;border:1px solid #bbcedb;background:white;color:#24536b}.science-equation{font:20px/1.5 "Cambria Math",serif;color:#2764e7;margin:16px 0}.science-warning{font-size:14px;color:#946029!important;line-height:1.6}.science-readout strong{font-size:26px;color:#087f83}@media print{.science-controls{display:none!important}}
'''

JS=r'''
(()=>{
const D=window.SCIENCE_TRACE;
const state={};let stamp=0;
const get=(kind)=>state[kind]||(state[kind]={t:0,playing:false,threshold:4});
const canvas=kind=>document.querySelector('#lab-'+kind);
const panel=kind=>document.querySelector(`[data-lab="${kind}"]`);
const heatCache=new Map();
const fmt=(n,d=3)=>Number(n).toFixed(d);
function dl(rows){return '<dl>'+rows.map(([a,b])=>`<dt>${a}</dt><dd>${b}</dd>`).join('')+'</dl>'}
function label(ctx,x,y,s,color='#415e74',size=17){ctx.fillStyle=color;ctx.font=`${size}px Microsoft YaHei`;ctx.fillText(s,x,y)}
function image(ctx,a,w,h,x,y,size,min,max,key,signed=false){
 let cache=heatCache.get(key);
 if(!cache){const off=document.createElement('canvas');off.width=w;off.height=h;const oc=off.getContext('2d'),im=oc.createImageData(w,h);for(let i=0;i<a.length;i++){
 const q=Math.max(0,Math.min(1,(a[i]-min)/(max-min||1)));let r,g,b;
 if(signed){const v=Math.abs(q-.5)*2,target=q<.5?[196,60,67]:[39,100,231];r=Math.round(239+(target[0]-239)*v);g=Math.round(244+(target[1]-244)*v);b=Math.round(248+(target[2]-248)*v)}
 else{const v=Math.asinh(10*q)/Math.asinh(10);r=9+v*218;g=22+v*221;b=37+v*211}
 im.data.set([r,g,b,255],i*4)}oc.putImageData(im,0,0);heatCache.set(key,off);cache=off}
 ctx.imageSmoothingEnabled=false;ctx.drawImage(cache,x,y,size,size);ctx.strokeStyle='#bdd0df';ctx.strokeRect(x,y,size,size)
}
function clear(ctx){ctx.clearRect(0,0,880,370);ctx.fillStyle='#eef3f8';ctx.fillRect(0,0,880,370)}
function extent(a,p=.99){const s=a.slice().sort((a,b)=>a-b);return [s[0],s[Math.floor((s.length-1)*p)]]}
const limits={raw:extent(D.point.raw),bg:extent(D.point.background),res:extent(D.point.residual),response:extent(D.point.response),trailRaw:extent(D.trail.raw),trailResp:extent(D.trail.frames.flatMap(f=>f.map),.995)};
const terms=D.photometry.pixels.filter(p=>p.weight>0),cum=[0];for(const p of terms)cum.push(cum.at(-1)+p.term);
const maxT={background:2,convolution:4095,photometry:terms.length,kernels:D.trail.frames.length-1,dedup:D.dedup.candidates.length};
function convAt(idx){const p=D.point,w=p.width,xx=idx%w,yy=Math.floor(idx/w),k=p.kernel_1d,r=(k.length-1)/2;let sum=0;let weights=0;for(let j=0;j<k.length;j++)for(let i=0;i<k.length;i++){const x=Math.max(0,Math.min(w-1,xx+i-r)),y=Math.max(0,Math.min(w-1,yy+j-r)),at=y*w+x;const wt=k[i]*k[j];if(p.raw[at]>=0){sum+=wt*p.residual[at];weights+=wt}}return sum/weights}
function kernelAt(f,x,y){const t=D.trail,w=t.width,cy=Math.floor(f.kernel_rows/2),cx=Math.floor(f.kernel_cols/2);let sum=0;for(let j=0;j<f.kernel_rows;j++)for(let i=0;i<f.kernel_cols;i++){const xx=x+i-cx,yy=y+j-cy;if(xx>=0&&yy>=0&&xx<w&&yy<t.height)sum+=f.kernel[j*f.kernel_cols+i]*t.whitened[yy*w+xx]}return sum/f.norm}
function dedupStep(n){let kept=[],actions=[];const a=D.dedup.candidates;for(let i=0;i<n;i++){const d=a[i];let match=null,closest=null;for(let j=0;j<kept.length;j++){let diff=Math.abs(kept[j].pa_deg-d.pa_deg)%180;if(diff>90)diff=180-diff;let th=kept[j].pa_deg*Math.PI/180,dist=Math.abs(-Math.sin(th)*(d.x-kept[j].x)+Math.cos(th)*(d.y-kept[j].y));const pair={rank:j+1,angle:diff,distance:dist};if(!closest||dist<closest.distance)closest=pair;if(diff<=3&&dist<3){match=pair;break}}
 const record={index:i,kept:!match,pair:match||closest};actions.push(record);if(!match)kept.push(d)}return {kept,actions}}
function render(kind){const el=panel(kind);if(!el)return;const s=get(kind),ca=canvas(kind),ctx=ca.getContext('2d');const t=Math.max(0,Math.min(maxT[kind],Math.round(s.t)));s.t=t;clear(ctx);let values='',caption='';
 if(kind==='background'){
 const p=D.point,arrays=[p.raw,p.background,p.residual],lim=[limits.raw,limits.bg,limits.res];for(let i=0;i<3;i++){const x=15+i*290;label(ctx,x,24,['I · 原始像素 / DN','B · 局部背景 / DN','I − B · 净信号 / DN'][i]);if(i<=t)image(ctx,arrays[i],64,64,x,44,255,...lim[i],'bg'+i);else{ctx.fillStyle='#dce5ed';ctx.fillRect(x,44,255,255);label(ctx,x+65,174,'等待计算步骤')};if(i<2)label(ctx,x+267,175,i===0?'−':'=', '#2764e7',24)}
 const idx=31*64+32;values=dl([['当前阶段',`${t+1} / 3`],['固定像素',`(32, 31)`],['原始 DN',fmt(p.raw[idx])],['背景 B',fmt(p.background[idx])],['I − B',t===2?fmt(p.residual[idx]):'待相减']]);
 label(ctx,15,335,'色标分别注明 DN 范围；相减是数值运算，不是图像亮度变换。', '#586f81',16);caption='阶段：'+['原始观测','估计每个位置的背景','逐像素相减，保留残差'][t];
 }else if(kind==='convolution'){
 const p=D.point,w=64,x=t%w,y=Math.floor(t/w),v=convAt(t),k=p.kernel_1d,r=(k.length-1)/2;
 label(ctx,15,24,'输入残差 · I−B / DN');image(ctx,p.residual,w,w,15,44,255,...limits.res,'conv-res');
 ctx.strokeStyle='#e6b35a';ctx.lineWidth=2;ctx.strokeRect(15+(x-r)*255/w,44+(y-r)*255/w,k.length*255/w,k.length*255/w);ctx.lineWidth=1;
 label(ctx,303,24,'Gaussian 权重 K');const kw=k.flatMap(a=>k.map(b=>a*b));image(ctx,kw,k.length,k.length,303,70,192,0,Math.max(...kw),'gaussian-kernel');
 label(ctx,543,24,'标准化响应 R′ 与候选');image(ctx,p.response,w,w,543,44,255,...limits.response,'conv-response');
 const cell=255/w;ctx.fillStyle='#e0e8f0';const row=Math.floor(t/w),col=t%w;ctx.fillRect(543+(col+1)*cell,44+row*cell,255-(col+1)*cell,cell);ctx.fillRect(543,44+(row+1)*cell,255,255-(row+1)*cell);
 const peaks=p.thresholds.find(a=>a.threshold===s.threshold).peaks;let visited=0;ctx.strokeStyle='#6be7d5';ctx.lineWidth=2;for(const q of peaks)if(q[1]*w+q[0]<=t){visited++;ctx.strokeRect(543+q[0]*cell-6,44+q[1]*cell-6,12,12)}ctx.lineWidth=1;
 ctx.font='10px Consolas';ctx.textAlign='center';for(let j=0;j<k.length;j++)for(let i=0;i<k.length;i++){ctx.fillStyle=kw[j*k.length+i]>.025?'#112e48':'#b4ccd9';ctx.fillText(fmt(kw[j*k.length+i],3),303+(i+.5)*192/k.length,70+(j+.57)*192/k.length)}ctx.textAlign='start';
 label(ctx,303,289,'Σ K = '+fmt(kw.reduce((a,b)=>a+b,0),6),'#2764e7',15);label(ctx,15,335,`逐项乘积之和 = ${fmt(v,6)} DN；当前处理位置 (${x},${y})`, '#265875',18);
 values=dl([['像素序号',`${t+1} / 4096`],['裁剪坐标',`(${x}, ${y})`],['浏览器 ΣK·d',fmt(v,6)],['Rust 滤波值',fmt(p.filtered[t],6)],['差值',Math.abs(v-p.filtered[t]).toExponential(1)],['标准化响应',fmt(p.response[t])],['已访问候选',`${visited} / ${peaks.length}`]]);caption=`响应图按行扫描；青框为阈值 ${s.threshold} 下已访问的真实候选。点击左图可定位像素。`;
 }else if(kind==='photometry'){
 const p=D.photometry,raw=p.pixels.map(a=>a.value),lim=extent(raw);label(ctx,15,24,'真实像素 · 孔径与背景环');image(ctx,raw,25,25,15,43,285,...lim,'photo-raw');
 const scale=285/25,cx=15+(p.center[0]+.5)*scale,cy=43+(p.center[1]+.5)*scale;
 ctx.lineWidth=1.5;for(const [r,color] of [[p.annulus_inner,'#4b8dff'],[p.annulus_outer,'#4b8dff'],[p.radius,'#62dbc7']]){ctx.strokeStyle=color;ctx.beginPath();ctx.arc(cx,cy,r*scale,0,2*Math.PI);ctx.stroke()}
 for(const q of terms.slice(0,t)){ctx.fillStyle=`rgba(44,211,166,${.18+.45*q.weight})`;ctx.fillRect(15+q.x*scale,43+q.y*scale,scale,scale)}
 if(t){const q=terms[t-1];ctx.strokeStyle='#ffba5b';ctx.lineWidth=2;ctx.strokeRect(15+q.x*scale,43+q.y*scale,scale,scale)}
 label(ctx,356,24,'累积净通量 F / DN');const lo=Math.min(0,...cum),hi=Math.max(...cum),gx=370,gy=306,gw=440,gh=230;
 ctx.strokeStyle='#c1cfda';ctx.beginPath();ctx.moveTo(gx,gy-gh);ctx.lineTo(gx,gy);ctx.lineTo(gx+gw,gy);ctx.stroke();
 for(let j=0;j<=3;j++)label(ctx,gx-12,gy-(hi-lo)*j/3/(hi-lo||1)*gh,fmt(lo+(hi-lo)*j/3,0),'#697e90',14);
 ctx.strokeStyle='#087f83';ctx.lineWidth=3;ctx.beginPath();for(let i=0;i<=t;i++){const xx=gx+i/terms.length*gw,yy=gy-(cum[i]-lo)/(hi-lo)*gh;i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy)}ctx.stroke();ctx.lineWidth=1;
 label(ctx,gx,340,'0', '#687d90',16);label(ctx,gx+gw-35,340,`${terms.length} 像素`, '#687d90',16);
 const q=t?terms[t-1]:null;values=dl([['背景 b',fmt(p.background,3)],['加入像素',`${t} / ${terms.length}`],['像素 DN',q?fmt(q.value,2):'—'],['面积权重 a',q?fmt(q.weight,4):'—'],['a·(I−b)',q?fmt(q.term,4):'—'],['累积 F',fmt(cum[t],4)],['最终 Rust F',fmt(p.flux,4)]]);
 values+=`<div class="science-equation">F = Σaᵢ(Iᵢ−b)</div><div class="science-equation">m_inst = ${t===terms.length?fmt(-2.5*Math.log10(cum[t]),6):'完成后计算'}</div>`;
 caption=t===terms.length?'65 项累加已完成，误差 = '+Math.abs(cum[t]-p.flux).toExponential(1)+'；颜色只标记已加入的真实像素。':'蓝圈：6–10 px 背景环；青圈：4 px 孔径；逐像素加入 a·(I−b)。';
 }else if(kind==='kernels'){
 const p=D.trail,f=p.frames[t],maxK=Math.max(...f.kernel.map(Math.abs));
 label(ctx,15,24,'同一真实拖影 · 原始 DN');image(ctx,p.raw,64,64,15,43,210,...limits.trailRaw,'trail-raw');
 ctx.strokeStyle='#e8b454';ctx.strokeRect(15+32*210/64-5,43+32*210/64-5,10,10);
 label(ctx,290,24,'当前白化核 K̃');const size=190,kside=Math.max(f.kernel_rows,f.kernel_cols),padded=new Array(kside*kside).fill(0),oy=Math.floor((kside-f.kernel_rows)/2),ox=Math.floor((kside-f.kernel_cols)/2);for(let y=0;y<f.kernel_rows;y++)for(let x=0;x<f.kernel_cols;x++)padded[(y+oy)*kside+x+ox]=f.kernel[y*f.kernel_cols+x];image(ctx,padded,kside,kside,290,53,size,-maxK,maxK,'kernel'+t,true);
 label(ctx,558,24,'当前核的真实响应图');image(ctx,f.map,64,64,558,43,210,...limits.trailResp,'trail-map'+t);
 let best=0;for(let i=1;i<=t;i++)if(p.frames[i].z>p.frames[best].z)best=i;
 const allMax=Math.max(...p.frames.map(a=>a.z)),allMin=Math.min(0,...p.frames.map(a=>a.z));
 const chartY=v=>343-(v-allMin)/(allMax-allMin)*43,zeroY=chartY(0);ctx.strokeStyle='#9eb3c5';ctx.beginPath();ctx.moveTo(55,zeroY);ctx.lineTo(815,zeroY);ctx.stroke();label(ctx,5,304,fmt(allMax,0),'#75899a',12);label(ctx,25,346,fmt(allMin,0),'#75899a',12);
 for(let i=0;i<=t;i++){const xx=55+i/72*760,yy=chartY(p.frames[i].z);ctx.strokeStyle=i===best?'#087f83':'#7397bc';ctx.beginPath();ctx.moveTo(xx,zeroY);ctx.lineTo(xx,yy);ctx.stroke()}
 for(const i of [0,24,48,72])label(ctx,55+i/72*760-9,362,'#'+(i+1),'#75899a',11);
 label(ctx,290,262,`核色标 ±${maxK.toExponential(1)}`, '#586e82',13);label(ctx,558,267,`响应色标 ${fmt(limits.trailResp[0],1)} 至 ${fmt(limits.trailResp[1],1)}`, '#586e82',13);
 label(ctx,25,289,'固定像素 (32,32) 的逐假设响应 z；每条线对应一次实际核计算', '#586e82',15);
 const direct=kernelAt(f,32,32);values=dl([['假设序号',`${t+1} / 73`],['核长度',fmt(f.length,2)+' px'],['实际方向',fmt(f.angle,2)+'°'],['z 当前核',fmt(f.z,4)],['Λ 已检验最大',fmt(p.frames[best].z,4)],['最佳核序号',String(best+1)],['内积复算误差',Math.abs(direct-f.z).toExponential(1)]]);
 caption=`核像素保持等比例；暖色为负、蓝色为正、零为浅灰。核图按 ±max|K̃| 显示；响应图使用固定色标。`;
 }else if(kind==='dedup'){
 const a=D.dedup.candidates,out=dedupStep(t),acts=out.actions;label(ctx,20,24,'真实候选坐标 · 横轴 x / px，纵轴 y / px');
 const X=x=>40+x/4096*510,Y=y=>44+y/4096*282;ctx.strokeStyle='#c6d5e0';for(let v=0;v<=4096;v+=1024){ctx.beginPath();ctx.moveTo(X(v),44);ctx.lineTo(X(v),326);ctx.moveTo(40,Y(v));ctx.lineTo(550,Y(v));ctx.stroke();label(ctx,X(v)-8,352,v,'#75899a',13);label(ctx,1,Y(v)+5,v,'#75899a',11)}
 for(let i=0;i<a.length;i++){const act=acts.find(v=>v.index===i);const color=!act?'#b2c0cd':act.kept?'#087f83':'#be6c36';ctx.fillStyle=color;ctx.beginPath();ctx.arc(X(a[i].x),Y(a[i].y),i===t-1?7:4,0,2*Math.PI);ctx.fill();let lx=X(a[i].x)+8,ly=Y(a[i].y)+2;if(i>=1&&i<=3){lx+=24;ly+=(i-2)*19;ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(X(a[i].x)+5,Y(a[i].y));ctx.lineTo(lx-3,ly-4);ctx.stroke()}label(ctx,lx,ly,String(i+1),color,14)}
 label(ctx,596,59,'保留 / 抑制判断','#245b6c',20);for(let i=0;i<acts.length;i++)label(ctx,598,91+i*27,`#${i+1}  ${acts[i].kept?'保留':'同脊线抑制'}`,acts[i].kept?'#087f83':'#a66132',17);
 const a0=acts.at(-1);values=dl([['已处理',`${t} / ${a.length}`],['保留代表',String(out.kept.length)],['当前候选',t?'#'+t:'—'],['最近比较方向差',a0?.pair?fmt(a0.pair.angle,4)+'°':'—'],['对应垂直距离',a0?.pair?fmt(a0.pair.distance,4)+' px':'—'],['本次判断',a0?(a0.kept?'保留':'抑制'):'等待']]);caption='绿色：保留；棕色：同脊线抑制；灰色：尚未处理。坐标及判断来自真实候选。';
 }
 el.querySelector('[data-output="values"]').innerHTML=values;el.querySelector('[data-output="caption"]').textContent=caption;
 el.querySelector('[data-action="seek"]').value=t;el.querySelector('[data-output="progress"]').textContent=`${t} / ${maxT[kind]}`;el.querySelector('[data-action="play"]').textContent=s.playing?'暂停计算':'播放计算';
 if(kind==='convolution')el.querySelector('[data-output="threshold"]').textContent=s.threshold;
 if(kind==='kernels')el.querySelector('[data-action="hypothesis"]').value=t;
}
for(const el of document.querySelectorAll('[data-lab]')){
 const kind=el.dataset.lab,s=get(kind);el.querySelector('[data-action="seek"]').max=maxT[kind];
 el.querySelector('[data-action="seek"]').oninput=e=>{s.t=+e.target.value;s.playing=false;render(kind)};
 el.querySelector('[data-action="reset"]').onclick=()=>{s.t=0;s.playing=false;render(kind)};
 el.querySelector('[data-action="play"]').onclick=()=>{if(document.body.classList.contains('still'))document.querySelector('#motion').click();if(s.t>=maxT[kind])s.t=0;s.playing=!s.playing;stamp=performance.now();render(kind)};
 el.querySelector('[data-action="step"]').onclick=()=>{s.playing=false;s.t=Math.min(maxT[kind],s.t+1);render(kind)};
 if(kind==='convolution'){el.querySelector('[data-action="threshold"]').oninput=e=>{s.threshold=+e.target.value;render(kind)};canvas(kind).onclick=e=>{const r=e.currentTarget.getBoundingClientRect(),x=(e.clientX-r.left)*880/r.width,y=(e.clientY-r.top)*370/r.height;if(x>=15&&x<270&&y>=44&&y<299){s.t=Math.floor((y-44)/255*64)*64+Math.floor((x-15)/255*64);s.playing=false;render(kind)}}}
 if(kind==='kernels'){const sel=el.querySelector('select');D.trail.frames.forEach((f,i)=>{const op=document.createElement('option');op.value=i;op.textContent=i===0?'点源核':`#${i+1} L=${fmt(f.length,1)} φ=${fmt(f.angle,2)}°`;sel.append(op)});sel.onchange=e=>{s.t=+e.target.value;s.playing=false;render(kind)}}
 render(kind);
}
let last=performance.now(),debt=0;
function tick(now){const elapsed=Math.min(.1,(now-last)/1000);last=now;const active=document.querySelector('.slide.active [data-lab]');if(active){const kind=active.dataset.lab,s=get(kind);if(s.playing&&!document.body.classList.contains('paused')&&!document.body.classList.contains('still')){debt+=elapsed;const duration={background:2,convolution:.003,photometry:.14,kernels:.22,dedup:.85}[kind];if(debt>=duration){const n=Math.floor(debt/duration);debt%=duration;s.t=Math.min(maxT[kind],s.t+n);if(s.t>=maxT[kind])s.playing=false;render(kind)}}else debt=0}else debt=0;requestAnimationFrame(tick)}requestAnimationFrame(tick);
window.scienceRender=()=>{const el=document.querySelector('.slide.active [data-lab]');if(el)render(el.dataset.lab)};
document.querySelector('#replay').addEventListener('click',()=>{const el=document.querySelector('.slide.active [data-lab]');if(el){const s=get(el.dataset.lab);s.t=0;s.playing=true;render(el.dataset.lab)}});
window.__science={state,render,maxT,convAt,kernelAt,dedupStep,seek:(kind,t)=>{get(kind).t=t;get(kind).playing=false;render(kind)},verify:()=>{
 let ce=0;for(let i=0;i<4096;i++)ce=Math.max(ce,Math.abs(convAt(i)-D.point.filtered[i]));
 let ke=0;for(const f of D.trail.frames)ke=Math.max(ke,Math.abs(kernelAt(f,32,32)-f.z));
 const pe=Math.abs(cum.at(-1)-D.photometry.flux);
 return {convolutionMaxAbsError:ce,kernelCenterMaxAbsError:ke,photometryAbsError:pe,photometryMInst:-2.5*Math.log10(cum.at(-1)),thresholdCounts:D.point.thresholds.map(x=>({threshold:x.threshold,count:x.peaks.length})),dedupFinalCount:dedupStep(D.dedup.candidates.length).kept.length,traceHypotheses:D.trail.frames.length};
}};
let printBackup=null;
addEventListener('beforeprint',()=>{printBackup=JSON.parse(JSON.stringify(state));for(const kind of Object.keys(maxT)){get(kind).playing=false;get(kind).t=maxT[kind];render(kind)}});
addEventListener('afterprint',()=>{if(printBackup){for(const [kind,s] of Object.entries(printBackup)){state[kind]=s;render(kind)}printBackup=null}});
})();
'''
