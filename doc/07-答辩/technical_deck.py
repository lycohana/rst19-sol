"""Build the offline defense deck from retained evidence and Rust numerical traces.

Run: python doc/07-答辩/build_deck.py
All illustrative SVGs are explicitly distinguished from measurements.
"""
from pathlib import Path
import json, math, hashlib, subprocess, html, shutil, re
from science_slides import revise, CSS as SCIENCE_CSS, JS as SCIENCE_JS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVID = ROOT / 'doc/05-论文答辩/latex-论文/evidence'
records = []
for name in ['frame_A.png','frame_B.png']:
    p=HERE/'figs'/name
    records.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
motion=ROOT/'doc/05-论文答辩/latex-论文/figs/research/moving_contact.png'
records.append({'path':motion.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(motion.read_bytes()).hexdigest()})
shutil.copy2(motion,HERE/'figs/moving_contact.png')
def read(rel):
    p = ROOT / rel
    records.append({'path': rel, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    return json.loads(p.read_text(encoding='utf-8-sig'))
c = read('doc/05-论文答辩/latex-论文/evidence/computed_results.json')
g = read('tmp/uni_A3/glrt.json')
t = read('tmp/flag_on/02-rst19-sources/trail_catalog.json')
h = read('tmp/glrt_val/h0_correlated.json')
w = read('tmp/glrt_val/h0_white.json')
for rel in ['backend/src/detection.rs','backend/src/filtering.rs','backend/src/measurement.rs',
            'backend/src/empirical_photometry.rs','backend/src/glrt.rs','backend/src/trail_catalog.rs',
            'backend/src/source_audit.rs','desktop/src/ui.rs','desktop/src/task_suites.rs',
            'backend/src/sequence_registration.rs','backend/src/sequence_assignment.rs',
            'backend/src/sequence_motion.rs','backend/src/sequence_motion_recovery.rs',
            'tools/glrt_validate/ref_glrt.py']:
    p=ROOT/rel
    records.append({'path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})

BLUE='#2764e7'; TEAL='#087f83'; ORANGE='#b65a21'; RED='#c34850'; MUTED='#68768b'
def svg(content, width=760, height=360, label='原理示意图'):
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{label}">{content}</svg>'
def txt(x,y,s,size=18,color=MUTED,anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{html.escape(str(s))}</text>'
def line(x1,y1,x2,y2,color='#d3dce6',dash=''):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" {dash}/>'
def path(points,color=BLUE,extra=''):
    return '<path d="M '+' L '.join(f'{x:.2f},{y:.2f}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="3" {extra}/>'
def legend(items):
    return '<div class="legend">'+''.join(f'<span><i style="background:{co}"></i>{label}</span>' for label,co in items)+'</div>'
def block(n,title,body):
    return f'<div class="explain reveal"><span class="ordinal">{n}</span><div><h3>{title}</h3><p>{body}</p></div></div>'
def formula(s,cap=''):
    return f'<div class="formula">{s}</div><p class="micro">{cap}</p>'
def pill(s,kind='code'):
    return f'<span class="pill {kind}">{s}</span>'
def chart_bars(rows,maxv,unit='',height=350):
    out=''
    for i,(label,v,co) in enumerate(rows):
        y=35+i*78
        out+=txt(10,y+21,label,19,'#273b52')
        out+=f'<rect x="200" y="{y}" width="410" height="32" rx="3" fill="#e9eef5"/>'
        out+=f'<rect class="bar" x="200" y="{y}" width="{max(2,410*v/maxv):.2f}" height="32" rx="3" fill="{co}"/>'
        out+=txt(625,y+23,f'{v:.4f} {unit}',18,co)
    return svg(out,780,height,'留存结果条形图')
def field():
    out='<rect width="700" height="390" rx="10" fill="#091626"/>'
    for i in range(105):
        x=(i*97+33)%680+10; y=(i*i*47+23)%370+10
        out+=f'<circle cx="{x}" cy="{y}" r="{0.7+(i%5)*0.25}" fill="#cbdce9" opacity="{.25+(i%6)*.11}"/>'
    for x,y in [(125,130),(240,245),(515,265),(320,75)]:
        out+=f'<circle cx="{x}" cy="{y}" r="6" fill="#a5dff2" opacity=".25"/><circle cx="{x}" cy="{y}" r="2.7" fill="white"/><circle class="ring" cx="{x}" cy="{y}" r="19" fill="none" stroke="#73d4cf" stroke-width="2"/>'
    out+=txt(20,366,'合成星场 · 滤波与局部极大提取示意',17,'#9cb4c9')
    return svg(out,700,390,'合成星场上的检测示意，非实测图像')
def background():
    out=''
    for j in range(7):
        for i in range(12):
            v=int(228-3*i-1.7*j)
            out+=f'<rect x="{30+i*29}" y="{42+j*29}" width="28" height="28" fill="rgb({v-15},{v},{min(255,v+15)})"/>'
    out+=txt(30,30,'图像分块',19,BLUE)+txt(30,282,'局部背景 B(x,y) 与噪声 σ(x,y)',18)
    out+=line(401,147,448,147,BLUE)+txt(470,60,'响应标准化',22,BLUE)
    out+=path([(470+i*2,155-70*math.exp(-((i-40)/9)**2)) for i in range(115)])
    out+=line(470,188,710,188,ORANGE,'stroke-dasharray="5 5"')+txt(475,216,'阈值用于提案',18,ORANGE)
    out+=txt(475,258,'响应尺度 × 局部噪声修正',18)
    return svg(out,760,320,'背景分块和响应标准化示意')
def aperture():
    out='<defs><radialGradient id="psf"><stop stop-color="#fff"/><stop offset=".15" stop-color="#badff7"/><stop offset="1" stop-color="#132a41"/></radialGradient></defs><rect x="30" y="5" width="420" height="330" rx="8" fill="#0a192b"/>'
    out+='<circle cx="240" cy="170" r="100" fill="url(#psf)"/>'
    for r,co in [(62,TEAL),(112,BLUE),(145,BLUE)]:
        out+=f'<circle class="ring" cx="240" cy="170" r="{r}" fill="none" stroke="{co}" stroke-width="3"/>'
    out+=line(240,170,500,105,TEAL)+txt(510,109,'测量孔径',23,TEAL)
    out+=line(360,215,500,250,BLUE)+txt(510,257,'局部背景环',23,BLUE)
    out+=txt(48,318,'几何示意 · 半径非按比例',15,'#b1c4d5')
    return svg(out,760,350,'孔径与背景环示意')
def kernel():
    return '<canvas id="kernel" width="760" height="300" aria-label="点源与线段卷积核演示"></canvas><div class="kernel-controls"><label>线段长度 <input id="length" type="range" min="0" max="120" value="65"><output id="lengthOut">65 px</output></label><label>方向 φ <input id="angle" type="range" min="0" max="179" value="35"><output id="angleOut">35°</output></label></div>'
def thresholds():
    out=''
    for y,val in [(300,4),(220,5),(140,6),(60,7)]:
        out+=line(70,y,685,y)+txt(50,y+6,val,16,anchor='end')
    for x,label in [(85,'1'),(375,'10'),(665,'100')]: out+=txt(x,330,label,18,anchor='middle')
    coarse=g['refinement']['thresholds_coarse']; fine=g['thresholds']
    for data,co in [(coarse,TEAL),(fine,BLUE)]:
        pts=[(85+i*290,300-(data[f'fa_{fa}']-4)*80) for i,fa in enumerate([1,10,100])]
        out+=path(pts,co,'class="draw"')
        for (x,y),fa in zip(pts,[1,10,100]): out+=f'<circle cx="{x}" cy="{y}" r="5" fill="{co}"/>'+txt(x,y-14,f'{data[f"fa_{fa}"]:.3f}',17,co,'middle')
    out+=txt(72,23,'阈值 u（标准化响应单位）',18)+txt(435,359,'目标虚警数 / 帧 · 对数轴',18)
    return svg(out,760,380,'同一次原型运行的粗搜索与精化阈值')
def corr():
    out=''
    for j,v in enumerate([0,200,400,600]):
        y=300-v*.4; out+=line(70,y,710,y)+txt(54,y+5,v,15,anchor='end')
    for i,row in enumerate(h['boxcar']):
        x=118+i*155; obs=row['peaks_above_4sigma_mean']; pred=row['prediction_neff']
        for xx,val,co in [(x,obs,BLUE),(x+44,pred,ORANGE)]:
            out+=f'<rect class="bar" x="{xx}" y="{300-val*.4}" width="32" height="{val*.4}" fill="{co}"/>'+txt(xx+16,290-val*.4,f'{val:.0f}',17,co,'middle')
        out+=txt(x+38,330,f'l = {row["l"]:g} px',17,anchor='middle')
    out+=txt(70,32,'4σ 以上局部极大数 / 帧',19)
    return svg(out,760,355,'相关噪声实验实测假峰与被否定的预测')

slides=[]
def add(ch,title,sub,body,claim,source,kind='code',dark=False,notes=''):
    i=len(slides)+1
    slides.append(f'''<section class="slide {'dark' if dark else ''}" data-title="{html.escape(title)}" data-chapter="{ch}" aria-label="第{i}页：{title}">
    <header><div class="kicker">RST19 <span>/</span> {ch}</div>{pill({'code':'代码实现','measured':'留存实测','illustration':'原理示意','boundary':'结论边界'}[kind],kind)}</header>
    <h1>{title}</h1><p class="subtitle">{sub}</p><div class="content">{body}</div>
    <div class="claim">{claim}</div><footer><span>{source}</span><span>{i:02d} / 20</span></footer>
    <aside class="speaker"><h2>来源与讲解备注</h2><p>{notes or claim}</p><p class="source-detail">{source}</p></aside></section>''')

add('研究问题','从像素结构到可核查的测量结果',
    '面向仪器参数不完备条件的天基星图检测、经验测光与拖影分析',
    '<div class="cover-grid"><div><div class="cover-line">恒星检测<br><em>与经验星等标定</em></div><p class="lead">以真实程序说明处理方法<br>以留存证据界定科学结论</p><div class="cover-topics"><span>01 点源目录</span><span>02 经验星等</span><span>03 拖影目录</span></div></div><div class="cover-image"><img src="figs/frame_A.png" alt="事件 A 的真实图像裁剪"><span class="vertical-label">真实观测 / 事件 A</span><div class="image-meta">594 px <small>模型估计下界</small><br>96.61° <small>像平面方向</small></div></div></div>',
    '研究对象为图像中的信号结构；物理身份需要外部星表或星历的进一步核验。',
    '观测裁剪：figs/frame_A.png · 方法核对：当前 Rust 源码 · 证据快照：2026-10-20',dark=True,kind='boundary')

add('01 / 观测与任务','同一星图包含不同的信号形态',
    '点源、连续拖影与非均匀短条，需要不同的测量模型和验证方式。',
    '<div class="morph-grid"><div class="specimen">'+field()+'<h3>点源</h3><p>局部峰值 → 孔径通量 → 质量证据</p><span class="micro">合成示意</span></div><div class="specimen"><div class="photo"><img src="figs/frame_A.png" alt="真实连续拖影 A"></div><h3>连续拖影 A</h3><p>方向与长度搜索 → 沿脊线检验</p><span class="micro">真实裁剪 · 第 1 帧</span></div><div class="specimen"><div class="photo"><img src="figs/frame_B.png" alt="真实非均匀短条 B"></div><h3>非均匀结构 B</h3><p>单帧形态证据 → 多历元运动验证</p><span class="micro">真实裁剪 · 第 15 帧</span></div></div>',
    '形态分类、算法轨迹与天体身份属于不同证据层级。','图像：figs/frame_A.png / frame_B.png · backend/src/detection.rs / glrt.rs',kind='boundary')

add('01 / 程序架构','两条处理分支，共享同一观测输入',
    '图形界面通过 worker 调用 Rust 库；点源测光与附加拖影目录保留各自的输出。',
    '<div class="architecture"><div class="input-node">FITS 原始图像<small>像素 · 掩膜 · 头信息</small></div><div class="branch"><div class="flow-row reveal"><b>点源分支</b><span>背景 / 噪声</span><i>→</i><span>峰值提案</span><i>→</i><span>测量 / 质量</span><i>→</i><strong>source_catalog.csv</strong></div><div class="flow-row reveal"><b>星等分支</b><span>星表匹配</span><i>→</i><span>重测孔径</span><i>→</i><span>稳健 WLS</span><i>→</i><strong>calibration.json</strong></div><div class="flow-row teal reveal"><b>拖影分支</b><span>白化 / 核库</span><i>→</i><span>GLRT 搜索</span><i>→</i><span>形态 / 去重</span><i>→</i><strong>trail_catalog.json</strong></div></div></div>'+legend([('点源及测光',BLUE),('附加拖影目录',TEAL)]),
    '拖影目录由 source_audit 的库层入口接入，开关跟随 source 参数下发。','source_audit.rs:178–209 · trail_catalog.rs:107–184 · desktop/src/ui.rs:717–720',notes='执行关系：source_audit 先写点源产物，再调用 trail_catalog::report；测光套件在后续步骤使用源表。图中连接表示数据依赖，不表示并行执行。')

add('02 / 点源检测','背景估计使局部响应具有可比尺度',
    '程序估计背景与噪声图，再对滤波响应作稳健标准化与局部噪声修正。',
    '<div class="split"><div class="visual">'+background()+legend([('滤波响应',BLUE),('提案阈值',ORANGE)])+'</div><div>'+block('1','背景与噪声','默认背景块大小为 128 px；稳健统计抑制亮源对尺度估计的影响。')+block('2','滤波响应标准化','响应裁剪尺度与局部噪声比共同构成分母，噪声比设有 0.25 下限。')+formula('R′ = (R − B<sub>R</sub>) / [s<sub>R</sub> · max(σ/med σ, 0.25)]','R 为滤波响应；B_R、s_R 为响应背景与尺度。')+'</div></div>',
    '提案响应是工程标准化量；其数值不能直接解释为整幅星图的全局虚警概率。','backend/src/filtering.rs:156–194 · detection.rs::DetectionOptions',kind='code')

add('02 / 点源检测','候选提案与质量筛选分阶段执行',
    '默认 Gaussian 生成局部峰值；hybrid 合并 Gaussian 与两个高斯差分（DoG）尺度的提案。',
    '<div class="split"><div class="visual">'+field()+'</div><div>'+block('1','形成响应图','Gaussian 平滑后提取局部极大；hybrid 额外计算 DoG 响应。')+block('2','合并与近邻抑制','合并邻近候选，记录 proposal_methods 和 proposal_scales。')+block('3','逐源测量','在候选位置计算通量、形态、支持像素数与质量标志。')+'<div class="small-callout">当前库默认：Gaussian；局部去混叠默认关闭。具体运行以参数记录为准。</div></div></div>',
    '算法提供多种提案模式；模式可用性与某次运行实际启用的模式必须分别表述。','detection.rs:14–69 · filtering.rs::response_proposals / merge_proposals',kind='illustration')

add('02 / 测光','由孔径内的净信号计算仪器星等',
    '经验测光程序在匹配位置重新测量；背景环用于估计局部背景及其不确定度。',
    '<div class="split"><div class="visual">'+aperture()+legend([('测量孔径',TEAL),('背景环',BLUE)])+'</div><div>'+formula('F = Σ a<sub>i</sub> (I<sub>i</sub> − b)','a_i 为像素在孔径内的面积权重，b 为局部背景。')+formula('m<sub>inst</sub> = −2.5 log<sub>10</sub> F','当前经验测光代码使用孔径通量 F，未在此处除以曝光时间。')+formula('σ<sub>m</sub> ≈ 1.0857 · σ<sub>F</sub> / F','正通量条件下的一阶误差传播。')+'<p>曝光口径、孔径与背景定义必须随零点共同保留。</p></div></div>',
    '源目录通量与经验测光重测通量可以不同；比较前需核对测量口径。','empirical_photometry.rs::measure / aperture_coverage · :1986–1992',kind='code')

add('02 / 质量控制','质量通过是筛选结果，不等于恒星真值',
    '当前点源质量函数使用 18 项拒绝标志；通量信噪比与形态证据共同参与判断。',
    '<div class="split"><div class="funnel"><div class="funnel-a reveal"><small>冻结源表 · 候选</small><b>66 469</b></div><div class="funnel-b reveal"><small>质量通过</small><b>27 552</b></div><div class="fraction">通过率 <strong>41.45%</strong></div></div><div>'+block('A','信号证据','LOW_FLUX_SNR、NON_POSITIVE_FLUX、INSUFFICIENT_PSF_SUPPORT 等。')+block('B','形态证据','NARROW、BROAD、ELONGATED、UNRESOLVED_BLEND 等。')+block('C','数据异常','SATURATED、NEGATIVE_OVERFLOW、CODE_PATTERN、MASKED 等。')+'<div class="small-callout">标志具有误拒与漏拒的可能；通过率不表示精确率或完备度。</div></div></div>',
    '该计数来自留存源表汇总，本次幻灯片重制未重新执行全帧检测。','computed_results.json::source_count / quality_count · measurement.rs:6–31',kind='measured')

peaks=c['peak_counts_all']
add('02 / 数据诊断','异常高码值的集中分布需要显式处理',
    '冻结源表中，3990–3993 DN 附近出现候选峰值堆积；经验测光配置采用 3990 DN 峰值上限。',
    '<div class="split"><div class="visual">'+svg(''.join(f'<rect class="bar" x="{95+i*145}" y="{300-v*.60}" width="76" height="{v*.60}" fill="{BLUE}"/>'+txt(133+i*145,285-v*.60,v,24,BLUE,'middle')+txt(133+i*145,333,k,22,anchor='middle') for i,(k,v) in enumerate(peaks.items()))+txt(55,30,'候选峰值计数',20)+txt(575,365,'峰值 / DN',18),760,390,'冻结源表中异常高码值的峰值计数')+'</div><div>'+block('1','可测事实','四个码值对应候选峰值计数共 880 个；图中展示逐码值计数。')+block('2','程序处理','检测保留重复码值证据；经验测光以峰值上限及异常像素规则排除受影响样本。')+block('3','解释边界','该上限是当前数据的经验排除条件。缺少仪器读出说明时，不能直接确定物理满阱或饱和机制。')+'</div></div>',
    '统计图说明异常分布，仪器物理原因仍需要独立标定。','computed_results.json::peak_counts_all · calibration.json::settings.peak_ceiling · detection.rs::repeated_high_code_values',kind='measured')

add('03 / 经验标定','零点与色项属于两种不同的拟合模型',
    'Gaia 参考星等与仪器星等的差值，通过稳健筛选和加权最小二乘求解。',
    '<div class="model-columns"><div class="model reveal"><span class="model-id">M0 / 无色项</span>'+formula('g<sub>emp</sub> = m<sub>inst</sub> + ZP')+'<div class="metric">20.4232<small>± 0.0005 · 形式误差</small></div><p>18 985 个拟合内点<br>颜色未知的目标使用该模型</p></div><div class="model reveal"><span class="model-id">M1 / 含色项</span>'+formula('g<sub>emp</sub> = m<sub>inst</sub> + ZP + C(BP−RP)')+'<div class="metric">20.5008<small>± 0.0015 · 形式误差</small></div><p>C = −0.08520 ± 0.00157<br>18 979 个拟合内点</p></div></div><div class="method-strip">权重 w = 1/(σ²<sub>inst</sub> + σ²<sub>G</sub>) <span>→</span> 稳健剔除 <span>→</span> WLS 与协方差</div>',
    '零点随模型定义改变；20.4232 不能与含色项的拟合结果合并为同一个解。','computed_results.json::photometry · empirical_photometry.rs::fit_calibration / robust_calibration',kind='measured')

p=c['photometry']
add('03 / 独立检验','形式误差、拟合散度与留出误差回答不同问题',
    '五折检验覆盖全部具备颜色信息的留出样本；测试残差不再按全局内点预筛选。',
    '<div class="split"><div class="visual">'+chart_bars([('M0 零点形式误差',p['zero_point']['sigma'],TEAL),('M0 样本内散度',p['zero_point']['in_sample_scatter_mag'],BLUE),('M0 全样本留出',p['holdout']['order0']['holdout_sigma_mag'],ORANGE),('M1 全样本留出',p['holdout']['order1']['holdout_sigma_mag'],ORANGE)],.35,'mag',350)+'<p class="micro">条形长度为同一线性尺度；形式误差条最短，不能解释为单目标精度。</p></div><div>'+block('1','参数估计稳定性','零点形式误差约 0.0005 mag，条件于权重和模型。')+block('2','拟合内样本一致性','M0 内点散度约 0.142 mag，受到筛选条件影响。')+block('3','样本外预测','M0 留出约 0.3265 mag；M1 约 0.3258 mag，降低约 0.00077 mag。')+'</div></div>',
    '该留出结果支持经验预测误差的报告，尚不足以证明色项改进具有统计显著性。','computed_results.json::photometry.holdout · empirical_photometry.rs::cross_validate',kind='measured')

add('04 / 拖影模型','点源可表示为线段核的零长度极限',
    '点扩散函数（PSF）描述单位点源的成像形状；线段与其卷积生成拖影核。',
    '<div class="split"><div class="visual kernel-visual">'+kernel()+legend([('PSF 核',BLUE),('线段核 × PSF',TEAL)])+'</div><div>'+formula('K<sub>L,φ</sub> = PSF ∗ segment(L,φ)','每个核归一化为单位总流量；L → 0 时退化为点源。')+block('L','长度搜索','目录固定使用 16–1024 px 的 7 个长度。')+block('φ','方向搜索','24 个方向加点核，共 7 × 24 + 1 = 169 个粗假设。')+'<div class="small-callout">拖动长度和方向观察核形态。此图为解析原理演示，未执行生产检测。</div></div></div>',
    '核函数表示对象模型；统一核族不表示全部观测结构均符合均匀线段假设。','trail_catalog.rs:12–13 · glrt.rs::streak_kernel / Library',kind='illustration')

add('04 / 检测统计','白化与匹配滤波构成 GLRT 响应',
    '广义似然比检验（GLRT）在核族中选择最大响应；白化用于降低噪声的空间相关性。',
    '<div class="stat-flow"><div class="stat-node reveal"><b>背景扣除</b><p>d = I − B</p><small>观测像素</small></div><i>→</i><div class="stat-node reveal"><b>一致白化</b><p>d̃ = Wd<br>K̃ = WK</p><small>抑制噪声相关</small></div><i>→</i><div class="stat-node reveal"><b>核族响应</b><p>z<sub>θ</sub> = ⟨K̃<sub>θ</sub>, d̃⟩<br> / (σ<sub>w</sub> ‖K̃<sub>θ</sub>‖)</p><small>相同归一化口径</small></div></div><div class="model-columns compact"><div>'+formula('Λ = max<sub>θ</sub> z<sub>θ</sub>','Λ 用于提取检测候选。')+'</div><div>'+formula('LR = z<sub>best line</sub> − z<sub>point</sub>','代码中的 LR 是响应差统计量；正值支持线状形态。')+'</div></div>',
    '生产目录使用 Line 引擎；PSF 平滑后的方向积分与前缀和避免对每个线核重复逆 FFT。','glrt.rs:1–28 · mf_z_freq / line_z_map · trail_catalog.rs:112–120',notes='当前代码写成 ⟨K,W²d⟩/(σ_w‖WK‖)，与对称零相位 W 下的白化内积等价。LR 为实现中的变量名，不能直接解读为卡方分布的标准似然比统计量。')

add('04 / 阈值标定','阈值随搜索规模与精化次数改变',
    '当前 Rust 路径以合成噪声场拟合 Λ 尾部，估计相关面积，并为候选精化追加多重试验修正。',
    '<div class="split"><div class="visual">'+thresholds()+legend([('粗搜索阈值',TEAL),('精化后阈值',BLUE)])+'</div><div>'+formula('u = μ + β ln(N<sub>eff</sub> · m / N<sub>FA</sub>)','本式对应当前 Rust 的标定方式；μ、β 的定义依拟合对象而定。')+block('1','固定一次运行','PSF = 1.93 px；169 个粗假设；噪声标定图 1024²。')+block('2','支付精化代价','该运行 m = 4312；阈值增加 1.6547 个响应单位。')+block('3','解释统计口径','N_FA 是噪声模型下的目标期望虚警数，不是实图中已认证的误检数。')+'</div></div>',
    '原型阈值 7.0904 与目录阈值 7.0884 来自不同 PSF 配置，应分别保留。','快照：tmp/uni_A3/glrt.json · glrt.rs:2050–2139',kind='measured')

add('04 / 独立验证','相关噪声实验否定简单的 N/l² 假峰预测',
    '独立 NumPy 参照在固定 4σ 条件下比较局部极大计数；每个相关尺度保留 8 次重复。',
    '<div class="split"><div class="visual">'+corr()+legend([('实测局部极大数',BLUE),('被否定的 N/l² 预测',ORANGE)])+'</div><div>'+block('1','实际观测','相关长度从 1 增至 5 px，假峰均值仍由约 539 缓慢降至 358。')+block('2','被否定的预测','简单缩减独立样本数，预测约由 531 降至 21，明显偏低。')+block('3','验证范围','该实验属于独立参照的相关场测试；其核族、白化与生产目录配置不同。')+'</div></div>',
    '局部峰数、像素超越数与全帧最大值分布必须分别定义，避免混用统计公式。','tmp/glrt_val/h0_correlated.json::boxcar · tools/glrt_validate/h0_calibration.py',kind='measured')

profile=[28,22,19,24,62,53,347,669,1231,1252,795,566,236,176,132,45,39,36,33,17,14]
profile_svg=svg(''.join(line(65,280-v*.16,710,280-v*.16)+txt(52,286-v*.16,v+25,15,anchor='end') for v in [0,500,1000])+path([(65+i*31,280-(v-25)*.16) for i,v in enumerate(profile)],ORANGE,'class="draw"')+txt(65,29,'事件 B · 沿轴像素值 / DN',21)+''.join(txt(65+i*31,309,i*4,16,anchor='middle') for i in [0,5,10,15,20])+txt(340,346,'沿轴相对位置 / px（PA≈132°）',17,'#68768b','middle')+txt(535,64,'峰值 1252 DN',18,ORANGE),760,365,'历史沿轴像素采样记录，非当前 uniform 函数的输入序列')
add('04 / 适用范围','沿脊线检验限制均匀线段模型的适用性',
    '在未白化的 PSF 平滑图上采样，避免白化负瓣干扰形态检验。',
    '<div class="split"><div class="visual">'+profile_svg+legend([('B 的像素剖面记录',ORANGE)])+'</div><div>'+formula('duty ≥ 0.85 且 scatter ≤ 0.25','两项同时满足，uniform 才为 true。')+'<table><tr><th>对象 / 配置</th><th>duty</th><th>scatter</th></tr><tr><td>A · 固定 PSF 原型</td><td>0.9867</td><td>0.0606</td></tr><tr><td>A · 产品目录</td><td>0.9900</td><td>0.0576</td></tr></table><p>实现先筛选 ≥ 15% 峰值的样本，再以中位数定义 duty；scatter 使用超过半中位数的样本。</p><p class="micro">代码按阈值收集样本，未强制保留连续区间；不能宣称已检验全轴连续性。</p></div></div>',
    '当前均匀性标志是经验判据；非均匀团块链的检出仍存在局限。','glrt.rs:2698–2752 · uni_A3/glrt.json · 原型实测-A与B-20261020.md §9',kind='boundary')

add('05 / 实测结果','事件 A：方向可测，长度保留下界解释',
    '同一真实裁剪，在独立原型运行与产品目录中得到近似一致的长线方向和模型长度。',
    '<div class="result-grid"><div class="tall-image"><img src="figs/frame_A.png" alt="事件 A 真实观测"><div class="ridge-guide"></div><span>真实裁剪 · 像平面</span></div><div><div class="hero-metrics"><div>594.0<small>长度 / px</small></div><div>96.61°<small>方向 φ</small></div></div><table><tr><th>量</th><th>原型 PSF 1.93</th><th>目录 PSF 2.1513</th></tr><tr><td>Λ 响应</td><td>21 562.08</td><td>21 825.99</td></tr><tr><td>duty</td><td>0.9867</td><td>0.9900</td></tr><tr><td>scatter</td><td>0.0606</td><td>0.0576</td></tr></table><div class="small-callout">均匀长脊沿轴位置可漂移。报告重点为垂直于脊线的位置与方向；594 px 为模型估计的可见长度下界。</div><p class="micro">φ 在像素索引坐标中自 +x 向 +y 计，模 180°；并非直接给出的天球位置角。</p></div></div>',
    'Λ 是模型下的标准化响应；不能据此宣称二万 σ 的天体身份置信度。','tmp/uni_A3/glrt.json::detections[0] · tmp/flag_on/02-rst19-sources/trail_catalog.json',kind='measured',dark=True)

add('05 / 跨帧证据','单帧形态与多历元运动提供互补证据',
    '配准消除视场整体运动；门控分配建立关联；原图支持与时间拟合进一步约束轨迹。',
    '<div class="split"><div class="visual"><img class="motion-image" src="figs/moving_contact.png" alt="项目论文留存的多帧运动结构裁剪"><p class="micro">论文留存的序列裁剪；各分支可能重复描述同一观测结构。</p></div><div>'+block('1','统一坐标','普通路径使用仿射配准；快速路径主要扣除累积平移，两者误差口径不同。')+block('2','关联与时间拟合','半径门控分量内最小代价分配；读取实际时间拟合运动，注明 px/s 或参考帧间隔。')+block('3','原图复核','预测 ROI 必须获得原图支持，补入观测时保留 observation_source。')+'</div></div>',
    '算法分类为 moving 说明时序证据满足当前规则，物理身份仍需独立核验。','sequence_registration.rs · sequence_assignment.rs · sequence_motion.rs · sequence_motion_recovery.rs',kind='code')

on=ROOT/'tmp/flag_on/02-rst19-sources/source_catalog.csv'
off=ROOT/'tmp/flag_off/02-rst19-sources/source_catalog.csv'
same=on.exists() and off.exists() and on.read_bytes()==off.read_bytes()
if not same: raise RuntimeError('Expected retained on/off source catalogs to match')
for pth in [on,off]: records.append({'path':pth.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(pth.read_bytes()).hexdigest()})
add('06 / 工程实现','共线去重将重复候选整理为附加目录',
    '产品目录先筛选 uniform 且 L ≥ 40 px 的长候选，按 Λ 排序后进行同脊线抑制。',
    '<div class="split"><div class="visual">'+svg('<rect x="30" y="20" width="330" height="290" rx="8" fill="#e8eef6"/>'+line(115,45,270,285,TEAL)+''.join(f'<circle class="ring" cx="{115+i*31}" cy="{45+i*48}" r="12" fill="white" stroke="{BLUE}" stroke-width="3"/>' for i in range(6))+txt(50,342,'沿同一脊线的重复候选',19)+txt(391,165,'→',42,TEAL)+'<rect x="455" y="20" width="270" height="290" rx="8" fill="#e8f4f2"/>'+line(510,45,665,285,TEAL)+'<circle cx="572" cy="141" r="14" fill="white" stroke="#087f83" stroke-width="4"/>'+txt(470,342,'保留最高 Λ 的代表',19),760,365,'共线候选去重示意')+'</div><div>'+block('1','几何门控','轴方向差不超过 3°，候选至保留脊线的垂距小于 3 px。')+block('2','留存运行结果','1064 个长候选，经均匀性筛选与共线去重，最终输出 3 条目录记录。')+block('3','开关产物对照','启用 21 个文件，关闭 20 个文件；留存 source_catalog.csv 已逐字节复核一致。')+'<p class="micro">3 条目录记录为算法输出，不能直接解释为 3 个已确认空间目标。</p></div></div>',
    'GUI 默认启用拖影目录；开关由 --trail-catalog=on|off 下发到点源分析入口。','trail_catalog.rs:86–99 / 125–151 · desktop/src/ui.rs:1561 · 留存 on/off 产物',kind='measured')

add('07 / 结论','以测量定义与证据范围约束研究结论',
    '现有实现完成分层源检测、经验测光及附加拖影目录；后续检验需围绕模型失配与仪器误差展开。',
    '<div class="conclusion-grid"><div class="conclusion reveal"><span>01 / 点源</span><h2>候选与质量分层</h2><p>保留逐源通量、形态与拒绝标志，形成可追溯目录。</p><small>待补：独立标签下的精确率与完备度</small></div><div class="conclusion reveal"><span>02 / 星等</span><h2>经验标定与留出检验</h2><p>无色项零点 20.4232；全样本留出散度约 0.3265 mag。</p><small>待补：增益、平场与仪器响应标定</small></div><div class="conclusion reveal"><span>03 / 拖影</span><h2>形态假设与附加目录</h2><p>事件 A 获得方向与模型长度；共线去重保留目录代表。</p><small>待补：端点估计、非均匀核与虚警迁移检验</small></div></div><div class="closing">可复算的方法 <span>→</span> 可核查的结果 <span>→</span> 有边界的结论</div>',
    '科学贡献以实际实现和独立检验为依据；本页不主张已完成天体物理身份认证。','方法与实测快照：evidence/deck-evidence.json · 核对说明：设计与核验说明.md',kind='boundary',dark=True)

add('附录 / 证据索引','每一项定量结论均保留来源',
    '主讲页使用精简文案；来源抽屉与本页用于答辩追问、参数核对及复算定位。',
    '<div class="evidence-grid"><div><h3>01 / 点源与经验测光</h3><p>computed_results.json<br>calibration.json</p><small>冻结源表汇总、零点、色项、五折留出</small></div><div><h3>02 / 固定 PSF 原型</h3><p>tmp/uni_A3/glrt.json</p><small>PSF 1.93 px；169 粗假设；精化 4312 次</small></div><div><h3>03 / 产品拖影目录</h3><p>tmp/flag_on/02-rst19-sources/<br>trail_catalog.json</p><small>PSF 2.1513 px；3 条记录；阈值 7.0884</small></div><div><h3>04 / 独立参照实验</h3><p>tmp/glrt_val/h0_correlated.json<br>tools/glrt_validate/</p><small>NumPy 相关噪声实验；与生产配置区分</small></div></div><div class="audit-note">本次工作：当前源码核对、真实裁剪数值导出、浏览器逐项复算与页面核验。<br>全帧标定与完整序列结果采用留存记录；局部数值实验单独标明配置。</div>',
    '运行结果与当前源码分别记录 SHA-256；留存结果不冒称由本次源码重新运行生成。','完整路径与 SHA-256：evidence/deck-evidence.json · 按 N 查看每页来源与备注',kind='boundary')

slides,science_trace=revise(slides,g)
assert len(slides)==20

CSS=r'''
:root{--ink:#142b42;--muted:#586c80;--blue:#2764e7;--teal:#087f83;--paper:#f6f8fb;--line:#dbe3ec;--orange:#b65a21}*{box-sizing:border-box}html,body{margin:0;height:100%;overflow:hidden;background:#081321;font-family:"Microsoft YaHei","Segoe UI",sans-serif;color:var(--ink)}button,input,a{-webkit-tap-highlight-color:transparent}button{font:inherit;cursor:pointer}button:focus-visible,a:focus-visible,input:focus-visible{outline:3px solid #35c4c9;outline-offset:4px}#stage{position:fixed;inset:0 0 48px;display:grid;place-items:center;overflow:hidden}#deck{position:relative;width:1440px;height:810px;flex-shrink:0;transform-origin:center}.slide{position:absolute;inset:0;padding:35px 62px 30px;background:var(--paper);display:none;overflow:hidden}.slide.active{display:flex;flex-direction:column}.slide header{display:flex;justify-content:space-between;align-items:center;height:27px;margin-bottom:18px}.kicker{font-size:15px;font-weight:700;color:var(--blue);letter-spacing:.10em}.kicker span{color:#9aafc3;margin:0 14px}.pill{font-size:13px;letter-spacing:.07em;padding:6px 12px;border-radius:4px;background:#e7edfc;color:var(--blue)}.pill.measured{background:#dbefec;color:var(--teal)}.pill.boundary{background:#f4e9dc;color:#9b581e}.pill.illustration{background:#e9e7f6;color:#6353a0}h1{font-size:38px;letter-spacing:-1px;line-height:1.3;margin:0 0 12px;font-weight:700}h2{font-size:28px;line-height:1.5}h3{font-size:22px;line-height:1.4;margin:0 0 8px;font-weight:700}p{font-size:20px;line-height:1.7;margin:8px 0 14px}.subtitle{font-size:20px;color:var(--muted);line-height:1.65;margin:0;max-width:1250px}.content{flex:1;min-height:0;display:flex;flex-direction:column;justify-content:center;padding:22px 0 18px}.claim{font-size:19px;line-height:1.55;padding:15px 19px;border-left:4px solid var(--teal);background:#e7efef;flex-shrink:0}.slide footer{display:flex;justify-content:space-between;align-items:center;gap:20px;color:#687c8f;font-size:12px;line-height:1.4;margin-top:16px;min-height:20px}.slide footer>span:first-child{max-width:1190px}.split{display:grid;grid-template-columns:1.18fr 1fr;gap:44px;align-items:center}.visual{min-width:0}.visual svg{width:100%;max-height:355px;overflow:visible;display:block}.visual .legend{margin-top:16px}svg text{font-family:"Microsoft YaHei","Segoe UI",sans-serif}.explain{display:flex;gap:16px;margin-bottom:19px}.explain p{font-size:18px;line-height:1.7;margin:0;color:var(--muted)}.ordinal{font-size:16px;color:var(--blue);width:32px;height:32px;border:1px solid #baccea;border-radius:50%;display:grid;place-items:center;flex-shrink:0;margin-top:2px}.micro{font-size:14px!important;line-height:1.6!important;color:#697c8d;margin:8px 0!important}.formula{font-family:"Cambria Math","Times New Roman","Microsoft YaHei",serif;font-size:27px;line-height:1.6;color:var(--blue);margin:12px 0 5px}.formula sub{font-size:65%}.small-callout{background:#e9eef5;padding:15px 18px;font-size:18px;line-height:1.7;margin-top:14px}.legend{display:flex;gap:24px;justify-content:center;font-size:15px;color:var(--muted);flex-wrap:wrap}.legend span{display:flex;gap:9px;align-items:center}.legend i{display:inline-block;width:20px;height:4px;border-radius:1px}.dark{background:#0d2033;color:#e6eef7}.dark .kicker{color:#7ba7f4}.dark .subtitle,.dark p,.dark .micro{color:#a7bbcc}.dark .claim{background:#183348;color:#d1eeed;border-color:#58b9b9}.dark footer{color:#8fa9bd}.dark .small-callout{background:#183448;color:#c8dbe7}.dark table th{background:#19374c}.dark table td{border-color:#29475b}.dark .pill{background:#1c394c;color:#8bd1cf}.cover-grid{display:grid;grid-template-columns:1.4fr 1fr;gap:40px;height:100%;align-items:center}.cover-line{font-size:62px;line-height:1.35;letter-spacing:-1.8px;font-weight:700}.cover-line em{font-style:normal;color:#6bc3c1}.lead{font-size:24px;line-height:1.8;margin-top:23px}.cover-topics{display:flex;gap:24px;margin-top:30px;font-size:16px;color:#97b2cb}.cover-image{position:relative;height:400px;overflow:hidden;background:linear-gradient(135deg,#193a50,#081525);border:1px solid #2c4b60;display:flex;align-items:center;justify-content:flex-start}.cover-image img{height:100%;width:180px;object-fit:cover;opacity:.9}.vertical-label{position:absolute;right:20px;top:24px;font-size:13px;color:#a4bed1;writing-mode:vertical-rl;letter-spacing:4px}.image-meta{font-size:39px;line-height:1.5;margin-left:26px;color:#fff;font-variant-numeric:tabular-nums}.image-meta small{font-size:13px;display:block;color:#a8c0d2;margin-bottom:22px}.morph-grid{display:grid;grid-template-columns:1fr .8fr 1fr;gap:38px}.specimen{border-top:2px solid #bbcddd;padding-top:14px}.specimen>svg,.photo{width:100%;height:245px;object-fit:contain;background:#091626;margin-bottom:18px;border-radius:5px}.photo{display:flex;justify-content:center;overflow:hidden}.photo img{height:100%;max-width:100%;object-fit:contain}.specimen p{font-size:17px;line-height:1.7}.architecture{display:flex;gap:25px;align-items:center;margin-bottom:26px}.input-node{background:#142e49;color:#e4eff8;padding:34px 24px;text-align:center;font-size:24px;line-height:1.7;border-radius:5px;flex-shrink:0}.input-node small{display:block;font-size:14px;color:#a5bfd3}.branch{flex:1;min-width:0}.flow-row{display:flex;gap:12px;align-items:center;background:#e7edf7;border-left:3px solid var(--blue);padding:24px 17px;margin:14px 0;font-size:17px}.flow-row b{color:var(--blue);margin-right:7px;font-size:18px}.flow-row i{font-style:normal;color:#94a7be}.flow-row strong{font-size:15px;color:var(--blue);margin-left:auto}.flow-row.teal{background:#e1efec;border-color:var(--teal)}.flow-row.teal b,.flow-row.teal strong{color:var(--teal)}.funnel{text-align:center;padding:0 35px}.funnel-a,.funnel-b{padding:25px 20px 28px;background:#2764e7;color:white;clip-path:polygon(0 0,100% 0,88% 100%,12% 100%)}.funnel-b{background:#087f83;margin:5px 12% 0;clip-path:polygon(0 0,100% 0,86% 100%,14% 100%)}.funnel small{display:block;font-size:18px}.funnel b{display:block;font-size:51px;font-variant-numeric:tabular-nums;margin-top:8px}.fraction{font-size:20px;margin-top:20px;color:var(--teal)}.fraction strong{font-size:32px;margin-left:15px}.model-columns{display:grid;grid-template-columns:1fr 1fr;gap:40px}.model{border-top:3px solid var(--blue);padding:22px 25px;background:#eaf0f8}.model:nth-child(2){border-color:var(--teal);background:#e4efed}.model-id{font-size:15px;letter-spacing:2px;color:var(--muted)}.metric{font-size:48px;line-height:1.3;color:var(--ink);margin:18px 0 12px}.metric small{font-size:18px;margin-left:12px}.model p{font-size:18px;margin:8px 0}.method-strip{padding:20px 0 0;font-size:19px;text-align:center;color:var(--muted)}.method-strip span{padding:0 28px;color:var(--blue)}canvas{display:block;width:100%;height:auto;background:#0c1b2e;border-radius:5px}.kernel-controls{display:flex;gap:28px;margin:19px 0;font-size:16px}.kernel-controls label{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.kernel-controls input{width:135px;accent-color:var(--teal)}output{min-width:48px;color:var(--blue)}.stat-flow{display:flex;align-items:center;gap:27px;margin-bottom:26px}.stat-flow>i{color:#9eb4c8;font-size:36px;font-style:normal}.stat-node{flex:1;text-align:center;padding:22px 20px;background:#e7eef6;border-top:3px solid var(--blue);height:205px}.stat-node b{font-size:23px;color:var(--blue)}.stat-node p{font-size:24px;font-family:"Cambria Math",serif;margin:15px 0 9px;line-height:1.4}.stat-node small{font-size:15px;color:var(--muted)}.compact{gap:60px}.compact .formula{font-size:30px}.compact>div{border-bottom:1px solid #cbd9e4;padding:0 20px 13px}.result-grid{display:grid;grid-template-columns:310px 1fr;gap:60px;align-items:center}.tall-image{height:405px;position:relative;display:flex;justify-content:center;background:#06111e;overflow:hidden;border:1px solid #29475d}.tall-image img{height:100%;max-width:100%;object-fit:contain}.tall-image span{position:absolute;bottom:12px;right:13px;font-size:13px;color:#abc1d2;background:#0a1d30;padding:5px}.ridge-guide{position:absolute;width:2px;height:90%;left:49%;top:5%;background:#56c5ba;opacity:.25;transform:rotate(6.6deg);transform-origin:center}.hero-metrics{display:flex;gap:70px;margin-bottom:25px}.hero-metrics>div{font-size:58px;color:#74ceca;font-variant-numeric:tabular-nums;line-height:1.1}.hero-metrics small{display:block;font-size:17px;color:#a7c1d3;margin-top:12px}table{width:100%;border-collapse:collapse;margin:18px 0 14px;font-size:18px;line-height:1.5}th{text-align:left;background:#e5ecf4;padding:10px 12px;color:var(--muted);font-size:16px;font-weight:600}td{padding:10px 12px;border-bottom:1px solid #d7e0ea;font-variant-numeric:tabular-nums}.motion-image{max-height:330px;width:100%;object-fit:contain}.conclusion-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:38px}.conclusion{border-top:2px solid #4c8b9a;padding:22px 0}.conclusion>span{font-size:15px;letter-spacing:2px;color:#78c7c4}.conclusion h2{font-size:28px;margin:25px 0 18px}.conclusion p{font-size:21px;line-height:1.9;min-height:115px}.conclusion small{font-size:16px;line-height:1.8;color:#91aebe}.closing{text-align:center;font-size:27px;color:#d6e8ef;margin-top:25px}.closing span{padding:0 22px;color:#70c7c3}.evidence-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px 46px}.evidence-grid>div{border-top:1px solid #c6d4e1;padding:16px 0}.evidence-grid h3{font-size:19px;color:var(--blue)}.evidence-grid p{font:18px/1.7 Consolas,"Microsoft YaHei",sans-serif;margin:7px 0}.evidence-grid small{font-size:16px;color:var(--muted)}.audit-note{margin-top:20px;font-size:18px;line-height:1.8;background:#e8eef5;padding:15px 20px}.speaker{display:none;position:absolute;right:32px;top:90px;width:650px;padding:30px;background:#11283e;color:#e7f0f7;border:1px solid #5e8ea7;box-shadow:0 20px 100px #0008;z-index:20;max-height:610px;overflow:auto}.speaker h2{font-size:25px;margin:0 0 18px}.speaker p{font-size:20px;color:#d1e0eb}.speaker .source-detail{font:16px/1.9 Consolas,"Microsoft YaHei",sans-serif;overflow-wrap:anywhere;color:#89c3e5}.show-notes .active .speaker{display:block}#toolbar{position:fixed;height:48px;bottom:0;left:0;right:0;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:0 22px;color:#a5bacc;background:#0a1828;font-size:12px;z-index:50}#toolbar>div{display:flex;align-items:center;gap:10px}#toolbar button{border:1px solid #30475b;background:#13283c;color:#cbdfeb;border-radius:4px;padding:5px 10px;font-size:12px;line-height:1.5}#toolbar button:hover{background:#23445f}#toolbar button[aria-pressed=true]{border-color:#67c4c2;color:#7ce0dc}#toolbar button:disabled{opacity:.35;cursor:default}#progress{position:fixed;bottom:48px;left:0;height:3px;background:#62c2bd;z-index:51;transition:width .3s}#counter{font-variant-numeric:tabular-nums;min-width:50px;text-align:center}#toc{position:fixed;inset:0 0 48px;background:#091a2df5;z-index:60;display:none;padding:35px 5vw;overflow:auto;color:#dce9f4}#toc.open{display:block}#toc header{display:flex;justify-content:space-between;align-items:center}#toc h2{font-size:25px}#toc-close{border:1px solid #526f86;background:transparent;color:#dce9f4;padding:7px 17px;border-radius:4px}#toc-list{display:grid;grid-template-columns:repeat(4,1fr);gap:13px}#toc-list button{text-align:left;color:#dce9f4;background:#142d43;border:1px solid #2b4b63;padding:17px;border-radius:5px;font-size:15px;line-height:1.6;min-height:100px}#toc-list button span{display:block;color:#75c7c3;font-size:12px;margin-bottom:8px}#toc-list button:hover{background:#234c67}#live{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}.active .reveal{animation:rise .7s both}.active .reveal:nth-child(2){animation-delay:.18s}.active .reveal:nth-child(3){animation-delay:.36s}.active .reveal:nth-child(4){animation-delay:.54s}.active .draw{stroke-dasharray:1400;stroke-dashoffset:1400;animation:draw 1.6s .25s forwards}.active .bar{transform-box:fill-box;transform-origin:bottom;animation:grow .85s .15s both}.active .scanner{animation:scan 5s ease-in-out infinite}.active .ring{animation:ring 2.8s ease-in-out infinite}.paused *{animation-play-state:paused!important}.still *{animation:none!important;transition:none!important;stroke-dashoffset:0!important}.still .reveal{opacity:1!important;transform:none!important}.still .bar{transform:none!important}@keyframes rise{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}@keyframes draw{to{stroke-dashoffset:0}}@keyframes grow{from{transform:scaleY(.01)}to{transform:scaleY(1)}}@keyframes scan{0%,100%{transform:translateX(0)}50%{transform:translateX(600px)}}@keyframes ring{0%,100%{opacity:.55}50%{opacity:1}}@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;stroke-dashoffset:0!important}}@media(max-width:800px){#toolbar{padding:0 8px}#toolbar .hint{display:none}#toolbar>div{gap:4px}#toolbar button{padding:5px 6px}#toc-list{grid-template-columns:repeat(2,1fr)}}@media print{@page{size:1440px 810px;margin:0}html,body{background:white;overflow:visible;width:1440px;height:auto;-webkit-print-color-adjust:exact;print-color-adjust:exact}#stage{position:static;display:block;overflow:visible}#deck{transform:none!important;height:auto!important;width:1440px}.slide{position:relative;display:flex!important;width:1440px;height:810px;break-after:page;page-break-after:always}.slide:last-child{break-after:auto;page-break-after:auto}#toolbar,#toc,#progress,.speaker{display:none!important}*{animation:none!important;transition:none!important;stroke-dashoffset:0!important}.reveal{opacity:1!important;transform:none!important}}
'''

JS=r'''
const slides=[...document.querySelectorAll('.slide')],deck=document.querySelector('#deck');let cur=0,paused=false,still=matchMedia('(prefers-reduced-motion: reduce)').matches;
const $=s=>document.querySelector(s);function fit(){deck.style.transform=`scale(${Math.min(innerWidth/1440,(innerHeight-48)/810)})`}
function show(n,update=true){cur=Math.max(0,Math.min(slides.length-1,Number.isFinite(n)?n:0));slides.forEach((s,i)=>{s.classList.toggle('active',i===cur);s.setAttribute('aria-hidden',i!==cur);s.inert=i!==cur});$('#counter').textContent=`${String(cur+1).padStart(2,'0')} / ${slides.length}`;$('#progress').style.width=`${(cur+1)/slides.length*100}%`;$('#prev').disabled=cur===0;$('#next').disabled=cur===slides.length-1;$('#live').textContent=`第${cur+1}页，${slides[cur].dataset.title}`;if(update)history.replaceState(null,'',`#${cur+1}`);if(cur===10)drawKernel()}
function toc(open){$('#toc').classList.toggle('open',open);$('#toc').inert=!open;$('#toc-button').setAttribute('aria-expanded',String(open));if(open)$('#toc-close').focus();else $('#toc-button').focus()}
function toggleStill(){still=!still;document.body.classList.toggle('still',still);$('#motion').setAttribute('aria-pressed',String(still));$('#motion').textContent=still?'动画：关闭':'动画：开启'}
function togglePause(){paused=!paused;document.body.classList.toggle('paused',paused);$('#pause').textContent=paused?'继续动画':'暂停动画';$('#pause').setAttribute('aria-pressed',String(paused))}
$('#prev').onclick=()=>show(cur-1);$('#next').onclick=()=>show(cur+1);$('#toc-button').onclick=()=>toc(!$('#toc').classList.contains('open'));$('#toc-close').onclick=()=>toc(false);$('#notes').onclick=()=>{document.body.classList.toggle('show-notes');$('#notes').setAttribute('aria-pressed',String(document.body.classList.contains('show-notes')))};$('#motion').onclick=toggleStill;$('#pause').onclick=togglePause;$('#replay').onclick=()=>{const s=slides[cur];s.classList.remove('active');void s.offsetWidth;s.classList.add('active')};$('#full').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen()}catch(e){$('#live').textContent='当前浏览器不支持全屏，请使用 F11。'}};
slides.forEach((s,i)=>{const b=document.createElement('button');b.innerHTML=`<span>${String(i+1).padStart(2,'0')} / ${s.dataset.chapter}</span>${s.dataset.title}`;b.onclick=()=>{show(i);toc(false)};$('#toc-list').append(b)});
document.addEventListener('keydown',e=>{if(e.target.matches('input,textarea,select')||e.altKey||e.ctrlKey||e.metaKey)return;if(e.key==='Escape'){toc(false);document.body.classList.remove('show-notes');$('#notes').setAttribute('aria-pressed','false');return}if(e.target.closest('#toc'))return;if(['ArrowRight','PageDown',' '].includes(e.key)&&e.target.tagName!=='BUTTON'){e.preventDefault();show(cur+1)}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(cur-1)}else if(e.key==='Home'){e.preventDefault();show(0)}else if(e.key==='End'){e.preventDefault();show(slides.length-1)}else if(e.key.toLowerCase()==='n')$('#notes').click();else if(e.key.toLowerCase()==='o')toc(!$('#toc').classList.contains('open'));else if(e.key.toLowerCase()==='r')$('#replay').click();else if(e.key.toLowerCase()==='f')$('#full').click();else if(e.key.toLowerCase()==='p')togglePause()});
window.addEventListener('resize',fit);window.addEventListener('hashchange',()=>show(parseInt(location.hash.slice(1),10)-1,false));let startX=0;deck.addEventListener('touchstart',e=>{startX=e.changedTouches[0].clientX},{passive:true});deck.addEventListener('touchend',e=>{if(e.target.closest('input,button'))return;const dx=e.changedTouches[0].clientX-startX;if(Math.abs(dx)>60)show(cur+(dx<0?1:-1))},{passive:true});
function drawKernel(){const ca=$('#kernel'),ctx=ca.getContext('2d');const L=Number($('#length').value),deg=Number($('#angle').value),a=deg*Math.PI/180;$('#lengthOut').textContent=L+' px';$('#angleOut').textContent=deg+'°';ctx.fillStyle='#0c1b2e';ctx.fillRect(0,0,760,300);ctx.font='18px Microsoft YaHei';ctx.fillStyle='#a9bfd1';ctx.fillText('PSF · 点源核',48,35);ctx.fillText('PSF ∗ 单位线段核',400,35);function heat(cx,cy,len,ang){const size=115,st=3,sigma=7;for(let y=-size;y<=size;y+=st)for(let x=-size;x<=size;x+=st){let v=0;const n=len===0?1:Math.max(8,Math.ceil(len));for(let k=0;k<n;k++){const p=n===1?0:len*(k/(n-1)-.5),dx=x-p*Math.cos(ang),dy=y-p*Math.sin(ang);v+=Math.exp(-(dx*dx+dy*dy)/(2*sigma*sigma))/n}const norm=len===0?1:Math.min(1,Math.sqrt(2*Math.PI)*sigma/Math.max(1,len));const q=Math.min(1,v/norm);ctx.fillStyle=`rgb(${Math.round(12+q*110)},${Math.round(27+q*192)},${Math.round(46+q*172)})`;ctx.fillRect(cx+x,cy+y,st,st)}}heat(185,160,0,0);heat(565,160,L,a);ctx.fillStyle='#91aabd';ctx.font='14px Microsoft YaHei';ctx.fillText('解析采样示意 · 每核归一化思路 · 显示亮度单独拉伸',150,289)}
$('#length').addEventListener('input',drawKernel);$('#angle').addEventListener('input',drawKernel);drawKernel();document.body.classList.toggle('still',still);$('#motion').setAttribute('aria-pressed',String(still));$('#motion').textContent=still?'动画：关闭':'动画：开启';$('#toc').inert=true;fit();show(parseInt(location.hash.slice(1)||'1',10)-1);window.__deck={show,fit,state:()=>({page:cur+1,total:slides.length,paused,still}),hold:()=>{if(!still)toggleStill()},resume:()=>{if(still)toggleStill()}};
'''

manifest={'generated_date':'2026-10-20','source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'working_tree':'pre-existing modifications; current file hashes recorded separately from retained runs',
 'scope':'Retained full-frame results plus newly captured numerical experiments on real FITS crops; no full-frame algorithm rerun',
 'inputs':records,'results':{'source_count':c['source_count'],'quality_count':c['quality_count'],
 'photometry':c['photometry'],'peak_counts_all':peaks,'prototype':{'detection':g['detections'][0],'meta':{k:v for k,v in g.items() if k!='detections'}},
 'production_trail_catalog':t,'correlated_noise':h['boxcar'],'on_off_source_catalog_byte_identical':same},
 'science_corrections':['Holdout uses all colour-complete held-out samples: 0.32654 / 0.32577 mag',
 'M0 zero point 20.4232 and M1 zero point 20.5008 must remain separate',
 'along_profile filters values by amplitude without enforcing a contiguous interval',
 'Lambda and LR are implemented response statistics, not astrophysical identity significance',
 'Pixel index PA is not celestial PA','Standalone PSF 1.93 and production PSF 2.1513 runs are separate'],
 'numerical_animation':{'trace_path':'evidence/science-traces.json','sha256':hashlib.sha256((HERE/'evidence/science-traces.json').read_bytes()).hexdigest(),
 'provenance':science_trace['provenance'],'ornamental_animations':'removed'}}
(HERE/'evidence').mkdir(exist_ok=True)
(HERE/'evidence/deck-evidence.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
JS=JS.replace("&&e.target.tagName!=='BUTTON'", "&&(e.key!==' '||e.target.tagName!=='BUTTON')")
CSS+='\n@media print{#live{display:none!important}.slide{flex-direction:column!important}}\n'
CSS=re.sub(r'@keyframes\s+[\w-]+\s*\{(?:[^{}]|\{[^{}]*\})*\}', '', CSS)
CSS=re.sub(r'animation(?:-[\w-]+)?\s*:[^;}]+;?', '', CSS)
CSS+=SCIENCE_CSS
JS=re.sub(r"function drawKernel\(\).*?drawKernel\(\);document.body.classList.toggle", "function drawKernel(){};document.body.classList.toggle",JS,flags=re.S)
JS=JS.replace('if(cur===10)drawKernel()', 'if(window.scienceRender)window.scienceRender()')
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark light"><title>RST19 · 天基星图检测与经验测光 · 答辩</title><style>'+CSS+'</style></head><body><main id="stage"><div id="deck">'+''.join(slides)+'</div></main><div id="progress"></div><nav id="toolbar" aria-label="演示控制"><div><button id="toc-button" aria-expanded="false" aria-controls="toc">目录 O</button><button id="prev" aria-label="上一页">←</button><span id="counter"></span><button id="next" aria-label="下一页">→</button><span class="hint">方向键 / 空格翻页 · N 来源 · R 重播</span></div><div><button id="motion" aria-pressed="false">动画：开启</button><button id="pause" aria-pressed="false">暂停动画</button><button id="replay">重播 R</button><button id="notes" aria-pressed="false">来源 N</button><button id="full">全屏 F</button></div></nav><div id="toc" role="dialog" aria-modal="true" aria-labelledby="toc-title"><header><h2 id="toc-title">答辩内容 / 20 页</h2><button id="toc-close">返回演示 Esc</button></header><div id="toc-list"></div></div><div id="live" role="status" aria-live="polite"></div><script>'+JS+'</script></body></html>'
page=page.replace('</title>', '</title><link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E%3Crect width=%2732%27 height=%2732%27 rx=%277%27 fill=%27%230d2033%27/%3E%3Cpath d=%27M8 25L23 6%27 stroke=%27%236bc3c1%27 stroke-width=%273%27/%3E%3C/svg%3E">')
page=page.replace('</body>', '<script>window.SCIENCE_TRACE='+json.dumps(science_trace,separators=(',',':'),ensure_ascii=False)+';</script><script>'+SCIENCE_JS+'</script></body>')
(HERE/'答辩幻灯片.html').write_text(page,encoding='utf-8')
print(f'Built {len(slides)} slides; {len(records)} source/evidence hashes; on/off source catalogs identical.')
