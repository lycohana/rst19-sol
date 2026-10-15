use crate::backend::{Backend, Job};
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Vec2};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const CYAN: Color32 = Color32::from_rgb(99, 232, 255);
const GOLD: Color32 = Color32::from_rgb(255, 209, 102);
const SOURCE_DOT_ZOOM: f32 = 1.35;
const SOURCE_RING_ZOOM: f32 = 2.1;

#[derive(Clone, Copy, Debug, PartialEq)]
struct SourceMarkerStyle {
    dot_radius: f32,
    dot_alpha: u8,
    ring_radius: f32,
    ring_width: f32,
    ring_alpha: u8,
}

fn source_marker_style(zoom: f32) -> SourceMarkerStyle {
    let ring_progress =
        ((zoom - SOURCE_DOT_ZOOM) / (SOURCE_RING_ZOOM - SOURCE_DOT_ZOOM)).clamp(0.0, 1.0);
    SourceMarkerStyle {
        dot_radius: 1.15 * (1.0 - ring_progress),
        dot_alpha: (210.0 * (1.0 - ring_progress)).round() as u8,
        ring_radius: 1.7 + 0.8 * ring_progress,
        ring_width: 1.0,
        ring_alpha: (220.0 * ring_progress).round() as u8,
    }
}

fn marker_color(color: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha)
}

struct JobProgress {
    label: String,
    fraction: Option<f32>,
    recent_steps: VecDeque<String>,
    started_at: Instant,
}

impl JobProgress {
    fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            fraction: None,
            recent_steps: VecDeque::from(["任务已启动，等待算法输出".into()]),
            started_at: Instant::now(),
        }
    }

    fn record(&mut self, line: &str) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }

        let step = match parse_progress_line(line) {
            Some((fraction, description)) => {
                self.fraction = Some(fraction);
                format!("{:.0}% · {description}", fraction * 100.0)
            }
            None => line.to_owned(),
        };
        let step = compact_log_line(&step);
        if self.recent_steps.back() == Some(&step) {
            return;
        }
        self.recent_steps.push_back(step);
        while self.recent_steps.len() > 5 {
            self.recent_steps.pop_front();
        }
    }
}

fn parse_progress_line(line: &str) -> Option<(f32, String)> {
    let line = line.trim_start();
    let (value, suffix) = line.split_once('%')?;
    let percent = value.trim().parse::<f32>().ok()?;
    if !percent.is_finite() || !(0.0..=100.0).contains(&percent) {
        return None;
    }
    let step = suffix
        .trim_start_matches(|character: char| {
            character.is_whitespace() || ['·', '|', '-', '—'].contains(&character)
        })
        .trim();
    Some((percent / 100.0, step.to_owned()))
}

fn compact_log_line(line: &str) -> String {
    let mut chars = line.chars();
    let mut compact = chars.by_ref().take(180).collect::<String>();
    if chars.next().is_some() {
        compact.push('…');
    }
    compact
}

fn job_label(request: &Value) -> String {
    match request["operation"].as_str() {
        Some("tool") => request["tool"]
            .as_str()
            .map(|name| title(name).to_owned())
            .unwrap_or_else(|| "算法任务".into()),
        Some("preview") => "读取 FITS 预览".into(),
        _ => "算法任务".into(),
    }
}

#[derive(Clone, Deserialize)]
struct Field {
    key: String,
    flag: String,
    help: String,
    required: bool,
    default: Value,
    choices: Vec<Value>,
    kind: String,
    nargs: Value,
    path: bool,
    repeat: bool,
}
#[derive(Clone, Deserialize)]
struct Tool {
    name: String,
    description: String,
    fields: Vec<Field>,
    #[serde(default)]
    native: bool,
}

pub struct Desktop {
    backend: Result<Backend, String>,
    job: Option<Job>,
    job_progress: Option<JobProgress>,
    tools: Vec<Tool>,
    selected_tool: usize,
    values: HashMap<String, String>,
    toggles: HashMap<String, bool>,
    search: String,
    data_dir: String,
    output_dir: String,
    frames: Vec<PathBuf>,
    selected_frame: Option<PathBuf>,
    preview_mode: String,
    texture: Option<egui::TextureHandle>,
    image_size: Vec2,
    zoom: f32,
    pan: Vec2,
    overlay: bool,
    rejected: bool,
    motion: bool,
    payload: Value,
    analysis: Value,
    analysis_path: Option<PathBuf>,
    sequence: Value,
    logs: String,
    status: String,
    environment: String,
    job_dir: Option<PathBuf>,
    tab: usize,
    source_search: String,
    active_source: Option<Value>,
    smoke: Option<PathBuf>,
    smoke_analysis: bool,
    started: Instant,
    screenshot_requested: bool,
}

fn text_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        _ => value.to_string(),
    }
}
fn title(name: &str) -> &str {
    match name {
        "rst19" => "单帧星点检测",
        "rst19-sequence" => "序列与运动分析",
        "rst19-mosaic" => "注册合成大图",
        "rst19-photometric-auto" => "自动星表与测光",
        "rst19-gaia-frame" => "单帧 Gaia 星表",
        "rst19-plate-solver" => "视场解算",
        "rst19-innovation" => "创新证据报告",
        "rst19-injection" => "注入回收实验",
        "rst19-sweep" => "检测参数扫描",
        _ => name.strip_prefix("rst19-").unwrap_or(name),
    }
}

fn field_title(key: &str) -> &str {
    match key {
        "fits" | "fits_path" | "frame_path" => "输入 FITS 文件",
        "data_dir" => "FITS 序列目录",
        "threshold_sigma" => "检测阈值 σ",
        "psf_fwhm" => "PSF 半高全宽 / px",
        "proposal_mode" => "候选提案模式",
        "min_flux_snr" => "孔径通量 SNR 下限",
        "sequence_json" => "序列分析 JSON",
        "catalog" => "离线星表 CSV",
        "json_out" | "out" => "输出文件",
        _ => key,
    }
}

// Array input never passes through a shell. Spaces and non-ASCII paths stay intact.
fn arguments(
    tool: &Tool,
    values: &HashMap<String, String>,
    toggles: &HashMap<String, bool>,
) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    for field in &tool.fields {
        if field.kind == "bool" {
            if *toggles.get(&field.key).unwrap_or(&false)
                != field.default.as_bool().unwrap_or(false)
            {
                args.push(field.flag.clone());
            }
            continue;
        }
        let value = values.get(&field.key).map(|s| s.trim()).unwrap_or("");
        if value.is_empty() {
            if field.required {
                return Err(format!("请填写 {}", field.key));
            }
            continue;
        }
        let multiple =
            field.repeat || (!field.nargs.is_null() && field.nargs != "?" && field.nargs != 1);
        if multiple {
            let list: Vec<Value> = serde_json::from_str(value).map_err(|_| {
                format!(
                    "{} 需要 JSON 数组，例如 [\"路径 1\", \"路径 2\"]",
                    field.key
                )
            })?;
            for (index, item) in list.iter().enumerate() {
                if !field.flag.is_empty() && (index == 0 || field.repeat) {
                    args.push(field.flag.clone());
                }
                if let Some(group) = item.as_array() {
                    args.extend(group.iter().map(text_value));
                } else {
                    args.push(text_value(item));
                }
            }
        } else {
            if !field.flag.is_empty() {
                args.push(field.flag.clone());
            }
            args.push(value.to_owned());
        }
    }
    Ok(args)
}

impl Desktop {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let mut style = (*cc.egui_ctx.style()).clone();
        style.visuals.panel_fill = Color32::from_rgb(10, 16, 30);
        style.visuals.window_fill = Color32::from_rgb(16, 26, 46);
        style.visuals.selection.bg_fill = Color32::from_rgb(30, 74, 100);
        style.spacing.item_spacing = Vec2::new(10.0, 9.0);
        cc.egui_ctx.set_style(style);
        let mut fonts = egui::FontDefinitions::default();
        let root = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_owned()));
        let font_paths = [
            root.as_ref()
                .map(|r| r.join("assets/NotoSansSC-Regular.otf")),
            root.map(|r| r.join("assets/NotoSansSC-Regular.ttf")),
            Some(PathBuf::from("C:/Windows/Fonts/msyh.ttc")),
            Some(PathBuf::from("/System/Library/Fonts/PingFang.ttc")),
            Some(PathBuf::from(
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            )),
        ];
        for path in font_paths.into_iter().flatten() {
            if let Ok(data) = std::fs::read(path) {
                fonts
                    .font_data
                    .insert("cjk".into(), egui::FontData::from_owned(data).into());
                for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                    fonts
                        .families
                        .entry(family)
                        .or_default()
                        .insert(0, "cjk".into());
                }
                break;
            }
        }
        cc.egui_ctx.set_fonts(fonts);
        let backend = Backend::discover();
        let project = backend
            .as_ref()
            .map(|b| b.root().to_owned())
            .unwrap_or_default();
        let data_dir = project
            .join("doc/00-项目资料/原始数据")
            .to_string_lossy()
            .to_string();
        let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let output_dir = home.join("RST19-results").to_string_lossy().to_string();
        let mut args = std::env::args().skip(1);
        let mut smoke = None;
        let mut smoke_analysis = false;
        let mut data_override = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--smoke-screenshot" => smoke = args.next().map(PathBuf::from),
                "--data-dir" => data_override = args.next(),
                "--smoke-analysis" => smoke_analysis = true,
                _ => {}
            }
        }
        let mut app = Self {
            backend,
            job: None,
            job_progress: None,
            tools: vec![],
            selected_tool: 0,
            values: HashMap::new(),
            toggles: HashMap::new(),
            search: String::new(),
            data_dir: data_override.unwrap_or(data_dir),
            output_dir,
            frames: vec![],
            selected_frame: None,
            preview_mode: "raw".into(),
            texture: None,
            image_size: Vec2::ZERO,
            zoom: 1.0,
            pan: Vec2::ZERO,
            overlay: true,
            rejected: false,
            motion: true,
            payload: Value::Null,
            analysis: Value::Null,
            analysis_path: None,
            sequence: Value::Null,
            logs: String::new(),
            status: "正在检查算法环境…".into(),
            environment: String::new(),
            job_dir: None,
            tab: 0,
            source_search: String::new(),
            active_source: None,
            smoke,
            smoke_analysis,
            started: Instant::now(),
            screenshot_requested: false,
        };
        app.scan_frames();
        match &app.backend {
            Ok(b) => match b.start(None, b.root()) {
                Ok(job) => {
                    app.job = Some(job);
                    app.job_progress = Some(JobProgress::new("检查算法环境"));
                }
                Err(e) => app.status = e,
            },
            Err(e) => app.status = e.clone(),
        }
        app
    }

    fn scan_frames(&mut self) {
        self.frames = std::fs::read_dir(&self.data_dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|p| {
                p.extension().is_some_and(|e| {
                    ["fits", "fit", "fts"].contains(&e.to_string_lossy().to_lowercase().as_str())
                })
            })
            .collect();
        self.frames.sort();
        self.selected_frame = self.frames.first().cloned();
    }

    fn select_tool(&mut self, index: usize) {
        self.selected_tool = index;
        self.values.clear();
        self.toggles.clear();
        for field in &self.tools[index].fields {
            let mut value = text_value(&field.default);
            if field.key == "data_dir" {
                value = self.data_dir.clone();
            }
            if ["fits", "fits_path", "frame_path"].contains(&field.key.as_str()) {
                value = self
                    .selected_frame
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
            }
            // Each run supplies its own output locations, instead of CLI relative defaults.
            if field.key.ends_with("out_dir") || field.key == "json_out" {
                value.clear();
            }
            self.values.insert(field.key.clone(), value);
            self.toggles
                .insert(field.key.clone(), field.default.as_bool().unwrap_or(false));
        }
    }

    fn start(&mut self, mut request: Value) {
        if self.job.is_some() {
            return;
        }
        let progress_label = job_label(&request);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let out = PathBuf::from(&self.output_dir).join(format!("run-{timestamp}"));
        if let Err(e) = std::fs::create_dir_all(&out) {
            self.status = format!("无法创建结果目录：{e}");
            return;
        }
        request["out_dir"] = json!(out);
        if request["operation"] == "tool" {
            let tool = &self.tools[self.selected_tool];
            for field in &tool.fields {
                if field.key.ends_with("out_dir")
                    && self.values.get(&field.key).is_none_or(|v| v.is_empty())
                {
                    let target = out.join(&field.key);
                    let args = request["args"].as_array_mut().unwrap();
                    args.push(json!(field.flag));
                    args.push(json!(target));
                }
            }
        }
        match self
            .backend
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|b| b.start(Some(request), &out))
        {
            Ok(job) => {
                self.job = Some(job);
                self.job_progress = Some(JobProgress::new(progress_label));
                self.job_dir = Some(out);
                self.logs.clear();
                self.status = "任务运行中…".into();
                self.active_source = None;
            }
            Err(e) => self.status = e,
        }
    }

    fn preview(&mut self) {
        if let Some(path) = &self.selected_frame {
            self.start(json!({"operation":"preview", "path":path, "mode":self.preview_mode}));
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        let events: Vec<Value> = self
            .job
            .as_ref()
            .map(|j| j.receiver.try_iter().take(500).collect())
            .unwrap_or_default();
        for event in events {
            if event["event"] == "log" {
                let line = event["text"].as_str().unwrap_or("");
                if let Some(progress) = &mut self.job_progress {
                    progress.record(line);
                    if let Some(fraction) = progress.fraction {
                        self.status = format!("任务运行中 · {:.0}%", fraction * 100.0);
                    }
                }
                self.logs.push_str(line);
                self.logs.push('\n');
                if self.logs.len() > 150_000 {
                    let mut boundary = self.logs.len() - 100_000;
                    while !self.logs.is_char_boundary(boundary) {
                        boundary += 1;
                    }
                    self.logs.drain(..boundary);
                }
            } else if event["event"] == "done" {
                let cancelled = self.job.as_ref().is_some_and(|j| j.cancelled);
                self.job.take();
                if cancelled {
                    self.status = "任务已取消；未完成的产物保留在任务目录".into();
                    continue;
                }
                if event["ok"] != true {
                    self.status = event["error"].as_str().unwrap_or("未知错误").into();
                    self.logs
                        .push_str(event["traceback"].as_str().unwrap_or(""));
                    continue;
                }
                let p = &event["payload"];
                if let Some(tools) = p["tools"].as_array() {
                    match serde_json::from_value::<Vec<Tool>>(json!(tools)) {
                        Ok(mut tools) => {
                            let primary = [
                                "rst19",
                                "rst19-sequence",
                                "rst19-mosaic",
                                "rst19-photometric-auto",
                                "rst19-gaia-frame",
                                "rst19-gaia-merge",
                                "rst19-innovation",
                                "rst19-injection",
                                "rst19-sweep",
                            ];
                            tools.sort_by_key(|tool| {
                                (
                                    primary
                                        .iter()
                                        .position(|name| *name == tool.name)
                                        .unwrap_or(primary.len()),
                                    tool.name.clone(),
                                )
                            });
                            self.tools = tools;
                            self.select_tool(0);
                            let native_count = self.tools.iter().filter(|tool| tool.native).count();
                            self.environment = format!(
                                "Rust 原生 UI · {native_count}/{} 项研究工具",
                                self.tools.len()
                            );
                            self.status = "环境已就绪".into();
                            if self.smoke.is_some() {
                                if self.smoke_analysis {
                                    match arguments(&self.tools[0], &self.values, &self.toggles) {
                                        Ok(args) => self.start(json!({"operation":"tool", "tool":"rst19", "args":args})),
                                        Err(e) => self.status = e,
                                    }
                                } else {
                                    self.preview();
                                }
                            }
                        }
                        Err(e) => self.status = format!("工具定义无效：{e}"),
                    }
                    continue;
                }
                if let Some(path) = p["preview"].as_str() {
                    match image::open(path) {
                        Ok(image) => {
                            let rgba = image.to_rgba8();
                            let size = [rgba.width() as usize, rgba.height() as usize];
                            self.texture = Some(ctx.load_texture(
                                "fits",
                                egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw()),
                                egui::TextureOptions::LINEAR,
                            ));
                            self.image_size = Vec2::new(
                                p["width"].as_f64().unwrap_or(size[0] as f64) as f32,
                                p["height"].as_f64().unwrap_or(size[1] as f64) as f32,
                            );
                            self.zoom = 1.0;
                            self.pan = Vec2::ZERO;
                        }
                        Err(e) => {
                            self.status = format!("预览读取失败：{e}");
                            continue;
                        }
                    }
                }
                if p["tool"] == "rst19" {
                    self.analysis = p["result"].clone();
                    self.analysis_path = self.analysis["path"].as_str().map(PathBuf::from);
                    self.selected_frame = self.analysis_path.clone();
                } else if p["tool"] == "rst19-sequence" {
                    self.sequence = p["result"].clone();
                }
                self.payload = p.clone();
                self.status = "任务完成 · 产物已保存".into();
            }
        }
        if self.job.is_some() {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
    }

    fn field_ui(&mut self, ui: &mut egui::Ui, field: &Field) {
        ui.push_id(&field.key, |ui| {
            if field.kind == "bool" {
                ui.checkbox(
                    self.toggles.entry(field.key.clone()).or_default(),
                    &field.key,
                )
                .on_hover_text(&field.help);
            } else {
                let multiple = field.repeat
                    || (!field.nargs.is_null() && field.nargs != "?" && field.nargs != 1);
                ui.label(
                    egui::RichText::new(format!(
                        "{}{}",
                        field_title(&field.key),
                        if field.required { " *" } else { "" }
                    ))
                    .color(Color32::from_rgb(160, 183, 208)),
                )
                .on_hover_text(&field.help);
                let value = self.values.entry(field.key.clone()).or_default();
                if !field.choices.is_empty() {
                    egui::ComboBox::from_id_salt("choice")
                        .selected_text(if value.is_empty() {
                            "使用默认值"
                        } else {
                            value.as_str()
                        })
                        .width(270.0)
                        .show_ui(ui, |ui| {
                            for choice in &field.choices {
                                let s = text_value(choice);
                                ui.selectable_value(value, s.clone(), s);
                            }
                        });
                } else {
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(value)
                                .desired_width(if field.path { 230.0 } else { 280.0 })
                                .hint_text(if multiple {
                                    "JSON 数组"
                                } else {
                                    "留空使用算法默认值"
                                }),
                        );
                        if field.path && ui.small_button("…").clicked() {
                            if multiple {
                                if let Some(paths) = rfd::FileDialog::new().pick_files() {
                                    if !paths.is_empty() {
                                        let paths = paths
                                            .iter()
                                            .map(|path| path.to_string_lossy().into_owned())
                                            .collect::<Vec<_>>();
                                        *value = serde_json::to_string(&paths).unwrap_or_default();
                                    }
                                }
                            } else {
                                let picked = if field.key.ends_with("dir") {
                                    rfd::FileDialog::new().pick_folder()
                                } else if field.key == "out" || field.key.ends_with("_out") {
                                    rfd::FileDialog::new().save_file()
                                } else {
                                    rfd::FileDialog::new().pick_file()
                                };
                                if let Some(path) = picked {
                                    *value = path.to_string_lossy().into_owned();
                                }
                            }
                        }
                    });
                }
                ui.label(
                    egui::RichText::new(&field.help)
                        .small()
                        .color(Color32::from_gray(135)),
                );
            }
            ui.add_space(3.0);
        });
    }

    fn viewer(&mut self, ui: &mut egui::Ui) {
        if self.analysis_path.is_some() && self.analysis_path == self.selected_frame {
            ui.horizontal_wrapped(|ui| {
                for (key, label) in [
                    ("candidate_count", "候选峰"),
                    ("returned_count", "已测候选"),
                    ("quality_count", "质量通过"),
                    ("rejected_count", "被拒候选"),
                ] {
                    ui.label(format!("{label} {}", self.analysis["detection"][key]));
                    ui.separator();
                }
                if self.analysis["detection"]["truncated"] == true {
                    ui.colored_label(GOLD, "候选返回已截断");
                }
                if let Some(magnitude) =
                    self.analysis["faintest_detected"]["instrumental_magnitude"].as_f64()
                {
                    ui.label(format!("最暗质量源 m_inst {magnitude:.3}"));
                }
            });
        }
        ui.horizontal(|ui| {
            ui.heading("观测视场");
            ui.add_enabled_ui(self.job.is_none(), |ui| {
                let old = self.preview_mode.clone();
                egui::ComboBox::from_id_salt("preview mode")
                    .selected_text(match self.preview_mode.as_str() {
                        "raw" => "原始显示",
                        "noise" => "噪声检查",
                        _ => "增强显示",
                    })
                    .show_ui(ui, |ui| {
                        for (key, label) in [
                            ("raw", "原始显示"),
                            ("enhanced", "增强显示"),
                            ("noise", "噪声检查"),
                        ] {
                            ui.selectable_value(&mut self.preview_mode, key.into(), label);
                        }
                    });
                if old != self.preview_mode {
                    self.preview();
                }
                if ui.button("读取预览").clicked() {
                    self.preview();
                }
            });
            if ui.button("适合窗口").clicked() {
                self.zoom = 1.0;
                self.pan = Vec2::ZERO;
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.overlay, "质量通过源");
            ui.checkbox(&mut self.rejected, "被拒候选");
            ui.checkbox(&mut self.motion, "运动轨迹");
            ui.label(egui::RichText::new("滚轮缩放 · 拖动平移 · 点击查看源").small());
        });
        let (rect, response) = ui.allocate_exact_size(
            ui.available_size().max(Vec2::splat(1.0)),
            Sense::click_and_drag(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 8.0, Color32::from_rgb(4, 8, 16));
        let Some(texture) = &self.texture else {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "选择 FITS 文件，读取观测视场",
                egui::FontId::proportional(22.0),
                Color32::from_rgb(100, 130, 160),
            );
            return;
        };
        let fit = (rect.width() / self.image_size.x).min(rect.height() / self.image_size.y);
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let old = self.zoom;
                self.zoom = (old * (scroll * 0.003).exp()).clamp(1.0, 30.0);
                let pointer = ui.input(|i| i.pointer.hover_pos()).unwrap_or(rect.center());
                let offset = pointer - rect.center();
                self.pan = (self.pan - offset) * (self.zoom / old) + offset;
            }
        }
        if response.dragged() {
            self.pan += ui.input(|i| i.pointer.delta());
        }
        let scale = fit * self.zoom;
        let marker_style = source_marker_style(self.zoom);
        let image_rect = Rect::from_center_size(rect.center() + self.pan, self.image_size * scale);
        painter.image(
            texture.id(),
            image_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        let exact = self.analysis_path.as_ref() == self.selected_frame.as_ref();
        let mut nearest: Option<(f32, Value)> = None;
        if exact && (self.overlay || self.rejected) {
            if let Some(sources) = self.analysis["detection"]["sources"].as_array() {
                for source in sources {
                    let quality = source["quality_passed"].as_bool().unwrap_or(false);
                    if (quality && !self.overlay) || (!quality && !self.rejected) {
                        continue;
                    }
                    let x = source["x"].as_f64().unwrap_or(0.0) as f32;
                    let y = source["y"].as_f64().unwrap_or(0.0) as f32;
                    let point = image_rect.min + Vec2::new(x, y) * scale;
                    if !rect.contains(point) {
                        continue;
                    }
                    let color = if quality {
                        CYAN
                    } else {
                        Color32::from_rgb(195, 165, 255)
                    };
                    if marker_style.dot_alpha > 0 {
                        painter.circle_filled(
                            point,
                            marker_style.dot_radius,
                            marker_color(color, marker_style.dot_alpha),
                        );
                    }
                    if marker_style.ring_alpha > 0 {
                        painter.circle_stroke(
                            point,
                            marker_style.ring_radius,
                            (
                                marker_style.ring_width,
                                marker_color(color, marker_style.ring_alpha),
                            ),
                        );
                    }
                    if let Some(pointer) = response.hover_pos() {
                        let distance = pointer.distance(point);
                        if distance < 10.0 && nearest.as_ref().is_none_or(|(d, _)| distance < *d) {
                            nearest = Some((distance, source.clone()));
                        }
                    }
                }
            }
        }
        if self.motion {
            let frame_index = self.sequence["frames"].as_array().and_then(|frames| {
                frames.iter().position(|frame| {
                    frame["path"].as_str().map(PathBuf::from).as_ref()
                        == self.selected_frame.as_ref()
                })
            });
            if let Some(index) = frame_index {
                if let Some(tracks) = self.sequence["tracks"].as_array() {
                    for track in tracks.iter().filter(|t| t["classification"] == "moving") {
                        if let Some(points) = track["points"].as_array() {
                            for point in points
                                .iter()
                                .filter(|p| p["frame_index"].as_u64() == Some(index as u64))
                            {
                                let position = image_rect.min
                                    + Vec2::new(
                                        point["x"].as_f64().unwrap_or(0.0) as f32,
                                        point["y"].as_f64().unwrap_or(0.0) as f32,
                                    ) * scale;
                                painter.circle_stroke(position, 7.0, (1.5, GOLD));
                                painter.text(
                                    position + Vec2::new(10.0, 0.0),
                                    egui::Align2::LEFT_CENTER,
                                    format!("T{}", track["track_id"]),
                                    egui::FontId::monospace(12.0),
                                    GOLD,
                                );
                            }
                        }
                    }
                }
                if let Some(tracks) = self.sequence["motion_features"].as_array() {
                    for track in tracks.iter().filter(|t| t["classification"] == "moving") {
                        if let Some(points) = track["points"].as_array() {
                            for point in points
                                .iter()
                                .filter(|p| p["frame_index"].as_u64() == Some(index as u64))
                            {
                                let position = image_rect.min
                                    + Vec2::new(
                                        point["x"].as_f64().unwrap_or(0.0) as f32,
                                        point["y"].as_f64().unwrap_or(0.0) as f32,
                                    ) * scale;
                                let color = Color32::from_rgb(255, 107, 214);
                                painter.circle_stroke(position, 8.0, (1.5, color));
                                painter.text(
                                    position + Vec2::new(12.0, 0.0),
                                    egui::Align2::LEFT_CENTER,
                                    format!(
                                        "L{} · {:.1} px",
                                        track["track_id"],
                                        point["length_px"].as_f64().unwrap_or(0.0)
                                    ),
                                    egui::FontId::monospace(12.0),
                                    color,
                                );
                                if let Some(pointer) = response.hover_pos() {
                                    let distance = pointer.distance(position);
                                    if distance < 12.0
                                        && nearest.as_ref().is_none_or(|(d, _)| distance < *d)
                                    {
                                        nearest =
                                            Some((distance, json!({"track":track, "point":point})));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some((_, source)) = nearest {
            response.clone().on_hover_text(format!(
                "ID {} · ({:.2}, {:.2})\n孔径 SNR {} · FWHM {}\n{}",
                source["detection_id"],
                source["x"].as_f64().unwrap_or(0.0),
                source["y"].as_f64().unwrap_or(0.0),
                source["flux_snr"],
                source["fwhm"],
                source["flags"]
            ));
            if response.clicked() {
                self.active_source = Some(source);
            }
        }
    }
}

impl eframe::App for Desktop {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.poll(ctx);
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("RST19").size(27.0).strong().color(CYAN));
                ui.label("星图研究工作台");
                ui.separator();
                ui.label(
                    egui::RichText::new("观测 / 检测 / 运动 / 测光 / 证据")
                        .color(Color32::from_rgb(145, 166, 201)),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some(job) = &mut self.job {
                        if ui.button("取消任务").clicked() {
                            job.cancel();
                        }
                        ui.spinner();
                    }
                    ui.label(&self.status);
                });
            });
        });
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new(&self.environment).small());
                ui.separator();
                ui.label(
                    egui::RichText::new(
                        "显示映射不改变原始 ADU；候选与质量通过源不等同于真实天体。",
                    )
                    .small(),
                );
            });
        });
        egui::SidePanel::left("workspace")
            .default_width(230.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("观测数据");
                ui.add_enabled_ui(self.job.is_none(), |ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.data_dir).desired_width(f32::INFINITY),
                    );
                    ui.horizontal(|ui| {
                        if ui.button("选择目录").clicked() {
                            if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                self.data_dir = path.to_string_lossy().into_owned();
                                self.scan_frames();
                                self.texture = None;
                            }
                        }
                        if ui.button("刷新").clicked() {
                            self.scan_frames();
                            self.texture = None;
                        }
                    });
                    ui.label(format!("{} 帧 FITS", self.frames.len()));
                    egui::ScrollArea::vertical()
                        .id_salt("frames")
                        .max_height(210.0)
                        .show(ui, |ui| {
                            let mut picked = None;
                            for (i, frame) in self.frames.iter().enumerate() {
                                if ui
                                    .selectable_label(
                                        self.selected_frame.as_ref() == Some(frame),
                                        format!(
                                            "{:02}  {}",
                                            i + 1,
                                            frame.file_name().unwrap().to_string_lossy()
                                        ),
                                    )
                                    .clicked()
                                {
                                    picked = Some(frame.clone());
                                }
                            }
                            if let Some(frame) = picked {
                                self.selected_frame = Some(frame.clone());
                                self.active_source = None;
                                self.texture = None;
                                for key in ["fits", "fits_path", "frame_path"] {
                                    if self.values.contains_key(key) {
                                        self.values.insert(
                                            key.into(),
                                            frame.to_string_lossy().into_owned(),
                                        );
                                    }
                                }
                                self.preview();
                            }
                        });
                });
                ui.separator();
                ui.heading("分析工具");
                ui.add(egui::TextEdit::singleline(&mut self.search).hint_text("搜索研究工具"));
                ui.add_enabled_ui(self.job.is_none(), |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("tools")
                        .show(ui, |ui| {
                            let mut selected = None;
                            for (index, tool) in self.tools.iter().enumerate().filter(|(_, t)| {
                                self.search.is_empty()
                                    || t.name.contains(&self.search)
                                    || t.description.contains(&self.search)
                            }) {
                                let label = format!(
                                    "{} · {}",
                                    if tool.native { "Rust" } else { "待迁移" },
                                    title(&tool.name)
                                );
                                if ui
                                    .selectable_label(self.selected_tool == index, label)
                                    .on_hover_text(&tool.description)
                                    .clicked()
                                {
                                    selected = Some(index);
                                }
                            }
                            if let Some(index) = selected {
                                self.select_tool(index);
                            }
                        });
                });
            });
        egui::SidePanel::right("parameters").default_width(340.0).width_range(300.0..=550.0).show(ctx, |ui| {
            ui.heading("任务参数");
            if self.tools.is_empty() {
                ui.label(&self.status);
                if self.job.is_none() && ui.button("重新检查环境").clicked() {
                    if let Ok(backend) = &self.backend { match backend.start(None, backend.root()) { Ok(job) => self.job = Some(job), Err(e) => self.status = e } }
                }
                return;
            }
            let tool = self.tools[self.selected_tool].clone();
            ui.label(egui::RichText::new(title(&tool.name)).strong().color(CYAN));
            ui.label(&tool.description);
            ui.add_enabled_ui(self.job.is_none(), |ui| {
                if ui.add_sized([ui.available_width(), 36.0], egui::Button::new("运行分析").fill(Color32::from_rgb(28, 77, 98))).clicked() {
                    match arguments(&tool, &self.values, &self.toggles) {
                        Ok(args) => self.start(json!({"operation":"tool", "tool":tool.name, "args":args, "mode":self.preview_mode})),
                        Err(e) => self.status = e,
                    }
                }
                egui::ScrollArea::vertical().id_salt("fields").show(ui, |ui| {
                    ui.label("结果保存目录");
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.output_dir).desired_width(230.0));
                        if ui.small_button("…").clicked() { if let Some(path) = rfd::FileDialog::new().pick_folder() { self.output_dir = path.to_string_lossy().into_owned(); } }
                    });
                    ui.separator();
                    for field in tool.fields.iter().filter(|f| f.required || ["threshold_sigma", "min_flux_snr", "proposal_mode", "psf_fwhm", "sequence_json"].contains(&f.key.as_str())) { self.field_ui(ui, field); }
                    egui::CollapsingHeader::new("全部参数与研究设置").show(ui, |ui| {
                        for field in tool.fields.iter().filter(|f| !(f.required || ["threshold_sigma", "min_flux_snr", "proposal_mode", "psf_fwhm", "sequence_json"].contains(&f.key.as_str()))) { self.field_ui(ui, field); }
                    });
                    if let Some(source) = &self.active_source {
                        ui.separator(); ui.heading("选中源");
                        ui.label(egui::RichText::new(serde_json::to_string_pretty(source).unwrap_or_default()).monospace().small());
                    }
                });
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut show_full_logs = false;
            if self.job.is_some() {
                if let Some(progress) = &self.job_progress {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("任务运行中").strong().color(CYAN));
                            ui.separator();
                            ui.label(egui::RichText::new(&progress.label).strong());
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    let elapsed = progress.started_at.elapsed().as_secs();
                                    ui.label(format!("已运行 {:02}:{:02}", elapsed / 60, elapsed % 60));
                                },
                            );
                        });
                        if let Some(fraction) = progress.fraction {
                            ui.add(
                                egui::ProgressBar::new(fraction)
                                    .desired_height(9.0)
                                    .text(format!("{:.0}%", fraction * 100.0)),
                            );
                        } else {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label("算法尚未报告百分比；正在显示实时运行步骤");
                            });
                        }
                        if !progress.recent_steps.is_empty() {
                            ui.label(egui::RichText::new("最近运行步骤 / 消息").small().weak());
                            let last = progress.recent_steps.len() - 1;
                            for (index, step) in progress.recent_steps.iter().enumerate() {
                                let color = if index == last {
                                    CYAN
                                } else {
                                    Color32::from_rgb(150, 165, 187)
                                };
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} {step}",
                                        if index == last { "›" } else { "·" }
                                    ))
                                    .small()
                                    .color(color),
                                );
                            }
                        }
                        if self.tab != 3 && ui.small_button("打开完整运行日志").clicked() {
                            show_full_logs = true;
                        }
                    });
                    ui.add_space(6.0);
                }
            }
            if show_full_logs {
                self.tab = 3;
            }
            ui.horizontal(|ui| {
                for (i, tab) in ["观测视场", "检测源表", "结构化证据", "运行日志", "产物浏览"].iter().enumerate() { ui.selectable_value(&mut self.tab, i, *tab); }
            });
            ui.separator();
            match self.tab {
                0 => self.viewer(ui),
                1 => {
                    if self.analysis_path.as_ref() != self.selected_frame.as_ref() { ui.label("当前帧尚未分析。源表来自上一分析帧。"); }
                    ui.horizontal(|ui| {
                        for key in ["candidate_count", "returned_count", "quality_count", "rejected_count"] { ui.label(format!("{}: {}", key, self.analysis["detection"][key])); }
                    });
                    ui.label("按检测 ID 搜索"); ui.text_edit_singleline(&mut self.source_search);
                    let all_sources = self.analysis["detection"]["sources"].as_array();
                    let indices: Vec<usize> = all_sources.map(|s| s.iter().enumerate().filter(|(_, s)| self.source_search.is_empty() || s["detection_id"].to_string().contains(&self.source_search)).map(|(i, _)| i).collect()).unwrap_or_default();
                    egui::ScrollArea::vertical().show_rows(ui, 25.0, indices.len(), |ui, range| {
                        for index in range { let s = &all_sources.unwrap()[indices[index]];
                            if ui.selectable_label(false, format!("ID {:>6}    x {:>9.2}    y {:>9.2}    SNR {:>9}    FWHM {:>8}    {}", s["detection_id"], s["x"].as_f64().unwrap_or(0.0), s["y"].as_f64().unwrap_or(0.0), s["flux_snr"], s["fwhm"], if s["quality_passed"] == true { "质量通过" } else { "被拒候选" })).clicked() { self.active_source = Some(s.clone()); }
                        }
                    });
                }
                2 => {
                    ui.label("保留完整输出字段、标定状态与证据来源；可从任务目录获取 JSON。");
                    let result = self.payload.get("result").filter(|r| !r.is_null()).unwrap_or(&self.payload);
                    egui::ScrollArea::both().show(ui, |ui| { evidence_tree(ui, "result", result); });
                }
                3 => { egui::ScrollArea::both().stick_to_bottom(true).show(ui, |ui| { ui.add(egui::TextEdit::multiline(&mut self.logs).font(egui::TextStyle::Monospace).desired_width(f32::INFINITY).interactive(false)); }); }
                _ => {
                    ui.label("每次任务使用独立目录，保留参数、stdout、stderr 和算法产物。");
                    if let Some(out) = &self.job_dir {
                        ui.label(out.to_string_lossy());
                        if ui.button("打开任务目录").clicked() { open_path(out); }
                        egui::ScrollArea::vertical().show(ui, |ui| { artifacts(ui, out, 0); });
                    }
                    ui.separator();
                    if ui.button("导入已有结果 JSON").clicked() {
                        if let Some(path) = rfd::FileDialog::new().add_filter("JSON", &["json"]).pick_file() {
                            match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str::<Value>(&s).map_err(|e| e.to_string())) {
                                Ok(result) => {
                                    if result.get("detection").is_some() { self.analysis_path = result["path"].as_str().map(PathBuf::from); self.analysis = result.clone(); }
                                    if result.get("frames").is_some() { self.sequence = result.clone(); }
                                    self.payload = json!({"result":result}); self.status = "已导入结果".into();
                                }, Err(e) => self.status = e,
                            }
                        }
                    }
                }
            }
        });
        if let Some(path) = self.smoke.clone() {
            if self.started.elapsed() > Duration::from_secs(55) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            if self.job.is_none()
                && !self.tools.is_empty()
                && self.texture.is_some()
                && !self.screenshot_requested
            {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                self.screenshot_requested = true;
            }
            let shots = ctx.input(|i| {
                i.events
                    .iter()
                    .filter_map(|e| {
                        if let egui::Event::Screenshot { image, .. } = e {
                            Some(image.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            });
            for shot in shots {
                let pixels: Vec<u8> = shot.pixels.iter().flat_map(|p| p.to_array()).collect();
                if let Some(image) =
                    image::RgbaImage::from_raw(shot.width() as u32, shot.height() as u32, pixels)
                {
                    let _ = image.save(&path);
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

fn open_path(path: &std::path::Path) {
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("explorer.exe").arg(path).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(path).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(path).spawn();
    }
}

fn evidence_tree(ui: &mut egui::Ui, key: &str, value: &Value) {
    ui.push_id(key, |ui| match value {
        Value::Object(map) => {
            egui::CollapsingHeader::new(format!("{key} · {} 项", map.len()))
                .default_open(key == "result")
                .show(ui, |ui| {
                    for (k, v) in map {
                        evidence_tree(ui, k, v);
                    }
                });
        }
        Value::Array(items) => {
            egui::CollapsingHeader::new(format!("{key} · {} 条", items.len())).show(ui, |ui| {
                // Large scientific arrays stay virtualized instead of formatting all rows every frame.
                egui::ScrollArea::vertical()
                    .id_salt(key)
                    .max_height(420.0)
                    .show_rows(ui, 26.0, items.len(), |ui, range| {
                        for index in range {
                            let item = &items[index];
                            let count = item
                                .as_object()
                                .map(|m| m.len())
                                .or_else(|| item.as_array().map(|a| a.len()));
                            if let Some(count) = count {
                                let label = format!("[{index}] · {count} 项");
                                ui.menu_button(label, |ui| {
                                    evidence_tree(ui, "item", item);
                                });
                            } else {
                                ui.label(format!("[{index}] {}", text_value(item)));
                            }
                        }
                    });
            });
        }
        _ => {
            ui.label(format!("{key}: {}", text_value(value)));
        }
    });
}
fn artifacts(ui: &mut egui::Ui, dir: &std::path::Path, depth: usize) {
    if depth > 4 {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut paths: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if path.is_dir() {
                egui::CollapsingHeader::new(name)
                    .id_salt(&path)
                    .show(ui, |ui| artifacts(ui, &path, depth + 1));
            } else if ui.link(name).clicked() {
                open_path(&path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_panel_uses_reported_percentages_and_keeps_recent_steps() {
        let (fraction, step) = parse_progress_line(" 67.0% · 检测完成").unwrap();
        assert!((fraction - 0.67).abs() < f32::EPSILON);
        assert_eq!(step, "检测完成");
        assert!(parse_progress_line("loading stage 67%").is_none());
        assert!(parse_progress_line("101% · invalid").is_none());

        let mut progress = JobProgress::new("单帧星点检测");
        progress.record("读取 FITS");
        assert!(progress.fraction.is_none());
        for index in 0..7 {
            progress.record(&format!("处理分块 {index}"));
        }
        assert_eq!(progress.recent_steps.len(), 5);
        assert_eq!(progress.recent_steps.back().unwrap(), "处理分块 6");
    }

    #[test]
    fn source_markers_adapt_from_overview_dots_to_zoomed_rings() {
        let overview = source_marker_style(1.0);
        assert!(overview.dot_radius > 1.0);
        assert!(overview.dot_alpha > 0);
        assert_eq!(overview.ring_alpha, 0);

        let transition = source_marker_style(1.7);
        assert!(transition.dot_alpha > 0);
        assert!(transition.ring_alpha > 0);

        let close_up = source_marker_style(3.0);
        assert_eq!(close_up.dot_alpha, 0);
        assert!(close_up.ring_alpha > 0);
        assert!(close_up.ring_radius > overview.dot_radius);
    }

    #[test]
    fn paths_and_arrays_do_not_become_shell_commands() {
        let tool: Tool = serde_json::from_value(json!({"name":"test", "description":"", "fields":[
            {"key":"file", "flag":"", "help":"", "required":true, "default":null, "choices":[], "kind":"value", "nargs":null, "path":true, "repeat":false},
            {"key":"inputs", "flag":"--inputs", "help":"", "required":false, "default":null, "choices":[], "kind":"value", "nargs":"+", "path":true, "repeat":false},
            {"key":"keep", "flag":"--disable", "help":"", "required":false, "default":true, "choices":[], "kind":"bool", "nargs":0, "path":false, "repeat":false}
        ]})).unwrap();
        let values = HashMap::from([
            ("file".into(), "C:/原始 数据/a.fits".into()),
            ("inputs".into(), "[\"a b\", \"$(echo x)\"]".into()),
        ]);
        let toggles = HashMap::from([("keep".into(), false)]);
        assert_eq!(
            arguments(&tool, &values, &toggles).unwrap(),
            [
                "C:/原始 数据/a.fits",
                "--inputs",
                "a b",
                "$(echo x)",
                "--disable"
            ]
        );
        assert!(arguments(&tool, &HashMap::new(), &toggles).is_err());
    }
}
