#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod backend;
mod ui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([1000.0, 680.0]),
        ..Default::default()
    };
    eframe::run_native(
        "RST19 · 星图研究工作台",
        options,
        Box::new(|cc| Ok(Box::new(ui::Desktop::new(cc)))),
    )
}
