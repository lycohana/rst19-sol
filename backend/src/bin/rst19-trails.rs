fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    std::process::exit(rst19_core::single_frame_trail_audit::run_cli(&args));
}
