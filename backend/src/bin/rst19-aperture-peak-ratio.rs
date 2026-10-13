fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    std::process::exit(rst19_core::aperture_peak_ratio::run_cli(&args));
}
