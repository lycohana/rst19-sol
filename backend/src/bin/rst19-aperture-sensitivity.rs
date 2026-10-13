fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    std::process::exit(rst19_core::aperture_sensitivity::run_cli(&args));
}
