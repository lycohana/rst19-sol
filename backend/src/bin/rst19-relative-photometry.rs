fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    std::process::exit(rst19_core::relative_photometry::run_cli(&args));
}
