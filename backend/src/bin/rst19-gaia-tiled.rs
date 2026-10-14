fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    std::process::exit(rst19_core::gaia_tiled::run_cli(&args));
}
