fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(rst19_core::spatial_injection::run_cli(&args));
}
