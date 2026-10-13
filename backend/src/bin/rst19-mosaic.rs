fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if let Err(error) = rst19_core::mosaic::run_cli(&args) {
        eprintln!("rst19-mosaic: {error}");
        std::process::exit(2);
    }
}
