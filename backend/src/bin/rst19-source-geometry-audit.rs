fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if let Err(error) = rst19_core::source_geometry_audit::run_cli(&args) {
        eprintln!("rst19-source-geometry-audit: {error}");
        std::process::exit(2);
    }
}
