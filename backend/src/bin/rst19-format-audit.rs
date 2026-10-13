fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if let Err(error) = rst19_core::format_audit::run_cli(&args) {
        eprintln!("rst19-format-audit: {error}");
        std::process::exit(2);
    }
}
