fn main() {
    if let Err(error) = rst19_core::attitude::run_cli(&std::env::args().skip(1).collect::<Vec<_>>())
    {
        eprintln!("rst19-aux-audit: {error}");
        std::process::exit(2);
    }
}
