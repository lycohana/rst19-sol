fn main() {
    if let Err(error) = rst19_core::sequence_cli::run(&std::env::args().skip(1).collect::<Vec<_>>())
    {
        eprintln!("rst19-sequence: {error}");
        std::process::exit(2);
    }
}
