fn main() {
    if let Err(error) =
        rst19_core::innovation_cli::run(&std::env::args().skip(1).collect::<Vec<_>>())
    {
        eprintln!("rst19-innovation: {error}");
        std::process::exit(2);
    }
}
