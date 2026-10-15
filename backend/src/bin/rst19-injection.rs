fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|v| v == "--help" || v == "-h") {
        println!("{}", rst19_core::injection::HELP);
        return;
    }
    match rst19_core::injection::run_desktop(&args) {
        Ok(result) => println!("{result}"),
        Err(error) => {
            eprintln!("rst19-injection: {error}");
            std::process::exit(2);
        }
    }
}
