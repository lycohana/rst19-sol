fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|v| v == "--help" || v == "-h") {
        println!("{}", rst19_core::native_package::HELP);
        return;
    }
    match rst19_core::native_package::run(&args) {
        Ok(value) => println!(
            "{}",
            serde_json::to_string_pretty(&value).expect("package result")
        ),
        Err(error) => {
            eprintln!("rst19-package: {error}");
            std::process::exit(2);
        }
    }
}
