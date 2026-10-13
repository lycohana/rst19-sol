fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match rst19_core::innovation_package::run_cli(&args) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("innovation-package error: {error}");
            std::process::exit(2);
        }
    }
}
