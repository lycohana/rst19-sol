fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match rst19_core::detection_sweep::run_cli(&args) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("rst19-sweep: {error}");
            std::process::exit(2);
        }
    }
}
