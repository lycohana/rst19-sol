fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match rst19_core::catalog_merge::run_cli(&args) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("gaia-merge: {error}");
            std::process::exit(2);
        }
    }
}
