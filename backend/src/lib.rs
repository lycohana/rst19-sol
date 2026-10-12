#![recursion_limit = "512"]
//! Native scientific implementation. No Python interpreter, FFI, or subprocess calls.
pub mod compatible_spatial;
pub mod coordinates;
pub mod deblend;
pub mod detection;
pub mod filtering;
pub mod filtering32;
pub mod fits;
pub mod measurement;
pub mod models;
pub mod precision;
pub mod psf;
pub mod spatial;
pub mod statistics;
pub mod statistics32;

pub type Result<T> = std::result::Result<T, String>;
