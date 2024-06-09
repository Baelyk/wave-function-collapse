use log::debug;
use wasm_bindgen::prelude::*;

pub use crate::wave::WaveFunctionCollapse;

mod pattern;
mod pixel;
mod wave;

fn setup_logging() -> Result<(), fern::InitError> {
    fern::Dispatch::new()
        .chain(fern::Output::call(console_log::log))
        .apply()?;
    Ok(())
}

#[wasm_bindgen(start)]
pub fn init() -> Result<(), String> {
    setup_logging().map_err(|e| e.to_string())?;
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    debug!("Hello from Rust!");
    Ok(())
}
