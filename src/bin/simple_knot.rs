use image::io::Reader as ImageReader;
use wave_function_collapse::*;

fn main() {
    let source = ImageReader::open("data/SimpleKnot.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgba8();

    wave_function_collapse(&source, 3, 3, 200, 200, true, true, 21);

    println!("FINISHED");
}
