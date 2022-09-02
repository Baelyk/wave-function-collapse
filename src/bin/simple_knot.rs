use image::io::Reader as ImageReader;
use wave_function_collapse::*;

fn main() {
    let source = ImageReader::open("data/Flowers.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgba8();

    wave_function_collapse(&source, 3, 3, 100, 100, true);
    println!("FINISHED");
}
