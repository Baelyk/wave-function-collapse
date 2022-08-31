use image::io::Reader as ImageReader;
use wave_function_collapse::*;

fn main() {
    let source = ImageReader::open("data/Rooms.png")
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();

    wave_function_collapse(&source, 3, 3, 300, 300);
    println!("FINISHED");
}
