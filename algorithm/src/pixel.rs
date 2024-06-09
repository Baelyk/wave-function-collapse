use image::Rgba;

use crate::pattern::Pattern;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Pixel(pub [u8; 4]);

impl std::ops::Index<usize> for Pixel {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::Deref for Pixel {
    type Target = [u8; 4];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl FromIterator<Pixel> for Pattern {
    fn from_iter<I: IntoIterator<Item = Pixel>>(iter: I) -> Self {
        let pixels = iter.into_iter().collect();
        Pattern(pixels)
    }
}

impl From<Pixel> for Rgba<u8> {
    fn from(pixel: Pixel) -> Self {
        Self::from(pixel.0)
    }
}

impl From<Rgba<u8>> for Pixel {
    fn from(rgba: Rgba<u8>) -> Self {
        Self(rgba.0)
    }
}
