use bit_set::BitSet;
use image::RgbaImage;
use rustc_hash::FxHashMap as HashMap;

use crate::pixel::Pixel;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Direction {
    Top,
    Left,
    Bottom,
    Right,
}

impl Direction {
    fn opposite(&self) -> Self {
        match self {
            Direction::Top => Direction::Bottom,
            Direction::Left => Direction::Right,
            Direction::Bottom => Direction::Top,
            Direction::Right => Direction::Left,
        }
    }

    fn to_index(self) -> usize {
        match self {
            Direction::Top => 0,
            Direction::Left => 1,
            Direction::Bottom => 2,
            Direction::Right => 3,
        }
    }
}

pub const DIRECTIONS: [Direction; 4] = [
    Direction::Top,
    Direction::Left,
    Direction::Bottom,
    Direction::Right,
];

pub type PatternSet = BitSet;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Pattern(pub Vec<Pixel>);

impl Pattern {
    /// Get the part of the pattern that would overlap with the pattern to the direction of this
    /// pattern by removing the opposite sides last row/col, e.g. to get the left overlap remove
    /// the rightmost column, or to get the bottom overlap remove the top row.
    ///
    /// Note that when comparing to determine adjacent relations, seeing which patterns
    /// can be e.g. right of 1 compares between the right of 1 and the left of the other patterns.
    fn overlap(&self, direction: &Direction, width: u32, height: u32) -> Self {
        let (width, height) = (width as usize, height as usize);
        let length = self.0.len();
        assert_eq!(length, width * height);
        self.0
            .iter()
            .enumerate()
            .filter(|&(i, _)| match direction {
                Direction::Top => i < length - width,
                Direction::Left => i % width != width - 1,
                Direction::Bottom => i >= width,
                Direction::Right => i % width != 0,
            })
            .map(|(_, p)| *p)
            .collect()
    }

    /// The identity map for Pattern. A little silly, but this way D_8 can be a constant.
    fn identity(self, _: u32, _: u32) -> Self {
        self
    }

    /// Action of r in D_8 on a pattern, 90 degree clockwise rotation.
    fn rotate(self, width: u32, height: u32) -> Self {
        // a b c    g d a
        // d e f -> h e b
        // g h i    i f c
        // This will only work with square patterns
        assert_eq!(width, height);
        let mut pixels: Vec<(u32, Pixel)> = self
            .0
            .iter()
            .enumerate()
            // Index to u32, copy pixel
            .map(|(i, p)| (i as u32, *p))
            // Index to coordinates
            .map(|(i, p)| (i % width, i / width, p))
            // Rotate
            .map(|(x, y, p)| (width - 1 - y, x, p))
            // Coordinates back to index
            .map(|(x, y, p)| (x + y * width, p))
            .collect();
        pixels.sort_by_key(|(i, _)| *i);
        let pixels = pixels.into_iter().map(|(_, p)| p).collect();
        Pattern(pixels)
    }

    /// Action of s in D_8 on a pattern, horizontal reflection.
    fn reflect(self, width: u32, _: u32) -> Self {
        // a b c    c b a
        // d e f -> f e d
        // g h i    i h g
        let mut pixels: Vec<(u32, Pixel)> = self
            .0
            .iter()
            .enumerate()
            // Index to u32, copy pixel
            .map(|(i, p)| (i as u32, *p))
            // Index to coordinates
            .map(|(i, p)| (i % width, i / width, p))
            // Reflect
            .map(|(x, y, p)| (width - 1 - x, y, p))
            // Coordinates back to index
            .map(|(x, y, p)| (x + y * width, p))
            .collect();
        pixels.sort_by_key(|(i, _)| *i);
        let pixels = pixels.into_iter().map(|(_, p)| p).collect();
        Pattern(pixels)
    }

    /// Get the orbit of D8 containing this pattern, but only the different patterns (skip the
    /// identity action).
    fn orbit_of_d8(&self, width: u32, height: u32) -> [Pattern; 7] {
        type D8PatternAction = fn(Pattern, u32, u32) -> Pattern;

        const E: D8PatternAction = Pattern::identity;
        const R: D8PatternAction = Pattern::rotate;
        const S: D8PatternAction = Pattern::reflect;

        // D_8 (less the identity) is r, r^2, r^3, s, sr, sr^2, sr^3
        const D8: [[D8PatternAction; 4]; 7] = [
            [R, E, E, E],
            [R, R, E, E],
            [R, R, R, E],
            [S, E, E, E],
            [S, R, E, E],
            [S, R, R, E],
            [S, R, R, R],
        ];

        D8.map(|g| {
            g.iter()
                // Reverse the action so it can be written as a vec in the same order
                // as the expression (sr means rotate then reflect)
                .rev()
                .fold(self.clone(), |p, g| g(p, width, height))
        })
    }
}

/// The color of this pattern, the times this pattern occured, this patterns adjacency
/// compatibilities
struct PatternInfo(Pixel, u32, [PatternSet; DIRECTIONS.len()]);
impl PatternInfo {
    fn color(&self) -> Pixel {
        self.0
    }

    fn count(&self) -> u32 {
        self.1
    }

    fn adjacency(&self) -> &[PatternSet; DIRECTIONS.len()] {
        &self.2
    }
}

pub type PatternId = usize;
pub struct Patterns(Vec<PatternInfo>);

impl Patterns {
    pub fn from_image(
        image: &RgbaImage,
        pattern_width: u32,
        pattern_height: u32,
        sym: bool,
    ) -> Self {
        let (width, height) = image.dimensions();

        // Get the `pattern_width` by `pattern_height` patterns with top-left corner at (x, y),
        // wrapping around the sides of the image.
        let mut patterns = HashMap::default();
        for x in 0..width {
            for y in 0..height {
                let mut pixels = Vec::with_capacity((pattern_width * pattern_height) as usize);
                for dy in 0..pattern_height {
                    for dx in 0..pattern_width {
                        let pixel = *image.get_pixel((x + dx) % width, (y + dy) % height);
                        pixels.push(pixel.into());
                    }
                }
                // Count occurences of patterns as well
                patterns
                    .entry(Pattern(pixels))
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }

        // Augment pattern data with rotations and reflections
        if sym {
            patterns.clone().into_iter().for_each(|(pattern, count)| {
                pattern
                    .orbit_of_d8(pattern_width, pattern_height)
                    .into_iter()
                    .for_each(|p| {
                        patterns
                            .entry(p)
                            .and_modify(|c| *c += count)
                            .or_insert(count);
                    });
            });
        }

        // Sort the patterns by the number of times they occured to allow for deterministic
        // generation.
        let mut patterns: Vec<(Pattern, u32)> = patterns.into_iter().collect();
        patterns.sort_by_key(|(_, count)| *count);

        // From here on, we no longer care about the source image

        // Extract the relevant information from the patterns
        // 1. The color of the top-left corner
        // 2. The times this pattern occurred
        // 3. Its adjacency compatibilities
        let info = patterns
            .iter()
            .map(|(pattern, count)| {
                let pixel = pattern.0[0];
                let adjacency = DIRECTIONS.map(|direction| {
                    let overlap = pattern.overlap(&direction, pattern_width, pattern_height);
                    // Patterns than can be to the `direction` of this pattern
                    patterns
                        .iter()
                        .enumerate()
                        .filter(|(_, (other, _))| {
                            overlap
                                == other.overlap(
                                    &direction.opposite(),
                                    pattern_width,
                                    pattern_height,
                                )
                        })
                        .map(|(i, _)| i)
                        .collect()
                });
                PatternInfo(pixel, *count, adjacency)
            })
            .collect();

        Patterns(info)
    }

    fn set(&self) -> PatternSet {
        BitSet::with_capacity(self.len())
    }

    pub fn set_all(&self) -> PatternSet {
        (0..self.len()).collect()
    }

    pub fn set_with(&self, pattern: PatternId) -> PatternSet {
        let mut set = self.set();
        set.insert(pattern);
        set
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Get the number of times each pattern occured in the source image, where patterns not in the
    /// mask are set to zero.
    pub fn frequencies<'a>(&'a self, mask: &'a PatternSet) -> impl Iterator<Item = u32> + 'a {
        self.0
            .iter()
            .enumerate()
            .map(|(i, info)| if mask.contains(i) { info.count() } else { 0 })
    }

    /// Get the patterns that can be to the `direction` of `pattern`
    pub fn compatibles(&self, pattern: PatternId, direction: &Direction) -> &PatternSet {
        &self.0[pattern].adjacency()[direction.to_index()]
    }

    /// Get the display color of this pattern, obtained from the top-left pixel.
    pub fn color(&self, pattern: PatternId) -> Pixel {
        self.0[pattern].color()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern_builder(reds: &[u8]) -> Pattern {
        let pixels = reds.into_iter().map(|r| Pixel([*r, 0, 0, 0])).collect();
        Pattern(pixels)
    }

    #[test]
    fn test_rotate() {
        // 0 1 2
        // 3 4 5
        // 6 7 8
        let pattern = Pattern(vec![
            Pixel([0, 0, 0, 0]),
            Pixel([1, 0, 0, 0]),
            Pixel([2, 0, 0, 0]),
            Pixel([3, 0, 0, 0]),
            Pixel([4, 0, 0, 0]),
            Pixel([5, 0, 0, 0]),
            Pixel([6, 0, 0, 0]),
            Pixel([7, 0, 0, 0]),
            Pixel([8, 0, 0, 0]),
        ]);
        // 6 3 0
        // 7 4 1
        // 8 5 2
        let rotation = Pattern(vec![
            Pixel([6, 0, 0, 0]),
            Pixel([3, 0, 0, 0]),
            Pixel([0, 0, 0, 0]),
            Pixel([7, 0, 0, 0]),
            Pixel([4, 0, 0, 0]),
            Pixel([1, 0, 0, 0]),
            Pixel([8, 0, 0, 0]),
            Pixel([5, 0, 0, 0]),
            Pixel([2, 0, 0, 0]),
        ]);
        assert_eq!(pattern.rotate(3, 3), rotation);
    }

    #[test]
    fn test_reflect() {
        // 0 1 2
        // 3 4 5
        // 6 7 8
        let pattern = Pattern(vec![
            Pixel([0, 0, 0, 0]),
            Pixel([1, 0, 0, 0]),
            Pixel([2, 0, 0, 0]),
            Pixel([3, 0, 0, 0]),
            Pixel([4, 0, 0, 0]),
            Pixel([5, 0, 0, 0]),
            Pixel([6, 0, 0, 0]),
            Pixel([7, 0, 0, 0]),
            Pixel([8, 0, 0, 0]),
        ]);
        // 2 1 0
        // 5 4 3
        // 8 7 6
        let reflection = Pattern(vec![
            Pixel([2, 0, 0, 0]),
            Pixel([1, 0, 0, 0]),
            Pixel([0, 0, 0, 0]),
            Pixel([5, 0, 0, 0]),
            Pixel([4, 0, 0, 0]),
            Pixel([3, 0, 0, 0]),
            Pixel([8, 0, 0, 0]),
            Pixel([7, 0, 0, 0]),
            Pixel([6, 0, 0, 0]),
        ]);
        assert_eq!(pattern.reflect(3, 3), reflection)
    }

    #[test]
    fn test_d8() {
        // 0 1 2
        // 3 4 5
        // 6 7 8
        let pattern = pattern_builder(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let orbit = [
            // 6 3 0
            // 7 4 1
            // 8 5 2
            pattern_builder(&[6, 3, 0, 7, 4, 1, 8, 5, 2]),
            // 8 7 6
            // 5 4 3
            // 2 1 0
            pattern_builder(&[8, 7, 6, 5, 4, 3, 2, 1, 0]),
            // 2 5 8
            // 1 4 7
            // 0 3 6
            pattern_builder(&[2, 5, 8, 1, 4, 7, 0, 3, 6]),
            // 2 1 0
            // 5 4 3
            // 8 7 6
            pattern_builder(&[2, 1, 0, 5, 4, 3, 8, 7, 6]),
            // 0 3 6
            // 1 4 7
            // 2 5 8
            pattern_builder(&[0, 3, 6, 1, 4, 7, 2, 5, 8]),
            // 6 7 8
            // 3 4 5
            // 0 1 2
            pattern_builder(&[6, 7, 8, 3, 4, 5, 0, 1, 2]),
            // 8 5 2
            // 7 4 1
            // 6 3 0
            pattern_builder(&[8, 5, 2, 7, 4, 1, 6, 3, 0]),
        ];
        assert_eq!(pattern.orbit_of_d8(3, 3), orbit)
    }
}
