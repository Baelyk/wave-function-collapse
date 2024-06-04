use bit_set::BitSet;
use image::codecs::gif::GifEncoder;
use image::Frame;
use image::Rgba;
use image::RgbaImage;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::fs::File;
use std::hash::Hash;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

type PatternSet = BitSet;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Direction {
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

const DIRECTIONS: [Direction; 4] = [
    Direction::Top,
    Direction::Left,
    Direction::Bottom,
    Direction::Right,
];

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
struct Pixel([u8; 4]);

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

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct Pattern(Vec<Pixel>);

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

type PatternId = usize;
struct Patterns(Vec<PatternInfo>);

impl Patterns {
    fn from_image(image: &RgbaImage, pattern_width: u32, pattern_height: u32, sym: bool) -> Self {
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

    fn set_all(&self) -> PatternSet {
        (0..self.len()).collect()
    }

    fn set_with(&self, pattern: PatternId) -> PatternSet {
        let mut set = self.set();
        set.insert(pattern);
        set
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    /// Get the number of times each pattern occured in the source image, where patterns not in the
    /// mask are set to zero.
    fn frequencies<'a>(&'a self, mask: &'a PatternSet) -> impl Iterator<Item = u32> + 'a {
        self.0
            .iter()
            .enumerate()
            .map(|(i, info)| if mask.contains(i) { info.count() } else { 0 })
    }

    /// Get the patterns that can be to the `direction` of `pattern`
    fn compatibles(&self, pattern: PatternId, direction: &Direction) -> &PatternSet {
        &self.0[pattern].adjacency()[direction.to_index()]
    }

    /// Get the display color of this pattern, obtained from the top-left pixel.
    fn color(&self, pattern: PatternId) -> Pixel {
        self.0[pattern].color()
    }
}

fn calculate_entropy(mask: &PatternSet, patterns: &Patterns) -> f32 {
    // Entropy is -sum(p_i * log(p_i)), p_i = occurences_of_i / all_occurences
    let frequencies: Vec<f32> = patterns
        .frequencies(mask)
        .map(|f| f as f32)
        .filter(|f| *f != 0.0)
        .collect();
    let total: f32 = frequencies.iter().sum();
    -frequencies
        .iter()
        .map(|f| f / total)
        .map(|p| p * p.ln())
        .sum::<f32>()
}

fn get_neighbors(cell: usize, width: u32, height: u32) -> [Option<usize>; 4] {
    const WRAP: bool = true;
    let width = width as usize;
    let height = height as usize;
    // Top, Left, Bottom, Right
    if WRAP {
        [
            // Top
            Some(if cell >= width {
                cell - width
            } else {
                width * height - width + cell
            }),
            // Left
            Some(if cell % width > 0 {
                cell - 1
            } else {
                (cell / width) * width + width - 1
            }),
            // Bottom
            Some(if cell + width < width * height {
                cell + width
            } else {
                cell % width
            }),
            // Right
            Some(if (cell + 1) % width != 0 {
                cell + 1
            } else {
                (cell / width) * width
            }),
        ]
    } else {
        [
            // Top
            cell.checked_sub(width),
            // Left
            cell.checked_sub(1),
            // Bottom
            if cell + width < width * height {
                Some(cell + width)
            } else {
                None
            },
            // Right
            if (cell + 1) % width != 0 {
                Some(cell + 1)
            } else {
                None
            },
        ]
    }
}

type Wave = Vec<PatternSet>;

fn initialize(patterns: &Patterns, width: u32, height: u32) -> (Wave, Vec<f32>) {
    log("1");
    // Initialize the wave so each cell is in a superposition of all patterns
    let all_patterns = patterns.set_all();
    let mut wave: Vec<PatternSet> = Vec::with_capacity((width * height) as usize);
    wave.resize((width * height) as usize, all_patterns.clone());
    log("2");

    // Each cell has the same entropy at the start
    let entropy = calculate_entropy(&all_patterns, patterns);
    let entropy = [entropy].repeat(wave.len());
    log("3");

    (wave, entropy)
}

fn observe(
    wave: &Wave,
    entropy: &[f32],
    patterns: &Patterns,
    rng: &mut impl Rng,
) -> Option<Collapse> {
    // Get the cell with the least nonzero entropy
    if let Some((cell, _)) = entropy
        .iter()
        .enumerate()
        .filter(|&(_, e)| *e != 0.0)
        .reduce(|(i, min), (j, e)| if e < min { (j, e) } else { (i, min) })
    {
        // Randomly pick a pattern from the cell's possible patterns, weighted by times that
        // pattern occurred in the seed
        let dist = WeightedIndex::new(patterns.frequencies(&wave[cell])).unwrap();
        let pattern = dist.sample(rng);
        Some((cell, pattern))
    } else {
        // All cells are collapsed
        None
    }
}

enum PropagationResult {
    Success(HashMap<usize, PatternSet>),
    Contradiction,
}

fn propagate(
    collapse: Collapse,
    wave: &mut Wave,
    patterns: &Patterns,
    width: u32,
    height: u32,
) -> PropagationResult {
    let (cell, pattern) = collapse;
    // Starting with the collapsed cell, propagating changes
    let mut queue: Vec<(usize, PatternSet)> = vec![(cell, patterns.set_with(pattern))];
    // Keep track of changed cells to know which to recalculate entropy for
    let mut changed = BitSet::with_capacity(patterns.len());

    while !queue.is_empty() {
        // `cell` is the cell we're propagating the collapse through, and `post_collapse` is
        // the set of patterns this cell could be as a result of the collapse.
        let (cell, post_collapse) = queue.pop().unwrap();
        debug_assert!(!post_collapse.is_empty());

        // Only propagate from this cell if it will have fewer options
        let prev = wave[cell].len();
        wave[cell].intersect_with(&post_collapse);

        // If this cell has no possibilities, the algorithm has run into a contradiction
        if wave[cell].is_empty() {
            return PropagationResult::Contradiction;
        }

        if prev != wave[cell].len() {
            changed.insert(cell);
            // Propagate changes to neighbors
            get_neighbors(cell, width, height)
                .into_iter()
                .zip(DIRECTIONS.into_iter())
                .filter(|(neighbor, _)| neighbor.is_some())
                .for_each(|(neighbor, direction)| {
                    // Compatibles is the set of all patterns that can be to `direction` of any
                    // of this cell's possible patterns.
                    let mut compatibles = wave[cell]
                        .iter()
                        .map(|i| patterns.compatibles(i, &direction));
                    let mut compatible = compatibles.next().unwrap().clone();
                    compatibles.for_each(|set| compatible.union_with(set));
                    queue.push((neighbor.unwrap(), compatible));
                });
        }
    }

    PropagationResult::Success(
        changed
            .into_iter()
            .map(|cell| (cell, wave[cell].clone()))
            .collect(),
    )
}

type Collapse = (usize, PatternId);
type CollapseHistory = Vec<(Collapse, HashMap<usize, PatternSet>)>;

#[wasm_bindgen]
pub fn wave_function_collapse(
    image_buffer: &[u8],
    n: u32,
    m: u32,
    width: u32,
    height: u32,
    anim: bool,
    sym: bool,
    seed: u32,
) -> Vec<u8> {
    log("Hello!");
    let image: RgbaImage = image::load_from_memory(image_buffer)
        .expect("Image should load")
        .into_rgba8();

    log("a");
    // Find the patterns from the source image
    let patterns = Patterns::from_image(&image, n, m, sym);
    eprintln!("Found {} patterns", patterns.len());

    log("b");
    // Number of digits to display the number of cells
    let rem_width = format!("{}", width * height).len();

    log("c");
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed.into());
    let (mut wave, mut entropy) = initialize(&patterns, width, height);
    log("c1");

    // Observation stack
    let mut history: CollapseHistory = Vec::new();
    log("c2");

    // Overserve-propagate-update loop
    let mut restarts = 0;
    let mut iter_num = 0;
    log("c3");
    log("d");
    loop {
        log("loop");
        iter_num += 1;
        // Observe and collapse a new cell or break if there are no unobserved cells
        let collapse: Collapse = match observe(&wave, &entropy, &patterns, &mut rng) {
            Some((c, p)) => (c, p),
            None => break,
        };

        // Logging
        let remaining = entropy.iter().filter(|&e| *e != 0.0).count() as u32;
        let percent = 100.0 * ((width * height) - remaining) as f32 / (width * height) as f32;
        eprint!(
            "\r{}: {} obs, {} restarts, {:>rem_width$} rem, {:2.0}%",
            iter_num,
            history.len(),
            restarts,
            remaining,
            percent
        );

        // Propagate this collapse
        match propagate(collapse, &mut wave, &patterns, width, height) {
            // Update entropies
            PropagationResult::Success(changes) => {
                changes
                    .iter()
                    .for_each(|(cell, mask)| entropy[*cell] = calculate_entropy(mask, &patterns));
                history.push((collapse, changes))
            }
            // Restart
            PropagationResult::Contradiction => {
                eprintln!("\nRestarting...");
                restarts += 1;
                (wave, entropy) = initialize(&patterns, width, height);
                history.drain(0..);
                // Observe again
                continue;
            }
        }
    }
    log("e");

    eprintln!(
        "\rFinished after {} restarts and {} iterations",
        restarts,
        iter_num - 1,
    );
    log("f");

    // Animate
    if anim {
        animate(&history, &patterns, width, height, 10);
    }
    log("g");
    let image = draw(&wave, &patterns, width as usize, height as usize, 10);
    log("h");
    //image
    //.save("data/output.png")
    //.expect("Unable to save final png");
    log("g");

    let mut bytes = Vec::new();

    image
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("Unable to write");

    bytes
}

fn animate(history: &CollapseHistory, patterns: &Patterns, width: u32, height: u32, scale: u32) {
    unimplemented!()
}

fn put_pixel_scaled(data: &mut [u8], scale: u32, width: u32, x: u32, y: u32, color: [u8; 4]) {
    for dx in 0..scale {
        for dy in 0..scale {
            let (x, y) = (scale * x + dx, scale * y + dy);
            // 4 channels, there are scale * width pixels in a row
            let index = 4 * (x + y * scale * width) as usize;
            data[index..index + 4]
                .iter_mut()
                .zip(color.iter())
                .for_each(|(data, channel)| *data = *channel);
        }
    }
}

/// Get the average color of patterns in the mask weighted by their frequency
fn get_average_color(mask: &PatternSet, patterns: &Patterns) -> Pixel {
    let frequencies: Vec<f32> = patterns.frequencies(mask).map(|f| f as f32).collect();
    let total: f32 = frequencies.iter().sum();
    Pixel(
        frequencies
            .iter()
            .enumerate()
            .map(|(pattern, freq)| patterns.color(pattern).map(|c| c as f32 * freq / total))
            .reduce(|a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
            .unwrap()
            .map(|c| c as u8),
    )
}

fn draw(
    wave: &[PatternSet],
    patterns: &Patterns,
    width: usize,
    height: usize,
    scale: usize,
) -> image::RgbaImage {
    let mut image = image::RgbaImage::new((width * scale) as u32, (height * scale) as u32);
    (0..width).for_each(|x| {
        (0..height).for_each(|y| {
            let possibilities = &wave[(x + width * y) as usize];
            let color = get_average_color(possibilities, patterns);
            (0..scale).for_each(|dx| {
                (0..scale).for_each(|dy| {
                    image.put_pixel(
                        (scale * x + dx) as u32,
                        (scale * y + dy) as u32,
                        Rgba(*color),
                    );
                })
            })
        })
    });
    //image.save(filename).unwrap();
    image
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

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen(start)]
pub fn greet() {
    log("Hello from Rust!");
}
