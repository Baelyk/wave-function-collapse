use image::codecs::gif::GifEncoder;
use image::Frame;
use image::Rgba;
use image::RgbaImage;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::fs::File;
use std::hash::Hash;
use webp_animation::Encoder;

/// A set of `usize`s to act as a pattern set
#[derive(Debug, Clone, PartialEq, Eq)]
struct Set(Vec<bool>);

impl Set {
    fn with_size(size: usize) -> Self {
        Set(vec![false; size])
    }

    fn with_all(size: usize) -> Self {
        Set(vec![true; size])
    }

    fn with_only(n: usize, size: usize) -> Self {
        let mut set = Self::with_size(size);
        set[n] = true;
        set
    }

    fn contains(&self, n: usize) -> bool {
        self[n]
    }

    fn len(&self) -> usize {
        self.iter().count()
    }

    fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }

    fn iter(&'_ self) -> impl Iterator<Item = usize> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter(|(_, &included)| included)
            .map(|(i, _)| i)
    }
}

impl std::ops::Index<usize> for Set {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Set {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl std::ops::BitAnd<&Set> for &Set {
    type Output = Set;

    fn bitand(self, rhs: &Set) -> Self::Output {
        assert!(self.0.len() == rhs.0.len());
        self.0
            .iter()
            .zip(rhs.0.iter())
            .map(|(l, r)| *l && *r)
            .collect()
    }
}

impl std::ops::BitOr<&Set> for &Set {
    type Output = Set;

    fn bitor(self, rhs: &Set) -> Self::Output {
        assert!(self.0.len() == rhs.0.len());
        self.0
            .iter()
            .zip(rhs.0.iter())
            .map(|(l, r)| *l || *r)
            .collect()
    }
}
impl std::fmt::Display for Set {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let contents = self.iter().enumerate().fold(String::new(), |acc, (i, n)| {
            if i == 0 {
                format!("{n}")
            } else {
                format!("{acc}, {n}")
            }
        });
        write!(f, "{{{}}}", contents)
    }
}

impl FromIterator<bool> for Set {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        Set(iter.into_iter().collect())
    }
}

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
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct Pattern(Vec<Pixel>);

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
type PatternSet = Set;
struct Patterns(Vec<PatternInfo>);

impl Patterns {
    fn from_image(image: &RgbaImage, pattern_width: u32, pattern_height: u32) -> Self {
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

        // Sort the patterns by the number of times they occured to allow for deterministic
        // generation.
        let mut patterns: Vec<(Pattern, u32)> = patterns.into_iter().collect();
        patterns.sort_by_key(|(_, count)| *count);

        // TODO: Augment pattern data with rotations and reflections

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
                        .map(|(other, _)| {
                            overlap
                                == other.overlap(
                                    &direction.opposite(),
                                    pattern_width,
                                    pattern_height,
                                )
                        })
                        .collect()
                });
                PatternInfo(pixel, *count, adjacency)
            })
            .collect();

        Patterns(info)
    }

    fn set(&self) -> PatternSet {
        Set::with_size(self.len())
    }

    fn set_all(&self) -> PatternSet {
        Set::with_all(self.len())
    }

    fn set_with(&self, pattern: PatternId) -> PatternSet {
        Set::with_only(pattern, self.len())
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

fn initialize(
    image: &RgbaImage,
    n: u32,
    m: u32,
    width: u32,
    height: u32,
) -> (Patterns, Wave, Vec<f32>) {
    // Find the patterns from the source image
    let patterns = Patterns::from_image(image, n, m);
    eprintln!("Found {} patterns", patterns.len());

    // Initialize the wave so each cell is in a superposition of all patterns
    let all_patterns = patterns.set_all();
    let wave: Vec<PatternSet> = (0..(width * height))
        .map(|_| all_patterns.clone())
        .collect();

    // Each cell has the same entropy at the start
    let entropy = calculate_entropy(&all_patterns, &patterns);
    let entropy = [entropy].repeat(wave.len());

    (patterns, wave, entropy)
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
    let mut changes: HashMap<usize, PatternSet> = HashMap::default();

    while !queue.is_empty() {
        // `cell` is the cell we're propagating the collapse through, and `post_collapse` is
        // the set of patterns this cell could be as a result of the collapse.
        let (cell, post_collapse) = queue.pop().unwrap();
        assert!(!post_collapse.is_empty());

        // Only propagate from this cell if it will have fewer options
        let prev = wave[cell].len();
        wave[cell] = &wave[cell] & &post_collapse;

        // If this cell has no possibilities, the algorithm has run into a contradiction
        if wave[cell].is_empty() {
            return PropagationResult::Contradiction;
        }

        if prev != wave[cell].len() {
            changes.insert(cell, wave[cell].clone());
            // Propagate changes to neighbors
            get_neighbors(cell, width, height)
                .into_iter()
                .zip(DIRECTIONS.into_iter())
                .filter(|(neighbor, _)| neighbor.is_some())
                .for_each(|(neighbor, direction)| {
                    // Compatibles is the set of all patterns that can be to `direction` of any
                    // of this cell's possible patterns.
                    let compatibles = wave[cell]
                        .iter()
                        .map(|i| patterns.compatibles(i, &direction))
                        .fold(patterns.set(), |a, b| &a | b);
                    queue.push((neighbor.unwrap(), compatibles));
                });
        }
    }

    PropagationResult::Success(changes)
}

type Collapse = (usize, PatternId);
type CollapseHistory = Vec<(Collapse, HashMap<usize, PatternSet>)>;

pub fn wave_function_collapse(
    image: &RgbaImage,
    n: u32,
    m: u32,
    width: u32,
    height: u32,
    anim: bool,
    seed: u64,
) {
    // Number of digits to display the number of cells
    let rem_width = format!("{}", width * height).len();

    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let (mut patterns, mut wave, mut entropy) = initialize(image, n, m, width, height);
    let mut restarts = 0;
    // Observation stack
    let mut history: CollapseHistory = Vec::new();
    // Overserve-propagate-update loop
    let mut iter_num = 0;
    let time = std::time::Instant::now();
    loop {
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
                history.push((collapse, changes.clone()));
                changes
                    .into_iter()
                    .for_each(|(cell, mask)| entropy[cell] = calculate_entropy(&mask, &patterns));
            }
            // Restart
            PropagationResult::Contradiction => {
                eprintln!("\nRestarting...");
                restarts += 1;
                (patterns, wave, entropy) = initialize(image, n, m, width, height);
                history.drain(0..);
                // Observe again
                continue;
            }
        }
    }
    println!(
        "\rFinished after {} restarts and {} iterations, {}us per iter",
        restarts,
        iter_num - 1,
        time.elapsed().as_micros() / iter_num
    );

    // Animate
    if anim {
        animate(&history, &patterns, width, height, 10);
        let image = draw(&wave, &patterns, width as usize, height as usize, 10);
        image
            .save("data/output.png")
            .expect("Unable to save final png");
    }
}

fn animate(history: &CollapseHistory, patterns: &Patterns, width: u32, height: u32, scale: u32) {
    let file = File::create("data/output.gif").unwrap();
    let mut gif = GifEncoder::new_with_speed(file, 30);
    gif.set_repeat(image::codecs::gif::Repeat::Infinite)
        .unwrap();
    let mut encoder = Encoder::new((scale * width, scale * height)).unwrap();
    let format_width = format!("{}", history.len()).len();

    // The first frame is the average of all the patterns
    let background = get_average_color(&patterns.set_all(), patterns);
    let mut frame = background.repeat((scale * width * scale * height) as usize);

    let duration = 5;
    // Delay is ms per iteration, bounded in [20, 1000]
    let delay = (duration * 1000) / history.len() as i32;
    let delay = std::cmp::max(20, delay);
    let delay = std::cmp::min(1000, delay);
    let skip_by = std::cmp::max(1, history.len() as i32 / (duration * (1000 / delay))) as usize;
    // Create the next frame by copying the last and only modifying the changed cells
    history.iter().enumerate().for_each(|(i, (_, diffs))| {
        eprint!(
            "\rAnimating {:format_width$}/{:format_width$}",
            i + 1,
            history.len()
        );
        if i % skip_by == 0 {
            encoder
                .add_frame(&frame, (i / skip_by) as i32 * delay)
                .unwrap();
            let gif_frame = Frame::from_parts(
                RgbaImage::from_vec(scale * width, scale * height, frame.clone()).unwrap(),
                0,
                0,
                image::Delay::from_numer_denom_ms(delay as u32, 1),
            );
            gif.encode_frame(gif_frame).unwrap();
        }
        diffs.iter().for_each(|(index, mask)| {
            let index = *index as u32;
            let (x, y) = (index % width, index / width);
            let color = get_average_color(mask, patterns);
            put_pixel_scaled(&mut frame, scale, width, x, y, *color)
        });
    });
    encoder
        .add_frame(&frame, (history.len() / skip_by + 1) as i32 * delay)
        .unwrap();
    let gif_frame = Frame::from_parts(
        RgbaImage::from_vec(scale * width, scale * height, frame.clone()).unwrap(),
        0,
        0,
        image::Delay::from_numer_denom_ms(((history.len() / skip_by + 1) as i32 * delay) as u32, 1),
    );
    gif.encode_frame(gif_frame.clone()).unwrap();
    println!("delay: {:?}", gif_frame.delay());

    let data = encoder
        .finalize(1000 + ((history.len() / skip_by + 1) as i32 * delay))
        .unwrap();
    std::fs::write("data/output.webp", data).unwrap();
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
}
