use image::codecs::gif::GifEncoder;
use image::io::Reader as ImageReader;
use image::Frame;
use image::ImageResult;
use image::Rgb;
use image::Rgba;
use image::RgbaImage;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::hash::{Hash, Hasher};
use webp_animation::Encoder;

pub fn add(left: usize, right: usize) -> usize {
    left + right
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

    fn to_index(&self) -> usize {
        match self {
            Direction::Top => 0,
            Direction::Left => 1,
            Direction::Bottom => 2,
            Direction::Right => 3,
        }
    }

    fn from_index(index: usize) -> Self {
        match index {
            0 => Direction::Top,
            1 => Direction::Left,
            2 => Direction::Bottom,
            3 => Direction::Right,
            _ => panic!("Invalid direction index {index}"),
        }
    }
}

const DIRECTIONS: [Direction; 4] = [
    Direction::Top,
    Direction::Left,
    Direction::Bottom,
    Direction::Right,
];

type PatternId = usize;

struct Counter<T: Eq + Hash> {
    counter: HashMap<T, u32>,
}

impl<T: Eq + Hash> Counter<T> {
    fn with_capacity(capacity: usize) -> Self {
        Counter {
            counter: HashMap::with_capacity(capacity),
        }
    }

    fn insert(&mut self, element: T) {
        self.counter
            .entry(element)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, T, u32> {
        self.counter.iter()
    }

    pub fn len(&self) -> usize {
        self.counter.len()
    }

    pub fn into_keys(self) -> std::collections::hash_map::IntoKeys<T, u32> {
        self.counter.into_keys()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct Pattern {
    pixels: Vec<Rgb<u8>>,
}

impl Pattern {
    /// Get the part of the pattern that would overlap with the pattern to the direction of this
    /// pattern by removing the opposite sides last row/col, e.g. to get the left overlap remove
    /// the rightmost column, or to get the bottom overlap remove the top row.
    ///
    /// Note that when comparing to determine adjacent relations, seeing which patterns
    /// can be e.g. right of 1 compares between the right of 1 and the left of the other patterns.
    fn overlap(&self, direction: &Direction, width: usize, height: usize) -> Vec<Rgb<u8>> {
        self.pixels
            .iter()
            .enumerate()
            .filter(|&(i, _)| match direction {
                Direction::Top => i < self.pixels.len() - width,
                Direction::Left => i % width != width - 1,
                Direction::Bottom => i >= width,
                Direction::Right => i % width != 0,
            })
            .map(|(_, p)| *p)
            .collect()
    }

    fn render(&self, filename: String, width: usize, height: usize, scale: usize) {
        let mut image = image::RgbImage::new((width * scale) as u32, (height * scale) as u32);
        (0..width).for_each(|x| {
            (0..height).for_each(|y| {
                let color = self.pixels[x + width * y];
                (0..scale).for_each(|dx| {
                    (0..scale).for_each(|dy| {
                        image.put_pixel((scale * x + dx) as u32, (scale * y + dy) as u32, color);
                    })
                })
            })
        });
        image.save(filename).unwrap();
    }
}

struct Patterns {
    patterns: Vec<Pattern>,
    /// PatternNum -> (pattern_occurences, adjacency compatibilities)
    info: HashMap<usize, (u32, [HashSet<usize>; 4])>,
}

impl Patterns {
    fn new(patterns: Counter<Pattern>, width: usize, height: usize) -> Self {
        let mut info = HashMap::with_capacity(patterns.len());

        // Find the others this pattern can overlap with
        patterns
            .iter()
            .enumerate()
            .for_each(|(i, (pattern, count))| {
                let overlapable = DIRECTIONS.map(|direction| {
                    let overlap = pattern.overlap(&direction, width, height);
                    // List of patterns than can be to the `direction` of this pattern
                    patterns
                        .iter()
                        .enumerate()
                        .filter(|&(_, (other, _))| {
                            overlap == other.overlap(&direction.opposite(), width, height)
                        })
                        .map(|(j, _)| j)
                        .collect()
                });
                info.insert(i, (*count, overlapable));
            });

        let patterns: Vec<Pattern> = patterns.into_keys().collect();

        patterns.iter().enumerate().for_each(|(i, pattern)| {
            pattern.render(format!("data/pattern_{i}.png"), width, height, 10)
        });

        Patterns { patterns, info }
    }

    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    fn frequencies(&self, mask: &HashSet<usize>) -> Vec<u32> {
        (0..self.len())
            .map(|i| {
                if mask.contains(&i) {
                    self.info.get(&i).unwrap().0
                } else {
                    0
                }
            })
            .collect()
    }

    /// Get the patterns that can be to the `direction` of `pattern`
    fn compatibles(&self, pattern: usize, direction: &Direction) -> HashSet<usize> {
        self.info.get(&pattern).unwrap().1[direction.to_index()].clone()
    }

    fn top_left_color(&self, pattern: usize) -> Rgb<u8> {
        self.patterns[pattern].pixels[0]
    }
}

type Image = image::RgbImage; // image::ImageBuffer<Rgb<u8>, Vec<u8>>;

fn find_patterns(image: &Image, n: u32, m: u32) -> Patterns {
    let (width, height) = image.dimensions();

    // TODO: Augment pattern data with rotations and reflections
    // Get the n by m patterns with top-left corner at (x, y), wrapping around the sides of the
    // image.
    let mut patterns = Counter::with_capacity((width * height) as usize);
    for x in 0..width {
        for y in 0..height {
            let mut pixels = Vec::with_capacity((n * m) as usize);
            for dy in 0..m {
                for dx in 0..n {
                    let pixel = image.get_pixel((x + dx) % width, (y + dy) % height);
                    pixels.push(*pixel);
                }
            }
            patterns.insert(Pattern { pixels });
        }
    }

    Patterns::new(patterns, n as usize, m as usize)
}

fn calculate_entropy(mask: &HashSet<usize>, patterns: &Patterns) -> f32 {
    // Entropy is -sum(p_i * log(p_i)), p_i = occurences_of_i / all_occurences
    let frequencies: Vec<f32> = patterns
        .frequencies(mask)
        .into_iter()
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

type Wave = Vec<HashSet<PatternId>>;

fn initialize(
    image: &Image,
    n: u32,
    m: u32,
    width: u32,
    height: u32,
) -> (Patterns, Wave, Vec<f32>) {
    println!("Finding patterns");
    let patterns = find_patterns(image, n, m);

    println!("Creating wave");
    let all_patterns: HashSet<usize> = (0..patterns.len()).collect();
    let wave: Vec<HashSet<usize>> = (0..(width * height))
        .map(|_| HashSet::from(all_patterns.clone()))
        .collect();
    println!("freqs: {:?}", patterns.frequencies(&wave[0]));
    println!("Calculating entropies");
    let entropy: Vec<f32> = wave
        .iter()
        .map(|mask| calculate_entropy(mask, &patterns))
        .collect();

    (patterns, wave, entropy)
}

fn observe(wave: &Wave, entropy: &Vec<f32>, patterns: &Patterns) -> Option<(usize, PatternId)> {
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
        let mut rng = thread_rng();
        let pattern = dist.sample(&mut rng);
        Some((cell, pattern))
    } else {
        None
    }
}

enum PropagationResult {
    Success(HashMap<usize, HashSet<PatternId>>),
    Contradiction,
}

fn propagate(
    collapse: Collapse,
    wave: &Wave,
    patterns: &Patterns,
    width: u32,
    height: u32,
) -> PropagationResult {
    let (cell, pattern) = collapse;
    let mut wave = wave.clone();
    // Starting with the collapsed cell, propagating changes
    let mut queue: Vec<(usize, HashSet<PatternId>)> = vec![(cell, [pattern].into())];
    // Keep track of changed cells to know which to recalculate entropy for
    let mut changes: HashMap<usize, HashSet<PatternId>> = HashMap::new();

    while !queue.is_empty() {
        // `cell` is the cell we're propagating the collapse through, and `post_collapse` is
        // the set of patterns this cell could be as a result of the collapse.
        let (cell, post_collapse) = queue.pop().unwrap();
        assert!(!post_collapse.is_empty());

        // Only propagate from this cell if this cell has fewer options
        let prev = wave[cell].len();
        wave[cell] = wave[cell].intersection(&post_collapse).copied().collect();

        // If this cell has no possibilities, the algorithm has run into a contradiction
        if wave[cell].is_empty() {
            println!("\n contradiction with {cell}");
            return PropagationResult::Contradiction;
        }

        if prev != wave[cell].len() {
            // Insert (overwrite, even) these changes
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
                        .map(|&i| patterns.compatibles(i, &direction))
                        .reduce(|a, b| {
                            let mut a = a;
                            a.extend(b);
                            a
                        })
                        .expect(&format!(
                            "Unnexpected lack of possibilities for {:?} of {:?}",
                            direction, wave[cell]
                        ));
                    queue.push((neighbor.unwrap(), compatibles));
                });
        }
    }

    PropagationResult::Success(changes)
}

type Collapse = (usize, PatternId);
type PatternSet = HashSet<PatternId>;
type CollapseHistory = Vec<(Collapse, HashMap<usize, PatternSet>)>;

pub fn wave_function_collapse(image: &Image, n: u32, m: u32, width: u32, height: u32) {
    // Number of digits to display the number of cells
    let rem_width = format!("{}", width * height).len();

    let (mut patterns, mut wave, mut entropy) = initialize(image, n, m, width, height);
    let mut restarts = 0;
    // Observation stack
    let mut history: CollapseHistory = Vec::new();
    // Overserve-propagate-update loop
    let mut iter_num: usize = 0;
    loop {
        iter_num += 1;
        // Observe and collapse a new cell or break if there are no unobserved cells
        let collapse: Collapse = match observe(&wave, &entropy, &patterns) {
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
        match propagate(collapse, &wave, &patterns, width, height) {
            // Update entropies
            PropagationResult::Success(changes) => {
                history.push((collapse, changes.clone()));
                changes.into_iter().for_each(|(cell, mask)| {
                    wave[cell] = mask;
                    entropy[cell] = calculate_entropy(&wave[cell], &patterns)
                });
            }
            // Restart
            PropagationResult::Contradiction => {
                eprintln!("\nRestarting...");
                restarts += 1;
                (patterns, wave, entropy) = initialize(image, n, m, width, height);
                history.drain(1..);
                iter_num = 0;
                // Observe again
                continue;
            }
        }
    }
    println!(
        "\rFinished after {} restarts and {} iterations",
        restarts,
        iter_num - 1
    );

    // Animate
    animate(&history, &patterns, width, height, 10);
    let mut image = draw(&wave, &patterns, width as usize, height as usize, 10);
    image.save("data/output.png");
}

fn animate(history: &CollapseHistory, patterns: &Patterns, width: u32, height: u32, scale: u32) {
    let file = File::create("data/output.gif").unwrap();
    let mut gif = GifEncoder::new_with_speed(file, 10);
    gif.set_repeat(image::codecs::gif::Repeat::Infinite)
        .unwrap();
    let mut encoder = Encoder::new((scale * width, scale * height)).unwrap();
    let format_width = format!("{}", history.len()).len();

    // The first frame is the average of all the patterns
    let all_patterns: HashSet<usize> = (0..patterns.len()).collect();
    let background = get_average_color(&all_patterns, &patterns);
    let mut frame = background.repeat((scale * width * scale * height) as usize);

    let duration = 5;
    let delay = std::cmp::max(20, (duration * 1000) / history.len() as i32);
    let skip_by = std::cmp::max(1, history.len() as i32 / (duration * (1000 / delay))) as usize;
    println!("{} {} {}", delay, 1000.0 / delay as f32, skip_by);
    // Create the next frame by copying the last and only modifying the changed cells
    history.iter().enumerate().for_each(|(i, (_, diffs))| {
        eprint!(
            "\rAnimating {:format_width$}/{:format_width$}",
            i + 1,
            history.len()
        );
        if i % skip_by == 0 {
            eprint!("rending {i}");
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
            put_pixel_scaled(&mut frame, scale, width, x, y, color)
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

fn get_average_color(mask: &HashSet<usize>, patterns: &Patterns) -> [u8; 4] {
    let frequencies: Vec<f32> = patterns
        .frequencies(mask)
        .into_iter()
        .map(|f| f as f32)
        .collect();
    let total: f32 = frequencies.iter().sum();
    let rgb = Rgb(frequencies
        .iter()
        .enumerate()
        .map(|(pattern, freq)| {
            patterns
                .top_left_color(pattern)
                .0
                .map(|c| c as f32 * freq / total)
        })
        .reduce(|a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
        .unwrap()
        .map(|c| c as u8))
    .0;
    [rgb[0], rgb[1], rgb[2], 255]
}

fn draw(
    wave: &Vec<HashSet<usize>>,
    patterns: &Patterns,
    width: usize,
    height: usize,
    scale: usize,
) -> image::RgbaImage {
    let mut image = image::RgbaImage::new((width * scale) as u32, (height * scale) as u32);
    (0..width).for_each(|x| {
        (0..height).for_each(|y| {
            let possibilities = &wave[(x + width * y) as usize];
            let color = get_average_color(possibilities, &patterns);
            (0..scale).for_each(|dx| {
                (0..scale).for_each(|dy| {
                    image.put_pixel(
                        (scale * x + dx) as u32,
                        (scale * y + dy) as u32,
                        Rgba(color),
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

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
