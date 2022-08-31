use image::io::Reader as ImageReader;
use image::Rgb;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::hash::{Hash, Hasher};

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

    pub fn probability_of(&self, pattern: usize) -> f32 {
        self.info.get(&pattern).unwrap().0 as f32 / self.len() as f32
    }

    fn get_pattern_occurances(&self, pattern: usize) -> u32 {
        self.info.get(&pattern).unwrap().0
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

pub fn wave_function_collapse(image: &Image, n: u32, m: u32, width: u32, height: u32) {
    // Output gif setup
    let mut file = File::create("data/output.gif").unwrap();
    let mut encoder = image::codecs::gif::GifEncoder::new(file);
    encoder
        .set_repeat(image::codecs::gif::Repeat::Infinite)
        .unwrap();

    println!("Finding patterns");
    let patterns = find_patterns(image, n, m);

    println!("Creating wave");
    let all_patterns: HashSet<usize> = (0..patterns.len()).collect();
    let mut wave: Vec<HashSet<usize>> = (0..(width * height))
        .map(|_| HashSet::from(all_patterns.clone()))
        .collect();
    println!("freqs: {:?}", patterns.frequencies(&wave[0]));
    println!("Calculating entropies");
    let mut entropy: Vec<f32> = wave
        .iter()
        .map(|mask| calculate_entropy(mask, &patterns))
        .collect();

    // maybe just use reduce instead of min
    let mut iter_num = 0;
    let mut now = std::time::Instant::now();
    let mut spent = vec![];
    let mut rng = thread_rng();
    while let Some((min_entropy_index, _)) = entropy
        .iter()
        .enumerate()
        .filter(|&(_, e)| *e != 0.0)
        .reduce(|(i, min), (j, e)| if e < min { (j, e) } else { (i, min) })
    {
        eprint!(
            "\r{iter_num}, {} remaining, {} entropy, {}us",
            entropy.iter().filter(|&e| *e != 0.0).count(),
            entropy.iter().sum::<f32>(),
            now.elapsed().as_micros()
        );
        spent.push(now.elapsed());
        now = std::time::Instant::now();
        // COLLAPSE this cell into a random pattern based on the pattern distribution in the
        // source image
        let dist = WeightedIndex::new(patterns.frequencies(&wave[min_entropy_index])).unwrap();
        let pattern = dist.sample(&mut rng);

        // PROPAGATE this collapse through neighbors
        let mut queue: Vec<(usize, HashSet<usize>)> = vec![(min_entropy_index, [pattern].into())];
        let mut changed: HashSet<usize> = HashSet::new();

        while !queue.is_empty() {
            // `cell` is the cell we're propagating the collapse through, and `post_collapse` is
            // the set of patterns this cell could be as a result of the collapse.
            let (cell, post_collapse) = queue.pop().unwrap();
            assert!(!post_collapse.is_empty());

            // Only propagate from this cell if this cell has fewer options
            let prev = wave[cell].clone();
            let count = wave[cell].len();
            wave[cell] = wave[cell].intersection(&post_collapse).copied().collect();
            if count != wave[cell].len() {
                changed.insert(cell);
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
            assert!(!wave[cell].is_empty());
        }

        // UPDATE ENTROPIES of changed cells
        changed
            .iter()
            .for_each(|&cell| entropy[cell] = calculate_entropy(&wave[cell], &patterns));

        // Draw this frame
        if iter_num % 1000 == 0 {
            let image = draw(&wave, &patterns, width as usize, height as usize, 10);
            let frame = image::Frame::new(image);
            encoder.encode_frame(frame).unwrap();
        }
        iter_num += 1;
    }

    let avg = (spent.iter().sum::<std::time::Duration>() / iter_num as u32).as_micros();
    println!("avg loop time: {}us", avg);

    // Save the final image
    let image = draw(&wave, &patterns, width as usize, height as usize, 10);
    image.save("data/output.png");
}

fn put_pixel_scaled(image: &mut Image, scale: u32, x: u32, y: u32, color: Rgb<u8>) {
    for dx in 0..scale {
        for dy in 0..scale {
            image.put_pixel(scale * x + dx, scale * y + dy, color);
        }
    }
}

fn get_average_color(mask: &HashSet<usize>, patterns: &Patterns) -> image::Rgba<u8> {
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
    image::Rgba([rgb[0], rgb[1], rgb[2], 255])
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
                    image.put_pixel((scale * x + dx) as u32, (scale * y + dy) as u32, color);
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

    #[test]
    fn test_neighbors() {
        let width = 50;
        let height = 50;
        let scale = 10;
        let mut image = image::RgbImage::new((width * scale) as u32, (height * scale) as u32);
        let cell = 280;
        let neighbors = get_neighbors(cell, width, height);
        //assert_eq!(neighbors, [Some(2475), Some(24), Some(75), Some(26)]);
        put_pixel_scaled(
            &mut image,
            scale,
            cell as u32 % width,
            cell as u32 / width,
            Rgb([255, 0, 0]),
        );
        neighbors
            .iter()
            .zip([(100, 100), (0, 255), (255, 0), (255, 255)])
            .for_each(|(cell, (g, b))| {
                let cell = cell.unwrap() as u32;
                put_pixel_scaled(
                    &mut image,
                    scale,
                    cell as u32 % width,
                    cell as u32 / width,
                    Rgb([0, g, b]),
                );
            });
        image.save("data/test_neighbors.png").unwrap();
    }
}
