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

        // TODO: I'm dubious on the merit of these enumerates
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

    fn frequencies(&self, mask: &Vec<bool>) -> Vec<u32> {
        self.info
            .values()
            .enumerate()
            .map(|(i, &(c, _))| if mask[i] { c } else { 0 })
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
    //let image = ImageReader::open("data/SimpleKnot.png")
    //.unwrap()
    //.decode()
    //.unwrap()
    //.to_rgb8();
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
            println!("({x}, {y}) is {:?}", image.get_pixel(x, y));
            patterns.insert(Pattern { pixels });
        }
    }

    Patterns::new(patterns, n as usize, m as usize)
}

fn calculate_entropy(mask: &Vec<bool>, patterns: &Patterns) -> f32 {
    let frequencies = patterns.frequencies(mask);
    let total: u32 = frequencies.iter().sum();
    -mask
        .iter()
        .enumerate()
        .filter(|&(_, possible)| *possible)
        .map(|(i, _)| {
            let p = frequencies[i] as f32 / total as f32;
            p * p.log2()
        })
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

    println!("Finding patterns");
    let patterns = find_patterns(image, n, m);
    (0..patterns.len()).for_each(|i| {
        println!("{i}:");
        DIRECTIONS.iter().for_each(|direction| {
            println!(
                "  {:?}: {:?}",
                direction,
                patterns.compatibles(i, direction)
            );
        });
    });

    println!("Creating wave");
    let mut wave = vec![vec![true; patterns.len()]; (width * height) as usize];
    println!("Calculating entropies");
    let mut entropy: Vec<f32> = wave
        .iter()
        .map(|mask| calculate_entropy(mask, &patterns))
        .collect();

    // maybe just use reduce instead of min
    let mut iter_num = 0;
    while let Some((min_entropy_index, _)) = entropy
        .iter()
        .enumerate()
        .filter(|&(_, e)| *e != 0.0)
        .reduce(|(i, min), (j, e)| if e < min { (j, e) } else { (i, min) })
    {
        eprint!(
            "\r{iter_num}, {} remaining, {} entropy",
            entropy.iter().filter(|&e| *e != 0.0).count(),
            entropy.iter().sum::<f32>()
        );
        let mut rng = thread_rng();
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
            // `pre_collapse` is the set of patterns this cell could be pre-collapse
            let pre_collapse: HashSet<usize> = wave[cell]
                .iter()
                .enumerate()
                .filter(|(_, &possible)| possible)
                .map(|(i, _)| i)
                .collect();
            // `difference` is the patterns that are no longer possible
            let mut difference = pre_collapse.difference(&post_collapse).peekable();
            if difference.peek().is_some() {
                difference.for_each(|&i| wave[cell][i] = false);
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
                            .enumerate()
                            .filter(|(_, &possible)| possible)
                            .map(|(i, _)| patterns.compatibles(i, &direction))
                            .reduce(|a, b| {
                                let mut a = a;
                                a.extend(b);
                                a
                            })
                            .expect("Unnexpected lack of possibilities");
                        queue.push((neighbor.unwrap(), compatibles));
                    });
            }
        }

        // UPDATE ENTROPIES of changed cells
        changed
            .iter()
            .for_each(|&cell| entropy[cell] = calculate_entropy(&wave[cell], &patterns));

        // Draw this frame
        let image = draw(&wave, &patterns, width as usize, height as usize, 10);
        let mut frame = image::Frame::new(image);
        encoder.encode_frame(frame);
        iter_num += 1;
    }

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

fn get_average_color(mask: &Vec<bool>, patterns: &Patterns) -> Rgb<u8> {
    let frequencies: Vec<f32> = patterns
        .frequencies(mask)
        .into_iter()
        .map(|f| f as f32)
        .collect();
    let total: f32 = frequencies.iter().sum();
    Rgb(frequencies
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
}

fn draw(
    wave: &Vec<Vec<bool>>,
    patterns: &Patterns,
    width: usize,
    height: usize,
    scale: usize,
) -> image::RgbaImage {
    let mut image = image::RgbaImage::new((width * scale) as u32, (height * scale) as u32);
    (0..width).for_each(|x| {
        (0..height).for_each(|y| {
            let possibilities: Vec<usize> = wave[(x + width * y) as usize]
                .iter()
                .enumerate()
                .filter(|(_, &possible)| possible)
                .map(|(i, _)| i)
                .collect();
            let color = if possibilities.len() == 1 {
                let pattern = possibilities[0];
                patterns.top_left_color(pattern)
            } else {
                get_average_color(&wave[(x + width * y) as usize], &patterns)
            }
            .0;
            let color = image::Rgba([color[0], color[1], color[2], 255]);
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
