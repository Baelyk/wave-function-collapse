use crate::pattern;
use crate::pixel::Pixel;
use bit_set::BitSet;
use image::RgbaImage;
use log::debug;
use pattern::PatternId;
use pattern::PatternSet;
use pattern::Patterns;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::ImageData;
use web_sys::OffscreenCanvasRenderingContext2d;

type Collapse = (usize, PatternId);

enum TickResult {
    Done,
    Success(BitSet),
    Contradiction(Collapse),
}

#[wasm_bindgen]
pub enum CollapseState {
    Done,
    InProgress,
    Contradiction,
}

#[wasm_bindgen]
pub struct WaveFunctionCollapse {
    patterns: Patterns,
    width: u32,
    height: u32,
    wave: Vec<PatternSet>,
    entropy: Vec<f32>,
    rng: SmallRng,
    colors: Vec<u8>,
    changes: BitSet,
}

#[wasm_bindgen]
impl WaveFunctionCollapse {
    /// Create a new WaveFunctionCollapse from an image buffer
    #[wasm_bindgen(constructor)]
    pub fn new(
        source_image_buffer: &[u8],
        n: u32,
        m: u32,
        width: u32,
        height: u32,
        sym: bool,
        seed: u32,
    ) -> Self {
        let source_image: RgbaImage = image::load_from_memory(source_image_buffer)
            .expect("Image should load")
            .into_rgba8();

        // Find the patterns from the source image
        let patterns = Patterns::from_image(&source_image, n, m, sym);
        eprintln!("Found {} patterns", patterns.len());

        // Seeded RNG for determinism when picking which pattern to collapse to
        let rng = rand::rngs::SmallRng::seed_from_u64(seed.into());
        let num_cells = (width * height)
            .try_into()
            .expect("u32 should fit in usize");
        let (wave, entropy) = Self::initialize(&patterns, num_cells);

        let initial_color = get_average_color(&wave[0], &patterns);
        let colors = initial_color
            .into_iter()
            .cycle()
            .take(4 * num_cells)
            .collect();

        let changes = BitSet::with_capacity(num_cells);

        WaveFunctionCollapse {
            patterns,
            width,
            height,
            wave,
            entropy,
            rng,
            colors,
            changes,
        }
    }

    /// Create initial `wave` and `entropy` Vecs
    fn initialize(patterns: &Patterns, num_cells: usize) -> (Vec<PatternSet>, Vec<f32>) {
        // Initialize the wave so each cell is in a superposition of all patterns
        let all_patterns = patterns.set_all();
        let mut wave: Vec<PatternSet> = Vec::with_capacity(num_cells as usize);
        wave.resize(num_cells as usize, all_patterns.clone());

        // Each cell has the same entropy at the start
        let entropy = calculate_entropy(&all_patterns, patterns);
        let entropy = [entropy].repeat(wave.len());

        (wave, entropy)
    }

    //#[wasm_bindgen]
    //pub async fn collapse(&mut self) -> bool {
    //loop {
    //debug!("loop");
    //match self.tick() {
    //WaveFunctionCollapseResult::Success => continue,
    //WaveFunctionCollapseResult::Done => return true,
    //WaveFunctionCollapseResult::Contradiction => return false,
    //}
    //}
    //}
    //

    pub fn collapse_for_ticks(&mut self, ticks: usize) -> CollapseState {
        let mut tick_count = 0;
        while tick_count < ticks {
            tick_count += 1;
            match self.tick() {
                TickResult::Success(_) => continue,
                TickResult::Done => return CollapseState::Done,
                TickResult::Contradiction(_) => CollapseState::Contradiction,
            };
        }
        CollapseState::InProgress
    }

    #[wasm_bindgen]
    pub async fn collapse_and_draw(&mut self, ctx: &OffscreenCanvasRenderingContext2d) -> bool {
        debug!("Initial draw");
        self.draw(ctx);
        loop {
            debug!("loop and draw");
            let result = self.tick();
            match result {
                TickResult::Success(_) => continue,
                TickResult::Done => break,
                TickResult::Contradiction(_) => break,
            }
        }
        debug!("{:?}", ctx);
        true
    }

    /// Run a single observe-propagate-update cycle
    fn tick(&mut self) -> TickResult {
        // Observe and collapse a new cell or break if there are no unobserved cells
        let collapse: Collapse = match self.observe() {
            Some((c, p)) => (c, p),
            None => return TickResult::Done,
        };

        // Propagate this collapse
        let Ok(changes) = self.propagate(collapse) else {
            return TickResult::Contradiction(collapse);
        };

        changes.iter().for_each(|cell| {
            // Update entropies
            self.entropy[cell] = calculate_entropy(&self.wave[cell], &self.patterns);
            // Update colors
            self.update_cell_color(cell);
            // Mark as changed
            self.changes.insert(cell);
        });

        TickResult::Success(changes)
    }

    fn observe(&mut self) -> Option<Collapse> {
        // Get the cell with the least nonzero entropy
        if let Some((cell, _)) = self
            .entropy
            .iter()
            .enumerate()
            .filter(|&(_, e)| *e != 0.0)
            .reduce(|(i, min), (j, e)| if e < min { (j, e) } else { (i, min) })
        {
            // Randomly pick a pattern from the cell's possible patterns, weighted by times that
            // pattern occurred in the seed
            let dist = WeightedIndex::new(self.patterns.frequencies(&self.wave[cell])).unwrap();
            let pattern = dist.sample(&mut self.rng);
            Some((cell, pattern))
        } else {
            // All cells are collapsed
            None
        }
    }

    fn propagate(&mut self, collapse: Collapse) -> Result<BitSet, ()> {
        let (cell, pattern) = collapse;
        // Starting with the collapsed cell, propagating changes
        let mut queue: Vec<(usize, PatternSet)> = vec![(cell, self.patterns.set_with(pattern))];
        // Keep track of changed cells to know which to recalculate entropy for
        let mut changed = BitSet::with_capacity(self.patterns.len());

        while !queue.is_empty() {
            // `cell` is the cell we're propagating the collapse through, and `post_collapse` is
            // the set of patterns this cell could be as a result of the collapse.
            let (cell, post_collapse) = queue.pop().unwrap();
            debug_assert!(!post_collapse.is_empty());

            // Only propagate from this cell if it will have fewer options
            let prev = self.wave[cell].len();
            self.wave[cell].intersect_with(&post_collapse);

            // If this cell has no possibilities, the algorithm has run into a contradiction
            if self.wave[cell].is_empty() {
                return Err(());
            }

            if prev != self.wave[cell].len() {
                changed.insert(cell);
                // Propagate changes to neighbors
                get_neighbors(cell, self.width, self.height)
                    .into_iter()
                    .zip(crate::pattern::DIRECTIONS.into_iter())
                    .filter(|(neighbor, _)| neighbor.is_some())
                    .for_each(|(neighbor, direction)| {
                        // Compatibles is the set of all patterns that can be to `direction` of any
                        // of this cell's possible patterns.
                        let mut compatibles = self.wave[cell]
                            .iter()
                            .map(|i| self.patterns.compatibles(i, &direction));
                        let mut compatible = compatibles.next().unwrap().clone();
                        compatibles.for_each(|set| compatible.union_with(set));
                        queue.push((neighbor.unwrap(), compatible));
                    });
            }
        }

        Ok(changed)
    }

    fn update_cell_color(&mut self, cell: usize) {
        let color = get_average_color(&self.wave[cell], &self.patterns);
        self.colors[4 * cell] = color[0];
        self.colors[4 * cell + 1] = color[1];
        self.colors[4 * cell + 2] = color[2];
        self.colors[4 * cell + 3] = color[3];
    }

    fn draw_cell(&self, ctx: &OffscreenCanvasRenderingContext2d, cell: usize) {
        let color = get_average_color(&self.wave[cell], &self.patterns);
        ctx.set_fill_style(&format!("rgb({}, {}, {})", color[0], color[1], color[2]).into());
        let x: f64 = f64::try_from(cell as u32 % self.width).unwrap();
        let y: f64 = f64::try_from(cell as u32 / self.width).unwrap();
        //debug!("{} {:?} {:?} ({}, {})", cell, color, ctx.fill_style(), x, y);
        ctx.fill_rect(x, y, 1.0, 1.0);
    }

    #[wasm_bindgen]
    pub fn draw(&self, ctx: &OffscreenCanvasRenderingContext2d) {
        // If the canvas is the same size as the image, use `put_image_data`
        let canvas = ctx.canvas();
        if canvas.width() == self.width && canvas.height() == self.height {
            let image = ImageData::new_with_u8_clamped_array_and_sh(
                Clamped(&self.colors),
                self.width,
                self.height,
            )
            .expect("unable to create image");
            ctx.put_image_data(&image, 0.0, 0.0)
                .expect("unable to draw");
            return;
        }

        // Otherwise, draw each cell
        (0..self.width * self.height).for_each(|cell| self.draw_cell(ctx, cell as usize));
    }

    pub fn draw_changes(&mut self, ctx: &OffscreenCanvasRenderingContext2d) {
        self.changes
            .iter()
            .for_each(|cell| self.draw_cell(ctx, cell as usize));
        self.changes.clear();
    }

    #[wasm_bindgen(getter)]
    pub fn colors(&mut self) -> *const u8 {
        self.colors.as_ptr()
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
