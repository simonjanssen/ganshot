use rand::{RngExt, rngs::ThreadRng};
use rand_distr::{Distribution, Normal};

use crate::gan::data::Point2;

const STD_DEV: f64 = 0.1;

// --- Core Triplet Types and Implementations

pub struct Gaussian2 {
    nx: Normal<f64>,
    ny: Normal<f64>,
}

impl Gaussian2 {
    pub fn builder(c: Point2) -> Gaussian2Builder {
        Gaussian2Builder { c }
    }

    pub fn sample(&self, rng: &mut ThreadRng) -> Point2 {
        let x = self.nx.sample(rng);
        let y = self.ny.sample(rng);
        Point2 { x, y }
    }
}

pub struct Gaussian2Builder {
    c: Point2,
}

impl Gaussian2Builder {
    pub fn build(self) -> Gaussian2 {
        let Point2 { x, y } = self.c;
        let nx = Normal::new(x, STD_DEV).unwrap();
        let ny = Normal::new(y, STD_DEV).unwrap();
        Gaussian2 { nx, ny }
    }
}

pub struct TripleGaussian2 {
    g1: Gaussian2,
    g2: Gaussian2,
    g3: Gaussian2,
}

impl TripleGaussian2 {
    pub fn builder(c1: Point2, c2: Point2, c3: Point2) -> TripleGaussian2Builder {
        TripleGaussian2Builder { c1, c2, c3 }
    }

    pub fn sample(&self, rng: &mut ThreadRng) -> Point2 {
        let choice = rng.random_range(0..3);
        match choice {
            0 => self.g1.sample(rng),
            1 => self.g2.sample(rng),
            _ => self.g3.sample(rng),
        }
    }
}

pub struct TripleGaussian2Builder {
    c1: Point2,
    c2: Point2,
    c3: Point2,
}

impl TripleGaussian2Builder {
    pub fn build(self) -> TripleGaussian2 {
        let g1 = Gaussian2::builder(self.c1).build();
        let g2 = Gaussian2::builder(self.c2).build();
        let g3 = Gaussian2::builder(self.c3).build();
        TripleGaussian2 { g1, g2, g3 }
    }
}
