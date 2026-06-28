use rand::RngExt;
use rand_distr::{Distribution, Normal};

use crate::data::commons::Coord2;

const STD_DEV: f64 = 0.1;

// --- Geometries ---
mod inner {
    use crate::data::commons::{Coord2, Geometry};

    pub struct Point {
        center: Coord2,
    }

    impl Geometry for Point {
        const N: usize = 1;
        type Outline = [Coord2; Self::N];

        fn to_outline(&self) -> Self::Outline {
            [self.center]
        }
    }

    impl Point {
        pub fn new(center: Coord2) -> Self {
            Self { center }
        }
    }
}

pub use inner::Point;

// --- Distributions ---
pub struct Gaussian {
    nx: Normal<f64>,
    ny: Normal<f64>,
}

impl Gaussian {
    pub fn new(center: Coord2) -> Self {
        let (x, y) = center;
        let nx = Normal::new(x, STD_DEV).unwrap();
        let ny = Normal::new(y, STD_DEV).unwrap();
        Self { nx, ny }
    }
}

impl Distribution<Point> for Gaussian {
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Point {
        let x = self.nx.sample(rng);
        let y = self.ny.sample(rng);
        Point::new((x, y))
    }
}

pub struct GaussianTriplet {
    g1: Gaussian,
    g2: Gaussian,
    g3: Gaussian,
}

impl GaussianTriplet {
    pub fn new(c1: Coord2, c2: Coord2, c3: Coord2) -> Self {
        let g1 = Gaussian::new(c1);
        let g2 = Gaussian::new(c2);
        let g3: Gaussian = Gaussian::new(c3);
        Self { g1, g2, g3 }
    }
}

impl Distribution<Point> for GaussianTriplet {
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Point {
        let g = match rng.random_range(0..3) {
            0 => &self.g1,
            1 => &self.g2,
            _ => &self.g3,
        };
        g.sample(rng)
    }
}

impl Default for GaussianTriplet {
    fn default() -> Self {
        Self::new((0., -1.), (-0.707, 0.707), (0.707, 0.707))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{MyBackend, select_device},
        data::commons::{Batch, Batcher, Dataset, Geometry},
    };
    use burn::{data::dataloader::DataLoaderBuilder, tensor::Shape};

    #[test]
    fn iterate_batches() {
        let device = select_device();

        let (batch_size, seed, num_workers) = (64, 42, 4);
        let n = 10_000;
        let sampler = GaussianTriplet::default();
        let dataset = Dataset::new(sampler, n);
        let batcher = Batcher::from(&dataset);

        let dataloader = DataLoaderBuilder::new(batcher)
            .batch_size(batch_size)
            .shuffle(seed)
            .num_workers(num_workers)
            .set_device(device)
            .build(dataset);

        let batches: Vec<Batch<MyBackend>> = dataloader.iter().collect();
        assert_eq!(batches.len(), n.div_ceil(batch_size));

        let batch = &batches[0];
        assert_eq!(batch.tensor.shape(), Shape::new([batch_size, Point::N * 2]));
    }
}
