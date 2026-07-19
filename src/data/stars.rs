use rand::{RngExt, rngs::ThreadRng};
use rand_distr::Distribution;

// --- Geometries ---

/// make it impossible to init Star except for ::new()
mod geometries {
    use crate::data::commons::{Coord2, Geometry};

    pub struct Star {
        center: Coord2,
        size: f64,
        rotation: f64,
    }

    impl Geometry for Star {
        const N: usize = 10;
        type Outline = [Coord2; Self::N];

        /// Return the outline coordinates with wich one could draw the star's outer bounds
        /// This is a sequence of 10 coordinates (5 star tips, 5 star valleys) returned in counter-clockwise order
        ///
        /// Even indices are the outer tips (on the circle of radius `size`), odd indices are the inner
        /// valleys (on the circle of radius `size * sin(18°)/sin(54°)`, the classic pentagram ratio).
        /// With `rotation == 0.0` the first tip points straight up along the +y axis; `rotation` is
        /// measured in turns (a full counter-clockwise revolution is `1.0`).
        fn to_outline(&self) -> Self::Outline {
            // Classic pentagram inner/outer radius ratio: sin(18°) / sin(54°).
            let inner_ratio = (18.0_f64.to_radians()).sin() / (54.0_f64.to_radians()).sin();
            let inner_radius = self.size * inner_ratio;

            // First tip points up (+y) at rotation 0; `rotation` turns are added counter-clockwise.
            let base = std::f64::consts::FRAC_PI_2 + self.rotation * std::f64::consts::TAU;
            let step = std::f64::consts::TAU / 5.0;
            let half_step = step / 2.0;

            let mut outline = [(0.0, 0.0); 10];
            for i in 0..5 {
                let tip_angle = base + i as f64 * step;
                let valley_angle = tip_angle + half_step;
                outline[2 * i] = (
                    self.center.0 + self.size * tip_angle.cos(),
                    self.center.1 + self.size * tip_angle.sin(),
                );
                outline[2 * i + 1] = (
                    self.center.0 + inner_radius * valley_angle.cos(),
                    self.center.1 + inner_radius * valley_angle.sin(),
                );
            }
            outline
        }
    }

    impl Star {
        pub fn new(center: Coord2, size: f64, rotation: f64) -> Self {
            assert!(
                (0.0..=1.0).contains(&rotation),
                "rotation is out of range [0, 1]!"
            );
            assert!(size > 0.0, "size must be greater than 0!");
            Self {
                center,
                size,
                rotation,
            }
        }
    }
}

pub use geometries::Star;

// --- Distributions ---
pub struct RandomStars {
    rng: ThreadRng,
}

impl RandomStars {
    pub fn new() -> Self {
        Self { rng: rand::rng() }
    }
}

impl Iterator for RandomStars {
    type Item = Star;
    fn next(&mut self) -> Option<Self::Item> {
        Some(Star::new(
            (
                self.rng.random_range(-5.0..5.0),
                self.rng.random_range(-5.0..5.0),
            ),
            self.rng.random_range(1.0..5.0),
            self.rng.random_range(0.0..1.0),
        ))
    }
}

impl Distribution<Star> for RandomStars {
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Star {
        Star::new(
            (rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)),
            rng.random_range(1.0..5.0),
            rng.random_range(0.0..1.0),
        )
    }
}
