use ganshot::{
    data::{commons::Geometry, stars::RandomStars},
    training::recorder::plot_outlines,
};
use rand_distr::Distribution;

fn main() {
    let mut rng = rand::rng();

    // Sample a handful of stars and flatten each outline into (x1, y1, x2, y2, ...).
    let outlines: Vec<Vec<f64>> = (&RandomStars)
        .sample_iter(&mut rng)
        .take(5)
        .map(|star| {
            star.to_outline()
                .into_iter()
                .flat_map(|(x, y)| [x, y])
                .collect()
        })
        .collect();

    // Plot as a single epoch (epoch label 0).
    plot_outlines(vec![0], vec![outlines]);
}
