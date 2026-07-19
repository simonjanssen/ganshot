use ganshot::{
    data::{commons::Geometry, stars::RandomStars},
    training::recorder::plot_outlines,
};

fn main() {
    // Sample a handful of stars and flatten each outline into (x1, y1, x2, y2, ...).
    let stars = RandomStars::new();
    let outlines: Vec<Vec<f64>> = stars
        .take(5)
        .map(|star| {
            star.to_outline()
                .into_iter()
                .flat_map(|(x, y)| [x, y])
                .collect()
        })
        .collect();

    // Plot as a single epoch (epoch label 0).
    plot_outlines(vec![0], vec![outlines]).write_html("./checkpoints/generator_outlines.html");
}
