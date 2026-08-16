use burn::{
    module::{AutodiffModule, Module},
    record::CompactRecorder,
};
use ganshot::{
    backend::{MyAutodiffBackend, select_device},
    data::{
        commons::{Geometry, plot_2x3},
        stars::Star,
    },
    models::gan::generator::{GeneratorConfig, sample_z_fixed},
};
use rand::{SeedableRng, rngs::StdRng};

fn main() {
    let device = select_device();

    // Matches how the generator was saved during training (CompactRecorder,
    // path "./checkpoints/generator-{epochs}"; the extension is added by the recorder).
    let model_path = "./checkpoints/generator-250";
    let recorder = CompactRecorder::new();

    let z_dim = 8;
    let h1_dim = 32;
    let h2_dim = 128;
    // Must match the trained model: real_dim = Star::N * 2 (x, y per outline point).
    let real_dim = Star::N * 2;
    let config = GeneratorConfig::new(z_dim, h1_dim, h2_dim, real_dim);

    let mut generator = config.init::<MyAutodiffBackend>(&device);
    generator = generator
        .load_file(model_path, &recorder, &device)
        .expect("Failed to load trained generator!");
    let generator_valid = generator.valid();

    // Seed the RNG so the shared base latent vector (and thus the sweep) is reproducible.
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    // Which latent dimension to sweep, from `--dim <N>` (defaults to 0).
    let fixed_dim = std::env::args()
        .skip_while(|a| a != "--dim")
        .nth(1)
        .map(|v| {
            v.parse::<usize>()
                .expect("--dim must be a non-negative integer")
        })
        .unwrap_or(0);
    assert!(
        fixed_dim < z_dim,
        "--dim ({fixed_dim}) must be < z_dim ({z_dim})"
    );
    let fixed_steps = 6;
    let z_valid = sample_z_fixed(z_dim, fixed_dim, fixed_steps, &mut rng, &device);

    let fake_valid = generator_valid.forward(z_valid);
    let [_rows, cols] = fake_valid.dims();
    let flat: Vec<f32> = fake_valid.into_data().to_vec().unwrap();
    let outlines: Vec<Vec<f64>> = flat
        .chunks_exact(cols)
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();

    let geometries: Vec<_> = outlines.into_iter().map(|v| Star::from_vec(v)).collect();
    let geometries: [Star; 6] = geometries.try_into().expect("Not 6 geometries!");
    let plot = plot_2x3(&geometries);
    plot.write_html("tmp/generate_stars.html");
}
