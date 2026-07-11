use burn::{
    module::{AutodiffModule, Module},
    record::CompactRecorder,
};
use ganshot::{
    backend::{MyAutodiffBackend, select_device},
    data::{commons::Geometry, stars::Star},
    models::generator::{GeneratorConfig, sample_z_fixed},
    training::recorder::plot_outlines,
};
use rand::{SeedableRng, rngs::StdRng};

fn main() {
    let device = select_device();

    // Matches how the generator was saved during training (CompactRecorder,
    // path "./checkpoints/generator-{epochs}"; the extension is added by the recorder).
    let model_path = "./checkpoints/generator-250";
    let recorder = CompactRecorder::new();

    let z_dim = 8;
    let nb_hidden = 100;
    // Must match the trained model: real_dim = Star::N * 2 (x, y per outline point).
    let real_dim = Star::N * 2;
    let config = GeneratorConfig::new(z_dim, nb_hidden, real_dim);

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
    let fixed_steps = 101;
    let z_valid = sample_z_fixed(z_dim, fixed_dim, fixed_steps, &mut rng, &device);

    let fake_valid = generator_valid.forward(z_valid);
    let [_rows, cols] = fake_valid.dims();
    let flat: Vec<f32> = fake_valid.into_data().to_vec().unwrap();
    let outlines: Vec<Vec<f64>> = flat
        .chunks_exact(cols)
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();

    // Each sweep step becomes its own slider position holding a single outline,
    // so scrubbing the slider shows how the star morphs along latent dim `fixed_dim`.
    let steps: Vec<usize> = (0..fixed_steps).collect();
    let grouped: Vec<Vec<Vec<f64>>> = outlines.into_iter().map(|outline| vec![outline]).collect();
    plot_outlines(steps, grouped).write_html(format!(
        "./checkpoints/generator_outlines_dim{fixed_dim}.html"
    ));
}
