use burn::tensor::Shape;
use ganshot::{
    backend::{MyAutodiffBackend, select_device},
    models::generator::{GeneratorConfig, sample_z, sample_z_fixed},
};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn generator_forward_pass() {
    let device = select_device();

    let (z_dim, nb_hidden, batch_size, real_dim) = (8, 100, 64, 2);
    let config = GeneratorConfig::new(z_dim, nb_hidden, real_dim);
    let generator = config.init::<MyAutodiffBackend>(&device);

    let mut rng = rand::rng();
    let z = sample_z([batch_size, z_dim], &mut rng, &device);

    let output = generator.forward(z);
    assert_eq!(output.shape(), Shape::new([batch_size, 2]));
}

#[test]
fn sample_z_fixed_sweeps_one_dim() {
    let device = select_device();

    let (z_dim, fixed_dim, fixed_steps) = (8, 3, 5);
    let mut rng = StdRng::seed_from_u64(42);
    let z =
        sample_z_fixed::<_, MyAutodiffBackend>(z_dim, fixed_dim, fixed_steps, &mut rng, &device);

    assert_eq!(z.shape(), Shape::new([fixed_steps, z_dim]));

    let data = z.into_data();
    let values = data.to_vec::<f32>().unwrap();

    for step in 0..fixed_steps {
        let row = &values[step * z_dim..(step + 1) * z_dim];

        // The fixed dimension is swept linearly from -1.0 to 1.0.
        let expected = -1.0 + step as f32 * (2.0 / (fixed_steps - 1) as f32);
        assert!((row[fixed_dim] - expected).abs() < 1e-6);

        // Every other dimension stays constant across all rows.
        for dim in 0..z_dim {
            if dim != fixed_dim {
                assert!((row[dim] - values[dim]).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn sample_z_fixed_single_step() {
    let device = select_device();

    let mut rng = StdRng::seed_from_u64(7);
    let z = sample_z_fixed::<_, MyAutodiffBackend>(8, 0, 1, &mut rng, &device);

    assert_eq!(z.shape(), Shape::new([1, 8]));
    let values = z.into_data().to_vec::<f32>().unwrap();
    // With a single step the swept dimension sits at the range start.
    assert!((values[0] - (-1.0)).abs() < 1e-6);
}
