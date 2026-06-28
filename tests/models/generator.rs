use burn::tensor::Shape;
use ganshot::{
    backend::{MyAutodiffBackend, select_device},
    models::generator::{GeneratorConfig, sample_z},
};

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
