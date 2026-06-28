use burn::{Tensor, tensor::Shape};
use ganshot::{
    backend::{MyAutodiffBackend, select_device},
    models::discriminator::DiscriminatorConfig,
};

#[test]
fn discriminator_forward_pass() {
    let device = select_device();

    let (nb_hidden, batch_size, real_dim) = (100, 64, 2);
    let config = DiscriminatorConfig::new(real_dim, nb_hidden);
    let discriminator = config.init::<MyAutodiffBackend>(&device);

    let x = Tensor::empty([batch_size, 2], &device);
    let output = discriminator.forward(x);
    assert_eq!(output.shape(), Shape::new([batch_size, 1]));
}
