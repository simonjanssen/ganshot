use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Distribution, Float, Shape, TensorCreationOptions, TensorData, backend::Backend},
};
use rand::Rng;

#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    linear1: Linear<B>,
    relu1: Relu,
    linear2: Linear<B>,
}

impl<B: Backend> Generator<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.relu1.forward(x);
        self.linear2.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct GeneratorConfig {
    z_dim: usize,
    nb_hidden: usize,
    output_dim: usize,
}

impl GeneratorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Generator<B> {
        Generator {
            linear1: LinearConfig::new(self.z_dim, self.nb_hidden).init(device),
            relu1: Relu::new(),
            linear2: LinearConfig::new(self.nb_hidden, self.output_dim).init(device),
        }
    }
}

/// Sample a random generator input tensor with Normal(mean=0.0, std=1.0)
pub fn sample_z<S, R, B, const D: usize>(
    shape: S,
    rng: &mut R,
    options: impl Into<TensorCreationOptions<B>>,
) -> Tensor<B, D, Float>
where
    S: Into<Shape>,
    R: Rng,
    B: Backend,
{
    let tensor_data = TensorData::random::<f32, _, _>(shape, Distribution::Normal(0.0, 1.0), rng);
    Tensor::from_data(tensor_data, options)
}
