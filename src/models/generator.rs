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

/// Sample a generator input where every row shares the same random base latent
/// vector, except `fixed_dim`, which is swept linearly from -1.0 to 1.0 across
/// the `fixed_steps` rows. Useful for probing what a single latent dimension
/// controls. Pass a seeded `rng` to reproduce the base vector.
///
/// Returns a `[fixed_steps, z_dim]` tensor.
pub fn sample_z_fixed<R, B>(
    z_dim: usize,
    fixed_dim: usize,
    fixed_steps: usize,
    rng: &mut R,
    options: impl Into<TensorCreationOptions<B>>,
) -> Tensor<B, 2>
where
    R: Rng,
    B: Backend,
{
    assert!(
        fixed_dim < z_dim,
        "fixed_dim ({fixed_dim}) must be < z_dim ({z_dim})"
    );

    // Sample the shared base latent vector once (batch size 1).
    let base = TensorData::random::<f32, _, _>([z_dim], Distribution::Normal(0.0, 1.0), rng)
        .to_vec::<f32>()
        .expect("base latent vector should be f32");

    // Linear step across [-1.0, 1.0]; 0.0 when there is a single step.
    let step = if fixed_steps > 1 {
        2.0 / (fixed_steps - 1) as f32
    } else {
        0.0
    };

    // Replicate the base row for each step, overwriting `fixed_dim` with the swept value.
    let mut flat = Vec::with_capacity(fixed_steps * z_dim);
    for i in 0..fixed_steps {
        let mut row = base.clone();
        row[fixed_dim] = -1.0 + i as f32 * step;
        flat.extend_from_slice(&row);
    }

    let tensor_data = TensorData::new(flat, [fixed_steps, z_dim]);
    Tensor::from_data(tensor_data, options)
}
