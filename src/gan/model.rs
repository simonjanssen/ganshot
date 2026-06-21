use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu, Sigmoid},
    tensor::backend::Backend,
};

// --- GAN/Generator

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
}

impl GeneratorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Generator<B> {
        Generator {
            linear1: LinearConfig::new(self.z_dim, self.nb_hidden).init(device),
            relu1: Relu::new(),
            linear2: LinearConfig::new(self.nb_hidden, 2).init(device),
        }
    }
}

// --- GAN/Discriminator

#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    linear1: Linear<B>,
    relu1: Relu,
    linear2: Linear<B>,
    sigmoid1: Sigmoid,
}

impl<B: Backend> Discriminator<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.relu1.forward(x);
        let x = self.linear2.forward(x);
        self.sigmoid1.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct DiscriminatorConfig {
    nb_hidden: usize,
}

impl DiscriminatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Discriminator<B> {
        Discriminator {
            linear1: LinearConfig::new(2, self.nb_hidden).init(device),
            relu1: Relu::new(),
            linear2: LinearConfig::new(self.nb_hidden, 1).init(device),
            sigmoid1: Sigmoid::new(),
        }
    }
}
