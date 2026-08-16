use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu, Sigmoid},
    tensor::backend::Backend,
};

#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    linear1: Linear<B>,
    relu1: Relu,
    linear2: Linear<B>,
    relu2: Relu,
    linear3: Linear<B>,
    sigmoid1: Sigmoid,
}

impl<B: Backend> Discriminator<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.relu1.forward(x);
        let x = self.linear2.forward(x);
        let x = self.relu2.forward(x);
        let x = self.linear3.forward(x);
        self.sigmoid1.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct DiscriminatorConfig {
    input_dim: usize,
    h1_dim: usize,
    h2_dim: usize,
}

impl DiscriminatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Discriminator<B> {
        Discriminator {
            linear1: LinearConfig::new(self.input_dim, self.h1_dim).init(device),
            relu1: Relu::new(),
            linear2: LinearConfig::new(self.h1_dim, self.h2_dim).init(device),
            relu2: Relu::new(),
            linear3: LinearConfig::new(self.h2_dim, 1).init(device),
            sigmoid1: Sigmoid::new(),
        }
    }
}
