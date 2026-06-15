#![recursion_limit = "256"]

use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};
use ganshot::{
    model::ModelConfig,
    training::{TrainingConfig, train},
};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = Default::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
        &device,
        Default::default(),
    );
    let artifact_dir = "./tmp/";
    train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
