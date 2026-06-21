use burn::{backend::wgpu::WgpuDevice, optim::AdamConfig};
use ganshot::{
    ARTIFACT_DIR,
    backend::MyAutodiffBackend,
    mnist::{
        model::ModelConfig,
        training::{TrainingConfig, train},
    },
};

fn main() {
    let device = WgpuDevice::default();

    train::<MyAutodiffBackend>(
        ARTIFACT_DIR,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
