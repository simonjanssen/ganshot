use burn::optim::AdamConfig;
use ganshot::{
    ARTIFACT_DIR,
    backend::{MyAutodiffBackend, select_device},
    mnist::{
        model::ModelConfig,
        training::{TrainingConfig, train},
    },
};

fn main() {
    let device = select_device();

    train::<MyAutodiffBackend>(
        ARTIFACT_DIR,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );
}
