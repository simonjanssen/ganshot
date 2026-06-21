use burn::optim::AdamConfig;
use ganshot::{
    ARTIFACT_DIR,
    backend::{MyAutodiffBackend, init_backend},
    mnist::{
        model::ModelConfig,
        training::{TrainingConfig, train},
    },
};

fn main() {
    let device = Default::default();
    init_backend(&device);

    train::<MyAutodiffBackend>(
        ARTIFACT_DIR,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
