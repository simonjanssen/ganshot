use std::time::Instant;

use burn::{
    config::Config,
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};

use crate::{mnist::data::MnistBatcher, mnist::training::TrainingConfig};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MnistItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;
    let batcher = MnistBatcher::default();
    let batch = batcher.batch(vec![item], &device);
    let t0 = Instant::now();
    let output = model.forward(batch.images);
    println!("forward pass in {:?}", t0.elapsed());
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {predicted} Expected {label}");
}
