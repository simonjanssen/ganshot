use burn::data::dataset::Dataset;
use ganshot::{
    ARTIFACT_DIR,
    backend::{MyBackend, select_device},
    mnist::inference::infer,
};

fn main() {
    let device = select_device();

    infer::<MyBackend>(
        ARTIFACT_DIR,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
