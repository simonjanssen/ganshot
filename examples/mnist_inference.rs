use burn::data::dataset::Dataset;
use ganshot::{
    ARTIFACT_DIR,
    backend::{MyBackend, init_backend},
    mnist::inference::infer,
};

fn main() {
    let device = Default::default();
    init_backend(&device);

    infer::<MyBackend>(
        ARTIFACT_DIR,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
