use burn::{backend::Wgpu, data::dataset::Dataset};
use ganshot::{ARTIFACT_DIR, mnist::inference::infer};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = Default::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
        &device,
        Default::default(),
    );

    infer::<MyBackend>(
        ARTIFACT_DIR,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
