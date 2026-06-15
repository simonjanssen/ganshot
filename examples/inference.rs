use burn::{backend::Wgpu, data::dataset::Dataset};
use ganshot::inference::infer;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = Default::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
        &device,
        Default::default(),
    );
    let artifact_dir = "./tmp/";

    infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
