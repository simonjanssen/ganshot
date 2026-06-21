use burn::{backend::wgpu::WgpuDevice, data::dataset::Dataset};
use ganshot::{ARTIFACT_DIR, backend::MyBackend, mnist::inference::infer};

fn main() {
    let device = WgpuDevice::default();

    infer::<MyBackend>(
        ARTIFACT_DIR,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
