use std::sync::Once;

use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};

pub type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;

pub static INIT_BACKEND: Once = Once::new();

pub fn init_backend(device: &WgpuDevice) {
    INIT_BACKEND.call_once(|| {
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
            device,
            Default::default(),
        );
    });
}
