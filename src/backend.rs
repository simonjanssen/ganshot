use std::sync::Once;

use burn::backend::Autodiff;
use burn::tensor::Device;

#[cfg(target_os = "macos")]
use burn::backend::{Wgpu, wgpu::WgpuDevice};

#[cfg(not(target_os = "macos"))]
use burn::backend::Flex;

// The backend is a *compile-time* type, not a runtime value: a single
// `select_device` call is monomorphized for one backend, so it cannot return
// different concrete devices at runtime. The platform choice therefore happens
// here via `cfg`, and `select_device` just builds the matching default device
// and runs any one-time setup that backend needs.

/// Apple Silicon: GPU via the Wgpu/Metal backend.
#[cfg(target_os = "macos")]
pub type MyBackend = Wgpu<f32, i32>;

/// Everywhere else: CPU-only, pure-Rust Flex backend.
#[cfg(not(target_os = "macos"))]
pub type MyBackend = Flex<f32, i32>;

pub type MyAutodiffBackend = Autodiff<MyBackend>;

#[cfg(target_os = "macos")]
static INIT_BACKEND: Once = Once::new();

/// One-time Metal graphics-adapter setup for the Wgpu backend.
#[cfg(target_os = "macos")]
fn init_backend(device: &WgpuDevice) {
    INIT_BACKEND.call_once(|| {
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
            device,
            Default::default(),
        );
    });
}

/// Factory that returns the default device for the platform's [`MyBackend`],
/// performing any backend initialization it requires.
pub fn select_device() -> Device<MyBackend> {
    let device = Device::<MyBackend>::default();

    #[cfg(target_os = "macos")]
    init_backend(&device);

    device
}
