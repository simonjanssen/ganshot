use anyhow::Error;
use burn::backend::wgpu::WgpuDevice;
use ganshot::{backend::MyAutodiffBackend, gan::training};

fn main() -> Result<(), Error> {
    let device = WgpuDevice::default();

    training::run::<MyAutodiffBackend>(device)?;
    Ok(())
}
