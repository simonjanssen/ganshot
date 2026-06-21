use anyhow::Error;
use burn::backend::{Autodiff, Wgpu};
use ganshot::gan::training;

fn main() -> Result<(), Error> {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = Default::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
        &device,
        Default::default(),
    );
    training::run::<MyAutodiffBackend>(device)?;
    Ok(())
}
