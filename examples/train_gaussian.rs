use ganshot::{
    backend::{MyAutodiffBackend, select_device},
    data::points::GaussianTriplet,
    training,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = select_device();
    let sampler = GaussianTriplet::default();
    training::runner::run::<MyAutodiffBackend, _, _>(device, sampler)?;
    Ok(())
}
