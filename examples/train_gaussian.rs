use anyhow::Error;
use ganshot::{
    backend::{MyAutodiffBackend, select_device},
    data::points::GaussianTriplet,
    training,
};

fn main() -> Result<(), Error> {
    let device = select_device();
    let sampler = GaussianTriplet::default();
    training::runner::run::<MyAutodiffBackend, _, _>(device, sampler)?;
    Ok(())
}
