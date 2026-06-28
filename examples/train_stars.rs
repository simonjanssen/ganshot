use anyhow::Error;
use ganshot::{
    backend::{MyAutodiffBackend, select_device},
    data::stars::RandomStars,
    training,
};

fn main() -> Result<(), Error> {
    let device = select_device();
    let sampler = RandomStars {};
    training::runner::run::<MyAutodiffBackend, _, _>(device, sampler)?;
    Ok(())
}
