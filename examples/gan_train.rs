use anyhow::Error;
use ganshot::{
    backend::{MyAutodiffBackend, init_backend},
    gan::training,
};

fn main() -> Result<(), Error> {
    let device = Default::default();
    init_backend(&device);

    training::run::<MyAutodiffBackend>(device)?;
    Ok(())
}
