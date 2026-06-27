use anyhow::Error;
use ganshot::{
    backend::{MyAutodiffBackend, select_device},
    gan::training,
};

fn main() -> Result<(), Error> {
    let device = select_device();

    training::run::<MyAutodiffBackend>(device)?;
    Ok(())
}
