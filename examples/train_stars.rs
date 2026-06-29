use ganshot::{
    backend::{MyAutodiffBackend, select_device},
    data::stars::RandomStars,
    training,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = select_device();
    training::runner::run::<MyAutodiffBackend, _, _>(device, RandomStars)?;
    Ok(())
}
