use ganshot::{
    backend::{MyAutodiffBackend, select_device},
    data::stars::RandomStars,
    training,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = select_device();
    let stars = RandomStars::new();
    training::runner::run::<MyAutodiffBackend, _, _>(device, stars)?;
    Ok(())
}
