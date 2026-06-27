use burn::{
    Tensor,
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    tensor::Shape,
};
use ganshot::{
    backend::{MyAutodiffBackend, MyBackend, select_device},
    gan::{
        data::{TripletBatch, TripletBatcher, TripletDataset, sample_z},
        model::{DiscriminatorConfig, GeneratorConfig},
    },
};

#[test]
fn generator_forward_pass() {
    let device = select_device();

    let (z_dim, nb_hidden, batch_size) = (8, 100, 64);
    let config = GeneratorConfig::new(z_dim, nb_hidden);
    let generator = config.init::<MyAutodiffBackend>(&device);

    let mut rng = rand::rng();
    let z = sample_z([batch_size, z_dim], &mut rng, &device);

    let output = generator.forward(z);
    assert_eq!(output.shape(), Shape::new([batch_size, 2]));
}

#[test]
fn discriminator_forward_pass() {
    let device = select_device();

    let (nb_hidden, batch_size) = (100, 64);
    let config = DiscriminatorConfig::new(nb_hidden);
    let discriminator = config.init::<MyAutodiffBackend>(&device);

    let x = Tensor::empty([batch_size, 2], &device);
    let output = discriminator.forward(x);
    assert_eq!(output.shape(), Shape::new([batch_size, 1]));
}

#[test]
fn load_dataset() {
    let dataset = TripletDataset::train();
    assert_eq!(dataset.len(), 10000);

    let item = dataset.get(42);
    assert!(item.is_some());
}

#[test]
fn iterate_batches() {
    let device = select_device();

    let (batch_size, seed, num_workers) = (64, 42, 4);
    let batcher = TripletBatcher::default();
    let dataset = TripletDataset::train();

    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(num_workers)
        .set_device(device)
        .build(dataset);

    let batch: Option<TripletBatch<MyBackend>> = dataloader.iter().next();
    assert!(batch.is_some());

    let batch = batch.unwrap();
    assert_eq!(batch.points.shape(), Shape::new([batch_size, 2]));
}
