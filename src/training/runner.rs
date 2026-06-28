use std::time::Instant;

use anyhow::Error;
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, cast::ToElement},
};
use rand_distr::Distribution;

use crate::{
    data::commons::{Batcher, Dataset, Geometry},
    models::{
        discriminator::DiscriminatorConfig,
        generator::{GeneratorConfig, sample_z},
    },
    training::recorder::{ARTIFACT_DIR, create_artifact_dir, plot_loss, plot_outlines},
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 100)]
    pub epochs: usize,
    #[config(default = 512)]
    pub batch_size: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 4)]
    pub num_workers: usize,
    pub config_g: GeneratorConfig,
    pub config_d: DiscriminatorConfig,
    pub optimizer_g: AdamConfig,
    pub optimizer_d: AdamConfig,
    #[config(default = 5e-4)]
    pub lr: f64,
}

pub fn run<B: AutodiffBackend, G: Geometry + 'static, D: Distribution<G>>(
    device: B::Device,
    sampler: D,
) -> Result<(), Error> {
    create_artifact_dir(ARTIFACT_DIR);

    println!("Loading config..");
    let z_dim = 8;
    let nb_hidden = 100;
    let real_dim = G::N * 2;
    let mut rng = rand::rng();

    let config_g = GeneratorConfig::new(z_dim, nb_hidden, real_dim);
    let config_d = DiscriminatorConfig::new(real_dim, nb_hidden);
    let optimizer_g = AdamConfig::new();
    let optimizer_d = AdamConfig::new();
    let config = TrainingConfig::new(config_g, config_d, optimizer_g, optimizer_d);

    println!("Init models..");
    let mut generator = config.config_g.init::<B>(&device);
    let mut discriminator = config.config_d.init::<B>(&device);
    let mut optim_g = config.optimizer_g.init();
    let mut optim_d = config.optimizer_d.init();

    println!("Init data loaders..");
    let dataset = Dataset::new(sampler, 10_000);
    let batcher = Batcher::from(&dataset);
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(dataset);

    let mut log_epochs = Vec::with_capacity(config.epochs);
    let mut log_loss_g = Vec::with_capacity(config.epochs);
    let mut log_loss_d = Vec::with_capacity(config.epochs);

    let mut log_epochs_outlines = Vec::with_capacity(config.epochs);
    let mut log_outlines = Vec::with_capacity(config.epochs);

    let n_sample_valid = 5;

    println!("Starting training..");
    for epoch in 1..config.epochs + 1 {
        if (epoch - 1).is_multiple_of(5) {
            // sample from generator
            let generator_valid = generator.valid();
            let z_valid = sample_z([n_sample_valid, z_dim], &mut rng, &device);
            let fake_valid = generator_valid.forward(z_valid);
            let [_rows, cols] = fake_valid.dims();
            let flat: Vec<f32> = fake_valid.into_data().to_vec().unwrap();
            let outlines: Vec<Vec<f64>> = flat
                .chunks_exact(cols)
                .map(|row| row.iter().map(|&v| v as f64).collect())
                .collect();
            log_epochs_outlines.push(epoch - 1);
            log_outlines.push(outlines);
            plot_outlines(log_epochs_outlines.clone(), log_outlines.clone());
        }

        let t = Instant::now();
        let (mut sum_g, mut n_g) = (0.0_f32, 0u32);
        let (mut sum_d, mut n_d) = (0.0_f32, 0u32);

        for real_batch in dataloader.iter() {
            let real = real_batch.tensor;
            let batch_size = real.dims()[0];

            // --- Train discriminator on the full batch ---
            // The fake samples are detached so the discriminator update does not
            // backprop into (or modify) the generator.
            let z = sample_z([batch_size, z_dim], &mut rng, &device);
            let fake = generator.forward(z);
            let d_on_real = discriminator.forward(real);
            let d_on_fake = discriminator.forward(fake.clone().detach());
            let loss_d = -d_on_fake.neg().add_scalar(1.0).log().mean() - d_on_real.log().mean();
            sum_d += loss_d.clone().into_scalar().to_f32();
            n_d += 1;

            let grads = loss_d.backward();
            let grads: GradientsParams = GradientsParams::from_grads(grads, &discriminator);
            discriminator = optim_d.step(config.lr, discriminator, grads);

            // --- Train generator against the just-updated discriminator ---
            // Non-saturating generator loss: minimize -log(D(fake)) instead of
            // log(1 - D(fake)), which provides stronger gradients early in training.
            let d_on_fake = discriminator.forward(fake);
            let loss_g = -d_on_fake.log().mean();
            sum_g += loss_g.clone().into_scalar().to_f32();
            n_g += 1;

            let grads = loss_g.backward();
            let grads: GradientsParams = GradientsParams::from_grads(grads, &generator);
            generator = optim_g.step(config.lr, generator, grads);
        }

        let avg_loss_g = sum_g / n_g.max(1) as f32;
        let avg_loss_d = sum_d / n_d.max(1) as f32;
        let dt_train = t.elapsed();
        let t = Instant::now();

        log_epochs.push(epoch);
        log_loss_g.push(avg_loss_g);
        log_loss_d.push(avg_loss_d);
        plot_loss(log_epochs.clone(), log_loss_g.clone(), log_loss_d.clone());
        let dt_plot = t.elapsed();

        println!(
            "[Train - Epoch {:>3}] avg_loss_g {:>8.3} | avg_loss_d {:>8.3} | {:?} train | {:?} plot",
            epoch, avg_loss_g, avg_loss_d, dt_train, dt_plot
        );
    }

    let path_checkpoint = format!("{}/generator-{:>3}", ARTIFACT_DIR, config.epochs);
    generator.save_file(path_checkpoint, &CompactRecorder::new())?;

    Ok(())
}
