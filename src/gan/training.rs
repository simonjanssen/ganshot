use anyhow::Error;
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, cast::ToElement},
};

use crate::{
    ARTIFACT_DIR,
    gan::{
        data::{TripletBatcher, TripletDataset, sample_z},
        model::{DiscriminatorConfig, GeneratorConfig},
        record::{plot_distr, plot_loss},
    },
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 350)]
    pub epochs: usize,
    #[config(default = 256)]
    pub batch_size: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 4)]
    pub num_workers: usize,
    pub config_g: GeneratorConfig,
    pub config_d: DiscriminatorConfig,
    pub optimizer_g: AdamConfig,
    pub optimizer_d: AdamConfig,
    #[config(default = 1e-3)]
    pub lr: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(device: B::Device) -> Result<(), Error> {
    create_artifact_dir(ARTIFACT_DIR);

    println!("Loading config..");
    let z_dim = 8;
    let nb_hidden = 100;
    let mut rng = rand::rng();

    let config_g = GeneratorConfig::new(z_dim, nb_hidden);
    let config_d = DiscriminatorConfig::new(nb_hidden);
    let optimizer_g = AdamConfig::new();
    let optimizer_d = AdamConfig::new();
    let config = TrainingConfig::new(config_g, config_d, optimizer_g, optimizer_d);

    println!("Init models..");
    let mut generator = config.config_g.init::<B>(&device);
    let mut discriminator = config.config_d.init::<B>(&device);
    let mut optim_g = config.optimizer_g.init();
    let mut optim_d = config.optimizer_d.init();

    println!("Init data loaders..");
    let batcher = TripletBatcher::default();
    let dataset = TripletDataset::train();
    let dataset_len = dataset.len();
    let (x_gt, y_gt) = dataset.to_xy();
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset);

    println!("Starting training..");

    let mut log_epochs = Vec::with_capacity(config.epochs);
    let mut log_loss_g = Vec::with_capacity(config.epochs);
    let mut log_loss_d = Vec::with_capacity(config.epochs);

    let mut snap_epochs: Vec<usize> = Vec::new();
    let mut snap_x_g: Vec<Vec<f32>> = Vec::new();
    let mut snap_y_g: Vec<Vec<f32>> = Vec::new();

    for epoch in 1..config.epochs + 1 {
        let prev_epoch = epoch - 1;
        if prev_epoch.is_multiple_of(10) {
            // evaluate generator
            println!("Sampling from generator..");
            let generator_valid = generator.valid();
            let z = sample_z([dataset_len, z_dim], &mut rng, &device);
            let generator_sample = generator_valid.forward(z);
            let values: Vec<f32> = generator_sample.into_data().to_vec().unwrap();
            let x_g: Vec<f32> = values.iter().step_by(2).copied().collect();
            let y_g: Vec<f32> = values.iter().skip(1).step_by(2).copied().collect();

            snap_epochs.push(prev_epoch);
            snap_x_g.push(x_g);
            snap_y_g.push(y_g);
            plot_distr(
                snap_epochs.clone(),
                x_gt.clone(),
                y_gt.clone(),
                snap_x_g.clone(),
                snap_y_g.clone(),
            );
        }

        let (mut sum_g, mut n_g) = (0.0_f32, 0u32);
        let (mut sum_d, mut n_d) = (0.0_f32, 0u32);

        for (iteration, real_batch) in dataloader.iter().enumerate() {
            // Forward passes
            //println!("Running generator forward pass..");
            let z = sample_z([real_batch.points.dims()[0], z_dim], &mut rng, &device);
            let fake_batch = generator.forward(z);

            //println!("Running discriminator forward passes..");
            let d_scores_on_real = discriminator.forward(real_batch.points);
            let d_scores_on_fake = discriminator.forward(fake_batch);

            //println!("Running backward passes..");
            if iteration.is_multiple_of(2) {
                // Non-saturating generator loss: minimize -log(D(fake)) instead of
                // log(1 - D(fake)), which provides stronger gradients early in training.
                let loss = -d_scores_on_fake.log().mean();
                sum_g += loss.clone().into_scalar().to_f32();
                n_g += 1;

                let grads = loss.backward();
                let grads: GradientsParams = GradientsParams::from_grads(grads, &generator);
                generator = optim_g.step(config.lr, generator, grads);
            } else {
                let loss = -d_scores_on_fake.neg().add_scalar(1.0).log().mean()
                    - d_scores_on_real.log().mean();
                sum_d += loss.clone().into_scalar().to_f32();
                n_d += 1;

                let grads = loss.backward();
                let grads: GradientsParams = GradientsParams::from_grads(grads, &discriminator);
                discriminator = optim_d.step(config.lr, discriminator, grads);
            }
        }

        let avg_loss_g = sum_g / n_g.max(1) as f32;
        let avg_loss_d = sum_d / n_d.max(1) as f32;

        log_epochs.push(epoch);
        log_loss_g.push(avg_loss_g);
        log_loss_d.push(avg_loss_d);
        plot_loss(log_epochs.clone(), log_loss_g.clone(), log_loss_d.clone());

        println!(
            "[Train - Epoch {:>3}] avg_loss_g {:>8.3} | avg_loss_d {:>8.3}",
            epoch, avg_loss_g, avg_loss_d,
        );
    }

    let path_checkpoint = format!("{}/generator-{:>3}", ARTIFACT_DIR, config.epochs);
    generator.save_file(path_checkpoint, &CompactRecorder::new())?;

    Ok(())
}
