# Copilot instructions for `ganshot`

`ganshot` is a Rust (edition 2024) machine-learning playground built on the
[Burn](https://burn.dev) framework. It contains two self-contained examples: a
GAN that learns a toy 2D triple-Gaussian distribution, and a small CNN MNIST
classifier. Run on the Wgpu/Metal backend.

## Build, test, lint

CI (`.github/workflows/quality.yml`) runs with `RUSTFLAGS="-D warnings"`, so warnings fail. Match it locally:

```sh
cargo fmt --all -- --check
cargo clippy --all-features --all-targets --no-deps -- -D warnings
cargo check --all-features --all-targets
cargo test --all-features --all-targets   # integration + unit tests
cargo test --all-features --doc           # doctests
```

Run a single test by name (substring filter), e.g. one of the `tests/gan.rs` cases:

```sh
cargo test --all-features --test gan generator_forward_pass
```

Run the examples (these are the real entrypoints; there is no binary target):

```sh
cargo run --example gan_train        # trains the GAN, writes plots to ./checkpoints
cargo run --example mnist_train      # trains the MNIST CNN
cargo run --example mnist_inference  # loads ./checkpoints model and predicts one sample
```

Dependency hygiene (`.github/workflows/audit.yml`) runs on `Cargo.{toml,lock}` changes:

```sh
cargo audit --deny warnings        # config in .cargo/audit.toml
cargo deny check licenses          # config in .cargo/deny.toml (permissive licenses only)
```

## Architecture

- `src/backend.rs` is the single source of truth for the compute backend.
  Everything refers to `MyBackend` (`Wgpu<f32, i32>`) and `MyAutodiffBackend`
  (`Autodiff<MyBackend>`); never hard-code a backend elsewhere. `init_backend`
  wires up the Metal graphics adapter via a `std::sync::Once`.
- `src/lib.rs` exposes `gan`, `mnist`, and `backend` modules plus
  `ARTIFACT_DIR = "./checkpoints"`, the output dir for saved models and plots.
- The two examples are deliberately parallel but use **different training
  styles**:
  - `mnist` uses Burn's high-level `Learner` / `SupervisedTraining` with
    `TrainStep`/`InferenceStep` impls on the model (`src/mnist/model.rs`) and
    built-in metrics/checkpointing.
  - `gan` uses a **hand-written training loop** (`src/gan/training.rs`):
    discriminator and generator are stepped manually each batch with
    `optim.step(lr, model, grads)`. Fakes are `.detach()`ed for the
    discriminator update; the generator uses the non-saturating
    `-log(D(fake))` loss. Read the inline comments there before changing loss
    or update order.
- Each example follows the same module layout: `data` (dataset + `Batcher`),
  `model` (network + config), `training`, and example-specific extras
  (`gan/record.rs` for plots, `mnist/inference.rs`). The GAN's synthetic data
  generator lives in `src/gan/data/triplet.rs`.
- Visualization uses `plotly` to write standalone HTML into `ARTIFACT_DIR`
  (`loss.html`, `generator.html`); there is no live dashboard.

## Conventions

- **Model definition pattern**: a `#[derive(Module, Debug)]` struct holding
  Burn layers, paired with a `#[derive(Config, Debug)]` `XConfig` struct whose
  `init<B: Backend>(&self, device: &B::Device) -> X<B>` constructs it. Follow
  this pattern (see `gan/model.rs`, `mnist/model.rs`) for any new model.
- Hyperparameters are `Config` fields with `#[config(default = ...)]`; prefer
  adding a defaulted config field over a magic constant.
- Generic over the backend `B` with trait bounds (`Backend` /
  `AutodiffBackend`); concrete backend types are only named in `backend.rs`,
  examples, and `tests/`.
- Models/configs are persisted with `CompactRecorder`; MNIST also saves its
  `TrainingConfig` to `config.json` and reloads it for inference.
- `create_artifact_dir` wipes `ARTIFACT_DIR` at the start of every training
  run — assume artifacts are not preserved across runs.
- Random sampling of generator noise goes through `gan::data::sample_z`
  (Normal(0,1)); use it rather than constructing distributions inline.
- `ARTIFACT_DIR` (`/checkpoints*`) and `target` are git-ignored.
