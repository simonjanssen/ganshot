use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    tensor::{Distribution, TensorCreationOptions},
};
use rand::Rng;

use crate::gan::data::triplet::TripleGaussian2;

// --- Library Imports

pub mod triplet;

// --- Core/Shared Types

/// Sample a random generator input tensor with Normal(mean=0.0, std=1.0)
pub fn sample_z<S, R, B, const D: usize>(
    shape: S,
    rng: &mut R,
    options: impl Into<TensorCreationOptions<B>>,
) -> Tensor<B, D, Float>
where
    S: Into<Shape>,
    R: Rng,
    B: Backend,
{
    let tensor_data = TensorData::random::<f32, _, _>(shape, Distribution::Normal(0.0, 1.0), rng);
    Tensor::from_data(tensor_data, options)
}

#[derive(Clone, Debug)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

// --- Dataset/Dataloaders

#[derive(Clone, Debug)]
pub struct TripletItem {
    pub point: Point2,
}

pub struct TripletDataset {
    dataset: Vec<TripletItem>,
}

impl Dataset<TripletItem> for TripletDataset {
    fn get(&self, index: usize) -> Option<TripletItem> {
        self.dataset.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl TripletDataset {
    pub fn train() -> Self {
        Self::new()
    }

    fn new() -> Self {
        let triplet = TripleGaussian2::builder(
            Point2 { x: 0., y: -1. },
            Point2 {
                x: -0.707,
                y: 0.707,
            },
            Point2 { x: 0.707, y: 0.707 },
        )
        .build();
        let mut rng = rand::rng();
        let dataset = std::iter::repeat_with(|| triplet.sample(&mut rng))
            .take(10000)
            .map(|point| TripletItem { point })
            .collect();
        Self { dataset }
    }

    pub fn to_xy(&self) -> (Vec<f64>, Vec<f64>) {
        let mut x = Vec::with_capacity(self.len());
        let mut y = Vec::with_capacity(self.len());
        for item in self.dataset.clone() {
            x.push(item.point.x);
            y.push(item.point.y);
        }
        (x, y)
    }
}

#[derive(Clone, Debug)]
pub struct TripletBatch<B: Backend> {
    pub points: Tensor<B, 2>,
}

#[derive(Clone, Debug, Default)]
pub struct TripletBatcher {}

impl<B: Backend> Batcher<B, TripletItem, TripletBatch<B>> for TripletBatcher {
    fn batch(&self, items: Vec<TripletItem>, device: &B::Device) -> TripletBatch<B> {
        let tensors = items
            .iter()
            .map(|item| TensorData::from([item.point.x, item.point.y]).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 2]))
            .collect();
        let points = Tensor::cat(tensors, 0);
        TripletBatch { points }
    }
}
