use std::marker::PhantomData;

use burn::{
    Tensor,
    data::{dataloader::batcher::Batcher as BurnBatcher, dataset::Dataset as BurnDataset},
    tensor::{TensorData, backend::Backend},
};
use plotly::{
    Layout, Plot, Scatter,
    layout::{Axis, GridPattern, LayoutGrid},
};

/// 2D-Coordinate
pub type Coord2 = (f64, f64);

pub trait Geometry {
    const N: usize;
    type Outline: IntoIterator<Item = Coord2> + Copy + Send + Sync + std::fmt::Debug + 'static;

    /// Create an uninit version of Self
    fn empty() -> Self;

    /// Return the outline coordinates with wich one could draw the geometry's outer bounds (ordering matters)
    fn to_outline(&self) -> Self::Outline;

    fn set_outline(&mut self, outline: Self::Outline);

    fn from_outline(outline: Self::Outline) -> Self
    where
        Self: Sized,
    {
        let mut s = Self::empty();
        s.set_outline(outline);
        s
    }

    fn from_vec(v: Vec<f64>) -> Self;

    fn traces(&self) -> Vec<Box<Scatter<f64, f64>>>;

    fn plot(&self) -> Plot {
        let mut plot = Plot::new();
        for trace in self.traces() {
            plot.add_trace(trace);
        }
        plot
    }
}

pub struct Dataset<G: Geometry> {
    dataset: Vec<G::Outline>,
    mean: f64,
    std: f64,
}

impl<G: Geometry> Dataset<G> {
    pub fn new<I: Iterator<Item = G>>(iterator: I, n: usize) -> Self {
        let dataset: Vec<_> = iterator.take(n).map(|i| i.to_outline()).collect();

        assert!(!dataset.is_empty(), "empty dataset");

        // Standardize all coordinates to zero mean / unit variance. A single
        // pooled mean and std (shared by x and y) keeps the normalization
        // isotropic so the star shapes are not distorted. Computed in one pass
        // over the dataset via running sums (var = E[v²] - E[v]²); coordinates
        // live in a small range, so the naive single-pass form is accurate.
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0.0;
        for outline in &dataset {
            for (x, y) in *outline {
                sum += x + y;
                sum_sq += x * x + y * y;
                count += 2.0;
            }
        }

        let mean = sum / count;
        let std = (sum_sq / count - mean * mean).sqrt();
        assert!(std > 0.0, "std zero");

        Self { dataset, mean, std }
    }
}

impl<G: Geometry> BurnDataset<G::Outline> for Dataset<G> {
    fn get(&self, index: usize) -> Option<G::Outline> {
        self.dataset.get(index).copied()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[derive(Clone, Debug)]
pub struct Batch<B: Backend> {
    pub tensor: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct Batcher<G: Geometry> {
    mean: f64,
    std: f64,
    _marker: PhantomData<fn() -> G>,
}

impl<G: Geometry> From<&Dataset<G>> for Batcher<G> {
    fn from(value: &Dataset<G>) -> Self {
        Self {
            mean: value.mean,
            std: value.std,
            _marker: PhantomData,
        }
    }
}

impl<B: Backend, G: Geometry> BurnBatcher<B, G::Outline, Batch<B>> for Batcher<G> {
    fn batch(&self, items: Vec<G::Outline>, device: &B::Device) -> Batch<B> {
        let rows = items.len();
        let cols = G::N * 2;

        // Flatten the whole batch into a single contiguous buffer (one
        // allocation, preallocated to the exact size) and build a single
        // [rows, cols] tensor. Avoids per-item Vecs, per-item tensors and the
        // extra copy a `Tensor::cat` would incur.
        let mut flat = Vec::with_capacity(rows * cols);
        for outline in items {
            for (x, y) in outline {
                flat.push(x);
                flat.push(y);
            }
        }

        let data = TensorData::new(flat, [rows, cols]).convert::<B::FloatElem>();
        let batched = Tensor::<B, 2>::from_data(data, device)
            .sub_scalar(self.mean)
            .div_scalar(self.std);
        Batch { tensor: batched }
    }
}

/// Plot a 2x3 grid of 6 Geometries
pub fn plot_2x3<G: Geometry>(geometries: &[G; 6]) -> Plot {
    let mut plot = Plot::new();
    let (rows, cols) = (2, 3);

    let mut layout = Layout::new()
        .show_legend(false)
        .width(400 * cols)
        .height(400 * rows)
        .grid(
            LayoutGrid::new()
                .rows(rows)
                .columns(cols)
                .pattern(GridPattern::Independent),
        );

    for (i, geometry) in geometries.iter().enumerate() {
        // plotly names the first subplot's axes "x"/"y" and the rest "x2".."x6"
        let suffix = if i == 0 {
            String::new()
        } else {
            (i + 1).to_string()
        };
        let (x_id, y_id) = (format!("x{suffix}"), format!("y{suffix}"));

        for trace in geometry.traces() {
            plot.add_trace(trace.x_axis(&x_id).y_axis(&y_id));
        }

        let (x_axis, y_axis) = {
            // rectangular frame, no grid/ticks/labels
            let bare_axis = || {
                Axis::new()
                    .show_grid(false)
                    .zero_line(false)
                    .show_tick_labels(false)
                    .show_line(true)
                    .mirror(true)
            };

            (bare_axis(), bare_axis().scale_ratio(1.0))
        };
        let y_axis = y_axis.scale_anchor(&x_id);
        layout = match i {
            0 => layout.x_axis(x_axis).y_axis(y_axis),
            1 => layout.x_axis2(x_axis).y_axis2(y_axis),
            2 => layout.x_axis3(x_axis).y_axis3(y_axis),
            3 => layout.x_axis4(x_axis).y_axis4(y_axis),
            4 => layout.x_axis5(x_axis).y_axis5(y_axis),
            _ => layout.x_axis6(x_axis).y_axis6(y_axis),
        };
    }

    plot.set_layout(layout);
    plot
}
