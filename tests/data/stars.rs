use burn::{data::dataloader::DataLoaderBuilder, tensor::Shape};
use ganshot::{
    backend::{MyBackend, select_device},
    data::{
        commons::{Batch, Batcher, Dataset, Geometry},
        stars::{RandomStars, Star},
    },
};

const EPS: f64 = 1e-9;

/// Classic pentagram inner/outer radius ratio: sin(18°) / sin(54°).
fn inner_ratio() -> f64 {
    (18.0_f64.to_radians()).sin() / (54.0_f64.to_radians()).sin()
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

#[test]
fn returns_ten_points() {
    let star = Star::new((0.0, 0.0), 2.0, 0.0);
    let outline = star.to_outline();
    assert_eq!(outline.len(), 10);
}

#[test]
fn tips_lie_on_outer_circle_and_valleys_on_inner_circle() {
    let center = (1.5, -2.0);
    let size = 3.0;
    let star = Star::new(center, size, 0.0);
    let outline = star.to_outline();

    // Even indices are tips (outer radius), odd indices are valleys (inner radius).
    for (i, &p) in outline.iter().enumerate() {
        let r = dist(p, center);
        if i % 2 == 0 {
            assert!((r - size).abs() < EPS, "tip {i} radius {r} != {size}");
        } else {
            let expected = size * inner_ratio();
            assert!(
                (r - expected).abs() < EPS,
                "valley {i} radius {r} != {expected}"
            );
        }
    }
}

#[test]
fn first_tip_points_up_for_zero_rotation() {
    let size = 2.0;
    let star = Star::new((0.0, 0.0), size, 0.0);
    let outline = star.to_outline();
    let first = outline[0];
    assert!(first.0.abs() < EPS, "x should be ~0, got {}", first.0);
    assert!(
        (first.1 - size).abs() < EPS,
        "y should be ~{size}, got {}",
        first.1
    );
}

#[test]
fn centroid_matches_center() {
    let center = (4.0, -1.0);
    let star = Star::new(center, 2.5, 0.3);
    let outline = star.to_outline();
    let sum = outline
        .iter()
        .fold((0.0, 0.0), |acc, p| (acc.0 + p.0, acc.1 + p.1));
    let centroid = (sum.0 / outline.len() as f64, sum.1 / outline.len() as f64);
    assert!((centroid.0 - center.0).abs() < EPS);
    assert!((centroid.1 - center.1).abs() < EPS);
}

#[test]
fn full_turn_is_invariant() {
    let zero = Star::new((0.5, 0.5), 1.7, 0.0).to_outline();
    let full = Star::new((0.5, 0.5), 1.7, 1.0).to_outline();
    for (a, b) in zero.iter().zip(full.iter()) {
        assert!(dist(*a, *b) < EPS, "{a:?} != {b:?}");
    }
}

#[test]
fn points_are_counter_clockwise() {
    // Shoelace signed area is positive for CCW-ordered polygons.
    let outline = Star::new((0.0, 0.0), 2.0, 0.0).to_outline();
    let mut area = 0.0;
    for i in 0..outline.len() {
        let (x1, y1) = outline[i];
        let (x2, y2) = outline[(i + 1) % outline.len()];
        area += x1 * y2 - x2 * y1;
    }
    assert!(area > 0.0, "expected CCW (positive area), got {area}");
}

#[test]
fn iterate_batches() {
    let device = select_device();

    let (batch_size, seed, num_workers) = (64, 42, 4);
    let n = 10_000;
    let dataset = Dataset::new(RandomStars, n);
    let batcher = Batcher::from(&dataset);

    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(num_workers)
        .set_device(device)
        .build(dataset);

    let batches: Vec<Batch<MyBackend>> = dataloader.iter().collect();
    assert_eq!(batches.len(), n.div_ceil(batch_size));

    let batch = &batches[0];
    assert_eq!(batch.tensor.shape(), Shape::new([batch_size, Star::N * 2]));
}
