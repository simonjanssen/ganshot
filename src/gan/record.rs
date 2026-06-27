use plotly::{
    Layout, Plot, Scatter,
    common::{AxisSide, Line, Mode, Visible},
    layout::{Axis, Slider, SliderCurrentValue, SliderStepBuilder},
};

use crate::ARTIFACT_DIR;

/// Maximum number of points plotted per distribution to keep the plots legible.
const MAX_PLOT_POINTS: usize = 250;

/// Plot model losses over epochs
pub fn plot_loss(epochs: Vec<usize>, loss_g: Vec<f32>, loss_d: Vec<f32>) {
    let mut plot = Plot::new();
    let layout = Layout::new()
        .show_legend(true)
        .x_axis(Axis::new().title("epoch"))
        .y_axis(Axis::new().title("loss_g"))
        .y_axis2(
            Axis::new()
                .title("loss_d")
                .overlaying("y")
                .side(AxisSide::Right),
        );
    let path_plot = format!("{}/loss.html", ARTIFACT_DIR);

    let trace_loss_g = Scatter::new(epochs.clone(), loss_g)
        .mode(Mode::Lines)
        .name("loss_g")
        .line(Line::new().color("#1f77b4"));
    let trace_loss_d = Scatter::new(epochs, loss_d)
        .mode(Mode::Lines)
        .name("loss_d")
        .line(Line::new().color("#ff7f0e"))
        .y_axis("y2");

    plot.add_trace(trace_loss_g);
    plot.add_trace(trace_loss_d);
    plot.set_layout(layout);
    plot.write_html(path_plot);
}

/// Plot ground truth and generated data distributions across epochs.
///
/// Produces a single HTML file with a slider to browse the generator output at
/// each recorded epoch. The ground truth distribution stays fixed while the
/// generator samples (`x_g`/`y_g`, one inner `Vec` per recorded epoch) change
/// with the slider position.
pub fn plot_distr(
    epochs: Vec<usize>,
    mut x_gt: Vec<f64>,
    mut y_gt: Vec<f64>,
    x_g: Vec<Vec<f32>>,
    y_g: Vec<Vec<f32>>,
) {
    let mut plot = Plot::new();
    let n = epochs.len();

    // Clip every distribution to the first `MAX_PLOT_POINTS` points to avoid clutter.
    x_gt.truncate(MAX_PLOT_POINTS);
    y_gt.truncate(MAX_PLOT_POINTS);

    // Ground truth is trace index 0 and stays visible for every slider step.
    let trace_gt = Scatter::new(x_gt, y_gt)
        .mode(Mode::Markers)
        .name("ground truth");
    plot.add_trace(trace_gt);

    // One generator trace per recorded epoch; only the last is visible initially.
    for (i, (mut xs, mut ys)) in x_g.into_iter().zip(y_g).enumerate() {
        xs.truncate(MAX_PLOT_POINTS);
        ys.truncate(MAX_PLOT_POINTS);
        let visible = if i + 1 == n {
            Visible::True
        } else {
            Visible::False
        };
        let trace_g = Scatter::new(xs, ys)
            .mode(Mode::Markers)
            .name("generator")
            .visible(visible);
        plot.add_trace(trace_g);
    }

    // Each slider step toggles which generator trace is visible. The visibility
    // array spans all traces: index 0 (ground truth) is always on, generator
    // trace `i` is on only for step `i`.
    let steps: Vec<_> = epochs
        .iter()
        .enumerate()
        .map(|(i, epoch)| {
            let mut visible = Vec::with_capacity(n + 1);
            visible.push(Visible::True);
            for j in 0..n {
                visible.push(if j == i {
                    Visible::True
                } else {
                    Visible::False
                });
            }
            SliderStepBuilder::new()
                .label(epoch.to_string())
                .push_restyle(Scatter::<f64, f64>::modify_visible(visible))
                .build()
                .unwrap()
        })
        .collect();

    let slider = Slider::new()
        .active(n.max(1) as i32 - 1)
        .current_value(SliderCurrentValue::new().prefix("Epoch: "))
        .steps(steps);

    let layout = Layout::new()
        .show_legend(true)
        .width(700)
        .height(700)
        .x_axis(Axis::new().title("x").range(vec![-3.0, 3.0]))
        .y_axis(
            Axis::new()
                .title("y")
                .range(vec![-3.0, 3.0])
                .scale_anchor("x")
                .scale_ratio(1.0),
        )
        .sliders(vec![slider]);

    plot.set_layout(layout);
    let path_plot = format!("{}/generator.html", ARTIFACT_DIR);
    plot.write_html(path_plot);
}
