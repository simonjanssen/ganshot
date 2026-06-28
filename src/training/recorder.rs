use plotly::{
    Layout, Plot, Scatter,
    common::{Line, Marker, Mode, Visible},
    layout::{Axis, Slider, SliderCurrentValue, SliderStepBuilder},
};

pub static ARTIFACT_DIR: &str = "./checkpoints";
const MAX_PLOT_POINTS: usize = 250;

pub fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Plot the generator loss against the discriminator loss over epochs.
///
/// Every epoch defines a point `(loss_g, loss_d)`; the full trajectory of these
/// points is drawn as a fixed line, and a slider moves a marker along the trace
/// to visualize the training dynamics epoch by epoch.
pub fn plot_loss(epochs: Vec<usize>, loss_g: Vec<f32>, loss_d: Vec<f32>) {
    let mut plot = Plot::new();
    let n = epochs.len();
    let path_plot = format!("{}/loss.html", ARTIFACT_DIR);

    // Trace index 0 is the full trajectory line, always visible for every step.
    let trace_path = Scatter::new(loss_g.clone(), loss_d.clone())
        .mode(Mode::LinesMarkers)
        .name("trajectory")
        .line(Line::new().color("#1f77b4"));
    plot.add_trace(trace_path);

    // One single-point marker per epoch; only the last is visible initially.
    for (i, (g, d)) in loss_g.iter().zip(&loss_d).enumerate() {
        let visible = if i + 1 == n {
            Visible::True
        } else {
            Visible::False
        };
        let trace_point = Scatter::new(vec![*g], vec![*d])
            .mode(Mode::Markers)
            .name("current")
            .marker(Marker::new().size(12).color("#ff7f0e"))
            .visible(visible);
        plot.add_trace(trace_point);
    }

    // Each slider step toggles which marker is visible. The visibility array
    // spans all traces: index 0 (trajectory) is always on, marker `i` is on
    // only for step `i`.
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
                .push_restyle(Scatter::<f32, f32>::modify_visible(visible))
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
        .x_axis(Axis::new().title("loss_g"))
        .y_axis(
            Axis::new()
                .title("loss_d")
                .scale_anchor("x")
                .scale_ratio(1.0),
        )
        .sliders(vec![slider]);

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

/// Plot generator outlines across epochs with a slider.
///
/// `outlines` is indexed as `outlines[epoch][outline]`, where each innermost
/// `Vec` holds one outline's coordinates flattened as `(x1, y1, x2, y2, ...)`.
/// Outlines with a single point (length 2) are drawn as markers only; longer
/// outlines are drawn as connected lines with markers. A slider toggles which
/// epoch's group of outlines is visible, mirroring [`plot_distr`].
pub fn plot_outlines(epochs: Vec<usize>, outlines: Vec<Vec<Vec<f64>>>) {
    let mut plot = Plot::new();
    let n = epochs.len();

    // Number of traces contributed by each epoch, used to build the global
    // per-trace visibility array spanning every epoch for each slider step.
    let counts: Vec<usize> = outlines.iter().map(Vec::len).collect();

    // Add every epoch's outlines as individual traces; only the last epoch's
    // group is visible initially.
    for (i, epoch_outlines) in outlines.into_iter().enumerate() {
        let is_last = i + 1 == n;
        for outline in epoch_outlines {
            let mut xs = Vec::with_capacity(outline.len() / 2);
            let mut ys = Vec::with_capacity(outline.len() / 2);
            for pair in outline.chunks_exact(2) {
                xs.push(pair[0]);
                ys.push(pair[1]);
            }

            // A single point can only be drawn as a marker; there is no line to draw.
            let mode = if xs.len() <= 1 {
                Mode::Markers
            } else {
                // Close the outline by repeating the first point so the edge
                // between the last and first vertex is drawn.
                xs.push(xs[0]);
                ys.push(ys[0]);
                Mode::Lines
            };

            let visible = if is_last {
                Visible::True
            } else {
                Visible::False
            };
            let trace = Scatter::new(xs, ys)
                .mode(mode)
                .show_legend(false)
                .visible(visible);
            plot.add_trace(trace);
        }
    }

    // Each slider step makes visible exactly the trace-group belonging to that
    // epoch. The visibility array spans all traces in epoch order.
    let steps: Vec<_> = epochs
        .iter()
        .enumerate()
        .map(|(i, epoch)| {
            let mut visible = Vec::with_capacity(counts.iter().sum());
            for (j, &count) in counts.iter().enumerate() {
                for _ in 0..count {
                    visible.push(if j == i {
                        Visible::True
                    } else {
                        Visible::False
                    });
                }
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
        .show_legend(false)
        .width(700)
        .height(700)
        .x_axis(Axis::new().title("x"))
        .y_axis(Axis::new().title("y").scale_anchor("x").scale_ratio(1.0))
        .sliders(vec![slider]);
    plot.set_layout(layout);

    let path_plot = format!("{}/generator_outlines.html", ARTIFACT_DIR);
    plot.write_html(path_plot);
}
