use plotly::{
    Layout, Plot, Scatter,
    common::{Line, Marker, Mode, Visible},
    layout::{Axis, Slider, SliderCurrentValue, SliderStepBuilder},
};

pub static ARTIFACT_DIR: &str = "./checkpoints";

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
        .mode(Mode::Lines)
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

/// Plot generator outlines across epochs with a slider.
///
/// `outlines` is indexed as `outlines[epoch][outline]`, where each innermost
/// `Vec` holds one outline's coordinates flattened as `(x1, y1, x2, y2, ...)`.
/// Single-point outlines (length 2) of an epoch are merged into one marker
/// trace; multi-point outlines are each drawn as their own closed line. A
/// slider toggles which epoch's group of outlines is visible, mirroring
/// [`plot_distr`].
pub fn plot_outlines(epochs: Vec<usize>, outlines: Vec<Vec<Vec<f64>>>) {
    let mut plot = Plot::new();
    let n = epochs.len();

    // Number of traces contributed by each epoch, used to build the global
    // per-trace visibility array spanning every epoch for each slider step.
    // Filled while building traces, since a mix of points and polygons makes
    // the count content-dependent.
    let mut counts: Vec<usize> = Vec::with_capacity(n);

    // Fixed, square axis range covering every coordinate across all epochs, so
    // the coordinate system stays put while sliding instead of rescaling per
    // epoch. Even-indexed values are x, odd-indexed are y.
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for epoch_outlines in &outlines {
        for outline in epoch_outlines {
            for &v in outline {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        lo = -1.0;
        hi = 1.0;
    }
    let pad = (hi - lo) * 0.05;
    let range = vec![lo - pad, hi + pad];

    // Add every epoch's outlines as traces; only the last epoch's group is
    // visible initially. Multi-point outlines each become their own closed line
    // trace (so lines don't bridge separate polygons), while all single-point
    // outlines of an epoch are merged into a single marker trace to avoid one
    // trace per point.
    for (i, epoch_outlines) in outlines.into_iter().enumerate() {
        let is_last = i + 1 == n;
        let mut traces_this_epoch = 0;

        // Accumulated single-point outlines for this epoch.
        let mut point_xs = Vec::new();
        let mut point_ys = Vec::new();

        for outline in epoch_outlines {
            // A single point (or empty) has no edge to draw: collect it for the
            // shared marker trace instead of emitting a per-point trace.
            if outline.len() <= 2 {
                if outline.len() == 2 {
                    point_xs.push(outline[0]);
                    point_ys.push(outline[1]);
                }
                continue;
            }

            let mut xs = Vec::with_capacity(outline.len() / 2 + 1);
            let mut ys = Vec::with_capacity(outline.len() / 2 + 1);
            for pair in outline.chunks_exact(2) {
                xs.push(pair[0]);
                ys.push(pair[1]);
            }
            // Close the outline by repeating the first point so the edge between
            // the last and first vertex is drawn.
            xs.push(xs[0]);
            ys.push(ys[0]);

            let visible = if is_last {
                Visible::True
            } else {
                Visible::False
            };
            let trace = Scatter::new(xs, ys)
                .mode(Mode::Lines)
                .show_legend(false)
                .visible(visible);
            plot.add_trace(trace);
            traces_this_epoch += 1;
        }

        // One combined marker trace for all single-point outlines of the epoch.
        if !point_xs.is_empty() {
            let visible = if is_last {
                Visible::True
            } else {
                Visible::False
            };
            let trace = Scatter::new(point_xs, point_ys)
                .mode(Mode::Markers)
                .show_legend(false)
                .visible(visible);
            plot.add_trace(trace);
            traces_this_epoch += 1;
        }

        counts.push(traces_this_epoch);
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
        .x_axis(Axis::new().title("x").range(range.clone()))
        .y_axis(
            Axis::new()
                .title("y")
                .range(range)
                .scale_anchor("x")
                .scale_ratio(1.0),
        )
        .sliders(vec![slider]);
    plot.set_layout(layout);

    let path_plot = format!("{}/generator_outlines.html", ARTIFACT_DIR);
    plot.write_html(path_plot);
}
