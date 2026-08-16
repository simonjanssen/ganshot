use std::fs;

use anyhow::{Error, anyhow};

use ganshot::data::human_pose::{BodyJoint, HumanPoses};
use plotly::{Layout, Plot, Scatter, common::Mode, layout::Axis};

struct SemanticHumanPose {
    r_ankle: BodyJoint,
    r_knee: BodyJoint,
    r_hip: BodyJoint,
    l_hip: BodyJoint,
    l_knee: BodyJoint,
    l_ankle: BodyJoint,
    pelvis: BodyJoint,
    thorax: BodyJoint,
    upper_neck: BodyJoint,
    head_top: BodyJoint,
    r_wrist: BodyJoint,
    r_elbow: BodyJoint,
    r_shoulder: BodyJoint,
    l_shoulder: BodyJoint,
    l_elbow: BodyJoint,
    l_wrist: BodyJoint,
}

impl From<[BodyJoint; 16]> for SemanticHumanPose {
    fn from(value: [BodyJoint; 16]) -> Self {
        Self {
            r_ankle: value[0],
            r_knee: value[1],
            r_hip: value[2],
            l_hip: value[3],
            l_knee: value[4],
            l_ankle: value[5],
            pelvis: value[6],
            thorax: value[7],
            upper_neck: value[8],
            head_top: value[9],
            r_wrist: value[10],
            r_elbow: value[11],
            r_shoulder: value[12],
            l_shoulder: value[13],
            l_elbow: value[14],
            l_wrist: value[15],
        }
    }
}

struct JointTrace {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl JointTrace {
    pub fn new() -> Self {
        Self {
            xs: Vec::new(),
            ys: Vec::new(),
        }
    }

    pub fn push(&mut self, joint: BodyJoint) {
        self.xs.push(joint.x);
        self.ys.push(joint.y);
    }
}

impl From<JointTrace> for Box<Scatter<f64, f64>> {
    fn from(value: JointTrace) -> Self {
        Scatter::new(value.xs, value.ys).mode(Mode::LinesMarkers)
    }
}

fn main() -> Result<(), Error> {
    let path_ann = "data/mpii_human_pose_v1_u12_2/annotations.json";
    let mut human_poses = HumanPoses::from_annotations_json(path_ann)?;
    println!("poses: {}", human_poses.len());

    let raw_pose = human_poses.next().ok_or(anyhow!("No pose!"))?;
    let raw_joints = raw_pose.joints().ok_or(anyhow!("No joints!"))?;
    let pose: SemanticHumanPose = raw_joints.into();

    let mut plot = Plot::new();

    // right foot: r_ankle -> r_knee -> r_hip
    let trace_right_foot: Box<Scatter<f64, f64>> = {
        let mut trace = JointTrace::new();
        trace.push(pose.r_ankle);
        trace.push(pose.r_knee);
        trace.push(pose.r_hip);
        trace.push(pose.pelvis);
        trace.into()
    };
    plot.add_trace(trace_right_foot);

    // left foot: l_ankle -> l_knee -> l_hip
    let trace_left_foot: Box<Scatter<f64, f64>> = {
        let mut trace = JointTrace::new();
        trace.push(pose.l_ankle);
        trace.push(pose.l_knee);
        trace.push(pose.l_hip);
        trace.push(pose.pelvis);
        trace.into()
    };
    plot.add_trace(trace_left_foot);

    // header/stem
    let trace_header: Box<Scatter<f64, f64>> = {
        let mut trace = JointTrace::new();
        trace.push(pose.head_top);
        trace.push(pose.upper_neck);
        trace.push(pose.thorax);
        trace.push(pose.pelvis);
        trace.into()
    };
    plot.add_trace(trace_header);

    // left arm
    let trace_left_arm: Box<Scatter<f64, f64>> = {
        let mut trace = JointTrace::new();
        trace.push(pose.l_wrist);
        trace.push(pose.l_elbow);
        trace.push(pose.l_shoulder);
        trace.push(pose.thorax);
        trace.into()
    };
    plot.add_trace(trace_left_arm);

    // right arm
    let trace_right_arm: Box<Scatter<f64, f64>> = {
        let mut trace = JointTrace::new();
        trace.push(pose.r_wrist);
        trace.push(pose.r_elbow);
        trace.push(pose.r_shoulder);
        trace.push(pose.thorax);
        trace.into()
    };
    plot.add_trace(trace_right_arm);

    // image coordinates: y grows downwards, so plot the axis descending
    let y_min = raw_joints.iter().map(|j| j.y).fold(f64::INFINITY, f64::min);
    let y_max = raw_joints
        .iter()
        .map(|j| j.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let pad = 0.05 * (y_max - y_min);

    // rectangular frame, no grid/ticks/labels
    let bare_axis = || {
        Axis::new()
            .show_grid(false)
            .zero_line(false)
            .show_tick_labels(false)
            .show_line(true)
            .mirror(true)
    };

    let layout = Layout::new()
        .show_legend(false)
        .width(500)
        .height(500)
        .x_axis(bare_axis())
        .y_axis(
            bare_axis()
                .scale_anchor("x")
                .scale_ratio(1.0)
                .range(vec![y_max + pad, y_min - pad]),
        );
    plot.set_layout(layout);

    let html = plot.to_html();
    fs::write("./tmp/human.html", html)?;

    Ok(())
}
