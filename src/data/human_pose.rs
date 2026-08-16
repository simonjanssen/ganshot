use rand::seq::SliceRandom;
use std::{fs::File, io::BufReader, iter::FusedIterator, time::Instant};

use serde::Deserialize;

use crate::Result;
use crate::data::commons::{Coord2, Geometry};
use crate::error::LoadDatasetError;

#[derive(Deserialize, Debug, Copy, Clone)]
pub struct BodyJoint {
    id: usize,
    pub x: f64,
    pub y: f64,
}

impl From<(usize, Coord2)> for BodyJoint {
    fn from(value: (usize, Coord2)) -> Self {
        let (id, (x, y)) = value;
        Self { id, x, y }
    }
}

impl From<BodyJoint> for Coord2 {
    fn from(value: BodyJoint) -> Self {
        (value.x, value.y)
    }
}

#[derive(Deserialize)]
struct Person {
    points: Option<Vec<BodyJoint>>,
}

impl Person {
    pub fn flip_y(&mut self) {
        if let Some(joints) = &mut self.points {
            let y_min = joints.iter().map(|j| j.y).fold(f64::INFINITY, f64::min);
            let y_max = joints.iter().map(|j| j.y).fold(f64::NEG_INFINITY, f64::max);
            let y_diff = y_max - y_min;
            for joint in joints.iter_mut() {
                joint.y = y_diff - joint.y;
            }
        }
    }
}

#[derive(Deserialize)]
struct Annotation {
    people: Vec<Person>,
}

// --- Geometries ---
mod geometries {
    use plotly::{Layout, Plot, Scatter, common::Mode, layout::Axis};

    use crate::data::{
        commons::{Coord2, Geometry},
        human_pose::BodyJoint,
    };

    #[derive(Clone, Copy)]
    struct HumanPoseRepr {
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

    impl From<[BodyJoint; 16]> for HumanPoseRepr {
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

    impl Into<[BodyJoint; 16]> for HumanPoseRepr {
        fn into(self) -> [BodyJoint; 16] {
            [
                self.r_ankle,
                self.r_knee,
                self.r_hip,
                self.l_hip,
                self.l_knee,
                self.l_ankle,
                self.pelvis,
                self.thorax,
                self.upper_neck,
                self.head_top,
                self.r_wrist,
                self.r_elbow,
                self.r_shoulder,
                self.l_shoulder,
                self.l_elbow,
                self.l_wrist,
            ]
        }
    }

    pub struct HumanPose {
        repr: Option<HumanPoseRepr>,
    }

    impl Geometry for HumanPose {
        const N: usize = 16;
        type Outline = [Coord2; Self::N];

        fn empty() -> Self {
            Self { repr: None }
        }

        fn set_outline(&mut self, outline: Self::Outline) {
            let joints = std::array::from_fn(|i| BodyJoint::from((i, outline[i])));
            self.repr = Some(joints.into());
        }

        fn to_outline(&self) -> Self::Outline {
            if let Some(repr) = &self.repr {
                let repr = *repr;
                let joints: [BodyJoint; 16] = repr.into();
                joints.map(Into::into)
            } else {
                panic!("Empty pose doesn't have an outline!")
            }
        }

        fn traces(&self) -> Vec<Box<Scatter<f64, f64>>> {
            let Some(pose) = self.repr else {
                panic!("Cannot plot empty human pose!")
            };

            let limb = |label: &str, joints: [BodyJoint; 4]| {
                let xs = joints.iter().map(|j| j.x).collect();
                let ys = joints.iter().map(|j| j.y).collect();
                Scatter::new(xs, ys).mode(Mode::LinesMarkers).name(label)
            };

            vec![
                limb(
                    "right leg",
                    [pose.r_ankle, pose.r_knee, pose.r_hip, pose.pelvis],
                ),
                limb(
                    "left leg",
                    [pose.l_ankle, pose.l_knee, pose.l_hip, pose.pelvis],
                ),
                limb(
                    "spine",
                    [pose.head_top, pose.upper_neck, pose.thorax, pose.pelvis],
                ),
                limb(
                    "left arm",
                    [pose.l_wrist, pose.l_elbow, pose.l_shoulder, pose.thorax],
                ),
                limb(
                    "right arm",
                    [pose.r_wrist, pose.r_elbow, pose.r_shoulder, pose.thorax],
                ),
            ]
        }

        fn plot(&self) -> Plot {
            let mut plot = Plot::new();
            for trace in self.traces() {
                plot.add_trace(trace);
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
            plot.set_layout(
                Layout::new()
                    .show_legend(false)
                    .width(500)
                    .height(500)
                    .x_axis(x_axis)
                    .y_axis(y_axis.scale_anchor("x")),
            );
            plot
        }
    }

    impl HumanPose {
        pub fn new(joints: [BodyJoint; 16]) -> Self {
            Self {
                repr: Some(joints.into()),
            }
        }
    }
}

pub use geometries::HumanPose;

// --- Distributions ---
pub struct HumanPoses {
    poses: Vec<HumanPose>,
}

impl HumanPoses {
    pub fn from_annotations_json(path: &str) -> Result<Self> {
        let t0 = Instant::now();
        let annotations: Vec<Annotation> = {
            let file = File::open(path).map_err(|e| LoadDatasetError::Io {
                path: path.into(),
                source: e,
            })?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader).map_err(|e| LoadDatasetError::Json {
                path: path.into(),
                source: e,
            })
        }?;
        println!("Parse annotations: {:?}", t0.elapsed());

        let t0 = Instant::now();
        let mut poses: Vec<HumanPose> = Vec::new();
        let mut invalid = 0u32;
        for annotation in annotations {
            for mut person in annotation.people {
                person.flip_y();
                if let Some(mut points) = person.points {
                    if points.len() == HumanPose::N {
                        points.sort_by_key(|p| p.id);
                        poses.push(HumanPose::new(points.try_into().unwrap()));
                    } else {
                        invalid += 1
                    }
                }
            }
        }
        poses.shuffle(&mut rand::rng());
        println!("Parse poses: {:?}", t0.elapsed());
        println!("invalid poses: {}", invalid);
        Ok(Self { poses })
    }

    pub fn len(&self) -> usize {
        self.poses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }
}

impl Iterator for HumanPoses {
    type Item = HumanPose;
    fn next(&mut self) -> Option<Self::Item> {
        self.poses.pop()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.poses.len(), Some(self.poses.len()))
    }
}

impl FusedIterator for HumanPoses {}
