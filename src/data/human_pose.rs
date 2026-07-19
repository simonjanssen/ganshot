use anyhow::Error;
use rand::seq::SliceRandom;
use std::{fs::File, io::BufReader, iter::FusedIterator, time::Instant};

use serde::Deserialize;

use crate::data::commons::Geometry;

#[derive(Deserialize, Debug, Copy, Clone)]
pub struct BodyJoint {
    id: usize,
    x: f64,
    y: f64,
}

#[derive(Deserialize)]
struct Person {
    points: Option<Vec<BodyJoint>>,
}

#[derive(Deserialize)]
struct Annotation {
    people: Vec<Person>,
}

// --- Geometries ---
mod geometries {
    use crate::data::{
        commons::{Coord2, Geometry},
        human_pose::BodyJoint,
    };

    pub struct HumanPose {
        joints: [BodyJoint; 16],
    }

    impl Geometry for HumanPose {
        const N: usize = 16;
        type Outline = [Coord2; Self::N];

        fn to_outline(&self) -> Self::Outline {
            self.joints.map(|j| (j.x, j.y))
        }
    }

    impl HumanPose {
        pub fn new(joints: [BodyJoint; 16]) -> Self {
            Self { joints }
        }
    }
}

pub use geometries::HumanPose;

// --- Distributions ---
pub struct HumanPoses {
    poses: Vec<HumanPose>,
}

impl HumanPoses {
    pub fn from_annotations_json(path: &str) -> Result<Self, Error> {
        let t0 = Instant::now();
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let annotations: Vec<Annotation> = serde_json::from_reader(reader)?;
        println!("Parse annotations: {:?}", t0.elapsed());
        let t0 = Instant::now();
        let mut poses: Vec<HumanPose> = Vec::new();
        let mut invalid = 0u32;
        for annotation in annotations {
            for person in annotation.people {
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
