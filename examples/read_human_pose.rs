use anyhow::Error;
use burn::data::dataset::Dataset as DatasetExt;
use ganshot::data::{
    commons::{Batcher, Dataset},
    human_pose::HumanPoses,
};

fn main() -> Result<(), Error> {
    let path_ann = "data/mpii_human_pose_v1_u12_2/annotations.json";
    let human_poses = HumanPoses::from_annotations_json(path_ann)?;
    println!("poses: {}", human_poses.len());
    let dataset = Dataset::new(human_poses, 10_000);
    println!("dataset: {}", dataset.len());
    let batcher = Batcher::from(&dataset);
    Ok(())
}
