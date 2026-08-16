use ganshot::{
    backend::{MyAutodiffBackend, select_device},
    data::human_pose::HumanPoses,
    training,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path_ann = "data/mpii_human_pose_v1_u12_2/annotations.json";
    let human_poses = HumanPoses::from_annotations_json(path_ann)?;
    println!("poses: {}", human_poses.len());

    let device = select_device();

    training::runner::run::<MyAutodiffBackend, _, _>(device, human_poses)?;
    Ok(())
}
