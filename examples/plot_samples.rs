use anyhow::{Error, anyhow};

use ganshot::data::{
    commons::{Geometry, plot_2x3},
    human_pose::HumanPoses,
    stars::RandomStars,
};
use plotly::Plot;

fn plot_samples<S: Iterator<Item = G>, G: Geometry>(samples: S) -> Result<Plot, Error> {
    let poses: Vec<_> = samples.take(6).collect();
    let poses = poses.try_into().map_err(|_err| anyhow!("Not 6 poses!"))?;
    Ok(plot_2x3(&poses))
}

fn main() -> Result<(), Error> {
    // RandomStars dataset
    {
        let samples = RandomStars::new();
        let plot = plot_samples(samples)?;
        plot.write_html("tmp/star.html");
    }

    // HumanPoses dataset
    {
        let path_ann = "data/mpii_human_pose_v1_u12_2/annotations.json";
        let samples = HumanPoses::from_annotations_json(path_ann)?;
        println!("poses: {}", samples.len());
        let plot = plot_samples(samples)?;
        plot.write_html("tmp/human.html");
    }
    Ok(())
}
