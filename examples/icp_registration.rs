use pointclouds_core::PointCloud;
use pointclouds_registration::{icp_point_to_point, apply_transform, IcpParams};

fn main() {
    // Create a simple source cloud: points along x-axis
    let source = PointCloud::from_xyz(
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
    );
    println!("Source: {} points", source.len());

    // Target is source shifted by [1.0, 0.5, 0.0]
    let target = PointCloud::from_xyz(
        source.x.iter().map(|v| v + 1.0).collect(),
        source.y.iter().map(|v| v + 0.5).collect(),
        source.z.clone(),
    );
    println!("Target: {} points (shifted by [1.0, 0.5, 0.0])", target.len());

    // Run ICP
    let params = IcpParams::default();
    let result = icp_point_to_point(&source, &target, &params);
    println!("ICP converged: {}", result.converged);
    println!("ICP iterations: {}", result.num_iterations);
    println!("ICP RMSE: {:.6}", result.rmse);
    println!("ICP fitness: {:.4}", result.fitness);
    println!(
        "Translation: [{:.4}, {:.4}, {:.4}]",
        result.transform.translation[0],
        result.transform.translation[1],
        result.transform.translation[2]
    );

    // Apply transform to source
    let aligned = apply_transform(&source, &result.transform);
    println!("Aligned first point: {:?}", aligned.point(0));
    println!("Target first point: {:?}", target.point(0));
}
