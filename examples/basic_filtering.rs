use pointclouds_core::PointCloud;
use pointclouds_filters::{passthrough_filter, voxel_downsample};

fn main() {
    // Create a synthetic point cloud: 1000 random-ish points
    let n = 1000;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.731) % 10.0).collect();
    let y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.419) % 10.0).collect();
    let z: Vec<f32> = (0..n).map(|i| (i as f32 * 0.257) % 10.0).collect();
    let cloud = PointCloud::from_xyz(x, y, z);
    println!("Original cloud: {} points", cloud.len());

    // Passthrough filter: keep points where x is in [2.0, 8.0]
    let filtered = passthrough_filter(&cloud, 'x', 2.0, 8.0);
    println!("After passthrough (x in [2.0, 8.0]): {} points", filtered.len());

    // Voxel downsample with voxel size 1.0
    let downsampled = voxel_downsample(&filtered, 1.0);
    println!("After voxel downsample (size=1.0): {} points", downsampled.len());

    // Show bounding box
    let aabb = downsampled.aabb();
    println!(
        "Bounding box: min={:?}, max={:?}",
        aabb.min, aabb.max
    );
}
