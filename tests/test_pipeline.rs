use pointclouds_core::PointCloud;
use pointclouds_filters::{passthrough_filter, voxel_downsample};
use pointclouds_normals::estimate_normals;
use pointclouds_segmentation::ransac_plane;

/// End-to-end pipeline: load → filter → downsample → estimate normals → segment
#[test]
fn pipeline_filter_downsample_normals_segment() {
    // Create a synthetic scene: a ground plane + an elevated cluster
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();

    // Ground plane at z ≈ 0 (100 points on a 10×10 grid)
    // Small z variation avoids kiddo bucket overflow on coplanar points
    for i in 0..10 {
        for j in 0..10 {
            x.push(i as f32 * 0.1);
            y.push(j as f32 * 0.1);
            z.push((i * 10 + j) as f32 * 1e-5);
        }
    }

    // Elevated cluster at z ≈ 5 (25 points)
    for i in 0..5 {
        for j in 0..5 {
            x.push(i as f32 * 0.05 + 0.3);
            y.push(j as f32 * 0.05 + 0.3);
            z.push(5.0 + (i * 5 + j) as f32 * 1e-5);
        }
    }

    let cloud = PointCloud::from_xyz(x, y, z);
    assert_eq!(cloud.len(), 125);

    // Step 1: Passthrough filter on z axis to keep ground plane only
    let ground = passthrough_filter(&cloud, 'z', -0.5, 0.5);
    assert_eq!(ground.len(), 100);

    // Step 2: Voxel downsample
    let downsampled = voxel_downsample(&ground, 0.15);
    assert!(downsampled.len() > 0);
    assert!(downsampled.len() < ground.len());

    // Step 3: Normal estimation (should produce normals for all points)
    let normals = estimate_normals(&downsampled, 5);
    assert_eq!(normals.nx.len(), downsampled.len());

    // For a flat plane at z=0, normals should be approximately (0, 0, ±1)
    for i in 0..normals.nx.len() {
        let nz_abs = normals.nz[i].abs();
        assert!(
            nz_abs > 0.8,
            "normal z component should be close to ±1 for flat plane, got {}",
            nz_abs
        );
    }

    // Step 4: RANSAC on the full cloud should find the dominant plane
    let (model, inliers) = ransac_plane(&cloud, 0.1, 100);
    // The dominant plane should be the ground plane (z=0), so normal ≈ (0,0,1)
    assert!(
        model.normal[2].abs() > 0.9,
        "RANSAC should find z-plane, got normal {:?}",
        model.normal
    );
    // Should find approximately 100 ground-plane inliers
    assert!(
        inliers.len() >= 80,
        "expected >=80 inliers, got {}",
        inliers.len()
    );
}

#[test]
fn pipeline_voxel_then_normals() {
    // Simple test: create cloud, downsample, compute normals
    let cloud = PointCloud::from_xyz(
        (0..1000).map(|i| (i % 100) as f32 * 0.01).collect(),
        (0..1000).map(|i| (i / 100) as f32 * 0.01).collect(),
        (0..1000).map(|i| i as f32 * 1e-6).collect(),
    );

    let ds = voxel_downsample(&cloud, 0.05);
    assert!(ds.len() < cloud.len());

    let normals = estimate_normals(&ds, 5);
    assert_eq!(normals.nx.len(), ds.len());

    // All normals should be unit length
    for i in 0..normals.nx.len() {
        let len = (normals.nx[i].powi(2) + normals.ny[i].powi(2) + normals.nz[i].powi(2)).sqrt();
        assert!(
            (len - 1.0).abs() < 0.1,
            "normal should be unit length, got {}",
            len
        );
    }
}
