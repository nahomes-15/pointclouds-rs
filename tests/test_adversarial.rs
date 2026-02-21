//! Adversarial edge-case integration tests.
//!
//! These tests probe degenerate, boundary, and pathological inputs across
//! the full crate stack to verify no panics, no infinite loops, and
//! consistent error handling.

use pointclouds_core::PointCloud;

// ────────────────── PointCloud core ──────────────────

#[test]
fn empty_cloud_operations() {
    let cloud = PointCloud::new();
    assert!(cloud.is_empty());
    assert_eq!(cloud.len(), 0);
    assert_eq!(cloud.to_array(), Vec::<f32>::new());
    assert!(cloud.iter_points().next().is_none());

    let aabb = cloud.aabb();
    assert!(aabb.is_empty());

    // select on empty
    let selected = cloud.select(&[]);
    assert!(selected.is_empty());

    // select_inverse on empty
    let inv = cloud.select_inverse(&[]);
    assert!(inv.is_empty());
}

#[test]
fn single_point_cloud() {
    let cloud = PointCloud::from_xyz(vec![42.0], vec![-1.0], vec![0.0]);
    assert_eq!(cloud.len(), 1);
    assert_eq!(cloud.point(0), [42.0, -1.0, 0.0]);

    let aabb = cloud.aabb();
    assert!(aabb.contains(&[42.0, -1.0, 0.0]));

    let selected = cloud.select(&[0]);
    assert_eq!(selected.len(), 1);

    let inv = cloud.select_inverse(&[0]);
    assert!(inv.is_empty());
}

#[test]
fn cloud_with_inf_values() {
    let cloud = PointCloud::from_xyz(
        vec![f32::INFINITY, f32::NEG_INFINITY, 1.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    );
    assert_eq!(cloud.len(), 3);

    // AABB should handle inf gracefully (skip non-finite)
    let aabb = cloud.aabb();
    // The finite point should be contained
    assert!(aabb.contains(&[1.0, 0.0, 0.0]));
}

#[test]
fn cloud_with_nan_values() {
    let cloud = PointCloud::from_xyz(
        vec![f32::NAN, 1.0, 2.0],
        vec![0.0, f32::NAN, 0.0],
        vec![0.0, 0.0, 0.0],
    );
    assert_eq!(cloud.len(), 3);

    // AABB should skip NaN points
    let aabb = cloud.aabb();
    assert!(aabb.contains(&[2.0, 0.0, 0.0]));
}

#[test]
fn from_array_zero_length() {
    let cloud = PointCloud::from_array(&[], 0);
    assert!(cloud.is_empty());
}

// ────────────────── KdTree ──────────────────

#[test]
fn kdtree_single_point() {
    use pointclouds_spatial::KdTree;

    let cloud = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
    let tree = KdTree::build(&cloud);

    // knn with k=1 on single-point tree
    let (idx, dist) = tree.knn(&[1.0, 2.0, 3.0], 1);
    assert_eq!(idx, vec![0]);
    assert!(dist[0] < 1e-6);

    // knn with k=100 on single-point tree
    let (idx, _) = tree.knn(&[0.0, 0.0, 0.0], 100);
    assert_eq!(idx.len(), 1);

    // radius search with huge radius
    let idx = tree.radius_search(&[0.0, 0.0, 0.0], 1e6);
    assert_eq!(idx.len(), 1);
}

#[test]
fn kdtree_inf_query() {
    use pointclouds_spatial::KdTree;

    let cloud = PointCloud::from_xyz(vec![0.0, 1.0], vec![0.0; 2], vec![0.0; 2]);
    let tree = KdTree::build(&cloud);

    // Query with infinity should return empty (not finite)
    let (idx, _) = tree.knn(&[f32::INFINITY, 0.0, 0.0], 1);
    assert!(idx.is_empty());

    let idx = tree.radius_search(&[f32::INFINITY, 0.0, 0.0], 1.0);
    assert!(idx.is_empty());
}

#[test]
fn kdtree_radius_search_zero_radius() {
    use pointclouds_spatial::KdTree;

    let cloud = PointCloud::from_xyz(vec![0.0], vec![0.0], vec![0.0]);
    let tree = KdTree::build(&cloud);

    let idx = tree.radius_search(&[0.0, 0.0, 0.0], 0.0);
    assert!(idx.is_empty(), "zero radius should return empty");
}

#[test]
fn kdtree_radius_search_inf_radius() {
    use pointclouds_spatial::KdTree;

    let cloud = PointCloud::from_xyz(vec![0.0], vec![0.0], vec![0.0]);
    let tree = KdTree::build(&cloud);

    let idx = tree.radius_search(&[0.0, 0.0, 0.0], f32::INFINITY);
    assert!(
        idx.is_empty(),
        "infinite radius should return empty (not finite)"
    );
}

// ────────────────── Filters ──────────────────

#[test]
fn voxel_downsample_single_point() {
    use pointclouds_filters::voxel_downsample;

    let cloud = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
    let result = voxel_downsample(&cloud, 0.5);
    assert_eq!(result.len(), 1);
}

#[test]
fn voxel_downsample_empty_cloud() {
    use pointclouds_filters::voxel_downsample;

    let cloud = PointCloud::new();
    let result = voxel_downsample(&cloud, 1.0);
    assert!(result.is_empty());
}

#[test]
fn passthrough_filter_inverted_range() {
    use pointclouds_filters::passthrough_filter;

    let cloud = PointCloud::from_xyz(vec![1.0, 5.0, 10.0], vec![0.0; 3], vec![0.0; 3]);
    // min > max — should return empty or zero points
    let result = passthrough_filter(&cloud, 'x', 10.0, 1.0);
    assert!(result.is_empty());
}

#[test]
fn sor_on_single_point() {
    use pointclouds_filters::statistical_outlier_removal;

    let cloud = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
    // SOR with k=10 on a 1-point cloud should not panic
    let result = statistical_outlier_removal(&cloud, 10, 1.0);
    assert!(result.len() <= 1);
}

#[test]
fn ror_on_single_point() {
    use pointclouds_filters::radius_outlier_removal;

    let cloud = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
    let result = radius_outlier_removal(&cloud, 1.0, 2);
    // Single point has 0 neighbors within radius, so it should be removed
    assert!(result.is_empty());
}

// ────────────────── Normals ──────────────────

#[test]
fn normals_two_identical_points() {
    use pointclouds_normals::estimate_normals;

    let cloud = PointCloud::from_xyz(vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]);
    // Two identical points — degenerate covariance. Should not panic.
    let normals = estimate_normals(&cloud, 2);
    assert_eq!(normals.nx.len(), 2);
    // Normals may be arbitrary but should be finite
    for i in 0..2 {
        assert!(normals.nx[i].is_finite());
        assert!(normals.ny[i].is_finite());
        assert!(normals.nz[i].is_finite());
    }
}

#[test]
fn normals_k_larger_than_cloud() {
    use pointclouds_normals::estimate_normals;

    let cloud = PointCloud::from_xyz(vec![0.0, 1.0, 2.0], vec![0.0; 3], vec![0.0; 3]);
    let normals = estimate_normals(&cloud, 100);
    assert_eq!(normals.nx.len(), 3);
}

// ────────────────── Registration ──────────────────

#[test]
fn icp_single_point_clouds() {
    use pointclouds_registration::{icp_point_to_point, IcpParams};

    let source = PointCloud::from_xyz(vec![0.0], vec![0.0], vec![0.0]);
    let target = PointCloud::from_xyz(vec![1.0], vec![0.0], vec![0.0]);
    let params = IcpParams::default();
    let result = icp_point_to_point(&source, &target, &params);
    // Should not panic, should produce some result
    assert!(result.rmse.is_finite());
}

#[test]
fn icp_max_iterations_zero() {
    use pointclouds_registration::{icp_point_to_point, IcpParams};

    let cloud = PointCloud::from_xyz(vec![0.0, 1.0], vec![0.0; 2], vec![0.0; 2]);
    let params = IcpParams {
        max_iterations: 0,
        ..IcpParams::default()
    };
    let result = icp_point_to_point(&cloud, &cloud, &params);
    assert_eq!(result.num_iterations, 0);
}

#[test]
fn icp_plane_empty_source_nonempty_target() {
    use pointclouds_core::Normals;
    use pointclouds_registration::{icp_point_to_plane, IcpParams};

    let source = PointCloud::new();
    let target = PointCloud::from_xyz(vec![1.0], vec![0.0], vec![0.0]);
    let normals = Normals {
        nx: vec![0.0],
        ny: vec![0.0],
        nz: vec![1.0],
    };
    let params = IcpParams::default();
    let result = icp_point_to_plane(&source, &target, &normals, &params).unwrap();
    assert!(result.transform.is_identity(1e-6));
    assert!(!result.converged);
}

// ────────────────── Segmentation ──────────────────

#[test]
fn ransac_single_point() {
    use pointclouds_segmentation::ransac_plane;

    let cloud = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
    let (model, inliers) = ransac_plane(&cloud, 0.01, 100);
    // Cannot fit a plane with 1 point — should return empty
    assert!(inliers.is_empty() || model.normal.iter().all(|v| v.is_finite()));
}

#[test]
fn ransac_two_points() {
    use pointclouds_segmentation::ransac_plane;

    let cloud = PointCloud::from_xyz(vec![0.0, 1.0], vec![0.0; 2], vec![0.0; 2]);
    let (model, inliers) = ransac_plane(&cloud, 0.01, 100);
    assert!(inliers.is_empty() || model.normal.iter().all(|v| v.is_finite()));
}

#[test]
fn euclidean_cluster_all_same_point() {
    use pointclouds_segmentation::euclidean_cluster;

    let n = 50;
    let cloud = PointCloud::from_xyz(vec![1.0; n], vec![2.0; n], vec![3.0; n]);
    let clusters = euclidean_cluster(&cloud, 0.5, 1, 1000);
    // All points at the same location = one cluster
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), n);
}

#[test]
fn euclidean_cluster_min_larger_than_cloud() {
    use pointclouds_segmentation::euclidean_cluster;

    let cloud = PointCloud::from_xyz(vec![0.0, 1.0], vec![0.0; 2], vec![0.0; 2]);
    let clusters = euclidean_cluster(&cloud, 0.5, 100, 1000);
    // min_size > cloud size, no clusters should be returned
    assert!(clusters.is_empty());
}

// ────────────────── IO ──────────────────

#[test]
fn read_ply_corrupted_header() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "not_a_ply_file\ngarbage data here\n").unwrap();
    tmp.flush().unwrap();

    let result = pointclouds_io::read_ply(tmp.path());
    assert!(result.is_err());
}

#[test]
fn read_pcd_corrupted_header() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "# this is not a valid PCD\ngarbage\n").unwrap();
    tmp.flush().unwrap();

    let result = pointclouds_io::read_pcd(tmp.path());
    assert!(result.is_err());
}

#[test]
fn read_ply_truncated_binary() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut tmp = NamedTempFile::new().unwrap();
    // Valid header claiming 100 vertices, but no binary data
    write!(
        tmp,
        "ply\nformat binary_little_endian 1.0\nelement vertex 100\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    )
    .unwrap();
    tmp.flush().unwrap();

    let result = pointclouds_io::read_ply(tmp.path());
    assert!(result.is_err(), "truncated binary PLY should fail");
}

// ────────────────── Cross-crate pipeline with degenerate input ──────────────────

#[test]
fn full_pipeline_on_3_points() {
    use pointclouds_filters::voxel_downsample;
    use pointclouds_normals::estimate_normals;
    use pointclouds_segmentation::ransac_plane;

    // Minimal viable cloud for a full pipeline
    let cloud = PointCloud::from_xyz(
        vec![0.0, 1.0, 0.5],
        vec![0.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0],
    );

    // Downsample (should keep all or reduce slightly)
    let ds = voxel_downsample(&cloud, 0.5);
    assert!(ds.len() >= 1);

    // Normals (may be degenerate but should not panic)
    let normals = estimate_normals(&ds, 3);
    assert_eq!(normals.nx.len(), ds.len());

    // RANSAC (3 points define exactly one plane)
    let (model, inliers) = ransac_plane(&cloud, 0.01, 10);
    assert!(model.normal.iter().all(|v| v.is_finite()));
    assert_eq!(inliers.len(), 3);
}

#[test]
fn select_and_select_inverse_are_complements() {
    let cloud = PointCloud::from_xyz(
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        vec![10.0, 11.0, 12.0, 13.0, 14.0],
        vec![20.0, 21.0, 22.0, 23.0, 24.0],
    );

    let indices = vec![1, 3];
    let selected = cloud.select(&indices);
    let inverse = cloud.select_inverse(&indices);

    assert_eq!(selected.len() + inverse.len(), cloud.len());

    // Union of x values should equal original
    let mut all_x: Vec<f32> = selected.x.iter().chain(inverse.x.iter()).copied().collect();
    all_x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut orig_x = cloud.x.clone();
    orig_x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(all_x, orig_x);
}
