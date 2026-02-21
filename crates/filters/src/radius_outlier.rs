use pointclouds_core::PointCloud;
use pointclouds_spatial::KdTree;

pub fn radius_outlier_removal(cloud: &PointCloud, radius: f32, min_neighbors: usize) -> PointCloud {
    if cloud.is_empty() {
        return PointCloud::new();
    }

    let tree = KdTree::build(cloud);
    let keep: Vec<usize> = (0..cloud.len())
        .filter(|&i| {
            let q = [cloud.x[i], cloud.y[i], cloud.z[i]];
            tree.radius_search(&q, radius).len() >= min_neighbors
        })
        .collect();

    cloud.select(&keep)
}

#[cfg(test)]
mod tests {
    use super::radius_outlier_removal;
    use pointclouds_core::PointCloud;
    use proptest::prelude::*;

    #[test]
    fn radius_outlier_removes_isolated_points() {
        // Three close points plus one far-away isolated point
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, 0.2, 100.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        );
        // radius=0.5 captures the 3 close points as mutual neighbors;
        // min_neighbors=2 means a point needs at least 2 points (including itself)
        // within the radius. The isolated point at x=100 has only itself.
        let result = radius_outlier_removal(&cloud, 0.5, 2);
        assert_eq!(result.len(), 3);
        for i in 0..result.len() {
            assert!(result.x[i] < 1.0, "isolated point should have been removed");
        }
    }

    #[test]
    fn radius_outlier_keeps_dense_cluster() {
        // All points are close together -- none should be removed
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, 0.2, 0.3, 0.4],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
        );
        // radius=1.0 is large enough to capture all; min_neighbors=2
        let result = radius_outlier_removal(&cloud, 1.0, 2);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn radius_outlier_empty_cloud() {
        let cloud = PointCloud::new();
        let result = radius_outlier_removal(&cloud, 1.0, 2);
        assert!(result.is_empty());
    }

    proptest! {
        #[test]
        fn radius_outlier_never_increases_count(
            pts in prop::collection::vec(
                (-100.0f32..100.0f32, -100.0f32..100.0f32, -100.0f32..100.0f32),
                0..300
            ),
            radius in 0.01f32..10.0f32,
            min_neighbors in 1usize..10,
        ) {
            let cloud = PointCloud::from_xyz(
                pts.iter().map(|p| p.0).collect(),
                pts.iter().map(|p| p.1).collect(),
                pts.iter().map(|p| p.2).collect(),
            );
            let result = radius_outlier_removal(&cloud, radius, min_neighbors);
            prop_assert!(result.len() <= cloud.len());
        }
    }
}
