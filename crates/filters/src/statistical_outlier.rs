use pointclouds_core::PointCloud;
use pointclouds_spatial::KdTree;

pub fn statistical_outlier_removal(cloud: &PointCloud, k: usize, std_mul: f32) -> PointCloud {
    if cloud.is_empty() || k == 0 {
        return PointCloud::new();
    }

    // Single point: no neighbors to compare against, keep it
    if cloud.len() == 1 {
        return cloud.clone();
    }

    let tree = KdTree::build(cloud);

    // For each point, compute the mean distance to its k nearest neighbors.
    // knn returns the query point itself as the nearest (distance 0), so we
    // request k+1 neighbors and skip the first one (self-match).
    let mean_dists: Vec<f32> = (0..cloud.len())
        .map(|i| {
            let q = [cloud.x[i], cloud.y[i], cloud.z[i]];
            if !q[0].is_finite() || !q[1].is_finite() || !q[2].is_finite() {
                return f32::INFINITY;
            }
            let (_, dists) = tree.knn(&q, k + 1);
            // Skip the first neighbor (self at distance 0).
            // If we got fewer than k+1 results, use all non-self results.
            let neighbor_dists = if dists.len() > 1 {
                &dists[1..]
            } else {
                &dists[..]
            };
            if neighbor_dists.is_empty() {
                return f32::INFINITY;
            }
            let sum: f32 = neighbor_dists.iter().sum();
            sum / neighbor_dists.len() as f32
        })
        .collect();

    // Compute global mean and standard deviation of mean distances,
    // considering only finite values.
    let finite_dists: Vec<f32> = mean_dists
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .collect();

    if finite_dists.is_empty() {
        return PointCloud::new();
    }

    let n = finite_dists.len() as f32;
    let global_mean: f32 = finite_dists.iter().sum::<f32>() / n;
    let variance: f32 = finite_dists
        .iter()
        .map(|d| (d - global_mean).powi(2))
        .sum::<f32>()
        / n;
    let global_stddev = variance.sqrt();

    let threshold = global_mean + std_mul * global_stddev;

    let keep: Vec<usize> = (0..cloud.len())
        .filter(|&i| mean_dists[i] <= threshold)
        .collect();

    cloud.select(&keep)
}

#[cfg(test)]
mod tests {
    use super::statistical_outlier_removal;
    use pointclouds_core::PointCloud;
    use proptest::prelude::*;

    #[test]
    fn sor_removes_outliers() {
        // Dense cluster around the origin, plus one far-away outlier
        let mut x = vec![0.0, 0.1, -0.1, 0.05, -0.05];
        let mut y = vec![0.0, 0.1, -0.1, 0.05, -0.05];
        let mut z = vec![0.0, 0.1, -0.1, 0.05, -0.05];
        // Outlier far away
        x.push(100.0);
        y.push(100.0);
        z.push(100.0);

        let cloud = PointCloud::from_xyz(x, y, z);
        // k=4 neighbors, std_mul=1.0 should remove the outlier
        let result = statistical_outlier_removal(&cloud, 4, 1.0);

        // The outlier at (100,100,100) should have been removed
        assert_eq!(result.len(), 5);
        for i in 0..result.len() {
            let p = result.point(i);
            // All remaining points should be near origin
            assert!(p[0].abs() <= 0.2, "unexpected x={}", p[0]);
            assert!(p[1].abs() <= 0.2, "unexpected y={}", p[1]);
            assert!(p[2].abs() <= 0.2, "unexpected z={}", p[2]);
        }
    }

    #[test]
    fn sor_keeps_inliers() {
        // A symmetric grid of points where every point has similar neighbor
        // distances, so nothing should be classified as an outlier.
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for ix in 0..3 {
            for iy in 0..3 {
                for iz in 0..3 {
                    x.push(ix as f32);
                    y.push(iy as f32);
                    z.push(iz as f32);
                }
            }
        }
        let cloud = PointCloud::from_xyz(x, y, z); // 27 points in a 3x3x3 grid
        let result = statistical_outlier_removal(&cloud, 5, 3.0);

        // With a generous std_mul on a symmetric grid, all points survive
        assert_eq!(result.len(), cloud.len());
    }

    #[test]
    fn sor_empty_cloud() {
        let cloud = PointCloud::new();
        let result = statistical_outlier_removal(&cloud, 5, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn sor_single_point() {
        let cloud = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
        let result = statistical_outlier_removal(&cloud, 5, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result.point(0), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn sor_k_zero_returns_empty() {
        let cloud = PointCloud::from_xyz(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]);
        let result = statistical_outlier_removal(&cloud, 0, 1.0);
        assert!(result.is_empty());
    }

    proptest! {
        #[test]
        fn sor_never_increases_count(
            pts in prop::collection::vec(
                (-100.0f32..100.0f32, -100.0f32..100.0f32, -100.0f32..100.0f32),
                0..200
            ),
            k in 1usize..10,
            std_mul in 0.5f32..3.0f32,
        ) {
            let cloud = PointCloud::from_xyz(
                pts.iter().map(|p| p.0).collect(),
                pts.iter().map(|p| p.1).collect(),
                pts.iter().map(|p| p.2).collect(),
            );
            let result = statistical_outlier_removal(&cloud, k, std_mul);
            prop_assert!(result.len() <= cloud.len());
        }
    }
}
