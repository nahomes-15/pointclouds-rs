use pointclouds_core::PointCloud;
use pointclouds_spatial::KdTree;
use std::collections::VecDeque;

/// Extracts clusters from a point cloud using Euclidean distance-based
/// connected-component analysis.
///
/// Points whose pairwise distance is less than or equal to `distance_threshold`
/// are considered connected. Each connected component forms a cluster. Only
/// clusters whose size falls within `[min_size, max_size]` are returned.
///
/// The returned clusters are sorted by size (largest first), and the indices
/// within each cluster are sorted in ascending order.
pub fn euclidean_cluster(
    cloud: &PointCloud,
    distance_threshold: f32,
    min_size: usize,
    max_size: usize,
) -> Vec<Vec<usize>> {
    if cloud.is_empty() || distance_threshold <= 0.0 || min_size == 0 {
        return Vec::new();
    }

    let tree = KdTree::build(cloud);
    let n = cloud.len();
    let mut visited = vec![false; n];
    let mut clusters = Vec::new();

    for i in 0..n {
        if visited[i] {
            continue;
        }

        let mut cluster = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(i);
        visited[i] = true;

        while let Some(current) = queue.pop_front() {
            cluster.push(current);

            if cluster.len() > max_size {
                break;
            }

            let query = [cloud.x[current], cloud.y[current], cloud.z[current]];
            let neighbors = tree.radius_search(&query, distance_threshold);

            for neighbor in neighbors {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        if cluster.len() >= min_size && cluster.len() <= max_size {
            cluster.sort_unstable();
            clusters.push(cluster);
        }
    }

    // Sort clusters by size, largest first
    clusters.sort_by_key(|c| std::cmp::Reverse(c.len()));
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use pointclouds_core::PointCloud;
    use proptest::prelude::*;
    use std::collections::HashSet;

    #[test]
    fn two_separated_clusters() {
        // Cluster A: points around (0,0,0)
        // Cluster B: points around (100,100,100)
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, 0.2, 100.0, 100.1, 100.2],
            vec![0.0, 0.1, 0.0, 100.0, 100.1, 100.0],
            vec![0.0, 0.0, 0.1, 100.0, 100.0, 100.1],
        );

        let clusters = euclidean_cluster(&cloud, 1.0, 1, 100);
        assert_eq!(clusters.len(), 2);
        // Both clusters should have 3 points each
        assert_eq!(clusters[0].len(), 3);
        assert_eq!(clusters[1].len(), 3);

        // Verify the two clusters contain non-overlapping indices
        let set_a: HashSet<usize> = clusters[0].iter().copied().collect();
        let set_b: HashSet<usize> = clusters[1].iter().copied().collect();
        assert!(set_a.is_disjoint(&set_b));

        // Verify all 6 indices are covered
        let all: HashSet<usize> = set_a.union(&set_b).copied().collect();
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn single_dense_cluster() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, 0.2, 0.3],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        );

        let clusters = euclidean_cluster(&cloud, 0.5, 1, 100);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 4);
        assert_eq!(clusters[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn empty_cloud() {
        let cloud = PointCloud::new();
        let clusters = euclidean_cluster(&cloud, 1.0, 1, 100);
        assert!(clusters.is_empty());
    }

    #[test]
    fn min_size_filter() {
        // 2 points close together + 1 isolated point
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, 50.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        );

        // min_size=2 should exclude the isolated point's cluster of size 1
        let clusters = euclidean_cluster(&cloud, 1.0, 2, 100);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 2);
        assert_eq!(clusters[0], vec![0, 1]);
    }

    #[test]
    fn max_size_filter() {
        // 4 points close together
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, 0.2, 0.3],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        );

        // max_size=2 should exclude the cluster of size 4
        let clusters = euclidean_cluster(&cloud, 1.0, 1, 2);
        assert!(clusters.is_empty());
    }

    #[test]
    fn clusters_sorted_largest_first() {
        // Group A: 2 points around (0,0,0)
        // Group B: 3 points around (50,0,0)
        // Group C: 1 point at (100,0,0)
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, 50.0, 50.1, 50.2, 100.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );

        let clusters = euclidean_cluster(&cloud, 1.0, 1, 100);
        assert_eq!(clusters.len(), 3);
        assert_eq!(clusters[0].len(), 3); // Group B
        assert_eq!(clusters[1].len(), 2); // Group A
        assert_eq!(clusters[2].len(), 1); // Group C
    }

    #[test]
    fn zero_distance_threshold_returns_empty() {
        let cloud = PointCloud::from_xyz(vec![0.0], vec![0.0], vec![0.0]);
        let clusters = euclidean_cluster(&cloud, 0.0, 1, 100);
        assert!(clusters.is_empty());
    }

    #[test]
    fn negative_distance_threshold_returns_empty() {
        let cloud = PointCloud::from_xyz(vec![0.0], vec![0.0], vec![0.0]);
        let clusters = euclidean_cluster(&cloud, -1.0, 1, 100);
        assert!(clusters.is_empty());
    }

    #[test]
    fn zero_min_size_returns_empty() {
        let cloud = PointCloud::from_xyz(vec![0.0], vec![0.0], vec![0.0]);
        let clusters = euclidean_cluster(&cloud, 1.0, 0, 100);
        assert!(clusters.is_empty());
    }

    #[test]
    fn indices_within_each_cluster_are_sorted() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, 0.2, 50.0, 50.1],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
        );

        let clusters = euclidean_cluster(&cloud, 1.0, 1, 100);
        for cluster in &clusters {
            for window in cluster.windows(2) {
                assert!(window[0] < window[1]);
            }
        }
    }

    proptest! {
        #[test]
        fn cluster_indices_are_valid(
            pts in prop::collection::vec(
                (-100.0f32..100.0, -100.0f32..100.0, -100.0f32..100.0),
                1..50
            ),
            threshold in 0.1f32..10.0,
        ) {
            let n = pts.len();
            let cloud = PointCloud::from_xyz(
                pts.iter().map(|p| p.0).collect(),
                pts.iter().map(|p| p.1).collect(),
                pts.iter().map(|p| p.2).collect(),
            );

            let clusters = euclidean_cluster(&cloud, threshold, 1, n);
            for cluster in &clusters {
                for &idx in cluster {
                    prop_assert!(idx < n, "Index {} out of bounds (n={})", idx, n);
                }
            }
        }

        #[test]
        fn cluster_indices_are_unique(
            pts in prop::collection::vec(
                (-100.0f32..100.0, -100.0f32..100.0, -100.0f32..100.0),
                1..50
            ),
            threshold in 0.1f32..10.0,
        ) {
            let n = pts.len();
            let cloud = PointCloud::from_xyz(
                pts.iter().map(|p| p.0).collect(),
                pts.iter().map(|p| p.1).collect(),
                pts.iter().map(|p| p.2).collect(),
            );

            let clusters = euclidean_cluster(&cloud, threshold, 1, n);
            let mut all_indices: Vec<usize> = clusters.into_iter().flatten().collect();
            let total = all_indices.len();
            all_indices.sort_unstable();
            all_indices.dedup();
            prop_assert_eq!(all_indices.len(), total, "Duplicate indices found across clusters");
        }
    }
}
