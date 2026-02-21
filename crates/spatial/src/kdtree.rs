use kiddo::float::distance::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use pointclouds_core::PointCloud;
use std::num::NonZero;

/// A KdTree for efficient spatial queries on 3D point clouds.
///
/// Built on top of kiddo v5's `ImmutableKdTree`, which uses a cache-optimized
/// layout for faster queries than the mutable variant. The tree is built once
/// from a slice of points and cannot be modified afterwards.
///
/// The tree stores `u32` indices mapping back to the original PointCloud.
#[derive(Debug, Clone)]
pub struct KdTree {
    tree: ImmutableKdTree<f32, u32, 3, 32>,
    num_points: usize,
}

impl KdTree {
    /// Build a KdTree from a PointCloud.
    ///
    /// Points are extracted into a contiguous `[[f32; 3]]` slice and fed to
    /// kiddo's `ImmutableKdTree` constructor, which builds a balanced,
    /// cache-optimized tree in one pass.
    pub fn build(cloud: &PointCloud) -> Self {
        let n = cloud.len();
        if n == 0 {
            return Self {
                tree: ImmutableKdTree::new_from_slice(&[]),
                num_points: 0,
            };
        }

        let points: Vec<[f32; 3]> = (0..n)
            .map(|i| [cloud.x[i], cloud.y[i], cloud.z[i]])
            .collect();

        let tree = ImmutableKdTree::new_from_slice(&points);

        Self {
            tree,
            num_points: n,
        }
    }

    /// Returns the number of points in the tree.
    pub fn len(&self) -> usize {
        self.num_points
    }

    /// Returns true if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.num_points == 0
    }

    /// Find the `k` nearest neighbours to `query`.
    ///
    /// Returns `(indices, distances)` where distances are **Euclidean**
    /// (not squared), sorted in ascending order by distance.
    ///
    /// Edge cases:
    /// - Returns empty if `k == 0`, cloud is empty, or query contains NaN.
    /// - If `k > len()`, returns all points.
    pub fn knn(&self, query: &[f32; 3], k: usize) -> (Vec<usize>, Vec<f32>) {
        if k == 0 || self.is_empty() || !query.iter().all(|v| v.is_finite()) {
            return (Vec::new(), Vec::new());
        }

        let nz_k = NonZero::new(k).unwrap();
        let results = self.tree.nearest_n::<SquaredEuclidean>(query, nz_k);

        let mut indices = Vec::with_capacity(results.len());
        let mut distances = Vec::with_capacity(results.len());
        for nn in results {
            indices.push(nn.item as usize);
            distances.push(nn.distance.sqrt());
        }

        (indices, distances)
    }

    /// Find the `k` nearest neighbours to `query`, returning only indices.
    ///
    /// This is faster than [`knn`] when distances are not needed (e.g.,
    /// normal estimation) because it skips the sqrt computation and
    /// distance vector allocation.
    pub fn knn_indices(&self, query: &[f32; 3], k: usize) -> Vec<usize> {
        if k == 0 || self.is_empty() || !query.iter().all(|v| v.is_finite()) {
            return Vec::new();
        }

        let nz_k = NonZero::new(k).unwrap();
        let results = self.tree.nearest_n::<SquaredEuclidean>(query, nz_k);

        results.iter().map(|nn| nn.item as usize).collect()
    }

    /// Find all points within `radius` (Euclidean distance) of `query`.
    ///
    /// Returns indices of points where `euclidean_dist <= radius`.
    ///
    /// Edge cases:
    /// - Returns empty if radius <= 0, cloud is empty, radius is non-finite,
    ///   or query contains NaN.
    pub fn radius_search(&self, query: &[f32; 3], radius: f32) -> Vec<usize> {
        if self.is_empty()
            || radius <= 0.0
            || !radius.is_finite()
            || !query.iter().all(|v| v.is_finite())
        {
            return Vec::new();
        }

        let radius_sq = radius * radius;

        // kiddo's `within_unsorted` uses strict `<`. To include points
        // exactly on the boundary (dist == radius), we query with a tiny
        // epsilon added, then post-filter with `<=`.
        let query_radius_sq = radius_sq + f32::EPSILON * radius_sq.max(1.0);

        let results = self
            .tree
            .within_unsorted::<SquaredEuclidean>(query, query_radius_sq);

        let mut indices: Vec<usize> = results
            .into_iter()
            .filter(|nn| nn.distance <= radius_sq)
            .map(|nn| nn.item as usize)
            .collect();

        // Sort by index for deterministic output order
        indices.sort_unstable();

        indices
    }
}

#[cfg(test)]
mod tests {
    use super::KdTree;
    use pointclouds_core::PointCloud;
    use proptest::prelude::*;

    #[test]
    fn knn_returns_expected_neighbors() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, 1.0, 2.0, 10.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        );
        let tree = KdTree::build(&cloud);
        let (idx, dist) = tree.knn(&[0.2, 0.0, 0.0], 2);
        assert_eq!(idx, vec![0, 1]);
        assert!(dist[0] <= dist[1]);
    }

    #[test]
    fn radius_search_finds_points() {
        let cloud = PointCloud::from_xyz(vec![0.0, 0.5, 2.0], vec![0.0; 3], vec![0.0; 3]);
        let tree = KdTree::build(&cloud);
        let mut idx = tree.radius_search(&[0.0, 0.0, 0.0], 0.75);
        idx.sort();
        assert_eq!(idx, vec![0, 1]);
    }

    #[test]
    fn knn_empty_cloud() {
        let cloud = PointCloud::new();
        let tree = KdTree::build(&cloud);
        let (idx, dist) = tree.knn(&[0.0, 0.0, 0.0], 5);
        assert!(idx.is_empty());
        assert!(dist.is_empty());
    }

    #[test]
    fn knn_k_zero() {
        let cloud = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
        let tree = KdTree::build(&cloud);
        let (idx, dist) = tree.knn(&[0.0, 0.0, 0.0], 0);
        assert!(idx.is_empty());
        assert!(dist.is_empty());
    }

    #[test]
    fn knn_nan_query() {
        let cloud = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
        let tree = KdTree::build(&cloud);
        let (idx, dist) = tree.knn(&[f32::NAN, 0.0, 0.0], 1);
        assert!(idx.is_empty());
        assert!(dist.is_empty());
    }

    #[test]
    fn radius_search_empty_cloud() {
        let cloud = PointCloud::new();
        let tree = KdTree::build(&cloud);
        let idx = tree.radius_search(&[0.0, 0.0, 0.0], 10.0);
        assert!(idx.is_empty());
    }

    #[test]
    fn radius_search_negative_radius() {
        let cloud = PointCloud::from_xyz(vec![0.0], vec![0.0], vec![0.0]);
        let tree = KdTree::build(&cloud);
        let idx = tree.radius_search(&[0.0, 0.0, 0.0], -1.0);
        assert!(idx.is_empty());
    }

    #[test]
    fn knn_k_larger_than_cloud() {
        let cloud = PointCloud::from_xyz(vec![0.0, 1.0, 2.0], vec![0.0; 3], vec![0.0; 3]);
        let tree = KdTree::build(&cloud);
        let (idx, _dist) = tree.knn(&[0.0, 0.0, 0.0], 100);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn knn_distances_are_sorted() {
        let cloud = PointCloud::from_xyz(vec![0.0, 3.0, 1.0, 7.0, 2.0], vec![0.0; 5], vec![0.0; 5]);
        let tree = KdTree::build(&cloud);
        let (_idx, dist) = tree.knn(&[0.5, 0.0, 0.0], 5);
        for w in dist.windows(2) {
            assert!(w[0] <= w[1], "distances not sorted: {:?}", dist);
        }
    }

    #[test]
    fn radius_search_exact_boundary() {
        // Place a point at exactly distance 1.0 from the origin
        let cloud = PointCloud::from_xyz(vec![1.0, 5.0], vec![0.0; 2], vec![0.0; 2]);
        let tree = KdTree::build(&cloud);
        let idx = tree.radius_search(&[0.0, 0.0, 0.0], 1.0);
        // Point at distance exactly 1.0 should be included (<=)
        assert!(
            idx.contains(&0),
            "point at exact boundary should be included, got {:?}",
            idx
        );
        // Point at distance 5.0 should NOT be included
        assert!(!idx.contains(&1));
    }

    proptest! {
        #[test]
        fn knn_returns_at_most_k_results(
            pts in prop::collection::vec(
                (-100.0f32..100.0f32, -100.0f32..100.0f32, -100.0f32..100.0f32),
                1..200
            ),
            k in 1usize..50,
        ) {
            let cloud = PointCloud::from_xyz(
                pts.iter().map(|p| p.0).collect(),
                pts.iter().map(|p| p.1).collect(),
                pts.iter().map(|p| p.2).collect(),
            );
            let tree = KdTree::build(&cloud);
            let (idx, dist) = tree.knn(&[0.0, 0.0, 0.0], k);
            prop_assert!(idx.len() <= k);
            prop_assert!(idx.len() <= pts.len());
            prop_assert_eq!(idx.len(), dist.len());
        }

        #[test]
        fn radius_search_results_are_within_radius(
            pts in prop::collection::vec(
                (-100.0f32..100.0f32, -100.0f32..100.0f32, -100.0f32..100.0f32),
                1..200
            ),
            radius in 0.1f32..50.0f32,
        ) {
            let cloud = PointCloud::from_xyz(
                pts.iter().map(|p| p.0).collect(),
                pts.iter().map(|p| p.1).collect(),
                pts.iter().map(|p| p.2).collect(),
            );
            let tree = KdTree::build(&cloud);
            let idx = tree.radius_search(&[0.0, 0.0, 0.0], radius);
            for &i in &idx {
                let dx = pts[i].0;
                let dy = pts[i].1;
                let dz = pts[i].2;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                prop_assert!(
                    dist <= radius + f32::EPSILON * 10.0,
                    "point {} at dist {} exceeds radius {}",
                    i,
                    dist,
                    radius,
                );
            }
        }
    }
}
