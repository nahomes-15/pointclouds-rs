use hashbrown::HashMap;
use pointclouds_core::PointCloud;
use rayon::prelude::*;

// ────────────────── Union-Find ──────────────────

struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            // Path splitting: point to grandparent
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = self.parent[x as usize];
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // Union by rank
        match self.rank[ra as usize].cmp(&self.rank[rb as usize]) {
            std::cmp::Ordering::Less => self.parent[ra as usize] = rb,
            std::cmp::Ordering::Greater => self.parent[rb as usize] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb as usize] = ra;
                self.rank[ra as usize] += 1;
            }
        }
    }
}

// ────────────────── Grid cell key ──────────────────

type CellKey = (i32, i32, i32);

#[inline]
fn cell_key(x: f32, y: f32, z: f32, inv_r: f32) -> CellKey {
    (
        (x * inv_r).floor() as i32,
        (y * inv_r).floor() as i32,
        (z * inv_r).floor() as i32,
    )
}

/// Half-neighborhood offsets (14 of 27): self-cell + 13 forward neighbors.
/// This avoids checking each pair twice.
const HALF_OFFSETS: [(i32, i32, i32); 14] = [
    // self
    (0, 0, 0),
    // 13 "forward" neighbors (lexicographic order, skip negatives of these)
    (1, 0, 0),
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (1, -1, -1),
    (0, 1, 0),
    (0, 1, 1),
    (0, 1, -1),
    (0, 0, 1),
];

// ────────────────── Public API ──────────────────

/// Extracts clusters from a point cloud using Euclidean distance-based
/// connected-component analysis.
///
/// Points whose pairwise distance is less than or equal to `distance_threshold`
/// are considered connected. Each connected component forms a cluster. Only
/// clusters whose size falls within `[min_size, max_size]` are returned.
///
/// The returned clusters are sorted by size (largest first), and the indices
/// within each cluster are sorted in ascending order.
///
/// Uses grid-based spatial hashing with union-find for O(n) average-case
/// performance, parallelized with rayon for candidate pair generation.
pub fn euclidean_cluster(
    cloud: &PointCloud,
    distance_threshold: f32,
    min_size: usize,
    max_size: usize,
) -> Vec<Vec<usize>> {
    if cloud.is_empty() || distance_threshold <= 0.0 || min_size == 0 {
        return Vec::new();
    }

    let n = cloud.len();
    let inv_r = 1.0 / distance_threshold;
    let r2 = distance_threshold * distance_threshold;

    // Step 1: Assign points to grid cells, skipping non-finite points.
    let mut grid: HashMap<CellKey, Vec<u32>> = HashMap::new();
    for i in 0..n {
        let (x, y, z) = (cloud.x[i], cloud.y[i], cloud.z[i]);
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            continue;
        }
        let key = cell_key(x, y, z, inv_r);
        grid.entry(key).or_default().push(i as u32);
    }

    // Step 2: Collect cell keys for parallel iteration.
    let cells: Vec<CellKey> = grid.keys().copied().collect();

    // Step 3: For each cell, compare against half-neighborhood and collect
    // union pairs. Process in parallel chunks, collect edges per chunk.
    let edge_chunks: Vec<Vec<(u32, u32)>> = cells
        .par_iter()
        .map(|&(cx, cy, cz)| {
            let mut edges = Vec::new();
            let cell_a = &grid[&(cx, cy, cz)];

            for &(dx, dy, dz) in &HALF_OFFSETS {
                let neighbor_key = (cx + dx, cy + dy, cz + dz);
                let cell_b = match grid.get(&neighbor_key) {
                    Some(b) => b,
                    None => continue,
                };

                let same_cell = dx == 0 && dy == 0 && dz == 0;

                for (ai, &idx_a) in cell_a.iter().enumerate() {
                    let ax = cloud.x[idx_a as usize];
                    let ay = cloud.y[idx_a as usize];
                    let az = cloud.z[idx_a as usize];

                    let start = if same_cell { ai + 1 } else { 0 };
                    for &idx_b in &cell_b[start..] {
                        let dx = ax - cloud.x[idx_b as usize];
                        let dy = ay - cloud.y[idx_b as usize];
                        let dz = az - cloud.z[idx_b as usize];
                        if dx * dx + dy * dy + dz * dz <= r2 {
                            edges.push((idx_a, idx_b));
                        }
                    }
                }
            }
            edges
        })
        .collect();

    // Step 4: Union all edges sequentially (union-find is nearly O(1) amortized).
    let mut uf = UnionFind::new(n);
    for chunk in &edge_chunks {
        for &(a, b) in chunk {
            uf.union(a, b);
        }
    }

    // Step 5: Extract connected components.
    let mut components: HashMap<u32, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = uf.find(i as u32);
        components.entry(root).or_default().push(i);
    }

    // Step 6: Filter by size, sort indices within each cluster, sort by size.
    let mut clusters: Vec<Vec<usize>> = components
        .into_values()
        .filter(|c| c.len() >= min_size && c.len() <= max_size)
        .collect();

    clusters.par_iter_mut().for_each(|c| c.sort_unstable());
    // Sort by size descending, break ties by smallest index ascending
    // for deterministic output regardless of rayon scheduling order.
    clusters.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
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
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, 0.2, 100.0, 100.1, 100.2],
            vec![0.0, 0.1, 0.0, 100.0, 100.1, 100.0],
            vec![0.0, 0.0, 0.1, 100.0, 100.0, 100.1],
        );

        let clusters = euclidean_cluster(&cloud, 1.0, 1, 100);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].len(), 3);
        assert_eq!(clusters[1].len(), 3);

        let set_a: HashSet<usize> = clusters[0].iter().copied().collect();
        let set_b: HashSet<usize> = clusters[1].iter().copied().collect();
        assert!(set_a.is_disjoint(&set_b));

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
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, 50.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        );

        let clusters = euclidean_cluster(&cloud, 1.0, 2, 100);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 2);
        assert_eq!(clusters[0], vec![0, 1]);
    }

    #[test]
    fn max_size_filter() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, 0.2, 0.3],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        );

        let clusters = euclidean_cluster(&cloud, 1.0, 1, 2);
        assert!(clusters.is_empty());
    }

    #[test]
    fn clusters_sorted_largest_first() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, 50.0, 50.1, 50.2, 100.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );

        let clusters = euclidean_cluster(&cloud, 1.0, 1, 100);
        assert_eq!(clusters.len(), 3);
        assert_eq!(clusters[0].len(), 3);
        assert_eq!(clusters[1].len(), 2);
        assert_eq!(clusters[2].len(), 1);
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

    /// Transitivity: if A is near B and B is near C, all three must be in the
    /// same cluster even if A and C are > distance_threshold apart.
    #[test]
    fn transitive_connectivity() {
        // Three points in a chain: 0--0.4--0.4 with threshold 0.5
        // dist(0, 2) = 0.8 > 0.5, but they're connected via point 1
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.4, 0.8],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        );
        let clusters = euclidean_cluster(&cloud, 0.5, 1, 100);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    /// Points that span multiple grid cells but are still within threshold
    /// must be connected.
    #[test]
    fn cross_cell_boundary() {
        // Two points just under threshold apart, straddling a cell boundary
        // With r=1.0, cell boundary at x=1.0; points at 0.99 and 1.01
        let cloud = PointCloud::from_xyz(vec![0.99, 1.01], vec![0.0, 0.0], vec![0.0, 0.0]);
        let clusters = euclidean_cluster(&cloud, 1.0, 1, 100);
        assert_eq!(
            clusters.len(),
            1,
            "cross-cell points should be in same cluster"
        );
        assert_eq!(clusters[0].len(), 2);
    }

    /// Dense stress test: many points in a small area should not OOM or panic.
    #[test]
    fn dense_cell_stress() {
        let n = 5000;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let y = vec![0.0; n];
        let z = vec![0.0; n];
        let cloud = PointCloud::from_xyz(x, y, z);
        let clusters = euclidean_cluster(&cloud, 0.01, 1, n + 1);
        // All points are within a few cells, should form clusters
        assert!(!clusters.is_empty());
        let total: usize = clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total, n);
    }

    /// Non-finite points should be silently excluded, not cause panics.
    #[test]
    fn nan_and_inf_excluded() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.1, f32::NAN, f32::INFINITY, 0.2],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let clusters = euclidean_cluster(&cloud, 0.5, 1, 100);
        // Non-finite points are excluded from grid, so they become
        // singleton components (size 1). The 3 finite close points
        // should form one cluster.
        let big: Vec<_> = clusters.iter().filter(|c| c.len() >= 2).collect();
        assert_eq!(big.len(), 1);
        assert_eq!(big[0].len(), 3);
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

        /// Every point must appear in exactly one component (no lost points).
        #[test]
        fn all_points_assigned(
            pts in prop::collection::vec(
                (-50.0f32..50.0, -50.0f32..50.0, -50.0f32..50.0),
                1..100
            ),
            threshold in 0.1f32..5.0,
        ) {
            let n = pts.len();
            let cloud = PointCloud::from_xyz(
                pts.iter().map(|p| p.0).collect(),
                pts.iter().map(|p| p.1).collect(),
                pts.iter().map(|p| p.2).collect(),
            );

            // min_size=1, max_size=n to capture all clusters
            let clusters = euclidean_cluster(&cloud, threshold, 1, n);
            let total: usize = clusters.iter().map(|c| c.len()).sum();
            prop_assert_eq!(total, n, "Lost {} points", n - total);
        }
    }
}
