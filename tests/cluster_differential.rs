//! Differential correctness tests for the grid-based union-find clustering.
//!
//! Compares the optimized `euclidean_cluster` against a brute-force reference
//! implementation to catch any silent regressions from the algorithm rewrite.

use pointclouds_core::PointCloud;
use pointclouds_segmentation::euclidean_cluster;
use rand::prelude::*;

// ────────────────── Brute-force reference ──────────────────

/// O(n^2) brute-force connected-components clustering for correctness reference.
fn brute_force_cluster(
    cloud: &PointCloud,
    distance_threshold: f32,
    min_size: usize,
    max_size: usize,
) -> Vec<Vec<usize>> {
    let n = cloud.len();
    if n == 0 || distance_threshold <= 0.0 || min_size == 0 {
        return Vec::new();
    }

    let r2 = distance_threshold * distance_threshold;

    // Build adjacency via O(n^2) pairwise distance check
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    for i in 0..n {
        let (xi, yi, zi) = (cloud.x[i], cloud.y[i], cloud.z[i]);
        if !xi.is_finite() || !yi.is_finite() || !zi.is_finite() {
            continue;
        }
        for j in (i + 1)..n {
            let (xj, yj, zj) = (cloud.x[j], cloud.y[j], cloud.z[j]);
            if !xj.is_finite() || !yj.is_finite() || !zj.is_finite() {
                continue;
            }
            let dx = xi - xj;
            let dy = yi - yj;
            let dz = zi - zj;
            if dx * dx + dy * dy + dz * dz <= r2 {
                union(&mut parent, i, j);
            }
        }
    }

    // Extract components
    let mut components: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        components.entry(root).or_default().push(i);
    }

    let mut clusters: Vec<Vec<usize>> = components
        .into_values()
        .filter(|c| c.len() >= min_size && c.len() <= max_size)
        .collect();

    for c in &mut clusters {
        c.sort_unstable();
    }
    clusters.sort_by_key(|c| std::cmp::Reverse(c.len()));
    clusters
}

/// Normalize cluster output for comparison: sort indices within each cluster,
/// then sort clusters by (size desc, first index asc) for deterministic ordering
/// when sizes are equal.
fn normalize(mut clusters: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    for c in &mut clusters {
        c.sort_unstable();
    }
    clusters.sort_by(|a, b| {
        b.len()
            .cmp(&a.len())
            .then_with(|| a.first().cmp(&b.first()))
    });
    clusters
}

// ────────────────── 1. Differential correctness ──────────────────

#[test]
fn differential_random_small_clouds() {
    let mut rng = StdRng::seed_from_u64(42);

    for trial in 0..200 {
        let n = rng.gen_range(2..80);
        let threshold = rng.gen_range(0.5f32..5.0);

        let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-20.0..20.0)).collect();
        let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-20.0..20.0)).collect();
        let z: Vec<f32> = (0..n).map(|_| rng.gen_range(-20.0..20.0)).collect();
        let cloud = PointCloud::from_xyz(x, y, z);

        let got = normalize(euclidean_cluster(&cloud, threshold, 1, n));
        let expected = normalize(brute_force_cluster(&cloud, threshold, 1, n));

        assert_eq!(
            got, expected,
            "trial {}: n={}, threshold={:.2} — clusters differ",
            trial, n, threshold
        );
    }
}

#[test]
fn differential_medium_clouds() {
    let mut rng = StdRng::seed_from_u64(99);

    for trial in 0..20 {
        let n = rng.gen_range(500..2000);
        let threshold = rng.gen_range(1.0f32..8.0);

        let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-50.0..50.0)).collect();
        let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-50.0..50.0)).collect();
        let z: Vec<f32> = (0..n).map(|_| rng.gen_range(-50.0..50.0)).collect();
        let cloud = PointCloud::from_xyz(x, y, z);

        let got = normalize(euclidean_cluster(&cloud, threshold, 1, n));
        let expected = normalize(brute_force_cluster(&cloud, threshold, 1, n));

        assert_eq!(
            got, expected,
            "trial {}: n={}, threshold={:.2} — clusters differ",
            trial, n, threshold
        );
    }
}

// ────────────────── 2. Boundary / precision tests ──────────────────

#[test]
fn points_exactly_at_threshold() {
    // Two points exactly distance_threshold apart should be connected (<=)
    let cloud = PointCloud::from_xyz(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let clusters = euclidean_cluster(&cloud, 1.0, 1, 100);
    let norm = normalize(clusters);
    assert_eq!(
        norm.len(),
        1,
        "points at exact threshold should be connected"
    );
    assert_eq!(norm[0], vec![0, 1]);
}

#[test]
fn points_just_beyond_threshold() {
    // Two points just beyond threshold should NOT be connected
    let d = 1.0f32 + 1e-4;
    let cloud = PointCloud::from_xyz(vec![0.0, d], vec![0.0, 0.0], vec![0.0, 0.0]);
    let clusters = euclidean_cluster(&cloud, 1.0, 1, 100);
    assert_eq!(
        clusters.len(),
        2,
        "points beyond threshold should be separate"
    );
}

#[test]
fn points_on_cell_boundaries() {
    // Points at exact cell boundary: x = k * threshold
    // Place points at 0.999999, 1.000001 (straddling cell boundary at 1.0)
    let eps = 1e-5_f32;
    let cloud = PointCloud::from_xyz(vec![1.0 - eps, 1.0 + eps], vec![0.0, 0.0], vec![0.0, 0.0]);
    let clusters = euclidean_cluster(&cloud, 1.0, 1, 100);
    assert_eq!(
        clusters.len(),
        1,
        "points straddling cell boundary should be connected"
    );
}

#[test]
fn very_large_coordinates() {
    // Points at large coordinates should still cluster correctly
    let base = 1e6_f32;
    let cloud = PointCloud::from_xyz(
        vec![base, base + 0.1, base + 0.2, base + 100.0],
        vec![base, base, base, base],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let clusters = euclidean_cluster(&cloud, 0.5, 1, 100);
    assert_eq!(clusters.len(), 2);
    assert_eq!(clusters[0].len(), 3);
    assert_eq!(clusters[1].len(), 1);
}

#[test]
fn very_small_threshold() {
    // With tiny threshold, each point should be its own cluster
    let cloud = PointCloud::from_xyz(
        vec![0.0, 1.0, 2.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    );
    let clusters = euclidean_cluster(&cloud, 1e-6, 1, 100);
    assert_eq!(clusters.len(), 3);
    for c in &clusters {
        assert_eq!(c.len(), 1);
    }
}

// ────────────────── 3. Metamorphic invariants ──────────────────

#[test]
fn shuffled_cloud_same_membership() {
    let mut rng = StdRng::seed_from_u64(55);
    let n = 200;
    let threshold = 2.0;

    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let z: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let cloud = PointCloud::from_xyz(x.clone(), y.clone(), z.clone());

    let original_clusters = euclidean_cluster(&cloud, threshold, 1, n);

    // Build a point-to-cluster-id mapping from original
    let mut membership_orig = vec![usize::MAX; n];
    for (cid, cluster) in original_clusters.iter().enumerate() {
        for &idx in cluster {
            membership_orig[idx] = cid;
        }
    }

    // Shuffle point order
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rng);
    let sx: Vec<f32> = perm.iter().map(|&i| x[i]).collect();
    let sy: Vec<f32> = perm.iter().map(|&i| y[i]).collect();
    let sz: Vec<f32> = perm.iter().map(|&i| z[i]).collect();
    let shuffled = PointCloud::from_xyz(sx, sy, sz);

    let shuffled_clusters = euclidean_cluster(&shuffled, threshold, 1, n);

    // Build membership for shuffled (map back to original indices)
    let mut membership_shuffled = vec![usize::MAX; n];
    for (cid, cluster) in shuffled_clusters.iter().enumerate() {
        for &new_idx in cluster {
            let orig_idx = perm[new_idx];
            membership_shuffled[orig_idx] = cid;
        }
    }

    // Same number of clusters
    assert_eq!(original_clusters.len(), shuffled_clusters.len());

    // Same co-membership: if two points were in the same cluster originally,
    // they must be in the same cluster after shuffling
    for i in 0..n {
        for j in (i + 1)..n {
            let same_orig = membership_orig[i] == membership_orig[j];
            let same_shuf = membership_shuffled[i] == membership_shuffled[j];
            assert_eq!(
                same_orig, same_shuf,
                "points {} and {} have different co-membership after shuffle",
                i, j
            );
        }
    }
}

#[test]
fn translated_cloud_same_membership() {
    let mut rng = StdRng::seed_from_u64(66);
    let n = 150;
    let threshold = 3.0;

    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let z: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let cloud = PointCloud::from_xyz(x.clone(), y.clone(), z.clone());

    let original = normalize(euclidean_cluster(&cloud, threshold, 1, n));

    // Translate all points by a large offset
    let offset = 12345.0_f32;
    let tx: Vec<f32> = x.iter().map(|v| v + offset).collect();
    let ty: Vec<f32> = y.iter().map(|v| v + offset).collect();
    let tz: Vec<f32> = z.iter().map(|v| v + offset).collect();
    let translated = PointCloud::from_xyz(tx, ty, tz);

    let translated_result = normalize(euclidean_cluster(&translated, threshold, 1, n));

    assert_eq!(
        original, translated_result,
        "translation should not change cluster membership"
    );
}

#[test]
fn duplicate_points_stable() {
    // Inserting duplicate points should not break clustering
    let cloud = PointCloud::from_xyz(
        vec![0.0, 0.0, 0.0, 10.0, 10.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
    );
    let clusters = euclidean_cluster(&cloud, 1.0, 1, 100);
    assert_eq!(clusters.len(), 2);
    // The three duplicates at origin should be in one cluster
    let big = &clusters[0];
    assert_eq!(big.len(), 3);
    assert!(big.contains(&0) && big.contains(&1) && big.contains(&2));
}

// ────────────────── 4. Determinism / soak test ──────────────────

#[test]
fn determinism_1000_runs() {
    let mut rng = StdRng::seed_from_u64(77);
    let n = 500;
    let threshold = 2.0;

    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-20.0..20.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-20.0..20.0)).collect();
    let z: Vec<f32> = (0..n).map(|_| rng.gen_range(-20.0..20.0)).collect();
    let cloud = PointCloud::from_xyz(x, y, z);

    let reference = euclidean_cluster(&cloud, threshold, 1, n);

    for run in 0..1000 {
        let result = euclidean_cluster(&cloud, threshold, 1, n);
        assert_eq!(result, reference, "non-deterministic output on run {}", run);
    }
}

// ────────────────── 5. Density / radius sweep ──────────────────

#[test]
fn density_radius_sweep() {
    // Not an assertion-heavy test — prints a table for manual inspection.
    // Catches scaling cliffs if any radius/density combo panics or hangs.
    let sizes = [100, 1000, 5000];
    let radii = [0.5, 1.0, 2.0, 5.0];

    println!("\n=== Clustering Density/Radius Sweep ===");
    println!(
        "{:>8} {:>8} {:>10} {:>10} {:>10}",
        "N", "radius", "clusters", "max_sz", "time_ms"
    );

    for &n in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-20.0..20.0)).collect();
        let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-20.0..20.0)).collect();
        let z: Vec<f32> = (0..n).map(|_| rng.gen_range(-20.0..20.0)).collect();
        let cloud = PointCloud::from_xyz(x, y, z);

        for &r in &radii {
            let t0 = std::time::Instant::now();
            let clusters = euclidean_cluster(&cloud, r, 1, n);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let max_sz = clusters.iter().map(|c| c.len()).max().unwrap_or(0);

            println!(
                "{:>8} {:>8.1} {:>10} {:>10} {:>10.2}",
                n,
                r,
                clusters.len(),
                max_sz,
                ms
            );

            // Sanity: no lost points
            let total: usize = clusters.iter().map(|c| c.len()).sum();
            assert_eq!(total, n, "lost points at n={}, r={}", n, r);

            // No NaN in indices
            for c in &clusters {
                for &idx in c {
                    assert!(idx < n);
                }
            }
        }
    }
}
