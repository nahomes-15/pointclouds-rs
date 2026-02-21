use pointclouds_core::PointCloud;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// A 3D plane model in the form `n . x + d = 0`, where `n` is a unit normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneModel {
    pub normal: [f32; 3],
    pub d: f32,
}

impl PlaneModel {
    /// Computes the absolute distance from a point to this plane.
    /// Assumes `normal` is a unit vector.
    #[inline]
    pub fn distance_to_point(&self, point: &[f32; 3]) -> f32 {
        (self.normal[0] * point[0] + self.normal[1] * point[1] + self.normal[2] * point[2] + self.d)
            .abs()
    }
}

impl Default for PlaneModel {
    fn default() -> Self {
        Self {
            normal: [0.0, 0.0, 1.0],
            d: 0.0,
        }
    }
}

/// Fits a plane to the point cloud using the RANSAC algorithm.
///
/// Uses a random (non-deterministic) seed. For reproducible results, use
/// [`ransac_plane_seeded`] instead.
pub fn ransac_plane(
    cloud: &PointCloud,
    distance_threshold: f32,
    iterations: usize,
) -> (PlaneModel, Vec<usize>) {
    let seed = rand::thread_rng().next_u64();
    ransac_plane_seeded(cloud, distance_threshold, iterations, seed)
}

/// Fits a plane to the point cloud using the RANSAC algorithm with a
/// deterministic seed for the random number generator.
///
/// # Algorithm
///
/// 1. Pre-generate all random samples upfront for determinism.
/// 2. Distribute iteration batches across threads with rayon.
/// 3. Each thread independently finds its local best model.
/// 4. Merge thread-local results to find the global best.
/// 5. Apply adaptive early termination based on inlier ratio.
/// 6. Return the best plane model and its inlier indices.
pub fn ransac_plane_seeded(
    cloud: &PointCloud,
    distance_threshold: f32,
    iterations: usize,
    seed: u64,
) -> (PlaneModel, Vec<usize>) {
    let n = cloud.len();

    if n < 3 {
        return (PlaneModel::default(), Vec::new());
    }

    // Pre-extract points into contiguous array for cache-friendly access
    let points: Vec<[f32; 3]> = (0..n)
        .map(|i| [cloud.x[i], cloud.y[i], cloud.z[i]])
        .collect();

    // Pre-generate all random samples for determinism
    let mut rng = StdRng::seed_from_u64(seed);
    let samples: Vec<(usize, usize, usize)> = (0..iterations)
        .filter_map(|_| sample_three_distinct(n, &mut rng))
        .collect();

    let use_parallel = n >= 10_000 && samples.len() >= 16;

    let (best_model, _best_count) = if use_parallel {
        // Parallel: each batch of iterations runs independently
        samples
            .par_iter()
            .filter_map(|&(i0, i1, i2)| {
                let model = fit_plane_from_three_points(&points[i0], &points[i1], &points[i2])?;
                let count = count_inliers(&points, &model, distance_threshold);
                Some((model, count))
            })
            .reduce_with(|a, b| if a.1 >= b.1 { a } else { b })
            .unwrap_or((PlaneModel::default(), 0))
    } else {
        // Sequential with adaptive early termination
        let mut best_model = PlaneModel::default();
        let mut best_inlier_count: usize = 0;

        for (iter, &(i0, i1, i2)) in samples.iter().enumerate() {
            let model = match fit_plane_from_three_points(&points[i0], &points[i1], &points[i2]) {
                Some(m) => m,
                None => continue,
            };

            let inlier_count = count_inliers(&points, &model, distance_threshold);

            if inlier_count > best_inlier_count {
                best_inlier_count = inlier_count;
                best_model = model;

                // Adaptive early termination
                let w = best_inlier_count as f64 / n as f64;
                if w > 0.5 {
                    let needed = (1.0 - 0.999f64).ln() / (1.0 - w.powi(3)).ln();
                    if (iter as f64) > needed {
                        break;
                    }
                }
            }
        }

        (best_model, best_inlier_count)
    };

    // Collect inlier indices for the best model
    let inliers: Vec<usize> = (0..n)
        .filter(|&j| best_model.distance_to_point(&points[j]) <= distance_threshold)
        .collect();

    (best_model, inliers)
}

/// Count inliers sequentially.
#[inline]
fn count_inliers(points: &[[f32; 3]], model: &PlaneModel, threshold: f32) -> usize {
    points
        .iter()
        .filter(|p| model.distance_to_point(p) <= threshold)
        .count()
}

/// Samples 3 distinct indices in [0, n).
fn sample_three_distinct(n: usize, rng: &mut StdRng) -> Option<(usize, usize, usize)> {
    if n < 3 {
        return None;
    }
    let i0 = rng.gen_range(0..n);
    let mut i1 = rng.gen_range(0..n);
    // Retry until distinct from i0
    let mut attempts = 0;
    while i1 == i0 {
        if attempts > 100 {
            return None;
        }
        i1 = rng.gen_range(0..n);
        attempts += 1;
    }
    let mut i2 = rng.gen_range(0..n);
    attempts = 0;
    while i2 == i0 || i2 == i1 {
        if attempts > 100 {
            return None;
        }
        i2 = rng.gen_range(0..n);
        attempts += 1;
    }
    Some((i0, i1, i2))
}

/// Fits a plane through 3 points, returning `None` if they are collinear.
fn fit_plane_from_three_points(p0: &[f32; 3], p1: &[f32; 3], p2: &[f32; 3]) -> Option<PlaneModel> {
    // Vectors from p0 to p1 and p0 to p2
    let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

    // Cross product: v1 x v2
    let nx = v1[1] * v2[2] - v1[2] * v2[1];
    let ny = v1[2] * v2[0] - v1[0] * v2[2];
    let nz = v1[0] * v2[1] - v1[1] * v2[0];

    let len = (nx * nx + ny * ny + nz * nz).sqrt();

    if len < 1e-10 {
        // Points are collinear (or coincident)
        return None;
    }

    // Normalize
    let normal = [nx / len, ny / len, nz / len];
    let d = -(normal[0] * p0[0] + normal[1] * p0[1] + normal[2] * p0[2]);

    Some(PlaneModel { normal, d })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pointclouds_core::PointCloud;
    use proptest::prelude::*;

    #[test]
    fn fit_xy_plane() {
        // Points on z=0
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                x.push(i as f32 * 0.1);
                y.push(j as f32 * 0.1);
                z.push(0.0);
            }
        }
        let cloud = PointCloud::from_xyz(x, y, z);
        let (model, inliers) = ransac_plane_seeded(&cloud, 0.01, 100, 42);

        // Normal should be approximately (0, 0, +/-1)
        assert!(
            model.normal[2].abs() > 0.99,
            "Expected normal z-component near +/-1, got {:?}",
            model.normal
        );
        assert!(model.d.abs() < 0.01, "Expected d near 0, got {}", model.d);
        assert_eq!(inliers.len(), 400);
    }

    #[test]
    fn fit_offset_plane() {
        // Points on z=5
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                x.push(i as f32);
                y.push(j as f32);
                z.push(5.0);
            }
        }
        let cloud = PointCloud::from_xyz(x, y, z);
        let (model, inliers) = ransac_plane_seeded(&cloud, 0.01, 100, 42);

        // Normal should be approximately (0, 0, +/-1)
        assert!(
            model.normal[2].abs() > 0.99,
            "Expected normal z-component near +/-1, got {:?}",
            model.normal
        );
        // d should be approximately -5 or 5 depending on normal direction
        assert!(
            (model.d.abs() - 5.0).abs() < 0.01,
            "Expected |d| near 5, got {}",
            model.d
        );
        assert_eq!(inliers.len(), 100);
    }

    #[test]
    fn fit_tilted_plane() {
        // Points on x + y + z = 1
        // We'll generate points where z = 1 - x - y
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                let xv = i as f32 * 0.1;
                let yv = j as f32 * 0.1;
                let zv = 1.0 - xv - yv;
                x.push(xv);
                y.push(yv);
                z.push(zv);
            }
        }
        let cloud = PointCloud::from_xyz(x, y, z);
        let (model, inliers) = ransac_plane_seeded(&cloud, 0.01, 100, 42);

        // The plane x + y + z = 1 has normal (1,1,1)/sqrt(3) and d = -1/sqrt(3)
        // or equivalently n = (-1,-1,-1)/sqrt(3) and d = 1/sqrt(3)
        let expected_n_mag = 1.0 / 3.0f32.sqrt();

        // Check that each component of normal has equal magnitude
        let nx = model.normal[0].abs();
        let ny = model.normal[1].abs();
        let nz = model.normal[2].abs();

        assert!(
            (nx - expected_n_mag).abs() < 0.05,
            "nx={} expected ~{}",
            nx,
            expected_n_mag
        );
        assert!(
            (ny - expected_n_mag).abs() < 0.05,
            "ny={} expected ~{}",
            ny,
            expected_n_mag
        );
        assert!(
            (nz - expected_n_mag).abs() < 0.05,
            "nz={} expected ~{}",
            nz,
            expected_n_mag
        );

        // All 100 points should be inliers
        assert_eq!(inliers.len(), 100);
    }

    #[test]
    fn plane_with_outliers() {
        // Grid of points on z=0 (inliers) + distant outliers at z=100
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();

        // Inliers: a 7x7 grid on z=0 (49 points)
        for i in 0..7 {
            for j in 0..7 {
                x.push(i as f32);
                y.push(j as f32);
                z.push(0.0);
            }
        }

        // Outliers at z=100
        for i in 0..10 {
            x.push(i as f32);
            y.push(i as f32);
            z.push(100.0);
        }

        let cloud = PointCloud::from_xyz(x, y, z);
        let (model, inliers) = ransac_plane_seeded(&cloud, 0.1, 200, 42);

        // The plane should fit z=0 (the dominant plane)
        assert!(
            model.normal[2].abs() > 0.9,
            "Expected normal z-component to be dominant, got {:?}",
            model.normal
        );

        // All 49 inlier points should be detected, outliers excluded
        assert!(
            inliers.len() >= 49,
            "Expected at least 49 inliers, got {}",
            inliers.len()
        );
        // Outliers should not be included
        for &idx in &inliers {
            assert!(
                cloud.z[idx].abs() < 1.0,
                "Outlier point {} (z={}) was incorrectly classified as inlier",
                idx,
                cloud.z[idx]
            );
        }
    }

    #[test]
    fn empty_cloud() {
        let cloud = PointCloud::new();
        let (model, inliers) = ransac_plane_seeded(&cloud, 0.1, 100, 42);
        assert_eq!(model.normal, [0.0, 0.0, 1.0]);
        assert_eq!(model.d, 0.0);
        assert!(inliers.is_empty());
    }

    #[test]
    fn fewer_than_3_points() {
        // 2 points: not enough to define a plane
        let cloud = PointCloud::from_xyz(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
        let (model, inliers) = ransac_plane_seeded(&cloud, 0.1, 100, 42);
        assert_eq!(model.normal, [0.0, 0.0, 1.0]);
        assert_eq!(model.d, 0.0);
        assert!(inliers.is_empty());

        // 1 point
        let cloud = PointCloud::from_xyz(vec![0.0], vec![0.0], vec![0.0]);
        let (_model, inliers) = ransac_plane_seeded(&cloud, 0.1, 100, 42);
        assert!(inliers.is_empty());
    }

    #[test]
    fn distance_to_point_works() {
        // Plane: z = 0  =>  normal = (0,0,1), d = 0
        let model = PlaneModel {
            normal: [0.0, 0.0, 1.0],
            d: 0.0,
        };
        assert!((model.distance_to_point(&[0.0, 0.0, 0.0]) - 0.0).abs() < 1e-6);
        assert!((model.distance_to_point(&[1.0, 2.0, 3.0]) - 3.0).abs() < 1e-6);
        assert!((model.distance_to_point(&[0.0, 0.0, -5.0]) - 5.0).abs() < 1e-6);

        // Plane: x + y + z = 3, i.e. normal = (1,1,1)/sqrt(3), d = -3/sqrt(3)
        let s3 = 3.0f32.sqrt();
        let model2 = PlaneModel {
            normal: [1.0 / s3, 1.0 / s3, 1.0 / s3],
            d: -3.0 / s3,
        };
        // Point (1,1,1) is on the plane
        assert!(model2.distance_to_point(&[1.0, 1.0, 1.0]) < 1e-5);
        // Point (0,0,0) is at distance 3/sqrt(3) = sqrt(3)
        assert!((model2.distance_to_point(&[0.0, 0.0, 0.0]) - s3).abs() < 1e-5);
    }

    #[test]
    fn seeded_is_deterministic() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );

        let (m1, i1) = ransac_plane_seeded(&cloud, 0.01, 50, 123);
        let (m2, i2) = ransac_plane_seeded(&cloud, 0.01, 50, 123);

        assert_eq!(m1.normal, m2.normal);
        assert_eq!(m1.d, m2.d);
        assert_eq!(i1, i2);
    }

    #[test]
    fn exactly_3_points() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0],
        );
        let (model, inliers) = ransac_plane_seeded(&cloud, 0.01, 100, 42);
        // These 3 points define z=0 plane
        assert!(model.normal[2].abs() > 0.99);
        assert_eq!(inliers.len(), 3);
    }

    proptest! {
        #[test]
        fn inliers_are_within_threshold(
            plane_pts in prop::collection::vec(
                (-10.0f32..10.0, -10.0f32..10.0),
                10..50
            ),
            threshold in 0.01f32..1.0,
            seed in 0u64..10000,
        ) {
            // Generate points on the z=0 plane
            let n = plane_pts.len();
            let cloud = PointCloud::from_xyz(
                plane_pts.iter().map(|p| p.0).collect(),
                plane_pts.iter().map(|p| p.1).collect(),
                vec![0.0; n],
            );

            let (model, inliers) = ransac_plane_seeded(&cloud, threshold, 100, seed);

            for &idx in &inliers {
                let point = [cloud.x[idx], cloud.y[idx], cloud.z[idx]];
                let dist = model.distance_to_point(&point);
                prop_assert!(
                    dist <= threshold + 1e-5,
                    "Inlier {} has distance {} > threshold {}",
                    idx, dist, threshold
                );
            }
        }
    }
}
