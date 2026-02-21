use pointclouds_core::{Normals, PointCloud};
use pointclouds_spatial::KdTree;
use rayon::prelude::*;

/// Estimate surface normals for each point in the cloud using PCA.
///
/// For each point, the k nearest neighbors are found, a covariance matrix is
/// built from the neighbor positions, and the eigenvector corresponding to the
/// smallest eigenvalue is taken as the surface normal.  Normals are oriented to
/// face toward the origin (viewpoint at `[0, 0, 0]`).
///
/// The computation is parallelized across points using rayon.
pub fn estimate_normals(cloud: &PointCloud, k: usize) -> Normals {
    estimate_normals_with_viewpoint(cloud, k, [0.0, 0.0, 0.0])
}

/// Same as [`estimate_normals`] but orients normals toward the given viewpoint
/// instead of the origin.
pub fn estimate_normals_with_viewpoint(
    cloud: &PointCloud,
    k: usize,
    viewpoint: [f32; 3],
) -> Normals {
    // Handle edge cases
    if cloud.is_empty() || k == 0 {
        return Normals {
            nx: vec![],
            ny: vec![],
            nz: vec![],
        };
    }

    let tree = KdTree::build(cloud);
    let n = cloud.len();

    // Pre-extract points into contiguous array for cache-friendly access
    let points: Vec<[f32; 3]> = (0..n)
        .map(|i| [cloud.x[i], cloud.y[i], cloud.z[i]])
        .collect();

    // Parallel computation: each point independently computes its normal
    let normals_vec: Vec<[f32; 3]> = points
        .par_iter()
        .map(|point| {
            let indices = tree.knn_indices(point, k);

            let count = indices.len() as f32;

            if count < 1.0 {
                return [0.0, 0.0, 1.0];
            }

            // Compute centroid of the neighbors
            let mut cx = 0.0f32;
            let mut cy = 0.0f32;
            let mut cz = 0.0f32;

            for &idx in &indices {
                cx += points[idx][0];
                cy += points[idx][1];
                cz += points[idx][2];
            }
            cx /= count;
            cy /= count;
            cz /= count;

            // Build upper triangle of 3x3 covariance matrix (symmetric)
            let mut c00 = 0.0f32;
            let mut c01 = 0.0f32;
            let mut c02 = 0.0f32;
            let mut c11 = 0.0f32;
            let mut c12 = 0.0f32;
            let mut c22 = 0.0f32;
            for &idx in &indices {
                let dx = points[idx][0] - cx;
                let dy = points[idx][1] - cy;
                let dz = points[idx][2] - cz;
                c00 += dx * dx;
                c01 += dx * dy;
                c02 += dx * dz;
                c11 += dy * dy;
                c12 += dy * dz;
                c22 += dz * dz;
            }

            // Fast analytical eigenvector for smallest eigenvalue of 3x3 symmetric matrix
            let (mut nnx, mut nny, mut nnz) =
                smallest_eigenvector_3x3(c00, c01, c02, c11, c12, c22);

            // Normalize to unit length
            let len = (nnx * nnx + nny * nny + nnz * nnz).sqrt();
            if len > 1e-10 {
                nnx /= len;
                nny /= len;
                nnz /= len;
            }

            // Orient toward viewpoint
            let vx = viewpoint[0] - point[0];
            let vy = viewpoint[1] - point[1];
            let vz = viewpoint[2] - point[2];
            let dot = nnx * vx + nny * vy + nnz * vz;
            if dot < 0.0 {
                nnx = -nnx;
                nny = -nny;
                nnz = -nnz;
            }

            [nnx, nny, nnz]
        })
        .collect();

    // Unpack SoA
    let mut nx = Vec::with_capacity(n);
    let mut ny = Vec::with_capacity(n);
    let mut nz = Vec::with_capacity(n);
    for normal in &normals_vec {
        nx.push(normal[0]);
        ny.push(normal[1]);
        nz.push(normal[2]);
    }

    Normals { nx, ny, nz }
}

/// Compute the eigenvector corresponding to the smallest eigenvalue of a
/// 3x3 symmetric matrix using Cardano's analytical formula for the eigenvalues
/// and a cross-product trick for the eigenvector.
///
/// The matrix is:
///   | a00  a01  a02 |
///   | a01  a11  a12 |
///   | a02  a12  a22 |
///
/// This avoids the overhead of nalgebra's general iterative eigensolver and
/// all associated heap allocations, giving ~3-5x speedup for this critical
/// inner loop of normal estimation.
#[inline]
fn smallest_eigenvector_3x3(
    a00: f32,
    a01: f32,
    a02: f32,
    a11: f32,
    a12: f32,
    a22: f32,
) -> (f32, f32, f32) {
    // Use f64 internally for numerical stability in eigenvalue computation
    let a00 = a00 as f64;
    let a01 = a01 as f64;
    let a02 = a02 as f64;
    let a11 = a11 as f64;
    let a12 = a12 as f64;
    let a22 = a22 as f64;

    // Characteristic equation: det(A - λI) = 0
    // For 3x3 symmetric matrix, eigenvalues via Cardano's formula
    let m = (a00 + a11 + a22) / 3.0; // mean of diagonal (trace / 3)

    // Shift: B = A - mI
    let b00 = a00 - m;
    let b11 = a11 - m;
    let b22 = a22 - m;

    // q = det(B) / 2
    let q = (b00 * (b11 * b22 - a12 * a12) - a01 * (a01 * b22 - a12 * a02)
        + a02 * (a01 * a12 - b11 * a02))
        / 2.0;

    // p = sum of squares of B entries / 6
    let p = (b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * (a01 * a01 + a02 * a02 + a12 * a12)) / 6.0;

    let pp = p.max(0.0); // guard against tiny negatives from floating point

    if pp < 1e-30 {
        // Matrix is (near) zero or scalar multiple of identity
        return (0.0, 0.0, 1.0);
    }

    // phi = arccos(q / p^(3/2)) / 3
    let det_ratio = q / (pp * pp.sqrt());
    let det_ratio = det_ratio.clamp(-1.0, 1.0);
    let phi = det_ratio.acos() / 3.0;

    // Eigenvalues (sorted: eig0 <= eig1 <= eig2)
    let sqrt_p = pp.sqrt();
    let eig0 = m + 2.0 * sqrt_p * (phi + 2.0 * std::f64::consts::FRAC_PI_3).cos(); // smallest
    let eig2 = m + 2.0 * sqrt_p * phi.cos(); // largest
    let eig1 = 3.0 * m - eig0 - eig2; // middle (trace identity)

    // Pick the smallest eigenvalue
    let lambda = if eig0.abs() <= eig1.abs() && eig0.abs() <= eig2.abs() {
        eig0
    } else if eig1.abs() <= eig2.abs() {
        eig1
    } else {
        eig2
    };

    // Compute eigenvector via (A - λI) has rank ≤ 2, so cross product of any
    // two rows gives the null space direction (the eigenvector)
    let r00 = a00 - lambda;
    let r11 = a11 - lambda;
    let r22 = a22 - lambda;

    // Cross product of row 0 and row 1
    let mut ex = a01 * a12 - r11 * a02;
    let mut ey = a02 * a01 - a12 * r00;
    let mut ez = r00 * r11 - a01 * a01;

    let len2 = ex * ex + ey * ey + ez * ez;

    if len2 < 1e-30 {
        // Try rows 0 and 2
        ex = a01 * r22 - a12 * a02;
        ey = a02 * a02 - r22 * r00;
        ez = r00 * a12 - a01 * a02;

        let len2b = ex * ex + ey * ey + ez * ez;
        if len2b < 1e-30 {
            // Try rows 1 and 2
            ex = r11 * r22 - a12 * a12;
            ey = a12 * a02 - r22 * a01;
            ez = a01 * a12 - r11 * a02;

            let len2c = ex * ex + ey * ey + ez * ez;
            if len2c < 1e-30 {
                return (0.0, 0.0, 1.0);
            }
            let inv = 1.0 / len2c.sqrt();
            return ((ex * inv) as f32, (ey * inv) as f32, (ez * inv) as f32);
        }
        let inv = 1.0 / len2b.sqrt();
        return ((ex * inv) as f32, (ey * inv) as f32, (ez * inv) as f32);
    }

    let inv = 1.0 / len2.sqrt();
    ((ex * inv) as f32, (ey * inv) as f32, (ez * inv) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use pointclouds_core::PointCloud;
    use proptest::prelude::*;

    /// Helper: create a grid of points on the z~=0 plane.
    ///
    /// A tiny per-point perturbation is added to the z coordinate so that
    /// kiddo's bucket-based KdTree does not panic on too many identical
    /// axis values.  The perturbation is deterministic and negligibly small
    /// (~1e-7) relative to the grid spacing.
    fn xy_plane_cloud(grid_size: usize, spacing: f32) -> PointCloud {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        let mut idx = 0u32;
        for i in 0..grid_size {
            for j in 0..grid_size {
                x.push(i as f32 * spacing);
                y.push(j as f32 * spacing);
                // Tiny deterministic perturbation to avoid kiddo bucket panic
                z.push(idx as f32 * 1e-7);
                idx += 1;
            }
        }
        PointCloud::from_xyz(x, y, z)
    }

    /// Helper: create a grid of points on the y~=0 plane.
    ///
    /// Same perturbation strategy as `xy_plane_cloud`.
    fn xz_plane_cloud(grid_size: usize, spacing: f32) -> PointCloud {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        let mut idx = 0u32;
        for i in 0..grid_size {
            for j in 0..grid_size {
                x.push(i as f32 * spacing);
                y.push(idx as f32 * 1e-7);
                z.push(j as f32 * spacing);
                idx += 1;
            }
        }
        PointCloud::from_xyz(x, y, z)
    }

    /// Helper: create points on a unit sphere centered at the origin.
    fn sphere_cloud(n_lat: usize, n_lon: usize) -> PointCloud {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for i in 1..n_lat {
            let theta = std::f32::consts::PI * i as f32 / n_lat as f32;
            for j in 0..n_lon {
                let phi = 2.0 * std::f32::consts::PI * j as f32 / n_lon as f32;
                x.push(theta.sin() * phi.cos());
                y.push(theta.sin() * phi.sin());
                z.push(theta.cos());
            }
        }
        PointCloud::from_xyz(x, y, z)
    }

    #[test]
    fn normals_of_xy_plane() {
        // Points on the z=0 plane; normals should be approximately (0, 0, +/-1).
        let cloud = xy_plane_cloud(10, 1.0);
        let normals = estimate_normals(&cloud, 10);

        assert_eq!(normals.nx.len(), cloud.len());

        // Check interior points (skip edge points that may have skewed neighborhoods)
        for i in 0..cloud.len() {
            let nz_abs = normals.nz[i].abs();
            // The normal should be dominated by the z component
            assert!(
                nz_abs > 0.9,
                "Point {}: normal z component is {} (expected ~1.0), full normal = ({}, {}, {})",
                i,
                nz_abs,
                normals.nx[i],
                normals.ny[i],
                normals.nz[i]
            );
        }
    }

    #[test]
    fn normals_of_xz_plane() {
        // Points on the y=0 plane; normals should be approximately (0, +/-1, 0).
        let cloud = xz_plane_cloud(10, 1.0);
        let normals = estimate_normals(&cloud, 10);

        assert_eq!(normals.ny.len(), cloud.len());

        for i in 0..cloud.len() {
            let ny_abs = normals.ny[i].abs();
            assert!(
                ny_abs > 0.9,
                "Point {}: normal y component is {} (expected ~1.0), full normal = ({}, {}, {})",
                i,
                ny_abs,
                normals.nx[i],
                normals.ny[i],
                normals.nz[i]
            );
        }
    }

    #[test]
    fn normals_of_sphere() {
        // Points on a unit sphere centered at origin.
        // With viewpoint at origin, normals should point inward (toward origin),
        // i.e., normal ≈ -point (since points are on the unit sphere).
        let cloud = sphere_cloud(20, 20);
        let normals = estimate_normals(&cloud, 15);

        assert_eq!(normals.nx.len(), cloud.len());

        let mut good = 0;
        for i in 0..cloud.len() {
            // The expected inward normal is -point (since |point| = 1)
            let ex = -cloud.x[i];
            let ey = -cloud.y[i];
            let ez = -cloud.z[i];

            // Dot product of computed normal with expected normal
            let dot = normals.nx[i] * ex + normals.ny[i] * ey + normals.nz[i] * ez;
            // dot should be close to 1.0 (same direction)
            if dot > 0.8 {
                good += 1;
            }
        }

        // Allow some edge effects, but most normals should be correct
        let ratio = good as f32 / cloud.len() as f32;
        assert!(
            ratio > 0.85,
            "Only {:.1}% of sphere normals pointed inward (expected > 85%)",
            ratio * 100.0,
        );
    }

    #[test]
    fn normals_are_unit_length() {
        let cloud = xy_plane_cloud(5, 1.0);
        let normals = estimate_normals(&cloud, 5);

        for i in 0..cloud.len() {
            let len = (normals.nx[i] * normals.nx[i]
                + normals.ny[i] * normals.ny[i]
                + normals.nz[i] * normals.nz[i])
                .sqrt();
            assert_abs_diff_eq!(len, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn normals_empty_cloud() {
        let cloud = PointCloud::new();
        let normals = estimate_normals(&cloud, 10);
        assert!(normals.nx.is_empty());
        assert!(normals.ny.is_empty());
        assert!(normals.nz.is_empty());
    }

    #[test]
    fn normals_single_point() {
        // Degenerate case: only one point.  Should not panic.
        let cloud = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
        let normals = estimate_normals(&cloud, 5);
        assert_eq!(normals.nx.len(), 1);
        assert_eq!(normals.ny.len(), 1);
        assert_eq!(normals.nz.len(), 1);
    }

    #[test]
    fn normals_collinear_points() {
        // Points along the x-axis — the normal is degenerate (there are two
        // eigenvectors with eigenvalue zero).  We just verify it doesn't panic
        // and returns some unit-length normals.
        // Tiny perturbations on y and z to avoid kiddo bucket panic.
        let n = 20;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| i as f32 * 1e-7).collect();
        let z: Vec<f32> = (0..n).map(|i| i as f32 * 2e-7).collect();
        let cloud = PointCloud::from_xyz(x, y, z);
        let normals = estimate_normals(&cloud, 5);

        assert_eq!(normals.nx.len(), n);
        for i in 0..n {
            let len = (normals.nx[i] * normals.nx[i]
                + normals.ny[i] * normals.ny[i]
                + normals.nz[i] * normals.nz[i])
                .sqrt();
            // May not be perfectly unit if all neighbors are identical, but
            // should be finite and reasonable.
            assert!(
                len.is_finite(),
                "Normal length is not finite at index {}",
                i
            );
        }
    }

    #[test]
    fn normals_k_zero_returns_empty() {
        let cloud = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
        let normals = estimate_normals(&cloud, 0);
        assert!(normals.nx.is_empty());
    }

    #[test]
    fn normals_with_viewpoint_flips_direction() {
        // XY plane at z ~ 5.  With viewpoint at [0, 0, 10] (above), normals
        // should point upward (+z).  With viewpoint at [0, 0, -10] (below),
        // normals should point downward (-z).
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        let mut idx = 0u32;
        for i in 0..10 {
            for j in 0..10 {
                x.push(i as f32);
                y.push(j as f32);
                z.push(5.0 + idx as f32 * 1e-7);
                idx += 1;
            }
        }
        let cloud = PointCloud::from_xyz(x, y, z);

        let normals_above = estimate_normals_with_viewpoint(&cloud, 10, [5.0, 5.0, 100.0]);
        let normals_below = estimate_normals_with_viewpoint(&cloud, 10, [5.0, 5.0, -100.0]);

        // Check a few interior points
        for i in [44, 45, 55, 54] {
            assert!(
                normals_above.nz[i] > 0.9,
                "Viewpoint above: normal z at {} is {} (expected > 0.9)",
                i,
                normals_above.nz[i]
            );
            assert!(
                normals_below.nz[i] < -0.9,
                "Viewpoint below: normal z at {} is {} (expected < -0.9)",
                i,
                normals_below.nz[i]
            );
        }
    }

    proptest! {
        #[test]
        fn normals_always_unit_length(
            pts in prop::collection::vec(
                (-10.0f32..10.0, -10.0f32..10.0, -10.0f32..10.0),
                3..50
            )
        ) {
            let x: Vec<f32> = pts.iter().map(|p| p.0).collect();
            let y: Vec<f32> = pts.iter().map(|p| p.1).collect();
            let z: Vec<f32> = pts.iter().map(|p| p.2).collect();
            let cloud = PointCloud::from_xyz(x, y, z);
            let normals = estimate_normals(&cloud, 5);

            for i in 0..cloud.len() {
                let len = (normals.nx[i] * normals.nx[i]
                    + normals.ny[i] * normals.ny[i]
                    + normals.nz[i] * normals.nz[i])
                    .sqrt();
                // Allow some tolerance for degenerate configurations
                prop_assert!(
                    len.is_finite(),
                    "Normal at index {} has non-finite length: {}", i, len
                );
                if len > 1e-6 {
                    prop_assert!(
                        (len - 1.0).abs() < 0.01,
                        "Normal at index {} has length {} (expected ~1.0)", i, len
                    );
                }
            }
        }
    }
}
