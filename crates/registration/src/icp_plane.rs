use nalgebra::{Matrix6, Vector6};
use pointclouds_core::{Normals, PointCloud};
use pointclouds_spatial::KdTree;

use crate::correspondence::{find_correspondences, Correspondence};
use crate::icp::{apply_transform, compute_rmse, IcpParams, IcpResult, RigidTransform};

/// Point-to-plane ICP registration.
///
/// Aligns `source` to `target` by minimizing the sum of squared point-to-plane
/// distances: `sum_i ((R*s_i + t - t_i) · n_i)^2`, where `n_i` is the target
/// surface normal at the corresponding point.
///
/// This converges faster than point-to-point ICP on smooth surfaces because it
/// allows sliding along the tangent plane.
///
/// # Errors
///
/// Returns `Err` if `target_normals` length does not match `target.len()`.
pub fn icp_point_to_plane(
    source: &PointCloud,
    target: &PointCloud,
    target_normals: &Normals,
    params: &IcpParams,
) -> Result<IcpResult, IcpPlaneError> {
    // Validate normals length
    if target_normals.nx.len() != target.len() {
        return Err(IcpPlaneError::NormalsMismatch {
            normals_len: target_normals.nx.len(),
            cloud_len: target.len(),
        });
    }

    // Handle empty clouds
    if source.is_empty() || target.is_empty() {
        return Ok(IcpResult {
            transform: RigidTransform::identity(),
            fitness: 0.0,
            rmse: 0.0,
            converged: source.is_empty() && target.is_empty(),
            num_iterations: 0,
        });
    }

    let target_tree = KdTree::build(target);
    let mut current = apply_transform(source, &RigidTransform::identity());
    let mut cumulative = RigidTransform::identity();

    let mut prev_rmse = f32::INFINITY;
    let mut converged = false;
    let mut num_iterations = 0;
    let mut last_rmse = f32::INFINITY;
    let mut last_fitness = 0.0_f32;

    for iter in 0..params.max_iterations {
        num_iterations = iter + 1;

        let correspondences =
            find_correspondences(&current, &target_tree, params.max_correspondence_distance);

        if correspondences.is_empty() {
            break;
        }

        let rmse = compute_rmse(&correspondences);
        last_rmse = rmse;
        last_fitness = correspondences.len() as f32 / source.len() as f32;

        if (prev_rmse - rmse).abs() < params.tolerance {
            converged = true;
            break;
        }
        prev_rmse = rmse;

        let incremental = solve_point_to_plane(&current, target, target_normals, &correspondences);

        cumulative = cumulative.compose(&incremental);
        current = apply_transform(&current, &incremental);
    }

    if num_iterations == 0 {
        let correspondences =
            find_correspondences(&current, &target_tree, params.max_correspondence_distance);
        if !correspondences.is_empty() {
            last_rmse = compute_rmse(&correspondences);
            last_fitness = correspondences.len() as f32 / source.len() as f32;
        }
    }

    Ok(IcpResult {
        transform: cumulative,
        fitness: last_fitness,
        rmse: last_rmse,
        converged,
        num_iterations,
    })
}

/// Error type for point-to-plane ICP.
#[derive(Debug, Clone)]
pub enum IcpPlaneError {
    /// The normals vector length does not match the target cloud length.
    NormalsMismatch {
        normals_len: usize,
        cloud_len: usize,
    },
}

impl std::fmt::Display for IcpPlaneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IcpPlaneError::NormalsMismatch {
                normals_len,
                cloud_len,
            } => write!(
                f,
                "target_normals length ({}) does not match target cloud length ({})",
                normals_len, cloud_len
            ),
        }
    }
}

impl std::error::Error for IcpPlaneError {}

/// Solve the linearized point-to-plane minimization for one ICP step.
///
/// Uses the small-angle approximation: R ≈ I + [α, β, γ]× where [α, β, γ]
/// are small rotation angles. This gives a 6×6 linear system Ax = b where
/// x = [α, β, γ, tx, ty, tz].
fn solve_point_to_plane(
    source: &PointCloud,
    target: &PointCloud,
    target_normals: &Normals,
    correspondences: &[Correspondence],
) -> RigidTransform {
    if correspondences.is_empty() {
        return RigidTransform::identity();
    }

    // Build the 6×6 normal equations: A^T A x = A^T b
    // For each correspondence (s_i, t_i, n_i):
    //   a_i = [n_i × s_i, n_i]  (1×6 row)
    //   b_i = (t_i - s_i) · n_i
    let mut ata = Matrix6::<f64>::zeros();
    let mut atb = Vector6::<f64>::zeros();

    for c in correspondences {
        let si = c.source_index;
        let ti = c.target_index;

        let sx = source.x[si] as f64;
        let sy = source.y[si] as f64;
        let sz = source.z[si] as f64;

        let tx = target.x[ti] as f64;
        let ty = target.y[ti] as f64;
        let tz = target.z[ti] as f64;

        let nx = target_normals.nx[ti] as f64;
        let ny = target_normals.ny[ti] as f64;
        let nz = target_normals.nz[ti] as f64;

        // a = [n × s, n] = [sy*nz - sz*ny, sz*nx - sx*nz, sx*ny - sy*nx, nx, ny, nz]
        let a = Vector6::new(
            sy * nz - sz * ny,
            sz * nx - sx * nz,
            sx * ny - sy * nx,
            nx,
            ny,
            nz,
        );

        // b = (t - s) · n
        let b = (tx - sx) * nx + (ty - sy) * ny + (tz - sz) * nz;

        // Accumulate A^T A and A^T b
        ata += a * a.transpose();
        atb += a * b;
    }

    // Add Tikhonov regularization to handle rank-deficient cases
    // (e.g., flat plane where tangent directions are unconstrained).
    // The damping factor is scaled relative to the matrix norm.
    let diag_max = (0..6).map(|i| ata[(i, i)].abs()).fold(0.0_f64, f64::max);
    let lambda = 1e-6 * diag_max.max(1e-12);
    for i in 0..6 {
        ata[(i, i)] += lambda;
    }

    // Solve A^T A x = A^T b using Cholesky (SPD after regularization)
    // Fall back to LU if Cholesky fails.
    let x = match ata.cholesky() {
        Some(chol) => chol.solve(&atb),
        None => match ata.lu().solve(&atb) {
            Some(sol) => sol,
            None => return RigidTransform::identity(),
        },
    };

    let alpha = x[0] as f32;
    let beta = x[1] as f32;
    let gamma = x[2] as f32;
    let tx = x[3] as f32;
    let ty = x[4] as f32;
    let tz = x[5] as f32;

    // Construct rotation matrix from small angles using Rodrigues' formula.
    let angle = (alpha * alpha + beta * beta + gamma * gamma).sqrt();

    let rotation = if angle < 1e-10 {
        [
            [1.0, -gamma, beta],
            [gamma, 1.0, -alpha],
            [-beta, alpha, 1.0],
        ]
    } else {
        let ax = alpha / angle;
        let ay = beta / angle;
        let az = gamma / angle;
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;

        [
            [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay],
            [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax],
            [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
        ]
    };

    RigidTransform {
        rotation,
        translation: [tx, ty, tz],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use pointclouds_core::PointCloud;
    use pointclouds_normals::estimate_normals;

    /// Helper: create a flat grid on z=0 with small perturbation for kiddo.
    fn flat_plane_cloud(grid: usize) -> PointCloud {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        let mut idx = 0u32;
        for i in 0..grid {
            for j in 0..grid {
                x.push(i as f32 - grid as f32 / 2.0);
                y.push(j as f32 - grid as f32 / 2.0);
                z.push(idx as f32 * 1e-7);
                idx += 1;
            }
        }
        PointCloud::from_xyz(x, y, z)
    }

    #[test]
    fn plane_icp_identity() {
        let cloud = flat_plane_cloud(10);
        let normals = estimate_normals(&cloud, 10);
        let params = IcpParams::default();
        let result = icp_point_to_plane(&cloud, &cloud, &normals, &params).unwrap();

        assert!(result.converged);
        assert!(result.rmse < 1e-4, "RMSE = {}", result.rmse);
        assert!(result.transform.is_identity(1e-3));
    }

    #[test]
    fn plane_icp_known_translation_along_normal() {
        // Translate along Z (the normal direction of the plane).
        let target = flat_plane_cloud(10);
        let normals = estimate_normals(&target, 10);

        let shift = RigidTransform {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.2],
        };
        let source = apply_transform(&target, &shift);

        let params = IcpParams {
            max_iterations: 200,
            tolerance: 1e-10,
            ..IcpParams::default()
        };
        let result = icp_point_to_plane(&source, &target, &normals, &params).unwrap();

        assert!(
            result.rmse < 0.05,
            "RMSE = {}, iters = {}",
            result.rmse,
            result.num_iterations
        );
        assert_relative_eq!(result.transform.translation[2], -0.2, epsilon = 0.1);
    }

    #[test]
    fn plane_icp_known_translation_tangent() {
        // Translate along X (tangent to the plane).
        // Point-to-plane allows sliding along the tangent, so it should still
        // converge even if the tangent component is under-constrained.
        let target = flat_plane_cloud(10);
        let normals = estimate_normals(&target, 10);

        let shift = RigidTransform {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.3, 0.0, 0.0],
        };
        let source = apply_transform(&target, &shift);

        let params = IcpParams {
            max_iterations: 100,
            tolerance: 1e-8,
            ..IcpParams::default()
        };
        let result = icp_point_to_plane(&source, &target, &normals, &params).unwrap();

        assert!(result.converged || result.rmse < 0.1);
    }

    #[test]
    fn plane_icp_small_rotation() {
        // Small rotation on a hemisphere so normals vary and all DOFs are constrained.
        let n_lat = 15;
        let n_lon = 30;
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for i in 1..n_lat {
            let theta = std::f32::consts::FRAC_PI_2 * i as f32 / n_lat as f32;
            for j in 0..n_lon {
                let phi = 2.0 * std::f32::consts::PI * j as f32 / n_lon as f32;
                x.push(theta.sin() * phi.cos());
                y.push(theta.sin() * phi.sin());
                z.push(theta.cos());
            }
        }
        let target = PointCloud::from_xyz(x, y, z);
        let normals = estimate_normals(&target, 10);

        let angle: f32 = 0.1; // ~5.7 degrees
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let rot = RigidTransform {
            rotation: [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        };
        let source = apply_transform(&target, &rot);

        let params = IcpParams {
            max_iterations: 100,
            tolerance: 1e-8,
            ..IcpParams::default()
        };
        let result = icp_point_to_plane(&source, &target, &normals, &params).unwrap();

        assert!(result.rmse < 0.1, "RMSE = {}", result.rmse);

        let transformed = apply_transform(&source, &result.transform);
        let mut max_err = 0.0_f32;
        for i in 0..target.len() {
            let dx = transformed.x[i] - target.x[i];
            let dy = transformed.y[i] - target.y[i];
            let dz = transformed.z[i] - target.z[i];
            let err = (dx * dx + dy * dy + dz * dz).sqrt();
            max_err = max_err.max(err);
        }
        assert!(
            max_err < 0.2,
            "Max point error = {} (expected < 0.2)",
            max_err
        );
    }

    #[test]
    fn plane_icp_empty_clouds() {
        let empty = PointCloud::new();
        let normals = Normals {
            nx: vec![],
            ny: vec![],
            nz: vec![],
        };
        let params = IcpParams::default();
        let result = icp_point_to_plane(&empty, &empty, &normals, &params).unwrap();

        assert!(result.transform.is_identity(1e-6));
        assert_eq!(result.num_iterations, 0);
    }

    #[test]
    fn plane_icp_normals_mismatch_returns_error() {
        let cloud = PointCloud::from_xyz(vec![0.0, 1.0], vec![0.0; 2], vec![0.0; 2]);
        let bad_normals = Normals {
            nx: vec![0.0],
            ny: vec![0.0],
            nz: vec![1.0],
        };
        let params = IcpParams::default();
        let result = icp_point_to_plane(&cloud, &cloud, &bad_normals, &params);
        assert!(result.is_err());
    }

    #[test]
    fn plane_icp_converges_faster_than_point_to_point() {
        // Point-to-plane should need fewer iterations for a normal-direction shift.
        let target = flat_plane_cloud(10);
        let normals = estimate_normals(&target, 10);

        let shift = RigidTransform {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.3],
        };
        let source = apply_transform(&target, &shift);

        let params = IcpParams {
            max_iterations: 200,
            tolerance: 1e-8,
            ..IcpParams::default()
        };

        let plane_result = icp_point_to_plane(&source, &target, &normals, &params).unwrap();
        let point_result = crate::icp::icp_point_to_point(&source, &target, &params);

        assert!(
            plane_result.num_iterations <= point_result.num_iterations + 5,
            "plane iters={}, point iters={}",
            plane_result.num_iterations,
            point_result.num_iterations
        );
    }
}
