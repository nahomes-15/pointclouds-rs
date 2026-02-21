use nalgebra::{Matrix3, Vector3, SVD};
use pointclouds_core::PointCloud;
use pointclouds_spatial::KdTree;

use crate::correspondence::{find_correspondences, Correspondence};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidTransform {
    pub rotation: [[f32; 3]; 3],
    pub translation: [f32; 3],
}

impl RigidTransform {
    pub fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        }
    }

    pub fn is_identity(&self, eps: f32) -> bool {
        let id = Self::identity();
        for r in 0..3 {
            for c in 0..3 {
                if (self.rotation[r][c] - id.rotation[r][c]).abs() > eps {
                    return false;
                }
            }
        }
        for a in 0..3 {
            if self.translation[a].abs() > eps {
                return false;
            }
        }
        true
    }

    /// Apply the rigid transform to a single point: R * p + t
    pub fn apply_to_point(&self, p: &[f32; 3]) -> [f32; 3] {
        let r = &self.rotation;
        let t = &self.translation;
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
        ]
    }

    /// Compose two transforms: apply `self` first, then `other`.
    ///
    /// Result: R_new = other.R * self.R, t_new = other.R * self.t + other.t
    pub fn compose(&self, other: &RigidTransform) -> RigidTransform {
        let r_self = mat3_from_arrays(&self.rotation);
        let r_other = mat3_from_arrays(&other.rotation);
        let t_self = Vector3::new(
            self.translation[0],
            self.translation[1],
            self.translation[2],
        );
        let t_other = Vector3::new(
            other.translation[0],
            other.translation[1],
            other.translation[2],
        );

        let r_new = r_other * r_self;
        let t_new = r_other * t_self + t_other;

        RigidTransform {
            rotation: mat3_to_arrays(&r_new),
            translation: [t_new[0], t_new[1], t_new[2]],
        }
    }
}

/// Apply a rigid transform to all points in a cloud, returning a new cloud.
pub fn apply_transform(cloud: &PointCloud, transform: &RigidTransform) -> PointCloud {
    let n = cloud.len();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);

    for i in 0..n {
        let p = [cloud.x[i], cloud.y[i], cloud.z[i]];
        let tp = transform.apply_to_point(&p);
        x.push(tp[0]);
        y.push(tp[1]);
        z.push(tp[2]);
    }

    PointCloud::from_xyz(x, y, z)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IcpParams {
    pub max_iterations: usize,
    pub tolerance: f32,
    pub max_correspondence_distance: f32,
}

impl Default for IcpParams {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-5,
            max_correspondence_distance: f32::INFINITY,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IcpResult {
    pub transform: RigidTransform,
    pub fitness: f32,
    pub rmse: f32,
    pub converged: bool,
    pub num_iterations: usize,
}

/// Point-to-point ICP registration using SVD.
///
/// Aligns `source` to `target` by iteratively finding nearest-neighbor
/// correspondences and computing the optimal rigid transform via SVD
/// decomposition of the cross-covariance matrix.
pub fn icp_point_to_point(
    source: &PointCloud,
    target: &PointCloud,
    params: &IcpParams,
) -> IcpResult {
    // Handle empty clouds
    if source.is_empty() || target.is_empty() {
        return IcpResult {
            transform: RigidTransform::identity(),
            fitness: 0.0,
            rmse: 0.0,
            converged: source.is_empty() && target.is_empty(),
            num_iterations: 0,
        };
    }

    // Build KdTree on target
    let target_tree = KdTree::build(target);

    // Working copy of the source cloud that we transform each iteration
    let mut current = apply_transform(source, &RigidTransform::identity());

    // Cumulative transform (accumulated across iterations)
    let mut cumulative = RigidTransform::identity();

    let mut prev_rmse = f32::INFINITY;
    let mut converged = false;
    let mut num_iterations = 0;
    let mut last_rmse = f32::INFINITY;
    let mut last_fitness = 0.0_f32;

    for iter in 0..params.max_iterations {
        num_iterations = iter + 1;

        // Find correspondences between current (transformed) source and target
        let correspondences =
            find_correspondences(&current, &target_tree, params.max_correspondence_distance);

        if correspondences.is_empty() {
            break;
        }

        // Compute RMSE for this iteration
        let rmse = compute_rmse(&correspondences);
        last_rmse = rmse;
        last_fitness = correspondences.len() as f32 / source.len() as f32;

        // Check convergence
        if (prev_rmse - rmse).abs() < params.tolerance {
            converged = true;
            break;
        }
        prev_rmse = rmse;

        // Compute the incremental transform from correspondences
        let incremental = compute_rigid_transform_svd(&current, target, &correspondences);

        // Accumulate: cumulative = incremental composed after cumulative
        cumulative = cumulative.compose(&incremental);

        // Apply incremental transform to current points
        current = apply_transform(&current, &incremental);
    }

    // If we never entered the loop or had no correspondences, compute final metrics
    if num_iterations == 0 {
        let correspondences =
            find_correspondences(&current, &target_tree, params.max_correspondence_distance);
        if !correspondences.is_empty() {
            last_rmse = compute_rmse(&correspondences);
            last_fitness = correspondences.len() as f32 / source.len() as f32;
        }
    }

    IcpResult {
        transform: cumulative,
        fitness: last_fitness,
        rmse: last_rmse,
        converged,
        num_iterations,
    }
}

/// Compute the optimal rigid transform (rotation + translation) that aligns
/// the corresponding source points to target points, using SVD.
fn compute_rigid_transform_svd(
    source: &PointCloud,
    target: &PointCloud,
    correspondences: &[Correspondence],
) -> RigidTransform {
    let n = correspondences.len();
    if n == 0 {
        return RigidTransform::identity();
    }

    // Compute centroids of corresponding source and target points
    let mut src_centroid = Vector3::new(0.0_f32, 0.0, 0.0);
    let mut tgt_centroid = Vector3::new(0.0_f32, 0.0, 0.0);

    for c in correspondences {
        let si = c.source_index;
        let ti = c.target_index;
        src_centroid += Vector3::new(source.x[si], source.y[si], source.z[si]);
        tgt_centroid += Vector3::new(target.x[ti], target.y[ti], target.z[ti]);
    }

    let n_f = n as f32;
    src_centroid /= n_f;
    tgt_centroid /= n_f;

    // Build cross-covariance matrix H = sum (src_i - src_centroid)(tgt_i - tgt_centroid)^T
    let mut h = Matrix3::<f32>::zeros();

    for c in correspondences {
        let si = c.source_index;
        let ti = c.target_index;
        let src_pt = Vector3::new(source.x[si], source.y[si], source.z[si]) - src_centroid;
        let tgt_pt = Vector3::new(target.x[ti], target.y[ti], target.z[ti]) - tgt_centroid;
        h += src_pt * tgt_pt.transpose();
    }

    // SVD of H
    let svd = SVD::new(h, true, true);
    let u = svd.u.expect("SVD should produce U matrix");
    let mut v_t = svd.v_t.expect("SVD should produce V^T matrix");

    // Handle reflection: if det(V * U^T) < 0, negate last column of V
    // V = v_t.transpose()
    let v = v_t.transpose();
    let det = (v * u.transpose()).determinant();

    if det < 0.0 {
        // Negate the last row of v_t (equivalent to negating last column of V)
        v_t[(2, 0)] = -v_t[(2, 0)];
        v_t[(2, 1)] = -v_t[(2, 1)];
        v_t[(2, 2)] = -v_t[(2, 2)];
    }

    let rotation = v_t.transpose() * u.transpose();
    let translation = tgt_centroid - rotation * src_centroid;

    RigidTransform {
        rotation: mat3_to_arrays(&rotation),
        translation: [translation[0], translation[1], translation[2]],
    }
}

/// Compute the root mean square error from correspondence distances.
pub(crate) fn compute_rmse(correspondences: &[Correspondence]) -> f32 {
    if correspondences.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = correspondences
        .iter()
        .map(|c| c.distance * c.distance)
        .sum();
    (sum_sq / correspondences.len() as f32).sqrt()
}

/// Convert a nalgebra Matrix3 to a [[f32; 3]; 3] array (row-major).
fn mat3_to_arrays(m: &Matrix3<f32>) -> [[f32; 3]; 3] {
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
}

/// Convert a [[f32; 3]; 3] array to a nalgebra Matrix3.
fn mat3_from_arrays(a: &[[f32; 3]; 3]) -> Matrix3<f32> {
    Matrix3::new(
        a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use pointclouds_core::PointCloud;
    use proptest::prelude::*;

    /// Helper: create a simple cube-like point cloud with 8 corners.
    fn cube_cloud() -> PointCloud {
        PointCloud::from_xyz(
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        )
    }

    /// Helper: create a translated version of a cloud.
    fn translate_cloud(cloud: &PointCloud, dx: f32, dy: f32, dz: f32) -> PointCloud {
        let t = RigidTransform {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [dx, dy, dz],
        };
        apply_transform(cloud, &t)
    }

    #[test]
    fn identity_transform() {
        let cloud = cube_cloud();
        let params = IcpParams::default();
        let result = icp_point_to_point(&cloud, &cloud, &params);

        assert!(
            result.transform.is_identity(1e-4),
            "Transform should be near identity: {:?}",
            result.transform
        );
        assert!(
            result.rmse < 1e-4,
            "RMSE should be near 0, got {}",
            result.rmse
        );
        assert_relative_eq!(result.fitness, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn known_translation() {
        let source = cube_cloud();
        let target = translate_cloud(&source, 1.0, 0.0, 0.0);
        let params = IcpParams {
            max_iterations: 100,
            tolerance: 1e-8,
            max_correspondence_distance: f32::INFINITY,
        };

        let result = icp_point_to_point(&source, &target, &params);

        assert!(result.converged, "ICP should converge");
        assert!(
            result.rmse < 1e-3,
            "RMSE should be near 0, got {}",
            result.rmse
        );

        // Translation should be approximately [1, 0, 0]
        let t = result.transform.translation;
        assert_relative_eq!(t[0], 1.0, epsilon = 0.05);
        assert_relative_eq!(t[1], 0.0, epsilon = 0.05);
        assert_relative_eq!(t[2], 0.0, epsilon = 0.05);
    }

    #[test]
    fn known_rotation_small_angle_z() {
        // ICP is a local optimizer and struggles with large rotations.
        // Use a 30-degree rotation which is well within convergence basin.
        // Use a dense asymmetric point cloud centered at the origin.
        let angle: f32 = std::f32::consts::FRAC_PI_6; // 30 degrees
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Create a dense cloud: points along an arc + some off-axis points
        let mut sx = Vec::new();
        let mut sy = Vec::new();
        let mut sz = Vec::new();
        for i in 0..40 {
            let t = i as f32 * 0.25 - 5.0;
            sx.push(t);
            sy.push(0.0);
            sz.push(0.0);
        }
        // Add asymmetric off-axis points
        for i in 0..20 {
            let t = i as f32 * 0.25;
            sx.push(0.0);
            sy.push(t);
            sz.push(0.0);
        }

        let source = PointCloud::from_xyz(sx, sy, sz);

        // Apply rotation manually
        let rot_transform = RigidTransform {
            rotation: [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        };
        let target = apply_transform(&source, &rot_transform);

        let params = IcpParams {
            max_iterations: 200,
            tolerance: 1e-10,
            max_correspondence_distance: f32::INFINITY,
        };

        let result = icp_point_to_point(&source, &target, &params);

        assert!(result.converged, "ICP should converge");
        assert!(
            result.rmse < 0.05,
            "RMSE should be small, got {}",
            result.rmse
        );

        // Verify the transform recovers the rotation by checking that
        // applying it to source produces something close to target.
        let transformed = apply_transform(&source, &result.transform);
        for i in 0..source.len() {
            assert_relative_eq!(transformed.x[i], target.x[i], epsilon = 0.15);
            assert_relative_eq!(transformed.y[i], target.y[i], epsilon = 0.15);
            assert_relative_eq!(transformed.z[i], target.z[i], epsilon = 0.15);
        }

        // The rotation matrix should approximate Rz(30deg)
        let r = result.transform.rotation;
        assert_relative_eq!(r[0][0], cos_a, epsilon = 0.1);
        assert_relative_eq!(r[0][1], -sin_a, epsilon = 0.1);
        assert_relative_eq!(r[1][0], sin_a, epsilon = 0.1);
        assert_relative_eq!(r[1][1], cos_a, epsilon = 0.1);
        assert_relative_eq!(r[2][2], 1.0, epsilon = 0.1);
    }

    #[test]
    fn converges_flag() {
        let cloud = cube_cloud();
        let params = IcpParams::default();
        let result = icp_point_to_point(&cloud, &cloud, &params);

        assert!(result.converged, "ICP on identical clouds should converge");
    }

    #[test]
    fn empty_clouds() {
        let empty = PointCloud::new();
        let params = IcpParams::default();
        let result = icp_point_to_point(&empty, &empty, &params);

        assert!(
            result.transform.is_identity(1e-6),
            "Empty clouds should give identity transform"
        );
        assert_eq!(result.num_iterations, 0);
    }

    #[test]
    fn empty_source_nonempty_target() {
        let empty = PointCloud::new();
        let target = cube_cloud();
        let params = IcpParams::default();
        let result = icp_point_to_point(&empty, &target, &params);

        assert!(result.transform.is_identity(1e-6));
        assert_eq!(result.num_iterations, 0);
        assert!(!result.converged);
    }

    #[test]
    fn max_correspondence_distance_filters() {
        // Source at x = 0, 1, 2, ..., 9
        // Target at x = 0, 1, 2, ..., 9 + 100 (one far-away point)
        let source = PointCloud::from_xyz(
            (0..10).map(|i| i as f32).collect(),
            vec![0.0; 10],
            vec![0.0; 10],
        );
        // Target is the same, but shifted by 0.1 so we have slightly different positions
        let target = PointCloud::from_xyz(
            (0..10).map(|i| i as f32 + 0.1).collect(),
            vec![0.0; 10],
            vec![0.0; 10],
        );

        // With tight max_correspondence_distance vs loose
        let tight_params = IcpParams {
            max_iterations: 1,
            tolerance: 1e-8,
            max_correspondence_distance: 0.01, // Very tight: 0.1 offset > 0.01
            ..IcpParams::default()
        };
        let loose_params = IcpParams {
            max_iterations: 1,
            tolerance: 1e-8,
            max_correspondence_distance: f32::INFINITY,
            ..IcpParams::default()
        };

        let tight_result = icp_point_to_point(&source, &target, &tight_params);
        let loose_result = icp_point_to_point(&source, &target, &loose_params);

        // Tight distance should find fewer (or zero) correspondences
        assert!(
            tight_result.fitness <= loose_result.fitness,
            "Tight distance filter should reduce fitness: tight={}, loose={}",
            tight_result.fitness,
            loose_result.fitness,
        );
    }

    #[test]
    fn apply_transform_identity() {
        let cloud = cube_cloud();
        let identity = RigidTransform::identity();
        let result = apply_transform(&cloud, &identity);

        for i in 0..cloud.len() {
            assert_relative_eq!(result.x[i], cloud.x[i], epsilon = 1e-6);
            assert_relative_eq!(result.y[i], cloud.y[i], epsilon = 1e-6);
            assert_relative_eq!(result.z[i], cloud.z[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn apply_transform_translation() {
        let cloud = PointCloud::from_xyz(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]);
        let t = RigidTransform {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [10.0, 20.0, 30.0],
        };
        let result = apply_transform(&cloud, &t);

        assert_relative_eq!(result.x[0], 11.0, epsilon = 1e-6);
        assert_relative_eq!(result.y[0], 23.0, epsilon = 1e-6);
        assert_relative_eq!(result.z[0], 35.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], 12.0, epsilon = 1e-6);
        assert_relative_eq!(result.y[1], 24.0, epsilon = 1e-6);
        assert_relative_eq!(result.z[1], 36.0, epsilon = 1e-6);
    }

    #[test]
    fn compose_transforms() {
        // T1: translate by (1, 0, 0)
        let t1 = RigidTransform {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [1.0, 0.0, 0.0],
        };
        // T2: translate by (0, 2, 0)
        let t2 = RigidTransform {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 2.0, 0.0],
        };

        // Composing T1 then T2 should give translation (1, 2, 0)
        let composed = t1.compose(&t2);
        assert_relative_eq!(composed.translation[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(composed.translation[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(composed.translation[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn compose_rotation_then_translation() {
        // T1: rotate 90 degrees around Z
        let t1 = RigidTransform {
            rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        };
        // T2: translate by (1, 0, 0)
        let t2 = RigidTransform {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [1.0, 0.0, 0.0],
        };

        // Apply T1 then T2 to point (1, 0, 0):
        // After T1: (0, 1, 0) (rotated 90 deg)
        // After T2: (1, 1, 0) (translated)
        let composed = t1.compose(&t2);
        let p = composed.apply_to_point(&[1.0, 0.0, 0.0]);
        assert_relative_eq!(p[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(p[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(p[2], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn apply_to_point_rotation() {
        // Rotate 90 degrees around Z: (1, 0, 0) -> (0, 1, 0)
        let t = RigidTransform {
            rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        };
        let result = t.apply_to_point(&[1.0, 0.0, 0.0]);
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result[2], 0.0, epsilon = 1e-6);
    }

    proptest! {
        #[test]
        fn icp_on_identity_gives_small_rmse(
            pts in prop::collection::vec(
                (-10.0f32..10.0f32, -10.0f32..10.0f32, -10.0f32..10.0f32),
                4..50
            ),
        ) {
            let cloud = PointCloud::from_xyz(
                pts.iter().map(|p| p.0).collect(),
                pts.iter().map(|p| p.1).collect(),
                pts.iter().map(|p| p.2).collect(),
            );
            let params = IcpParams::default();
            let result = icp_point_to_point(&cloud, &cloud, &params);

            prop_assert!(
                result.rmse < 0.01,
                "RMSE should be tiny for identity case, got {}",
                result.rmse
            );
        }
    }
}
