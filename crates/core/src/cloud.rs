use crate::{Aabb, CloudView};

#[derive(Debug, Clone, PartialEq)]
pub struct PointCloud {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub normals: Option<Normals>,
    pub colors: Option<Colors>,
    pub intensity: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Normals {
    pub nx: Vec<f32>,
    pub ny: Vec<f32>,
    pub nz: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Colors {
    pub r: Vec<u8>,
    pub g: Vec<u8>,
    pub b: Vec<u8>,
}

impl PointCloud {
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            normals: None,
            colors: None,
            intensity: None,
        }
    }

    pub fn from_xyz(x: Vec<f32>, y: Vec<f32>, z: Vec<f32>) -> Self {
        assert_eq!(x.len(), y.len(), "x and y must have same length");
        assert_eq!(x.len(), z.len(), "x and z must have same length");

        Self {
            x,
            y,
            z,
            normals: None,
            colors: None,
            intensity: None,
        }
    }

    pub fn from_array(data: &[f32], num_points: usize) -> Self {
        assert_eq!(
            data.len(),
            num_points * 3,
            "interleaved xyz input must have num_points * 3 floats"
        );

        let mut x = Vec::with_capacity(num_points);
        let mut y = Vec::with_capacity(num_points);
        let mut z = Vec::with_capacity(num_points);

        for chunk in data.chunks_exact(3).take(num_points) {
            x.push(chunk[0]);
            y.push(chunk[1]);
            z.push(chunk[2]);
        }

        Self::from_xyz(x, y, z)
    }

    pub fn view_from_array(data: &[f32], num_points: usize) -> CloudView<'_> {
        CloudView::from_interleaved_xyz(data, num_points)
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(self.x.len(), self.y.len());
        debug_assert_eq!(self.x.len(), self.z.len());
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    pub fn aabb(&self) -> Aabb {
        Aabb::from_xyz(&self.x, &self.y, &self.z)
    }

    pub fn point(&self, i: usize) -> [f32; 3] {
        [self.x[i], self.y[i], self.z[i]]
    }

    pub fn iter_points(&self) -> impl Iterator<Item = [f32; 3]> + '_ {
        self.x
            .iter()
            .zip(&self.y)
            .zip(&self.z)
            .map(|((x, y), z)| [*x, *y, *z])
    }

    pub fn select(&self, indices: &[usize]) -> Self {
        let mut x = Vec::with_capacity(indices.len());
        let mut y = Vec::with_capacity(indices.len());
        let mut z = Vec::with_capacity(indices.len());

        for &idx in indices {
            assert!(idx < self.len(), "index out of bounds in select");
            x.push(self.x[idx]);
            y.push(self.y[idx]);
            z.push(self.z[idx]);
        }

        let normals = self.normals.as_ref().map(|n| Normals {
            nx: indices.iter().map(|&idx| n.nx[idx]).collect(),
            ny: indices.iter().map(|&idx| n.ny[idx]).collect(),
            nz: indices.iter().map(|&idx| n.nz[idx]).collect(),
        });

        let colors = self.colors.as_ref().map(|c| Colors {
            r: indices.iter().map(|&idx| c.r[idx]).collect(),
            g: indices.iter().map(|&idx| c.g[idx]).collect(),
            b: indices.iter().map(|&idx| c.b[idx]).collect(),
        });

        let intensity = self
            .intensity
            .as_ref()
            .map(|it| indices.iter().map(|&idx| it[idx]).collect());

        Self {
            x,
            y,
            z,
            normals,
            colors,
            intensity,
        }
    }

    /// Select all points NOT in the given index set.
    ///
    /// This is the complement of [`select`]: if `select` returns points at
    /// the given indices, `select_inverse` returns all the rest.
    ///
    /// The returned cloud preserves the relative order of the retained points.
    ///
    /// # Panics
    ///
    /// Panics if any index in `indices` is out of bounds.
    pub fn select_inverse(&self, indices: &[usize]) -> Self {
        let n = self.len();
        let mut exclude = vec![false; n];
        for &idx in indices {
            assert!(idx < n, "index out of bounds in select_inverse");
            exclude[idx] = true;
        }

        let kept: Vec<usize> = (0..n).filter(|&i| !exclude[i]).collect();
        self.select(&kept)
    }

    pub fn to_array(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len() * 3);
        for i in 0..self.len() {
            out.push(self.x[i]);
            out.push(self.y[i]);
            out.push(self.z[i]);
        }
        out
    }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::PointCloud;
    use proptest::prelude::*;

    #[test]
    fn new_is_empty() {
        let cloud = PointCloud::new();
        assert!(cloud.is_empty());
        assert_eq!(cloud.len(), 0);
    }

    #[test]
    fn from_xyz_builds_cloud() {
        let cloud = PointCloud::from_xyz(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]);
        assert_eq!(cloud.len(), 2);
        assert_eq!(cloud.point(0), [1.0, 3.0, 5.0]);
        assert_eq!(cloud.point(1), [2.0, 4.0, 6.0]);
    }

    #[test]
    fn from_array_deinterleaves() {
        let arr = vec![1.0, 10.0, 100.0, 2.0, 20.0, 200.0];
        let cloud = PointCloud::from_array(&arr, 2);
        assert_eq!(cloud.x, vec![1.0, 2.0]);
        assert_eq!(cloud.y, vec![10.0, 20.0]);
        assert_eq!(cloud.z, vec![100.0, 200.0]);
    }

    #[test]
    fn to_array_interleaves() {
        let cloud = PointCloud::from_xyz(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]);
        assert_eq!(cloud.to_array(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn roundtrip_from_array_to_array() {
        let src = vec![0.0, 1.0, 2.0, 3.0, -4.0, 5.0, 6.0, 7.0, 8.0];
        let cloud = PointCloud::from_array(&src, 3);
        assert_eq!(cloud.to_array(), src);
    }

    #[test]
    fn select_subsets_points() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, 1.0, 2.0, 3.0],
            vec![10.0, 11.0, 12.0, 13.0],
            vec![20.0, 21.0, 22.0, 23.0],
        );
        let selected = cloud.select(&[3, 1]);
        assert_eq!(selected.x, vec![3.0, 1.0]);
        assert_eq!(selected.y, vec![13.0, 11.0]);
        assert_eq!(selected.z, vec![23.0, 21.0]);
    }

    #[test]
    fn iter_points_yields_xyz_tuples() {
        let cloud = PointCloud::from_xyz(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]);
        let pts: Vec<[f32; 3]> = cloud.iter_points().collect();
        assert_eq!(pts, vec![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]);
    }

    #[test]
    fn aabb_contains_all_points() {
        let cloud = PointCloud::from_xyz(vec![-1.0, 2.0], vec![3.0, -4.0], vec![5.0, 6.0]);
        let aabb = cloud.aabb();
        for p in cloud.iter_points() {
            assert!(aabb.contains(&p));
        }
    }

    #[test]
    fn aabb_ignores_nan() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, f32::NAN, 2.0],
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        );
        let aabb = cloud.aabb();
        assert!(aabb.contains(&[0.0, 1.0, 4.0]));
        assert!(aabb.contains(&[2.0, 3.0, 6.0]));
        assert!(!aabb.contains(&[f32::NAN, 2.0, 5.0]));
    }

    #[test]
    fn select_inverse_basic() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, 1.0, 2.0, 3.0],
            vec![10.0, 11.0, 12.0, 13.0],
            vec![20.0, 21.0, 22.0, 23.0],
        );
        // Exclude indices 0 and 2
        let inv = cloud.select_inverse(&[0, 2]);
        assert_eq!(inv.len(), 2);
        assert_eq!(inv.x, vec![1.0, 3.0]);
        assert_eq!(inv.y, vec![11.0, 13.0]);
        assert_eq!(inv.z, vec![21.0, 23.0]);
    }

    #[test]
    fn select_inverse_empty_indices() {
        let cloud = PointCloud::from_xyz(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]);
        let inv = cloud.select_inverse(&[]);
        assert_eq!(inv.len(), 2);
        assert_eq!(inv.x, cloud.x);
    }

    #[test]
    fn select_inverse_all_indices() {
        let cloud = PointCloud::from_xyz(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]);
        let inv = cloud.select_inverse(&[0, 1]);
        assert!(inv.is_empty());
    }

    #[test]
    fn select_inverse_with_normals() {
        let mut cloud = PointCloud::from_xyz(vec![0.0, 1.0, 2.0], vec![0.0; 3], vec![0.0; 3]);
        cloud.normals = Some(crate::Normals {
            nx: vec![0.1, 0.2, 0.3],
            ny: vec![0.4, 0.5, 0.6],
            nz: vec![0.7, 0.8, 0.9],
        });
        let inv = cloud.select_inverse(&[1]); // exclude index 1
        assert_eq!(inv.len(), 2);
        let normals = inv.normals.as_ref().unwrap();
        assert_eq!(normals.nx, vec![0.1, 0.3]);
        assert_eq!(normals.ny, vec![0.4, 0.6]);
        assert_eq!(normals.nz, vec![0.7, 0.9]);
    }

    #[test]
    fn select_inverse_duplicate_indices() {
        // Duplicate indices should be treated the same as single occurrence
        let cloud = PointCloud::from_xyz(vec![0.0, 1.0, 2.0], vec![0.0; 3], vec![0.0; 3]);
        let inv = cloud.select_inverse(&[1, 1, 1]);
        assert_eq!(inv.len(), 2);
        assert_eq!(inv.x, vec![0.0, 2.0]);
    }

    #[test]
    #[should_panic]
    fn from_xyz_panics_on_mismatch() {
        let _ = PointCloud::from_xyz(vec![1.0], vec![2.0, 3.0], vec![4.0]);
    }

    proptest! {
        #[test]
        fn roundtrip_preserves_interleaved_data(
            pts in prop::collection::vec((-1000.0f32..1000.0f32, -1000.0f32..1000.0f32, -1000.0f32..1000.0f32), 0..500)
        ) {
            let mut flat = Vec::with_capacity(pts.len() * 3);
            for (x, y, z) in &pts {
                flat.push(*x);
                flat.push(*y);
                flat.push(*z);
            }
            let cloud = PointCloud::from_array(&flat, pts.len());
            prop_assert_eq!(cloud.to_array(), flat);
        }

        #[test]
        fn aabb_contains_all_finite_points(
            pts in prop::collection::vec((-1000.0f32..1000.0f32, -1000.0f32..1000.0f32, -1000.0f32..1000.0f32), 1..500)
        ) {
            let mut x = Vec::with_capacity(pts.len());
            let mut y = Vec::with_capacity(pts.len());
            let mut z = Vec::with_capacity(pts.len());
            for (px, py, pz) in pts {
                x.push(px);
                y.push(py);
                z.push(pz);
            }
            let cloud = PointCloud::from_xyz(x, y, z);
            let aabb = cloud.aabb();
            for p in cloud.iter_points() {
                prop_assert!(aabb.contains(&p));
            }
        }

        #[test]
        fn select_never_changes_length_to_more_than_indices(
            data in prop::collection::vec((-10.0f32..10.0f32, -10.0f32..10.0f32, -10.0f32..10.0f32), 1..200),
            idxs in prop::collection::vec(0usize..200, 0..200)
        ) {
            let n = data.len();
            let cloud = PointCloud::from_xyz(
                data.iter().map(|p| p.0).collect(),
                data.iter().map(|p| p.1).collect(),
                data.iter().map(|p| p.2).collect(),
            );
            let valid: Vec<usize> = idxs.into_iter().filter(|i| *i < n).collect();
            let out = cloud.select(&valid);
            prop_assert_eq!(out.len(), valid.len());
        }
    }
}
