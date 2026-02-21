use hashbrown::HashMap;
use pointclouds_core::PointCloud;

#[derive(Default, Clone, Copy)]
struct VoxelAccum {
    sx: f32,
    sy: f32,
    sz: f32,
    n: usize,
}

pub fn voxel_downsample(cloud: &PointCloud, voxel_size: f32) -> PointCloud {
    assert!(
        voxel_size.is_finite() && voxel_size > 0.0,
        "voxel_size must be > 0 and finite"
    );

    if cloud.is_empty() {
        return PointCloud::new();
    }

    let mut bins: HashMap<(i32, i32, i32), VoxelAccum> = HashMap::new();

    for i in 0..cloud.len() {
        let px = cloud.x[i];
        let py = cloud.y[i];
        let pz = cloud.z[i];
        if !px.is_finite() || !py.is_finite() || !pz.is_finite() {
            continue;
        }

        let key = (
            (px / voxel_size).floor() as i32,
            (py / voxel_size).floor() as i32,
            (pz / voxel_size).floor() as i32,
        );

        let entry = bins.entry(key).or_default();
        entry.sx += px;
        entry.sy += py;
        entry.sz += pz;
        entry.n += 1;
    }

    if bins.is_empty() {
        return PointCloud::new();
    }

    let mut keys: Vec<(i32, i32, i32)> = bins.keys().copied().collect();
    keys.sort_unstable();

    let mut x = Vec::with_capacity(keys.len());
    let mut y = Vec::with_capacity(keys.len());
    let mut z = Vec::with_capacity(keys.len());

    for key in keys {
        let a = bins.get(&key).expect("bin key should exist");
        let denom = a.n as f32;
        x.push(a.sx / denom);
        y.push(a.sy / denom);
        z.push(a.sz / denom);
    }

    PointCloud::from_xyz(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::voxel_downsample;
    use pointclouds_core::PointCloud;
    use proptest::prelude::*;

    #[test]
    fn voxel_downsample_reduces_points() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5],
            vec![0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5],
            vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
        );
        let out = voxel_downsample(&cloud, 1.0);
        assert_eq!(out.len(), 1);
        assert!((out.x[0] - 0.25).abs() < 1e-6);
        assert!((out.y[0] - 0.25).abs() < 1e-6);
        assert!((out.z[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn voxel_downsample_empty_cloud() {
        let out = voxel_downsample(&PointCloud::new(), 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn voxel_downsample_single_point() {
        let cloud = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
        let out = voxel_downsample(&cloud, 1.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out.point(0), [1.0, 2.0, 3.0]);
    }

    proptest! {
        #[test]
        fn voxel_downsample_never_increases_points(
            pts in prop::collection::vec((-100.0f32..100.0f32, -100.0f32..100.0f32, -100.0f32..100.0f32), 1..3000),
            voxel_size in 0.01f32..10.0f32,
        ) {
            let cloud = PointCloud::from_xyz(
                pts.iter().map(|p| p.0).collect(),
                pts.iter().map(|p| p.1).collect(),
                pts.iter().map(|p| p.2).collect(),
            );
            let out = voxel_downsample(&cloud, voxel_size);
            prop_assert!(out.len() <= cloud.len());
        }
    }
}
