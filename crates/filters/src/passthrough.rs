use pointclouds_core::PointCloud;

pub fn passthrough_filter(cloud: &PointCloud, axis: char, min: f32, max: f32) -> PointCloud {
    if cloud.is_empty() {
        return PointCloud::new();
    }

    let mut keep = Vec::new();
    for i in 0..cloud.len() {
        let v = match axis {
            'x' | 'X' => cloud.x[i],
            'y' | 'Y' => cloud.y[i],
            'z' | 'Z' => cloud.z[i],
            _ => panic!("axis must be one of x/y/z"),
        };

        if v.is_finite() && v >= min && v <= max {
            keep.push(i);
        }
    }

    cloud.select(&keep)
}

#[cfg(test)]
mod tests {
    use super::passthrough_filter;
    use pointclouds_core::PointCloud;
    use proptest::prelude::*;

    fn sample_cloud() -> PointCloud {
        // 5 points with distinct x, y, z values for easy reasoning
        PointCloud::from_xyz(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![100.0, 200.0, 300.0, 400.0, 500.0],
        )
    }

    #[test]
    fn passthrough_x_axis() {
        let cloud = sample_cloud();
        let result = passthrough_filter(&cloud, 'x', 2.0, 4.0);
        assert_eq!(result.len(), 3);
        assert_eq!(result.x, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn passthrough_y_axis() {
        let cloud = sample_cloud();
        let result = passthrough_filter(&cloud, 'y', 20.0, 40.0);
        assert_eq!(result.len(), 3);
        assert_eq!(result.y, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn passthrough_z_axis() {
        let cloud = sample_cloud();
        let result = passthrough_filter(&cloud, 'z', 200.0, 400.0);
        assert_eq!(result.len(), 3);
        assert_eq!(result.z, vec![200.0, 300.0, 400.0]);
    }

    #[test]
    fn passthrough_empty_cloud() {
        let cloud = PointCloud::new();
        let result = passthrough_filter(&cloud, 'x', 0.0, 10.0);
        assert!(result.is_empty());
    }

    #[test]
    fn passthrough_no_points_in_range() {
        let cloud = sample_cloud();
        // x values are 1..5, so range 10..20 excludes everything
        let result = passthrough_filter(&cloud, 'x', 10.0, 20.0);
        assert!(result.is_empty());
    }

    #[test]
    fn passthrough_all_points_in_range() {
        let cloud = sample_cloud();
        // x values are 1..5, range 0..10 includes everything
        let result = passthrough_filter(&cloud, 'x', 0.0, 10.0);
        assert_eq!(result.len(), cloud.len());
    }

    proptest! {
        #[test]
        fn passthrough_result_within_bounds(
            pts in prop::collection::vec(
                (-100.0f32..100.0f32, -100.0f32..100.0f32, -100.0f32..100.0f32),
                1..500
            ),
            min_val in -50.0f32..0.0f32,
            max_val in 0.0f32..50.0f32,
        ) {
            let cloud = PointCloud::from_xyz(
                pts.iter().map(|p| p.0).collect(),
                pts.iter().map(|p| p.1).collect(),
                pts.iter().map(|p| p.2).collect(),
            );
            let result = passthrough_filter(&cloud, 'x', min_val, max_val);
            for i in 0..result.len() {
                prop_assert!(result.x[i] >= min_val, "x={} < min={}", result.x[i], min_val);
                prop_assert!(result.x[i] <= max_val, "x={} > max={}", result.x[i], max_val);
            }
        }
    }
}
