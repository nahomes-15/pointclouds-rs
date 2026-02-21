use pointclouds_core::PointCloud;
use pointclouds_spatial::KdTree;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correspondence {
    pub source_index: usize,
    pub target_index: usize,
    pub distance: f32,
}

/// Find correspondences between source points and the nearest points in the
/// target cloud (represented by its KdTree).
///
/// For each point in `source`, the nearest neighbor in `target_tree` is found.
/// Only correspondences with distance <= `max_distance` are returned.
pub fn find_correspondences(
    source: &PointCloud,
    target_tree: &KdTree,
    max_distance: f32,
) -> Vec<Correspondence> {
    let mut correspondences = Vec::with_capacity(source.len());

    for i in 0..source.len() {
        let query = [source.x[i], source.y[i], source.z[i]];
        let (indices, distances) = target_tree.knn(&query, 1);

        if let (Some(&target_idx), Some(&dist)) = (indices.first(), distances.first()) {
            if dist <= max_distance {
                correspondences.push(Correspondence {
                    source_index: i,
                    target_index: target_idx,
                    distance: dist,
                });
            }
        }
    }

    correspondences
}

#[cfg(test)]
mod tests {
    use super::*;
    use pointclouds_core::PointCloud;
    use pointclouds_spatial::KdTree;

    #[test]
    fn find_correspondences_identical_clouds() {
        let cloud = PointCloud::from_xyz(
            vec![0.0, 1.0, 2.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        );
        let tree = KdTree::build(&cloud);

        let corrs = find_correspondences(&cloud, &tree, f32::INFINITY);

        assert_eq!(corrs.len(), 3);
        for c in &corrs {
            assert_eq!(c.source_index, c.target_index);
            assert!(
                c.distance.abs() < 1e-6,
                "Expected distance ~0, got {}",
                c.distance
            );
        }
    }

    #[test]
    fn find_correspondences_with_max_distance() {
        // Source: points at x = 0, 1, 10
        // Target: points at x = 0, 1, 2
        let source = PointCloud::from_xyz(
            vec![0.0, 1.0, 10.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        );
        let target = PointCloud::from_xyz(
            vec![0.0, 1.0, 2.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        );
        let tree = KdTree::build(&target);

        // With max_distance = 3.0, the point at x=10 should be excluded
        // (nearest target point is x=2, distance=8)
        let corrs = find_correspondences(&source, &tree, 3.0);

        assert_eq!(corrs.len(), 2);
        assert_eq!(corrs[0].source_index, 0);
        assert_eq!(corrs[1].source_index, 1);
    }

    #[test]
    fn find_correspondences_empty_source() {
        let source = PointCloud::new();
        let target = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
        let tree = KdTree::build(&target);

        let corrs = find_correspondences(&source, &tree, f32::INFINITY);
        assert!(corrs.is_empty());
    }

    #[test]
    fn find_correspondences_empty_target() {
        let source = PointCloud::from_xyz(vec![1.0], vec![2.0], vec![3.0]);
        let target = PointCloud::new();
        let tree = KdTree::build(&target);

        let corrs = find_correspondences(&source, &tree, f32::INFINITY);
        assert!(corrs.is_empty());
    }
}
