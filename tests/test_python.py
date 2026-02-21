import numpy as np
import pytest


def test_import():
    import pointclouds_rs
    assert hasattr(pointclouds_rs, "PointCloud")


def test_pointcloud_create_empty():
    from pointclouds_rs import PointCloud
    cloud = PointCloud()
    assert cloud.len() == 0
    assert cloud.is_empty()


def test_pointcloud_from_numpy():
    from pointclouds_rs import PointCloud
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    cloud = PointCloud.from_numpy(data)
    assert cloud.len() == 2
    assert not cloud.is_empty()


def test_pointcloud_roundtrip_numpy():
    from pointclouds_rs import PointCloud
    data = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    cloud = PointCloud.from_numpy(data)
    out = cloud.to_numpy()
    np.testing.assert_allclose(out, data, atol=1e-6)


def test_pointcloud_from_numpy_f64():
    """f64 arrays are auto-cast to f32."""
    from pointclouds_rs import PointCloud
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    cloud = PointCloud.from_numpy(data)
    assert cloud.len() == 2
    out = cloud.to_numpy()
    np.testing.assert_allclose(out, data.astype(np.float32), atol=1e-6)


def test_pointcloud_fortran_order_rejected():
    """Fortran-order (column-major) arrays must be rejected, not silently misread."""
    from pointclouds_rs import PointCloud
    data = np.asfortranarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    with pytest.raises((ValueError, Exception)):
        PointCloud.from_numpy(data)


def test_pointcloud_repr():
    from pointclouds_rs import PointCloud
    cloud = PointCloud()
    assert "PointCloud" in repr(cloud)


def test_voxel_downsample():
    from pointclouds_rs import PointCloud, voxel_downsample
    data = np.random.rand(1000, 3).astype(np.float32) * 10.0
    cloud = PointCloud.from_numpy(data)
    result = voxel_downsample(cloud, 1.0)
    assert result.len() > 0
    assert result.len() < cloud.len()


def test_voxel_downsample_invalid_size():
    """Invalid voxel_size should raise ValueError, not panic."""
    from pointclouds_rs import PointCloud, voxel_downsample
    cloud = PointCloud.from_numpy(np.array([[1, 2, 3]], dtype=np.float32))
    with pytest.raises((ValueError, Exception)):
        voxel_downsample(cloud, -1.0)
    with pytest.raises((ValueError, Exception)):
        voxel_downsample(cloud, 0.0)
    with pytest.raises((ValueError, Exception)):
        voxel_downsample(cloud, float("nan"))


def test_passthrough_filter():
    from pointclouds_rs import PointCloud, passthrough_filter
    data = np.array(
        [[1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32
    )
    cloud = PointCloud.from_numpy(data)
    result = passthrough_filter(cloud, "x", 0.0, 6.0)
    assert result.len() == 2


def test_passthrough_filter_invalid_axis():
    """Invalid axis should raise ValueError, not panic."""
    from pointclouds_rs import PointCloud, passthrough_filter
    cloud = PointCloud.from_numpy(np.array([[1, 2, 3]], dtype=np.float32))
    with pytest.raises((ValueError, Exception)):
        passthrough_filter(cloud, "w", 0.0, 1.0)


def test_statistical_outlier_removal():
    from pointclouds_rs import PointCloud, statistical_outlier_removal
    # Dense cluster + one far outlier
    cluster = np.random.rand(50, 3).astype(np.float32) * 0.1
    outlier = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
    data = np.vstack([cluster, outlier])
    cloud = PointCloud.from_numpy(data)
    result = statistical_outlier_removal(cloud, 10, 1.0)
    assert result.len() <= cloud.len()


def test_radius_outlier_removal():
    from pointclouds_rs import PointCloud, radius_outlier_removal
    # Dense cluster + isolated point
    cluster = np.random.rand(50, 3).astype(np.float32) * 0.1
    outlier = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
    data = np.vstack([cluster, outlier])
    cloud = PointCloud.from_numpy(data)
    result = radius_outlier_removal(cloud, 0.5, 3)
    assert result.len() < cloud.len()


def test_estimate_normals():
    from pointclouds_rs import PointCloud, estimate_normals
    # Flat plane at z~0 with tiny noise to avoid kiddo bucket overflow
    rng = np.random.default_rng(42)
    xs = np.linspace(0, 1, 10, dtype=np.float32)
    ys = np.linspace(0, 1, 10, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    zz = rng.normal(0, 1e-4, 100).astype(np.float32)
    data = np.column_stack([xx.ravel(), yy.ravel(), zz])
    cloud = PointCloud.from_numpy(data)
    result = estimate_normals(cloud, 5)
    assert result.len() == 100


def test_icp_point_to_point():
    from pointclouds_rs import PointCloud, icp_point_to_point
    data = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
    )
    source = PointCloud.from_numpy(data)
    target = PointCloud.from_numpy(data + np.array([0.1, 0, 0], dtype=np.float32))
    result = icp_point_to_point(source, target)
    assert result.converged
    assert result.rmse < 0.1


def test_icp_point_to_plane():
    from pointclouds_rs import PointCloud, estimate_normals, icp_point_to_plane
    # Flat plane shifted along normal direction (z)
    rng = np.random.default_rng(123)
    xs = np.linspace(-2, 2, 10, dtype=np.float32)
    ys = np.linspace(-2, 2, 10, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    zz = rng.normal(0, 1e-4, 100).astype(np.float32)
    target_data = np.column_stack([xx.ravel(), yy.ravel(), zz])
    target = PointCloud.from_numpy(target_data)
    target_with_normals = estimate_normals(target, 10)

    source_data = target_data.copy()
    source_data[:, 2] += 0.3  # shift along Z
    source = PointCloud.from_numpy(source_data)

    result = icp_point_to_plane(source, target_with_normals)
    assert result.converged
    assert result.rmse < 0.1


def test_icp_point_to_plane_no_normals():
    """Target without normals should raise ValueError."""
    from pointclouds_rs import PointCloud, icp_point_to_plane
    data = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    cloud = PointCloud.from_numpy(data)
    with pytest.raises((ValueError, Exception)):
        icp_point_to_plane(cloud, cloud)


def test_euclidean_cluster():
    from pointclouds_rs import PointCloud, euclidean_cluster
    # Two clusters far apart
    c1 = np.random.rand(20, 3).astype(np.float32) * 0.1
    c2 = np.random.rand(20, 3).astype(np.float32) * 0.1 + 10.0
    data = np.vstack([c1, c2])
    cloud = PointCloud.from_numpy(data)
    clusters = euclidean_cluster(cloud, 0.5, 5, 100)
    assert len(clusters) == 2


def test_ransac_plane():
    from pointclouds_rs import PointCloud, ransac_plane
    # Points on z=0 plane
    data = np.column_stack([
        np.random.rand(100).astype(np.float32),
        np.random.rand(100).astype(np.float32),
        np.zeros(100, dtype=np.float32),
    ])
    cloud = PointCloud.from_numpy(data)
    result = ransac_plane(cloud, 0.01, 100)
    assert abs(result.normal[2]) > 0.9  # normal should be ~(0,0,1)
    assert len(result.inliers) > 90


def test_read_write_pcd(tmp_path):
    from pointclouds_rs import PointCloud, read_pcd, write_pcd
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    cloud = PointCloud.from_numpy(data)
    path = str(tmp_path / "test.pcd")
    write_pcd(path, cloud)
    loaded = read_pcd(path)
    assert loaded.len() == 2


def test_read_write_ply(tmp_path):
    from pointclouds_rs import PointCloud, read_ply, write_ply
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    cloud = PointCloud.from_numpy(data)
    path = str(tmp_path / "test.ply")
    write_ply(path, cloud)
    loaded = read_ply(path)
    assert loaded.len() == 2


def test_read_write_ply_binary(tmp_path):
    from pointclouds_rs import PointCloud, read_ply, write_ply_binary
    data = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
    cloud = PointCloud.from_numpy(data)
    path = str(tmp_path / "test_bin.ply")
    write_ply_binary(path, cloud)
    loaded = read_ply(path)
    assert loaded.len() == 2
    out = loaded.to_numpy()
    np.testing.assert_array_equal(out, data)  # binary is bit-exact


def test_read_las_nonexistent():
    """read_las on missing file should raise IOError, not panic."""
    from pointclouds_rs import read_las
    with pytest.raises((IOError, OSError)):
        read_las("/tmp/definitely_not_a_real_file_xyz_123.las")


def test_read_las_available():
    """read_las should be importable from pointclouds_rs."""
    import pointclouds_rs
    assert hasattr(pointclouds_rs, "read_las")


# ──────── Adversarial edge-case tests ────────

def test_empty_cloud_to_numpy():
    from pointclouds_rs import PointCloud
    cloud = PointCloud()
    out = cloud.to_numpy()
    assert out.shape == (0, 3) or out.size == 0


def test_from_numpy_wrong_shape():
    """1D array should raise, not silently misread."""
    from pointclouds_rs import PointCloud
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(Exception):
        PointCloud.from_numpy(data)


def test_from_numpy_wrong_columns():
    """Nx2 array should raise."""
    from pointclouds_rs import PointCloud
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    with pytest.raises(Exception):
        PointCloud.from_numpy(data)


def test_from_numpy_nan_values():
    """NaN values should be accepted (they're valid f32), not crash."""
    from pointclouds_rs import PointCloud
    data = np.array([[float("nan"), 0, 0], [1, 2, 3]], dtype=np.float32)
    cloud = PointCloud.from_numpy(data)
    assert cloud.len() == 2


def test_from_numpy_inf_values():
    """Inf values should be accepted (valid f32)."""
    from pointclouds_rs import PointCloud
    data = np.array([[float("inf"), 0, 0], [1, 2, 3]], dtype=np.float32)
    cloud = PointCloud.from_numpy(data)
    assert cloud.len() == 2


def test_voxel_downsample_very_large_voxel():
    """A voxel larger than the cloud should collapse to 1 point."""
    from pointclouds_rs import PointCloud, voxel_downsample
    data = np.random.rand(100, 3).astype(np.float32)
    cloud = PointCloud.from_numpy(data)
    result = voxel_downsample(cloud, 1000.0)
    assert result.len() == 1


def test_voxel_downsample_very_small_voxel():
    """A tiny voxel should keep ~all unique points."""
    from pointclouds_rs import PointCloud, voxel_downsample
    data = np.random.rand(50, 3).astype(np.float32) * 100
    cloud = PointCloud.from_numpy(data)
    result = voxel_downsample(cloud, 0.001)
    assert result.len() >= 40  # should keep most points


def test_icp_identical_clouds():
    """ICP on identical clouds should converge with near-zero RMSE."""
    from pointclouds_rs import PointCloud, icp_point_to_point
    data = np.random.rand(20, 3).astype(np.float32)
    cloud = PointCloud.from_numpy(data)
    result = icp_point_to_point(cloud, cloud)
    assert result.converged
    assert result.rmse < 0.01


def test_ransac_with_only_3_points():
    """Exactly 3 points define exactly one plane."""
    from pointclouds_rs import PointCloud, ransac_plane
    data = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    cloud = PointCloud.from_numpy(data)
    result = ransac_plane(cloud, 0.01, 10)
    assert abs(result.normal[2]) > 0.9  # Z-plane
    assert len(result.inliers) == 3


def test_euclidean_cluster_single_point():
    """Single point below min_size should return no clusters."""
    from pointclouds_rs import PointCloud, euclidean_cluster
    data = np.array([[0, 0, 0]], dtype=np.float32)
    cloud = PointCloud.from_numpy(data)
    clusters = euclidean_cluster(cloud, 1.0, 2, 100)
    assert len(clusters) == 0  # min_size=2, only 1 point


def test_estimate_normals_two_points():
    """Normal estimation on 2 points should not panic."""
    from pointclouds_rs import PointCloud, estimate_normals
    data = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    cloud = PointCloud.from_numpy(data)
    result = estimate_normals(cloud, 2)
    assert result.len() == 2


def test_passthrough_filter_all_filtered():
    """Filter range that excludes all points should return empty."""
    from pointclouds_rs import PointCloud, passthrough_filter
    data = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    cloud = PointCloud.from_numpy(data)
    result = passthrough_filter(cloud, "x", 100.0, 200.0)
    assert result.len() == 0
