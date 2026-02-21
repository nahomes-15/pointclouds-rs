use pointclouds_core::PointCloud;
use pointclouds_io::{read_pcd, write_pcd};

#[test]
fn pcd_write_then_read_roundtrip() {
    let cloud = PointCloud::from_xyz(
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    );

    let dir = std::env::temp_dir().join("pointclouds_rs_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("roundtrip.pcd");

    write_pcd(&path, &cloud).unwrap();
    let loaded = read_pcd(&path).unwrap();

    assert_eq!(loaded.len(), cloud.len());
    for i in 0..cloud.len() {
        assert!((loaded.x[i] - cloud.x[i]).abs() < 1e-4);
        assert!((loaded.y[i] - cloud.y[i]).abs() < 1e-4);
        assert!((loaded.z[i] - cloud.z[i]).abs() < 1e-4);
    }

    // Cleanup
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn pcd_empty_cloud_roundtrip() {
    let cloud = PointCloud::new();
    let dir = std::env::temp_dir().join("pointclouds_rs_test_empty");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("empty.pcd");

    write_pcd(&path, &cloud).unwrap();
    let loaded = read_pcd(&path).unwrap();
    assert_eq!(loaded.len(), 0);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn read_sample_bunny_pcd() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest_dir).join("data/bunny.pcd");
    if path.exists() {
        let cloud = read_pcd(&path).unwrap();
        assert!(cloud.len() > 0, "bunny.pcd should have points");
    }
}
