//! Real-world-style integration tests that exercise end-to-end pipelines
//! the way a production user would: KITTI-like obstacle detection, realistic
//! ICP registration, multi-format IO roundtrips, and large-cloud scaling.

use pointclouds_core::{Colors, Normals, PointCloud};
use pointclouds_filters::{statistical_outlier_removal, voxel_downsample};
use pointclouds_io::{read_ply, write_pcd, write_pcd_binary, write_ply, write_ply_binary};
use pointclouds_normals::estimate_normals;
use pointclouds_registration::{
    apply_transform, icp_point_to_plane, icp_point_to_point, IcpParams, RigidTransform,
};
use pointclouds_segmentation::{euclidean_cluster, ransac_plane_seeded};
use rand::prelude::*;
use std::time::Instant;

// ────────────────── helpers ──────────────────

/// Build a KITTI-like synthetic scene with a deterministic seed.
fn build_kitti_scene(seed: u64) -> (PointCloud, usize) {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();

    // Ground plane: ~50k points, x/y in [-20, 20], z ≈ 0 with small noise
    let ground_count = 50_000;
    for _ in 0..ground_count {
        x.push(rng.gen_range(-20.0f32..20.0));
        y.push(rng.gen_range(-20.0f32..20.0));
        z.push(rng.gen_range(-0.05f32..0.05));
    }

    // 3 obstacle clusters, ~2k points each, box-like around known centers
    let centers = [[5.0f32, 3.0, 1.0], [-8.0, -5.0, 0.8], [12.0, -10.0, 1.5]];
    let obstacle_pts_each = 2_000;
    for center in &centers {
        for _ in 0..obstacle_pts_each {
            x.push(center[0] + rng.gen_range(-0.8f32..0.8));
            y.push(center[1] + rng.gen_range(-0.8f32..0.8));
            z.push(center[2] + rng.gen_range(-0.5f32..0.5));
        }
    }

    // Outlier noise: ~1k random points scattered widely
    let outlier_count = 1_000;
    for _ in 0..outlier_count {
        x.push(rng.gen_range(-30.0f32..30.0));
        y.push(rng.gen_range(-30.0f32..30.0));
        z.push(rng.gen_range(-5.0f32..10.0));
    }

    let _total = ground_count + obstacle_pts_each * 3 + outlier_count;
    (PointCloud::from_xyz(x, y, z), ground_count)
}

/// Build a hemisphere surface with deterministic points.
fn build_hemisphere(n: usize, seed: u64, radius: f32) -> PointCloud {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);

    for _ in 0..n {
        // Uniform sampling on upper hemisphere via rejection
        loop {
            let px = rng.gen_range(-1.0f32..1.0);
            let py = rng.gen_range(-1.0f32..1.0);
            let r2 = px * px + py * py;
            if r2 < 1.0 {
                let pz = (1.0 - r2).sqrt();
                x.push(px * radius);
                y.push(py * radius);
                z.push(pz * radius);
                break;
            }
        }
    }
    PointCloud::from_xyz(x, y, z)
}

// ────────────────── Test 1: KITTI-like full pipeline ──────────────────

#[test]
fn test_full_pipeline_kitti_like() {
    let t0 = Instant::now();

    let (scene, _ground_count) = build_kitti_scene(42);
    let raw_count = scene.len();
    assert_eq!(raw_count, 57_000);

    // Step 1: voxel downsample
    let t_ds = Instant::now();
    let downsampled = voxel_downsample(&scene, 0.2);
    let ds_time = t_ds.elapsed();
    let ds_count = downsampled.len();

    // Downsample ratio should be reasonable (keep 5%–90%)
    let ratio = ds_count as f64 / raw_count as f64;
    assert!(
        ratio > 0.05 && ratio < 0.90,
        "downsample ratio {:.2} out of expected range",
        ratio
    );

    // Step 2: statistical outlier removal
    let t_sor = Instant::now();
    let cleaned = statistical_outlier_removal(&downsampled, 20, 2.0);
    let sor_time = t_sor.elapsed();

    // Should remove some outliers but keep most points
    assert!(cleaned.len() > ds_count / 2, "SOR removed too many points");
    assert!(cleaned.len() < ds_count, "SOR removed nothing");

    // Step 3: RANSAC ground plane
    let t_ransac = Instant::now();
    let (plane, inliers) = ransac_plane_seeded(&cleaned, 0.15, 500, 42);
    let ransac_time = t_ransac.elapsed();

    // Plane normal should be approximately vertical
    assert!(
        plane.normal[2].abs() > 0.9,
        "plane normal z={:.3}, expected ~vertical",
        plane.normal[2]
    );

    // Most cleaned points come from ground, so inliers should be a majority
    let inlier_ratio = inliers.len() as f64 / cleaned.len() as f64;
    assert!(
        inlier_ratio > 0.3,
        "ground inlier ratio {:.2} too low",
        inlier_ratio
    );

    // Step 4: separate obstacles from ground
    let t_seg = Instant::now();
    let obstacles = cleaned.select_inverse(&inliers);
    assert!(
        !obstacles.is_empty(),
        "no obstacle points after ground removal"
    );

    // Step 5: cluster obstacles
    let clusters = euclidean_cluster(&obstacles, 1.0, 10, 10_000);
    let seg_time = t_seg.elapsed();

    // Should find approximately 3 obstacle clusters
    assert!(
        clusters.len() >= 2 && clusters.len() <= 6,
        "expected ~3 clusters, found {}",
        clusters.len()
    );

    // Verify no NaN in final obstacle cloud
    for i in 0..obstacles.len() {
        let p = obstacles.point(i);
        assert!(
            p[0].is_finite() && p[1].is_finite() && p[2].is_finite(),
            "NaN/Inf found in obstacle point {}",
            i
        );
    }

    let total_time = t0.elapsed();

    // Print timing summary (visible with --nocapture)
    println!("=== KITTI-like Pipeline Timing ===");
    println!("  Raw points:      {}", raw_count);
    println!("  After downsample: {} ({:.1}%)", ds_count, ratio * 100.0);
    println!("  After SOR:       {}", cleaned.len());
    println!(
        "  Ground inliers:  {} ({:.1}%)",
        inliers.len(),
        inlier_ratio * 100.0
    );
    println!("  Obstacle points: {}", obstacles.len());
    println!("  Clusters found:  {}", clusters.len());
    for (i, c) in clusters.iter().enumerate() {
        println!("    cluster {}: {} points", i, c.len());
    }
    println!("  ---");
    println!("  Downsample:  {:>8.2?}", ds_time);
    println!("  SOR:         {:>8.2?}", sor_time);
    println!("  RANSAC:      {:>8.2?}", ransac_time);
    println!("  Segment:     {:>8.2?}", seg_time);
    println!("  Total:       {:>8.2?}", total_time);
}

// ────────────────── Test 2: ICP realistic registration ──────────────────

#[test]
fn test_icp_realistic_registration() {
    // Build a hemisphere "target" surface
    let target = build_hemisphere(500, 99, 5.0);

    // Known transform: small translation + small rotation around Z
    let angle: f32 = 0.05; // ~2.9 degrees
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let known_transform = RigidTransform {
        rotation: [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.3, -0.2, 0.1],
    };

    let source = apply_transform(&target, &known_transform);

    // Point-to-point ICP
    let params = IcpParams {
        max_iterations: 100,
        tolerance: 1e-6,
        ..IcpParams::default()
    };

    let t0 = Instant::now();
    let result_p2p = icp_point_to_point(&source, &target, &params);
    let p2p_time = t0.elapsed();

    assert!(result_p2p.converged, "point-to-point ICP did not converge");
    assert!(
        result_p2p.rmse < 0.5,
        "point-to-point RMSE {:.4} too high",
        result_p2p.rmse
    );

    // Verify recovered translation is close to ground truth (inverse)
    let recovered_t = result_p2p.transform.translation;
    let expected_t = [
        -known_transform.translation[0],
        -known_transform.translation[1],
        -known_transform.translation[2],
    ];
    for i in 0..3 {
        assert!(
            (recovered_t[i] - expected_t[i]).abs() < 1.0,
            "translation[{}]: recovered={:.3}, expected~={:.3}",
            i,
            recovered_t[i],
            expected_t[i]
        );
    }

    // Point-to-plane ICP (needs normals on target)
    let target_normals = estimate_normals(&target, 15);

    let t1 = Instant::now();
    let result_p2pl = icp_point_to_plane(&source, &target, &target_normals, &params).unwrap();
    let p2pl_time = t1.elapsed();

    assert!(result_p2pl.converged, "point-to-plane ICP did not converge");
    assert!(
        result_p2pl.rmse < 0.5,
        "point-to-plane RMSE {:.4} too high",
        result_p2pl.rmse
    );

    println!("=== ICP Registration ===");
    println!("  Target points: {}", target.len());
    println!(
        "  Known transform: t=[{:.2}, {:.2}, {:.2}], rot_z={:.1}°",
        known_transform.translation[0],
        known_transform.translation[1],
        known_transform.translation[2],
        angle.to_degrees()
    );
    println!(
        "  Point-to-point: RMSE={:.6}, iters={}, time={:?}",
        result_p2p.rmse, result_p2p.num_iterations, p2p_time
    );
    println!(
        "  Point-to-plane: RMSE={:.6}, iters={}, time={:?}",
        result_p2pl.rmse, result_p2pl.num_iterations, p2pl_time
    );

    // Document convergence comparison
    if result_p2pl.num_iterations <= result_p2p.num_iterations {
        println!("  -> Point-to-plane converged in <= iterations (as expected)");
    } else {
        println!(
            "  -> Point-to-plane used more iterations ({} vs {}); \
             this can happen on curved surfaces where the linearization \
             assumption is less accurate",
            result_p2pl.num_iterations, result_p2p.num_iterations
        );
    }
}

// ────────────────── Test 3: IO roundtrip with all fields ──────────────────

#[test]
fn test_io_roundtrip_real_world_fields() {
    let n = 100;
    let mut rng = StdRng::seed_from_u64(77);

    // Build cloud with xyz + normals + colors
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let z: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();

    // Generate unit-ish normals
    let nx: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let ny: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let nz: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let r: Vec<u8> = (0..n).map(|_| rng.gen()).collect();
    let g: Vec<u8> = (0..n).map(|_| rng.gen()).collect();
    let b: Vec<u8> = (0..n).map(|_| rng.gen()).collect();

    let mut cloud = PointCloud::from_xyz(x.clone(), y.clone(), z.clone());
    cloud.normals = Some(Normals {
        nx: nx.clone(),
        ny: ny.clone(),
        nz: nz.clone(),
    });
    cloud.colors = Some(Colors {
        r: r.clone(),
        g: g.clone(),
        b: b.clone(),
    });

    let tmp_dir = tempfile::tempdir().unwrap();

    // ── PLY ASCII roundtrip ──
    let ply_ascii_path = tmp_dir.path().join("test.ply");
    write_ply(&ply_ascii_path, &cloud).unwrap();
    let loaded_ascii = read_ply(&ply_ascii_path).unwrap();

    assert_eq!(loaded_ascii.len(), n);
    for i in 0..n {
        assert!(
            (loaded_ascii.x[i] - x[i]).abs() < 1e-4,
            "PLY ASCII x mismatch at {}",
            i
        );
        assert!(
            (loaded_ascii.y[i] - y[i]).abs() < 1e-4,
            "PLY ASCII y mismatch at {}",
            i
        );
        assert!(
            (loaded_ascii.z[i] - z[i]).abs() < 1e-4,
            "PLY ASCII z mismatch at {}",
            i
        );
    }
    let ln = loaded_ascii
        .normals
        .as_ref()
        .expect("PLY ASCII lost normals");
    for i in 0..n {
        assert!(
            (ln.nx[i] - nx[i]).abs() < 1e-4,
            "PLY ASCII nx mismatch at {}",
            i
        );
    }
    let lc = loaded_ascii.colors.as_ref().expect("PLY ASCII lost colors");
    assert_eq!(lc.r, r, "PLY ASCII red channel mismatch");
    assert_eq!(lc.g, g, "PLY ASCII green channel mismatch");
    assert_eq!(lc.b, b, "PLY ASCII blue channel mismatch");

    // ── PLY binary roundtrip ──
    let ply_bin_path = tmp_dir.path().join("test_bin.ply");
    write_ply_binary(&ply_bin_path, &cloud).unwrap();
    let loaded_bin = read_ply(&ply_bin_path).unwrap();

    assert_eq!(loaded_bin.len(), n);
    // Binary should be bit-exact for floats
    for i in 0..n {
        assert_eq!(loaded_bin.x[i], x[i], "PLY binary x mismatch at {}", i);
        assert_eq!(loaded_bin.y[i], y[i], "PLY binary y mismatch at {}", i);
        assert_eq!(loaded_bin.z[i], z[i], "PLY binary z mismatch at {}", i);
    }
    let bn = loaded_bin
        .normals
        .as_ref()
        .expect("PLY binary lost normals");
    for i in 0..n {
        assert_eq!(bn.nx[i], nx[i], "PLY binary nx mismatch at {}", i);
        assert_eq!(bn.ny[i], ny[i], "PLY binary ny mismatch at {}", i);
        assert_eq!(bn.nz[i], nz[i], "PLY binary nz mismatch at {}", i);
    }
    let bc = loaded_bin.colors.as_ref().expect("PLY binary lost colors");
    assert_eq!(bc.r, r, "PLY binary red mismatch");
    assert_eq!(bc.g, g, "PLY binary green mismatch");
    assert_eq!(bc.b, b, "PLY binary blue mismatch");

    // ── PCD binary roundtrip (xyz only) ──
    let pcd_bin_path = tmp_dir.path().join("test.pcd");
    write_pcd_binary(&pcd_bin_path, &cloud).unwrap();
    let loaded_pcd = pointclouds_io::read_pcd(&pcd_bin_path).unwrap();
    assert_eq!(loaded_pcd.len(), n);
    for i in 0..n {
        assert_eq!(loaded_pcd.x[i], x[i], "PCD binary x mismatch at {}", i);
        assert_eq!(loaded_pcd.y[i], y[i], "PCD binary y mismatch at {}", i);
        assert_eq!(loaded_pcd.z[i], z[i], "PCD binary z mismatch at {}", i);
    }

    // ── PCD ASCII roundtrip (xyz only) ──
    let pcd_ascii_path = tmp_dir.path().join("test_ascii.pcd");
    write_pcd(&pcd_ascii_path, &cloud).unwrap();
    let loaded_pcd_ascii = pointclouds_io::read_pcd(&pcd_ascii_path).unwrap();
    assert_eq!(loaded_pcd_ascii.len(), n);
    for i in 0..n {
        assert!(
            (loaded_pcd_ascii.x[i] - x[i]).abs() < 1e-4,
            "PCD ASCII x mismatch at {}",
            i
        );
    }

    println!("=== IO Roundtrip ===");
    println!("  {} points with normals + colors", n);
    println!("  PLY ASCII:  xyz ✓  normals ✓  colors ✓");
    println!("  PLY binary: xyz ✓  normals ✓  colors ✓  (bit-exact)");
    println!("  PCD binary: xyz ✓  (normals/colors not supported in PCD writer)");
    println!("  PCD ASCII:  xyz ✓  (normals/colors not supported in PCD writer)");
}

// ────────────────── Test 4: Large cloud scaling ──────────────────

#[test]
#[ignore] // too heavy for default CI; run with: cargo test -- --ignored
fn test_large_cloud_scaling() {
    let n = 2_000_000;
    let mut rng = StdRng::seed_from_u64(12345);

    println!(
        "=== Large Cloud Scaling ({:.1}M points) ===",
        n as f64 / 1e6
    );

    let t_gen = Instant::now();
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-100.0f32..100.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(-100.0f32..100.0)).collect();
    let z: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..20.0)).collect();
    let cloud = PointCloud::from_xyz(x, y, z);
    println!("  Generation:  {:>8.2?}", t_gen.elapsed());

    // Voxel downsample
    let t_ds = Instant::now();
    let downsampled = voxel_downsample(&cloud, 0.5);
    let ds_time = t_ds.elapsed();
    assert!(downsampled.len() > 0);
    assert!(downsampled.len() < n);
    println!(
        "  Downsample:  {:>8.2?}  ({} -> {} pts)",
        ds_time,
        n,
        downsampled.len()
    );

    // SOR on downsampled
    let t_sor = Instant::now();
    let cleaned = statistical_outlier_removal(&downsampled, 10, 2.0);
    let sor_time = t_sor.elapsed();
    assert!(cleaned.len() > 0);
    println!(
        "  SOR:         {:>8.2?}  ({} -> {} pts)",
        sor_time,
        downsampled.len(),
        cleaned.len()
    );

    // RANSAC
    let t_ransac = Instant::now();
    let (_plane, inliers) = ransac_plane_seeded(&cleaned, 0.3, 200, 42);
    let ransac_time = t_ransac.elapsed();
    assert!(inliers.len() > 0);
    println!(
        "  RANSAC:      {:>8.2?}  ({} inliers)",
        ransac_time,
        inliers.len()
    );

    let total = t_gen.elapsed();
    println!("  Total:       {:>8.2?}", total);
    println!("  No panics, no OOM.");
}
