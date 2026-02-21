#!/usr/bin/env python3
"""
KITTI-style obstacle detection pipeline using pointclouds_rs.

Usage:
    python kitti_obstacle_detection.py              # synthetic scene
    python kitti_obstacle_detection.py scene.pcd    # from file
"""

import sys
import time
import platform

import numpy as np

import pointclouds_rs as pcrs


# ── Synthetic scene generation ───────────────────────────────────────────────


def generate_kitti_scene(seed=42):
    """Generate a synthetic KITTI-like point cloud scene.

    Components:
      - Ground plane: ~60k points, x in [-30,30], y in [-20,20], z ~ 0
      - Car cluster 1: ~3k points centered at (8, 3, 0.8)
      - Car cluster 2: ~3k points centered at (-5, -8, 0.8)
      - Pedestrian cluster: ~500 points centered at (3, -2, 0.9)
      - Random outlier noise: ~1.5k points
    """
    rng = np.random.default_rng(seed)

    parts = []

    # Ground plane
    n_ground = 60_000
    gx = rng.uniform(-30, 30, n_ground).astype(np.float32)
    gy = rng.uniform(-20, 20, n_ground).astype(np.float32)
    gz = rng.normal(0, 0.03, n_ground).astype(np.float32)
    parts.append(np.column_stack([gx, gy, gz]))

    # Car 1: box ~4m x 1.8m x 1.5m
    n_car = 3_000
    cx, cy, cz = 8.0, 3.0, 0.8
    car1 = np.column_stack([
        rng.uniform(cx - 2.0, cx + 2.0, n_car),
        rng.uniform(cy - 0.9, cy + 0.9, n_car),
        rng.uniform(cz - 0.0, cz + 1.5, n_car),
    ]).astype(np.float32)
    parts.append(car1)

    # Car 2: box ~4m x 1.8m x 1.5m
    cx2, cy2, cz2 = -5.0, -8.0, 0.8
    car2 = np.column_stack([
        rng.uniform(cx2 - 2.0, cx2 + 2.0, n_car),
        rng.uniform(cy2 - 0.9, cy2 + 0.9, n_car),
        rng.uniform(cz2 - 0.0, cz2 + 1.5, n_car),
    ]).astype(np.float32)
    parts.append(car2)

    # Pedestrian: ~0.5m x 0.5m x 1.8m
    n_ped = 500
    px, py, pz = 3.0, -2.0, 0.9
    ped = np.column_stack([
        rng.uniform(px - 0.25, px + 0.25, n_ped),
        rng.uniform(py - 0.25, py + 0.25, n_ped),
        rng.uniform(pz - 0.0, pz + 1.8, n_ped),
    ]).astype(np.float32)
    parts.append(ped)

    # Outlier noise
    n_noise = 1_500
    noise = np.column_stack([
        rng.uniform(-35, 35, n_noise),
        rng.uniform(-25, 25, n_noise),
        rng.uniform(-3, 8, n_noise),
    ]).astype(np.float32)
    parts.append(noise)

    return np.vstack(parts)


# ── Pipeline ─────────────────────────────────────────────────────────────────


def run_pipeline(cloud):
    """Run the full KITTI obstacle detection pipeline. Returns a results dict."""
    results = {}
    results["raw_points"] = cloud.len()

    # Step 1: voxel downsample
    t0 = time.perf_counter()
    downsampled = pcrs.voxel_downsample(cloud, 0.15)
    results["downsample_ms"] = (time.perf_counter() - t0) * 1000
    results["after_downsample"] = downsampled.len()

    # Step 2: statistical outlier removal
    t0 = time.perf_counter()
    cleaned = pcrs.statistical_outlier_removal(downsampled, 20, 2.0)
    results["sor_ms"] = (time.perf_counter() - t0) * 1000
    results["after_sor"] = cleaned.len()

    # Step 3: RANSAC ground plane segmentation
    t0 = time.perf_counter()
    plane = pcrs.ransac_plane(cleaned, 0.15, 500)
    results["ransac_ms"] = (time.perf_counter() - t0) * 1000
    results["plane_normal"] = plane.normal
    results["plane_inliers"] = len(plane.inliers)

    # Step 4: separate ground from obstacles
    t0 = time.perf_counter()
    obstacles = cleaned.select_inverse(plane.inliers)
    results["after_ground_removal"] = obstacles.len()

    # Step 5: cluster obstacles
    clusters = pcrs.euclidean_cluster(obstacles, 0.8, 10, 20_000)
    results["cluster_ms"] = (time.perf_counter() - t0) * 1000
    results["num_clusters"] = len(clusters)
    results["cluster_sizes"] = [len(c) for c in clusters]

    return results


def run_open3d_comparison(data):
    """Run comparable Open3D pipeline for timing comparison."""
    try:
        import open3d as o3d
    except ImportError:
        return None

    results = {}

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data.astype(np.float64))

    t0 = time.perf_counter()
    down = pcd.voxel_down_sample(0.15)
    results["downsample_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cl, ind = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    cleaned = down.select_by_index(ind)
    results["sor_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    plane_model, inliers = cleaned.segment_plane(
        distance_threshold=0.15, ransac_n=3, num_iterations=500
    )
    results["ransac_ms"] = (time.perf_counter() - t0) * 1000
    results["plane_inliers"] = len(inliers)

    t0 = time.perf_counter()
    obstacles = cleaned.select_by_index(inliers, invert=True)
    labels = np.array(obstacles.cluster_dbscan(eps=0.8, min_points=10))
    n_clusters = labels.max() + 1 if len(labels) > 0 else 0
    results["cluster_ms"] = (time.perf_counter() - t0) * 1000
    results["num_clusters"] = n_clusters

    return results


# ── Output ───────────────────────────────────────────────────────────────────


def print_results(results, o3d_results=None):
    print()
    print("=" * 60)
    print("  KITTI Obstacle Detection Pipeline — pointclouds_rs")
    print("=" * 60)
    print()
    print(f"  Python {platform.python_version()} | {platform.system()} {platform.machine()}")
    print()
    print("  Pipeline steps:")
    print(f"    Raw input:          {results['raw_points']:>8,} points")
    print(f"    After downsample:   {results['after_downsample']:>8,} points  ({results['downsample_ms']:>7.1f} ms)")
    print(f"    After SOR:          {results['after_sor']:>8,} points  ({results['sor_ms']:>7.1f} ms)")
    print(f"    Ground inliers:     {results['plane_inliers']:>8,} points  ({results['ransac_ms']:>7.1f} ms)")
    print(f"    Obstacle points:    {results['after_ground_removal']:>8,} points")
    print(f"    Clusters found:     {results['num_clusters']:>8}          ({results['cluster_ms']:>7.1f} ms)")
    print()
    n = results["plane_normal"]
    print(f"  Ground plane normal:  [{n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f}]")
    print()
    print("  Detected clusters:")
    for i, size in enumerate(results["cluster_sizes"]):
        label = "car-sized" if size > 500 else "pedestrian-sized" if size > 50 else "small"
        print(f"    [{i}] {size:>5} points  ({label})")
    print()

    total_ms = results["downsample_ms"] + results["sor_ms"] + results["ransac_ms"] + results["cluster_ms"]
    print(f"  Total pipeline time:  {total_ms:>7.1f} ms")

    if o3d_results is not None:
        o3d_total = o3d_results["downsample_ms"] + o3d_results["sor_ms"] + o3d_results["ransac_ms"] + o3d_results["cluster_ms"]
        print()
        print("  Open3D comparison:")
        print(f"    Downsample:  {o3d_results['downsample_ms']:>7.1f} ms  (vs {results['downsample_ms']:>7.1f} ms)")
        print(f"    SOR:         {o3d_results['sor_ms']:>7.1f} ms  (vs {results['sor_ms']:>7.1f} ms)")
        print(f"    RANSAC:      {o3d_results['ransac_ms']:>7.1f} ms  (vs {results['ransac_ms']:>7.1f} ms)")
        print(f"    Clustering:  {o3d_results['cluster_ms']:>7.1f} ms  (vs {results['cluster_ms']:>7.1f} ms)")
        print(f"    Total:       {o3d_total:>7.1f} ms  (vs {total_ms:>7.1f} ms)")
        if o3d_total > 0:
            print(f"    Speedup:     {o3d_total / total_ms:.1f}x")
    else:
        print()
        print("  Open3D not installed; skipped comparison.")

    print()
    print("=" * 60)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    # Load or generate point cloud
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Loading point cloud from: {path}")
        if path.endswith(".las") or path.endswith(".laz"):
            cloud = pcrs.read_las(path)
        elif path.endswith(".ply"):
            cloud = pcrs.read_ply(path)
        else:
            cloud = pcrs.read_pcd(path)
        data = cloud.to_numpy()
    else:
        print("Generating synthetic KITTI-like scene...")
        data = generate_kitti_scene(seed=42)
        cloud = pcrs.PointCloud.from_numpy(data)

    # Run pipeline
    results = run_pipeline(cloud)

    # Optional Open3D comparison
    o3d_results = run_open3d_comparison(data)

    # Print results
    print_results(results, o3d_results)


if __name__ == "__main__":
    main()
