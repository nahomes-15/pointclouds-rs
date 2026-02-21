#!/usr/bin/env python3
"""
Aerial LiDAR processing pipeline using pointclouds_rs.

Demonstrates the library handling large-scale aerial-style point clouds
with per-step timing and throughput reporting.

Usage:
    python aerial_lidar.py                  # synthetic scene (~2.5M pts)
    python aerial_lidar.py --quick          # smaller synthetic (~250k pts)
    python aerial_lidar.py scene.las        # from file
"""

import argparse
import time
import platform

import numpy as np

import pointclouds_rs as pcrs


# ── Synthetic aerial scene generation ────────────────────────────────────────


def generate_aerial_scene(seed=42, scale=1.0):
    """Generate a synthetic aerial LiDAR scene.

    Args:
        seed: Random seed for reproducibility.
        scale: Multiplier for point counts (1.0 = ~2.5M, 0.1 = ~250k).

    Components:
      - Terrain baseline: undulating ground surface
      - Buildings: several flat-roofed structures at various heights
      - Trees: several hemisphere-like clusters
    """
    rng = np.random.default_rng(seed)

    parts = []
    area_x, area_y = 500.0, 500.0  # 500m x 500m survey area

    # ── Terrain baseline ──
    # Undulating ground with gentle hills: z = sin-based elevation + noise
    n_terrain = int(2_000_000 * scale)
    tx = rng.uniform(0, area_x, n_terrain).astype(np.float32)
    ty = rng.uniform(0, area_y, n_terrain).astype(np.float32)
    # Gentle terrain undulation
    tz = (
        2.0 * np.sin(tx * 0.02) * np.cos(ty * 0.015)
        + rng.normal(0, 0.05, n_terrain)
    ).astype(np.float32)
    parts.append(np.column_stack([tx, ty, tz]))

    # ── Buildings ──
    buildings = [
        # (center_x, center_y, width, depth, height)
        (100, 120, 30, 20, 12),
        (250, 300, 40, 40, 18),
        (350, 100, 25, 25, 8),
        (400, 400, 50, 30, 15),
        (150, 350, 20, 20, 10),
    ]
    n_per_building = int(50_000 * scale)
    for cx, cy, w, d, h in buildings:
        # Roof points (top surface)
        n_roof = int(n_per_building * 0.7)
        bx = rng.uniform(cx - w / 2, cx + w / 2, n_roof).astype(np.float32)
        by = rng.uniform(cy - d / 2, cy + d / 2, n_roof).astype(np.float32)
        bz = np.full(n_roof, h, dtype=np.float32) + rng.normal(0, 0.02, n_roof).astype(np.float32)
        parts.append(np.column_stack([bx, by, bz]))

        # Wall points (vertical surfaces)
        n_wall = n_per_building - n_roof
        side = rng.integers(0, 4, n_wall)
        wx = np.empty(n_wall, dtype=np.float32)
        wy = np.empty(n_wall, dtype=np.float32)
        wz = rng.uniform(0, h, n_wall).astype(np.float32)
        for i in range(n_wall):
            if side[i] == 0:  # north wall
                wx[i] = rng.uniform(cx - w / 2, cx + w / 2)
                wy[i] = cy + d / 2
            elif side[i] == 1:  # south wall
                wx[i] = rng.uniform(cx - w / 2, cx + w / 2)
                wy[i] = cy - d / 2
            elif side[i] == 2:  # east wall
                wx[i] = cx + w / 2
                wy[i] = rng.uniform(cy - d / 2, cy + d / 2)
            else:  # west wall
                wx[i] = cx - w / 2
                wy[i] = rng.uniform(cy - d / 2, cy + d / 2)
        parts.append(np.column_stack([wx, wy, wz]))

    # ── Trees ──
    trees = [
        # (center_x, center_y, radius, height_base)
        (50, 50, 5, 0),
        (180, 200, 6, 0),
        (300, 450, 4, 0),
        (420, 250, 7, 0),
        (80, 400, 5, 0),
        (450, 50, 4, 0),
        (200, 80, 5, 0),
        (350, 350, 6, 0),
    ]
    n_per_tree = int(20_000 * scale)
    for cx, cy, r, base in trees:
        # Hemisphere canopy
        n_canopy = int(n_per_tree * 0.8)
        # Rejection sampling for hemisphere
        count = 0
        canopy_pts = []
        while count < n_canopy:
            batch = 2 * n_canopy
            px = rng.uniform(-1, 1, batch)
            py = rng.uniform(-1, 1, batch)
            r2 = px**2 + py**2
            mask = r2 < 1.0
            px, py, r2 = px[mask], py[mask], r2[mask]
            pz = np.sqrt(1.0 - r2)
            pts = np.column_stack([
                (cx + px[:n_canopy - count] * r).astype(np.float32),
                (cy + py[:n_canopy - count] * r).astype(np.float32),
                (base + 4 + pz[:n_canopy - count] * r).astype(np.float32),
            ])
            canopy_pts.append(pts)
            count += len(pts)
        parts.append(np.vstack(canopy_pts)[:n_canopy])

        # Trunk
        n_trunk = n_per_tree - n_canopy
        trunk_x = cx + rng.normal(0, 0.15, n_trunk).astype(np.float32)
        trunk_y = cy + rng.normal(0, 0.15, n_trunk).astype(np.float32)
        trunk_z = rng.uniform(base, base + 4, n_trunk).astype(np.float32)
        parts.append(np.column_stack([trunk_x, trunk_y, trunk_z]))

    return np.vstack(parts)


# ── Pipeline ─────────────────────────────────────────────────────────────────


def run_pipeline(cloud):
    """Run aerial LiDAR processing pipeline with per-step timing."""
    results = {}
    results["raw_points"] = cloud.len()

    # Bounding box from numpy
    data = cloud.to_numpy()
    bb_min = data.min(axis=0)
    bb_max = data.max(axis=0)
    results["bbox_min"] = bb_min.tolist()
    results["bbox_max"] = bb_max.tolist()
    results["bbox_extent"] = (bb_max - bb_min).tolist()

    # Step 1: voxel downsample
    t0 = time.perf_counter()
    downsampled = pcrs.voxel_downsample(cloud, 0.5)
    results["downsample_ms"] = (time.perf_counter() - t0) * 1000
    results["after_downsample"] = downsampled.len()

    # Step 2: estimate normals
    t0 = time.perf_counter()
    with_normals = pcrs.estimate_normals(downsampled, 15)
    results["normals_ms"] = (time.perf_counter() - t0) * 1000
    results["after_normals"] = with_normals.len()

    # Step 3: RANSAC ground plane
    t0 = time.perf_counter()
    plane = pcrs.ransac_plane(with_normals, 0.3, 300)
    results["ransac_ms"] = (time.perf_counter() - t0) * 1000
    results["plane_normal"] = plane.normal
    results["plane_inliers"] = len(plane.inliers)

    # Step 4: separate ground / non-ground
    t0 = time.perf_counter()
    non_ground = with_normals.select_inverse(plane.inliers)
    results["non_ground_points"] = non_ground.len()

    # Step 5: cluster non-ground objects
    clusters = pcrs.euclidean_cluster(non_ground, 2.0, 20, 100_000)
    results["cluster_ms"] = (time.perf_counter() - t0) * 1000
    results["num_clusters"] = len(clusters)
    results["cluster_sizes"] = sorted([len(c) for c in clusters], reverse=True)

    return results


# ── Output ───────────────────────────────────────────────────────────────────


def print_results(results):
    print()
    print("=" * 64)
    print("  Aerial LiDAR Processing Pipeline — pointclouds_rs")
    print("=" * 64)
    print()
    print(f"  Python {platform.python_version()} | {platform.system()} {platform.machine()}")
    print()
    print(f"  Input:  {results['raw_points']:>12,} points")
    bb = results["bbox_extent"]
    print(f"  Extent: {bb[0]:.1f} x {bb[1]:.1f} x {bb[2]:.1f} m")
    print()
    print("  Pipeline steps                  Points       Time")
    print("  " + "-" * 54)
    print(f"  Voxel downsample (0.5m)   {results['after_downsample']:>10,}   {results['downsample_ms']:>8.1f} ms")
    print(f"  Estimate normals (k=15)   {results['after_normals']:>10,}   {results['normals_ms']:>8.1f} ms")
    print(f"  RANSAC ground plane       {results['plane_inliers']:>10,}   {results['ransac_ms']:>8.1f} ms")
    print(f"  Non-ground points         {results['non_ground_points']:>10,}")
    print(f"  Euclidean clustering      {results['num_clusters']:>10}   {results['cluster_ms']:>8.1f} ms")
    print()

    n = results["plane_normal"]
    print(f"  Ground normal: [{n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f}]")
    print()

    # Cluster summary
    sizes = results["cluster_sizes"]
    if sizes:
        print(f"  Clusters: {len(sizes)} total")
        # Show top 10
        for i, s in enumerate(sizes[:10]):
            label = "building" if s > 5000 else "tree" if s > 500 else "small"
            print(f"    [{i:>2}] {s:>7,} points  ({label})")
        if len(sizes) > 10:
            print(f"    ... and {len(sizes) - 10} more")
    print()

    total_ms = (
        results["downsample_ms"]
        + results["normals_ms"]
        + results["ransac_ms"]
        + results["cluster_ms"]
    )
    throughput = results["raw_points"] / (total_ms / 1000) if total_ms > 0 else 0
    print(f"  Total pipeline time:  {total_ms:>8.1f} ms")
    print(f"  Throughput:           {throughput:>8,.0f} pts/sec")
    print()
    print("=" * 64)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Aerial LiDAR processing demo with pointclouds_rs"
    )
    parser.add_argument("input", nargs="?", help="Input point cloud file (.pcd, .ply, .las)")
    parser.add_argument("--quick", action="store_true", help="Use smaller synthetic scene (~250k pts)")
    args = parser.parse_args()

    if args.input:
        path = args.input
        print(f"Loading point cloud from: {path}")
        if path.endswith(".las") or path.endswith(".laz"):
            cloud = pcrs.read_las(path)
        elif path.endswith(".ply"):
            cloud = pcrs.read_ply(path)
        else:
            cloud = pcrs.read_pcd(path)
    else:
        scale = 0.1 if args.quick else 1.0
        n_approx = "~250k" if args.quick else "~2.5M"
        print(f"Generating synthetic aerial scene ({n_approx} points)...")
        t0 = time.perf_counter()
        data = generate_aerial_scene(seed=42, scale=scale)
        gen_ms = (time.perf_counter() - t0) * 1000
        print(f"  Generated {len(data):,} points in {gen_ms:.0f} ms")
        cloud = pcrs.PointCloud.from_numpy(data)

    results = run_pipeline(cloud)
    print_results(results)


if __name__ == "__main__":
    main()
