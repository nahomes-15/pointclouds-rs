"""
Head-to-head benchmark: pointclouds-rs vs Open3D.

Usage:
    pip install open3d pointclouds-rs numpy
    python tests/bench_vs_open3d.py

Benchmarks voxel downsample, normal estimation, RANSAC plane fitting,
and passthrough filter on 100K and 1M random points.
"""

import copy
import time
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_random_cloud(n: int, seed: int = 42) -> np.ndarray:
    """Generate n random 3D points in [0, 10)^3."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 3), dtype=np.float64) * 10.0


def bench(fn, *, warmup: int = 2, repeats: int = 10) -> float:
    """Return median wall-clock seconds over `repeats` after `warmup` runs."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# Open3D wrappers
# ---------------------------------------------------------------------------

def o3d_voxel_downsample(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size)


def o3d_estimate_normals(pcd, k):
    pcd.estimate_normals(
        search_param=__import__("open3d").geometry.KDTreeSearchParamKNN(knn=k)
    )
    return pcd


def o3d_ransac_plane(pcd, dist, iters):
    return pcd.segment_plane(
        distance_threshold=dist, ransac_n=3, num_iterations=iters
    )


def o3d_passthrough(pcd, axis_idx, lo, hi):
    pts = np.asarray(pcd.points)
    mask = (pts[:, axis_idx] >= lo) & (pts[:, axis_idx] <= hi)
    return pcd.select_by_index(np.where(mask)[0])


# ---------------------------------------------------------------------------
# pointclouds-rs wrappers
# ---------------------------------------------------------------------------

def pcr_voxel_downsample(cloud, voxel_size):
    import pointclouds_rs as pcr
    return pcr.voxel_downsample(cloud, voxel_size)


def pcr_estimate_normals(cloud, k):
    import pointclouds_rs as pcr
    return pcr.estimate_normals(cloud, k)


def pcr_ransac_plane(cloud, dist, iters):
    import pointclouds_rs as pcr
    return pcr.ransac_plane(cloud, dist, iters)


def pcr_passthrough(cloud, axis, lo, hi):
    import pointclouds_rs as pcr
    return pcr.passthrough_filter(cloud, axis, lo, hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(n_points: int):
    import open3d as o3d
    import pointclouds_rs as pcr

    print(f"\n{'='*60}")
    print(f"  Benchmark: {n_points:,} points")
    print(f"{'='*60}")

    pts64 = make_random_cloud(n_points)
    pts32 = pts64.astype(np.float32)

    # Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts64)

    # pointclouds-rs point cloud
    cloud = pcr.PointCloud.from_numpy(pts32)

    results = []

    # --- Voxel Downsample ---
    voxel_size = 0.1
    t_o3d = bench(lambda: o3d_voxel_downsample(pcd, voxel_size))
    t_pcr = bench(lambda: pcr_voxel_downsample(cloud, voxel_size))
    speedup = t_o3d / t_pcr if t_pcr > 0 else float("inf")
    results.append(("Voxel Downsample", t_o3d, t_pcr, speedup))

    # --- Passthrough Filter ---
    t_o3d = bench(lambda: o3d_passthrough(pcd, 2, 2.0, 8.0))
    t_pcr = bench(lambda: pcr_passthrough(cloud, "z", 2.0, 8.0))
    speedup = t_o3d / t_pcr if t_pcr > 0 else float("inf")
    results.append(("Passthrough Filter", t_o3d, t_pcr, speedup))

    # --- Normal Estimation (k=15) ---
    t_o3d = bench(lambda: o3d_estimate_normals(copy.deepcopy(pcd), 15), warmup=1, repeats=3)
    t_pcr = bench(lambda: pcr_estimate_normals(cloud, 15), warmup=1, repeats=3)
    speedup = t_o3d / t_pcr if t_pcr > 0 else float("inf")
    results.append(("Normal Estimation k=15", t_o3d, t_pcr, speedup))

    # --- RANSAC Plane Fitting ---
    t_o3d = bench(lambda: o3d_ransac_plane(pcd, 0.05, 1000), repeats=5)
    t_pcr = bench(lambda: pcr_ransac_plane(cloud, 0.05, 1000), repeats=5)
    speedup = t_o3d / t_pcr if t_pcr > 0 else float("inf")
    results.append(("RANSAC Plane (1000 iter)", t_o3d, t_pcr, speedup))

    # --- Print results ---
    print(f"\n{'Algorithm':<28} {'Open3D':>12} {'pcr':>12} {'Speedup':>10}")
    print("-" * 64)
    for name, to, tp, sp in results:
        print(f"{name:<28} {to*1000:>10.2f}ms {tp*1000:>10.2f}ms {sp:>8.1f}x")

    return results


if __name__ == "__main__":
    print("pointclouds-rs vs Open3D — Head-to-Head Benchmark")
    print("=" * 60)

    all_results = []
    for n in [100_000, 1_000_000]:
        all_results.extend(run_comparison(n))

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    speedups = [r[3] for r in all_results]
    print(f"  Min speedup: {min(speedups):.1f}x")
    print(f"  Max speedup: {max(speedups):.1f}x")
    print(f"  Avg speedup: {sum(speedups)/len(speedups):.1f}x")
    target_met = all(s >= 3.0 for s in speedups)
    print(f"  Target (>=3x): {'PASS' if target_met else 'MIXED — see individual results'}")
