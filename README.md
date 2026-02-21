# pointclouds-rs

**The Polars of Point Clouds.**

`pointclouds-rs` is a high-performance Rust-native point cloud processing library
with first-class Python bindings. It targets **>=3x speedup over Open3D** on
representative workloads, using an SoA-first data model, f32 pipelines, and
efficient NumPy interop.

## Features

- **SoA data model** — cache-friendly, SIMD-ready layout (`x`, `y`, `z` as separate `Vec<f32>`)
- **f32-first** — aligned with real sensor output (LiDAR, depth cameras)
- **NumPy interop** — `PointCloud.from_numpy()` / `.to_numpy()` with automatic f64→f32 coercion
- **Batteries included** — filters, normals, ICP registration, segmentation, multi-format I/O
- **Safe Rust** — `#![forbid(unsafe_code)]` on all crates
- **Cross-platform wheels** — `pip install pointclouds-rs` on Linux, macOS, Windows

## Quickstart (Python)

```python
import numpy as np
import pointclouds_rs as pcr

# Create from NumPy (zero-copy for contiguous f32)
points = np.random.rand(100_000, 3).astype(np.float32)
cloud = pcr.PointCloud.from_numpy(points)

# Voxel downsample
ds = pcr.voxel_downsample(cloud, voxel_size=0.05)
print(f"Downsampled: {cloud.len()} → {ds.len()} points")

# Passthrough filter
filtered = pcr.passthrough_filter(cloud, "z", 0.2, 0.8)

# Estimate normals
with_normals = pcr.estimate_normals(cloud, k=15)

# RANSAC plane fitting
result = pcr.ransac_plane(cloud, distance_threshold=0.01, iterations=1000)
print(f"Plane normal: {result.normal}, inliers: {len(result.inliers)}")

# ICP registration
source = pcr.PointCloud.from_numpy(np.random.rand(1000, 3).astype(np.float32))
target = pcr.PointCloud.from_numpy(np.random.rand(1000, 3).astype(np.float32))
icp = pcr.icp_point_to_point(source, target, max_iterations=50)
print(f"ICP converged: {icp.converged}, RMSE: {icp.rmse:.6f}")

# Euclidean clustering
clusters = pcr.euclidean_cluster(cloud, distance_threshold=0.05, min_size=10, max_size=10000)

# File I/O
pcr.write_pcd("output.pcd", cloud)
pcr.write_ply("output.ply", cloud)
loaded = pcr.read_pcd("output.pcd")
```

## Quickstart (Rust)

```rust
use pointclouds_core::PointCloud;
use pointclouds_filters::voxel_downsample;
use pointclouds_normals::estimate_normals;
use pointclouds_registration::{icp_point_to_point, IcpParams};
use pointclouds_segmentation::ransac_plane;
use pointclouds_io::{read_pcd, write_pcd};

// Create a point cloud
let cloud = PointCloud::from_xyz(
    vec![0.0, 1.0, 2.0],
    vec![0.0, 0.0, 0.0],
    vec![0.0, 0.0, 0.0],
);

// Voxel downsample
let ds = voxel_downsample(&cloud, 0.5);

// Estimate normals
let normals = estimate_normals(&cloud, 10);

// ICP registration
let result = icp_point_to_point(&source, &target, &IcpParams::default());

// RANSAC plane
let (model, inliers) = ransac_plane(&cloud, 0.01, 1000);

// I/O
write_pcd("output.pcd", &cloud).unwrap();
let loaded = read_pcd("output.pcd").unwrap();
```

## Crates

| Crate | Description |
|-------|-------------|
| `pointclouds-core` | SoA `PointCloud`, `CloudView`, `Aabb`, point types |
| `pointclouds-spatial` | KD-tree (kiddo-backed, O(log n) queries) |
| `pointclouds-filters` | Voxel downsample, passthrough, statistical/radius outlier removal |
| `pointclouds-normals` | PCA-based normal estimation via SVD |
| `pointclouds-registration` | Point-to-point ICP, rigid transforms, correspondences |
| `pointclouds-segmentation` | Euclidean clustering, RANSAC plane fitting |
| `pointclouds-io` | PCD (ASCII + binary), PLY (ASCII), LAS file I/O |
| `pointclouds-python` | PyO3 + maturin Python bindings |

## Building

```bash
# Rust
cargo test --workspace
cargo bench --workspace

# Python
cd crates/python
maturin develop --release
pytest ../../tests/test_python.py -v
```

## Performance

Target: **>=3x faster than Open3D** on 1M-point workloads.

Run benchmarks:
```bash
cargo bench --workspace
```

See [BENCHMARKS.md](BENCHMARKS.md) for detailed results.

## License

Dual-licensed under either:

- MIT license (`LICENSE-MIT`)
- Apache License, Version 2.0 (`LICENSE-APACHE`)

at your option.
