# Benchmarks

## Environment

- **CPU**: Apple M4 Max (arm64)
- **RAM**: 36 GB
- **OS**: macOS 15.5 (Darwin 24.6.0)
- **Rust**: 1.92.0 (release profile, LTO)
- **Python**: 3.14.2 (for pipeline benchmarks)
- **Benchmark tool**: [Criterion.rs](https://github.com/bheisler/criterion.rs), 100 samples per measurement

## Reproducing

```bash
# All Criterion benchmarks (Rust-only, no Python needed)
cargo bench -p pointclouds-benches

# Individual suites
cargo bench -p pointclouds-benches --bench bench_voxel
cargo bench -p pointclouds-benches --bench bench_kdtree
cargo bench -p pointclouds-benches --bench bench_normals
cargo bench -p pointclouds-benches --bench bench_icp
cargo bench -p pointclouds-benches --bench bench_filters

# End-to-end pipeline benchmarks (Python, requires release wheel)
maturin develop --release --manifest-path crates/python/Cargo.toml
python examples/python/kitti_obstacle_detection.py
python examples/python/aerial_lidar.py --quick
```

## Criterion results

All numbers are wall-clock median from 100 Criterion samples.

### Voxel Downsample

| Points | Time |
|--------|------|
| 10K | 61 us |
| 100K | 703 us |
| 1M | 8.3 ms |

### KD-Tree queries (per query, tree already built)

| Points in tree | KNN (k=10) | Radius search |
|---------------|-----------|---------------|
| 100K | 1.47 us | 235 ns |
| 1M | 2.13 us | 419 ns |

### Normal estimation (PCA, k=10, rayon-parallel)

| Points | Time |
|--------|------|
| 10K | 1.4 ms |
| 100K | 15.8 ms |

### ICP point-to-point (max 50 iterations)

| Points | Time |
|--------|------|
| 1K | 466 us |
| 10K | 5.15 ms |

### Passthrough filter

| Points | Time |
|--------|------|
| 100K | 372 us |
| 1M | 5.5 ms |

### Statistical outlier removal (k=10)

| Points | Time |
|--------|------|
| 10K | 11.2 ms |
| 100K | 128 ms |

### Radius outlier removal

| Points | Time |
|--------|------|
| 10K | 1.35 ms |
| 100K | 19.1 ms |

## Euclidean clustering

Uses grid-based spatial hashing + union-find (not KD-tree BFS). Rayon-parallel
candidate pair generation. Measured via `examples/python/aerial_lidar.py --quick`:

| Points | Radius | Time | Algorithm |
|--------|--------|------|-----------|
| 161K non-ground | 2.0m | 16 ms | grid + union-find |

## Summary table

| Algorithm | 10K | 100K | 1M |
|-----------|-----|------|-----|
| Voxel downsample | 61 us | 703 us | 8.3 ms |
| KD-tree KNN k=10 (per query) | -- | 1.47 us | 2.13 us |
| KD-tree radius (per query) | -- | 235 ns | 419 ns |
| Normal estimation k=10 | 1.4 ms | 15.8 ms | -- |
| ICP point-to-point | 466 us (1K) | 5.15 ms | -- |
| Passthrough filter | -- | 372 us | 5.5 ms |
| SOR k=10 | 11.2 ms | 128 ms | -- |
| Radius outlier removal | 1.35 ms | 19.1 ms | -- |

## End-to-end pipeline benchmarks

These run from Python using the release wheel (`maturin develop --release`).

### KITTI obstacle detection (68K points)

```
Raw input:            68,000 points
After downsample:     58,053 points  (    3.9 ms)
After SOR:            56,819 points  (  153.0 ms)
Ground inliers:       52,424 points  (    2.1 ms)
Obstacle points:       4,395 points
Clusters found:            3          (    2.9 ms)
Total pipeline time:    162 ms
```

### Aerial LiDAR (241K points, --quick mode)

```
Voxel downsample (0.5m)      208,090       13.2 ms
Estimate normals (k=15)      208,090       54.8 ms
RANSAC ground plane           45,346        2.3 ms
Non-ground points            162,744
Euclidean clustering              21       16.4 ms
Total pipeline time:      87 ms
Throughput:           2,783,172 pts/sec
```

## Current limits

- **SOR dominates small pipelines**: Statistical outlier removal accounts for ~95% of
  the KITTI pipeline time (153 ms of 162 ms). It uses per-point KNN which is inherently
  O(n * k * log n). Replacing with a grid-based approach would help but is not yet done.
- **Normal estimation at 1M+**: Not yet benchmarked at 1M points. At 100K it takes 16 ms
  (rayon-parallel), but scaling behavior above 500K is untested.
- **ICP scales quadratically with correspondence search**: Point-to-point ICP at 10K
  takes 5.1 ms. At 100K+ it would be slow without subsampling or correspondence caching.
- **No SIMD**: SoA layout is SIMD-friendly but no explicit SIMD intrinsics are used yet.
  The compiler auto-vectorizes some loops but there is room for improvement.
- **Clustering edge explosion**: The grid + union-find clustering generates O(n * k)
  candidate pairs where k is the average cell occupancy. With very small radius relative
  to point density, cells become dense and pair generation dominates. Works well for
  typical LiDAR densities.
- **Single-threaded ICP**: ICP iterations are sequential (each depends on the previous).
  Only the KD-tree queries within each iteration are parallelizable.
- **No Open3D comparison data in CI**: The Open3D comparison numbers in earlier versions
  were measured manually. Automated comparison is not part of the test suite.
