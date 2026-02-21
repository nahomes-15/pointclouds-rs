# Benchmarks

All benchmarks use [Criterion.rs](https://github.com/bheisler/criterion.rs) with 100
samples per measurement. Run on macOS arm64 (Apple Silicon), Rust 1.92.0, release profile.

## Running

```bash
# All benchmarks
cargo bench --workspace

# Individual benchmarks
cargo bench -p pointclouds-benches --bench bench_voxel
cargo bench -p pointclouds-benches --bench bench_kdtree
cargo bench -p pointclouds-benches --bench bench_normals
cargo bench -p pointclouds-benches --bench bench_icp
cargo bench -p pointclouds-benches --bench bench_filters
```

## Results

### Voxel Downsample

| Points | Mean | Notes |
|--------|------|-------|
| 10K | 58.6 us | |
| 100K | 671.0 us | |
| 1M | 7.57 ms | ~132K points/ms |

### KD-Tree Queries (per query)

| Points in tree | KNN (k=10) | Radius Search |
|---------------|-----------|---------------|
| 100K | 303.5 ns | 71.6 ns |
| 1M | 348.1 ns | 88.5 ns |

KD-Tree query times scale logarithmically — going from 100K to 1M points (10x)
increases query time by only ~15%.

### Normal Estimation (PCA, k=10)

| Points | Mean |
|--------|------|
| 10K | 11.3 ms |
| 100K | 125.9 ms |

### ICP Point-to-Point (max 50 iterations)

| Points | Mean |
|--------|------|
| 1K | 393.2 us |
| 10K | 4.98 ms |

### Passthrough Filter

| Points | Mean |
|--------|------|
| 100K | 354.4 us |
| 1M | 5.13 ms |

### Statistical Outlier Removal (k=10)

| Points | Mean |
|--------|------|
| 10K | 8.80 ms |
| 100K | 106.5 ms |

### Radius Outlier Removal

| Points | Mean |
|--------|------|
| 10K | 1.43 ms |
| 100K | 22.75 ms |

## Summary Table

| Algorithm | 1K | 10K | 100K | 1M |
|-----------|-----|------|-------|------|
| Voxel Downsample | -- | 58.6 us | 671 us | 7.57 ms |
| KD-Tree KNN k=10 (per query) | -- | -- | 304 ns | 348 ns |
| KD-Tree Radius (per query) | -- | -- | 71.6 ns | 88.5 ns |
| Normal Estimation k=10 | -- | 11.3 ms | 126 ms | -- |
| ICP Point-to-Point | 393 us | 4.98 ms | -- | -- |
| Passthrough Filter | -- | -- | 354 us | 5.13 ms |
| Statistical Outlier k=10 | -- | 8.80 ms | 107 ms | -- |
| Radius Outlier Removal | -- | 1.43 ms | 22.8 ms | -- |

## Open3D Comparison

Run the comparison script:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install open3d numpy maturin
maturin develop --release --manifest-path crates/python/Cargo.toml
python tests/bench_vs_open3d.py
```

### Results (macOS arm64, Apple Silicon, Python 3.12, Open3D 0.19)

| Algorithm | 100K Open3D | 100K pcr | Speedup | 1M Open3D | 1M pcr | Speedup |
|-----------|------------|---------|---------|----------|--------|---------|
| Voxel Downsample | 9.85 ms | 5.47 ms | **1.8x** | 264 ms | 54.7 ms | **4.8x** |
| Passthrough Filter | 5.31 ms | 0.34 ms | **15.5x** | 54.3 ms | 5.01 ms | **10.8x** |
| Normal Estimation k=15 | 49.7 ms | 19.7 ms | **2.5x** | 715 ms | 280 ms | **2.6x** |
| RANSAC Plane (1000 iter) | 9.23 ms | 2.93 ms | **3.1x** | 83.2 ms | 27.2 ms | **3.1x** |

**Average speedup: 5.5x** across all algorithms and sizes.

At 1M points (the target scale):
- Voxel Downsample: **4.8x** faster
- Passthrough Filter: **10.8x** faster
- RANSAC Plane: **3.1x** faster
- Normal Estimation: **2.6x** (bounded by KdTree query speed — kiddo vs FLANN)

### Optimizations applied

- **Normal estimation**: Rayon-parallelized per-point computation + analytical 3x3
  eigensolver (Cardano's formula) replacing nalgebra's general iterative solver.
  Improved from 0.3x to 2.5x vs Open3D.
- **RANSAC**: Rayon-parallelized iteration batches + pre-extracted contiguous point
  array + adaptive early termination. Improved from 0.4x to 3.1x vs Open3D.
