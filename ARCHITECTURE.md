# Architecture

## Workspace Layout

```
pointclouds-rs/
├── crates/
│   ├── core/           # Foundation: PointCloud, Aabb, CloudView, traits
│   ├── spatial/        # KD-tree (kiddo-backed), octree/voxel grid stubs
│   ├── filters/        # Voxel downsample, passthrough, outlier removal
│   ├── normals/        # PCA-based normal estimation via SVD
│   ├── registration/   # ICP (point-to-point), rigid transforms
│   ├── segmentation/   # Euclidean clustering, RANSAC plane fitting
│   ├── io/             # PCD, PLY, LAS file I/O
│   └── python/         # PyO3 + maturin bindings
├── benches/            # Criterion benchmarks
├── tests/              # Integration tests + Python tests
├── examples/           # Rust usage examples
└── data/               # Sample PCD files
```

## Dependency Graph

```
                    core
                     │
                  spatial
                 ╱   │   ╲
           filters normals registration
                     │
              segmentation
                     │
                   python (depends on all)
                     │
                    io (depends on core only + las crate)
```

## Core Data Model

### SoA (Structure of Arrays) Design

The central `PointCloud` struct uses SoA layout for cache efficiency:

```rust
pub struct PointCloud {
    pub x: Vec<f32>,     // X coordinates (contiguous)
    pub y: Vec<f32>,     // Y coordinates (contiguous)
    pub z: Vec<f32>,     // Z coordinates (contiguous)
    pub normals: Option<Normals>,    // Optional SoA normals
    pub colors: Option<Colors>,      // Optional SoA colors (u8)
    pub intensity: Option<Vec<f32>>, // Optional per-point intensity
}
```

This design:
- Maximizes cache utilization for axis-aligned operations
- Enables future SIMD vectorization over contiguous coordinate arrays
- Avoids AoS padding overhead
- Supports optional attributes without allocation when unused

### f32-First Pipeline

All internal computation uses `f32`:
- Matches LiDAR/depth camera native precision
- 2x throughput vs f64 for SIMD operations
- Sufficient precision for point cloud processing (mm-scale accuracy)

## Spatial Indexing

The `KdTree` wraps the `kiddo` crate (v4) for O(log n) spatial queries:
- `knn(query, k)` — k-nearest neighbors with Euclidean distances
- `radius_search(query, radius)` — all points within radius
- Owns data (no lifetime ties to source cloud)
- Handles edge cases: empty clouds, NaN queries, boundary conditions

## Algorithm Implementations

### Normal Estimation (PCA)
For each point: find k-nearest neighbors → compute covariance matrix →
eigendecomposition → smallest eigenvector = normal. Normals oriented toward
viewpoint (default: origin).

### ICP Registration (SVD)
Iterative closest point with SVD-based optimal rigid transform:
1. Find nearest-neighbor correspondences via KdTree
2. Compute cross-covariance matrix
3. SVD decomposition → rotation matrix
4. Handle reflection (det < 0) by negating column
5. Convergence via RMSE change tolerance

### RANSAC Plane Fitting
Random sample consensus with seeded RNG:
1. Sample 3 points → fit plane via cross product
2. Count inliers within distance threshold
3. Track best model across iterations
4. Deterministic variant via `ransac_plane_seeded()`

### Euclidean Clustering
BFS-based connected component extraction:
1. Build KdTree
2. BFS from unvisited points using radius search
3. Filter clusters by min/max size
4. Sort by size (largest first)

## Safety

- All public crate APIs use `#![forbid(unsafe_code)]`
- No unsafe in the dependency-free `core` crate
- Python bindings use PyO3's safe abstractions
- `kiddo` and `nalgebra` handle their own internal unsafe

## Python Bindings

- Built with PyO3 + maturin
- ABI3 stable (Python >= 3.9)
- `PointCloud.from_numpy()` reads contiguous f32 Nx3 arrays
- All algorithms exposed as module-level functions
- Custom result classes (`IcpResult`, `PlaneResult`) with `__repr__`

## Testing Strategy

- Unit tests in each module with `#[cfg(test)]`
- Property-based tests via `proptest` (roundtrip invariants, output bounds)
- Integration tests in `tests/` (pipeline, I/O roundtrip)
- Python tests via `pytest` (15 tests covering all bindings)
- Criterion benchmarks for performance regression detection
