use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pointclouds_core::PointCloud;
use pointclouds_spatial::KdTree;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_cloud(n: usize, seed: u64) -> PointCloud {
    let mut rng = StdRng::seed_from_u64(seed);
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..100.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..100.0)).collect();
    let z: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..100.0)).collect();
    PointCloud::from_xyz(x, y, z)
}

fn bench_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree_knn_10");
    for size in [100_000, 1_000_000] {
        let cloud = random_cloud(size, 42);
        let tree = KdTree::build(&cloud);
        // Fixed query point in the middle of the range
        let query = [50.0f32, 50.0, 50.0];
        group.bench_with_input(BenchmarkId::new("pointclouds-rs", size), &size, |b, _| {
            b.iter(|| tree.knn(&query, 10))
        });
    }
    group.finish();
}

fn bench_radius_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree_radius_search");
    for size in [100_000, 1_000_000] {
        let cloud = random_cloud(size, 42);
        let tree = KdTree::build(&cloud);
        let query = [50.0f32, 50.0, 50.0];
        group.bench_with_input(BenchmarkId::new("pointclouds-rs", size), &size, |b, _| {
            b.iter(|| tree.radius_search(&query, 0.1))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_knn, bench_radius_search);
criterion_main!(benches);
