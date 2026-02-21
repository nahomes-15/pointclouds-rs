use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pointclouds_core::PointCloud;
use pointclouds_filters::{
    passthrough_filter, radius_outlier_removal, statistical_outlier_removal,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_cloud(n: usize, seed: u64) -> PointCloud {
    let mut rng = StdRng::seed_from_u64(seed);
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..100.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..100.0)).collect();
    let z: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..100.0)).collect();
    PointCloud::from_xyz(x, y, z)
}

fn bench_passthrough(c: &mut Criterion) {
    let mut group = c.benchmark_group("passthrough_filter");
    for size in [100_000, 1_000_000] {
        let cloud = random_cloud(size, 42);
        group.bench_with_input(
            BenchmarkId::new("pointclouds-rs", size),
            &cloud,
            |b, cloud| b.iter(|| passthrough_filter(cloud, 'x', 25.0, 75.0)),
        );
    }
    group.finish();
}

fn bench_statistical_outlier(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_outlier_removal_k10");
    for size in [10_000, 100_000] {
        let cloud = random_cloud(size, 42);
        group.bench_with_input(
            BenchmarkId::new("pointclouds-rs", size),
            &cloud,
            |b, cloud| b.iter(|| statistical_outlier_removal(cloud, 10, 1.0)),
        );
    }
    group.finish();
}

fn bench_radius_outlier(c: &mut Criterion) {
    let mut group = c.benchmark_group("radius_outlier_removal");
    for size in [10_000, 100_000] {
        let cloud = random_cloud(size, 42);
        group.bench_with_input(
            BenchmarkId::new("pointclouds-rs", size),
            &cloud,
            |b, cloud| b.iter(|| radius_outlier_removal(cloud, 0.5, 5)),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_passthrough,
    bench_statistical_outlier,
    bench_radius_outlier
);
criterion_main!(benches);
