use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pointclouds_core::PointCloud;
use pointclouds_normals::estimate_normals;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_cloud(n: usize, seed: u64) -> PointCloud {
    let mut rng = StdRng::seed_from_u64(seed);
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..100.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..100.0)).collect();
    let z: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..100.0)).collect();
    PointCloud::from_xyz(x, y, z)
}

fn bench_estimate_normals(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_normals_k10");
    for size in [10_000, 100_000] {
        let cloud = random_cloud(size, 42);
        group.bench_with_input(
            BenchmarkId::new("pointclouds-rs", size),
            &cloud,
            |b, cloud| b.iter(|| estimate_normals(cloud, 10)),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_estimate_normals);
criterion_main!(benches);
