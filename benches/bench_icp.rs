use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pointclouds_core::PointCloud;
use pointclouds_registration::{apply_transform, icp_point_to_point, IcpParams, RigidTransform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_cloud(n: usize, seed: u64) -> PointCloud {
    let mut rng = StdRng::seed_from_u64(seed);
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..100.0)).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..100.0)).collect();
    let z: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..100.0)).collect();
    PointCloud::from_xyz(x, y, z)
}

fn bench_icp_point_to_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("icp_point_to_point");
    let params = IcpParams::default();

    // Translation of (0.1, 0, 0)
    let translation = RigidTransform {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.1, 0.0, 0.0],
    };

    for size in [1_000, 10_000] {
        let source = random_cloud(size, 42);
        let target = apply_transform(&source, &translation);
        group.bench_with_input(BenchmarkId::new("pointclouds-rs", size), &size, |b, _| {
            b.iter(|| icp_point_to_point(&source, &target, &params))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_icp_point_to_point);
criterion_main!(benches);
