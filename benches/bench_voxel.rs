use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pointclouds_core::PointCloud;
use pointclouds_filters::voxel_downsample;

fn random_cloud(n: usize) -> PointCloud {
    let x = (0..n).map(|i| i as f32 * 0.001).collect();
    let y = (0..n).map(|i| i as f32 * 0.002).collect();
    let z = (0..n).map(|i| i as f32 * 0.003).collect();
    PointCloud::from_xyz(x, y, z)
}

fn bench_voxel(c: &mut Criterion) {
    let mut group = c.benchmark_group("voxel_downsample");
    for size in [10_000, 100_000, 1_000_000] {
        let cloud = random_cloud(size);
        group.bench_with_input(
            BenchmarkId::new("pointclouds-rs", size),
            &cloud,
            |b, cloud| b.iter(|| voxel_downsample(cloud, 0.05)),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_voxel);
criterion_main!(benches);
