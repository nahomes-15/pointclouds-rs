use crate::cloud::PyPointCloud;
use pyo3::prelude::*;

#[pyfunction(name = "euclidean_cluster")]
pub fn euclidean_cluster_py(
    cloud: &PyPointCloud,
    distance_threshold: f32,
    min_size: usize,
    max_size: usize,
) -> Vec<Vec<usize>> {
    pointclouds_segmentation::euclidean_cluster(
        &cloud.inner,
        distance_threshold,
        min_size,
        max_size,
    )
}

#[pyclass(name = "PlaneResult")]
#[derive(Debug, Clone)]
pub struct PyPlaneResult {
    #[pyo3(get)]
    pub normal: [f32; 3],
    #[pyo3(get)]
    pub d: f32,
    #[pyo3(get)]
    pub inliers: Vec<usize>,
}

#[pymethods]
impl PyPlaneResult {
    pub fn __repr__(&self) -> String {
        format!(
            "PlaneResult(normal={:?}, d={:.4}, inliers={})",
            self.normal,
            self.d,
            self.inliers.len()
        )
    }
}

#[pyfunction(name = "ransac_plane")]
pub fn ransac_plane_py(
    cloud: &PyPointCloud,
    distance_threshold: f32,
    iterations: usize,
) -> PyPlaneResult {
    let (model, inliers) =
        pointclouds_segmentation::ransac_plane(&cloud.inner, distance_threshold, iterations);
    PyPlaneResult {
        normal: model.normal,
        d: model.d,
        inliers,
    }
}
