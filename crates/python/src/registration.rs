use crate::cloud::PyPointCloud;
use pyo3::prelude::*;

#[pyclass(name = "IcpResult")]
#[derive(Debug, Clone)]
pub struct PyIcpResult {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub fitness: f32,
    #[pyo3(get)]
    pub rmse: f32,
    #[pyo3(get)]
    pub num_iterations: usize,
    #[pyo3(get)]
    pub translation: [f32; 3],
    #[pyo3(get)]
    pub rotation: [[f32; 3]; 3],
}

#[pymethods]
impl PyIcpResult {
    pub fn __repr__(&self) -> String {
        format!(
            "IcpResult(converged={}, rmse={:.6}, iterations={})",
            self.converged, self.rmse, self.num_iterations
        )
    }
}

#[pyfunction(name = "icp_point_to_point")]
#[pyo3(signature = (source, target, max_iterations=50, tolerance=1e-5, max_correspondence_distance=f32::INFINITY))]
pub fn icp_point_to_point_py(
    source: &PyPointCloud,
    target: &PyPointCloud,
    max_iterations: usize,
    tolerance: f32,
    max_correspondence_distance: f32,
) -> PyIcpResult {
    let params = pointclouds_registration::IcpParams {
        max_iterations,
        tolerance,
        max_correspondence_distance,
    };
    let result =
        pointclouds_registration::icp_point_to_point(&source.inner, &target.inner, &params);
    PyIcpResult {
        converged: result.converged,
        fitness: result.fitness,
        rmse: result.rmse,
        num_iterations: result.num_iterations,
        translation: result.transform.translation,
        rotation: result.transform.rotation,
    }
}

#[pyfunction(name = "icp_point_to_plane")]
#[pyo3(signature = (source, target, max_iterations=50, tolerance=1e-5, max_correspondence_distance=f32::INFINITY))]
pub fn icp_point_to_plane_py(
    source: &PyPointCloud,
    target: &PyPointCloud,
    max_iterations: usize,
    tolerance: f32,
    max_correspondence_distance: f32,
) -> PyResult<PyIcpResult> {
    // target must have normals
    let target_normals = target.inner.normals.as_ref().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "target cloud must have normals for point-to-plane ICP. \
             Use estimate_normals(target, k) first.",
        )
    })?;

    let params = pointclouds_registration::IcpParams {
        max_iterations,
        tolerance,
        max_correspondence_distance,
    };
    let result = pointclouds_registration::icp_point_to_plane(
        &source.inner,
        &target.inner,
        target_normals,
        &params,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyIcpResult {
        converged: result.converged,
        fitness: result.fitness,
        rmse: result.rmse,
        num_iterations: result.num_iterations,
        translation: result.transform.translation,
        rotation: result.transform.rotation,
    })
}

#[pyfunction(name = "apply_transform")]
pub fn apply_transform_py(
    cloud: &PyPointCloud,
    rotation: [[f32; 3]; 3],
    translation: [f32; 3],
) -> PyPointCloud {
    let transform = pointclouds_registration::RigidTransform {
        rotation,
        translation,
    };
    let out = pointclouds_registration::apply_transform(&cloud.inner, &transform);
    PyPointCloud { inner: out }
}
