use crate::cloud::PyPointCloud;
use pyo3::prelude::*;

#[pyfunction(name = "estimate_normals")]
pub fn estimate_normals_py(cloud: &PyPointCloud, k: usize) -> PyResult<PyPointCloud> {
    let normals = pointclouds_normals::estimate_normals(&cloud.inner, k);
    let mut out = cloud.inner.clone();
    out.normals = Some(normals);
    Ok(PyPointCloud { inner: out })
}
