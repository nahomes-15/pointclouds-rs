use crate::cloud::PyPointCloud;
use pyo3::prelude::*;

#[pyfunction(name = "read_pcd")]
pub fn read_pcd_py(path: &str) -> PyResult<PyPointCloud> {
    let cloud = pointclouds_io::read_pcd(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(PyPointCloud { inner: cloud })
}

#[pyfunction(name = "write_pcd")]
pub fn write_pcd_py(path: &str, cloud: &PyPointCloud) -> PyResult<()> {
    pointclouds_io::write_pcd(path, &cloud.inner)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction(name = "read_ply")]
pub fn read_ply_py(path: &str) -> PyResult<PyPointCloud> {
    let cloud = pointclouds_io::read_ply(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(PyPointCloud { inner: cloud })
}

#[pyfunction(name = "write_ply")]
pub fn write_ply_py(path: &str, cloud: &PyPointCloud) -> PyResult<()> {
    pointclouds_io::write_ply(path, &cloud.inner)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction(name = "write_ply_binary")]
pub fn write_ply_binary_py(path: &str, cloud: &PyPointCloud) -> PyResult<()> {
    pointclouds_io::write_ply_binary(path, &cloud.inner)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction(name = "read_las")]
pub fn read_las_py(path: &str) -> PyResult<PyPointCloud> {
    let cloud = pointclouds_io::read_las(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(PyPointCloud { inner: cloud })
}
