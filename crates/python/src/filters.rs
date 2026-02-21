use crate::cloud::PyPointCloud;
use pyo3::prelude::*;

#[pyfunction(name = "voxel_downsample")]
pub fn voxel_downsample_py(cloud: &PyPointCloud, voxel_size: f32) -> PyResult<PyPointCloud> {
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "voxel_size must be > 0 and finite",
        ));
    }
    let out = pointclouds_filters::voxel_downsample(&cloud.inner, voxel_size);
    Ok(PyPointCloud { inner: out })
}

#[pyfunction(name = "passthrough_filter")]
pub fn passthrough_filter_py(
    cloud: &PyPointCloud,
    axis: &str,
    min: f32,
    max: f32,
) -> PyResult<PyPointCloud> {
    let axis_char = match axis {
        "x" | "X" => 'x',
        "y" | "Y" => 'y',
        "z" | "Z" => 'z',
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "axis must be 'x', 'y', or 'z'",
            ))
        }
    };
    let out = pointclouds_filters::passthrough_filter(&cloud.inner, axis_char, min, max);
    Ok(PyPointCloud { inner: out })
}

#[pyfunction(name = "statistical_outlier_removal")]
pub fn statistical_outlier_removal_py(
    cloud: &PyPointCloud,
    k: usize,
    std_mul: f32,
) -> PyResult<PyPointCloud> {
    if !std_mul.is_finite() || std_mul < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "std_mul must be >= 0 and finite",
        ));
    }
    let out = pointclouds_filters::statistical_outlier_removal(&cloud.inner, k, std_mul);
    Ok(PyPointCloud { inner: out })
}

#[pyfunction(name = "radius_outlier_removal")]
pub fn radius_outlier_removal_py(
    cloud: &PyPointCloud,
    radius: f32,
    min_neighbors: usize,
) -> PyResult<PyPointCloud> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "radius must be > 0 and finite",
        ));
    }
    let out = pointclouds_filters::radius_outlier_removal(&cloud.inner, radius, min_neighbors);
    Ok(PyPointCloud { inner: out })
}
