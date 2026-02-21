#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

mod cloud;
mod filters;
mod io;
mod normals;
mod registration;
mod segmentation;

#[pymodule]
fn pointclouds_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core
    m.add_class::<cloud::PyPointCloud>()?;

    // Filters
    m.add_function(wrap_pyfunction!(filters::voxel_downsample_py, m)?)?;
    m.add_function(wrap_pyfunction!(filters::passthrough_filter_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        filters::statistical_outlier_removal_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(filters::radius_outlier_removal_py, m)?)?;

    // Normals
    m.add_function(wrap_pyfunction!(normals::estimate_normals_py, m)?)?;

    // Registration
    m.add_class::<registration::PyIcpResult>()?;
    m.add_function(wrap_pyfunction!(registration::icp_point_to_point_py, m)?)?;
    m.add_function(wrap_pyfunction!(registration::icp_point_to_plane_py, m)?)?;
    m.add_function(wrap_pyfunction!(registration::apply_transform_py, m)?)?;

    // Segmentation
    m.add_class::<segmentation::PyPlaneResult>()?;
    m.add_function(wrap_pyfunction!(segmentation::euclidean_cluster_py, m)?)?;
    m.add_function(wrap_pyfunction!(segmentation::ransac_plane_py, m)?)?;

    // IO
    m.add_function(wrap_pyfunction!(io::read_pcd_py, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_pcd_py, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_ply_py, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_ply_py, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_ply_binary_py, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_las_py, m)?)?;

    Ok(())
}
