use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[pyclass(name = "PointCloud")]
#[derive(Debug, Clone)]
pub struct PyPointCloud {
    pub(crate) inner: pointclouds_core::PointCloud,
}

#[pymethods]
impl PyPointCloud {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: pointclouds_core::PointCloud::new(),
        }
    }

    /// Create a PointCloud from an Nx3 NumPy array.
    ///
    /// Accepts f32 or f64 arrays. f64 arrays are cast to f32 automatically.
    /// The array must be C-contiguous (row-major). Fortran-order arrays are
    /// rejected to prevent silent data corruption.
    #[staticmethod]
    pub fn from_numpy(py: Python<'_>, array: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        // Try f32 first
        if let Ok(arr) = array.downcast::<PyArray2<f32>>() {
            return Self::from_f32_array(arr);
        }
        // Try f64 and auto-cast
        if let Ok(arr) = array.downcast::<PyArray2<f64>>() {
            return Self::from_f64_array(py, arr);
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected NumPy array with dtype float32 or float64, shape (N, 3)",
        ))
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let data = self.inner.to_array();
        Ok(PyArray2::from_vec2_bound(
            py,
            &data.chunks(3).map(|c| c.to_vec()).collect::<Vec<_>>(),
        )?)
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn __repr__(&self) -> String {
        format!("PointCloud(n={})", self.inner.len())
    }
}

impl PyPointCloud {
    fn from_f32_array(array: &Bound<'_, PyArray2<f32>>) -> PyResult<Self> {
        // Reject non-C-contiguous arrays (e.g., Fortran-order) to prevent
        // silent data corruption from misinterpreted memory layout.
        if !array.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous (row-major). \
                 Use numpy.ascontiguousarray(arr) to convert.",
            ));
        }
        let readonly = array.readonly();
        let shape = readonly.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "expected shape (N, 3)",
            ));
        }
        let slice = readonly.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("failed to read array as contiguous slice")
        })?;
        let cloud = pointclouds_core::PointCloud::from_array(slice, shape[0]);
        Ok(Self { inner: cloud })
    }

    fn from_f64_array(py: Python<'_>, array: &Bound<'_, PyArray2<f64>>) -> PyResult<Self> {
        if !array.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous (row-major). \
                 Use numpy.ascontiguousarray(arr) to convert.",
            ));
        }
        let readonly = array.readonly();
        let shape = readonly.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "expected shape (N, 3)",
            ));
        }
        let slice = readonly.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("failed to read array as contiguous slice")
        })?;
        // Cast f64 → f32
        let f32_data: Vec<f32> = slice.iter().map(|&v| v as f32).collect();
        let cloud = pointclouds_core::PointCloud::from_array(&f32_data, shape[0]);
        // Suppress unused warning — py is needed for API consistency
        let _ = py;
        Ok(Self { inner: cloud })
    }
}
