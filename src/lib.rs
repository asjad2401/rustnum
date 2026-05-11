use ndarray::Axis;
use rayon::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// ── helpers ──────────────────────────────────────────────────────────────────

#[inline(always)]
fn relu_f64(x: f64) -> f64 {
    x.max(0.0)
}

#[inline(always)]
fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ── module ───────────────────────────────────────────────────────────────────

#[pymodule]
mod rustnum {
    use super::*;

    /// relu(x)
    ///
    /// Element-wise Rectified Linear Unit.  Identical call signature to what
    /// you'd write by hand in NumPy:  ``np.maximum(x, 0)``
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray (float64, any shape)
    ///
    /// Returns
    /// -------
    /// numpy.ndarray  — same shape as input
    #[pyfunction]
    fn relu<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, f64>) -> Bound<'py, PyArrayDyn<f64>> {
        let arr = x.as_array();
        let out = arr.mapv(relu_f64);
        out.into_pyarray(py)
    }

    /// sigmoid(x)
    ///
    /// Element-wise logistic sigmoid:  1 / (1 + exp(-x))
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray (float64, any shape)
    ///
    /// Returns
    /// -------
    /// numpy.ndarray  — same shape as input, values in (0, 1)
    #[pyfunction]
    fn sigmoid<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'py, f64>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        let arr = x.as_array();
        let out = arr.mapv(sigmoid_f64);
        out.into_pyarray(py)
    }

    /// softmax(x, axis=-1)
    ///
    /// Softmax over the last axis (or any axis you choose).  Numerically
    /// stable: subtracts max before exp to prevent overflow.
    ///
    /// Parameters
    /// ----------
    /// x    : numpy.ndarray (float64, any shape)
    /// axis : int  — axis along which to normalise (default -1)
    ///
    /// Returns
    /// -------
    /// numpy.ndarray  — same shape as input, sums to 1.0 along `axis`
    #[pyfunction]
    #[pyo3(signature = (x, axis=-1))]
    fn softmax<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'py, f64>,
        axis: i64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let arr = x.as_array();
        let ndim = arr.ndim();

        // normalise negative axis
        let ax = if axis < 0 {
            (ndim as i64 + axis) as usize
        } else {
            axis as usize
        };

        if ax >= ndim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "axis {axis} out of bounds for array with {ndim} dimensions"
            )));
        }

        let mut out = arr.to_owned();

        // normalise every 1-D slice along `ax` in parallel across CPU cores
        out.lanes_mut(Axis(ax))
            .into_iter()
            .par_bridge()
            .for_each(|mut lane| {
                let max = lane.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                lane.mapv_inplace(|v| (v - max).exp());
                let sum: f64 = lane.iter().sum();
                lane.mapv_inplace(|v| v / sum);
            });

        Ok(out.into_pyarray(py))
    }
}
