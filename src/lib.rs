// downcast is deprecated in PyO3 0.28 in favour of a not-yet-stable dtype-aware
// API in rust-numpy. Suppressed until upstream settles on the replacement.
#![allow(deprecated)]
use ndarray::{ArrayD, ArrayViewD, Axis};
use numpy::{Element, IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{exceptions::PyTypeError, prelude::*};
use rayon::prelude::*;
use std::{iter::Sum, ops::{Div, Sub}};

// ── float trait ───────────────────────────────────────────────────────────────
//
// Abstracts over f32/f64 so all implementations are written once.

trait FloatElem:
    Copy + Send + Sync + PartialOrd + Sum + Sub<Output = Self> + Div<Output = Self> + Element + 'static
{
    const NEG_INFINITY: Self;
    fn float_exp(self) -> Self;
    fn relu(self) -> Self;
    fn sigmoid(self) -> Self;
}

impl FloatElem for f64 {
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    fn float_exp(self) -> Self   { f64::exp(self) }
    fn relu(self) -> Self        { f64::max(self, 0.0) }
    fn sigmoid(self) -> Self     { 1.0 / (1.0 + f64::exp(-self)) }
}

impl FloatElem for f32 {
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    fn float_exp(self) -> Self   { f32::exp(self) }
    fn relu(self) -> Self        { f32::max(self, 0.0) }
    fn sigmoid(self) -> Self     { 1.0 / (1.0 + f32::exp(-self)) }
}

// ── generic implementations ───────────────────────────────────────────────────

fn relu_impl<F: FloatElem>(arr: ArrayViewD<F>) -> ArrayD<F> {
    arr.mapv(F::relu)
}

fn sigmoid_impl<F: FloatElem>(arr: ArrayViewD<F>) -> ArrayD<F> {
    arr.mapv(F::sigmoid)
}

fn softmax_impl<F: FloatElem>(arr: ArrayViewD<F>, ax: usize) -> ArrayD<F> {
    let mut out = arr.to_owned();
    out.lanes_mut(Axis(ax))
        .into_iter()
        .par_bridge()
        .for_each(|mut lane| {
            let max = lane.fold(F::NEG_INFINITY, |a, &b| if b > a { b } else { a });
            lane.mapv_inplace(|v| (v - max).float_exp());
            let sum: F = lane.iter().copied().sum();
            lane.mapv_inplace(|v| v / sum);
        });
    out
}

// ── python module ─────────────────────────────────────────────────────────────

#[pymodule]
mod rustnum {
    use super::*;

    /// relu(x)
    ///
    /// Element-wise Rectified Linear Unit. Accepts float32 or float64 arrays.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray (float32 or float64, any shape)
    ///
    /// Returns
    /// -------
    /// numpy.ndarray — same shape and dtype as input
    #[pyfunction]
    fn relu<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(arr) = x.downcast::<PyArrayDyn<f64>>() {
            Ok(relu_impl(arr.readonly().as_array()).into_pyarray(py).into_any())
        } else if let Ok(arr) = x.downcast::<PyArrayDyn<f32>>() {
            Ok(relu_impl(arr.readonly().as_array()).into_pyarray(py).into_any())
        } else {
            Err(PyTypeError::new_err("relu: expected float32 or float64 array"))
        }
    }

    /// sigmoid(x)
    ///
    /// Element-wise logistic sigmoid: 1 / (1 + exp(-x)).
    /// Accepts float32 or float64 arrays.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray (float32 or float64, any shape)
    ///
    /// Returns
    /// -------
    /// numpy.ndarray — same shape and dtype as input, values in (0, 1)
    #[pyfunction]
    fn sigmoid<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(arr) = x.downcast::<PyArrayDyn<f64>>() {
            Ok(sigmoid_impl(arr.readonly().as_array()).into_pyarray(py).into_any())
        } else if let Ok(arr) = x.downcast::<PyArrayDyn<f32>>() {
            Ok(sigmoid_impl(arr.readonly().as_array()).into_pyarray(py).into_any())
        } else {
            Err(PyTypeError::new_err("sigmoid: expected float32 or float64 array"))
        }
    }

    /// softmax(x, axis=-1)
    ///
    /// Numerically stable softmax over any axis. Accepts float32 or float64 arrays.
    ///
    /// Parameters
    /// ----------
    /// x    : numpy.ndarray (float32 or float64, any shape)
    /// axis : int — axis along which to normalise (default -1)
    ///
    /// Returns
    /// -------
    /// numpy.ndarray — same shape and dtype as input, sums to 1.0 along `axis`
    #[pyfunction]
    #[pyo3(signature = (x, axis=-1))]
    fn softmax<'py>(
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        axis: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        fn resolve_ax(axis: i64, ndim: usize) -> PyResult<usize> {
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
            Ok(ax)
        }

        if let Ok(arr) = x.downcast::<PyArrayDyn<f64>>() {
            let arr = arr.readonly();
            let ax = resolve_ax(axis, arr.ndim())?;
            Ok(softmax_impl(arr.as_array(), ax).into_pyarray(py).into_any())
        } else if let Ok(arr) = x.downcast::<PyArrayDyn<f32>>() {
            let arr = arr.readonly();
            let ax = resolve_ax(axis, arr.ndim())?;
            Ok(softmax_impl(arr.as_array(), ax).into_pyarray(py).into_any())
        } else {
            Err(PyTypeError::new_err("softmax: expected float32 or float64 array"))
        }
    }
}
