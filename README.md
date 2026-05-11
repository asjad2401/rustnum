# rustnum

A NumPy-compatible Python library backed by Rust for performance-critical operations.

Call it exactly like NumPy — no code changes needed on your side. The Rust engine runs underneath.

```python
import rustnum
import numpy as np

x64 = np.random.randn(1000, 1000)          # float64
x32 = x64.astype(np.float32)               # float32

rustnum.sigmoid(x64)   # float64 → float64, 1.9× faster than NumPy
rustnum.sigmoid(x32)   # float32 → float32, output dtype always matches input
rustnum.softmax(x64)   # 3.3× faster, parallelised across CPU cores
rustnum.relu(x32)      # same as np.maximum(x, 0)
```

---

## Why

NumPy is general-purpose. It doesn't include activation functions (`relu`, `sigmoid`,
`softmax`) because those belong to higher-level libraries. In practice, every DL
project implements them by hand, usually as a chain of NumPy operations — each one
allocating a temporary array, none of them as fast as a single fused Rust pass.

`rustnum` fills that gap: a thin Python API over a compiled Rust core.

---

## Installation

> Wheels are not yet on PyPI. Build from source for now.

**Prerequisites:** Rust toolchain (`rustup`), Python ≥ 3.8, `maturin`

```bash
git clone https://github.com/asjad2401/rustnum
cd rustnum
pip install maturin
maturin develop --release   # builds + installs into your active venv
```

---

## Functions

All functions accept **float32 or float64** arrays. Output dtype always matches input — no silent upcasting.

### `rustnum.relu(x)`
Element-wise Rectified Linear Unit.

```python
rustnum.relu(x)          # equivalent to np.maximum(x, 0)
```

### `rustnum.sigmoid(x)`
Element-wise logistic sigmoid: `1 / (1 + exp(-x))`

```python
rustnum.sigmoid(x)       # no single NumPy equivalent — users usually write the formula
```

### `rustnum.softmax(x, axis=-1)`
Numerically stable softmax over any axis.

```python
rustnum.softmax(x)           # over last axis (default)
rustnum.softmax(x, axis=0)   # over first axis
```

Subtracts the per-slice max before `exp` to prevent overflow — the same approach
used by PyTorch and JAX internally.

---

## Benchmarks

200 runs, AMD x86-64, Python 3.12.

**1000×1000, 200 runs, AMD x86-64, Python 3.12**

| Function | dtype | NumPy | rustnum | Speedup |
|---|---|---|---|---|
| `relu` | f64 | 1.41 ms | 1.35 ms | ~1× |
| `relu` | f32 | 0.50 ms | 0.57 ms | ~1× |
| `sigmoid` | f64 | 11.66 ms | 6.08 ms | **1.9×** |
| `sigmoid` | f32 | 3.96 ms | 3.44 ms | **1.15×** |
| `softmax` | f64 | 11.94 ms | 3.63 ms | **3.3×** |
| `softmax` | f32 | 4.00 ms | 1.83 ms | **2.2×** |

`relu` matches NumPy because `np.maximum` is already SIMD-vectorised.
`sigmoid` wins because NumPy chains multiple ufunc calls; Rust fuses them in one pass.
`softmax` is parallelised across CPU cores via rayon — each row processed independently.
f32 is faster in absolute terms (half the memory bandwidth); relative speedups are smaller because NumPy's f32 path is also faster.

---

## Stack

| Layer | Tool |
|---|---|
| Python bindings | [PyO3](https://github.com/PyO3/pyo3) |
| NumPy array bridge | [rust-numpy](https://github.com/PyO3/rust-numpy) |
| N-dimensional arrays | [ndarray](https://github.com/rust-ndarray/ndarray) |
| Parallelism | [rayon](https://github.com/rayon-rs/rayon) |
| Build & packaging | [maturin](https://github.com/PyO3/maturin) |

---

## Roadmap

- [x] Parallel softmax with `rayon` (3× speedup)
- [x] `f32` support — both dtypes accepted, output matches input
- [ ] More activations: `leaky_relu`, `elu`, `gelu`
- [ ] PyPI release

---

## Contributing

This is an early-stage open-source project. Issues and PRs welcome.

See `DEVLOG.md` for a running log of design decisions and benchmarks.

---

## License

MIT
