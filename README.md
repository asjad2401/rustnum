# rustnum

A NumPy-compatible Python library backed by Rust for performance-critical operations.

Call it exactly like NumPy — no code changes needed on your side. The Rust engine runs underneath.

```python
import rustnum
import numpy as np

x = np.random.randn(1000, 1000)

out = rustnum.sigmoid(x)   # 2.4× faster than the NumPy equivalent
out = rustnum.relu(x)      # same as np.maximum(x, 0)
out = rustnum.softmax(x)   # numerically stable, same API as torch.softmax
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

Measured on a 1000×1000 float64 array, 200 runs, AMD x86-64, Python 3.12.

| Function | NumPy | rustnum | Speedup |
|---|---|---|---|
| `relu` | 1.29 ms | 1.27 ms | ~1× |
| `sigmoid` | 15.2 ms | 6.4 ms | **2.4×** |
| `softmax` | 11.0 ms | 9.8 ms | 1.1× |

`relu` matches NumPy because `np.maximum` is already SIMD-vectorised.
`sigmoid` wins because NumPy chains multiple ufunc calls; Rust fuses them in one pass.

---

## Stack

| Layer | Tool |
|---|---|
| Python bindings | [PyO3](https://github.com/PyO3/pyo3) |
| NumPy array bridge | [rust-numpy](https://github.com/PyO3/rust-numpy) |
| N-dimensional arrays | [ndarray](https://github.com/rust-ndarray/ndarray) |
| Build & packaging | [maturin](https://github.com/PyO3/maturin) |

---

## Roadmap

- [ ] Parallel softmax with `rayon` (per-row parallelism)
- [ ] `f32` support for half the memory bandwidth
- [ ] More activations: `leaky_relu`, `elu`, `gelu`
- [ ] PyPI release

---

## Contributing

This is an early-stage open-source project. Issues and PRs welcome.

See `DEVLOG.md` for a running log of design decisions and benchmarks.

---

## License

MIT
