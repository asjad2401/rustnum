# rustnum — Developer Log

A Python library with a NumPy-compatible API backed by Rust for performance.
Users `import rustnum` and call functions with the exact same signatures they
already know — no code changes on their side.

---

## Stack

| Layer | Tool | Purpose |
|---|---|---|
| Rust library | `cdylib` crate | compiled native extension |
| Python bindings | [PyO3](https://github.com/PyO3/pyo3) v0.28 | exposes Rust fns as Python callables |
| Array bridge | [rust-numpy](https://github.com/PyO3/rust-numpy) v0.28 | zero-copy NumPy ↔ ndarray conversion |
| Array ops | [ndarray](https://github.com/rust-ndarray/ndarray) v0.17 | N-dimensional array ops in Rust |
| Build / packaging | [maturin](https://github.com/PyO3/maturin) v1.13 | compiles + installs as a pip wheel |
| Parallelism | [rayon](https://github.com/rayon-rs/rayon) v1 | data parallelism across CPU cores |

---

## Session 1 — Project setup & first functions

### Goal
Pick a small set of functions that:
- Are compute-bound (benefit from native speed)
- Have a well-understood, stable API
- Are **absent from NumPy core** (relu, sigmoid, softmax — all common in DL, none are `np.*`)

### What we built

**`rustnum.relu(x)`**
Element-wise ReLU. Equivalent to `np.maximum(x, 0)`.
Identical signature; accepts any-shape float64 ndarray.

**`rustnum.sigmoid(x)`**
Element-wise logistic sigmoid: `1 / (1 + exp(-x))`.
No native NumPy equivalent — users currently write the formula by hand.

**`rustnum.softmax(x, axis=-1)`**
Numerically-stable softmax over any axis (subtracts max before exp).
No native NumPy equivalent — users currently write 3-line boilerplate.

### Key design decisions

**Zero-copy array passing**
`PyReadonlyArrayDyn` borrows the NumPy buffer directly — no Python→Rust copy on
input. The output is a fresh Rust-owned ndarray promoted to a NumPy array via
`into_pyarray`.

**Numerically stable softmax**
Subtracts the per-row max before `exp()` to prevent overflow. This is the
standard trick; our implementation mirrors what PyTorch/JAX do internally.

**`--release` build**
maturin is invoked with `--release` so LLVM optimisations (auto-vectorisation,
inlining) are active. Debug builds are ~10× slower.

### Benchmark results (1000×1000 float64, 200 runs, AMD x86-64)

| Function | NumPy | rustnum | Speedup |
|---|---|---|---|
| relu | 1.287 ms | 1.269 ms | ~1× |
| sigmoid | 15.226 ms | 6.382 ms | **2.4×** |
| softmax | 11.002 ms | 9.757 ms | 1.1× |

**relu**: NumPy's `maximum` is already SIMD-vectorised; parity is expected.
**sigmoid**: Large win — NumPy calls into a generic ufunc chain (`exp` + division);
Rust inlines and vectorises the whole thing in one pass.
**softmax**: Moderate win — the lane-by-lane iterator isn't yet parallelised.

### Known gaps / next steps

- [x] Parallel softmax with `rayon` → see Session 2
- [ ] `f32` support (half the memory bandwidth → potentially 2× throughput)
- [ ] `leaky_relu(x, alpha=0.01)` and `elu(x)` (same pattern, easy wins)
- [ ] SIMD explicit hints (or `std::simd` nightly) for relu
- [ ] Benchmark on larger arrays to see cache-miss behaviour
- [ ] Publish to PyPI as a real package

---

## Session 2 — Parallel softmax with rayon

### Goal
Replace the sequential lane-by-lane softmax loop with a parallel one using rayon,
without changing the public API at all.

### What changed

Added `rayon = "1"` and enabled `ndarray`'s `rayon` feature flag. In `softmax`,
replaced the sequential `for` loop with `.into_iter().par_bridge().for_each()`.

`par_bridge()` is rayon's adapter for bridging any standard `Iterator` into its
parallel scheduler. We use it because `LanesMut<IxDyn>` doesn't implement
`IntoParallelIterator` directly (ndarray's parallel support targets fixed-dimension
types natively; dynamic-dimension lanes need the bridge).

Each lane (1-D slice along the softmax axis) is independent — no shared mutable
state — so parallelising is safe and correct by construction.

### Benchmark results (200 runs, AMD x86-64, Python 3.12)

| Array shape | NumPy | rustnum | Speedup |
|---|---|---|---|
| 1000×1000 f64 | 11.348 ms | 3.720 ms | **3.05×** |
| 32×128×512 f64 | 26.135 ms | 8.123 ms | **3.22×** |

Correctness verified against NumPy on both 2D and 3D inputs with non-default axis.

### Known gaps / next steps

- [ ] `f32` support
- [ ] `leaky_relu`, `elu`, `gelu`
- [ ] Native `IntoParallelIterator` for `LanesMut<IxDyn>` (upstream ndarray contribution opportunity)
- [ ] PyPI release
