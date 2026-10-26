# Linear algebra

These functions accept linear Quantities and bare numeric ndarrays/scalars.
Each operand has one uniform unit. Points and implicit list/tuple magnitudes
are unsupported. NumPy controls numerical algorithms, broadcasting, shape,
dtype, singular-matrix errors and available options; inputs are not modified.

| NumPy function | Result unit and options |
| --- | --- |
| `linalg.solve(a, b)` | Solution unit `b.unit / a.unit`; native matrix/vector and stacked forms |
| `linalg.tensorsolve(a, b, axes=None)` | Solution unit `b.unit / a.unit`; native axes permutation |
| `linalg.lstsq(a, b, rcond=None)` | `(solution, residuals, rank, singular_values)` with units described below |
| `linalg.inv(a)` | Reciprocal input unit |
| `linalg.pinv(a, rcond=None, hermitian=False, *, rtol=...)` | Reciprocal input unit; native cutoff selection and broadcasting |
| `linalg.tensorinv(a, ind=2)` | Reciprocal input unit; native tensor shape/ind rules |
| `linalg.matrix_power(a, n)` | Input unit raised to the bare integer n, including zero and negative powers |
| `linalg.det(a)` | Input unit raised to the matrix order; stacked results share that order |
| `linalg.norm(x, ord=None, axis=None, keepdims=False)` | Input unit; vector `ord=0` gives a dimensionless nonzero count |
| `linalg.vector_norm(x, *, axis=None, keepdims=False, ord=2)` | Input unit; `ord=0` gives a dimensionless nonzero count |
| `linalg.matrix_norm(x, *, keepdims=False, ord='fro')` | Input unit; operates on the last two axes |
| `linalg.cond(x, p=None)` | Dimensionless Quantity |
| `linalg.matrix_rank(A, tol=None, hermitian=False, *, rtol=None)` | Bare NumPy integer or integer ndarray |
| `linalg.qr(a, mode='reduced')` | Dimensionless Q and input-unit R; reduced, complete and r modes |
| `linalg.cholesky(a, *, upper=False)` | Input unit raised to the exact power 1/2 |
| `linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)` | Dimensionless U/Vh and input-unit singular values; only singular values when compute_uv=False |
| `linalg.svdvals(x)` | Input-unit singular values |
| `linalg.eig(a)`, `linalg.eigh(a, UPLO='L')` | Input-unit eigenvalues and dimensionless eigenvectors |
| `linalg.eigvals(a)`, `linalg.eigvalsh(a, UPLO='L')` | Input-unit eigenvalues |

Bare numerical operands are dimensionless. Solving does not require a and b
to have the same dimensions: their quotient determines the solution's unit.
Full extra unit scales remain exact in the result unit; they do not alter
NumPy's numerical kernels. A physical matrix with differently dimensioned
rows or columns cannot be stored in a single Quantity.

```python
import numpy as np
from unitx import Quantity
from unitx.si import si

resistance = Quantity(np.array([[3.0, 1.0], [1.0, 2.0]]), si.ohm)
voltage = Quantity(np.array([9.0, 8.0]), si.V)
current = np.linalg.solve(resistance, voltage)
np.testing.assert_allclose(current.value_in(si.A), [2, 3])
np.testing.assert_allclose((resistance @ current).value_in(si.V), voltage.value)
inverse = np.linalg.inv(resistance)
np.testing.assert_allclose((resistance @ inverse).value_in(si.one), np.eye(2), atol=1e-14)
```

## Least squares and cutoffs

`lstsq` returns four items:

- The solution has unit `b.unit / a.unit`.
- Residuals are sums of squared absolute errors and have unit `b.unit**2`,
  including empty residual arrays for underdetermined or rank-deficient cases.
- Rank is NumPy's bare integer.
- Singular values retain a's unit.

`rcond` in `lstsq`/`pinv`, and `rtol` in `pinv`/`matrix_rank`, are relative
cutoffs. Bare numbers/arrays follow NumPy. Dimensionless Quantity cutoffs
normalize their complete scale before use. Absolute `matrix_rank(tol=...)`
cutoffs can be Quantities compatible with A's unit; they are converted to A's
stored unit. Bare absolute thresholds are in A's stored numerical domain, so
changing A's unit can change the rank obtained with the same bare threshold.
Use a Quantity threshold to specify the same physical resolution.
Cutoff lists/tuples are accepted as numerical parameter arrays, not magnitudes.
NumPy controls defaults, broadcasting and mutually exclusive cutoff errors;
legacy `lstsq(rcond=-1)` behavior is preserved.

```python
lengths = Quantity(np.diag([1.0, 0.005]), si.m)
assert np.linalg.matrix_rank(lengths) == 2
assert np.linalg.matrix_rank(lengths, tol=1 * si.cm) == 1
assert np.linalg.matrix_rank(lengths.to(si.cm), tol=1 * si.cm) == 1
```

## Norms and normalization

Vector norms and matrix norms, including Frobenius, nuclear and spectral
norms, retain the complete input unit. The vector `ord=0` operation counts
nonzero entries and returns a dimensionless Quantity. Points are unsupported:
the norm of coordinate readings would depend on the coordinate origin.
Dividing by a norm gives a dimensionless normalized quantity.

```python
signal = Quantity(np.array([3.0, 4.0]), si.V)
amplitude = np.linalg.vector_norm(signal)
assert amplitude.unit == si.V and amplitude.value == 5
np.testing.assert_allclose((signal / amplitude).value_in(si.one), [0.6, 0.8])
```

## Factorizations and eigenproblems

QR, SVD and eigenvector results retain NumPy's namedtuple fields (`Q`/`R`,
`U`/`S`/`Vh`, `eigenvalues`/`eigenvectors`) and support tuple unpacking. Each
component is a Quantity. Singular values, eigenvalues and R retain the full
input unit, including exact extra scales. Orthogonal/unitary vectors use the
coherent dimensionless unit. Cholesky factors carry the exact square root of
the full unit. The reconstruction equations therefore retain their units:

```python
q, r = np.linalg.qr(resistance)
np.testing.assert_allclose((q @ r).value_in(si.ohm), resistance.value)
u, s, vh = np.linalg.svd(resistance, full_matrices=False)
np.testing.assert_allclose(((u * s) @ vh).value_in(si.ohm), resistance.value)
eigenvalues, eigenvectors = np.linalg.eigh(resistance)
np.testing.assert_allclose(
    (resistance @ eigenvectors).value_in(si.ohm),
    (eigenvectors * eigenvalues).value_in(si.ohm),
)
lower = np.linalg.cholesky(resistance)
np.testing.assert_allclose((lower @ lower.T).value_in(si.ohm), resistance.value)
```

QR supports `mode='reduced'`, `'complete'` and `'r'`; the last returns just R.
`'raw'` and deprecated modes are rejected. Raw QR packs dimensionless
Householder reflectors and input-unit triangular entries in the same array,
which cannot be represented by one uniform unit. Extract numerical values
explicitly when using that low-level representation.

NumPy controls stacked/rectangular matrix shapes, complex conjugation, result
dtypes and numerical failures. `eigh`/`eigvalsh` use the selected `UPLO`
triangle, and Cholesky uses the triangle selected by `upper`; unitx does not
add a Hermitian-symmetry check. None of these functions accepts an `out` buffer.

NumPy's static annotations can still describe dispatched results as ndarrays
or Any. These runtime implementations do not replace NumPy's stubs; see the
[typing guide](../typing.md#numpy-component-inference).

References: [NumPy linear algebra](https://numpy.org/doc/stable/reference/routines.linalg.html),
[least-squares results](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html),
[rank thresholds](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html),
[QR modes](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html),
[SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html).

[NumPy support matrix](../numpy.md) · [Documentation index](../README.md)
