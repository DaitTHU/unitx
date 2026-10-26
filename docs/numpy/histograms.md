# Histograms and covariance

NumPy controls numerical algorithms, dtype, bin selection, shape validation,
NaNs and weight restrictions. unitx supplies coordinate alignment and result
units. Plain numeric arrays/scalars and Quantity/point arrays are supported.

## Covariance and correlation

`np.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
aweights=None, *, dtype=None)` returns a Quantity in the squared difference
unit of m. With y, its difference unit must be compatible and its values scale
to m's difference unit. Coordinate origins do not affect centered covariance.
Relative weights and ddof can be coherent dimensionless Quantities.

Different dimensions cannot share a single covariance matrix unit: combining
length and time would need m², m·s and s² in different cells. Use
[scalar covariance](../statistics.md) for the cross term.
`np.corrcoef(x, y=None, rowvar=True, *, dtype=None)` returns dimensionless
coefficients and allows independently dimensioned variables. Native deprecated
bias/ddof options and their warnings follow the installed NumPy version.

## Bins, edges and weights

`histogram`, `histogram_bin_edges`, `histogram2d`, `histogramdd` and `digitize`
align explicit edges and range endpoints to each input coordinate. Bare
integers select bin counts; strings select native 1-D bin algorithms.
Dimensional samples require explicit units on edges/bounds. Points require
compatible point edges, preserving their coordinate in returned edges.
Bare edge lists are valid for dimensionless samples; lists of scalar quantities
or a Quantity array may specify dimensional edges.

A single histogramdd Quantity matrix has shape (N, D) and one unit for all
columns. A coordinate sequence `(x, y, z)` permits independently dimensioned
1-D arrays. `histogram2d` has NumPy's shared-edge-vector or per-axis bins rules.

Without density or Quantity weights, counts are bare numeric arrays with native
dtypes (2-D/ND counts can be floating). Quantity weights give counts in the
weight unit. `density=True` cancels the weight unit and returns the inverse
product of coordinate difference units. Thus density integrates to one when
multiplied by bin widths; its entries need not sum to one. Points are not weights.
`digitize` returns bare integer indices. Weighted `bincount` takes bare integer
indices, supports minlength, and preserves a Quantity weight's unit.

```python
import numpy as np
from unitx import Quantity
from unitx.si import si

x = Quantity(np.array([0.0, 1, 2, 3]), si.m)
density, edges = np.histogram(x, bins=2, density=True)
assert density.unit == si.m**-1 and edges.unit == si.m
np.testing.assert_allclose(np.sum(density * np.diff(edges)).value, 1)
y = Quantity(np.array([0.0, 1, 3, 4]), si.s)
density2, xe, ye = np.histogram2d(x, y, bins=2, density=True)
integral = np.sum(density2 * np.diff(xe)[:, None] * np.diff(ye)[None, :])
np.testing.assert_allclose(integral.value, 1)
```

These functions have no output buffers and do not modify inputs. Automatic
bin algorithms with unsupported weights fail according to NumPy.

References: [NumPy cov](https://numpy.org/doc/stable/reference/generated/numpy.cov.html),
[histogram](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html),
[histogramdd](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html).
