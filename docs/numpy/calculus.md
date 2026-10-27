# Interpolation, integration, differences and gradients

```python
import numpy as np
from unitx import Quantity, Unit
```

## Linear interpolation

`np.interp(x, xp, fp, left=None, right=None, period=None)` evaluates a
one-dimensional piecewise linear interpolant. Use linear Quantities, bare
numeric scalars/ndarrays, or numeric lists/tuples. Bare inputs are dimensionless,
including zero. AffineQuantity and LogQuantity points are unsupported; convert
explicitly first. At least one of x, xp or fp must be a Quantity for NumPy to
dispatch to unitx; a Quantity supplied only as left/right is insufficient.

`x` and `xp` must have compatible dimensions, including any logarithmic degree.
Query values convert to `xp.unit` before interpolation. `xp` must be a nonempty,
finite, strictly increasing real 1-D array; duplicates, decreasing/unsorted
coordinates and NaN/Inf are rejected. The check uses the float64 coordinates
passed to NumPy, so coordinates that collapse together at that precision are
also rejected. A single sample is allowed. Query x may be a real scalar or an
array of any shape and need not be sorted. Results have the same shape as x.

`fp` must be a real or complex 1-D array with the same length as xp. The result
is always a Quantity retaining the complete `fp.unit`, including an exact
extra scale. Supplied left/right values must be numeric scalars or scalar
Quantities with fp's dimension; compatible units convert to `fp.unit`.
Bare values remain dimensionless and do not implicitly acquire fp's unit.
Defaults repeat the first/last fp value outside the sample interval, rather
than extrapolating its slope. A scaled dimensionless fp retains its stored
coordinates, while bare dimensionless boundaries convert to those coordinates.

NumPy controls float64/complex128 evaluation, dtype acceptance, query NaN/Inf
and sample-value NaN/Inf behavior. Fraction/Decimal scalars enter its floating
domain; object, boolean, masked and subclass arrays are unsupported.
The wrapper does not modify any input. Non-None `period` is unsupported, as are
`out`, `dtype` and `where`. See
[NumPy interp](https://numpy.org/doc/stable/reference/generated/numpy.interp.html)
for the numerical definition; unitx additionally checks the coordinate order
and the unit rules above.

```python
sample_time = Quantity(np.array([0.0, 1, 3]), 's')
sample_current = Quantity(np.array([0.0, 2, 6]), 'A')
time = Quantity(np.array([0.0, 500, 1000, 2000, 3000]), 'ms')
current = np.interp(time, sample_time, sample_current)
assert current.unit == sample_current.unit
np.testing.assert_allclose(current.value, [0, 1, 2, 4, 6])
assert np.trapezoid(current, time).isclose(Quantity(9, 'C'))
np.testing.assert_allclose(np.gradient(current, time).value_in('A/s'), 2)
np.testing.assert_allclose(np.diff(current).value, [1, 1, 2, 2])
```

## Trapezoidal integration

`np.trapezoid(y, x=x)` accepts a Quantity for y, x, or both and returns a
Quantity with unit `y.unit * x.unit`; bare inputs are dimensionless. With
`x=None`, scalar `dx` supplies the coordinate unit instead (default: dimensionless
1). NumPy dispatches this function on y and x, so at least one of those must be
a Quantity to enter its handler. When only dx is a Quantity and y is a plain
numeric array, NumPy's native arithmetic also works, finishing with the
Quantity `sum` method. This returns the spacing unit; it does not extend
point-coordinate integration.
Explicit x makes dx irrelevant. Numeric arrays, scalar spacing, axis handling,
coordinate order (including reversed or nonmonotonic x), broadcasting, and
empty intervals follow [NumPy trapezoid](https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html).
No coordinate sorting, interpolation, dtype option, or output buffer is added.

## Differences and gradients

`np.diff(q, n=1, axis=-1)` computes differences in q's stored coordinates and
retains its unit for every order n. `prepend` and `append` may be numeric
scalars/arrays or Quantities; their dimensions must match the input, and their
values convert to the input unit before differencing. Bare boundaries are
dimensionless, including bare zero. A bare input with a Quantity boundary is
also dimensionless. Scaled dimensionless boundaries therefore normalize to the
input's coordinates. Explicit list inputs/boundaries remain unsupported.

With `n=0`, diff returns its input object unchanged and ignores axis and
boundaries. Other orders follow [NumPy diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html)
for shapes, scalar-boundary expansion, dtype promotion, unsigned wraparound,
and empty results. A difference is not a derivative: its unit is not divided
by a coordinate unit or raised to n.

`np.gradient(f, *spacing, axis=None, edge_order=1)` accepts numeric or Quantity
input and spacings. Each spacing is a scalar step or one-dimensional coordinate
array; bare inputs and spacings are dimensionless. A single scalar step applies
to every selected axis. Otherwise supply one spacing per selected axis, in
the order given by axis. Each component has unit `f.unit / spacing.unit`,
including exact Unit factors. No spacing means dimensionless unit steps.
The kernels use stored numerical coordinates; no implicit SI conversion is
needed because each result retains the corresponding unit ratio.

One selected axis returns one Quantity; multiple axes return a tuple of
Quantities, which may have different dimensions. No selected axes returns an
empty tuple. Output shapes, nonuniform/decreasing coordinates, edge_order,
dtype, and numerical warnings follow
[NumPy gradient](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html).
Coordinates are not sorted or interpolated. Neither function accepts out,
where, or dtype. These NumPy entry points require at least one Quantity among
their dispatched inputs; the existing NumPy stub limitations still apply.

```python
time = Quantity(np.array([0.0, 1, 3, 6]), 's')
distance = Quantity(np.array([0.0, 2, 6, 12]), 'm')
increments = np.diff(distance)
assert increments.unit == Unit('m')
velocity = np.gradient(distance, time, edge_order=2)
assert velocity.unit == Unit('m/s')
np.testing.assert_allclose(velocity.value, 2)
assert np.trapezoid(velocity, time).isclose(distance[-1] - distance[0])
```

[NumPy overview and support matrix](../numpy.md) · [Documentation index](../README.md)
