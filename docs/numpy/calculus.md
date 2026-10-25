# Integration, differences and gradients

```python
import numpy as np
from unitx import Quantity, Unit
```

## Trapezoidal integration

`np.trapezoid(y, x=x)` accepts a Quantity for y, x, or both and returns a
Quantity with unit `y.unit * x.unit`; bare inputs are dimensionless. With
`x=None`, scalar `dx` supplies the coordinate unit instead (default: dimensionless
1). NumPy dispatches this function on y and x, so at least one of those must be
a Quantity to enter unitx; a Quantity supplied only as dx is insufficient.
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
