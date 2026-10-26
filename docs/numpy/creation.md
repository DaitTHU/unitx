# Allocation, grids and padding

`zeros_like`, `ones_like`, `empty_like` and `full_like` retain the template's
complete unit or point coordinate. NumPy controls dtype, order, shape and device
options; unsupported magnitude dtypes remain rejected. Bare `full_like` fill
values are **readings in the template's unit**, including list/tuple fills.
Quantity fills convert to that unit; point fills must be compatible points.
The template must carry the unit: NumPy does not dispatch `full_like` on its
fill argument alone. Allocations own their storage, and `empty_like` leaves
values uninitialized.

`linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
axis=0, *, device=None)` aligns compatible endpoints to the first wrapped
endpoint's complete unit or point coordinate. Bare endpoints are dimensionless.
Points produce points; `retstep=True` returns a **difference** step with the
coordinate's delta unit. NumPy controls array endpoints, broadcasting, integer
rounding, endpoint and empty/singleton behavior.

`geomspace` has the same endpoint alignment, dtype and axis rules but accepts
only linear quantities. Negative and complex endpoints follow NumPy's kernel.
`logspace` requires dimensionless linear exponents and base, normalizes their
full scales, and returns a coherent dimensionless Quantity. For a dimensional
logarithmic grid, use `geomspace` or explicitly multiply a dimensionless grid
by a reference quantity.

`meshgrid(*xi, copy=True, sparse=False, indexing='xy')` retains each axis's own
unit or coordinate independently; bare numeric axes stay bare. Different
dimensions are allowed. Copy, sparse and indexing behavior follows NumPy;
`copy=False` can expose overlapping storage, so copy before writing.

`pad(array, pad_width, mode='constant', **kwargs)` supports all eleven NumPy
built-in modes. Bare `constant_values` and `end_values` are stored readings;
wrapped fills convert as for `full_like`, including nested per-axis pairs.
The default fill is a zero reading. Statistical padding and odd reflection
operate on stored readings and retain point coordinates. Callbacks are
unsupported because they would expose unitless mutable views. NumPy validates
mode-specific keywords, widths, rounding and errors; `empty` leaves padded
values uninitialized. Padding allocates new storage.

`np.lib.stride_tricks.sliding_window_view(x, window_shape, axis=None, *,
subok=False, writeable=False)` retains units/coordinates and NumPy's shared
storage. It is read-only by default; explicitly writable windows can overlap.
`subok` does not enable ndarray subclasses in any of these APIs.

```python
import numpy as np
from unitx import AffineQuantity, Quantity, Unit

t, dt = np.linspace(Quantity(0, 's'), Quantity(1000, 'ms'), 5, retstep=True)
assert dt == Quantity(0.25, 's')
temperature = np.linspace(AffineQuantity(0, 'degC'), AffineQuantity(293.15, 'K'), 5)
assert temperature.unit == AffineQuantity(0, 'degC').unit
filled = np.full_like(t, Quantity(500, 'ms'))
np.testing.assert_allclose(filled.value, 0.5)
x = np.geomspace(Quantity(1, 'cm'), Quantity(1, 'm'), 3)
assert x.unit == Unit('cm')
xx, tt = np.meshgrid(x, t, indexing='ij', sparse=True)
assert xx.unit == x.unit and tt.unit == t.unit
padded = np.pad(x, 1, mode='edge')
windows = np.lib.stride_tricks.sliding_window_view(padded, 3)
assert windows.unit == x.unit and not windows.value.flags.writeable
```

[NumPy overview](../numpy.md) · [NumPy allocation reference](https://numpy.org/doc/stable/reference/routines.array-creation.html)
