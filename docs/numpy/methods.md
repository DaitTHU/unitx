# Direct array methods

For NumPy magnitudes, Quantity provides `reshape`, `ravel`, `copy`, `transpose`,
`squeeze`, `sum`, `mean` and `std`. AffineQuantity and LogQuantity provide the same methods
except `sum`. Mean preserves a point's coordinate; standard deviation returns
its difference unit. Logarithmic statistics operate on readings; convert to
physical values explicitly before averaging power or amplitude.

`reshape((m, n))` and `reshape(m, n)` accept `order='C'/'F'/'A'` and
`copy=None/True/False`, following NumPy's view/copy rules. `ravel` accepts
`C/F/A/K` and copies only when needed. `copy(order='C')` creates independent
ndarray storage, including for a NumPy scalar. Its C default follows
[ndarray.copy](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html);
`np.copy(q)` keeps NumPy's K default. Native reshape of a NumPy scalar to `()`
can remain a NumPy scalar; ndarray input remains ndarray with the same dtype.
Python scalar magnitudes require the scalar APIs or explicit array construction.

`transpose()` reverses the axes; `transpose((2, 0, 1))` and
`transpose(2, 0, 1)` select a permutation. Axis inputs are positional.
`squeeze(axis=None)` removes all length-one axes, or only the selected integer
or tuple of axes. Both retain the complete unit, coordinate family and dtype,
and share storage for ndarray inputs. A fully squeezed ndarray remains a
zero-dimensional ndarray; a NumPy scalar remains a NumPy scalar. Invalid
axes raise native NumPy errors without changing the input.

Reductions accept `axis`, `dtype`, `out`, `keepdims` and `where`. Sum also
accepts a compatible `initial` (bare zero is the additive identity); std
supports `ddof`, `correction` and a compatible precomputed `mean`. These use
the existing [statistics](statistics.md) and
[fixed-unit output](writes.md#output-buffers) rules, including returning
the original out object and staging writes. Numerical result dtypes follow
NumPy; a reduction need not preserve the input dtype or ndarray storage.
Direct methods retain the wrapper in static types; general NumPy function
inference still has the [documented limits](../typing.md#numpy-magnitude-types).

```python
import numpy as np
from unitx import Quantity

samples = Quantity(np.arange(6.0), 'm').reshape(2, 3)
np.testing.assert_allclose(samples.mean(axis=1).value, [1, 4])
assert samples.sum(initial=Quantity(100, 'cm')) == Quantity(16, 'm')
assert np.shares_memory(samples.ravel().value, samples.value)
assert not np.shares_memory(samples.copy().value, samples.value)
channels = samples.reshape(2, 1, 3).transpose(2, 0, 1).squeeze(-1)
np.testing.assert_array_equal(channels.value, samples.value.T)
assert np.trapezoid(np.array([1.0, 2, 3]), dx=Quantity(2, 's')) == Quantity(8, 's')
```


[NumPy overview](../numpy.md) · [Type hints](../typing.md)
