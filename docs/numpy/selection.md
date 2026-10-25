# Selection and clipping

```python
import numpy as np
from unitx import Quantity, Unit
```

`np.where(condition, x, y)` aligns both branches to the first Quantity's unit,
or to x's coordinate when both branches are points. The condition must be a
bare boolean scalar or plain boolean ndarray. Both branches must be compatible,
even when the condition selects only one. Shape broadcasting and dtype promotion
follow NumPy. The one-argument form does not accept a Quantity condition; use
`np.where(q > threshold)` on the bare comparison mask to obtain indices.

`np.minimum(x, y)` and `np.maximum(x, y)` use the same alignment rules and
support the documented ufunc options, including masked writes to `out`.
Their reduce/accumulate methods are unsupported. `fmin/fmax` share the same
alignment and output rules, choosing the non-NaN operand when only one is NaN.
NaNs propagate according to NumPy's minimum/maximum rules. See the
[output buffer contract](writes.md#output-buffers) for staging and casting.

`np.clip(a, lower, upper)` requires a Quantity or point as a, and converts the
bounds into a's unit or coordinate. `None` disables an edge; both edges may be
`None`. Alternatively, use `np.clip(a, min=lower, max=upper)`; do not mix the
two naming conventions. The input is positional. Bounds broadcast against a;
reversed bounds follow NumPy and select the upper bound. Ufunc call options
(`where`, `dtype`, `casting`, `order`, `subok`) and a Quantity/point `out` use
the controlled [output rules](writes.md#output-buffers). Unit conversion can
evaluate all input elements even with a mask; `where` controls kernel evaluation
and output writes.

Bare numbers and arrays are coherent dimensionless values, including zero.
Dimensional branches and bounds must carry units. Scaled dimensionless
quantities align their scales; logarithmic differences retain their full
dimension. Affine points select only against affine points, and logarithmic
readings only against compatible readings with the same power/amplitude
convention. A point and a difference cannot be mixed. Selection preserves
readings and does not perform physical averaging.

```python
positions = Quantity(np.array([-1.0, 0.5, 3.0]), 'm')
bounded = np.clip(positions, Quantity(0, 'cm'), Quantity(200, 'cm'))
cleaned = np.where(positions > Quantity(0, 'm'), bounded, Quantity(0, 'm'))
np.testing.assert_allclose(cleaned.value, [0, 0.5, 2])
```

`np.min/max` (also `np.amin/amax`) preserve the input unit or coordinate.
`initial` must carry a compatible unit or point coordinate, even for
dimensionless input. A mask requires an explicit initial value, as in NumPy.
Empty reductions, axes, NaNs and `keepdims` follow NumPy. These reductions
write the exact result shape to `out`; elementwise selection and clipping
allow the ufunc output broadcasting rules.

`np.nanmin/max` share the same unit, point, mask, initial and output rules,
while omitting NaNs. An all-NaN slice returns NaN and emits RuntimeWarning;
an empty reduction still needs an initial value. A warning promoted to an
exception leaves `out` unchanged.

`np.argmin/argmax` and `np.nanargmin/nanargmax` return bare indices and accept
one integer axis or `axis=None`, `keepdims`, and a bare integer ndarray `out`
of exactly the result shape. They also accept affine and logarithmic points
and operate on stored readings. The NaN variants omit NaNs and raise
ValueError for all-NaN slices before writing `out`. NumPy's caveat remains:
`nanargmin` results cannot be trusted for slices containing only NaNs and
positive infinity; the corresponding `nanargmax` case is NaNs and negative
infinity. unitx delegates these numerical choices to NumPy.

See NumPy's [where](https://numpy.org/doc/stable/reference/generated/numpy.where.html),
[clip](https://numpy.org/doc/stable/reference/generated/numpy.clip.html) and
[minimum](https://numpy.org/doc/stable/reference/generated/numpy.minimum.html)
references for the underlying numerical behavior.
See also [nanmin](https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html),
[nanargmin](https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html)
and [nanargmax](https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html)
for NaN and infinity behavior.

[NumPy overview and support matrix](../numpy.md) · [Documentation index](../README.md)
