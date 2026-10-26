# Vector, tensor and discrete products

All inputs are ordinary linear Quantities or bare numeric scalars/ndarrays.
Bare operands are dimensionless; lists/tuples are metadata or operand sequences,
not implicit magnitude arrays. Points are unsupported.

| Operation | Result unit and options |
| --- | --- |
| `dot`, `inner`, `outer`, `vdot`, `tensordot`, `kron` | Product of both input units; native shape and contraction rules |
| `vecdot`, `linalg.vecdot` | Product of input units; conjugate the first input, with native axis/broadcast rules |
| `einsum` | Product of each operand's unit, counted once per occurrence |
| `einsum_path` | Bare path list and diagnostic string; no numerical quantity output |
| `cross` | Product of input units; axisa/axisb/axisc/axis follow NumPy |
| `trace` | Preserve input unit; offset, axis1/axis2, dtype and out |
| `convolve`, `correlate` | Product of input units; native mode and one-dimensional rules |
| `linalg.multi_dot` | Product of all matrix/vector units; native dimension and ordering rules |

Contractions sum numerical products; they do not multiply the unit by the
number of summed terms. Complete extra unit scales remain exact in the result
unit. Each input array still has one uniform unit; a matrix with differently
dimensioned columns cannot be represented by a single Quantity.

NumPy controls conjugation: `vdot`/`vecdot` conjugate the first operand,
`correlate` conjugates the second, and `dot`/`inner`/`einsum` do not insert
conjugation. `convolve` and `correlate` are discrete sums. Multiply by a
sampling interval explicitly when approximating a continuous integral.

`einsum` and `einsum_path` support both string and interleaved integer-subscript
forms, ellipses and native optimization/path options. Single-operand view
operations retain NumPy's memory sharing and writeability.

```python
import numpy as np
from unitx import Quantity
from unitx.si import si

force = Quantity(np.array([[1.0, 2.0], [3.0, 4.0]]), si.N)
displacement = Quantity(np.array([0.5, 1.0]), si.m)
work = np.tensordot(force, displacement, axes=([1], [0]))
assert work.unit == si.N * si.m
np.testing.assert_array_equal(work.value, [2.5, 5.5])
np.testing.assert_array_equal(np.einsum('ij,j->i', force, displacement).value, work.value)
```

`dot`, `outer`, `trace`, `einsum`, `multi_dot` and the `vecdot` ufunc support
[staged output writes](writes.md#output-buffers). Functions require the exact
result shape. Ufunc output rules apply to `vecdot`; its kernel accepts axis
and axes, but not where. `einsum` uses its casting option for the kernel and
final write, defaulting to `safe`; other function outputs use `same_kind`.
As elsewhere, target storage/dtype does not choose the computation kernel,
and unit conversion can require a floating result before final casting.
`cross` follows NumPy's version-dependent two-component-vector behavior:
deprecation warnings point to the caller, and removed support raises ValueError. Native invalid shapes, axes and unsupported options raise
before output mutation.

See the official [NumPy products reference](https://numpy.org/doc/stable/reference/routines.linalg.html)
and [einsum options](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).

[NumPy support matrix](../numpy.md) · [Documentation index](../README.md)
