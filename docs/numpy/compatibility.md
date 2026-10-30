# Compatibility sweep and remaining boundaries

This guide collects additional supported paths and the remaining boundaries.
The [support matrix](../numpy.md#support-matrix) is the API contract;
being listed in NumPy's inventory does not itself imply unitx support.

## Additional supported paths

- `block` aligns all leaves to the first wrapped leaf's complete unit or point
  coordinate and follows NumPy's nested-list/shape rules.
- `choose(a, choices, out=None, mode='raise')` takes bare integer indices and
  compatible choices, including points; its output uses staged fixed-unit
  writes. `select(condlist, choicelist, default=0)` takes bare boolean conditions
  and compatible choices. Its default fills otherwise-unselected positions:
  bare defaults are stored readings, and wrapped defaults convert to the
  retained unit/coordinate, as for padding.
- `nan_to_num(x, copy=True, nan=0, posinf=None, neginf=None)` retains the wrapper.
  Bare replacements are readings; wrapped replacements align to the input.
  Native complex behavior replaces real and imaginary components separately.
  `copy=False` on ndarray magnitudes stages all computation before writing
  into the same Quantity/point and buffer. Read-only, conversion, casting and
  shape failures preserve the input. Scalars follow native allocation rules.
- `ediff1d` flattens stored values before differencing and returns the complete
  difference unit, including for points. Optional `to_begin`/`to_end` values
  must be compatible differences; bare values are dimensionless.
- `cumulative_sum` retains linear units. `cumulative_prod` normalizes
  dimensionless input and returns a coherent dimensionless result. Native
  axis/dtype/include_initial rules apply, including the required axis for
  multidimensional input. Both support staged `out` writes and reject points.
- `array_equal` aligns compatible units/coordinates, requires equal shapes,
  and forwards `equal_nan`. `array_equiv` allows native broadcasting. Different
  dimensions, point families/conventions, or point/difference operands return
  False. Conversion and equality use NumPy numerical semantics, including
  floating rounding, rather than scalar Quantity's exact comparison.
- `all`/`any` use stored readings, including coordinate zero for points. They
  return bare booleans and accept axis, keepdims, a bare boolean where mask
  and a staged boolean ndarray out. `isreal`/`iscomplex` and their `*obj` forms
  return native bare predicates; `isposinf`/`isneginf` also accept boolean out.
- `shares_memory`/`may_share_memory` inspect magnitudes without converting
  units; native max_work options apply. `resize` retains units/coordinates
  and NumPy's flat repetition behavior. `trim_zeros` uses stored zero readings
  and preserves native views. `unstack` returns a tuple of wrapped views.
  `astype` retains complete units/coordinates, supports native dtype/copy/device
  options and rejects unsupported magnitude dtypes. `sort_complex` retains
  linear units; points are unsupported because the result is complex.
- `i0` requires dimensionless linear input and normalizes its full scale.
  `ldexp` preserves the first input's complete linear unit with dimensionless
  integer exponents, reconstructing `frexp`. `spacing` and `nextafter` use
  stored floating representations and retain linear units; `nextafter` aligns
  its second operand. These representation-dependent operations need not
  commute with changing the stored unit. `heaviside` uses the sign of any
  linear input and a coherent dimensionless value at zero.
- `linalg.matmul`, `linalg.outer`, `linalg.cross` and `linalg.tensordot` return
  product units with their stricter native Array API shapes and keywords.
  `linalg.trace` retains linear units, while `linalg.diagonal` also retains
  point coordinates and native view flags. `matvec` and `vecmat` preserve
  product units, native conjugation, call options and staged out.

```python
import numpy as np
from unitx import Quantity, Unit

q = Quantity(np.array([1.0, np.nan, 3.0]), 'm')
clean = np.nan_to_num(q, nan=Quantity(200, 'cm'))
prefix = np.cumulative_sum(clean, include_initial=True)
np.testing.assert_array_equal(prefix.value, [0, 1, 3, 6])
assert np.array_equal(clean, clean.to('cm'))
blocks = np.block([[clean], [clean.to('cm')]])
assert blocks.unit == Unit('m') and blocks.shape == (2, 3)
mantissa, exponent = np.frexp(clean)
assert np.array_equal(np.ldexp(mantissa, exponent), clean)
assert np.shares_memory(clean, clean[1:])
```

## Inventory audit and version policy

The sweep examined the public inventories exposed by
[`get_overridable_numpy_array_functions`](https://numpy.org/doc/stable/reference/generated/numpy.testing.overrides.get_overridable_numpy_array_functions.html)
and [`get_overridable_numpy_ufuncs`](https://numpy.org/doc/stable/reference/generated/numpy.testing.overrides.get_overridable_numpy_ufuncs.html),
then compared the remaining entries with the single-unit numeric-array model.
Inventories include constructors with `like=`, record/string functions, internal
ufuncs and aliases; counting their names is not a measure of useful coverage.
The matrix documents supported behavior and options rather than claiming full
NumPy parity or Array API conformance.

NumPy 2.1 is the minimum. Array API cumulative functions, `unstack`, the strict
`linalg` functions and `astype` are available there. `matvec`/`vecmat` require
NumPy 2.2 and are looked up only when available. Version-dependent native
keywords, such as `unique(sorted=...)` and newer `trim_zeros` options, remain
the installed NumPy's responsibility: unsupported options raise rather than
being silently emulated. Device is restricted by NumPy's CPU backend. General
ufunc signature/axes remapping and NumPy 2.5's `out=...` extension are outside
the documented call options. NumPy stubs still do not infer arbitrary dispatched
Quantity results; see [typing](../typing.md#numpy-magnitude-types).

| Remaining category | Boundary / explicit numerical exit |
| --- | --- |
| `array`, `asarray`, `asanyarray`, contiguity/require coercions | Implicit stripping remains rejected. Use `.value`/`.value_in()` when an external API needs numbers. |
| Bare constructors, random sampling, index constructors, dtype promotion helpers | Construct numerical arrays first, then attach a unit; `like=q` is not an implicit unit constructor. |
| Dimensional `power` with varying or empty exponent arrays | Rejected. Nonempty constant exponent arrays preserve numerical broadcasting and one common result unit. No per-element-unit result is constructed. |
| `polyfit`, polynomial coefficients/roots, `vander`, dimensional `slogdet` | Coefficients or logarithms need heterogeneous units/reference scales; no misleading single-unit wrapper is returned. |
| `piecewise`, `apply_along_axis`, `apply_over_axes` | Callback input/output units and mutable views require a separate explicit contract. |
| Bitwise, gcd/lcm, string, datetime, record and masked-array APIs | Outside the supported numeric magnitude/unit model. |
| Automatic complex-domain `lib.scimath` functions | Use explicit complex numeric magnitudes and the supported ufuncs. |
| File and text I/O (`save*`, `loadtxt`, `savetxt`, etc.) | Explicitly choose numeric units and separate metadata; the scalar serialization format does not serialize ndarray storage. |
| Dimensional cumulative/segmented products | Explicitly rejected: partial results can have different units. Ordinary dimensionless inputs normalize to one coherent unit. |
| Packed raw QR | Reflectors and triangular entries have different units; extract numerical values explicitly for this low-level representation. |

[NumPy overview](../numpy.md) · [Array writes and methods](writes.md)
