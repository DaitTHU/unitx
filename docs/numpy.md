# NumPy support

Install `unitx[numpy]` for NumPy >= 2.1. Core imports and Python scalar
arithmetic do not import NumPy.

```python
import numpy as np
from unitx import Quantity, Unit

metre = Unit('m')
a = np.arange(6.0).reshape(2, 3)
q = a * metre
assert q.value is a and (metre * a).value is a
speed = q / Quantity(2, 's')
centimetres = q.value_in('cm')
column_means = np.mean(q, axis=0)
mask = q > Quantity(100, 'cm')
```

One Quantity holds one unit for its entire magnitude. Supported magnitudes
are plain ndarrays and NumPy numeric scalars with integer, floating, or complex
dtype. Object, boolean, string, structured, datetime, masked arrays, ndarray
subclasses, and other backends are unsupported. Boolean arrays are accepted
as masks and returned by predicates. Construct ndarrays explicitly: list/tuple
magnitudes and implicit Fraction/Decimal object arrays are rejected.

## Values, storage, and precision

`.value`, `.value_in(unit)`, and `.to(unit).value` are explicit numerical exits.
`np.asarray(q)` and `np.array(q)` raise TypeError even for dimensionless q;
use `np.asarray(q.value_in(target_unit))`.

Construction, unit-only multiplication/division, and an identity conversion
preserve the magnitude object. Factor/SymbolicConstant operands change the
exact unit scale. Indexing, iteration, `.T`, reshape, and transpose follow
NumPy's scalar/view/copy rules; `.flatten(order='C')` copies. Zero-dimensional
arrays cannot be iterated. `.shape`, `.ndim`, `.dtype`, and `.size` are available
for NumPy magnitudes. Python scalars do not acquire array attributes. Exposed
values and views may share writable storage.

Unit algebra stays exact. Nonidentity conversion uses `np.multiply`: integral
rational ratios become Python int multipliers; nonintegral rational ratios
become one Python float multiplier; irrational ratios use Factor's approximate
value. NumPy controls dtype promotion, rounding, and overflow. Conversion
allocates and never writes to the input. Integer overflow, multiplier
conversion errors, and float overflow/underflow remain possible, including for
extended precision inputs. Conversion never creates Fraction/Decimal arrays.

Python Decimal and Fraction scalar operands (including Quantity magnitudes
such as `SI.h` and `SI.kB`) enter NumPy computations as Python floats. NumPy
then controls promotion, so a float32 array need not become float64. This
conversion rounds, may overflow or underflow, and does not preserve extended
precision from a Decimal/Fraction scalar. Constants and scalar-only arithmetic
retain their original types and Decimal context rules. Object arrays remain
unsupported. Formal unit powers retain exact Fraction/Decimal exponents even
though the numerical kernel uses floating exponents.

Comparisons broadcast and return bare boolean results. Equal unit factors
compare native values; otherwise both operands convert to coherent SI values.
Different dimensions yield False equality / True inequality masks; ordering
raises DimensionError. Bare numbers/arrays are dimensionless. These numerical
rules are distinct from Python scalar Quantity's exact comparison and hash
rules. All NumPy magnitudes, including scalars and zero-dimensional arrays,
produce unhashable Quantities. `bool(q)` delegates to `bool(q.value)`.

## Support matrix

Unit-bearing results are Quantities, including dimensionless values and scalar
reductions. Predicates and indices are bare results. Unsupported functions,
methods, mixed array types, and keywords fail explicitly.

| ufunc / Python operation | Result rule |
| --- | --- |
| add, subtract / `+`, `-` | Align compatible dimensions to the first Quantity's unit |
| multiply, divide/true_divide, matmul / `*`, `/`, `@` | Combine units |
| equal, not_equal, less, less_equal, greater, greater_equal / comparisons | Bare boolean results |
| positive, negative, absolute, fabs, conjugate / unary `+`, `-`, `abs` | Preserve unit |
| square, sqrt, cbrt, reciprocal | Exact unit exponents 2, 1/2, 1/3, -1 |
| power, float_power / `**` | Dimensional bases require a finite real scalar exponent; dimensionless bases/exponents normalize their scales |
| sin, cos, tan, exp, expm1, log, log2, log10, log1p | Require coherent dimensionless values; return dimensionless Quantity |
| isfinite, isnan, isinf, signbit | Bare boolean results |
| add.reduce, add.accumulate | Preserve unit; accept axis, dtype, out, plus keepdims for reduce |

Ufunc `__call__` accepts `where`, `dtype`, `casting`, `order`, `subok`, and
single-output `out` as described below. Each kernel validates its own options.
`where` is a bare boolean scalar
or plain boolean ndarray; masked-out positions without an output buffer follow
NumPy's uninitialized-result behavior. Numeric outputs must stay in the
supported dtype domain. Product reductions, at, outer, and reduceat are
unsupported. Python `**` also accepts Fraction/Decimal exponents, retaining
exact unit exponents while passing a float exponent to the NumPy kernel.
Angles retain unitx's dimensionless model: deg/rad normalize before these
functions, including exp/log. Python scalar numerical semantics remain native.

| NumPy function | Unit/result rule | Supported options |
| --- | --- | --- |
| reshape, transpose | Preserve unit | shape/order/copy; axes |
| concatenate, stack | Convert to first Quantity's unit; bare arrays remain dimensionless | axis, dtype, casting, out; dtype and out are mutually exclusive |
| sum, mean | Preserve unit | axis, dtype, keepdims, out |
| median | Preserve unit | axis, keepdims, out; overwrite_input=False |
| std, var | Preserve unit / square unit | axis, dtype, ddof, keepdims, out |
| cumsum | Preserve unit | axis, dtype, out |
| argmin, argmax | Bare indices | axis, keepdims, out |
| isclose, allclose | Bare mask / Python bool | rtol, atol, equal_nan |

Supply the first magnitude argument positionally; concatenate/stack take a
sequence. Options follow each supported signature, including positional out
checks. Function and reduction outputs follow the rules below;
overwrite_input=True is rejected. Unsupported initial, reduction/statistical
where, mean, correction, and other keywords raise TypeError.
There is no `__setitem__`; `q += other` creates a result and rebinds q.

## Output buffers

Supported single-output ufunc calls, add.reduce/accumulate, and the numeric
functions listed with `out` above accept a Quantity whose magnitude is a
writable plain numeric ndarray. `out=target`, `out=(target,)`, and positional
outputs for ufuncs return that same target; functions accept `out=target` or
the positional output argument in their NumPy signature, without a tuple.
The target's magnitude object, dtype, and unit stay
fixed. Zero-dimensional, empty, and noncontiguous output arrays are supported.
NumPy scalars and Python scalars cannot serve as output buffers.

```python
target = Quantity(np.full((2, 3), -1.0), 'cm')
assert np.add(q, q, out=target, where=a > 0) is target
```

The operation first computes its ordinary result in temporary storage, then
converts that result to the target unit and casts it into a staging buffer.
The numerical kernel uses the input dtypes and any explicit `dtype`; the
target dtype is used for the final cast. This is not a promise of NumPy's
output-dependent kernel selection or allocation-free execution. Both the
result and `where` mask must broadcast to the target shape for ufunc calls.
Function and reduction outputs must have exactly the ordinary result's shape,
including axes retained by `keepdims`; they do not broadcast into a larger
target. Only true mask
positions are written; false positions retain their original target values.
Unit conversion also skips false positions. Input/output overlap and shared
slices are handled through temporaries; a mask sharing output storage is copied.

The natural result dimension must match the target dimension. For example,
multiplying lengths can write to an area target, but cannot write back to a
length target. `ScaledUnit` factors participate in the conversion. Every numeric
result, including a coherent or scaled dimensionless result, requires a Quantity
target; writing it to a bare ndarray would discard its unit. Boolean predicates
and comparisons instead accept only a bare boolean ndarray output.
`argmin` and `argmax` accept only a bare integer ndarray output, with the
same exact shape requirement as other function outputs.

Final casting uses `casting='same_kind'` by default, or the explicitly requested
NumPy casting policy. It does not silently enable `unsafe`; nonintegral unit
conversion into an integer target normally fails. `same_kind` can narrow float
precision, whereas `safe` rejects such narrowing. Dimension, broadcast, read-only,
casting, and temporary computation/conversion failures leave the target unchanged.
The final copy is not a general transaction against concurrent changes or
external buffer failures. Temporaries cost memory proportional to the result
and target; `out` provides controlled writes, not a zero-allocation guarantee.

Functions and add.reduce/accumulate use `same_kind` for the final cast and do
not accept a new `casting` keyword. Concatenate and stack use their existing
`casting` option for the final write as well as the kernel; their `dtype` and
`out` options cannot be combined. Index outputs also use `same_kind`, including
its allowance for integer narrowing. These policies intentionally do not
reproduce every function's native output casting rule. An explicit `dtype`
controls the temporary numerical computation for supported reductions; the
target dtype does not select the accumulator or kernel dtype.

```python
totals = Quantity(np.empty(3), 'cm')
assert np.sum(q, axis=0, out=totals) is totals
indices = np.empty(3, dtype=np.intp)
assert np.argmax(q, axis=0, out=indices) is indices
```

`out=None` and ufunc `out=(None,)` allocate ordinary results.
Multiple-output ufuncs and product reductions remain unsupported.
Output writes preserve the magnitude's type and dtype, so the covariant
`Quantity[T]` contract remains valid; no setter or arbitrary replacement of T
is introduced. NumPy's own stubs do not accept Quantity outputs or infer wrapped
return types; these runtime calls may require a local type-checker suppression.

## Close comparisons and typing

`q.isclose(other, rel_tol=1e-9, abs_tol=0)` uses the symmetric formula in
coherent SI coordinates:
`abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)`.
`np.isclose` instead uses `abs(a-b) <= atol + rtol*abs(b)` with defaults
rtol=1e-5 and atol=1e-8; `np.allclose` reduces that mask. Tolerances broadcast
and must be finite, nonnegative, and real. Relative tolerance is dimensionless;
absolute tolerance matches the operands' dimension. Nonzero bare absolute
tolerance is dimensionless, so dimensional `np.isclose` calls need a Quantity
atol or zero. Matching infinities are close; NaNs require equal_nan=True.
The NumPy path enters at least float64 before subtraction, preserving
longdouble when present; large integers may lose precision. Mixed NumPy
operations accept supported arrays/scalars or Python int/float/complex,
Decimal, and Fraction scalars under the conversion rule above.

```python
np.isclose(q, Quantity(a * 100, 'cm'), atol=Quantity(1e-8, 'm'))
q.isclose(Quantity(a * 100, 'cm'), abs_tol=Quantity(1e-8, 'm'))
```

`_numerics.py` defines `Scalar`, `Array`, and `Magnitude = Array | Scalar`
without importing NumPy. Array is an input capability protocol, not a guarantee
of other backend support. Construction and unit-only operators preserve
Quantity[T], as does `.T`. Numerical array operations, conversions, indexing,
and iteration return `Quantity[NDArray[Any] | numpy.number[Any]]`; `.flatten()`
returns `Quantity[NDArray[Any]]`. `.value` therefore exposes native NumPy
methods such as reshape, astype, and sum. `.dtype` exposes a NumPy dtype.
Changed dtypes are not falsely annotated as the original dtype, and possible
zero-dimensional-array to NumPy-scalar results remain in the result union.
Boolean comparisons and isclose retain the NumPy-independent Array protocol.

`Scalar` is nonrecursive: its arithmetic methods return object at the protocol
boundary, while equality still returns bool. Concrete scalar and array result
types are described by Quantity's public overloads. Custom scalar magnitudes
must still return valid scalar results. This avoids recursively comparing
NumPy's many arithmetic overloads against Scalar's own return protocol.

`_numpy_typing.py` imports concrete NumPy types only for static analysis; core
runtime imports still do not load NumPy. Strict scalar-consumer checks also
run with NumPy absent. Checking the full library's optional NumPy annotations
requires NumPy; `pyright tests/typing` checks scalar consumers without it.
All checks share `pyrightconfig.json`; the test runner selects the input
directories and skips optional cases when their dependencies are unavailable.

NumPy's left-operand and function stubs do not generally describe every
unit-aware dispatch. With NumPy 2.5 stubs and Pyright 1.1.414, the documented
`np.asarray([3000, 4000, 5000]) * si.K` expression retains a Quantity type
without an explicit dtype. This path has strict regression coverage. NumPy
2.1 and 2.4 stubs can still infer Any here; runtime support remains NumPy >= 2.1.
`np.array([...]) * unit` can also become Any because its unknown-dtype array
matches NumPy's object-array multiplication fallback. `unit * array` or
`Quantity(array, unit)` provide alternatives across the supported versions.
Arbitrary shape/dtype inference is not promised, including the dtype NumPy
infers for unannotated input lists.

A direct `np.expm1(q)` works at runtime but NumPy's stubs may infer Any or ndarray
rather than Quantity for that intermediate result. The full example's final
Quantity types are checked independently; this does not imply that NumPy's
ufunc stubs preserve the wrapper at every intermediate call. For an explicit
numerical boundary, use `np.expm1(np.asarray(q.value_in(si.one)))` after
checking dimensions and normalizing the scale.

Use `.value` or `.value_in(unit)` at Matplotlib plotting boundaries too.
`.to()` returns a new Quantity; write `q = q.to(unit)` to retain the conversion.
It has no `inplace` option. NumPy-free scalar consumers retain working type hints.

References: [NumPy dispatch](https://numpy.org/doc/stable/reference/arrays.classes.html),
[promotion](https://numpy.org/doc/stable/reference/arrays.promotion.html),
[isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html).
