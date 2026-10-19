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
| add.reduce, add.accumulate | Preserve unit; accept axis and dtype, plus keepdims for reduce |

Ufunc `__call__` accepts `where`, `dtype`, `casting`, `order`, `subok`, and empty
`out`. Each kernel validates its own options. `where` is a bare boolean scalar
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
| concatenate, stack | Convert to first Quantity's unit; bare arrays remain dimensionless | axis, dtype, casting |
| sum, mean | Preserve unit | axis, dtype, keepdims |
| median | Preserve unit | axis, keepdims; overwrite_input=False |
| std, var | Preserve unit / square unit | axis, dtype, ddof, keepdims |
| cumsum | Preserve unit | axis, dtype |
| argmin, argmax | Bare indices | axis, keepdims |
| isclose, allclose | Bare mask / Python bool | rtol, atol, equal_nan |

Supply the first magnitude argument positionally; concatenate/stack take a
sequence. Options follow each supported signature, including positional out
checks. Writable `out` and overwrite_input=True are rejected. Unsupported
initial, statistical where, mean, correction, and other keywords raise TypeError.
There is no `__setitem__`; `q += other` creates a result and rebinds q.

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
operations require supported arrays/scalars or Python int/float/complex.

```python
np.isclose(q, Quantity(a * 100, 'cm'), atol=Quantity(1e-8, 'm'))
q.isclose(Quantity(a * 100, 'cm'), abs_tol=Quantity(1e-8, 'm'))
```

`_numerics.py` defines `Scalar`, `Array`, and `Magnitude = Scalar | Array`
without importing NumPy. Array is a typing protocol, not a guarantee of other
backend support. Construction and unit-only operators preserve Quantity[T];
numerical array results conservatively use Array. NumPy's left-operand and
function stubs cannot generally infer wrapped unitx results; use
Quantity(array, unit) or unit * array for precise construction types. Arbitrary
shape/dtype inference is not promised. NumPy-free scalar consumers retain
working type hints.

References: [NumPy dispatch](https://numpy.org/doc/stable/reference/arrays.classes.html),
[promotion](https://numpy.org/doc/stable/reference/arrays.promotion.html),
[isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html).
