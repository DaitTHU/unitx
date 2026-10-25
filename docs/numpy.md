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

AffineQuantity points and LogQuantity readings share point storage rules; see
[affine operations](affine.md#comparison-tolerances-and-arrays) and
[logarithmic statistics](logarithmic.md#comparison-and-arrays).
Mean/median preserve points, std returns differences, var returns squared
differences, and point sums/products are rejected.

## Values, storage, and precision

`.value`, `.value_in(unit)`, and `.to(unit).value` are explicit numerical exits.
`np.asarray(q)` and `np.array(q)` raise TypeError even for dimensionless q;
use `np.asarray(q.value_in(target_unit))`.

Construction, unit-only multiplication/division, and an identity conversion
preserve the magnitude object. Factor/SymbolicConstant operands change the
exact unit scale. Indexing, iteration, `.T`, and the supported shape transformations
follow NumPy's scalar/view/copy rules; `.flatten(order='C')` copies. Zero-dimensional
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

Use this table as the supported API boundary. Jump to [writes](#indexed-assignment-and-augmented-arithmetic),
[differences](#differences-and-gradients), [masked statistics](#masked-sums-and-statistics),
[products](#product-reductions), [out](#output-buffers), or
[multiple outputs](#multiple-output-decompositions) for detailed keyword and storage rules.
See [weighted averages and quantiles](numpy/statistics.md#weighted-averages-and-quantiles)
for dimensionless weights, probabilities and method choices.

Unit-bearing results are Quantities, including dimensionless values and scalar
reductions. Predicates and indices are bare results. Unsupported functions,
methods, mixed array types, and keywords fail explicitly.

| ufunc / Python operation | Result rule |
| --- | --- |
| add, subtract / `+`, `-` | Align compatible dimensions to the first Quantity's unit |
| minimum, maximum | Elementwise selection; align to the first Quantity's unit or point coordinate |
| multiply, divide/true_divide, matmul / `*`, `/`, `@` | Combine units |
| equal, not_equal, less, less_equal, greater, greater_equal / comparisons | Bare boolean results |
| positive, negative, absolute, fabs, conjugate / unary `+`, `-`, `abs` | Preserve unit |
| square, sqrt, cbrt, reciprocal | Exact unit exponents 2, 1/2, 1/3, -1 |
| power, float_power / `**` | Dimensional bases require a finite real scalar exponent; dimensionless bases/exponents normalize their scales |
| sin, cos, tan, exp, expm1, log, log2, log10, log1p | Require coherent dimensionless values; return dimensionless Quantity |
| arcsin, arccos, arctan | Require coherent dimensionless values; return Quantity in rad |
| hypot | Compatible input dimensions; align to the first Quantity's unit |
| arctan2 | Compatible input dimensions; return Quantity in rad |
| isfinite, isnan, isinf, signbit | Bare boolean results |
| modf | Fractional and integral parts in the input unit |
| frexp | Mantissa in the input unit; bare integer binary exponent |
| add.reduce, add.accumulate | Preserve unit; accept axis, dtype, out, plus keepdims, initial, where for reduce |
| multiply.reduce | Product unit rules below; accepts axis, dtype, keepdims, initial, where, out |

Ufunc `__call__` accepts `where`, `dtype`, `casting`, `order`, `subok`, and
`out` as described below (two outputs for modf/frexp). Each kernel validates its own options.
`where` is a bare boolean scalar
or plain boolean ndarray; masked-out positions without an output buffer follow
NumPy's uninitialized-result behavior. Numeric outputs must stay in the
supported dtype domain. Product accumulation, at, outer, and reduceat are
unsupported. Python `**` also accepts Fraction/Decimal exponents, retaining
exact unit exponents while passing a float exponent to the NumPy kernel.
Angles retain unitx's dimensionless model: deg/rad normalize before these
functions, including exp/log. Python scalar numerical semantics remain native.

| NumPy function | Unit/result rule | Supported options |
| --- | --- | --- |
| reshape, transpose | Preserve unit | shape/order/copy; axes |
| squeeze, expand_dims | Preserve unit or point coordinate | axis |
| moveaxis, swapaxes, rollaxis | Preserve unit or point coordinate | source/destination; axis1/axis2; axis/start |
| ravel | Preserve unit or point coordinate | order: C, F, A, K |
| matrix_transpose, linalg.matrix_transpose | Swap the last two axes; preserve unit or point coordinate | At least two dimensions; transpose's permute_dims alias is also supported |
| atleast_1d, atleast_2d, atleast_3d | Preserve each input's unit or point coordinate independently | One or more positional numeric scalars/ndarrays, quantities or points |
| broadcast_to, broadcast_arrays | Preserve each input's unit or point coordinate independently | shape, subok; broadcasting does not align units |
| flip, fliplr, flipud, rot90, roll | Preserve unit or point coordinate | axis; k/axes for rot90; shift/axis for roll |
| repeat, tile, copy | Preserve unit or point coordinate | repeats/axis; reps; order/subok for copy |
| shape, ndim, size | Bare shape tuple or integer | size accepts axis |
| concatenate, stack | Convert to first Quantity's unit; bare arrays remain dimensionless | axis, dtype, casting, out; dtype and out are mutually exclusive |
| sum, nansum | Preserve unit | axis, dtype, keepdims, initial, where, out |
| mean, nanmean | Preserve unit or point coordinate | axis, dtype, keepdims, where, out |
| min/amin/nanmin, max/amax/nanmax | Preserve unit or point coordinate | axis, keepdims, initial, where, out |
| where | Select between compatible quantities or points | condition, x, y, all positional; no out |
| clip | Preserve input unit or point coordinate | a_min/a_max or min/max, out; ufunc call options |
| prod, nanprod | Product unit rules below; nanprod counts selected non-NaN elements | axis, dtype, keepdims, initial, where, out |
| median, nanmedian | Preserve unit or point coordinate | axis, keepdims, out; overwrite_input=False |
| std/nanstd, var/nanvar | Difference unit / square difference unit | axis, dtype, ddof, keepdims, where, out |
| average | Preserve sample unit; returned weight sum is a dimensionless Quantity | axis, weights, returned, keepdims; linear Quantity only |
| quantile, percentile, nanquantile, nanpercentile | Preserve sample unit; dimensionless q and weights | axis, out, method, keepdims, weights; overwrite_input=False; linear Quantity only |
| cumsum | Preserve unit | axis, dtype, out |
| argmin/nanargmin, argmax/nanargmax | Bare indices | axis, keepdims, out |
| interp | Align x to xp's unit; retain fp's unit | finite strictly increasing 1-D xp; scalar left/right, period=None; no out |
| trapezoid | Multiply ordinate and coordinate units | x, scalar dx, axis; no out |
| diff | Preserve input unit | n, axis, prepend, append; no out |
| gradient | Divide input unit by each coordinate unit | scalar spacings or coordinate arrays, axis, edge_order; no out |
| isclose, allclose | Bare mask / Python bool | rtol, atol, equal_nan |

Supply the first magnitude argument positionally; concatenate/stack take a
sequence. Options follow each supported signature, including positional out
checks. Function and reduction outputs follow the rules below;
overwrite_input=True is rejected. Reduction/statistical `where` is supported
for sum/nansum, mean/nanmean, std/nanstd, var/nanvar, add.reduce, prod/nanprod,
multiply.reduce, and min/max/nanmin/nanmax; `initial` is supported for
sum/nansum/add.reduce, prod/nanprod/multiply.reduce, and min/max/nanmin/nanmax. Precomputed `mean`,
`correction`, and other unsupported keywords raise TypeError.
Indexed assignment and augmented arithmetic on ndarrays follow the fixed-target
rules below. Python and NumPy scalar magnitudes retain ordinary result/rebinding
behavior for augmented arithmetic.

## Shape transformations

`squeeze`, `expand_dims`, `moveaxis`, `swapaxes`, `rollaxis`, and `ravel`
operate on magnitudes without conversion. Like reshape and transpose, they
retain the complete unit or point coordinate and return the same family:
Quantity, AffineQuantity, or LogQuantity. Only the NumPy function forms are
supported; these functions have no `out` option.

Axis rules and errors follow NumPy, including negative and tuple axes where
supported. Squeezing an ndarray whose axes all have length one produces a
zero-dimensional ndarray magnitude. Views share storage and its writeability
with the input.
`ravel` returns a contiguous one-dimensional magnitude, copying only when needed;
it can copy a noncontiguous input even when reshape could return a view.
Its C/F/A/K orders follow NumPy. Prefer `moveaxis` for new code; `rollaxis` is
also supported for compatibility with existing NumPy code.

`matrix_transpose` and `linalg.matrix_transpose` swap the last two axes of an
array with at least two dimensions. `permute_dims` is NumPy's alias of
`transpose` and supports the same axes argument.

`atleast_1d/2d/3d` and `broadcast_arrays` accept multiple positional inputs.
Each Quantity or point retains its own complete unit or coordinate; bare
numeric scalars/ndarrays produce bare ndarrays. Inputs may have different
dimensions or coordinate families because these functions only rearrange
storage. Bare list/tuple inputs and unsupported array types remain unsupported.
One input to `atleast_*` returns one result; multiple inputs return NumPy's
sequence container. `broadcast_arrays` always returns a sequence, even for one
input. Shape incompatibilities follow NumPy.

`broadcast_to` returns a read-only view. `broadcast_arrays` retains NumPy's
view flags and warnings; broadcast elements may alias the same storage, so
use `copy` before writing. `flip`, `fliplr`, `flipud`, and `rot90` retain
NumPy's views, including negative strides. `roll`, `repeat`, and `tile` follow
NumPy's allocation rules. `copy(order='K', subok=False)` creates independent,
writable storage, including when the source is read-only. `subok` on copy and
broadcasting is forwarded to NumPy for the plain supported magnitudes; it does
not enable ndarray subclasses.

`shape`, `ndim`, and `size` inspect the magnitude and return bare metadata,
including for scalar quantities. All these function forms preserve the wrapper
family when returning unit-bearing values and have no `out` option.

```python
import numpy as np
from unitx import Quantity

samples = Quantity(np.arange(6).reshape(1, 2, 3), 'cm')
matrix = np.squeeze(samples, axis=0)
assert matrix.shape == (2, 3)
assert np.moveaxis(matrix, 0, -1).shape == (3, 2)
assert matrix.unit == samples.unit
assert np.shares_memory(matrix.value, samples.value)

times = Quantity(np.array([0.0, 1.0, 2.0]), 's')
distance_grid, time_grid = np.broadcast_arrays(matrix, times)
assert distance_grid.unit == matrix.unit and time_grid.unit == times.unit
assert time_grid.shape == (2, 3)
readonly = np.broadcast_to(times, (2, 3))
assert not readonly.value.flags.writeable
independent = np.copy(readonly)
assert independent.value.flags.writeable
assert not np.shares_memory(independent.value, times.value)
assert np.shape(np.flip(independent, axis=-1)) == (2, 3)
assert np.ndim(independent) == 2 and np.size(independent, axis=0) == 2
```

## Selection and clipping

See [selection and clipping](numpy/selection.md) for `where`, `clip`,
`minimum/maximum`, and `min/max`, including coordinate and bound rules.
The NaN extrema and index functions share those coordinate rules and omit NaNs.

## Indexed assignment and augmented arithmetic

See [array writes](numpy/writes.md#indexed-assignment-and-augmented-arithmetic)
for fixed storage, indexed assignment, aliases and augmented operations.

## Differences and gradients

See [integration, differences and gradients](numpy/calculus.md) for
`interp`, `trapezoid`, `diff`, `gradient`, coordinate spacing and result units.

## Masked sums and statistics

See [reductions and statistics](numpy/statistics.md#masked-sums-and-statistics)
for sum, mean, standard deviation, variance, initial values and input masks.
The [NaN statistics](numpy/statistics.md#nan-statistics) omit NaN samples using
the same unit and output rules as their ordinary counterparts.

## Product reductions

See [product reductions](numpy/statistics.md#product-reductions) for uniform
selection counts, dimensionless normalization and the unit of each product.

## Output buffers

See [output buffers](numpy/writes.md#output-buffers) for shape, conversion,
casting, overlap and unchanged-target guarantees.

## Multiple-output decompositions

See [multiple outputs](numpy/writes.md#multiple-output-decompositions) for
`modf`, `frexp`, independent output units and staging all writes together.

## Close comparisons

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

See the [typing guide](typing.md) for magnitude annotations, optional-dependency
checks, and NumPy stub limitations. Use `.value` or `.value_in(unit)` at plotting
boundaries. `.to()` returns a new Quantity and has no inplace option.

References: [NumPy dispatch](https://numpy.org/doc/stable/reference/arrays.classes.html),
[promotion](https://numpy.org/doc/stable/reference/arrays.promotion.html),
[isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html).

[Documentation index](README.md) · [Typing](typing.md)
