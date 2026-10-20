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
| concatenate, stack | Convert to first Quantity's unit; bare arrays remain dimensionless | axis, dtype, casting, out; dtype and out are mutually exclusive |
| sum | Preserve unit | axis, dtype, keepdims, initial, where, out |
| mean | Preserve unit | axis, dtype, keepdims, where, out |
| prod | Product unit rules below | axis, dtype, keepdims, initial, where, out |
| median | Preserve unit | axis, keepdims, out; overwrite_input=False |
| std, var | Preserve unit / square unit | axis, dtype, ddof, keepdims, where, out |
| cumsum | Preserve unit | axis, dtype, out |
| argmin, argmax | Bare indices | axis, keepdims, out |
| trapezoid | Multiply ordinate and coordinate units | x, scalar dx, axis; no out |
| diff | Preserve input unit | n, axis, prepend, append; no out |
| gradient | Divide input unit by each coordinate unit | scalar spacings or coordinate arrays, axis, edge_order; no out |
| isclose, allclose | Bare mask / Python bool | rtol, atol, equal_nan |

Supply the first magnitude argument positionally; concatenate/stack take a
sequence. Options follow each supported signature, including positional out
checks. Function and reduction outputs follow the rules below;
overwrite_input=True is rejected. Reduction/statistical `where` is supported
for sum, mean, std, var, add.reduce, prod, and multiply.reduce; `initial` is
supported only for sum/add.reduce and prod/multiply.reduce. Precomputed `mean`,
`correction`, and other unsupported keywords raise TypeError.
Indexed assignment and augmented arithmetic on ndarrays follow the fixed-target
rules below. Python and NumPy scalar magnitudes retain ordinary result/rebinding
behavior for augmented arithmetic.

## Upgrading from 0.8

Version 0.9 adds indexed writes and changes ndarray augmented arithmetic from
rebinding to updating the existing buffer. Aliases now observe `q += other` and
the other supported augmented operators. The array, dtype, shape, and unit stay
fixed, so integer `q /= 2` and dimensional `q **= 2` can raise where the old
rebinding behavior succeeded. Python and NumPy scalar arithmetic is unchanged.

To retain the previous allocation and rebinding behavior, write `q = q + other`,
`q = q / 2`, or `q = q ** 2` explicitly. Use ordinary multiplication and rebinding
when changing units or dimensions. Indexed assignment uses `same_kind` casting;
it does not inherit ndarray assignment's implicit float-to-integer truncation.
The supported array domain and explicit unit-removal rules are unchanged.

## Indexed assignment and augmented arithmetic

`q[index] = value` writes into a writable plain numeric ndarray magnitude.
The magnitude object, dtype, shape, and unit stay fixed. Values must be supported
numeric scalars/ndarrays or Quantities with compatible dimensions; Quantity
values convert to the destination unit. Bare values are coherent dimensionless
values, including zero: use `Quantity(0, q.unit)` to clear a dimensional array.
Decimal/Fraction scalars enter the same floating domain as other NumPy operations.
Lists/tuples are allowed as indices, not as assigned magnitudes.

```python
positions = Quantity(np.array([1.0, 2, 3]), 'm')
positions[0] = Quantity(50, 'cm')
positions[positions > Quantity(1, 'm')] = Quantity(0, 'm')
np.testing.assert_array_equal(positions.value, [0.5, 0, 0])
```

Basic slices, integer/boolean indexing, broadcasting, new axes, scalar elements,
and repeated advanced indices follow [NumPy indexed assignment](https://numpy.org/doc/stable/user/basics.indexing.html#assigning-values-to-indexed-arrays).
Array indices must be plain integer/boolean ndarrays; masked arrays and ndarray
subclasses are rejected. Boolean masks are bare arrays, as returned by comparisons.
Repeated indices use NumPy assignment semantics; no accumulation at repeated
positions or particular winner among conflicting repeated writes is promised.
NumPy scalar magnitudes are immutable; explicit zero-dimensional ndarrays are
writable. Read-only arrays reject assignment, including empty selections.

Assignment stages unit conversion and `same_kind` casting **before** the native
indexed write. Shape and index errors then follow NumPy's unchanged-target rule;
casting failures occur only in temporary storage. Assignment supports RHS views
overlapping the target; index arrays are snapshotted before the write. Float
narrowing and integer narrowing remain allowed by `same_kind`, while float to
integer and complex to real writes fail. This deliberately differs from NumPy's
ordinary assignment, which can silently truncate floats into integers.

`+=`, `-=`, `*=`, `/=`, `**=`, and `@=` on ndarray Quantities return the same
Quantity and update its original array. Each computes its ordinary result first,
then uses the controlled output conversion, broadcasting, and `same_kind` casting
rules. The natural result dimension must match the destination: multiplying a
length by a time or squaring a length fails without modifying the destination.
Dimensionless factors, Units, and Quantities may scale the numerical values while
the destination unit stays fixed; exact Factor/SymbolicConstant operands first
participate in ordinary exact unit algebra, then convert back to that fixed unit.
Unit changes remain available through ordinary arithmetic and rebinding.

```python
positions = Quantity(np.array([1.0, 2, 3]), 'm')
alias = positions
buffer = positions.value
positions += Quantity(50, 'cm')
positions *= Quantity(2, 'km/m')
assert positions is alias and positions.value is buffer
np.testing.assert_allclose(buffer, [3000, 5000, 7000])
positions[1:] += Quantity(100, 'cm')
np.testing.assert_allclose(buffer, [3000, 5001, 7001])
```

All aliases observe successful writes. Basic indexed augmented assignment acts
on its view; advanced indexed augmented assignment computes on a copy and writes
back, so repeated indices do not accumulate repeatedly. Python/NumPy scalars
continue to create new Quantities and may change their magnitude type or unit.
An ndarray target that cannot represent a result raises instead of silently
rebinding to a wider dtype or a different unit. Use `q = q / 2` when an integer
array needs a floating result. `__setitem__` and augmented arithmetic preserve
the covariant magnitude contract because no operation replaces T, its dtype,
or its storage. Concrete ndarray annotations retain their dtype after successful
augmented arithmetic; a generic Array annotation may also describe an immutable
NumPy scalar and therefore retains the broader result type.

These operations use temporaries; they do not promise zero allocation or general
rollback against concurrent mutation or external buffer failures. Output staging
reuses an already independent result when its dtype matches the target; otherwise
it stages a cast or an overlapping value. Identity scaling still retains storage,
and nonidentity scaling follows the algorithm documented above. Run
`python benchmarks/numpy_writes.py` from the checkout to measure output,
assignment, and augmented-operation timings and traced peak allocations. Timings
are machine dependent; they are not a performance guarantee.

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
including exact ScaledUnit factors. No spacing means dimensionless unit steps.
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

## Masked sums and statistics

`np.sum(q)` and `np.add.reduce(q)` preserve q's unit, including ScaledUnit and
scaled dimensionless units. Their default axes are all axes and axis 0,
respectively. Both accept `initial`, a numeric scalar or scalar Quantity,
including NumPy scalars and zero-dimensional arrays. A Quantity initial must
have the input's dimension and is converted to the input unit, then added once
to every result slice. Bare nonzero initial values are coherent dimensionless
values; they do not implicitly acquire the input unit. Bare zero is accepted
as the additive identity in any dimension, while a Quantity zero retains its
declared dimension. Empty or fully masked sums return zero in the input unit,
or the converted initial when supplied. Explicit `initial=None` uses NumPy's
first-element semantics and fails on empty reductions.

The input and explicit `dtype` select the accumulator; initial does not promote
it. NumPy casts the converted initial into that dtype before summation, so an
integer accumulator can truncate a fractional initial. Decimal/Fraction initial
values enter the NumPy floating domain. Complex initial values follow NumPy's
casting rules, even for complex zero. Unsupported initial arrays, bools, or
incompatible dimensions fail before writing an output.

`where` on sum, add.reduce, mean, std, and var is a bare boolean scalar or plain
boolean ndarray broadcast to the input shape. It selects **input elements**;
all result positions are written when `out` is supplied. Different slices may
have different selected counts. Mean and standard deviation retain the input
unit; variance has its square. Statistics use stored coordinates, including
for scaled dimensionless input. NumPy controls dtype, complex statistics, NaN/Inf,
empty-sample warnings, and degrees of freedom (N in `N - ddof` is the selected count).
Masks do not imply automatic NaN omission. These operations use the same staged
output rules as other reductions, including exact shape and unchanged targets
on computation, unit, shape, or casting errors. `add.accumulate`, cumsum, and
median do not gain `where` or `initial`.

```python
samples = Quantity(np.array([[1.0, 2, 3], [4, 5, 6]]), 'm')
selected = np.array([[True, False, True], [False, True, True]])
totals = np.sum(samples, axis=1, where=selected, initial=Quantity(50, 'cm'))
assert totals.unit == samples.unit
np.testing.assert_allclose(totals.value, [4.5, 11.5])
means = np.mean(samples, axis=1, where=selected)
np.testing.assert_allclose(means.value, [2, 5.5])
```

The numerical rules follow [NumPy sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html),
[mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html), and
[var](https://numpy.org/doc/stable/reference/generated/numpy.var.html), with the
unit and output constraints above.

## Product reductions

`np.prod(q)` reduces all axes by default; `np.multiply.reduce(q)` defaults to
axis 0, following NumPy. Both accept scalar, negative, or tuple axes, `axis=None`,
`axis=()`, and `keepdims`. For dimensional input with unit U, each result has
unit `U**N`, where N is the number of selected input elements in that slice.
Zero-valued elements still count. All output slices must have the same N;
otherwise unitx raises DimensionError because a single Quantity cannot hold
different dimensions. `where` accepts the same bare boolean masks as ufunc
calls and broadcasts to the input shape. It selects input elements; every
output position is written when an output buffer is supplied.

An empty or fully masked product has value 1 and unit `U**0`, before applying
`initial`. If the output itself has no elements, the unit uses the nominal
reduced-axis extent; scalar `where=False` sets that extent to zero.
Dimensionless inputs, including deg and km/m, first convert to coherent
dimensionless values and produce unit `Unit()`. Their masked slices may have
different counts because the result still has a single coherent unit.

`initial` is a bare numeric scalar or a scalar Quantity, including NumPy
scalars and zero-dimensional arrays. A bare initial is dimensionless; a
Quantity initial contributes its own unit once, so the result unit is
`U**N * initial.unit`. An explicit zero still contributes its declared unit.
The default is dimensionless 1. `initial=None` uses NumPy's first-element
semantics and fails for empty reductions. NumPy controls the accumulator dtype,
including casting the initial value to that dtype; initial does not select
a wider numerical domain. Decimal/Fraction initial values enter NumPy as floats.

```python
lengths = Quantity(np.array([[2.0, 3, 4], [5, 6, 7]]), 'm')
selected = np.array([True, False, True])
areas = np.prod(lengths, axis=1, where=selected)
assert areas.unit == Unit('m2')
np.testing.assert_array_equal(areas.value, [8, 35])
weighted = np.multiply.reduce(lengths, axis=1, where=selected, initial=Quantity(2, 's'))
assert weighted.unit == Unit('m2 s')
```

Both entry points support the output rules below, including exact result
shape and unchanged targets on computation, unit, shape, or casting errors.
`cumprod` and `multiply.accumulate` remain unsupported: partial products
generally require different units at different positions. The supported
axis, empty-product, initial, and mask behavior follows
[NumPy prod](https://numpy.org/doc/stable/reference/generated/numpy.prod.html)
and [ufunc.reduce](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduce.html),
with the additional single-unit and explicit-output rules described here.

## Output buffers

Supported single-output ufunc calls, add.reduce/accumulate, multiply.reduce, and the numeric
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

Functions and supported reductions/accumulations use `same_kind` for the final cast and do
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

`out=None` and ufunc `out=(None,)` allocate ordinary single-output results.
The supported multiple-output calls are described below; other multiple-output
ufuncs and product accumulations remain unsupported.
Output writes preserve the magnitude's type and dtype, so the covariant
`Quantity[T]` contract remains valid; no arbitrary replacement of T is introduced.
NumPy's own stubs do not accept Quantity outputs or infer wrapped
return types; these runtime calls may require a local type-checker suppression.

## Multiple-output decompositions

`np.modf(q)` returns `(fractional, integral)`, both Quantities with q's unit.
`np.frexp(q)` returns `(mantissa, exponent)`, with q's unit on the mantissa and
a bare integer ndarray or NumPy scalar for the binary exponent. These functions
decompose **stored coordinates**, including for ScaledUnit and scaled
dimensionless inputs: convert q to the desired unit before decomposing it.
Changing the input unit can change both numerical outputs. There is no implicit
conversion to coherent dimensionless values. Thus `fractional + integral`
reconstructs q. For frexp, reconstruct with
`Quantity(np.ldexp(mantissa.value, exponent), mantissa.unit)`; separately
computing `np.exp2(exponent)` can overflow for large finite inputs. Dtype,
signed-zero, NaN/Inf, and real-input restrictions follow
[NumPy modf](https://numpy.org/doc/stable/reference/generated/numpy.modf.html) and
[frexp](https://numpy.org/doc/stable/reference/generated/numpy.frexp.html).
Only their ordinary `__call__` operation is supported.

Both accept `out=(first, second)` or positional output arguments. Each `None`
allocates that output; `(None, None)` allocates both. Every unit-bearing output
requires a Quantity with a writable plain numeric ndarray; frexp's exponent
requires a bare integer ndarray. The existing conversion, casting, and `where`
rules apply independently to each output. Supplied buffers must all have the
common broadcast shape of the inputs, mask, and outputs. An allocated output
also has that shape. False mask positions retain supplied values; without a
buffer those positions are unspecified.

```python
parts = Quantity(np.array([1.25, -2.5]), 'm')
fractional = Quantity(np.empty(2), 'cm')
integral = Quantity(np.empty(2), 'm')
f, i = np.modf(parts, out=(fractional, integral))
assert f is fractional and i is integral
np.testing.assert_array_equal(f.value, [25, -50])
np.testing.assert_array_equal(i.value, [1, -2])
```

All supplied outputs are converted and staged before any is written. A shape,
unit, casting, read-only, computation, or conversion failure leaves **all**
targets unchanged. Input/output overlap and masks sharing output storage are
supported. The two output buffers may share storage only when their elements
do not overlap; overlapping outputs raise ValueError before writing. This
avoids one result overwriting another. Final writes have the same concurrent
mutation and external-buffer limitations as single outputs, and require
temporary storage. The covariance and NumPy-stub limitations above still apply.

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
