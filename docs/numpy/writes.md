# Array writes and output buffers

```python
import numpy as np
from unitx import Quantity, Unit
```

The output examples use this input:

```python
a = np.arange(6.0).reshape(2, 3)
q = Quantity(a, 'm')
```

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
and nonidentity scaling follows the algorithm in the [overview](../numpy.md#values-storage-and-precision). Run
`python benchmarks/numpy_writes.py` from the checkout to measure output,
assignment, and augmented-operation timings and traced peak allocations. Timings
are machine dependent; they are not a performance guarantee.

## Output buffers

Supported single-output ufunc calls, add.reduce/accumulate, multiply.reduce, and the numeric
functions listed with `out` in the [support matrix](../numpy.md#support-matrix)
accept a Quantity whose magnitude is a writable plain numeric ndarray.
`out=target`, `out=(target,)`, and positional
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
result and `where` mask must broadcast to the target shape for ufunc calls and clip.
Other function and reduction outputs must have exactly the ordinary result's shape,
including axes retained by `keepdims`; they do not broadcast into a larger
target. Only true mask
positions are written; false positions retain their original target values.
Unit conversion also skips false positions. Input/output overlap and shared
slices are handled through temporaries; a mask sharing output storage is copied.

The natural result dimension must match the target dimension. For example,
multiplying lengths can write to an area target, but cannot write back to a
length target. `Unit` factors participate in the conversion. Every numeric
result, including a coherent or scaled dimensionless result, requires a Quantity
target; writing it to a bare ndarray would discard its unit. Boolean predicates
and comparisons instead accept only a bare boolean ndarray output.
`argmin`, `argmax`, `nanargmin` and `nanargmax` accept only a bare integer ndarray output, with the
same exact shape requirement as other function outputs.

Final casting uses `casting='same_kind'` by default, or the explicitly requested
NumPy casting policy. It does not silently enable `unsafe`; nonintegral unit
conversion into an integer target normally fails. `same_kind` can narrow float
precision, whereas `safe` rejects such narrowing. Dimension, broadcast, read-only,
casting, and temporary computation/conversion failures leave the target unchanged.
The final copy is not a general transaction against concurrent changes or
external buffer failures. Temporaries cost memory proportional to the result
and target; `out` provides controlled writes, not a zero-allocation guarantee.

Functions and supported reductions/accumulations use `same_kind` for the final
cast unless they expose `casting`; reductions do not accept a new `casting`
keyword. Clip accepts ufunc call options, including `casting`, and permits
output broadcasting. Concatenate and stack use their existing
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
decompose **stored coordinates**, including for Unit and scaled
dimensionless inputs: convert q to the desired unit before decomposing it.
Changing the input unit can change both numerical outputs. There is no implicit
conversion to coherent dimensionless values. Thus `fractional + integral`
reconstructs q. For frexp, reconstruct with
`Quantity(np.ldexp(mantissa.value, exponent), mantissa.unit)`; separately
computing `np.exp2(exponent)` can overflow for large finite inputs. Dtype,
signed-zero, NaN/Inf, and real-input restrictions follow
[NumPy modf](https://numpy.org/doc/stable/reference/generated/numpy.modf.html) and
[frexp](https://numpy.org/doc/stable/reference/generated/numpy.frexp.html).
`np.divmod(a, b)` aligns compatible linear inputs and returns a dimensionless
quotient plus a remainder in the alignment unit. For example, 250 cm divided
by 1 m gives a quotient of 2 and a remainder of 50 cm. Both outputs require
Quantity buffers. Points and incompatible dimensions are rejected.
Only these functions' ordinary `__call__` operation is supported.

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

[NumPy overview and support matrix](../numpy.md) · [Documentation index](../README.md)
