# Type hints

unitx ships inline annotations and `py.typed`. The parameter in `Quantity[T]`
or `AffineQuantity[T]` / `LogQuantity[T]` describes the magnitude type, not a physical dimension.
Units and compatible dimensions are validated at runtime.

## Scalar construction and arithmetic

```pycon
>>> from decimal import Decimal
>>> from unitx import Quantity
>>> from unitx.si import si
>>> q = Decimal('1.25') * si.m
>>> type(q.value) is Decimal
True
>>> (q / si.s).value is q.value
True

```

Constructing a Quantity or changing its unit product without numerical
arithmetic preserves T. Conversion and arithmetic can change the result type;
annotations use the corresponding result domain. `unitx.math` returns
`Quantity[float]` for supported Python real scalars and `Quantity[Array]`
for its array overloads. `unitx.time` clocks return `AffineQuantity[int]`;
duration adapters use ordinary Quantity objects. `time.from_datetime()` returns
`AffineQuantity[int]`; `time.to_datetime()` returns an aware datetime.

`unitx.serialization.from_dict()` returns a union of the supported unit and
scalar quantity classes. Use `isinstance` to narrow it. The magnitude type is
`int | float | Fraction | Decimal`; the input record determines its runtime type.
See [scalar data exchange](serialization.md).

A signature such as `Quantity[float]` does not require metres rather than
seconds. An `Annotated` label alone does not teach a type checker dimensional
algebra. unitx does not currently supply dimension-parameterized quantities
or a call-validation decorator.

## Float calculations

Float addition/subtraction, ordinary int/float multiplication and division
retain a float-compatible result domain, and integer powers of float-compatible
quantities do too. Explicit int/int addition/subtraction keeps its wider exact
result type. For example,
`distance / duration` can return `Quantity[float]` directly when both inputs
have that annotation. Fractional powers can produce complex
magnitudes and keep a wider result type; `unitx.math.sqrt` is the real-valued
float boundary for scalar square roots.

Integer magnitudes raised to literal exponents 0 through 25 retain
`Quantity[int]`; literal exponents -20 through -1 return `Quantity[float]`.
These finite Literal overloads follow the approach used by
[typeshed's integer annotations](https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi).
An unrestricted integer exponent keeps a wider real result type because its
sign is unknown. This also lets computed Decimal constants such as `SI.re`
retain their type through formulas containing positive integer powers, without
casts or changes to their numerical calculation.

Use `q.astype(float)` to convert the magnitude explicitly while retaining the
complete unit. It returns `Quantity[float]` for Python scalar input. This is a
real numeric conversion, unlike `typing.cast`, which does not change a value.
Decimal constants can participate directly in float formulas; the implicit
precision downgrade is described in [scalar behavior](scalars.md). Use an
explicit float conversion at a result boundary when its annotation must be
narrowed:

```pycon
>>> from unitx.si import SI, si
>>> length = 3.0 * si.m
>>> energy = (2.5 * si.V * SI.e).to(si.J).astype(float)
>>> duration = (length / SI.c).to(si.s).astype(float)
>>> type(duration.value) is float
True

```

`SI` keeps its existing integer and Decimal constants. Conversion to float
rounds, can overflow, and does not preserve Decimal precision. Scalar
`astype` accepts `int`, `float`, `complex`, `Fraction` and `Decimal`;
their native constructors control errors. NumPy magnitudes use their own
`astype` operation and numeric dtypes, retaining ndarray/NumPy-scalar results.
Only the dtype argument is supported; this method does not change units.

Unit conversion retains wider annotations. Python's float annotation also
accepts integer values, so even an object
annotated `Quantity[float]` may store an int: `Quantity[float](1, 'cm').to('m')`
really produces a Fraction. The float addition/subtraction overloads prioritize
ordinary scientific float inputs: if an object annotated `Quantity[float]`
actually stores an int, aligned addition can likewise produce a Fraction,
despite the inferred float result. Concrete int/Fraction overloads preserve
their wider result domains; runtime arithmetic remains exact for these inputs.
Use `q.to(target).astype(float)` when an output must have a float magnitude.
Mixed float/Decimal overloads likewise infer float for scientific formulas.
If a float annotation hides an int, its interaction with Decimal can instead
retain Decimal at runtime. Explicit int/Decimal inputs infer Decimal.

## Quantity subclasses

Preserving the magnitude parameter T does not preserve a Quantity subclass.
Quantity's ordinary binary arithmetic, unary operators and `.to()` construct
ordinary Quantity objects. A subclass must explicitly implement the operations
that preserve its own invariants. Scalar augmented operators delegate to the
corresponding binary operation; their annotations must describe the subclass's
results too.
For example, the [Impedance extension](https://github.com/DaitTHU/unitx/blob/main/examples/impedance.md)
preserves its type for addition of two Impedances, parallel composition and
conversion. Adding an ordinary Quantity returns an ordinary Quantity in either
operand order. Its local overrides specialize the base class's wider input
domain; they are not a template for preserving every array or coordinate overload.

## NumPy magnitude types

The `unitx._numerics` package defines `Scalar`, `Array`, and `Magnitude = Array | Scalar`
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

`_numerics/numpy_typing.py` imports concrete NumPy types only for static analysis; core
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

The same NumPy stub limitation applies to the supported vector/tensor products,
`einsum`, and shape transformations,
broadcasting, rearrangement, repetition and copying,
`where`, `clip`,
`minimum/maximum`, `average`, `quantile`, `percentile`, the supported `nan*`
statistics, and `interp`: runtime results
preserve the documented Quantity/point wrappers, but NumPy may annotate them
as ndarrays or Any. NumPy's `out` annotations also accept only ndarrays.
The `moveaxis` and `rollaxis` input annotations likewise require ndarrays
and can reject a Quantity/point argument during static checking, despite
working at runtime. Their installed type consumers check the dispatch protocol
directly; runtime tests use ordinary NumPy calls.
Runtime support and installed-consumer checks cover
these calls; unitx does not replace NumPy's stubs.

The optional [Matplotlib adapter](matplotlib.md) can handle the plotting
boundary directly, so `ax.plot(t, q)` needs no manual magnitude extraction.
Other external-library boundaries still use `.value` or `.value_in(unit)`.
`.to()` returns a new Quantity; write `q = q.to(unit)` to retain the conversion.
It has no `inplace` option. NumPy-free scalar consumers retain working type hints.

For `unitx.math` array results, the NumPy-independent Array protocol avoids
unknown optional types in strict scalar users without NumPy. Use
`np.asarray(result.value)` when additional native ndarray methods are needed
in static analysis. Result dtype and shape remain determined by NumPy.

## Verify a consumer

The test suite runs strict scalar fixtures with and without NumPy and checks
array consumers separately when NumPy is installed. `tests/typing` and
`tests/numpy_typing` are static fixtures excluded from pytest collection.
The small `check_asarray.py` regression requires NumPy >= 2.5 stubs; other array
fixtures cover the full supported NumPy range. Notebook checks extract the
actual code and add selected type assertions, rather than maintaining copies
of scientific examples. Runtime test annotations describe fixtures and test
data; they do not replace static `assert_type()` checks of public inference.

```sh
python -m pyright --pythonpath /path/to/your/python
```

Use an interpreter containing NumPy for full-library and array checks.
An installed-consumer check should run outside the checkout and confirm
that the wheel contains `unitx/py.typed`; source-tree inference alone does
not establish that annotations were packaged correctly.

[Documentation index](README.md) · [Scalar numerical behavior](scalars.md) ·
[NumPy support](numpy.md)
