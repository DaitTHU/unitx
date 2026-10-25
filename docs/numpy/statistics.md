# Reductions and statistics

```python
import numpy as np
from unitx import Quantity, Unit
```

## Masked sums and statistics

`np.sum(q)` and `np.add.reduce(q)` preserve q's unit, including Unit and
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

Both entry points support the output [output rules](writes.md#output-buffers), including exact result
shape and unchanged targets on computation, unit, shape, or casting errors.
`cumprod` and `multiply.accumulate` remain unsupported: partial products
generally require different units at different positions. The supported
axis, empty-product, initial, and mask behavior follows
[NumPy prod](https://numpy.org/doc/stable/reference/generated/numpy.prod.html)
and [ufunc.reduce](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduce.html),
with the additional single-unit and explicit-output rules described here.

## Weighted averages and quantiles

`np.average`, `np.quantile`, and `np.percentile` accept a linear `Quantity` as
their first, positional argument. Results retain its complete unit, including
an exact extra scale or a scaled dimensionless unit. AffineQuantity and
LogQuantity points are unsupported; convert explicitly before using these
functions. The `nan*` variants remain unsupported.

Weights and the `q` parameter must be dimensionless: use bare numeric scalars,
plain numeric ndarrays, numeric lists/tuples, or dimensionless Quantities.
Quantity parameters convert to coherent dimensionless values before entering
NumPy; a logarithmic difference such as dB is not an ordinary dimensionless
parameter. Boolean, object, masked and subclass arrays remain unsupported.
Fraction/Decimal scalars follow the usual NumPy floating conversion boundary;
object arrays of them are unsupported. These rules do not change the sample
Quantity's construction rules.

### Average

`np.average(samples, axis=None, weights=None, returned=False, *, keepdims=False)`
computes `sum(samples * weights) / sum(weights)` in stored sample coordinates.
It supports negative and tuple axes. With `axis=None`, weights must have the
same shape as the samples. With explicit axes, weights may have the full sample
shape or the shape of the selected axes, in the given axis order. For example,
samples of shape `(2, 3, 4)` accept weights of shape `(3,)` with `axis=1`,
or `(4, 2)` with `axis=(2, 0)`. Arbitrary broadcastable shapes are not accepted.
NumPy validates shapes and raises ZeroDivisionError for a zero weight sum.
Negative weights follow NumPy's average rules.

With `returned=True`, the result is `(average, sum_of_weights)`: the first is
a Quantity in the sample unit; the second is a Quantity in `Unit()`, with the
same shape as the average (including `keepdims`). Weight sums use the normalized
dimensionless weights. Without weights, this is the number of averaged elements
per slice. NumPy determines both magnitude dtypes. `average` has no `out`,
`dtype`, `where` or `initial` option.

```python
samples = Quantity(np.array([[1.0, 2, 3], [4, 5, 6]]), 'm')
weights = Quantity(np.array([1.0, 2, 1]), Unit('m/cm'))  # coherent weights: 100, 200, 100
means, weight_sums = np.average(samples, axis=1, weights=weights, returned=True)
assert means.unit == samples.unit and weight_sums.unit == Unit()
np.testing.assert_allclose(means.value, [2, 5])
np.testing.assert_allclose(weight_sums.value, [400, 400])
```

### Quantile and percentile

Both functions accept `axis`, `out`, `overwrite_input=False`, `method`,
`keepdims`, and keyword-only `weights`. `quantile` uses `q` in `[0, 1]`;
`percentile` uses `q` in `[0, 100]` **after dimensionless normalization**.
For example, `Quantity(0.5, '')` requests the median with quantile and the
0.5th percentile with percentile. Scalar `q` reduces the selected sample axes;
a vector `q` adds a leading axis indexing the requested levels. `keepdims=True`
retains reduced sample axes with extent one.

Without weights, NumPy's supported `method` choices are available, defaulting
to `linear`: interpolate adjacent sorted samples at position `q * (N - 1)`
for quantile. With weights, explicitly select `method="inverted_cdf"`, which
selects a sample using the weighted empirical cumulative distribution. Other
methods, including the default `linear`, reject weights. Weight shapes follow
the average rules above: full sample shape, or the selected-axis shape with
explicit `axis` (a 1-D vector for one axis). NumPy validates probability ranges,
methods and nonnegative quantile weights, and determines numerical dtypes,
NaN/Inf behavior, empty-sample results and warnings. Complex samples are rejected.
These calls do not omit NaNs automatically.

`out` follows the existing [output rules](writes.md#output-buffers): a Quantity
with a compatible unit, fixed ndarray storage, exact result shape, and
`same_kind` casting. Computation, conversion, shape or casting errors leave
the target unchanged, including when it aliases samples, probabilities or
weights. `overwrite_input=True` is rejected; otherwise sample storage is not
modified except through an explicitly supplied overlapping `out`. `dtype`,
`where`, `initial` and the old `interpolation` keyword are unsupported.

```python
quartiles = np.quantile(samples, [0.25, 0.75], axis=1, keepdims=True)
assert quartiles.unit == samples.unit and quartiles.shape == (2, 2, 1)
np.testing.assert_allclose(quartiles.value[..., 0], [[1.5, 4.5], [2.5, 5.5]])
medians = np.percentile(samples, 50, axis=1, weights=[1, 2, 1], method='inverted_cdf')
np.testing.assert_allclose(medians.value, [2, 5])
```

See [NumPy average](https://numpy.org/doc/stable/reference/generated/numpy.average.html),
[quantile](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html) and
[percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
for the numerical definitions. unitx adds the unit, parameter and output rules
described here and uses the installed NumPy kernel.

[NumPy overview and support matrix](../numpy.md) · [Documentation index](../README.md)
