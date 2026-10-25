# Joining, splitting and indexing

These NumPy function forms preserve the complete unit or point coordinate.
The first data argument is positional. NumPy controls shape, axis, dtype,
copy/view behavior and kernel errors within the supported input domain.
Quantity, AffineQuantity and LogQuantity are supported; methods with the same
names on these wrappers are not added.

## Joining samples

`hstack`, `vstack`, `dstack` and `column_stack` take a sequence of samples.
Like `concatenate` and `stack`, they convert all samples to the first wrapped
input's unit or coordinate. Bare numeric scalars/ndarrays are dimensionless;
bare list/tuple sample magnitudes remain unsupported. Different physical
dimensions, points mixed with differences, different point families and
different logarithmic conventions are rejected. Compatible logarithmic
references may differ: for example, dBW readings can join dBm readings.

`hstack` and `vstack` accept NumPy's keyword-only `dtype` and `casting`.
`dstack` and `column_stack` have neither option. These four functions have no
`axis` or `out` parameter. They retain NumPy's rules for promoting scalars and
one-dimensional inputs to higher-dimensional arrays.

`append(arr, values, axis=None)` and `insert(arr, obj, values, axis=None)`
use the same alignment rules, including when only `values` is wrapped.
Both allocate a new result and flatten the inputs as specified by NumPy when
`axis=None`. `insert` retains the destination magnitude's dtype after unit
conversion, so fractional values may be truncated when inserting into an
integer array. NumPy's distinction between `obj=0` and `obj=[0]` is retained.

```python
import numpy as np
from unitx import AffineQuantity, Quantity

distance = Quantity(np.array([1.0, 2.0]), 'm')
joined = np.hstack((distance, Quantity(np.array([300.0]), 'cm')))
assert joined.unit == distance.unit
np.testing.assert_allclose(joined.value, [1, 2, 3])
temperature = np.append(
    AffineQuantity(np.array([0.0]), 'degC'),
    AffineQuantity(np.array([273.15]), 'K'),
)
np.testing.assert_allclose(temperature.value, [0, 0], atol=1e-12)
```

## Splitting and selecting

`split`, `array_split`, `hsplit`, `vsplit` and `dsplit` return a list of wrapped
arrays, each retaining the source unit or coordinate. Views, empty sections,
writeability and storage sharing follow NumPy. `split` requires equal division
when passed a section count; `array_split` also permits unequal division.
Only `split` and `array_split` accept an `axis` argument.

`take` accepts `indices`, `axis`, `out` and `mode='raise'/'wrap'/'clip'`.
`take_along_axis` accepts `indices` and `axis`; supply `axis` explicitly for
portability because NumPy 2.1 requires it and newer NumPy defaults to -1.
`compress(condition, a, axis=None, out=None)` and `extract(condition, arr)`
use bare boolean conditions. `extract` returns a flattened selection.
`delete(arr, obj, axis=None)` returns a copy with selected entries removed.

Indices, section boundaries and insertion/deletion positions are bare integer
scalars, sequences or plain integer ndarrays, with slices or boolean indices
where NumPy accepts them. Conditions are bare boolean scalars, sequences or
plain boolean ndarrays. Quantities and points cannot serve as structural
indices or conditions, including dimensionless quantities. Boolean `insert`
indices retain the installed NumPy version's semantics, which changed in
NumPy 2.1.2. Other malformed indices and axes follow NumPy's errors.

`take` and `compress` support fixed-unit output buffers. The result is computed,
converted to the target unit or compatible coordinate, shape checked and cast
with `same_kind` before writing. The target object, ndarray, unit and dtype
stay fixed. Unit, shape, index, casting and read-only errors leave the target
unchanged, including when it aliases the source or indices. Other functions
on this page have no `out` option.

## Diagonals and triangular arrays

`diagonal(a, offset=0, axis1=0, axis2=1)` extracts diagonals and retains NumPy's
read-only view. `diag(v, k=0)` extracts a diagonal from a matrix or constructs
a matrix from a vector. `diagflat(v, k=0)` flattens the input and constructs a
diagonal matrix. `tril(m, k=0)` and `triu(m, k=0)` retain the lower or upper
triangle on the last two axes, following NumPy's rules for vectors and batches.

Constructed or masked entries contain zero **in the retained unit or
coordinate**. For example, `diag` on Celsius points fills off-diagonal entries
with 0 °C; `tril` on dBm readings fills the masked triangle with 0 dBm.
Coordinate zero is not necessarily physical zero. Convert to physical linear
quantities before these operations when physical zero is intended. Extraction
does not introduce new zero readings. All these operations retain NumPy's
dtype, strides, storage sharing and writeability.

NumPy's type stubs describe ndarray results and may reject wrapper inputs for
`take_along_axis`, despite runtime dispatch. See the [typing guide](../typing.md)
for this boundary. The semantics above follow NumPy's official
[take](https://numpy.org/doc/stable/reference/generated/numpy.take.html),
[insert](https://numpy.org/doc/stable/reference/generated/numpy.insert.html) and
[diagonal](https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html)
references, with unitx's explicit alignment and fixed-output rules.

[NumPy support](../numpy.md)
