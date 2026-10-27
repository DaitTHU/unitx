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
`np.asarray(q)` and `np.array(q)` raise `ImplicitConversionError` (a TypeError)
even for dimensionless q;
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

## Direct array methods

Quantity provides `reshape`, `ravel`, `copy`, `sum`, `mean` and `std` for NumPy
magnitudes. Two coordinate families provide the same methods except `sum`;
mean preserves a point and std returns a difference. These methods preserve
the wrapper in static types. See [array methods](numpy/methods.md) for shape,
copy, reduction, output and dtype rules with runnable examples.

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
| minimum, maximum, fmin, fmax | Elementwise selection; align to the first Quantity's unit or point coordinate; fmin/fmax use NumPy's NaN selection |
| multiply, divide/true_divide, matmul / `*`, `/`, `@` | Combine units |
| equal, not_equal, less, less_equal, greater, greater_equal / comparisons | Bare boolean results |
| positive, negative, absolute, fabs, conjugate / unary `+`, `-`, `abs` | Preserve unit |
| square, sqrt, cbrt, reciprocal | Exact unit exponents 2, 1/2, 1/3, -1 |
| power, float_power / `**` | Dimensional bases require a finite real scalar exponent; dimensionless bases/exponents normalize their scales |
| sin, cos, tan, sinh, cosh, tanh, arcsinh, arccosh, arctanh, exp, expm1, exp2, log, log2, log10, log1p | Require coherent dimensionless values; return dimensionless Quantity |
| logaddexp, logaddexp2 | Normalize both dimensionless inputs; return dimensionless Quantity |
| floor, ceil, trunc, rint | Round stored magnitudes; retain unit or point coordinate |
| sign | Dimensionless Quantity; complex convention follows NumPy |
| copysign | First input's unit; second input supplies only the sign and may have any linear unit |
| remainder/mod, fmod | Compatible linear inputs; remainder in the first Quantity's unit |
| floor_divide, divmod | Compatible linear inputs; dimensionless quotient, remainder in the first Quantity's unit |
| degrees/rad2deg, radians/deg2rad | Dimensionless angular input; equivalent angle in deg or rad |
| arcsin, arccos, arctan | Require coherent dimensionless values; return Quantity in rad |
| hypot | Compatible input dimensions; align to the first Quantity's unit |
| arctan2 | Compatible input dimensions; return Quantity in rad |
| isfinite, isnan, isinf, signbit | Bare boolean results |
| modf | Fractional and integral parts in the input unit |
| frexp | Mantissa in the input unit; bare integer binary exponent |
| vecdot | Conjugate first input; product unit; axis/axes options |
| matvec, vecmat (NumPy >= 2.2) | Product units; native matrix/vector shapes and vecmat conjugation; ordinary call options and out |
| ldexp | Input unit; dimensionless integer binary exponents; reconstructs frexp |
| nextafter, spacing | Stored-magnitude floating neighbors/spacing in the complete input unit; nextafter aligns compatible linear operands |
| heaviside | Sign of any linear input, dimensionless value at zero; coherent dimensionless result |
| add.reduce, add.accumulate | Preserve unit; accept axis, dtype, out, plus keepdims, initial, where for reduce |
| multiply.reduce | Product unit rules below; accepts axis, dtype, keepdims, initial, where, out |
| subtract, remainder, fmod: reduce/accumulate/reduceat | Preserve linear unit; initial must be compatible |
| minimum, maximum, fmin, fmax: reduce/accumulate/reduceat | Preserve unit or point coordinate |
| multiply, divide/true_divide, floor_divide, logaddexp, logaddexp2: methods | Dimensionless inputs normalize their scales; multiply.reduce retains its general product rule |
| outer, at | Binary outer uses the underlying call's unit rule; supported fixed-unit at updates stage repeated-index writes; see [method details](numpy/writes.md#ufunc-methods) |

Ufunc `__call__` accepts `where`, `dtype`, `casting`, `order`, `subok`, and
`out` as described below (two outputs for modf/frexp/divmod). Each kernel validates its own options.
`where` is a bare boolean scalar
or plain boolean ndarray; masked-out positions without an output buffer follow
NumPy's uninitialized-result behavior. Numeric outputs must stay in the
supported dtype domain. Method support and limitations are specified in
[ufunc methods](numpy/writes.md#ufunc-methods). Python `**` also accepts Fraction/Decimal exponents, retaining
exact unit exponents while passing a float exponent to the NumPy kernel.
Angles retain unitx's dimensionless model: deg/rad normalize before these
functions, including exp/log. Python scalar arithmetic follows the separate [scalar rules](scalars.md).

| NumPy function | Unit/result rule | Supported options |
| --- | --- | --- |
| block, choose, select | Align compatible quantities or points | Nested blocks; choose mode/out; select bare default is a stored reading; [compatibility details](numpy/compatibility.md) |
| cumulative_sum, cumulative_prod | Linear sum / coherent dimensionless product | axis, dtype, include_initial, staged out |
| ediff1d | Difference unit, including points | Compatible difference to_begin/to_end |
| nan_to_num | Preserve unit or point coordinate | copy; replacements are readings or compatible wrapped values; copy=False stages writes |
| all, any, array_equal, array_equiv, isreal, iscomplex, isrealobj, iscomplexobj, isposinf, isneginf | Bare predicates | [Predicate rules](numpy/compatibility.md) |
| resize, trim_zeros, unstack, astype, shares_memory, may_share_memory, sort_complex, i0 | Storage, metadata or coherent numerical results | [Compatibility sweep](numpy/compatibility.md) |
| linalg.matmul, linalg.outer, linalg.cross, linalg.tensordot, linalg.trace, linalg.diagonal | Product/input units; diagonal also preserves points | Native strict Array API shapes and keywords |
| full_like, linspace, geomspace, logspace, meshgrid, pad, lib.stride_tricks.sliding_window_view | Units, coordinates, difference steps and independent axes | [Allocation and grid rules](numpy/creation.md) |
| copyto, put, put_along_axis, putmask, place, fill_diagonal | Convert into fixed target units/coordinates; stage writes | [Batch write rules](numpy/writes.md#batch-writes) |
| dot, inner, outer, vdot, linalg.vecdot, tensordot, einsum, kron, cross, convolve, correlate, linalg.multi_dot | Product units; [contraction rules](numpy/contractions.md) | Native shape/axis/mode/path rules; out on dot, outer, einsum, multi_dot |
| einsum_path | Bare path and diagnostic string | Native subscript/optimization options |
| trace | Preserve linear unit | offset, axis1, axis2, dtype, out |
| linalg.norm | Preserve linear unit; ord=0 returns a dimensionless nonzero count | ord, axis, keepdims; native vector/matrix rules |
| linalg.vector_norm, linalg.matrix_norm | Preserve linear unit; vector ord=0 is dimensionless | Native ord/keepdims; vector axis; [linear algebra rules](numpy/linalg.md) |
| linalg.solve, linalg.tensorsolve, linalg.lstsq | Solution unit is b.unit / a.unit; lstsq has per-result units | Native shapes/axes/cutoffs; [linear systems](numpy/linalg.md) |
| linalg.inv, linalg.pinv, linalg.tensorinv | Reciprocal input unit | Native hermitian/cutoff/ind options |
| linalg.matrix_power, linalg.det | Input unit raised to integer power or matrix order | Native powers, square/stacked shapes |
| linalg.cond, linalg.matrix_rank | Dimensionless Quantity or bare integer rank | Native orders and absolute/relative cutoffs |
| linalg.qr | Dimensionless Q and input-unit R; mode='r' returns only R | reduced, complete, r; raw and deprecated modes unsupported |
| linalg.cholesky | Exact square root of the input unit | upper; native positive-definite/stacked rules |
| linalg.svd, linalg.svdvals | Input-unit singular values; dimensionless U and Vh | Native full_matrices, compute_uv, hermitian on svd |
| linalg.eig, linalg.eigh, linalg.eigvals, linalg.eigvalsh | Input-unit eigenvalues; dimensionless eigenvectors when returned | Native stacked/complex rules; UPLO on Hermitian routines |
| fft.fft/ifft, fft.fft2/ifft2, fft.fftn/ifftn, fft.rfft/irfft, fft.rfft2/irfft2, fft.rfftn/irfftn, fft.hfft/ihfft | Preserve the complete linear input unit | Native n/s, axis/axes and norm; staged out; [FFT rules](numpy/fft.md) |
| fft.fftshift, fft.ifftshift | Preserve unit or point coordinate | Native axes and odd/even length behavior; no out |
| fft.fftfreq, fft.rfftfreq | Quantity spacing produces reciprocal-unit Quantity frequencies; bare spacing retains NumPy's bare result | Native n, scalar d and device; works through Quantity arithmetic |
| ptp | Complete difference unit, including points | axis, out, keepdims |
| cov, corrcoef | Squared difference unit / dimensionless coefficients | Native matrix/options; [covariance rules](numpy/histograms.md) |
| histogram, histogram2d, histogramdd, histogram_bin_edges, digitize, bincount | Coordinate edges, weight counts, inverse-coordinate density, or bare indices | [Histogram rules](numpy/histograms.md) |
| nancumsum | Preserve linear unit | axis, dtype, out |
| cumprod, nancumprod | Normalize dimensionless inputs; dimensional cumulative products unsupported | axis, dtype, out |
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
| concatenate, stack | Align to the first Quantity's unit or point coordinate; bare arrays remain dimensionless | axis, dtype, casting, out; dtype and out are mutually exclusive |
| hstack, vstack, dstack, column_stack | Align to the first Quantity's unit or point coordinate | hstack/vstack accept dtype and casting; no out |
| append, insert | Align to the first Quantity's unit or point coordinate | axis; insert also accepts obj; no out |
| split, array_split, hsplit, vsplit, dsplit | List of results in the input unit or point coordinate | indices_or_sections; split/array_split also accept axis |
| take, take_along_axis | Preserve unit or point coordinate | indices, axis; take also accepts mode and out |
| compress, extract, delete | Preserve unit or point coordinate | bare boolean condition; delete accepts obj; compress/delete accept axis; compress accepts out |
| diagonal, diag, diagflat, tril, triu | Preserve unit or point coordinate, including filled zero readings | offset/axis1/axis2 for diagonal; k for the others; no out |
| sort, partition | Preserve unit or point coordinate | axis, kind, order; sort accepts stable; partition requires bare integer kth |
| argsort, argpartition, lexsort | Bare integer indices; lexsort keys keep independent units | axis; argsort/argpartition follow their sort/partition options |
| searchsorted | Bare insertion indices; align compatible samples | side, bare integer sorter |
| unique, unique_all/counts/inverse/values | Wrapped values; bare indices and counts | unique accepts return_index/inverse/counts, axis, equal_nan, version-available sorted |
| isin, intersect1d, union1d, setdiff1d, setxor1d | Align compatible inputs; isin returns a mask, sets return wrapped values | Native assume_unique, invert, kind and return_indices where applicable |
| nonzero, flatnonzero, argwhere, count_nonzero | Bare indices or counts of nonzero readings | count_nonzero accepts axis and keepdims |
| zeros_like, ones_like, empty_like | Preserve unit or point coordinate | dtype, order, subok, shape, device; plain numeric outputs only |
| real, imag | Preserve linear Quantity unit; views follow NumPy | No options; points are unsupported |
| real_if_close | Preserve linear Quantity unit; native real/complex dtype decision | tol; points are unsupported |
| angle | Phase of a linear Quantity in rad or deg | deg; points are unsupported |
| round/around, fix | Round stored magnitudes; preserve unit or point coordinate | decimals/out for round; out for fix |
| unwrap | Unwrap a dimensionless linear phase; preserve its input unit | discont, axis, period; bare thresholds are radians |
| sinc | Normalize the complete dimensionless input scale; return dimensionless Quantity | NumPy's sin(pi*x)/(pi*x), including its value 1 at zero; no out |
| sum, nansum | Preserve unit | axis, dtype, keepdims, initial, where, out |
| mean, nanmean | Preserve unit or point coordinate | axis, dtype, keepdims, where, out |
| min/amin/nanmin, max/amax/nanmax | Preserve unit or point coordinate | axis, keepdims, initial, where, out |
| where | Select between compatible quantities or points | condition, x, y, all positional; no out |
| clip | Preserve input unit or point coordinate | a_min/a_max or min/max, out; ufunc call options |
| prod, nanprod | Product unit rules below; nanprod counts selected non-NaN elements | axis, dtype, keepdims, initial, where, out |
| median, nanmedian | Preserve unit or point coordinate | axis, keepdims, out; overwrite_input=False |
| std/nanstd, var/nanvar | Difference unit / square difference unit | axis, dtype, ddof/correction, mean, keepdims, where, out |
| average | Preserve sample unit; returned weight sum is a dimensionless Quantity | axis, weights, returned, keepdims; linear Quantity only |
| quantile, percentile, nanquantile, nanpercentile | Preserve sample unit; dimensionless q and weights | axis, out, method, keepdims, weights; overwrite_input=False; linear Quantity only |
| cumsum | Preserve unit | axis, dtype, out |
| argmin/nanargmin, argmax/nanargmax | Bare indices | axis, keepdims, out |
| interp | Align x to xp's unit; retain fp's unit | finite strictly increasing 1-D xp; scalar left/right, period=None; no out |
| trapezoid | Multiply ordinate and coordinate units | x, scalar dx, axis; no out |
| diff | Preserve linear unit; point inputs return differences for n > 0 | n, axis, compatible point/value prepend and append; no out |
| gradient | Divide input difference unit by each coordinate difference unit | scalar difference spacings or coordinate arrays, axis, edge_order; no out |
| isclose, allclose | Bare mask / Python bool | rtol, atol, equal_nan |

Supply the first magnitude argument positionally; concatenate/stack take a
sequence. Options follow each supported signature, including positional out
checks. Function and reduction outputs follow the rules below;
overwrite_input=True is rejected. Reduction/statistical `where` is supported
for sum/nansum, mean/nanmean, std/nanstd, var/nanvar, add.reduce, prod/nanprod,
multiply.reduce, and min/max/nanmin/nanmax; `initial` is supported for
sum/nansum/add.reduce, prod/nanprod/multiply.reduce, and min/max/nanmin/nanmax.
std/var/nanstd/nanvar accept precomputed `mean` and dimensionless `correction`;
the mean must preserve sample quantity/point semantics and is converted to the
sample reference. NumPy controls shape and ddof/correction conflicts.
Indexed assignment and augmented arithmetic on ndarrays follow the fixed-target
rules below. Python and NumPy scalar magnitudes retain ordinary result/rebinding
behavior for augmented arithmetic.

## Shape transformations

`squeeze`, `expand_dims`, `moveaxis`, `swapaxes`, `rollaxis`, and `ravel`
operate on magnitudes without conversion. Like reshape and transpose, they
retain the complete unit or point coordinate and return the same family:
Quantity, AffineQuantity, or LogQuantity. Ravel also has a direct method;
these functions have no `out` option.

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

## Joining, splitting and indexing

See [joining, splitting and indexing](numpy/indexing.md) for compatible inputs,
point coordinates, structural indices, diagonal zero filling, views and output
buffers.

## Sorting, searching and sets

`sort` and `partition` operate on the stored readings, without changing units
or point coordinates. `argsort`, `argpartition`, and `lexsort` return bare
indices. Each `lexsort` key retains its own unit or coordinate: keys need not
have compatible physical dimensions. Supply keys as numeric scalars/ndarrays
or wrapped samples, not implicit list magnitudes.

`searchsorted`, `isin`, and the four binary set functions align inputs to the
first wrapped sample's unit or coordinate. They share the joining rules:
bare numbers/ndarrays are dimensionless, points and differences cannot mix,
and point families and logarithmic conventions must match. Unit alignment
uses the numerical conversion rules above, including floating rounding and
overflow; set membership has NumPy's equality semantics, without tolerance.

`unique` wraps only its values; its optional index, inverse and count outputs
stay bare. `unique_all`, `unique_counts`, and `unique_inverse` retain NumPy's
namedtuple type and fields, with only `values` wrapped. `unique_values` returns
one wrapped array. Native NaN behavior and output ordering follow the selected
function. NumPy 2.3 introduced `unique(sorted=...)`; older NumPy rejects that
keyword. `nonzero`, `flatnonzero`, `argwhere`, and `count_nonzero` use stored
readings, including coordinate zero for points. None of these functions has
an `out` option. Indices such as `kth` and `sorter` must be bare integers.

## Allocation and complex components

See [allocation, grids and padding](numpy/creation.md) for `full_like`,
`linspace`, `geomspace`, `logspace`, `meshgrid`, `pad` and sliding-window views.

`zeros_like(q)`, `ones_like(q)` and `empty_like(q)` allocate numeric storage while preserving
the complete unit or point coordinate. Native dtype, order, shape overrides,
writeability and errors apply. Zero- and one-filled points mean readings of zero and one in their retained
coordinate, as for diagonal construction. `empty_like` leaves values
uninitialized. Boolean, object, string and other unsupported magnitude dtypes
remain rejected; `subok` does not enable ndarray subclasses.

`real(q)` and `imag(q)` operate on linear quantities, including complex
magnitudes, and retain the full unit. Real/imaginary views share storage with
the original wherever NumPy does, including writeability. Points are not
supported for these component operations. For NumPy magnitudes, `q.real` and
`q.imag` provide the same components and preserve NumPy's native result types.
These properties also let NumPy's `real`/`imag` type hints retain the Quantity
wrapper and component dtype; see [typing](typing.md#numpy-component-inference).

```python
samples = Quantity(np.array([1.0 + 2.0j, 3.0 + 4.0j]), 'V')
buffer = np.empty_like(samples, shape=(3, 2))
buffer[:] = samples
np.testing.assert_array_equal(np.real(buffer).value[0], [1, 3])
np.testing.assert_array_equal(np.imag(buffer).value[0], [2, 4])
```

## Norms

See [linear algebra](numpy/linalg.md) for the full group,
including solve, least-squares results, inverse units and rank thresholds.

`np.linalg.norm(q, ord=None, axis=None, keepdims=False)` accepts linear
quantities. Vector norms and matrix norms (including Frobenius, nuclear and
spectral norms) retain the complete input unit. NumPy controls supported orders,
axes, shape, dtype, NaNs and numerical errors. The vector `ord=0` operation
counts nonzero entries and returns a dimensionless Quantity. Points are
unsupported; a norm of coordinate readings would depend on their origin.

Dividing by the norm produces a dimensionless normalized quantity:

```python
voltage = Quantity(np.array([3.0, 4.0]), 'V')
length = np.linalg.norm(voltage)
assert length.unit == Unit('V') and length.value == 5
direction = voltage / length
np.testing.assert_allclose(direction.value_in(''), [0.6, 0.8])
```

## Powers and uniform result units

Dimensional bases require one finite real scalar exponent, including a
zero-dimensional exponent array or scalar dimensionless Quantity. Exponent
arrays with one or more axes are rejected, even when all entries are equal.
In particular, elementwise powers that would produce m, m² and m³ in different
positions do not return a per-element-unit array or an object-array fallback.

```python
import numpy as np
from unitx import Quantity, Unit

lengths = Quantity(np.array([2.0, 3.0]), 'm')
assert np.power(lengths, 2).unit == Unit('m**2')
# np.power(lengths, np.array([1, 2])) raises TypeError:
# "Dimensional powers require one scalar exponent."
```

Ordinary dimensionless bases can use dimensionless array exponents and normal
NumPy broadcasting. Both operands' complete unit scales normalize first, so
percent and km/m do not behave like their unscaled stored magnitudes.
The result has the coherent dimensionless unit. Nonzero logarithmic degrees
do not qualify as ordinary dimensionless values.

## Mathematical and complex functions

Hyperbolic functions, their inverses, `exp2` and both `logaddexp` variants
normalize dimensionless inputs to coherent values. Inverse hyperbolic results
are dimensionless; inverse trigonometric results remain in radians.
NumPy controls real/complex domains, dtype, broadcasting, NaNs and warnings.

`sinc(x)` accepts linear dimensionless Quantities and normalizes their full
unit scale before calling NumPy's `sin(pi*x)/(pi*x)` kernel, with value 1 at
zero. It returns a coherent dimensionless Quantity; points and logarithmic
differences are unsupported. Equivalent dimensionless units produce the same
function values:

```python
x = Quantity(np.array([-50.0, 0.0, 50.0]), 'cm/m')
y = np.sinc(x)
assert y.unit == Unit()
np.testing.assert_allclose(y.value, [2 / np.pi, 1, 2 / np.pi])
```

`floor`, `ceil`, `trunc`, `rint`, `round`/`around` and `fix` round the **stored
magnitude**, retaining the complete unit or point coordinate. Thus rounding
metres and rounding centimetres have different physical resolutions. `decimals`
is a bare integer. Output buffers follow the existing staged-write rules.
`fix` forwards NumPy's deprecation warning where the installed NumPy version
emits one; prefer `trunc` in new code.
`sign` returns a dimensionless Quantity; `copysign` retains its first input's
unit and takes only the sign of its second linear input, regardless of dimension.

`floor_divide`, `remainder`/`mod`, `fmod` and `divmod` require compatible linear
inputs, aligned to the first Quantity's unit. Floor division returns a
dimensionless Quantity; remainders retain the alignment unit. `divmod` returns
both with the multiple-output staging rules. Bare values are dimensionless,
including zero. Points are unsupported for these operations.

`angle(z, deg=False)` accepts a linear Quantity of any dimension and returns
its complex phase in rad (deg when requested). `real_if_close(z, tol=100)`
retains the input unit and NumPy's dtype-dependent tolerance rule. Both reject
points. `degrees`/`rad2deg` and `radians`/`deg2rad` return an equivalent angle in
the named target unit: the source unit remains authoritative. For example,
`np.deg2rad(180 * si.deg)` is pi radians; coherent dimensionless input already
represents radians. Bare NumPy calls retain NumPy's ordinary conventions.

`unwrap(p, discont=None, axis=-1, *, period=2*pi)` requires a dimensionless
linear phase. It normalizes to radians for NumPy's kernel, then restores p's
complete unit. A bare `period` or `discont` is in radians; a dimensionless
Quantity threshold is converted to radians. Thresholds must be finite real
scalars, with positive period and nonnegative discont. Scaled dimensionless
units and degrees work with the default period. No output buffer is supported.

```python
phase = Quantity(np.array([170.0, -170.0, -160.0]), 'deg')
np.testing.assert_allclose(np.unwrap(phase).value, [170, 190, 200])
turns, remainder = np.divmod(Quantity(np.array([250.0]), 'cm'), Quantity(1.0, 'm'))
np.testing.assert_array_equal(turns.value, [2])
np.testing.assert_array_equal(remainder.value, [50])
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
