# Scalar statistics

`unitx.statistics` enhances eight functions from Python's `statistics` module.
It works without NumPy. Samples are iterables of Python int, float, Fraction
or Decimal values, linear Quantities, affine points, or logarithmic readings.
Bare numbers are dimensionless; bool, complex and NumPy magnitudes are
unsupported. A generator is consumed once.

```pycon
>>> from unitx import statistics
>>> from unitx.si import si
>>> statistics.mean([1 * si.m, 200 * si.cm, 3 * si.m]) == 2 * si.m
True
>>> statistics.mean([20 @ si.degC, 30 @ si.degC]) == 25 @ si.degC
True
>>> statistics.pvariance([20 @ si.degC, 30 @ si.degC]) == 25 * si.K**2
True
>>> statistics.covariance(
...     [1 * si.m, 2 * si.m, 3 * si.m], [1 * si.s, 2 * si.s, 4 * si.s]
... ) == 1.5 * si.m * si.s
True

```

| Function | Result |
| --- | --- |
| mean(data) | First sample's complete unit or coordinate |
| fmean(data, weights=None) | Same unit/coordinate, always float magnitude |
| variance(data, xbar=None), pvariance(data, mu=None) | Squared difference unit; sample/population variance |
| stdev(data, xbar=None), pstdev(data, mu=None) | Difference unit; sample/population standard deviation |
| covariance(x, y, /) | Float magnitude, product of the two difference units |
| correlation(x, y, /, *, method='linear') | Float magnitude, dimensionless; linear or ranked |

All results retain a Quantity or point wrapper, including dimensionless
results. Each variable uses its first sample's complete unit or coordinate,
including extra scales. Units within a variable must be compatible. Points
and differences cannot mix. Bare values may mix with dimensionless Quantities
but do not acquire a dimensional unit or a point coordinate implicitly.

Point means retain points. Dispersion uses the coordinate's difference unit.
Logarithmic statistics operate on readings; convert to physical quantities
explicitly for power/amplitude statistics. Precomputed `xbar`/`mu` must match
the sample's quantity/point semantics and are converted to its reference.
The backend does not verify that a supplied center is the actual mean.

`fmean` weights are ordinary dimensionless numbers or dimensionless scalar
Quantities; complete scales normalize before use. Weight validity and sample
count errors follow the standard library. Logarithmic differences are not
ordinary dimensionless weights.

Mean and variance use the standard-library numerical kernels after unit
alignment, without an extra float conversion. Decimal/Fraction domains are
retained when alignment and the kernel retain them; mixed numerical domains
follow their supported native arithmetic. Standard deviations follow Python's
sqrt domain: Decimal can retain Decimal, while integer/Fraction results enter
float. Covariance, linear correlation and fmean explicitly use float values,
including for Decimal/Fraction samples; these functions do not promise arbitrary
precision. Ranked correlation compares original scalar observations for ranks
and ties before computing its floating coefficient.

StatisticsError is the standard-library exception, re-exported by this module.
Unit incompatibility raises DimensionError, unsupported inputs raise TypeError.
NaNs are not automatically omitted. NumPy retains its separate array algorithms,
axes and dtypes: `statistics.variance` uses N-1 by default, while `np.var` uses N.

This finite adapter does not add the median/mode/quantiles families, geometric
or harmonic means, regressions, distributions or KDE. Some native statistics
calls already work directly on linear Quantities; their coordinate and exact
comparison contracts need separate assessment before adding further adapters.

Reference: [Python statistics](https://docs.python.org/3/library/statistics.html).

[Documentation index](README.md) · [NumPy statistics](numpy/statistics.md)
