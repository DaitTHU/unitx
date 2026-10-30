# Scientific calculations at explicit boundaries

These recipes distinguish physical values, coordinate readings and numerical
solver/storage values. All examples use the existing quantity model: one array
has one unit, and leaving that model is an explicit choice.

## Tolerances, zero and stored readings

`np.isclose(a, b)` uses `abs(a-b) <= atol + rtol*abs(b)` with a nonzero bare
`atol=1e-8`. A bare nonzero tolerance is dimensionless, so that default is invalid
for dimensional data. Supply a Quantity tolerance or `atol=0` for a purely
relative comparison. `q.isclose` instead uses a symmetric max-based scale and
defaults to `abs_tol=0`; its parameter names and formula are deliberately distinct.

```python
import numpy as np
from unitx import AffineQuantity, LogQuantity, Quantity
from unitx.si import SI, si

length = Quantity(np.array([1.0, 2.0]), si.m)
nearby = Quantity(np.array([100.0001, 200.0001]), si.cm)
assert np.all(np.isclose(length, nearby, rtol=0, atol=2e-6 * si.m))
assert np.all(np.isclose(length, length.to(si.cm), atol=0))
assert np.all(length.isclose(nearby, rel_tol=0, abs_tol=2e-6 * si.m))
```

Selection branches and clipping bounds are physical operands. A bare zero is
not a zero of arbitrary dimensions or an absolute point. Attach a compatible
unit, or construct a point in the same coordinate family. By contrast,
`nan_to_num` replacement parameters are stored readings: bare `nan=0` means zero
in the current unit or coordinate. Changing units first can therefore change
what a bare replacement means. Wrapped replacements explicitly fix the physical
value or coordinate point.

```python
q = Quantity(np.array([-1.0, np.nan, 2.0]), si.m)
nonnegative = np.clip(q, 0 * si.m, None)
selected = np.where(q > 0 * si.m, q, 0 * si.m)
np.testing.assert_array_equal(selected.value, [0, 0, 2])
np.testing.assert_array_equal(np.nan_to_num(nonnegative, nan=300 * si.cm).value, [0, 3, 2])
temperatures = AffineQuantity(np.array([np.nan, 20.0]), si.degC)
filled = np.nan_to_num(temperatures, nan=0)
assert filled[0] == 0 @ si.degC  # Not absolute zero.
absolute_zero = np.nan_to_num(temperatures, nan=0 @ si.K)
np.testing.assert_allclose(absolute_zero[0].value, -273.15)
```

A mean of logarithmic readings is a coordinate statistic. For power readings
it corresponds to a geometric mean of physical powers, not their arithmetic
mean. Standard deviation and variance refer to logarithmic differences and
squared differences; they are not watts and watts squared.

```python
levels = LogQuantity(np.array([0.0, 20.0]), si.dBm)
assert levels.mean() == 10 @ si.dBm
powers = levels.to_quantity(si.mW)
np.testing.assert_allclose(powers.mean().value, 50.5)
np.testing.assert_allclose(levels.mean().to_quantity(si.mW).value, 10)
assert levels.var().unit == si.dB**2
```

## Named units express a choice

Dimensions cannot choose between energy and torque, Hz and Bq, or Gy and Sv.
Choose the desired representation explicitly with `to`; the original formal
identity is unchanged. `simplify` only combines like units and `compact` selects
prefixes. Neither infers a physical quantity kind or searches the catalogue for
the shortest spelling. A new preference API is unnecessary for a fixed target.

```python
work = (2 * si.N) * (3 * si.m)
assert work.to(si.J) == 6 * si.J
assert work.unit == si.N * si.m
```

## Spectral position and density are different transformations

For positive finite **vacuum** wavelengths, ordinary frequency is `nu = c/lam`
and photon energy is `h*nu`. Angular frequency instead introduces `2*pi`.
A refractive medium requires its dispersion relation; these vacuum formulas
must not be applied to a material wavelength by changing a unit label.

For a spectral power density, conservation of power requires
`P_nu = P_lam * abs(dlam/dnu) = P_lam * lam**2/c`. Frequency decreases along an
increasing wavelength grid, so reverse both arrays before integrating in
increasing frequency order. Ordinary `.to()` only changes compatible units;
it does not apply this physical Jacobian.

```python
lam = Quantity(np.linspace(400.0, 800.0, 2001), si.nm)
nu = (SI.c / lam).to(si.THz)
photon_energy = (SI.h * nu).to(si.eV)
assert np.all(photon_energy.value > 0)
p_lam = Quantity(np.ones(lam.shape), si.W / si.nm)
p_nu = (p_lam * lam**2 / SI.c).to(si.W / si.THz)
power_lam = np.trapezoid(p_lam, x=lam)
power_nu = np.trapezoid(p_nu[::-1], x=nu[::-1])
np.testing.assert_allclose(power_nu.value_in(si.W), power_lam.value_in(si.W), rtol=1e-6)
```

This is a finite sampled-grid check with quadrature error, not exact numerical
integration. The [blackbody notebook](https://github.com/DaitTHU/unitx/blob/main/examples/blackbody.ipynb)
uses the same distinction for Planck spectra. Direct formulas remain the public
baseline; no implicit equivalency context or transform graph is needed.

## Scalar random sampling without NumPy

A standard-library `random.Random` owns independent state. Choose one numerical
unit before drawing, then reattach it. Validate and convert parameters before
consuming state. Gaussian widths are nonnegative differences; a point's origin
must not be applied to its width. Decimal/Fraction inputs enter the random
kernel's float domain explicitly. A Gaussian in log readings is a different
physical distribution from a Gaussian in power.

```python
from math import isfinite
from random import Random

center, width = 10.0 * si.cm, 2.0 * si.mm
location = float(center.value_in(si.m))
spread = float(width.value_in(si.m))
assert isfinite(location) and isfinite(spread) and spread >= 0
rng = Random(20261028)
state = rng.getstate()
draw = rng.gauss(location, spread) * si.m
rng.setstate(state)
assert draw == rng.gauss(location, spread) * si.m
```

The sampling and validation need no NumPy. This recipe keeps standard-library
seed/state behavior visible, without adding a second random API. For repeated array Monte
Carlo, see the [pendulum example](https://github.com/DaitTHU/unitx/blob/main/examples/pendulum_uncertainty.ipynb).

[NumPy support](numpy.md) · [SciPy callbacks](scipy.md) · [Array storage](array-storage.md)
