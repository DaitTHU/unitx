# unitx

[![PyPI version](https://img.shields.io/pypi/v/unitx)](https://pypi.org/project/unitx/)
[![Python versions](https://img.shields.io/pypi/pyversions/unitx)](https://pypi.org/project/unitx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/DaitTHU/unitx/blob/main/LICENSE)

A modern, intuitive unit library for Python 3.12+.
Exact unit algebra, expressive type hints, and NumPy integration.

Write physical formulas with units and constants in Python. `unitx` provides
unit conversion and dimensional analysis for Python scalars and integrates
with NumPy array arithmetic, broadcasting, and supported functions.
Unit scales and rational exponents retain their exact structure.

The library is in early development; public APIs may change during the 0.x series.

## Install

Python **3.12 or later** is required. Install unitx with:

```sh
pip install unitx
```

There are no mandatory runtime dependencies. If you already use NumPy **2.1
or later**, this installation also lets you use units in your NumPy calculations.
For NumPy work in a new environment or with an older NumPy version, use the
following command to install unitx and a compatible NumPy version:

```sh
pip install 'unitx[numpy]'
```

For development, install into your active Python environment:

```sh
pip install -e '.[test,numpy]'
```

This README describes the current source tree. For an older release, consult
its matching Git tag.

## Start with units and constants

The `unitx.si` submodule provides ready-to-use units and physical constants
based on the International System of Units (SI), through two read-only
namespaces:

- `si` contains units such as `si.m`, `si.s`, and `si.MeV`, including common
  prefixes and non-SI units.
- `SI` contains physical constants expressed in SI units, such as the speed
  of light `SI.c` and the Planck constant `SI.h`.

In the import below, `unitx.si` is the module path; the imported `si` and `SI`
are the namespaces you use directly in formulas:

```pycon
>>> from unitx.si import SI, si
>>> distance = 3 * si.m
>>> elapsed = 2 * si.s
>>> speed = distance / elapsed
>>> print(speed)
1.5 m/s
>>> print(speed.to(si.cm / si.s))
150.0 cm/s
>>> print(SI.c)
299792458 m/s
>>> rest_energy = (SI.me * SI.c**2).to(si.MeV)
>>> round(float(rest_energy.value), 6)
0.510999

```

Multiplying a number by a unit constructs a Quantity. Multiplication and
division combine units; addition and subtraction align compatible quantities
to the left operand's unit. Incompatible conversions and arithmetic raise
`DimensionError`.

## Units in NumPy: blackbody radiation

Attach units to NumPy arrays, then use broadcasting and NumPy functions to
evaluate a physical formula. Here, a column of wavelengths and three
temperatures broadcast to give Planck's spectral radiance.

Add `matplotlib` to your environment to plot the result:

```python
import numpy as np
from matplotlib import pyplot as plt

from unitx.si import SI, si

lam = np.linspace(0.01, 3, 100)[:, None] * si.um  # wavelength
T = np.asarray([3000, 4000, 5000]) * si.K  # temperature

nu = SI.c / lam  # frequency
radiance = 2 * SI.h * SI.c**2 / (lam**5 * np.expm1(SI.h * nu / (SI.kB * T)))
radiance = radiance.to('kW/sr/m2/nm')  # spectral radiance per unit wavelength

plt.plot(lam.value, radiance.value)
plt.xlabel(f'Wavelength [{lam.unit}]')
plt.ylabel(f'Spectral radiance [{radiance.unit}]')
plt.legend([f'T = {t}' for t in T])
plt.title('Blackbody radiation')
plt.show()
```

![Blackbody spectral radiance](https://raw.githubusercontent.com/DaitTHU/unitx/main/docs/images/blackbody.png)

One Quantity wraps the entire ndarray with one unit. NumPy handles broadcasting;
unitx checks dimensions and combines units. `SI.h` and `SI.kB` retain Decimal
values in the catalogue and enter NumPy calculations as floats.

`.to()` returns a new Quantity. `.value` exposes the numerical data for plotting;
`.value_in(unit)` also converts it to a chosen unit. Implicit unit removal with
`np.asarray(quantity)` or `np.array(quantity)` raises TypeError.

The [blackbody notebook](https://github.com/DaitTHU/unitx/blob/main/examples/blackbody.ipynb) uses `np.asarray([...]) * unit`
without an explicit dtype. This automatic construction inference is checked
with NumPy 2.5 stubs and Pyright; the example also runs on NumPy 2.1. Older
NumPy stubs may still infer Any for this expression. NumPy's own stubs can
infer different types for `np.array` and `np.asarray`;
`np.array([...]) * unit` can become Any. `unit * array` and
`Quantity(array, unit)` provide alternative construction forms. Arbitrary
shape/dtype inference and wrapped return types for every NumPy function are not
promised. See the [supported operations, storage rules, and typing limits](https://github.com/DaitTHU/unitx/blob/main/docs/numpy.md).

The [physics notebooks](https://github.com/DaitTHU/unitx/blob/main/examples/README.md) also cover damped motion, Gaussian
wave-packet spreading, optical attenuation, and correlated measurement uncertainty.
They include explanations and numerical checks; plots are generated when run,
keeping saved notebooks small.

## Values and conversion

Use `Quantity` and `Unit` directly when constructing values from strings:

```pycon
>>> from unitx import Quantity, Unit
>>> length = Quantity(1, 'm')
>>> print(length + Quantity(50, 'cm'))
3/2 m
>>> length.value_in('cm')
100
>>> Quantity(1, 'cm').value_in('m')
Fraction(1, 100)
>>> Quantity(1, 'm') == Quantity(100, 'cm')
True
>>> Unit('m') == Unit('cm')
False

```

A quantity's `.value` is paired with its `.unit`. Units retain their formal
identity: metre and centimetre are different units, although quantities can
compare equal after accounting for their scales.

Python scalar operations follow the magnitude's numerical domain. Decimal
operations retain Decimal context behavior; Fraction magnitudes provide exact
rational arithmetic. Scalar equality compares exact stored values, so binary
float `0.1` is distinct from Fraction(1, 10). Use symmetric `.isclose()` for
approximate comparisons:

```pycon
>>> Quantity(0.1 + 0.2, 'm').isclose(Quantity(0.3, 'm'))
True
>>> a, b = Quantity(1, 'm'), Quantity(100.05, 'cm')
>>> a.isclose(b, rel_tol=0, abs_tol=1 * si.mm)
True

```

NumPy numerical operations instead follow the documented array conversion,
promotion, rounding, and overflow rules. Array comparisons return boolean
masks, and NumPy-backed quantities are unhashable.

Read the [scalar numerical contract](https://github.com/DaitTHU/unitx/blob/main/docs/scalars.md) for Decimal conversion,
exact comparisons, powers, tolerances, and optional SymPy factorization.

## Expressions and custom units

Parse symbols, registered names, prefixes, products, division, and rational
powers:

```pycon
>>> Unit('kg m/s2') == Unit('kg*m/s**2')
True
>>> Unit('m^(1/2)') ** 2 == si.m
True
>>> Unit('ms') == Unit('m s')
False
>>> Unit('[astronomical unit] / s') == si.au / si.s
True

```

Multiplication and division associate left to right: write `m/(s A)` for a
shared denominator. Multiword names can stand alone; use `[registered name]`
inside expressions. Readable `str()` output is not a guaranteed parser input.

Define a unit from an existing unit or a positive finite quantity:

```pycon
>>> stride = (2 * si.m).define('demoStride', 'demo stride', prefixable=False)
>>> (3 * stride).value_in('m')
6
>>> Unit('demo stride') == stride
True

```

Definitions register spellings in the current process and return a new Unit.
Use `Factor` or `SymbolicConstant` with a unit when an additional exact unit
scale is intended; ordinary numbers construct quantities.

See the [unit and quantity guide](https://github.com/DaitTHU/unitx/blob/main/docs/units.md) for ScaledUnit, dynamic
registration, spelling-specific prefix permissions, catalogue conventions,
copying, and serialization.

## Notebook display and formatting

`Unit`, `ScaledUnit`, and `Quantity` provide Notebook LaTeX and HTML display.
Use `.to_latex()` for a math fragment without `$` delimiters, or `.to_html()`
for an escaped HTML fragment. No IPython dependency is required.

```python
from unitx.quantity import Quantity

speed = Quantity(12.3456, 'm/s')
assert f'{speed:.2f}' == '12.35 m/s'
assert format(speed, '.2f;latex') == speed.to_latex('.2f')
assert format(speed, '.2f;siunitx') == r'\qty{12.35}{m.s^{-1}}'
```

See [display and formatting](https://github.com/DaitTHU/unitx/blob/main/docs/display.md)
for syntax, exact-scale rendering, escaping, and array limitations.

## Scope and types

- Supported magnitudes are Python scalars and plain numeric NumPy
  arrays/scalars. Other array backends, ndarray subclasses, object arrays,
  and implicit list/tuple magnitudes are unsupported.
- Conversion uses multiplication by a scale. Offset temperature scales such
  as Celsius and Fahrenheit are unsupported.
- Covered NumPy ufuncs, functions, and reductions support unit-aware `out`
  buffers. Writable ndarray Quantities also support indexed assignment and
  augmented arithmetic, keeping the destination array, dtype, and unit fixed.
  Results requiring heterogeneous units remain unsupported. See the
  [NumPy support matrix](https://github.com/DaitTHU/unitx/blob/main/docs/numpy.md).
- The package ships inline annotations and `py.typed`. Construction and
  unit-only operations preserve the magnitude type; `.T` also retains it.
  Array calculations and conversion use concrete ndarray/NumPy-scalar result
  unions, making native `.value` methods available without claiming an
  unchanged dtype. Scalar protocols avoid recursive arithmetic return types
  so editor analysis remains responsive.

The package root exports `Quantity`, `Unit`, `Dimension`, `DIMENSIONLESS`,
`ONE`, `TEN`, and `PI`; `SI` and `si` are imported from `unitx.si`.
Other building blocks are available from their modules. Modules beginning
with `_` are internal.

## Documentation and development

- [Unit and quantity guide](https://github.com/DaitTHU/unitx/blob/main/docs/units.md)
- [Scalar numerical behavior](https://github.com/DaitTHU/unitx/blob/main/docs/scalars.md)
- [NumPy support matrix and typing](https://github.com/DaitTHU/unitx/blob/main/docs/numpy.md)
- [Migration from 0.3.0](https://github.com/DaitTHU/unitx/blob/main/docs/migration.md)
- [Changelog](https://github.com/DaitTHU/unitx/blob/main/CHANGELOG.md)

Run the checks from the repository root:

```sh
pytest -q
ruff check .
ruff format --check .
python -m doctest README.md docs/units.md docs/scalars.md
```

Pytest includes runtime tests, source doctests, and Pyright checks. CI covers
Python 3.12–3.15, optional SymPy, and minimum/current NumPy versions. Strict
scalar consumers are also checked without NumPy installed.

Publishing a GitHub Release triggers the [release workflow](https://github.com/DaitTHU/unitx/blob/main/.github/workflows/release.yml)
to validate and upload wheel/sdist distributions to PyPI. Manual workflow runs
validate and store distributions without uploading them. Release tags must
match the project version.

Licensed under the [MIT License](https://github.com/DaitTHU/unitx/blob/main/LICENSE).
