# unitx

[![PyPI version](https://img.shields.io/pypi/v/unitx)](https://pypi.org/project/unitx/)
[![Python versions](https://img.shields.io/pypi/pyversions/unitx)](https://pypi.org/project/unitx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Write Python formulas with ready-to-use units and physical constants. `unitx`
handles scalar and optional NumPy quantities, unit conversion, and dimensional
analysis, with exact rational unit scales and exponents.

`unitx` is in early development. Public APIs may change during the 0.x series.

## Installation

Requires **Python 3.12 or later**, with no mandatory runtime dependencies.

Install a published release:

```sh
python -m pip install unitx
```

To use the current development version, run this from the repository root:

```sh
python -m pip install -e .
```

This README describes the current source tree, including unreleased changes.
For a published release, consult the README at its matching Git tag.

## Quick start

Start with the two namespaces in `unitx.si`: lowercase `si` provides units,
and uppercase `SI` provides physical constants. Their members are ready to use
in ordinary Python formulas:

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

```

Multiplying a scalar by a unit constructs a quantity. Divide quantities to
compute rates, multiply them to combine physical quantities, and call `.to()`
with the unit you want. A quantity's `.value` and `.unit` expose its magnitude
and unit separately.

## Units and constants with SI

Use the same import for both namespaces:

```python
from unitx.si import SI, si
```

| Namespace | Members | Examples |
| --- | --- | --- |
| `si` | Units (`Unit`) | `si.m`, `si.ns`, `si.kg`, `si.N`, `si.J`, `si.MeV` |
| `SI` | Physical constants (`Quantity`) | `SI.c`, `SI.h`, `SI.hbar`, `SI.e`, `SI.kB`, `SI.me` |

Common prefixed units and units such as hours, litres, electronvolts, and
astronomical units are included. Units and constants can be combined directly:

```pycon
>>> rest_energy = (SI.me * SI.c**2).to(si.MeV)
>>> round(float(rest_energy.value), 6)
0.510999
>>> (2 * si.h).to(si.s).value
7200
>>> (1 * si.us).to(si.ns).value
1000

```

Both namespaces are read-only and do not need to be instantiated. `si` exposes
a selection of convenient attributes; other registered prefix combinations
remain available through `Unit`, for example `Unit('Tm')`. Importing `unitx`
initializes the predefined unit catalogue. Defining a custom unit registers its
text spellings without adding an attribute to `si`.

SI defining constants retain their exact decimal values. Measured constants
are stored nominal values; measurement uncertainties are not propagated.
Derived Decimal constants are evaluated in the Decimal context active at
import. Pi and the positive roots used by the Wien displacement constants
(`SI.b` for wavelength and `SI.b_` for frequency) use fixed Decimal
approximations with about 50 significant digits. Raising the context precision
does not add accuracy beyond those stored approximations.

## Constructing units and quantities

For unit expressions and explicit construction, import the classes:

```pycon
>>> from unitx.factor import Factor
>>> from unitx.quantity import Quantity
>>> from unitx.unit import Unit
>>> Quantity(50, 'cm') == 50 * si.cm
True
>>> Unit('astronomical unit') == si.au
True

```

`Quantity(value, unit)` accepts a unit expression or an existing `Unit` object.
`value * unit` constructs a quantity, and `value / unit` constructs a quantity
with the reciprocal unit. Construction preserves the original magnitude and
its type.

Multiplication is symmetric: `unit * value` and `value * unit` both construct
`Quantity` objects and preserve the magnitude's type. Dividing a unit by an
`int`, `Fraction`, or finite `Decimal` constructs a quantity with an exact
`Fraction` reciprocal, so `degree / 60` loses no precision. Float divisors use
native floating-point division without Factor's denominator limit. Booleans
are not accepted as right-hand numeric operands. A `Factor` or
`SymbolicConstant` on either side of a unit instead produces a `ScaledUnit`.

```pycon
>>> metre = Unit('m')
>>> metre * 2 == 2 * metre
True
>>> (metre * 2).value
2
>>> double_metre = metre * Factor(2)
>>> print(double_metre)
2⋅m
>>> (3 * double_metre).to('m').value
6
>>> arcmin = si.deg / 60
>>> arcmin.value
Fraction(1, 60)
>>> arcmin.to(si.arcmin).value
Fraction(1, 1)

```

Multiplying a `ScaledUnit` by an ordinary number also constructs a quantity,
keeping the exact scale in its unit and preserving the supplied magnitude.
Combining a Unit with a Quantity continues to return a Quantity.

Quantities also accept `Factor` and `SymbolicConstant` operands on either side
of multiplication or division. Multiplication and right-hand division preserve
the magnitude and its type, combining the exact factor with the unit instead.
Left-hand division takes the reciprocal magnitude and unit. Ordinary numeric
operands continue to act on the magnitude: `q * 2` and `q * Factor(2)` represent
equal physical quantities but can have different `.value` and `.unit` pairs.

```pycon
>>> from unitx.factor import PI
>>> hbar = SI.h / (2 * PI)
>>> print(hbar)
6.62607015E-34 J⋅s/(2⋅π)
>>> hbar.value == SI.h.value
True
>>> hbar * (2 * PI) == SI.h
True
>>> round(float(hbar.to('J*s').value) / 1e-34, 8)
1.05457182

```

The local `hbar` above retains the exact scale, whereas predefined `SI.hbar`
provides its numerical value in ordinary `J*s` units. `.value` is always the
magnitude paired with `.unit`, including its exact scale. Use
`.to('J*s').value` to evaluate the local expression in joule seconds;
conversion involving an irrational factor evaluates it approximately.
String output combines the scale and formal units into one product, with
grouped denominator factors and multiplication dots between adjacent numbers.

## Conversion and arithmetic

Use `.to()` to obtain a new quantity in a unit with the same physical dimensions,
or `.value_in(unit)` to obtain only its numerical value. The target unit is
required and accepts a string or `Unit`, including `ScaledUnit`.
`.value_in(unit)` returns the same value and concrete type as `.to(unit).value`,
without constructing a temporary quantity. Both follow the conversion rules
below. A conversion with scale one preserves the original magnitude; Decimal
magnitudes stay Decimal and other conversions can change the numerical type.
`.value` reads the original stored magnitude.
Addition and subtraction convert the right operand to the left operand's unit:

```pycon
>>> length = Quantity(1, 'm')
>>> print(length + Quantity(50, 'cm'))
3/2 m
>>> length.value_in('cm')
100
>>> Quantity(1, 'cm').value_in('m')
Fraction(1, 100)

```

Multiplication and division combine units. Powers apply to both the magnitude
and the unit. Multiplication or division by a `Unit` alone preserves the
magnitude unchanged.

Conversion, addition, subtraction, and ordering across incompatible dimensions
raise `unitx.exceptions.DimensionError`. Equality across incompatible dimensions
returns `False`.

For repeated conversions, reuse a target `Unit` object to avoid parsing the
expression on every call.

## Unit expressions

Expressions support registered symbols, singular English names, SI prefixes,
multiplication, division, parentheses, and rational powers:

```pycon
>>> Unit('kg m/s2') == Unit('kg*m/s**2')
True
>>> Unit('m^(1/2)') ** 2 == Unit('m')
True
>>> Unit('micrometre') == Unit('um')
True
>>> Unit('ms') == Unit('m s')
False

```

- Multiplication can use `*` or whitespace. Adjacent letters form one unit
  token: `ms` is a millisecond, while `m s` is metre times second.
- Powers can use adjacent numbers, `^`, or `**`. Exponents written as ratios
  require parentheses, as in `m^(1/2)`; decimal exponents such as `m^0.5` are exact.
  Adjacent integer superscripts are also accepted: `m²`, `s⁻¹²`, and `(m/s)²`.
  Superscript fractions and mixed exponent notations are not supported.
- Multiplication and division associate left to right. Use `m/(s A)` to put
  both `s` and `A` in the denominator; `m/s A` means `(m/s)*A`.
- Unit symbols and names are case-sensitive. `u`, `µ`, and `μ` are accepted
  micro prefixes. Use lowercase `k` for kilo; `K` denotes kelvin.
- A registered multiword or hyphenated name is recognized before implicit multiplication
  when it occupies the entire input. Whitespace between its words is treated
  as a single space. Inside expressions, use `[name]` to make a name one factor,
  for example `[astronomical unit] / s` or `[volt-ampere reactive]**2`.
  Symbols and `Unit` object composition remain available.

Square brackets reference registered names and name aliases only: `[metre]`
works, while `[m]` and `[m/s]` do not. The contents are never evaluated as an
expression. Empty, nested, and unmatched brackets are errors. Put permitted
name prefixes inside the brackets, as in `[kiloastronomical unit]`; `k[name]`
does not apply a prefix. Single and double quotes retain their angle-unit
meaning rather than acting as string delimiters. Bare compound names are
not guessed from adjacent words, and ordinary parentheses still group products.

The unit table contains `s`, `m`, `g`, `A`, `K`, `mol`, and `cd`, plus the
dimensionless identity. Importing `unitx` also initializes the common units in
`unitx.si`, including named derived units such as `N`. Mass scales are referenced
to kilograms: `g` has factor `1/1000`, and `kg` has factor `1`.

## Defining units

Call `.define()` on a `Unit`, `ScaledUnit`, or positive real `Quantity` to
register a new unit and return it:

```pycon
>>> newton = Unit('kg*m/s2').define('demoN', 'demo newton')
>>> minute = (60 * Unit('s')).define('demoMin', 'demo minute', prefixable=False)
>>> Quantity(2, minute).to('s').value
120
>>> from unitx.factor import PI
>>> degree = ((PI / 180) * Unit()).define('demoDeg', name='demo degree', prefixable=False)
>>> degree.factor == PI / 180
True
>>> Quantity(1, 'kN').to(newton).value
1000

```

Multiplying or dividing a unit by a `Factor` or `SymbolicConstant` creates a
`ScaledUnit`, which preserves an exact additional
scale through unit arithmetic. Its `scale`
is relative to its formal unit product; its `factor` is the total scale to the
coherent SI unit. An ordinary `Unit` has an implicit additional scale of one;
only `ScaledUnit` stores that scale. Unit multiplication, division, and powers
involving a `ScaledUnit` keep that type even when the scale cancels to one.
A scale of one compares equal to the corresponding `Unit` and has the same
hash. Calling `.define()` returns an ordinary `Unit` with its own formal
identity, sharing the definition's dimension and exact SI factor.

`ScaledUnit` supports shallow copying, deep copying, and pickling while
preserving its exact scale and formal units. Restoring custom units requires
their definitions to be registered in the receiving process.

Definitions affect the current process and leave the source object unchanged.
The symbol is positional-only. The primary `name` accepts positional or keyword
input. Additional name aliases are optional positional arguments after `name`,
such as `Unit('m').define('span', 'spanunit', 'spanlength', 'spanmeasure')`.
Pass the primary name positionally when supplying name aliases; omitting the
aliases gives an empty tuple. `symbol_aliases` accepts a string or tuple of
strings; it and `prefixable` must be passed by keyword.
Symbols and symbol aliases must each be a single printable unit token.
Names and name aliases may contain multiple such tokens separated by single
ASCII spaces or hyphens, without leading or trailing separators. Decimal digits
and parser operators are not allowed in any token. The primary symbol and `name` are
required and must be nonempty; empty aliases are ignored. Square brackets and
the superscript characters `⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻` are reserved for parser syntax.

```pycon
>>> span = (2 * si.m).define('span', 'survey span', 'survey distance')
>>> Unit('survey span') == span
True
>>> Unit('kilosurvey distance') == Unit('kspan')
True
>>> Unit('survey span') / si.s == Unit('span / s')
True
>>> Unit('[survey span] / [second]') == Unit('span / s')
True

```

Bare multiword names are recognized only as complete inputs. For example,
`Unit('astronomical unit / arcsec')` does not recognize `astronomical unit`
as a factor; use `[astronomical unit] / arcsec`, symbols, or Unit objects instead.

SI prefixes are enabled by default. `prefixable: PrefixPolicy = True` accepts
`True` for all spellings, `False` for none, a single spelling string, or a tuple
of exact symbols, names, and aliases supplied in that definition. A string such
as `prefixable='Da'` is equivalent to `prefixable=('Da',)`. Only the selected
spellings accept prefixes; an empty tuple disables prefixes. Unknown spellings
and non-string entries are errors. For example,
`prefixable=('arcsec', 'arcsecond')` enables all SI prefixes on those two
spellings, while other spellings of that unit remain unprefixable.

The tuple selects **unit spellings**, not prefix symbols. Prefix names attach
directly to the first word of a multiword name, as in `kiloastronomical unit`;
a separate `kilo astronomical unit` is not accepted. `PrefixPolicy` is available
from `unitx.unitregistry` for annotations and means `bool | str | tuple[str, ...]`.

```pycon
>>> tick = si.s.define(
...     'demoTick',
...     'survey interval',
...     symbol_aliases='demoT',
...     prefixable=('demoT', 'survey interval'),
... )
>>> Unit('[millisurvey interval]') == Unit('mdemoT')
True
>>> Quantity(1000, 'mdemoT').to(tick).value
Fraction(1, 1)
>>> Unit('mdemoT').expression()
'mdemoT'

```

Prefix permissions are checked before alias normalization. Canonical identity
still uses the original registered unit and canonical prefix, independent of
the input alias. Preferred spellings use the first permitted symbol or alias;
if only a name supports a prefix, `SingleUnit.symbol` falls back to that name
and `Unit.expression()` emits a `[name]` reference. Conversely, `SingleUnit.name`
falls back to a symbol if no name allows that prefix. Repr and pickle preserve
the same formal unit, rather than replacing the prefix with an extra scale.

`arcsec` and `arcsecond` accept prefixes, for example `marcsec` and
`milliarcsecond`; `″` and `"` do not. `as` continues to mean attosecond,
and `mas` is not an arcsecond alias. `a` (are) and `ha` (hectare) are separate
unprefixable units; `Pa` remains pascal. `yr` remains exactly 365 days.
`a_j` is the Julian year of exactly 365.25 days (31,557,600 seconds), while
`mo` is one twelfth of that interval, rather than a variable calendar month.
The name exceptions `megohm` and `kilohm` are supported alongside the older
`megaohm` and `kiloohm` spellings; `gigaohm` retains both vowels. No general
vowel-elision rule is applied.

`ly` is defined using the speed of light and the Julian year; `eV` uses the
elementary charge and volt. Both reuse the defining constants exposed by `SI`.
`mHg` is the registered metre-of-mercury unit and `mmHg` is its milli-prefixed
form; `760 * si.mmHg` converts to `si.atm` with value 1 under the library's
existing pressure convention. Prefixes do not stack: `mmmHg` and `kmmHg` are
invalid. `kt` retains its meaning as a knot alias; tonne remains unprefixable.

`si.mas` and `si.uas` provide milli- and microarcseconds as namespace shortcuts;
their registered expressions are `marcsec` and `µarcsec`. The strings `mas`
and `uas` are not registered aliases. `si.Da` and `si.u` now share one formal
unit identity; only the spelling `Da` accepts prefixes, such as `kDa`.
`si.mTorr` is available, and Torr accepts SI prefixes. Tonne uses `t` or
`tonne`; the ambiguous spelling `ton` is not registered.

`si.rpm` represents rotational frequency: `60 * si.rpm` converts to `si.Hz`
with value 1. For angular velocity, multiply that frequency by `2 * PI` and
convert to `si.rad / si.s`; directly converting frequency to `rad/s` does
not insert the physical factor of 2π. `si.c` is a speed unit equal to the
speed of light, while `SI.c` is the corresponding physical quantity.

The spelling `u` denotes the unified atomic mass unit when used on its own.
ASCII micro spellings such as `um` and `us` remain supported without warnings,
so entering non-ASCII characters is optional. `um`, `µm`, `μm`, and
`micrometre` all mean micrometres; the explicit product `u m` instead means
unified atomic mass unit times metre.

Registration raises `UnitRegistryError` if a new spelling, including a generated
prefixed form, already resolves to a unit or conflicts with another spelling
in the same definition. A multiword spelling must not replace an existing
implicit product such as `metre second`. Validation finishes before the
registry changes.

The registry and spelling rules live in `unitx.unitregistry`; parsing expressions
does not change unit definitions. A parser instance caches only a successful
result; failed parses may be retried after a definition is added. Registration
preserves existing valid spellings and validates before updating its indexes.
Concurrent registry writes and isolated registry contexts are not supported.

Quantity definitions require positive finite `int`, `Fraction`, `Decimal`, or
`float` magnitudes. Prefer `Fraction` or `Decimal` for exact scales. Floats use
Factor's denominator limit and can warn or round to zero and fail, for example
with `1e-10`. Defining a unit does not evaluate symbolic scales numerically.

## Units, dimensions, and equality

A `Unit` retains its formal factors. A `Quantity` compares physical values
across compatible units:

```pycon
>>> Unit('m') == Unit('cm')
False
>>> Quantity(1, 'm') == Quantity(100, 'cm')
True
>>> len({Quantity(1, 'm'), Quantity(100, 'cm')})
1
>>> from unitx.dimension import Dimension
>>> speed.dimension == Dimension(L=1, T=-1)
True

```

Equal quantities have equal hashes when their magnitudes are hashable. Different
formal unit expressions can have the same physical dimension and scale without
being equal as `Unit` objects.

An expression can be dimensionless and still carry a scale. Plain numbers added
to or subtracted from dimensionless quantities represent unscaled values:

```pycon
>>> scaled = Quantity(1, 'km/m')
>>> scaled.is_dimensionless()
True
>>> scaled == 1000
True
>>> (scaled + 1).to('').value
Fraction(1001, 1)

```

## Numerical behavior

Exact unit algebra and numerical magnitude evaluation have different rules:

- Formal exponents use one rule in Monomial, Dimension, Factor, Unit, and
  Quantity: int/Fraction/Decimal values stay exact; floats use their shortest
  decimal text without a denominator limit. Thus `0.1` means `1/10` and
  `1e-10` remains nonzero. Use `Fraction(1, 3)` for exactly one third:
  the float `1 / 3` means the decimal `0.3333333333333333` in formal exponents.
  Factor's numeric scale inputs retain their separate denominator-limit rules.
- Quantity handles units; the magnitudes determine numerical arithmetic.
  Operators use the magnitudes in operand order and preserve native promotion,
  rounding, signals, and exceptions. Decimal arithmetic returns Decimal and
  follows its active context. Mixing Decimal with float or Fraction raises
  `TypeError`, as it does for the bare numbers; choose a common numerical type
  explicitly when needed.
- Conversion with a rational scale keeps int/Fraction values exact. Decimal
  conversion combines the value and scale exactly internally, then rounds once
  to Decimal in the active context. Irrational unit scales are approximated as
  floats, using their shortest decimal representations for Decimal conversion.
  A scale of one preserves the magnitude without arithmetic or rounding.
  Direct `Factor(value)` retains its separate exact Fraction evaluation rules.
- Addition and subtraction align the right operand with the left quantity's
  unit before numerical arithmetic. Integers being aligned with a Decimal
  magnitude are scaled in the Decimal domain, so `Decimal('1') * si.m + 50 *
  si.cm` returns a Decimal magnitude of `1.5`. Explicit Fraction/float operands
  keep their own numerical domains. Finite-precision conversion and arithmetic
  can round; changing the result unit can therefore change rounding errors.
- Unary `+` and `-` apply the magnitude's native operators and preserve the unit.
  Decimal results remain Decimal and follow its context precision, rounding,
  signals, and exceptions; unary `+` can therefore change the numerical value.
- Integer division produces a float; Fraction arithmetic remains rational.
  Use Fraction magnitudes explicitly for exact rational results. Powers pass
  the original exponent to the magnitude: `2 ** 2.0` returns float and
  `2 ** Decimal('0.5')` returns Decimal. Decimal magnitudes require int or
  Decimal exponents, as in Python. The unit follows the common formal exponent
  rule above; numerical evaluation still uses the original exponent object.
  Exponents must be finite real scalars other than bool. Numerical powers can
  round or produce complex values, following the magnitude's rules.
- Truth values follow the magnitude, so zero quantities are false regardless
  of unit. Comparisons with equal unit factors use native magnitude operators,
  including independent `!=`, Decimal signals/exceptions and complex ordering
  errors. For different unit factors, builtin real values and complex
  components are aligned exactly to avoid rounding-dependent equality or
  hashes. Ordinary numeric subclasses use the same alignment as their base
  types. Decimal/float signals apply to native comparisons with equal factors;
  cross-factor alignment compares exact rational coefficients and does not
  perform an additional comparison merely to reproduce signals. Decimal/Fraction
  comparisons are supported natively.
- Equality is exact, without a tolerance. Finite floats are compared as their
  stored binary rational values. Floating-point conversions and irrational
  powers may round; use `isclose()` for approximate real scalar comparisons.
  It uses the symmetric condition
  `abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)` in coherent SI
  units, with defaults `rel_tol=1e-9` and `abs_tol=0`. Swapping quantities
  preserves the result. Nonzero absolute tolerances for dimensional quantities
  must carry matching units; plain operands and plain absolute tolerances
  represent unscaled dimensionless values. Tolerances must be finite and
  nonnegative. Different dimensions raise `DimensionError`. NaNs are never
  close, and infinities are close only to infinities of the same sign.
  Finite rational coefficients stay exact; residual irrational unit factors
  are approximated. With both tolerances zero, `isclose()` uses exact equality.

```pycon
>>> from decimal import Decimal, localcontext
>>> from fractions import Fraction
>>> with localcontext() as context:
...     context.prec = 2
...     q = Quantity(Decimal('1.2345'), 'm')
...     print((+q).value, (-q).value, (q * 2).value)
1.2 -1.2 2.5
>>> Quantity(Decimal('0.1'), 'm').to('cm').value
Decimal('10')
>>> (Quantity(Fraction(1), 'm') / Quantity(3, 's')).value
Fraction(1, 3)
>>> (Quantity(2, 'm') ** 2.0).value
4.0
>>> bool(Quantity(0, 'm'))
False
>>> Quantity(0.1, '') == Quantity(Fraction(1, 10), '')
False
>>> Quantity(0.1 + 0.2, 'm').isclose(Quantity(0.3, 'm'))
True
>>> a, b = Quantity(1, 'm'), Quantity(100.05, 'cm')
>>> a.isclose(b, rel_tol=0, abs_tol=Quantity(1, 'mm'))
True
>>> b.isclose(a, rel_tol=0, abs_tol=Quantity(1, 'mm'))
True

```

The built-in integer factorization backend is bounded. Unresolved factors emit
`IncompleteFactorWarning` and can limit simplification and comparisons. SymPy
is an optional, more capable factorization backend:

```sh
python -m pip install sympy
```

## Current scope

- Optional NumPy support accepts plain numeric ndarrays and NumPy numeric
  scalars. Install `unitx[numpy]`; see the [support matrix and numerical
  contract](docs/numpy.md). Array operations preserve units and comparisons
  return masks. Single-output ufuncs accept unit-aware `out` buffers; implicit
  unit stripping is rejected.
- Custom magnitudes must support scalar arithmetic, return a Python `bool`
  from equality, and defer raw operations with `Unit` operands to its reflected
  operators.
- Scalar operations use the magnitude's supported operand types, including
  complex numbers and custom scalar magnitudes. Custom magnitudes must also
  defer operations with Quantity operands so that Quantity can handle units.
  Ordering follows the magnitude's native rules.
- Unit conversion is multiplicative. Offset temperature scales such as Celsius
  and Fahrenheit are not supported.

## Building blocks

| Class | Import module | Purpose |
| --- | --- | --- |
| `si` / `SI` | `unitx.si` | Ready-to-use units and physical constants |
| `Quantity` | `unitx.quantity` | Scalar or NumPy magnitudes paired with units, conversion, and arithmetic |
| `Unit` | `unitx.unit` | Compound units, scaling factors, and physical dimensions |
| `ScaledUnit` | `unitx.unit` | Formal unit products with an additional exact scale |
| `Dimension` | `unitx.dimension` | Seven physical dimensions with rational exponents |
| `SingleUnit` | `unitx.singleunit` | One registered unit with an optional prefix |
| `Factor` | `unitx.factor` | Positive scaling expressions and magnitude scaling |
| `UnitParser` | `unitx.unitparser` | Parsing unit expressions into formal products |
| `Monomial` | `unitx.monomial` | Immutable formal products with rational exponents |

The package includes inline type annotations and a `py.typed` marker. Construction
and operations with `Unit` preserve the magnitude type in type hints; conversion
and numerical arithmetic use result unions where the concrete type depends on
runtime values or units. Modules whose names start with `_` are internal.

## Upgrading from 0.3.0

Version 0.4.0 replaced the implementation previously published on PyPI as 0.3.0.
It introduced incompatible changes:

- Python 3.12 or later is required; Python 3.11 is no longer supported.
- The former `unitx.utils` namespace was removed. Classes can be imported from
  their modules, as shown above; the current source tree also exports `Unit`,
  `Quantity`, and `Dimension` at the package root.
- `Unit` no longer defines `len()` or special truth-value behavior. Use
  `unit == Unit('')` to check the formal identity, or `unit.is_dimensionless()`
  to check physical dimensions.
- Conversion and arithmetic can produce different magnitude types, especially
  for Decimal inputs. Review the [numerical behavior](#numerical-behavior)
  before relying on a result's concrete type or approximate equality.

## Development

Install the development version and test dependencies:

```sh
python -m pip install -e ".[test]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The pytest suite includes doctests and Pyright checks for the typed API. CI runs
the suite with both the built-in and SymPy factorization backends.

## Releases

The [release workflow](.github/workflows/release.yml) validates the source,
builds a wheel and source distribution, checks their metadata, and tests the
installed wheel. Manual workflow runs store the distributions without uploading
them to PyPI.

Publishing a GitHub release runs the same checks and uploads the distributions
to PyPI. The release tag must match the project version, for example `v0.4.0`
for version `0.4.0`. Configure a PyPI Trusted Publisher for the repository,
workflow `release.yml`, and GitHub environment `pypi` before publishing.

See [CHANGELOG.md](CHANGELOG.md) for release history. Licensed under the
[MIT License](LICENSE).
