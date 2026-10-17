# unitx

Physical units, scalar quantities, and dimensional analysis for Python, with
exact rational unit scales and exponents.

unitx is in early development. Public APIs may change during the 0.x series.

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

## Upgrading from 0.3.0

Version 0.4.0 replaces the implementation previously published on PyPI as 0.3.0.
It includes incompatible changes:

- Python 3.12 or later is required; Python 3.11 is no longer supported.
- Import classes from their modules. The former top-level exports and
  `unitx.utils` namespace are no longer provided:

  ```python
  from unitx.dimension import DIMENSIONLESS, Dimension
  from unitx.quantity import Quantity
  from unitx.unit import Unit
  ```

- `Unit` no longer defines `len()` or special truth-value behavior. Use
  `unit == Unit('')` to check the formal identity, or `unit.is_dimensionless()`
  to check physical dimensions; these are different questions.
- Conversion and arithmetic can produce different magnitude types, especially
  for Decimal inputs. Review the [numerical behavior](#numerical-behavior)
  before relying on a result's concrete type or approximate equality.

## Quick start

Import the classes from their modules:

```pycon
>>> from unitx.quantity import Quantity
>>> from unitx.unit import Unit
>>> distance = 3 * Unit('m')
>>> elapsed = Quantity(2, 's')
>>> speed = distance / elapsed
>>> print(speed)
1.5 m/s
>>> print(speed.to('cm/s'))
150.0 cm/s

```

`Quantity(value, unit)` accepts a unit expression or an existing `Unit` object.
Multiplying a scalar by a unit constructs a quantity; dividing a scalar by a
unit constructs a quantity with the reciprocal unit. Construction preserves the
original magnitude and its type.

These are directional construction operators: `value * unit` constructs a
`Quantity`, and `factor * unit` constructs a `ScaledUnit` when the factor is a
`Factor` or `SymbolicConstant`. The reverse forms `unit * value` and
`unit * factor` are unsupported and raise `TypeError`.

## Conversion and arithmetic

Use `.to()` to convert between units with the same physical dimensions.
Addition and subtraction convert the right operand to the left operand's unit:

```pycon
>>> length = Quantity(1, 'm')
>>> print(length + Quantity(50, 'cm'))
3/2 m
>>> length.to('cm').value
100
>>> Quantity(1, 'cm').to('m').value
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
- Multiplication and division associate left to right. Use `m/(s A)` to put
  both `s` and `A` in the denominator; `m/s A` means `(m/s)*A`.
- Unit symbols and names are case-sensitive. `u`, `µ`, and `μ` are accepted
  micro prefixes. Use lowercase `k` for kilo; `K` denotes kelvin.

The current unit table contains `s`, `m`, `g`, `A`, `K`, `mol`, and `cd`, plus
the dimensionless identity. Mass scales are referenced to kilograms: `g` has
factor `1/1000`, and `kg` has factor `1`. Named derived units such as `N` are
not pre-registered; use expressions such as `kg*m/s2` or define them explicitly.

## Defining units

Call `.define()` on a `Unit`, `ScaledUnit`, or positive real `Quantity` to
register a new unit and return it:

```pycon
>>> newton = Unit('kg*m/s2').define('N', name='newton')
>>> minute = (60 * Unit('s')).define('min', name='minute', prefixable=False)
>>> Quantity(2, minute).to('s').value
120
>>> from unitx.factor import PI
>>> degree = ((PI / 180) * Unit()).define('deg', name='degree', prefixable=False)
>>> degree.factor == PI / 180
True
>>> Quantity(1, 'kN').to(newton).value
1000

```

Multiplying a `Factor` or `SymbolicConstant` by a unit creates a `ScaledUnit`,
which preserves an exact additional scale through unit arithmetic. Its `scale`
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
The symbol, names, and aliases must each be a single printable unit token,
without whitespace, decimal digits, or parser operators. The primary symbol
and `name` are required and must be nonempty; empty aliases are ignored.

SI prefixes are enabled by default; set `prefixable=False` to disable them.
Registration raises `UnitRegistryError` if a new spelling, including a generated
prefixed form, already resolves to a unit or conflicts with another spelling
in the same definition. Validation finishes before the registry changes.

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

- Rational unit scales and exponents remain exact. Prefer `Fraction` or
  `Decimal` for exact exponents; float exponents may be approximated with an
  `InexactFloatWarning`.
- Conversion and arithmetic can change the magnitude's type. Finite `Decimal`
  operands are converted exactly to `Fraction`, without Decimal context
  rounding.
- Other magnitudes use native numerical arithmetic. In particular, integer
  division produces a float, and nonintegral magnitude powers can round or
  produce complex values.
- Equality is exact, without a tolerance. Finite floats are compared as their
  stored binary rational values. Floating-point conversions and irrational
  powers may round; compare their values approximately in a common unit when
  appropriate.

```pycon
>>> from decimal import Decimal
>>> from fractions import Fraction
>>> Quantity(Decimal('0.1'), 'm').to('cm').value
10
>>> (Quantity(Fraction(1), 'm') / Quantity(3, 's')).value
Fraction(1, 3)
>>> Quantity(0.1, '') == Quantity(Fraction(1, 10), '')
False

```

The built-in integer factorization backend is bounded. Unresolved factors emit
`IncompleteFactorWarning` and can limit simplification and comparisons. SymPy
is an optional, more capable factorization backend:

```sh
python -m pip install sympy
```

## Current scope

- **NumPy scalars, arrays, and NumPy operations are not supported yet. NumPy
  integration is planned.** Structural compatibility with a magnitude protocol
  does not imply NumPy support.
- Custom magnitudes must support scalar arithmetic, return a Python `bool`
  from equality, and defer raw operations with `Unit` operands to its reflected
  operators.
- Scalar multiplication and division of quantities currently accept `int`,
  `float`, `Fraction`, and `Decimal`. Complex magnitudes are supported, including
  results of fractional powers; ordering follows the magnitude's native rules.
- Unit conversion is multiplicative. Offset temperature scales such as Celsius
  and Fahrenheit are not supported.

## Building blocks

| Class | Import module | Purpose |
| --- | --- | --- |
| `Quantity` | `unitx.quantity` | Scalar values paired with units, conversion, and arithmetic |
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
