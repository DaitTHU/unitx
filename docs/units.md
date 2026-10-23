# Units and quantities

A Quantity pairs one numerical magnitude with one linear Unit. Start with the
`si` namespace, or construct from strings when reading units from data.

```pycon
>>> from unitx import Quantity, Unit
>>> from unitx.si import SI, si

```

## Constructing units and quantities

For unit expressions and explicit construction, import the classes:

```pycon
>>> from unitx import Quantity, Unit
>>> Quantity(50, 'cm') == 50 * si.cm
True
>>> Unit('astronomical unit') == si.au
True

```

`SingleUnit(text)` resolves one symbol or name without parsing an expression.
Use `SingleUnit.from_name(name)` to accept only registered names and name aliases,
including permitted prefixes. Names are matched exactly; `from_name('kilometre')`
accepts the name, while `from_name('km')` raises `UnitSymbolError`.
Import `SingleUnit` from `unitx.singleunit`.

Unit spellings are case-sensitive: `k` is the kilo prefix and `K` is kelvin.
`km` means kilometre; `K m` means kelvin times metre. The former uppercase
kilo alias `K` is no longer accepted, so unregistered spellings such as `Km`
and `Kg` raise `UnitSymbolError`. Micro accepts canonical `µ`, Unicode `μ`,
and keyboard-friendly ASCII `u` without warnings.

`Quantity(value, unit)` accepts a linear unit expression, an existing Unit, or
an AffineUnit or LogUnit descriptor projected to its linear side. Quantity.unit is always Unit.
`Unit(text)` always returns Unit, including `Unit('degC')` for Celsius differences.
Absolute readings use AffineQuantity for linear coordinates and LogQuantity for
logarithmic coordinates, or `value @ coordinate`; see
[affine and logarithmic coordinates](affine.md) for their separate point rules.
`value * unit` constructs a quantity, and `value / unit` constructs a quantity
with the reciprocal unit. Construction preserves the original magnitude and
its type.

Multiplication is symmetric: `unit * value` and `value * unit` both construct
`Quantity` objects and preserve the magnitude's type. Dividing a unit by an
`int`, `Fraction`, or finite `Decimal` constructs a quantity with an exact
`Fraction` reciprocal, so `degree / 60` loses no precision. Float divisors use
native floating-point division without Factor's denominator limit. Booleans
are not accepted as right-hand numeric operands. A `Factor` or
`SymbolicConstant` on either side of a unit instead produces a `Unit`.

```pycon
>>> from unitx.factor import Factor
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

Multiplying a scaled `Unit` by an ordinary number also constructs a quantity,
keeping the exact scale in its unit and preserving the supplied magnitude.
Combining a Unit with a Quantity continues to return a Quantity.

Quantities also accept `Factor` and `SymbolicConstant` operands on either side
of multiplication or division. Multiplication and right-hand division preserve
the magnitude and its type, combining the exact factor with the unit instead.
Left-hand division takes the reciprocal magnitude and unit. Ordinary numeric
operands continue to act on the magnitude: `q * 2` and `q * Factor(2)` represent
equal physical quantities but can have different `.value` and `.unit` pairs.

```pycon
>>> from unitx import pi
>>> hbar = SI.h / (2 * pi)
>>> print(hbar)
6.62607015E-34 J⋅s/(2⋅π)
>>> hbar.value == SI.h.value
True
>>> hbar * (2 * pi) == SI.h
True
>>> round(float(hbar.to('J*s').value) / 1e-34, 8)
1.05457182

```

The local `hbar` above retains the exact scale in its Unit, whereas predefined `SI.hbar`
provides its numerical value in ordinary `J*s` units. `.value` is always the
magnitude paired with `.unit`, including its exact scale. Use
`.to('J*s').value` to evaluate the local expression in joule seconds;
conversion involving an irrational factor evaluates it approximately.
String output combines the scale and formal units into one product, with
grouped denominator factors and multiplication dots between adjacent numbers.


## Conversion and arithmetic

Use `.to()` to obtain a new quantity in a unit with the same physical dimensions,
or `.value_in(unit)` to obtain only its numerical value. The target unit is
required and accepts a string or `Unit`, including a scale specified with
`Unit(text, scale=Factor(...))`.
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
>>> from unitx import Dimension
>>> speed = 3 * si.m / (2 * si.s)
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


## Defining units

See [custom units](custom-units.md) for extra exact scales, `.define()` signatures,
spelling-specific prefix policies, validation and process-local reconstruction.

## Unicode unit characters

See the [catalogue's Unicode table](catalogues/si.md#unicode-unit-characters) for
accepted aliases and the distinction between an alias and a compound symbol.

[Documentation index](README.md) · [Scalar numerical behavior](scalars.md) ·
[Points and differences](affine.md)


## Complete compatibility and physical dimensions

`unit.dimension` is the complete algebraic signature used by conversion,
arithmetic, comparison and hashing. `unit.dimension.physical` projects onto the
seven physical axes. Named exponents and Dimension tuple/dictionary/text views
also describe only those seven axes. `unit.is_dimensionless()` is a physical
question, so dB is physically dimensionless while remaining algebraically
incompatible with an ordinary ratio. See [logarithmic quantities](logarithmic.md).
