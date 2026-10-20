# Units and quantities

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
>>> from unitx import PI
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

