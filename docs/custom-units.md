# Custom units

Define a multiplicative unit from an existing unit, exact scale or positive
finite scalar quantity. Absolute coordinate definitions have their own
[AffineUnit API](affine.md#defining-coordinates).

```pycon
>>> from unitx import Quantity, Unit
>>> from unitx.si import si

```

## Register a unit

Call `.define()` on a `Unit`, `ScaledUnit`, or positive real `Quantity` to
register a new unit and return it:

```pycon
>>> newton = Unit('kg*m/s2').define('demoN', 'demo newton')
>>> minute = (60 * Unit('s')).define('demoMin', 'demo minute', prefixable=False)
>>> Quantity(2, minute).to('s').value
120
>>> from unitx.factor import pi
>>> degree = ((pi / 180) * Unit()).define('demoDeg', name='demo degree', prefixable=False)
>>> degree.factor == pi / 180
True
>>> Quantity(1, 'kN').to(newton).value
1000

```

## Exact extra scales

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

## Definition signature and spellings

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

## Prefix permissions

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

## Validation and reconstruction

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


The [catalogue](catalogue.md#spelling-and-scale-conventions) explains the
built-in units' prefix exceptions and scale conventions.

[Documentation index](README.md) · [Units and quantities](units.md)
