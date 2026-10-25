# Migration notes

## Upgrading to 0.15.1

Catalogue lookup is now `catalogue.unit(expression)`. Replace calls such as
`si.parse('m/s')`, `imperial.parse('ft/s')` and `gaussian.parse('statC')` with
`si.unit('m/s')`, `imperial.unit('ft/s')` and `gaussian.unit('statC')`.
The old catalogue `parse()` method is removed. The input remains a string,
resolved in the catalogue's own environment, and the result is a linear Unit.
`Registry.parse()` and `UnitParser.parse()` retain their names and behavior.

## Upgrading to 0.14.0

### Coordinate families and algebraic dimensions

- Use `LogUnit(reference, delta_unit=..., kind=...)` for logarithmic coordinates;
  AffineUnit now defines linear coordinates only and has no `kind` argument.
- `si.dBm`, `si.dBW` and `si.dBV` are LogUnit descriptors. `value @ descriptor`
  produces LogQuantity; use `LogQuantity(value, 'dBm')` for explicit construction.
- `.to()` and `.value_in()` stay within the point's coordinate family. Replace
  logarithmic `.to('W')` with `.to_quantity('W')`, and physical-point
  `.to('dBm')` with `LogQuantity.from_quantity(physical_quantity, 'dBm')`.
- Complete algebraic compatibility is `left.dimension == right.dimension`.
  Use `.dimension.physical` for seven-axis physical comparisons. Physical
  dimensionless tests and physical exponent views continue to ignore the internal
  logarithmic axis. Independent public `log_degree` properties are removed.
- Scalar JSON format version 2 retains its derived `log_degree` wire field and
  adds `log_unit`/`log_quantity` tags. Coordinate family and mapping are validated
  together; logarithmic data tagged as an affine coordinate must be migrated.
- LogUnit preserves its defining `reference` and `at` rather than storing an
  evaluated zero-reading reference. Its `origin` property is a numerical view.
  Anonymous logarithmic identity compares the exact anchor, difference Unit and
  convention; equivalent readings can still compare equal across coordinates.
  Logarithmic JSON records use `reference` and `at`. Regenerate development
  records that used the earlier normalized `origin` schema from their original
  definitions; an approximate origin cannot recover an exact defining anchor.


### Catalogue isolation and definition identity

The unit value is now represented by `Unit` itself, including its exact `scale`.
Use `Unit(base, scale=factor)` or combine units with `Factor`/`SymbolicConstant`;
the separate scaled-unit class and its constructor are removed. Named coordinate
references and difference units also preserve extra scales.

Import `UnitDefinition` and `PrefixPolicy` from `unitx.unitdefinition`. Definition
construction validates its own metadata before it can be bound. Module-level
registry table views and spelling resolver functions are removed; applications
use `Registry.parse()` and `Registry.include(*units)` instead. Each include call
validates the whole batch before changing the environment.

Use `from unitx.imperial import imperial` for the ready-to-use Imperial
catalogue, following the existing `from unitx.si import si` structure.
Its `unit()` uses the catalogue environment. Default `Quantity(..., 'm')` and
`.to('cm')` retain their existing behavior; extra catalogue imports do not
change default spellings.

`.define()` no longer registers into the default environment implicitly.
Outside a `with registry.defining():` block,
Unit, Quantity and AffineUnit `.define()` calls
create standalone named objects. Their arithmetic and conversions work
immediately. To make strings parse, explicitly include the object in a Registry,
define related units inside `with registry.defining():`, or use
`registry.define(source, symbol, name)` for one linear unit. Applications
that intentionally extend the default table can import `default_registry` from
`unitx.unitregistry` and call its `define()` or `include()` method.

Definition blocks restore their enclosing selection even after an exception.
Nested blocks use the innermost explicit environment. Blocks follow ContextVar
semantics for asyncio task inheritance; they do not synchronize shared registries
or change default string parsing.

Catalogue classes collect their public Unit and named AffineUnit attributes only
after the body finishes. Class keywords `units=` and `registry=` select additional
shared definitions or an explicit receiving environment. Helpers return ordinary
objects; their unexported temporary definitions are not collected. Replace
`Unit('newSpelling')` within a class body with the earlier unit object. An enclosing
explicit definition block continues to apply normally during class execution.
Module-level declarations provide structured persistence locations. Identity,
conversion data and prefix policy are final before `.define()` returns. A failed
collection leaves the receiving spelling indexes unchanged. Named affine
coordinates retain bound reference and difference definitions and can use private
units without default registration.

Formal identity now uses shared definition objects rather than symbols.
Independent blocks and local catalogue declarations create independent units;
use `include()` to share existing definitions. Import the same module-level
catalogue before restoring its records in another process. Definition IDs are
internal persistence details; public `identity` and `identity_namespace` options
are removed. Compatible cross-environment quantities still interact normally.

The name defaults to the symbol, allowing `.define('twip')`.
`Registry(prefixable=False)` supplies the default in explicit definition blocks
and `registry.define()`; individual policies override it. UnitCatalog does not
supply construction defaults. Imperial, historical English and US definitions
use explicit `Registry(prefixable=False).defining()` blocks to disable prefixes
from construction onward. Fahrenheit and Rankine are accessed as
`imperial.degF` and `imperial.degR` instead of SI attributes or default strings.

JSON records advance to version 2; version 1 is explicitly rejected because it
cannot distinguish same-name private identities. Pickle records use the current
definition IDs and metadata shape; older pickle payloads are not supported. See
[serialization migration](serialization.md#migrating-version-1-records) and
[catalogue reconstruction](custom-units.md#reconstruction-and-exchange).


## Upgrading to 0.13.2: clock readings are points

The five `unitx.time` clock functions now return `AffineQuantity[int]` instead
of `Quantity[int]`, still storing integer nanoseconds. Subtract two readings
from the same clock to get a duration Quantity. Adding two readings and
multiplying a reading are rejected. `.unit` is now an AffineUnit;
`.unit.delta_unit` gives the linear ns unit. Use `.value_in('s')` for a numeric
timestamp in seconds. `sleep()`, `from_timedelta()` and `to_timedelta()` keep
their duration contracts. `from_datetime()` imports an aware datetime as an
integer ns Unix point; `to_datetime()` exports a Unix-referenced time point as
an aware datetime (UTC by default). Existing AffineUnit references support
custom known epochs. Naive datetimes are rejected, and export rounds to
microseconds with ties to even. Clock identity and epoch labels remain outside
the model, so mixing different clocks is still not detected.

## Upgrading to 0.13.0

The release adds `unitx.math` and `unitx.time` without changing ordinary
Quantity arithmetic. Mathematical wrappers always return Quantity, including
bare scalar calls. Their Python scalar backend uses float results; this is
not an arbitrary-precision math API. Clock functions return integer ns,
including `time()`; they do not mirror the standard library's float seconds.

The guides have been reorganized around the [documentation index](README.md).
The former math-time guide is now [standard-library enhancements](stdlib.md).
Unit registration, catalogue conventions and typing have their own guides.

## Upgrading from 0.11.0

- Import factor constants as `one`, `two`, `ten`, and `pi`; the former
  uppercase names are removed. `ln10` is available from `unitx.factor`.
- Unit and AffineUnit attributes are immutable. Build a new unit instead of
  assigning or deleting internal state.
- Use `value @ si.degC` or `AffineQuantity(value, 'degC')` for an absolute
  temperature reading. `value * si.degC` and ordinary Quantity construction
  represent temperature differences. The same point/difference distinction
  applies to logarithmic coordinates; see [affine coordinates](affine.md).
- `.to()` preserves the point class. Use `.to_quantity('K')`, `.to_quantity('W')`
  or `.to_quantity('V')` to explicitly obtain an ordinary physical quantity.
- The built-in atmosphere unit `atm` no longer accepts SI prefixes. Represent
  scaled magnitudes with an explicit Quantity, or register a separate unit.

## Upgrading from 0.8.x


Version 0.9 adds indexed writes and changes ndarray augmented arithmetic from
rebinding to updating the existing buffer. Aliases now observe `q += other` and
the other supported augmented operators. The array, dtype, shape, and unit stay
fixed, so integer `q /= 2` and dimensional `q **= 2` can raise where the old
rebinding behavior succeeded. Python and NumPy scalar arithmetic is unchanged.

To retain the previous allocation and rebinding behavior, write `q = q + other`,
`q = q / 2`, or `q = q ** 2` explicitly. Use ordinary multiplication and rebinding
when changing units or dimensions. Indexed assignment uses `same_kind` casting;
it does not inherit ndarray assignment's implicit float-to-integer truncation.
The supported array domain and explicit unit-removal rules are unchanged.


## Upgrading from 0.3.0

Version 0.4.0 replaced the implementation previously published on PyPI as 0.3.0.
It introduced incompatible changes:

- Python 3.12 or later is required; Python 3.11 is no longer supported.
- The former `unitx.utils` namespace was removed. Classes can be imported from
  their modules, with the current documented imports; the current source tree also exports `Unit`,
  `Quantity`, and `Dimension` at the package root.
- `Unit` no longer defines `len()` or special truth-value behavior. Use
  `unit == Unit('')` to check the formal identity, or `unit.is_dimensionless()`
  to check physical dimensions.
- Conversion and arithmetic can produce different magnitude types, especially
  for Decimal inputs. Review the [scalar numerical behavior](scalars.md)
  before relying on a result's concrete type or approximate equality.


[Documentation index](README.md)
