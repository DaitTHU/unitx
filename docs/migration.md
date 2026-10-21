# Migration notes

## Upgrading from 0.11.0

- Import factor constants as `one`, `two`, `ten`, and `pi`; the former
  uppercase names are removed. `ln10` is available from `unitx.factor`.
- Unit and ScaledUnit attributes are immutable. Build a new unit instead of
  assigning or deleting internal state.
- Use `value @ si.degC` or `AffineQuantity(value, 'degC')` for an absolute
  temperature reading. `value * si.degC` and ordinary Quantity construction
  represent temperature differences. The same point/difference distinction
  applies to logarithmic coordinates; see [affine coordinates](affine.md).
- `.to()` preserves the point class. Use `.to_quantity('K')`, `.to_quantity('W')`
  or `.to_quantity('V')` to explicitly obtain an ordinary physical quantity.
- The built-in atmosphere unit `atm` no longer accepts SI prefixes. Represent
  scaled magnitudes with an explicit Quantity, or register a separate unit.

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
  for Decimal inputs. Review the [scalar numerical behavior](scalars.md)
  before relying on a result's concrete type or approximate equality.

