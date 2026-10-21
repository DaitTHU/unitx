# Migration notes

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
