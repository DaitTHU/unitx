# Upgrading from 0.3.0

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

