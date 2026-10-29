# Unit catalogues

Use catalogues directly in formulas; no Registry setup is needed:

```pycon
>>> from unitx import Quantity, Unit
>>> from unitx.si import si
>>> from unitx.imperial import imperial
>>> (3 * imperial.ft).value_in(si.m)
Fraction(1143, 1250)
>>> imperial.unit('ft/s') == imperial.ft / si.s
True

```

| Catalogue | Import | Reference |
| --- | --- | --- |
| SI and common scientific units | `from unitx.si import si, SI` | [Units, constants, prefixes and Unicode spellings](si.md) |
| Imperial, historical English and US units | `from unitx.imperial import imperial, imperial_historical, us` | [Length, mass, volume, force and pressure conventions](imperial.md) |
| Information | `from unitx.information import information` | [Bits, bytes, nats, hartleys and binary prefixes](information.md) |
| CGS electromagnetism | `from unitx.cgs import cgs, gaussian, esu, emu, hl` | [Mechanical definitions and explicit SI correspondence](cgs.md) |

## The UnitCatalog entry point

`si`, `imperial`, `information`, `cgs`, `gaussian`, `esu`, `emu` and `hl` share the public `UnitCatalog` base.
A catalogue exposes unit attributes and a `unit()` entry point, using its
configured spelling environment. `si.unit()` uses the existing default
environment; Imperial and CGS parsing use independent environments.
The base preserves the existing non-instantiable, read-only namespace behavior.
UnitCatalog collects public unit attributes after the class body finishes.
Existing objects can also be shared with `units=`. Use earlier unit objects to
construct later definitions; new spellings become parseable after collection.
See [catalogue construction](../custom-units.md#defining-a-catalogue) for the
prefix permissions, structured persistence locations and optional class keywords.

A catalogue selects available units and parser spellings. It does not select
coherent base units or expand quantities in a chosen unit system.

## Default text stays fixed

`Quantity(3, 'm')` and `.to('cm')` keep using the default environment. Importing
a catalogue never adds its spellings to that environment. For example, default
`gal` denotes galileo (acceleration), while Imperial `gal` denotes gallon:

```pycon
>>> Unit('gal') == si.Gal
True
>>> imperial.unit('gal') == imperial.gal
True
>>> (1 * imperial.gal).value_in(si.L)
Fraction(454609, 100000)

```

Pass the catalogue's Unit object to `.to()` or `.value_in()`. A source quantity's
dimension never changes the meaning of a target string. Unit objects carry
bound immutable definitions; quantities do not retain a mutable registry.
Compatible quantities from different environments can be added or converted.

For custom units in ordinary code, use `with registry.defining():`; class
declarations remain available to catalogue maintainers. See
[custom units and registries](../custom-units.md#independent-registries).
For persisted identities, see [serialization](../serialization.md).

The entry-point design draws on Astropy's separation of
[directly usable unit objects and enabled parser names](https://docs.astropy.org/en/stable/units/standard_units.html#enabling-other-units).
Unlike [Pint's registry ownership rule](https://pint.readthedocs.io/en/stable/getting/pint-in-your-projects.html),
unitx allows compatible quantities from different environments to interact.
It retains unitx's own exact formal identity and numeric rules.

[Documentation index](../README.md) · [Units and quantities](../units.md)
