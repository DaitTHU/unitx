# Unit catalogues

Use the SI catalogue directly in formulas; no Registry setup is needed:

```pycon
>>> from unitx import Unit, UnitCatalog
>>> from unitx.si import si
>>> si.parse('metre') == si.m == Unit('m')
True

```

See the [SI reference](si.md) for units, constants, prefixes and Unicode spellings.
`si` is a read-only UnitCatalog namespace using the default parsing environment.
Custom catalogues collect their public unit attributes after the class body
finishes and provide their own `parse()` entry point. Attribute names do not
become parser aliases. Ordinary strings keep using the default environment.

A catalogue selects available units and parser spellings. It does not select
coherent base units or expand quantities in a chosen unit system.
See [custom units and registries](../custom-units.md#defining-a-catalogue)
for construction, explicit definition blocks and reconstruction boundaries.

[Documentation index](../README.md)
