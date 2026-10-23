# Imperial units

Use a catalogue directly in formulas. No registry setup is needed:

```pycon
>>> from unitx import Quantity, Unit
>>> from unitx.imperial import imperial
>>> from unitx.si import si, SI
>>> (3 * imperial.ft).value_in(si.m)
Fraction(1143, 1250)
>>> imperial.parse('ft/s') == imperial.ft / si.s
True
>>> SI.c.value_in(imperial.parse('ft/s'))
Fraction(374740572500, 381)

```

The submodule exports the immutable `imperial` unit namespace, following
`from unitx.si import si`. Its definitions live in `unitx/imperial/units.py`.
Physical constants remain in `SI`; convert them explicitly when needed.

## Imperial conventions

All scales below are exact. Decimal input in the definitions is not a
binary floating-point approximation.

| Attribute | Meaning |
| --- | --- |
| `inch` | International inch, 0.0254 m; parser symbol `in` |
| `ft`, `yd`, `mi` | International foot (12 inches), yard (3 feet), mile (1760 yards) |
| `lb`, `oz`, `st`, `ton` | Avoirdupois pound (0.45359237 kg), ounce (1/16 lb), stone (14 lb), long ton (2240 lb) |
| `gal`, `qt`, `pt`, `fl_oz` | Imperial gallon (4.54609 L), quart (1/4 gal), pint (1/8 gal), fluid ounce (1/160 gal) |
| `lbf`, `psi` | Pound-force (lb × 9.80665 m/s²), pound-force per square inch |

`imperial` uses British volume and long-ton conventions. `us` contains the
US-specific survey, liquid/dry volume and short-ton definitions;
`imperial_historical` contains stated historical English conventions. Import them
from `unitx.imperial`. Temperature intervals and coordinates are available as
`imperial.degR` and `imperial.degF`. All three catalogues disable prefixes for
new definitions; explicitly included SI units retain their prefix permissions.
The exact foot convention follows [NIST's international foot definition](https://www.nist.gov/pml/us-surveyfoot);
conversion conventions are also tabulated in
[NIST SP 811](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b8).

`imperial.parse()` accepts the catalogue's units and their names, plus explicitly
shared SI definitions: metre, second, gram, kelvin, newton, pascal, litre and
hour, including their permitted prefixes and aliases. Products, exact rational
powers, `[name]` references and parser precedence work as in `Unit(...)`.
It does not consult private registrations in the default table.

Use Unit objects or `imperial.parse()` results as conversion targets. A bare
string still uses the default environment. In particular, `Unit('gal')` denotes
SI galileo (acceleration); `imperial.parse('gal')` denotes Imperial gallon.
Physical constants remain available from `SI` and can be explicitly converted
to these units. This catalogue does not provide a second constant namespace.

## US and historical conventions

```pycon
>>> from unitx.imperial import imperial_historical, us
>>> (1 * us.us_survey_foot).value_in(si.m)
Fraction(1200, 3937)
>>> (1 * us.us_cable).value_in(si.m)
Fraction(27432, 125)
>>> (1 * us.us_gal).value_in(si.L)
Fraction(473176473, 125000000)
>>> us.parse('[inch of water]') == us.inH2O
True
>>> (1 * imperial_historical.ramsden_chain).value_in(imperial.ft)
100

```

The legacy survey foot is exactly 1200/3937 m; the survey mile is 5280 such
feet. It is distinct from the international foot. `us.us_cable` instead uses
720 international feet, exactly 219.456 m, following the modern convention in
[NIST's revised conversion table](https://www.nist.gov/pml/us-surveyfoot/revised-unit-conversion-factors).
`section` and `township` remain nominal areas based on the international mile;
they do not describe the measured area of an individual surveyed parcel.

`imperial.btu_IT` is exactly 1055.05585262 J. The thermochemical `btu_th` is
exactly `(453.59237 * 5/9)` thermochemical calories, with 1 calorie = 4.184 J.
The code derives it from the pound, Rankine interval and `si.cal`, retaining
the rational value instead of a truncated decimal. These definitions agree with
[NIST REFPROP's Btu conventions](https://www.nist.gov/system/files/documents/2018/05/23/refprop10a.pdf).

`us.circular_mil` is the area of a circle one mil in diameter:
`pi/4 * imperial.th**2`. Its factor retains symbolic pi. `us.inHg` and
`us.inH2O` use conventional column densities of 13595.1 and 1000 kg/m³,
respectively, and standard gravity 9.80665 m/s². Their exact scales under these
conventions are 3386.388640341 and 249.08891 Pa. They represent fixed
conventions, rather than temperature-dependent fluid properties; see
[NIST's pressure conversion table](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b9).
The conventional mercury scale is distinct from converting a column through
the catalogue's `si.Torr` scale.

The water-column attribute is spelled `inH2O` in Python. Its parser symbol is
`inH₂O` with a subscript two; the ASCII name `inch of water` is also accepted.
ASCII digits in unit tokens remain reserved for parser exponents, so the text
`inH2O` is not a registered spelling.

[Catalogue overview](README.md) · [SI units and constants](si.md) ·
[Documentation index](../README.md)
