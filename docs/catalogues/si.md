# SI units and physical constants

```pycon
>>> from unitx import Quantity, Unit
>>> from unitx.si import SI, si
>>> si.unit('metre') == si.m == Unit('metre')
True
>>> SI.c == 299792458 * si.m / si.s
True

```

`unitx.si` provides read-only namespaces: lowercase `si` contains ready-to-use
units and coordinate descriptors; uppercase `SI` contains physical-constant
Quantities. `Particle` contains particle data, while `au`, `nu` and `Planck`
contain physical unit scales expressed as SI Quantities.
Importing unitx initializes this catalogue.

`si.em` is a read-only coefficient view for common electromagnetic formulas:
`k1 = 1/(4*pi*epsilon0)`, `alpha = 1`, `k2 = mu0/(4*pi)`, and vacuum light
speed `c`. Its Quantities use float magnitudes derived from CODATA 2022 mu0
and exact BIPM c. `alpha` here is the Lorentz-force coefficient; `SI.alpha`
remains the fine-structure constant. See the
[coefficient definitions and formulas](cgs.md#shared-formulas-with-explicit-coefficients).
These attributes do not affect unit parsing or activate a physical conversion
context. `si.c` remains the speed Unit; `si.em.c` is a physical Quantity.

Unit definitions and prefixes follow the
[BIPM SI Brochure](https://www.bipm.org/en/publications/si-brochure).
Measured constants use [CODATA 2022](https://physics.nist.gov/cuu/pdf/all.pdf)
central values. Non-SI entries retain the specific conventions documented below.

## Available groups

The following are examples of namespace attributes, not an exhaustive list
of accepted parser spellings:

| Group | Examples in `si` |
| --- | --- |
| SI bases | s, m, kg, A, K, mol, cd |
| Named derived units | rad, sr, Hz, N, Pa, J, W, C, V, F, ohm, S, Wb, T, H, lm, lx, Bq, Gy, Sv, kat |
| Common prefixes | ns, ms, um, cm, km, MeV, kW, mA |
| Angles and ratios | turn, cycle, deg, arcmin, arcsec, mas, uas, percent, permille, ppm |
| Time and frequency | min, h, d, yr, a_j, a_g, a_t, a_s, a_c, kyr, Myr, Gyr, mo_j, rpm |
| Length, area and volume | au, pc, ly, nmile, angstrom, a, ha, b, L |
| Mass and mechanics | t, Da, u, c, Gal, knot, gf, kgf |
| Pressure and energy | bar, atm, mHg, mmHg, Torr, mTorr, Wh, eV, cal |
| Spectral flux density | Jy, nJy, uJy, mJy, kJy |
| Radioactivity | Bq, kBq, MBq, GBq, Ci, pCi, nCi, uCi, mCi, kCi |
| Electrical variants | VA, var, mho |
| Temperature and logarithmic coordinates | degC, dBm, dBW, dBV |
| Logarithmic differences | Np, B, dB |

The parser accepts permitted prefixes even when a convenience attribute is
not listed in `si`: construct `Unit('...')` for those spellings. Namespace
attributes are predefined; dynamically registering a unit does not add one.
For the complete definitions see [unit source](https://github.com/DaitTHU/unitx/blob/main/src/unitx/si/units.py)
and [SI bases](https://github.com/DaitTHU/unitx/blob/main/src/unitx/si/bases.py).

## Constants and precision

`SI` includes the seven SI defining constants, plus constants such as hbar,
G, mu0, epsilon0, me, mp, mn, alpha and the Wien displacement constants.
Particle data are available from `from unitx.si import Particle`; the existing
`unitx.si.constants.Particle` import remains supported.

Defining decimal values are stored using Decimal/int magnitudes. Measured
constants carry the digits in the catalogue; Quantity does not attach an
uncertainty model to them. Derived constants are numerical values calculated
when the catalogue loads in an isolated Decimal context: 28 digits,
`ROUND_HALF_EVEN`, exponent limits -999999 to 999999, and traps for invalid
operations, division by zero and overflow. Caller precision, rounding, traps,
flags and exponent limits do not change those stored values; import leaves the
caller context unchanged. Increasing Decimal precision later does not
recompute them. Subsequent arithmetic still follows the caller's context. The catalogue does not promise arbitrary-precision constants.
See the [data update policy](../compatibility.md#constants-and-catalogue-data).
See [constant source](https://github.com/DaitTHU/unitx/blob/main/src/unitx/si/constants.py)
for the stored definitions. Known computed Decimal constants, including `SI.re`,
retain `Quantity[Decimal]` annotations through the underlying Quantity operations.

Float formulas can use Decimal constants directly, for example `0.5 * SI.e`.
Mixed built-in float/Decimal `+`, `-`, `*`, `/` calculations return float after
Decimal evaluation. This implicit precision downgrade and its context/range
limits are documented in [scalar behavior](../scalars.md#arithmetic-and-result-types).

`unitx.pi` is a symbolic Factor base with a float approximation for numerical
evaluation. It supports exact cancellation in unit scales. It is distinct
from predefined `SI.hbar`, whose numerical magnitude is expressed in ordinary
J s units. See [symbolic factors](../scalars.md#symbolic-factors-and-pi) and
[exact extra scales](../custom-units.md#exact-extra-scales).

## Atomic, natural and Planck scales

```pycon
>>> from unitx.si import Particle, Planck, au, nu
>>> Particle.me is SI.me
True
>>> au.length is SI.a0
True
>>> # A physical length equal to three Bohr radii:
>>> length = 3 * au.length
>>> float((length / au.length).value_in(''))
3.0
>>> nu.velocity is SI.c
True
>>> Planck.length.unit == si.m
True

```

| Namespace | Convention | Representative scales |
| --- | --- | --- |
| `au` | Hartree atomic units: ħ = e = mₑ = 4πε₀ = 1 | `length` (Bohr radius), `energy` (Hartree), `time`, `charge` |
| `nu` | Electron-mass natural units: c = ħ = mₑ = ε₀ = 1 | `length` (reduced electron Compton wavelength), `energy`, `charge = sqrt(ε₀ ħ c)` |
| `Planck` | c = ħ = G = kB = 1 | `mass`, `length`, `time`, `energy`, `temperature` |

All three expose `velocity`, `action`, `mass`, `energy`, `momentum`, `length`
and `time`. `au` and `nu` also expose `permittivity`, `force`, `charge`,
`current`, `charge_density`, `electric_potential`, `electric_field`,
`electric_dipole_moment`, `electric_quadrupole_moment`,
`electric_polarizability`, `magnetic_flux_density`, `magnetic_dipole_moment`
and `magnetizability`. `au` and `Planck` expose `temperature`.

The atomic magnetic dipole scale is **twice** the Bohr magneton. In `nu`,
the elementary charge has numerical value sqrt(4πα); other natural-unit
conventions need not use the electron mass or this charge normalization.
`Planck` uses ħ rather than h and does not select a charge convention.

These attributes are **Quantities**, not Unit objects accepted as `.to()`
targets. Divide by a scale and use `.value_in('')` to obtain a dimensionless
number, or multiply a number by a scale to construct a physical quantity.
The namespaces do not activate cross-dimension conversion or register new
parser spellings. In particular, `si.au` and `Unit('au')` still mean the
astronomical unit; `au.length` is the atomic length scale.
All derived scales inherit the catalogue's finite numerical precision and
lack of uncertainty propagation.

## Spelling and scale conventions

`si.turn` and `si.cycle` share one definition: exactly `2*pi rad`, or 360 degrees.
The parser accepts both `turn` and `cycle`. These are angular units: `cycle/s`
converts to `rad/s` with factor `2*pi`; bare Hz remains `1/s`. Neither spelling
accepts prefixes. `si.ppm` is exactly `1e-6` and accepts no prefixes. It does not
infer mass, volume or molar fractions from its name.

`si.Jy` is the jansky, exactly `1e-26 W/(m**2*Hz)`, following the
[NRAO definition](https://public.nrao.edu/ask/how-does-one-measure-the-flux-of-a-radio-source/).
It accepts decimal SI prefixes; nJy, uJy, mJy and kJy are convenience attributes.
This is spectral flux density per frequency. Converting to a per-wavelength
density requires a spectral coordinate and its Jacobian, rather than an ordinary
unit conversion. Binary prefixes require explicit permission; metres reject `Kim`.

`si.Ci` is the curie, a non-SI activity unit defined as exactly
`37_000_000_000 Bq` by [CGPM Resolution 7 (1964)](https://www.bipm.org/en/committees/cg/cgpm/12-1964/resolution-7).
It accepts decimal SI prefixes; pCi, nCi, uCi, mCi and kCi are convenience
attributes. `Unit('curie')` and `Unit('microcurie')` accept the corresponding
names. Both Unicode micro spellings and ASCII `uCi` are supported.

```pycon
>>> (1 * si.uCi).value_in(si.Bq)
37000
>>> (1 * si.Bq).value_in(si.Ci)
Fraction(1, 37000000000)

```

Symbols and names are case-sensitive. Lowercase `k` is kilo and uppercase
`K` is kelvin: `km` is kilometre; `K m` is kelvin times metre. Unsupported
`Km`/`Kg` spellings raise UnitSymbolError. Micro accepts canonical `µ`, Unicode
`μ`, and ASCII `u` without warnings. The mass reference scale is kg: g has
SI factor 1/1000. Angle units remain physically dimensionless, retaining their
formal identity and scale; deg has exact factor pi/180.

`arcsec` and `arcsecond` accept prefixes, for example `marcsec` and
`milliarcsecond`; `″` and `"` do not. `as` continues to mean attosecond,
and `mas` is not an arcsecond alias. `a` (are) and `ha` (hectare) are separate
unprefixable units; `Pa` remains pascal.
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
with value 1. For angular velocity, multiply that frequency by `2 * pi` and
convert to `si.rad / si.s`; directly converting frequency to `rad/s` does
not insert the physical factor of 2π. `si.c` is a speed unit equal to the
speed of light, while `SI.c` is the corresponding physical quantity.

The spelling `u` denotes the unified atomic mass unit when used on its own.
ASCII micro spellings such as `um` and `us` remain supported without warnings,
so entering non-ASCII characters is optional. `um`, `µm`, `μm`, and
`micrometre` all mean micrometres; the explicit product `u m` instead means
unified atomic mass unit times metre.

## Years and calendar boundaries

`si.yr` and `si.a_j` are the same Unit, displayed as `yr`.
The spellings `yr`, `a_j`, `year`, `Julian year` and `annum` mean exactly
365.25 days, following the
[IAU scientific-year convention](https://iauarchive.eso.org/publications/proceedings_rules/units/).
IAU recommends `a`; unitx uses `yr` because `a` already denotes the are
(100 m²). Parsing never chooses between area and time from context.
Decimal prefixes are supported, including `si.kyr`, `si.Myr` and `si.Gyr`.

| Attribute / symbol | Parsed name | Days of 86,400 seconds | Convention |
| --- | --- | --- | --- |
| `yr` / `a_j` | `Julian year`, `year`, `annum` | 365.25 | Exact scientific year |
| `a_c` | `common year` | 365 | Fixed common-year interval |
| `a_g` | `Gregorian year` | 365.2425 | Exact 400-year calendar average |
| `a_t` | `tropical year` | 365.24219 | Fixed UCUM approximation |
| `a_s` | `sidereal year` | 365.256363004 | Fixed IERS tabulated approximation |

The qualified symbols `a_j`, `a_g`, `a_t` and their numerical definitions
follow [UCUM](https://ucum.org/ucum). This is not a UCUM parser: unitx retains
its own grammar, allows decimal prefixes on these years, and keeps `a` for
area. `a_c` and `a_s` are **unitx-specific symbols**.
The Gregorian average follows 97 leap days per 400-year cycle;
see [USNO's explanation](https://aa.usno.navy.mil/faq/leap_years).
The sidereal reference comes from the
[IERS useful-constants table](https://hpiers.obspm.fr/eop-pc/models/constants.html),
which cites Simon et al. (1994).

unitx stores each decimal scale exactly for reproducible conversion. The
astronomical interpretation of tropical and sidereal years is approximate:
neither is an exact period valid at every epoch, and the stored digits do
not represent an uncertainty model. In particular, UCUM's tropical-year
convention differs from the IERS table's 365.242190402 days. More digits in a
different convention do not make it the correct choice for a particular dataset.

```pycon
>>> si.yr is si.a_j
True
>>> (1 * si.Myr).value_in(si.yr)
1000000
>>> (1 * si.a_g).value_in(si.s)
31556952
>>> from decimal import Decimal
>>> (1 * si.a_t).value_in(si.s) == Decimal('31556925.216')
True

```

A dataset's `y` label still needs its source definition. For example,
[NUBASE2020, section 2.3](https://www-nds.iaea.org/amdc/ame2020/NUBASE2020.pdf)
uses 31,556,926 seconds, which differs from all five built-in years.
Express that interval directly, without changing the default registry:

```pycon
>>> dataset_year = (31_556_926 * si.s).define('dataset_year', prefixable=False)
>>> (2 * dataset_year).value_in(si.s)
63113852
>>> 1 * dataset_year == 1 * si.a_t
False

```

`si.mo_j`, parsed as `mo_j` or `Julian month`, is the fixed mean Julian month:
exactly one twelfth of `si.yr`, or 2,629,800 seconds, following UCUM.
Unqualified `mo` / `month` are not provided. A calendar month requires a date
and a calendar; a fixed mean month cannot implement that operation.
A fixed 366-day interval is simply `366 * si.d`. Other orbital periods, such
as anomalistic and draconic years, require an explicit model or reference
convention and are not currently catalogued. Calendar arithmetic and
astronomical ephemerides remain outside fixed-factor unit conversion.
See the [migration notes](../migration.md#upgrading-to-0180)
for the changed meaning of `yr` and stored-data migration.


## Unicode unit characters

Unicode includes dedicated unit characters in
[Letterlike Symbols](https://www.unicode.org/charts/PDF/U2100.pdf) and
[CJK Compatibility](https://www.unicode.org/charts/PDF/U3300.pdf).
The following characters are accepted as symbol aliases for existing units;
they preserve canonical unit identity, hashes and output spellings.
Load the catalogue with `from unitx.si import si` (also loaded by `unitx`).
Use strings such as `Unit('㍴')`; these aliases do not add Python attributes to `si`.

| Character | Code point | Canonical spelling / namespace |
| --- | --- | --- |
| ℃ | U+2103 | °C / si.degC |
| ℓ | U+2113 | L / si.L |
| Ω | U+2126 | Ω / si.ohm |
| K | U+212A | K / si.K |
| Å | U+212B | Å / si.angstrom (already supported) |
| ℧ | U+2127 | mho / si.mho (already supported) |
| ㍳ | U+3373 | au / si.au |
| ㍴ | U+3374 | bar / si.bar |
| ㍶ | U+3376 | pc / si.pc |
| ㎈ | U+3388 | cal / si.cal |
| ㎐ | U+3390 | Hz / si.Hz |
| ㎩ | U+33A9 | Pa / si.Pa |
| ㎭ | U+33AD | rad / si.rad |
| ㏃ | U+33C3 | Bq / si.Bq |
| ㏅ | U+33C5 | cd / si.cd |
| ㏉ | U+33C9 | Gy / si.Gy |
| ㏊ | U+33CA | ha / si.ha |
| ㏏ | U+33CF | kn / si.knot |
| ㏐ | U+33D0 | lm / si.lm |
| ㏓ | U+33D3 | lx / si.lx |
| ㏖ | U+33D6 | mol / si.mol |
| ㏛ | U+33DB | sr / si.sr |
| ㏜ | U+33DC | Sv / si.Sv |
| ㏝ | U+33DD | Wb / si.Wb |

Aliases follow each unit's existing prefix policy: `m㍴`, `kΩ` and `mℓ`
are accepted, while temperature coordinates disable prefixes.
`Unit('℃')` denotes a temperature difference, and
`AffineQuantity(0, '℃')` denotes a Celsius point, following the same rules as
`degC`. The characters do not change point/difference semantics.

```pycon
>>> Unit('㍴') == si.bar
True
>>> Unit('㎩') == si.Pa
True
>>> Unit('kΩ') == si.kohm
True
>>> from unitx import AffineQuantity
>>> from unitx.imperial import imperial
>>> AffineQuantity(32, imperial.unit('℉')) == AffineQuantity(0, '℃')
True

```

Other Unicode characters already contain prefixes or compound expressions:

| Characters | Corresponding unit expressions (not character aliases) |
| --- | --- |
| ㍱, ㎪, ㎫, ㎬ | hPa, kPa, MPa, GPa |
| ㎏, ㎎, ㎍ | kg, mg, ug |
| ㎞, ㎝, ㎜, ㎛, ㎚ | km, cm, mm, um, nm |
| ㎡, ㎢, ㎠, ㎟ | m², km², cm², mm² |
| ㎥, ㎦, ㎤, ㎣ | m³, km³, cm³, mm³ |
| ㎧, ㎨, ㎮, ㎯ | m/s, m/s², rad/s, rad/s² |
| ㎑, ㎒, ㎓, ㎔ | kHz, MHz, GHz, THz |
| ㏀, ㏁, ㏈ | kΩ, MΩ, dB |
| ㎉, ㎖, ㎕ | kcal, mL, uL |
| ㏞, ㏟ | V/m, A/m |

These characters are listed for reference and are not currently accepted as
aliases. Registering `㎏` as an alias of `g`, for example, would assign the
wrong scale; defining another unit for it would lose the formal identity of
`kg`. The parser does not apply blanket Unicode compatibility normalization.
The block also contains byte, horsepower, inch, pH, mathematical operators,
and Japanese phonetic unit forms; their presence in Unicode does not imply
support in this catalogue. For portable input, use ordinary unit expressions.


[Documentation index](../README.md) · [Unit expressions](../units.md#unit-expressions) ·
[Custom units](../custom-units.md)
