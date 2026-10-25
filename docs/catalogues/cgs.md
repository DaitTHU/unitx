# CGS and Gaussian electromagnetism

Import the catalogue and the explicit electromagnetic conversion entry point:

```pycon
>>> from unitx.cgs import cgs, gaussian
>>> from unitx.si import si, SI
>>> from fractions import Fraction
>>> (3 * cgs.dyn).to(si.N).value
Fraction(3, 100000)
>>> (3 * cgs.dyn * (2 * cgs.cm)).to(cgs.erg).value
6
>>> cgs.parse('erg/cm') == cgs.erg / cgs.cm
True
>>> (1 * si.J).value_in(cgs.g * cgs.cm**2 / cgs.s**2)
10000000

```

`cgs` contains only the shared mechanical units. `gaussian` inherits `cgs` and
adds Gaussian electromagnetic units, `convert()` and formula coefficients.
Both are read-only UnitCatalog namespaces with independent parsing environments.
Gaussian parsing explicitly includes the inherited mechanical definitions.
Pass their Unit objects to ordinary conversions; bare strings keep using the
default environment. Importing the package adds no default spellings.

```pycon
>>> gaussian.cm is cgs.cm and gaussian.erg is cgs.erg
True
>>> gaussian.parse('erg/statC') == cgs.erg / gaussian.statC
True
>>> hasattr(cgs, 'statC')
False

```

## Exact mechanical units

| Attribute / symbol | Singular name | Definition | Exact SI value |
| --- | --- | --- | --- |
| `cm` | centimetre | shared `si.cm` | 10⁻² m |
| `g` | gram | shared `si.g` | 10⁻³ kg |
| `s` | second | shared `si.s` | s |
| `Gal` | galileo | shared `si.Gal` | 10⁻² m/s² |
| `dyn` | dyne | g cm/s² | 10⁻⁵ N |
| `erg` | erg | dyn cm | 10⁻⁷ J |
| `Ba` | barye | dyn/cm² | 10⁻¹ Pa |
| `P` | poise | dyn s/cm² | 10⁻¹ Pa s |
| `St` | stokes | cm²/s | 10⁻⁴ m²/s |

New entries accept SI prefixes on their canonical symbol and singular name,
for example `cgs.parse('cP')`, `cgs.parse('cSt')` and `cgs.parse('millidyne')`.
They have no additional aliases. Reused SI definitions retain all their existing
spellings and prefix policies, including `gal` for galileo. `cm`, `g`, `s` and
`Gal` are the same objects as the SI entries. New definitions retain their own
formal identities. Values agree with the
[NIST CGS table](https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-5-units-outside-si).

Base-unit expansion uses an explicit target Unit, as above; there is no automatic
unit-system expansion operation.

## Gaussian units have mechanical dimensions

The following attributes belong to `gaussian`; the mechanical catalogue `cgs`
does not expose or parse them.

| Attribute / symbol | Singular name | Exact definition |
| --- | --- | --- |
| `statC` | statcoulomb | g^(1/2) cm^(3/2)/s |
| `statA` | statampere | statC/s |
| `statV` | statvolt | erg/statC |
| `statF` | statfarad | statC/statV, equivalent to cm |
| `G` | gauss | dyn/statC |
| `Mx` | maxwell | G cm² |
| `Oe` | oersted | same mechanical scale as G, independent definition |

These entries also accept SI prefixes on their symbols and singular names.
Other historical spellings such as `Fr`, `esu` and `Gs` are not registered.
Rational powers and mechanical factors remain exact.

Module-level definitions support repr/pickle reconstruction after this package
is loaded. JSON restoration also requires a receiving Registry that resolves the
record's expression with the same definitions:

```pycon
>>> from unitx import Registry
>>> from unitx.serialization import to_dict, from_dict
>>> receiver = Registry([gaussian.statC, cgs.erg])
>>> from_dict(to_dict(2 * gaussian.statC), registry=receiver) == 2 * gaussian.statC
True

```

`statC` and `Mx` have equal dimensions and factors, as do `G` and `Oe`, but each
has a different formal identity. Quantity follows its ordinary numerical rules:
`1 * gaussian.statC == 1 * gaussian.Mx` is true. A Quantity does not store a physical kind
such as charge or flux. Choosing a conversion kind supplies that interpretation
at the conversion boundary; it cannot check physical kinds throughout a formula.

## Explicit SI ↔ Gaussian conversion

`gaussian.convert(quantity, target, *, kind=None)` accepts a linear Quantity
and a Unit target. Parse strings with the appropriate catalogue first.

```pycon
>>> charge = gaussian.convert(1.0 * si.C, gaussian.statC)
>>> round(charge.value, 3)
2997924579.802
>>> abs(gaussian.convert(charge, si.C).value - 1) < 1e-15
True
>>> capacitance = gaussian.convert(2 * cgs.cm, si.F, kind='capacitance')
>>> round(capacitance.value * 1e12, 4)
2.2253

```

Let λ = √kₑ, where kₑ = 1/(4πε₀) is a **dimensioned** SI quantity. The fixed
rule table is:

| `kind` | SI representative | Gaussian representative | SI → Gaussian multiplier |
| --- | --- | --- | --- |
| `charge` | C | statC | λ |
| `current` | A | statA | λ |
| `potential` | V | statV | 1/λ |
| `electric_field` | V/m | statV/cm | 1/λ |
| `magnetic_flux_density` | T | G | c/λ |
| `magnetic_flux` | Wb | Mx | c/λ |
| `capacitance` | F | statF | λ² |
| `magnetic_field_strength` | A/m | Oe | 4πλ/c |

The inverse divides by the same multiplier. Explicit `kind` checks the complete
dimensions of both endpoints, including the logarithmic axis. Either endpoint
may use another unit of the corresponding dimension. No multistep search occurs.

Without `kind`, the seven named Gaussian representatives in the table select
their own route, including prefixes and additional exact Unit scales. The SI end
may be any unit of the corresponding dimension. Selection uses bound definition
identity, so an independently defined unit called `statC` does not inherit a route.
Other Gaussian expressions, including `statV/cm` and a length interpreted as
capacitance, require `kind`. Formal cancellations that leave a named unit are
recognized normally. With no kind and equal complete dimensions, conversion
uses ordinary linear scaling. An explicit kind always checks both endpoints.

Unknown kind names raise ValueError; unsupported object types raise TypeError.
Unmatched routes and incompatible dimensions raise DimensionError. D, P and M,
charge/current densities, other circuit quantities, EMU, full ESU,
Heaviside–Lorentz and medium constitutive parameters are outside this API.

The result is a new ordinary Quantity. `.to()`, low-level linear conversion,
equality, hashing, ordering, addition, array assignment and in-place arithmetic
keep their existing dimension rules. For example, `(1 * gaussian.statC).to(si.C)`
raises DimensionError; convert explicitly before comparing or writing.

## Numerical domains and precision

| Operation | Magnitudes | Result domain |
| --- | --- | --- |
| Mechanical / same-dimension scaling | ordinary supported magnitudes | ordinary conversion rules |
| Cross-representation conversion | Python int, float, Fraction | float |
| Cross-representation conversion | Python complex | complex |
| Cross-representation conversion | supported numeric NumPy scalars / plain ndarrays | native NumPy promotion |
| Cross-representation conversion | Decimal, bool, third-party scalar domains | TypeError |

Input arrays are never written. Broadcasting occurs through subsequent ordinary
Quantity arithmetic. Object/bool arrays and unsupported NumPy subclasses retain
the existing rejection rules. No implicit NumPy import occurs on scalar paths.
Static results use `Quantity[int | Fraction | float]` for real inputs, because
the same-dimension path can remain exact; array results use `Quantity[Array]`,
the NumPy-independent numerical protocol. A particular output dtype is not promised.

The coefficient objects use Python float magnitudes and support the same
initial Python/NumPy arithmetic domains. Decimal formulas require separately
prepared coefficients and are not supported by these objects.

Mechanical definitions are exact. Physical correspondence uses the library's
CODATA 2022 central value μ₀ = 1.25663706127 × 10⁻⁶ H/m and exact
c = 299792458 m/s, deriving ε₀ = 1/(μ₀c²), kₑ and λ. Constants and π are evaluated
as Python floats. Unit scales and the physical coefficient are combined before
one multiplication of the input magnitude; subsequent rounding and overflow
follow Python/NumPy. This API does not propagate measurement uncertainty.
[CODATA 2022](https://physics.nist.gov/cuu/pdf/all.pdf) gives μ₀ relative standard
uncertainty 1.6 × 10⁻¹⁰. After the
[2019 SI redefinition](https://www.bipm.org/en/committees/cg/cgpm/26-2018/resolution-1),
μ₀ is measured, so `1 G ≈ 10⁻⁴ T` and `1 Oe ≈ 1000/(4π) A/m` are approximate
correspondences here. Historical exact SI-embedded electromagnetic units are
not supplied under the same names.

The mechanical representation and conversion relations were cross-checked
against [Pint's Gaussian definitions](https://raw.githubusercontent.com/hgrecco/pint/master/pint/default_en.txt).
unitx uses the explicit entry point above and its own limited kind table.

## Shared formulas with explicit coefficients

`si.em` and `gaussian.em` group read-only `k1`, `alpha`, `k2` and `c`
attributes, all ordinary Quantities. These are frozen coefficient objects,
referenced by Final namespace attributes; they are not parsing environments.
`k2 = k1/(alpha*c²)` is derived; in Gaussian it simplifies to `alpha = 1/c`.
Here `alpha` is the Lorentz-force coefficient, unrelated to fine-structure
constant `SI.alpha`.

The float coefficients are derived from the existing SI constant data. `SI`
retains its physical constants and their existing numerical domains. `si.c` is
a speed Unit, while `si.em.c` is a Quantity representing the same speed.
Choosing either view does not activate a conversion context.

| Coefficient | SI | Gaussian |
| --- | --- | --- |
| `k1` | 1/(4πε₀) | 1 |
| `alpha` | 1 | 1/c |
| `k2` | μ₀/(4π) | 1/c |

```pycon
>>> def coulomb_force(q1, q2, r, em):
...     return em.k1 * q1 * q2 / r**2
>>> q_si, r = 1e-9 * si.C, 2.0 * si.cm
>>> q_g = gaussian.convert(q_si, gaussian.statC)
>>> f_si = coulomb_force(q_si, q_si, r, si.em)
>>> f_g = coulomb_force(q_g, q_g, r, gaussian.em)
>>> import math
>>> math.isclose(f_g.value_in(si.N), f_si.value_in(si.N), rel_tol=2e-15)
True

```

For the Lorentz force use **F** = q(**E** + alpha **v** × **B**), with inputs
prepared in the same representation as the coefficients. Vacuum energy density
is E²/(8π k1) + alpha B²/(8π k2); it reduces to (E² + B²)/(8π) in Gaussian.
A sphere's SI capacitance 4πε₀R converts to Gaussian length R via
`kind='capacitance'`.

No active system or context manager changes existing objects or Python formulas.
There is no separate `cgs.constants` namespace. Ordinary SI constants can be
converted to mechanical units. For electromagnetic constants, choose a supported
numerical magnitude before mapping; for example, the shipped `SI.e` uses Decimal:

```pycon
>>> from unitx import Quantity
>>> electron_charge = gaussian.convert(Quantity(float(SI.e.value), SI.e.unit), gaussian.statC)
>>> round(electron_charge.value * 1e10, 6)
4.803205

```

[Catalogue overview](README.md) · [Units and quantities](../units.md)
