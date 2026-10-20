# Changelog

## Unreleased

### Added

- Support prod and multiply.reduce with exact product units, scalar initial
  values (including Quantities), boolean input masks, and staged output writes.
  Require equal selected counts for dimensional outputs; normalize dimensionless
  inputs before reduction. Document empty products and accumulator dtype rules.
- Support trapezoid integration with Quantity ordinates, coordinates, or scalar
  spacing. Combine their units directly, preserving native coordinate order
  and axis behavior without manual value extraction or unit reconstruction.

### Improved

- Keep Python-comparison-to-ufunc mapping in the optional NumPy adapter and
  replace redundant Fractionable type guards with direct isinstance checks.

## [0.8.3] - 2026-10-20

### Added

- Support unit-aware `out` for sum, mean, median, std, var, cumsum,
  concatenate, stack, and add.reduce/accumulate. Require the exact result shape,
  preserve target identity, unit, and dtype, and stage computation, conversion,
  and casting before writing. Argmin/argmax accept bare integer ndarray outputs.
- Document function/reduction output casting and add regressions for scaled
  units, positional outputs, overlapping/strided buffers, and unchanged outputs
  on failure, plus installed-consumer type checks.

## [0.8.2] - 2026-10-20

### Fixed

- Support Decimal and Fraction scalar operands in NumPy calculations, including
  SI constants such as h and kB. Convert them to Python floats at the NumPy
  boundary without changing stored constants or Python scalar arithmetic.
  Keep exact Fraction/Decimal unit exponents in dimensional NumPy powers.
- Remove recursive arithmetic return constraints from the Scalar protocol to
  avoid excessive type-checker work against NumPy overloads. Preserve scalar
  boolean equality and public Quantity result overloads.

### Improved

- Describe array arithmetic, conversion, indexing, and iteration with concrete
  ndarray/NumPy-numeric-scalar result unions. Preserve the magnitude type and
  dtype for transpose, expose NumPy dtype metadata, and type flatten as ndarray.
- Add strict NumPy method-completion and no-NumPy scalar-consumer checks,
  installed-wheel validation, and a runnable blackbody example using asarray.
- Rewrite the README as a concise user guide, retaining detailed unit, scalar,
  and migration documentation under docs/. Document NumPy stub limitations and
  validate the example's supported type-inference path without an explicit dtype.

## [0.8.1] - 2026-10-19

### Added

- Support `out` for covered single-output NumPy ufunc calls. Numeric results
  write to compatible Quantity ndarray buffers after unit conversion; boolean
  comparisons and predicates write to bare boolean ndarrays. Preserve target
  identity, dtype, unit, and masked-out values, including overlapping views.
- Stage computation, unit conversion, and casting before writing; validation
  and temporary computation failures leave the target unchanged. Function and
  reduction outputs, item assignment, and multi-output ufuncs remain unsupported.

## [0.8.0] - 2026-10-19

### Added

- Add optional NumPy >= 2.1 support through `unitx[numpy]`, with one Quantity
  wrapping an entire plain numeric ndarray or NumPy numeric scalar. Preserve
  input objects in both `array * unit` and `unit * array` construction.
- Support unit-aware conversion, broadcasting arithmetic and comparisons,
  matrix multiplication, indexing, iteration, array metadata, transpose, and
  flattening. Preserve NumPy scalar indexing and view/copy behavior.
- Implement a documented set of NumPy ufuncs and functions, including roots,
  dimensionless transcendental functions, concatenation, statistics, predicates,
  and position queries. Keep exact unit exponents and ScaledUnit factors.
- Extend Quantity.isclose with symmetric array comparisons and broadcasting
  tolerances; implement NumPy isclose/allclose with their own defaults and
  asymmetric formula. Validate tolerance dimensions, finiteness, and signs.
- Add runtime and installed-consumer typing coverage, import-order and optional
  dependency checks, and CI jobs for minimum and current NumPy versions.

### Changed

- Define `Scalar`, `Array`, and `Magnitude = Scalar | Array` in `_numerics`.
  Preserve concrete magnitude types at construction and in unit-only operations;
  use conservative result types for array numerical operations.
- Keep Python scalar arithmetic, Decimal behavior, and exact scalar comparisons
  independent of the optional NumPy adapter. Array conversion follows the
  documented NumPy promotion, rounding, and overflow rules.

### NumPy scope

- Document the supported function/keyword matrix and precision/storage rules in
  `docs/numpy.md`. Other array backends, ndarray subclasses, object/boolean
  magnitudes, and implicit list/tuple conversion are unsupported.
- Reject implicit unit removal by `np.asarray(q)` or `np.array(q)`; use `.value`
  or `.value_in(unit)` explicitly. NumPy magnitudes produce unhashable Quantities.
- Reject writable out, item assignment, and operations that would require
  heterogeneous units. Python augmented assignment creates and rebinds a result.

## [0.7.1] - 2026-10-19

### Added

- Accept a single exact spelling string for `prefixable`, equivalent to a
  one-element tuple, with the same spelling validation before registration.
- Expand the `si` namespace's prefixed shortcuts and common units, including
  percent, permille, permyriad, rotational frequency (`rpm`), speed of light
  (`c`), gram-force, volt-ampere, ampere-hour, mho, and nit. Enable Torr prefixes.
- Add `SI.kappa` and the wavelength and frequency Wien displacement constants,
  `SI.b` and `SI.b_`.

### Changed

- Use fixed Decimal approximations with about 50 significant digits for pi and
  the positive roots in the Wien displacement laws. Derived constants still use
  the Decimal context active at import.
- Treat `u` and `Da` as spellings of one formal unit, with prefixes enabled only
  on `Da`. Rebuild stored units when migrating from the separate `u` identity;
  the mass conversion scale is unchanged. Use `kDa` rather than `kilodalton`.
- Keep ASCII micro spellings such as `um` and `us` convenient to type without
  warnings. The explicit product `u m` retains its separate mass-times-length
  meaning.

## [0.7.0] - 2026-10-19

### Added

- Add `[registered name]` references as single factors in unit expressions,
  including products, reciprocals, parentheses, and rational powers. References
  resolve names and name aliases only, with whitespace normalization and no
  expression evaluation or nested quoting.
- Support standalone hyphenated names and adjacent Unicode integer superscripts
  such as `m²`, `s⁻¹²`, and `[metre]²`.
- Extend `prefixable: PrefixPolicy = True` on all `.define()` entry points to
  accept a tuple of exact registered symbols, names, or aliases. Only those
  spellings accept prefixes; True enables all spellings and False or () disables
  all spellings. `PrefixPolicy` is `bool | tuple[str, ...]`.
- Enable prefixes on `arcsec` and `arcsecond` while keeping angle punctuation
  unprefixable. Register unprefixable are (`a`) and hectare (`ha`), and reactive
  power (`var`, name `volt-ampere reactive`). Keep `as` as attosecond and `yr`
  as a 365-day year.
- Add the Julian year `a_j` (exactly 31,557,600 seconds) and its twelfth,
  `mo`, as a fixed month interval distinct from variable calendar months.
- Add light year (`ly`), unified atomic mass unit (`u`), and common prefixed
  namespace attributes, including attosecond (`as_`) and petaelectronvolt (`PeV`).

### Changed

- Move definitions, spelling resolution, and registration validation into
  `unitx.unitregistry`, shared by SingleUnit and UnitParser. Preserve formal identities,
  exact SI factors, existing `.define()` defaults, and validation before mutation.
- Remove the superseded private `_unit_data` module. Name definition records
  `PrefixDefinition` and `UnitDefinition`, with explicit definition indexes and
  alias-to-symbol mappings.
- Select permitted canonical output spellings after applying binding policies.
  When only a name permits a prefix, expressions use `[name]` so repr and pickle
  preserve the prefixed identity. Expose `PrefixPolicy` for type annotations.
- Use the finite spelling exceptions `megohm` and `kilohm` in preferred names,
  retaining `megaohm` and `kiloohm` as accepted inputs.
- Reuse Unicode superscript tables, handle malformed name brackets through
  the common lexical error path, and simplify definition initialization,
  preferred spelling selection, and generated prefix-name validation.
- Share coherent SI units and the seven defining constants through private
  bases. Derive `ly` from `c` and the Julian year, and `eV` from `e` and `V`,
  preserving the public `si` and `SI` namespaces and constant magnitude types.

### Fixed

- Register metre of mercury (`mHg`) as the prefixable unit and construct `mmHg`
  from its milli prefix. Preserve `760 mmHg = 1 atm` and the existing Torr scale,
  keep the definition exact, and reject stacked prefixes such as `kmmHg`.

### Migration

- Square brackets and `⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻` are now reserved grammar characters and can
  no longer occur in registered symbols or names. Rename custom spellings that
  previously used them before rebuilding or unpickling those definitions.
- Long names remain whole-input lookups unless explicitly referenced in brackets.
  `str()` remains readable display and is not a general serialization format;
  use `expression()` for formal unit products and repr for ScaledUnit state.
- `mmHg` now has the formal identity of milli-prefixed `mHg`; rebuild stored
  units when moving between versions. Its pressure conversion scale is unchanged.
  `kmmHg` and names with repeated prefixes are no longer accepted.
- Registering `u` makes ASCII micro spellings such as `um` issue the existing
  ambiguity warning. Their meaning is unchanged; use `µm`, `μm`, or `micrometre`
  to avoid the warning. `kt` remains a knot alias and tonne stays unprefixable.

## [0.6.3] - 2026-10-18

### Added

- Add `Quantity.value_in(unit)` to return the same magnitude and concrete type as `.to(unit).value` without constructing a temporary quantity. The target unit is required and accepts a string or Unit, including ScaledUnit.
- Add `Quantity.isclose()` for symmetric real scalar comparisons with relative and absolute tolerances in coherent SI units. Nonzero absolute tolerances for dimensional quantities must carry matching units. Rational coefficients remain exact; residual irrational factors are approximated. Zero tolerances use exact equality.
- Make quantity truth values follow their magnitudes, so zero quantities are false regardless of unit.

### Changed

- Apply native magnitude operators in operand order for arithmetic, unary operations, and numerical powers, preserving numerical promotion, Decimal contexts, signals, and exceptions. Support the scalar operand types accepted by the magnitude, including complex values and custom scalars.
- Keep Decimal conversions and arithmetic in Decimal. Rational conversions combine the magnitude and scale exactly before rounding once in the active context; identity conversions preserve the original magnitude. Addition and subtraction align integer operands in the Decimal domain when paired with Decimal magnitudes. Derived SI constants use Decimal arithmetic and `Decimal(str(math.pi))` where an approximate pi is required, with evaluation in the context active at import.
- Interpret float formal exponents by their shortest decimal text, without a denominator limit, consistently across Monomial, Dimension, Factor, SymbolicConstant, Unit, and Quantity. For example, `0.1` represents `1/10` and `1e-10` remains nonzero. Numeric scale inputs to Factor retain their existing denominator-limit and warning rules.
- Cache rationalized Factor decompositions lazily, independently of numerical value evaluation, while preserving equality, hashing, copying, and serialization.
- Update public type hints, examples, and regression coverage for conversion, native arithmetic, comparisons, and formal exponents.

### Fixed

- Preserve native comparisons, including independent `!=`, Decimal signals, and complex ordering errors, when unit factors agree. Across different factors, compare builtin real values and complex components using exact coefficients, retaining consistent hashes for equal quantities and native unhashability for custom magnitudes.

### Migration

- Decimal arithmetic no longer converts operands to Fraction automatically. Mixing Decimal with float or Fraction in arithmetic now raises the native TypeError; explicitly choose a common numerical type when needed. Decimal conversion and derived SI constants can therefore have different result types and rounding from earlier releases.
- Quantity numerical powers use the original exponent object; for example, an integer magnitude raised to `2.0` produces a float. Use `Fraction(1, 3)` for an exactly one-third formal exponent: the float `1 / 3` now represents the decimal `0.3333333333333333` and does not simplify as exactly one third.

## [0.6.2] - 2026-10-18

### Added

- Support multiplication and division between Quantity and Factor or SymbolicConstant in both operand orders. Multiplication and right-hand division preserve the magnitude and its type, combining the exact factor with the unit's scale; left-hand division takes the reciprocal magnitude and scales the reciprocal unit. Ordinary numeric operands continue to operate on the magnitude.
- Preserve generic magnitude types in the corresponding public type hints, with runtime and static regression coverage for exact scaling and reciprocal operations.

### Changed

- Combine the compact exact scale and formal units in ScaledUnit string output, grouping denominator factors into one product, such as `J⋅s/(2⋅π)`.
- Display quantities with scaled units using the combined product and multiplication dots between adjacent numeric factors, such as `3⋅60⋅s`, without evaluating the exact scale. String output remains readable display rather than parser input; `repr()` retains reconstructible state.
- Document exact-factor quantity arithmetic and the distinction between a symbolic expression such as `SI.h / (2 * PI)` and predefined `SI.hbar`, which continues to expose its numerical value in ordinary `J*s` units.

## [0.6.1] - 2026-10-17

### Changed

- Make multiplication of ordinary numeric magnitudes with Unit and ScaledUnit symmetric: both operand orders construct a Quantity and preserve the magnitude type. Zero and negative magnitudes are supported; float magnitudes retain their stored values without Factor rationalization.
- Return Quantity when dividing Unit or ScaledUnit by an ordinary number. Integer, Fraction, and finite Decimal divisors produce exact Fraction reciprocals, so `si.deg / 60` preserves the exact arcminute definition; float divisors use native division.
- Keep Factor and SymbolicConstant multiplication and division as exact ScaledUnit operations. To retain the former explicit scaling behavior, use `unit * Factor(value)` or `unit / Factor(value)`.

## [0.6.0] - 2026-10-17

### Added

- Add `unitx.si` with read-only `si` unit and `SI` physical-constant namespaces, plus particle data in `unitx.si.constants.Particle`. Provide common SI and non-SI units, exact SI defining constants, and derived physical quantities ready for use in formulas.
- Support multiplying or dividing a `Unit` or `ScaledUnit` by a positive finite int, Fraction, Decimal, float, Factor, or SymbolicConstant on the right. These operations return a `ScaledUnit` and preserve its exact additional scale. Ordinary scalar magnitudes on the left still construct quantities.
- Accept multiword primary names and name aliases when registering units. Parse complete multiword names before implicit products, support directly attached prefix names, and reject definitions that would replace existing unit products.
- Test predefined-unit conversions, electronvolt scaling, constant values, namespace access, fresh-process initialization, and public type inference. Introduce the ready-to-use SI namespaces first in the README.

### Changed

- Change the definition signature to `.define(symbol, /, name, *name_aliases, symbol_aliases=(), prefixable=True)` across Unit, ScaledUnit, and Quantity. The primary name accepts positional or keyword input; pass extra name aliases positionally instead of using the removed `name_aliases=` keyword.
- Initialize the standard unit catalogue when importing unitx. Previously unregistered standard spellings, such as `N` and `min`, now resolve immediately; reuse predefined units rather than registering those spellings again.
- Export Unit, Quantity, Dimension, DIMENSIONLESS, ONE, PI, TEN, and the `si` module at the package root. Use `from unitx.si import SI, si` for the constant and unit namespaces.

### Limitations

- Multiword names are recognized only as complete inputs. Use unit symbols or combine Unit objects inside products, reciprocals, parentheses, and powers. Registration still excludes decimal digits and parser operators, including hyphens, from names.
- Physical constants store nominal measured values without uncertainty propagation. Derived constants involving numerical approximations are evaluated numerically.

## [0.5.0] - 2026-10-17

### Added

- Add `ScaledUnit` in `unitx.unit` for formal unit products with an additional exact positive scale. Multiplying or dividing a `Factor` or `SymbolicConstant` by a unit constructs a `ScaledUnit`.
- Preserve explicit scales through unit multiplication, division, rational powers, quantity conversion, readable display, reconstructible representations, copying, and pickling. A scale of one compares and hashes equal to the corresponding ordinary `Unit`.
- Add `Unit.expression()` to return the formal unit product in parser syntax, without any additional `ScaledUnit` scale.
- Add `.define()` on `Unit`, `ScaledUnit`, and `Quantity` objects to register units in the current process. Definitions return an ordinary `Unit` with its own formal identity and the source's physical dimension, incorporating quantity magnitudes into the registered SI factor without changing the source object.
- Require a nonempty primary symbol and name, with optional symbol aliases, name aliases, and configurable SI prefix support. All spellings must be single printable unit tokens; empty aliases are ignored.
- Add `UnitRegistryError` for conflicts with existing unit spellings or ambiguous prefix meanings. Validate symbols, names, aliases, and generated prefixed forms before changing the registry.
- Accept positive finite int, Fraction, Decimal, and float magnitudes for quantity definitions. Fraction and Decimal scales stay exact; float scales follow Factor's denominator limit and can warn or round to zero and fail. Restoring custom units in another process requires their definitions to be registered there.

### Changed

- Resolve dynamically registered unit symbols, names, aliases, and prefixed forms without relying on import-time unit-length limits.
- Display quantities with an explicit scale other than one using parentheses around the scaled unit, without numerically evaluating symbolic scales.
- Escape symbols in `SingleUnit.__repr__` so registered symbols containing quotes or backslashes can be reconstructed.

## [0.4.0] - 2026-10-16

Replace the implementation previously published on PyPI as 0.3.0. The 0.0.x
entries below retain this repository's development milestones.

### Changed

- Require Python 3.12 or later; Python 3.11 is no longer supported.
- Use module imports for `Quantity`, `Unit`, `Dimension`, and `DIMENSIONLESS`; the former top-level exports and `unitx.utils` namespace are no longer provided.
- Remove `Unit` length and special truth-value behavior. Check formal identity with `unit == Unit('')`, or physical dimensions with `unit.is_dimensionless()`.
- Compare and hash scalar quantities using exact rational coefficients and remaining irrational unit factors. Finite floats retain their stored binary rational values for equality, without a tolerance.
- Convert finite Decimal magnitudes exactly to Fraction for scaling and arithmetic, independent of the Decimal precision context; numerical results can change type.
- Normalize quantity exponents once for both the magnitude and unit, with approximate native evaluation for nonintegral magnitude powers.

### Fixed

- Interpret plain numbers as unscaled values when adding to or subtracting from scaled dimensionless quantities.
- Preserve the original magnitude and type when multiplying or dividing a quantity by a Unit.
- Cancel common irrational factors before evaluating quantity ordering, retaining exact ordering when the ratio is rational.

### Added

- Support Fraction and Decimal scalar multiplication and division alongside int and float.
- Distribute inline type information with `py.typed` and check numerical and custom scalar contracts with Pyright.
- Add a release workflow that validates version tags, checks distributions and the installed wheel, and publishes GitHub releases through PyPI Trusted Publishing.
- Document construction, conversion, unit syntax, numerical rules, supported units, and migration from 0.3.0. NumPy integration remains planned and is not supported in this release.

## [0.0.7] - 2026-10-15

- Add `Unit` with string construction for compound unit expressions, including registered symbols, names, prefixes, and aliases.
- Support unit multiplication and division, rational powers, and cancellation while preserving exact rational exponents.
- Expose unit symbols, coherent SI scaling factors, physical dimensions, and dimensionless checks; dimensionless expressions can retain nontrivial scaling factors.
- Define equality and hashing by formal unit factors regardless of insertion order, preserving distinct prefixed unit identities.
- Provide readable Unicode display and reconstructible `repr` strings for compound units and rational powers.
- Add executable usage examples and regression tests for alias normalization, equality and hashing, exact and approximate factors, dimensions, power cancellation, conversion round trips, state preservation, and invalid inputs.

## [0.0.6] - 2026-10-15

- Add `UnitParser` to parse unit expressions into `Monomial[SingleUnit]`, resolving symbols, names, prefixes, and aliases through `SingleUnit`.
- Support explicit and implicit multiplication, division, parentheses, and exact rational powers using adjacent numbers, `^`, or `**`, including decimal exponents and parenthesized fractions.
- Support empty and dimensionless expressions, the literal identity `1` in products such as `m 1/s`, and single-factor leading reciprocals such as `/s` and `/(m s)`.
- Cache successful parse results so repeated calls return the same object, while allowing failed attempts to be retried.
- Add `UnitSyntaxError` with diagnostics for malformed expressions, invalid numeric bases, zero fraction denominators, and redundant suffixes.
- Use spaces for multiplication in ambiguous prefix warning suggestions, matching the supported expression syntax.
- Add grammar documentation, executable examples, and regression tests for exact parsing, whitespace, diagnostics, caching, retries, and warning behavior and attribution.

## [0.0.5] - 2026-10-15

- Add built-in SI prefix data and initial unit definitions for `s`, `m`, `g`, `A`, `K`, `mol`, and `cd`, plus the dimensionless unit, with exact factors, dimensions, and prefix restrictions.
- Add immutable, hashable `SingleUnit` with symbol and name parsing, canonical identity, dimension and factor queries, prefix inspection and removal, copying, and serialization.
- Support symbol and name aliases, including `meter`, micro prefix variants `u` and `μ`, and the uppercase `K` alias for kilo, while preserving canonical symbols and names.
- Add `UnitSymbolError` for invalid unit inputs and `UnitSymbolWarning` for potentially misleading prefix aliases, with warnings attributed to user call sites.
- Add `Factor` hashing consistent with equality to rational numbers and individual `SymbolicConstant` instances.
- Add regression tests for unit definitions, alias normalization, exact factors, immutability, serialization, and warning behavior, filtering, and attribution.

## [0.0.4] - 2026-10-15

- Add `Factor` for exact positive scaling expressions with rational exponents, multiplicative arithmetic, and exact rational or approximate float evaluation, along with predefined `ONE`, `TWO`, and `TEN` instances.
- Add immutable `SymbolicConstant` with symbolic identity and float approximations, along with the predefined `PI` constant.
- Support factor decomposition, compact formatting, and magnitude scaling that converts finite `Decimal` magnitudes exactly before multiplication.
- Provide built-in bounded integer factorization and primality testing, optional SymPy support, and `IncompleteFactorWarning` for unresolved factorization or unproven primality.
- Display negative `Monomial` powers as denominator terms, with parentheses around multiple denominator factors.
- Attribute numerical warnings to user call sites without manually passing warning stack levels.
- Raise the minimum supported Python version from 3.11 to 3.12.
- Extend CI to test both built-in and SymPy backends on Python 3.12–3.15.

## [0.0.3] - 2026-10-14

- Add immutable, hashable `Monomial` for formal products with rational exponents.
- Support multiplication, division, rational powers, inverses, and numerator/denominator decomposition.
- Provide explicit exponent queries, component views, Unicode formatting, and exact representation round trips.
- Add tests for monomial arithmetic, normalization, immutability, hashing, copying, and serialization.

## [0.0.2] - 2026-10-13

- Add immutable physical dimensions with rational exponents and dimensionless normalization.
- Support dimension arithmetic, exponent aliases, and exact representation round trips.
- Add numerical conversion utilities with inexact float warnings and Unicode exponent formatting.
- Add tests for dimensions, numerical conversion, formatting, copying, and serialization.

## [0.0.1] - 2026-10-13

- Project initialization and setup.
