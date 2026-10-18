# Changelog

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
