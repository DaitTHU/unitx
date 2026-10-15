# Changelog

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
