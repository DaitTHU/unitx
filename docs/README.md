# unitx documentation

Use the [project README](../README.md) for installation and a first calculation.
These guides describe the current source tree; select a release tag when
reading about an older version.

## Choose a guide

| Task | Guide |
| --- | --- |
| Construct, combine, convert or compare quantities | [Units and quantities](units.md) |
| Find units, constants, prefixes and Unicode spellings | [Catalogue](catalogue.md) |
| Define an exact scale or register a new unit | [Custom units](custom-units.md) |
| Understand Fraction, Decimal, float and exact equality | [Scalar numerical behavior](scalars.md) |
| Work with absolute temperatures or logarithmic readings | [Points and differences](affine.md) |
| Call unit-aware mathematical functions or read a clock | [Standard-library enhancements](stdlib.md) |
| Calculate with arrays, masks, reductions and output buffers | [NumPy support](numpy.md) |
| Format text or export LaTeX, HTML and siunitx | [Display and formatting](display.md) |
| Understand annotations and NumPy inference limits | [Typing](typing.md) |
| Update code written for an earlier release | [Migration notes](migration.md) |
| Run scientific examples | [Physics notebooks](../examples/README.md) |
| Test or contribute a change | [Contributing](../CONTRIBUTING.md) |

For a first read, follow units and quantities, then the catalogue. Read the
scalar numerical guide before relying on concrete result types or exact
comparisons. Array users should also read NumPy's storage and conversion rules.

## Choose an object

| Object | Represents | Example |
| --- | --- | --- |
| `Unit` | A formal multiplicative unit | `si.m`, `Unit('kg m/s2')` |
| `ScaledUnit` | A unit product with an additional exact scale | `si.m * Factor(2)` |
| `Quantity[T]` | A magnitude of type T paired with a linear unit | `3 * si.m`, `5 * si.K` |
| `AffineUnit` | A coordinate origin and difference scale | `si.degC`, `si.dBm` |
| `AffineQuantity[T]` | An absolute reading in a coordinate | `20 @ si.degC`, `30 @ si.dBm` |

Formal unit identity, physical dimension and conversion scale are distinct.
For example, metres and centimetres are different Unit objects, while
`1 * si.m == 100 * si.cm`. Points retain their class after conversion, even
to a zero-origin coordinate such as K or W.

## Imports and boundaries

The package root exports `Quantity`, `AffineQuantity`, `Unit`, `AffineUnit`,
`Dimension`, `DIMENSIONLESS`, `one`, `ten`, `pi`, and the `si`, `math`, and
`time` modules. Import the unit and constant namespaces with
`from unitx.si import si, SI`. `Factor`, `ScaledUnit` and other building blocks
remain available from their own modules; names beginning with `_` are internal.

Unit algebra is exact within its documented symbolic/factorization model.
Numerical magnitudes follow their own arithmetic and backend rules. Neither
symbolic pi nor Decimal input implies arbitrary-precision numerical functions.
Type annotations describe magnitude types; physical dimensions are checked
at runtime.

NumPy and SymPy are optional. Core scalar use requires neither. Supported
NumPy inputs are plain numeric arrays/scalars, not arbitrary duck arrays,
masked arrays, or object arrays. The explicit `.value` and `.value_in(unit)`
exits let callers choose where to hand numerical data to another library.
