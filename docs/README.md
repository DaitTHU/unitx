# unitx documentation

Use the [project README](https://github.com/DaitTHU/unitx/blob/main/README.md)
for installation and a first calculation.
These guides describe the current source tree; select a release tag when
reading about an older version.

## Choose a guide

| Task | Guide |
| --- | --- |
| Construct, combine, convert or compare quantities | [Units and quantities](units.md) |
| Choose and parse a unit catalogue | [Unit catalogues](catalogues/README.md) |
| Find SI units, constants, prefixes and Unicode spellings | [SI reference](catalogues/si.md) |
| Find Imperial units and their exact conventions | [Imperial reference](catalogues/imperial.md) |
| Calculate with CGS units and convert Gaussian electromagnetism | [CGS reference](catalogues/cgs.md) |
| Define an exact scale or register a new unit | [Custom units and registries](custom-units.md) |
| Understand Fraction, Decimal, float and exact equality | [Scalar numerical behavior](scalars.md) |
| Work with absolute temperatures | [Points and differences](affine.md) |
| Work with logarithmic differences and readings | [Logarithmic quantities](logarithmic.md) |
| Call unit-aware mathematical functions or read a clock | [Standard-library enhancements](stdlib.md) |
| Calculate with arrays, masks, reductions and output buffers | [NumPy support](numpy.md) |
| Format text or export LaTeX, HTML and siunitx | [Display and formatting](display.md) |
| Save and restore scalar quantities as JSON-compatible records | [Scalar data exchange](serialization.md) |
| Understand annotations and NumPy inference limits | [Typing](typing.md) |
| Update code written for an earlier release | [Migration notes](migration.md) |
| Run scientific examples | [Scientific examples](https://github.com/DaitTHU/unitx/blob/main/examples/README.md) |
| Test or contribute a change | [Contributing](https://github.com/DaitTHU/unitx/blob/main/CONTRIBUTING.md) |
| Understand model and implementation boundaries | [Architecture](architecture.md) |

For a first read, follow units and quantities, then the catalogue. Read the
scalar numerical guide before relying on concrete result types or exact
comparisons. Array users should also read NumPy's storage and conversion rules.

## Choose an object

| Object | Represents | Example |
| --- | --- | --- |
| `Unit` | A formal unit product with an optional exact scale | `si.m`, `Unit('m', scale=Factor(2))` |
| `Quantity[T]` | A magnitude of type T paired with a linear unit | `3 * si.m`, `5 * si.K` |
| `AffineUnit` | A linear coordinate origin and difference scale | `si.degC` |
| `LogUnit` | A logarithmic physical reference, scale and convention | `si.dBm` |
| `AffineQuantity[T]` | A reading in a linear affine coordinate | `20 @ si.degC` |
| `LogQuantity[T]` | A reading in a logarithmic coordinate | `30 @ si.dBm` |

Formal unit identity, physical dimension and conversion scale are distinct.
For example, metres and centimetres are different Unit objects, while
`1 * si.m == 100 * si.cm`. Points retain their class after conversion, even
to a zero-origin coordinate such as K.

## Imports and boundaries

The package root exports `Quantity`, `AffineQuantity`, `Unit`, `AffineUnit`,
`LogUnit`, `LogQuantity`, `Dimension`, `DIMENSIONLESS`, `UnitCatalog`, `Registry`, `one`, `ten`, `pi`, the `si`,
`imperial`, `cgs`, `math`, and `time` modules. Import the unit and constant namespaces with
`from unitx.si import si, SI`. `Factor` and other building blocks
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
