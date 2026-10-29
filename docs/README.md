# unitx documentation

Use the [project README](https://github.com/DaitTHU/unitx/blob/main/README.md)
for installation and a first calculation.
These guides describe the current source tree; select a release tag when
reading about an older version.

## Choose a guide

| Task | Guide |
| --- | --- |
| Construct, combine, convert or compare quantities | [Units and quantities](units.md) |
| Expand SI base units, combine like units, choose prefixes or obtain a value–unit pair | [Unit representations](representation.md) |
| Choose and parse a unit catalogue | [Unit catalogues](catalogues/README.md) |
| Find SI units, constants, prefixes and Unicode spellings | [SI reference](catalogues/si.md) |
| Calculate with bit, byte, nat, hartley and binary multiples | [Information units](catalogues/information.md) |
| Find Imperial units and their exact conventions | [Imperial reference](catalogues/imperial.md) |
| Calculate with CGS units and convert Gaussian, ESU, EMU and H–L electromagnetism | [CGS reference](catalogues/cgs.md) |
| Define an exact scale or register a new unit | [Custom units and registries](custom-units.md) |
| Understand Fraction, Decimal, float and exact equality | [Scalar numerical behavior](scalars.md) |
| Work with absolute temperatures | [Points and differences](affine.md) |
| Work with logarithmic differences and readings | [Logarithmic quantities](logarithmic.md) |
| Call unit-aware mathematical functions or read a clock | [Standard-library enhancements](stdlib.md) |
| Compute scalar means, dispersion and paired statistics | [Scalar statistics](statistics.md) |
| Calculate with arrays, masks, reductions and output buffers | [NumPy support](numpy.md) |
| Use Bessel functions and Simpson integration | [SciPy support](scipy.md) |
| Plot quantities with automatic axis conversion | [Matplotlib plots](matplotlib.md) |
| Format text or export LaTeX, HTML and siunitx | [Display and formatting](display.md) |
| Choose tolerances, readings, spectral transforms or sampling scales | [Scientific boundaries](scientific-boundaries.md) |
| Save numeric arrays with complete coordinate metadata | [Array storage](array-storage.md) |
| Save and restore scalar quantities as JSON-compatible records | [Scalar data exchange](serialization.md) |
| Use finite typed NumPy integration, gradient and allocation entries | [Typed NumPy entries](typing.md#finite-typed-numpy-entries) |
| Understand annotations and NumPy inference limits | [Typing](typing.md) |
| Check public API, compatibility and data update promises | [Compatibility policy](compatibility.md) |
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
`imperial`, `information`, `cgs`, `math`, `statistics`, and `time` modules. Import the unit and constant namespaces with
`from unitx.si import si, SI`. `Factor` and other building blocks
remain available from their own modules; names beginning with `_` are internal.
Public magnitude annotations live in `unitx.typing`. The supported imports,
subclass boundary and 1.x promises are defined in the [compatibility policy](compatibility.md).

Unit algebra is exact within its documented symbolic/factorization model.
Numerical magnitudes follow their own arithmetic and backend rules. Neither
symbolic pi nor Decimal input implies arbitrary-precision numerical functions.
Type annotations describe magnitude types; physical dimensions are checked
at runtime.

NumPy, SciPy and SymPy are optional. Core scalar use requires neither. Supported
NumPy inputs are plain numeric arrays/scalars, not arbitrary duck arrays,
masked arrays, or object arrays. The explicit `.value` and `.value_in(unit)`
exits let callers choose where to hand numerical data to another library.
For coherent dimensionless numerical values, `q.n` is an explicit shorthand
for `q.value_in(si.one)` with the same complete compatibility checks.
