# Changelog

## [Unreleased]

## [0.18.0] - 2026-10-29

### Added

- Add the independent `unitx.information.information` catalogue with bit/shannon,
  eight-bit byte, nat and hartley. Information is dimensionless with natural-log
  normalization: bit = ln(2) nat and hartley = ln(10) nat. Keep default B/b
  spellings bound to bel/barn.
- Add exact IEC binary prefixes Ki through Qi, enabled only by explicit unit
  permissions. Extend the existing `prefixable` argument with `PrefixFamily`
  and per-spelling mappings, retaining bool/string/tuple decimal shorthand.
  Keep permissions immutable and enforce them in parsing, output spelling,
  compact representation and reconstruction. Select decimal/binary candidate
  collections separately and keep common decimal permission checks direct.
  Ordinary metres reject Kim.
- Add `si.turn`/`si.cycle` as one exact 2*pi-radian definition, `si.ppm` as an
  exact 1e-6 ratio, and jansky `si.Jy` with decimal prefix shortcuts.
- Expose symbolic `ln2` from `unitx.factor`, preserving exact cancellation in
  information-unit conversions without changing the scalar JSON schema.

## [0.17.8] - 2026-10-29

### Added

- Add unit-aware `math.sinpi` and `math.cospi` with period reduction before
  numerical pi multiplication, exact integer/half-integer nodes and scalar/NumPy
  support, including scaled dimensionless inputs.
- Add finite typed `unitx.numpy` entries for wrapped trapezoid integration,
  gradients and zeros/ones/empty/full-like allocations, reusing NumPy dispatch.

### Changed

- Preserve known NumPy floating dtypes through all real unary math adapters.
- Align scalar float `to` and `value_in` inference with `n` and scientific float
  arithmetic, retaining exact int/Fraction/Decimal runtime behavior and documenting
  the distinction between float annotations and guaranteed float magnitudes.
- Use `unitx.typing.__all__` as the explicit public re-export declaration,
  removing redundant same-name import aliases.

## [0.17.7] - 2026-10-29

### Changed

- Reduce repeated NumPy input-recognition and ufunc-dispatch allocations with
  fixed function groups and an empty-keyword path, preserving numerical,
  protocol-override and staged-output behavior. Record same-machine timing
  samples for common operations and complete scientific fitting.
- Reuse exact identity factors and cancellations; derive empty and single-term
  unit metadata directly and reuse evaluated factors at NumPy scaling boundaries.
- Use native Python argument binding for direct array methods, with explicit
  forwarding to the existing NumPy handlers instead of runtime signature binding.
- Keep powered-product identities inside Factor and Dimension, normalize Factor
  arithmetic operands once, and derive Dimension projections and display views
  directly from sparse monomials while retaining canonical axis order.
- Reuse normalized rational exponents and sparse views, and remove a redundant
  coordinate check after linear-operand validation.

## [0.17.6] - 2026-10-28

### Added

- Add validated float64 comparisons with NumPy, Pint, Astropy and unyt, warmed
  scalar/array/fit profiling commands, and a reproducible performance study.

### Changed

- Accumulate composite-unit factors and dimensions in private mutable monomials
  and freeze each product once, preserving exact scale and formal identity.
- Reuse one immutable dimensionless unit throughout the NumPy adapter instead
  of repeatedly constructing equivalent empty units.

## [0.17.5] - 2026-10-28

### Added

- Add typed `min`, `max` and `var` methods for linear, affine and logarithmic
  arrays, reusing compatible initials, squared difference units and staged out.
- Support nonempty constant exponent arrays for dimensional `power` and
  `float_power`, preserving broadcasting and checking every exponent exactly.
- Expose magnitude annotations through `unitx.typing` without importing NumPy.
- Document the 1.x public API, data compatibility, constant updates and deprecation
  policy, plus executable fitting, array storage and numerical-boundary recipes.
- Validate Matplotlib scatter/fill paths and document explicit point-errorbar
  conversion; record scalar, array-write and complete-fit performance baselines.

### Fixed

- Evaluate derived catalogue constants in an isolated 28-digit Decimal context,
  independent of caller precision, rounding, traps and exponent bounds, without
  changing the caller context. Subsequent user arithmetic keeps its own context.
- Normalize exact scalar point-conversion results at the NumPy boundary, so
  integer Kelvin replacements/selections in Celsius arrays do not create object
  dtypes or fail numeric fill operations.

## [0.17.4] - 2026-10-28

### Added

- Add typed `transpose` and `squeeze` methods for Quantity, AffineQuantity and
  LogQuantity, preserving full units, coordinate families and NumPy views.
- Add `unitx.math.conj` and complex inputs to `unitx.math.exp` for typed complex
  phases, preserving complete units and the existing NumPy dispatch rules.

### Changed

- Expose `n` as an ordinary read-only property and retain float inference for
  scientific scalar formulas, with exact runtime conversion rules unchanged.
- Preserve known NumPy dtypes through the supported `unitx.math` phase functions,
  mixed real/complex algebra, absolute values and same-dtype augmented products.
- Retain ndarray results and numeric dtypes for slices, boolean indices,
  ellipses, new axes and tuples starting with one of these indices.
- Infer complex64/complex128 for float32/float64 multiplication and division
  with Python complex scalars, including reflected operations.
- Infer concrete NumPy dtype and container types for explicit `astype` calls,
  and NumPy scalar types for complete sum/mean reductions and their numerical
  exits, reducing the need for consumer casts and numerical conversions.

## [0.17.3] - 2026-10-27

### Added

- Support `diff` and `gradient` on affine/logarithmic point arrays, returning
  differences and using coordinate difference units for derivatives. Align
  point boundaries explicitly; retain the original object for `diff(n=0)`.
- Document reusable trapezoidal weights using `np.gradient` and an endpoint
  correction.
- Add `UnitOperationError` for point/difference semantic violations and its
  `ImplicitConversionError` subclass for silent array unit removal. Both remain
  catchable as TypeError and UnitxError; native numerical errors stay native.

### Changed

- Delegate direct array methods through a private lazy decorator, retaining
  explicit signatures, method defaults and the existing NumPy unit/output rules.
- Retain known floating/complex NumPy dtypes in linear conversions, supported
  arithmetic and default sum/mean results, including possible NumPy scalars.
  Preserve numeric dtypes in indexing/iteration; narrow exact Fraction
  arithmetic and complex scalar result domains.

### Fixed

- Update the generator annotation on registry definition scopes to avoid the
  deprecated Iterator overload of `contextmanager` in current type stubs.

## [0.17.2] - 2026-10-27

### Added

- Add direct `reshape`, `ravel`, `copy`, `sum`, `mean` and `std` methods for
  NumPy-backed quantities, retaining typed wrappers and fixed-unit output rules.
  Shape/copy methods preserve known ndarray dtypes; reductions preserve the
  concrete output type when `out` is supplied.
- Add corresponding point methods except sum: shape/copy and mean retain the
  coordinate family, while standard deviation returns a linear difference.

### Changed

- Move Quantity's complete static interface into the bundled `quantity.pyi`,
  with checked runtime implementation annotations and no duplicated overloads.
  Preserve scalar inference without NumPy, the Decimal factory, components,
  representations and existing subclass behavior.
- Clarify dimensional power/cumulative-product and heterogeneous covariance
  boundaries in the NumPy guides, and execute the direct-method guide in tests.

### Fixed

- Allow `np.trapezoid` with bare numeric array samples and Quantity `dx` to
  finish its native arithmetic through the direct Quantity sum method.

## [0.17.1] - 2026-10-27

### Added

- Add read-only `q.n` as an explicit dimensionless numerical exit, equivalent
  to `q.value_in(si.one)` with full scale and compatibility checks.
- Add `Quantity.to_base()` for coherent SI base products and `simplify()` for
  combining like-dimensional factors without choosing named derived units.
- Add `Quantity.compact()` for readable SI prefixes, including negative and
  rational powers and one array-wide prefix selected from finite magnitudes.
- Add `as_pair()` to linear quantities, affine points and logarithmic readings,
  retaining the actual magnitude and complete unit or coordinate without copying.
- Document representation choices and verify their installed scalar/array
  runtime and type consumers without NumPy and with minimum/current dependencies.

### Fixed

- Explain failed conversions between units with matching physical dimensions
  but different logarithmic degrees, identifying both units instead of showing
  two apparently identical dimensions for conversions such as dB to one.

## [0.17.0] - 2026-10-27

Stabilize the documented NumPy integration developed throughout 0.16.x, with
complete scientific workflows and minimum/current installed compatibility.
The supported model remains one unit per array with explicit numerical exits.

### Fixed

- Correct NumPy selection and statistics guides to describe supported extrema
  methods and dimensionless multiply.accumulate/reduceat.

### Changed

- Check installed-wheel runtime and types with minimum and current NumPy,
  Matplotlib and SciPy in CI, preserving scalar checks without NumPy.
- Add complete coordinate-statistics, circuit-solve and sampled-spectrum
  regression flows across units and output-buffer boundaries.

## [0.16.13] - 2026-10-27

### Added

- Finish the NumPy compatibility sweep: block/choose/select, nan_to_num,
  ediff1d, cumulative_sum/prod, array equality/truth and complex/infinity
  predicates, storage sharing, resize/trim/unstack/astype, i0, binary floating
  helpers, strict linalg Array API functions and version-available matvec/vecmat.
  Document the inventory audit, version gates and explicit remaining boundaries.
- Add fixed-unit NumPy copyto, put, put_along_axis, putmask, place and
  fill_diagonal, plus applicable ufunc reduce, accumulate, reduceat, outer and
  at methods. Stage conversion, casts and repeated-index updates before writes.
- Complete NumPy allocation and grid support with full_like, compatible-unit
  linspace/geomspace, dimensionless logspace, independent-unit meshgrid,
  built-in padding modes and sliding-window views. Preserve point coordinates,
  difference steps, native dtype and storage behavior.

## [0.16.12] - 2026-10-26

### Added

- Add the NumPy-free `unitx.statistics` scalar adapter with mean, fmean,
  sample/population variance and standard deviation, covariance and linear/ranked
  correlation. Preserve units, point/difference semantics and native scalar
  numerical domains; document the explicit floating statistical kernels.
- Support NumPy ptp, cov/corrcoef, histogram families and bin edges, digitize,
  weighted bincount, nancumsum and dimensionless cumulative products. Preserve
  coordinate edges, weight units and inverse-coordinate density units.
- Support precomputed means and correction on NumPy std/var and NaN variants,
  with compatible quantity/point conversion and staged outputs.

## [0.16.11] - 2026-10-26

### Added

- Support NumPy's complex, real and Hermitian FFT families for linear
  quantities, preserving complete units, native transform options and staged
  output writes. Add `fftshift`/`ifftshift` for quantities and points.
- Document and verify `fftfreq`/`rfftfreq` with Quantity sampling intervals:
  their existing arithmetic produces reciprocal-unit frequency axes without
  an additional public adapter.

## [0.16.10] - 2026-10-26

### Added

- Support `np.sinc` for linear dimensionless quantities, normalizing their
  complete unit scale before NumPy's normalized sinc calculation.
- Support NumPy QR, Cholesky and singular-value decompositions, plus
  `eig`/`eigh`/`eigvals`/`eigvalsh`. Preserve per-component units and NumPy's
  named result fields; support stacked/complex inputs and native options.
  QR supports reduced, complete and r modes; packed raw arrays cannot express
  one uniform physical unit.

## [0.16.9] - 2026-10-26

### Added

- Support `np.linalg.norm` for linear quantities, retaining the complete input
  unit for vector and matrix norms. Vector `ord=0` returns a dimensionless
  nonzero count. Preserve NumPy's order, axis, dtype and keepdims behavior;
  reject affine/logarithmic points.
- Complete the roadmap's linear-system and matrix-measure group with `solve`,
  `tensorsolve`, `lstsq`, `inv`, `pinv`, `tensorinv`, `matrix_power`, `det`,
  `vector_norm`, `matrix_norm`, `cond` and `matrix_rank`. Preserve each result's
  unit, including least-squares residuals/singular values, and support explicit
  absolute and dimensionless relative cutoff quantities.

### Fixed

- Add NumPy-magnitude `Quantity.real` and `Quantity.imag` properties with native
  component result types and views. NumPy's structural component annotations
  now infer `np.real(q)` and `np.imag(q)` as Quantity objects with the matching
  real dtype, including complex64/complex128 arrays and NumPy scalars.
- Clarify the remaining NumPy function-dispatch inference limits and why
  general numerical array results can include NumPy scalars or promoted dtypes.

## [0.16.8] - 2026-10-26

### Added

- Support NumPy vector, tensor and discrete products: `dot`, `inner`, `outer`,
  `vdot`, `vecdot`/`linalg.vecdot`, `tensordot`, `einsum`, `einsum_path`, `kron`,
  `cross`, `trace`, `convolve`, `correlate` and `linalg.multi_dot`. Preserve
  product units, native conjugation and applicable staged output writes.
- Support `scipy.special.jv` directly for dimensionless Quantity inputs and
  add the explicit `unitx.scipy.simpson` integration adapter. Add the optional
  `scipy` extra and minimum/current SciPy validation; core imports remain free
  of NumPy/SciPy runtime dependencies.

### Fixed

- Infer integer magnitudes for small nonnegative literal powers, and float
  magnitudes for small negative literal powers, using Quantity overloads.
  Computed Decimal constants such as `SI.re` now retain their numerical type
  through their original formulas without constant-level casts or numerical
  changes. General integer exponents retain a wider result domain.

## [0.16.7] - 2026-10-26

### Added

- Support NumPy hyperbolic and inverse hyperbolic functions, `exp2`,
  `logaddexp/logaddexp2`, `fmin/fmax`, stored-coordinate rounding, `sign`,
  `copysign`, compatible-unit remainders, floor division and `divmod`.
  Preserve staged single- and multiple-output writes.
- Support `real_if_close`, `angle`, `unwrap` and degree/radian conversions
  with explicit phase units, plus `ones_like` retaining units or coordinates.
- Permit built-in float/Decimal mixing in linear Quantity `+`, `-`, `*`, `/`
  and augmented arithmetic. Convert floats via `Decimal(repr(x))`, align and
  calculate under the active Decimal context, then return float. Document the
  implicit precision downgrade; pure Decimal arithmetic and exact comparisons
  retain their existing contracts.
- Infer float addition/subtraction results for ordinary float scientific
  formulas, while preserving the wider concrete integer/Fraction paths and
  documenting the hidden-integer inference exception.

## [0.16.6] - 2026-10-25

### Added

- Support NumPy sorting, partitioning, searching, set operations, `unique` and
  its Array API result variants, and nonzero indices/counts. Retain complete
  units or point coordinates for values and return bare indices/masks/counts.
- Support NumPy `zeros_like`, `empty_like`, `real` and `imag` for scientific
  array workflows, preserving units and native dtype/storage behavior.
- Add `Quantity.astype(dtype)` for explicit magnitude conversion without
  changing units. Refine float-compatible scalar multiplication, division and
  integer-power annotations so ordinary formulas need fewer casts.
- Add the optional Matplotlib adapter `unitx.matplotlib.register()` for direct
  `ax.plot(t, q)`, compatible axis conversion, default unit labels and unit
  switching. Core imports retain their optional-dependency boundary.

### Fixed

- Scalar quantities and points raise TypeError for iteration and len, allowing
  standard iterable detection and scalar plotting limits to work correctly.

## [0.16.5] - 2026-10-25

### Added

- Support NumPy `hstack`, `vstack`, `dstack`, `column_stack`, `append` and
  `insert` with compatible unit or point-coordinate alignment. Support `split`,
  `array_split`, `hsplit`, `vsplit`, `dsplit`, `take`, `take_along_axis`,
  `compress`, `extract` and `delete`, preserving units, coordinates and NumPy's
  storage rules. `take` and `compress` stage output writes before mutation.
- Support NumPy `diagonal`, `diag`, `diagflat`, `tril` and `triu`. Filled zeros
  are readings in the retained unit or coordinate, including for points.

## [0.16.4] - 2026-10-25

### Added

- Support NumPy `atleast_1d/2d/3d`, `broadcast_to`, `broadcast_arrays`,
  `matrix_transpose` (including `linalg.matrix_transpose`), `flip/fliplr/flipud`,
  `rot90`, `roll`, `repeat`, `tile` and `copy`. Preserve complete units or point
  coordinates, NumPy's views/copies, writeability and warnings. Multiple inputs
  retain their own units or coordinates independently; bare inputs stay bare.
- Support NumPy `shape`, `ndim` and `size` as bare array metadata, including
  scalar quantities. Document and verify the existing `permute_dims` alias.

## [0.16.3] - 2026-10-25

### Added

- Support NumPy `squeeze`, `expand_dims`, `moveaxis`, `swapaxes`, `rollaxis`
  and `ravel`, preserving complete units or point coordinates. Retain NumPy's
  axis rules, dtype, shared views, read-only storage and necessary copies.

## [0.16.2] - 2026-10-25

### Added

- Support NumPy `nanmin`, `nanmax`, `nanargmin` and `nanargmax`, preserving
  units or point coordinates for extrema and returning bare indices. Retain
  NumPy's all-NaN warnings/errors and stage output writes before mutation.
- Support `nansum` with the existing sum unit, initial and mask rules, and
  `nanprod` with unit powers determined by selected non-NaN element counts.
  Dimensional product slices require equal counts; dimensionless products
  normalize to coherent values and may have different counts.

## [0.16.1] - 2026-10-25

### Added

- Support NumPy `nanmean`, `nanmedian`, `nanstd`, `nanvar`, `nanquantile` and
  `nanpercentile`, retaining the corresponding statistical unit and coordinate
  rules. Omit NaN samples with NumPy's dtype, mask, weight and warning behavior;
  preserve staged output writes, including when warnings become exceptions.
  Weighted NaN quantiles support one integer axis or `axis=None`, with full-shape
  weights for multidimensional samples, due to NumPy 2.1/2.5 kernel limitations.

## [0.16.0] - 2026-10-25

### Added

- Support NumPy `average`, `quantile` and `percentile` for linear Quantities,
  preserving sample units and normalizing dimensionless weights/probabilities.
  `average(returned=True)` includes a dimensionless Quantity weight sum;
  weighted quantiles require `method="inverted_cdf"` and stage `out` writes.
- Support nonperiodic NumPy `interp` for linear quantities with finite,
  strictly increasing one-dimensional sample coordinates. Align query and
  boundary units, preserve sample-value units, and compose with existing
  integration, difference and gradient functions.

## [0.15.2] - 2026-10-25

### Added

- Support NumPy three-argument `where`, `clip` (including `min`/`max` bounds),
  and elementwise `minimum`/`maximum` on linear quantities. Align compatible
  units or point coordinates, preserve wrappers, and stage masked output writes.
  Bare values remain dimensionless, including zero.

### Fixed

- Require compatible coordinate conventions when selecting point extrema,
  including `min/max` initial values. Power and amplitude logarithmic readings
  require explicit conversion before selection together.

### Documentation

- Complete the NumPy support matrix and boundary coverage for `min/max`
  (including `amin/amax`), selection, clipping, and NumPy stub limitations.
- Split detailed NumPy contracts into selection, statistics, calculus and
  output-buffer guides, retaining the overview and existing section anchors.

## [0.15.1] - 2026-10-25

### Changed

- Rename `UnitCatalog.parse(expression)` to `UnitCatalog.unit(expression)` across
  built-in and custom catalogues. Remove the old method; keep `Registry.parse()`
  and `UnitParser.parse()` unchanged. Update examples, typing consumers and
  migration guidance.

### Added

- Extend `unitx.cgs` with independent `esu`, `emu` and `hl` catalogues, exact
  mechanical unit definitions, formula coefficients and explicit SI correspondence
  for the same eight electromagnetic kinds as Gaussian. Share ESU electric and
  EMU magnetic definitions with Gaussian; represent Heaviside–Lorentz units with
  descriptive attributes and documented library-specific parser spellings.

## [0.15.0] - 2026-10-25

### Added

- Add `unitx.cgs` with a mechanical `cgs` catalogue and a `gaussian` catalogue
  inheriting its units with an independent parser. Gaussian electromagnetic
  definitions and `gaussian.convert()` for eight SI/Gaussian kinds share one
  namespace. Read-only `si.em` and `gaussian.em` objects group the formula
  coefficients with their defining equations and data sources.
  Physical correspondence uses the library's CODATA 2022 constants; ordinary
  dimension-strict conversion and arithmetic remain unchanged.

### Internal

- Group numerical, formatting and electromagnetic support in private packages.
  Preserve public module paths, the `unitx._numerics` scalar entry point, and
  lazy optional NumPy loading.

## [0.14.4] - 2026-10-24

### Fixed

- Correct the Impedance example's mixed-addition result annotations and retain
  its specialized scalar `+=` result types. Document the subclass boundary and
  check it with runtime and static consumers.

### Documentation

- Host the MkDocs guides on GitHub Pages with automatic deployment from main.
  Keep Read the Docs deployment pending transfer of the project name.

## [0.14.3] - 2026-10-24

### Added

- Allow `si.dB(reference, kind=...)` to construct a logarithmic coordinate.
  Named watts and volts, including SI prefixes, supply conventional defaults;
  common references reuse dBm, dBW and dBV. Other units require an explicit kind.
- Add a standalone Impedance subclass example with series addition, parallel
  `//` composition, unit conversion and a complex RLC circuit calculation,
  plus a notebook that traces its frequency response on a Smith chart.

### Internal

- Consolidate static consumers under scalar and NumPy typing fixtures, checking
  actual notebook source instead of duplicate blackbody examples. Keep the
  NumPy 2.5 asarray regression separately gated and exclude static fixtures from
  pytest collection.
- Complete runtime-test signature annotations and check them in the existing CI
  and release lint steps.

## [0.14.2] - 2026-10-24

### Documentation

- Add a MkDocs site with pinned build requirements, Mermaid diagrams,
  Read the Docs configuration and strict CI checks for internal links and anchors.
  Hosting can use an available project name; no documentation domain is assumed.
- Fix the README's logarithmic-guide link for package-index rendering.

### Internal

- Execute the README's Python code blocks directly and check the blackbody
  plot's units, curve count and numerical data.
- Extend installed-wheel runtime and type consumers with mathematical functions,
  integer clock readings, duration arithmetic and datetime precision checks.
- Add reproducible scalar benchmarks for parsing, conversion, comparison and
  exact factors, recording the environment and timing parameters as JSON.

## [0.14.1] - 2026-10-24

### Documentation

- Clarify concurrent Registry reads, application synchronization during
  registration, the absence of a freeze/snapshot API, and custom-definition
  sharing within and across processes.

## [0.14.0] - 2026-10-24

### Added

- `Registry` provides independent parsing environments, atomic `include(*units)`
  and explicit `with registry.defining():` blocks with configurable prefix defaults.
  `UnitCatalog` collects read-only unit attributes and provides catalogue parsing.
- `unitx.imperial` supplies British Imperial, historical English and US customary
  catalogues with independent spellings and exact documented scales, including
  US survey feet, thermochemical Btu, circular mil and conventional water/mercury
  pressure units. Importing a catalogue does not change default parsing.

### Changed

- **Breaking:** `Unit` now stores its complete exact extra scale; `ScaledUnit` is
  removed. Construct scaled units with `Unit(base, scale=...)` or Factor operands.
- **Breaking:** `.define()` creates standalone definitions outside an explicit
  Registry block. Extend default parsing explicitly with `default_registry`.
  Formal identity follows shared definition objects rather than symbols;
  independent same-name definitions remain distinct. Import `UnitDefinition`
  and `PrefixPolicy` from `unitx.unitdefinition`; registry table exports are removed.
- **Breaking:** `LogUnit` and `LogQuantity` replace logarithmic uses of AffineUnit
  and AffineQuantity. Point `.to()` stays within its coordinate family; use
  `.to_quantity()` and `LogQuantity.from_quantity()` at physical boundaries.
- **Breaking:** Dimension equality includes the internal logarithmic axis.
  Use `.physical` for seven-axis physical comparisons; separate `log_degree`
  attributes are removed.
- **Breaking:** Fahrenheit and Rankine move from SI/default parsing to
  `imperial.degF` and `imperial.degR`.
- **Breaking:** Scalar JSON uses format 2 with definition IDs and an optional
  receiving Registry; format 1 and older pickle payloads are unsupported.
  Reconstruction requires existing definitions and checks their metadata.
- Logarithmic coordinates retain exact `reference` and `at` anchors; `origin`
  is a derived numerical view. Affine defining readings that require an
  unrepresentable symbolic origin sum are rejected; supply the origin at `at=0`.

### Fixed

- Preserve private coordinate mappings through retained linear projections,
  and reconstruct private coordinate/point reprs even when their symbols coincide
  with default logarithmic difference units.

## [0.13.4] - 2026-10-23

### Fixed

- Preserve exact extra scales and structural unit formatting for affine points
  in anonymous zero-origin linear coordinates across text, LaTeX, HTML and siunitx.
  Stored readings and named coordinate labels are unchanged.

### Added

- Add an Arrhenius fitting notebook with absolute temperatures, activation-energy
  estimates and standard errors, a fitted curve and first-order rate predictions.

## [0.13.3] - 2026-10-22

### Added

- `unitx.serialization.to_dict()` / `from_dict()` exchange scalar Quantities,
  affine points and their units as versioned JSON-compatible records. Preserve
  int/float/Fraction/Decimal magnitudes, exact extra scales, formal unit identity
  and coordinate references; validate existing registry meanings without
  registering units. NumPy, complex magnitudes and subclasses are not supported.
- Add `SerializationError` for invalid records and failed restoration, and an
  independent `VersionError` for unsupported explicit versions. Both derive
  from `UnitxError` and `ValueError`; decoding preserves underlying causes.

## [0.13.2] - 2026-10-22

### Added

- `unitx.time.from_datetime()` and `to_datetime()` explicitly convert aware
  datetimes and Unix-referenced affine time points, including custom known
  epoch coordinates. Import preserves integer ns; export rounds once to
  microseconds with ties to even and defaults to UTC.
- `SingleUnit.from_name(name)` constructs a single unit from a registered name
  or name alias, including permitted prefixes, without accepting symbol spellings.

### Changed

- Clock functions in `unitx.time` return `AffineQuantity[int]` points in ns.
  Point subtraction returns a duration Quantity; point addition and
  multiplicative point arithmetic are rejected. Duration adapters are unchanged.
  Clock domains remain untagged; only compare/subtract readings from the same context.
- Clarify process-wide registration and unsupported concurrent registry access.
- Expand the README with a runnable blackbody example and a theme-aware SVG plot.

### Internal

- Route name-only parsing through `SingleUnit.from_name()`, clarify recursive-descent
  grammar, and remove a duplicate NumPy comparison branch.

## [0.13.1] - 2026-10-22

### Added

- Public `unitx.si.au`, `nu` and `Planck` read-only namespaces of SI-valued
  atomic, electron-mass natural, and Planck scales. These are physical
  Quantities, not parser registrations or implicit conversion contexts.
- Export the existing `Particle` namespace directly from `unitx.si`.

## [0.13.0] - 2026-10-21

### Added

- `unitx.math`: real roots, trigonometric/inverse trigonometric functions,
  exponentials/logarithms, hypot and atan2, with explicit units and optional NumPy support.
- `unitx.time`: time, monotonic, perf_counter, process_time and thread_time
  return integer nanosecond Quantities using the standard library's native ns clocks.
  Scalar duration adapters provide sleep and exact timedelta import, with
  nearest-microsecond, ties-to-even export.
- NumPy arcsin/arccos/arctan, hypot and arctan2 dispatch, including unit-aware outputs.

### Changed

- Rewrite the README around scalar formulas, points/differences, math/time and
  NumPy examples. Add a documentation index and focused catalogue, custom-unit,
  typing and contributing guides; move version history out of the NumPy reference.
- Rename the math/time guide to `docs/stdlib.md` (Standard-library enhancements).
  CI and release validation now execute all public Markdown doctests.

## [0.12.1] - 2026-10-21

### Fixed

- Exclude the internal affine/logarithmic design discussion from the repository
  and source distributions. Public API documentation remains in docs/affine.md.

## [0.12.0] - 2026-10-21

### Added

- AffineUnit coordinates for Celsius, Fahrenheit, dBm, dBW and dBV, with
  explicit point/difference arithmetic, conversions and NumPy statistics/writes.
- Rankine and logarithmic difference units Np/B/dB; logarithmic degree tracking
  for compound units and explicit power/amplitude ratio conversion.
- The symbolic natural logarithm of ten, `ln10`.

### Changed

- Public factor constants use lowercase `one`, `two`, `ten`, `pi`, and `ln10`;
  uppercase names are removed. Update imports and references accordingly.
- Unit and ScaledUnit now enforce immutable attributes, including deletion;
  copying and pickle reconstruction preserve unit identity and exact scales.
- Unit(text) always returns a linear Unit, including degC/degF and the difference
  scales of dBm/dBW/dBV. Quantity.unit is Unit; ordinary Quantity conversion
  never applies an origin or changes into an affine object.
- Independent AffineQuantity represents absolute points, including zero-origin
  K/W/V coordinates. Use value @ unit or AffineQuantity(value, unit). Point
  conversion preserves point identity; to_quantity(unit) explicitly evaluates
  an ordinary physical Quantity. Point subtraction returns a Quantity difference.
- AffineUnit descriptors (including si.degC) project to a linear Unit for
  multiplication, division, powers, Quantity construction and linear conversion;
  @ and AffineQuantity use their complete coordinate. Registered projections
  preserve the spelling; anonymous descriptors use delta_unit.
- AffineUnit(reference, at=0, delta_unit=None, kind=None) defines a coordinate
  from a known correspondence; its origin is normalized immediately. Registered
  spellings supply both linear difference metadata and an absolute coordinate.
  AffineUnit has no multiplicative factor and does not accept string construction.
- Exact unit ratios cancel before affine origin evaluation, avoiding precision
  loss from unnecessary SI floating intermediates. Decimal point differences
  combine alignment and subtraction before rounding. AffineQuantity.isclose
  defaults to zero relative tolerance and accepts a Quantity difference tolerance.
- Linear scalar point comparisons retain symbolic unit factors instead of
  deciding equality from floating conversions. Finite `math.pi` is a rational
  approximation and does not equal symbolic pi; zero-tolerance isclose follows
  the same rule. Unsupported symbolic sums require an explicit conversion or
  a nonzero difference tolerance.
- Keep shared numerical scaling in the conversion layer, with point arithmetic,
  comparison and tolerance rules in AffineQuantity.
- Add explicit Unicode unit aliases without compatibility normalization;
  aliases preserve canonical unit identity, prefix policies and coordinate metadata.
- Disable SI prefixes on the built-in atmosphere unit `atm`.

## [0.11.0] - 2026-10-21

### Changed

- Share attribute assignment/deletion protection across immutable algebra and
  unit objects while preserving their slot-based instance layout.
- SingleUnit now holds shared immutable prefix and unit definitions. Metadata and
  spelling selection use these references directly; canonical identity, prefix
  permissions, and text-based reconstruction retain their existing behavior.
  Built-in tables derive their indexes from self-contained definition records;
  the shared NO_PREFIX identity is separate from the actual prefix indexes.
  Resolvers return shared definitions directly; parsing and dynamic registration
  reuse them rather than resolving canonical symbols again. Registration preflight
  reads prefix names and aliases from the same definitions. Ohm spelling
  exceptions use one full-name mapping shared by parsing, output, and preflight.
- PrefixDefinition and UnitDefinition accept a string or tuple of strings for
  symbol aliases and normalize strings to one-element tuples. Symbols and names
  precede required keyword-only scale factors and dimensions.
- Iterate prefix spellings lazily instead of constructing a combined tuple for
  each lookup.

### Removed

- Remove the uppercase `K` alias for kilo. Use `k` for kilo and `K` for kelvin;
  `Km` and `Kg` now fail unless
  explicitly registered. Micro aliases `u` and `μ` remain supported.

## [0.10.1] - 2026-10-21

### Fixed

- Correct the display import regression test to allow an already-loaded optional
  SymPy factorization backend. Display itself must not load additional optional
  backends. Runtime display behavior is unchanged from 0.10.0.

## [0.10.0] - 2026-10-21

### Added

- Notebook LaTeX/HTML display and explicit `to_latex()` / `to_html()` for
  Unit, ScaledUnit and Quantity, with exact extra multipliers and escaped labels.
- Quantity formatting with native magnitude specs and explicit `;text`, `;latex`,
  `;html` and `;siunitx` output selection; Unit format selection and alignment.
- Slash unit fractions by default and explicit `to_latex(fraction=False)`
  signed-power output, independent of Fraction magnitude rendering.
- Mathematical scientific notation and NaN/Inf display for ordinary real scalars,
  plus siunitx v3 export through `to_siunitx()`. No new runtime dependencies.

## [0.9.0] - 2026-10-20

### Added

- Support indexed assignment to ndarray Quantities with compatible-unit
  conversion, fixed destination dtype, staged same_kind casting, and native
  indexing/broadcasting rules, including overlapping RHS values and index aliases.
- Support true ndarray augmented arithmetic (`+=`, `-=`, `*=`, `/=`, `**=`, `@=`)
  while keeping the original Quantity, buffer, dtype, and unit. Reject dimension
  changes and incompatible casts without modifying the target. Scalar magnitudes
  retain ordinary result/rebinding semantics.
- Add a reproducible benchmark for controlled array writes and peak allocations.

### Improved

- Reuse independent numerical temporaries with matching destination dtype when
  staging outputs, removing a redundant full-size allocation and copy.

### Changed

- Array augmented arithmetic now updates shared storage instead of rebinding to
  a new result. Aliases observe writes; integer true division or dimensional
  changes can now fail where ordinary arithmetic and rebinding would succeed.

## [0.8.6] - 2026-10-20

### Added

- Support diff with compatible boundary values converted to the input unit,
  and gradient with scalar spacing or coordinate arrays carrying units. Return
  each gradient component in the ordinate unit divided by that axis's
  coordinate unit, including mixed units across axes.
- Support input masks for sum, add.reduce, mean, std, and var, including staged
  output writes. Preserve units, selected-sample statistics, and native NumPy
  dtype, empty-sample warnings, and degrees-of-freedom behavior.
- Support scalar initial values for sum/add.reduce, converting compatible
  Quantities to the input unit. Bare nonzero initial values remain dimensionless;
  bare zero is an additive identity. Document accumulator casting and empty sums.

## [0.8.5] - 2026-10-20

### Added

- Support modf and frexp with units on both modf parts and the frexp mantissa;
  return the binary exponent as a bare integer. Decompose stored coordinates
  without implicitly normalizing scaled dimensionless units.
- Support both outputs, positional buffers, partial None outputs, and boolean
  masks. Stage every output before writing, preserve target units and dtypes,
  and reject overlapping outputs. Validate scalar, empty, strided, and shared
  storage cases, and use ldexp to reconstruct extreme finite floating values.

### Improved

- Consolidate Quantity-first NumPy function dispatch and reuse output staging
  for single-output and multiple-output calls.
- Use absolute documentation and image links in the README for PyPI rendering.

## [0.8.4] - 2026-10-20

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

### Examples

- Replace the blackbody script with a notebook and add damped motion, free
  Gaussian wave-packet propagation, optical attenuation, and correlated
  measurement-uncertainty notebooks. Keep calculations as Quantities and derive
  plot labels from their units. Check execution, typing, and output-free notebook
  files without adding Jupyter as a library or test dependency.

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
