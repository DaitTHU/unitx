# Unit model and implementation boundaries

A Unit stores a formal product and an additional exact scale. Named units and
prefixes retain their own formal identities; conversion-equivalent expressions
are not merged. Quantity supplies the magnitude; AffineQuantity and LogQuantity
supply readings in their respective coordinate families.

## Source layout

Public objects live in modules such as `unitx.unit`,
`unitx.quantity` and `unitx.unitregistry`. The standard-library enhancements
are `unitx.math`, `unitx.statistics` and `unitx.time`. SI, Imperial, information
and CGS units each have a catalogue package.

Implementations are grouped by responsibility:

| Package | Responsibility |
| --- | --- |
| `_numerics/` | Scalar protocols and number theory, exact scaling, linear conversion, optional NumPy kernels and annotations |
| `numpy.py` / `numpy.pyi` | Finite public wrapped-input adapters and static overloads, using the existing NumPy dispatch |
| `scipy.py` | Finite, lazily loaded SciPy integration at an explicit numerical boundary |
| `matplotlib.py` | Optional, explicitly registered axis conversion at Matplotlib's numeric plotting boundary |
| `_formatting/` | Unicode unit notation shared with the parser, and rich rendering |
| `_electromagnetism/` | Fixed electromagnetic coefficients and explicit physical correspondence |
| `_registry/` | Prefix data, root definitions and reconstruction lookup |
| `_coordinates/` | Affine/log mappings, coordinate definitions, resolution and point helpers |

The `_numerics` package entry owns scalar protocols, rational normalization and
the built-in number-theory helpers. Consumer annotations are exported through
`unitx.typing`; the private implementation location is not an extension API.
The package entry does not import its scaling, conversion or optional NumPy submodules.
Importing public `unitx.numpy` explicitly requires NumPy; ordinary scalar imports
and `unitx.typing` do not.
Other private package initializers are lightweight and import no implementations;
reading Unicode notation does not load the rich renderer. Small shared object
and operand boundaries remain at the root rather than in a generic utilities bag.

Numeric scaling and linear conversion stay outside the coordinate package.
Ordinary linear operations therefore do not need coordinate implementations;
NumPy kernels are still imported only at array boundaries.

The CGS package keeps its mechanical and four electromagnetic catalogues in
`cgs/units.py`. The private `_EMConversion` base in `_electromagnetism/conversion.py`
owns the explicit physical conversion boundary and its magnitude overloads.
The fixed rule tables and their definition-identity indexes are constructed on
first use, after the catalogue definitions exist, then cached per catalogue.
The charge and magnetic transforms derive from the formula coefficients;
magnetic H additionally uses the declared rationalization convention.
This avoids an import cycle without adding an active system or mutable registry.
`gaussian`, `esu`, `emu` and `hl` inherit mechanical units from `cgs` and explicitly
include them in independent parsers. Each exposes units, conversion and formula
coefficients; mechanical `cgs` has no electromagnetic entries. Shared electric
ESU/Gaussian and magnetic EMU/Gaussian definitions retain the same identities.
Formula coefficients are frozen `_EMCoefficients` objects; their defining
equations live in `_electromagnetism/coefficients.py`. The shared vacuum-permeability
input lives in SI bases so `SI.mu0` and `si.em` derive from one value.
Each electromagnetic catalogue's `convert()` selects a declared physical kind
and checks complete dimensions before numerical scaling. Its rules do not enter
ordinary linear conversion, comparison or array writing. Quantities store no
active system or electromagnetic kind. Cross-CGS correspondence goes explicitly
through SI. See the [CGS guide](catalogues/cgs.md).

## Cohesive internal records

Conversion data groups the exact factor and algebraic Dimension. Dimension
composes Monomial to own all rational exponents, including an internal logarithmic
axis. Equality and hashing compare the complete algebraic signature; `.physical`
projects onto the seven physical axes. Its named exponents, tuple, dictionary and
human-readable string describe only those physical axes. Definitions own this data;
a composite Unit derives it from its terms
and extra scale. It is an immutable summary, not a second independent definition.
The readable symbol is derived from the formal product.

Unit derives conversion data by collecting nonidentity Factor and Dimension
terms in one traversal, then asking each model to combine its powered terms.
Factor and Dimension own the empty/single-term and batched product paths.
This removes redundant numerical metadata work without
dropping formal unit terms. Empty Monomial powers reuse the immutable identity
after validating the exponent. Factor and Dimension retain their exponent-one
paths to reuse the containing object and its caches before entering Monomial.
Nontrivial products accumulate rational exponents in private mutable monomials
and freeze once, with Factor validating the resulting symbolic bases. A frozen
snapshot owns its mapping and remains unchanged if accumulation continues.
Public Monomial, Factor, Dimension and Unit objects retain their immutable
value and hashing contracts.

Factor normalizes arithmetic operands before applying exact product/quotient
rules. Identity operations and exact cancellation can reuse existing algebraic
values. Already normalized Fraction exponents pass through unchanged across
algebra layers. Numerical caches do not determine algebraic identity or classification.
NumPy scaling consumes Factor's evaluated int/Fraction/float domain instead of
separately decomposing it first.

Dimension's physical projection filters the sparse monomial directly. Display
and dictionary views share canonical physical-axis iteration without expanding
a seven-element tuple. `as_tuple()` remains the fixed-layout boundary used by
serialization and SI base-unit expansion.

Unit spellings group canonical symbols, names, aliases and prefix permissions.
Their validation produces the accepted spellings before registration. Definition
identity and coordinate data remain separate from spelling rules.
Prefix definitions carry their decimal/binary family. `True` and spelling-list
permissions enable decimal prefixes; PrefixFamily and spelling mappings can
select either family or both. Validation, parsing, canonical output and compact
representation consult the same stored policy. Catalogue prefix shortcuts use
the shared `_prefixed_units` helper; they do not create new unit definitions.

A registry's bindings own the symbol, name and alias indexes together. The
registry adapts public unit objects to definitions, stages a complete binding
batch, and commits only after all conflicts have been checked. The binding
implementation does not import Unit, Quantity or coordinate objects.
This failure-invariance guarantee supplies no locking or atomic visibility to
concurrent readers. Concurrent parsing is supported while bindings remain
unchanged; registration requires application synchronization. A catalogue's
read-only namespace does not freeze those bindings, and Registry exposes no
freeze or snapshot API. See the
[concurrent-use contract](custom-units.md#concurrent-use-and-read-only-access).

Packing fields is justified by shared operations and invariants, not by a target
slot count. Numeric caches, parser state and simple value/unit pairs do not need
another object merely to shorten their containing classes.

## Definitions and catalogues

The built-in root table uses UnitDefinition directly. Derived catalogue entries
use the ordinary Unit, Quantity or AffineUnit `.define()` method. Catalogue-wide
construction policies use explicit `Registry.defining()` blocks; no catalogue
wrapper duplicates definition signatures or changes metadata after construction.

UnitCatalog collects exported objects and delegates binding and publication.
The definition object determines formal identity. Registry membership, Python
attribute names and persistence locations do not determine that identity.

## Linear and logarithmic coordinates

AffineUnit describes a linear coordinate with a physical origin and complete
difference Unit. LogUnit describes a logarithmic coordinate with a positive
physical reference, logarithmic difference Unit and power/amplitude convention.
Their point types are AffineQuantity and LogQuantity. Each point class owns its
state, public operations and result construction. Shared point functions only
align operands, evaluate magnitudes and compare readings; they never choose a
concrete result class. Public `convert()` handles linear Unit-to-Unit scaling only.

`.to()` stays within its coordinate family. `.to_quantity()` explicitly evaluates
a physical Quantity; `LogQuantity.from_quantity(quantity, coordinate)` enters a
logarithmic coordinate. These boundaries prevent an implicit exponential or
logarithm from entering ordinary Quantity conversions.

`Unit.__call__` is a narrow convenience boundary for the formal dB unit. It
delegates lazily to the logarithmic module, which constructs an ordinary
LogUnit or reuses a named descriptor. Default conventions recognize watt and
volt definition identities with their prefixes; they are not inferred from
dimensions or symbol strings. Calling another Unit raises TypeError.

The difference-unit factor uses the neper scale. The mapping coefficient is
`ln(10)/2` for power and `ln(10)` for amplitude, so a reading displacement `x - at`
represents the ratio `10**((x - at) * delta.factor / coefficient)` relative to
the defining physical reference. Bel has factor `ln(10)/2` and
neper has factor one. Logarithmic degree is independent of the seven physical
dimensions and prevents implicit conversion of dB into an ordinary ratio. Its
exponent is stored only in Dimension, with no parallel `log_degree` state.

This is a closed pair of mappings, without user callbacks, a transformation
graph or a plugin registry. Array kernels and scalar numerical domains retain
their existing evaluation rules. Public `kind`, `to_ratio()` and `to_log()`
contracts remain explicit; the internal logarithmic axis is not a constructor
argument or public exponent attribute.

## Stored state and dependencies

Ordinary UnitDefinition stores only `_key`, `_spelling` and `_conversion`, plus
weak-reference support. A private `_CoordinateDefinition` extends it with one
immutable mapping. Its inherited conversion data describes the named linear
difference identity. The descriptor and definition share the same mapping
object, so retaining a bound linear projection also retains its coordinate
metadata without a second global coordinate index.

`_AffineMapping` owns the physical origin and complete difference Unit.
`_LogMapping` owns the positive physical reference, its defining reading `at`,
difference Unit and power/amplitude convention. Its zero-reading origin is a
derived numerical value; it cannot determine identity or reconstruction.
They validate and implement mathematical
conversion and comparison, without spelling, registration or presentation rules.
AffineUnit and LogUnit each store only `_mapping` and optional `_definition`.
Defining a coordinate gives it a distinct named difference identity and binds
only inside an explicit definition scope or catalogue.

Quantity, AffineQuantity and LogQuantity each own `_value` and `_unit` directly.
A stateless point operand marker lets linear operations defer to reflected point
operations without importing either concrete point family. A separate stateless
linear-unit interface projects descriptors at construction and unit-operation
boundaries. Neither interface selects mappings or concrete result types.

Ordinary linear arithmetic and conversion do not import coordinate
implementations. The explicit `value @ Unit` convenience loads the affine point
entry at that boundary. Registry's binding kernel accepts immutable definition records; coordinate definitions
enumerate their own reference/difference dependencies. Display, NumPy and
serialization remain explicit integration boundaries. Numeric caches, parser
state and reconstruction indexes remain separate from mathematical identity.

Direct array methods declare their public signatures explicitly, letting Python
bind arguments and defaults. A shared helper in `_numerics/methods.py` lazily
forwards their arguments to the same NumPy dispatch used by function calls.
Coordinate and linear result construction stays in the existing handlers.
Reshape keeps its explicit positional-shape normalization.

NumPy operand validation, numerical normalization, scaling and unit alignment
live in `_numerics/numpy_operands.py`. The dispatcher and the creation, histogram,
selection and write adapters use these shared rules without routing back through
the dispatcher. This module imports no operation or write handlers; validation
is still performed at each public boundary. Output staging and concrete operation
handlers retain their existing owners. Product adapters reuse the first operand's
complete Unit before accumulating later operands, preserving its extra scale and
formal identity without constructing an initial identity product.

The point marker's type parameter describes the result of adding a difference,
not the original magnitude type. This prevents the open scalar protocol from
claiming point operands and preserves concrete result types without naming the
coordinate families in Quantity's annotations.

| Object | Authoritative state and derived data |
| --- | --- |
| Monomial | Base/exponent mapping; multiplication and powers produce new mappings |
| SymbolicConstant | Symbol and approximate value under its mathematical construction contract |
| Factor | Exact monomial; evaluation and rationalized decomposition are derived caches |
| Dimension | Complete algebraic monomial; physical-axis views are projections |
| PrefixDefinition | Symbol, aliases, names, exact factor and decimal/binary family |
| UnitDefinition | Independent identity, validated spelling policy and conversion data; accepted spellings derive from the policy |
| SingleUnit | Shared prefix and unit definitions; display spellings respect their permissions |
| Unit | Formal monomial and extra exact scale; total factor/dimension derive from them |
| Quantity, AffineQuantity, LogQuantity | Stored magnitude and complete unit/coordinate on each concrete class |
| AffineUnit, LogUnit | Mapping and optional named definition; a coordinate definition shares that mapping |
| Registry | Spelling bindings and default prefix policy; lookup indexes refer to existing definitions |
| UnitParser | Input/environment, per-call token state and one successful-result cache |

The operand interfaces have empty slots. Catalogues remain class namespaces;
their registry is a class attribute, not per-instance state. The private mapping
records are not a public transformation or plugin API.

Keeping mapping records is deliberate: anonymous descriptors, named descriptors
and retained linear projections share one immutable mathematical definition.
Descriptors contain no forwarding methods for mathematical conversion; readings
and the explicit integration adapters call the mapping directly. Removing this
sharing while preserving the string/projection entry points would require
another association mechanism or duplicate the defining state. A separate
coordinate registry and removal of those public entry points are not part of
the model.

## Documentation boundaries

Module docstrings describe responsibility and dependency boundaries. Class
docstrings describe construction, identity and invariants; method docstrings
describe the local input/result contract. They should explain what a caller
needs without reproducing a guide's complete support matrix.

The [project README](https://github.com/DaitTHU/unitx/blob/main/README.md) is the
short runnable introduction. The [guide index](README.md) routes users to the
current public contracts; the [NumPy matrix](numpy.md#support-matrix) owns the
array support list. Migration notes describe changes from older behavior.
Architecture explains ownership and dependencies rather than duplicating every
private slot. Keep measured costs and source fingerprints in the
[performance report](https://github.com/DaitTHU/unitx/blob/main/benchmarks/performance.md).

## Investigating performance

Separate repeated setup from per-value work before changing a model:

- Reuse bound Unit objects, including powers and conversion targets, outside
  numerical loops. A UnitParser caches only its own successful result;
  `Unit(text)`, `Registry.parse()` and catalogue `unit()` create fresh parsers.
- Profile formal-product construction separately from magnitude kernels.
  Factor and Dimension already batch nontrivial products; repeated metadata
  derivation in Unit is still a concrete place to measure allocations.
- Keep scalar rational/Decimal conversion separate from NumPy scaling and
  dispatch costs. Short-array fixed overhead and large-array memory traffic
  need different measurements. Output staging must retain its failure and
  overlapping-storage guarantees.

These are investigation boundaries, not measured speedup claims. A cache or
shared abstraction needs a demonstrated repeated cost and an explicit lifetime;
it must not merge formal definitions, change registry lookup, or infer exact
identity from numerical approximations. Use the existing warmed profiles and
uninstrumented benchmarks before expanding state or changing public APIs.
