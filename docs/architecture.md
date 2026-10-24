# Unit model and implementation boundaries

A Unit stores a formal product and an additional exact scale. Named units and
prefixes retain their own formal identities; conversion-equivalent expressions
are not merged. Quantity supplies the magnitude; AffineQuantity and LogQuantity
supply readings in their respective coordinate families.

## Source layout

Public objects keep their existing modules, such as `unitx.unit`,
`unitx.quantity` and `unitx.unitregistry`. Private registry support lives in
`_registry/`: prefix data, root definitions and reconstruction lookup.
Private coordinate support lives in `_coordinates/`: the two mappings,
coordinate definitions and resolution, logarithmic evaluation and point helpers.
Both private package initializers remain lightweight and import no implementations.

Shared numeric scaling, linear conversion and operand interfaces stay outside
the coordinate package. Ordinary linear operations therefore do not need to
import coordinate implementations. SI and Imperial definitions remain in their
respective catalogue packages.

## Cohesive internal records

Conversion data groups the exact factor and algebraic Dimension. Dimension
composes Monomial to own all rational exponents, including an internal logarithmic
axis. Equality and hashing compare the complete algebraic signature; `.physical`
projects onto the seven physical axes. Its named exponents, tuple, dictionary and
human-readable string describe only those physical axes. Definitions own this data;
a composite Unit derives it from its terms
and extra scale. It is an immutable summary, not a second independent definition.
The readable symbol is derived from the formal product.

Unit spellings group canonical symbols, names, aliases and prefix permissions.
Their validation produces the accepted spellings before registration. Definition
identity and coordinate data remain separate from spelling rules.

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

The point marker's type parameter describes the result of adding a difference,
not the original magnitude type. This prevents the open scalar protocol from
claiming point operands and preserves concrete result types without naming the
coordinate families in Quantity's annotations.

| Object | Stored fields |
| --- | --- |
| Monomial | `_components` |
| SymbolicConstant | `_symbol`, `_value` |
| Factor | `_monomial`, `_value_cache`, `_decompose_cache` |
| Dimension | `_monomial` |
| PrefixDefinition | `symbol`, `symbol_aliases`, `names`, `factor` |
| _UnitSpellings | `symbols`, `names`, `prefixable`, `accepted` |
| _ConversionData | `factor`, `dimension` |
| UnitDefinition | `_key`, `_spelling`, `_conversion`, `__weakref__` |
| _CoordinateDefinition | inherited definition fields, `_mapping` |
| SingleUnit | `_prefix`, `_unit` |
| Unit | `_monomial`, `_scale`, `_conversion` |
| Quantity, AffineQuantity, LogQuantity | `_value`, `_unit` on each concrete class |
| _AffineMapping | `_origin`, `_delta_unit` |
| _LogMapping | `_reference`, `_at`, `_delta_unit`, `_kind` |
| AffineUnit, LogUnit | `_mapping`, `_definition` on each concrete class |
| Registry | `_bindings`, `_prefixable` |
| _Bindings | `definitions`, `names`, `aliases` |
| UnitParser | `_expression`, `_registry`, `_result`, `_tokens`, `_token` |
| Token | `kind`, `value` |

The operand interfaces have empty slots. Catalogues remain class namespaces;
their registry is a class attribute, not per-instance state. The private mapping
records are not a public transformation or plugin API.

Keeping mapping records is deliberate: anonymous descriptors, named descriptors
and retained linear projections share one immutable mathematical definition.
Descriptors contain no forwarding methods for mathematical conversion; readings
and the explicit integration adapters call the mapping directly. Removing this
sharing while preserving current string/projection entry points would require
another association mechanism or duplicate the defining state. A separate
coordinate registry and removal of those public entry points are not part of
this refactor.
