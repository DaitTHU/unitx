# Public API and compatibility policy

This is the compatibility policy for the forthcoming 1.x series. The current
0.x series can still make documented breaking changes; see the
[migration guide](migration.md). This policy does not announce a 1.0 release.

## Supported imports and extension boundaries

Compatibility covers names in the package root's `__all__`, documented public
members of those objects, and the following documented module entries:

| Entry | Supported surface |
| --- | --- |
| `unitx.si`, `unitx.imperial`, `unitx.cgs` | Documented catalogues, constants and explicit electromagnetic conversions |
| `unitx.math`, `unitx.statistics`, `unitx.time` | Documented functions |
| `unitx.scipy`, `unitx.matplotlib` | Documented finite adapters; registration remains explicit |
| `unitx.serialization` | `to_dict`, `from_dict` and their documented record schema |
| `unitx.typing` | `Scalar`, `Array`, `Magnitude`, `NumpyArray`, `NumpyValue` |
| `unitx.exceptions` | Exported exception and warning classes and their documented inheritance |
| `unitx.factor`, `unitx.monomial`, `unitx.dimension` | Documented algebra objects, constants and methods |
| `unitx.unitdefinition`, `unitx.singleunit`, `unitx.unitparser`, `unitx.unitregistry`, `unitx.unitcatalog` | Documented definition, parser, registry and catalogue interfaces |
| `unitx.unit`, `unitx.quantity`, `unitx.affineunit`, `unitx.affinequantity`, `unitx.logunit`, `unitx.logquantity` | Named public classes, plus `unitx.quantity.quantity` |

An import made internally by one of these modules is not automatically a
re-export. Underscore names, private packages, storage slots and dispatch tables
are internal. In particular, numerical protocols have a public import route
through `unitx.typing`; their original private module identity need not move.

Subclass construction and explicit subclass overrides remain available as in
the [Impedance example](https://github.com/DaitTHU/unitx/blob/main/examples/impedance.md).
Inherited calculations return the documented Quantity or point family; they do
not promise to preserve arbitrary subclasses, extra state or overridden
constructors. Private hooks are not an extension API. Array protocols describe
capabilities, not support for an arbitrary third-party backend.

Types describe magnitude domains and object families, not dimensions. Correcting
an unsound annotation or narrowing an overly broad result can happen in a patch
with a changelog entry. Removing a supported call requires the ordinary
compatibility process. Inference through NumPy/SciPy functions also depends on
their installed stubs; unitx cannot promise identical inference across all
checker and dependency versions.

## Data and reconstruction

Throughout 1.x, the reader will accept valid version 2 records written by 1.0,
under the documented definition-environment prerequisites. New writer schemas
must have a new format version and preserve the version 2 read path. The schema
version is independent of the package version. Version 1 records remain
unsupported; their [migration procedure](serialization.md#migrating-version-1-records)
requires an appropriate older reader. Unknown future schemas raise VersionError.

Built-in definition IDs and published unit/coordinate meanings remain stable
throughout 1.x. Moving implementation modules must retain resolution of those
IDs. Custom definitions must still be available with the same identities and
meanings in the receiving registry; the format is not a self-contained unit
registry or a general definition exchange language. Missing/conflicting
definitions fail explicitly. Symbols alone are not reconstruction metadata.

Pickle and evaluable repr are for trusted Python environments with the required
definitions, not a long-term or cross-version storage format. No cross-release
pickle compatibility is promised. Keep JSON metadata plus numeric arrays for
archival workflows; see [array storage](array-storage.md). Scalar JSON support
does not imply that array magnitudes can be passed directly to `to_dict`.

## Constants and catalogue data

Exact defining values and formal unit factors are not changed silently. A
correction that changes a published unit's meaning is a breaking change.
Additional nonconflicting names/units may be added in a minor release; existing
accepted text retains its meaning. Registry lookup does not infer a physical
kind from dimensions.

Measured constants use the source edition documented in the
[SI guide](catalogues/si.md). A new data edition may update numerical constants
and the explicit physical transforms that depend on them in a minor release,
with source edition, affected entries and numerical consequences recorded in
the changelog. Patch updates are reserved for documented data/implementation
errors. Existing unit definitions that use nominal measured inputs, such as Da,
retain their published factors within 1.x; updating constant data does not
silently redefine these units. Pin the package version and record the constants
used when reproducing numerical research.

Derived catalogue values use the fixed import context documented in the SI
guide. Its precision and numerical convention are part of the 1.x contract.
Finite digits do not mean exact mathematical constants or propagated measurement
uncertainties. User arithmetic remains in the caller's numerical domain. NumPy
results and transcendental functions are not guaranteed bitwise identical
across platforms and backend versions.

## Exceptions and failure effects

Depend on documented exception categories, not exact message wording or traceback
layout. DimensionError means incompatible complete dimensions; UnitOperationError
is a TypeError for invalid quantity/point operations, and ImplicitConversionError
is its unit-stripping subclass. Parsing/registration and serialization errors
retain the categories documented in their guides. Native numerical, shape,
dtype and domain errors may remain TypeError, ValueError, OverflowError or the
backend's own numerical error; not every failure is a UnitxError.

Warnings exported by `unitx.exceptions` retain their documented meaning and
are reported at the caller boundary. Native numerical warnings remain native.
Registration validates before committing bindings. Documented array writes
stage calculation, conversion and validation before modifying fixed buffers;
this is not rollback for arbitrary hardware/backend failures. Registry use
still requires application synchronization during mutation.

## Dependencies and deprecation

Core scalar use has no mandatory runtime dependencies. Optional libraries stay
lazy at their documented boundaries. Supported minimum Python and optional
library versions are recorded in `pyproject.toml`; CI tests minimum and current
combinations. Raising a minimum version requires a minor release, a changelog
notice and the affected installation/test matrix. A patch does not raise a
minimum except to address a disclosed security issue. Compatibility with future
upstream releases is established by validation, not assumed.

Within 1.x, intentional removal or incompatible behavior changes require a
DeprecationWarning at the caller boundary, a documented replacement and migration
example, and at least one minor release and six months of notice. Removal then
waits for a major release. Correctness fixes may restore the already-documented
contract in a patch; they must identify affected behavior and add a regression.
If a defect cannot be fixed without changing the promised semantics, it follows
the breaking-change process instead of silently redefining the contract.
