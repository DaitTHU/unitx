# Scalar data exchange

`unitx.serialization.to_dict()` returns a versioned record made of dictionaries,
lists, strings, null and an integer format version. `from_dict()` restores it.
Use Python's JSON functions, or another transport for the same data structure:

```pycon
>>> import json
>>> from decimal import Decimal
>>> from fractions import Fraction
>>> from unitx import AffineQuantity, AffineUnit, Quantity, Unit
>>> from unitx.serialization import from_dict, to_dict
>>> original = Quantity(Fraction(1, 3), 'cm')
>>> text = json.dumps(to_dict(original), allow_nan=False)
>>> restored = from_dict(json.loads(text))
>>> type(restored.value) is Fraction
True
>>> restored.value, restored.unit
(Fraction(1, 3), Unit('cm'))
>>> restored == original
True

```

The module supports `Unit`, `AffineUnit`, `LogUnit`, scalar `Quantity`,
`AffineQuantity` and `LogQuantity`. Import the two functions from `unitx.serialization`;
they are not package-root exports. No optional dependency is required.

## Stored numbers

Magnitudes must be exact instances of Python `int`, `float`, `Fraction` or
`Decimal`. Booleans, complex numbers, numeric subclasses, custom magnitudes,
NumPy scalars and arrays are rejected with TypeError. A round trip preserves
the stored magnitude's type and value without converting its unit.

| Type tag | `value` field | Meaning |
| --- | --- | --- |
| `int` | A canonical decimal integer string | Retains integers beyond the exact range of a JSON consumer's floating-point numbers |
| `fraction` | Two integer strings, `[numerator, denominator]` | Exact rational, with positive denominator |
| `float` | `float.hex()` text | Exact finite binary value, including subnormals and signed zero; `inf`, `-inf` and `nan` are also supported |
| `decimal` | Decimal text | Retains sign, coefficient, trailing zeros and exponent; infinities and NaNs are supported |

Float NaNs preserve their category; NaN payload bits and sign are not part of
the format. Decimal NaN payloads and signaling status are retained. Export
and import do not round a Decimal magnitude in the active context. Invalid
Decimal strings are rejected even when the caller disables InvalidOperation
traps, without setting flags in the caller's context.

All numerical data except the format version use strings. Consequently strict
JSON encoding with `allow_nan=False` works even for nonfinite magnitudes.
Python's normal integer-string conversion and Decimal representation limits
still apply.

## Units and points

The format preserves the formal product, including prefixes and rational
exponents. Equal physical scale does not merge distinct units such as Hz and
Bq, and `km/m` retains its nontrivial dimensionless scale. Every Unit record
stores its exact scale separately from its formal product, including when that
scale is one:

```pycon
>>> from unitx import pi
>>> from unitx.unit import Unit
>>> scaled = Unit('m', scale=pi ** Fraction(1, 3))
>>> rebuilt = from_dict(json.loads(json.dumps(to_dict(scaled))))
>>> type(rebuilt) is Unit and rebuilt.scale == scaled.scale
True
>>> rebuilt**3 == scaled**3
True

```

An AffineUnit stores its linear physical origin and difference unit. A LogUnit
stores a positive physical reference, its defining reading `at`, logarithmic
difference unit and convention. Logarithmic exchange retains this anchor without
evaluating a zero-reading reference.
Registered coordinates also retain their canonical symbol and stable definition ID;
anonymous coordinates remain anonymous. Points retain their concrete AffineQuantity or LogQuantity family:

```pycon
>>> point = AffineQuantity(Decimal('20.00'), 'degC')
>>> rebuilt = from_dict(json.loads(json.dumps(to_dict(point))))
>>> type(rebuilt) is AffineQuantity and rebuilt.unit == point.unit
True
>>> rebuilt.value.as_tuple() == point.value.as_tuple()
True
>>> epoch = AffineUnit(Quantity(946684800, 's'), delta_unit='s')
>>> custom = AffineQuantity(Fraction(1, 2), epoch)
>>> from_dict(to_dict(custom)).unit == epoch
True

```

This preserves the existing model's information. It does not add clock identity,
timezone or source labels to time points. Point arithmetic and the existing
hashability rules are unchanged.

## Registry requirements

Import resolves spellings against the default environment, or the explicit
`Registry` supplied as `from_dict(record, registry=receiver)`. Both formal
definition IDs and physical metadata must match. Share catalogue definitions
with `Registry([imperial.ft, si.s])`.

Custom definitions follow these sharing rules:

| Scope | Application responsibility |
| --- | --- |
| Within one process | Include the existing unit objects in each receiving Registry; this shares their formal identities. |
| Across processes | Put reusable definitions in a module-level UnitCatalog in an application-controlled module. Import compatible versions of that catalogue in each process and include the required objects in the receiving Registry. Catalogue persistence locations are derived automatically. |
| Definitions without a published catalogue location | Their UUID handles refer to live definitions in the originating process; they do not provide cross-process reconstruction. This includes units kept only in ordinary blocks or local classes. |

Repeating `.define()` creates a new identity, even with identical spelling,
dimension and scale. It does not recreate a saved definition. JSON carries
definition references and validation data, not an installable definition bundle.
Decoding never imports catalogue modules, constructs missing definitions,
registers spellings or replaces definitions. Unknown spellings raise
SerializationError with the original UnitSymbolError retained as `__cause__`.
Neither function changes the input object or record. Applications that mutate a
receiving Registry must also follow its
[synchronization contract](custom-units.md#concurrent-use-and-read-only-access)
throughout decoding.

A linear unit record contains its canonical symbols, stable definition IDs and its combined
dimension, SI Factor and logarithmic degree. Import checks these against the
resolved unit. A registered coordinate must also match its saved affine origin
or logarithmic reference anchor, formal reference and difference units, and
logarithmic kind.
Conflicts raise SerializationError rather than silently changing the result.

These checks describe the resulting unit, not a complete registry snapshot.
They do not compare names, aliases, prefix policies or each individual unit's
conversion metadata when changes cancel within a product. The receiving
application remains responsible for compatible unit definitions. Factor checks
compare exact formal state: a changed symbolic approximation or a different
unresolved-integer decomposition can cause rejection even if physical values
are approximately equal. No floating-point tolerance is used to infer equality.

For catalogue quantities, select the required definitions explicitly:

```pycon
>>> from unitx import Registry
>>> from unitx.imperial import imperial
>>> from unitx.si import si
>>> receiver = Registry([imperial.ft, si.s])
>>> original = 3 * imperial.unit('ft/s')
>>> from_dict(to_dict(original), registry=receiver) == original
True

```

The receiver must have compatible bindings for every term. A product containing
two independent definitions with the same spelling cannot be represented by one
receiving spelling environment. Pickle can preserve such a product by ID.

## Version 2 record format

Only the outermost object has `"version": 2`; every object record has a `type`.
For example, an integer length is represented as:

```json
{
  "version": 2,
  "type": "quantity",
  "value": {"type": "int", "value": "3"},
  "unit": {
    "type": "unit",
    "expression": "m",
    "scale": [],
    "identity": [["", "m", {"module": "unitx.unitregistry", "catalogue": "", "symbol": "m"}, "1", "1"]],
    "dimension": [["0", "1"], ["1", "1"], ["0", "1"], ["0", "1"], ["0", "1"], ["0", "1"], ["0", "1"]],
    "factor": [],
    "log_degree": ["0", "1"]
  }
}
```

The object records have these exact fields:

| `type` | Fields in addition to `type` |
| --- | --- |
| `quantity`, `affine_quantity`, `log_quantity` | `value` scalar record and matching linear/affine/logarithmic `unit` record |
| `unit` | `expression`, `scale`, `identity`, `dimension`, `factor`, `log_degree` |
| `affine_unit` | `origin` linear Quantity record, `delta_unit` linear unit record, `kind` null, `symbol` canonical registered string or null, `definition_id` stable ID or null |
| `log_unit` | `reference` linear Quantity record, `at` scalar record, `delta_unit` linear unit record, `kind` `power`/`amplitude`, `symbol` canonical registered string or null, `definition_id` stable ID or null |

`identity` is a sorted list of `[prefix_symbol, unprefixed_symbol, definition_id,
numerator, denominator]` terms; the last two strings encode the nonzero exponent.
Definition IDs are structured records: `{"module": ..., "catalogue": ..., "symbol": ...}`
for published definitions, or `{"uuid": "<integer>"}` for process-local definitions.
UUIDs remain typed handles internally; the integer string is their transport form.
No module/class/symbol information is encoded in a string or split during lookup.
Built-in root definitions use module `unitx.unitregistry` and an empty catalogue.

`dimension` lists seven rational pairs in T, L, M, I, Theta, N, J order.
`log_degree` is another rational pair in the existing version 2 wire format.
It is derived from Dimension at encoding and folded back into the complete
Dimension on validation; it is not a separate runtime state or public attribute.

Factor component lists preserve the formal integer/symbolic bases and rational
exponents. The empty list represents one. Each component has an `exponent`
pair and one of these base descriptions:

| Component `type` | Other fields |
| --- | --- |
| `prime` | `value`: prime integer string |
| `symbolic` | `symbol`: literal label, `value`: positive finite float.hex() approximation |
| `unfactored` | `value`: unresolved integer string, `limit`: stored factorization-limit integer string |

Symbolic labels are reconstructed as SymbolicConstant objects, never evaluated
as code. Their mathematical construction requirements remain the caller's
responsibility. Unresolved bases keep the existing incomplete-factorization
limitations and emit IncompleteFactorWarning on reconstruction.

Unknown versions, tags, missing or extra fields and invalid nested records are
rejected. The version is independent of the package release number; future
incompatible record changes require a new format version. This is a data
format, not a Python class/subclass reconstruction protocol. No `eval`, dynamic
class imports, JSON encoder subclasses, file I/O or configurable codec registry
are involved. Applications parsing external input should apply their own size
and resource limits.

Python pickle is a separate mechanism with its own reconstruction rules. Current
unitx pickles retain definition IDs, but pickle payloads produced by older unitx
formats are not supported. Use this versioned JSON format for data exchange.

## Migrating version 1 records

The writer emits version 2 and the reader requires it. Version 1 identified
units by symbols, so it cannot prove that same-named
private definitions share identity. Reading it raises VersionError. Decode old records with unitx 0.13.3 to recover their measurements. Rebuild
them using explicitly verified current definitions, then encode with the current
implementation. Do not merely change the version number: linear terms
need stable IDs and coordinate records need a `definition_id` field.

For private definitions, place reusable declarations in a module-level UnitCatalog
and import that catalogue in the receiving process. Independently repeating
`.define()` creates a new UUID and does not recreate a saved definition. There is no automatic
registration or migration of untrusted definitions.

## Errors

Import the error classes from `unitx.exceptions`. The public boundaries are:

| Operation and failure | Exception |
| --- | --- |
| `to_dict()` receives an unsupported object or magnitude | TypeError |
| `from_dict()` receives a non-dictionary top-level input | TypeError |
| A record specifies an unsupported integer format version | VersionError |
| A record has no version or the version is not an integer | SerializationError |
| Invalid fields, missing definitions, conflicting unit meanings or failed reconstruction | SerializationError |

VersionError and SerializationError both inherit UnitxError and ValueError.
They are independent classes: VersionError is available for explicit version
checks elsewhere in unitx as well. It currently describes the record format
version, not the installed package version. Catch the two classes separately
for different recovery actions, or catch UnitxError for both:

```pycon
>>> from unitx.exceptions import SerializationError, UnitxError, VersionError
>>> try:
...     from_dict({'version': 1})
... except VersionError:
...     print('Unsupported record version')
Unsupported record version
>>> try:
...     from_dict({'version': 2, 'type': 'unknown'})
... except SerializationError:
...     print('Invalid record')
Invalid record

```

Underlying numeric, parsing and coordinate exceptions, including
UnitSymbolError, UnitSyntaxError and DimensionError, are retained as
`error.__cause__` when wrapped. Explicit record-validation errors use
SerializationError directly. Exceptions outside the expected validation
families, such as RuntimeError, are not wrapped.

The return type of `from_dict()` is a union of the supported object classes;
use `isinstance` when code requires a specific class. Its scalar magnitude type
is the supported union, since arbitrary input data cannot statically determine
the original numerical type.

Numerical encodings follow the Python documentation for
[float hexadecimal conversion](https://docs.python.org/3/library/stdtypes.html#float.hex),
[Decimal construction](https://docs.python.org/3/library/decimal.html#decimal.Decimal)
and [JSON data exchange](https://docs.python.org/3/library/json.html).

[Documentation index](README.md) · [Scalar behavior](scalars.md) · [Custom units](custom-units.md)
