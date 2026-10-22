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

The module supports `Unit`, `ScaledUnit`, `AffineUnit`, scalar `Quantity` and
scalar `AffineQuantity`. Import the two functions from `unitx.serialization`;
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
Bq, and `km/m` retains its nontrivial dimensionless scale. A ScaledUnit stores
its additional Factor separately, including when that Factor is one:

```pycon
>>> from unitx import pi
>>> from unitx.unit import ScaledUnit
>>> scaled = ScaledUnit('m', pi ** Fraction(1, 3))
>>> rebuilt = from_dict(json.loads(json.dumps(to_dict(scaled))))
>>> type(rebuilt) is ScaledUnit and rebuilt.scale == scaled.scale
True
>>> rebuilt**3 == scaled**3
True

```

An AffineUnit stores its physical origin, difference unit and logarithmic kind.
Registered coordinates also retain their canonical registered symbol;
anonymous coordinates remain anonymous. Points stay AffineQuantity objects:

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

Import resolves spellings against the receiving process's existing registry.
Register custom units explicitly during application initialization before
decoding records that refer to them. Unknown spellings raise SerializationError
with the original UnitSymbolError retained as `__cause__`.
Decoding never registers or replaces definitions, and neither function changes
the input object or record.

A linear unit record contains its canonical formal identity and its combined
dimension, SI Factor and logarithmic degree. Import checks these against the
resolved unit. A registered coordinate must also match the saved physical
origin, its formal reference and difference units, and logarithmic kind.
Conflicts raise SerializationError rather than silently changing the result.

These checks describe the resulting unit, not a complete registry snapshot.
They do not compare names, aliases, prefix policies or each individual unit's
conversion metadata when changes cancel within a product. The receiving
application remains responsible for compatible unit definitions. Factor checks
compare exact formal state: a changed symbolic approximation or a different
unresolved-integer decomposition can cause rejection even if physical values
are approximately equal. No floating-point tolerance is used to infer equality.

## Version 1 record format

Only the outermost object has `"version": 1`; every object record has a `type`.
For example, an integer length is represented as:

```json
{
  "version": 1,
  "type": "quantity",
  "value": {"type": "int", "value": "3"},
  "unit": {
    "type": "unit",
    "expression": "m",
    "identity": [["", "m", "1", "1"]],
    "dimension": [["0", "1"], ["1", "1"], ["0", "1"], ["0", "1"], ["0", "1"], ["0", "1"], ["0", "1"]],
    "factor": [],
    "log_degree": ["0", "1"]
  }
}
```

The object records have these exact fields:

| `type` | Fields in addition to `type` |
| --- | --- |
| `quantity`, `affine_quantity` | `value` scalar record, `unit` linear/affine unit record respectively |
| `unit` | `expression`, `identity`, `dimension`, `factor`, `log_degree` |
| `scaled_unit` | The unit fields plus `scale`, a Factor component list |
| `affine_unit` | `origin` linear Quantity record, `delta_unit` linear unit record, `kind` null/`power`/`amplitude`, `symbol` canonical registered string or null |

`identity` is a sorted list of `[prefix_symbol, unprefixed_symbol, numerator,
denominator]` terms; the last two strings encode the nonzero exponent.
`dimension` lists seven rational pairs in T, L, M, I, Theta, N, J order.
`log_degree` is another rational pair.

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
...     from_dict({'version': 2})
... except VersionError:
...     print('Unsupported record version')
Unsupported record version
>>> try:
...     from_dict({'version': 1, 'type': 'unknown'})
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
