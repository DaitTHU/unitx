"""Versioned, JSON-compatible records for scalar quantities and units.

Use `to_dict` and `from_dict` around the standard library's JSON
functions. Decoding resolves existing units without registering definitions.
See `docs/serialization.md` for the version 1 format and its boundaries.
"""

from __future__ import annotations

# Structural exchange reads internal state, as display and pickle do, without
# adding public container interfaces to the algebra and coordinate classes.
# pyright: reportPrivateUsage=false
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import cast

from ._numerics import Fractionable, Magnitude, UnfactoredInteger
from .affinequantity import AffineQuantity
from .affineunit import AffineUnit, _as_affine_unit
from .exceptions import SerializationError, UnitxError, VersionError
from .factor import Factor, SymbolicConstant
from .quantity import Quantity
from .unit import ScaledUnit, Unit

__all__ = ['from_dict', 'to_dict']

type _Object = Unit | AffineUnit | Quantity[Fractionable] | AffineQuantity[Fractionable]


def to_dict(
    value: Unit | AffineUnit | Quantity[Magnitude] | AffineQuantity[Magnitude], /
) -> dict[str, object]:
    """Return a new version 1 record, suitable for `json.dumps(..., allow_nan=False)`.

    Support exact Unit, ScaledUnit, AffineUnit, Quantity and AffineQuantity
    classes. Magnitudes must be exact int, float, Fraction or Decimal classes;
    booleans, subclasses, complex numbers and NumPy objects are unsupported.
    No conversion or global registration is performed.

    >>> import json
    >>> q = Quantity(Fraction(1, 3), 'cm')
    >>> from_dict(json.loads(json.dumps(to_dict(q)))) == q
    True
    """
    return {'version': 1, **_encode(value)}


def from_dict(data: object, /) -> _Object:
    """Restore a version 1 record using existing unit definitions.

    Non-dict input raises TypeError. An unsupported integer format version
    raises VersionError. Missing or malformed fields and failed restoration
    raise SerializationError. Underlying parsing, numeric and coordinate
    errors are retained as `__cause__`.
    Never evaluate Python text, import named classes, or register units.
    """
    if type(data) is not dict:
        error_message = 'Expected a serialization record dictionary.'
        raise TypeError(error_message)
    record = cast(dict[str, object], data)
    version = record.get('version')
    if type(version) is not int:
        error_message = 'Serialized record requires an integer format version.'
        raise SerializationError(error_message)
    if version != 1:
        error_message = f'Unsupported serialization format version {version}; expected 1.'
        raise VersionError(error_message)
    try:
        return _decode({key: value for key, value in record.items() if key != 'version'})
    except (SerializationError, VersionError):
        raise
    except (TypeError, ValueError, OverflowError, UnitxError) as error:
        error_message = 'Cannot restore the serialized unitx record.'
        raise SerializationError(error_message) from error


def _record(data: object, keys: set[str] | None = None) -> dict[str, object]:
    if type(data) is not dict:
        error_message = 'Expected a serialization record dictionary.'
        raise SerializationError(error_message)
    record = cast(dict[str, object], data)
    if keys is not None and record.keys() != keys:
        error_message = f'Expected exactly these record fields: {sorted(keys)}.'
        raise SerializationError(error_message)
    return record


def _text(value: object) -> str:
    if type(value) is not str:
        error_message = 'Expected a string field.'
        raise SerializationError(error_message)
    return value


def _list(value: object) -> list[object]:
    if type(value) is not list:
        error_message = 'Expected a serialization list.'
        raise SerializationError(error_message)
    return cast(list[object], value)


def _integer(value: object) -> int:
    text = _text(value)
    result = int(text)
    if str(result) != text:
        error_message = 'Expected a canonical integer string.'
        raise SerializationError(error_message)
    return result


def _fraction_data(value: Fraction) -> list[str]:
    return [str(value.numerator), str(value.denominator)]


def _fraction(value: object) -> Fraction:
    parts = _list(value)
    if len(parts) != 2:
        error_message = 'Expected a [numerator, denominator] pair.'
        raise SerializationError(error_message)
    numerator, denominator = (_integer(part) for part in parts)
    if denominator <= 0:
        error_message = 'Fraction denominator must be positive.'
        raise SerializationError(error_message)
    return Fraction(numerator, denominator)


def _scalar_data(value: object) -> dict[str, object]:
    if type(value) is int:
        return {'type': 'int', 'value': str(value)}
    if type(value) is float:
        return {'type': 'float', 'value': value.hex()}
    if type(value) is Fraction:
        return {'type': 'fraction', 'value': _fraction_data(value)}
    if type(value) is Decimal:
        return {'type': 'decimal', 'value': str(value)}
    error_message = 'Serialization requires an int, float, Fraction or Decimal magnitude.'
    raise TypeError(error_message)


def _scalar(data: object) -> Fractionable:
    record = _record(data, {'type', 'value'})
    kind = _text(record['type'])
    value = record['value']
    if kind == 'fraction':
        return _fraction(value)
    if kind == 'int':
        return _integer(value)
    if kind == 'float':
        return _float(value)
    if kind == 'decimal':
        # Invalid decimal strings must fail even if the caller disables traps.
        with localcontext() as context:
            context.traps[InvalidOperation] = True
            try:
                return Decimal(_text(value))
            except InvalidOperation as error:
                error_message = 'Invalid Decimal string.'
                raise SerializationError(error_message) from error
    error_message = f'Unknown scalar type {kind!r}.'
    raise SerializationError(error_message)


def _float(value: object) -> float:
    text = _text(value)
    result = float.fromhex(text)
    if result.hex() != text:
        error_message = 'Expected a canonical float.hex() string.'
        raise SerializationError(error_message)
    return result


def _factor_data(factor: Factor) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for base, exponent in factor._monomial.components():
        if isinstance(base, SymbolicConstant):
            entry: dict[str, object] = {
                'type': 'symbolic',
                'symbol': base.symbol,
                'value': base.value.hex(),
            }
        elif isinstance(base, UnfactoredInteger):
            entry = {'type': 'unfactored', 'value': str(base), 'limit': str(base.limit)}
        else:
            entry = {'type': 'prime', 'value': str(base)}
        entry['exponent'] = _fraction_data(exponent)
        result.append(entry)
    return result


def _factor(data: object) -> Factor:
    components: dict[int | SymbolicConstant, Fraction] = {}
    for item in _list(data):
        record = _record(item)
        kind = _text(record.get('type'))
        if kind == 'symbolic':
            _record(record, {'type', 'symbol', 'value', 'exponent'})
            base = SymbolicConstant(_text(record['symbol']), _float(record['value']))
        elif kind in ('prime', 'unfactored'):
            keys = {'type', 'value', 'exponent'}
            if kind == 'unfactored':
                keys.add('limit')
            _record(record, keys)
            base = _integer(record['value'])
            if kind == 'unfactored':
                limit = _integer(record['limit'])
                base = UnfactoredInteger(base, limit)
        else:
            error_message = f'Unknown factor base type {kind!r}.'
            raise SerializationError(error_message)
        exponent = _fraction(record['exponent'])
        if not exponent or base in components:
            error_message = 'Factor components require unique bases and nonzero exponents.'
            raise SerializationError(error_message)
        components[base] = exponent
    return Factor.from_dict(components)


def _unit_data(unit: Unit) -> dict[str, object]:
    record: dict[str, object] = {
        'type': 'scaled_unit' if type(unit) is ScaledUnit else 'unit',
        'expression': unit.expression(),
        'identity': _unit_identity(unit),
        'dimension': [_fraction_data(exp) for exp in unit.dimension.as_tuple()],
        'factor': _factor_data(unit.factor),
        'log_degree': _fraction_data(unit.log_degree),
    }
    if type(unit) is ScaledUnit:
        record['scale'] = _factor_data(unit.scale)
    return record


def _unit_identity(unit: Unit) -> list[list[str]]:
    return sorted(
        [base.prefix_symbol, base.unprefixed_symbol, *_fraction_data(exponent)]
        for base, exponent in unit._monomial.components()
    )


def _identity(data: object) -> list[list[str]]:
    terms: list[list[str]] = []
    identities: set[tuple[str, str]] = set()
    for item in _list(data):
        term = _list(item)
        if len(term) != 4:
            error_message = 'Expected [prefix, symbol, numerator, denominator] terms.'
            raise SerializationError(error_message)
        prefix, symbol = _text(term[0]), _text(term[1])
        exponent = _fraction(term[2:])
        if not exponent or (prefix, symbol) in identities:
            error_message = 'Unit identity requires unique terms and nonzero exponents.'
            raise SerializationError(error_message)
        identities.add((prefix, symbol))
        terms.append([prefix, symbol, *_fraction_data(exponent)])
    return sorted(terms)


def _unit(data: object) -> Unit:
    record = _record(data)
    kind = _text(record.get('type'))
    keys = {'type', 'expression', 'identity', 'dimension', 'factor', 'log_degree'}
    if kind == 'scaled_unit':
        keys.add('scale')
    elif kind != 'unit':
        error_message = 'Expected a linear unit record.'
        raise SerializationError(error_message)
    _record(record, keys)
    unit = Unit(_text(record['expression']))
    if kind == 'scaled_unit':
        unit = ScaledUnit(unit, _factor(record['scale']))
    dimension = _list(record['dimension'])
    if len(dimension) != 7:
        error_message = 'Expected seven dimension exponents.'
        raise SerializationError(error_message)
    if (
        _identity(record['identity']) != _unit_identity(unit)
        or tuple(_fraction(exp) for exp in dimension) != unit.dimension.as_tuple()
        or _fraction(record['log_degree']) != unit.log_degree
        or _factor(record['factor']) != unit.factor
    ):
        error_message = 'Serialized unit meaning conflicts with the current registry.'
        raise SerializationError(error_message)
    return unit


def _encode(value: object) -> dict[str, object]:
    if type(value) is Unit or type(value) is ScaledUnit:
        return _unit_data(value)
    if type(value) is AffineUnit:
        record: dict[str, object] = {
            'type': 'affine_unit',
            'origin': _encode(value.origin),
            'delta_unit': _encode(value.delta_unit),
            'kind': value.kind,
            'symbol': value.symbol if value._definition is not None else None,
        }
        return record
    if type(value) is Quantity or type(value) is AffineQuantity:
        quantity = cast(Quantity[Magnitude] | AffineQuantity[Magnitude], value)
        return {
            'type': 'quantity' if type(quantity) is Quantity else 'affine_quantity',
            'value': _scalar_data(quantity.value),
            'unit': _encode(quantity.unit),
        }
    error_message = 'Expected a supported scalar quantity or unit, without subclassing.'
    raise TypeError(error_message)


def _decode(data: object) -> _Object:
    record = _record(data)
    kind = _text(record.get('type'))
    if kind in ('unit', 'scaled_unit'):
        return _unit(record)
    if kind == 'affine_unit':
        return _coordinate(record)
    if kind in ('quantity', 'affine_quantity'):
        _record(record, {'type', 'value', 'unit'})
        value = _scalar(record['value'])
        if kind == 'quantity':
            return Quantity(value, _unit(record['unit']))
        return AffineQuantity(value, _coordinate(record['unit']))
    error_message = f'Unknown serialized object type {kind!r}.'
    raise SerializationError(error_message)


def _coordinate(data: object) -> AffineUnit:
    record = _record(data, {'type', 'origin', 'delta_unit', 'kind', 'symbol'})
    if record['type'] != 'affine_unit':
        error_message = 'Expected an affine unit record.'
        raise SerializationError(error_message)
    origin_record = _record(record['origin'], {'type', 'value', 'unit'})
    if origin_record['type'] != 'quantity':
        error_message = 'Coordinate origin must be a linear quantity.'
        raise SerializationError(error_message)
    origin = Quantity(_scalar(origin_record['value']), _unit(origin_record['unit']))
    delta = _unit(record['delta_unit'])
    kind = record['kind']
    if kind is not None and kind not in ('power', 'amplitude'):
        error_message = 'Unknown logarithmic coordinate kind.'
        raise SerializationError(error_message)
    coordinate = AffineUnit(origin, delta_unit=delta, kind=kind)
    symbol = record['symbol']
    if symbol is None:
        return coordinate
    registered = _as_affine_unit(_text(symbol))
    # Require a registered coordinate, rather than silently using a zero-origin unit.
    if registered._definition is None or (
        registered.origin.unit != coordinate.origin.unit
        or registered.origin.value != coordinate.origin.value
        or registered.delta_unit != coordinate.delta_unit
        or registered.kind != coordinate.kind
    ):
        error_message = 'Serialized coordinate conflicts with the current registry.'
        raise SerializationError(error_message)
    return registered
