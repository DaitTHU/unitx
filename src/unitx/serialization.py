"""Versioned, JSON-compatible records for scalar quantities and units.

Use `to_dict` and `from_dict` around the standard library's JSON
functions. Decoding resolves existing units without registering definitions.
See `docs/serialization.md` for the version 2 format and its boundaries.
"""

from __future__ import annotations

# Structural exchange reads internal state, as display and pickle do, without
# adding public container interfaces to the algebra and coordinate classes.
# pyright: reportPrivateUsage=false
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import cast

from ._coordinates import as_coordinate
from ._numerics import Fractionable, Magnitude, UnfactoredInteger
from ._reconstruction import DefinitionId, _definition_id, _matches_id
from .affinequantity import AffineQuantity
from .affineunit import AffineUnit
from .dimension import Dimension
from .exceptions import SerializationError, UnitxError, VersionError
from .factor import Factor, SymbolicConstant
from .logquantity import LogQuantity
from .logunit import LogUnit
from .quantity import Quantity
from .unit import Unit
from .unitregistry import Registry

__all__ = ['from_dict', 'to_dict']

type _Object = (
    Unit
    | AffineUnit
    | LogUnit
    | Quantity[Fractionable]
    | AffineQuantity[Fractionable]
    | LogQuantity[Fractionable]
)


def to_dict(
    value: Unit
    | AffineUnit
    | LogUnit
    | Quantity[Magnitude]
    | AffineQuantity[Magnitude]
    | LogQuantity[Magnitude],
    /,
) -> dict[str, object]:
    """Return a new version 2 record, suitable for `json.dumps(..., allow_nan=False)`.

    Support exact Unit, AffineUnit, LogUnit, Quantity, AffineQuantity and LogQuantity classes.
    Magnitudes must be exact int, float, Fraction or Decimal classes;
    booleans, subclasses, complex numbers and NumPy objects are unsupported.
    No conversion or global registration is performed.

    >>> import json
    >>> q = Quantity(Fraction(1, 3), 'cm')
    >>> from_dict(json.loads(json.dumps(to_dict(q)))) == q
    True
    """
    return {'version': 2, **_encode(value)}


def from_dict(data: object, /, *, registry: Registry | None = None) -> _Object:
    """Restore a version 2 record using existing unit definitions.

    Non-dict input raises TypeError. An unsupported integer format version
    raises VersionError. Missing or malformed fields and failed restoration
    raise SerializationError. Underlying parsing, numeric and coordinate
    errors are retained as `__cause__`.
    Resolve in the explicit `registry` or the default environment, checking
    stable definition IDs and physical metadata. Version 1 is unsupported.
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
    if version != 2:
        error_message = f'Unsupported serialization format version {version}; expected 2.'
        raise VersionError(error_message)
    try:
        payload = {key: value for key, value in record.items() if key != 'version'}
        return _decode(payload, registry)
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
        'type': 'unit',
        'expression': unit.expression(),
        'scale': _factor_data(unit.scale),
        'identity': _unit_identity(unit),
        'dimension': [_fraction_data(exp) for exp in unit.dimension.as_tuple()],
        'factor': _factor_data(unit.factor),
        'log_degree': _fraction_data(unit.dimension._monomial.exponent_of('_log')),
    }
    return record


def _id_data(identity: DefinitionId) -> dict[str, object]:
    if isinstance(identity, int):
        return {'uuid': str(identity)}
    module, catalogue, symbol = identity
    return {'module': module, 'catalogue': catalogue, 'symbol': symbol}


def _id(data: object) -> DefinitionId:
    record = _record(data)
    if record.keys() == {'uuid'}:
        value = _integer(record['uuid'])
        if 0 <= value < 2**128:
            return value
        error_message = 'Expected a 128-bit UUID integer.'
        raise SerializationError(error_message)
    _record(record, {'module', 'catalogue', 'symbol'})
    return _text(record['module']), _text(record['catalogue']), _text(record['symbol'])


def _unit_identity(unit: Unit) -> list[list[object]]:
    return sorted(
        (
            [
                base.prefix_symbol,
                base.unprefixed_symbol,
                _id_data(_definition_id(base._unit)),
                *_fraction_data(exponent),
            ]
            for base, exponent in unit._monomial.components()
        ),
        key=lambda term: (term[0], term[1]),
    )


def _identity(data: object) -> list[tuple[str, str, DefinitionId, Fraction]]:
    terms: list[tuple[str, str, DefinitionId, Fraction]] = []
    identities: set[tuple[str, DefinitionId]] = set()
    for item in _list(data):
        term = _list(item)
        if len(term) != 5:
            error_message = (
                'Expected [prefix, symbol, definition ID, numerator, denominator] terms.'
            )
            raise SerializationError(error_message)
        prefix, symbol = _text(term[0]), _text(term[1])
        definition_id = _id(term[2])
        exponent = _fraction(term[3:])
        if not exponent or (prefix, definition_id) in identities:
            error_message = 'Unit identity requires unique terms and nonzero exponents.'
            raise SerializationError(error_message)
        identities.add((prefix, definition_id))
        terms.append((prefix, symbol, definition_id, exponent))
    return sorted(terms, key=lambda term: (term[0], term[1]))


def _matches_identity(unit: Unit, data: object) -> bool:
    saved = _identity(data)
    actual = sorted(
        unit._monomial.components(),
        key=lambda term: (term[0].prefix_symbol, term[0].unprefixed_symbol),
    )
    return len(saved) == len(actual) and all(
        prefix == base.prefix_symbol
        and symbol == base.unprefixed_symbol
        and exponent == actual_exponent
        and _matches_id(base._unit, identity)
        for (prefix, symbol, identity, exponent), (base, actual_exponent) in zip(saved, actual)
    )


def _unit(data: object, registry: Registry | None = None) -> Unit:
    record = _record(data)
    kind = _text(record.get('type'))
    keys = {'type', 'expression', 'scale', 'identity', 'dimension', 'factor', 'log_degree'}
    if kind != 'unit':
        error_message = 'Expected a linear unit record.'
        raise SerializationError(error_message)
    _record(record, keys)
    expression = _text(record['expression'])
    unit = Unit(expression) if registry is None else registry.parse(expression)
    unit = Unit(unit, scale=_factor(record['scale']))
    dimension = _list(record['dimension'])
    if len(dimension) != 7:
        error_message = 'Expected seven dimension exponents.'
        raise SerializationError(error_message)
    if (
        not _matches_identity(unit, record['identity'])
        or Dimension._with_log(
            Dimension._from_iter(_fraction(exp) for exp in dimension),
            _fraction(record['log_degree']),
        )
        != unit.dimension
        or _factor(record['factor']) != unit.factor
    ):
        error_message = 'Serialized unit meaning conflicts with the current registry.'
        raise SerializationError(error_message)
    return unit


def _encode(value: object) -> dict[str, object]:
    if type(value) is Unit:
        return _unit_data(value)
    if type(value) in (AffineUnit, LogUnit):
        value = cast(AffineUnit | LogUnit, value)
        record: dict[str, object] = {
            'type': 'affine_unit' if type(value) is AffineUnit else 'log_unit',
            'delta_unit': _encode(value.delta_unit),
            'kind': value._kind(),
            'symbol': value.symbol if value._definition is not None else None,
            'definition_id': _id_data(_definition_id(value._definition))
            if value._definition is not None
            else None,
        }
        if isinstance(value, LogUnit):
            record['reference'] = _encode(value.reference)
            record['at'] = _scalar_data(value.at)
        else:
            record['origin'] = _encode(value.origin)
        return record
    if type(value) in (Quantity, AffineQuantity, LogQuantity):
        quantity = cast(
            Quantity[Magnitude] | AffineQuantity[Magnitude] | LogQuantity[Magnitude], value
        )
        return {
            'type': 'quantity'
            if type(quantity) is Quantity
            else 'affine_quantity'
            if type(quantity) is AffineQuantity
            else 'log_quantity',
            'value': _scalar_data(quantity.value),
            'unit': _encode(quantity.unit),
        }
    error_message = 'Expected a supported scalar quantity or unit, without subclassing.'
    raise TypeError(error_message)


def _decode(data: object, registry: Registry | None = None) -> _Object:
    record = _record(data)
    kind = _text(record.get('type'))
    if kind == 'unit':
        return _unit(record, registry)
    if kind in ('affine_unit', 'log_unit'):
        return _coordinate(record, registry)
    if kind in ('quantity', 'affine_quantity', 'log_quantity'):
        _record(record, {'type', 'value', 'unit'})
        value = _scalar(record['value'])
        if kind == 'quantity':
            return Quantity(value, _unit(record['unit'], registry))
        coordinate = _coordinate(record['unit'], registry)
        if kind == 'affine_quantity' and isinstance(coordinate, AffineUnit):
            return AffineQuantity(value, coordinate)
        if kind == 'log_quantity' and isinstance(coordinate, LogUnit):
            return LogQuantity(value, coordinate)
        error_message = 'Serialized point type conflicts with its coordinate family.'
        raise SerializationError(error_message)
    error_message = f'Unknown serialized object type {kind!r}.'
    raise SerializationError(error_message)


def _coordinate(data: object, registry: Registry | None = None) -> AffineUnit | LogUnit:
    record = _record(data)
    logarithmic = record.get('type') == 'log_unit'
    fields = {'reference', 'at'} if logarithmic else {'origin'}
    _record(data, {'type', 'delta_unit', 'kind', 'symbol', 'definition_id'} | fields)
    if record['type'] not in ('affine_unit', 'log_unit'):
        error_message = 'Expected an affine unit record.'
        raise SerializationError(error_message)
    reference_record = _record(
        record['reference' if logarithmic else 'origin'], {'type', 'value', 'unit'}
    )
    if reference_record['type'] != 'quantity':
        error_message = 'Coordinate reference must be a linear quantity.'
        raise SerializationError(error_message)
    reference = Quantity(
        _scalar(reference_record['value']), _unit(reference_record['unit'], registry)
    )
    delta = _unit(record['delta_unit'], registry)
    kind = record['kind']
    if kind is not None and kind not in ('power', 'amplitude'):
        error_message = 'Unknown logarithmic coordinate kind.'
        raise SerializationError(error_message)
    if record['type'] == 'affine_unit' and kind is None:
        coordinate: AffineUnit | LogUnit = AffineUnit(reference, delta_unit=delta)
    elif record['type'] == 'log_unit' and kind in ('power', 'amplitude'):
        at = _scalar(record['at'])
        coordinate = LogUnit(reference, delta_unit=delta, kind=kind, at=at)
    else:
        error_message = 'Serialized coordinate type conflicts with its mapping.'
        raise SerializationError(error_message)
    symbol = record['symbol']
    if symbol is None:
        if record['definition_id'] is not None:
            error_message = 'Anonymous coordinates cannot have a definition ID.'
            raise SerializationError(error_message)
        return coordinate
    spelling = _text(symbol)
    registered = as_coordinate(spelling if registry is None else registry.parse(spelling))
    registered_reference = (
        registered.reference if isinstance(registered, LogUnit) else registered.origin
    )
    # Require a registered coordinate, rather than silently using a zero-origin unit.
    if registered._definition is None or (
        not _matches_id(registered._definition, _id(record['definition_id']))
        or registered_reference.unit != reference.unit
        or registered._mapping != coordinate._mapping
    ):
        error_message = 'Serialized coordinate conflicts with the current registry.'
        raise SerializationError(error_message)
    return registered
