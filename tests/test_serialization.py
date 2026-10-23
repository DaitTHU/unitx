"""JSON exchange preserves stored numbers, formal units and coordinate identity."""

import copy
import json
import math
import os
import subprocess
import sys
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from unitx import (
    AffineQuantity,
    AffineUnit,
    LogQuantity,
    LogUnit,
    Quantity,
    Unit,
    serialization,
    time,
)
from unitx import unitregistry as registry
from unitx._numerics import UnfactoredInteger
from unitx.exceptions import (
    DimensionError,
    IncompleteFactorWarning,
    SerializationError,
    UnitSymbolError,
    UnitSyntaxError,
    UnitxError,
    VersionError,
)
from unitx.factor import Factor, SymbolicConstant, ln10, pi
from unitx.serialization import from_dict, to_dict
from unitx.si import si


def _register(source, *args, **kwargs):
    """Explicitly bind test definitions needed by default string round trips."""
    from unitx.unitregistry import default_registry

    result = source.define(*args, **kwargs)
    default_registry.include(result)
    return result


def exchange(value, *, registry=None):
    return from_dict(json.loads(json.dumps(to_dict(value), allow_nan=False)), registry=registry)


def test_serialization_version_two_record():
    # An independently specified wire record fixes the public format.
    record = {
        'version': 2,
        'type': 'quantity',
        'value': {'type': 'fraction', 'value': ['1', '3']},
        'unit': {
            'type': 'unit',
            'expression': 'm',
            'scale': [],
            'identity': [
                [
                    '',
                    'm',
                    {'module': 'unitx.unitregistry', 'catalogue': '', 'symbol': 'm'},
                    '1',
                    '1',
                ]
            ],
            'dimension': [['0', '1'], ['1', '1'], *[['0', '1'] for _ in range(5)]],
            'factor': [],
            'log_degree': ['0', '1'],
        },
    }
    result = from_dict(record)
    assert type(result) is Quantity and type(result.value) is Fraction
    assert result.value == Fraction(1, 3) and result.unit == Unit('m')
    assert to_dict(Quantity(Fraction(1, 3), 'm')) == record


@pytest.fixture
def isolated_registry():
    tables = (
        registry.default_registry._bindings.definitions,
        registry.default_registry._bindings.names,
        registry.default_registry._bindings.aliases,
    )
    snapshots = tuple(table.copy() for table in tables)
    yield
    for table, snapshot in zip(tables, snapshots):
        table.clear()
        table.update(snapshot)


@pytest.mark.parametrize(
    'value',
    [
        0,
        -(2**100 + 1),
        0.1,
        -0.0,
        5e-324,
        float.fromhex('0x1.fffffffffffffp+1023'),
        Fraction(0),
        Fraction(-17, 31),
        Decimal('-0.000'),
        Decimal('1.2300'),
        Decimal('1E+1000'),
    ],
)
@pytest.mark.parametrize('quantity_class', [Quantity, AffineQuantity])
def test_scalar_json_roundtrip(value, quantity_class):
    quantity = quantity_class(value, 'cm')
    before = repr(quantity)
    restored = exchange(quantity)
    assert type(restored) is quantity_class
    assert type(restored.value) is type(value)
    assert restored.value == value
    if isinstance(value, Decimal):
        assert restored.value.as_tuple() == value.as_tuple()
    if isinstance(value, float):
        assert restored.value.hex() == value.hex()
    assert restored.unit == quantity.unit
    assert hash(restored.unit) == hash(quantity.unit)
    if quantity_class is Quantity:
        assert hash(restored) == hash(quantity)
    assert repr(quantity) == before


@pytest.mark.parametrize('quantity_class', [Quantity, AffineQuantity])
@pytest.mark.parametrize('value', [float('inf'), -float('inf'), float('nan')])
def test_float_specials_are_strict_json(value, quantity_class):
    restored = exchange(quantity_class(value, 'm'))
    assert type(restored.value) is float
    assert math.isnan(restored.value) if math.isnan(value) else restored.value == value


@pytest.mark.parametrize('text', ['Infinity', '-Infinity', 'NaN42', '-sNaN17'])
def test_decimal_specials_and_context(text):
    original = Quantity(Decimal(text), 's')
    with localcontext() as context:
        context.prec = 2
        context.clear_flags()
        restored = exchange(original)
        assert restored.value.as_tuple() == original.value.as_tuple()
        assert not any(context.flags.values())


@pytest.mark.parametrize(
    'unit',
    [
        Unit(),
        Unit('km/m'),
        Unit('Hz'),
        Unit('Bq'),
        Unit('um/s^(1/3)'),
        Unit('dB/m'),
        Unit('dB^2'),
        Unit('m', scale=Factor(1)),
        Unit('m/s', scale=pi ** Fraction(-1, 3)),
        Unit('cm', scale=(Factor(2) ** Fraction(1, 2)) * ln10),
    ],
)
def test_unit_identity_scale_and_future_arithmetic(unit):
    restored = exchange(unit)
    assert type(restored) is type(unit)
    assert restored == unit
    assert hash(restored) == hash(unit)
    assert {unit: 'same'}[restored] == 'same'
    assert restored.factor == unit.factor
    assert restored**3 / restored == unit**2
    quantity = exchange(Quantity(Fraction(2, 7), unit))
    assert quantity.unit == unit
    assert quantity.value_in(unit) == Fraction(2, 7)


@pytest.mark.parametrize('name', ['degC', 'dBm', 'dBW', 'dBV', 'K', 'ns'])
def test_registered_and_zero_origin_coordinates(name):
    quantity_type = LogQuantity if name in ('dBm', 'dBW', 'dBV') else AffineQuantity
    quantity = quantity_type(Decimal('12.500'), name)
    for original in (quantity, quantity.unit):
        restored = exchange(original)
        assert restored == original
        if isinstance(original, (AffineUnit, LogUnit)):
            assert hash(restored) == hash(original)
            assert restored.symbol == original.symbol
    physical_unit = quantity.unit.origin.unit
    assert exchange(quantity).to_quantity(physical_unit) == quantity.to_quantity(physical_unit)


@pytest.mark.parametrize('kind', [None, 'power', 'amplitude'])
def test_anonymous_coordinates_preserve_origin_and_explicit_scales(kind):
    if kind is None:
        coordinate = AffineUnit(
            Quantity(Fraction(-7, 3), Unit('s', scale=Factor(3))),
            delta_unit=Unit('ms', scale=Factor(7)),
        )
    else:
        coordinate = LogUnit(
            Quantity(Fraction(2, 3), Unit('W', scale=Factor(3))),
            delta_unit=Unit('dB', scale=Factor(2)),
            kind=kind,
        )
    point = (AffineQuantity if kind is None else LogQuantity)(Fraction(5, 4), coordinate)
    restored = exchange(point)
    assert restored.unit == coordinate
    assert restored.unit.origin.unit == coordinate.origin.unit
    assert restored.unit.delta_unit == coordinate.delta_unit
    if kind is not None:
        assert restored.unit.kind == kind
    assert restored.to_quantity(coordinate.origin.unit) == point.to_quantity(coordinate.origin.unit)
    assert (restored + Quantity(1, coordinate.delta_unit)) - restored == Quantity(
        1, coordinate.delta_unit
    )


def test_datetime_known_epoch():
    from datetime import datetime, timezone

    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    coordinate = AffineUnit(time.from_datetime(epoch), delta_unit='s')
    point = Fraction(123456789, 1000) @ coordinate
    assert time.to_datetime(exchange(point)) == time.to_datetime(point)


def test_decimal_context_cannot_round_origin_or_magnitude():
    coordinate = AffineUnit(Quantity(Decimal('123456.789'), 's'), delta_unit='ms')
    point = AffineQuantity(Decimal('0.000000123400'), coordinate)
    data = to_dict(point)
    with localcontext() as context:
        context.prec = 2
        context.clear_flags()
        restored = from_dict(data)
        assert restored.unit.origin.value == Fraction(123456789, 1000)
        assert restored.value.as_tuple() == point.value.as_tuple()
        assert not any(context.flags.values())


def test_custom_names_prefix_permissions_and_quoted_symbols(isolated_registry):
    custom = _register(
        Quantity(Fraction(3, 7), 'm'),
        "sx'\\",
        'serialization length',
        prefixable='serialization length',
    )
    prefixed = Unit('[kiloserialization length] / s')
    for original in (custom, prefixed, Quantity(Decimal('1.20'), prefixed)):
        assert exchange(original) == original
    coordinate = _register(
        AffineUnit(Quantity(2, custom), delta_unit=custom), 'sxepoch', 'serialization epoch'
    )
    assert exchange(3 @ coordinate).unit == coordinate
    assert exchange(coordinate).symbol == 'sxepoch'


def test_unknown_units_fail_without_registration(isolated_registry):
    unit = _register(Quantity(2, 'm'), 'sxmissing', 'serialization missing')
    data = to_dict(Quantity(3, unit))
    registry.default_registry._bindings.definitions.pop('sxmissing')
    snapshots = [
        dict(getattr(registry.default_registry._bindings, name))
        for name in ('definitions', 'names', 'aliases')
    ]
    with pytest.raises(SerializationError) as caught:
        from_dict(data)
    assert isinstance(caught.value.__cause__, UnitSymbolError)
    assert snapshots == [
        dict(getattr(registry.default_registry._bindings, name))
        for name in ('definitions', 'names', 'aliases')
    ]


@pytest.mark.parametrize('replacement', [Quantity(3, 'm'), Quantity(2, 's')])
def test_same_symbol_changed_meaning_rejected(isolated_registry, replacement):
    unit = _register(Quantity(2, 'm'), 'sxdrift', 'serialization drift')
    data = to_dict(unit)
    registry.default_registry._bindings.definitions.pop('sxdrift')
    registry.default_registry._bindings.names.pop('serialization drift')
    _register(replacement, 'sxdrift', 'serialization drift')
    with pytest.raises(SerializationError, match='conflicts'):
        from_dict(data)


def test_equal_physical_scale_does_not_replace_formal_identity():
    data = to_dict(Unit('Hz'))
    data['expression'] = 'Bq'
    with pytest.raises(SerializationError, match='conflicts'):
        from_dict(data)


def test_registry_does_not_rebind_changed_alias_identity(isolated_registry):
    # Prefix permission can make an alias the reconstruction spelling. Moving
    # that spelling to an equal-scale distinct unit must not change identity.
    source = _register(
        Unit('m'),
        'sxprimary',
        'serialization primary',
        symbol_aliases='sxalias',
        prefixable='sxalias',
    )
    data = to_dict(Unit('ksxalias'))
    registry.default_registry._bindings.aliases.pop('sxalias')
    _register(source, 'sxalias', 'serialization alias')
    with pytest.raises(SerializationError, match='conflicts'):
        from_dict(data)


def test_coordinate_reference_cannot_drift(isolated_registry):
    coordinate = _register(AffineUnit(Quantity(2, 's')), 'sxcoord', 'serialization coordinate')
    data = to_dict(coordinate)
    registry.default_registry._bindings.definitions.pop('sxcoord')
    registry.default_registry._bindings.names.pop('serialization coordinate')
    _register(AffineUnit(Quantity(3, 's')), 'sxcoord', 'serialization coordinate')
    with pytest.raises(SerializationError, match='conflicts'):
        from_dict(data)


def test_unfactored_scale_retains_formal_state():
    with pytest.warns(IncompleteFactorWarning):
        scale = Factor.from_dict({UnfactoredInteger(1000003 * 1000033, 1000): Fraction(1, 3)})
    quantity = Quantity(Fraction(1, 7), Unit('m', scale=scale))
    with pytest.warns(IncompleteFactorWarning):
        restored = exchange(quantity)
    assert restored.unit.scale == scale
    (base,) = restored.unit.scale._monomial.bases()
    assert type(base) is UnfactoredInteger
    assert base.limit == 1000


def test_symbolic_scale_is_reconstructed_without_evaluating_label():
    constant = SymbolicConstant('e', math.e)
    unit = Unit('m', scale=constant ** Fraction(1, 3))
    assert exchange(unit).scale == unit.scale


@pytest.mark.parametrize('data', [None, [], 'text', 1])
def test_records_require_dictionaries(data):
    with pytest.raises(TypeError, match='dictionary'):
        from_dict(data)


@pytest.mark.parametrize('version', [None, True, '1', 1.0])
def test_version_is_required_exact_integer(version):
    data = to_dict(Unit('m'))
    data['version'] = version
    with pytest.raises(SerializationError, match='version'):
        from_dict(data)


@pytest.mark.parametrize('version', [0, -1, 1, 3])
def test_serialization_invalid_unsupported_version(version):
    data = to_dict(Unit('m'))
    data['version'] = version
    with pytest.raises(VersionError, match=f'version {version}; expected 2') as caught:
        from_dict(data)
    assert type(caught.value) is VersionError
    assert isinstance(caught.value, (UnitxError, ValueError))
    assert not isinstance(caught.value, SerializationError)


def test_serialization_error_catch_boundaries():
    for error_class in (SerializationError, VersionError):
        error = error_class('test error')
        assert isinstance(error, UnitxError)
        assert isinstance(error, ValueError)
    with pytest.raises(SerializationError) as caught:
        from_dict({'version': 2, 'type': 'quantity'})
    assert type(caught.value) is SerializationError
    assert caught.value.__cause__ is None
    with pytest.raises(TypeError) as caught_type:
        from_dict([])
    assert not isinstance(caught_type.value, UnitxError)


@pytest.mark.parametrize(
    'expression, cause', [('missingserializationunit', UnitSymbolError), ('m +', UnitSyntaxError)]
)
def test_serialization_parse_error_cause(expression, cause):
    data = to_dict(Unit('m'))
    data['expression'] = expression
    with pytest.raises(SerializationError) as caught:
        from_dict(data)
    assert isinstance(caught.value.__cause__, cause)
    assert not isinstance(caught.value.__cause__, SerializationError)


@pytest.mark.parametrize(
    'kind, value, cause',
    [
        ('int', 'bad integer', ValueError),
        ('float', '0x1p+99999999', OverflowError),
        ('decimal', 'bad decimal', InvalidOperation),
    ],
)
def test_serialization_numeric_error_cause(kind, value, cause):
    data = to_dict(Quantity(1, 'm'))
    data['value'] = {'type': kind, 'value': value}
    with pytest.raises(SerializationError) as caught:
        from_dict(data)
    assert isinstance(caught.value.__cause__, cause)


def test_serialization_internal_error_propagates(monkeypatch):
    def fail(data, registry):
        error_message = 'Unexpected implementation failure.'
        raise RuntimeError(error_message)

    monkeypatch.setattr(serialization, '_decode', fail)
    with pytest.raises(RuntimeError, match='Unexpected implementation failure'):
        from_dict(to_dict(Unit('m')))


@pytest.mark.parametrize('field', ['version', 'type', 'expression', 'factor', 'identity'])
def test_missing_fields_fail(field):
    data = to_dict(Unit('m'))
    data.pop(field)
    with pytest.raises(SerializationError):
        from_dict(data)


def test_unknown_fields_and_tags_rejected():
    data = to_dict(Unit('m'))
    data['extra'] = None
    with pytest.raises(SerializationError, match='fields'):
        from_dict(data)
    data = to_dict(Quantity(1, 'm'))
    data['type'] = 'eval'
    with pytest.raises(SerializationError, match='Unknown'):
        from_dict(data)
    data = to_dict(Quantity(1, 'm'))
    data['value']['type'] = 'complex'
    with pytest.raises(SerializationError, match='Unknown'):
        from_dict(data)


@pytest.mark.parametrize(
    'field, value',
    [
        ('identity', None),
        ('identity', [[]]),
        (
            'identity',
            [['', 'm', {'module': 'unitx.unitregistry', 'catalogue': '', 'symbol': 'm'}, '0', '1']],
        ),
        (
            'identity',
            [
                [
                    '',
                    'm',
                    {'module': 'unitx.unitregistry', 'catalogue': '', 'symbol': 'm'},
                    '1',
                    '1',
                ],
                [
                    '',
                    'm',
                    {'module': 'unitx.unitregistry', 'catalogue': '', 'symbol': 'm'},
                    '2',
                    '1',
                ],
            ],
        ),
        ('dimension', []),
        ('dimension', [['0', '0']] * 7),
        ('log_degree', '0'),
        ('factor', None),
        ('factor', [{'type': 'unknown'}]),
        ('factor', [{'type': 'prime', 'value': '4', 'exponent': ['1', '1']}]),
        ('factor', [{'type': 'prime', 'value': '2', 'exponent': ['0', '1']}]),
        ('factor', [{'type': 'prime', 'value': '2', 'exponent': ['1', '1']}] * 2),
    ],
)
def test_invalid_unit_metadata_is_rejected_without_mutation(field, value):
    data = to_dict(Unit('m'))
    data[field] = value
    before = copy.deepcopy(data)
    registry_before = dict(registry.default_registry._bindings.definitions)
    with pytest.raises(SerializationError):
        from_dict(data)
    assert data == before
    assert registry.default_registry._bindings.definitions == registry_before


def test_nested_type_boundaries():
    linear = to_dict(Quantity(1, 'K'))
    linear['unit'] = to_dict(AffineQuantity(0, 'degC'))['unit']
    with pytest.raises(SerializationError, match='linear unit'):
        from_dict(linear)
    point = to_dict(AffineQuantity(20, 'degC'))
    point['unit'] = to_dict(Quantity(1, 'K'))['unit']
    with pytest.raises(SerializationError, match='fields'):
        from_dict(point)
    coordinate = to_dict(AffineQuantity(20, 'degC').unit)
    coordinate['origin']['type'] = 'affine_quantity'
    with pytest.raises(SerializationError, match='linear quantity'):
        from_dict(coordinate)


def test_invalid_coordinate_kind_and_dimension():
    data = to_dict(AffineUnit(Quantity(0, 's')))
    data['kind'] = 'invalid'
    with pytest.raises(SerializationError, match='kind'):
        from_dict(data)
    data['kind'] = None
    data['delta_unit'] = to_dict(Unit('m'))
    data['delta_unit'].pop('version')
    with pytest.raises(SerializationError) as caught:
        from_dict(data)
    assert isinstance(caught.value.__cause__, DimensionError)


def test_factor_symbol_conflicts_are_rejected():
    data = to_dict(Unit('m', scale=pi))
    data['scale'] = [
        {'type': 'symbolic', 'symbol': 'x', 'value': value.hex(), 'exponent': ['1', '1']}
        for value in (math.pi, math.e)
    ]
    with pytest.raises(SerializationError) as caught:
        from_dict(data)
    assert isinstance(caught.value.__cause__, ValueError)
    assert 'Conflicting values' in str(caught.value.__cause__)


@pytest.mark.parametrize(
    'scalar',
    [
        {'type': 'int', 'value': 3},
        {'type': 'int', 'value': '01'},
        {'type': 'fraction', 'value': ['1', '0']},
        {'type': 'fraction', 'value': ['1', '-2']},
        {'type': 'fraction', 'value': [1, 2]},
        {'type': 'float', 'value': '0.1'},
        {'type': 'float', 'value': '0x1p+99999999'},
        {'type': 'decimal', 'value': 'no number'},
    ],
)
def test_invalid_scalars_rejected(scalar):
    data = to_dict(Quantity(1, 'm'))
    data['value'] = scalar
    with localcontext() as context:
        context.traps[InvalidOperation] = False
        context.clear_flags()
        with pytest.raises(SerializationError):
            from_dict(data)
        assert not any(context.flags.values())


@pytest.mark.parametrize('value', [True, 1j, '1'])
def test_unsupported_magnitudes_rejected(value):
    with pytest.raises(TypeError, match='magnitude'):
        to_dict(Quantity(value, 'm'))


def test_subclasses_are_not_silently_normalized():
    class Integer(int):
        pass

    class CustomQuantity(Quantity):
        pass

    class CustomUnit(Unit):
        pass

    for original in (CustomQuantity(1, 'm'), CustomUnit('m'), Quantity(1, CustomUnit('m'))):
        with pytest.raises(TypeError, match='subclassing'):
            to_dict(original)
    with pytest.raises(TypeError, match='magnitude'):
        to_dict(Quantity(Integer(1), 'm'))


def test_numpy_is_explicitly_unsupported():
    np = pytest.importorskip('numpy')
    for value in (np.float64(1), np.int64(1), np.array(1), np.array([1, 2])):
        with pytest.raises(TypeError, match='magnitude'):
            to_dict(Quantity(value, 'm'))


def test_result_and_input_records_do_not_alias():
    original = Quantity(Fraction(1, 3), Unit('m', scale=pi))
    data = to_dict(original)
    before = copy.deepcopy(data)
    restored = from_dict(data)
    assert data == before
    data['value']['value'][0] = '100'
    assert restored.value == original.value == Fraction(1, 3)
    assert to_dict(original) == before


def test_cross_process_json_without_numpy():
    quantity = AffineQuantity(Fraction(123, 7), si.degF)
    source = Path(__file__).resolve().parents[1] / 'src'
    script = f"""import sys, json, importlib.util
sys.path.insert(0, {str(source)!r})
assert importlib.util.find_spec('numpy') is None
from unitx.serialization import from_dict, to_dict
from unitx.si import si
from unitx.si import si
restored = from_dict(json.load(sys.stdin), registry=si._registry)
print(json.dumps(to_dict(restored), allow_nan=False))
assert 'numpy' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, '-I', '-S', '-c', script],
        input=json.dumps(to_dict(quantity)),
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
        env=os.environ.copy(),
    )
    assert from_dict(json.loads(result.stdout), registry=si._registry) == quantity


@pytest.mark.parametrize(
    'identity',
    [
        'unitx.si:m',
        {},
        {'module': 'unitx.unitregistry', 'catalogue': ''},
        {'module': 1, 'catalogue': '', 'symbol': 'm'},
        {'module': 'unitx.unitregistry', 'catalogue': '', 'symbol': 'm', 'extra': ''},
        {'uuid': '0', 'symbol': 'm'},
        {'uuid': True},
        {'uuid': 1},
        {'uuid': '-1'},
        {'uuid': str(2**128)},
        {'uuid': '01'},
    ],
)
def test_serialization_invalid_definition_id(identity):
    record = to_dict(Unit('m'))
    record['identity'][0][2] = identity
    with pytest.raises(SerializationError):
        from_dict(record)
