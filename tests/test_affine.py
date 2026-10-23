"""Coordinate semantics, exact linear conversions, and logarithmic differences."""

import copy
import math
import operator
import pickle
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx import LogQuantity, LogUnit
from unitx.affinequantity import AffineQuantity
from unitx.affineunit import AffineUnit
from unitx.exceptions import DimensionError, UnitRegistryError
from unitx.factor import Factor, SymbolicConstant, ln10, pi
from unitx.quantity import Quantity
from unitx.si import si
from unitx.unit import Unit
from unitx.unitregistry import default_registry


def _register(source, *args, **kwargs):
    """Explicitly bind test definitions needed by default string round trips."""
    from unitx.unitregistry import default_registry

    result = source.define(*args, **kwargs)
    default_registry.include(result)
    return result


@pytest.mark.parametrize(
    'x, source, target, expected',
    [
        (0, 'degC', 'K', Fraction(27315, 100)),
        (100, 'degC', si.degF, 212),
        (32, si.degF, 'degC', 0),
        (0, si.degF, 'K', Fraction(45967, 180)),
        (10, si.degR, 'K', Fraction(50, 9)),
        (30, 'dBm', 'dBW', 0),
        (0, 'dBW', 'dBm', 30),
        (0, 'dBm', 'mW', 1),
        (20, 'dBm', 'mW', 100),
        (20, 'dBV', 'V', 10),
    ],
)
def test_reference_conversion(x, source, target, expected):
    quantity_type = (
        LogQuantity if isinstance(source, str) and source.startswith('dB') else AffineQuantity
    )
    q = quantity_type(x, source)
    physical_exit = (
        quantity_type is LogQuantity and isinstance(target, str) and target in ('W', 'mW', 'V')
    )
    result = q.to_quantity(target) if physical_exit else q.to(target)
    value = result.value
    assert value == expected
    if not physical_exit:
        assert value == q.value_in(target)
    assert q.value == x


@pytest.mark.parametrize('precision', [2, 8, 28])
def test_decimal_round_once(precision):
    with localcontext() as ctx:
        ctx.prec = precision
        value = Decimal('1.23456789')
        result = AffineQuantity(value, si.degF).to('degC').value
        exact = (Fraction(value) - 32) * Fraction(5, 9)
        assert result == Decimal(exact.numerator) / Decimal(exact.denominator)
        assert isinstance(result, Decimal)
        assert (LogQuantity(Decimal('30'), 'dBm') - LogQuantity(0, 'dBW')).value == 0
        assert LogQuantity(Decimal(20), 'dBm').to_quantity('mW').value == Decimal(100)


def test_point_and_difference_arithmetic():
    assert AffineQuantity(30, 'degC') - AffineQuantity(68, si.degF) == Quantity(10, 'K')
    assert AffineQuantity(20, 'degC') + Quantity(18, si.degR) == AffineQuantity(30, 'degC')
    assert Quantity(18, si.degR) + AffineQuantity(20, 'degC') == AffineQuantity(30, 'degC')
    assert LogQuantity(30, 'dBm') - LogQuantity(-10, 'dBW') == Quantity(10, 'dB')
    assert LogQuantity(20, 'dBm') + Quantity(10, 'dB') == LogQuantity(30, 'dBm')
    assert LogQuantity(20, 'dBV') - Quantity(10, 'dB') == LogQuantity(10, 'dBV')


@pytest.mark.parametrize('op', [operator.add, operator.mul, operator.truediv])
def test_point_forbidden_binary(op):
    with pytest.raises(TypeError):
        op(AffineQuantity(10, 'degC'), AffineQuantity(20, 'degC'))


@pytest.mark.parametrize(
    'op',
    [
        operator.neg,
        operator.abs,
        lambda q: q**0,
        lambda q: q * 2,
        lambda q: 2 / q,
        lambda q: q / 2,
        lambda q: Quantity(2, 'm') * q,
    ],
)
def test_point_forbidden_other(op):
    with pytest.raises(TypeError):
        op(LogQuantity(20, 'dBm'))


@pytest.mark.parametrize('text', ['degC/m', 'dBm/Hz', 'degC/degC', 'degC**0', 'dBm**2', '/degC'])
def test_unit_linear_products(text):
    assert isinstance(Unit(text), Unit)


def test_logarithmic_algebra_and_ratio():
    assert ln10 == SymbolicConstant('ln(10)', math.log(10))
    assert pickle.loads(pickle.dumps(ln10)) == ln10
    assert Unit('B').factor == ln10 / 2
    assert Quantity(20, 'dB').to('B') == Quantity(2, 'B')
    assert Quantity(20, 'dB').to_ratio(kind='power') == Quantity(100, '')
    assert Quantity(20, 'dB').to_ratio(kind='amplitude') == Quantity(10, '')
    assert Quantity(100, '').to_log('dB', kind='power').value == 20
    assert Quantity(1, 'km/m').to_log('dB', kind='power').value == 30
    assert Unit('dB/m').dimension == (si.dB / si.m).dimension
    assert Unit('dB**2').dimension == si.dB.dimension**2
    assert Unit('dB/Np').dimension == si.one.dimension
    assert (Quantity(2, 'dB/m') * Quantity(10, 'm')).to_ratio(kind='power').value == 100
    assert hash(Quantity(0, 'dB')) == hash(Quantity(0, 'Np'))
    assert not (Quantity(0, 'dB') == 0)
    with pytest.raises(DimensionError):
        Quantity(20, 'dB').to('')
    with pytest.raises(DimensionError):
        Quantity(1, 'dB/m').to_ratio(kind='power')
    with pytest.raises(TypeError):
        Quantity(1, 'dB') + 1


def test_point_comparison_and_hash():
    assert AffineQuantity(0, 'degC') == AffineQuantity(Fraction(27315, 100), 'K')
    assert LogQuantity(30, 'dBm') == LogQuantity(0, 'dBW')
    assert not (LogQuantity(0, 'dBm') == Quantity(1, 'mW'))
    with pytest.raises(DimensionError):
        LogQuantity(0, 'dBm') < Quantity(1, 'mW')
    with pytest.raises(DimensionError):
        LogQuantity(0, 'dBm') - LogQuantity(0, 'dBV')
    with pytest.raises(TypeError):
        hash(AffineQuantity(1, 'degC'))
    assert AffineQuantity(2**53 + 1, 'degC') != AffineQuantity(float(2**53 + 1), 'degC')


def test_affine_tolerances():
    q = AffineQuantity(20, 'degC')
    assert q.isclose(AffineQuantity(21, 'degC'), rel_tol=0, abs_tol=Quantity(1, 'K'))
    with pytest.raises(ValueError, match='rel_tol=0'):
        q.isclose(q, rel_tol=1e-9)
    with pytest.raises(TypeError):
        q.isclose(q, rel_tol=0, abs_tol=AffineQuantity(1, 'degC'))
    with pytest.raises(TypeError):
        q.isclose(q, rel_tol=0, abs_tol=1)


@pytest.mark.parametrize('name', ['degC', si.degF, 'dBm', 'dBV'])
def test_identity_copy_pickle_and_display(name):
    quantity_type = LogQuantity if name in ('dBm', 'dBV') else AffineQuantity
    unit = quantity_type(0, name).unit
    assert isinstance(unit, (AffineUnit, LogUnit))
    assert unit == pickle.loads(pickle.dumps(unit))
    assert copy.copy(unit) is unit
    assert copy.deepcopy(unit) is unit
    assert eval(repr(unit)) == unit
    q = quantity_type(1, unit)
    assert eval(repr(q)).value == q.value
    assert unit.symbol in q.to_html()
    assert unit.symbol in q.to_latex()
    assert 'parse-units=false' in q.to_siunitx()
    assert (-20) @ unit == quantity_type(-20, name)
    with pytest.raises(AttributeError):
        unit._logarithm = None


def test_custom_definition_and_atomic_conflict():
    tables = (
        default_registry._bindings.definitions,
        default_registry._bindings.names,
        default_registry._bindings.aliases,
    )
    snapshots = [dict(x) for x in tables]
    try:
        coordinate = AffineUnit(Quantity(10, 'K'), delta_unit=Unit('K'))
        assert pickle.loads(pickle.dumps(coordinate)) == coordinate
        registered = _register(coordinate, 'refTemp', 'reference temperature')
        assert AffineQuantity(2, registered).value_in('K') == 12
        before = [dict(x) for x in tables]
        with pytest.raises(UnitRegistryError):
            _register(coordinate, 'refTemp', 'another reference temperature')
        assert [dict(x) for x in tables] == before
        log = _register(Factor(2) * Unit('dB'), 'doubleDecibel', 'double decibel')
        assert log.dimension == si.dB.dimension
        assert Quantity(10, log).to_ratio(kind='power').value == 100
    finally:
        for table, snapshot in zip(tables, snapshots):
            table.clear()
            table.update(snapshot)


def test_domains_and_reference_validation():
    with pytest.raises(ValueError):
        LogUnit(Quantity(0, 'W'), delta_unit=Unit('dB'), kind='power')
    with pytest.raises(DimensionError):
        AffineUnit(Quantity(1, 'W'), delta_unit=Unit('K'))
    with pytest.raises(ValueError):
        LogQuantity.from_quantity(Quantity(-1, 'W'), 'dBm')
    assert LogQuantity.from_quantity(Quantity(0, 'W'), 'dBm').value == -math.inf
    assert LogQuantity(-math.inf, 'dBm').to_quantity('W').value == 0
    with pytest.raises(TypeError):
        AffineQuantity(1j, 'degC')
    with pytest.raises(TypeError):
        (pi * Unit('m')) * AffineQuantity(0, 'degC')


def test_exact_zero_tolerance_and_log_reference_alignment():
    point = AffineQuantity(0.1, 'degC')
    other = AffineQuantity(Decimal('273.25'), 'K')
    assert not (point == other)
    assert point.isclose(other, rel_tol=0, abs_tol=0) == (point == other)
    with pytest.raises(TypeError):
        LogQuantity(0, 'dBm').isclose(Quantity(1, 'mW'), rel_tol=0, abs_tol=0)
    with pytest.raises(TypeError):
        AffineQuantity(True, 'degC')


def test_cross_log_scale_comparison_retains_symbolic_identity():
    np_watt = LogUnit(Quantity(1, 'W'), delta_unit=Unit('Np'), kind='power')
    assert LogQuantity(0, np_watt) == LogQuantity(0, 'dBW')
    converted = LogQuantity(1, 'dBW').to(np_watt)
    assert converted != LogQuantity(1, 'dBW')
    assert not converted.isclose(LogQuantity(1, 'dBW'), rel_tol=0, abs_tol=0)
    with pytest.raises(TypeError, match='approximate'):
        LogQuantity(0, np_watt) == LogQuantity(30, 'dBm')


def test_affine_unit_independent_type():
    from unitx.unit import Unit

    coordinate = AffineQuantity(0, 'degC').unit
    assert isinstance(si.degC, AffineUnit)
    assert isinstance(Unit('degC'), Unit)
    assert not isinstance(coordinate, Unit)
    assert not issubclass(AffineUnit, Unit)
    assert not hasattr(coordinate, 'factor')
    assert not hasattr(coordinate, '_monomial')
    assert Unit(coordinate.expression()) == Unit('degC')
    assert not (si.K == coordinate)
    assert not (coordinate == si.K)
    assert Quantity(1, 'K').to('degC') == Quantity(1, 'degC')
    assert Quantity(1, si.degF).to('K') == Quantity(Fraction(5, 9), 'K')
    assert isinstance(Unit('degC', scale=Factor(2)), Unit)
    assert AffineUnit(AffineQuantity(1, 'degC')).origin == Quantity(Fraction(5483, 20), si.K)


def test_unit_definition_coordinate_metadata():
    coordinate = AffineUnit(10 * si.K).define('referenceStep')
    definition = coordinate._definition
    assert definition is not None
    assert definition._mapping is coordinate._mapping
    assert definition._mapping.origin == 10 * si.K
    assert definition._mapping.delta_unit == si.K
    assert definition.factor == Factor(1)
    assert definition.dimension == si.K.dimension
    assert definition.prefixable is False
    assert default_registry._bindings.definitions[si.dBm.symbol]._mapping is si.dBm._mapping
    assert si.dBm._mapping.origin == 1 * si.mW
    assert not hasattr(default_registry._bindings.definitions['K'], '_mapping')
    assert not hasattr(default_registry._bindings.definitions['K'], 'coordinate')


@pytest.mark.parametrize(
    'spelling, coordinate',
    [
        ('(degC)', si.degC),
        ('[degree Celsius]', si.degC),
        ('degC * 1', si.degC),
        ('(dBm)', si.dBm),
        ('[decibel-milliwatt]', si.dBm),
        ('dBm * 1', si.dBm),
    ],
)
def test_affine_quantity_coordinate_expression(spelling, coordinate):
    expected = 3 @ coordinate
    quantity_type = LogQuantity if isinstance(coordinate, LogUnit) else AffineQuantity
    assert quantity_type(3, spelling) == expected
    assert quantity_type(3, Unit(spelling)) == expected


def test_logarithmic_complex_hash_uses_exact_component_keys():
    left = Quantity(1 + 2j, 'B')
    right = Quantity(10 + 20j, 'dB')
    assert left == right
    assert hash(left) == hash(right)
    assert {left: 'value'}[right] == 'value'
    with pytest.raises(TypeError):
        left.to_ratio(kind='power')


@pytest.mark.parametrize('value', [0j, 1j])
def test_point_invalid_complex(value):
    assert Quantity(value, 'K').to('degC').value == value
    with pytest.raises(TypeError):
        AffineQuantity(value, 'K')
    with pytest.raises(TypeError):
        AffineQuantity(0, 'degC').isclose(AffineQuantity(0, 'degC'), rel_tol=0j)
    with pytest.raises(TypeError):
        AffineQuantity(0, 'degC').isclose(AffineQuantity(0, 'degC'), abs_tol=Quantity(0j, 'K'))


@pytest.mark.parametrize('precision', [2, 8, 28])
def test_affine_unit_reference_point(precision):
    with localcontext() as ctx:
        ctx.prec = precision
        celsius = AffineUnit(Decimal('273.15') * si.K)
        fahrenheit = AffineUnit(0 @ celsius, at=Decimal('32'), delta_unit=si.degR)
        assert celsius.delta_unit == si.K
        assert celsius.origin.value == Fraction(5463, 20)
        assert fahrenheit.origin.value == Fraction(45967, 180)
        assert fahrenheit.origin.unit == si.K
        assert AffineQuantity(32, fahrenheit).value_in(celsius) == 0
        assert AffineQuantity(212, fahrenheit).value_in(celsius) == 100
        assert AffineUnit(0 @ fahrenheit).delta_unit == si.degR
        assert AffineUnit(0 @ fahrenheit).origin == fahrenheit.origin
        assert hash(fahrenheit) == hash(pickle.loads(pickle.dumps(fahrenheit)))
        assert eval(repr(fahrenheit)) == fahrenheit
        assert copy.copy(fahrenheit) is fahrenheit
        assert copy.deepcopy(fahrenheit) is fahrenheit


def test_affine_unit_log_reference_point():
    unit = LogUnit(1 * si.W, at=30, delta_unit=si.dB, kind='power')
    assert unit.origin == 1 * si.mW
    assert LogQuantity(30, unit).to_quantity(si.W).value == 1
    assert LogQuantity(0, unit).to_quantity(si.mW).value == 1
    nested = LogUnit(LogQuantity(30, si.dBm), at=30, delta_unit=si.dB, kind='power')
    assert nested.origin == 1 * si.mW
    assert not isinstance(nested.origin.unit, AffineUnit)
    amplitude = LogUnit(10 * si.V, at=20, delta_unit=si.dB, kind='amplitude')
    assert amplitude.origin == 1 * si.V
    assert LogQuantity(20, amplitude).to_quantity(si.V).value == 10
    assert pickle.loads(pickle.dumps(unit)) == unit
    assert eval(repr(unit)) == unit


def test_affine_unit_reference_registration():
    tables = (
        default_registry._bindings.definitions,
        default_registry._bindings.names,
        default_registry._bindings.aliases,
    )
    snapshots = [dict(x) for x in tables]
    try:
        unit = _register(
            AffineUnit(AffineQuantity(0, si.degC), at=32, delta_unit=si.degR),
            'customFahrenheit',
            'custom Fahrenheit',
            symbol_aliases='customF',
        )
        assert unit.origin == Quantity(Fraction(45967, 180), si.K)
        assert (
            unit == AffineQuantity(0, 'customF').unit == AffineQuantity(0, 'custom Fahrenheit').unit
        )
        assert AffineQuantity(212, unit).value_in(si.degC) == 100
        assert hash(unit) == hash(pickle.loads(pickle.dumps(unit)))
        assert eval(repr(unit)) == unit
        definition = default_registry._bindings.definitions['customFahrenheit']
        assert definition._mapping is unit._mapping
        assert definition._mapping.origin == Quantity(Fraction(45967, 180), si.K)
    finally:
        for table, snapshot in zip(tables, snapshots):
            table.clear()
            table.update(snapshot)


@pytest.mark.parametrize('reference', ['degC', 'm', 1, None])
def test_affine_unit_invalid_reference(reference):
    with pytest.raises(TypeError, match='Reference must be a Quantity'):
        AffineUnit(reference)


@pytest.mark.parametrize('at', [True, 1j, '32', Quantity(32, ''), [32]])
def test_affine_unit_invalid_reading_type(at):
    with pytest.raises(TypeError, match='at must have a real Python scalar'):
        AffineUnit(0 * si.K, at=at)


@pytest.mark.parametrize('at', [float('nan'), float('inf'), Decimal('NaN'), Decimal('-Infinity')])
def test_affine_unit_invalid_reading_finiteness(at):
    with pytest.raises(ValueError, match='at must be finite'):
        AffineUnit(0 * si.K, at=at)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), Decimal('NaN')])
def test_affine_unit_invalid_reference_finiteness(value):
    with pytest.raises(ValueError, match='Reference must be finite'):
        AffineUnit(Quantity(value, si.K))


def test_affine_unit_invalid_log_default():
    with pytest.raises(TypeError, match='delta_unit'):
        LogUnit(1 * si.W, kind='power')


def test_affine_unit_stored_float():
    unit = AffineUnit(Quantity(0.1, si.K), at=0.1)
    assert unit.origin.value == 0
    unit = AffineUnit(Quantity(0.1, si.K))
    assert unit.origin.value == Fraction(0.1)
    assert unit.origin.value != Fraction(1, 10)


def test_affine_unit_reference_array_invalid():
    np = pytest.importorskip('numpy')
    with pytest.raises(TypeError, match='Reference must have a real Python scalar'):
        AffineUnit(Quantity(np.array([0, 1]), si.K))


@pytest.mark.parametrize(
    'unit',
    [si.degC, si.degF, si.dBm] + [AffineUnit(100 * si.K)],
)
def test_affine_unit_independent_display(unit):
    assert str(unit) == unit.symbol
    assert format(unit, '>20') == format(unit.symbol, '>20')
    assert format(unit, 'text') == unit.symbol
    assert format(unit, 'latex') == unit.to_latex()
    assert format(unit, 'html') == unit.to_html()
    assert format(unit, 'siunitx') == unit.to_siunitx()
    assert unit._repr_latex_() == '$' + unit.to_latex() + '$'
    assert unit._repr_html_() == unit.to_html()
    assert copy.copy(unit) is unit
    assert copy.deepcopy(unit) is unit
    assert not unit.is_dimensionless()


def test_affine_quantity_point_identity():
    point = AffineQuantity(20, 'degC')
    delta = point - AffineQuantity(10, 'degC')
    assert type(point) is AffineQuantity
    assert type(delta) is Quantity
    assert type(point.to(si.K)) is AffineQuantity
    assert delta.unit == si.K
    assert (point + delta).unit is point.unit
    assert AffineUnit(1 * si.one).is_dimensionless()
    assert not (point == Quantity(20, 'degC'))


def test_quantity_linear_log_target():
    assert Quantity(1, '').to_log('dBm', kind='power').value == 0


@pytest.mark.parametrize(('kind', 'reading'), [('power', 15), ('amplitude', 30)])
def test_quantity_scaled_logarithmic_difference(kind, reading):
    delta = Unit(si.dB, scale=Factor(2))
    level = Quantity(Decimal(reading), delta)
    ratio = level.to_ratio(kind=kind)
    assert ratio.value == Decimal('1000')
    assert isinstance(ratio.value, Decimal)
    assert ratio.to_log(delta, kind=kind).value == level.value
    named = delta.define('privateLogStep')
    assert Quantity(level.value, named).to_ratio(kind=kind) == ratio


@pytest.mark.parametrize(('kind', 'reading'), [('power', 5), ('amplitude', 10)])
def test_affine_unit_scaled_logarithmic_reference(kind, reading):
    from unitx import Registry
    from unitx.serialization import from_dict, to_dict

    reference = Unit(si.W, scale=pi)
    delta = Unit(si.dB, scale=Factor(2))
    coordinate = LogUnit(2 * reference, delta_unit=delta, kind=kind).define('logReading')
    point = reading @ coordinate
    assert point.to_quantity(reference).value == 20
    other_kind = 'amplitude' if kind == 'power' else 'power'
    other = LogUnit(2 * reference, delta_unit=delta, kind=other_kind)
    assert point.to(other).to_quantity(reference).value == 20
    assert (point + 2 * delta) - point == 2 * delta
    assert pickle.loads(pickle.dumps(point)) == point
    assert from_dict(to_dict(point), registry=Registry([coordinate])) == point
