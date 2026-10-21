"""Coordinate semantics, exact linear conversions, and logarithmic differences."""

import copy
import math
import operator
import pickle
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx.affinequantity import AffineQuantity
from unitx.affineunit import AffineUnit
from unitx.exceptions import DimensionError, UnitRegistryError
from unitx.factor import Factor, SymbolicConstant, ln10, pi
from unitx.quantity import Quantity
from unitx.si import si
from unitx.unit import Unit
from unitx.unitregistry import UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL


@pytest.mark.parametrize(
    'x, source, target, expected',
    [
        (0, 'degC', 'K', Fraction(27315, 100)),
        (100, 'degC', 'degF', 212),
        (32, 'degF', 'degC', 0),
        (0, 'degF', 'K', Fraction(45967, 180)),
        (10, 'degR', 'K', Fraction(50, 9)),
        (30, 'dBm', 'dBW', 0),
        (0, 'dBW', 'dBm', 30),
        (0, 'dBm', 'mW', 1),
        (20, 'dBm', 'mW', 100),
        (20, 'dBV', 'V', 10),
    ],
)
def test_reference_conversion(x, source, target, expected):
    q = AffineQuantity(x, source)
    value = q.value_in(target)
    assert value == expected
    assert value == q.to(target).value
    assert q.value == x


@pytest.mark.parametrize('precision', [2, 8, 28])
def test_decimal_round_once(precision):
    with localcontext() as ctx:
        ctx.prec = precision
        value = Decimal('1.23456789')
        result = AffineQuantity(value, 'degF').to('degC').value
        exact = (Fraction(value) - 32) * Fraction(5, 9)
        assert result == Decimal(exact.numerator) / Decimal(exact.denominator)
        assert isinstance(result, Decimal)
        assert (AffineQuantity(Decimal('30'), 'dBm') - AffineQuantity(0, 'dBW')).value == 0
        assert AffineQuantity(Decimal(20), 'dBm').to('mW').value == Decimal(100)


def test_point_and_difference_arithmetic():
    assert AffineQuantity(30, 'degC') - AffineQuantity(68, 'degF') == Quantity(10, 'K')
    assert AffineQuantity(20, 'degC') + Quantity(18, 'degR') == AffineQuantity(30, 'degC')
    assert Quantity(18, 'degR') + AffineQuantity(20, 'degC') == AffineQuantity(30, 'degC')
    assert AffineQuantity(30, 'dBm') - AffineQuantity(-10, 'dBW') == Quantity(10, 'dB')
    assert AffineQuantity(20, 'dBm') + Quantity(10, 'dB') == AffineQuantity(30, 'dBm')
    assert AffineQuantity(20, 'dBV') - Quantity(10, 'dB') == AffineQuantity(10, 'dBV')


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
        op(AffineQuantity(20, 'dBm'))


@pytest.mark.parametrize('text', ['degC/m', 'dBm/Hz', 'degC/degC', 'degC**0', 'dBm**2', '/degF'])
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
    assert Unit('dB/m').log_degree == 1
    assert Unit('dB**2').log_degree == 2
    assert Unit('dB/Np').log_degree == 0
    assert (Quantity(2, 'dB/m') * Quantity(10, 'm')).to_ratio(kind='power').value == 100
    assert hash(Quantity(0, 'dB')) == hash(Quantity(0, 'Np'))
    assert not (Quantity(0, 'dB') == 0)
    with pytest.raises(DimensionError):
        Quantity(20, 'dB').to('')
    with pytest.raises(DimensionError):
        Quantity(1, 'dB/m').to_ratio(kind='power')
    with pytest.raises(DimensionError):
        Quantity(1, 'dB') + 1


def test_point_comparison_and_hash():
    assert AffineQuantity(0, 'degC') == AffineQuantity(Fraction(27315, 100), 'K')
    assert AffineQuantity(30, 'dBm') == AffineQuantity(0, 'dBW')
    assert not (AffineQuantity(0, 'dBm') == Quantity(1, 'mW'))
    with pytest.raises(DimensionError):
        AffineQuantity(0, 'dBm') < Quantity(1, 'mW')
    with pytest.raises(DimensionError):
        AffineQuantity(0, 'dBm') - AffineQuantity(0, 'dBV')
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


@pytest.mark.parametrize('name', ['degC', 'degF', 'dBm', 'dBV'])
def test_identity_copy_pickle_and_display(name):
    unit = AffineQuantity(0, name).unit
    assert isinstance(unit, AffineUnit)
    assert unit == pickle.loads(pickle.dumps(unit))
    assert copy.copy(unit) is unit
    assert copy.deepcopy(unit) is unit
    assert eval(repr(unit)) == unit
    q = AffineQuantity(1, unit)
    assert eval(repr(q)).value == q.value
    assert unit.symbol in q.to_html()
    assert unit.symbol in q.to_latex()
    assert 'parse-units=false' in q.to_siunitx()
    assert (-20) @ unit == AffineQuantity(-20, name)
    with pytest.raises(AttributeError):
        unit._kind = None


def test_custom_definition_and_atomic_conflict():
    tables = UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL
    snapshots = [dict(x) for x in tables]
    try:
        coordinate = AffineUnit(Quantity(10, 'K'), delta_unit=Unit('K'))
        assert pickle.loads(pickle.dumps(coordinate)) == coordinate
        registered = coordinate.define('refTemp', 'reference temperature')
        assert AffineQuantity(2, registered).value_in('K') == 12
        before = [dict(x) for x in tables]
        with pytest.raises(UnitRegistryError):
            coordinate.define('refTemp', 'another reference temperature')
        assert [dict(x) for x in tables] == before
        log = (Factor(2) * Unit('dB')).define('doubleDecibel', 'double decibel')
        assert log.log_degree == 1
        assert Quantity(10, log).to_ratio(kind='power').value == 100
    finally:
        for table, snapshot in zip(tables, snapshots):
            table.clear()
            table.update(snapshot)


def test_domains_and_reference_validation():
    with pytest.raises(ValueError):
        AffineUnit(Quantity(0, 'W'), delta_unit=Unit('dB'), kind='power')
    with pytest.raises(DimensionError):
        AffineUnit(Quantity(1, 'W'), delta_unit=Unit('K'))
    with pytest.raises(ValueError):
        AffineQuantity(-1, 'W').to('dBm')
    assert AffineQuantity(0, 'W').to('dBm').value == -math.inf
    assert AffineQuantity(-math.inf, 'dBm').to('W').value == 0
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
        AffineQuantity(0, 'dBm').isclose(Quantity(1, 'mW'), rel_tol=0, abs_tol=0)
    with pytest.raises(TypeError):
        AffineQuantity(True, 'degC')


def test_cross_log_scale_comparison_retains_symbolic_identity():
    np_watt = AffineUnit(Quantity(1, 'W'), delta_unit=Unit('Np'), kind='power')
    assert AffineQuantity(0, np_watt) == AffineQuantity(0, 'dBW')
    converted = AffineQuantity(1, 'dBW').to(np_watt)
    assert converted != AffineQuantity(1, 'dBW')
    assert not converted.isclose(AffineQuantity(1, 'dBW'), rel_tol=0, abs_tol=0)
    with pytest.raises(TypeError, match='approximate'):
        AffineQuantity(0, np_watt) == AffineQuantity(30, 'dBm')


def test_affine_unit_independent_type():
    from unitx.unit import ScaledUnit

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
    assert Quantity(1, 'degF').to('K') == Quantity(Fraction(5, 9), 'K')
    assert isinstance(ScaledUnit('degC', Factor(2)), Unit)
    assert AffineUnit(AffineQuantity(1, 'degC')).origin == Quantity(Fraction(5483, 20), si.K)


def test_unified_definition_reference_marks_coordinates():
    from unitx.singleunit import SingleUnit
    from unitx.unitregistry import NO_PREFIX, UnitDefinition

    definition = UnitDefinition(
        'referenceStep',
        'reference step',
        factor=Factor(1),
        dimension=si.K.dimension,
        reference=(Fraction(10), 'K'),
        delta='K',
        prefixable=False,
    )
    assert definition.reference == (10, 'K')
    assert SingleUnit._from_definitions(NO_PREFIX, definition).factor == Factor(1)
    assert UNIT_DEFINITIONS[si.dBm.symbol].reference == (1, 'mW')
    assert UNIT_DEFINITIONS['K'].reference is None
    with pytest.raises(ValueError, match='require a physical reference'):
        UnitDefinition(
            'invalidStep', 'invalid step', factor=Factor(1), dimension=si.K.dimension, delta='K'
        )
    with pytest.raises(ValueError, match='positive'):
        UnitDefinition(
            'invalidStep',
            'invalid step',
            factor=Factor(1),
            dimension=si.W.dimension,
            reference=(0, 'W'),
            delta='dB',
            kind='power',
            prefixable=False,
        )


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
    unit = AffineUnit(1 * si.W, at=30, delta_unit=si.dB, kind='power')
    assert unit.origin == 1 * si.mW
    assert AffineQuantity(30, unit).value_in(si.W) == 1
    assert AffineQuantity(0, unit).value_in(si.mW) == 1
    nested = AffineUnit(AffineQuantity(30, si.dBm), at=30, delta_unit=si.dB, kind='power')
    assert nested.origin == 1 * si.mW
    assert not isinstance(nested.origin.unit, AffineUnit)
    amplitude = AffineUnit(10 * si.V, at=20, delta_unit=si.dB, kind='amplitude')
    assert amplitude.origin == 1 * si.V
    assert AffineQuantity(20, amplitude).value_in(si.V) == 10
    assert pickle.loads(pickle.dumps(unit)) == unit
    assert eval(repr(unit)) == unit


def test_affine_unit_reference_registration():
    tables = UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL
    snapshots = [dict(x) for x in tables]
    try:
        unit = AffineUnit(AffineQuantity(0, si.degC), at=32, delta_unit=si.degR).define(
            'customFahrenheit', 'custom Fahrenheit', symbol_aliases='customF'
        )
        assert unit.origin == Quantity(Fraction(45967, 180), si.K)
        assert (
            unit == AffineQuantity(0, 'customF').unit == AffineQuantity(0, 'custom Fahrenheit').unit
        )
        assert AffineQuantity(212, unit).value_in(si.degC) == 100
        assert hash(unit) == hash(pickle.loads(pickle.dumps(unit)))
        assert eval(repr(unit)) == unit
        definition = UNIT_DEFINITIONS['customFahrenheit']
        assert definition.reference == (Fraction(45967, 180), 'K')
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
    with pytest.raises(TypeError, match='explicit delta_unit'):
        AffineUnit(1 * si.W, kind='power')


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
    [AffineQuantity(0, text).unit for text in ('degC', 'degF', 'dBm')] + [AffineUnit(100 * si.K)],
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
