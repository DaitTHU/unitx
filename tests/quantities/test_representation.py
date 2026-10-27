"""Explicit numerical exits and physical representation changes."""

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.exceptions import DimensionError
from unitx.factor import pi
from unitx.si import si


@pytest.mark.parametrize('value', [1, 1.25, Fraction(1, 3), Decimal('1.25'), 1 + 2j])
def test_quantity_n_identity(value: int | float | Fraction | Decimal | complex) -> None:
    assert Quantity(value, si.one).n is value


def test_quantity_n_scaled() -> None:
    assert (1 * si.km / si.m).n == 1000
    assert (25 * si.percent).n == Fraction(1, 4)
    assert (180 * si.deg).n == pytest.approx(3.141592653589793)
    assert Quantity(1 + 2j, si.percent).n == 0.01 + 0.02j
    assert Quantity(2, Unit('', scale=pi)).n == pytest.approx(6.283185307179586)
    with localcontext() as context:
        context.prec = 2
        assert Quantity(Decimal('12.345'), si.percent).n == Decimal('0.12')


@pytest.mark.parametrize('unit', [si.m, si.dB, si.dB**2, si.dB / si.m])
def test_quantity_n_invalid_dimension(unit: Unit) -> None:
    with pytest.raises(DimensionError):
        _ = Quantity(1, unit).n


def test_quantity_n_invalid_assignment() -> None:
    with pytest.raises(AttributeError, match='read-only'):
        Quantity(1, si.one).n = 2


def test_quantity_n_point_boundary() -> None:
    assert not hasattr(AffineQuantity, 'n')
    assert not hasattr(LogQuantity, 'n')


def test_quantity_n_numpy() -> None:
    np = pytest.importorskip('numpy')
    values = np.array([25.0, 50.0])
    quantity = Quantity(values, si.percent)
    np.testing.assert_array_equal(quantity.n, [0.25, 0.5])
    np.testing.assert_array_equal(values, [25, 50])
    assert Quantity(values, si.one).n is values
    scalar = Quantity(np.array(25.0), si.percent).n
    assert scalar.shape == ()
    np.testing.assert_allclose(scalar, 0.25)
    with pytest.raises(TypeError):
        np.asarray(quantity)


@pytest.mark.parametrize(
    ('source', 'target'),
    [
        (si.J, si.kg * si.m**2 / si.s**2),
        (si.cm, si.m),
        (si.km / si.m, si.one),
        (si.deg, si.one),
        (si.degC, si.K),
        (si.mol / si.cd, si.mol / si.cd),
        (si.N ** Fraction(-1, 3), (si.kg * si.m / si.s**2) ** Fraction(-1, 3)),
    ],
)
def test_quantity_to_base_units(source: Unit, target: Unit) -> None:
    quantity = Quantity(Decimal('1.25'), source)
    result = quantity.to_base()
    assert result.unit == target
    assert result.value == quantity.value_in(target)
    assert type(result.value) is Decimal


def test_quantity_to_base_exact_scale() -> None:
    quantity = Quantity(Fraction(1, 3), Unit('cm', scale=pi))
    result = quantity.to_base()
    assert result.unit == si.m
    assert result.value == quantity.value_in(si.m)
    assert Quantity(10**100, si.km).to_base().value == 10**103
    result = (Fraction(3, 7) * si.cm).to_base()
    assert result.value == Fraction(3, 700)
    assert (1 * si.dB / si.Np).to_base().unit == si.one


@pytest.mark.parametrize('unit', [si.dB, si.dB**2, si.dB / si.m])
def test_quantity_to_base_invalid_logarithmic(unit: Unit) -> None:
    with pytest.raises(DimensionError, match='nonzero logarithmic degree'):
        (1 * unit).to_base()


def test_quantity_to_base_numpy() -> None:
    np = pytest.importorskip('numpy')
    values = np.array([1, 2, 3])
    result = Quantity(values, si.cm).to_base()
    assert result.unit == si.m and result.value.dtype.kind == 'f'
    np.testing.assert_allclose(result.value, [0.01, 0.02, 0.03])
    np.testing.assert_array_equal(values, [1, 2, 3])
    assert Quantity(values, si.m).to_base().value is values


@pytest.mark.parametrize(
    ('source', 'target', 'value'),
    [
        (si.cm * si.m**2, si.m**3, Fraction(1, 100)),
        (si.km / si.cm, si.one, 100000),
        (si.percent, si.one, Fraction(1, 100)),
        (si.N * si.m, si.N * si.m, 1),
        (si.cm, si.cm, 1),
        (si.m ** Fraction(1, 3) / si.cm ** Fraction(1, 3), si.one, 100 ** (1 / 3)),
    ],
)
def test_quantity_simplify_units(source: Unit, target: Unit, value: int | Fraction | float) -> None:
    quantity = Quantity(1, source)
    result = quantity.simplify()
    assert result.unit == target
    if isinstance(value, float):
        assert result.value == pytest.approx(value)
    else:
        assert result.value == value
    assert result.simplify().unit == result.unit
    assert result.simplify().value is result.value
    assert quantity.unit == source and quantity.value == 1


def test_quantity_simplify_order_and_scale() -> None:
    first = Quantity(Decimal('2'), Unit(si.cm * si.m**2, scale=pi)).simplify()
    second = Quantity(Decimal('2'), Unit(si.m**2 * si.cm, scale=pi)).simplify()
    assert first.unit == second.unit == Unit(si.m**3, scale=pi)
    assert first.value == second.value == Decimal('0.02')
    assert Quantity(Fraction(1, 3), si.percent).simplify().value == Fraction(1, 300)


def test_quantity_simplify_logarithmic() -> None:
    assert (1 * si.dB).simplify().unit == si.dB
    assert (1 * si.dB * si.percent).simplify().unit == si.dB
    assert (1 * si.dB / si.Np).simplify().unit == si.one
    with pytest.raises(DimensionError):
        _ = (1 * si.dB * si.percent).simplify().n


def test_quantity_simplify_numpy() -> None:
    np = pytest.importorskip('numpy')
    values = np.array([1, 2, 3])
    quantity = Quantity(values, si.cm * si.m**2)
    result = quantity.simplify()
    assert result.unit == si.m**3
    np.testing.assert_allclose(result.value, [0.01, 0.02, 0.03])
    np.testing.assert_array_equal(values, [1, 2, 3])


@pytest.mark.parametrize(
    ('quantity', 'target'),
    [
        (Quantity(Fraction(1, 10**9), si.m), si.nm),
        (Quantity(Fraction(1, 10**9), si.m**3), si.mm**3),
        (Quantity(Fraction(1, 100), si.m**2), si.dm**2),
        (Quantity(1000, si.W), si.kW),
        (Quantity(-1000, si.W), si.kW),
        (Quantity(1000, si.g), si.kg),
        (Quantity(1, si.kg), si.kg),
        (Quantity(10**9, si.m**-3), si.mm**-3),
        (Quantity(Fraction(1, 10), si.m ** Fraction(1, 3)), si.mm ** Fraction(1, 3)),
        (Quantity(10**100, si.m), Unit('Qm')),
        (Quantity(Fraction(1, 10**100), si.m), Unit('qm')),
    ],
)
def test_quantity_compact_units(quantity: Quantity[int | Fraction], target: Unit) -> None:
    result = quantity.compact()
    assert result.unit == target
    assert result.value == quantity.value_in(target)
    assert result.compact().unit == target


def test_quantity_compact_decimal_and_scale() -> None:
    with localcontext() as context:
        context.prec = 2
        quantity = Quantity(Decimal('0.0000012345'), si.m)
        result = quantity.compact()
        assert result.unit == si.um and result.value == Decimal('1.2')
    scaled = Quantity(1, Unit('m', scale=pi))
    result = scaled.compact()
    assert result.unit == si.m
    assert result.value == scaled.value_in(si.m)


@pytest.mark.parametrize('value', [0, float('nan'), float('inf'), Decimal('NaN')])
def test_quantity_compact_no_scale(value: int | float | Decimal) -> None:
    quantity = Quantity(value, si.mm)
    result = quantity.compact()
    assert result.unit == quantity.unit and result.value is value


def test_quantity_compact_boundaries() -> None:
    assert Quantity(999.9999999999999, si.m).compact().unit == si.m
    assert Quantity(1000.0, si.m).compact().unit == si.km
    assert Quantity(Fraction(999999999999999999, 10**15), si.m).compact().unit == si.m
    assert (1000 * si.deg).compact().unit == si.deg
    assert (1000 * si.m / si.s).compact().unit == si.m / si.s
    custom = si.m.define('compactNameOnly', name='compact name', prefixable='compact name')
    result = (1000 * custom).compact()
    assert result.value == 1
    assert result.unit.expression() == '[kilocompact name]'


def test_quantity_compact_invalid_complex() -> None:
    with pytest.raises(TypeError, match='real Python scalar'):
        Quantity(1j, si.m).compact()


def test_quantity_compact_numpy() -> None:
    np = pytest.importorskip('numpy')
    values = np.array([np.nan, -1e-6, 2e-6, np.inf])
    quantity = Quantity(values, si.m)
    result = quantity.compact()
    assert result.unit == si.um
    np.testing.assert_allclose(result.value, [np.nan, -1, 2, np.inf])
    np.testing.assert_allclose(values, [np.nan, -1e-6, 2e-6, np.inf])
    for values in (np.array([]), np.array([np.nan, np.inf]), np.array([0, 0])):
        result = Quantity(values, si.mm).compact()
        assert result.unit == si.mm and result.value is values
    integer = np.array([np.iinfo(np.int64).min], dtype=np.int64)
    assert Quantity(integer, si.m).compact().unit == Unit('Em')
    assert Quantity(np.array(1000), si.m).compact().unit == si.km
    assert Quantity(np.array([1000], dtype=np.longdouble), si.m).compact().unit == si.km
    with pytest.raises(TypeError, match='real magnitudes'):
        Quantity(np.array([1j]), si.m).compact()


def test_quantity_as_pair() -> None:
    quantity = Quantity(Decimal('1.25'), Unit('cm', scale=pi))
    value, unit = quantity.as_pair()
    assert value is quantity.value and unit is quantity.unit
    assert Quantity(value, unit).unit == quantity.unit
    assert Quantity(value, unit).value is quantity.value
    value, unit = quantity.to(si.m).as_pair()
    assert unit == si.m and value == quantity.value_in(si.m)


def test_quantity_as_pair_points() -> None:
    celsius = 20 @ si.degC
    level = 30 @ si.dBm
    value, affine = celsius.as_pair()
    assert value is celsius.value and affine is celsius.unit
    assert AffineQuantity(value, affine) == celsius
    value, logarithmic = level.as_pair()
    assert value is level.value and logarithmic is level.unit
    assert LogQuantity(value, logarithmic) == level
    with pytest.raises(TypeError):
        iter(celsius)


def test_quantity_as_pair_numpy() -> None:
    np = pytest.importorskip('numpy')
    values = np.arange(3.0)
    quantity = Quantity(values, si.m)
    value, unit = quantity.as_pair()
    assert value is values and unit is quantity.unit
    value[0] = 5
    assert quantity.value[0] == 5
    assert len(tuple(quantity)) == 3
    for point in (AffineQuantity(values, si.degC), LogQuantity(values, si.dBm)):
        value, coordinate = point.as_pair()
        assert value is values and coordinate is point.unit
