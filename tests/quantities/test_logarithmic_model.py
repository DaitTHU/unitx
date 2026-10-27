"""Complete algebra and explicit physical boundaries of logarithmic readings."""

from __future__ import annotations

import copy
import operator
import pickle
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx import (
    DIMENSIONLESS,
    AffineQuantity,
    AffineUnit,
    Dimension,
    LogQuantity,
    LogUnit,
    Quantity,
    Registry,
    Unit,
)
from unitx._numerics.conversion import compatible, convert
from unitx.exceptions import DimensionError, SerializationError
from unitx.logunit import LogKind
from unitx.serialization import from_dict, to_dict
from unitx.si import si


def test_dimension_complete_and_physical_views() -> None:
    log = si.dB.dimension
    assert log != DIMENSIONLESS
    assert log.physical is DIMENSIONLESS
    assert log.as_tuple() == DIMENSIONLESS.as_tuple()
    assert log.as_dict() == {}
    assert str(log) == '1'
    assert si.dB.is_dimensionless()
    assert Quantity(1, si.dB).is_dimensionless()
    assert not compatible(si.dB, si.one)
    assert len({log, DIMENSIONLESS}) == 2
    attenuation = Unit('dB/m').dimension
    assert attenuation.physical == Dimension(L=-1)
    assert attenuation.L == -1
    assert attenuation != Dimension(L=-1)
    assert Unit('dB/Np').dimension is DIMENSIONLESS
    assert log / log is DIMENSIONLESS
    assert log**0 is DIMENSIONLESS
    assert (log ** Fraction(-1, 3)) ** -3 == log
    assert ((log * Dimension(L=1)) ** 0.1) ** 10 == log * Dimension(L=1)
    assert copy.copy(log) is log
    assert copy.deepcopy(log) is log
    assert eval(repr(log)) == log
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(attenuation, protocol=protocol)) == attenuation


@pytest.mark.parametrize('unit', [si.dB, si.dB**2, si.dB / si.m])
def test_quantity_complete_compatibility(unit: Unit) -> None:
    plain = Unit('' if unit.is_dimensionless() else '/m')
    quantity = 1 * unit
    assert quantity.dimension.physical == plain.dimension
    assert quantity != 1 * plain
    with pytest.raises(DimensionError):
        quantity.to(plain)
    with pytest.raises(DimensionError):
        quantity + 1 * plain
    with pytest.raises(DimensionError):
        quantity.isclose(1 * plain)
    assert 0 * unit != 0 * plain
    assert {0 * unit: 'log', 0 * plain: 'physical'}[0 * unit] == 'log'


@pytest.mark.parametrize(
    ('source', 'target'),
    [
        (si.dB, si.one),
        (si.one, si.dB),
        (si.dB**2, si.dB),
        (si.dB / si.m, si.m**-1),
        (si.Np, si.one),
    ],
)
def test_quantity_invalid_logarithmic_conversion_diagnostic(source: Unit, target: Unit) -> None:
    with pytest.raises(DimensionError, match='logarithmic degrees differ') as caught:
        (1 * source).value_in(target)
    message = str(caught.value)
    assert f'unit {str(source)!r}' in message
    assert f'unit {str(target)!r}' in message
    assert 'physical dimensions match' in message


def test_log_unit_concrete_family() -> None:
    assert type(si.degC) is AffineUnit
    assert type(si.dBm) is LogUnit
    assert not isinstance(si.dBm, AffineUnit)
    assert type(20 @ si.degC) is AffineQuantity
    assert type(30 @ si.dBm) is LogQuantity
    assert not isinstance(30 @ si.dBm, AffineQuantity)
    assert not hasattr(si.degC, 'kind')
    assert not hasattr(si.dB, 'log_degree')
    assert not hasattr(si.dB.dimension, 'log_degree')
    assert Quantity(2, si.dBm).unit == Unit('dBm')
    assert (2 * si.dBm).unit == Unit('dBm')
    assert type((30 @ si.dBm) + 2 * si.dB) is LogQuantity
    assert type((30 @ si.dBm) - (0 @ si.dBW)) is Quantity
    assert type(+(30 @ si.dBm)) is LogQuantity


def test_log_quantity_explicit_physical_boundary() -> None:
    level = 30 @ si.dBm
    assert level.to_quantity(si.W) == 1 * si.W
    assert level.to(si.dBW) == 0 @ si.dBW
    assert LogQuantity.from_quantity(1 * si.W, si.dBm) == level
    assert LogQuantity.from_quantity(0 * si.W, si.dBm).value == -float('inf')
    assert (-float('inf') @ si.dBm).to_quantity(si.W).value == 0
    with pytest.raises(ValueError):
        LogQuantity.from_quantity(-1 * si.W, si.dBm)
    with pytest.raises(DimensionError):
        LogQuantity.from_quantity(1 * si.V, si.dBm)
    with pytest.raises(TypeError):
        AffineQuantity(30, si.dBm)
    with pytest.raises(TypeError):
        LogQuantity(1, si.W)
    with pytest.raises(TypeError):
        (1 @ si.W).to(si.dBm)
    with pytest.raises(TypeError):
        level.to(si.W)
    with pytest.raises(TypeError):
        AffineUnit(1 * si.W, kind='power')
    for source, target in ((si.dBm, si.W), (si.W, si.dBm), (si.degC, si.K)):
        with pytest.raises(TypeError, match='Linear conversion requires Unit'):
            convert(1, source, target)


@pytest.mark.parametrize('precision', [2, 28, 50])
def test_log_unit_defining_anchor_exactness(precision: int) -> None:
    coordinate = LogUnit(1 * si.W, at=1, delta_unit=si.dB, kind='power')
    with localcontext() as context:
        context.prec = precision
        assert coordinate.reference == 1 * si.W
        assert coordinate.at == 1
        assert (Decimal(1) @ coordinate).to_quantity(si.W).value == Decimal(1)
        assert LogQuantity.from_quantity(Decimal(1) * si.W, coordinate).value == Decimal(1)
        assert (Decimal(1) @ coordinate).to(si.dBm).value == Decimal(30)
        assert Decimal(1) @ coordinate == 30 @ si.dBm
        for restored in (
            pickle.loads(pickle.dumps(coordinate)),
            eval(repr(coordinate)),
            from_dict(to_dict(coordinate)),
        ):
            assert restored == coordinate
            assert hash(restored) == hash(coordinate)
            assert restored.reference == 1 * si.W
            assert restored.at == 1
            assert (Decimal(1) @ restored).to_quantity(si.W).value == Decimal(1)


def test_log_unit_definition_does_not_evaluate_origin() -> None:
    coordinate = LogUnit(1 * si.W, at=10**6, delta_unit=si.dB, kind='power')
    assert (10**6 @ coordinate).to_quantity(si.W) == 1 * si.W
    assert from_dict(to_dict(coordinate)) == coordinate


def test_log_unit_anchor_identity_retains_exact_definition() -> None:
    coordinate = LogUnit(1 * si.W, at=1, delta_unit=si.dB, kind='power')
    approximated = LogUnit(coordinate.origin, delta_unit=si.dB, kind='power')
    assert coordinate != approximated
    assert len({coordinate, approximated}) == 2


@pytest.mark.parametrize('precision', [2, 8, 28])
def test_log_quantity_decimal_domains(precision: int) -> None:
    with localcontext() as context:
        context.prec = precision
        level = Decimal('30') @ si.dBm
        assert level.to_quantity(si.W).value == Decimal(1)
        assert isinstance(level.to_quantity(si.W).value, Decimal)
        assert LogQuantity.from_quantity(Decimal(1) * si.W, si.dBm).value == Decimal(30)
        assert (level - Decimal(0) @ si.dBW).value == Decimal(0)
        exact = (2**60 + 1) @ si.dBm
        assert exact != float(2**60 + 1) @ si.dBm


@pytest.mark.parametrize('kind', ['power', 'amplitude'])
def test_log_unit_registry_reconstruction(kind: LogKind) -> None:
    coordinate = LogUnit(1 * si.W, delta_unit=si.dB, kind=kind).define('privateLevel')
    registry = Registry([coordinate])
    point = LogQuantity(Fraction(3, 2), registry.parse('privateLevel'))
    assert type(point.unit) is LogUnit
    for value in (point, coordinate):
        assert pickle.loads(pickle.dumps(value)) == value
        assert from_dict(to_dict(value), registry=registry) == value
    record = to_dict(point)
    record['type'] = 'affine_quantity'
    with pytest.raises(SerializationError):
        from_dict(record, registry=registry)


def test_numpy_complete_dimension_writes() -> None:
    np = pytest.importorskip('numpy')
    target = Quantity(np.array([1.0, 2.0]), si.dB)
    saved = target.value.copy()
    for operation in (
        lambda: target.__setitem__(slice(None), Quantity(np.array([3.0, 4.0]), si.one)),
        lambda: np.add(target, target, out=Quantity(np.empty(2), si.one)),
    ):
        with pytest.raises(DimensionError):
            operation()
        np.testing.assert_array_equal(target.value, saved)
    for function in (np.exp, np.sin):
        with pytest.raises(DimensionError):
            function(target)
    assert np.var(target).dimension == si.dB.dimension**2


def test_numpy_log_quantity_storage_and_family() -> None:
    np = pytest.importorskip('numpy')
    raw = np.array([0.0, 20.0, 30.0])
    point = raw @ si.dBm
    assert type(point) is LogQuantity
    assert point.value is raw
    assert type(point[:2]) is LogQuantity
    assert type(point.T) is LogQuantity
    assert type(point.flatten()) is LogQuantity
    assert type(np.mean(point)) is LogQuantity
    assert type(np.std(point)) is Quantity
    restored = LogQuantity.from_quantity(point.to_quantity(si.W), si.dBm)
    np.testing.assert_allclose(restored.value, raw)
    point[:] = LogQuantity(np.array([-30.0, -10.0, 0.0]), si.dBW)
    np.testing.assert_allclose(raw, [0, 20, 30])
    saved = raw.copy()
    affine = AffineQuantity(np.array([1.0, 2.0, 3.0]), si.W)
    for operation in (
        lambda: point.__setitem__(slice(None), affine),
        lambda: np.add(point, 1 * si.dB, out=affine),
        lambda: operator.iadd(point, 1 * si.W),
    ):
        with pytest.raises(DimensionError):
            operation()
        np.testing.assert_array_equal(raw, saved)
    with pytest.raises(TypeError):
        np.asarray(point)
