"""Scalar statistics preserve units, coordinates and native numerical domains."""

import statistics as native
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, statistics
from unitx.exceptions import DimensionError
from unitx.factor import Factor, pi
from unitx.si import si


@pytest.mark.parametrize('name', ['mean', 'fmean', 'variance', 'pvariance', 'stdev', 'pstdev'])
@pytest.mark.parametrize('kind', [int, float, Fraction, Decimal])
def test_statistics_numeric_domain(name: str, kind: Any) -> None:
    values = [kind(1), kind(2), kind(4)]
    unit = Factor(2) * pi * si.m
    result = getattr(statistics, name)(Quantity(value, unit) for value in values)
    expected = getattr(native, name)(values)
    assert type(result.value) is type(expected) and result.value == expected
    assert result.unit == (unit**2 if name in ('variance', 'pvariance') else unit)


@pytest.mark.parametrize('constructor,unit', [(AffineQuantity, 'degC'), (LogQuantity, 'dBm')])
@pytest.mark.parametrize('name', ['mean', 'fmean', 'variance', 'pvariance', 'stdev', 'pstdev'])
def test_statistics_points(constructor: Any, unit: str, name: str) -> None:
    values = [20, 25, 30]
    samples = [constructor(value, unit) for value in values]
    result = getattr(statistics, name)(samples)
    expected = getattr(native, name)(values)
    assert result.value == expected
    if name in ('mean', 'fmean'):
        assert type(result) is constructor and result.unit is samples[0].unit
    else:
        delta = samples[0].unit.delta_unit
        assert type(result) is Quantity
        assert result.unit == (delta**2 if name in ('variance', 'pvariance') else delta)


def test_statistics_alignment() -> None:
    assert statistics.mean([1 * si.m, 200 * si.cm, 3 * si.m]) == 2 * si.m
    assert statistics.mean([1, Quantity(200, 'cm/m')]) == 1.5 * si.one
    assert statistics.mean([20 @ si.degC, 303.15 @ si.K]).value == pytest.approx(25)
    assert statistics.variance([1 * si.m, 200 * si.cm, 3 * si.m], 200 * si.cm) == si.m**2 * 1
    assert statistics.pvariance([20 @ si.degC, 30 @ si.degC], 298.15 @ si.K) == 25 * si.K**2


def test_statistics_decimal_context() -> None:
    with localcontext() as context:
        context.prec = 6
        values = [Decimal('1.23456'), Decimal('2.34567'), Decimal('3.45678')]
        samples = [Quantity(value, 'm') for value in values]
        assert statistics.mean(samples).value == native.mean(values)
        assert statistics.variance(samples).value == native.variance(values)
        assert statistics.stdev(samples).value == native.stdev(values)


def test_statistics_fmean_weights() -> None:
    weights = [Quantity(100, 'cm/m'), 2 * si.one, 1 * si.one]
    assert statistics.fmean([1 * si.m, 200 * si.cm, 3 * si.m], weights) == 2 * si.m
    assert statistics.fmean([20 @ si.degC, 30 @ si.degC], [1, 3]) == 27.5 @ si.degC
    with pytest.raises(ValueError):
        statistics.fmean([1 * si.m], [1, 2])


def test_statistics_covariance() -> None:
    x = [1 * si.m, 200 * si.cm, 3 * si.m]
    y = [1 * si.s, 2000 * si.ms, 4 * si.s]
    assert statistics.covariance(iter(x), iter(y)) == 1.5 * si.m * si.s
    assert statistics.correlation(x, y).value == pytest.approx(
        native.correlation([1, 2, 3], [1, 2, 4])
    )
    assert (
        statistics.covariance([20 @ si.degC, 30 @ si.degC], [1 * si.s, 2 * si.s]) == 5 * si.K * si.s
    )
    assert (
        statistics.covariance([Decimal('1'), Decimal('2')], [Fraction(1), Fraction(3)]).value == 1.0
    )


def test_statistics_ranked_exact() -> None:
    base = 10**100
    x = [Quantity(base + i, 'm') for i in (0, 1, 1, 2)]
    y = [20 @ si.degC, 30 @ si.degC, 30 @ si.degC, 40 @ si.degC]
    assert statistics.correlation(x, y, method='ranked').value == 1.0
    with pytest.raises(ValueError):
        statistics.correlation(x, y, method='invalid')


@pytest.mark.parametrize('name', ['mean', 'fmean', 'variance', 'pvariance', 'stdev', 'pstdev'])
def test_statistics_invalid_empty(name: str) -> None:
    with pytest.raises(statistics.StatisticsError):
        getattr(statistics, name)([])


@pytest.mark.parametrize('data', [[1 * si.m, 1 * si.s], [1 * si.m, 1], [20 @ si.degC, 1 @ si.dBm]])
def test_statistics_invalid_dimension(data: list[Any]) -> None:
    with pytest.raises(DimensionError):
        statistics.mean(data)


@pytest.mark.parametrize('data', [[20 @ si.degC, 1 * si.K], [True], [1j]])
def test_statistics_invalid_type(data: list[Any]) -> None:
    with pytest.raises(TypeError):
        statistics.mean(data)


def test_statistics_invalid_center() -> None:
    with pytest.raises(TypeError):
        statistics.variance([20 @ si.degC, 30 @ si.degC], 25 * si.K)
    with pytest.raises(DimensionError):
        statistics.variance([1 * si.m, 2 * si.m], 1 * si.s)


def test_statistics_invalid_weights() -> None:
    with pytest.raises(DimensionError):
        statistics.fmean([1 * si.m], [1 * si.s])
    with pytest.raises(TypeError):
        statistics.fmean([1 * si.m], [20 @ si.degC])


def test_statistics_invalid_pairs() -> None:
    for name in ('covariance', 'correlation'):
        with pytest.raises(statistics.StatisticsError):
            getattr(statistics, name)([1 * si.m], [1 * si.s, 2 * si.s])
    with pytest.raises(statistics.StatisticsError):
        statistics.correlation([1, 1], [1, 2])
