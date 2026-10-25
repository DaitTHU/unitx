"""Explicit magnitude conversion preserves exact unit identity."""

from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest

from unitx import Quantity, Unit
from unitx.factor import Factor, pi


@pytest.mark.parametrize(
    'value,dtype',
    [
        (value, dtype)
        for value in [2, 1.25, Fraction(5, 4), Decimal('1.25')]
        for dtype in [int, float, complex, Fraction, Decimal]
        if not (isinstance(value, Fraction) and dtype is Decimal)
    ],
)
def test_quantity_astype_scalar(value: Any, dtype: Any) -> None:
    unit = Factor(2) * pi * Unit('cm')
    q = Quantity(value, unit)
    expected = dtype(value)
    result = q.astype(dtype)
    assert type(result) is Quantity and result.unit is q.unit
    assert type(result.value) is dtype and result.value == expected
    assert q.value is value


def test_quantity_astype_float_boundary() -> None:
    q: Quantity[float] = Quantity(1, 'cm')
    converted = q.to('m')
    assert type(converted.value) is Fraction
    result = converted.astype(float)
    assert type(result.value) is float and result.value == 0.01
    assert converted.value == Fraction(1, 100)


@pytest.mark.parametrize('dtype', [bool, str, object, 'float64'])
def test_quantity_astype_invalid_dtype(dtype: Any) -> None:
    with pytest.raises(TypeError, match='Scalar astype requires'):
        Quantity(1, 'm').astype(dtype)


def test_quantity_astype_invalid_value() -> None:
    with pytest.raises(TypeError):
        Quantity(1j, 'm').astype(float)


def test_quantity_astype_invalid_fraction_decimal() -> None:
    with pytest.raises(TypeError):
        Quantity(Fraction(1, 2), 'm').astype(Decimal)


def test_quantity_astype_native_overflow() -> None:
    q = Quantity(10**400, 'm')
    with pytest.raises(OverflowError):
        q.astype(float)
    assert q.value == 10**400
