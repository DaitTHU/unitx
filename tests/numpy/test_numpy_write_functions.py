"""Batch writes retain storage and stage every failure."""

from collections.abc import Callable
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.exceptions import DimensionError

np = pytest.importorskip('numpy')

OPERATIONS = [
    lambda a, v: np.copyto(a, v),
    lambda a, v: np.put(a, [0, 3], v),
    lambda a, v: np.put_along_axis(a, np.array([[0], [1]]), v, axis=1),
    lambda a, v: np.putmask(a, np.array([[True, False], [False, True]]), v),
    lambda a, v: np.place(a, np.array([[True, False], [False, True]]), v),
    lambda a, v: np.fill_diagonal(a, v),
]


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize(
    'family,unit,source_unit,source_value',
    [
        (Quantity, 'm', 'cm', 250.0),
        (AffineQuantity, 'degC', 'K', 275.65),
        (LogQuantity, 'dBm', 'dBW', -27.5),
    ],
)
def test_write_functions_conversion(
    operation: Callable[..., Any], family: Any, unit: str, source_unit: str, source_value: float
) -> None:
    backing = np.zeros((2, 4))
    a = family(backing[:, ::2], unit)
    raw = a.value.copy()
    assert operation(raw, 2.5) is None
    assert operation(a, family(source_value, source_unit)) is None
    np.testing.assert_allclose(a.value, raw, atol=1e-12)
    np.testing.assert_array_equal(backing[:, 1::2], 0)
    assert a.unit == family(0.0, unit).unit


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize(
    'value',
    [
        Quantity(2.0, 's'),
        0.0,
        np.ones(2),
        AffineQuantity(0.0, 'K'),
        Quantity(2j, 'm'),
        Quantity(1.0, 'cm'),
    ],
)
def test_write_functions_invalid(operation: Callable[..., Any], value: Any) -> None:
    a = Quantity(np.full((2, 2), 7, dtype=int), 'm')
    with pytest.raises((DimensionError, TypeError, ValueError)):
        operation(a, value)
    np.testing.assert_array_equal(a.value, 7)


@pytest.mark.parametrize('operation', OPERATIONS)
def test_write_functions_readonly(operation: Callable[..., Any]) -> None:
    a = Quantity(np.ones((2, 2)), 'm')
    a.value.flags.writeable = False
    with pytest.raises(ValueError, match='read-only'):
        operation(a, Quantity(2.0, 'm'))
    np.testing.assert_array_equal(a.value, 1)


def test_copyto_mask_alias_casting() -> None:
    raw = np.array([1, 1, 0, 1], dtype=np.int8)
    q = Quantity(raw, '')
    mask = raw.view(bool)
    np.copyto(q, Quantity(np.array([2, 3, 4, 5]), ''), where=mask)
    np.testing.assert_array_equal(raw, [2, 3, 0, 5])
    q = Quantity(np.array([1.0, 2.0, 3.0, 4.0]), 'm')
    np.copyto(q[1:], q[:-1])
    np.testing.assert_array_equal(q.value, [1, 1, 2, 3])
    integer = Quantity(np.zeros(2, dtype=int), 'm')
    np.copyto(integer, Quantity(150.0, 'cm'), casting='unsafe')
    np.testing.assert_array_equal(integer.value, 1)


def test_put_repeated_indices_and_modes() -> None:
    q = Quantity(np.zeros(4), 'm')
    np.put(q, [0, 0, 8], Quantity(np.array([100.0, 200.0, 300.0]), 'cm'), mode='wrap')
    np.testing.assert_array_equal(q.value, [3, 0, 0, 0])
    original = q.value.copy()
    with pytest.raises(IndexError):
        np.put(q, [0, 9], Quantity(8.0, 'm'))
    np.testing.assert_array_equal(q.value, original)


def test_putmask_place_native_repetition() -> None:
    for func in (np.putmask, np.place):
        q = Quantity(np.arange(6.0), 'm')
        mask = np.array([False, True, True, False, True, True])
        values = np.array([100.0, 200.0])
        expected = q.value.copy()
        func(expected, mask, values / 100)
        func(q, mask, Quantity(values, 'cm'))
        np.testing.assert_array_equal(q.value, expected)


@pytest.mark.parametrize('operation', OPERATIONS)
def test_write_functions_cast_overflow(operation: Callable[..., Any]) -> None:
    q = Quantity(np.ones((2, 2), dtype=np.float32), 'm')
    with np.errstate(over='raise'), pytest.raises(FloatingPointError):
        operation(q, Quantity(1e300, 'm'))
    np.testing.assert_array_equal(q.value, 1)


def test_put_indices_alias() -> None:
    raw = np.array([2, 0, 1])
    q = Quantity(raw, Unit())
    np.put(q, raw, Quantity(np.array([10, 20, 30]), Unit()))
    np.testing.assert_array_equal(raw, [20, 30, 10])
