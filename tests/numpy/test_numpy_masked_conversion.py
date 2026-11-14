"""Elementwise masks cover unit conversion as well as the numerical kernel."""

from typing import Any

import pytest

from unitx import AffineQuantity, Quantity
from unitx.exceptions import DimensionError
from unitx.si import si

np = pytest.importorskip('numpy')


@pytest.mark.parametrize(
    'operation',
    [
        np.add,
        np.subtract,
        np.minimum,
        np.maximum,
        np.fmin,
        np.fmax,
        np.remainder,
        np.fmod,
        np.floor_divide,
        np.hypot,
        np.arctan2,
        np.nextafter,
    ],
)
@pytest.mark.parametrize('with_out', [False, True])
def test_numpy_ufunc_masked_alignment(operation: Any, with_out: bool) -> None:
    left = Quantity(np.array([[1000.0], [2000.0]]), si.mm)
    right = Quantity(np.array([[1.0, 1e308, 2.0]]), si.m)
    mask = np.array([[True, False, True], [False, False, True]])
    expected = operation(
        Quantity(np.array([1000.0, 1000.0, 2000.0]), si.mm),
        Quantity(np.array([1.0, 2.0, 2.0]), si.m),
    )
    out = Quantity(np.full(mask.shape, -7.0), expected.unit) if with_out else None

    with np.errstate(all='raise'):
        result = operation(left, right, where=mask, out=out)

    assert result.unit == expected.unit
    np.testing.assert_array_equal(result.value[mask], expected.value)
    if out is not None:
        assert result is out
        np.testing.assert_array_equal(out.value[~mask], -7)
    np.testing.assert_array_equal(right.value, [[1.0, 1e308, 2.0]])


@pytest.mark.parametrize(
    'operation',
    [np.sin, np.exp, np.log, np.arcsin, np.arctan, np.degrees, np.radians],
)
def test_numpy_ufunc_masked_dimensionless(operation: Any) -> None:
    unit = si.km / si.m
    source = Quantity(np.array([0.0005, 1e308]), unit)
    expected = operation(Quantity(np.array([0.0005]), unit))
    out = Quantity(np.full(2, -7.0), expected.unit)

    with np.errstate(all='raise'):
        assert operation(source, out=out, where=np.array([True, False])) is out

    np.testing.assert_array_equal(out.value, [expected.value[0], -7])


@pytest.mark.parametrize('operation', [np.logaddexp, np.logaddexp2, np.heaviside])
def test_numpy_ufunc_masked_dimensionless_pair(operation: Any) -> None:
    source = Quantity(np.array([0.0005, 1e308]), si.km / si.m)
    expected = operation(source[:1], source[:1])
    out = Quantity(np.full(2, -7.0), expected.unit)

    with np.errstate(all='raise'):
        assert operation(source, source, out=out, where=np.array([True, False])) is out

    np.testing.assert_array_equal(out.value, [expected.value[0], -7])


@pytest.mark.parametrize(
    'operation', [np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal]
)
def test_numpy_ufunc_masked_comparison(operation: Any) -> None:
    left = Quantity(np.array([1.0, 1e308]), si.km)
    right = Quantity(np.array([1000.0, 1e308]), si.m)
    mask = np.array([True, False])
    out = np.array([False, True])

    with np.errstate(all='raise'):
        assert operation(left, right, out=out, where=mask) is out

    np.testing.assert_array_equal(out, [operation(1000, 1000), True])


@pytest.mark.parametrize('operation', [np.power, np.float_power])
def test_numpy_ufunc_masked_power_base(operation: Any) -> None:
    source = Quantity(np.array([0.0005, 1e308]), si.km / si.m)
    out = Quantity(np.full(2, -7.0), si.one)
    with np.errstate(all='raise'):
        operation(source, 2, out=out, where=np.array([True, False]))
    np.testing.assert_array_equal(out.value, [0.25, -7])


def test_numpy_divmod_masked_conversion() -> None:
    source = Quantity(np.array([2500.0, 1.0]), si.mm)
    divisor = Quantity(np.array([1.0, 1e308]), si.m)
    quotient = Quantity(np.full(2, -7.0), si.one)
    remainder = Quantity(np.full(2, -7.0), si.cm)
    with np.errstate(all='raise'):
        result = np.divmod(
            source, divisor, out=(quotient, remainder), where=np.array([True, False])
        )
    assert result[0] is quotient and result[1] is remainder
    np.testing.assert_array_equal(quotient.value, [2, -7])
    np.testing.assert_array_equal(remainder.value, [50, -7])


@pytest.mark.parametrize('family', [Quantity, AffineQuantity])
def test_numpy_clip_masked_bounds(family: Any) -> None:
    source = family(np.array([500.0, 600.0]), si.mK)
    lower = family(np.array([1.0, 1e308]), si.K)
    upper = family(np.array([2.0, 1e308]), si.K)
    out = family(np.full(2, -7.0), si.mK)
    with np.errstate(all='raise'):
        assert np.clip(source, lower, upper, out=out, where=np.array([True, False])) is out
    np.testing.assert_array_equal(out.value, [1000, -7])


@pytest.mark.parametrize('mask', [False, np.bool_(False), np.array(False), np.empty(0, dtype=bool)])
def test_numpy_ufunc_masked_scalar(mask: Any) -> None:
    out = Quantity(np.full(np.shape(mask), -7.0), si.mm)
    with np.errstate(all='raise'):
        assert np.add(Quantity(1.0, si.mm), Quantity(1e308, si.m), out=out, where=mask) is out
    np.testing.assert_array_equal(out.value, np.full(np.shape(mask), -7.0))


def test_numpy_ufunc_masked_overlap() -> None:
    storage = np.array([1000.0, 1e308, 2000.0, 3000.0])
    source = Quantity(storage[:-1], si.m)
    out = Quantity(storage[1:], si.mm)
    with np.errstate(all='raise'):
        assert np.add(out, source, out=out, where=np.array([False, False, True])) is out
    np.testing.assert_array_equal(storage, [1000, 1e308, 2000, 2003000])


def test_numpy_ufunc_masked_conversion_invalid_overflow() -> None:
    out = Quantity(np.full(2, -7.0), si.mm)
    source = Quantity(np.array([1.0, 1e308]), si.m)
    with np.errstate(all='raise'), pytest.raises(FloatingPointError, match='overflow'):
        np.add(out, source, out=out, where=np.array([False, True]))
    np.testing.assert_array_equal(out.value, [-7, -7])


def test_numpy_ufunc_masked_conversion_invalid_dimension() -> None:
    out = Quantity(np.full(2, -7.0), si.mm)
    with pytest.raises(DimensionError, match='incompatible'):
        np.add(out, Quantity(np.array([1.0, 1e308]), si.s), out=out, where=False)
    np.testing.assert_array_equal(out.value, [-7, -7])
