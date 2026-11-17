"""Domain exceptions allow selective catches while preserving TypeError."""

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity
from unitx.exceptions import DimensionError, ImplicitConversionError, UnitOperationError, UnitxError
from unitx.si import si


@pytest.mark.parametrize('point', [20 @ si.degC, 30 @ si.dBm])
def test_unit_operation_error_point_arithmetic(
    point: AffineQuantity[int] | LogQuantity[int],
) -> None:
    with pytest.raises(UnitOperationError) as caught:
        point * 2
    assert isinstance(caught.value, TypeError)
    assert isinstance(caught.value, UnitxError)
    with pytest.raises(UnitOperationError):
        point + point


@pytest.mark.parametrize('sample', [Quantity(1, si.m), 20 @ si.degC, 30 @ si.dBm])
def test_implicit_conversion_error(
    sample: Quantity[int] | AffineQuantity[int] | LogQuantity[int],
) -> None:
    with pytest.raises(ImplicitConversionError) as caught:
        sample.__array__()
    assert isinstance(caught.value, TypeError)
    assert isinstance(caught.value, UnitxError)
    assert isinstance(caught.value, UnitOperationError)


def test_unit_operation_error_native_numeric() -> None:
    with pytest.raises(ZeroDivisionError) as caught:
        Quantity(1, si.m) / 0
    assert not isinstance(caught.value, UnitxError)


def test_dimension_error_conversion_context() -> None:
    with pytest.raises(DimensionError) as caught:
        (1 * si.cm).to(si.s)
    assert "'cm' (dimension L)" in str(caught.value)
    assert "'s' (dimension T)" in str(caught.value)


def test_dimension_error_array_addition_context() -> None:
    np = pytest.importorskip('numpy')
    with pytest.raises(DimensionError) as caught:
        np.add(Quantity(np.ones(2), si.m), 1 * si.s)
    assert "Cannot add quantities with units 'm' (L) and 's' (T)" in str(caught.value)


def test_dimension_error_exponential_hint() -> None:
    np = pytest.importorskip('numpy')
    with pytest.raises(DimensionError) as caught:
        np.exp(Quantity(np.ones(2), si.s))
    assert 'exp requires dimensionless input' in str(caught.value)
    assert "unit 's'" in str(caught.value)
    assert 'Divide by a reference quantity' in str(caught.value)


def test_unit_operation_error_point_addition_hint() -> None:
    with pytest.raises(UnitOperationError, match='Subtract points to obtain a difference'):
        (20 @ si.degC) + (10 @ si.degC)


@pytest.mark.parametrize('operation', ['to', 'value_in', 'construct'])
@pytest.mark.parametrize('magnitude', [0, 1, 2])
def test_quantity_invalid_unit_hint(operation: str, magnitude: int) -> None:
    source = 3 / si.s
    target = magnitude / si.s
    with pytest.raises(TypeError, match='got a Quantity') as caught:
        if operation == 'construct':
            Quantity(3, target)
        else:
            getattr(source, operation)(target)
    message = str(caught.value)
    assert "use si.s**-1 or '1/s'" in message
    assert 'quantity.unit' in message
    assert source.to(si.s**-1) == source.to('1/s') == source
    assert source.value_in(target.unit) == 3


def test_quantity_invalid_unit_array_hint() -> None:
    np = pytest.importorskip('numpy')
    with pytest.raises(TypeError, match='got a Quantity'):
        (3 / si.s).to(np.array([1, 2]) / si.s)
