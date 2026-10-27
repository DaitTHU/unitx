"""Domain exceptions allow selective catches while preserving TypeError."""

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity
from unitx.exceptions import ImplicitConversionError, UnitOperationError, UnitxError
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
